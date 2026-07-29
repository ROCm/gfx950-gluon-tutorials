"""Shared helpers for the CDNA4 (gfx950) Flash Attention Gluon kernels.

Split by who uses what: the ``@gluon.jit`` helpers are called from inside
``fmha_v3.py`` / ``fmha_v4.py``; the rest is host-side plumbing used by those kernels'
launchers and by ``bench.py`` (problem metadata, input generation, the torch
correctness reference, and the FLOP model).
"""

import math

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# ---------------------------------------------------------------------------
# Kernel-side helpers
# ---------------------------------------------------------------------------


@gluon.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: gl.constexpr = 8):
    """Remap program IDs to distribute work evenly across XCDs.

    Un-round-robins one grid axis so consecutive ids land on the same XCD. Note
    it is keyed on GRID_MN alone, so at GRID_MN == NUM_XCDS it is the identity.
    """
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@gluon.jit
def _nan_propagating_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    """Reduce-max using IEEE 754 maximum (propagates NaN)."""
    return gl.reduce(x, axis, _nan_propagating_max)


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


class MetaData:
    """Host-side descriptor for one attention problem.

    The tutorial kernels are non-causal and take ``bhsd`` or ``bshd`` only, so
    this carries just the fields their launchers and bench.py read.
    """

    causal = False
    layout = None
    max_seqlens_q = 0
    max_seqlens_k = 0

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale


def get_shape_from_layout(q, k, metadata):
    if metadata.layout == "bhsd":
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == "bshd":
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    else:
        raise ValueError(f"Unsupported layout: {metadata.layout}")
    return batch, nheads_q, nheads_k, head_size


def get_strides_from_layout(q, k, v, o, metadata):
    """Strides reordered to the kernel's (batch, head, seq, dim) convention."""
    if metadata.layout == "bhsd":
        order = (0, 1, 2, 3)
    elif metadata.layout == "bshd":
        order = (0, 2, 1, 3)
    else:
        raise ValueError(f"Unsupported layout: {metadata.layout}")
    return tuple(tuple(t.stride(i) for i in order) for t in (q, k, v, o))


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, requires_grad=False):
    torch.manual_seed(20)

    def rand(heads, n_ctx):
        shape = (Z, heads, n_ctx, D_HEAD) if layout == "bhsd" else (Z, n_ctx, heads, D_HEAD)
        return torch.randn(shape, dtype=dtype, device="cuda", requires_grad=requires_grad)

    if layout not in ("bhsd", "bshd"):
        raise ValueError(f"Unsupported layout: {layout}")
    q = rand(HQ, N_CTX_Q)
    k = rand(HK, N_CTX_K)
    v = rand(HK, N_CTX_K)

    md = MetaData(sm_scale=D_HEAD**-0.5)
    md.max_seqlens_q = N_CTX_Q
    md.max_seqlens_k = N_CTX_K
    md.layout = layout
    return q, k, v, md


def sdpa_reference(q, k, v, causal=False, sm_scale=None):
    """Reference attention with GQA/MQA support, in fp32. Expects a bhsd frame.

    Computed and returned in fp32 even for fp16/bf16 inputs, so the reference does
    not carry the output format's own rounding error. That makes the comparison in
    _check_output measure only what the kernel did. torch picks a memory-efficient
    backend for fp32, so the [S, S] score matrix is never materialized.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    q, k, v = q.float(), k.float(), v.float()
    if q.shape[1] != k.shape[1]:
        r = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    if not causal:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=sm_scale
        )
    M, N = q.shape[2], k.shape[2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    mask = torch.tril(torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=N - M)
    scores.masked_fill_(~mask, float("-inf"))
    p = torch.nan_to_num(torch.softmax(scores, dim=-1))
    return torch.matmul(p, v)


def _check_output(o, o_ref, atol=1e-3, rtol=1e-3):
    """Compare against the fp32 reference, with the floor in units of the output format.

    A fixed absolute tolerance is the wrong test for this kernel. The FA output is a
    convex combination of the value rows, so its magnitude falls as ~1/sqrt(S): a short
    sequence produces *larger* numbers and therefore larger absolute rounding error even
    though its relative accuracy is identical. With a constant atol the verdict ends up
    turning on which side of a power of two the single largest element lands, which is
    not a correctness signal -- it made bf16 fail at S=1024 and S=4096 but pass at 2048.

    So the floor scales with the reference instead. sdpa_reference returns fp32, so the
    only rounding in the comparison is the kernel's own, and the measured max error is
    0.5-0.95 ULP of the output dtype at every length in both dtypes -- i.e. the kernel
    is accurate to within one rounding step of the format it writes. Two ULP at the top
    of the range is a ~3x margin on that, and orders of magnitude under any real bug.
    """
    out_dtype = o.dtype
    o, o_ref = o.float(), o_ref.float()
    scale = o_ref.abs().max().item()
    if scale > 0 and math.isfinite(scale):
        # finfo().eps is the relative spacing at 1.0: 2**-7 bf16, 2**-10 fp16. The
        # reference is fp32, so the only rounding left is the kernel's own store to
        # out_dtype -- at most half an ULP. Two ULP at the top of the range is a
        # 4x margin on that, and still an order of magnitude under any real bug.
        atol = max(atol, 2 * torch.finfo(out_dtype).eps * scale)
    diff = (o - o_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    try:
        torch.testing.assert_close(o, o_ref, check_dtype=False, atol=atol, rtol=rtol)
        return True, max_diff, mean_diff
    except AssertionError:
        return False, max_diff, mean_diff


def compute_flops(B, HQ, M, N, D, causal):
    """Total FLOPs for the FA forward pass: two B*HQ*M*N*D matmuls."""
    if causal:
        valid = ((N**2 + N) / 2) if M > N else (M * N - ((M**2 - M) / 2))
        flops_per_matmul = 2.0 * B * HQ * valid * D
    else:
        flops_per_matmul = 2.0 * B * HQ * M * N * D
    return 2 * flops_per_matmul
