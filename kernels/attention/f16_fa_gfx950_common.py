"""Shared helpers for CDNA4 gfx950 Flash Attention Gluon kernels."""

import datetime
import json
import math

import torch
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async

def get_mma_type_for_arch(arch: str) -> str:
    """MI350 specialization: only gfx950 is supported."""
    if arch == "gfx950":
        return "mfma_cdna4"
    raise ValueError(f"MI350-only kernel: unsupported GPU architecture {arch}")


# ---------------------------------------------------------------------------
# Gluon kernel helpers
# ---------------------------------------------------------------------------

@gluon.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: gl.constexpr = 8):
    """Remap program IDs to distribute work evenly across XCDs."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (tall_xcds * pids_per_xcd
               + (xcd - tall_xcds) * (pids_per_xcd - 1)
               + local_pid)
    return pid


@gluon.jit
def _nan_propagating_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    """Reduce-max using IEEE 754 maximum (propagates NaN)."""
    return gl.reduce(x, axis, _nan_propagating_max)


@gluon.jit
def do_mma(MMA_TYPE: gl.constexpr, a, b, c):
    """MI350 path: always use CDNA4 MFMA."""
    return mfma_cdna4(a, b, c)


# ---------------------------------------------------------------------------
# Non-pipelined inner loop
# ---------------------------------------------------------------------------

@gluon.jit
def attn_fwd_inner(
    acc, l_i, m_i, q_dot, kt_ptrs, v_ptrs, offs_n, offs_d,
    kt_offs_d, kt_offs_n, start_m,
    stride_kn, stride_vk,
    block_start, block_end,
    kt_smem, v_smem,
    seqlen_q, seqlen_k,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    PRE_LOAD_V: gl.constexpr, MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    MMA_TYPE: gl.constexpr,
    kt_blocked_layout: gl.constexpr, blocked_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Inner attention loop over K/V blocks with shared memory staging."""
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    SEQQ = seqlen_q if VARLEN else MAX_SEQLENS_Q
    for block_n in range(block_start, block_end):
        start_n = block_n * BLOCK_N

        if PRE_LOAD_V:
            if MASK_STEPS:
                v_mask = (start_n + offs_n[:, None]) < SEQK
                if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                    v_mask = v_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v_global = gl.load(v_ptrs)
            v_smem.store(v_global)

        if MASK_STEPS:
            kt_mask = (start_n + kt_offs_n[None, :]) < SEQK
            if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
            kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        else:
            kt_global = gl.load(kt_ptrs)
        kt_smem.store(kt_global)

        k_t = kt_smem.load(kt_blocked_layout)
        kt_dot = gl.convert_layout(k_t, kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(MMA_TYPE, q_dot, kt_dot, qk)
        qk = qk * qk_scale

        if MASK_STEPS and IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            causal_boundary = causal_offs_m[:, None] + (SEQK - SEQQ)
            causal_mask = causal_offs_n[None, :] <= causal_boundary
            qk = gl.where(causal_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                    dtype=gl.float32, layout=mma_layout))

        if MASK_STEPS:
            bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            bound_mask = bound_offs[None, :] < SEQK
            qk = gl.where(bound_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                   dtype=gl.float32, layout=mma_layout))

        m_ij = nan_propagating_max(qk, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        # Varlen ragged-causal rows can attend to zero keys, leaving m_new == -inf.
        if VARLEN:
            m_sub = gl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_sub = m_new
        p = gl.exp2(qk - m_sub[:, None])
        l_ij = gl.sum(p, axis=1)
        alpha = gl.exp2(m_i - m_sub)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        m_i = m_new

        if not PRE_LOAD_V:
            if MASK_STEPS:
                v_mask = (start_n + offs_n[:, None]) < SEQK
                if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                    v_mask = v_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v_global = gl.load(v_ptrs)
            v_smem.store(v_global)

        v = v_smem.load(blocked_layout)
        p_cast = p.to(v.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        v_dot = gl.convert_layout(v, v_dot_layout)
        acc = do_mma(MMA_TYPE, p_dot, v_dot, acc)

        kt_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk

    return acc, l_i, m_i, kt_ptrs, v_ptrs


# ---------------------------------------------------------------------------
# Pipelined inner loop helpers (CDNA4 async copy path)
# ---------------------------------------------------------------------------

@gluon.jit
def issue_async_load_k(
    kt_smem, k_base, start_n,
    stride_kn, stride_kk,
    seqlen_k,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    VARLEN: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = kt_offs_d[:, None] * stride_kk + (start_n + kt_offs_n[None, :]) * stride_kn

    if MASK_STEPS:
        kt_mask = (start_n + kt_offs_n[None, :]) < SEQK
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets)
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_v(
    v_smem, v_base, start_n,
    stride_vk, stride_vn,
    seqlen_k,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    VARLEN: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)
    v_offsets = (start_n + v_offs_n[:, None]) * stride_vk + v_offs_d[None, :] * stride_vn

    if MASK_STEPS:
        v_mask = (start_n + v_offs_n[:, None]) < SEQK
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets, mask=v_mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets)
    cdna4_async.commit_group()


@gluon.jit
def compute_dot1_qk(
    q_dot, kt_dot,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    mma_layout: gl.constexpr,
):
    """Dot1: compute QK^T from register operands. Returns unscaled qk scores."""
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
    qk = do_mma("mfma_cdna4", q_dot, kt_dot, qk)
    return qk


@gluon.jit
def compute_softmax(
    acc, l_i, m_i, qk, start_n, start_m,
    seqlen_q, seqlen_k,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    VARLEN: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Online softmax with optional masking."""
    SEQK = seqlen_k if VARLEN else MAX_SEQLENS_K
    SEQQ = seqlen_q if VARLEN else MAX_SEQLENS_Q
    if MASK_STEPS:
        qk_scaled = qk * qk_scale

        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            causal_boundary = causal_offs_m[:, None] + (SEQK - SEQQ)
            causal_mask = causal_offs_n[None, :] <= causal_boundary
            qk_scaled = gl.where(causal_mask, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                    dtype=gl.float32, layout=mma_layout))

        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        bound_mask = bound_offs[None, :] < SEQK
        qk_scaled = gl.where(bound_mask, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                            dtype=gl.float32, layout=mma_layout))

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        # Varlen ragged-causal rows can attend to zero keys, leaving m_new == -inf.
        if VARLEN:
            m_sub = gl.where(m_new == float("-inf"), 0.0, m_new)
        else:
            m_sub = m_new
        p = gl.exp2(qk_scaled - m_sub[:, None])
    else:
        # FMA-friendly unmasked path.
        m_ij = nan_propagating_max(qk, axis=1) * qk_scale
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        m_sub = m_new
        p = gl.exp2(qk * qk_scale - m_sub[:, None])

    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_sub)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


@gluon.jit
def compute_dot2_pv(acc, p, v_smem, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr):
    """Dot2: Compute P @ V and accumulate."""
    v_dot = cdna4_async.load_shared_relaxed(v_smem, v_dot_layout)
    p_cast = p.to(v_dot.dtype)
    p_dot = gl.convert_layout(p_cast, p_dot_layout)
    acc = do_mma("mfma_cdna4", p_dot, v_dot, acc)
    return acc


# ---------------------------------------------------------------------------
# Benchmark / reference helpers
# ---------------------------------------------------------------------------

class MetaData:
    max_seqlens_q = 0
    max_seqlens_k = 0
    causal = False
    layout = None
    cu_seqlens_q = None
    cu_seqlens_k = None
    varlen = False
    num_contexts = 0
    total_q = 0
    total_k = 0

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def need_causal(self):
        self.causal = True

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        """Enable ragged/thd mode from FAv3-style cu_seqlens indptr tensors."""
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        self.max_seqlens_q = int(seqlens_q.max())
        self.max_seqlens_k = int(seqlens_k.max())
        self.total_q = int(cu_seqlens_q[-1])
        self.total_k = int(cu_seqlens_k[-1])

    def check_args(self, q, k, v, o):
        assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
        if self.varlen:
            assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, \
                "varlen/thd mode expects 3D (total_tokens, heads, head_dim) q/k/v"
            assert self.cu_seqlens_q is not None and self.cu_seqlens_k is not None
            assert self.layout == 'thd'
            assert self.cu_seqlens_q.is_cuda and self.cu_seqlens_k.is_cuda
            assert self.cu_seqlens_q.dtype == torch.int32 and self.cu_seqlens_k.dtype == torch.int32
            assert self.total_q == q.shape[0], \
                f"cu_seqlens_q[-1]={self.total_q} != q tokens {q.shape[0]}"
            assert self.total_k == k.shape[0], \
                f"cu_seqlens_k[-1]={self.total_k} != k tokens {k.shape[0]}"
        else:
            assert q.dim() == 4
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1]
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert o.shape == q.shape
        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, self)
        assert (nheads_q % nheads_k) == 0
        assert head_size <= 256
        assert self.layout is not None


def get_shape_from_layout(q, k, metadata):
    if metadata.layout == 'bhsd':
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == 'bshd':
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    elif metadata.layout == 'thd':
        _, nheads_q, head_size = q.shape
        nheads_k = k.shape[1]
        batch = metadata.num_contexts
    else:
        raise ValueError(f"Unsupported layout: {metadata.layout}")
    return batch, nheads_q, nheads_k, head_size


def get_strides_from_layout(q, k, v, o, metadata):
    if metadata.layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif metadata.layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    elif metadata.layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        raise ValueError(f"Unsupported layout: {metadata.layout}")
    return q_strides, k_strides, v_strides, o_strides


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, requires_grad=False):
    torch.manual_seed(20)
    if layout == 'bhsd':
        q = torch.randn((Z, HQ, N_CTX_Q, D_HEAD), dtype=dtype, device="cuda", requires_grad=requires_grad)
        k = torch.randn((Z, HK, N_CTX_K, D_HEAD), dtype=dtype, device="cuda", requires_grad=requires_grad)
        v = torch.randn((Z, HK, N_CTX_K, D_HEAD), dtype=dtype, device="cuda", requires_grad=requires_grad)
    elif layout == 'bshd':
        q = torch.randn((Z, N_CTX_Q, HQ, D_HEAD), dtype=dtype, device="cuda", requires_grad=requires_grad)
        k = torch.randn((Z, N_CTX_K, HK, D_HEAD), dtype=dtype, device="cuda", requires_grad=requires_grad)
        v = torch.randn((Z, N_CTX_K, HK, D_HEAD), dtype=dtype, device="cuda", requires_grad=requires_grad)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    sm_scale = D_HEAD ** -0.5
    md = MetaData(sm_scale=sm_scale)
    md.max_seqlens_q = N_CTX_Q
    md.max_seqlens_k = N_CTX_K
    md.layout = layout
    return q, k, v, md


def sdpa_reference(q, k, v, causal=False, sm_scale=None):
    """Reference attention with GQA/MQA support."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    if q.shape[1] != k.shape[1]:
        r = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(r, dim=1)
        v = v.repeat_interleave(r, dim=1)
    if not causal:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=sm_scale
        )
    M, N = q.shape[2], k.shape[2]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    mask = torch.tril(torch.ones(M, N, device=q.device, dtype=torch.bool),
                      diagonal=N - M)
    scores.masked_fill_(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p)
    return torch.matmul(p, v.float()).to(q.dtype)


def _check_output(o, o_ref, atol=1e-3, rtol=1e-3):
    diff = (o - o_ref).abs()
    max_diff  = diff.max().item()
    mean_diff = diff.mean().item()
    try:
        torch.testing.assert_close(o, o_ref, check_dtype=False, atol=atol, rtol=rtol)
        return True, max_diff, mean_diff
    except AssertionError:
        return False, max_diff, mean_diff


def compute_flops(B, HQ, M, N, D, causal):
    """Total FLOPs for FA forward pass."""
    if causal:
        valid = ((N**2 + N) / 2) if M > N else (M * N - ((M**2 - M) / 2))
        flops_per_matmul = 2.0 * B * HQ * valid * D
    else:
        flops_per_matmul = 2.0 * B * HQ * M * N * D
    return 2 * flops_per_matmul


def print_results(results, check_only=False):
    """Print benchmark or accuracy-check results as a formatted table."""
    has_accuracy = results and results[0].get('acc') is not None
    has_perf     = results and results[0].get('tflops') is not None

    print()
    print("=" * 110)
    print("Flash Attention (Gluon / gfx950) Benchmark")
    print("=" * 110)
    print()

    if check_only:
        hdr = (f"{'B':>4} {'HQ':>4} {'HK':>4} {'SeqQ':>7} {'SeqK':>7} {'D':>4} "
               f"{'Causal':>6} | {'Acc':>5} {'MaxDiff':>9} {'MeanDiff':>9} | Config")
    else:
        hdr = (f"{'B':>4} {'HQ':>4} {'HK':>4} {'SeqQ':>7} {'SeqK':>7} {'D':>4} "
               f"{'Causal':>6} | {'TFLOPS':>10} {'ms':>8}")
        if has_accuracy:
            hdr += f" | {'Acc':>5} {'MaxDiff':>9}"

    print(hdr)
    print("-" * len(hdr))

    total_tflops = 0
    valid_count  = 0
    pass_count   = 0
    total_configs = len(results)

    for r in results:
        causal_str = "yes" if r['causal'] else "no"

        if check_only:
            a   = r.get('acc')
            md  = r.get('maxd')
            mnd = r.get('meand')
            a_str   = "PASS" if a else "FAIL"
            md_str  = f"{md:.2e}"  if md  is not None and md  != float('inf') else "ERR"
            mnd_str = f"{mnd:.2e}" if mnd is not None and mnd != float('inf') else "ERR"
            cfg = r.get('cfg') or ""
            line = (f"{r['B']:>4} {r['HQ']:>4} {r['HK']:>4} {r['M']:>7} {r['N']:>7} "
                    f"{r['D']:>4} {causal_str:>6} | {a_str:>5} {md_str:>9} {mnd_str:>9} | {cfg}")
            if a:
                pass_count += 1
        else:
            if r['tflops'] is None:
                tf_str = "ERROR"
                ms_str = "N/A"
            else:
                tf_str = f"{r['tflops']:.1f}"
                ms_str = f"{r['ms']:.3f}"
                total_tflops += r['tflops']
                valid_count  += 1
            line = (f"{r['B']:>4} {r['HQ']:>4} {r['HK']:>4} {r['M']:>7} {r['N']:>7} "
                    f"{r['D']:>4} {causal_str:>6} | {tf_str:>10} {ms_str:>8}")
            if has_accuracy:
                a   = r.get('acc')
                md  = r.get('maxd')
                a_str  = "PASS" if a else "FAIL"
                md_str = f"{md:.2e}" if md is not None and md != float('inf') else "ERR"
                line += f" | {a_str:>5} {md_str:>9}"
                if a:
                    pass_count += 1

        print(line)

    print("-" * len(hdr))

    if not check_only and valid_count > 0:
        avg_tf = total_tflops / valid_count
        print(f"{'Average':>49} | {avg_tf:>10.1f}")

    print()
    if has_accuracy or check_only:
        print(f"Accuracy: torch.testing.assert_close(output, torch_ref, check_dtype=False, atol=1e-3, rtol=1e-3)")
        print(f"Reference: PyTorch scaled_dot_product_attention")
        print(f"Gluon: {pass_count}/{total_configs} PASS")
    if not check_only:
        print("Units: TFLOPS")
    print()


def save_results_json(results, path):
    """Save benchmark results to JSON."""
    out = {
        "description": "Gluon FA gfx950 benchmark results",
        "date": datetime.datetime.now().isoformat(),
        "results": [],
        "summary": {},
    }
    total_tf = 0
    n = 0
    for r in results:
        entry = {
            "B": r["B"], "HQ": r["HQ"], "HK": r["HK"],
            "M": r["M"], "N": r["N"], "D": r["D"], "causal": r["causal"],
        }
        if r.get("tflops") is not None:
            entry["gluon"] = {"tflops": round(r["tflops"], 1), "config": r.get("cfg", "")}
            total_tf += r["tflops"]
            n += 1
        out["results"].append(entry)
    if n:
        out["summary"] = {"gluon_avg_tflops": round(total_tf / n, 1)}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
