"""
FMHA v4 + causal masking: fmha_v4's kernel and grid launch, with the causal bounds and the
mask added, and as little else changed as possible. It is the starting point that fmha_v5.py
improves on, and the home of ``causal_mask``, which both use.

What changes against fmha_v4:

* A 256-row q-block m attends to (m + 1) * 256 keys: the K/V walk stops at the diagonal,
  and only its last four tiles are masked.
* The grid takes the longest q-blocks first (start_m reversed), so the short ones fill the
  tail of the grid.
* The mask is applied in a mem stage (or plain code), never inside a dot cluster: in the
  loop only the last pair's odd half scores a diagonal tile (tile n-4), and its branch on
  that is a uniform scalar branch; tiles n-3..n-1 are the odd tail and the drain.
* The drain's LDS slots are compile-time constants again (n is a multiple of 4): a runtime
  n made them runtime indices, whose LDS addresses no longer fold into the ds_read offsets
  and cost ~90 spilled VGPRs.

The mask (triton-tickets#812). ``causal_mask`` knows the MFMA layout, so per wave each 32x32
block of a score tile is one of three states -- fully kept, fully masked, or the diagonal
block, whose per-register keep-test is a compare against a compile-time column (the
generated bodies are in fmha_causal_mask.py). The plain ``gl.where(row - col >= delta)``
(FA_MASK_IMPL=generic) materializes row - col for all 32 registers of a thread; LLVM hoists
those and keeps them live, and the kernel spills.

What this kernel cannot fix is the launch: uneven workgroups leave CUs idle at every
handover, the per-workgroup prologue and epilogue are paid for blocks of 4 tiles as for
128, and the result is ~75% of fmha_v4's non-causal TFLOPS. fmha_v5.py runs the same
pipeline persistently.

Build environment as fmha_v4 (see README.md): llirSched plugin, MachineSink disabled.
"""

import os
import sys

# The out-of-tree LLIR scheduler ships as an LLVM pass plugin. Loaded via
# LLVM_PASS_PLUGIN_PATH, it resolves LLVM symbols (e.g. llvm::CallbackVH::anchor)
# from libtriton at dlopen time, which requires libtriton in the *global* symbol
# scope. CPython loads C-extensions RTLD_LOCAL by default, so the plugin dies with
# a bogus "undefined symbol" even though libtriton defines it -- and only on a
# *fresh* compile, so a warm ~/.triton/cache hides the breakage. Opt into
# RTLD_GLOBAL before the first `import triton`. ``bench.py`` does the same; doing
# it here too covers harnesses that import this module directly.
if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    if "triton" in sys.modules:
        # Too late for the flag above: libtriton is already loaded RTLD_LOCAL.
        # dlopen-ing a loaded object with RTLD_GLOBAL promotes its symbols into
        # the global scope, which is all the plugin needs.
        import ctypes

        try:
            ctypes.CDLL(
                os.path.join(os.path.dirname(sys.modules["triton"].__file__), "_C", "libtriton.so"),
                mode=ctypes.RTLD_GLOBAL,
            )
        except OSError:
            pass  # best effort; bench.py sets the flag early enough on its own

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd import slice as amd_slice
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language._layouts import (
    DotOperandLayout,
    DistributedLinearLayout,
    PaddedSharedLayout,
)


from fmha_v4_map_elementwise import rescale_acc_pack64, rescale_row
from fmha_causal_mask import causal_mask_pack32, causal_select_pack32
from common import (
    get_shape_from_layout,
    get_strides_from_layout,
    MetaData,
    nan_propagating_max,
    remap_xcd,
)


# ---------------------------------------------------------------------------
# The eight per-tile operations of the rotated 4-cluster loop.
#
# These are the names README.md uses; each of the four warp_pipeline clusters below
# holds two of them:
#
#   dot_qk (DOT1) -- Q * K^T MFMA -> qk scores
#   dot_pv (DOT2) -- P * V   MFMA -> acc
#   VEC1   -- softmax numerator          (new row-max + exp2 burst -> p, alpha)
#   VEC2   -- softmax denominator + acc  (sum p, acc rescale, l_i, p->fp16 cast)
#   LRK    -- local-read  K  (LDS -> regs)
#   LRV    -- local-read  V  (LDS -> regs)
#   ACK    -- async-copy  K  (global -> LDS)
#   ACV    -- async-copy  V  (global -> LDS)
#
# ---------------------------------------------------------------------------

# Lazy-rescale threshold, in log2 (exp2) units. Skip the accumulator correction
# while a row's tile-max stays within this margin of its running max; exp2(8)=256
# of headroom is trivially safe for the fp32 accumulator. (Matches FlyDSL's 8.0.)
LAZY_RESCALE_THRESHOLD = tl.constexpr(8.0)


# FUTURE WORK -- off by default, and left in place rather than reverted so it can be picked
# up. `FA_Q_DIRECT_LDS=1` loads the Q tile with the same global-to-LDS DMA the loop uses for
# K/V instead of staging it through VGPRs, which removes the blanket `s_waitcnt vmcnt(0)`
# the register path forces.
#
# It does what it says: prologue 16429 -> 13313 cycles per workgroup, 8 `global_load` and 8
# `ds_write` gone, VGPRs 256 -> 246, and both paths validate bit-identically. It is not the
# default because the *loop* regresses 2.9% (in-loop MFMA efficiency 94.5% -> 91.8%) for two
# reasons that live outside this kernel:
#
#   - deferring the `q_dot` read past the K/V allocation extends `q_smem`'s live range, so
#     Triton can no longer overlay it with `kt_smem`/`v_smem`; LDS goes 68016 -> 134560 B and
#     the K/V offsets stop fitting a `ds_read`'s 16-bit `offset:` field;
#   - the lower register pressure makes the allocator reassign across the whole function, and
#     the llirSched interleave lands differently on the result.
#
# Picking it up means addressing one of those, not this code. With the switch off, the
# generated prologue, loop and epilogue are byte-identical to the build before this existed.
Q_DIRECT_TO_LDS = tl.constexpr(os.environ.get("FA_Q_DIRECT_LDS", "0") == "1")


@gluon.jit
def sc_vec1(qk, m_run, qk_scale: gl.constexpr, SCALE_ON_Q: gl.constexpr):
    """VEC1: softmax numerator -- new row-max + exp2 burst (DOT2 cluster), LAZY.

    From the qk scores produced by DOT1 this iteration, computes the (lazily
    updated) running max and the unnormalized probabilities p = exp2(qk*scale -
    m_new), and the rescale factor alpha = exp2(m_run - m_new). Five eighths of p
    are exponentiated here; the remaining three slices are handed over raw for
    VEC2 to finish (see the split below). All of it is carried to the next
    iteration, where VEC2 and DOT2 consume it.

    LAZY rescale: instead of always bumping the max to max(m_run, rowmax(qk)),
    the max is only advanced for rows whose tile-max exceeds the running max by
    more than LAZY_RESCALE_THRESHOLD. For a "stable" row m_new == m_run, so
    alpha = exp2(0) = 1 and VEC2's ``acc *= alpha`` is a no-op -- the correction
    is deferred. p = exp2(qk*scale - m_run) can then exceed 1 (up to ~2**thr) but
    acc and l_i stay in the same lagging frame, so acc / l_i is unchanged. Softmax
    shift-invariance makes this exact; the threshold only bounds the exp2 range.
    """
    # SCALE_ON_Q: qk already carries qk_scale because it was folded
    # into Q before the loop, so the row max needs no scale multiply here.
    if SCALE_ON_Q:
        m_ij = nan_propagating_max(qk, axis=1)
    else:
        m_ij = nan_propagating_max(qk, axis=1) * qk_scale
    m_cand = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
    # Per-row lazy gate: advance the max only where the jump exceeds the threshold;
    # otherwise keep the (stale) running max so the rescale is skipped. Deliberately
    # branchless -- gl.where keeps control flow uniform here, inside a dot cluster,
    # where a branch would be scheduled ahead of the first MFMA and stall the matrix
    # core. The branch that actually elides the acc multiply lives in rescale_lazy(),
    # in a mem cluster, where its cost lands against memory latency instead.
    m_new = gl.where(m_cand - m_run > LAZY_RESCALE_THRESHOLD, m_cand, m_run)
    # Slice the [BLOCK_M, BLOCK_N] = 256x64 score tile into EIGHT 256x8 column slices
    # T0..T7 and split the softmax numerator 5:3 -- sub + exp2 on T0..T4 stay here in
    # VEC1 (DOT2), and BOTH the sub and the exp2 for T5/T6/T7 move to VEC2 (DOT1).
    #
    # Why 3/8: the two dot clusters share a fixed co-execution budget of 384 cycles each,
    # and moving elementwise work between them is the only way to level them. At 3/8 the
    # softmax asks 324 cycles of DOT2 and 356 of DOT1, both inside with margin on either
    # side; eighths are the granularity that makes that ratio reachable. README section 5
    # sweeps it.
    #
    # The sub moves with its exp2. A subtract left behind in VEC1 has its only consumer in
    # the other cluster, and the scheduler sinks it there -- a pure fsub carries no chain
    # edge, so no barrier holds it in place. Carrying the RAW slices qk_5..qk_7 across
    # means the sub is born in the cluster that consumes it.
    #
    # Register-neutral: p_0123 (4 slices) + p_4 + qk_5 + qk_6 + qk_7 carries the same
    # element count as the full p, and the result is exact since exp2 is elementwise.
    qk_lo, qk_hi = _split_halves(qk)        # 2 x [256, 32]
    qk_a, qk_b = _split_halves(qk_lo)       # 4 x [256, 16]
    qk_c, qk_d = _split_halves(qk_hi)
    qk_0, qk_1 = _split_halves(qk_a)        # 8 x [256, 8]
    qk_2, qk_3 = _split_halves(qk_b)
    qk_4, qk_5 = _split_halves(qk_c)
    qk_6, qk_7 = _split_halves(qk_d)
    if SCALE_ON_Q:
        p_0 = gl.exp2(qk_0 - m_new[:, None])
        p_1 = gl.exp2(qk_1 - m_new[:, None])
        p_2 = gl.exp2(qk_2 - m_new[:, None])
        p_3 = gl.exp2(qk_3 - m_new[:, None])
        p_4 = gl.exp2(qk_4 - m_new[:, None])
    else:
        # SCALE_ON_Q=False: the scale has to be applied per element, so slice the
        # fma's input instead and keep the mul+sub fused (gl.fma -> one
        # llvm.fmuladd), otherwise llirSched double-counts the un-contracted pair.
        p_0 = gl.exp2(gl.fma(qk_0, qk_scale, -m_new[:, None]))
        p_1 = gl.exp2(gl.fma(qk_1, qk_scale, -m_new[:, None]))
        p_2 = gl.exp2(gl.fma(qk_2, qk_scale, -m_new[:, None]))
        p_3 = gl.exp2(gl.fma(qk_3, qk_scale, -m_new[:, None]))
        p_4 = gl.exp2(gl.fma(qk_4, qk_scale, -m_new[:, None]))
    p_0123 = _concat_halves(_concat_halves(p_0, p_1), _concat_halves(p_2, p_3))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p_0123, p_4, qk_5, qk_6, qk_7, alpha


@gluon.jit
def _split_halves(x):
    """Split a 2D tensor into its left/right column halves, FREE of any code.

    reshape/permute/split are pure layout re-interpretations; each half keeps the
    SAME distributed (MFMA) layout as ``x``, so the split is a plain partition of
    each lane's registers -- no data movement, no v_mov, no extra instruction.
    ``assert_trivial=True`` makes the compiler *prove* that at build time (it fails
    the compile if the conversion would need real work). Same idea as
    split_subtile() in the upstream mxfp_fa_gfx1250 Gluon example.
    """
    half: gl.constexpr = x.shape[1] // 2
    a0 = amd_slice(x, [x.shape[0], half], [0, 0])
    a1 = amd_slice(x, [x.shape[0], half], [0, half])
    return a0, a1


@gluon.jit
def _concat_halves(x, y):
    """Inverse of _split_halves -- rejoin left/right halves, also code-free."""
    gl.static_assert(x.type.layout == y.type.layout)
    layout: gl.constexpr = x.type.layout
    shape: gl.constexpr = [x.shape[0], x.shape[1] + y.shape[1]]
    a = gl.join(x, y).permute(0, 2, 1).reshape(shape)
    return gl.convert_layout(a, layout, assert_trivial=True)


# FA_MASK_IMPL picks the mask body causal_mask() uses when its caller does not: "select"
# (default, branch-free), "3state" (per-wave branches), or "generic" -- a plain gl.where on
# row - col, kept for comparison: it is the generic masking triton-tickets#812 is about.
MASK_IMPL = os.environ.get("FA_MASK_IMPL", "select")
MASK_GENERIC = tl.constexpr(MASK_IMPL == "generic")
MASK_3STATE = tl.constexpr(MASK_IMPL == "3state")


@gluon.jit
def causal_mask(qk, delta, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                THREE_STATE: gl.constexpr = None):
    """Causal mask for one score tile: keep (r, c) iff r - c >= delta.

    ``delta`` is the tile's first key column minus the block's first query row,
    so it is 0 for the first diagonal tile and grows by BLOCK_N per tile.

    The 3-state form (default) knows the MFMA layout: per wave, each 32x32 block of
    the tile is fully kept, fully masked, or the diagonal block, and only the last
    needs per-element work -- a compare against a compile-time column (see
    fmha_causal_mask.py). The generic ``gl.where`` form materializes row - col for
    all 32 registers of a thread, which LLVM hoists and keeps live, and spills.

    ``THREE_STATE`` picks the scalar body: True branches per wave on the block state
    (cheapest when most blocks are fully kept -- plain code with a runtime delta),
    False compares every register (branch-free: what a warp-pipelined loop needs).
    None takes the FA_MASK_IMPL default.
    """
    layout: gl.constexpr = qk.type.layout
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
    if MASK_GENERIC:
        keep = (rows[:, None] - cols[None, :]) >= delta
        return gl.where(keep, qk, float("-inf"))
    gl.static_assert(BLOCK_N == 64)
    gl.static_assert(BLOCK_M == 32 * gl.num_warps())
    # d: row - col - delta per element; register 0 of each thread holds column 4*(lane//32),
    # so d0 is the per-lane quantity the scalar body compares against its constants.
    d = rows[:, None] - cols[None, :] - delta
    if (MASK_3STATE if THREE_STATE is None else THREE_STATE):
        # k: 32 * wave - delta, uniform per wave (rows of a wave are 32*wave + lane%32).
        k = (rows[:, None] // 32) * 32 - delta + cols[None, :] * 0
        qk, = gl.map_elementwise(causal_mask_pack32, qk, d, k, pack=32)
    else:
        qk, = gl.map_elementwise(causal_select_pack32, qk, d, pack=32)
    return qk


@gluon.jit
def rescale_lazy(acc, l_i, alpha):
    """The deferred per-row correction, hoisted into the PRIOR mem stage.

    acc *= alpha and l_i *= alpha, skipped per wave when no row advanced, run in
    the mem2 stage -- paired with the LRK/ACV loads -- instead of sitting ahead of
    the DOT1 QK mfma. alpha was produced by VEC1 in THIS tile's DOT2; applying the
    correction here (before the NEXT tile's DOT2 accumulates) keeps acc in the
    advanced frame while overlapping the branch/control with LDS/global latency.

    PER-WAVE LAZY SKIP: ``gl.map_elementwise`` hands each thread's elements to a
    scalar function (``pack`` = elements per thread), whose ``if alpha != 1``
    becomes one branch per thread. The backend lowers it to ``s_and_saveexec`` +
    ``s_cbranch_execz``, so each wavefront independently skips the rescale when
    none of ITS rows advanced -- no cross-warp reduction. Rows that did not
    advance carry alpha == 1, so the multiply is exact wherever it does run.
    With BLOCK_M=256, BLOCK_DMODEL=128 and 8 warps a thread holds 64 accumulator
    elements (all in one row) and one row of l_i; see fmha_v4_map_elementwise.py.
    """
    alpha_b = gl.broadcast(alpha[:, None], acc)[0]
    acc, = gl.map_elementwise(rescale_acc_pack64, acc, alpha_b, pack=64)
    l_i = gl.map_elementwise(rescale_row, l_i, alpha)
    return acc, l_i


@gluon.jit
def sc_vec2(l_i, p_0123, p_4, qk_5, qk_6, qk_7, m_new,
            p_dot_layout: gl.constexpr, out_dtype: gl.constexpr,
            qk_scale: gl.constexpr, SCALE_ON_Q: gl.constexpr):
    """VEC2: softmax denominator + DOT2-operand convert (DOT1 cluster).

    The accumulator/denominator rescale (acc *= alpha, l_i *= alpha) has been
    hoisted OUT into rescale_lazy() in the PRIOR mem2 stage, leaving only the
    unconditional denominator update (l_i += sum p) and the p->fp16 convert that
    prepares the DOT2 operand. p was produced by VEC1 in the previous iteration and
    l_i's frame was already corrected by the preceding rescale_lazy, so the split
    ``l_i * alpha`` (in rescale_lazy) ``+ l_ij`` (here) is exact; for stable rows
    alpha == 1. This leaves the DOT1 stage as just [sum + cvt] + the QK mfma, with
    no branch/control ahead of the mfma.

    It also runs the deferred fraction of VEC1's exp2 burst (VEC1 hands over the
    un-exp'd argument t_r), balancing VALU between the two DOT stages.
    """
    # The deferred 3/8 -- their subtracts happen here too, not in VEC1, so the
    # ops live in the stage that consumes them (see sc_vec1). Then rejoin; slice and
    # concat are both free (see _split_halves).
    if SCALE_ON_Q:
        p_5 = gl.exp2(qk_5 - m_new[:, None])
        p_6 = gl.exp2(qk_6 - m_new[:, None])
        p_7 = gl.exp2(qk_7 - m_new[:, None])
    else:
        p_5 = gl.exp2(gl.fma(qk_5, qk_scale, -m_new[:, None]))
        p_6 = gl.exp2(gl.fma(qk_6, qk_scale, -m_new[:, None]))
        p_7 = gl.exp2(gl.fma(qk_7, qk_scale, -m_new[:, None]))
    p_4567 = _concat_halves(_concat_halves(p_4, p_5), _concat_halves(p_6, p_7))
    p = _concat_halves(p_0123, p_4567)
    l_ij = gl.sum(p, axis=1)
    p_dot = gl.convert_layout(p.to(out_dtype), p_dot_layout)
    l_i = l_i + l_ij
    return l_i, p_dot


# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------

def get_gluon_cdna_autotune_configs():
    # Simplified tutorial baseline: the single most performant config for the
    # focus shape (D=128, non-causal). Full autotune space is in git history.
    #
    # llvm_fn_attrs amdgpu-agpr-alloc="0,0" forces 0 AGPRs, so the accumulators live in
    # VGPRs. Left to itself the backend parks them in AGPRs and moves them in and out with
    # v_accvgpr; in the 2x-unrolled loop those moves land on the critical path and cost
    # several percent of throughput. The tuple form is required -- as a string, "0,0" is
    # split on its comma.
    return [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2,
                       'llvm_fn_attrs': (("amdgpu-agpr-alloc", "0,0"),)}, num_warps=8),
    ]


GLUON_AUTOTUNE_KEYS = ['IS_CAUSAL', 'N_CTX', 'HQ', 'HK']


# ---------------------------------------------------------------------------
# Main Gluon kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=get_gluon_cdna_autotune_configs(),
    key=GLUON_AUTOTUNE_KEYS,
)
@gluon.jit
def gluon_attn_fwd(Q, K, V, SM_SCALE: gl.constexpr, L, Out,
                   stride_qz, stride_qh, stride_qm, stride_qk,
                   stride_kz, stride_kh, stride_kn, stride_kk,
                   stride_vz, stride_vh, stride_vk, stride_vn,
                   stride_oz, stride_oh, stride_om, stride_on,
                   HQ: gl.constexpr, HK: gl.constexpr,
                   N_CTX: gl.constexpr,
                   IS_CAUSAL: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr,
                   SCALE_ON_Q: gl.constexpr = True):
    """
    Gluon FMHA forward kernel (AMD CDNA4 / gfx950).
    Grid: (num_heads_q, num_m_blocks, batch)
    """
    num_warps: gl.constexpr = gl.num_warps()

    off_h_q = gl.program_id(0)
    off_h_q = remap_xcd(off_h_q, HQ)
    start_m  = gl.program_id(1)
    if IS_CAUSAL:
        # Longest-first: block m attends to (m + 1) * BLOCK_M keys, so dispatch the
        # last m-block first and let the short ones fill the tail of the grid.
        start_m = (N_CTX // BLOCK_M) - 1 - start_m
    off_z    = gl.program_id(2)
    off_h_k  = off_h_q * HK // HQ

    mma_layout: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, 16],
                                              transposed=True, warps_per_cta=[num_warps, 1])
    k_width:          gl.constexpr = 8
    threads_per_warp: gl.constexpr = 64
    pv_k_width:       gl.constexpr = 4

    q_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=k_width)
    kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=k_width)
    p_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=pv_k_width)
    v_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=pv_k_width)

    # Store layout for the O tile (BLOCK_M x BLOCK_DMODEL = 256 x 128). `order=[1, 0]` puts
    # the D axis contiguous, and `threads_per_warp=[4, 16]` spreads 16 lanes along it: 16
    # lanes x `size_per_thread`=8 covers all 128 columns of a row, so a quarter-warp holds
    # one full row as 256 contiguous bytes, and the 8 bf16 per lane are exactly the
    # `dwordx4` the store wants.
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, threads_per_warp // 4],
        warps_per_cta=[num_warps, 1], order=[1, 0])

    # The LSE store is 1-D over BLOCK_M and wants consecutive rows on consecutive lanes, so
    # it needs its own M-major arrangement: `threads_per_warp=[16, 4]` puts 16 lanes along M
    # and `lse` goes out as 2 contiguous `global_store_dword`. Reusing `blocked_layout` for it
    # would leave 4 lanes along M and 8 stride-4 stores -- invisible at S=8192, where LSE is
    # 1 KB against O's 64 KB, and a 3% regression at S=2048, where the epilogue is a third of
    # the dispatch.
    lse_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1], order=[1, 0])

    offs_m_layout:    gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout:    gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    offs_m_lse_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=lse_layout)
    mma_m_layout:     gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    offs_m    = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d    = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_m_lse = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_lse_layout)

    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    # `FA_Q_DIRECT_LDS=1` loads the Q tile with the same global-to-LDS DMA the loop uses for
    # K/V, so it never passes through VGPRs and the register path's blanket
    # `s_waitcnt vmcnt(0)` disappears. A buffer DMA has no per-element mask, and the kernel's
    # shape scope only guarantees `N_CTX % BLOCK_N == 0`, so a partial M tile falls back.
    # Everything below is gated: with the switch off this is the register path, byte for
    # byte. The DMA wins the prologue and loses more in the loop.
    Q_DMA: gl.constexpr = Q_DIRECT_TO_LDS and N_CTX % BLOCK_M == 0

    if Q_DMA:
        # A DMA-writable *padded* shared layout, plus an address layout to build the offsets
        # from. Both are the V pattern extended along dim 0: V is [64, 128], Q is [256, 128],
        # same contiguous dimension, so only the row bases grow -- 13 offset bases become 15,
        # and the per-thread register bases 4 become 6 (32768 elements / 512 threads = 64
        # each, eight 16-byte DMA granules). Bits: reg 6 + lane 6 + warp 3 = 15 = 256 x 128.
        q_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, 8]],
            offset_bases=[
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                [16, 0], [32, 0], [64, 0], [128, 0],
                [1, 0], [2, 0], [4, 0], [8, 0]
            ],
            cga_layout=[],
            shape=[BLOCK_M, BLOCK_DMODEL])
        q_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [64, 0], [128, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]],
            block_bases=[],
            shape=[BLOCK_M, BLOCK_DMODEL])
        q_smem = gl.allocate_shared_memory(
            Q.dtype.element_ty, [BLOCK_M, BLOCK_DMODEL], layout=q_smem_layout)
        q_ad: gl.constexpr = gl.SliceLayout(dim=1, parent=q_async_layout)
        q_an: gl.constexpr = gl.SliceLayout(dim=0, parent=q_async_layout)
        q_off = (gl.arange(0, BLOCK_M, layout=q_ad)[:, None] * stride_qm
                 + gl.arange(0, BLOCK_DMODEL, layout=q_an)[None, :] * stride_qk)
        # Issued at the top of the kernel and *not* waited for: it is the oldest async group,
        # so the prologue's existing `wait_group(2)` before the first QK drains it together
        # with K[0]. `q_dot` is read there, and the scale applied to it, because with the DMA
        # Q is never in registers on the way in. Issuing it further down cost 3505 cycles.
        cdna4_async.buffer_load_to_shared(
            q_smem, q_base + start_m * BLOCK_M * stride_qm, q_off)
        cdna4_async.commit_group()   # ACQ
    else:
        q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
            vec=8, per_phase=1, max_phase=16, order=[1, 0])
        q_smem = gl.allocate_shared_memory(
            Q.dtype.element_ty, [BLOCK_M, BLOCK_DMODEL], layout=q_smem_layout)
        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_mask = offs_m[:, None] < N_CTX
        q = gl.load(q_ptrs, mask=q_mask, other=0.0)
        # SCALE_ON_Q (the default): fold qk_scale into Q ONCE here, outside the loop.
        # qk then already carries the scale, so VEC1's per-element fma(qk, qk_scale, -m_new)
        # collapses to a plain subtract and the row max needs no scale multiply -- it also
        # shortens the row-max dependency chain. Costs one extra rounding of Q: max error
        # 1.22e-04 vs 6.10e-05, both inside the 1e-3 tolerance. Worth ~+0.5%.
        if SCALE_ON_Q:
            q = (q.to(gl.float32) * qk_scale).to(Q.dtype.element_ty)
        q_smem.store(q)
        q_dot = q_smem.load(q_dot_layout)

    m_i  = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i  = gl.full([BLOCK_M], 1.0,           dtype=gl.float32, layout=mma_m_layout)
    acc  = gl.zeros([BLOCK_M, BLOCK_DMODEL],  dtype=gl.float32, layout=mma_layout)

    # Simplified tutorial kernel: non-causal, K length a multiple of BLOCK_N, so
    # every K/V block is full and unmasked (no causal / no ragged-tail masking).
    n_blocks = (N_CTX + BLOCK_N - 1) // BLOCK_N
    if IS_CAUSAL:
        # Keys past the block's last query row are fully masked: stop there. The last
        # BLOCK_M // BLOCK_N tiles straddle the diagonal and are the only masked ones.
        gl.static_assert(N_CTX % BLOCK_M == 0)
        gl.static_assert(BLOCK_M // BLOCK_N == 4)
        n_blocks = (start_m + 1) * (BLOCK_M // BLOCK_N)


    # Single supported config: D=128, BLOCK_N=64, 8 warps. The full per-
    # (BLOCK_DMODEL, BLOCK_N, num_warps) layout dispatch was dropped for the tutorial.
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
        [0, 16], [0, 32],
        [0, 1], [0, 2], [0, 4], [0, 8]
    ]
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
        [16, 0], [32, 0],
        [1, 0], [2, 0], [4, 0], [8, 0]
    ]
    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
        warp_bases=[[0, 1], [0, 2], [0, 4]],
        block_bases=[],
        shape=[BLOCK_DMODEL, BLOCK_N])
    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
        warp_bases=[[1, 0], [2, 0], [4, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_DMODEL])

    kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, 8]],
        offset_bases=kt_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DMODEL, BLOCK_N])
    v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, 32]],
        offset_bases=v_offset_bases,
        cga_layout=[],
        shape=[BLOCK_N, BLOCK_DMODEL])

    BUF_DEPTH: gl.constexpr = 2
    kt_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty, [BUF_DEPTH, BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
    v_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty, [BUF_DEPTH, BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)


    # === Rotated 4-cluster pipelined inner loop (inlined) ===
    # Whole [0, n_blocks) K/V range; every block full and unmasked.
    # 4 pipeline stages (s0 = this tile's output .. s3 = K prefetch 3-ahead). The
    # softmax numerator (VEC1) is rotated one stage ahead so its exp2 burst lands
    # after the P*V MFMA and feeds the NEXT iteration's dot_pv:
    #   dot_pv s0  VEC2 s0  LRV s0 | dot_qk s1  VEC1 s1 | LRK s2  ACV s2 | ACK s3
    # LDS is double-buffered (BUF_DEPTH=2) for K and V; deeper stages ride in regs
    # and in-flight async copies.
    block_start = 0
    block_end = n_blocks

    # The main loop is 2x-unrolled over [block_start, block_end-3), so that range
    # may hold an odd tile over. Computed from constexpr N_CTX/BLOCK_N (block_end
    # itself is a runtime value here), so the guard below resolves at compile time
    # and costs nothing when it is false -- seqlen 16320 builds a byte-identical
    # opcode stream either way.
    NUM_BLOCKS: gl.constexpr = (N_CTX + BLOCK_N - 1) // BLOCK_N
    # When (n_blocks-3) is odd the 2x-unrolled loop leaves one tile over. It is
    # always an "even" tile (LDS slots cur=0/next=1) and it is always tile n-4,
    # so the drain below needs no change -- it already keys its slots off
    # (index - block_start) % BUF_DEPTH at runtime.
    # Causal: n_blocks = 4 * (start_m + 1), so n_blocks - 3 is always odd.
    ODD_TAIL: gl.constexpr = IS_CAUSAL or (NUM_BLOCKS - 3) % 2 == 1

    # Fixed async-copy offset (intra-tile pattern) computed once; each tile loads
    # from a base pointer advanced by a constant step, so the offset never changes.
    kt_ad: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_an: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_off = (gl.arange(0, BLOCK_DMODEL, layout=kt_ad)[:, None] * stride_kk
              + gl.arange(0, BLOCK_N, layout=kt_an)[None, :] * stride_kn)
    v_an: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_ad: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_off = (gl.arange(0, BLOCK_N, layout=v_an)[:, None] * stride_vk
             + gl.arange(0, BLOCK_DMODEL, layout=v_ad)[None, :] * stride_vn)
    kt_step = BLOCK_N * stride_kn   # per-tile base pointer advance
    v_step = BLOCK_N * stride_vk

    if not Q_DMA:
        # No-op when nothing has been issued, which is the case on the register path. With Q
        # in flight it would drain ACQ and expose the very latency the DMA path hides, so it
        # is gated rather than deleted.
        cdna4_async.wait_group(0)


    # Intended steady-state async depth: keep 2*BUF_DEPTH-2 == 2 commit groups in
    # flight, so a wait_group(2) before each LDS read drains exactly the tile being
    # read (the oldest of 3 outstanding). The two loop reads use WAIT_LOOP-1 though:
    # the LLVM backend derives a too-loose s_waitcnt vmcnt from wait_group(2) under
    # this kernel's register pressure, letting an LDS ds_read race ahead of its
    # global->LDS async copy. Waiting for one fewer group forces a tight enough vmcnt
    # (the extra-drained group is not yet needed) and costs no measured performance.
    WAIT_LOOP: gl.constexpr = 2 * BUF_DEPTH - 2  # == 2

    # -- Prologue ----------------------------------------------------------
    # Prime the rotated pipeline for output tile 0: compute the FULL ahead-work
    # for tile 0 (qk[0], m_new[0], and the exp2 burst p[0]/alpha[0]) and the K
    # regs for tile 1, plus stage K[0..2] / V[0..1] into LDS. K is prefetched
    # 3-ahead so three K tiles (0,1,2) must be staged into the 2 K slots -- slot
    # 0 is reused for K[2] after LRK[0] reads K[0] (guarded by a barrier).
    #
    # Commit order: K0, V0, K1, (barrier) K2, V1  ->  end pending {K2, V1},
    # matching the loop's steady-state entry condition.
    cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_base, kt_off)
    cdna4_async.commit_group()  # ACK[0]
    cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base, v_off)
    cdna4_async.commit_group()   # ACV[0]
    cdna4_async.buffer_load_to_shared(kt_smem.index(1), k_base + kt_step, kt_off)
    cdna4_async.commit_group()  # ACK[1]

    cdna4_async.wait_group(2)                                       # Q and K[0] complete
    # SCALE_ON_Q (the default): fold qk_scale into Q ONCE, outside the loop. qk then
    # already carries the scale, so VEC1's per-element fma(qk, qk_scale, -m_new) collapses
    # to a plain subtract and the row max needs no scale multiply -- it also shortens the
    # row-max dependency chain. Costs one extra rounding of Q: max error 1.22e-04 vs
    # 6.10e-05, both inside the 1e-3 tolerance. Worth ~+0.5%.
    #
    # It is applied *here*, to the operand read out of LDS, not to Q on its way in. Same
    # arithmetic in the same order and still once per kernel, but it no longer forces Q
    # through VGPRs, which is what lets the load above be a DMA. q_dot lives in registers
    # for every iteration, so scaling it here is scaling it once.
    if Q_DMA:
        # Q was DMA'd, so it is read here -- the wait_group(2) above covered ACQ -- and the
        # scale is applied to the operand instead of to Q on its way in. Same arithmetic in
        # the same order, still once per kernel, since q_dot lives in registers for every
        # iteration.
        q_dot = q_smem.load(q_dot_layout)
        if SCALE_ON_Q:
            q_dot = (q_dot.to(gl.float32) * qk_scale).to(Q.dtype.element_ty)
    kt0 = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)                    # LRK[0] -> K regs tile 0
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
    qk = mfma_cdna4(q_dot, kt0, qk)  # dot_qk[0] -> qk[0]
    if IS_CAUSAL:
        qk = causal_mask(qk, -start_m * BLOCK_M, BLOCK_M, BLOCK_N)
    m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_i, qk_scale, SCALE_ON_Q)   # VEC1[0] -> m_new[0], p[0], alpha[0]=0

    gl.barrier()                                                   # WAR: LRK[0] ds_read vs K[2] write
    cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_base + 2 * kt_step, kt_off)
    cdna4_async.commit_group()  # ACK[2] (slot0 reuse)
    cdna4_async.wait_group(1)                                       # K[1] complete
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(1), kt_dot_layout)                 # LRK[1] -> K regs tile 1
    cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + v_step, v_off)
    cdna4_async.commit_group()   # ACV[1]

    # -- Main loop (2x-unrolled: even tile then odd tile) ------------------
    # Runs output tiles [block_start, block_end-3). Unrolling by BUF_DEPTH=2
    # makes the ping-pong LDS slots compile-time constants (0/1) instead of a
    # runtime `% BUF_DEPTH`, dropping the slot arithmetic from the hot loop. An
    # odd tail tile (constexpr) is handled after the loop.
    # Rescale for tile 0's DOT2, using the prologue's alpha_c[0]. The loop's
    # mem2 stages carry the rescale for every later tile, so the DOT1 sc_vec2 is
    # left as just sum+cvt with no branch ahead of the QK mfma.
    acc, l_i = rescale_lazy(acc, l_i, alpha_c)

    main_loop_pairs = (block_end - 3 - block_start) // 2
    for pair_idx in tl.range(0, main_loop_pairs):
        block_n = block_start + pair_idx * 2

        # even tile (block_n): LDS slots cur=0, next=1
        with warp_pipeline_stage("dot1"):
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_dot, qk)   # dot_qk
            l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)   # VEC2 (sum+cvt; rescale in prior mem2)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)   # LRV
            cdna4_async.buffer_load_to_shared(kt_smem.index(1), k_base + (block_n + 3) * kt_step, kt_off)   # ACK
            cdna4_async.commit_group()
        with warp_pipeline_stage("dot2"):
            acc = mfma_cdna4(p_dot, v_dot, acc)   # dot_pv
            m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)   # VEC1
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)   # LRK
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # rescale for next DOT1, overlapped w/ mem latency

        # odd tile (block_n+1): LDS slots cur=1, next=0
        with warp_pipeline_stage("dot1"):
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_dot, qk)   # dot_qk
            l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)   # VEC2 (sum+cvt; rescale in prior mem2)
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)   # LRV
            cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_base + (block_n + 4) * kt_step, kt_off)   # ACK
            cdna4_async.commit_group()
            if IS_CAUSAL:
                # The last pair's odd half scores tile n-4, the first diagonal tile.
                if pair_idx == main_loop_pairs - 1:
                    qk = causal_mask(qk, 0, BLOCK_M, BLOCK_N)
        with warp_pipeline_stage("dot2"):
            acc = mfma_cdna4(p_dot, v_dot, acc)   # dot_pv
            m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)   # VEC1
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(1), kt_dot_layout)   # LRK
            cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + (block_n + 3) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # rescale for next DOT1, overlapped w/ mem latency

    # -- Odd tail tile (constexpr; only when n_blocks is even) -------------
    if ODD_TAIL:
        block_n = block_start + main_loop_pairs * 2
        with warp_pipeline_stage("dot1"):
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_dot, qk)   # dot_qk
            l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)   # VEC2
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)   # LRV
            cdna4_async.buffer_load_to_shared(kt_smem.index(1), k_base + (block_n + 3) * kt_step, kt_off)   # ACK
            cdna4_async.commit_group()
            if IS_CAUSAL:
                qk = causal_mask(qk, BLOCK_N, BLOCK_M, BLOCK_N)   # tile n-3
        with warp_pipeline_stage("dot2"):
            acc = mfma_cdna4(p_dot, v_dot, acc)   # dot_pv
            m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)   # VEC1
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)   # LRK
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # rescale for the drain's first DOT1

    # -- Drain (last 3 output tiles, no OOB global prefetch) ---------------
    # After the loop: outputs [.., n-4] done; K[0..n-1] and V[0..n-2] in LDS
    # (V[n-1] still to load); carried kt_dot=K regs tile n-2, m_run=m_new[n-3],
    # p_c=p[n-3], alpha_c=alpha[n-3]; pending async {V[n-3],K[n-1],V[n-2]}.
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    if IS_CAUSAL:
        # n_blocks is a runtime multiple of 4 here, so the slots are still compile-time
        # constants -- keep them so: runtime slot indices cost ~90 spilled VGPRs of LDS
        # addresses that can no longer fold into the ds_read offsets.
        s_nm3: gl.constexpr = 1
        s_nm2: gl.constexpr = 0
        s_nm1: gl.constexpr = 1
    else:
        s_nm3 = ((nm3 - block_start) % BUF_DEPTH).to(tl.int32)
        s_nm2 = ((nm2 - block_start) % BUF_DEPTH).to(tl.int32)
        s_nm1 = ((nm1 - block_start) % BUF_DEPTH).to(tl.int32)

    # output tile n-3 (also issues the final V prefetch, ACV[n-1])
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
    qk = mfma_cdna4(q_dot, kt_dot, qk)   # dot_qk[n-2]
    if IS_CAUSAL:
        qk = causal_mask(qk, 2 * BLOCK_N, BLOCK_M, BLOCK_N)
    cdna4_async.wait_group(2)                                           # V[n-3] complete
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm3), v_dot_layout)                    # LRV[n-3]
    l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)  # VEC2[n-3] (rescale done in loop's last mem2)
    acc = mfma_cdna4(p_dot, v_dot, acc)                                  # dot_pv[n-3]
    m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)  # VEC1[n-2] -> m_new, p[n-2]
    acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # rescale for tile n-2
    gl.barrier()                                                       # WAR: LRV[n-3] vs V[n-1] write
    cdna4_async.buffer_load_to_shared(v_smem.index(s_nm1), v_base + nm1 * v_step, v_off)
    cdna4_async.commit_group()   # ACV[n-1]
    cdna4_async.wait_group(2)                                           # K[n-1] complete
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(s_nm1), kt_dot_layout)                 # LRK[n-1] -> K regs tile n-1

    # output tile n-2
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
    qk = mfma_cdna4(q_dot, kt_dot, qk)   # dot_qk[n-1]
    if IS_CAUSAL:
        qk = causal_mask(qk, 3 * BLOCK_N, BLOCK_M, BLOCK_N)
    cdna4_async.wait_group(1)                                           # V[n-2] complete
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm2), v_dot_layout)                    # LRV[n-2]
    l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)  # VEC2[n-2]
    acc = mfma_cdna4(p_dot, v_dot, acc)                                  # dot_pv[n-2]
    m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)  # VEC1[n-1] -> m_new, p[n-1]
    acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # rescale for tile n-1

    # output tile n-1 (final; no further dot_qk / prefetch)
    cdna4_async.wait_group(0)                                           # V[n-1] complete
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm1), v_dot_layout)                    # LRV[n-1]
    l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)  # VEC2[n-1]
    acc = mfma_cdna4(p_dot, v_dot, acc)                                  # dot_pv[n-1]


    m_i = m_run
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]

    o_base  = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs  = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_mask  = offs_m[:, None] < N_CTX
    # Downcast first, then convert the layout: `convert_layout` out of the mma layout goes
    # through LDS, and doing it in bf16 halves the bytes that round trip.
    acc_out = acc.to(Out.dtype.element_ty)
    acc_blocked = gl.convert_layout(acc_out, blocked_layout)
    gl.store(o_ptrs, acc_blocked, mask=o_mask)

    l_ptrs = L + off_z * HQ * N_CTX + off_h_q * N_CTX + offs_m_lse
    l_mask = offs_m_lse < N_CTX
    lse = m_i / 1.44269504089 + gl.log2(l_i) / 1.44269504089
    lse_blocked = gl.convert_layout(lse, offs_m_lse_layout)
    gl.store(l_ptrs, lse_blocked, mask=l_mask)


# ---------------------------------------------------------------------------
# Metadata / input helpers (adapted from flash_attention.py)
# ---------------------------------------------------------------------------

LAST_LSE = None   # the last launch's log-sum-exp, for tests


def run_gluon_attention(q, k, v, o, metadata: MetaData, scale_on_q: bool = True):
    """Run gluon_attn_fwd on the given inputs and write output into o.

    Simplified tutorial kernel: non-causal self-attention (Q and K share one N_CTX),
    N_CTX must be a multiple of BLOCK_N (64), and the head dim must be a power of
    two (used directly as BLOCK_DMODEL, no padding). All hold for the tutorial.
    """
    if metadata.causal:
        assert metadata.max_seqlens_q % 256 == 0, "causal path needs seqlen % BLOCK_M (256) == 0"
    assert metadata.max_seqlens_k % 64 == 0, "K seqlen must be a multiple of BLOCK_N (64)"
    assert metadata.max_seqlens_q == metadata.max_seqlens_k, "combined N_CTX requires Q seqlen == K seqlen"
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
    assert head_size & (head_size - 1) == 0, "head dim must be a power of two"
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, metadata)

    M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)

    def grid(META):
        return (nheads_q, triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']), batch)

    global LAST_LSE
    LAST_LSE = M
    gluon_attn_fwd[grid](
        q, k, v, metadata.sm_scale, M, o,
        *q_strides, *k_strides, *v_strides, *o_strides,
        HQ=nheads_q, HK=nheads_k,
        N_CTX=metadata.max_seqlens_q,
        IS_CAUSAL=bool(metadata.causal),
        BLOCK_DMODEL=head_size,
        # Where qk_scale is applied: on Q before the loop (the default) or per
        # element inside VEC1.
        SCALE_ON_Q=scale_on_q,
    )
