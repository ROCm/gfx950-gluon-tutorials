"""
FAv4: CDNA4 (gfx950) Flash Attention forward kernel with LAZY softmax rescaling.

This is fav3.py with one algorithmic change: the online-softmax accumulator
correction (``acc *= alpha``) is applied *lazily*. Softmax is shift-invariant --
subtracting the running max only exists to keep the exp2 argument bounded -- so
the running max is allowed to LAG: it is only bumped (and the accumulator
rescaled) when a tile's max exceeds the running max by more than
``LAZY_RESCALE_THRESHOLD`` (in log2 units; 8 -> a 2**8 = 256x safety margin,
trivially within fp32's dynamic range). When the max is stable the correction is
skipped, so p = exp2(score - m_lag) can rise up to ~256 but the final acc / l_i
(both carried in the same lagging frame) is unchanged. Same idea as ROCm/FlyDSL
`dualwave_swp_lazy_rescale` and Flash-Attention-4's deferred/threshold correction.

NOTE: this first cut gates the max update with a per-row ``gl.where`` (branchless):
the numerics are the lazy ones (validated for correctness), but the acc multiply
still executes with alpha=1 on skipped rows. Turning that into a uniform branch
that actually elides the multiply's VALU (the real perf win) is a follow-up that
must co-design with the llir interleave -- hence: test correctness with the llir
scheduler DISABLED first.

This file implements a CDNA4 (gfx950) Flash Attention forward kernel.

It contains the Gluon kernel, its single autotune config, and the host launcher
(``run_gluon_attention``); correctness and benchmarking live in ``bench.py``. This
tutorial copy is simplified to the single most-performant path: non-causal, head
dim 128, K length a multiple of ``BLOCK_N`` (64).

This kernel matches the pipeline architecture of the kernel from the “FAV3 Unmatched” series, but translated to Gluon instead of Triton.

To get close to matching the LLIR from the original kernel in its docker container with modified Triton and LLVM, we can apply these environment variables:

(Note that comparison and measurement used b1 d128 hq64 sq16384 causal0, BLOCK_M=256 BLOCK_N=64 NUM_STAGES=4 num_warps=8):

- ``DISABLE_LLVM_OPT=disable-vector-combine`` -- needed to match the LLIR, but also better performance in some environments.
- ``AMDGCN_SCALARIZE_PACKED_FOPS=1`` -- emits one scalar fp op per element instead of packed ``v2f16`` ops, so the LLIR/AMDGCN reads op-per-element and diffs cleanly against the docker environment dumps.  Readability/IR-match only; no perf effect.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language._layouts import (
    DotOperandLayout,
    DistributedLinearLayout,
    PaddedSharedLayout,
)


from f16_fa_gfx950_common import (
    compute_dot1_qk,
    compute_dot2_pv,
    compute_softmax,
    get_shape_from_layout,
    get_strides_from_layout,
    MetaData,
    nan_propagating_max,
    remap_xcd,
)


# ---------------------------------------------------------------------------
# Logical sub-cluster primitives for the matched rotated 4-cluster loop.
#
# The hot loop is composed from eight named logical sub-clusters:
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


@gluon.jit
def sc_vec1(qk, m_run, qk_scale: gl.constexpr):
    """VEC1: softmax numerator -- new row-max + exp2 burst (DOT2 cluster), LAZY.

    From the qk scores produced by DOT1 this iteration, computes the (lazily
    updated) running max and the unnormalized probabilities p = exp2(qk*scale -
    m_new) and rescale factor alpha = exp2(m_run - m_new), carried to the next
    iteration (consumed by VEC2 and DOT2).

    LAZY rescale: instead of always bumping the max to max(m_run, rowmax(qk)),
    the max is only advanced for rows whose tile-max exceeds the running max by
    more than LAZY_RESCALE_THRESHOLD. For a "stable" row m_new == m_run, so
    alpha = exp2(0) = 1 and VEC2's ``acc *= alpha`` is a no-op -- the correction
    is deferred. p = exp2(qk*scale - m_run) can then exceed 1 (up to ~2**thr) but
    acc and l_i stay in the same lagging frame, so acc / l_i is unchanged. Softmax
    shift-invariance makes this exact; the threshold only bounds the exp2 range.
    """
    m_ij = nan_propagating_max(qk, axis=1) * qk_scale
    m_cand = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
    # Per-row lazy gate: advance the max only where the jump exceeds the threshold;
    # otherwise keep the (stale) running max so the rescale is skipped. Branchless
    # for the correctness-first pass -- gl.where keeps control flow uniform so the
    # warp pipeline is undisturbed; a real uniform branch (to actually elide the
    # acc multiply) is the perf follow-up co-designed with the llir interleave.
    m_new = gl.where(m_cand - m_run > LAZY_RESCALE_THRESHOLD, m_cand, m_run)
    # Fuse qk*scale - m_new at the source (gl.fma -> single llvm.fmuladd) instead
    # of leaving it as fmul+fsub. The backend only contracts those into v_pk_fma
    # AFTER llirSched runs, so an un-fused pair is double-counted by the interleave
    # weight (2+2 vs the real 2), making its co-exec groups come out half full.
    p = gl.exp2(gl.fma(qk, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def _rescale(acc, l_i, alpha):
    """warp_predicate body: apply the deferred per-row correction.

    Runs only for wavefronts holding at least one row with alpha < 1 (see
    sc_vec2). Rows that did not advance carry alpha == 1, so their multiply is a
    no-op -- correct even though the whole lane executes under one exec mask.
    """
    return acc * alpha[:, None], l_i * alpha


@gluon.jit
def rescale_lazy(acc, l_i, alpha):
    """opt3: the deferred per-row correction, hoisted into the PRIOR mem stage.

    Same warp_predicate rescale as before (acc *= alpha, l_i *= alpha, skipped
    per-wave when no row advanced), but pulled OUT of sc_vec2 so it runs in the
    mem2 stage -- paired with the LRK/ACV loads -- instead of sitting ahead of the
    DOT1 QK mfma. alpha was produced by VEC1 in THIS tile's DOT2; applying the
    correction here (before the NEXT tile's DOT2 accumulates) keeps acc in the
    advanced frame while overlapping the branch/control with LDS/global latency.

    PER-WAVE LAZY SKIP: ``gl.warp_predicate`` keyed on ``alpha < 1`` lowers to
    ``s_and_saveexec`` + ``s_cbranch_execz``, so each wavefront independently skips
    the rescale when none of ITS rows advanced -- no cross-warp reduction. Rows
    that did not advance carry alpha == 1, so their multiply is a no-op.
    """
    need = alpha < 1.0
    acc, l_i = gl.warp_predicate(need, (acc, l_i), _rescale, args=(alpha, ))
    return acc, l_i


@gluon.jit
def sc_vec2(l_i, p, p_dot_layout: gl.constexpr, out_dtype: gl.constexpr):
    """VEC2: softmax denominator + DOT2-operand convert (DOT1 cluster).

    opt3: the accumulator/denominator rescale (acc *= alpha, l_i *= alpha) has been
    hoisted OUT into rescale_lazy() in the PRIOR mem2 stage, leaving only the
    unconditional denominator update (l_i += sum p) and the p->fp16 convert that
    prepares the DOT2 operand. p was produced by VEC1 in the previous iteration and
    l_i's frame was already corrected by the preceding rescale_lazy, so the split
    ``l_i * alpha`` (in rescale_lazy) ``+ l_ij`` (here) is exact; for stable rows
    alpha == 1. This leaves the DOT1 stage as just [sum + cvt] + the QK mfma, with
    no branch/control ahead of the mfma.
    """
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
    # llvm_fn_attrs amdgpu-agpr-alloc="0,0" forces 0 AGPRs (VGPR-only). By default
    # the backend parks accumulators in AGPRs and shuffles them with v_accvgpr moves;
    # with the 2x-unrolled loop that shuffling lands on the critical path and costs
    # ~50 TFLOPS (~803 VGPR-only vs ~754 with AGPRs), so VGPR-only is required here,
    # not just cosmetic. Tuple form required: the string form splits "0,0" on its comma.
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
                   BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr):
    """
    Gluon Flash Attention Forward Kernel (AMD CDNA4 / gfx950).
    Grid: (num_heads_q, num_m_blocks, batch)
    """
    num_warps: gl.constexpr = gl.num_warps()

    off_h_q = gl.program_id(0)
    off_h_q = remap_xcd(off_h_q, HQ)
    start_m  = gl.program_id(1)
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

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1], order=[1, 0])

    offs_m_layout:    gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout:    gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_m_layout:     gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    offs_m    = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d    = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)

    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0])
    q_smem = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_M, BLOCK_DMODEL], layout=q_smem_layout)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)
    q_smem.store(q)
    q_dot = q_smem.load(q_dot_layout)

    m_i  = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i  = gl.full([BLOCK_M], 1.0,           dtype=gl.float32, layout=mma_m_layout)
    acc  = gl.zeros([BLOCK_M, BLOCK_DMODEL],  dtype=gl.float32, layout=mma_layout)

    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    # Simplified tutorial kernel: non-causal, K length a multiple of BLOCK_N, so
    # every K/V block is full and unmasked (no causal / no ragged-tail masking).
    n_blocks = (N_CTX + BLOCK_N - 1) // BLOCK_N


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
    # must contain an even number of tiles. Rather than emit a runtime odd-tail
    # tile, require it statically. (n_blocks-3) even <=> n_blocks odd <=>
    # N_CTX == BLOCK_N * (odd). e.g. seqlen 16320/16448 are OK; 16384
    # (n_blocks=256) is NOT. Computed from constexpr N_CTX/BLOCK_N (block_end
    # itself is a runtime value here).
    NUM_BLOCKS: gl.constexpr = (N_CTX + BLOCK_N - 1) // BLOCK_N
    gl.static_assert(
        (NUM_BLOCKS - 3) % 2 == 0,
        "N_CTX must give an odd n_blocks = (N_CTX + BLOCK_N - 1)//BLOCK_N so the "
        "2x-unrolled main loop needs no odd-tail tile",
    )

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

    cdna4_async.wait_group(2)                                       # K[0] complete
    kt0 = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)                    # LRK[0] -> K regs tile 0
    qk = compute_dot1_qk(q_dot, kt0, BLOCK_M, BLOCK_N, mma_layout)  # dot_qk[0] -> qk[0]
    m_run, p_c, alpha_c = sc_vec1(qk, m_i, qk_scale)   # VEC1[0] -> m_new[0], p[0], alpha[0]=0

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
    # opt3: rescale for tile 0's DOT2, using the prologue's alpha_c[0]. The loop's
    # mem2 stages carry the rescale for every later tile, so the DOT1 sc_vec2 is
    # left as just sum+cvt with no branch ahead of the QK mfma.
    acc, l_i = rescale_lazy(acc, l_i, alpha_c)

    main_loop_pairs = (block_end - 3 - block_start) // 2
    for pair_idx in tl.range(0, main_loop_pairs):
        block_n = block_start + pair_idx * 2

        # even tile (block_n): LDS slots cur=0, next=1
        with warp_pipeline_stage("dot1"):
            l_i, p_dot = sc_vec2(l_i, p_c, p_dot_layout, q_dot.dtype)   # VEC2 (sum+cvt; rescale in prior mem2)
            qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)   # dot_qk
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)   # LRV
            cdna4_async.buffer_load_to_shared(kt_smem.index(1), k_base + (block_n + 3) * kt_step, kt_off)   # ACK
            cdna4_async.commit_group()
        with warp_pipeline_stage("dot2"):
            acc = mfma_cdna4(p_dot, v_dot, acc)   # dot_pv
            m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale)   # VEC1
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)   # LRK
            cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # opt3: rescale for next DOT1, overlapped w/ mem latency

        # odd tile (block_n+1): LDS slots cur=1, next=0
        with warp_pipeline_stage("dot1"):
            l_i, p_dot = sc_vec2(l_i, p_c, p_dot_layout, q_dot.dtype)   # VEC2 (sum+cvt; rescale in prior mem2)
            qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)   # dot_qk
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem1"):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)   # LRV
            cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_base + (block_n + 4) * kt_step, kt_off)   # ACK
            cdna4_async.commit_group()
        with warp_pipeline_stage("dot2"):
            acc = mfma_cdna4(p_dot, v_dot, acc)   # dot_pv
            m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale)   # VEC1
        cdna4_async.wait_group(WAIT_LOOP - 1)
        with warp_pipeline_stage("mem2"):
            kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(1), kt_dot_layout)   # LRK
            cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + (block_n + 3) * v_step, v_off)   # ACV
            cdna4_async.commit_group()
            acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # opt3: rescale for next DOT1, overlapped w/ mem latency

    # (odd-tail tile removed; the static_assert above guarantees it is never needed)

    # -- Drain (last 3 output tiles, no OOB global prefetch) ---------------
    # After the loop: outputs [.., n-4] done; K[0..n-1] and V[0..n-2] in LDS
    # (V[n-1] still to load); carried kt_dot=K regs tile n-2, m_run=m_new[n-3],
    # p_c=p[n-3], alpha_c=alpha[n-3]; pending async {V[n-3],K[n-1],V[n-2]}.
    nm3 = block_end - 3
    nm2 = block_end - 2
    nm1 = block_end - 1
    s_nm3 = ((nm3 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm2 = ((nm2 - block_start) % BUF_DEPTH).to(tl.int32)
    s_nm1 = ((nm1 - block_start) % BUF_DEPTH).to(tl.int32)

    # output tile n-3 (also issues the final V prefetch, ACV[n-1])
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)   # dot_qk[n-2]
    cdna4_async.wait_group(2)                                           # V[n-3] complete
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm3), v_dot_layout)                    # LRV[n-3]
    l_i, p_dot = sc_vec2(l_i, p_c, p_dot_layout, q_dot.dtype)  # VEC2[n-3] (rescale done in loop's last mem2)
    acc = mfma_cdna4(p_dot, v_dot, acc)                                  # dot_pv[n-3]
    m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale)  # VEC1[n-2] -> m_new, p[n-2]
    acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # opt3: rescale for tile n-2
    gl.barrier()                                                       # WAR: LRV[n-3] vs V[n-1] write
    cdna4_async.buffer_load_to_shared(v_smem.index(s_nm1), v_base + nm1 * v_step, v_off)
    cdna4_async.commit_group()   # ACV[n-1]
    cdna4_async.wait_group(2)                                           # K[n-1] complete
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(s_nm1), kt_dot_layout)                 # LRK[n-1] -> K regs tile n-1

    # output tile n-2
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)   # dot_qk[n-1]
    cdna4_async.wait_group(1)                                           # V[n-2] complete
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm2), v_dot_layout)                    # LRV[n-2]
    l_i, p_dot = sc_vec2(l_i, p_c, p_dot_layout, q_dot.dtype)  # VEC2[n-2]
    acc = mfma_cdna4(p_dot, v_dot, acc)                                  # dot_pv[n-2]
    m_run, p_c, alpha_c = sc_vec1(qk, m_run, qk_scale)  # VEC1[n-1] -> m_new, p[n-1]
    acc, l_i = rescale_lazy(acc, l_i, alpha_c)   # opt3: rescale for tile n-1

    # output tile n-1 (final; no further dot_qk / prefetch)
    cdna4_async.wait_group(0)                                           # V[n-1] complete
    v_dot = cdna4_async.load_shared_relaxed(v_smem.index(s_nm1), v_dot_layout)                    # LRV[n-1]
    l_i, p_dot = sc_vec2(l_i, p_c, p_dot_layout, q_dot.dtype)  # VEC2[n-1]
    acc = mfma_cdna4(p_dot, v_dot, acc)                                  # dot_pv[n-1]


    m_i = m_run
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]

    o_base  = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs  = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_mask  = offs_m[:, None] < N_CTX
    acc_blocked = gl.convert_layout(acc, blocked_layout)
    gl.store(o_ptrs, acc_blocked.to(Out.dtype.element_ty), mask=o_mask)

    l_ptrs = L + off_z * HQ * N_CTX + off_h_q * N_CTX + offs_m
    l_mask = offs_m < N_CTX
    lse = m_i / 1.44269504089 + gl.log2(l_i) / 1.44269504089
    lse_blocked = gl.convert_layout(lse, offs_m_layout)
    gl.store(l_ptrs, lse_blocked, mask=l_mask)


# ---------------------------------------------------------------------------
# Metadata / input helpers (adapted from flash_attention.py)
# ---------------------------------------------------------------------------

def run_gluon_attention(q, k, v, o, metadata: MetaData):
    """Run gluon_attn_fwd on the given inputs and write output into o.

    Simplified tutorial kernel: non-causal self-attention (Q and K share one N_CTX),
    N_CTX must be a multiple of BLOCK_N (64), and the head dim must be a power of
    two (used directly as BLOCK_DMODEL, no padding). All hold for the tutorial.
    """
    assert not metadata.causal, "simplified FAV3 tutorial kernel supports non-causal only"
    assert metadata.max_seqlens_k % 64 == 0, "K seqlen must be a multiple of BLOCK_N (64)"
    assert metadata.max_seqlens_q == metadata.max_seqlens_k, "combined N_CTX requires Q seqlen == K seqlen"
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
    assert head_size & (head_size - 1) == 0, "head dim must be a power of two"
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, metadata)

    M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)

    def grid(META):
        return (nheads_q, triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']), batch)

    gluon_attn_fwd[grid](
        q, k, v, metadata.sm_scale, M, o,
        *q_strides, *k_strides, *v_strides, *o_strides,
        HQ=nheads_q, HK=nheads_k,
        N_CTX=metadata.max_seqlens_q,
        IS_CAUSAL=False,
        BLOCK_DMODEL=head_size,
    )
