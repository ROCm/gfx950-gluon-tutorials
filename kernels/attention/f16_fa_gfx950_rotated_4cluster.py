"""
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
from triton.experimental.gluon.language._layouts import (
    DotOperandLayout,
    DistributedLinearLayout,
    PaddedSharedLayout,
)


from f16_fa_gfx950_common import (
    compute_dot1_qk,
    compute_dot2_pv,
    compute_softmax,
    do_mma,
    get_shape_from_layout,
    get_strides_from_layout,
    issue_async_load_k,
    issue_async_load_v,
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

@gluon.jit
def sc_vec1(qk, m_run, start_n, start_m,
            qk_scale: gl.constexpr,
            MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
            N_CTX: gl.constexpr,
            BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
            mma_layout: gl.constexpr,
            mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr):
    """VEC1: softmax numerator -- (mask +) new row-max + exp2 burst (DOT2 cluster).

    From the qk scores produced by DOT1 this iteration, computes the new running
    max m_new = max(m_run, rowmax(qk)*scale), the unnormalized probabilities
    p = exp2(qk*scale - m_new), and the rescale factor alpha = exp2(m_run - m_new).
    p and alpha are carried to the next iteration (consumed by VEC2 and DOT2).
    This is the expensive transcendental group, paired with the P*V MFMA so the
    exp throughput overlaps the matrix engine.

    When MASK_STEPS the scores are scaled and masked (causal + K-bound) before
    the max/exp2 (mirroring ``compute_softmax``). The unmasked branch keeps the
    FMA-friendly form (scale folded into the max and exp2 inputs); MASK_STEPS is
    constexpr so that branch is identical to the unmasked-only schedule after DCE.
    """
    if MASK_STEPS:
        qk_sm = qk * qk_scale
        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            causal_boundary = causal_offs_n[None, :]
            causal_mask = causal_offs_m[:, None] >= causal_boundary
            qk_sm = gl.where(causal_mask, qk_sm, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                         dtype=gl.float32, layout=mma_layout))
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        bound_mask = bound_offs[None, :] < N_CTX
        qk_sm = gl.where(bound_mask, qk_sm, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                    dtype=gl.float32, layout=mma_layout))
        m_ij = nan_propagating_max(qk_sm, axis=1)
        m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_sm - m_new[:, None])
        alpha = gl.exp2(m_run - m_new)
    else:
        m_ij = nan_propagating_max(qk, axis=1) * qk_scale
        m_new = gl.maximum(m_run, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk * qk_scale - m_new[:, None])
        alpha = gl.exp2(m_run - m_new)
    return m_new, p, alpha


@gluon.jit
def sc_vec2(acc, l_i, p, alpha, p_dot_layout: gl.constexpr, out_dtype: gl.constexpr):
    """VEC2: softmax denominator + accumulator correction (DOT1 cluster).

    Updates the running denominator (l_i = l_i*alpha + sum p) with one cross-lane
    sum reduction, rescales the accumulator (acc *= alpha), and casts p to fp16
    with the layout convert that prepares the operand for the immediately-
    following DOT2. p and alpha were produced by VEC1 in the *previous* iteration
    (the carried previous-tile probabilities).

    Op order matches the reference VEC2 split exactly: row-sum first (l_ij), then the
    accumulator rescale (acc *= alpha), then the running-denominator update
    (l_i = l_i*alpha + l_ij), then the p->fp16 cast.
    """
    l_ij = gl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    p_dot = gl.convert_layout(p.to(out_dtype), p_dot_layout)
    return acc, l_i, p_dot


@gluon.jit
def sc_dot_pv(acc, p_dot, v_dot):
    """dot_pv: P @ V -> acc (p already cast in VEC2, V already in registers)."""
    return do_mma("mfma_cdna4", p_dot, v_dot, acc)


@gluon.jit
def sc_lr(smem_slot, dot_layout: gl.constexpr):
    """LRK / LRV: local-read a tile from LDS into registers."""
    return cdna4_async.load_shared_relaxed(smem_slot, dot_layout)


# ---------------------------------------------------------------------------
# Autotune configs
# ---------------------------------------------------------------------------

def get_gluon_cdna_autotune_configs():
    # Simplified tutorial baseline: the single most performant config for the
    # focus shape (D=128, non-causal). Full autotune space is in git history.
    return [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_warps=8),
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
    mma_offs_n_col:   gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row:   gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
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
    b0 = block_start
    issue_async_load_k(kt_smem.index(0), k_base, (b0 + 0) * BLOCK_N,
                       stride_kn, stride_kk, N_CTX, False, N_CTX, False,
                       BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, kt_async_layout)  # ACK[0]
    issue_async_load_v(v_smem.index(0), v_base, (b0 + 0) * BLOCK_N,
                       stride_vk, stride_vn, N_CTX, False, N_CTX, False,
                       BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, v_async_layout)   # ACV[0]
    issue_async_load_k(kt_smem.index(1), k_base, (b0 + 1) * BLOCK_N,
                       stride_kn, stride_kk, N_CTX, False, N_CTX, False,
                       BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, kt_async_layout)  # ACK[1]

    n0 = 0
    cdna4_async.wait_group(2)                                       # K[0] complete
    kt0 = sc_lr(kt_smem.index(0), kt_dot_layout)                    # LRK[0] -> K regs tile 0
    qk = compute_dot1_qk(q_dot, kt0, BLOCK_M, BLOCK_N, mma_layout)  # dot_qk[0] -> qk[0]
    m_run, p_c, alpha_c = sc_vec1(qk, m_i, n0, start_m, qk_scale,   # VEC1[0] -> m_new[0], p[0], alpha[0]=0
                                  False, IS_CAUSAL, N_CTX,
                                  BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row)

    gl.barrier()                                                   # WAR: LRK[0] ds_read vs K[2] write
    issue_async_load_k(kt_smem.index(0), k_base, (b0 + 2) * BLOCK_N,
                       stride_kn, stride_kk, N_CTX, False, N_CTX, False,
                       BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, kt_async_layout)  # ACK[2] (slot0 reuse)
    cdna4_async.wait_group(1)                                       # K[1] complete
    kt_dot = sc_lr(kt_smem.index(1), kt_dot_layout)                 # LRK[1] -> K regs tile 1
    issue_async_load_v(v_smem.index(1), v_base, (b0 + 1) * BLOCK_N,
                       stride_vk, stride_vn, N_CTX, False, N_CTX, False,
                       BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, v_async_layout)   # ACV[1]

    # -- Main loop (full rotated body) -------------------------------------
    # Runs output tiles [block_start, block_end-3): the last full iteration
    # whose K prefetch (ACK[i+3]) is still in bounds. The final three tiles are
    # drained below without out-of-bounds global prefetch.
    for block_n in tl.range(block_start, block_end - 3):
        cur_slot = ((block_n - block_start) % BUF_DEPTH).to(tl.int32)
        nxt_slot = ((block_n + 1 - block_start) % BUF_DEPTH).to(tl.int32)
        ack_n = ((block_n + 3) * BLOCK_N).to(tl.int32)
        acv_n = ((block_n + 2) * BLOCK_N).to(tl.int32)
        ahead_n = 0

        # cluster 0 DOT1: dot_qk[i+1] (s1) then VEC2[i] (s0). dot_qk leads so
        # iglp.opt prefixes the MFMA.
        with warp_pipeline_stage("dot1", priority=0):
            qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)  # dot_qk s1 -> qk[i+1]
            acc, l_i, p_dot = sc_vec2(acc, l_i, p_c, alpha_c,                 # VEC2 s0 (sum + acc rescale
                                      p_dot_layout, q_dot.dtype)              #         + l_i + p->fp16 cast)

        cdna4_async.wait_group(WAIT_LOOP - 1)  # V[i] complete (for LRV[i]); -1: backend vmcnt workaround

        # cluster 1 MEM1: LRV[i] (stage 0) then ACK[i+3] (stage 3).
        with warp_pipeline_stage("mem1", priority=1):
            v_dot = sc_lr(v_smem.index(cur_slot), v_dot_layout)               # LRV   s0
            issue_async_load_k(kt_smem.index(nxt_slot), k_base, ack_n,
                               stride_kn, stride_kk, N_CTX, False, N_CTX, False,
                               BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, kt_async_layout)  # ACK s3

        # cluster 2 DOT2: dot_pv[i] (s0) then VEC1[i+1] (s1: new max + exp2
        # burst).  The exp2 lands AFTER the P*V MFMA and feeds the next iteration's dot_pv.
        with warp_pipeline_stage("dot2", priority=0):
            acc = sc_dot_pv(acc, p_dot, v_dot)                                # dot_pv s0
            m_run, p_c, alpha_c = sc_vec1(qk, m_run, ahead_n, start_m, qk_scale,  # VEC1 s1 -> m_new, p[i+1]
                                          False, IS_CAUSAL, N_CTX,
                                          BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row)

        cdna4_async.wait_group(WAIT_LOOP - 1)  # K[i+2] complete (for LRK[i+2]); -1: backend vmcnt workaround

        # cluster 3 MEM2: LRK[i+2] (stage 2) then ACV[i+2] (stage 2).
        with warp_pipeline_stage("mem2", priority=1):
            kt_dot = sc_lr(kt_smem.index(cur_slot), kt_dot_layout)            # LRK   s2 -> K regs tile i+2
            issue_async_load_v(v_smem.index(cur_slot), v_base, acv_n,
                               stride_vk, stride_vn, N_CTX, False, N_CTX, False,
                               BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, v_async_layout)   # ACV s2

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
    nm2_n = 0
    nm1_n = 0

    # output tile n-3 (also issues the final V prefetch, ACV[n-1])
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)   # dot_qk[n-2]
    cdna4_async.wait_group(2)                                           # V[n-3] complete
    v_dot = sc_lr(v_smem.index(s_nm3), v_dot_layout)                    # LRV[n-3]
    acc, l_i, p_dot = sc_vec2(acc, l_i, p_c, alpha_c, p_dot_layout, q_dot.dtype)  # VEC2[n-3]
    acc = sc_dot_pv(acc, p_dot, v_dot)                                  # dot_pv[n-3]
    m_run, p_c, alpha_c = sc_vec1(qk, m_run, nm2_n, start_m, qk_scale,  # VEC1[n-2] -> m_new, p[n-2]
                                  False, IS_CAUSAL, N_CTX,
                                  BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row)
    gl.barrier()                                                       # WAR: LRV[n-3] vs V[n-1] write
    issue_async_load_v(v_smem.index(s_nm1), v_base, (nm1 * BLOCK_N).to(tl.int32),
                       stride_vk, stride_vn, N_CTX, False, N_CTX, False,
                       BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL, v_async_layout)   # ACV[n-1]
    cdna4_async.wait_group(2)                                           # K[n-1] complete
    kt_dot = sc_lr(kt_smem.index(s_nm1), kt_dot_layout)                 # LRK[n-1] -> K regs tile n-1

    # output tile n-2
    qk = compute_dot1_qk(q_dot, kt_dot, BLOCK_M, BLOCK_N, mma_layout)   # dot_qk[n-1]
    cdna4_async.wait_group(1)                                           # V[n-2] complete
    v_dot = sc_lr(v_smem.index(s_nm2), v_dot_layout)                    # LRV[n-2]
    acc, l_i, p_dot = sc_vec2(acc, l_i, p_c, alpha_c, p_dot_layout, q_dot.dtype)  # VEC2[n-2]
    acc = sc_dot_pv(acc, p_dot, v_dot)                                  # dot_pv[n-2]
    m_run, p_c, alpha_c = sc_vec1(qk, m_run, nm1_n, start_m, qk_scale,  # VEC1[n-1] -> m_new, p[n-1]
                                  False, IS_CAUSAL, N_CTX,
                                  BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row)

    # output tile n-1 (final; no further dot_qk / prefetch)
    cdna4_async.wait_group(0)                                           # V[n-1] complete
    v_dot = sc_lr(v_smem.index(s_nm1), v_dot_layout)                    # LRV[n-1]
    acc, l_i, p_dot = sc_vec2(acc, l_i, p_c, alpha_c, p_dot_layout, q_dot.dtype)  # VEC2[n-1]
    acc = sc_dot_pv(acc, p_dot, v_dot)                                  # dot_pv[n-1]


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
