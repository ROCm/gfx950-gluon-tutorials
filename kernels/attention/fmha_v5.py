"""
FMHA v5: fmha_v4's pipeline run persistently, with causal masking.

The rotated 4-cluster loop, the lazy rescale and the softmax split are fmha_v4's,
unchanged -- the hot loop of a non-causal run compiles to the same body. What is new is
everything around the loop, because causal attention changes the shape of the work: a
256-row q-block m attends to (m + 1) * 256 keys, so blocks range from 4 to 128 K/V tiles,
and only the last four tiles of each block touch the diagonal.

Persistent, in balanced pairs. Launched as an ordinary grid, those uneven workgroups
leave CUs idle at every handover (~10 us, ~6% of all CU time at the tutorial shape), which
non-causal's uniform workgroups never see. Here one workgroup per CU walks a fixed list of
q-blocks, in pairs (M-1-j, j) whose tiles add up to the same (M + 1) * 4, so every
workgroup ends together. Jobs are numbered (batch, head)-major and the workgroups sharing an
XCD take consecutive jobs, so an XCD's L2 serves one or two (batch, head) at a time.

Overlapped hand-over. A persistent workgroup can prefetch the next q-block while it
finishes this one: Q straight into the MFMA operand registers (the current Q is dead after
the last QK) and the first K/V tiles into the ring. O leaves through LDS in two explicit
32 KB halves -- small enough that the size-sorted LDS allocator still puts the K/V ring at
the bottom, where its ds_read offsets fit 16 bits -- and the LSE straight from the row
layout. Per q-block the hand-over then costs ~3 us, the same as non-causal pays.

The mask (triton-tickets#812). The four diagonal tiles' VEC1s are the last three loop
iterations and the drain's. The loop is split in two warp-pipelined loops -- plain pairs,
then the masked pairs -- because a mask branch inside a single loop splits the PV cluster's
block, IGroupLP loses its interleave there, and that costs ~5% on every tile. The mask itself
is the issue's 3-state scheme on the MFMA layout (fmha_causal_mask.py): per wave each 32x32
block is fully kept, fully masked, or the diagonal block, whose per-register keep-test is a
compare against a compile-time column. In the loop that is a branch-free v_cmp/v_cndmask
pair per register with no extra VGPRs; in plain code the per-wave states branch. Together
the masks cost ~0.6% of the run. In the drain, waves whose rows are entirely masked skip
the QK MFMA (gl.amd.warp_id, when the Triton build has it).

Everything lives in registers at the edge: the kernel is at 256 VGPRs, and several choices
below exist only to keep it from spilling (see the comments at each).
"""

import os
import sys

if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

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

from common import get_shape_from_layout, get_strides_from_layout, MetaData
from fmha_v4 import _split_halves, rescale_lazy, sc_vec1, sc_vec2
from fmha_v4_causal import causal_mask

# Waves whose rows the diagonal masks out entirely skip the drain's QK MFMA. That needs a
# per-wave branch, i.e. gl.amd.warp_id(); without it (or with FA_SKIP_MASKED_WAVES=0) every
# wave computes the tile and the mask discards it.
SKIP_MASKED_WAVES = tl.constexpr(hasattr(gl.amd, "warp_id")
                                 and os.environ.get("FA_SKIP_MASKED_WAVES", "1") == "1")

# Running-max start value. Finite, so that a row whose first tile is fully masked keeps
# p = 0 and alpha = 1 instead of exp2(-inf - -inf) = NaN (it cannot happen when a block's
# first tile is tile 0, but costs nothing). The first real tile still moves it -- the jump
# exceeds the lazy-rescale threshold -- and its alpha = exp2(-1e30 - m) = 0 clears acc and l.
M_INIT = tl.constexpr(-1.0e30)

# FA_WG_TIMING=1 records s_memrealtime at NSLOT points of every q-block (see
# scripts/fa_qblock_timing.py). The stamps perturb the kernel a little: read them as a
# breakdown, and time with rocprof.
NSLOT = tl.constexpr(8)


@gluon.jit
def _stamp(T, slot, layout: gl.constexpr):
    """Instrumentation: store [realtime, HW_ID, XCC_ID] at T[slot * 4 ..]."""
    z = gl.zeros([4], gl.int32, layout)
    t = gl.inline_asm_elementwise("s_memrealtime $0\ns_waitcnt lgkmcnt(0)", "=s,v", [z],
                                  dtype=gl.int64, is_pure=False, pack=1)
    hw = gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s,v", [z],
                                   dtype=gl.int32, is_pure=False, pack=1)
    xcc = gl.inline_asm_elementwise("s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s,v", [z],
                                    dtype=gl.int32, is_pure=False, pack=1)
    i = gl.arange(0, 4, layout)
    v = gl.where(i == 0, t, gl.where(i == 1, hw.to(gl.int64), xcc.to(gl.int64)))
    gl.store(T + slot * 4 + i, v, mask=i < 3)


@gluon.jit
def _delta(i, n, BLOCK_N: gl.constexpr):
    """Causal-mask delta of an n-tile q-block's tile i: its first key column minus the
    block's first query row. 0 for the first diagonal tile (n-4), 3 * BLOCK_N for the last."""
    return (i - n + 4) * BLOCK_N


@gluon.jit
def _pipe_pair(q_dot, kt_dot, acc, l_i, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c,
               kt_smem, v_smem, k_base, v_base, kt_off, v_off, kt_step, v_step,
               block_n, block_end,
               MASKED: gl.constexpr,
               mma_layout: gl.constexpr, p_dot_layout: gl.constexpr,
               v_dot_layout: gl.constexpr, kt_dot_layout: gl.constexpr,
               qk_scale: gl.constexpr, SCALE_ON_Q: gl.constexpr,
               BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, WAIT: gl.constexpr):
    """Two iterations (tiles block_n, block_n+1) of fmha_v4's rotated 4-cluster loop.

    With MASKED, mem1 masks the score tile that the following dot2's VEC1 consumes (tile
    block_n+1, then block_n+2) -- unconditionally, so the body stays branch-free. The odd
    half's K prefetch is clamped to the last tile: the loop runs one iteration more than
    fmha_v4's, and that one would otherwise fetch tile n.
    """
    # even tile (block_n): LDS slots cur=0, next=1
    with warp_pipeline_stage("dot1"):
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_dot, qk)
        l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)
    cdna4_async.wait_group(WAIT)
    with warp_pipeline_stage("mem1"):
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
        cdna4_async.buffer_load_to_shared(kt_smem.index(1), k_base + (block_n + 3) * kt_step, kt_off)
        cdna4_async.commit_group()
        if MASKED:
            qk = causal_mask(qk, _delta(block_n + 1, block_end, BLOCK_N), BLOCK_M, BLOCK_N, False)
    with warp_pipeline_stage("dot2"):
        acc = mfma_cdna4(p_dot, v_dot, acc)
        m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)
    cdna4_async.wait_group(WAIT)
    with warp_pipeline_stage("mem2"):
        kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
        cdna4_async.buffer_load_to_shared(v_smem.index(0), v_base + (block_n + 2) * v_step, v_off)
        cdna4_async.commit_group()
        acc, l_i = rescale_lazy(acc, l_i, alpha_c)

    # odd tile (block_n+1): LDS slots cur=1, next=0
    with warp_pipeline_stage("dot1"):
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_dot, qk)
        l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)
    cdna4_async.wait_group(WAIT)
    with warp_pipeline_stage("mem1"):
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)
        k_next = gl.minimum(block_n + 4, block_end - 1)
        cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_base + k_next * kt_step, kt_off)
        cdna4_async.commit_group()
        if MASKED:
            qk = causal_mask(qk, _delta(block_n + 2, block_end, BLOCK_N), BLOCK_M, BLOCK_N, False)
    with warp_pipeline_stage("dot2"):
        acc = mfma_cdna4(p_dot, v_dot, acc)
        m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)
    cdna4_async.wait_group(WAIT)
    with warp_pipeline_stage("mem2"):
        kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(1), kt_dot_layout)
        cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + (block_n + 3) * v_step, v_off)
        cdna4_async.commit_group()
        acc, l_i = rescale_lazy(acc, l_i, alpha_c)
    return kt_dot, acc, l_i, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c


@gluon.jit
def _job(t, wg_x, IS_CAUSAL: gl.constexpr, NUM_WG: gl.constexpr, JOBS_PER_BH: gl.constexpr,
         NUM_M: gl.constexpr, NUM_BLOCKS: gl.constexpr, TILES_PER_M: gl.constexpr):
    """(batch * head, q-block, #K/V tiles) of this workgroup's t-th q-block."""
    if IS_CAUSAL:
        job = (t // 2) * NUM_WG + wg_x
        jj = job % JOBS_PER_BH
        short = t % 2   # the long block first, then its short partner
        start_m = jj + (1 - short) * (NUM_M - 1 - 2 * jj)
        n_blocks = (start_m + 1) * TILES_PER_M
    else:
        job = t * NUM_WG + wg_x
        start_m = job % JOBS_PER_BH
        n_blocks = NUM_BLOCKS + start_m * 0
    return job // JOBS_PER_BH, start_m, n_blocks


@gluon.jit
def gluon_attn_fwd(Q, K, V, SM_SCALE: gl.constexpr, L, Out,
                   # The strides are compile-time constants: the kernel is specialized on the
                   # shape anyway, and as runtime values they are 16 SGPRs held through the
                   # whole persistent loop (+0.5% on causal).
                   stride_qz: gl.constexpr, stride_qh: gl.constexpr, stride_qm: gl.constexpr, stride_qk: gl.constexpr,
                   stride_kz: gl.constexpr, stride_kh: gl.constexpr, stride_kn: gl.constexpr, stride_kk: gl.constexpr,
                   stride_vz: gl.constexpr, stride_vh: gl.constexpr, stride_vk: gl.constexpr, stride_vn: gl.constexpr,
                   stride_oz: gl.constexpr, stride_oh: gl.constexpr, stride_om: gl.constexpr, stride_on: gl.constexpr,
                   HQ: gl.constexpr, HK: gl.constexpr, BATCH: gl.constexpr,
                   N_CTX: gl.constexpr,
                   IS_CAUSAL: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr,
                   NUM_WG: gl.constexpr,
                   SCALE_ON_Q: gl.constexpr = True,
                   Timing=None, TIMING: gl.constexpr = False):
    """Persistent FMHA forward. Grid: (NUM_WG,), one workgroup per CU."""
    num_warps: gl.constexpr = gl.num_warps()
    pid = gl.program_id(0)

    mma_layout: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, 16],
                                              transposed=True, warps_per_cta=[num_warps, 1])
    q_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=8)
    kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=8)
    p_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=4)
    v_dot_layout:  gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=4)
    mma_m_layout:  gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    q_m: gl.constexpr = gl.SliceLayout(dim=1, parent=q_dot_layout)
    q_d: gl.constexpr = gl.SliceLayout(dim=0, parent=q_dot_layout)
    if TIMING:
        t_layout: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    # K/V ring: the layouts are fmha_v4's. Allocated once and kept across q-blocks, because
    # the next q-block's first tiles are prefetched into it while this one drains.
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
        offset_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],
                      [0, 16], [0, 32], [0, 1], [0, 2], [0, 4], [0, 8]],
        cga_layout=[],
        shape=[BLOCK_DMODEL, BLOCK_N])
    v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, 32]],
        offset_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
                      [16, 0], [32, 0], [1, 0], [2, 0], [4, 0], [8, 0]],
        cga_layout=[],
        shape=[BLOCK_N, BLOCK_DMODEL])
    BUF_DEPTH: gl.constexpr = 2
    kt_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty, [BUF_DEPTH, BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
    v_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty, [BUF_DEPTH, BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
    kt_ad: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_an: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_off = (gl.arange(0, BLOCK_DMODEL, layout=kt_ad)[:, None] * stride_kk
              + gl.arange(0, BLOCK_N, layout=kt_an)[None, :] * stride_kn)
    v_an: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_ad: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_off = (gl.arange(0, BLOCK_N, layout=v_an)[:, None] * stride_vk
             + gl.arange(0, BLOCK_DMODEL, layout=v_ad)[None, :] * stride_vn)
    kt_step: gl.constexpr = BLOCK_N * stride_kn
    v_step: gl.constexpr = BLOCK_N * stride_vk
    # fmha_v4's loop wait, one async group tighter than the ring needs. Relaxing it races
    # here too (the max error drifts from run to run).
    WAIT: gl.constexpr = 2 * BUF_DEPTH - 3

    # ---- persistent work split ------------------------------------------------------
    NUM_M: gl.constexpr = N_CTX // BLOCK_M
    gl.static_assert(N_CTX % BLOCK_M == 0)
    TILES_PER_M: gl.constexpr = BLOCK_M // BLOCK_N
    NUM_BLOCKS: gl.constexpr = N_CTX // BLOCK_N
    # The loop runs (n - 2) / 2 pairs, so n must be even; causal n = 4 * (m + 1).
    gl.static_assert(NUM_BLOCKS % 2 == 0)
    if IS_CAUSAL:
        gl.static_assert(NUM_M % 2 == 0)
        gl.static_assert(TILES_PER_M == 4)
        JOBS_PER_BH: gl.constexpr = NUM_M // 2
        QB_PER_JOB: gl.constexpr = 2
    else:
        JOBS_PER_BH: gl.constexpr = NUM_M
        QB_PER_JOB: gl.constexpr = 1
    NUM_JOBS: gl.constexpr = BATCH * HQ * JOBS_PER_BH
    gl.static_assert(NUM_WG % 8 == 0)
    # The NUM_WG / 8 workgroups that share an XCD (the hardware places workgroup i on XCD
    # i % 8) take consecutive job numbers.
    wg_x = (pid % 8) * (NUM_WG // 8) + pid // 8
    n_qblocks = (NUM_JOBS - wg_x + NUM_WG - 1) // NUM_WG * QB_PER_JOB
    PER_WG: gl.constexpr = (NUM_JOBS + NUM_WG - 1) // NUM_WG * QB_PER_JOB

    # Every launched workgroup has at least one job (the launcher caps the grid), and
    # saying so keeps LLVM from guarding the persistent loop against zero trips.
    gl.assume(n_qblocks > 0)
    # Prefetch for the first q-block -- exactly what every later one gets from its
    # predecessor: Q straight into the MFMA operand layout, first K/V tiles into the ring.
    bh_n, sm_n, nb_n = _job(0, wg_x, IS_CAUSAL, NUM_WG, JOBS_PER_BH, NUM_M, NUM_BLOCKS,
                            TILES_PER_M)
    q_next = gl.load(Q + (bh_n // HQ) * stride_qz + (bh_n % HQ) * stride_qh
                     + sm_n * BLOCK_M * stride_qm
                     + gl.arange(0, BLOCK_M, layout=q_m)[:, None] * stride_qm
                     + gl.arange(0, BLOCK_DMODEL, layout=q_d)[None, :] * stride_qk)
    k_pre = K + (bh_n // HQ) * stride_kz + ((bh_n % HQ) * HK // HQ) * stride_kh
    v_pre = V + (bh_n // HQ) * stride_vz + ((bh_n % HQ) * HK // HQ) * stride_vh
    cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_pre, kt_off)
    cdna4_async.commit_group()  # ACK[0]
    cdna4_async.buffer_load_to_shared(v_smem.index(0), v_pre, v_off)
    cdna4_async.commit_group()  # ACV[0]
    cdna4_async.buffer_load_to_shared(kt_smem.index(1), k_pre + kt_step, kt_off)
    cdna4_async.commit_group()  # ACK[1]

    # disable_licm: hoisting per-q-block setup (LDS addresses, mask constants) out of
    # this loop keeps it live across the hot loop, where it spills.
    for t in tl.range(0, n_qblocks, disable_licm=True):
        tb = (pid * PER_WG + t) * NSLOT
        if TIMING:
            _stamp(Timing, tb, t_layout)
        bh, start_m, n_blocks = _job(t, wg_x, IS_CAUSAL, NUM_WG, JOBS_PER_BH, NUM_M,
                                     NUM_BLOCKS, TILES_PER_M)
        off_h_q = bh % HQ
        off_z = bh // HQ
        k_base = K + off_z * stride_kz + (off_h_q * HK // HQ) * stride_kh
        v_base = V + off_z * stride_vz + (off_h_q * HK // HQ) * stride_vh
        block_end = n_blocks

        if SCALE_ON_Q:
            q_dot = (q_next.to(gl.float32) * qk_scale).to(Q.dtype.element_ty)
        else:
            q_dot = q_next
        m_i = gl.full([BLOCK_M], M_INIT, dtype=gl.float32, layout=mma_m_layout)
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
        acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mma_layout)

        # -- Prologue (fmha_v4's; ACK[0], ACV[0], ACK[1] were prefetched) ------------
        cdna4_async.wait_group(2)
        kt0 = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt0, qk)  # dot_qk[0]
        if IS_CAUSAL:
            # Diagonal only for block 0; otherwise every wave takes the keep-all branch.
            qk = causal_mask(qk, _delta(0, block_end, BLOCK_N), BLOCK_M, BLOCK_N, True)
        m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_i, qk_scale, SCALE_ON_Q)
        gl.barrier()
        cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_base + 2 * kt_step, kt_off)
        cdna4_async.commit_group()  # ACK[2]
        cdna4_async.wait_group(1)
        kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(1), kt_dot_layout)
        cdna4_async.buffer_load_to_shared(v_smem.index(1), v_base + v_step, v_off)
        cdna4_async.commit_group()   # ACV[1]
        acc, l_i = rescale_lazy(acc, l_i, alpha_c)
        if TIMING:
            _stamp(Timing, tb + 1, t_layout)

        # -- Main loop: n-2 iterations in pairs, warp-pipelined -------------------
        # One iteration more than fmha_v4's loop: that removes its odd tail (n-3 is odd
        # for every causal block) and one drain stage.
        main_loop_pairs = (block_end - 2) // 2
        # n >= 4, so the loop always runs. Saying so removes LLVM's zero-trip guard,
        # whose second path to the drain otherwise costs ~90 spilled VGPRs.
        gl.assume(main_loop_pairs > 0)
        if IS_CAUSAL:
            # The diagonal's VEC1s are the last three iterations: those pairs (one for
            # block 0) run as a second, masked, loop. Its trip count is a runtime value;
            # a constant 1 would be folded away and its stages would leak.
            plain_pairs = gl.maximum(main_loop_pairs - 2, 0)
        else:
            plain_pairs = main_loop_pairs
        for pair_idx in tl.range(0, plain_pairs):
            kt_dot, acc, l_i, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = _pipe_pair(
                q_dot, kt_dot, acc, l_i, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c,
                kt_smem, v_smem, k_base, v_base, kt_off, v_off, kt_step, v_step,
                pair_idx * 2, block_end, False,
                mma_layout, p_dot_layout, v_dot_layout, kt_dot_layout, qk_scale, SCALE_ON_Q,
                BLOCK_M, BLOCK_N, WAIT)
        if TIMING:
            _stamp(Timing, tb + 2, t_layout)
        if IS_CAUSAL:
            for pair_idx in tl.range(plain_pairs, main_loop_pairs):
                kt_dot, acc, l_i, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = _pipe_pair(
                    q_dot, kt_dot, acc, l_i, m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c,
                    kt_smem, v_smem, k_base, v_base, kt_off, v_off, kt_step, v_step,
                    pair_idx * 2, block_end, True,
                    mma_layout, p_dot_layout, v_dot_layout, kt_dot_layout, qk_scale, SCALE_ON_Q,
                    BLOCK_M, BLOCK_N, WAIT)
        if TIMING:
            _stamp(Timing, tb + 3, t_layout)

        # -- Drain: tiles n-2 and n-1 ----------------------------------------------
        # Pending {K[n-1] (dup), V[n-1]}; kt_dot = K[n-1], p_c = p[n-2].
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        if IS_CAUSAL and SKIP_MASKED_WAVES:
            # Tile n-1 is the diagonal's far end: only waves whose last row reaches its
            # first column (32 * wave + 31 >= delta, i.e. waves 6 and 7) have anything
            # to keep, so the rest skip its QK and let the mask fill in -inf. Plain code
            # with no barrier before the mask, so a per-wave branch is safe. (The same
            # skip around the PVs would save more, but the accumulator then merges two
            # paths and the register allocator spills ~170 VGPRs.)
            if gl.amd.warp_id() * 32 + 31 >= _delta(block_end - 1, block_end, BLOCK_N):
                qk = mfma_cdna4(q_dot, kt_dot, qk)   # dot_qk[n-1]
        else:
            qk = mfma_cdna4(q_dot, kt_dot, qk)   # dot_qk[n-1]
        if IS_CAUSAL:
            qk = causal_mask(qk, _delta(block_end - 1, block_end, BLOCK_N), BLOCK_M, BLOCK_N, True)
        cdna4_async.wait_group(1)            # V[n-2] complete
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
        l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)
        acc = mfma_cdna4(p_dot, v_dot, acc)  # dot_pv[n-2]
        m_run, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, alpha_c = sc_vec1(qk, m_run, qk_scale, SCALE_ON_Q)
        acc, l_i = rescale_lazy(acc, l_i, alpha_c)
        cdna4_async.wait_group(0)            # V[n-1] complete
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(1), v_dot_layout)

        # -- Prefetch the next q-block: Q into registers (q_dot is dead), its first K/V
        # tiles into the ring (this block's last reads of slots kt0/kt1/v0 are done once
        # every wave passes the barrier). They overlap the last PV, the epilogue and the
        # next q-block's setup. The last q-block refetches itself (drained after the loop).
        gl.barrier()
        bh_n, sm_n, nb_n = _job(gl.minimum(t + 1, n_qblocks - 1), wg_x, IS_CAUSAL, NUM_WG,
                                JOBS_PER_BH, NUM_M, NUM_BLOCKS, TILES_PER_M)
        # Offsets computed here, not hoisted: kept live across the hot loop they spill.
        q_next = gl.load(Q + (bh_n // HQ) * stride_qz + (bh_n % HQ) * stride_qh
                         + sm_n * BLOCK_M * stride_qm
                         + gl.arange(0, BLOCK_M, layout=q_m)[:, None] * stride_qm
                         + gl.arange(0, BLOCK_DMODEL, layout=q_d)[None, :] * stride_qk)
        k_pre = K + (bh_n // HQ) * stride_kz + ((bh_n % HQ) * HK // HQ) * stride_kh
        v_pre = V + (bh_n // HQ) * stride_vz + ((bh_n % HQ) * HK // HQ) * stride_vh
        cdna4_async.buffer_load_to_shared(kt_smem.index(0), k_pre, kt_off)
        cdna4_async.commit_group()  # ACK[0] of the next q-block
        cdna4_async.buffer_load_to_shared(v_smem.index(0), v_pre, v_off)
        cdna4_async.commit_group()  # ACV[0]
        cdna4_async.buffer_load_to_shared(kt_smem.index(1), k_pre + kt_step, kt_off)
        cdna4_async.commit_group()  # ACK[1]
        if TIMING:
            _stamp(Timing, tb + 4, t_layout)

        l_i, p_dot = sc_vec2(l_i, p_c_0123, p_c_4, qk_c_5, qk_c_6, qk_c_7, m_run, p_dot_layout, q_dot.dtype, qk_scale, SCALE_ON_Q)
        acc = mfma_cdna4(p_dot, v_dot, acc)  # dot_pv[n-1]
        if TIMING:
            _stamp(Timing, tb + 5, t_layout)

        # -- Epilogue --------------------------------------------------------------
        acc = acc * (1.0 / l_i)[:, None]
        o_base = Out + off_z * stride_oz + off_h_q * stride_oh
        acc_out = acc.to(Out.dtype.element_ty)
        # O leaves through LDS in two explicit 32 KB halves: each is smaller than the
        # K/V buffers, so the size-sorted allocator still puts the K/V ring at the bottom
        # of LDS, where its ds_read offsets fit 16 bits (a single 64 KB conversion
        # scratch would be placed first and push them above 64 KB); both halves are
        # written before one barrier; and they never have to coexist with a Q staging
        # buffer. Storing straight from the MFMA layout instead makes every 8-byte store
        # span 32 rows and costs ~4 us per q-block.
        HALF_D: gl.constexpr = BLOCK_DMODEL // 2
        oh_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8], threads_per_warp=[64 // (HALF_D // 8), HALF_D // 8],
            warps_per_cta=[num_warps, 1], order=[1, 0])
        os_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
        o_s0 = gl.allocate_shared_memory(Out.dtype.element_ty, [BLOCK_M, HALF_D], layout=os_layout)
        o_s1 = gl.allocate_shared_memory(Out.dtype.element_ty, [BLOCK_M, HALF_D], layout=os_layout)
        o_lo, o_hi = _split_halves(acc_out)
        o_s0.store(o_lo)
        o_s1.store(o_hi)
        om = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, oh_layout))
        od = gl.arange(0, HALF_D, layout=gl.SliceLayout(0, oh_layout))
        o_ptrs = o_base + om[:, None] * stride_om + od[None, :] * stride_on
        gl.store(o_ptrs, o_s0.load(oh_layout))
        gl.store(o_ptrs + HALF_D * stride_on, o_s1.load(oh_layout))

        # The LSE goes straight from the row layout: the two lanes that share a row
        # write the same value, which costs nothing next to an LDS round trip.
        lse = m_run / 1.44269504089 + gl.log2(l_i) / 1.44269504089
        offs_m_row = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_m_layout)
        gl.store(L + off_z * HQ * N_CTX + off_h_q * N_CTX + offs_m_row, lse)
        if TIMING:
            _stamp(Timing, tb + 6, t_layout)
    cdna4_async.wait_group(0)   # the final, redundant prefetch


LAST_TIMING = None   # FA_WG_TIMING=1: [q-blocks, NSLOT, 4] stamps of the last launch
LAST_LSE = None      # the last launch's log-sum-exp, for tests
CONFIG = dict(BLOCK_M=256, BLOCK_N=64, num_warps=8, waves_per_eu=2,
              llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),))


def num_cus(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def run_gluon_attention(q, k, v, o, metadata: MetaData, scale_on_q: bool = True):
    """Persistent launch, one workgroup per CU.

    Self-attention (Q and K share one length), head dim 128, length a multiple of 256 (512
    for causal: blocks are paired), bhsd or bshd. Causal needs length >= 1024: with a single
    pair per (batch, head) every trip count becomes a compile-time 1, MLIR folds the
    warp-pipelined loops away, and their stage markers land in the outer loop.
    """
    S = metadata.max_seqlens_q
    assert S == metadata.max_seqlens_k, "self-attention only (Q and K share one length)"
    assert S % 256 == 0, "length must be a multiple of BLOCK_M (256)"
    if metadata.causal:
        assert S % 512 == 0 and S >= 1024, "causal needs length % 512 == 0 and >= 1024"
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
    assert head_size == 128, "head dim 128 only"
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, metadata)
    M = torch.empty((batch, nheads_q, S), device=q.device, dtype=torch.float32)
    jobs = batch * nheads_q * ((S // 512) if metadata.causal else (S // 256))
    # One workgroup per CU, but never more than there are jobs (every workgroup must get
    # one), and a multiple of the 8 XCDs for the job interleave.
    num_wg = min(num_cus(q.device), jobs)
    num_wg -= num_wg % 8
    assert num_wg > 0, "needs at least 8 jobs"
    timing = {}
    if os.environ.get("FA_WG_TIMING") == "1":
        per_wg = -(-jobs // num_wg) * (2 if metadata.causal else 1)
        global LAST_TIMING
        LAST_TIMING = torch.zeros((num_wg * per_wg, NSLOT.value, 4), device=q.device,
                                  dtype=torch.int64)
        timing = dict(Timing=LAST_TIMING, TIMING=True)
    global LAST_LSE
    LAST_LSE = M
    names = [f"stride_{t}{d}" for t, ds in (("q", "zhmk"), ("k", "zhnk"), ("v", "zhkn"), ("o", "zhmn"))
             for d in ds]
    strides = dict(zip(names, [s for grp in (q_strides, k_strides, v_strides, o_strides) for s in grp]))
    gluon_attn_fwd[(num_wg,)](
        q, k, v, metadata.sm_scale, M, o,
        **strides,
        HQ=nheads_q, HK=nheads_k, BATCH=batch,
        N_CTX=S,
        IS_CAUSAL=bool(metadata.causal),
        BLOCK_M=CONFIG["BLOCK_M"], BLOCK_DMODEL=head_size, BLOCK_N=CONFIG["BLOCK_N"],
        NUM_WG=num_wg,
        SCALE_ON_Q=scale_on_q,
        num_warps=CONFIG["num_warps"], waves_per_eu=CONFIG["waves_per_eu"],
        llvm_fn_attrs=CONFIG["llvm_fn_attrs"],
        **timing,
    )
