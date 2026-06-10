##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
##############################################################################

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def get_pids(
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        # pid remapping on xcds
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def v1_sliceMN(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """
    MXFP4 GEMM kernel with M+N slicing and tile-paired scale async copy.

    Same M+N split skeleton as a16w16/v8_sliceMN: A split along M into a_top/a_bot,
    B split along N into b_left/b_right, output C is a 2x2 grid of quadrants
    (C_tl, C_tr, C_bl, C_br). Loop unrolled by 2 so buffer indices alternate
    naturally without runtime computation.

    MXFP4 deltas vs v8_sliceMN:
      * Inputs are packed FP4 (uint8 with two e2m1 nibbles per byte).
      * Each tile carries an e8m0 scale tile alongside it.
      * Every `AC X` issues `AC X` AND `AC X_sc` in the same commit group —
        scales travel into LDS via buffer_load_to_shared (no register
        roundtrip, no ds_write). Each scale half-tile is 128*8 = 1024 bytes
        across 256 threads = 4 bytes/thread, lowering to a single
        `buffer_load_dword ... offen lds`.
      * Every `LR X` issues `LR X` AND `LR X_sc` (smem.load with the MFMA
        scale layout) so the MFMA scale operand is ready right before each DOT.
      * MFMA is `gl.amd.cdna4.mfma_scaled` with format="e2m1".

    Tile dimensions (BLOCK_M=BLOCK_N=BLOCK_K=256, SCALE_GROUP_SIZE=32):
      A_t, A_b               : [BLOCK_M//2, BLOCK_K//2] = [128, 128] uint8 packed FP4
      B_l, B_r               : [BLOCK_N//2, BLOCK_K//2] = [128, 128] uint8 packed FP4
      A_sc_t, A_sc_b         : [BLOCK_M//2, BLOCK_K//SCALE_GROUP_SIZE] = [128, 8] uint8 e8m0
      B_sc_l, B_sc_r         : [BLOCK_N//2, BLOCK_K//SCALE_GROUP_SIZE] = [128, 8] uint8 e8m0
      C_tl, C_tr, C_bl, C_br : [BLOCK_M//2, BLOCK_N//2] = [128, 128] bf16

    Tensor shapes:
      A:        (M, K//2)  uint8, K-contiguous (packed FP4, two nibbles per byte)
      B:        (N, K//2)  uint8, K-contiguous (packed FP4)
      A_scales: (M, K//32) uint8 e8m0
      B_scales: (N, K//32) uint8 e8m0
      C:        (M, N)     bfloat16
    """

    SCALE_GROUP_SIZE: gl.constexpr = 32

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # ------------------------------------------------------------------
    # Global load layouts (half-M, half-N)
    # Each shape covers the K//2 packed-FP4 byte width along K.
    # ------------------------------------------------------------------
    # Half-M global load layout for A: [128, 128] = [BLOCK_M//2, BLOCK_K//2]
    # v0_sliceN's gLoadLayoutA had reg_bases [..., [128, 0]] covering the full
    # [256, 128] tile; drop that base for the half-M slice.
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K // 2],
    )
    # Half-N global load layout for B: [128, 128] = [BLOCK_N//2, BLOCK_K//2]
    # Identical to v0_sliceN's gLoadLayoutB (already half-N).
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N // 2, BLOCK_K // 2],
    )

    # ------------------------------------------------------------------
    # Scale global load layouts (half tile = [128, 8] uint8 = 1024 bytes)
    # 4 warps * 64 threads = 256 threads. 1024 / 256 = 4 bytes/thread = single
    # buffer_load_dword (this is the buffer_load_to_lds_dword the v1 design
    # spec calls out).
    # ------------------------------------------------------------------
    # sizePerThread=[4, 1]: 4 uint8 along M (contiguous, since M is the fast
    # dim in the source storage, see bench.gen_mxfp4_inputs which transposes
    # the (K//32, M) allocation).
    blocked_scales_half: gl.constexpr = gl.BlockedLayout(
        [4, 1],
        [32, 2],
        [1, 4],
        [0, 1],
    )

    # ------------------------------------------------------------------
    # Shared memory layouts
    # ------------------------------------------------------------------
    # Half-M padded shared layout for A tile: [128, 128]. Same padding spec
    # ([[1024, 32]]) as v0_sliceN's sharedLayoutA; drop the [128, 0] M base
    # (only 7 M bases needed for M=128 instead of 8 for M=256).
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 32]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [],
        [BLOCK_M // 2, BLOCK_K // 2],
    )
    # Half-N padded shared layout for B tile: [128, 128]. Same as v0_sliceN's
    # sharedLayoutB (which was already half-N).
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 32]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [],
        [BLOCK_N // 2, BLOCK_K // 2],
    )

    # Shared layout for half scale tile: [128, 8] uint8 = 1024 bytes.
    # SwizzledSharedLayout(1, 1, 1) = identity swizzle (same shape as v0_sliceN
    # used for its scale smem).
    sharedScaleLayout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    # ------------------------------------------------------------------
    # MFMA + dot-operand + scale operand layouts
    # ------------------------------------------------------------------
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 2]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    # Half-tile scale register layouts. v0_sliceN passed the full-A tile
    # [BLOCK_M, ...] for A; here A is split along M so the layout shape is
    # halved.
    scale_a_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_M // 2, BLOCK_K // SCALE_GROUP_SIZE]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N // 2, BLOCK_K // SCALE_GROUP_SIZE]
    )

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    # ------------------------------------------------------------------
    # SMEM allocations
    # Tile + scale slots, each double-buffered. With scales sharing the
    # commit group of their tile, only one in-flight count needs tracking.
    # ------------------------------------------------------------------
    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K // 2], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K // 2], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [nBuffers, BLOCK_N // 2, BLOCK_K // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [nBuffers, BLOCK_N // 2, BLOCK_K // 2], sharedLayoutB
    )
    smem_a_sc_t = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty,
        [nBuffers, BLOCK_M // 2, BLOCK_K // SCALE_GROUP_SIZE],
        sharedScaleLayout,
    )
    smem_a_sc_b = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty,
        [nBuffers, BLOCK_M // 2, BLOCK_K // SCALE_GROUP_SIZE],
        sharedScaleLayout,
    )
    smem_b_sc_l = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty,
        [nBuffers, BLOCK_N // 2, BLOCK_K // SCALE_GROUP_SIZE],
        sharedScaleLayout,
    )
    smem_b_sc_r = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty,
        [nBuffers, BLOCK_N // 2, BLOCK_K // SCALE_GROUP_SIZE],
        sharedScaleLayout,
    )

    # ------------------------------------------------------------------
    # Tile global offsets (half-M for A, half-N for B). Like v8_sliceMN we
    # pre-compute base and _next so a_base/b_base advance by 2*BLOCK_K only
    # once per unrolled iteration.
    # ------------------------------------------------------------------
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutA))
    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + (BLOCK_M // 2) * stride_am
    a_top_offsets_next = a_top_offsets + (BLOCK_K // 2) * stride_ak
    a_bot_offsets_next = a_bot_offsets + (BLOCK_K // 2) * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutB))
    b_left_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_right_offsets = b_left_offsets + (BLOCK_N // 2) * stride_bn
    b_left_offsets_next = b_left_offsets + (BLOCK_K // 2) * stride_bk
    b_right_offsets_next = b_right_offsets + (BLOCK_K // 2) * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # ------------------------------------------------------------------
    # Scale global offsets (half-M for A scales, half-N for B scales).
    # Scale bases advance with the tile bases (a_scales_ptr/b_scales_ptr
    # bump in the loop, mirroring v0_sliceN).
    # ------------------------------------------------------------------
    offs_ks_a = gl.arange(0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales_half))
    offs_asm_top = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, blocked_scales_half))) % M
    a_sc_top_offsets = offs_asm_top[:, None] * stride_asm + offs_ks_a[None, :] * stride_ask
    a_sc_bot_offsets = a_sc_top_offsets + (BLOCK_M // 2) * stride_asm
    a_sc_top_offsets_next = a_sc_top_offsets + (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
    a_sc_bot_offsets_next = a_sc_bot_offsets + (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask

    offs_ks_b = gl.arange(0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales_half))
    offs_bsn_left = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, blocked_scales_half))) % N
    b_sc_left_offsets = offs_bsn_left[:, None] * stride_bsn + offs_ks_b[None, :] * stride_bsk
    b_sc_right_offsets = b_sc_left_offsets + (BLOCK_N // 2) * stride_bsn
    b_sc_left_offsets_next = b_sc_left_offsets + (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk
    b_sc_right_offsets_next = b_sc_right_offsets + (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    # ------------------------------------------------------------------
    # Accumulators: four 128x128 quadrants
    # ------------------------------------------------------------------
    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # ==================================================================
    # Prologue
    # ==================================================================
    # AC iter 0 --> buffer 0 (base offsets). 4 commit groups, each with
    # a tile+scale pair (matches the diagram's "AC X, X_sc[0]" rows).
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_l.index(g_idx), b_scales_ptr, b_sc_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_t.index(g_idx), a_scales_ptr, a_sc_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_base, a_bot_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_b.index(g_idx), a_scales_ptr, a_sc_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_r.index(g_idx), b_scales_ptr, b_sc_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    # AC iter 1 --> buffer 1 (_next offsets). Same 4 pairs.
    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets_next)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_l.index(g_idx), b_scales_ptr, b_sc_left_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_base, a_top_offsets_next)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_t.index(g_idx), a_scales_ptr, a_sc_top_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_base, a_bot_offsets_next)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_b.index(g_idx), a_scales_ptr, a_sc_bot_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets_next)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_r.index(g_idx), b_scales_ptr, b_sc_right_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    a_base += (BLOCK_K // 2) * stride_ak * 2
    b_base += (BLOCK_K // 2) * stride_bk * 2
    a_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask * 2
    b_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk * 2

    # Wait until B_left[0] and A_top[0] (with their scales) are retired
    # (6 groups still in flight), then bring registers in for the first DOT.
    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).permute([1, 0]).load(dot_b_layout)
    b_sc_left = smem_b_sc_l.index(0).load(scale_b_layout)
    a_top = smemA_top.index(0).load(dot_a_layout)
    a_sc_top = smem_a_sc_t.index(0).load(scale_a_layout)

    gl.assume(iterMax > 3)

    # ==================================================================
    # Main loop (k = 0, 2, 4, ..., iterMax - 4, step 2)
    # Each outer iteration runs 4 regions x 2 sub-iterations = 8 regions.
    # Buffer index alternates buf0/buf1 between sub-iterations.
    # ==================================================================
    for k in range(0, iterMax - 2, 2):

        # ----------------------------------------------------------------
        # Sub-iteration 0: consume buffer 0, prefetch into buffer 0 (base offsets)
        # ----------------------------------------------------------------

        # Region 0: C_tl = DOT(a_top, b_left)
        acc_tl = gl.amd.cdna4.mfma_scaled(
            a=a_top,
            a_scale=a_sc_top,
            a_format="e2m1",
            b=b_left,
            b_scale=b_sc_left,
            b_format="e2m1",
            acc=acc_tl,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dot_a_layout)
        a_sc_bot = smem_a_sc_b.index(0).load(scale_a_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_l.index(0), b_scales_ptr, b_sc_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        # Region 1: C_bl = DOT(a_bot, b_left)
        acc_bl = gl.amd.cdna4.mfma_scaled(
            a=a_bot,
            a_scale=a_sc_bot,
            a_format="e2m1",
            b=b_left,
            b_scale=b_sc_left,
            b_format="e2m1",
            acc=acc_bl,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).permute([1, 0]).load(dot_b_layout)
        b_sc_right = smem_b_sc_r.index(0).load(scale_b_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_t.index(0), a_scales_ptr, a_sc_top_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        # Region 2: C_tr = DOT(a_top, b_right)
        acc_tr = gl.amd.cdna4.mfma_scaled(
            a=a_top,
            a_scale=a_sc_top,
            a_format="e2m1",
            b=b_right,
            b_scale=b_sc_right,
            b_format="e2m1",
            acc=acc_tr,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).permute([1, 0]).load(dot_b_layout)
        b_sc_left = smem_b_sc_l.index(1).load(scale_b_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_b.index(0), a_scales_ptr, a_sc_bot_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        # Region 3: C_br = DOT(a_bot, b_right)
        acc_br = gl.amd.cdna4.mfma_scaled(
            a=a_bot,
            a_scale=a_sc_bot,
            a_format="e2m1",
            b=b_right,
            b_scale=b_sc_right,
            b_format="e2m1",
            acc=acc_br,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dot_a_layout)
        a_sc_top = smem_a_sc_t.index(1).load(scale_a_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_r.index(0), b_scales_ptr, b_sc_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        # ----------------------------------------------------------------
        # Sub-iteration 1: consume buffer 1, prefetch into buffer 1 (_next offsets)
        # ----------------------------------------------------------------

        # Region 0: C_tl = DOT(a_top, b_left)
        acc_tl = gl.amd.cdna4.mfma_scaled(
            a=a_top,
            a_scale=a_sc_top,
            a_format="e2m1",
            b=b_left,
            b_scale=b_sc_left,
            b_format="e2m1",
            acc=acc_tl,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dot_a_layout)
        a_sc_bot = smem_a_sc_b.index(1).load(scale_a_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_left_offsets_next)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_l.index(1), b_scales_ptr, b_sc_left_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        # Region 1: C_bl = DOT(a_bot, b_left)
        acc_bl = gl.amd.cdna4.mfma_scaled(
            a=a_bot,
            a_scale=a_sc_bot,
            a_format="e2m1",
            b=b_left,
            b_scale=b_sc_left,
            b_format="e2m1",
            acc=acc_bl,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).permute([1, 0]).load(dot_b_layout)
        b_sc_right = smem_b_sc_r.index(1).load(scale_b_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(1), a_base, a_top_offsets_next)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_t.index(1), a_scales_ptr, a_sc_top_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        # Region 2: C_tr = DOT(a_top, b_right)
        acc_tr = gl.amd.cdna4.mfma_scaled(
            a=a_top,
            a_scale=a_sc_top,
            a_format="e2m1",
            b=b_right,
            b_scale=b_sc_right,
            b_format="e2m1",
            acc=acc_tr,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).permute([1, 0]).load(dot_b_layout)
        b_sc_left = smem_b_sc_l.index(0).load(scale_b_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(1), a_base, a_bot_offsets_next)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_a_sc_b.index(1), a_scales_ptr, a_sc_bot_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        # Region 3: C_br = DOT(a_bot, b_right)
        acc_br = gl.amd.cdna4.mfma_scaled(
            a=a_bot,
            a_scale=a_sc_bot,
            a_format="e2m1",
            b=b_right,
            b_scale=b_sc_right,
            b_format="e2m1",
            acc=acc_br,
        )
        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dot_a_layout)
        a_sc_top = smem_a_sc_t.index(0).load(scale_a_layout)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(1), b_base, b_right_offsets_next)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smem_b_sc_r.index(1), b_scales_ptr, b_sc_right_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        a_base += (BLOCK_K // 2) * stride_ak * 2
        b_base += (BLOCK_K // 2) * stride_bk * 2
        a_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask * 2
        b_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk * 2

    # ==================================================================
    # Epilogue: last 2 iterations (no new async copies). Natural-pipeline
    # epilogue interleaves stores with the final MFMAs.
    # ==================================================================

    gStoreLayoutC_local: gl.constexpr = gStoreLayoutC

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC_local))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC_local))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + (BLOCK_N // 2) * stride_cn
    c_bl_offsets = c_tl_offsets + (BLOCK_M // 2) * stride_cm
    c_br_offsets = c_bl_offsets + (BLOCK_N // 2) * stride_cn

    # Iter iterMax - 2 (l_idx = (iterMax - 2) % 2)
    acc_tl = gl.amd.cdna4.mfma_scaled(
        a=a_top, a_scale=a_sc_top, a_format="e2m1",
        b=b_left, b_scale=b_sc_left, b_format="e2m1",
        acc=acc_tl,
    )
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dot_a_layout)
    a_sc_bot = smem_a_sc_b.index(l_idx).load(scale_a_layout)

    acc_bl = gl.amd.cdna4.mfma_scaled(
        a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
        b=b_left, b_scale=b_sc_left, b_format="e2m1",
        acc=acc_bl,
    )
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).permute([1, 0]).load(dot_b_layout)
    b_sc_right = smem_b_sc_r.index(l_idx).load(scale_b_layout)

    acc_tr = gl.amd.cdna4.mfma_scaled(
        a=a_top, a_scale=a_sc_top, a_format="e2m1",
        b=b_right, b_scale=b_sc_right, b_format="e2m1",
        acc=acc_tr,
    )
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).permute([1, 0]).load(dot_b_layout)
    b_sc_left = smem_b_sc_l.index(g_idx).load(scale_b_layout)

    acc_br = gl.amd.cdna4.mfma_scaled(
        a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
        b=b_right, b_scale=b_sc_right, b_format="e2m1",
        acc=acc_br,
    )
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dot_a_layout)
    a_sc_top = smem_a_sc_t.index(g_idx).load(scale_a_layout)

    # Iter iterMax - 1: final MFMAs interleaved with stores
    acc_tl = gl.amd.cdna4.mfma_scaled(
        a=a_top, a_scale=a_sc_top, a_format="e2m1",
        b=b_left, b_scale=b_sc_left, b_format="e2m1",
        acc=acc_tl,
    )
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dot_a_layout)
    a_sc_bot = smem_a_sc_b.index(g_idx).load(scale_a_layout)

    acc_bl = gl.amd.cdna4.mfma_scaled(
        a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
        b=b_left, b_scale=b_sc_left, b_format="e2m1",
        acc=acc_bl,
    )
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).permute([1, 0]).load(dot_b_layout)
    b_sc_right = smem_b_sc_r.index(g_idx).load(scale_b_layout)

    c_tl = acc_tl.to(c_ptr.type.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_tl, c_base, c_tl_offsets)

    acc_tr = gl.amd.cdna4.mfma_scaled(
        a=a_top, a_scale=a_sc_top, a_format="e2m1",
        b=b_right, b_scale=b_sc_right, b_format="e2m1",
        acc=acc_tr,
    )

    c_bl = acc_bl.to(c_ptr.type.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_bl, c_base, c_bl_offsets)

    acc_br = gl.amd.cdna4.mfma_scaled(
        a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
        b=b_right, b_scale=b_sc_right, b_format="e2m1",
        acc=acc_br,
    )

    c_tr = acc_tr.to(c_ptr.type.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_tr, c_base, c_tr_offsets)

    c_br = acc_br.to(c_ptr.type.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_br, c_base, c_br_offsets)


def matmul(a, b, a_scales, b_scales):
    # A:        (M, K//2)  uint8 K-contiguous (packed FP4)
    # B:        (N, K//2)  uint8 K-contiguous (packed FP4)
    # A_scales: (M, K//32) uint8 e8m0
    # B_scales: (N, K//32) uint8 e8m0
    M = a.shape[0]
    K_packed = a.shape[1]
    K = K_packed * 2
    N = b.shape[0]

    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 256
    num_warps = 4

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4

    v1_sliceMN[grid](
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )
    return c
