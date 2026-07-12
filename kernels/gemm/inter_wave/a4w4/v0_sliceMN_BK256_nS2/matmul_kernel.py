##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
##############################################################################

"""
v0_sliceMN_BK256_nS2 -- 8-wave warp-pipeline MXFP4 (a4w4) GEMM, M/N-sliced.

This is the a4w4 (4-bit, MXFP4/e2m1) analogue of
inter_wave/a16w16. The 8-wave skeleton -- 8 warps ([2,4] =
2 waves/SIMD), the 2x2 [128x128] quadrant slicing with four separate
double-buffered LDS allocations, the warp_pipeline_stage wave-level schedule,
no-AGPR, and the spill-free store-side pointer-walk epilogue -- is copied from
the fp16 8-wave kernel. The MXFP4 numerics come from the 4-wave
a4w4/v1_sliceMN kernel:

  * operands are packed FP4 (uint8, two e2m1 nibbles per byte), so the K-step
    is BLOCK_K=256 logical = 128 bytes / row into LDS;
  * every group of 32 e2m1 elements shares an 8-bit e8m0 scale. Each tile
    carries a [128, 8] uint8 scale half-tile that streams straight into LDS
    (buffer_load_to_shared, no ds_write) in the SAME commit group as its tile;
  * MFMA is `mfma_scaled(..., a_format="e2m1", ..., b_format="e2m1")` with the
    e8m0 scale operands, instr_shape [16,16,128], k_width=16;
  * B is stored (N, K//2); the LDS tile is loaded with `.permute([1,0])` to
    feed the MFMA as a logical (K, N) operand;
  * output C is bf16.

The 8-wave global-load layouts are the 4-wave a4w4 layouts with one register
base promoted to a third warp base (warpsPerCTA [2,2] -> [2,4]); the scale
global-load blocked layout gains the extra warp along M. The padded shared tile
layouts and the identity scale shared layout are warp-independent and reused
verbatim from the 4-wave a4w4 kernel.
"""

import torch
import triton
import triton.language as tl
from common import get_pids
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 256

NUM_WARPS = 8
WARPS_M = 2
WARPS_N = 4

NUM_XCDS = 8
GROUP_SIZE_M = 4

SCALE_GROUP_SIZE = 32

MIN_K = 4 * BLOCK_K
KERNEL_NAME = "v0_sliceMN_BK256_nS2"


@gluon.jit
def v0_sliceMN_BK256_nS2(
    a_ptr,
    b_ptr,
    c_ptr,  #
    a_scales_ptr,
    b_scales_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bn,
    stride_bk,  #
    stride_cm,
    stride_cn,  #
    stride_asm,
    stride_ask,  #
    stride_bsn,
    stride_bsk,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    SCALE_GROUP_SIZE: gl.constexpr = 32

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # ---- 8-wave global-load layouts (4-wave a4w4 + 1 extra warp dim) ----
    # A half-M tile [BLOCK_M//2, BLOCK_K//2] = [128, 128] packed-FP4 bytes;
    # warps tile M (3 bits = 8 warps).
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0], [4, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K // 2],
    )
    # B half-N tile [BLOCK_N//2, BLOCK_K//2] = [128, 128]; warps tile N.
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0], [4, 0]],
        block_bases=[],
        shape=[BLOCK_N // 2, BLOCK_K // 2],
    )

    # ---- scale global-load blocked layout (half tile = [128, 8] uint8) ----
    # Keep 4 contiguous bytes/thread along M (order[0,1]) so the async copy to
    # LDS lowers to a single `buffer_load_dword ... lds` (the LDS DMA needs dword
    # granularity). The half-tile is only 1024 bytes = 256 dword-threads, but an
    # 8-warp kernel spans 512 threads, so warpsPerCTA=[2,4] over-covers M by 2x:
    # 256 threads issue the dword loads, the other 256 are masked. The 4-wave
    # layout was [4,1],[32,2],[1,4]; only warpsPerCTA changes ([1,4]->[2,4]).
    blocked_scales_half: gl.constexpr = gl.BlockedLayout(
        [4, 1],
        [32, 2],
        [2, 4],
        [0, 1],
    )

    # ---- padded shared tile layouts (warp-independent, reused from 4-wave a4w4) ----
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
    sharedScaleLayout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    scale_a_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_M // 2, BLOCK_K // SCALE_GROUP_SIZE]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N // 2, BLOCK_K // SCALE_GROUP_SIZE]
    )

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

    # ---- load-side pointer-walk offsets ----
    # All four A/B quadrants (and both K-buffers) share ONE within-tile offset
    # tensor; the top/bot, left/right, and even/odd(_next) variants are reached
    # by adding SCALAR deltas to the (uniform) base pointer instead of holding a
    # separate [128x128] offset tensor per variant. This keeps the four f32
    # accumulators resident under the 256-VGPR (2 waves/SIMD) budget: 8 tile +
    # 8 scale offset tensors -> 2 tile + 2 scale offset tensors.
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutA))
    a_tile_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutB))
    b_tile_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    offs_ks_a = gl.arange(0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales_half))
    offs_asm = (
        pid_m * BLOCK_M + gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, blocked_scales_half))
    ) % M
    a_sc_offsets = offs_asm[:, None] * stride_asm + offs_ks_a[None, :] * stride_ask

    offs_ks_b = gl.arange(0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales_half))
    offs_bsn = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, blocked_scales_half))
    ) % N
    b_sc_offsets = offs_bsn[:, None] * stride_bsn + offs_ks_b[None, :] * stride_bsk

    # Scalar (uniform) base-pointer deltas for the quadrant / K-buffer variants.
    a_half_m = (BLOCK_M // 2) * stride_am  # a_top -> a_bot
    b_half_n = (BLOCK_N // 2) * stride_bn  # b_left -> b_right
    a_k2 = (BLOCK_K // 2) * stride_ak  # even -> odd (_next) K-step
    b_k2 = (BLOCK_K // 2) * stride_bk
    a_sc_half_m = (BLOCK_M // 2) * stride_asm
    b_sc_half_n = (BLOCK_N // 2) * stride_bsn
    a_sc_k = (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
    b_sc_k = (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # ---- Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits, tile+scale each) ----
    cdna4_async.buffer_load_to_shared(smemB_left.index(0), b_base, b_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_b_sc_l.index(0), b_scales_ptr, b_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_top.index(0), a_base, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_a_sc_t.index(0), a_scales_ptr, a_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half_m, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(
        smem_a_sc_b.index(0), a_scales_ptr + a_sc_half_m, a_sc_offsets
    )
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemB_right.index(0), b_base + b_half_n, b_tile_offsets)
    cdna4_async.buffer_load_to_shared(
        smem_b_sc_r.index(0), b_scales_ptr + b_sc_half_n, b_sc_offsets
    )
    cdna4_async.commit_group()

    cdna4_async.buffer_load_to_shared(smemB_left.index(1), b_base + b_k2, b_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_b_sc_l.index(1), b_scales_ptr + b_sc_k, b_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_top.index(1), a_base + a_k2, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_a_sc_t.index(1), a_scales_ptr + a_sc_k, a_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_bot.index(1), a_base + a_half_m + a_k2, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(
        smem_a_sc_b.index(1), a_scales_ptr + a_sc_half_m + a_sc_k, a_sc_offsets
    )
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(
        smemB_right.index(1), b_base + b_half_n + b_k2, b_tile_offsets
    )
    cdna4_async.buffer_load_to_shared(
        smem_b_sc_r.index(1), b_scales_ptr + b_sc_half_n + b_sc_k, b_sc_offsets
    )
    cdna4_async.commit_group()

    a_base += (BLOCK_K // 2) * stride_ak * 2
    b_base += (BLOCK_K // 2) * stride_bk * 2
    a_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask * 2
    b_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk * 2

    cdna4_async.wait_group(6)
    b_left = smemB_left.index(0).permute([1, 0]).load(dot_b_layout)
    b_sc_left = smem_b_sc_l.index(0).load(scale_b_layout)
    a_top = smemA_top.index(0).load(dot_a_layout)
    a_sc_top = smem_a_sc_t.index(0).load(scale_a_layout)

    gl.assume(iterMax > 3)

    # ---- Main loop (2x unrolled): 8 (mfma + LR + AC) regions ----
    for k in tl.range(0, iterMax - 2, 2):
        # --- sub-iter 0 (buffer 0) ---
        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.cdna4.mfma_scaled(
                a=a_top,
                a_scale=a_sc_top,
                a_format="e2m1",
                b=b_left,
                b_scale=b_sc_left,
                b_format="e2m1",
                acc=acc_tl,
            )
        with warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.index(0).load(dot_a_layout)
            a_sc_bot = smem_a_sc_b.index(0).load(scale_a_layout)
            cdna4_async.buffer_load_to_shared(smemB_left.index(0), b_base, b_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_b_sc_l.index(0), b_scales_ptr, b_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.cdna4.mfma_scaled(
                a=a_bot,
                a_scale=a_sc_bot,
                a_format="e2m1",
                b=b_left,
                b_scale=b_sc_left,
                b_format="e2m1",
                acc=acc_bl,
            )
        with warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.index(0).permute([1, 0]).load(dot_b_layout)
            b_sc_right = smem_b_sc_r.index(0).load(scale_b_layout)
            cdna4_async.buffer_load_to_shared(smemA_top.index(0), a_base, a_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_a_sc_t.index(0), a_scales_ptr, a_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.cdna4.mfma_scaled(
                a=a_top,
                a_scale=a_sc_top,
                a_format="e2m1",
                b=b_right,
                b_scale=b_sc_right,
                b_format="e2m1",
                acc=acc_tr,
            )
        with warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.index(1).permute([1, 0]).load(dot_b_layout)
            b_sc_left = smem_b_sc_l.index(1).load(scale_b_layout)
            cdna4_async.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half_m, a_tile_offsets)
            cdna4_async.buffer_load_to_shared(
                smem_a_sc_b.index(0), a_scales_ptr + a_sc_half_m, a_sc_offsets
            )
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.cdna4.mfma_scaled(
                a=a_bot,
                a_scale=a_sc_bot,
                a_format="e2m1",
                b=b_right,
                b_scale=b_sc_right,
                b_format="e2m1",
                acc=acc_br,
            )
        with warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.index(1).load(dot_a_layout)
            a_sc_top = smem_a_sc_t.index(1).load(scale_a_layout)
            cdna4_async.buffer_load_to_shared(
                smemB_right.index(0), b_base + b_half_n, b_tile_offsets
            )
            cdna4_async.buffer_load_to_shared(
                smem_b_sc_r.index(0), b_scales_ptr + b_sc_half_n, b_sc_offsets
            )
            cdna4_async.commit_group()

        # --- sub-iter 1 (buffer 1, base + one K-step) ---
        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.cdna4.mfma_scaled(
                a=a_top,
                a_scale=a_sc_top,
                a_format="e2m1",
                b=b_left,
                b_scale=b_sc_left,
                b_format="e2m1",
                acc=acc_tl,
            )
        with warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.index(1).load(dot_a_layout)
            a_sc_bot = smem_a_sc_b.index(1).load(scale_a_layout)
            cdna4_async.buffer_load_to_shared(smemB_left.index(1), b_base + b_k2, b_tile_offsets)
            cdna4_async.buffer_load_to_shared(
                smem_b_sc_l.index(1), b_scales_ptr + b_sc_k, b_sc_offsets
            )
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.cdna4.mfma_scaled(
                a=a_bot,
                a_scale=a_sc_bot,
                a_format="e2m1",
                b=b_left,
                b_scale=b_sc_left,
                b_format="e2m1",
                acc=acc_bl,
            )
        with warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.index(1).permute([1, 0]).load(dot_b_layout)
            b_sc_right = smem_b_sc_r.index(1).load(scale_b_layout)
            cdna4_async.buffer_load_to_shared(smemA_top.index(1), a_base + a_k2, a_tile_offsets)
            cdna4_async.buffer_load_to_shared(
                smem_a_sc_t.index(1), a_scales_ptr + a_sc_k, a_sc_offsets
            )
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.cdna4.mfma_scaled(
                a=a_top,
                a_scale=a_sc_top,
                a_format="e2m1",
                b=b_right,
                b_scale=b_sc_right,
                b_format="e2m1",
                acc=acc_tr,
            )
        with warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.index(0).permute([1, 0]).load(dot_b_layout)
            b_sc_left = smem_b_sc_l.index(0).load(scale_b_layout)
            cdna4_async.buffer_load_to_shared(
                smemA_bot.index(1), a_base + a_half_m + a_k2, a_tile_offsets
            )
            cdna4_async.buffer_load_to_shared(
                smem_a_sc_b.index(1), a_scales_ptr + a_sc_half_m + a_sc_k, a_sc_offsets
            )
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.cdna4.mfma_scaled(
                a=a_bot,
                a_scale=a_sc_bot,
                a_format="e2m1",
                b=b_right,
                b_scale=b_sc_right,
                b_format="e2m1",
                acc=acc_br,
            )
        with warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.index(0).load(dot_a_layout)
            a_sc_top = smem_a_sc_t.index(0).load(scale_a_layout)
            cdna4_async.buffer_load_to_shared(
                smemB_right.index(1), b_base + b_half_n + b_k2, b_tile_offsets
            )
            cdna4_async.buffer_load_to_shared(
                smem_b_sc_r.index(1), b_scales_ptr + b_sc_half_n + b_sc_k, b_sc_offsets
            )
            cdna4_async.commit_group()
            a_base += a_k2 * 2
            b_base += b_k2 * 2
            a_scales_ptr += a_sc_k * 2
            b_scales_ptr += b_sc_k * 2

    # ---- Epilogue: last 2 K-steps, drain, 4-quadrant store ----
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([4, 8], [4, 16], [WARPS_M, WARPS_N], [1, 0])
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_quad_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tl_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_bl_base = c_tl_base + (BLOCK_M // 2) * stride_cm
    c_tr_base = c_tl_base + (BLOCK_N // 2) * stride_cn
    c_br_base = c_bl_base + (BLOCK_N // 2) * stride_cn

    # iter iterMax-2
    acc_tl = gl.amd.cdna4.mfma_scaled(
        a=a_top,
        a_scale=a_sc_top,
        a_format="e2m1",
        b=b_left,
        b_scale=b_sc_left,
        b_format="e2m1",
        acc=acc_tl,
    )
    cdna4_async.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dot_a_layout)
    a_sc_bot = smem_a_sc_b.index(l_idx).load(scale_a_layout)

    acc_bl = gl.amd.cdna4.mfma_scaled(
        a=a_bot,
        a_scale=a_sc_bot,
        a_format="e2m1",
        b=b_left,
        b_scale=b_sc_left,
        b_format="e2m1",
        acc=acc_bl,
    )
    cdna4_async.wait_group(4)
    b_right = smemB_right.index(l_idx).permute([1, 0]).load(dot_b_layout)
    b_sc_right = smem_b_sc_r.index(l_idx).load(scale_b_layout)

    acc_tr = gl.amd.cdna4.mfma_scaled(
        a=a_top,
        a_scale=a_sc_top,
        a_format="e2m1",
        b=b_right,
        b_scale=b_sc_right,
        b_format="e2m1",
        acc=acc_tr,
    )
    cdna4_async.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).permute([1, 0]).load(dot_b_layout)
    b_sc_left = smem_b_sc_l.index(g_idx).load(scale_b_layout)

    acc_br = gl.amd.cdna4.mfma_scaled(
        a=a_bot,
        a_scale=a_sc_bot,
        a_format="e2m1",
        b=b_right,
        b_scale=b_sc_right,
        b_format="e2m1",
        acc=acc_br,
    )
    cdna4_async.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dot_a_layout)
    a_sc_top = smem_a_sc_t.index(g_idx).load(scale_a_layout)

    # iter iterMax-1: complete ALL four accumulators FIRST, then convert + store,
    # so the dot/scale operands die before the store phase and nothing spills.
    acc_tl = gl.amd.cdna4.mfma_scaled(
        a=a_top,
        a_scale=a_sc_top,
        a_format="e2m1",
        b=b_left,
        b_scale=b_sc_left,
        b_format="e2m1",
        acc=acc_tl,
    )
    cdna4_async.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dot_a_layout)
    a_sc_bot = smem_a_sc_b.index(g_idx).load(scale_a_layout)

    acc_bl = gl.amd.cdna4.mfma_scaled(
        a=a_bot,
        a_scale=a_sc_bot,
        a_format="e2m1",
        b=b_left,
        b_scale=b_sc_left,
        b_format="e2m1",
        acc=acc_bl,
    )
    cdna4_async.wait_group(0)
    b_right = smemB_right.index(g_idx).permute([1, 0]).load(dot_b_layout)
    b_sc_right = smem_b_sc_r.index(g_idx).load(scale_b_layout)

    acc_tr = gl.amd.cdna4.mfma_scaled(
        a=a_top,
        a_scale=a_sc_top,
        a_format="e2m1",
        b=b_right,
        b_scale=b_sc_right,
        b_format="e2m1",
        acc=acc_tr,
    )
    acc_br = gl.amd.cdna4.mfma_scaled(
        a=a_bot,
        a_scale=a_sc_bot,
        a_format="e2m1",
        b=b_right,
        b_scale=b_sc_right,
        b_format="e2m1",
        acc=acc_br,
    )

    c_tl = gl.convert_layout(
        acc_tl.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False
    )
    gl.amd.cdna4.buffer_store(c_tl, c_tl_base, c_quad_offsets)
    c_bl = gl.convert_layout(
        acc_bl.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False
    )
    gl.amd.cdna4.buffer_store(c_bl, c_bl_base, c_quad_offsets)
    c_tr = gl.convert_layout(
        acc_tr.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False
    )
    gl.amd.cdna4.buffer_store(c_tr, c_tr_base, c_quad_offsets)
    c_br = gl.convert_layout(
        acc_br.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False
    )
    gl.amd.cdna4.buffer_store(c_br, c_br_base, c_quad_offsets)


def matmul_kernel_only(a, b, a_scales, b_scales, c):
    """Kernel-only entry. Shapes:
    a:        (M, K//2)  uint8 packed FP4, K-contiguous
    b:        (N, K//2)  uint8 packed FP4, K-contiguous
    a_scales: (M, K//32) uint8 e8m0
    b_scales: (N, K//32) uint8 e8m0
    c:        (M, N)     bf16 (pre-allocated)
    """
    M = a.shape[0]
    K = a.shape[1] * 2
    N = b.shape[0]
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    GRID_MN = grid_m * grid_n
    v0_sliceMN_BK256_nS2[(GRID_MN,)](
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
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no spills).
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    return c


def matmul(a, b, a_scales, b_scales, c=None):
    """C = dequant(A) @ dequant(B).T. a/b packed FP4 (M,K//2)/(N,K//2); output bf16."""
    M = a.shape[0]
    N = b.shape[0]
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    return matmul_kernel_only(a, b, a_scales, b_scales, c)
