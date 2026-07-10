##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
##############################################################################

"""
v2_mfma32x32x64_BK256_nS2 -- 8-wave warp-pipeline MXFP4 (a4w4) GEMM, M/N-sliced
tiles + combined (un-N-sliced) B scale so the B scale is read with the hardware
transpose (ds_read_b64_tr_b8) instead of byte-gather + v_perm.

Why v0's B scale degrades. v0 slices *everything* into the 2x2 [128x128] quadrant
grid, including the B scale (b_sc_left / b_sc_right, each [128, 8]). At 8 warps N
is tiled by WARPS_N=4 warps, so a [128, 8] B-scale half gives each thread only
4 bytes -- below the 64-bit width of ds_read_b64_tr_b8 -- and the B scale
degrades to per-byte ds_read_u8 + v_perm reassembly (the A scale, tiled by
WARPS_M=2, is unaffected and keeps the transpose read).

The fix (hybrid slice): keep the *tiles* M/N-sliced, but load the FULL
[BLOCK_N, NG] = [256, 8] B scale as ONE combined buffer (both the async
global->LDS fill and the LDS->register read). At [2,4] the un-sliced [256, 8]
gives each thread 8 bytes = 64 bits, so the local_load lowers to
ds_read_b64_tr_b8 with no v_perm. get_mfma_scale_layout([256,8]) is exactly the
per-quadrant get_mfma_scale_layout([128,8]) (== scale_b_layout) plus one extra
register base [128,0], so a zero-cost split + convert_layout recovers the left
(N in [0,128)) and right (N in [128,256)) [128, 8] halves that feed the
left/right MFMA columns -- the same operands v0 uses.

The async fill of the whole [256, 8] scale needs the right blocked layout. The
scale is N-contiguous in HBM (b_scales is (K/32, N).T, strides (1, N)), and
gfx950 direct-to-LDS cannot scatter: each warp's dword writes must land in ONE
contiguous LDS run. With v0's [4,1],[32,2],[2,4] a warp spans 128 N x 2 K, and
the two K groups are tileN=256 apart in LDS -> a 128-byte gap -> not coalesced
-> the load will not lower. So the B-scale fill uses [4,1],[64,1],[1,8]: each
warp = 64 N-lanes x 1 K-lane covers 256 N x 1 K = one contiguous 256-byte
K-column, and the 8 warps cover the 8 K groups. (v0's [128,8] halves happen to
coalesce because there tileN=128 == the warp's N-span, so no gap.)
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd import warp_pipeline_stage

from common import get_pids

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
KERNEL_NAME = "v2_mfma32x32x64_BK256_nS2"


@gluon.jit
def _bsc_load_split(smem_bsc, COMB: gl.constexpr, HALF: gl.constexpr,
                    HN: gl.constexpr, NG: gl.constexpr):
    """Read the combined [2*HN, NG] B scale from LDS with the hardware transpose
    (8 bytes/thread at [2,4] -> ds_read_b64_tr_b8, no v_perm), then split it into
    the two [HN, NG] N-halves (left, right) that feed the left/right MFMA columns.
    The split is a register slice and the convert_layout (slice-enc -> linear
    scale_b_layout) is free."""
    sb = smem_bsc.load(COMB)                            # [2*HN, NG] transpose read
    left, right = gl.split(gl.permute(sb.reshape([2, HN, NG]), [1, 2, 0]))
    return gl.convert_layout(left, HALF), gl.convert_layout(right, HALF)


@gluon.jit
def v2_mfma32x32x64_BK256_nS2(
    a_ptr, b_ptr, c_ptr,  #
    a_scales_ptr, b_scales_ptr,  #
    M, N, K,  #
    stride_am, stride_ak,  #
    stride_bn, stride_bk,  #
    stride_cm, stride_cn,  #
    stride_asm, stride_ask,  #
    stride_bsn, stride_bsk,  #
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
    WARPS_M: gl.constexpr, WARPS_N: gl.constexpr,  #
    GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
):
    SCALE_GROUP_SIZE: gl.constexpr = 32
    NG: gl.constexpr = BLOCK_K // SCALE_GROUP_SIZE      # scale groups along K = 8
    HN: gl.constexpr = BLOCK_N // 2                     # 128

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # ---- 8-wave global-load layouts (4-wave a4w4 + 1 extra warp dim) ----
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [16, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0]],
        warp_bases=[[2, 0], [4, 0], [8, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K // 2],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [16, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [1, 0], [32, 0], [64, 0]],
        warp_bases=[[2, 0], [4, 0], [8, 0]],
        block_bases=[],
        shape=[BLOCK_N // 2, BLOCK_K // 2],
    )

    # ---- A scale global-load blocked layout (half tile = [128, 8] uint8) ----
    # v0-style [4,1],[32,2],[2,4]: 256 dword-threads over-cover [128,8] by 2x.
    blocked_a_scales: gl.constexpr = gl.BlockedLayout([4, 1], [32, 2], [2, 4], [0, 1])
    # ---- B scale global-load blocked layout (FULL tile = [256, 8] uint8) ----
    # Each warp = 64 N-lanes x 1 K-lane -> 256 N x 1 K = ONE contiguous 256-byte
    # K-column; the 8 warps cover the 8 K groups. This is the layout that keeps the
    # direct-to-LDS write coalesced for the whole [256,8] (see the module docstring).
    blocked_b_scales: gl.constexpr = gl.BlockedLayout([4, 1], [64, 1], [1, 8], [0, 1])

    # ---- padded shared tile layouts (warp-independent, reused from 4-wave a4w4) ----
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
         [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        [], [BLOCK_M // 2, BLOCK_K // 2],
    )
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
         [1, 0], [32, 0], [64, 0], [2, 0], [4, 0], [8, 0], [16, 0]],
        [], [BLOCK_N // 2, BLOCK_K // 2],
    )
    sharedScaleLayout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 64], transposed=True, warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)
    scale_a_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_a_layout, [BLOCK_M // 2, NG])
    # Per-quadrant B-scale layout (mfma operand) and the combined full-N layout
    # (== per-quadrant + one register base [128,0]) used for the transpose read.
    scale_b_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_b_layout, [BLOCK_N // 2, NG])
    scale_b_comb_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(dot_b_layout, [BLOCK_N, NG])

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(a_ptr.type.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K // 2], sharedLayoutA)
    smemA_bot = gl.allocate_shared_memory(a_ptr.type.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K // 2], sharedLayoutA)
    smemB_left = gl.allocate_shared_memory(b_ptr.type.element_ty, [nBuffers, BLOCK_N // 2, BLOCK_K // 2], sharedLayoutB)
    smemB_right = gl.allocate_shared_memory(b_ptr.type.element_ty, [nBuffers, BLOCK_N // 2, BLOCK_K // 2], sharedLayoutB)
    smem_a_sc_t = gl.allocate_shared_memory(a_scales_ptr.type.element_ty, [nBuffers, BLOCK_M // 2, NG], sharedScaleLayout)
    smem_a_sc_b = gl.allocate_shared_memory(a_scales_ptr.type.element_ty, [nBuffers, BLOCK_M // 2, NG], sharedScaleLayout)
    # Combined B scale: ONE [BLOCK_N, NG] = [256, 8] buffer per K-buffer.
    smem_b_sc = gl.allocate_shared_memory(b_scales_ptr.type.element_ty, [nBuffers, BLOCK_N, NG], sharedScaleLayout)

    # ---- load-side pointer-walk offsets ----
    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutA))
    a_tile_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutB))
    b_tile_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    offs_ks_a = gl.arange(0, NG, gl.SliceLayout(0, blocked_a_scales))
    offs_asm = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, blocked_a_scales))) % M
    a_sc_offsets = offs_asm[:, None] * stride_asm + offs_ks_a[None, :] * stride_ask

    # B scale offsets cover the FULL N tile ([256, 8]) in one async copy.
    offs_ks_b = gl.arange(0, NG, gl.SliceLayout(0, blocked_b_scales))
    offs_bsn = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(1, blocked_b_scales))) % N
    b_sc_offsets = offs_bsn[:, None] * stride_bsn + offs_ks_b[None, :] * stride_bsk

    # Scalar (uniform) base-pointer deltas for the quadrant / K-buffer variants.
    a_half_m = (BLOCK_M // 2) * stride_am        # a_top -> a_bot
    b_half_n = (BLOCK_N // 2) * stride_bn        # b_left -> b_right
    a_k2 = (BLOCK_K // 2) * stride_ak            # even -> odd (_next) K-step
    b_k2 = (BLOCK_K // 2) * stride_bk
    a_sc_half_m = (BLOCK_M // 2) * stride_asm
    a_sc_k = NG * stride_ask
    b_sc_k = NG * stride_bsk

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfma_layout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # ---- Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits) ----
    # The combined B scale rides in the B_left commit group; the B_right group is
    # now tile-only (its scale is covered by the combined load).
    cdna4_async.buffer_load_to_shared(smemB_left.index(0), b_base, b_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_b_sc.index(0), b_scales_ptr, b_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_top.index(0), a_base, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_a_sc_t.index(0), a_scales_ptr, a_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half_m, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_a_sc_b.index(0), a_scales_ptr + a_sc_half_m, a_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemB_right.index(0), b_base + b_half_n, b_tile_offsets)
    cdna4_async.commit_group()

    cdna4_async.buffer_load_to_shared(smemB_left.index(1), b_base + b_k2, b_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_b_sc.index(1), b_scales_ptr + b_sc_k, b_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_top.index(1), a_base + a_k2, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_a_sc_t.index(1), a_scales_ptr + a_sc_k, a_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemA_bot.index(1), a_base + a_half_m + a_k2, a_tile_offsets)
    cdna4_async.buffer_load_to_shared(smem_a_sc_b.index(1), a_scales_ptr + a_sc_half_m + a_sc_k, a_sc_offsets)
    cdna4_async.commit_group()
    cdna4_async.buffer_load_to_shared(smemB_right.index(1), b_base + b_half_n + b_k2, b_tile_offsets)
    cdna4_async.commit_group()

    a_base += (BLOCK_K // 2) * stride_ak * 2
    b_base += (BLOCK_K // 2) * stride_bk * 2
    a_scales_ptr += NG * stride_ask * 2
    b_scales_ptr += NG * stride_bsk * 2

    cdna4_async.wait_group(6)
    b_left = smemB_left.index(0).permute([1, 0]).load(dot_b_layout)
    a_top = smemA_top.index(0).load(dot_a_layout)
    a_sc_top = smem_a_sc_t.index(0).load(scale_a_layout)
    b_sc_left, b_sc_right = _bsc_load_split(smem_b_sc.index(0), scale_b_comb_layout, scale_b_layout, HN, NG)

    gl.assume(iterMax > 3)

    # ---- Main loop (2x unrolled): 8 (mfma + LR + AC) regions ----
    for k in tl.range(0, iterMax - 2, 2):
        # --- sub-iter 0 (buffer 0) ---
        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                              b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_tl)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            a_bot = smemA_bot.index(0).load(dot_a_layout)
            a_sc_bot = smem_a_sc_b.index(0).load(scale_a_layout)
            cdna4_async.buffer_load_to_shared(smemB_left.index(0), b_base, b_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_b_sc.index(0), b_scales_ptr, b_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                              b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_bl)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            b_right = smemB_right.index(0).permute([1, 0]).load(dot_b_layout)
            cdna4_async.buffer_load_to_shared(smemA_top.index(0), a_base, a_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_a_sc_t.index(0), a_scales_ptr, a_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                              b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_tr)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            b_left = smemB_left.index(1).permute([1, 0]).load(dot_b_layout)
            cdna4_async.buffer_load_to_shared(smemA_bot.index(0), a_base + a_half_m, a_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_a_sc_b.index(0), a_scales_ptr + a_sc_half_m, a_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                              b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_br)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            a_top = smemA_top.index(1).load(dot_a_layout)
            a_sc_top = smem_a_sc_t.index(1).load(scale_a_layout)
            b_sc_left, b_sc_right = _bsc_load_split(smem_b_sc.index(1), scale_b_comb_layout, scale_b_layout, HN, NG)
            cdna4_async.buffer_load_to_shared(smemB_right.index(0), b_base + b_half_n, b_tile_offsets)
            cdna4_async.commit_group()

        # --- sub-iter 1 (buffer 1, base + one K-step) ---
        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tl = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                              b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_tl)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            a_bot = smemA_bot.index(1).load(dot_a_layout)
            a_sc_bot = smem_a_sc_b.index(1).load(scale_a_layout)
            cdna4_async.buffer_load_to_shared(smemB_left.index(1), b_base + b_k2, b_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_b_sc.index(1), b_scales_ptr + b_sc_k, b_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_bl = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                              b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_bl)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            b_right = smemB_right.index(1).permute([1, 0]).load(dot_b_layout)
            cdna4_async.buffer_load_to_shared(smemA_top.index(1), a_base + a_k2, a_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_a_sc_t.index(1), a_scales_ptr + a_sc_k, a_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tr = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                              b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_tr)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            b_left = smemB_left.index(0).permute([1, 0]).load(dot_b_layout)
            cdna4_async.buffer_load_to_shared(smemA_bot.index(1), a_base + a_half_m + a_k2, a_tile_offsets)
            cdna4_async.buffer_load_to_shared(smem_a_sc_b.index(1), a_scales_ptr + a_sc_half_m + a_sc_k, a_sc_offsets)
            cdna4_async.commit_group()

        cdna4_async.wait_group(5)
        with warp_pipeline_stage("mfma", priority=0):
            acc_br = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                              b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_br)
        with warp_pipeline_stage("mem", priority=1, fence_loads=True):
            a_top = smemA_top.index(0).load(dot_a_layout)
            a_sc_top = smem_a_sc_t.index(0).load(scale_a_layout)
            b_sc_left, b_sc_right = _bsc_load_split(smem_b_sc.index(0), scale_b_comb_layout, scale_b_layout, HN, NG)
            cdna4_async.buffer_load_to_shared(smemB_right.index(1), b_base + b_half_n + b_k2, b_tile_offsets)
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

    # iter iterMax-2 (b_sc_left/right for this step were prefetched at loop tail)
    acc_tl = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                      b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_tl)
    cdna4_async.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dot_a_layout)
    a_sc_bot = smem_a_sc_b.index(l_idx).load(scale_a_layout)

    acc_bl = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                      b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_bl)
    cdna4_async.wait_group(4)
    b_right = smemB_right.index(l_idx).permute([1, 0]).load(dot_b_layout)

    acc_tr = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                      b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_tr)
    cdna4_async.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).permute([1, 0]).load(dot_b_layout)

    acc_br = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                      b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_br)
    cdna4_async.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dot_a_layout)
    a_sc_top = smem_a_sc_t.index(g_idx).load(scale_a_layout)
    b_sc_left, b_sc_right = _bsc_load_split(smem_b_sc.index(g_idx), scale_b_comb_layout, scale_b_layout, HN, NG)

    # iter iterMax-1: complete ALL four accumulators FIRST, then convert + store.
    acc_tl = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                      b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_tl)
    cdna4_async.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dot_a_layout)
    a_sc_bot = smem_a_sc_b.index(g_idx).load(scale_a_layout)

    acc_bl = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                      b=b_left, b_scale=b_sc_left, b_format="e2m1", acc=acc_bl)
    cdna4_async.wait_group(0)
    b_right = smemB_right.index(g_idx).permute([1, 0]).load(dot_b_layout)

    acc_tr = gl.amd.cdna4.mfma_scaled(a=a_top, a_scale=a_sc_top, a_format="e2m1",
                                      b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_tr)
    acc_br = gl.amd.cdna4.mfma_scaled(a=a_bot, a_scale=a_sc_bot, a_format="e2m1",
                                      b=b_right, b_scale=b_sc_right, b_format="e2m1", acc=acc_br)

    c_tl = gl.convert_layout(acc_tl.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_tl, c_tl_base, c_quad_offsets)
    c_bl = gl.convert_layout(acc_bl.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_bl, c_bl_base, c_quad_offsets)
    c_tr = gl.convert_layout(acc_tr.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_tr, c_tr_base, c_quad_offsets)
    c_br = gl.convert_layout(acc_br.to(c_ptr.type.element_ty), layout=gStoreLayoutC, assert_trivial=False)
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
    v2_mfma32x32x64_BK256_nS2[(GRID_MN,)](
        a, b, c, a_scales, b_scales, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        WARPS_M=WARPS_M, WARPS_N=WARPS_N,
        GRID_MN=GRID_MN, NUM_XCDS=NUM_XCDS, GROUP_SIZE_M=GROUP_SIZE_M,
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
