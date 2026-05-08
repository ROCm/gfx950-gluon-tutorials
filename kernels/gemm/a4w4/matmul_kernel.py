##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
        ## pid remapping on xcds
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
def a4w4_kernel(
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
    MXFP4 GEMM kernel with N-slice and prefetched scales.
    B tile and B scales split along N dim by half.
    Double-buffered async_copy for tiles + double-buffered scale prefetch.

    Naming convention:
      - Tiles use smem buffer index 0/1 (double-buffered async_copy)
      - Scales use register buffers named buf0/buf1/buf2/buf3:
        buf1/buf3 = raw buffer_load results (prefetched, waiting for use)
        buf0/buf2 = scale registers after store-to-shared + load-from-shared

    Pipeline (step 2 loop, unrolled by 2):

      Prologue:
        iter 0:
          AC A0, B0_left + GR a_sc, b_sc_left --> commit group 0, scales -> buf1
          AC B0_right    + GR b_sc_right       --> commit group 1, scales -> buf1
          advance tile/scale ptrs
        iter 1:
          AC A1, B1_left + GR a_sc, b_sc_left --> commit group 2, scales -> buf3
          AC B1_right    + GR b_sc_right       --> commit group 3, scales -> buf3
          advance tile/scale ptrs
        wait_group(3), local_load A0/B0_left, store/load scales -> buf0

      Main loop (4 regions per 2 K iterations):
        Region 0: DOT_left(buf0)  | wait | LL B_right, store b_sc -> buf0 | AC A+B_left + GR scales -> buf1
        Region 1: DOT_right(buf0) | wait | LL A+B_left, store scales -> buf2 | AC B_right + GR b_sc -> buf1
                  advance ptrs
        Region 2: DOT_left(buf2)  | wait | LL B_right, store b_sc -> buf2 | AC A+B_left + GR scales -> buf3
        Region 3: DOT_right(buf2) | wait | LL A+B_left, store scales -> buf0 | AC B_right + GR b_sc -> buf3
                  advance ptrs

      Epilogue (last 2 iterations, no new AC):
        Region 0: DOT_left(buf0)  | wait | LL B_right, store b_sc -> buf0
        Region 1: DOT_right(buf0) | wait | LL A+B_left, store scales -> buf2
        Region 2: DOT_left(buf2)  | wait(0) | LL B_right, store b_sc -> buf2
                  store acc_left
        Region 3: DOT_right(buf2)
                  store acc_right

    A: (M, K//2) uint8, row-major (K-contiguous) -> tiles [256, 128]
    B: (N, K//2) uint8, row-major (K-contiguous) -> tiles [128, 128] (left/right halves)
    A_scales: (M, K//32) uint8 e8m0
    B_scales: (N, K//32) uint8 e8m0
    C: (M, N) bfloat16
    """

    SCALE_GROUP_SIZE: gl.constexpr = 32

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # -- Global load layout for A: [BLOCK_M, BLOCK_K//2] = [256, 128] --
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K // 2],
    )

    # -- Global load layout for B half: [BLOCK_N//2, BLOCK_K//2] = [128, 128] --
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N // 2, BLOCK_K // 2],
    )

    # Scale load layout for A: (256, 8)
    blocked_scales_a: gl.constexpr = gl.BlockedLayout(
        [8, 1],
        [32, 2],
        [1, 4],
        [0, 1],
    )

    # Scale load layout for B half: (128, 8)
    blocked_scales_b: gl.constexpr = gl.BlockedLayout(
        [4, 1],
        [32, 2],
        [1, 4],
        [0, 1],
    )

    # -- Shared memory layout for A: [BLOCK_M, BLOCK_K//2] = [256, 128] --
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
            [128, 0],
        ],
        [],
        [BLOCK_M, BLOCK_K // 2],
    )

    # -- Shared memory layout for B half: [BLOCK_N//2, BLOCK_K//2] = [128, 128] --
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

    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    # -- MFMA layouts --
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 2]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    scale_a_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N // 2, BLOCK_K // SCALE_GROUP_SIZE]
    )

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    # -- SMEM allocation --
    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [nBuffers, BLOCK_M, BLOCK_K // 2], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [nBuffers, BLOCK_N // 2, BLOCK_K // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [nBuffers, BLOCK_N // 2, BLOCK_K // 2], sharedLayoutB
    )
    smem_as = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty,
        [BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE],
        shared_scales,
    )
    smem_bs = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty,
        [BLOCK_N // 2, BLOCK_K // SCALE_GROUP_SIZE],
        shared_scales,
    )

    # -- Tile offsets --
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutA))
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_offsets_next = a_offsets + (BLOCK_K // 2) * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayoutB))
    b_left_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_right_offsets = b_left_offsets + (BLOCK_N // 2) * stride_bn
    b_left_offsets_next = b_left_offsets + (BLOCK_K // 2) * stride_bk
    b_right_offsets_next = b_right_offsets + (BLOCK_K // 2) * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # -- Scale offsets (fixed; base ptrs are advanced instead) --
    offs_ks = gl.arange(0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales_a))
    offs_asm = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, blocked_scales_a))) % M
    offs_as = offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    offs_as_next = offs_as + (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask

    offs_ks_b = gl.arange(0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales_b))
    offs_bsn_half = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N // 2, gl.SliceLayout(1, blocked_scales_b))
    ) % N
    offs_bs_left = offs_bsn_half[:, None] * stride_bsn + offs_ks_b[None, :] * stride_bsk
    offs_bs_right = offs_bs_left + (BLOCK_N // 2) * stride_bsn
    offs_bs_left_next = offs_bs_left + (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk
    offs_bs_right_next = offs_bs_right + (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    acc_left = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfma_layout)
    acc_right = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfma_layout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # ====== Prologue: prefetch iter 0 and iter 1 ======

    # -- iter 0: AC tiles + GR scales --
    # commit group 0: A0 + B0_left tiles, a_sc + b_sc_left scales
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(0), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
    a_sc_buf1 = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as)
    b_sc_left_buf1 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_left)
    gl.amd.cdna4.async_copy.commit_group()

    # commit group 1: B0_right tile, b_sc_right scale
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
    b_sc_right_buf1 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_right)
    gl.amd.cdna4.async_copy.commit_group()

    # -- iter 1: AC tiles + GR scales --
    # commit group 2: A1 + B1_left tiles, a_sc + b_sc_left scales
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(1), a_base, a_offsets_next)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_left_offsets_next)
    a_sc_buf3 = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as_next)
    b_sc_left_buf3 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_left_next)
    gl.amd.cdna4.async_copy.commit_group()

    # commit group 3: B1_right tile, b_sc_right scale
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(1), b_base, b_right_offsets_next
    )
    b_sc_right_buf3 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_right_next)
    gl.amd.cdna4.async_copy.commit_group()

    a_base += (BLOCK_K // 2) * stride_ak * 2
    b_base += (BLOCK_K // 2) * stride_bk * 2
    a_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask * 2
    b_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk * 2

    # -- Wait for iter 0 tiles (commit group 0) and prepare registers --
    gl.amd.cdna4.async_copy.wait_group(3)
    a = smemA.index(0).load(dot_a_layout)
    b_left = smemB_left.index(0).permute([1, 0]).load(dot_b_layout)

    smem_as.store(a_sc_buf1)
    smem_bs.store(b_sc_left_buf1)
    a_sc_reg_buf0 = smem_as.load(scale_a_layout)
    b_sc_left_reg_buf0 = smem_bs.load(scale_b_layout)

    gl.assume(iterMax > 3)

    # ====== Main loop (step 2, unrolled by 2) ======
    for k in range(0, iterMax - 2, 2):

        ########################################
        ## Region 0 (even iter, buffer 0 -> 1)
        ########################################
        g_idx = 0
        l_idx = 1

        # DOT_left
        acc_left = gl.amd.cdna4.mfma_scaled(
            a=a,
            a_scale=a_sc_reg_buf0,
            a_format="e2m1",
            b=b_left,
            b_scale=b_sc_left_reg_buf0,
            b_format="e2m1",
            acc=acc_left,
        )

        # Wait for B_right tile (commit group 1)
        gl.amd.cdna4.async_copy.wait_group(2)
        b_right = smemB_right.index(g_idx).permute([1, 0]).load(dot_b_layout)

        # Prepare b_sc_right from prefetched buf1
        smem_bs.store(b_sc_right_buf1)
        b_sc_right_reg_buf0 = smem_bs.load(scale_b_layout)

        # AC next A + B_left tiles + GR next a_sc + b_sc_left scales
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(g_idx), b_base, b_left_offsets
        )
        a_sc_buf1 = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as)
        b_sc_left_buf1 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_left)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1 (even iter, buffer 0 -> 1)
        ########################################

        # DOT_right
        acc_right = gl.amd.cdna4.mfma_scaled(
            a=a,
            a_scale=a_sc_reg_buf0,
            a_format="e2m1",
            b=b_right,
            b_scale=b_sc_right_reg_buf0,
            b_format="e2m1",
            acc=acc_right,
        )

        # Wait for next A + B_left tiles (commit group 2)
        gl.amd.cdna4.async_copy.wait_group(2)
        a_next = smemA.index(l_idx).load(dot_a_layout)
        b_left = smemB_left.index(l_idx).permute([1, 0]).load(dot_b_layout)

        # Prepare next a_sc + b_sc_left from prefetched buf3
        smem_as.store(a_sc_buf3)
        smem_bs.store(b_sc_left_buf3)
        a_sc_reg_buf2 = smem_as.load(scale_a_layout)
        b_sc_left_reg_buf2 = smem_bs.load(scale_b_layout)

        # AC next B_right tile + GR next b_sc_right scale
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(g_idx), b_base, b_right_offsets
        )
        b_sc_right_buf1 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_right)
        gl.amd.cdna4.async_copy.commit_group()

        ## -- loop unrolling --

        g_idx = 1
        l_idx = 0

        ########################################
        ## Region 2 (odd iter, buffer 1 -> 0)
        ########################################

        # DOT_left
        acc_left = gl.amd.cdna4.mfma_scaled(
            a=a_next,
            a_scale=a_sc_reg_buf2,
            a_format="e2m1",
            b=b_left,
            b_scale=b_sc_left_reg_buf2,
            b_format="e2m1",
            acc=acc_left,
        )

        # Wait for B_right tile
        gl.amd.cdna4.async_copy.wait_group(2)
        b_right = smemB_right.index(g_idx).permute([1, 0]).load(dot_b_layout)

        # Prepare b_sc_right from prefetched buf3
        smem_bs.store(b_sc_right_buf3)
        b_sc_right_reg_buf2 = smem_bs.load(scale_b_layout)

        # AC next A + B_left tiles + GR next a_sc + b_sc_left scales
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets_next)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(g_idx), b_base, b_left_offsets_next
        )
        a_sc_buf3 = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as_next)
        b_sc_left_buf3 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_left_next)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3 (odd iter, buffer 1 -> 0)
        ########################################

        # DOT_right
        acc_right = gl.amd.cdna4.mfma_scaled(
            a=a_next,
            a_scale=a_sc_reg_buf2,
            a_format="e2m1",
            b=b_right,
            b_scale=b_sc_right_reg_buf2,
            b_format="e2m1",
            acc=acc_right,
        )

        # Wait for next A + B_left tiles
        gl.amd.cdna4.async_copy.wait_group(2)
        a = smemA.index(l_idx).load(dot_a_layout)
        b_left = smemB_left.index(l_idx).permute([1, 0]).load(dot_b_layout)

        # Prepare next a_sc + b_sc_left from prefetched buf1
        smem_as.store(a_sc_buf1)
        smem_bs.store(b_sc_left_buf1)
        a_sc_reg_buf0 = smem_as.load(scale_a_layout)
        b_sc_left_reg_buf0 = smem_bs.load(scale_b_layout)

        # AC next B_right tile + GR next b_sc_right scale
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(g_idx), b_base, b_right_offsets_next
        )
        b_sc_right_buf3 = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs_right_next)
        gl.amd.cdna4.async_copy.commit_group()

        a_base += (BLOCK_K // 2) * stride_ak * 2
        b_base += (BLOCK_K // 2) * stride_bk * 2
        a_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask * 2
        b_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk * 2

    # ====== Epilogue: last 2 iterations (no new async copies) ======

    # -- Output offset setup --
    offs_cn_left = pid_n * BLOCK_N + gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr

    # -- iterMax - 2 --

    ########################################
    ## Epilogue Region 0
    ########################################
    g_idx = 0
    l_idx = 1

    acc_left = gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_sc_reg_buf0,
        a_format="e2m1",
        b=b_left,
        b_scale=b_sc_left_reg_buf0,
        b_format="e2m1",
        acc=acc_left,
    )

    gl.amd.cdna4.async_copy.wait_group(2)
    b_right = smemB_right.index(g_idx).permute([1, 0]).load(dot_b_layout)

    smem_bs.store(b_sc_right_buf1)
    b_sc_right_reg_buf0 = smem_bs.load(scale_b_layout)

    ########################################
    ## Epilogue Region 1
    ########################################
    acc_right = gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_sc_reg_buf0,
        a_format="e2m1",
        b=b_right,
        b_scale=b_sc_right_reg_buf0,
        b_format="e2m1",
        acc=acc_right,
    )

    gl.amd.cdna4.async_copy.wait_group(1)
    a_next = smemA.index(l_idx).load(dot_a_layout)
    b_left = smemB_left.index(l_idx).permute([1, 0]).load(dot_b_layout)

    smem_as.store(a_sc_buf3)
    smem_bs.store(b_sc_left_buf3)
    a_sc_reg_buf2 = smem_as.load(scale_a_layout)
    b_sc_left_reg_buf2 = smem_bs.load(scale_b_layout)

    # -- iterMax - 1 --

    ########################################
    ## Epilogue Region 2: final acc_left MFMA + store
    ########################################
    g_idx = 1

    # Output offsets for the full BLOCK_M x (BLOCK_N // 2) tile
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    c_left_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn_left[None, :]
    c_right_offsets = c_left_offsets + (BLOCK_N // 2) * stride_cn
    c_tile_base = c_base + pid_m * BLOCK_M * stride_cm

    acc_left = gl.amd.cdna4.mfma_scaled(
        a=a_next,
        a_scale=a_sc_reg_buf2,
        a_format="e2m1",
        b=b_left,
        b_scale=b_sc_left_reg_buf2,
        b_format="e2m1",
        acc=acc_left,
    )

    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).permute([1, 0]).load(dot_b_layout)
    smem_bs.store(b_sc_right_buf3)
    b_sc_right_reg_buf2 = smem_bs.load(scale_b_layout)

    c_left = acc_left.to(c_ptr.type.element_ty)
    c_left = gl.convert_layout(c_left, layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_left, c_tile_base, c_left_offsets)

    ########################################
    ## Epilogue Region 3: final acc_right MFMA + store
    ########################################
    acc_right = gl.amd.cdna4.mfma_scaled(
        a=a_next,
        a_scale=a_sc_reg_buf2,
        a_format="e2m1",
        b=b_right,
        b_scale=b_sc_right_reg_buf2,
        b_format="e2m1",
        acc=acc_right,
    )

    c_right = acc_right.to(c_ptr.type.element_ty)
    c_right = gl.convert_layout(c_right, layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c_right, c_tile_base, c_right_offsets)


def matmul(a, b, a_scales, b_scales):
    # A: (M, K//2) uint8, K-contiguous
    # B: (N, K//2) uint8, K-contiguous
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

    a4w4_kernel[grid](
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
