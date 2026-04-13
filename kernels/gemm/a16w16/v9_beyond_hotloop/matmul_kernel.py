import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import extract_slice


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
def v9_beyond_hotloop(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
):
    """
    Beyond the hot loop: L2 locality + interleaved epilogue on top of v8_sliceMN.

    Builds on v8's slice-both-M-and-N design (4 quadrant accumulators, unrolled by 2,
    pre-computed _next offsets) and adds:

    1. XCD-aware PID remapping + GROUP_SIZE_M workgroup swizzling for L2 cache locality.
    2. Interleaved epilogue using extract_slice: each 128x128 accumulator is split into
       two 64x128 sub-tiles along M. Stores for sub-tile i are pipelined with the MFMA
       computing sub-tile i+1, spreading write traffic over time.
    """

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # Half-M global load layout
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K],
    )
    # Half-N global load layout
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

    # Half-M padded shared layout
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [],
        [BLOCK_M // 2, BLOCK_K],
    )
    # Half-N padded shared layout
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
        ],
        [],
        [BLOCK_K, BLOCK_N // 2],
    )

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Two sets of offsets: base (even K-steps) and _next (odd K-steps).
    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + BLOCK_M * stride_am // 2
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    ## Prologue — same as v8
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(g_idx), a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(g_idx), a_base, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base, a_bot_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base, b_right_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.assume(iterMax > 3)

    ## Main loop — same as v8
    for k in range(0, iterMax - 2, 2):

        ## =============================================================
        ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ## =============================================================
        ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
        ## into buffer 1. AC uses _next offsets (odd K-step).
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_base, b_right_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    ## =========================================================================
    ## Epilogue with interleaved stores via extract_slice
    ## =========================================================================
    ## Each 128x128 accumulator is split into two 64x128 sub-tiles along M.
    ## Stores for sub-tile i are pipelined with the MFMA computing sub-tile i+1.

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    offs_cm_slice = gl.arange(0, BLOCK_M // 4, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_slice_offsets = stride_cm * offs_cm_slice[:, None] + stride_cn * offs_cn[None, :]

    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn

    # Sub-tile base addresses for the 2x2 x 2 = 8 sub-tiles
    # top-left quadrant: m[0:64], m[64:128] x n[0:128]
    c_tl0_base = c_base
    c_tl1_base = c_base + 64 * stride_cm
    # bottom-left quadrant: m[128:192], m[192:256] x n[0:128]
    c_bl0_base = c_base + BLOCK_M * stride_cm // 2
    c_bl1_base = c_bl0_base + 64 * stride_cm
    # top-right quadrant: m[0:64], m[64:128] x n[128:256]
    c_tr0_base = c_base + BLOCK_N * stride_cn // 2
    c_tr1_base = c_tr0_base + 64 * stride_cm
    # bottom-right quadrant: m[128:192], m[192:256] x n[128:256]
    c_br0_base = c_base + BLOCK_M * stride_cm // 2 + BLOCK_N * stride_cn // 2
    c_br1_base = c_br0_base + 64 * stride_cm

    ## Iter iterMax - 2: same 4-region pattern as main loop, no AC
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr)
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br)
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

    ## Iter iterMax - 1: interleaved sub-tile MFMAs + stores

    ## Region 0: acc_tl sub-tiles, interleaved with stores
    a_top_0 = extract_slice(a_top, [64, 64], [0, 0])
    acc_tl_0 = extract_slice(acc_tl, [64, 128], [0, 0])
    acc_tl_0 = gl.amd.cdna3.mfma(a_top_0, b_left, acc_tl_0)

    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    a_top_1 = extract_slice(a_top, [64, 64], [64, 0])
    acc_tl_1 = extract_slice(acc_tl, [64, 128], [64, 0])
    acc_tl_1 = gl.amd.cdna3.mfma(a_top_1, b_left, acc_tl_1)

    c_tl0 = acc_tl_0.to(a_ptr.dtype.element_ty)
    c_tl0 = gl.convert_layout(c_tl0, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_tl0, ptr=c_tl0_base, offsets=c_slice_offsets)

    ## Region 1: acc_bl sub-tiles
    a_bot_0 = extract_slice(a_bot, [64, 64], [0, 0])
    acc_bl_0 = extract_slice(acc_bl, [64, 128], [0, 0])
    acc_bl_0 = gl.amd.cdna3.mfma(a_bot_0, b_left, acc_bl_0)

    c_tl1 = acc_tl_1.to(a_ptr.dtype.element_ty)
    c_tl1 = gl.convert_layout(c_tl1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_tl1, ptr=c_tl1_base, offsets=c_slice_offsets)

    a_bot_1 = extract_slice(a_bot, [64, 64], [64, 0])
    acc_bl_1 = extract_slice(acc_bl, [64, 128], [64, 0])
    acc_bl_1 = gl.amd.cdna3.mfma(a_bot_1, b_left, acc_bl_1)

    c_bl0 = acc_bl_0.to(a_ptr.dtype.element_ty)
    c_bl0 = gl.convert_layout(c_bl0, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_bl0, ptr=c_bl0_base, offsets=c_slice_offsets)

    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    ## Region 2: acc_tr sub-tiles
    acc_tr_0 = extract_slice(acc_tr, [64, 128], [0, 0])
    acc_tr_0 = gl.amd.cdna3.mfma(a_top_0, b_right, acc_tr_0)

    c_bl1 = acc_bl_1.to(a_ptr.dtype.element_ty)
    c_bl1 = gl.convert_layout(c_bl1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_bl1, ptr=c_bl1_base, offsets=c_slice_offsets)

    acc_tr_1 = extract_slice(acc_tr, [64, 128], [64, 0])
    acc_tr_1 = gl.amd.cdna3.mfma(a_top_1, b_right, acc_tr_1)

    c_tr0 = acc_tr_0.to(a_ptr.dtype.element_ty)
    c_tr0 = gl.convert_layout(c_tr0, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_tr0, ptr=c_tr0_base, offsets=c_slice_offsets)

    ## Region 3: acc_br sub-tiles
    acc_br_0 = extract_slice(acc_br, [64, 128], [0, 0])
    acc_br_0 = gl.amd.cdna3.mfma(a_bot_0, b_right, acc_br_0)

    c_tr1 = acc_tr_1.to(a_ptr.dtype.element_ty)
    c_tr1 = gl.convert_layout(c_tr1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_tr1, ptr=c_tr1_base, offsets=c_slice_offsets)

    acc_br_1 = extract_slice(acc_br, [64, 128], [64, 0])
    acc_br_1 = gl.amd.cdna3.mfma(a_bot_1, b_right, acc_br_1)

    c_br0 = acc_br_0.to(a_ptr.dtype.element_ty)
    c_br0 = gl.convert_layout(c_br0, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_br0, ptr=c_br0_base, offsets=c_slice_offsets)

    c_br1 = acc_br_1.to(a_ptr.dtype.element_ty)
    c_br1 = gl.convert_layout(c_br1, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c_br1, ptr=c_br1_base, offsets=c_slice_offsets)


def matmul(a, b, c=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    v9_beyond_hotloop[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )
    return c
