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
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        # When GRID_MN cannot divide NUM_XCDS, some xcds will have
        # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        # We calculate the number of xcds that have pids_per_xcd pids as
        # tall_xcds
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        # Note that we need to consider the following two cases:
        # 1. the current pid is on a tall xcd
        # 2. the current pid is on a short xcd
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
def v8_beyond_hotloop(
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
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,  #
):
    """
    Local prefetch pipeline design

    Prologue

        AC A0, B0{left, right} --> buffer 0
        AC A1, B1{left, right} --> buffer 1
        async_wait buffer 0
        local_load A0, B0{left} <-- buffer 0

    InLoop

        acc{left} = DOT(A0, B0{left})
        local_load B0{right} <-- buffer 0
        AC A2, B2{left} --> buffer 0

        acc{right} = DOT(A0, B0{right})
        async_wait buffer 1
        local_load A1, B1{left} <-- buffer 1
        AC B2{right} --> buffer 0

    Epilogue (iterMax - 2)

        acc{left} = DOT(A, B{left})
        local_load B{right}

        acc{right} = DOT(A, B{right})
        local_load A, B{left} from next buffer

    Epilogue (iterMax - 1)

        acc{left} = DOT(A, B{left})  (sliced, interleaved with stores)
        local_load B{right}
        store(acc{left} slices)

        acc{right} = DOT(A, B{right})  (sliced, interleaved with stores)
        store(acc{right} slices)
    """

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

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
            [128, 0],
        ],
        [],
        [BLOCK_M, BLOCK_K],
    )
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
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc_left = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_right = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ## AC A0, B0{left, right} --> buffer 0
    ## AC A1, B1{left, right} --> buffer 1
    ## async_wait buffer 0
    ## local_load A0, B0{left} <-- buffer 0
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    gl.amd.cdna4.async_copy.wait_group(3)
    l_idx = 0
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

    gl.assume(iterMax > 3)

    for k in range(0, iterMax - 2, 2):
        ## In loop
        ## g_idx: buffer id for async copy
        ## l_idx: buffer id for local load
        ##
        ## Now with local prefetch, 3 independent things are happening in parallel:
        ##   1. async copy is filling buffer g_idx
        ##   2. local load is consuming data from buffer l_idx
        ##   3. DOT is doing compute with data from the previous local load.

        ########################################
        ## Region 0
        ########################################
        g_idx = 0
        l_idx = 1

        acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)

        gl.amd.cdna4.async_copy.wait_group(2)
        b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smemB_right.index(g_idx), dotOpLayoutB
        )

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(g_idx), b_base, b_left_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1
        ########################################
        acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

        gl.amd.cdna4.async_copy.wait_group(2)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(g_idx), b_base, b_right_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        ## ---------------------------------------------------------------------
        ## Loop unroll separator
        ## ---------------------------------------------------------------------

        g_idx = 1
        l_idx = 0

        ########################################
        ## Region 2
        ########################################

        acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)

        gl.amd.cdna4.async_copy.wait_group(2)
        b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smemB_right.index(g_idx), dotOpLayoutB
        )

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(g_idx), b_base, b_left_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3
        ########################################
        acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

        gl.amd.cdna4.async_copy.wait_group(2)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(g_idx), b_base, b_right_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    ## Epilogue
    gl.amd.cdna4.async_copy.wait_group(0)

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))

    ## iterMax - 2

    ########################################
    ## Region 0
    ########################################
    g_idx = 0
    l_idx = 1
    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)
    b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

    ########################################
    ## Region 1
    ########################################
    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

    ## iterMax - 1

    ########################################
    ## Region 2
    ########################################
    g_idx = 1

    offs_cm_slice = gl.arange(0, BLOCK_M // 4, gl.SliceLayout(1, gStoreLayoutC))
    c_slice_offsets = stride_cm * offs_cm_slice[:, None] + stride_cn * offs_cn[None, :]
    c00_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c01_base = c00_base + 64 * stride_cm
    c02_base = c01_base + 64 * stride_cm
    c03_base = c02_base + 64 * stride_cm

    c10_base = c00_base + BLOCK_N * stride_cn // 2
    c11_base = c10_base + 64 * stride_cm
    c12_base = c11_base + 64 * stride_cm
    c13_base = c12_base + 64 * stride_cm

    ## slice 0 m[0:64]n[0:128]
    a0 = extract_slice(a, [64, 64], [0, 0])
    acc00 = extract_slice(acc_left, [64, 128], [0, 0])
    acc00 = gl.amd.cdna3.mfma(a0, b_left, acc00)
    b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

    ## slice 1 m[64:128]n[0:128]
    a1 = extract_slice(a, [64, 64], [64, 0])
    acc01 = extract_slice(acc_left, [64, 128], [64, 0])
    acc01 = gl.amd.cdna3.mfma(a1, b_left, acc01)

    c00 = acc00.to(a_ptr.dtype.element_ty)
    c00 = gl.convert_layout(c00, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c00, ptr=c00_base, offsets=c_slice_offsets)

    ## slice 2 m[128:192]n[0:128]
    a2 = extract_slice(a, [64, 64], [128, 0])
    acc02 = extract_slice(acc_left, [64, 128], [128, 0])
    acc02 = gl.amd.cdna3.mfma(a2, b_left, acc02)
    c01 = acc01.to(a_ptr.dtype.element_ty)
    c01 = gl.convert_layout(c01, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c01, ptr=c01_base, offsets=c_slice_offsets)

    ## slice 3 m[192:256]n[0:128]
    a3 = extract_slice(a, [64, 64], [192, 0])
    acc03 = extract_slice(acc_left, [64, 128], [192, 0])
    acc03 = gl.amd.cdna3.mfma(a3, b_left, acc03)
    c02 = acc02.to(a_ptr.dtype.element_ty)
    c02 = gl.convert_layout(c02, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c02, ptr=c02_base, offsets=c_slice_offsets)

    ########################################
    ## Region 3
    ########################################
    ## slice 0 m[0:64]n[128:256]
    acc10 = extract_slice(acc_right, [64, 128], [0, 0])
    acc10 = gl.amd.cdna3.mfma(a0, b_right, acc10)
    c03 = acc03.to(a_ptr.dtype.element_ty)
    c03 = gl.convert_layout(c03, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c03, ptr=c03_base, offsets=c_slice_offsets)

    ## slice 1 m[64:128]n[128:256]
    acc11 = extract_slice(acc_right, [64, 128], [64, 0])
    acc11 = gl.amd.cdna3.mfma(a1, b_right, acc11)
    c10 = acc10.to(a_ptr.dtype.element_ty)
    c10 = gl.convert_layout(c10, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c10, ptr=c10_base, offsets=c_slice_offsets)

    ## slice 2 m[128:192]n[128:256]
    acc12 = extract_slice(acc_right, [64, 128], [128, 0])
    acc12 = gl.amd.cdna3.mfma(a2, b_right, acc12)
    c11 = acc11.to(a_ptr.dtype.element_ty)
    c11 = gl.convert_layout(c11, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c11, ptr=c11_base, offsets=c_slice_offsets)

    ## slice 3 m[192:256]n[128:256]
    acc13 = extract_slice(acc_right, [64, 128], [192, 0])
    acc13 = gl.amd.cdna3.mfma(a3, b_right, acc13)
    c12 = acc12.to(a_ptr.dtype.element_ty)
    c12 = gl.convert_layout(c12, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c12, ptr=c12_base, offsets=c_slice_offsets)

    c13 = acc13.to(a_ptr.dtype.element_ty)
    c13 = gl.convert_layout(c13, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c13, ptr=c13_base, offsets=c_slice_offsets)


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
    v8_beyond_hotloop[grid](
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
