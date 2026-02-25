import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def v7_slice(
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

        acc{left} = DOT(A, B{left})
        local_load B{right}
        store(acc{left})

        acc{right} = DOT(A, B{right})
        store(acc{right})
    """

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

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
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1
        ########################################
        acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

        gl.amd.cdna4.async_copy.wait_group(2)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
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
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_left.index(g_idx), b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3
        ########################################
        acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

        gl.amd.cdna4.async_copy.wait_group(2)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b_left = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_left.index(l_idx), dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB_right.index(g_idx), b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    ## Epilogue
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_left_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_right_offsets = c_left_offsets + BLOCK_N * stride_cn // 2

    ## iterMax - 2

    ########################################
    ## Region 0
    ########################################
    g_idx = 0
    l_idx = 1
    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)
    gl.amd.cdna4.async_copy.wait_group(0)
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

    acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)
    b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

    c_left = acc_left.to(a_ptr.dtype.element_ty)
    c_left = gl.convert_layout(c_left, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_left_offsets, stored_value=c_left)

    ########################################
    ## Region 3
    ########################################
    acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

    c_right = acc_right.to(a_ptr.dtype.element_ty)
    c_right = gl.convert_layout(c_right, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_right_offsets, stored_value=c_right)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    num_warps = 4
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    v7_slice[grid](
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
        num_warps=num_warps,
    )
    return c
