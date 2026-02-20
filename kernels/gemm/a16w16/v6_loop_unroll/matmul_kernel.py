import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def v6_loop_unroll(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
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

        AC A0, B0 --> buffer 0
        AC A1, B1 --> buffer 1
        async_wait buffer 0
        local_load A0, B0 <-- buffer 0

    InLoop

        DOT(A0, B0)
        async_wait buffer 1
        local_load A1, B1 <-- buffer 1
        AC A2, B2 --> buffer 0

    Epilogue

        DOT(A0, B0)
        async_wait buffer 1
        local_load A1, B1 <-- buffer 1

        DOT(A1, B1)
        store(acc)
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
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 128]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N],
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
            [0, 128],
        ],
        [],
        [BLOCK_K, BLOCK_N],
    )

    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ## AC A0, B0 --> buffer 0
    ## AC A1, B1 --> buffer 1
    ## async_wait buffer 0
    ## local_load A0, B0 <-- buffer 0
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    gl.amd.cdna4.async_copy.wait_group(1)
    l_idx = 0
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

    for k in range(0, iterMax - 1, 2):
        ## In loop
        ## g_idx: buffer id for async copy
        ## l_idx: buffer id for local load
        ##
        ## Now with local prefetch, 3 independent things are happening in parallel:
        ##   1. async copy is filling buffer g_idx
        ##   2. local load is consuming data from buffer l_idx
        ##   3. DOT is doing compute with data from buffer g_idx.
        g_idx = 0
        l_idx = 1

        acc = gl.amd.cdna3.mfma(a, b, acc)

        gl.amd.cdna4.async_copy.wait_group(0)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA.index(g_idx), a_base, a_offsets, mask=(k != (iterMax - 2))
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB.index(g_idx), b_base, b_offsets, mask=(k != (iterMax - 2))
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        g_idx = 1
        l_idx = 0

        acc = gl.amd.cdna3.mfma(a_next, b_next, acc)

        gl.amd.cdna4.async_copy.wait_group(0)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA.index(g_idx), a_base, a_offsets, mask=(k != (iterMax - 2))
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB.index(g_idx), b_base, b_offsets, mask=(k != (iterMax - 2))
        )
        gl.amd.cdna4.async_copy.commit_group()

        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    ## Epilogue
    ## iterMax - 1
    acc = gl.amd.cdna3.mfma(a, b, acc)
    c = acc.to(a_ptr.dtype.element_ty)

    # gStoreLayoutC: gl.constexpr = mfmaLayout
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [2, 32], [4, 1], [1, 0])
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask)


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
    v6_loop_unroll[grid](
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
