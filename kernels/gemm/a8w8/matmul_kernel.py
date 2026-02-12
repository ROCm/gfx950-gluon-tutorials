import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import extract_slice, sched_barrier
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy


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
        # 1. the currnt pid is on a tall xcd
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
def matmul_kernel(
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

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b0_init_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    b0_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_K * stride_bk
    b1_offsets = (
        offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn + BLOCK_N * stride_bn // 2
    )

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        tiles_per_warp=[2, 2],
        warps_per_cta=[2, 2],
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=32)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=32)

    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16], [2048, 32]],
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
        [BLOCK_M, BLOCK_K],
    )
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16], [2048, 32]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
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
    smemB0 = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB1 = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    acc0 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc1 = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax: gl.constexpr = gl.cdiv(K, BLOCK_K)

    ## Prologue
    ##
    ## AC A[0], B0[0], B1[0] --> buffer 0
    ## AC A[1], B0[1], B1[1] --> buffer 1
    ## async_wait buffer 0
    ## local_load (A+B0)[0] <-- buffer 0
    ##
    ## InLoop
    ##
    ## DOT(A, B0)[0]
    ## local_load B1[0] <-- buffer 0
    ## AC (A+B0)[2] --> buffer 0
    ##
    ## DOT(A, B1)[0]
    ## async_wait buffer 1
    ## local_load (A+B0)[1] <-- buffer 1
    ## AC B1[2] --> buffer 0
    ##
    ## Epilogue
    ##
    ## local_load B1[n-1]
    ## DOT(A{n-1}, B0{n-1})
    ## store(acc0)
    ## DOT(A{n-1}, B1{n-1})
    ## store(acc1)

    ## Prologue
    g_idx = 0
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_init_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    g_idx = 1
    cdna4_async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    cdna4_async_copy.buffer_load_to_shared(smemB0.index(g_idx), b_base, b0_offsets)
    cdna4_async_copy.commit_group()

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    cdna4_async_copy.buffer_load_to_shared(smemB1.index(g_idx), b_base, b1_offsets)
    cdna4_async_copy.commit_group()

    cdna4_async_copy.wait_group(3)
    l_idx = 0
    a = cdna4_async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(l_idx), dotOpLayoutB)

    cdna4_async_copy.wait_group(2)
    for k in range(0, iterMax - 1, 2):

        sched_barrier(0)

        ## DOT(A, B0)[0]
        ## LR B1[0]
        ## AC (A+B0)[2]
        acc0 = gl.amd.cdna4.mfma_scaled(a, None, "e5m2", b0, None, "e5m2", acc0)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(
            smemA.index(0), a_base, a_offsets, mask=(k != (iterMax - 2))
        )
        cdna4_async_copy.buffer_load_to_shared(
            smemB0.index(0), b_base, b0_offsets, mask=(k != (iterMax - 2))
        )
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        ## DOT(A, B1)[0]
        ## LR (A+B0)[1]
        ## AC B1[2]
        cdna4_async_copy.wait_group(2)
        acc1 = gl.amd.cdna4.mfma_scaled(a, None, "e5m2", b1, None, "e5m2", acc1)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(1), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(
            smemB1.index(0), b_base, b1_offsets, mask=(k != (iterMax - 2))
        )
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        ## ---------------------------------------------------------------------
        ## Loop unroll separator
        ## ---------------------------------------------------------------------
        ## DOT(A, B0)[1]
        ## LR B1[1]
        ## AC (A+B0)[3]
        cdna4_async_copy.wait_group(2)
        acc0 = gl.amd.cdna4.mfma_scaled(a, None, "e5m2", b0, None, "e5m2", acc0)
        b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(1), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(
            smemA.index(1), a_base, a_offsets, mask=(k != (iterMax - 2))
        )
        cdna4_async_copy.buffer_load_to_shared(
            smemB0.index(1), b_base, b0_offsets, mask=(k != (iterMax - 2))
        )
        cdna4_async_copy.commit_group()

        sched_barrier(0)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

        ## DOT(A, B1)[1]
        ## LR (A+B0)[1]
        ## AC B1[3]
        cdna4_async_copy.wait_group(2)
        acc1 = gl.amd.cdna4.mfma_scaled(a, None, "e5m2", b1, None, "e5m2", acc1)
        a = cdna4_async_copy.load_shared_relaxed(smemA.index(0), dotOpLayoutA)
        b0 = cdna4_async_copy.load_shared_relaxed(smemB0.index(0), dotOpLayoutB)

        cdna4_async_copy.buffer_load_to_shared(
            smemB1.index(1), b_base, b1_offsets, mask=(k != (iterMax - 2))
        )
        cdna4_async_copy.commit_group()

        cdna4_async_copy.wait_group(2)
        sched_barrier(0)

    cdna4_async_copy.wait_group(0)

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    offs_cm_slice = gl.arange(0, BLOCK_M // 4, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))

    c00_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c01_base = c00_base + 64 * stride_cm
    c02_base = c01_base + 64 * stride_cm
    c03_base = c02_base + 64 * stride_cm
    c_slice_offsets = stride_cm * offs_cm_slice[:, None] + stride_cn * offs_cn[None, :]

    c10_base = c00_base + BLOCK_N * stride_cn // 2
    c11_base = c10_base + 64 * stride_cm
    c12_base = c11_base + 64 * stride_cm
    c13_base = c12_base + 64 * stride_cm

    l_idx = (iterMax - 1) % 2

    # sched_barrier(0)

    ## slice 0 m[0:128]n[0:128]
    a0 = extract_slice(a, [64, 128], [0, 0])
    acc00 = extract_slice(acc0, [64, 128], [0, 0])
    acc00 = gl.amd.cdna4.mfma_scaled(a0, None, "e5m2", b0, None, "e5m2", acc00)
    c00 = acc00.to(tl.float16)
    c00 = gl.convert_layout(c00, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c00, ptr=c00_base, offsets=c_slice_offsets)

    ## slice 1 m[64:128]n[0:128]
    a1 = extract_slice(a, [64, 128], [64, 0])
    acc01 = extract_slice(acc0, [64, 128], [64, 0])
    acc01 = gl.amd.cdna4.mfma_scaled(a1, None, "e5m2", b0, None, "e5m2", acc01)
    c01 = acc01.to(tl.float16)
    c01 = gl.convert_layout(c01, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c01, ptr=c01_base, offsets=c_slice_offsets)

    ## slice 2 m[128:192]n[0:128]
    a2 = extract_slice(a, [64, 128], [128, 0])
    acc02 = extract_slice(acc0, [64, 128], [128, 0])
    acc02 = gl.amd.cdna4.mfma_scaled(a2, None, "e5m2", b0, None, "e5m2", acc02)
    c02 = acc02.to(tl.float16)
    c02 = gl.convert_layout(c02, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c02, ptr=c02_base, offsets=c_slice_offsets)

    ## slice 3 m[192:256]n[0:128]
    a3 = extract_slice(a, [64, 128], [192, 0])
    acc03 = extract_slice(acc0, [64, 128], [192, 0])
    acc03 = gl.amd.cdna4.mfma_scaled(a3, None, "e5m2", b0, None, "e5m2", acc03)
    c03 = acc03.to(tl.float16)
    c03 = gl.convert_layout(c03, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c03, ptr=c03_base, offsets=c_slice_offsets)

    ## We sync at the beginning of the epilogue
    b1 = cdna4_async_copy.load_shared_relaxed(smemB1.index(l_idx), dotOpLayoutB)
    # c0 = acc0.to(tl.float16)
    # c0 = gl.convert_layout(c0, layout=gStoreLayoutC)

    # sched_barrier(0)

    ## slice 0 m[0:64]n[128:256]
    # a0 = extract_slice(a, [64, 64], [0, 0])
    acc10 = extract_slice(acc1, [64, 128], [0, 0])
    acc10 = gl.amd.cdna4.mfma_scaled(a0, None, "e5m2", b1, None, "e5m2", acc10)
    c10 = acc10.to(tl.float16)
    c10 = gl.convert_layout(c10, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c10, ptr=c10_base, offsets=c_slice_offsets)

    ## slice 1 m[64:128]n[128:256]
    # a1 = extract_slice(a, [64, 64], [64, 0])
    acc11 = extract_slice(acc1, [64, 128], [64, 0])
    acc11 = gl.amd.cdna4.mfma_scaled(a1, None, "e5m2", b1, None, "e5m2", acc11)
    c11 = acc11.to(tl.float16)
    c11 = gl.convert_layout(c11, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c11, ptr=c11_base, offsets=c_slice_offsets)

    ## slice 2 m[128:192]n[128:256]
    # a2 = extract_slice(a, [64, 64], [128, 0])
    acc12 = extract_slice(acc1, [64, 128], [128, 0])
    acc12 = gl.amd.cdna4.mfma_scaled(a2, None, "e5m2", b1, None, "e5m2", acc12)
    c12 = acc12.to(tl.float16)
    c12 = gl.convert_layout(c12, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c12, ptr=c12_base, offsets=c_slice_offsets)

    ## slice 3 m[192:256]n[128:256]
    # a3 = extract_slice(a, [64, 64], [192, 0])
    acc13 = extract_slice(acc1, [64, 128], [192, 0])
    acc13 = gl.amd.cdna4.mfma_scaled(a3, None, "e5m2", b1, None, "e5m2", acc13)
    c13 = acc13.to(tl.float16)
    c13 = gl.convert_layout(c13, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(stored_value=c13, ptr=c13_base, offsets=c_slice_offsets)
    gl.amd.cdna3.buffer_store(stored_value=c13, ptr=c13_base, offsets=c_slice_offsets)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 128
    num_warps = 4
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    matmul_kernel[grid](
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
