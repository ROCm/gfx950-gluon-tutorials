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
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )

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
    MXFP4 GEMM kernel with local prefetch (v5 style).
    Double-buffered async_copy + local_load prefetch.

    Pipeline (3 things in parallel):
      1. async copy fills buffer g_idx (next+1 iteration's data)
      2. local load prefetches from buffer l_idx (next iteration's data)
      3. DOT computes with registers (current iteration's data)

    Prologue:
        AC A0, B0 --> buffer 0
        AC A1, B1 --> buffer 1
        wait buffer 0
        local_load A0, B0 <-- buffer 0
        load scales iter 0

    In loop:
        DOT(A, B, scales)
        wait buffer 1
        local_load A_next, B_next <-- buffer l_idx
        load scales for next iter
        AC A_next+1, B_next+1 --> buffer g_idx (masked on last iter)

    Epilogue:
        DOT(A, B, scales)
        wait last buffer
        local_load last A, B
        load last scales
        DOT(last A, last B, last scales)
        store(acc)

    A: (M, K//2) uint8, row-major (K-contiguous) -> tiles [256, 128]
    B: (N, K//2) uint8, row-major (K-contiguous) -> tiles [256, 128]
    A_scales: (M, K//32) uint8 e8m0
    B_scales: (N, K//32) uint8 e8m0
    C: (M, N) bfloat16
    """

    SCALE_GROUP_SIZE: gl.constexpr = 32

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # -- Global load layout (DistributedLinearLayout) --
    # Both A [256, 128] and B [256, 128] use the same layout.
    gLoadLayout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K // 2],
    )

    # Scale load layout: (256, 8)
    blocked_scales: gl.constexpr = gl.BlockedLayout(
        [8, 1],
        [32, 2],
        [1, 4],
        [0, 1],
    )

    # -- Shared memory layout (PaddedSharedLayout) --
    sharedLayout: gl.constexpr = gl.PaddedSharedLayout(
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
        [BLOCK_M, BLOCK_K // 2],
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
        dot_b_layout, [BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE]
    )

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    # -- SMEM allocation (double-buffered for A and B) --
    nBuffers: gl.constexpr = 2
    smemA = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [nBuffers, BLOCK_M, BLOCK_K // 2], sharedLayout
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [nBuffers, BLOCK_N, BLOCK_K // 2], sharedLayout
    )
    smem_as = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty,
        [BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE],
        shared_scales,
    )
    smem_bs = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty,
        [BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE],
        shared_scales,
    )

    # -- Offsets --
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayout))
    offs_ak = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayout))
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, gLoadLayout))
    offs_bk = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayout))
    b_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    offs_ks = gl.arange(
        0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales)
    )
    offs_asm = (
        pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, blocked_scales))
    ) % M
    offs_bsn = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(1, blocked_scales))
    ) % N
    offs_as = offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    offs_bs = offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)

    iterMax = gl.cdiv(K, BLOCK_K)

    # ====== Prologue ======
    # AC A0, B0 --> buffer 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA.index(0), a_base, a_offsets
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB.index(0), b_base, b_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    a_base += (BLOCK_K // 2) * stride_ak
    b_base += (BLOCK_K // 2) * stride_bk

    # AC A1, B1 --> buffer 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA.index(1), a_base, a_offsets
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB.index(1), b_base, b_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    a_base += (BLOCK_K // 2) * stride_ak
    b_base += (BLOCK_K // 2) * stride_bk

    # Wait for buffer 0, local_load A0, B0
    gl.amd.cdna4.async_copy.wait_group(1)
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(0), dot_a_layout)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smemB.index(0).permute([1, 0]), dot_b_layout
    )

    # Load scales for iter 0
    a_sc = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as)
    b_sc = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs)
    smem_as.store(a_sc)
    smem_bs.store(b_sc)
    a_sc_reg = smem_as.load(layout=scale_a_layout)
    b_sc_reg = smem_bs.load(layout=scale_b_layout)
    offs_as += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
    offs_bs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    # ====== Main loop ======
    for k in range(0, iterMax - 1):
        # 3 things in parallel:
        #   1. DOT with current a, b, scales (from prev local_load)
        #   2. local_load next iteration's data from buffer l_idx
        #   3. async copy next+1 iteration's data to buffer g_idx
        g_idx = k % 2
        l_idx = 1 - g_idx


        # DOT with current data
        acc = gl.amd.cdna4.mfma_scaled(
            a=a,
            a_scale=a_sc_reg,
            a_format="e2m1",
            b=b,
            b_scale=b_sc_reg,
            b_format="e2m1",
            acc=acc,
        )

        # Wait for next buffer and local_load
        gl.amd.cdna4.async_copy.wait_group(0)

        # Load scales
        a_sc = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as)
        b_sc = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs)

        #gl.amd.cdna3.sched_barrier(0)

        # Async copy next+1 iteration (masked on last iter)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA.index(g_idx), a_base, a_offsets, mask=(k != (iterMax - 2))
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB.index(g_idx), b_base, b_offsets, mask=(k != (iterMax - 2))
        )
        gl.amd.cdna4.async_copy.commit_group()

        a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smemA.index(l_idx), dot_a_layout
        )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            smemB.index(l_idx).permute([1, 0]), dot_b_layout
        )

        smem_as.store(a_sc)
        smem_bs.store(b_sc)
        #a_sc_reg = smem_as.load(layout=scale_a_layout)
        #b_sc_reg = smem_bs.load(layout=scale_b_layout)
        a_sc_reg = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_as, scale_a_layout)
        b_sc_reg = gl.amd.cdna4.async_copy.load_shared_relaxed(smem_bs, scale_b_layout)


        # Advance
        a_base += (BLOCK_K // 2) * stride_ak
        b_base += (BLOCK_K // 2) * stride_bk
        offs_as += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
        offs_bs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    # ====== Epilogue ======
    # Last DOT
    acc = gl.amd.cdna4.mfma_scaled(
        a=a,
        a_scale=a_sc_reg,
        a_format="e2m1",
        b=b,
        b_scale=b_sc_reg,
        b_format="e2m1",
        acc=acc,
    )

    # -- Store output --
    c = acc.to(c_ptr.type.element_ty)
    c = gl.convert_layout(c, layout=gStoreLayoutC, assert_trivial=False)
    offs_cm = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC)
    )
    offs_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC)
    )
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna4.buffer_store(c, c_ptr, offs_c, c_mask)


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
