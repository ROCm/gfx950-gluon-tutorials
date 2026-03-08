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
def amxfp4wmxfp4_kernel(
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
    MXFP4 GEMM kernel in v3_lds_padding style.
    Uses DistributedLinearLayout + PaddedSharedLayout + async_copy.
    Single-buffered, no prefetch.

    A: (M, K//2) uint8, row-major (K-contiguous) → tiles [256, 128]
    B: (N, K//2) uint8, row-major (K-contiguous) → tiles [256, 128]
    A_scales: (M, K//32) uint8 e8m0
    B_scales: (N, K//32) uint8 e8m0
    C: (M, N) bfloat16
    """

    SCALE_GROUP_SIZE: gl.constexpr = 32

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # -- Global load layout (DistributedLinearLayout) --
    #
    # Both A [256, 128] and B [256, 128] use the same layout since B is now
    # K-contiguous (N, K//2) instead of N-contiguous (K//2, N).
    #
    # For uint8 [256, 128]:
    #   4 consecutive K reg_bases for 128b loads: [0,1],[0,2],[0,4],[0,8]
    #   3 lane K bits: [0,16],[0,32],[0,64]
    #   3 reg M/N bits: [4,0],[8,0],[128,0]
    #   3 lane M/N bits: [16,0],[32,0],[64,0]
    #   2 warp M/N bits: [1,0],[2,0]
    gLoadLayout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K // 2],
    )

    # Scale load layout: (256, 8)
    blocked_scales: gl.constexpr = gl.BlockedLayout(
        [8, 1], [32, 2], [1, 4], [0, 1],
    )

    # -- Shared memory layout (PaddedSharedLayout) --
    #
    # Same layout for both A and B since both are [256, 128] uint8 tiles.
    # For uint8: byte = linear index, so padding [[1024,16],[2048,32]]
    sharedLayout: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16], [2048, 32]],
        [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],
            [16, 0], [32, 0], [64, 0],
            [1, 0], [2, 0], [4, 0], [8, 0], [128, 0],
        ],
        [],
        [BLOCK_M, BLOCK_K // 2],
    )

    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

    # -- MFMA layouts --
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 2]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=16)
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)
    scale_a_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE]
    )

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    # -- SMEM allocation (single-buffered) --
    # Both A and B tiles are [256, 128] with the same shared layout.
    smemA = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K // 2], sharedLayout)
    smemB = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K // 2], sharedLayout)
    smem_as = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty, [BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE], shared_scales,
    )
    smem_bs = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty, [BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE], shared_scales,
    )

    # -- Offsets --
    # A: (M, K//2) row-major
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayout))
    offs_ak = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayout))
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    # B: (N, K//2) row-major (K-contiguous) — same tile shape [256, 128] as A
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(1, gLoadLayout))
    offs_bk = gl.arange(0, BLOCK_K // 2, gl.SliceLayout(0, gLoadLayout))
    b_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Scales
    offs_ks = gl.arange(0, BLOCK_K // SCALE_GROUP_SIZE, gl.SliceLayout(0, blocked_scales))
    offs_asm = (
        pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, blocked_scales))
    ) % M
    offs_bsn = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(1, blocked_scales))
    ) % N
    offs_as = offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    offs_bs = offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)

    # Permute smemB [N, K//2] → [K//2, N] for dot_b_layout compatibility
    smemB_T = smemB.permute([1, 0])

    # -- Main loop (v3 style: async_copy, no prefetch) --
    for k in range(0, gl.cdiv(K, BLOCK_K)):
        # Async copy A and B to shared (both [256, 128] tiles)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA, a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB, b_base, b_offsets)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.wait_group(0)

        # Load from shared to registers
        # A: smemA [M, K//2] → dot_a_layout
        # B: smemB_T [K//2, N] (permuted view) → dot_b_layout
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA, dot_a_layout)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_T, dot_b_layout)

        # Load scales (explicit buffer_load → smem → registers)
        a_sc = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as)
        b_sc = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs)
        smem_as.store(a_sc)
        smem_bs.store(b_sc)
        a_sc_reg = smem_as.load(layout=scale_a_layout)
        b_sc_reg = smem_bs.load(layout=scale_b_layout)

        # Compute
        acc = gl.amd.cdna4.mfma_scaled(
            a=a, a_scale=a_sc_reg, a_format="e2m1",
            b=b, b_scale=b_sc_reg, b_format="e2m1",
            acc=acc,
        )

        # Advance
        a_base += (BLOCK_K // 2) * stride_ak
        b_base += (BLOCK_K // 2) * stride_bk
        offs_as += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
        offs_bs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

    # -- Store output --
    c = acc.to(c_ptr.type.element_ty)
    c = gl.convert_layout(c, layout=gStoreLayoutC, assert_trivial=False)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
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

    amxfp4wmxfp4_kernel[grid](
        a, b, c, a_scales, b_scales,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GRID_MN=GRID_MN, NUM_XCDS=NUM_XCDS, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
    )
    return c
