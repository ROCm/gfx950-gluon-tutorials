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
    stride_bk,
    stride_bn,
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
    MXFP4 GEMM kernel: C = A @ B
    A is (M, K//2) packed uint8 in e2m1 format, row-major
    B is (K//2, N) packed uint8 in e2m1 format, column-major
    A_scales is (M, K//32) in e8m0 format
    B_scales is (N, K//32) in e8m0 format
    C is (M, N) in bfloat16

    Uses explicit buffer_load/store with manual 2-stage double-buffer pipeline.
    Based on the aiter gluon kernel with fixed AMDMFMALayout API.
    """

    SCALE_GROUP_SIZE: gl.constexpr = 32

    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # -- Layouts --

    # Global load layout for A: (BLOCK_M, BLOCK_K // 2) packed elements
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[8, 1],
        order=[1, 0],
    )

    # Global load layout for B: (BLOCK_K // 2, BLOCK_N) packed elements
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 8],
        order=[0, 1],
    )

    # Global load layout for scales
    blocked_scales: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 8],
        order=[0, 1],
    )

    # Shared memory layouts
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
    shared_scales: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0, 1]
    )

    # MFMA layout: 3D instr_shape for CDNA4, 8 warps = [2, 4]
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[2, 4]
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    # Scale layouts: auto-generated to match MFMA thread assignment
    scale_a_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE]
    )
    scale_b_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE]
    )

    # Output store layout
    gStoreLayoutC: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [8, 1], [1, 0]
    )

    # -- Offset computation --

    # A offsets: (BLOCK_M, BLOCK_K // 2) — packed uint8
    offs_ak = gl.arange(0, BLOCK_K // 2, layout=gl.SliceLayout(0, blocked_mk))
    offs_am = (
        pid_m * BLOCK_M
        + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk))
    ) % M
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak

    # B offsets: (BLOCK_K // 2, BLOCK_N) — packed uint8
    offs_bk = gl.arange(0, BLOCK_K // 2, layout=gl.SliceLayout(1, blocked_kn))
    offs_bn = (
        pid_n * BLOCK_N
        + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kn))
    ) % N
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # Scale offsets: (M, K // SCALE_GROUP_SIZE) and (N, K // SCALE_GROUP_SIZE)
    offs_ks = gl.arange(
        0, BLOCK_K // SCALE_GROUP_SIZE, layout=gl.SliceLayout(0, blocked_scales)
    )
    offs_asm = (
        pid_m * BLOCK_M
        + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_scales))
    ) % M
    offs_bsn = (
        pid_n * BLOCK_N
        + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, blocked_scales))
    ) % N

    offs_as = offs_asm[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    offs_bs = offs_bsn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    # -- Shared memory allocation --
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_M, BLOCK_K // 2], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_K // 2, BLOCK_N], layout=shared_b
    )
    smem_as = gl.allocate_shared_memory(
        a_scales_ptr.type.element_ty,
        [BLOCK_M, BLOCK_K // SCALE_GROUP_SIZE],
        layout=shared_scales,
    )
    smem_bs = gl.allocate_shared_memory(
        b_scales_ptr.type.element_ty,
        [BLOCK_N, BLOCK_K // SCALE_GROUP_SIZE],
        layout=shared_scales,
    )

    # -- Prologue: load first tile from global to registers --
    a = gl.amd.cdna4.buffer_load(ptr=a_ptr, offsets=offs_a)
    a_scales = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as)
    b = gl.amd.cdna4.buffer_load(ptr=b_ptr, offsets=offs_b)
    b_scales = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs)

    smem_as.store(a_scales)
    smem_a.store(a)

    accumulator = gl.zeros(
        (BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )

    num_k_iter = gl.cdiv(K, BLOCK_K)

    # -- Main loop: 2-stage pipeline --
    for k in range(0, num_k_iter - 1):
        # Advance pointers for next iteration
        a_ptr += (BLOCK_K // 2) * stride_ak
        b_ptr += (BLOCK_K // 2) * stride_bk
        a_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
        b_scales_ptr += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

        # Load next A tile while processing current
        a = gl.amd.cdna4.buffer_load(ptr=a_ptr, offsets=offs_a)

        # Store current B and B_scales to SMEM
        smem_b.store(b)
        smem_bs.store(b_scales)

        # Load current tiles from SMEM to registers in dot layout
        curr_a = smem_a.load(layout=dot_a_layout)
        curr_a_scales = smem_as.load(layout=scale_a_layout)

        # Load next A scales
        a_scales = gl.amd.cdna4.buffer_load(ptr=a_scales_ptr, offsets=offs_as)

        curr_b_scales = smem_bs.load(layout=scale_b_layout)

        # Load next B and B scales
        b = gl.amd.cdna4.buffer_load(ptr=b_ptr, offsets=offs_b)
        b_scales = gl.amd.cdna4.buffer_load(ptr=b_scales_ptr, offsets=offs_bs)

        curr_b = smem_b.load(layout=dot_b_layout)

        # Scaled MFMA: accumulate with e2m1 format and e8m0 scales
        accumulator = gl.amd.cdna4.mfma_scaled(
            a=curr_a,
            a_scale=curr_a_scales,
            a_format="e2m1",
            b=curr_b,
            b_scale=curr_b_scales,
            b_format="e2m1",
            acc=accumulator,
        )

        # Store next A to SMEM for next iteration
        smem_a.store(a)
        smem_as.store(a_scales)

    # -- Epilogue: process last tile --
    smem_b.store(b)
    smem_bs.store(b_scales)
    curr_a = smem_a.load(layout=dot_a_layout)
    curr_b = smem_b.load(layout=dot_b_layout)
    curr_a_scales = smem_as.load(layout=scale_a_layout)
    curr_b_scales = smem_bs.load(layout=scale_b_layout)

    accumulator = gl.amd.cdna4.mfma_scaled(
        a=curr_a,
        a_scale=curr_a_scales,
        a_format="e2m1",
        b=curr_b,
        b_scale=curr_b_scales,
        b_format="e2m1",
        acc=accumulator,
    )

    # Convert and store output
    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, gStoreLayoutC)
    )
    offs_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, gStoreLayoutC)
    )
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    c = gl.convert_layout(c, layout=gStoreLayoutC, assert_trivial=False)
    gl.amd.cdna4.buffer_store(c, c_ptr, offs_c, c_mask)


def matmul(a, b, a_scales, b_scales):
    """
    Compute C = A @ B with MXFP4 (e2m1) inputs and e8m0 block scales.

    Args:
        a: (M, K//2) uint8 tensor — packed MXFP4 activations, row-major
        b: (K//2, N) uint8 tensor — packed MXFP4 weights, column-major
        a_scales: (M, K//32) uint8 tensor — e8m0 per-group scales for A
        b_scales: (N, K//32) uint8 tensor — e8m0 per-group scales for B
    Returns:
        c: (M, N) bfloat16 tensor
    """
    M = a.shape[0]
    K_packed = a.shape[1]
    K = K_packed * 2  # unpack: 2 elements per byte
    N = b.shape[1]

    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 256
    num_warps = 8

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (GRID_MN, 1)
    NUM_XCDS = 8
    GROUP_SIZE_M = 4

    amxfp4wmxfp4_kernel[grid](
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
