import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Memory-bound GEMM kernel for skinny matrices (small M) on AMD GFX950.

    Computes C[M, N] = A[M, K] x B[K, N] in fp16 with fp32 accumulation.

    Tile: BLOCK_M=32, BLOCK_N=256, BLOCK_K=64, 4 warps (2x2).
    Per iteration: A tile = 4 KB, B tile = 32 KB, total = 36 KB.

    Key optimizations:
      - 2-stage async pipeline (buffer_load_dwordx4 -> LDS -> ds_read -> MFMA)
      - .cg cache modifier on B loads (B is not shared across workgroups)
      - Large BLOCK_N to maximize B/A ratio for TCP utilization efficiency
    """

    NUM_STAGES: gl.constexpr = 2

    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ---- Global load layouts ----
    # A tile [BLOCK_M=32, BLOCK_K=64]: 2048 elements, 8 per thread, 1 buffer_load each
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [4, 0], [8, 0], [16, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )
    # B tile [BLOCK_K=64, BLOCK_N=256]: 16384 elements, 64 per thread, 4 buffer_loads each
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 64], [0, 128]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 8], [0, 16], [0, 32]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N],
    )

    # ---- Shared memory layouts ----
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [4, 0],
            [8, 0],
            [16, 0],
            [1, 0],
            [2, 0],
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
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 64],
            [0, 128],
        ],
        [],
        [BLOCK_K, BLOCK_N],
    )

    # ---- Allocate double-buffered shared memory ----
    smemA = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_M, BLOCK_K], sharedLayoutA
    )
    smemB = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_STAGES, BLOCK_K, BLOCK_N], sharedLayoutB
    )

    # ---- Compute offsets ----
    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # ---- MFMA compute layout ----
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)
    iterMax = gl.cdiv(K, BLOCK_K)

    # ---- Prologue: fill both pipeline buffers ----
    for s in range(NUM_STAGES):
        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(s), a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB.index(s), b_base, b_offsets, cache_modifier=".cg"
        )
        gl.amd.cdna4.async_copy.commit_group()
        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    # ---- Main loop ----
    for k in range(0, iterMax - NUM_STAGES):
        l_idx = k % NUM_STAGES

        gl.amd.cdna4.async_copy.wait_group(NUM_STAGES - 1)

        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
        acc = gl.amd.cdna3.mfma(a, b, acc)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(l_idx), a_base, a_offsets)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB.index(l_idx), b_base, b_offsets, cache_modifier=".cg"
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    # ---- Epilogue: drain remaining buffers ----
    gl.amd.cdna4.async_copy.wait_group(0)

    for s in range(NUM_STAGES):
        l_idx = (iterMax - NUM_STAGES + s) % NUM_STAGES
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
        acc = gl.amd.cdna3.mfma(a, b, acc)

    # ---- Store result ----
    c = acc.to(a_ptr.dtype.element_ty)
    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask)


def matmul(a, b, c=None, num_stages=2):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M = 32
    BLOCK_N = 256
    BLOCK_K = 64
    num_warps = 4
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return c
