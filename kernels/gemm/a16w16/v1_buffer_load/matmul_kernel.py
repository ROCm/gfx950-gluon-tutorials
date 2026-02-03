import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import torch

@gluon.jit
def v1_buffer_load(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak,  #
       stride_bk, stride_bn,  #
       stride_cm, stride_cn, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
       ):

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    gLoadLayoutA: gl.constexpr = gl.BlockedLayout(
        [1, 8], # sizePerThread
        [512 // BLOCK_K, BLOCK_K // 8], # threadsPerWarp
        [4, 1], # warpsPerCTA
        [1, 0]  # order
    )

    gLoadLayoutB: gl.constexpr = gl.BlockedLayout(
        [8, 1], # sizePerThread
        [BLOCK_K // 8, 512 // BLOCK_K], # threadsPerWarp
        [1, 4], # warpsPerCTA
        [0, 1]  # order
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    a_ptrs = a_base + a_offsets
    b_ptrs = b_base + b_offsets

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(version=4,
                                                    instr_shape=[16, 16, 32],
                                                    transposed=True,
                                                    warps_per_cta=[2, 2])

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfmaLayout)

    for k in range(0, gl.cdiv(K, BLOCK_K)):
        ga = gl.amd.cdna3.buffer_load(ptr=a_base, offsets=a_offsets, mask=offs_ak[None, :] < K - k * BLOCK_K, other=0.0)
        gb = gl.amd.cdna3.buffer_load(ptr=b_base, offsets=b_offsets, mask=offs_bk[:, None] < K - k * BLOCK_K, other=0.0)
        a = gl.convert_layout(ga, layout=dotOpLayoutA)
        b = gl.convert_layout(gb, layout=dotOpLayoutB)

        acc = gl.amd.cdna3.mfma(a, b, acc)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    c = acc.to(a_ptr.dtype.element_ty)

    gStoreLayoutC: gl.constexpr = mfmaLayout
    c = gl.convert_layout(c, layout=gStoreLayoutC)
    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_ptrs = c_base + c_offsets
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
    v1_buffer_load[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps)
    return c
