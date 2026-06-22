##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

"""
FP16/BF16 GEMM kernel for CDNA4 (gfx950) — 8-wave warp-pipeline.

Ported from AMD-Triton/gluon-kernels (kernels/cdna4/gemm/
f16_gemm_warp_pipeline_gfx950.py) into the gfx950 tutorial layout so the
tutorial's bench / rocprof / ATT tooling can be wired to it.

Design (contrast with the tutorial's 4-wave a16w16 kernels):
- 8 warps (warpsPerCTA = [2, 4]), vs 4 warps (2x2)
- 256 x 256 x 32 tiles (BLOCK_K=32), vs 256 x 256 x 64
- 3 shared-memory buffers (triple-buffered), vs 2 (double)
- DistributedLinearLayout async copies + PaddedSharedLayout LDS
- `warp_pipeline_stage` for wave-level instruction interleaving
  (mfma stage + merged mem/lds stage), instead of the LLIR scheduler
- B is pre-transposed so K is contiguous (stride_bk == 1)

NOTE: the llir+amdgcnas toolchain used by the 4-wave tutorial kernels does NOT
apply here — it is built around the 4-wave register/schedule model and fails RA
on this 8-wave kernel. This kernel relies on `warp_pipeline_stage` instead.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4

from common import get_pid_m_n, store_result, issue_async_load_a, issue_async_load_b

# Tile sizes matching the reference ttgir kernel.
# The wrapper pre-transposes B so K is contiguous (stride_bk=1).
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 32

# Warp layout: warpsPerCTA = [2, 4]
NUM_WARPS = 8
WARPS_M = 2
WARPS_N = 4

# PID mapping parameters
GROUP_M = 16
NUM_XCDS = 8

# Smallest K the 2x-unrolled, 3-buffer pipeline can handle (5 k-tiles).
MIN_K = 5 * BLOCK_K

# kernel function name — used as the rocprof/ATT include-regex
KERNEL_NAME = "gemm_async_warp_pipeline"


@gluon.jit
def gemm_async_warp_pipeline(
    a_ptr, b_ptr, c_ptr,  #
    M, N, K,  #
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,  #
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
    WARPS_M: gl.constexpr, WARPS_N: gl.constexpr,  #
    GROUP_M: gl.constexpr, NUM_XCDS: gl.constexpr,
):
    """3-buffer pipelined GEMM with warp_pipeline_stage for gfx950."""
    a_dtype: gl.constexpr = a_ptr.type.element_ty
    b_dtype: gl.constexpr = b_ptr.type.element_ty
    c_dtype: gl.constexpr = c_ptr.type.element_ty
    gl.static_assert(a_dtype.is_fp16() or a_dtype.is_bf16(), "Only fp16/bf16 supported for A")
    gl.static_assert(b_dtype.is_fp16() or b_dtype.is_bf16(), "Only fp16/bf16 supported for B")
    gl.static_assert(c_dtype.is_fp16() or c_dtype.is_bf16(), "Only fp16/bf16 supported for C")

    gl.assume(stride_am >= 0)
    gl.assume(stride_ak >= 0)
    gl.assume(stride_bk >= 0)
    gl.assume(stride_bn >= 0)
    gl.assume(stride_cm >= 0)
    gl.assume(stride_cn >= 0)

    # PID mapping (GROUP_M + XCD remap)
    XCD_CHUNK: gl.constexpr = 128
    pid_m, pid_n = get_pid_m_n(M, N, BLOCK_M, BLOCK_N, GROUP_M, XCD_CHUNK, NUM_XCDS)

    # Layouts
    MMA_LAYOUT: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    OPERAND_LAYOUT_A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA_LAYOUT, k_width=8)
    OPERAND_LAYOUT_B: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA_LAYOUT, k_width=8)

    A_ASYNC_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [16, 0], [32, 0], [64, 0], [128, 0]],
        warp_bases=[[1, 0], [2, 0], [4, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )

    B_ASYNC_LAYOUT: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [0, 16], [0, 32], [0, 64], [0, 128]],
        warp_bases=[[0, 1], [0, 2], [0, 4]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N],
    )

    SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16],
            [16, 0], [32, 0], [64, 0], [128, 0],
            [1, 0], [2, 0], [4, 0],
            [8, 0],
        ],
        [],
        [BLOCK_M, BLOCK_K],
    )

    SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
            [0, 16], [0, 32], [0, 64], [0, 128],
            [0, 1], [0, 2], [0, 4],
            [0, 8],
        ],
        [],
        [BLOCK_K, BLOCK_N],
    )

    STORE_LAYOUT_C: gl.constexpr = gl.BlockedLayout(
        [16, 8], [8, 8], [WARPS_M, WARPS_N], [1, 0]
    )

    # Base pointers
    a_base = a_ptr + (pid_m * BLOCK_M) * stride_am
    b_base = b_ptr + (pid_n * BLOCK_N) * stride_bn

    # Shared memory ring buffers (3 buffers)
    smemA = gl.allocate_shared_memory(a_dtype, [3, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    smemB = gl.allocate_shared_memory(b_dtype, [3, BLOCK_K, BLOCK_N], SHARED_LAYOUT_B)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, MMA_LAYOUT)

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # Prologue: issue async loads for first 3 tiles.
    # Separate commits for A and B give 6 groups in flight after prologue.
    # wait_group(N) means wait until N groups remain in flight.

    # Tile 0 into buffer 0
    issue_async_load_a(smemA.index(0), a_base, 0, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_b(smemB.index(0), b_base, 0, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
    cdna4_async.commit_group()

    # Tile 1 into buffer 1
    issue_async_load_a(smemA.index(1), a_base, 1, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_b(smemB.index(1), b_base, 1, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
    cdna4_async.commit_group()

    # Tile 2 into buffer 2
    issue_async_load_a(smemA.index(2), a_base, 2, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_b(smemB.index(2), b_base, 2, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
    cdna4_async.commit_group()

    # Wait for tile 0 (6 issued, wait_group(4) means 2 completed = tile 0)
    cdna4_async.wait_group(4)
    a_regs = cdna4_async.load_shared_relaxed(smemA.index(0), OPERAND_LAYOUT_A)
    b_regs = cdna4_async.load_shared_relaxed(smemB.index(0), OPERAND_LAYOUT_B)

    # Main loop: 2x unrolled, 2 stages (mfma + mem) per step.
    # Commit group invariant: 4 in-flight at step entry, issue 2 -> 6,
    # wait_group(4) drains the 2 oldest -> back to 4.
    main_loop_pairs = (num_k_tiles - 3) // 2

    for pair_idx in tl.range(0, main_loop_pairs):
        tile_even = pair_idx * 2
        tile_odd = pair_idx * 2 + 1

        buf_even = tile_even % 3
        buf_odd = tile_odd % 3
        buf_next = (tile_even + 2) % 3

        prefetch_even = tile_even + 3
        prefetch_odd = tile_odd + 3

        # Step 1: process even tile
        with warp_pipeline_stage("mfma", priority=0):
            acc = mfma_cdna4(a_regs, b_regs, acc)

        cdna4_async.wait_group(2)

        with warp_pipeline_stage("mem", priority=1):
            issue_async_load_a(smemA.index(buf_even), a_base, prefetch_even, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
            cdna4_async.commit_group()
            issue_async_load_b(smemB.index(buf_even), b_base, prefetch_even, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
            cdna4_async.commit_group()
            a_regs = cdna4_async.load_shared_relaxed(smemA.index(buf_odd), OPERAND_LAYOUT_A)
            b_regs = cdna4_async.load_shared_relaxed(smemB.index(buf_odd), OPERAND_LAYOUT_B)

        # Step 2: process odd tile
        with warp_pipeline_stage("mfma", priority=0):
            acc = mfma_cdna4(a_regs, b_regs, acc)

        cdna4_async.wait_group(2)

        with warp_pipeline_stage("mem", priority=1):
            issue_async_load_a(smemA.index(buf_odd), a_base, prefetch_odd, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
            cdna4_async.commit_group()
            issue_async_load_b(smemB.index(buf_odd), b_base, prefetch_odd, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
            cdna4_async.commit_group()
            a_regs = cdna4_async.load_shared_relaxed(smemA.index(buf_next), OPERAND_LAYOUT_A)
            b_regs = cdna4_async.load_shared_relaxed(smemB.index(buf_next), OPERAND_LAYOUT_B)

    # Tail: process remaining tiles without future loads
    tiles_processed = main_loop_pairs * 2
    tiles_remaining = num_k_tiles - tiles_processed

    # Tail tile 0: already in a_regs/b_regs
    acc = mfma_cdna4(a_regs, b_regs, acc)

    cdna4_async.wait_group(0)

    # Tail tile 1
    buf_tile1 = (tiles_processed - 2) % 3
    a_regs = cdna4_async.load_shared_relaxed(smemA.index(buf_tile1), OPERAND_LAYOUT_A)
    b_regs = cdna4_async.load_shared_relaxed(smemB.index(buf_tile1), OPERAND_LAYOUT_B)
    acc = mfma_cdna4(a_regs, b_regs, acc)

    # Tail tile 2
    buf_tile2 = (tiles_processed - 1) % 3
    a_regs = cdna4_async.load_shared_relaxed(smemA.index(buf_tile2), OPERAND_LAYOUT_A)
    b_regs = cdna4_async.load_shared_relaxed(smemB.index(buf_tile2), OPERAND_LAYOUT_B)
    acc = mfma_cdna4(a_regs, b_regs, acc)

    # NOTE: Extra tail tile when num_k_tiles is odd relative to pairs
    if tiles_remaining > 3:
        last_tile_idx = tiles_processed + 3
        buf_last = tiles_processed % 3
        issue_async_load_a(smemA.index(buf_last), a_base, last_tile_idx, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
        cdna4_async.commit_group()
        issue_async_load_b(smemB.index(buf_last), b_base, last_tile_idx, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
        cdna4_async.commit_group()
        cdna4_async.wait_group(0)
        a_regs = cdna4_async.load_shared_relaxed(smemA.index(buf_last), OPERAND_LAYOUT_A)
        b_regs = cdna4_async.load_shared_relaxed(smemB.index(buf_last), OPERAND_LAYOUT_B)
        acc = mfma_cdna4(a_regs, b_regs, acc)

    # Epilogue: store
    store_result(acc, c_ptr, c_dtype, pid_m, pid_n, M, N,
                 stride_cm, stride_cn, BLOCK_M, BLOCK_N, STORE_LAYOUT_C)


def matmul_kernel_only(a: torch.Tensor, b_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Kernel-only entry for fair benchmarking.

    a   : (M, K) contiguous
    b_t : (N, K) contiguous  — B pre-transposed so K is the contiguous dim
    c   : (M, N) pre-allocated output

    No transpose or allocation happens here, so do_bench / rocprof time only
    the GEMM kernel.
    """
    M, K = a.shape
    N = b_t.shape[0]

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m * grid_n,)

    gemm_async_warp_pipeline[grid](
        a, b_t, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_t.stride(1), b_t.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
        GROUP_M=GROUP_M,
        NUM_XCDS=NUM_XCDS,
        num_warps=NUM_WARPS,
    )
    return c


def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
    """Compute C = A @ B.

    a16w16-compatible signature. `b` is the (K, N) right-hand matrix (the same
    layout the tutorial's a16w16 bench passes). This kernel needs B with K
    contiguous, so B is transposed to (N, K) contiguous here. For fair timing
    use `matmul_kernel_only` with a pre-transposed `b_t` instead.
    """
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype in (torch.float16, torch.bfloat16) and b.dtype == a.dtype
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    N = b.shape[1]

    num_k_tiles = triton.cdiv(K, BLOCK_K)
    assert num_k_tiles >= 5, f"K={K} too small for 2x-unrolled pipeline (need K >= {MIN_K})"

    b_t = b.t().contiguous()  # (N, K) contiguous
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    return matmul_kernel_only(a, b_t, c)
