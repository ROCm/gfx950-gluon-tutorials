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
8-wave warp-pipeline FP16/BF16 GEMM — **3x-unrolled** variant.

Same kernel as matmul_kernel.py, but the main loop is unrolled by 3 instead of 2.
Because the LDS ring has 3 buffers, unrolling by 3 makes every buffer index a
COMPILE-TIME CONSTANT (0, 1, 2) within the loop body. That removes the runtime
`tile % 3` buffer-rotation arithmetic the 2x loop carries (the divide-by-3
`0xaaaaaaab` magic, the `0xc600` ring-span multiply, and the ~15 `s_sub`/
`v_subrev` wrap-subtracts) — those collapse into constant immediates baked into
the ds_read / buffer_load offsets.

Per iteration (triple_idx = j) processes tiles {3j, 3j+1, 3j+2} in buffers
{0, 1, 2} and prefetches {3j+3, 3j+4, 3j+5} into those same just-freed buffers.

Trade-off vs 2x: a 3-step body holds more operand register sets, which may raise
VGPR pressure (watch for spills / a drop below 2 waves/SIMD). Benchmark before
adopting.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4

from common import get_pids, store_result, issue_async_load_a, issue_async_load_b

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 32

NUM_WARPS = 8
WARPS_M = 2
WARPS_N = 4

# L2-locality PID mapping — copied from v9 (get_pids): XCD-aware remap +
# GROUP_SIZE_M swizzle. v9's optimal GROUP_SIZE_M for P=32 tiles/XCD is 4.
NUM_XCDS = 8
GROUP_SIZE_M = 4

MIN_K = 5 * BLOCK_K

# kernel function name — used as the rocprof/ATT include-regex (distinct from 2x)
KERNEL_NAME = "gemm_async_warp_pipeline_u3"


@gluon.jit
def gemm_async_warp_pipeline_u3(
    a_ptr, b_ptr, c_ptr,  #
    M, N, K,  #
    stride_am, stride_ak,  #
    stride_bk, stride_bn,  #
    stride_cm, stride_cn,  #
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,  #
    WARPS_M: gl.constexpr, WARPS_N: gl.constexpr,  #
    GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
):
    """3-buffer pipelined GEMM, 3x-unrolled (constant buffer indices)."""
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

    # XCD-aware PID remapping + GROUP_SIZE_M swizzle (v9 scheme, active at all sizes).
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

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

    a_base = a_ptr + (pid_m * BLOCK_M) * stride_am
    b_base = b_ptr + (pid_n * BLOCK_N) * stride_bn

    smemA = gl.allocate_shared_memory(a_dtype, [3, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    smemB = gl.allocate_shared_memory(b_dtype, [3, BLOCK_K, BLOCK_N], SHARED_LAYOUT_B)

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, MMA_LAYOUT)

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # Prologue: load tiles 0,1,2 into buffers 0,1,2 (6 commit groups).
    issue_async_load_a(smemA.index(0), a_base, 0, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_b(smemB.index(0), b_base, 0, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_a(smemA.index(1), a_base, 1, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_b(smemB.index(1), b_base, 1, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_a(smemA.index(2), a_base, 2, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
    cdna4_async.commit_group()
    issue_async_load_b(smemB.index(2), b_base, 2, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
    cdna4_async.commit_group()

    # Wait for tile 0 (6 issued, wait_group(4) => 2 done = tile 0)
    cdna4_async.wait_group(4)
    a_regs = cdna4_async.load_shared_relaxed(smemA.index(0), OPERAND_LAYOUT_A)
    b_regs = cdna4_async.load_shared_relaxed(smemB.index(0), OPERAND_LAYOUT_B)

    # Main loop: 3x unrolled. Buffer indices are the literal constants 0,1,2 in
    # every step, so no `tile % 3` arithmetic survives in the loop body.
    main_loop_triples = (num_k_tiles - 3) // 3

    for triple_idx in tl.range(0, main_loop_triples):
        # Step 0's GR drain is hoisted ABOVE the loop-index arithmetic below.
        # WarpPipeliner rejects pipelining if it meets an async-wait (this
        # wait_group) while a stage cluster is non-empty; the index arith
        # (base_tile/pf*) would populate that cluster first and silently disable
        # the warp-pipeline (no s_setprio, half the s_barriers). Hoisted here it
        # lands at an empty cluster boundary so pipelining is preserved. Steps
        # 1-2 keep wait_group inline (each follows a mem-stage border = empty
        # cluster, so no rejection there).
        cdna4_async.wait_group(2)
        base_tile = triple_idx * 3
        pf0 = base_tile + 3
        pf1 = base_tile + 4
        pf2 = base_tile + 5

        # Correctness-critical ordering (per step): wait_group(2) precedes the
        # mfma region, and each shared load (LR) precedes its global prefetch
        # (GR). The per-iteration s_barrier synchronizes only wave EXECUTION,
        # not the async GR memory, so draining the GR vmcnt ahead of the mfma is
        # what guarantees an LDS tile's filling copy has landed before any wave
        # (including the other 4-wave group) consumes it. The reverse layout
        # (wait_group after mfma + GR-first) races on the 3-buffer LDS ring and
        # passes the allclose check only by timing luck. Loads stay relaxed: the
        # no-alias hint elides a redundant LDS barrier the non-relaxed smem.load
        # form would insert here (~6% slower).

        # Step 0: process tile base_tile (buffer 0)
        with warp_pipeline_stage("mfma", priority=0):
            acc = mfma_cdna4(a_regs, b_regs, acc)
        with warp_pipeline_stage("mem", priority=1):
            a_regs = cdna4_async.load_shared_relaxed(smemA.index(1), OPERAND_LAYOUT_A)
            b_regs = cdna4_async.load_shared_relaxed(smemB.index(1), OPERAND_LAYOUT_B)
            issue_async_load_a(smemA.index(0), a_base, pf0, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
            cdna4_async.commit_group()
            issue_async_load_b(smemB.index(0), b_base, pf0, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
            cdna4_async.commit_group()

        # Step 1: process tile base_tile+1 (buffer 1)
        cdna4_async.wait_group(2)
        with warp_pipeline_stage("mfma", priority=0):
            acc = mfma_cdna4(a_regs, b_regs, acc)
        with warp_pipeline_stage("mem", priority=1):
            a_regs = cdna4_async.load_shared_relaxed(smemA.index(2), OPERAND_LAYOUT_A)
            b_regs = cdna4_async.load_shared_relaxed(smemB.index(2), OPERAND_LAYOUT_B)
            issue_async_load_a(smemA.index(1), a_base, pf1, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
            cdna4_async.commit_group()
            issue_async_load_b(smemB.index(1), b_base, pf1, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
            cdna4_async.commit_group()

        # Step 2: process tile base_tile+2 (buffer 2)
        cdna4_async.wait_group(2)
        with warp_pipeline_stage("mfma", priority=0):
            acc = mfma_cdna4(a_regs, b_regs, acc)
        with warp_pipeline_stage("mem", priority=1):
            # Next iteration's first tile (base+3) lands in buffer 0.
            a_regs = cdna4_async.load_shared_relaxed(smemA.index(0), OPERAND_LAYOUT_A)
            b_regs = cdna4_async.load_shared_relaxed(smemB.index(0), OPERAND_LAYOUT_B)
            issue_async_load_a(smemA.index(2), a_base, pf2, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
            cdna4_async.commit_group()
            issue_async_load_b(smemB.index(2), b_base, pf2, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
            cdna4_async.commit_group()

    # Tail: tiles_processed is a multiple of 3, so tiles_processed,{+1,+2} are
    # resident in buffers 0,1,2; a_regs/b_regs already hold tile tiles_processed.
    tiles_processed = main_loop_triples * 3
    tiles_remaining = num_k_tiles - tiles_processed  # 3, 4, or 5

    # Tail tile 0 (buffer 0, already in regs)
    acc = mfma_cdna4(a_regs, b_regs, acc)
    cdna4_async.wait_group(0)

    # Tail tile 1 (buffer 1)
    a_regs = cdna4_async.load_shared_relaxed(smemA.index(1), OPERAND_LAYOUT_A)
    b_regs = cdna4_async.load_shared_relaxed(smemB.index(1), OPERAND_LAYOUT_B)
    acc = mfma_cdna4(a_regs, b_regs, acc)

    # Tail tile 2 (buffer 2)
    a_regs = cdna4_async.load_shared_relaxed(smemA.index(2), OPERAND_LAYOUT_A)
    b_regs = cdna4_async.load_shared_relaxed(smemB.index(2), OPERAND_LAYOUT_B)
    acc = mfma_cdna4(a_regs, b_regs, acc)

    # Extra tile tiles_processed+3 (buffer 0) when num_k_tiles % 3 != 0
    if tiles_remaining > 3:
        issue_async_load_a(smemA.index(0), a_base, tiles_processed + 3, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
        cdna4_async.commit_group()
        issue_async_load_b(smemB.index(0), b_base, tiles_processed + 3, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
        cdna4_async.commit_group()
        cdna4_async.wait_group(0)
        a_regs = cdna4_async.load_shared_relaxed(smemA.index(0), OPERAND_LAYOUT_A)
        b_regs = cdna4_async.load_shared_relaxed(smemB.index(0), OPERAND_LAYOUT_B)
        acc = mfma_cdna4(a_regs, b_regs, acc)

    # Extra tile tiles_processed+4 (buffer 1) when num_k_tiles % 3 == 2
    if tiles_remaining > 4:
        issue_async_load_a(smemA.index(1), a_base, tiles_processed + 4, stride_am, stride_ak, BLOCK_M, BLOCK_K, A_ASYNC_LAYOUT)
        cdna4_async.commit_group()
        issue_async_load_b(smemB.index(1), b_base, tiles_processed + 4, stride_bk, stride_bn, BLOCK_K, BLOCK_N, B_ASYNC_LAYOUT)
        cdna4_async.commit_group()
        cdna4_async.wait_group(0)
        a_regs = cdna4_async.load_shared_relaxed(smemA.index(1), OPERAND_LAYOUT_A)
        b_regs = cdna4_async.load_shared_relaxed(smemB.index(1), OPERAND_LAYOUT_B)
        acc = mfma_cdna4(a_regs, b_regs, acc)

    store_result(acc, c_ptr, c_dtype, pid_m, pid_n, M, N,
                 stride_cm, stride_cn, BLOCK_M, BLOCK_N, STORE_LAYOUT_C)


def matmul_kernel_only(a: torch.Tensor, b_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Kernel-only entry (b_t pre-transposed (N,K) contiguous, c pre-allocated)."""
    M, K = a.shape
    N = b_t.shape[0]

    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    GRID_MN = grid_m * grid_n
    grid = (GRID_MN,)

    gemm_async_warp_pipeline_u3[grid](
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
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
    )
    return c


def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
    """C = A @ B. `b` is (K, N); transposed to (N, K) contiguous for the kernel."""
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype in (torch.float16, torch.bfloat16) and b.dtype == a.dtype
    assert a.is_cuda and b.is_cuda

    M, K = a.shape
    N = b.shape[1]

    num_k_tiles = triton.cdiv(K, BLOCK_K)
    assert num_k_tiles >= 5, f"K={K} too small (need K >= {MIN_K})"

    b_t = b.t().contiguous()
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    return matmul_kernel_only(a, b_t, c)
