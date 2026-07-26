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
a16w16 inter-wave (8-wave warp-pipeline) FP16/BF16 GEMM for gfx942 / MI300X.

Port of the gfx950 tutorial's `inter_wave/a16w16`: 8 warps/CTA = 2 waves/SIMD,
the 256x256 tile sliced into 2x2 [128x128] quadrants, and the hot loop split
into `warp_pipeline_stage` clusters so the two wave groups run a stage out of
phase -- while one group issues MFMA, the other issues its `buffer_load` /
`local_store` / `local_load` and absorbs the waits.

Everything about the memory pipeline is inherited from `intra_wave/`: no
`buffer_load_to_shared` on CDNA3, so global data goes HBM -> VGPR -> LDS -> VGPR;
and 64 KB of LDS holds exactly one 256x256x64 stage, so the four half-tile slots
are single-buffered and recycled one region after their last read.  See that
kernel's docstring for the derivation.

WHAT IS SPECIFIC TO THE 8-WAVE KERNEL
-------------------------------------
Two waves per SIMD split the unified 512-register file, so each wave gets **256
registers, VGPR and AGPR together** -- AGPRs buy no extra capacity here, which
is why this kernel runs with `amdgpu-agpr-alloc=0,0` while the 4-wave one
reserves 256 AGPRs.  Budgeting against 256 is what shapes the kernel:

    accumulators   4 x [128x128] f32 / (8 x 64)                  = 128
    dot operands   4 live half-tile *K-slices* (see below)        =  48
    global staging 4 half-tiles, [128x64] / (8 x 64)              =  32
                                                                    ---
                                                                    208  (+ ~24 addressing)

The dot operands are the whole story.  Loading a full [128x64] half-tile the way
the 4-wave kernel does costs 32 VGPRs for A and 16 for B, and holding all four
live is 96 -- which lands the kernel at 256 before addressing.  Measured: it
compiles to 256 VGPRs **with 28 spill slots** and runs at ~280 TFLOPS.

So the operands are read as **K=32 slices** of the same [128x64] LDS tile
(`smem.slice(0, 32, k_dim)` / `slice(32, 32, k_dim)`), halving each live operand
and giving 8 regions per K-step instead of 4:

    region      DOT                  LR (next region's operand)   LW / GR
    ---------------------------------------------------------------------
    R0   C_tl += A_t[k0] B_l[k0]     A_b[k0]                      -
    R1   C_bl += A_b[k0] B_l[k0]     B_r[k0]                      -
    R2   C_tr += A_t[k0] B_r[k0]     A_t[k1]                      -
    R3   C_br += A_b[k0] B_r[k0]     B_l[k1]                      A_top
    R4   C_tl += A_t[k1] B_l[k1]     A_b[k1]                      B_left
    R5   C_bl += A_b[k1] B_l[k1]     B_r[k1]                      A_bot
    R6   C_tr += A_t[k1] B_r[k1]     B_l[k0] of tile k+1          B_right
    R7   C_br += A_b[k1] B_r[k1]     A_t[k0] of tile k+1          -

Each operand register is overwritten exactly one region after its last use, so
four live slices (A_t, A_b, B_l, B_r) are enough -- no doubling.  Each LDS slot
is refilled one region after its last read (A_top read last in R2, refilled in
R3; B_left in R3/R4; A_bot in R4/R5; B_right in R5/R6), so one barrier per
region boundary is sufficient and correct.

Slicing K also fixes the cluster cadence.  A K-step is 256 mfma/SIMD = ~4096
cycles, so 8 regions put a cluster boundary every ~512 cycles -- the same
barrier rate as the gfx950 8-wave kernel.  An earlier BLOCK_K=32 variant of this
kernel (double-buffered LDS, 4 regions) hit the same register budget but halved
the contiguous run of every global load from 128 B to 64 B, doubling TCP
cache-line pressure; ablating its `buffer_load`s moved it from 386 to 503
TFLOPS.  Keeping BLOCK_K=64 for the *global* side and slicing only the *LDS
read* side is what gets both.

`local_store` bank conflicts: `SwizzledSharedLayout(8, 2, 8)` is conflict-free
for both directions, and slicing K does not disturb it -- the swizzle permutes
16 B chunks within a row, and a K=32 slice just restricts which chunk indices a
lane touches.
"""

import torch
import triton
from common import get_pids
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store, mfma

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
SPLIT_K: gl.constexpr = 32  # dot-operand K slice

NUM_WARPS = 8
WARPS_M = 2
WARPS_N = 4

NUM_XCDS = 8
GROUP_SIZE_M = 4

# main loop runs range(0, iterMax - 2); two K-steps are peeled
MIN_K = 4 * BLOCK_K
KERNEL_NAME = "a16w16_inter_wave_gfx942"


@gluon.jit
def a16w16_inter_wave_gfx942(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,  #
    SPLIT_K: gl.constexpr,  #
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # ---- global-load layouts (8-warp variant of the intra_wave layouts) -----
    # Eight lanes cover one whole 128 B row of the tile and consecutive lanes
    # walk consecutive rows: one buffer_load_dwordx4 reads 1024 contiguous
    # bytes = 8 back-to-back cache lines, and the matching local_store has
    # lanes 0-15 covering two adjacent rows = 256 B = all 64 banks once.
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 4]],
        warp_bases=[[0, 16], [0, 32], [0, 64]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # ---- LDS: one 16 KB slot per half-tile, 64 KB total ---------------------
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + (BLOCK_M // 2) * stride_am
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + (BLOCK_N // 2) * stride_bn

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 2)

    # ---- prologue -----------------------------------------------------------
    gB_left = buffer_load(ptr=b_base, offsets=b_left_offsets)
    gA_top = buffer_load(ptr=a_base, offsets=a_top_offsets)
    gA_bot = buffer_load(ptr=a_base, offsets=a_bot_offsets)
    gB_right = buffer_load(ptr=b_base, offsets=b_right_offsets)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    smemB_left.store(gB_left)
    smemA_top.store(gA_top)
    smemA_bot.store(gA_bot)
    smemB_right.store(gB_right)

    gB_left = buffer_load(ptr=b_base, offsets=b_left_offsets)
    gA_top = buffer_load(ptr=a_base, offsets=a_top_offsets)
    gA_bot = buffer_load(ptr=a_base, offsets=a_bot_offsets)
    gB_right = buffer_load(ptr=b_base, offsets=b_right_offsets)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    b_left = smemB_left.slice(0, SPLIT_K, 0).load(dotB)
    a_top = smemA_top.slice(0, SPLIT_K, 1).load(dotA)

    ## ------------------------------------------------------------------
    ## Main loop -- 8 (mfma | mem) cluster pairs per K-step.  The mem cluster
    ## carries the higher s_setprio so its address VALU keeps issuing while
    ## the other wave group is on the matrix unit.
    ## ------------------------------------------------------------------
    for k in range(0, iterMax - 2):

        ## ---- K-slice 0 --------------------------------------------------
        with warp_pipeline_stage("mfma", priority=0):
            acc_tl = mfma(a_top, b_left, acc_tl)
        with warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.slice(0, SPLIT_K, 1).load(dotA)

        with warp_pipeline_stage("mfma", priority=0):
            acc_bl = mfma(a_bot, b_left, acc_bl)
        with warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.slice(0, SPLIT_K, 0).load(dotB)

        with warp_pipeline_stage("mfma", priority=0):
            acc_tr = mfma(a_top, b_right, acc_tr)
        with warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.slice(SPLIT_K, SPLIT_K, 1).load(dotA)

        with warp_pipeline_stage("mfma", priority=0):
            acc_br = mfma(a_bot, b_right, acc_br)
        with warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.slice(SPLIT_K, SPLIT_K, 0).load(dotB)
            smemA_top.store(gA_top)  # LW A_top(k+1) -- slot free since R2
            gA_top = buffer_load(ptr=a_base, offsets=a_top_offsets)  # GR A_top(k+2)

        ## ---- K-slice 1 --------------------------------------------------
        with warp_pipeline_stage("mfma", priority=0):
            acc_tl = mfma(a_top, b_left, acc_tl)
        with warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.slice(SPLIT_K, SPLIT_K, 1).load(dotA)
            smemB_left.store(gB_left)  # LW B_left(k+1) -- slot free since R3
            gB_left = buffer_load(ptr=b_base, offsets=b_left_offsets)

        with warp_pipeline_stage("mfma", priority=0):
            acc_bl = mfma(a_bot, b_left, acc_bl)
        with warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.slice(SPLIT_K, SPLIT_K, 0).load(dotB)
            smemA_bot.store(gA_bot)  # LW A_bot(k+1) -- slot free since R4
            gA_bot = buffer_load(ptr=a_base, offsets=a_bot_offsets)

        with warp_pipeline_stage("mfma", priority=0):
            acc_tr = mfma(a_top, b_right, acc_tr)
        with warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.slice(0, SPLIT_K, 0).load(dotB)  # tile k+1
            smemB_right.store(gB_right)  # LW B_right(k+1) -- slot free since R5
            gB_right = buffer_load(ptr=b_base, offsets=b_right_offsets)

        with warp_pipeline_stage("mfma", priority=0):
            acc_br = mfma(a_bot, b_right, acc_br)
        with warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.slice(0, SPLIT_K, 1).load(dotA)  # tile k+1
            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

    ## ------------------------------------------------------------------
    ## Peel 1 (k = iterMax - 2): last LDS refill, nothing left to prefetch.
    ## ------------------------------------------------------------------
    acc_tl = mfma(a_top, b_left, acc_tl)
    a_bot = smemA_bot.slice(0, SPLIT_K, 1).load(dotA)

    acc_bl = mfma(a_bot, b_left, acc_bl)
    b_right = smemB_right.slice(0, SPLIT_K, 0).load(dotB)

    acc_tr = mfma(a_top, b_right, acc_tr)
    a_top = smemA_top.slice(SPLIT_K, SPLIT_K, 1).load(dotA)

    acc_br = mfma(a_bot, b_right, acc_br)
    b_left = smemB_left.slice(SPLIT_K, SPLIT_K, 0).load(dotB)
    smemA_top.store(gA_top)

    acc_tl = mfma(a_top, b_left, acc_tl)
    a_bot = smemA_bot.slice(SPLIT_K, SPLIT_K, 1).load(dotA)
    smemB_left.store(gB_left)

    acc_bl = mfma(a_bot, b_left, acc_bl)
    b_right = smemB_right.slice(SPLIT_K, SPLIT_K, 0).load(dotB)
    smemA_bot.store(gA_bot)

    acc_tr = mfma(a_top, b_right, acc_tr)
    b_left = smemB_left.slice(0, SPLIT_K, 0).load(dotB)
    smemB_right.store(gB_right)

    acc_br = mfma(a_bot, b_right, acc_br)
    a_top = smemA_top.slice(0, SPLIT_K, 1).load(dotA)

    ## ------------------------------------------------------------------
    ## Peel 2 (k = iterMax - 1): drain -- no LW, no GR, and the two LRs that
    ## would fetch tile iterMax are dropped.
    ## ------------------------------------------------------------------
    acc_tl = mfma(a_top, b_left, acc_tl)
    a_bot = smemA_bot.slice(0, SPLIT_K, 1).load(dotA)

    acc_bl = mfma(a_bot, b_left, acc_bl)
    b_right = smemB_right.slice(0, SPLIT_K, 0).load(dotB)

    acc_tr = mfma(a_top, b_right, acc_tr)
    a_top = smemA_top.slice(SPLIT_K, SPLIT_K, 1).load(dotA)

    acc_br = mfma(a_bot, b_right, acc_br)
    b_left = smemB_left.slice(SPLIT_K, SPLIT_K, 0).load(dotB)

    acc_tl = mfma(a_top, b_left, acc_tl)
    a_bot = smemA_bot.slice(SPLIT_K, SPLIT_K, 1).load(dotA)

    acc_bl = mfma(a_bot, b_left, acc_bl)
    b_right = smemB_right.slice(SPLIT_K, SPLIT_K, 0).load(dotB)

    acc_tr = mfma(a_top, b_right, acc_tr)
    acc_br = mfma(a_bot, b_right, acc_br)

    ## ------------------------------------------------------------------
    ## Epilogue: store straight from the mfma layout (no convert_layout, so
    ## no LDS scratch -- LDS is full to the byte).
    ## ------------------------------------------------------------------
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, mfmaLayout))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
    c_quad_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tl_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_bl_base = c_tl_base + (BLOCK_M // 2) * stride_cm
    c_tr_base = c_tl_base + (BLOCK_N // 2) * stride_cn
    c_br_base = c_bl_base + (BLOCK_N // 2) * stride_cn
    buffer_store(
        ptr=c_tl_base, offsets=c_quad_offsets, stored_value=acc_tl.to(c_ptr.dtype.element_ty)
    )
    buffer_store(
        ptr=c_bl_base, offsets=c_quad_offsets, stored_value=acc_bl.to(c_ptr.dtype.element_ty)
    )
    buffer_store(
        ptr=c_tr_base, offsets=c_quad_offsets, stored_value=acc_tr.to(c_ptr.dtype.element_ty)
    )
    buffer_store(
        ptr=c_br_base, offsets=c_quad_offsets, stored_value=acc_br.to(c_ptr.dtype.element_ty)
    )


def matmul_kernel_only(a: torch.Tensor, b_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Kernel-only entry; `b_t` is B pre-transposed to (N, K) contiguous."""
    M, K = a.shape
    N = b_t.shape[0]
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    a16w16_inter_wave_gfx942[(GRID_MN,)](
        a,
        b_t,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_t.stride(1),
        b_t.stride(0),  # stride_bk = 1 (K contiguous), stride_bn = K
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        # Forbid AGPRs: with 2 waves/SIMD the unified 512-register file is split
        # between the resident waves, so AGPRs add no capacity and only cost
        # v_accvgpr copies in the epilogue.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    return c


def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
    """C = A @ B.  `b` is (K, N); transposed to (N, K) contiguous for the kernel."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    N = b.shape[1]
    b_t = b.t().contiguous()
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    return matmul_kernel_only(a, b_t, c)
