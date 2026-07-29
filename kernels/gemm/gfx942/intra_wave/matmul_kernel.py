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
a16w16 intra-wave (4-wave) FP16/BF16 GEMM for gfx942 / MI300X (CDNA3).

Port of the gfx950 tutorial's `intra_wave/a16w16/v9_beyond_hotloop`: same 2x2
quadrant slicing of the 256x256 output tile, same B_left -> A_top -> A_bot ->
B_right region order, same XCD-aware PID remap + GROUP_SIZE_M swizzle.

Two things are genuinely different on CDNA3, and they drive the whole design:

1. NO `buffer_load_to_shared`.  gfx950 streams HBM -> LDS with an async copy and
   synchronises with `async_wait(n)`.  gfx942 has to go
   `buffer_load` (HBM -> VGPR) -> `local_store` (VGPR -> LDS) ->
   `local_load` (LDS -> VGPR).  The register round-trip costs 4 x 16 = 64 VGPRs
   of staging, and the LDS producer/consumer hazard is now closed by real
   `s_barrier`s (inserted by Triton's membar pass) instead of an async counter.

2. HALF THE LDS.  gfx950 has 160 KB/CU and double-buffers a 256x256x64 stage
   (2 x 64 KB).  gfx942 has 64 KB, which is *exactly one* stage -- double
   buffering is impossible at this tile size.

   Rather than shrink BLOCK_K to 32 (which would halve the contiguous run of
   each global load from 128 B to 64 B and roughly double TCP cache-line
   pressure), this kernel keeps BLOCK_K=64 and recycles LDS at *half-tile*
   granularity.  Each of the four half-tiles (A_top, A_bot, B_left, B_right)
   owns a single 16 KB slot, and a slot is refilled one region after its last
   read:

       region 0:  DOT C_tl  |  LR A_bot(k)     |  LW B_left(k+1)   |  GR B_left(k+2)
       region 1:  DOT C_bl  |  LR B_right(k)   |  LW A_top(k+1)    |  GR A_top(k+2)
       region 2:  DOT C_tr  |  LR B_left(k+1)  |  LW A_bot(k+1)    |  GR A_bot(k+2)
       region 3:  DOT C_br  |  LR A_top(k+1)   |  LW B_right(k+1)  |  GR B_right(k+2)

   Every slot's write lands strictly between its previous read and its next
   read, one region apart on both sides, so a single barrier per region is
   sufficient -- and one region is 256 mfma/4 = 64 x 16 = ~1024 cycles, which is
   plenty to absorb the barrier plus the LDS traffic.

   The pipeline depth this buys is the same as gfx950's double buffer:
   GR(k+2) -> LW(k+1) is one full K-step (~4096 cycles) of global latency
   hiding, and LR(k+1) -> DOT(k+1) is one region (~1024 cycles) of LDS latency
   hiding.

3. MFMA shape.  CDNA3's widest f16/bf16 16x16 intrinsic is
   `v_mfma_f32_16x16x16_f16` (K=16, 16 cycles) rather than CDNA4's
   `..._16x16x32_f16`.  `k_width=8` still gives `ds_read_b128`: the dot-operand
   K tile is 8 * (64/16) = 32, which Triton splits into two K=16 mfmas.

4. LDS layout.  gfx950 uses `PaddedSharedLayout([[512, 16]])`; padding is not
   affordable here (the four slots already fill LDS to the byte), so both
   operands use `SwizzledSharedLayout(8, 1, 8)`.  **per_phase=1, not 2**: the
   global-load layout has consecutive lanes walking consecutive rows, so lanes
   0-15 of a b128 access cover two adjacent rows, and per_phase=2 gives those
   two rows the same swizzle phase -- they hit the same 32 banks and every
   access replays.  Measured with SQ_LDS_BANK_CONFLICT: (8,2,8) is 0.667 replay
   cycles per useful cycle (13.33 cyc/LDS instr), (8,1,8) is exactly 0
   (8.00 cyc).  See tools/lds_conflict.py --sweep.

Register budget (per lane, 1 wave/SIMD so 256 VGPR + 256 AGPR are available):
    C accumulators   4 x [128x128] f32  = 256   -> AGPR (amdgpu-agpr-alloc=256)
    dot operands     4 x half-tile f16  = 128   -> VGPR
    global staging   4 x half-tile f16  =  64   -> VGPR
                                          ----
                                          448
"""

import os

import torch
import triton
from common import get_pids
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store, mfma

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64

NUM_WARPS = 4
WARPS_M = 2
WARPS_N = 2

NUM_XCDS = 8
GROUP_SIZE_M = 4

# main loop runs range(0, iterMax - 2) and two K-steps are peeled
MIN_K = 4 * BLOCK_K
KERNEL_NAME = "a16w16_intra_wave_gfx942"

# SwizzledSharedLayout(vec, per_phase, max_phase); see tools/lds_conflict.py.
# gl.constexpr instances because a @gluon.jit body may only read globals that
# are already constexpr.
SWZ_VEC = gl.constexpr(int(os.environ.get("GFX942_SWZ_VEC", 8)))
SWZ_PER_PHASE = gl.constexpr(int(os.environ.get("GFX942_SWZ_PER_PHASE", 1)))
SWZ_MAX_PHASE = gl.constexpr(int(os.environ.get("GFX942_SWZ_MAX_PHASE", 8)))


@gluon.jit
def a16w16_intra_wave_gfx942(
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
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # ---- global-load layouts -------------------------------------------------
    # Both operands are K-contiguous.  Eight lanes cover one whole 128 B row of
    # the tile (K=64 fp16), and *consecutive* lanes walk consecutive rows, so a
    # single buffer_load_dwordx4 touches 8 back-to-back cache lines (1024 B
    # contiguous).  Walking consecutive rows -- rather than the gfx950
    # tutorial's stride-16 rows -- is also what makes the following
    # `local_store` bank-conflict free: lanes 0-15 then cover two adjacent
    # 128 B rows = 256 B = all 64 banks exactly once.
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [16, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 16]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 4]],
        warp_bases=[[0, 32], [0, 64]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

    # ---- shared layouts ------------------------------------------------------
    # swizzle(vec=8, per_phase=1, max_phase=8): the 16 B chunk index inside a
    # 128 B row is xored with row % 8.  Measured conflict-free in both
    # directions; per_phase=2 (which an earlier version of this kernel used, and
    # which tools/layout_check.py's model wrongly reports as clean) costs 67%
    # extra LDS cycles.  Costs no LDS unlike padding, which matters because the
    # four slots below are exactly 64 KB.
    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(
        SWZ_VEC, SWZ_PER_PHASE, SWZ_MAX_PHASE, order=[1, 0]
    )
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(
        SWZ_VEC, SWZ_PER_PHASE, SWZ_MAX_PHASE, order=[0, 1]
    )

    # ---- mfma / dot-operand layouts -----------------------------------------
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

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

    ## ------------------------------------------------------------------
    ## Prologue: GR tile 0 -> LW tile 0 -> GR tile 1, then prime the two
    ## operands that region 0 consumes.  On exit `a_base`/`b_base` point at
    ## tile 2, which is what the loop body fetches.
    ## ------------------------------------------------------------------
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

    ## Pin the prologue's global-load order to the order the loop body consumes
    ## them in.  Without this the MachineScheduler sinks B_left's four loads to
    ## the end of the prologue, so the preheader issues the 16 loop-carried
    ## loads rotated by 12 relative to the loop body.  SIInsertWaitcnts joins
    ## the two paths with a per-register `max` of the two age scores, and under
    ## that rotation the four *oldest* in-flight loads inherit the scores of the
    ## four youngest -- collapsing region 0's waits from vmcnt(15)/(13)/(12) to
    ## vmcnt(3)/(1)/(0), i.e. draining the whole 16-deep queue built to hide HBM
    ## latency.  s_barrier has unmodeled side effects, so it fences the
    ## scheduler; Gluon exposes no sched.barrier.  See note.md Opt 5.
    gl.barrier()

    gA_top = buffer_load(ptr=a_base, offsets=a_top_offsets)
    gA_bot = buffer_load(ptr=a_base, offsets=a_bot_offsets)
    gB_right = buffer_load(ptr=b_base, offsets=b_right_offsets)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    b_left = smemB_left.load(dotOpLayoutB)
    a_top = smemA_top.load(dotOpLayoutA)

    ## Close the prologue's pending LDS reads before entering the loop.
    ##
    ## Without this, `b_left`/`a_top` above are still outstanding in membar's
    ## BlockInfo at the loop header. That set is joined into the loop-body
    ## input, where it collides with region 0's `smemB_left.store` as a WAR
    ## hazard -- one that can only occur on the *first* iteration. membar
    ## resolves it the only way it can, by planting a barrier inside the loop
    ## body, so every iteration pays for a one-time condition.
    ##
    ## Draining here costs one barrier in the prologue and removes one from
    ## every K-step: 3 barriers/iteration -> 2, and 3 `s_waitcnt lgkmcnt(0)`
    ## -> 1. See note.md Opt 3.
    gl.barrier()

    ## ------------------------------------------------------------------
    ## Main loop: four regions, each one DOT + one LR + one LW + one GR.
    ## ------------------------------------------------------------------
    for k in range(0, iterMax - 2):

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = mfma(a_top, b_left, acc_tl)

        a_bot = smemA_bot.load(dotOpLayoutA)  # LR A_bot(k)
        smemB_left.store(gB_left)  # LW B_left(k+1)
        gB_left = buffer_load(ptr=b_base, offsets=b_left_offsets)  # GR B_left(k+2)

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = mfma(a_bot, b_left, acc_bl)

        b_right = smemB_right.load(dotOpLayoutB)  # LR B_right(k)
        smemA_top.store(gA_top)  # LW A_top(k+1)
        gA_top = buffer_load(ptr=a_base, offsets=a_top_offsets)  # GR A_top(k+2)

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = mfma(a_top, b_right, acc_tr)

        b_left = smemB_left.load(dotOpLayoutB)  # LR B_left(k+1)
        smemA_bot.store(gA_bot)  # LW A_bot(k+1)
        gA_bot = buffer_load(ptr=a_base, offsets=a_bot_offsets)  # GR A_bot(k+2)

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = mfma(a_bot, b_right, acc_br)

        a_top = smemA_top.load(dotOpLayoutA)  # LR A_top(k+1)
        smemB_right.store(gB_right)  # LW B_right(k+1)
        gB_right = buffer_load(ptr=b_base, offsets=b_right_offsets)  # GR B_right(k+2)

        a_base += BLOCK_K * stride_ak
        b_base += BLOCK_K * stride_bk

    ## ------------------------------------------------------------------
    ## Peel 1 (k = iterMax - 2): still stores the last tile into LDS, but
    ## there is nothing left to prefetch from global.
    ## ------------------------------------------------------------------
    acc_tl = mfma(a_top, b_left, acc_tl)
    a_bot = smemA_bot.load(dotOpLayoutA)
    smemB_left.store(gB_left)

    acc_bl = mfma(a_bot, b_left, acc_bl)
    b_right = smemB_right.load(dotOpLayoutB)
    smemA_top.store(gA_top)

    acc_tr = mfma(a_top, b_right, acc_tr)
    b_left = smemB_left.load(dotOpLayoutB)
    smemA_bot.store(gA_bot)

    acc_br = mfma(a_bot, b_right, acc_br)
    a_top = smemA_top.load(dotOpLayoutA)
    smemB_right.store(gB_right)

    ## ------------------------------------------------------------------
    ## Peel 2 (k = iterMax - 1): drain -- no LW, no GR, and the two LRs that
    ## would fetch tile iterMax are dropped.
    ## ------------------------------------------------------------------
    acc_tl = mfma(a_top, b_left, acc_tl)
    a_bot = smemA_bot.load(dotOpLayoutA)

    acc_bl = mfma(a_bot, b_left, acc_bl)
    b_right = smemB_right.load(dotOpLayoutB)

    acc_tr = mfma(a_top, b_right, acc_tr)
    acc_br = mfma(a_bot, b_right, acc_br)

    ## ------------------------------------------------------------------
    ## Epilogue: store the four quadrants straight from the mfma layout.
    ##
    ## The gfx950 v9 kernel converts to a BlockedLayout first to widen the
    ## stores to dwordx4; that conversion needs an LDS scratch buffer, and LDS
    ## here is full to the byte.  Storing in the mfma layout (dwordx2, since a
    ## transposed 16x16 tile gives each lane 4 consecutive N elements) needs no
    ## scratch.  One shared within-quadrant offset tensor + four scalar bases
    ## keeps the epilogue's VGPR footprint low.
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


def _agpr_attrs():
    """force-agpr RA hint.

    The 256 f32 accumulator registers do not fit alongside the 192 VGPRs of dot
    operands + global staging, so the accumulators must live in AGPRs.  Pair
    with TRITON_FORCE_MFMA_AGPR=1, which additionally sets
    amdgpu-mfma-vgpr-form=0 so LLVM does not fall back to the VGPR form.
    """
    return "amdgpu-agpr-alloc=256" if os.environ.get("TRITON_FORCE_MFMA_AGPR") else ""


def matmul_kernel_only(a: torch.Tensor, b_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Kernel-only entry.

    `b_t` is B pre-transposed to (N, K) and contiguous; the kernel sees a
    logical (K, N) operand with K contiguous via the strides below.
    """
    M, K = a.shape
    N = b_t.shape[0]
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    a16w16_intra_wave_gfx942[(GRID_MN,)](
        a,
        b_t,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b_t.stride(1),
        b_t.stride(0),  # stride_bk = 1 (K contiguous), stride_bn = K
        c.stride(0),
        c.stride(1),  #
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
        GRID_MN=GRID_MN,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=NUM_WARPS,
        llvm_fn_attrs=_agpr_attrs(),
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
