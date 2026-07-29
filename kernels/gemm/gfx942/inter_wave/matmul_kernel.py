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

8 warps/CTA = 2 waves/SIMD, `warps_per_cta = [2, 4]`, tile 256x256x64, MFMA
`v_mfma_f32_16x16x16_f16`.  The loop is 8 regions -- four memory, four dot,
strictly alternating -- so the two wave groups ping-pong: while one group is on
the matrix unit the other issues its global loads, LDS reads and LDS writes.

WHOLE-TENSOR GLOBAL, SLICED LDS READS
-------------------------------------
Global and LDS *writes* move whole tensors:

    GR A / LW A     A[256 x 64]   ->  4 x buffer_load_dwordx4 / 4 x ds_write_b128
    GR B / LW B     B[64 x 256]   ->  4 x buffer_load_dwordx4 / 4 x ds_write_b128

LDS *reads* and the dots work on half-tiles, sliced out of the same buffers:

    A_t = A[0:128, :]   A_b = A[128:256, :]     B_l = B[:, 0:128]   B_r = B[:, 128:256]

The slice is along M (for A) or N (for B) -- never along K, which is the
swizzled dimension -- so `SwizzledSharedLayout(8, 2, 8)` is undisturbed: the
swizzle xors the 16 B chunk index within a row by (row / 2) % 8, a period of 16
rows, and both 128 and the N-half boundary are multiples of it.

ONE LDS BUFFER
--------------
    A[256 x 64] x 2 B  +  B[64 x 256] x 2 B  =  32 KB + 32 KB  =  64 KB
which is exactly MI300X's LDS, so there is no room for double buffering and none
is needed: the pipelining buffer is the *registers* holding the in-flight global
loads, not LDS.  LDS holds tile k while tile k+1 is in flight from HBM; the
`local_store` of tile k+1 lands only after the last `local_load` of tile k.

REGION SCHEDULE  (region k processes tile k and prefetches tile k+1)
--------------------------------------------------------------------
    region 0  mem   GR B[k+1] , LR A_t[k]     4 x buffer_load_dwordx4 + 8 x ds_read_b128
    region 1  dot   C_tl += A_t x B_l         32 x mfma
    region 2  mem   GR A[k+1] , LR B_r[k]     4 x buffer_load_dwordx4 + 4 x ds_read_b128
    region 3  dot   C_tr += A_t x B_r         32 x mfma
    region 4  mem   LR A_b[k] , LW B[k+1]     8 x ds_read_b128 + 4 x ds_write_b128
    region 5  dot   C_bl += A_b x B_l         32 x mfma
    region 6  mem   LR B_l[k+1] , LW A[k+1]   4 x ds_read_b128 + 4 x ds_write_b128
    region 7  dot   C_br += A_b x B_r         32 x mfma

WHY THIS ORDER
--------------
*Dot order* `C_tl -> C_tr -> C_bl -> C_br` and *read order*
`B_l -> A_t -> B_r -> A_b` are chosen together to halve the operand registers.
A_t is live only across regions 1-3 and A_b only across 5-7, so **the two share
one register set**; B_l (used in 1 and 5) and B_r (used in 3 and 7) both span the
whole body and need their own.  Per lane, replicated across the warp grid:

    A_t / A_b   [128x64] over WARPS_M=2   = 32 VGPR  (shared)
    B_l , B_r   [64x128] over WARPS_N=4   = 16 VGPR each
    C quadrants 4 x [128x128] f32         = 128 VGPR
    A,B staging whole tiles / 8 waves     = 32 VGPR

Loading all four half-tiles with independent registers would cost 96 instead of
64 and spills; that is what the earlier K=32-sliced version worked around, at
the price of twice as many regions.

*GR/LW order* is B before A on both sides.  B[k+1] is stored in region 4 and A
in region 6, each 2 dot regions (~1024 cycles) after its load, which is the
global latency cover.  B is stored first because `LR B_l[k+1]` in region 6 needs
it: B_l for the *next* iteration is read one region after B's own store, which
is what lets region 1 start on a dot with no preceding read.

*Hazards* are all closed by region boundaries: A is read in regions 0 and 4 and
written in 6; B is read in 2 (and 6, for k+1) and written in 4.
"""

import os

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

NUM_WARPS = 8
WARPS_M = 2
WARPS_N = 4

NUM_XCDS = 8
GROUP_SIZE_M = 4

# main loop runs range(0, iterMax - 1); one K-step is peeled
MIN_K = 3 * BLOCK_K
KERNEL_NAME = "a16w16_inter_wave_gfx942"

# SwizzledSharedLayout(vec, per_phase, max_phase) for both LDS buffers.
#
# per_phase=1, NOT 2.  The global-load layout has consecutive lanes walking
# consecutive rows, so lanes 0-15 of a ds_read_b128 cover two adjacent rows.
# With per_phase=2 those two rows share a swizzle phase, hit the same 32 banks,
# and every access replays: measured SQ_LDS_BANK_CONFLICT ratio 0.75, i.e. 14
# cycles per LDS instruction against a conflict-free 8.  per_phase=1 advances
# the phase every row and measures exactly 0 conflicts.  vec=8 keeps the 16 B
# ds_read_b128 / ds_write_b128 width.  See tools/lds_conflict.py --sweep.
#
# Overridable (and gl.constexpr instances, since a @gluon.jit body may only read
# globals that are already constexpr) so the sweep can drive them.
SWZ_VEC = gl.constexpr(int(os.environ.get("GFX942_SWZ_VEC", 8)))
SWZ_PER_PHASE = gl.constexpr(int(os.environ.get("GFX942_SWZ_PER_PHASE", 1)))
SWZ_MAX_PHASE = gl.constexpr(int(os.environ.get("GFX942_SWZ_MAX_PHASE", 8)))


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
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,  #
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    pid_m, pid_n = get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # ---- global-load layouts: whole tensors, not half-tiles ------------------
    # Eight lanes cover one whole 128 B row and consecutive lanes walk
    # consecutive rows, so a buffer_load_dwordx4 reads 1024 contiguous bytes (8
    # back-to-back cache lines) and the matching local_store has lanes 0-15
    # covering two adjacent rows = 256 B = all 64 banks exactly once.  The final
    # register base ([128,0] for A, [0,128] for B) is what widens these from the
    # half-tile layouts to the full [256x64] / [64x256] tensors: 32 elements per
    # lane = 64 B = 4 x dwordx4 per wave.
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [128, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=[[16, 0], [32, 0], [64, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K],
    )
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 128]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 1], [0, 2], [0, 4]],
        warp_bases=[[0, 16], [0, 32], [0, 64]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N],
    )

    sharedLayoutA: gl.constexpr = gl.SwizzledSharedLayout(
        SWZ_VEC, SWZ_PER_PHASE, SWZ_MAX_PHASE, order=[1, 0]
    )
    sharedLayoutB: gl.constexpr = gl.SwizzledSharedLayout(
        SWZ_VEC, SWZ_PER_PHASE, SWZ_MAX_PHASE, order=[0, 1]
    )

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfmaLayout, k_width=8)
    dotB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfmaLayout, k_width=8)

    # ---- LDS: one buffer, whole tensors, 64 KB exactly ----------------------
    smemA = gl.allocate_shared_memory(a_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], sharedLayoutA)
    smemB = gl.allocate_shared_memory(b_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], sharedLayoutB)

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    b_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)
    gl.assume(iterMax > 1)

    ## ------------------------------------------------------------------
    ## Prologue: GR B[0], A[0] -> LW B[0], A[0] -> LR B_l[0].
    ## B before A on both sides, matching the loop.
    ## ------------------------------------------------------------------
    gB = buffer_load(ptr=b_base, offsets=b_offsets)
    gA = buffer_load(ptr=a_base, offsets=a_offsets)
    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    smemB.store(gB)
    smemA.store(gA)

    b_l = smemB.slice(0, BLOCK_N // 2, 1).load(dotB)

    ## ------------------------------------------------------------------
    ## Main loop: 8 regions, mem and dot strictly alternating.  Each mem
    ## region carries the higher s_setprio so its address VALU keeps issuing
    ## while the other wave group holds the matrix unit.
    ## ------------------------------------------------------------------
    for k in range(0, iterMax - 1):

        ## region 0 -- mem: GR B[k+1], LR A_t[k]
        with warp_pipeline_stage("mem", priority=1):
            gB = buffer_load(ptr=b_base, offsets=b_offsets)
            a_t = smemA.slice(0, BLOCK_M // 2, 0).load(dotA)

        ## region 1 -- dot: C_tl
        with warp_pipeline_stage("mfma", priority=0):
            acc_tl = mfma(a_t, b_l, acc_tl)

        ## region 2 -- mem: GR A[k+1], LR B_r[k]
        with warp_pipeline_stage("mem", priority=1):
            gA = buffer_load(ptr=a_base, offsets=a_offsets)
            b_r = smemB.slice(BLOCK_N // 2, BLOCK_N // 2, 1).load(dotB)

        ## region 3 -- dot: C_tr  (last use of A_t; A_b may reuse its registers)
        with warp_pipeline_stage("mfma", priority=0):
            acc_tr = mfma(a_t, b_r, acc_tr)

        ## region 4 -- mem: LR A_b[k], LW B[k+1]  (B_r[k] read in r2, so free)
        with warp_pipeline_stage("mem", priority=1):
            a_b = smemA.slice(BLOCK_M // 2, BLOCK_M // 2, 0).load(dotA)
            smemB.store(gB)

        ## region 5 -- dot: C_bl  (last use of B_l[k])
        with warp_pipeline_stage("mfma", priority=0):
            acc_bl = mfma(a_b, b_l, acc_bl)

        ## region 6 -- mem: LR B_l[k+1], LW A[k+1]  (A_b[k] read in r4, so free)
        with warp_pipeline_stage("mem", priority=1):
            b_l = smemB.slice(0, BLOCK_N // 2, 1).load(dotB)
            smemA.store(gA)
            a_base += BLOCK_K * stride_ak
            b_base += BLOCK_K * stride_bk

        ## region 7 -- dot: C_br
        with warp_pipeline_stage("mfma", priority=0):
            acc_br = mfma(a_b, b_r, acc_br)

    ## ------------------------------------------------------------------
    ## Peel (k = iterMax - 1): drain.  B_l was read in the last region 6, so
    ## only the three remaining half-tile reads are left; no GR, no LW.
    ## ------------------------------------------------------------------
    a_t = smemA.slice(0, BLOCK_M // 2, 0).load(dotA)
    acc_tl = mfma(a_t, b_l, acc_tl)

    b_r = smemB.slice(BLOCK_N // 2, BLOCK_N // 2, 1).load(dotB)
    acc_tr = mfma(a_t, b_r, acc_tr)

    a_b = smemA.slice(BLOCK_M // 2, BLOCK_M // 2, 0).load(dotA)
    acc_bl = mfma(a_b, b_l, acc_bl)
    acc_br = mfma(a_b, b_r, acc_br)

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
