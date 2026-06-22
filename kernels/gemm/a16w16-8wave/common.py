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
Common utilities for the 8-wave warp-pipeline GEMM kernel (gfx950 / CDNA4).

Ported verbatim from AMD-Triton/gluon-kernels
(kernels/cdna4/gemm/f16_gemm_common_gfx950.py): PID mapping, store epilogue,
and async-load helpers used by the warp-pipeline kernel.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async


@gluon.jit
def get_pid_m_n(
    M, N,  #
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
    GROUP_M: gl.constexpr, XCD_CHUNK: gl.constexpr, NUM_XCDS: gl.constexpr,
):
    """PID mapping with GROUP_M tiling for L2 locality and XCD remapping for MI350."""
    pid0 = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    grid_mn = num_pid_m * num_pid_n

    # GROUP_M tiling for L2 cache locality
    num_pid_in_group = num_pid_n * GROUP_M
    group_id = pid0 // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m_local = pid0 % group_size_m
    pid_m_cache = first_pid_m + pid_m_local
    pid_n_cache = (pid0 % num_pid_in_group) // group_size_m

    pid_cache = pid_m_cache * num_pid_n + pid_n_cache
    pid_cache = gl.where(pid_cache < grid_mn, pid_cache, pid0)

    # XCD remapping for MI350
    full = (grid_mn // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)

    xcd = pid_cache % NUM_XCDS
    local_pid = pid_cache // NUM_XCDS
    chunk_id = local_pid // XCD_CHUNK
    pos = local_pid - chunk_id * XCD_CHUNK
    pid1_remap = chunk_id * (NUM_XCDS * XCD_CHUNK) + xcd * XCD_CHUNK + pos
    pid1 = gl.where(pid_cache < full, pid1_remap, pid_cache)

    pid_m = pid1 // num_pid_n
    pid_n = pid1 - pid_m * num_pid_n
    return pid_m, pid_n


@gluon.jit
def get_pids(
    M, N,  #
    BM: gl.constexpr, BN: gl.constexpr,  #
    GRID_MN: gl.constexpr, NUM_XCDS: gl.constexpr, GROUP_SIZE_M: gl.constexpr,
):
    """XCD-aware PID remapping + GROUP_SIZE_M swizzle, copied verbatim from the
    a16w16 v9 tutorial kernel (v9_beyond_hotloop). Active at any grid_mn (unlike
    get_pid_m_n, whose XCD_CHUNK=128 gate makes it a no-op below 1024 tiles)."""
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
def store_result(
    acc, c_ptr, c_dtype,  #
    pid_m, pid_n, M, N,  #
    stride_cm, stride_cn,  #
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,  #
    STORE_LAYOUT_C: gl.constexpr,
):
    """Convert accumulator and store to global memory with masking."""
    c = acc.to(c_dtype)
    c = gl.convert_layout(c, layout=STORE_LAYOUT_C)

    offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE_LAYOUT_C))
    offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, STORE_LAYOUT_C))

    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.cdna4.buffer_store(ptr=c_base, offsets=c_offsets, stored_value=c, mask=c_mask)


@gluon.jit
def issue_async_load_a(
    a_smem, a_base, k_idx,  #
    stride_am, stride_ak,  #
    BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr,  #
    A_ASYNC_LAYOUT: gl.constexpr,
):
    """Issue async load for A tile into shared memory. Does NOT commit."""
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=A_ASYNC_LAYOUT)
    offs_k_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=A_ASYNC_LAYOUT)

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_k = gl.arange(0, BLOCK_K, layout=offs_k_layout)

    k_start = k_idx * BLOCK_K
    a_offsets = offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak

    cdna4_async.buffer_load_to_shared(a_smem, a_base, a_offsets)
    # NOTE: Caller commits after both A and B are issued.


@gluon.jit
def issue_async_load_b(
    b_smem, b_base, k_idx,  #
    stride_bk, stride_bn,  #
    BLOCK_K: gl.constexpr, BLOCK_N: gl.constexpr,  #
    B_ASYNC_LAYOUT: gl.constexpr,
):
    """Issue async load for B tile into shared memory. Does NOT commit."""
    offs_k_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=B_ASYNC_LAYOUT)
    offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=B_ASYNC_LAYOUT)

    offs_k = gl.arange(0, BLOCK_K, layout=offs_k_layout)
    offs_n = gl.arange(0, BLOCK_N, layout=offs_n_layout)

    k_start = k_idx * BLOCK_K
    b_offsets = (k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

    cdna4_async.buffer_load_to_shared(b_smem, b_base, b_offsets)
    # NOTE: Caller commits after both A and B are issued.
