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
Benchmark + correctness driver for the 8-wave warp-pipeline a8w8 (BF8) GEMM.

Mirrors kernels/gemm/inter_wave/a16w16/bench.py (same --K / --rocprof /
--rotating-buffer-size args and the do_bench + rocprof-rotating-tensor
mechanisms) but with BF8 (float8_e5m2) inputs and an fp16 output, matching the
4-wave kernels/gemm/a8w8/bench.py numerics. B is pre-transposed to (N, K)
contiguous outside the timed region since the kernel needs K contiguous.
"""

import argparse
import os
import sys

import torch
import triton

# Put the shared kernels/gemm/utils/ on the path so the kernel's
# `from common import get_pids` resolves to the shared helper.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
# scripts/graph_timer.py: the tutorial's timing protocol (graph replay, warm-up, rotating sets).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts")
)
import graph_timer  # noqa: E402
from matmul_kernel import KERNEL_NAME, MIN_K, matmul_kernel_only  # noqa: E402

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_x_vals():
    return [
        (4096, 4096, 512),
        (4096, 4096, 1024),
        (4096, 4096, 2048),
        (4096, 4096, 3072),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
        (4096, 4096, 32768),
    ]


def get_gemm_sizes(selected_k=None):
    sizes = get_x_vals()

    if selected_k is None:
        return sizes

    filtered = [s for s in sizes if s[2] == selected_k]

    if not filtered:
        raise ValueError(
            f"No GEMM size found with K={selected_k}. "
            f"Available K values: {[k for _, _, k in sizes]}"
        )

    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description="8-wave warp-pipeline a8w8 (BF8) GEMM benchmark")
    parser.add_argument("--K", type=int, default=None, help="Select GEMM problem size with given K")
    parser.add_argument(
        "--rocprof",
        action="store_true",
        help="Profiler mode: instead of timing, issue --warmup-dispatches + --iters eager dispatches "
        "back to back for an external profiler (rocprofv3 --pmc / --att / --kernel-trace).",
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="profiler mode: dispatches after the warm-up"
    )
    parser.add_argument(
        "--warmup-dispatches",
        type=int,
        default=0,
        help="profiler mode: dispatches before the --iters ones (see graph_timer.profile_loop)",
    )
    parser.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=graph_timer.COLD_INPUT_MIB,
        help="MiB of input tensors to rotate through (at least 2 sets), so every dispatch reads "
        f"its inputs from HBM (default: {graph_timer.COLD_INPUT_MIB}, twice the 256 MiB MALL)",
    )
    graph_timer.add_timing_args(parser)
    return parser.parse_args()


def test_correctness(gemm_sizes):
    for M, N, K in gemm_sizes:
        if K < MIN_K:
            print(f"[inter_wave/a8w8] {M=} {N=} {K=}: SKIPPED (K < {MIN_K})")
            continue

        # BF8 inputs. a is (M, K) K-contiguous; b_t is B stored transposed (N, K)
        # contiguous (K contiguous, as the kernel requires). Math operand B = b_t.t().
        a = (torch.rand((M, K), device=DEVICE, dtype=torch.float16) - 0.5).to(torch.float8_e5m2)
        b_t = (torch.rand((N, K), device=DEVICE, dtype=torch.float16) - 0.5).to(torch.float8_e5m2)
        c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

        triton_output = matmul_kernel_only(a, b_t, c)
        torch_output = torch.matmul(a.to(torch.float16), b_t.t().to(torch.float16))

        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"[inter_wave/a8w8] {M=} {N=} {K=} dtype=f8: ✅ Triton and Torch match")
        else:
            max_diff = (triton_output - torch_output).abs().max().item()
            print(
                f"[inter_wave/a8w8] {M=} {N=} {K=} dtype=f8: ❌ Triton and Torch differ (max_diff={max_diff:.4f})"
            )


def gen_rotating_tensors(M, N, K, rotating_buffer_size_mb=graph_timer.COLD_INPUT_MIB):
    """Rotating (a, b_t, c) sets: enough that the inputs in rotation exceed rotating_buffer_size_mb.

    Dispatch i uses set i % count, so every dispatch reads its inputs from HBM. b_t is (N, K)
    contiguous (K contiguous), the layout the kernel consumes, so no transpose happens inside the
    timed loop.
    """
    count = graph_timer.rotating_sets(M * K + K * N, rotating_buffer_size_mb)  # 1 byte per fp8
    a_list, b_list, c_list = [], [], []
    for _ in range(count):
        a_list.append(torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2))
        b_list.append(torch.randn((N, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2))
        c_list.append(torch.empty((M, N), device=DEVICE, dtype=torch.float16))
    return a_list, b_list, c_list, count


def main():
    args = parse_args()

    print(f"[inter_wave/a8w8] kernel={KERNEL_NAME}")

    gemm_sizes = get_gemm_sizes(args.K)

    test_correctness(gemm_sizes)

    protocol = graph_timer.protocol_from_args(args)
    for M, N, K in gemm_sizes:
        if K < MIN_K:
            print(f"[inter_wave/a8w8] {M=} {N=} {K=}: SKIPPED (K < {MIN_K})")
            continue
        a_l, b_l, c_l, n_sets = gen_rotating_tensors(M, N, K, args.rotating_buffer_size)

        def launch(slot):
            matmul_kernel_only(a_l[slot], b_l[slot], c_l[slot])

        label = f"{KERNEL_NAME} {M}x{N}x{K} f8"
        if args.rocprof:
            graph_timer.profile_loop(launch, n_sets, args.warmup_dispatches, args.iters)
            print(
                f"[{label}] profiler mode: {args.warmup_dispatches} + {args.iters} dispatches "
                f"over {n_sets} rotating sets done"
            )
            continue
        result = graph_timer.time_dispatches(
            launch, n_sets, protocol, flops=2 * M * N * K, label=label
        )
        result.update(kernel=KERNEL_NAME, M=M, N=N, K=K, dtype="f8")
        graph_timer.print_result(result, args.result_json)
        a_l = b_l = c_l = None  # free the sets
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
