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
Benchmark + correctness driver for the 8-wave warp-pipeline GEMM.

Mirrors kernels/gemm/a16w16/bench.py: same --K / --dtype / --rocprof /
--rotating-buffer-size args and the same do_bench + rocprof-rotating-tensor
mechanisms, so the tutorial's rocprof kernel-trace and ATT tooling can be
pointed at this kernel (see collect_perf.py).

The only 8-wave-specific quirk: B is pre-transposed to (N, K) outside the timed
region (kernel-only timing) since these kernels need K contiguous.
"""

import argparse
import os
import sys

import torch
import triton

# Put the shared kernels/gemm/utils/ on the path so the kernel's
# `from common import get_pids` resolves to the shared helper.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))

from matmul_kernel import KERNEL_NAME, MIN_K, matmul_kernel_only  # noqa: E402

DEVICE = triton.runtime.driver.active.get_active_torch_device()

name_to_torch_type = {"fp16": torch.float16, "bf16": torch.bfloat16}


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


def get_dtypes(selected_dtype=None):
    default_dtypes = ["fp16", "bf16"]

    if selected_dtype is None:
        return default_dtypes

    if isinstance(selected_dtype, str):
        selected_dtype = [selected_dtype]

    invalid = set(selected_dtype) - set(default_dtypes)
    if invalid:
        raise ValueError(
            f"Unsupported dtype(s): {sorted(invalid)}. " f"Supported dtypes: {default_dtypes}"
        )

    return selected_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="8-wave warp-pipeline GEMM benchmark")
    parser.add_argument("--K", type=int, default=None, help="Select GEMM problem size with given K")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["fp16", "bf16"],
        default=None,
        help="Data type(s) to benchmark (default: fp16 bf16)",
    )
    parser.add_argument(
        "--rocprof",
        action="store_true",
        help="Rocprof mode: run kernel 1000 times without do_bench. "
        "Use with rocprofv3 --kernel-trace to measure timing externally.",
    )
    parser.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=512,
        help="Total size (MB) of rotating tensors (a, b, c) for rocprof mode. "
        "Should exceed GPU cache (L2+MALL) size. (default: 512)",
    )
    return parser.parse_args()


def test_correctness(dtype, gemm_sizes):
    torch_dtype = name_to_torch_type[dtype]

    for M, N, K in gemm_sizes:
        if K < MIN_K:
            print(f"[8wave] {M=} {N=} {K=} {dtype=}: SKIPPED (K < {MIN_K})")
            continue

        a = torch.rand((M, K), device=DEVICE, dtype=torch_dtype) - 0.5
        # b_t is B stored transposed (N, K) contiguous — K contiguous, as the
        # kernel requires. The math operand B is b_t.t() of shape (K, N).
        b_t = torch.rand((N, K), device=DEVICE, dtype=torch_dtype) - 0.5
        c = torch.empty((M, N), device=DEVICE, dtype=torch_dtype)

        triton_output = matmul_kernel_only(a, b_t, c)
        torch_output = torch.matmul(a, b_t.t())

        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"[8wave] {M=} {N=} {K=} {dtype=}: ✅ Triton and Torch match")
        else:
            max_diff = (triton_output - torch_output).abs().max().item()
            print(
                f"[8wave] {M=} {N=} {K=} {dtype=}: ❌ Triton and Torch differ (max_diff={max_diff:.4f})"
            )


def gen_rotating_tensors(M, N, K, torch_dtype, rotating_buffer_size_mb=512):
    """Allocate multiple copies of (a, b_t, c) to exceed GPU cache size.

    Each iteration of the benchmark loop uses a different copy via i % block_count,
    so cached data from the previous iteration is useless and the kernel always
    starts with cold caches. b_t is (N, K) contiguous (K contiguous) — the layout
    the kernel consumes, so no transpose happens inside the timed loop.
    """
    elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
    a_size = M * K * elem_bytes
    b_size = K * N * elem_bytes
    c_size = M * N * elem_bytes
    total_size = a_size + b_size + c_size

    block_count = max(1, rotating_buffer_size_mb * 1024 * 1024 // total_size)

    a_list, b_list, c_list = [], [], []
    for _ in range(block_count):
        a_list.append(torch.randn((M, K), device=DEVICE, dtype=torch_dtype))
        b_list.append(torch.randn((N, K), device=DEVICE, dtype=torch_dtype))
        c_list.append(torch.empty((M, N), device=DEVICE, dtype=torch_dtype))

    return a_list, b_list, c_list, block_count


def run_rocprof_iterations(dtypes, gemm_sizes, n_iters=1000, rotating_buffer_size_mb=512):
    """Run the kernel n_iters times for each dtype/size combo using rotating tensors.

    Rotating tensors ensure each iteration reads from different memory addresses,
    defeating GPU cache and producing cold-cache timings similar to real workloads.
    Designed to be wrapped by rocprofv3 --kernel-trace for external timing.
    """
    for dtype in dtypes:
        torch_dtype = name_to_torch_type[dtype]
        for M, N, K in gemm_sizes:
            if K < MIN_K:
                print(f"[8wave] {M=} {N=} {K=} {dtype=}: SKIPPED (K < {MIN_K})")
                continue
            a_list, b_list, c_list, block_count = gen_rotating_tensors(
                M, N, K, torch_dtype, rotating_buffer_size_mb
            )
            print(
                f"[8wave] {M=} {N=} {K=} {dtype=}: "
                f"rotating tensors: {block_count} copies, "
                f"{block_count * (M*K + K*N + M*N) * a_list[0].element_size() / 1024**2:.0f} MB"
            )
            # Warmup
            matmul_kernel_only(a_list[0], b_list[0], c_list[0])
            torch.cuda.synchronize()
            for i in range(n_iters):
                idx = i % block_count
                matmul_kernel_only(a_list[idx], b_list[idx], c_list[idx])
            torch.cuda.synchronize()
            print(f"[8wave] {M=} {N=} {K=} {dtype=}: {n_iters} iterations done")


def main():
    args = parse_args()

    print(f"[8wave] inter_wave/a16w16  kernel={KERNEL_NAME}")

    gemm_sizes = get_gemm_sizes(args.K)
    dtypes = get_dtypes(args.dtype)

    for dtype in dtypes:
        test_correctness(dtype, gemm_sizes)

    if args.rocprof:
        run_rocprof_iterations(
            dtypes,
            gemm_sizes,
            rotating_buffer_size_mb=args.rotating_buffer_size,
        )
        return

    configs = [
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=gemm_sizes,
            line_arg="dtype",
            line_vals=dtypes,
            line_names=dtypes,
            styles=[("green", "-"), ("yellow", "--"), ("red", "--")],
            ylabel="TFLOPS",
            plot_name="matmul-performance-inter_wave/a16w16",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, dtype):
        torch_dtype = name_to_torch_type[dtype]
        a = torch.randn((M, K), device=DEVICE, dtype=torch_dtype)
        b_t = torch.randn((N, K), device=DEVICE, dtype=torch_dtype)
        c_out = torch.empty((M, N), device=DEVICE, dtype=torch_dtype)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_kernel_only(a, b_t, c_out), quantiles=quantiles
        )

        def perf(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    print(f"\ninter_wave/a16w16 ({KERNEL_NAME}):")
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
