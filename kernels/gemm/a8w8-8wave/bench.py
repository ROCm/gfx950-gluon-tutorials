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

Mirrors kernels/gemm/a16w16-8wave/bench.py (same --K / --rocprof /
--rotating-buffer-size args and the do_bench + rocprof-rotating-tensor
mechanisms) but with BF8 (float8_e5m2) inputs and an fp16 output, matching the
4-wave kernels/gemm/a8w8/bench.py numerics. B is pre-transposed to (N, K)
contiguous outside the timed region since the kernel needs K contiguous.

--version selects the kernel subdir (only v1_sliceMN_BK128_nS2 exists here).
"""

import argparse
import importlib

import torch
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Versioned kernels live in subdirs, mirroring the a16w16-8wave layout.
VERSION_MAP = {1: "v1_sliceMN_BK128_nS2"}

# Rebound in main() once the selected version module is imported.
matmul_kernel_only = None
MIN_K = None
KERNEL_NAME = None


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
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        choices=sorted(VERSION_MAP),
        help="Kernel version: 1=v1_sliceMN_BK128_nS2. Default: 1",
    )
    return parser.parse_args()


def test_correctness(gemm_sizes):
    for M, N, K in gemm_sizes:
        if K < MIN_K:
            print(f"[a8w8-8wave] {M=} {N=} {K=}: SKIPPED (K < {MIN_K})")
            continue

        # BF8 inputs. a is (M, K) K-contiguous; b_t is B stored transposed (N, K)
        # contiguous (K contiguous, as the kernel requires). Math operand B = b_t.t().
        a = (torch.rand((M, K), device=DEVICE, dtype=torch.float16) - 0.5).to(torch.float8_e5m2)
        b_t = (torch.rand((N, K), device=DEVICE, dtype=torch.float16) - 0.5).to(torch.float8_e5m2)
        c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

        triton_output = matmul_kernel_only(a, b_t, c)
        torch_output = torch.matmul(a.to(torch.float16), b_t.t().to(torch.float16))

        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"[a8w8-8wave] {M=} {N=} {K=} dtype=f8: ✅ Triton and Torch match")
        else:
            max_diff = (triton_output - torch_output).abs().max().item()
            print(
                f"[a8w8-8wave] {M=} {N=} {K=} dtype=f8: ❌ Triton and Torch differ (max_diff={max_diff:.4f})"
            )


def gen_rotating_tensors(M, N, K, rotating_buffer_size_mb=512):
    """Allocate multiple copies of (a, b_t, c) to exceed GPU cache size.

    Each benchmark iteration uses a different copy via i % block_count, so the
    kernel always starts cold. b_t is (N, K) contiguous (K contiguous), the
    layout the kernel consumes, so no transpose happens inside the timed loop.
    """
    elem_bytes = 1  # float8_e5m2
    a_size = M * K * elem_bytes
    b_size = K * N * elem_bytes
    c_size = M * N * 2  # fp16 output
    total_size = a_size + b_size + c_size

    block_count = max(1, rotating_buffer_size_mb * 1024 * 1024 // total_size)

    a_list, b_list, c_list = [], [], []
    for _ in range(block_count):
        a_list.append(torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2))
        b_list.append(torch.randn((N, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2))
        c_list.append(torch.empty((M, N), device=DEVICE, dtype=torch.float16))

    return a_list, b_list, c_list, block_count


def run_rocprof_iterations(gemm_sizes, n_iters=1000, rotating_buffer_size_mb=512):
    """Run the kernel n_iters times for each size using rotating tensors.

    Rotating tensors defeat GPU cache and produce cold-cache timings. Designed
    to be wrapped by rocprofv3 --kernel-trace for external timing.
    """
    for M, N, K in gemm_sizes:
        if K < MIN_K:
            print(f"[a8w8-8wave] {M=} {N=} {K=}: SKIPPED (K < {MIN_K})")
            continue
        a_list, b_list, c_list, block_count = gen_rotating_tensors(M, N, K, rotating_buffer_size_mb)
        print(
            f"[a8w8-8wave] {M=} {N=} {K=}: "
            f"rotating tensors: {block_count} copies, "
            f"{block_count * (M*K + K*N + M*N*2) / 1024**2:.0f} MB"
        )
        # Warmup
        matmul_kernel_only(a_list[0], b_list[0], c_list[0])
        torch.cuda.synchronize()
        for i in range(n_iters):
            idx = i % block_count
            matmul_kernel_only(a_list[idx], b_list[idx], c_list[idx])
        torch.cuda.synchronize()
        print(f"[a8w8-8wave] {M=} {N=} {K=}: {n_iters} iterations done")


def main():
    args = parse_args()

    global matmul_kernel_only, MIN_K, KERNEL_NAME
    version_dir = VERSION_MAP[args.version]
    km = importlib.import_module(f"{version_dir}.matmul_kernel")
    matmul_kernel_only = km.matmul_kernel_only
    MIN_K = km.MIN_K
    KERNEL_NAME = km.KERNEL_NAME
    print(f"[a8w8-8wave] version={args.version} ({version_dir})  kernel={KERNEL_NAME}")

    gemm_sizes = get_gemm_sizes(args.K)

    test_correctness(gemm_sizes)

    if args.rocprof:
        run_rocprof_iterations(
            gemm_sizes,
            rotating_buffer_size_mb=args.rotating_buffer_size,
        )
        return

    configs = [
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=gemm_sizes,
            line_arg="dtype",
            line_vals=["f8"],
            line_names=["f8"],
            styles=[("green", "-")],
            ylabel="TFLOPS",
            plot_name="matmul-performance-a8w8-8wave",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, dtype):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
        b_t = torch.randn((N, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
        c_out = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_kernel_only(a, b_t, c_out), quantiles=quantiles
        )

        def perf(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    print(f"\na8w8-8wave ({KERNEL_NAME}):")
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
