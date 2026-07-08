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

import argparse

# The out-of-tree LLIR scheduler ships as an LLVM pass plugin (see ../../../plugins/).
# Loaded via LLVM_PASS_PLUGIN_PATH, it resolves LLVM symbols from libtriton at
# dlopen time, which requires libtriton in the *global* symbol scope. CPython
# loads C-extensions RTLD_LOCAL by default, so opt into RTLD_GLOBAL before the
# first `import triton`. Only takes effect when the plugin is in use.
import os
import sys

import torch

if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import triton

# Out-of-tree amdgcnas peephole (post-assembly): install the amdgcn-stage hook
# when TRITON_AMDGCNAS_PLUGIN is set. Pure-Python text transform, no rebuild.
if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plugins", "amdgcnas"
        ),
    )
    import amdgcnas_plugin
    from triton import knobs

    knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook
from matmul_kernel import matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()

name_to_torch_type = {"fp16": torch.float16, "bf16": torch.bfloat16}


def get_x_vals():
    return [
        (4096, 4096, 1024),
        (4096, 4096, 2048),
        (4096, 4096, 3072),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 4096, 16384),
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

    parser = argparse.ArgumentParser(description="GEMM benchmark")
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
        help="Total size (MB) of rotating tensors (a, b) for rocprof mode. "
        "Should exceed GPU cache (L2+MALL) size. (default: 512)",
    )
    return parser.parse_args()


def test_correctness(gemm_sizes):
    torch_dtype = torch.float16

    for M, N, K in gemm_sizes:
        a = torch.rand((M, K), device=DEVICE, dtype=torch_dtype) - 0.5
        b = torch.rand((N, K), device=DEVICE, dtype=torch_dtype).T - 0.5
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"[a8w8] {M=} {N=} {K=} dtype=f8: ✅ Triton and Torch match")
        else:
            print(f"[a8w8] {M=} {N=} {K=} dtype=f8: ❌ Triton and Torch differ")


def gen_rotating_tensors(M, N, K, rotating_buffer_size_mb=512):
    """Allocate multiple copies of (a, b) tensors to exceed GPU cache size.

    Each iteration of the benchmark loop uses a different copy via i % block_count,
    so cached data from the previous iteration is useless and the kernel always
    starts with cold caches.
    """
    torch_dtype = torch.float8_e5m2
    elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
    a_size = M * K * elem_bytes
    b_size = K * N * elem_bytes
    total_size = a_size + b_size

    block_count = max(1, rotating_buffer_size_mb * 1024 * 1024 // total_size)

    a_list, b_list = [], []
    for _ in range(block_count):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T.to(torch.float8_e5m2)
        a_list.append(a)
        b_list.append(b)

    return a_list, b_list, block_count


def run_rocprof_iterations(gemm_sizes, n_iters=1000, rotating_buffer_size_mb=512):
    """Run the kernel n_iters times for each size using rotating tensors.

    Rotating tensors ensure each iteration reads from different memory addresses,
    defeating GPU cache and producing cold-cache timings similar to real workloads.
    Designed to be wrapped by rocprofv3 --kernel-trace for external timing.
    """
    for M, N, K in gemm_sizes:
        a_list, b_list, block_count = gen_rotating_tensors(M, N, K, rotating_buffer_size_mb)
        print(
            f"[a8w8] {M=} {N=} {K=} dtype=f8: "
            f"rotating tensors: {block_count} copies, "
            f"{block_count * (M*K + K*N) * a_list[0].element_size() / 1024**2:.0f} MB"
        )
        # Warmup
        matmul(a_list[0], b_list[0])
        torch.cuda.synchronize()
        for i in range(n_iters):
            idx = i % block_count
            matmul(a_list[idx], b_list[idx])
        torch.cuda.synchronize()
        print(f"[a8w8] {M=} {N=} {K=} dtype=f8: {n_iters} iterations done")


def main():
    args = parse_args()

    gemm_sizes = get_gemm_sizes(args.K)

    test_correctness(gemm_sizes)

    if args.rocprof:
        run_rocprof_iterations(gemm_sizes, rotating_buffer_size_mb=args.rotating_buffer_size)
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
            plot_name="matmul-performance",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, dtype):
        torch_dtype = torch.float16
        a = torch.randn((M, K), device=DEVICE, dtype=torch_dtype)
        b = torch.randn((N, K), device=DEVICE, dtype=torch_dtype).T
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

        def perf(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    print("\na8w8_kernel:")
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
