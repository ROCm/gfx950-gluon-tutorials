import argparse

import torch
import triton
from matmul_kernel import matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()

name_to_torch_type = {"fp16": torch.float16, "bf16": torch.bfloat16}


def get_x_vals(selected_k=None):
    # M=32, N chosen so that N/BLOCK_N gives 256 workgroups (1 per CU on GFX950).
    # BLOCK_N=256, so N=65536 for 256 WGs.
    sizes = [
        (32, 65536, 1024),
        (32, 65536, 2048),
        (32, 65536, 4096),
        (32, 65536, 8192),
        (32, 65536, 16384),
    ]

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
    default_dtypes = ["fp16"]

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
    parser = argparse.ArgumentParser(description="Memory-bound GEMM benchmark")
    parser.add_argument("--K", type=int, default=None, help="Select GEMM problem size with given K")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["fp16"],
        default=None,
        help="Data type(s) to benchmark (default: fp16)",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=2,
        help="Number of async copy pipeline stages (default: 2)",
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


def test_correctness(dtype, gemm_sizes, num_stages):
    torch_dtype = name_to_torch_type[dtype]

    for M, N, K in gemm_sizes:
        a = torch.rand((M, K), device=DEVICE, dtype=torch_dtype) - 0.5
        b = torch.rand((N, K), device=DEVICE, dtype=torch_dtype).T - 0.5
        triton_output = matmul(a, b, num_stages=num_stages)
        torch_output = torch.matmul(a, b)
        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"{M=} {N=} {K=} {dtype=}: Triton and Torch match")
        else:
            print(f"{M=} {N=} {K=} {dtype=}: Triton and Torch differ")


def gen_rotating_tensors(M, N, K, torch_dtype, rotating_buffer_size_mb=512):
    """Allocate multiple copies of (a, b, c) tensors to exceed GPU cache size."""
    elem_bytes = torch.tensor([], dtype=torch_dtype).element_size()
    a_size = M * K * elem_bytes
    b_size = K * N * elem_bytes
    c_size = M * N * elem_bytes
    total_size = a_size + b_size + c_size

    block_count = max(1, rotating_buffer_size_mb * 1024 * 1024 // total_size)

    a_list, b_list, c_list = [], [], []
    for _ in range(block_count):
        a_list.append(torch.randn((M, K), device=DEVICE, dtype=torch_dtype))
        b_list.append(torch.randn((N, K), device=DEVICE, dtype=torch_dtype).T)
        c_list.append(torch.empty((M, N), device=DEVICE, dtype=torch_dtype))

    return a_list, b_list, c_list, block_count


def run_rocprof_iterations(
    dtypes, gemm_sizes, num_stages, n_iters=1000, rotating_buffer_size_mb=512
):
    """Run the kernel n_iters times for each dtype/size combo using rotating tensors."""
    for dtype in dtypes:
        torch_dtype = name_to_torch_type[dtype]
        for M, N, K in gemm_sizes:
            a_list, b_list, c_list, block_count = gen_rotating_tensors(
                M, N, K, torch_dtype, rotating_buffer_size_mb
            )
            print(
                f"{M=} {N=} {K=} {dtype=}: "
                f"rotating tensors: {block_count} copies, "
                f"{block_count * (M*K + K*N + M*N) * a_list[0].element_size() / 1024**2:.0f} MB"
            )
            # Warmup
            matmul(a_list[0], b_list[0], c_list[0], num_stages=num_stages)
            torch.cuda.synchronize()
            for i in range(n_iters):
                idx = i % block_count
                matmul(a_list[idx], b_list[idx], c_list[idx], num_stages=num_stages)
            torch.cuda.synchronize()
            print(f"{M=} {N=} {K=} {dtype=}: {n_iters} iterations done")


def main():
    args = parse_args()
    num_stages = args.num_stages

    gemm_sizes = get_x_vals(args.K)
    dtypes = get_dtypes(args.dtype)

    for dtype in dtypes:
        test_correctness(dtype, gemm_sizes, num_stages)

    if args.rocprof:
        run_rocprof_iterations(
            dtypes,
            gemm_sizes,
            num_stages,
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
            styles=[("green", "-")],
            ylabel="TFLOPS",
            plot_name=f"memory-bound-gemm-stages{num_stages}",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, dtype):
        torch_dtype = name_to_torch_type[dtype]
        a = torch.randn((M, K), device=DEVICE, dtype=torch_dtype)
        b = torch.randn((N, K), device=DEVICE, dtype=torch_dtype).T
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b, num_stages=num_stages), quantiles=quantiles
        )

        elem_bytes = a.element_size()
        total_bytes = (M * K + K * N + M * N) * elem_bytes
        bw_gbs = total_bytes * 1e-9 / (ms * 1e-3)

        def perf(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        print(f"  {M=:5d} {N=:5d} {K=:5d} {dtype}: " f"{perf(ms):7.1f} TFLOPS, {bw_gbs:7.1f} GB/s")

        return perf(ms), perf(max_ms), perf(min_ms)

    print(f"\nMemory-bound GEMM (num_stages={num_stages}):")
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
