import argparse

import torch
import triton

# from v0_naive.matmul_kernel import matmul
# from v1_buffer_load.matmul import mamtul
# from v2_async_copy.matmul import mamtul
# from v3_lds.matmul_kernel import matmul
from v4_global_prefetch.matmul_kernel import matmul

# from v7_beyond_hotloop.matmul import mamtul

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


def get_gemm_sizes():

    args = parse_args()
    selected_k = args.K
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


def get_dtypes():
    default_dtypes = ["fp16", "bf16"]

    args = parse_args()
    selected_dtype = args.dtype

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

    parser = argparse.ArgumentParser(description="GEMM benchmark")
    parser.add_argument("--K", type=int, default=None, help="Select GEMM problem size with given K")
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=["fp16", "bf16"],
        default=None,
        help="Data type(s) to benchmark (default: fp16 bf16)",
    )
    return parser.parse_args()


def test_correctness(dtype):
    if dtype == "f8":
        torch_dtype = torch.float16
    else:
        torch_dtype = name_to_torch_type[dtype]

    for M, N, K in get_gemm_sizes():
        a = torch.rand((M, K), device=DEVICE, dtype=torch_dtype) - 0.5
        b = torch.rand((N, K), device=DEVICE, dtype=torch_dtype).T - 0.5
        if dtype == "f8":
            a = a.to(torch.float8_e5m2)
            b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)
        if dtype == "f8":
            torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        else:
            torch_output = torch.matmul(a, b)
        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"{M=} {N=} {K=} {dtype=}: ✅ Triton and Torch match")
        else:
            print(f"{M=} {N=} {K=} {dtype=}: ❌ Triton and Torch differ")


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=get_gemm_sizes(),
        line_arg="dtype",
        line_vals=get_dtypes(),
        line_names=get_dtypes(),
        styles=[("green", "-"), ("yellow", "--"), ("red", "--")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)


@triton.testing.perf_report(configs)
def benchmark(M, N, K, dtype):
    if dtype == "f8":
        torch_dtype = torch.float16
    else:
        torch_dtype = name_to_torch_type[dtype]
    a = torch.randn((M, K), device=DEVICE, dtype=torch_dtype)
    b = torch.randn((N, K), device=DEVICE, dtype=torch_dtype).T
    if dtype == "f8":
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


for dtype in get_dtypes():
    test_correctness(dtype)

benchmark.run(show_plots=False, print_data=True)
