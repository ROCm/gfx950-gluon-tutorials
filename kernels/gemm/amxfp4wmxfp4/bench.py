import argparse

import torch
import triton
from matmul_kernel import matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def mxfp4_to_f32(x):
    """Unpack MXFP4 (e2m1) values from packed uint8 to float32."""
    # Each uint8 holds 2 FP4 values: low nibble and high nibble
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    """Convert e8m0 scale values to float32."""
    return 2 ** ((x - 127).to(torch.float32))


def generate_mxfp4_inputs(M, N, K):
    """Generate random MXFP4 packed tensors and e8m0 scales.

    Returns:
        a_fp4: (M, K//2) uint8 — packed MXFP4 activations
        b_fp4: (N, K//2) uint8 — packed MXFP4 weights (N,K//2 layout)
        a_scales: (M, K//32) uint8 — e8m0 scales for A
        b_scales: (N, K//32) uint8 — e8m0 scales for B
    """
    torch.manual_seed(42)

    # Generate random FP4 values packed in uint8
    # Each uint8 = (high_nibble << 4) | low_nibble, nibble values 0-15
    a_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    a_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    a_fp4 = (a_high << 4 | a_low).to(device=DEVICE)

    b_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=DEVICE)
    b_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=DEVICE)
    b_fp4 = b_low | b_high << 4

    # Generate e8m0 scales: values near 127 (=1.0 with bias 127)
    # Range [124, 128) gives scales 2^(-3) to 2^(0)
    M_pad = (M + 255) // 256 * 256
    a_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device=DEVICE
    ).T[:M]
    b_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=DEVICE
    ).T

    return a_fp4, b_fp4, a_scales, b_scales


def torch_reference(a_fp4, b_fp4, a_scales, b_scales, dtype=torch.bfloat16):
    """Compute reference GEMM by dequantizing MXFP4 to float32."""
    # Dequantize A and B
    a_f32 = mxfp4_to_f32(a_fp4)
    b_f32 = mxfp4_to_f32(b_fp4)

    # Expand scales: (M, K//32) -> (M, K) via repeat_interleave
    a_scales_f32 = e8m0_to_f32(
        a_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )
    b_scales_f32 = e8m0_to_f32(
        b_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )

    # Scale the dequantized values
    a_f32 = a_f32 * a_scales_f32
    b_f32 = b_f32 * b_scales_f32

    # Compute GEMM: A(M,K) @ B(N,K).T = C(M,N)
    return torch.mm(a_f32, b_f32.T).to(dtype)


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
    parser = argparse.ArgumentParser(description="MXFP4 GEMM benchmark")
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
        help="Total size (MB) of rotating tensors for rocprof mode. "
        "Should exceed GPU cache (L2+MALL) size. (default: 512)",
    )
    return parser.parse_args()


def test_correctness(gemm_sizes):
    for M, N, K in gemm_sizes:
        a_fp4, b_fp4, a_scales, b_scales = generate_mxfp4_inputs(M, N, K)

        # Kernel expects B as (N, K//2) row-major (K-contiguous)
        triton_output = matmul(a_fp4, b_fp4, a_scales, b_scales)

        # Reference: dequantize and compute in float32
        torch_output = torch_reference(a_fp4, b_fp4, a_scales, b_scales, dtype=torch.bfloat16)

        if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
            print(f"[amxfp4wmxfp4] {M=} {N=} {K=}: Triton and Torch match")
        else:
            max_diff = (triton_output - torch_output).abs().max().item()
            print(f"[amxfp4wmxfp4] {M=} {N=} {K=}: Triton and Torch differ (max_diff={max_diff:.4f})")


def gen_rotating_tensors(M, N, K, rotating_buffer_size_mb=512):
    """Allocate multiple copies of tensors to exceed GPU cache size."""
    elem_bytes = 1  # uint8
    # a: M*K//2, b: K//2*N, a_scales: M*K//32, b_scales: N*K//32
    a_size = M * (K // 2) * elem_bytes
    b_size = (K // 2) * N * elem_bytes
    as_size = M * (K // 32) * elem_bytes
    bs_size = N * (K // 32) * elem_bytes
    total_size = a_size + b_size + as_size + bs_size

    block_count = max(1, rotating_buffer_size_mb * 1024 * 1024 // total_size)

    a_list, b_list, as_list, bs_list = [], [], [], []
    for _ in range(block_count):
        a_fp4, b_fp4, a_scales, b_scales = generate_mxfp4_inputs(M, N, K)
        a_list.append(a_fp4)
        b_list.append(b_fp4)  # (N, K//2) K-contiguous layout for kernel
        as_list.append(a_scales)
        bs_list.append(b_scales)

    return a_list, b_list, as_list, bs_list, block_count


def run_rocprof_iterations(gemm_sizes, n_iters=1000, rotating_buffer_size_mb=512):
    """Run kernel n_iters times with rotating tensors for cache-cold profiling."""
    for M, N, K in gemm_sizes:
        a_list, b_list, as_list, bs_list, block_count = gen_rotating_tensors(
            M, N, K, rotating_buffer_size_mb
        )
        total_bytes = block_count * (
            M * (K // 2) + (K // 2) * N + M * (K // 32) + N * (K // 32)
        )
        print(
            f"[amxfp4wmxfp4] {M=} {N=} {K=}: "
            f"rotating tensors: {block_count} copies, "
            f"{total_bytes / 1024**2:.0f} MB"
        )
        # Warmup
        matmul(a_list[0], b_list[0], as_list[0], bs_list[0])
        torch.cuda.synchronize()
        for i in range(n_iters):
            idx = i % block_count
            matmul(a_list[idx], b_list[idx], as_list[idx], bs_list[idx])
        torch.cuda.synchronize()
        print(f"[amxfp4wmxfp4] {M=} {N=} {K=}: {n_iters} iterations done")


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
            line_vals=["mxfp4"],
            line_names=["mxfp4"],
            styles=[("green", "-")],
            ylabel="TFLOPS",
            plot_name="matmul-performance",
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, dtype):
        a_fp4, b_fp4, a_scales, b_scales = generate_mxfp4_inputs(M, N, K)
        quantiles = [0.5, 0.2, 0.8]
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a_fp4, b_fp4, a_scales, b_scales),
            quantiles=quantiles,
        )

        def perf(ms):
            return 2 * M * N * K * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    print("\namxfp4wmxfp4_kernel:")
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
