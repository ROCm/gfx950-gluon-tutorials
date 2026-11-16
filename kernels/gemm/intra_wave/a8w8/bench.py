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
import functools
import os
import sys

import torch
import triton

# Out-of-tree amdgcnas peephole (post-assembly): install the amdgcn-stage hook
# when TRITON_AMDGCNAS_PLUGIN is set. Pure-Python text transform, no rebuild.
if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "plugins",
            "amdgcnas",
        ),
    )
    import amdgcnas_plugin
    from triton import knobs

    knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook

# Put the shared kernels/gemm/utils/ on the path so the kernel's
# `from common import get_pids` resolves to the shared helper.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
# scripts/graph_timer.py: the tutorial's timing protocol (graph replay, warm-up, rotating sets).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "scripts")
)
import graph_timer  # noqa: E402
from matmul_kernel import matmul as _matmul  # noqa: E402

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
    parser.add_argument(
        "--schedule-hint",
        default=None,
        help='Triton schedule_hint compile option for the kernel ("" turns the MFMA scheduler '
        "off; default: the kernel's own choice, see matmul() in matmul_kernel.py)",
    )
    graph_timer.add_timing_args(parser)
    return parser.parse_args()


def test_correctness(matmul, gemm_sizes):
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


def gen_rotating_tensors(M, N, K, rotating_buffer_size_mb=graph_timer.COLD_INPUT_MIB):
    """Rotating (a, b) sets: enough that the inputs in rotation exceed rotating_buffer_size_mb.

    Dispatch i uses set i % count, so the data the previous dispatches left in L2 and the MALL is
    useless and every dispatch reads its inputs from HBM, as a kernel in a real model would.
    """
    count = graph_timer.rotating_sets(M * K + K * N, rotating_buffer_size_mb)  # 1 byte per fp8
    a_list, b_list = [], []
    for _ in range(count):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
        b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).T.to(torch.float8_e5m2)
        a_list.append(a)
        b_list.append(b)
    return a_list, b_list, count


def main():
    args = parse_args()
    matmul = _matmul
    if args.schedule_hint is not None:
        matmul = functools.partial(matmul, schedule_hint=args.schedule_hint)

    gemm_sizes = get_gemm_sizes(args.K)

    test_correctness(matmul, gemm_sizes)

    protocol = graph_timer.protocol_from_args(args)
    for M, N, K in gemm_sizes:
        a_l, b_l, n_sets = gen_rotating_tensors(M, N, K, args.rotating_buffer_size)

        def launch(slot):
            matmul(a_l[slot], b_l[slot])

        label = f"a8w8_kernel {M}x{N}x{K} f8"
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
        result.update(kernel="a8w8_kernel", M=M, N=N, K=K, dtype="f8")
        graph_timer.print_result(result, args.result_json)
        a_l = b_l = None  # free the sets
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
