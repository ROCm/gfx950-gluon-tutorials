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
"""Baseline benchmark for the FAV3 (rotated 4-cluster) Gluon flash-attention
kernel, ported from AMD-Triton/gluon-kernels (kernels/cdna4/fa).

Mirrors the GEMM tutorial bench.py: correctness vs a torch reference, do_bench
TFLOPS, a --rocprof iteration mode, and the out-of-tree plugin hooks so the same
harness can later measure base / llir / llir+plugins configs.
"""

import argparse
import os
import sys

import torch

# The out-of-tree LLIR scheduler ships as an LLVM pass plugin. Loaded via
# LLVM_PASS_PLUGIN_PATH, it resolves LLVM symbols from libtriton at dlopen time,
# which requires libtriton in the *global* symbol scope. CPython loads
# C-extensions RTLD_LOCAL by default, so opt into RTLD_GLOBAL before the first
# `import triton`. Only takes effect when the plugin is in use.
if os.environ.get("LLVM_PASS_PLUGIN_PATH"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)

import triton  # noqa: E402

# Out-of-tree amdgcnas peephole (post-assembly): install the amdgcn-stage hook
# when TRITON_AMDGCNAS_PLUGIN is set. Pure-Python text transform, no rebuild.
if os.environ.get("TRITON_AMDGCNAS_PLUGIN"):
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugins", "amdgcnas"),
    )
    import amdgcnas_plugin  # noqa: E402
    from triton import knobs  # noqa: E402

    knobs.runtime.add_stages_inspection_hook = amdgcnas_plugin.inspect_stages_hook

# Ported FAV3 kernel + shared helpers live alongside this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f16_fa_gfx950_common import (
    _check_output,
    compute_flops,
    input_helper,
    sdpa_reference,
)  # noqa: E402
from f16_fa_gfx950_rotated_4cluster import run_gluon_attention  # noqa: E402

DEVICE = triton.runtime.driver.active.get_active_torch_device()

name_to_torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}

# Sequence-length sweep (Q == K), the classic FA scaling axis.
SEQLENS = [1024, 2048, 4096, 8192, 16384]
# The causal reference materializes the full B*H*S*S fp32 scores matrix, so only
# validate at the smaller shapes; the perf sweep runs the full range.
CORRECTNESS_MAX_SEQLEN = 4096

# The perf sweep runs one line per causal setting; --causal-mode picks which.
CAUSAL_MODES = {"both": [False, True], "causal": [True], "noncausal": [False]}
CAUSAL_NAME = {False: "non-causal", True: "causal"}


def parse_args():
    p = argparse.ArgumentParser(description="FAV3 rotated-4cluster attention benchmark (gfx950)")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--layout", choices=["bhsd", "bshd"], default="bhsd")
    p.add_argument(
        "--causal-mode",
        choices=["both", "causal", "noncausal"],
        default="noncausal",
        help="which causal setting(s) to run (default: noncausal)",
    )
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hq", type=int, default=64, help="number of query heads")
    p.add_argument(
        "--hk", type=int, default=64, help="number of kv heads (hq%%hk==0; hq==hk is MHA)"
    )
    p.add_argument("--d", type=int, default=128, help="head dimension")
    p.add_argument("--seqlen", type=int, default=8192, help="sequence length (default: 8192)")
    p.add_argument("--sweep", action="store_true", help="sweep all seqlens instead of a single one")
    p.add_argument(
        "--rocprof",
        action="store_true",
        help="run the kernel n_iters times (no do_bench); wrap with rocprofv3 --kernel-trace",
    )
    p.add_argument("--n-iters", type=int, default=1000)
    return p.parse_args()


def make_inputs(B, HQ, HK, N_CTX, D, dtype, layout, causal):
    q, k, v, md = input_helper(B, HQ, HK, N_CTX, N_CTX, D, dtype, layout)
    if causal:
        md.need_causal()
    o = torch.empty_like(q)
    return q, k, v, o, md


def test_correctness(B, HQ, HK, N_CTX, D, dtype, layout, causal):
    q, k, v, o, md = make_inputs(B, HQ, HK, N_CTX, D, dtype, layout, causal)
    run_gluon_attention(q, k, v, o, md)
    o_ref = sdpa_reference(q, k, v, causal=causal, sm_scale=md.sm_scale)
    ok, max_diff, mean_diff = _check_output(o, o_ref)
    tag = "✅ match" if ok else "❌ MISMATCH"
    print(
        f"[FAV3] B={B} HQ={HQ} HK={HK} N={N_CTX} D={D} causal={causal} {layout} "
        f"{q.dtype}: {tag}  (max={max_diff:.2e} mean={mean_diff:.2e})"
    )
    return ok


def run_rocprof_iterations(args, torch_dtype, seqlens, causal_vals):
    for causal in causal_vals:
        for N_CTX in seqlens:
            q, k, v, o, md = make_inputs(
                args.batch, args.hq, args.hk, N_CTX, args.d, torch_dtype, args.layout, causal
            )
            run_gluon_attention(q, k, v, o, md)  # warmup + autotune
            torch.cuda.synchronize()
            for _ in range(args.n_iters):
                run_gluon_attention(q, k, v, o, md)
            torch.cuda.synchronize()
            print(f"[FAV3] N={N_CTX} causal={causal}: {args.n_iters} iterations done")


def main():
    args = parse_args()
    torch_dtype = name_to_torch_dtype[args.dtype]
    seqlens = SEQLENS if args.sweep else [args.seqlen]
    causal_vals = CAUSAL_MODES[args.causal_mode]

    # 1) Correctness. The causal reference materializes the full B*H*S*S fp32 scores
    #    matrix, so skip it past CORRECTNESS_MAX_SEQLEN; the non-causal reference uses
    #    the memory-efficient F.sdpa and checks at every seqlen.
    for causal in causal_vals:
        for N_CTX in seqlens:
            if causal and N_CTX > CORRECTNESS_MAX_SEQLEN:
                continue
            test_correctness(
                args.batch, args.hq, args.hk, N_CTX, args.d, torch_dtype, args.layout, causal
            )

    # 2) Rocprof mode: cold, external timing.
    if args.rocprof:
        run_rocprof_iterations(args, torch_dtype, seqlens, causal_vals)
        return

    # 3) do_bench TFLOPS sweep — one line per causal setting.
    configs = [
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=seqlens,
            line_arg="causal",
            line_vals=causal_vals,
            line_names=[CAUSAL_NAME[c] for c in causal_vals],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="TFLOPS",
            plot_name=(
                f"fav3-attn-B{args.batch}-HQ{args.hq}-HK{args.hk}"
                f"-D{args.d}-{args.layout}-{args.dtype}"
            ),
            args={},
        )
    ]

    @triton.testing.perf_report(configs)
    def benchmark(N_CTX, causal):
        q, k, v, o, md = make_inputs(
            args.batch, args.hq, args.hk, N_CTX, args.d, torch_dtype, args.layout, causal
        )
        fn = lambda: run_gluon_attention(q, k, v, o, md)  # noqa: E731
        fn()  # trigger autotune + warm compile before timing
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
        return compute_flops(args.batch, args.hq, N_CTX, N_CTX, args.d, causal) / ms * 1e-9

    print(
        f"\nFAV3 rotated-4cluster attention | B={args.batch} HQ={args.hq} HK={args.hk} "
        f"D={args.d} layout={args.layout} dtype={args.dtype} causal_mode={args.causal_mode}"
    )
    benchmark.run(show_plots=False, print_data=True)


if __name__ == "__main__":
    main()
