#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit passthrough of the same, subject to the
# following conditions:
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

"""Kernel-time TFLOPS for the flash-attention kernels, via ``rocprofv3``.

The GEMM tutorial measures TFLOPS from kernel timestamps rather than from
``do_bench`` wall time (``scripts/run_perf_table.py:run_rocprof_trace``): it wraps
``bench.py --rocprof`` in ``rocprofv3 --kernel-trace``, averages the final N
dispatch durations, and divides the FLOP count by that. Wall-clock timing instead
charges the kernel for host-side launch gaps, and those gaps also idle the GPU,
which lets clocks drift between dispatches.

This script applies the same protocol to ``kernels/attention/bench.py``, with the
FA FLOP count (2 GEMMs of B*HQ*M*N*D). It imports ``avg_kernel_time_ns`` and
``find_kernel_trace_csv`` from ``run_perf_table`` instead of reimplementing them, so
the two paths cannot drift apart in how they read a trace (numeric ``Dispatch_Id``
ordering before the final-N tail is taken -- a lexicographic sort would place
dispatch 999 after 1000).

Dispatch paths:

  --launch prepared  pre-bound PreparedKernel (default; matches the GEMM --prepared
                     path, so FA and GEMM numbers are directly comparable)
  --launch jit       the ordinary JIT wrapper, binding arguments per dispatch
  --launch both      run each and report the delta

Prepared launch is used for protocol consistency, not for a speedup: at S=16320 the
kernel runs 7 ms, so the async queue hides the ~40 us of per-dispatch host binding
and the two paths measure the same (order-balanced A/B/A/B: -0.24%, inside the
+-1.3% session spread). What it does buy is repeatability -- two independent 8-GPU
passes agreed to 0.17%, against +-1.3% for do_bench.

Two error sources dominate any of this, so quote them with a number: the 8 MI355X
dies on one board span 6.5% (they track the *slowest* XCD clock, r=0.89, not the
average), and a thermally cool part over-reads by 4-7% as DPM ramps into its
power-limited plateau.

Example (fav4 at the tutorial shape, plugin stack enabled):

    HIP_VISIBLE_DEVICES=1 FA_MODULE=fav4 \\
    DISABLE_LLVM_OPT=disable-machine-sink \\
    LLVM_PASS_PLUGIN_PATH=$PWD/plugins/llir_scheduler/libLlirSched.so \\
    LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \\
    python scripts/fa_kernel_time.py --seqlen 16320 --launch both
"""

import argparse
import os
import shutil
import subprocess
import sys

KERNEL_NAME = "gluon_attn_fwd"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTENTION_DIR = os.path.join(REPO_ROOT, "kernels", "attention")

# Reuse the GEMM path's trace reader rather than reimplementing it: sharing
# avg_kernel_time_ns is what actually guarantees FA and GEMM numbers mean the same
# thing (same Dispatch_Id ordering, same final-N tail, same arithmetic). It imports
# no torch/triton, so this stays a light stdlib-only driver.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_perf_table import avg_kernel_time_ns, find_kernel_trace_csv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="rocprofv3 kernel-time TFLOPS for FA (gfx950)")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--layout", choices=["bhsd", "bshd"], default="bhsd")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hq", type=int, default=64)
    p.add_argument("--hk", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=16320)
    # Flag names follow run_perf_table.py, this script's peer on the GEMM side.
    p.add_argument("--iters", type=int, default=1000, help="measured dispatches (default: 1000)")
    p.add_argument("--warmup", type=int, default=10, help="prepared-mode warmup dispatches")
    p.add_argument(
        "--rotating-sets",
        type=int,
        default=0,
        help="0 (default) derives the count from --rotating-buffer-size",
    )
    p.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=512,
        help="total working set (MB) spread across rotating sets, so dispatches read cold data "
        "instead of MALL-resident data (default: 512, same as the GEMM bench)",
    )
    p.add_argument(
        "--last-n",
        type=int,
        default=100,
        help="average over the final N dispatches, matching the GEMM protocol (default: 100)",
    )
    p.add_argument(
        "--scale-on-q",
        type=int,
        default=1,
        choices=[0, 1],
        help="0 applies qk_scale per element inside VEC1 instead of pre-scaling Q",
    )
    p.add_argument(
        "--launch",
        choices=["jit", "prepared", "both"],
        default="prepared",
        help="dispatch path. 'prepared' (default) is the reported configuration: it matches how "
        "the GEMM kernels are measured, so FA and GEMM numbers are directly comparable. 'jit' "
        "and 'both' remain for attributing a delta to the launch path itself",
    )
    p.add_argument(
        "--no-serialize",
        action="store_true",
        help="do not set AMD_SERIALIZE_KERNEL=3. The GEMM protocol sets it so each dispatch is "
        "timed in isolation; clearing it lets dispatches queue back-to-back",
    )
    p.add_argument("--trace-dir", default=None, help="where to keep the rocprofv3 traces")
    p.add_argument("--keep-trace", action="store_true")
    return p.parse_args()


def fa_flops(args):
    """Total FLOPs per dispatch: two B*HQ*M*N*D GEMMs, non-causal.

    Same expression as ``kernels/attention/common.py``'s ``compute_flops(..., causal=False)``,
    duplicated here so the script does not need to import torch/triton.
    """
    return 2 * (2.0 * args.batch * args.hq * args.seqlen * args.seqlen * args.d)


def collect(args, launch_mode, trace_root):
    """Run one rocprofv3 session and return its stats dict, or None on failure."""
    trace_dir = os.path.join(trace_root, f"fa_{launch_mode}_trace")
    if os.path.isdir(trace_dir):
        shutil.rmtree(trace_dir)

    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "-f",
        "csv",
        "--kernel-include-regex",
        KERNEL_NAME,
        "-d",
        trace_dir,
        "--",
        sys.executable,
        "bench.py",
        "--rocprof",
        "--dtype",
        args.dtype,
        "--layout",
        args.layout,
        "--batch",
        str(args.batch),
        "--hq",
        str(args.hq),
        "--hk",
        str(args.hk),
        "--d",
        str(args.d),
        "--seqlen",
        str(args.seqlen),
        "--n-iters",
        str(args.iters),
        "--rotating-sets",
        str(args.rotating_sets),
        "--rotating-buffer-size",
        str(args.rotating_buffer_size),
        "--scale-on-q",
        str(args.scale_on_q),
    ]
    if launch_mode == "prepared":
        cmd += ["--prepared", "--n-warmup", str(args.warmup)]

    env = os.environ.copy()
    if not args.no_serialize:
        env["AMD_SERIALIZE_KERNEL"] = "3"

    print(f"  rocprofv3: collecting {launch_mode} kernel trace ({args.iters} dispatches) ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ATTENTION_DIR)
    if proc.returncode != 0:
        print(f"  rocprofv3 FAILED (exit {proc.returncode})")
        for line in (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:]:
            print(f"    {line}")
        return None
    for line in proc.stdout.splitlines():
        if "match" in line or "MISMATCH" in line or "dispatches done" in line:
            print(f"    {line.strip()}")

    csv_path = find_kernel_trace_csv(trace_dir)
    if csv_path is None:
        print(f"  rocprofv3: no kernel_trace.csv under {trace_dir}")
        return None
    avg_ns, count = avg_kernel_time_ns(csv_path, KERNEL_NAME, last_n=args.last_n)
    if avg_ns is None:
        print(f"  rocprofv3: no rows matched '{KERNEL_NAME}' in {csv_path}")
        return None

    return {
        "mode": launch_mode,
        "dispatches": count,
        "tail": min(args.last_n, count),
        "avg_us": avg_ns / 1e3,
        "tflops": fa_flops(args) / avg_ns * 1e-3,
    }


def main():
    args = parse_args()
    if min(args.warmup, args.iters, args.last_n, args.rotating_buffer_size) <= 0:
        print("Error: --warmup, --iters, --last-n and --rotating-buffer-size must be positive")
        sys.exit(2)
    if args.rotating_sets < 0:
        print("Error: --rotating-sets must be >= 0 (0 derives the count)")
        sys.exit(2)
    trace_root = args.trace_dir or os.path.join(ATTENTION_DIR, "fa_kernel_time_traces")
    os.makedirs(trace_root, exist_ok=True)

    modes = ["jit", "prepared"] if args.launch == "both" else [args.launch]
    print(
        f"FA kernel-time TFLOPS | module={os.environ.get('FA_MODULE', 'fav3')} "
        f"B={args.batch} HQ={args.hq} HK={args.hk} N={args.seqlen} D={args.d} "
        f"{args.dtype} {args.layout} non-causal | {fa_flops(args) * 1e-12:.2f} TFLOP/dispatch"
    )
    print(
        f"  averaging the last {args.last_n} of {args.iters} dispatches"
        f"{'' if args.no_serialize else ', AMD_SERIALIZE_KERNEL=3'}"
    )

    results = [r for r in (collect(args, m, trace_root) for m in modes) if r]
    if not results:
        sys.exit(1)

    print(f"\n  {'launch':<10} {'dispatches':>10} {'final-N':>8} {'avg us':>10} {'TFLOPS':>9}")
    for r in results:
        print(
            f"  {r['mode']:<10} {r['dispatches']:>10} {r['tail']:>8} {r['avg_us']:>10.2f} "
            f"{r['tflops']:>9.1f}"
        )
    if len(results) == 2:
        jit, prep = results[0], results[1]
        delta = (prep["tflops"] - jit["tflops"]) / jit["tflops"] * 100
        print(
            f"\n  prepared vs jit: {delta:+.2f}% kernel time "
            f"({jit['avg_us'] - prep['avg_us']:+.2f} us/dispatch)"
        )

    if not args.keep_trace:
        shutil.rmtree(trace_root, ignore_errors=True)


if __name__ == "__main__":
    main()
