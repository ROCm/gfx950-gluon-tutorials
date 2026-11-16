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

"""Kernel-time TFLOPS for the flash-attention kernels, under the tutorial's timing protocol.

The GEMM and FA numbers are measured the same way: ``bench.py``'s timing protocol
(``scripts/graph_timer.py``, see ``docs/benchmarking.md``). A CUDA graph of back-to-back
dispatches over rotating (q, k, v, o) sets is replayed for a warm-up time and then for a
measurement window, and the result is the median replay time per dispatch. That is the
back-to-back state a kernel holds when it runs continuously, with no host launch, no host
synchronization and no event pair between dispatches. ``--kernel-trace`` adds a
``rocprofv3 --kernel-trace`` cross-check over the same window (no serialization).

This script runs ``kernels/attention/bench.py`` with the FA shape and prints the result with
the FA FLOP count (2 GEMMs of B*HQ*M*N*D). It used to time a ``rocprofv3`` trace of 1000
serialized dispatches (``AMD_SERIALIZE_KERNEL=3``) and average the last 100; serialization
puts a host round trip before every dispatch, which changes the power state the kernel runs
in, and the last 100 of 1000 are not yet the sustained state.

Two error sources dominate any of this, so quote them with a number: the 8 MI355X
dies on one board span 6.5% (they track the *slowest* XCD clock, r=0.89, not the
average), and a thermally cool part over-reads by 4-7% as DPM ramps into its
power-limited plateau -- use ``--protocol sustained`` for numbers that are published.

Example (fmha_v4 at the tutorial shape, plugin stack enabled):

    FA_MODULE=fmha_v4 \\
    DISABLE_LLVM_OPT=disable-machine-sink \\
    LLVM_PASS_PLUGIN_PATH=$PWD/plugins/llir_scheduler/libLlirSched.so \\
    python scripts/fa_kernel_time.py --seqlen 16320 --protocol sustained
"""

import argparse
import os
import sys

KERNEL_NAME = "gluon_attn_fwd"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTENTION_DIR = os.path.join(REPO_ROOT, "kernels", "attention")

# Share the GEMM path's timing driver, so FA and GEMM numbers mean the same thing. Neither
# module imports torch/triton at import time, so this stays a light driver.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_timer  # noqa: E402
from run_perf_table import run_timing  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Kernel-time TFLOPS for FA (gfx950)")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--layout", choices=["bhsd", "bshd"], default="bhsd")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hq", type=int, default=64)
    p.add_argument("--hk", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=16320)
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
        "--scale-on-q",
        type=int,
        default=1,
        choices=[0, 1],
        help="0 applies qk_scale per element inside VEC1 instead of pre-scaling Q",
    )
    p.add_argument(
        "--protocol",
        choices=sorted(graph_timer.PROTOCOLS),
        default=graph_timer.DEFAULT_PROTOCOL,
        help="quick (1 s warm-up + 4 s window) or sustained (60 s + 30 s, for published numbers)",
    )
    p.add_argument("--warmup-s", type=float, default=None, help="override the warm-up time")
    p.add_argument("--window-s", type=float, default=None, help="override the window length")
    p.add_argument(
        "--kernel-trace",
        action="store_true",
        help="also report the rocprofv3 --kernel-trace median over the window",
    )
    p.add_argument("--trace-dir", default=None, help="where to put the rocprofv3 trace")
    p.add_argument("--keep-trace", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.rotating_sets < 0 or args.rotating_buffer_size <= 0:
        print("Error: --rotating-sets must be >= 0 and --rotating-buffer-size positive")
        sys.exit(2)
    bench_args = [
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
        "--rotating-sets",
        str(args.rotating_sets),
        "--rotating-buffer-size",
        str(args.rotating_buffer_size),
        "--scale-on-q",
        str(args.scale_on_q),
    ]
    protocol_cli = ["--protocol", args.protocol]
    if args.warmup_s is not None:
        protocol_cli += ["--warmup-s", str(args.warmup_s)]
    if args.window_s is not None:
        protocol_cli += ["--window-s", str(args.window_s)]
    trace_dir = args.trace_dir or os.path.join(ATTENTION_DIR, "fa_kernel_time_trace")

    print(
        f"FA kernel-time TFLOPS | module={os.environ.get('FA_MODULE', 'fmha_v3')} "
        f"B={args.batch} HQ={args.hq} HK={args.hk} N={args.seqlen} D={args.d} "
        f"{args.dtype} {args.layout} non-causal"
    )
    result, trace_us = run_timing(
        ATTENTION_DIR,
        bench_args,
        os.environ.copy(),
        protocol_cli,
        trace_kernel=KERNEL_NAME if args.kernel_trace else None,
        trace_dir=trace_dir,
        keep_trace=args.keep_trace,
    )
    if result is None:
        sys.exit(1)

    print(f"\n  {'source':<14} {'us/dispatch':>12} {'TFLOPS':>9}")
    print(f"  {'graph replay':<14} {result['median_us']:>12.2f} {result['tflops']:>9.1f}")
    if trace_us:
        tf = result["flops"] / trace_us * 1e-6
        print(f"  {'kernel trace':<14} {trace_us:>12.2f} {tf:>9.1f}")
    gpu = result["gpu"]
    cond = f"protocol {result['protocol']}, GPU {gpu['bdf']}"
    if result["power_w"] is not None:
        cond += f", {result['power_w']:.0f} W, {result['clock_mhz']:.0f} MHz in the window"
    print(f"  measured under: {cond}")
    if args.kernel_trace and args.keep_trace:
        print(f"  kernel trace kept in {trace_dir}")


if __name__ == "__main__":
    main()
