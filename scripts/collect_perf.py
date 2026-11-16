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
Collect TFLOPS + MFMA efficiency for the 8-wave warp-pipeline (inter_wave) GEMM
kernels using rocprof. One tool for all three inter_wave kernels; the 4-wave
(intra_wave) kernels use scripts/run_perf_table.py instead.

  * TFLOPS  — bench.py's timing protocol (scripts/graph_timer.py: graph replay of
              back-to-back dispatches over rotating inputs, warm-up, measurement
              window; see docs/benchmarking.md). --rocprof adds a kernel-trace
              cross-check over the same window.
  * MFMA eff — rocprofv3 --att via scripts/run_att.py on one dispatch taken after
              --att-warmup-s of back-to-back dispatches, decoded by
              scripts/process_json.py, then scaled x2 for 2 waves/SIMD.
  * VGPRs/spills — parsed from the kernel's .amdgcn in a private Triton cache.

The inter_wave kernels schedule themselves via warp_pipeline_stage and run
"base" (no schedule_hint, no amdgcnas), so no env vars.

Usage:
    python scripts/collect_perf.py --kernel a16w16 --K 8192 --dtype fp16
    python scripts/collect_perf.py --kernel a8w8 --K 16384
    python scripts/collect_perf.py --kernel a4w4 --version 1 --K 32768 \
        --rotating-buffer-size 2048
    python scripts/collect_perf.py --kernel a16w16 --K 8192 --skip-att  # TFLOPS only
    python scripts/collect_perf.py --kernel a16w16 --K 8192 --protocol sustained  # published
"""

import argparse
import os
import shutil
import subprocess
import sys

GIT_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
SCRIPTS_DIR = os.path.join(GIT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import graph_timer  # noqa: E402
from run_perf_table import (  # noqa: E402
    ATT_WARMUP_S,
    clean_caches,
    parse_amdgcn_metadata,
    private_triton_env,
    run_att_steady,
    run_timing,
)

# 8 waves across a CU's 4 SIMDs => 2 waves/SIMD. process_json.py measures a
# SINGLE wave's MFMA-cycle fraction; with 2 co-resident waves interleaving MFMA
# issue on the same SIMD, per-SIMD utilization is ~WAVES_PER_SIMD x per-wave.
WAVES_PER_SIMD = 2

# Per-kernel knobs. `dtypes` is None for single-precision kernels (no --dtype);
# `versions` maps --version -> subdir for multi-version kernels (else None, and
# the jit kernel is named `kernel_name`). `att_type` selects the ATT buffer-size
# override in run_perf_table.write_att_config. `label` is the dtype shown in the
# results table (for a16w16 it is the chosen --dtype).
KERNELS = {
    "a16w16": dict(
        kernel_name="a16w16_kernel",
        dtypes=("fp16", "bf16"),
        versions=None,
        att_type="a16w16",
        label=None,
    ),
    "a8w8": dict(
        kernel_name="a8w8_kernel", dtypes=None, versions=None, att_type="a8w8", label="f8"
    ),
    "a4w4": dict(
        kernel_name=None,
        dtypes=None,
        versions={0: "v0_sliceMN", 1: "v1_combineBsc", 2: "v2_mfma32x32x64"},
        att_type="a4w4",
        label="mxfp4",
    ),
}


def scale_eff(raw_pct, factor):
    """Scale a 'NN.NN%' string by factor; return (raw_str, scaled_str)."""
    if raw_pct is None:
        return None, None
    val = float(raw_pct.rstrip("%")) * factor
    return raw_pct, f"{val:.2f}%"


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect TFLOPS + MFMA eff for an inter_wave GEMM kernel"
    )
    p.add_argument("--kernel", required=True, choices=sorted(KERNELS), help="inter_wave kernel")
    p.add_argument("--K", type=int, default=8192, help="K dimension (default: 8192)")
    p.add_argument("--M", type=int, default=4096, help="M dimension (default: 4096)")
    p.add_argument("--N", type=int, default=4096, help="N dimension (default: 4096)")
    p.add_argument(
        "--dtype", default="fp16", choices=["fp16", "bf16"], help="a16w16 only (default: fp16)"
    )
    p.add_argument("--version", type=int, default=0, help="version (a4w4 only): 0/1/2")
    p.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=graph_timer.COLD_INPUT_MIB,
        help="MiB of inputs to rotate through (default: %(default)s)",
    )
    p.add_argument("--skip-timing", action="store_true", help="Skip the timing run (TFLOPS)")
    p.add_argument("--skip-att", action="store_true", help="Skip ATT (MFMA efficiency)")
    p.add_argument(
        "--rocprof",
        action="store_true",
        help="also time the window with rocprofv3 --kernel-trace (no serialization)",
    )
    p.add_argument(
        "--protocol",
        choices=sorted(graph_timer.PROTOCOLS),
        default=graph_timer.DEFAULT_PROTOCOL,
        help="timing protocol: quick (1 s + 4 s) or sustained (60 s + 30 s). Default: %(default)s",
    )
    p.add_argument("--warmup-s", type=float, default=None, help="override the warm-up time")
    p.add_argument("--window-s", type=float, default=None, help="override the window length")
    p.add_argument(
        "--att-warmup-s",
        type=float,
        default=ATT_WARMUP_S,
        help="seconds of back-to-back dispatches before the ATT-traced one (default: %(default)s)",
    )
    args = p.parse_args()
    cfg = KERNELS[args.kernel]
    if cfg["versions"] is not None and args.version not in cfg["versions"]:
        p.error(
            f"--version {args.version} invalid for {args.kernel}; choose {sorted(cfg['versions'])}"
        )
    return args


def resolve(args):
    """Return (work_dir, kernel_name, dtype_label, bench_tail) for this run."""
    cfg = KERNELS[args.kernel]
    work_dir = os.path.join(GIT_ROOT, "kernels", "gemm", "inter_wave", args.kernel)
    kernel_name = cfg["versions"][args.version] if cfg["versions"] else cfg["kernel_name"]
    dtype_label = args.dtype if cfg["dtypes"] else cfg["label"]
    tail = []  # kernel-selecting bench.py args common to trace + att
    if cfg["dtypes"]:
        tail += ["--dtype", args.dtype]
    if cfg["versions"] is not None:
        tail += ["--version", str(args.version)]
    return work_dir, kernel_name, dtype_label, tail


def timing_cli(args):
    cli = ["--protocol", args.protocol, "--rotating-buffer-size", str(args.rotating_buffer_size)]
    if args.warmup_s is not None:
        cli += ["--warmup-s", str(args.warmup_s)]
    if args.window_s is not None:
        cli += ["--window-s", str(args.window_s)]
    return cli


def main():
    args = parse_args()
    work_dir, kernel_name, dtype_label, tail = resolve(args)
    cfg = KERNELS[args.kernel]
    ver = f", version={args.version}" if cfg["versions"] is not None else ""
    print("=" * 64)
    print(
        f"inter_wave/{args.kernel} 8-wave perf — {args.M}x{args.N}x{args.K} "
        f"{dtype_label}  (kernel={kernel_name}{ver})"
    )
    print("=" * 64)

    clean_caches(work_dir)
    env, cache_dir = private_triton_env(os.environ)
    bench_args = ["--K", str(args.K), *tail]
    timing, trace_us, mfma_eff_raw = None, None, None
    try:
        if not args.skip_timing:
            timing, trace_us = run_timing(
                work_dir,
                bench_args,
                env,
                timing_cli(args),
                trace_kernel=kernel_name if args.rocprof else None,
                trace_dir=os.path.join(work_dir, "rocprof_trace"),
            )
        if not args.skip_att:
            if timing is None:
                print("  ATT skipped: it needs the timing run's dispatch time for its warm-up")
            else:
                mfma_eff_raw = run_att_steady(
                    work_dir,
                    kernel_name,
                    bench_args,
                    env,
                    timing["median_us"],
                    cfg["att_type"],
                    args.att_warmup_s,
                )
        vgprs, spills = parse_amdgcn_metadata(kernel_name, cache_dir)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    tflops = timing["tflops"] if timing else None
    mfma_eff_raw, mfma_eff = scale_eff(mfma_eff_raw, WAVES_PER_SIMD)

    def fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "N/A"

    print()
    print("=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(
        f"| {'M':>5} | {'N':>5} | {'K':>6} | {'dtype':>5} | {'TFLOPS':>8} | "
        f"{'MFMA Eff':>9} | {'VGPRs':>5} | {'Spills':>6} |"
    )
    print(f"| {'-'*5} | {'-'*5} | {'-'*6} | {'-'*5} | {'-'*8} | {'-'*9} | {'-'*5} | {'-'*6} |")
    print(
        f"| {args.M:>5} | {args.N:>5} | {args.K:>6} | {dtype_label:>5} | "
        f"{fmt(round(tflops, 1) if tflops else None):>8} | {fmt(mfma_eff):>9} | "
        f"{fmt(vgprs):>5} | {fmt(spills):>6} |"
    )
    if mfma_eff is not None:
        print(f"\nMFMA Eff = {mfma_eff} (per-wave {mfma_eff_raw} x {WAVES_PER_SIMD} waves/SIMD)")
    if timing is not None:
        gpu = timing["gpu"]
        cond = (
            f"{timing['median_us']:.2f} us/dispatch, protocol {timing['protocol']}, "
            f"GPU {gpu['bdf']}"
        )
        if timing["power_w"] is not None:
            cond += f", {timing['power_w']:.0f} W, {timing['clock_mhz']:.0f} MHz in the window"
        print(f"TFLOPS measured under: {cond}")
        if trace_us:
            print(f"Kernel trace over the window: {timing['flops'] / trace_us * 1e-6:.1f} TFLOPS")


if __name__ == "__main__":
    main()
