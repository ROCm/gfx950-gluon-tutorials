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

  * TFLOPS  — rocprofv3 --kernel-trace around `bench.py --rocprof` (rotating
              tensors / cold cache), averaging the last 100 kernel dispatches.
  * MFMA eff — rocprofv3 --att via scripts/run_att.py, decoded by
              scripts/process_json.py, then scaled x2 for 2 waves/SIMD.
  * VGPRs/spills — parsed from the kernel's .amdgcn in the triton cache.

The inter_wave kernels schedule themselves via warp_pipeline_stage and run
"base" (no TRITON_ENABLE_LLIR_SCHED / TRITON_ENABLE_AMDGCN_AS), so no env vars.

Usage:
    python scripts/collect_perf.py --kernel a16w16 --K 8192 --dtype fp16
    python scripts/collect_perf.py --kernel a8w8 --K 16384
    python scripts/collect_perf.py --kernel a4w4 --version 1 --K 32768 \
        --rotating-buffer-size 2048
    python scripts/collect_perf.py --kernel a16w16 --K 8192 --skip-att  # TFLOPS only
"""

import argparse
import os
import shutil
import subprocess
import sys

GIT_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
SCRIPTS_DIR = os.path.join(GIT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from run_perf_table import (  # noqa: E402
    avg_kernel_time_ns,
    clean_caches,
    find_kernel_trace_csv,
    parse_amdgcn_metadata,
    parse_mfma_efficiency,
    write_att_config,
)

RUN_ATT = os.path.join(SCRIPTS_DIR, "run_att.py")

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
    p.add_argument("--rotating-buffer-size", type=int, default=512, help="MB (default: 512)")
    p.add_argument("--skip-trace", action="store_true", help="Skip rocprof kernel-trace (TFLOPS)")
    p.add_argument("--skip-att", action="store_true", help="Skip ATT (MFMA efficiency)")
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


def run_kernel_trace(args, work_dir, kernel_name, tail):
    """rocprofv3 --kernel-trace -> average kernel time -> TFLOPS."""
    trace_dir = os.path.join(work_dir, "rocprof_trace")
    if os.path.isdir(trace_dir):
        shutil.rmtree(trace_dir)

    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "-f",
        "csv",
        "--kernel-include-regex",
        kernel_name,
        "-d",
        trace_dir,
        "--",
        "python",
        "bench.py",
        "--rocprof",
        "--K",
        str(args.K),
        "--rotating-buffer-size",
        str(args.rotating_buffer_size),
        *tail,
    ]
    print(f"  rocprofv3 --kernel-trace: {kernel_name} K={args.K} ...")
    proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        print("  kernel-trace FAILED:")
        for line in (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:]:
            print("    " + line)
        return None

    csv_path = find_kernel_trace_csv(trace_dir)
    if csv_path is None:
        print(f"  no kernel_trace.csv under {trace_dir}")
        return None

    avg_ns, count = avg_kernel_time_ns(csv_path, kernel_name)
    if avg_ns is None:
        print(f"  no rows matched kernel '{kernel_name}' in {csv_path}")
        return None

    tflops = 2 * args.M * args.N * args.K * 1e-12 / (avg_ns * 1e-9)
    print(f"  rocprofv3: {count} dispatches, avg={avg_ns/1e3:.2f} us, tflops={tflops:.1f}")
    return tflops


def run_att(args, work_dir, kernel_name, att_type, tail):
    """rocprofv3 --att (via run_att.py) -> process_json.py -> MFMA efficiency."""
    write_att_config(kernel_name, work_dir, kernel_type=att_type)

    cmd = [
        sys.executable,
        RUN_ATT,
        "--att-output",
        "tmp",
        "python",
        "bench.py",
        "--K",
        str(args.K),
        *tail,
    ]
    print(f"  rocprofv3 --att (MFMA eff): {kernel_name} K={args.K} ...")
    env = os.environ.copy()
    env.setdefault("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/")
    proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True, env=env)
    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print("  ATT FAILED:")
        for line in combined.strip().splitlines()[-8:]:
            print("    " + line)
        return None

    mfma_eff = parse_mfma_efficiency(combined)
    if mfma_eff is None:
        print("  could not parse MFMA efficiency from ATT output")
        for line in combined.strip().splitlines()[-8:]:
            print("    " + line)
    return mfma_eff


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

    tflops = None if args.skip_trace else run_kernel_trace(args, work_dir, kernel_name, tail)
    mfma_eff_raw = (
        None if args.skip_att else run_att(args, work_dir, kernel_name, cfg["att_type"], tail)
    )
    mfma_eff_raw, mfma_eff = scale_eff(mfma_eff_raw, WAVES_PER_SIMD)
    vgprs, spills = parse_amdgcn_metadata(kernel_name)

    def fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "N/A"

    print()
    print("=" * 64)
    print("RESULTS (rocprof)")
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


if __name__ == "__main__":
    main()
