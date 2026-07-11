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
Collect TFLOPS + MFMA efficiency for the 8-wave warp-pipeline a8w8 (BF8) GEMM
using rocprof. Same mechanism as inter_wave/a16w16/collect_perf.py:

  * TFLOPS  — rocprofv3 --kernel-trace around `bench.py --rocprof` (rotating
              tensors / cold cache), averaging the last 100 kernel dispatches.
  * MFMA eff — rocprofv3 --att via scripts/run_att.py, decoded by
              scripts/process_json.py, then scaled x2 for 2 waves/SIMD.
  * VGPRs/spills — parsed from the kernel's .amdgcn in the triton cache.

Like inter_wave/a16w16, this kernel schedules itself via warp_pipeline_stage and runs
"base" (no TRITON_ENABLE_LLIR_SCHED / TRITON_ENABLE_AMDGCN_AS).

Usage:
    python collect_perf.py --K 8192
    python collect_perf.py --K 8192 --skip-att   # TFLOPS only
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

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_MAP = {1: "v1_sliceMN_BK128_nS2"}
KERNEL_NAME = None  # set in main() from --version
RUN_ATT = os.path.join(SCRIPTS_DIR, "run_att.py")

# 8 waves across a CU's 4 SIMDs => 2 waves/SIMD. process_json.py measures a
# SINGLE wave's MFMA-cycle fraction; with 2 co-resident waves interleaving MFMA
# issue on the same SIMD, per-SIMD utilization is ~WAVES_PER_SIMD x per-wave.
WAVES_PER_SIMD = 2


def scale_eff(raw_pct, factor):
    """Scale a 'NN.NN%' string by factor; return (raw_str, scaled_str)."""
    if raw_pct is None:
        return None, None
    val = float(raw_pct.rstrip("%")) * factor
    return raw_pct, f"{val:.2f}%"


def parse_args():
    p = argparse.ArgumentParser(description="Collect TFLOPS + MFMA eff for the a8w8 8-wave kernel")
    p.add_argument("--K", type=int, default=8192, help="K dimension (default: 8192)")
    p.add_argument("--M", type=int, default=4096, help="M dimension (default: 4096)")
    p.add_argument("--N", type=int, default=4096, help="N dimension (default: 4096)")
    p.add_argument("--rotating-buffer-size", type=int, default=512, help="MB (default: 512)")
    p.add_argument(
        "--version",
        type=int,
        default=1,
        choices=sorted(VERSION_MAP),
        help="Kernel version 1=v1_sliceMN_BK128_nS2 (default: 1)",
    )
    p.add_argument("--skip-trace", action="store_true", help="Skip rocprof kernel-trace (TFLOPS)")
    p.add_argument("--skip-att", action="store_true", help="Skip ATT (MFMA efficiency)")
    return p.parse_args()


def run_kernel_trace(args):
    """rocprofv3 --kernel-trace -> average kernel time -> TFLOPS."""
    trace_dir = os.path.join(WORK_DIR, "rocprof_trace")
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
        "python",
        "bench.py",
        "--rocprof",
        "--K",
        str(args.K),
        "--rotating-buffer-size",
        str(args.rotating_buffer_size),
        "--version",
        str(args.version),
    ]
    print(f"  rocprofv3 --kernel-trace: {KERNEL_NAME} K={args.K} ...")
    proc = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        print("  kernel-trace FAILED:")
        for line in (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:]:
            print("    " + line)
        return None

    csv_path = find_kernel_trace_csv(trace_dir)
    if csv_path is None:
        print(f"  no kernel_trace.csv under {trace_dir}")
        return None

    avg_ns, count = avg_kernel_time_ns(csv_path, KERNEL_NAME)
    if avg_ns is None:
        print(f"  no rows matched kernel '{KERNEL_NAME}' in {csv_path}")
        return None

    tflops = 2 * args.M * args.N * args.K * 1e-12 / (avg_ns * 1e-9)
    print(f"  rocprofv3: {count} dispatches, avg={avg_ns/1e3:.2f} us, tflops={tflops:.1f}")
    return tflops


def run_att(args):
    """rocprofv3 --att (via run_att.py) -> process_json.py -> MFMA efficiency."""
    write_att_config(KERNEL_NAME, WORK_DIR, kernel_type="a8w8")

    cmd = [
        sys.executable,
        RUN_ATT,
        "--att-output",
        "tmp",
        "python",
        "bench.py",
        "--K",
        str(args.K),
        "--version",
        str(args.version),
    ]
    print(f"  rocprofv3 --att (MFMA eff): {KERNEL_NAME} K={args.K} ...")
    env = os.environ.copy()
    env.setdefault("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/")
    proc = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True, env=env)
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
    global KERNEL_NAME
    KERNEL_NAME = VERSION_MAP[args.version]
    print("=" * 64)
    print(
        f"a8w8 8-wave warp-pipeline perf — {args.M}x{args.N}x{args.K} f8  (version={args.version}, {KERNEL_NAME})"
    )
    print("=" * 64)

    clean_caches(WORK_DIR)

    tflops = None if args.skip_trace else run_kernel_trace(args)
    mfma_eff_raw = None if args.skip_att else run_att(args)
    mfma_eff_raw, mfma_eff = scale_eff(mfma_eff_raw, WAVES_PER_SIMD)
    vgprs, spills = parse_amdgcn_metadata(KERNEL_NAME)

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
        f"| {args.M:>5} | {args.N:>5} | {args.K:>6} | {'f8':>5} | "
        f"{fmt(round(tflops, 1) if tflops else None):>8} | {fmt(mfma_eff):>9} | "
        f"{fmt(vgprs):>5} | {fmt(spills):>6} |"
    )
    if mfma_eff is not None:
        print(f"\nMFMA Eff = {mfma_eff} (per-wave {mfma_eff_raw} x {WAVES_PER_SIMD} waves/SIMD)")


if __name__ == "__main__":
    main()
