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
Collect VMEM-latency hardware counters for the 8-wave warp-pipeline GEMM.

Reuses the tutorial's existing counter mechanism (scripts/run_counter_collection.py
helpers) but targets this kernel (gemm_async_warp_pipeline[_u3], selected by
--unroll) instead of the a16w16 VERSION_MAP.

Default counters answer "how long are the buffer_load / VMEM stalls":
  VmemLatency                  avg VMEM instruction latency (cycles), derived
  TCP_TCC_READ_REQ_LATENCY_sum total L1(TCP)->L2(TCC) read-request latency
  TCP_TCC_READ_REQ_sum         L1->L2 read requests   (avg = LATENCY/REQ)
  TCP_TCP_LATENCY_sum          total TCP wave latency
  TCP_TA_TCP_STATE_READ_sum    TCP reads             (avg wave lat = LATENCY/READ)

Run with the no-AGPR config to match the current best kernel:
  TRITON_HIP_AGPR_ALLOC="0,0" python collect_counters.py --unroll 3 --K 8192 --dtype fp16
"""

import argparse
import os
import shutil
import subprocess
import sys

GIT_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
SCRIPTS_DIR = os.path.join(GIT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from run_counter_collection import (  # noqa: E402
    clean_triton_cache,
    find_counter_csv,
    parse_counter_csv,
    write_counters_yaml,
)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_COUNTERS = (
    "VmemLatency,"
    "TCP_TCC_READ_REQ_LATENCY_sum,TCP_TCC_READ_REQ_sum,"
    "TCP_TCP_LATENCY_sum,TCP_TA_TCP_STATE_READ_sum"
)


def parse_args():
    p = argparse.ArgumentParser(description="Collect VMEM-latency counters for the 8-wave kernel")
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--unroll", type=int, default=3, choices=[2, 3])
    p.add_argument("--rotating-buffer-size", type=int, default=512)
    p.add_argument("--counters", default=DEFAULT_COUNTERS, help="comma-separated counter list")
    return p.parse_args()


def main():
    args = parse_args()
    counters = [c.strip() for c in args.counters.split(",")]
    kernel_name = "gemm_async_warp_pipeline_u3" if args.unroll == 3 else "gemm_async_warp_pipeline"

    print("=" * 70)
    print(f"VMEM-latency counters — {kernel_name}  K={args.K} {args.dtype}  unroll={args.unroll}")
    print(f"AGPR alloc override: {os.environ.get('TRITON_HIP_AGPR_ALLOC', '(default)')}")
    print("=" * 70)

    clean_triton_cache()

    counters_yaml = os.path.join(WORK_DIR, "counters.yaml")
    write_counters_yaml(counters, counters_yaml)

    trace_dir = os.path.join(WORK_DIR, "counter_tmp")
    if os.path.isdir(trace_dir):
        shutil.rmtree(trace_dir)

    cmd = [
        "rocprofv3", "-i", counters_yaml,
        "--kernel-include-regex", kernel_name,
        "-d", trace_dir, "--output-format", "csv", "--",
        "python", "bench.py", "--rocprof",
        "--K", str(args.K), "--dtype", args.dtype, "--unroll", str(args.unroll),
        "--rotating-buffer-size", str(args.rotating_buffer_size),
    ]
    print(f"  rocprofv3 counters: {', '.join(counters)} ...")
    proc = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True)  # inherits env
    if proc.returncode != 0:
        print("  FAILED:")
        for line in (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-10:]:
            print("    " + line)
        return

    csv_path = find_counter_csv(trace_dir)
    if csv_path is None:
        print(f"  no counter_collection.csv under {trace_dir}")
        return

    avgs, n = parse_counter_csv(csv_path, counters, kernel_name)
    print(f"\n  {n} dispatches matched\n")
    print(f"  {'counter':<32} {'avg / dispatch':>18}")
    print(f"  {'-'*32} {'-'*18}")
    for c in counters:
        v = avgs.get(c)
        print(f"  {c:<32} {('%.1f' % v) if v is not None else 'N/A':>18}")

    # Derived average latencies
    def ratio(num, den):
        a, b = avgs.get(num), avgs.get(den)
        return a / b if a and b else None

    print("\n  Derived average latencies (cycles):")
    vmem = avgs.get("VmemLatency")
    l1l2 = ratio("TCP_TCC_READ_REQ_LATENCY_sum", "TCP_TCC_READ_REQ_sum")
    tcpwave = ratio("TCP_TCP_LATENCY_sum", "TCP_TA_TCP_STATE_READ_sum")
    if vmem is not None:
        print(f"    VMEM instruction latency (VmemLatency)        : {vmem:>10.1f}")
    if l1l2 is not None:
        print(f"    L1->L2 read-request latency (TCC_LAT/REQ)     : {l1l2:>10.1f}")
    if tcpwave is not None:
        print(f"    TCP wave latency (TCP_LAT/TA_TCP_STATE_READ)  : {tcpwave:>10.1f}")


if __name__ == "__main__":
    main()
