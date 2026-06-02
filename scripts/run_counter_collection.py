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
run_counter_collection.py

Automates hardware performance counter collection using rocprofv3 across
kernel versions and scheduler configs, then prints a summary table of
averaged counter values.

Usage:
    python scripts/run_counter_collection.py --kernel a16w16 --versions 7 8 --configs base llir \\
        --counters TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum --K 4096 --dtype fp16
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

# Duplicated from bench.py to avoid importing it (it pulls in torch/triton).
VERSION_MAP = {
    0: "v0_naive",
    1: "v1_buffer_load",
    2: "v2_async_copy",
    3: "v3_lds",
    4: "v4_global_prefetch",
    5: "v5_local_prefetch",
    6: "v6_loop_unroll",
    7: "v7_sliceN",
    8: "v8_sliceMN",
    9: "v9_beyond_hotloop",
}

CONFIG_ENV = {
    "base": {},
    "llir": {"TRITON_ENABLE_LLIR_SCHED": "1"},
    # RA hints (LLVM flag) only, without the amdgcnas post-assembly pass.
    # Gated by the `TRITON_ENABLE_AMDGPU_RA_HINTS` env var, supported
    # natively by the `gfx950-tutorial-v0.2` pin.
    "llir+ra": {
        "TRITON_ENABLE_LLIR_SCHED": "1",
        "TRITON_ENABLE_AMDGPU_RA_HINTS": "1",
    },
    "llir+amdgcnas": {
        "TRITON_ENABLE_LLIR_SCHED": "1",
        "TRITON_ENABLE_AMDGCN_AS": "1",
    },
}

TRITON_CACHE = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))


def get_git_root():
    """Return the absolute path to the git repository root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_counters_yaml(counters, path):
    """Write the rocprofv3 counters input file."""
    with open(path, "w") as f:
        f.write("jobs:\n")
        f.write("  - pmc:\n")
        for counter in counters:
            f.write(f"      - {counter}\n")


def clean_triton_cache():
    """Remove triton cache so the kernel is recompiled."""
    if os.path.isdir(TRITON_CACHE):
        shutil.rmtree(TRITON_CACHE)


def find_counter_csv(trace_dir):
    """Find the *_counter_collection.csv inside the rocprofv3 output directory.

    rocprofv3 creates: trace_dir/pass_N/<hostname>/<pid>_counter_collection.csv
    """
    pattern = os.path.join(trace_dir, "**", "*_counter_collection.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_counter_csv(csv_path, counters, kernel_name):
    """Parse the counter collection CSV and return average values.

    Returns a dict mapping counter name -> average value, only for rows
    whose Kernel_Name contains kernel_name.
    """
    totals = {c: [] for c in counters}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_name not in row["Kernel_Name"]:
                continue
            name = row["Counter_Name"]
            if name in totals:
                totals[name].append(float(row["Counter_Value"]))

    averages = {}
    for c in counters:
        vals = totals[c]
        averages[c] = sum(vals) / len(vals) if vals else None
    n_dispatches = len(totals[counters[0]]) if totals[counters[0]] else 0
    return averages, n_dispatches


def run_collection(version, config, counters, K, dtype, kernel="a16w16"):
    """Run rocprofv3 counter collection for one (version, config) pair.

    Returns a dict with version_dir, averages (counter->value), and n_dispatches.
    """
    git_root = get_git_root()

    if kernel == "a8w8":
        version_dir = "a8w8_kernel"
        work_dir = os.path.join(git_root, "kernels", "gemm", "a8w8")
    elif kernel == "a4w4":
        version_dir = "a4w4_kernel"
        work_dir = os.path.join(git_root, "kernels", "gemm", "a4w4")
    else:
        version_dir = VERSION_MAP[version]
        work_dir = os.path.join(git_root, "kernels", "gemm", "a16w16")

    result = {
        "version_dir": version_dir,
        "averages": {c: None for c in counters},
        "n_dispatches": 0,
    }

    clean_triton_cache()

    # Write counters.yaml
    counters_yaml = os.path.join(work_dir, "counters.yaml")
    write_counters_yaml(counters, counters_yaml)

    # Prepare output directory
    trace_dir = os.path.join(work_dir, "counter_tmp")
    if os.path.isdir(trace_dir):
        shutil.rmtree(trace_dir)

    # Build env
    env = os.environ.copy()
    for key in (
        "TRITON_ENABLE_LLIR_SCHED",
        "TRITON_ENABLE_AMDGCN_AS",
        "TRITON_ENABLE_AMDGPU_RA_HINTS",
    ):
        env.pop(key, None)
    env.update(CONFIG_ENV[config])

    # Build benchmark command
    bench_cmd = ["python", "bench.py", "--rocprof", "--K", str(K)]
    if kernel not in ("a8w8", "a4w4"):
        bench_cmd.extend(["--dtype", dtype, "--version", str(version)])

    cmd = [
        "rocprofv3",
        "-i",
        counters_yaml,
        "--kernel-include-regex",
        version_dir,
        "-d",
        trace_dir,
        "--output-format",
        "csv",
        "--",
    ] + bench_cmd

    label = f"v{version} ({version_dir})" if kernel not in ("a8w8", "a4w4") else kernel
    print(f"  Collecting counters: {label} config={config}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=work_dir,
        )
        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
            for line in lines[-5:]:
                print(f"    {line}")
            return result
    except Exception as e:
        print(f"  FAILED: {e}")
        return result

    # Parse results
    csv_path = find_counter_csv(trace_dir)
    if csv_path is None:
        print(f"  No counter_collection.csv found in {trace_dir}")
        return result

    averages, n_dispatches = parse_counter_csv(csv_path, counters, version_dir)
    result["averages"] = averages
    result["n_dispatches"] = n_dispatches
    print(f"  OK ({n_dispatches} dispatches)")

    return result


def print_table(config, rows, counters):
    """Print a markdown summary table for one config."""
    # Build header
    header = (
        "| Version              | " + " | ".join(f"{c:<28}" for c in counters) + " | Dispatches |"
    )
    sep_parts = (
        [
            "|----------------------|",
        ]
        + [f"{''.join(['-'] * 30)}|" for _ in counters]
        + ["------------|"]
    )
    sep = "".join(sep_parts)

    print(f"\nConfig: {config}")
    print(header)
    print(sep)
    for row in rows:
        version_dir = row["version_dir"]
        vals = []
        for c in counters:
            v = row["averages"][c]
            vals.append(f"{v:>28,.0f}" if v is not None else f"{'FAIL':>28}")
        n = row["n_dispatches"]
        disp = f"{n:>10,}" if n else f"{'FAIL':>10}"
        print(f"| {version_dir:<20} | " + " | ".join(vals) + f" | {disp} |")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect hardware performance counters using rocprofv3 and print averaged results."
    )
    parser.add_argument(
        "--kernel",
        choices=["a16w16", "a8w8", "a4w4"],
        default="a16w16",
        help="Kernel type to benchmark (default: a16w16)",
    )
    parser.add_argument(
        "--versions",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8],
        help="Kernel versions to benchmark (default: 5 6 7 8). Ignored for a8w8.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIG_ENV.keys()),
        default=list(CONFIG_ENV.keys()),
        help="Scheduler configs to test (default: all)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=4096,
        help="K dimension for GEMM (default: 4096)",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16"],
        help="Data type for benchmark (default: fp16). Ignored for a8w8.",
    )
    parser.add_argument(
        "--counters",
        required=True,
        help="Comma-separated list of hardware counters to collect "
        "(e.g. TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    counters = [c.strip() for c in args.counters.split(",")]

    if args.kernel in ("a8w8", "a4w4"):
        versions = [None]
    else:
        for v in args.versions:
            if v not in VERSION_MAP:
                print(f"Error: version {v} not in VERSION_MAP. Valid: {list(VERSION_MAP.keys())}")
                sys.exit(1)
        versions = args.versions

    # Collect results grouped by config
    results = {}
    for config in args.configs:
        print(f"\n{'='*60}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        results[config] = []
        for version in versions:
            row = run_collection(version, config, counters, args.K, args.dtype, kernel=args.kernel)
            results[config].append(row)

    # Print summary tables
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for config in args.configs:
        print_table(config, results[config], counters)


if __name__ == "__main__":
    main()
