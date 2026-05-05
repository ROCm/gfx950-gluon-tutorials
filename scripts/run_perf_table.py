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
run_perf_table.py

Automates running benchmarks across kernel versions and scheduler configs,
collecting TFLOPS, VGPR count, spills, and MFMA efficiency, then printing
a markdown performance table.

Usage:
    # a16w16 kernels (run from anywhere):
    python scripts/run_perf_table.py --kernel a16w16 --versions 5 6 7 8 --configs base llir llir+amdgcnas --K 4096 --dtype fp16

    # a8w8 kernel (run from anywhere):
    python scripts/run_perf_table.py --kernel a8w8 --configs llir+amdgcnas --K 8192

    # a4w4 kernel (run from anywhere):
    python scripts/run_perf_table.py --kernel a4w4 --configs llir+amdgcnas --K 8192

    # Use rocprofv3 for TFLOPS timing instead of do_bench:
    python scripts/run_perf_table.py --kernel a16w16 --configs llir+amdgcnas --versions 7 --K 8192 --dtype fp16 --rocprof
"""

import argparse
import csv
import glob
import json
import os
import re
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
    # Used to measure how much of the amdgcnas improvement comes from the
    # register-allocation hint vs. the post-assembly peephole / LICM.
    # Gated by the `TRITON_ENABLE_AMDGPU_RA_HINTS` env var (in
    # third_party/amd/backend/compiler.py and python/src/llvm.cc) and
    # supported natively by the `gfx950-tutorial-v0.1` pin.
    "llir+ra": {
        "TRITON_ENABLE_LLIR_SCHED": "1",
        "TRITON_ENABLE_AMDGPU_RA_HINTS": "1",
    },
    "llir+amdgcnas": {
        "TRITON_ENABLE_LLIR_SCHED": "1",
        "TRITON_ENABLE_AMDGCN_AS": "1",
    },
}

# (kernel, config) -> set of versions that have a published TFLOPS / MFMA-eff
# number in the tutorial. Pairs not in this set are skipped by default — they
# either crash at compile time (e.g. v0..v4 + llir segfault) or produce results
# that aren't part of the documented optimization story (e.g. v6 + llir+amdgcnas
# FAILs, v5 + amdgcnas spills 246 VGPRs).
#
# Single-kernel benchmarks (a8w8, a4w4) use `None` as the version sentinel,
# matching how main() already represents them.
#
# Pass --allow-unreported to bypass the gate (e.g. for development).
REPORTED_COMBINATIONS = {
    "a16w16": {
        "base": {0, 2, 3, 4, 5},
        "llir": {5, 6, 7},
        "llir+ra": {7},
        "llir+amdgcnas": {7, 8, 9},
    },
    "a8w8": {
        "base": {None},
        "llir": {None},
        "llir+amdgcnas": {None},
    },
    "a4w4": {
        "base": {None},
        "llir": {None},
        "llir+amdgcnas": {None},
    },
}


def is_reported(kernel, config, version):
    """Return True if this (kernel, config, version) appears in the tutorial."""
    return version in REPORTED_COMBINATIONS.get(kernel, {}).get(config, set())

TRITON_CACHE = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))

ATT_MATMUL_TEMPLATE = {
    "jobs": [
        {
            "kernel_include_regex": "",
            "kernel_exclude_regex": "",
            "kernel_iteration_range": "[15]",
            "advanced_thread_trace": True,
            "att_target_cu": 0,
            "att_shader_engine_mask": "0xF",
            "att_simd_select": "0xF",
            "att_buffer_size": "0x60000000",
        }
    ]
}

# Kernels with large instruction counts need bigger ATT buffers
ATT_BUFFER_SIZE_OVERRIDES = {
    "a4w4": "0x20000000",  # 512MB for MXFP4 (2x MFMA per iteration vs FP8)
}


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


def write_att_config(version_dir, work_dir=None, kernel_type="a16w16"):
    """Write att_matmul.json with kernel_include_regex set to the version dir name."""
    cfg = json.loads(json.dumps(ATT_MATMUL_TEMPLATE))
    cfg["jobs"][0]["kernel_include_regex"] = version_dir
    if kernel_type in ATT_BUFFER_SIZE_OVERRIDES:
        cfg["jobs"][0]["att_buffer_size"] = ATT_BUFFER_SIZE_OVERRIDES[kernel_type]
    att_path = os.path.join(work_dir, "att_matmul.json") if work_dir else "att_matmul.json"
    with open(att_path, "w") as f:
        json.dump(cfg, f, indent=4)


def clean_caches(work_dir=None):
    """Remove triton cache and local tmp/ directory."""
    if os.path.isdir(TRITON_CACHE):
        shutil.rmtree(TRITON_CACHE)
    tmp_dir = os.path.join(work_dir, "tmp") if work_dir else "tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)


def parse_tflops(output):
    """Parse TFLOPS from bench.py output.

    The benchmark table (pandas-style) has lines like:
        0  4096.0  4096.0  4096.0    123.456
    We grab the last float on the last such data row.
    """
    tflops = None
    for line in output.splitlines():
        # Match pandas-style data rows: index followed by M, N, K floats and TFLOPS
        m = re.match(r"\s*\d+\s+([\d.]+\s+[\d.]+\s+[\d.]+\s+.+)", line)
        if m:
            nums = re.findall(r"[\d.]+", m.group(1))
            if nums:
                tflops = float(nums[-1])
    return tflops


def parse_mfma_efficiency(output):
    """Parse MFMA efficiency from process_json.py JSON output."""
    m = re.search(r'"mfma efficiency"\s*:\s*"([\d.]+%)"', output)
    if m:
        return m.group(1)
    return None


def parse_amdgcn_metadata(version_dir):
    """Parse VGPRs and spills from the .amdgcn file in triton cache."""
    pattern = os.path.join(TRITON_CACHE, "*", f"{version_dir}*.amdgcn")
    files = glob.glob(pattern)
    if not files:
        return None, None

    vgprs = None
    spills = None
    for fpath in files:
        with open(fpath, "r") as f:
            for line in f:
                vm = re.search(r"\.vgpr_count:\s*(\d+)", line)
                if vm:
                    vgprs = int(vm.group(1))
                sm = re.search(r"\.vgpr_spill_count:\s*(\d+)", line)
                if sm:
                    spills = int(sm.group(1))
                if vgprs is not None and spills is not None:
                    break
        if vgprs is not None:
            break

    return vgprs, spills


def find_kernel_trace_csv(trace_dir):
    """Find the *_kernel_trace.csv file inside the rocprofv3 trace directory.

    rocprofv3 creates a subdirectory named after the node hostname, e.g.
    trace_dir/<hostname>/<pid>_kernel_trace.csv
    """
    pattern = os.path.join(trace_dir, "*", "*_kernel_trace.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    # Return the most recently modified one
    return max(files, key=os.path.getmtime)


def avg_kernel_time_ns(csv_path, kernel_name, last_n=100):
    """Return the average elapsed time in nanoseconds for the last_n matching rows.

    Early dispatches may have inflated times due to warmup effects.
    Averaging only the last_n dispatches gives steady-state timing.
    """
    durations = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_name in row["Kernel_Name"]:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                durations.append(end - start)
    if not durations:
        return None, 0
    tail = durations[-last_n:]
    return sum(tail) / len(tail), len(durations)


def run_rocprof_trace(version_dir, K, dtype, version, work_dir, env, kernel_type="a16w16"):
    """Run rocprofv3 --kernel-trace to collect kernel timestamps.

    Returns TFLOPS computed from the average kernel time, or None on failure.
    """
    M, N = 4096, 4096
    trace_dir = os.path.join(work_dir, f"{version_dir}_rocprof_trace")
    if os.path.isdir(trace_dir):
        shutil.rmtree(trace_dir)

    cmd = [
        "rocprofv3",
        "--kernel-trace",
        "-f",
        "csv",
        "--kernel-include-regex",
        version_dir,
        "-d",
        trace_dir,
        "--",
        "python",
        "bench.py",
        "--rocprof",
        "--K",
        str(K),
    ]
    if kernel_type not in ("a8w8", "a4w4"):
        cmd.extend(["--dtype", dtype, "--version", str(version)])

    rocprof_env = env.copy()
    rocprof_env["AMD_SERIALIZE_KERNEL"] = "3"

    print(f"  rocprofv3: collecting kernel trace for {version_dir} ...")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=rocprof_env,
            cwd=work_dir,
        )
        if proc.returncode != 0:
            print(f"  rocprofv3 FAILED (exit code {proc.returncode})")
            lines = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
            for line in lines[-5:]:
                print(f"    {line}")
            return None
    except Exception as e:
        print(f"  rocprofv3 FAILED: {e}")
        return None

    csv_path = find_kernel_trace_csv(trace_dir)
    if csv_path is None:
        print(f"  rocprofv3: no kernel_trace.csv found in {trace_dir}")
        return None

    avg_ns, count = avg_kernel_time_ns(csv_path, version_dir)
    if avg_ns is None:
        print(f"  rocprofv3: no rows matched kernel '{version_dir}' in {csv_path}")
        return None

    avg_us = avg_ns / 1e3
    tflops = 2 * M * N * K * 1e-12 / (avg_ns * 1e-9)
    print(f"  rocprofv3: {count} dispatches, avg={avg_us:.2f} us, tflops={tflops:.1f}")
    return tflops


def run_benchmark(version, config, K, dtype, kernel="a16w16", use_rocprof=False):
    """Run a single benchmark for the given version, config, and kernel type.

    Returns a dict with keys: tflops, vgprs, spills, mfma_eff, or None values on failure.
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
        "tflops": None,
        "vgprs": None,
        "spills": None,
        "mfma_eff": None,
    }

    clean_caches(work_dir)
    write_att_config(version_dir, work_dir, kernel_type=kernel)

    run_att_path = os.path.join(git_root, "scripts", "run_att.py")

    env = os.environ.copy()
    # Clear any previous config env vars
    for key in (
        "TRITON_ENABLE_LLIR_SCHED",
        "TRITON_ENABLE_AMDGCN_AS",
        "TRITON_ENABLE_AMDGPU_RA_HINTS",
    ):
        env.pop(key, None)
    # Set config-specific env vars
    env.update(CONFIG_ENV[config])

    cmd = [
        sys.executable,
        run_att_path,
        "--att-output",
        "tmp",
        "python",
        "bench.py",
        "--K",
        str(K),
    ]
    if kernel not in ("a8w8", "a4w4"):
        cmd.extend(["--dtype", dtype, "--version", str(version)])

    if kernel in ("a8w8", "a4w4"):
        print(f"  Running: {kernel} config={config}")
    else:
        print(f"  Running: v{version} ({version_dir}) config={config}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=work_dir,
        )
        combined = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            # Print last few lines of output for debugging
            lines = combined.strip().splitlines()
            for line in lines[-5:]:
                print(f"    {line}")
            if not use_rocprof:
                return result
        else:
            if not use_rocprof:
                result["tflops"] = parse_tflops(combined)
            result["mfma_eff"] = parse_mfma_efficiency(combined)

    except Exception as e:
        print(f"  FAILED: {e}")
        if not use_rocprof:
            return result

    vgprs, spills = parse_amdgcn_metadata(version_dir)
    result["vgprs"] = vgprs
    result["spills"] = spills

    # Run rocprofv3 to get TFLOPS from kernel timestamps
    if use_rocprof:
        tflops = run_rocprof_trace(
            version_dir, K, dtype, version, work_dir, env, kernel_type=kernel
        )
        result["tflops"] = tflops

    return result


def format_val(val, fmt=None):
    """Format a value for the table, returning 'FAIL' if None."""
    if val is None:
        return "FAIL"
    if fmt:
        return fmt.format(val)
    return str(val)


def print_table(config, rows):
    """Print a markdown table for one config."""
    print(f"\nConfig: {config}")
    print("| Version              | TFLOPS | VGPRs | Spills | MFMA Eff. |")
    print("|----------------------|--------|-------|--------|-----------|")
    for row in rows:
        version_dir = row["version_dir"]
        tflops = (
            format_val(row["tflops"], "{:.0f}")
            if isinstance(row["tflops"], float)
            else format_val(row["tflops"])
        )
        vgprs = format_val(row["vgprs"])
        spills = format_val(row["spills"])
        mfma_eff = format_val(row["mfma_eff"])
        print(f"| {version_dir:<20} | {tflops:>6} | {vgprs:>5} | {spills:>6} | {mfma_eff:>9} |")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmarks across kernel versions and configs, print a performance table."
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
        "--rocprof",
        action="store_true",
        help="Use rocprofv3 kernel-trace for TFLOPS instead of do_bench.",
    )
    parser.add_argument(
        "--allow-unreported",
        action="store_true",
        help="Run (version, config) pairs that do not have a published number "
        "in the tutorial. Off by default — unreported pairs include known "
        "crashes (e.g. v0..v4 + llir segfault) and configs that aren't part "
        "of the documented optimization story.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.kernel in ("a8w8", "a4w4"):
        # a8w8/a4w4 have a single kernel, --versions is ignored
        versions = [None]
    else:
        # Validate versions for a16w16
        for v in args.versions:
            if v not in VERSION_MAP:
                print(f"Error: version {v} not in VERSION_MAP. Valid: {list(VERSION_MAP.keys())}")
                sys.exit(1)
        versions = args.versions

    # Filter (config, version) pairs to those reported in the tutorial unless
    # --allow-unreported is set. Skipped pairs are listed up front so the user
    # sees exactly what was dropped.
    skipped = []
    for config in args.configs:
        for version in versions:
            if not args.allow_unreported and not is_reported(args.kernel, config, version):
                skipped.append((config, version))

    if skipped:
        print(f"\n{'='*60}")
        print("Skipped (not reported in tutorial; pass --allow-unreported to run):")
        for config, version in skipped:
            label = VERSION_MAP[version] if version is not None else args.kernel
            print(f"  {label} + {config}")
        print(f"{'='*60}")

    # Collect results grouped by config
    results = {}
    for config in args.configs:
        results[config] = []
        active_versions = [
            v for v in versions
            if args.allow_unreported or is_reported(args.kernel, config, v)
        ]
        if not active_versions:
            continue
        print(f"\n{'='*60}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        for version in active_versions:
            row = run_benchmark(
                version,
                config,
                args.K,
                args.dtype,
                kernel=args.kernel,
                use_rocprof=args.rocprof,
            )
            results[config].append(row)

    # Print summary tables
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for config in args.configs:
        if results[config]:
            print_table(config, results[config])


if __name__ == "__main__":
    main()
