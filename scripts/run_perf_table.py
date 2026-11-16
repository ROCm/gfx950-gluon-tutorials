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
    python scripts/run_perf_table.py --kernel a4w4 --versions 0 1 --configs llir+amdgcnas --K 8192

    # Published numbers: the sustained protocol (60 s warm-up + 30 s window per row):
    python scripts/run_perf_table.py --kernel a16w16 --configs llir+amdgcnas --versions 9 --K 8192 --dtype fp16 --protocol sustained

    # Cross-check the event timing with rocprofv3 --kernel-trace over the same window:
    python scripts/run_perf_table.py --kernel a16w16 --configs llir+amdgcnas --versions 7 --K 8192 --dtype fp16 --rocprof

Each row: TFLOPS from bench.py's timing protocol (scripts/graph_timer.py: graph replay of
back-to-back dispatches over rotating inputs, warm-up, then a measurement window; see
docs/benchmarking.md); MFMA efficiency from rocprofv3 --att on one dispatch taken after a
warm-up of --att-warmup-s seconds of back-to-back dispatches; VGPRs/spills from the kernel the
run compiled into a private Triton cache (the user's cache is never touched).
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_timer  # noqa: E402  (stdlib-only at import time)

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

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scheduler configs. The MFMA scheduler is part of the pinned Triton
# (triton-lang/triton#12209): the kernels that use it pass
# schedule_hint="mfma-schedule" themselves, so "llir" needs nothing from the
# harness and "base" turns the hint off on the bench.py command line. amdgcnas
# is still the out-of-tree post-assembly peephole, enabled through the
# environment. Each config adds one component on top of the previous one, so a
# perf table row's number reflects that stack.
CONFIGS = {
    "base": {"env": {}, "args": ["--schedule-hint", ""]},
    "llir": {"env": {}, "args": []},
    "llir+amdgcnas": {"env": {"TRITON_AMDGCNAS_PLUGIN": "1"}, "args": []},
}

# (kernel, config) -> set of versions that have a published TFLOPS / MFMA-eff
# number in the tutorial. Pairs not in this set are skipped by default — they
# either crash at compile time (e.g. v0..v4 + llir segfault) or produce results
# that aren't part of the documented optimization story (e.g. v5 + amdgcnas
# spills 246 VGPRs).
#
# Single-kernel benchmarks (a8w8) use `None` as the version sentinel.
# Multi-version kernels (a16w16, a4w4) use the numeric versions from VERSION_MAP
# (a16w16) or A4W4_VERSION_MAP (a4w4).
#
# Pass --allow-unreported to bypass the gate (e.g. for development).
REPORTED_COMBINATIONS = {
    "a16w16": {
        "base": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        "llir": {5, 6, 7, 8, 9},
        "llir+amdgcnas": {7, 8, 9},
    },
    "a8w8": {
        "base": {None},
        "llir": {None},
        "llir+amdgcnas": {None},
    },
    "a4w4": {
        "base": {0, 1},
        "llir": {0, 1},
        "llir+amdgcnas": {0, 1},
    },
}

# a4w4 has its own version → directory map. Each directory's matmul_kernel.py
# defines a kernel function whose name matches the directory (e.g. v0_sliceN
# function in v0_sliceN/matmul_kernel.py), so the same `version_dir` value
# serves both rocprof's kernel_include_regex and the bench.py --version arg.
A4W4_VERSION_MAP = {
    0: "v0_sliceN",
    1: "v1_sliceMN",
}


def is_reported(kernel, config, version):
    """Return True if this (kernel, config, version) appears in the tutorial."""
    return version in REPORTED_COMBINATIONS.get(kernel, {}).get(config, set())


# Each run compiles into a fresh private Triton cache (TRITON_CACHE_DIR in the subprocess env):
# the configs differ only in compile-time hooks that Triton's cache key does not see, so a shared
# cache could hand one config another config's binary, and deleting the user's cache instead
# (as this script used to) destroys other jobs' compiled kernels.
ATT_WARMUP_S = 1.0

ATT_MATMUL_TEMPLATE = {
    "jobs": [
        {
            "kernel_include_regex": "",
            "kernel_exclude_regex": "",
            "kernel_iteration_range": "[1]",  # set per run by write_att_config
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


def write_att_config(version_dir, work_dir=None, kernel_type="a16w16", iteration=1):
    """Write att_matmul.json: trace dispatch number ``iteration`` of the kernel ``version_dir``."""
    cfg = json.loads(json.dumps(ATT_MATMUL_TEMPLATE))
    cfg["jobs"][0]["kernel_include_regex"] = version_dir
    cfg["jobs"][0]["kernel_iteration_range"] = f"[{iteration}]"
    if kernel_type in ATT_BUFFER_SIZE_OVERRIDES:
        cfg["jobs"][0]["att_buffer_size"] = ATT_BUFFER_SIZE_OVERRIDES[kernel_type]
    att_path = os.path.join(work_dir, "att_matmul.json") if work_dir else "att_matmul.json"
    with open(att_path, "w") as f:
        json.dump(cfg, f, indent=4)


def clean_caches(work_dir=None):
    """Remove the local ATT output directory (tmp/). Triton's cache is left alone."""
    tmp_dir = os.path.join(work_dir, "tmp") if work_dir else "tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)


def private_triton_env(env):
    """A copy of ``env`` with TRITON_CACHE_DIR set to a fresh directory; returns (env, dir)."""
    cache_dir = tempfile.mkdtemp(prefix="perf_table_triton_cache_")
    env = dict(env)
    env["TRITON_CACHE_DIR"] = cache_dir
    return env, cache_dir


def timing_cli(args):
    """bench.py timing flags for the chosen protocol."""
    cli = ["--protocol", args.protocol]
    if args.warmup_s is not None:
        cli += ["--warmup-s", str(args.warmup_s)]
    if args.window_s is not None:
        cli += ["--window-s", str(args.window_s)]
    return cli


def run_timing(
    work_dir, bench_args, env, protocol_cli, trace_kernel=None, trace_dir=None, keep_trace=False
):
    """Run bench.py's timing protocol; return (result dict or None, kernel-trace median us or None).

    With ``trace_kernel``, the run happens under ``rocprofv3 --kernel-trace`` (no serialization)
    and the dispatches that start inside the measurement window give a kernel-trace median.
    """
    cmd = [sys.executable, "bench.py", *bench_args, *protocol_cli]
    if trace_kernel:
        if os.path.isdir(trace_dir):
            shutil.rmtree(trace_dir)
        cmd = [
            "rocprofv3",
            "--kernel-trace",
            "-f",
            "csv",
            "--kernel-include-regex",
            trace_kernel,
            "-d",
            trace_dir,
            "--",
        ] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work_dir)
    results = graph_timer.parse_results(proc.stdout)
    if proc.returncode != 0 or not results:
        print(f"  timing FAILED (exit code {proc.returncode})")
        for line in (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-5:]:
            print(f"    {line}")
        return None, None
    for line in proc.stdout.splitlines():
        if "match" in line or "differ" in line or line.startswith("[") and "us/dispatch" in line:
            print(f"  {line.strip()}")
    result = results[-1]
    trace_us = None
    if trace_kernel:
        csv_path = find_kernel_trace_csv(trace_dir)
        if csv_path:
            trace_us, n = graph_timer.trace_window_median(
                csv_path, trace_kernel, *result["window_monotonic_ns"]
            )
            if trace_us:
                print(f"  kernel trace: median {trace_us:.2f} us over the window's {n} dispatches")
        if not keep_trace:
            shutil.rmtree(trace_dir, ignore_errors=True)
    return result, trace_us


def run_att_steady(work_dir, kernel_name, bench_args, env, dispatch_us, kernel_type, warmup_s):
    """MFMA efficiency from ATT on one dispatch that follows ``warmup_s`` of back-to-back ones.

    bench.py's profiler mode issues 1 correctness-check dispatch, then W warm-up dispatches, then
    a few more; the ATT config names dispatch 1 + W + 1, so the traced dispatch runs in the
    back-to-back state rather than in the start-up transient (dispatch 15, ~3 ms after the start,
    where this script used to trace). ATT serializes only the traced dispatch.
    """
    warmup = graph_timer.warmup_dispatches_for(warmup_s, dispatch_us)
    write_att_config(kernel_name, work_dir, kernel_type=kernel_type, iteration=1 + warmup + 1)
    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "scripts", "run_att.py"),
        "--att-output",
        "tmp",
        sys.executable,
        "bench.py",
        *bench_args,
        "--rocprof",
        "--warmup-dispatches",
        str(warmup),
        "--iters",
        "3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=work_dir)
    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(f"  ATT FAILED (exit code {proc.returncode})")
        for line in combined.strip().splitlines()[-5:]:
            print(f"    {line}")
        return None
    eff = parse_mfma_efficiency(combined)
    print(f"  ATT: dispatch {1 + warmup + 1} ({warmup_s:g} s of warm-up), MFMA efficiency {eff}")
    return eff


def parse_mfma_efficiency(output):
    """Parse MFMA efficiency from process_json.py JSON output."""
    m = re.search(r'"mfma efficiency"\s*:\s*"([\d.]+%)"', output)
    if m:
        return m.group(1)
    return None


def parse_amdgcn_metadata(version_dir, cache_dir):
    """Parse VGPRs and spills from the kernel's .amdgcn in the Triton cache ``cache_dir``."""
    pattern = os.path.join(cache_dir, "*", f"{version_dir}*.amdgcn")
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


def run_benchmark(version, config, K, dtype, args, kernel="a16w16"):
    """Run one (version, config): timing, then ATT, then VGPR/spill parsing.

    Returns a dict with keys: tflops, us, power_w, clock_mhz, trace_tflops, vgprs, spills, mfma_eff.
    """
    git_root = get_git_root()

    if kernel == "a8w8":
        version_dir = "a8w8_kernel"
        work_dir = os.path.join(git_root, "kernels", "gemm", "intra_wave", "a8w8")
    elif kernel == "a4w4":
        version_dir = A4W4_VERSION_MAP[version]
        work_dir = os.path.join(git_root, "kernels", "gemm", "intra_wave", "a4w4")
    else:
        version_dir = VERSION_MAP[version]
        work_dir = os.path.join(git_root, "kernels", "gemm", "intra_wave", "a16w16")

    result = dict(version_dir=version_dir, tflops=None, us=None, power_w=None, clock_mhz=None)
    result.update(trace_tflops=None, vgprs=None, spills=None, mfma_eff=None)

    clean_caches(work_dir)
    env = os.environ.copy()
    # Clear any previous config env vars, then set this config's
    for key in (
        "TRITON_AMDGCNAS_PLUGIN",
        "TRITON_ENABLE_AMDGCN_AS",
    ):
        env.pop(key, None)
    env.update(CONFIGS[config]["env"])
    env, cache_dir = private_triton_env(env)

    bench_args = ["--K", str(K)]
    if kernel == "a16w16":
        bench_args += ["--dtype", dtype, "--version", str(version)]
    elif kernel == "a4w4":
        bench_args += ["--version", str(version)]
    bench_args += CONFIGS[config]["args"]

    if kernel == "a8w8":
        print(f"  Running: {kernel} config={config}")
    else:
        print(f"  Running: v{version} ({version_dir}) config={config}")
    try:
        timing, trace_us = run_timing(
            work_dir,
            bench_args,
            env,
            timing_cli(args),
            trace_kernel=version_dir if args.rocprof else None,
            trace_dir=os.path.join(work_dir, f"{version_dir}_rocprof_trace"),
        )
        if timing is not None:
            result.update(
                tflops=timing["tflops"],
                us=timing["median_us"],
                power_w=timing["power_w"],
                clock_mhz=timing["clock_mhz"],
                protocol=timing["protocol"],
                gpu=timing["gpu"],
            )
            if trace_us:
                result["trace_tflops"] = timing["flops"] / (trace_us * 1e-6) / 1e12
            if not args.skip_att:
                result["mfma_eff"] = run_att_steady(
                    work_dir,
                    version_dir,
                    bench_args,
                    env,
                    timing["median_us"],
                    kernel,
                    args.att_warmup_s,
                )
        result["vgprs"], result["spills"] = parse_amdgcn_metadata(version_dir, cache_dir)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return result


def format_val(val, fmt=None):
    """Format a value for the table, returning 'FAIL' if None."""
    if val is None:
        return "FAIL"
    if fmt:
        return fmt.format(val)
    return str(val)


def print_table(config, rows, with_trace=False):
    """Print a markdown table for one config."""
    print(f"\nConfig: {config}")
    trace_h, trace_d = (" Trace TF |", "----------|") if with_trace else ("", "")
    print(f"| Version              | TFLOPS |{trace_h} VGPRs | Spills | MFMA Eff. |")
    print(f"|----------------------|--------|{trace_d}-------|--------|-----------|")
    for row in rows:
        version_dir = row["version_dir"]
        tflops = format_val(row["tflops"], "{:.0f}")
        trace = f" {format_val(row['trace_tflops'], '{:.0f}'):>8} |" if with_trace else ""
        vgprs = format_val(row["vgprs"])
        spills = format_val(row["spills"])
        mfma_eff = format_val(row["mfma_eff"])
        print(
            f"| {version_dir:<20} | {tflops:>6} |{trace} {vgprs:>5} | {spills:>6} | {mfma_eff:>9} |"
        )
    print()


def print_conditions(rows):
    """One line per row: what the TFLOPS were measured under."""
    for row in rows:
        if row.get("us") is None:
            continue
        gpu = row.get("gpu") or {}
        cond = f"{row['us']:.2f} us/dispatch, protocol {row.get('protocol')}, GPU {gpu.get('bdf')}"
        if gpu.get("power_cap_w"):
            cond += f" (cap {gpu['power_cap_w']:.0f} W)"
        if row.get("power_w") is not None:
            cond += f", {row['power_w']:.0f} W, {row['clock_mhz']:.0f} MHz in the window"
        print(f"  {row['version_dir']}: {cond}")


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
        help="Kernel versions to benchmark (default: 5 6 7 8 for a16w16, "
        "valid 0 1 for a4w4). Ignored for a8w8.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(CONFIGS),
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
        help="Also time the window with rocprofv3 --kernel-trace (no serialization) and report "
        "the kernel-trace TFLOPS next to the event-timed ones.",
    )
    parser.add_argument(
        "--protocol",
        choices=sorted(graph_timer.PROTOCOLS),
        default=graph_timer.DEFAULT_PROTOCOL,
        help="bench.py timing protocol: quick (1 s warm-up + 4 s window) or sustained (60 s + 30 s, "
        "for published numbers). Default: %(default)s",
    )
    parser.add_argument("--warmup-s", type=float, default=None, help="override the warm-up time")
    parser.add_argument("--window-s", type=float, default=None, help="override the window length")
    parser.add_argument(
        "--att-warmup-s",
        type=float,
        default=ATT_WARMUP_S,
        help="seconds of back-to-back dispatches before the ATT-traced one (default: %(default)s)",
    )
    parser.add_argument("--skip-att", action="store_true", help="skip ATT (MFMA efficiency)")
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

    if args.att_warmup_s < 0:
        print("Error: --att-warmup-s must be >= 0")
        sys.exit(2)

    if args.kernel == "a8w8":
        # a8w8 has a single kernel, --versions is ignored
        versions = [None]
    elif args.kernel == "a4w4":
        # Validate versions for a4w4
        for v in args.versions:
            if v not in A4W4_VERSION_MAP:
                print(
                    f"Error: version {v} not in A4W4_VERSION_MAP. Valid: {list(A4W4_VERSION_MAP.keys())}"
                )
                sys.exit(1)
        versions = args.versions
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

    def _version_label(kernel, version):
        if version is None:
            return kernel
        if kernel == "a4w4":
            return A4W4_VERSION_MAP[version]
        return VERSION_MAP[version]

    if skipped:
        print(f"\n{'='*60}")
        print("Skipped (not reported in tutorial; pass --allow-unreported to run):")
        for config, version in skipped:
            print(f"  {_version_label(args.kernel, version)} + {config}")
        print(f"{'='*60}")

    # Collect results grouped by config
    results = {}
    for config in args.configs:
        results[config] = []
        active_versions = [
            v for v in versions if args.allow_unreported or is_reported(args.kernel, config, v)
        ]
        if not active_versions:
            continue
        print(f"\n{'='*60}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        for version in active_versions:
            row = run_benchmark(version, config, args.K, args.dtype, args, kernel=args.kernel)
            results[config].append(row)

    # Print summary tables
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for config in args.configs:
        if results[config]:
            print_table(config, results[config], with_trace=args.rocprof)
    print("Measured under:")
    for config in args.configs:
        print_conditions(results[config])


if __name__ == "__main__":
    main()
