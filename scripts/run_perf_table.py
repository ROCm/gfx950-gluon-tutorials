#!/usr/bin/env python3
"""
run_perf_table.py

Automates running benchmarks across kernel versions and scheduler configs,
collecting TFLOPS, VGPR count, spills, and MFMA efficiency, then printing
a markdown performance table.

Usage (run from kernels/gemm/a16w16/):
    python ../../../scripts/run_perf_table.py --versions 5 6 7 8 --configs base llir llir+amdgcnas --K 4096 --dtype fp16
"""

import argparse
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
    7: "v7_slice",
    8: "v8_beyond_hotloop",
}

CONFIG_ENV = {
    "base": {},
    "llir": {"TRITON_ENABLE_LLIR_SCHED": "1"},
    "llir+amdgcnas": {
        "TRITON_ENABLE_LLIR_SCHED": "1",
        "TRITON_ENABLE_AMDGCN_AS": "1",
    },
}

TRITON_CACHE = os.path.expanduser("~/.triton/cache")

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
            "att_buffer_size": "0x6000000",
        }
    ]
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


def write_att_config(version_dir):
    """Write att_matmul.json with kernel_include_regex set to the version dir name."""
    cfg = json.loads(json.dumps(ATT_MATMUL_TEMPLATE))
    cfg["jobs"][0]["kernel_include_regex"] = version_dir
    with open("att_matmul.json", "w") as f:
        json.dump(cfg, f, indent=4)


def clean_caches():
    """Remove triton cache and local tmp/ directory."""
    if os.path.isdir(TRITON_CACHE):
        shutil.rmtree(TRITON_CACHE)
    if os.path.isdir("tmp"):
        shutil.rmtree("tmp")


def parse_tflops(output):
    """Parse TFLOPS from bench.py output.

    The benchmark table has lines like:
        (4096, 4096, 4096)     123.456
    We grab the last float on the last such line.
    """
    tflops = None
    for line in output.splitlines():
        # Match lines that start with a tuple like (M, N, K) and end with floats
        m = re.search(r"\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)\s+(.+)", line)
        if m:
            # The last number on the line is the TFLOPS value
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


def run_benchmark(version, config, K, dtype):
    """Run a single benchmark for the given version and config.

    Returns a dict with keys: tflops, vgprs, spills, mfma_eff, or None values on failure.
    """
    version_dir = VERSION_MAP[version]
    result = {"version_dir": version_dir, "tflops": None, "vgprs": None, "spills": None, "mfma_eff": None}

    clean_caches()
    write_att_config(version_dir)

    git_root = get_git_root()
    run_att_path = os.path.join(git_root, "scripts", "run_att.py")

    env = os.environ.copy()
    # Clear any previous config env vars
    for key in ("TRITON_ENABLE_LLIR_SCHED", "TRITON_ENABLE_AMDGCN_AS"):
        env.pop(key, None)
    # Set config-specific env vars
    env.update(CONFIG_ENV[config])

    cmd = [
        sys.executable,
        run_att_path,
        "--att-output", "tmp",
        "python", "bench.py",
        "--K", str(K),
        "--dtype", dtype,
        "--version", str(version),
    ]

    print(f"  Running: v{version} ({version_dir}) config={config}")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        combined = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            # Print last few lines of output for debugging
            lines = combined.strip().splitlines()
            for line in lines[-5:]:
                print(f"    {line}")
            return result

        result["tflops"] = parse_tflops(combined)
        result["mfma_eff"] = parse_mfma_efficiency(combined)

    except Exception as e:
        print(f"  FAILED: {e}")
        return result

    vgprs, spills = parse_amdgcn_metadata(version_dir)
    result["vgprs"] = vgprs
    result["spills"] = spills

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
        tflops = format_val(row["tflops"], "{:.0f}") if isinstance(row["tflops"], float) else format_val(row["tflops"])
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
        "--versions",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8],
        help="Kernel versions to benchmark (default: 5 6 7 8)",
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
        help="Data type for benchmark (default: fp16)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate versions
    for v in args.versions:
        if v not in VERSION_MAP:
            print(f"Error: version {v} not in VERSION_MAP. Valid: {list(VERSION_MAP.keys())}")
            sys.exit(1)

    # Collect results grouped by config
    results = {}
    for config in args.configs:
        print(f"\n{'='*60}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        results[config] = []
        for version in args.versions:
            row = run_benchmark(version, config, args.K, args.dtype)
            results[config].append(row)

    # Print summary tables
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for config in args.configs:
        print_table(config, results[config])


if __name__ == "__main__":
    main()
