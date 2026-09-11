#!/usr/bin/env python3
"""Run memory-bound GEMM benchmark with rocprofv3 and compute bandwidth.

Usage:
    # Single K value:
    python run_bench.py --K 8192

    # All default K values (1024, 2048, 4096, 8192, 16384):
    python run_bench.py

    # Custom rotating buffer size:
    python run_bench.py --K 8192 --rotating-buffer-size 1024
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

M = 32
N = 65536
BLOCK_N = 256
ELEM_BYTES = 2  # fp16
KERNEL_NAME = "matmul_kernel"
DEFAULT_K_VALUES = [1024, 2048, 4096, 8192, 16384]


def find_kernel_trace_csv(trace_dir):
    """Find the *_kernel_trace.csv inside the rocprofv3 trace directory."""
    pattern = os.path.join(trace_dir, "*", "*_kernel_trace.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_kernel_times(csv_path, kernel_name, warmup_skip=1):
    """Parse kernel durations from rocprof CSV, grouped by Grid_Size_X.

    Returns dict mapping grid_size_x -> list of duration_ns.
    Skips the first `warmup_skip` dispatches per grid size.
    """
    groups = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_name not in row["Kernel_Name"]:
                continue
            grid_x = int(row["Grid_Size_X"])
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            groups.setdefault(grid_x, []).append(end - start)

    # Skip warmup dispatches
    for grid_x in groups:
        groups[grid_x] = groups[grid_x][warmup_skip:]

    return groups


def grid_x_to_k(grid_x, wg_size=256):
    """Derive K from Grid_Size_X.

    Grid_Size_X = num_workgroups * workgroup_size
    num_workgroups = (M / BLOCK_M) * (N / BLOCK_N)
    For fixed M=32, BLOCK_M=32, N=65536, BLOCK_N=256: num_workgroups = 256
    All K values produce the same grid, so we cannot distinguish K from grid_x alone.
    """
    num_wgs = grid_x // wg_size
    return num_wgs


def compute_bandwidth(k, avg_ns):
    """Compute bandwidth in GB/s from problem size and average kernel time."""
    total_bytes = (M * k + k * N + M * N) * ELEM_BYTES
    return total_bytes / avg_ns  # bytes/ns = GB/s


def run_rocprof(k_values, rotating_buffer_size):
    """Run rocprofv3 with bench.py for each K value and return results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results = []

    for k in k_values:
        trace_dir = os.path.join(script_dir, f"rocprof_run_k{k}")
        if os.path.isdir(trace_dir):
            shutil.rmtree(trace_dir)

        cmd = [
            "rocprofv3",
            "--kernel-trace",
            "--kernel-include-regex",
            KERNEL_NAME,
            "-d",
            trace_dir,
            "--",
            sys.executable,
            os.path.join(script_dir, "bench.py"),
            "--K",
            str(k),
            "--rocprof",
            "--rotating-buffer-size",
            str(rotating_buffer_size),
        ]

        print(f"Running: K={k}")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            for line in (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-5:]:
                print(f"    {line}")
            results.append((k, None, None, None))
            continue

        csv_path = find_kernel_trace_csv(trace_dir)
        if csv_path is None:
            print(f"  No kernel_trace.csv found in {trace_dir}")
            results.append((k, None, None, None))
            continue

        groups = parse_kernel_times(csv_path, KERNEL_NAME)
        if not groups:
            print(f"  No kernel dispatches found for '{KERNEL_NAME}'")
            results.append((k, None, None, None))
            continue

        # All K values produce the same grid size, so there should be one group
        for grid_x, durations in groups.items():
            if not durations:
                continue
            avg_ns = sum(durations) / len(durations)
            min_ns = min(durations)
            max_ns = max(durations)
            bw_avg = compute_bandwidth(k, avg_ns)
            bw_max = compute_bandwidth(k, min_ns)
            bw_min = compute_bandwidth(k, max_ns)
            results.append((k, bw_avg, bw_min, bw_max))
            print(
                f"  K={k}: {len(durations)} dispatches, "
                f"avg={avg_ns / 1e3:.2f} us, "
                f"BW={bw_avg:.0f} GB/s"
            )

        # Clean up trace directory
        shutil.rmtree(trace_dir, ignore_errors=True)

    return results


def print_results(results):
    """Print results as a markdown table."""
    print(f"\nMemory-bound GEMM: M={M}, N={N}, fp16")
    print(f"{'K':>6} | {'BW avg (GB/s)':>14} | {'BW min (GB/s)':>14} | {'BW max (GB/s)':>14}")
    print(f"{'------':>6}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 14}")
    for k, bw_avg, bw_min, bw_max in results:
        if bw_avg is None:
            print(f"{k:>6} | {'FAIL':>14} | {'FAIL':>14} | {'FAIL':>14}")
        else:
            print(f"{k:>6} | {bw_avg:>14.0f} | {bw_min:>14.0f} | {bw_max:>14.0f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run memory-bound GEMM benchmark with rocprofv3")
    parser.add_argument(
        "--K",
        type=int,
        nargs="+",
        default=None,
        help=f"K value(s) to benchmark (default: {DEFAULT_K_VALUES})",
    )
    parser.add_argument(
        "--rotating-buffer-size",
        type=int,
        default=512,
        help="Rotating buffer size in MB (default: 512)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    k_values = args.K if args.K else DEFAULT_K_VALUES

    results = run_rocprof(k_values, args.rotating_buffer_size)
    print_results(results)


if __name__ == "__main__":
    main()
