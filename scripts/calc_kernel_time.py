#!/usr/bin/env python3
"""Calculate the average elapsed time of a given kernel from a rocprof kernel trace CSV."""

import argparse
import csv
import sys


def calc_avg_kernel_time(csv_path, kernel_name, unit="us"):
    """Read a kernel trace CSV, filter by kernel_name (substring match), and return average elapsed time."""
    scale = {"ns": 1, "us": 1e3, "ms": 1e6}
    if unit not in scale:
        raise ValueError(f"Unknown unit '{unit}', choose from {list(scale.keys())}")

    durations = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_name in row["Kernel_Name"]:
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                durations.append(end - start)

    if not durations:
        print(f"No rows matched kernel name '{kernel_name}' in {csv_path}")
        sys.exit(1)

    avg = sum(durations) / len(durations)
    mn = min(durations)
    mx = max(durations)

    print(f"Kernel : {kernel_name}")
    print(f"File   : {csv_path}")
    print(f"Matches: {len(durations)}")
    print(f"Avg    : {avg / scale[unit]:.2f} {unit}")
    print(f"Min    : {mn / scale[unit]:.2f} {unit}")
    print(f"Max    : {mx / scale[unit]:.2f} {unit}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average kernel time from a rocprof kernel trace CSV."
    )
    parser.add_argument("csv_file", help="Path to the kernel trace CSV file")
    parser.add_argument("kernel_name", help="Kernel name (substring match) to filter rows")
    parser.add_argument(
        "--unit",
        choices=["ns", "us", "ms"],
        default="us",
        help="Time unit for output (default: us)",
    )
    args = parser.parse_args()
    calc_avg_kernel_time(args.csv_file, args.kernel_name, args.unit)


if __name__ == "__main__":
    main()
