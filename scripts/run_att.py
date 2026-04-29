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


import argparse
import glob
import os
import subprocess
import sys


def get_git_root():
    """
    Return the absolute path to the git repository root.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Failed to determine git repository root. " "Are you running this inside a git repo?"
        )


def run_rocprof_att(att_output_dir, python_cmd):
    env = os.environ.copy()
    env["ROCPROF_ATT_LIBRARY_PATH"] = "/var/lib/jenkins/att-decoder-v3-3.0.0-Linux/opt/rocm/lib/"

    cmd = [
        "rocprofv3",
        "--att",
        "-i",
        "att_matmul.json",
        "-d",
        att_output_dir,
        "--",
    ] + python_cmd

    print("Running rocprofv3 command:")
    print(" ".join(cmd))

    subprocess.run(cmd, env=env, check=True)


def find_ui_output_dir(att_output_dir):
    pattern = os.path.join(att_output_dir, "ui_*")
    matches = glob.glob(pattern)

    if not matches:
        raise RuntimeError(f"No ui_* directory found in {att_output_dir}")

    if len(matches) > 1:
        print("Warning: multiple ui_* directories found, using the first one:")
        for m in matches:
            print(f"  {m}")

    return matches[0]


def run_process_json(ui_output_dir):
    git_root = get_git_root()
    script_path = os.path.join(git_root, "scripts", "process_json.py")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"process_json.py not found at {script_path}")

    cmd = [
        sys.executable,
        script_path,
        ui_output_dir,
    ]

    print("Running process_json.py:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run rocprofv3 ATT and post-process traces")

    parser.add_argument(
        "--att-output", required=True, help="Output directory for rocprofv3 ATT (passed to -d)"
    )

    parser.add_argument(
        "python_cmd",
        nargs=argparse.REMAINDER,
        help="Python script and arguments to profile (after --)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.python_cmd:
        raise ValueError("You must provide a python command to run after --")

    run_rocprof_att(args.att_output, args.python_cmd)

    ui_dir = find_ui_output_dir(args.att_output)
    print(f"Found UI output directory: {ui_dir}")

    run_process_json(ui_dir)


if __name__ == "__main__":
    main()
