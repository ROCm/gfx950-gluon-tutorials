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


# rocprofiler-sdk 1.4 and later create the thread-trace queue directly through KFD and need the
# driver to report the queue's context-save sizes (KFD ABI 1.20). Older amdgpu drivers do not, and
# the queue is rejected with "AMDKFD_IOC_CREATE_QUEUE failed: Invalid argument". The SDK's HSA path
# (what ROCm 7.2 used) works there; ROCPROFILER_SQTT_FORCE_HSA=1 selects it.
KFD_QUEUE_ERROR = "AMDKFD_IOC_CREATE_QUEUE failed"
FORCE_HSA = "ROCPROFILER_SQTT_FORCE_HSA"


def run_rocprof_att(att_output_dir, python_cmd):
    env = os.environ.copy()
    # Allow user override via the standard env var. If not set, fall back to a
    # common install location for the rocprof-trace-decoder package (ROCm 7.0+
    # layout). On ROCm 7.0+, scripts/install_att_decoder.sh drops the .so at
    # /opt/rocm/lib/librocprof-trace-decoder.so by default.
    env.setdefault("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/")

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

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0 and KFD_QUEUE_ERROR in output and FORCE_HSA not in env:
        print(
            f"rocprofv3 could not create its KFD thread-trace queue (driver older than KFD ABI "
            f"1.20); retrying with {FORCE_HSA}=1 (the HSA path)"
        )
        env[FORCE_HSA] = "1"
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
    print(output)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=output)


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
