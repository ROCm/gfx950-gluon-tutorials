#!/usr/bin/env python3
##############################################################################
# MIT License
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
##############################################################################
"""Collect an ATT (Advanced Thread Trace) for one gfx942 GEMM kernel.

Wraps `rocprofv3 --att` the way the gfx950 tutorial's `scripts/run_att.py`
does, and verifies that the decoder produced the `ui_*` directory that
ATTViewer consumes.  The decoder library must be present:

    bash scripts/install_att_decoder.sh      # from the repo root

    python run_att.py inter_wave --K 8256 --out att_inter_wave
    python run_att.py intra_wave --K 8256 --out att_intra_wave

Re-execs itself as `--_driver` under rocprofv3.  The child dispatches only the
kernel under test, a fixed number of times, so `kernel_iteration_range` selects
a known steady-state dispatch and no torch call or correctness check pollutes
the dispatch stream.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

KERNEL_REGEX = {
    "intra_wave": "a16w16_intra_wave_gfx942",
    "inter_wave": "a16w16_inter_wave_gfx942",
}


def write_config(path, kernel_regex, iteration, buffer_size, target_cu, se_mask):
    cfg = {
        "jobs": [
            {
                "kernel_include_regex": kernel_regex,
                "kernel_exclude_regex": "",
                "kernel_iteration_range": f"[{iteration}]",
                "advanced_thread_trace": True,
                "att_target_cu": target_cu,
                "att_shader_engine_mask": se_mask,
                "att_simd_select": "0xF",
                "att_buffer_size": buffer_size,
            }
        ]
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    return cfg


def _load(name):
    """Import the kernel module (both files are called `matmul_kernel`)."""
    import importlib.util

    # the kernels do `from common import get_pids`
    sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
    spec = importlib.util.spec_from_file_location(
        f"gfx942_{name}", os.path.join(ROOT, name, "matmul_kernel.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _driver_main(argv):
    p = argparse.ArgumentParser(prog="run_att.py --_driver")
    p.add_argument("kernel", choices=sorted(KERNEL_REGEX))
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4864)
    p.add_argument("--K", type=int, default=8256)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--iters", type=int, default=8)
    args = p.parse_args(argv)

    # the 4-wave kernel needs the force-agpr hint before it compiles
    os.environ.setdefault("TRITON_FORCE_MFMA_AGPR", "1")
    import torch
    import triton

    mod = _load(args.kernel)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    dev = triton.runtime.driver.active.get_active_torch_device()
    M, N, K = args.M, args.N, args.K

    a = torch.randn((M, K), device=dev, dtype=dt) * 0.1
    b_t = torch.randn((N, K), device=dev, dtype=dt) * 0.1
    c = torch.empty((M, N), device=dev, dtype=dt)

    print(
        f"[run_att:driver] {args.kernel} {mod.KERNEL_NAME} {M}x{N}x{K} {args.dtype} "
        f"x{args.iters} dispatches"
    )
    for _ in range(args.iters):
        mod.matmul_kernel_only(a, b_t, c)
    torch.cuda.synchronize()
    print("[run_att:driver] done")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_driver":
        return _driver_main(sys.argv[2:])

    p = argparse.ArgumentParser(description="rocprofv3 ATT capture for a gfx942 GEMM kernel")
    p.add_argument("kernel", choices=sorted(KERNEL_REGEX))
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4864)
    p.add_argument("--K", type=int, default=8256)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--out", required=True, help="output directory for the trace")
    p.add_argument(
        "--iteration", type=int, default=4, help="which matching dispatch to trace (1-based)"
    )
    p.add_argument("--iters", type=int, default=8, help="dispatches the driver issues")
    p.add_argument("--buffer-size", default="0x10000000", help="ATT buffer size (default 256 MB)")
    p.add_argument("--target-cu", type=int, default=0)
    p.add_argument("--se-mask", default="0xF", help="shader-engine bitmask")
    p.add_argument("--gpu", default="0", help="HIP_VISIBLE_DEVICES for the run")
    args = p.parse_args()

    decoder = "/opt/rocm/lib/librocprof-trace-decoder.so"
    if not os.path.exists(decoder):
        sys.exit(
            f"error: {decoder} missing. Run install_att_decoder.sh first — without the\n"
            f"decoder rocprofv3 emits raw .att blobs and no ui_* directory."
        )

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    cfg_path = os.path.join(out, "att_config.json")
    cfg = write_config(
        cfg_path,
        KERNEL_REGEX[args.kernel],
        args.iteration,
        args.buffer_size,
        args.target_cu,
        args.se_mask,
    )
    print(f"ATT config: {json.dumps(cfg['jobs'][0])}")

    env = os.environ.copy()
    env.setdefault("ROCPROF_ATT_LIBRARY_PATH", "/opt/rocm/lib/")
    env["HIP_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("TRITON_FORCE_MFMA_AGPR", "1")

    cmd = [
        "rocprofv3",
        "--att",
        "-i",
        cfg_path,
        "-d",
        out,
        "--",
        sys.executable,
        os.path.abspath(__file__),
        "--_driver",
        args.kernel,
        "--M",
        str(args.M),
        "--N",
        str(args.N),
        "--K",
        str(args.K),
        "--dtype",
        args.dtype,
        "--iters",
        str(args.iters),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True, cwd=out)

    ui_dirs = sorted(glob.glob(os.path.join(out, "**", "ui_*"), recursive=True))
    if not ui_dirs:
        sys.exit(
            f"error: no ui_* directory under {out} — the decoder did not run.\n"
            f"Check that ROCPROF_ATT_LIBRARY_PATH points at {decoder}."
        )

    print(
        f"\nATT trace for {args.kernel} ({KERNEL_REGEX[args.kernel]}) "
        f"{args.M}x{args.N}x{args.K} {args.dtype}"
    )
    for d in ui_dirs:
        files = os.listdir(d)
        size = sum(
            os.path.getsize(os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f))
        )
        print(f"  {d}  ({len(files)} files, {size / 1024 ** 2:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
