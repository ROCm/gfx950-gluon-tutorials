#!/usr/bin/env python3
##############################################################################
# MIT License
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
##############################################################################
"""Measure LDS bank conflicts with rocprofv3 hardware counters, and sweep the
`SwizzledSharedLayout` parameters against them.

Counters (gfx942):
    SQ_LDS_BANK_CONFLICT   cycles LDS is stalled replaying a conflicted access
    SQ_LDS_IDX_ACTIVE      cycles LDS is servicing an indexed access
    SQ_INSTS_LDS           LDS instructions issued

The derived figure rocprof calls `LdsBankConflict` is

    SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT)

i.e. replay cycles per useful cycle. 0 means conflict-free; 1 means every access
costs one extra pass.

    python lds_conflict.py --kernel inter_wave                  # measure as-is
    python lds_conflict.py --kernel inter_wave --sweep          # sweep swizzles

The sweep re-execs the driver once per (vec, per_phase, max_phase) with
`GFX942_SWZ_*` set, clearing the Triton cache each time because the swizzle is
baked into the kernel at compile time and is not part of the cache key.
"""

import argparse
import csv
import glob
import itertools
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COUNTERS = ["SQ_LDS_BANK_CONFLICT", "SQ_LDS_IDX_ACTIVE", "SQ_INSTS_LDS"]
KERNEL_REGEX = {
    "intra_wave": "a16w16_intra_wave_gfx942",
    "inter_wave": "a16w16_inter_wave_gfx942",
}


def collect(kernel, gpu, M, N, K, dtype, iters, out, swz=None):
    """One rocprofv3 --pmc run; returns {counter: value} summed over dispatches."""
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    cfg = os.path.join(out, "pmc.yaml")
    with open(cfg, "w") as f:
        f.write("jobs:\n  - pmc:\n")
        for c in COUNTERS:
            f.write(f"      - {c}\n")

    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env["TRITON_CACHE_DIR"] = os.path.join(out, "tcache")
    if swz:
        env["GFX942_SWZ_VEC"], env["GFX942_SWZ_PER_PHASE"], env["GFX942_SWZ_MAX_PHASE"] = (
            str(v) for v in swz
        )

    cmd = [
        "rocprofv3",
        "-i",
        cfg,
        "--output-format",
        "csv",
        "-d",
        out,
        "--",
        sys.executable,
        os.path.join(HERE, "bench_prepared.py"),
        "--_driver",
        kernel,
        "--M",
        str(M),
        "--N",
        str(N),
        "--K",
        str(K),
        "--dtype",
        dtype,
        "--iters",
        str(iters),
        "--warmup",
        "5",
    ]
    # rocprofv3 exits non-zero even on success here, so judge by the artifact and
    # by the driver's own FATAL guard rather than by the return code.
    r = subprocess.run(cmd, env=env, cwd=out, capture_output=True, text=True)
    combined = r.stdout + r.stderr
    if "FATAL" in combined or "Traceback" in combined:
        tail = combined.strip().splitlines()[-4:]
        raise RuntimeError("driver failed: " + " | ".join(tail))

    vals = {c: 0.0 for c in COUNTERS}
    files = glob.glob(os.path.join(out, "**", "*counter_collection.csv"), recursive=True)
    if not files:
        tail = combined.strip().splitlines()[-3:]
        raise RuntimeError(f"no counter csv under {out}: " + " | ".join(tail))
    with open(files[0], newline="") as fh:
        for row in csv.DictReader(fh):
            if KERNEL_REGEX[kernel] not in row.get("Kernel_Name", ""):
                continue
            name = row["Counter_Name"]
            if name in vals:
                vals[name] += float(row["Counter_Value"])
    return vals


def report(tag, v):
    bc, idx, insts = v["SQ_LDS_BANK_CONFLICT"], v["SQ_LDS_IDX_ACTIVE"], v["SQ_INSTS_LDS"]
    useful = idx - bc
    ratio = bc / useful if useful > 0 else float("nan")
    print(
        f"  {tag:<22s} conflict={ratio:7.4f}  bank_conflict={bc:14,.0f} "
        f"idx_active={idx:14,.0f}  cyc/LDS_instr={idx / insts if insts else 0:6.2f}"
    )
    return ratio


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--kernel", choices=sorted(KERNEL_REGEX), default="inter_wave")
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4864)
    p.add_argument("--K", type=int, default=8256)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--out", default="/tmp/lds_conflict")
    a = p.parse_args()

    print(f"{a.kernel} {a.M}x{a.N}x{a.K} {a.dtype}  GPU{a.gpu}  {a.iters} dispatches")
    print("  conflict = SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT)\n")

    if not a.sweep:
        report(
            "as-committed",
            collect(a.kernel, a.gpu, a.M, a.N, a.K, a.dtype, a.iters, os.path.join(a.out, "base")),
        )
        return 0

    # vec must keep the 16 B (8 x fp16) dwordx4 access width usable; per_phase x
    # max_phase spans the swizzle period in rows.
    grid = [
        (v, pp, mp)
        for v, pp, mp in itertools.product([4, 8], [1, 2, 4], [4, 8, 16])
        if pp * mp <= 64
    ]
    results = {}
    for swz in grid:
        tag = f"vec={swz[0]} per={swz[1]} max={swz[2]}"
        try:
            v = collect(
                a.kernel,
                a.gpu,
                a.M,
                a.N,
                a.K,
                a.dtype,
                a.iters,
                os.path.join(a.out, "swz_%d_%d_%d" % swz),
                swz=swz,
            )
            results[swz] = report(tag, v)
        except Exception as e:
            print(f"  {tag:<22s} FAILED: {str(e)[:90]}")
    if results:
        best = min(results, key=results.get)
        print(
            f"\n  lowest conflict: vec={best[0]} per_phase={best[1]} max_phase={best[2]} "
            f"-> {results[best]:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
