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

"""Pick an `att_target_cu` that actually exists on the GPU you are about to trace.

A shader array exposes 9 CU slots but enables only 8 -- one is harvested for yield, and
*which* index differs per die (on this box the union of harvested indices across 8 dies is
{0,1,2,3,6,7,8}, so only 4 and 5 are safe everywhere). rocprofv3 takes a single
`att_target_cu` and **does not validate it**: aiming at a harvested CU gives exit 0, a few
KB of `.att`, no `se*_wv*.json` wave files, and `process_json.py` then dies with
`'NoneType' object is not iterable`. There is no "any CU" mode in the tool (`-1` aborts,
an out-of-range index is silently accepted), so discover a valid index instead.

This runs `att_cu_census.cpp` -- every workgroup reports the (XCC, SE, CU) it ran on, and
slots that never appear are the harvested ones -- then picks an index present in *every*
array the requested shader-engine mask selects. Results are cached per GPU (keyed by PCI
bus id), so it costs one sub-second launch per machine.

    # emit a ready-to-use config from a template
    python scripts/att_pick_cu.py --template kernels/attention/att_attn_se0.json \
                                  --out /tmp/att.json
    rocprofv3 --att -i /tmp/att.json -d /tmp/trace -- python bench.py --rocprof ...

    # or just ask which CU to use
    python scripts/att_pick_cu.py --se-mask 0x1 --print-cu
    python scripts/att_pick_cu.py --report          # full per-array census

Honours `HIP_VISIBLE_DEVICES`, so it inspects the same device the trace will run on.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS_SRC = os.path.join(HERE, "att_cu_census.cpp")
CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "gfx950-att-cu"
)
CU_SLOTS = 16  # CU_ID field is 4 bits


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--template", help="ATT config JSON to rewrite (att_target_cu is replaced)")
    p.add_argument("--out", help="where to write the rewritten config (default: stdout)")
    p.add_argument(
        "--se-mask",
        default=None,
        help="shader-engine bitmask the CU must be valid for, e.g. 0x1. Defaults to the "
        "template's att_shader_engine_mask, or every array if neither is given",
    )
    p.add_argument("--print-cu", action="store_true", help="print just the chosen CU index")
    p.add_argument("--report", action="store_true", help="print the per-array census")
    p.add_argument("--blocks", type=int, default=32768, help="census launch size")
    p.add_argument("--refresh", action="store_true", help="ignore the cached census")
    p.add_argument(
        "--hipcc",
        default=os.environ.get("HIPCC", "/opt/rocm/bin/hipcc"),
        help="hipcc used to build the census helper",
    )
    p.add_argument(
        "--arch", default=None, help="--offload-arch for the census (default: autodetect)"
    )
    return p.parse_args()


def gpu_key():
    """Identify the device the trace will use: PCI bus id, not the HIP index.

    HIP_VISIBLE_DEVICES renumbers devices, so the index alone would alias different dies
    across invocations -- and this whole problem is per-die.
    """
    try:
        import torch  # noqa: PLC0415  (optional: only for a stable cache key)

        props = torch.cuda.get_device_properties(0)
        return f"pci{props.pci_bus_id:02x}", props.gcnArchName.split(":")[0]
    except Exception:
        vis = os.environ.get("HIP_VISIBLE_DEVICES", "all")
        return f"vis{vis}", None


def build_census(hipcc, arch):
    os.makedirs(CACHE_DIR, exist_ok=True)
    binary = os.path.join(CACHE_DIR, f"att_cu_census.{arch or 'native'}")
    if os.path.exists(binary) and os.path.getmtime(binary) >= os.path.getmtime(CENSUS_SRC):
        return binary
    if not shutil.which(hipcc) and not os.path.exists(hipcc):
        sys.exit(f"att_pick_cu: {hipcc} not found; pass --hipcc or set HIPCC")
    cmd = [hipcc, "-O2", "-w", "-o", binary, CENSUS_SRC]
    if arch:
        cmd.insert(1, f"--offload-arch={arch}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"att_pick_cu: building the census failed:\n{proc.stdout}\n{proc.stderr}")
    return binary


def run_census(args):
    """Return {(xcc, se): cu_mask} for the current device, cached per die."""
    key, arch = gpu_key()
    cache = os.path.join(CACHE_DIR, f"census.{key}.json")
    if os.path.exists(cache) and not args.refresh:
        with open(cache) as f:
            return {tuple(int(k) for k in kv.split(",")): v for kv, v in json.load(f).items()}

    binary = build_census(args.hipcc, args.arch or arch)
    proc = subprocess.run([binary, str(args.blocks)], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"att_pick_cu: census run failed:\n{proc.stdout}\n{proc.stderr}")
    arrays = {}
    for m in re.finditer(r"xcc (\d+) se (\d+) cu_mask (0x[0-9A-Fa-f]+)", proc.stdout):
        arrays[(int(m.group(1)), int(m.group(2)))] = int(m.group(3), 16)
    if not arrays:
        sys.exit(f"att_pick_cu: census produced no data:\n{proc.stdout}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({f"{x},{s}": v for (x, s), v in arrays.items()}, f)
    return arrays


def pick_cu(arrays, se_mask):
    """Lowest CU index enabled in EVERY array the SE mask selects.

    Every selected array, not just one: the config names an SE mask rather than a single
    (XCC, SE), and a CU can be harvested in one XCC's array while present in another's.
    """
    selected = {k: v for k, v in arrays.items() if se_mask is None or (se_mask >> k[1]) & 1}
    if not selected:
        sys.exit(f"att_pick_cu: no arrays matched se_mask {se_mask:#x}")
    common = None
    for m in selected.values():
        common = m if common is None else (common & m)
    for cu in range(CU_SLOTS):
        if (common >> cu) & 1:
            return cu, selected
    sys.exit("att_pick_cu: no CU is enabled in every selected array")


def main():
    args = parse_args()
    template = None
    if args.template:
        with open(args.template) as f:
            template = json.load(f)

    se_mask = args.se_mask
    if se_mask is None and template:
        se_mask = template.get("jobs", [{}])[0].get("att_shader_engine_mask")
    se_mask = int(se_mask, 0) if isinstance(se_mask, str) else se_mask

    arrays = run_census(args)
    cu, selected = pick_cu(arrays, se_mask)

    if args.report:
        print(
            f"  census: {len(arrays)} arrays, se_mask={se_mask if se_mask is None else hex(se_mask)}"
        )
        for (x, s), m in sorted(arrays.items()):
            cus = " ".join(str(c) for c in range(CU_SLOTS) if (m >> c) & 1)
            gone = [c for c in range(9) if not (m >> c) & 1]
            tag = "  <- selected" if (x, s) in selected else ""
            print(f"    xcc{x} se{s}: enabled [{cus}]  harvested {gone}{tag}")
    if args.print_cu:
        print(cu)
        return
    if not template:
        if not args.report:
            print(f"att_target_cu = {cu}")
        return

    template.setdefault("jobs", [{}])[0]["att_target_cu"] = cu
    text = json.dumps(template, indent=4) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"  wrote {args.out} with att_target_cu = {cu}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
