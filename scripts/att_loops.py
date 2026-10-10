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
"""MFMA efficiency of every loop in a decoded ATT trace, however often it is entered.

process_json.py times one loop from its first entry to the first epilogue instruction,
which assumes the loop runs once per wave. A persistent kernel enters its inner loops
once per work item (fmha_v5: 32 q-blocks per workgroup, two inner loops for causal), so
this script attributes time instruction by instruction instead: every issue-to-next-issue
interval of an instruction inside a loop body counts toward that loop. Per loop it reports

  iterations      back-edge executions per wave
  cyc/iter        loop cycles per iteration (one iteration of fmha's 2x-unrolled loop is
                  two K/V tiles)
  share           the loop's fraction of the wave's lifetime
  MFMA eff        MFMA issue cycles / loop cycles, per wave and x waves-per-SIMD (the
                  per-SIMD figure the attention README quotes; see docs/mfma_efficiency.md)

Loops are found from backward s_cbranch/s_branch targets in code.json.

    python scripts/att_loops.py <ui_output_dir> [--waves-per-simd 2]
"""

import argparse
import json
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_json import get_mfma_cycles  # noqa: E402


def find_loops(code):
    """[(first_index, backedge_index)] of every loop with an executed backward branch."""
    by_addr = {c[5]: c[2] for c in code}
    loops = []
    for c in code:
        name, idx, addr, hit = c[0], c[2], c[5], c[6]
        if not (name.startswith("s_cbranch") or name.startswith("s_branch")) or hit == 0:
            continue
        parts = name.split()
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        imm = int(parts[1])
        if imm <= 32767:
            continue  # forward
        target = addr + 4 + 4 * (imm - 65536)
        if target in by_addr:
            loops.append((by_addr[target], idx))
    # one entry per body (a loop may have several back-edges; keep the outermost range)
    loops.sort(key=lambda r: (r[0], -r[1]))
    out = []
    for lo, hi in loops:
        if out and out[-1][0] == lo:
            continue
        out.append((lo, hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--waves-per-simd", type=int, default=2)
    a = ap.parse_args()
    with open(os.path.join(a.folder, "code.json")) as f:
        data = json.load(f)
    code = data["code"] if isinstance(data, dict) else data
    mfma_cyc = {c[2]: get_mfma_cycles(c[0]) for c in code}
    loops = find_loops(code)
    if not loops:
        sys.exit("no loops found")
    files = sorted(glob(os.path.join(a.folder, "se*_sm*_sl*_wv*.json")))
    acc = {lh: {"t": 0, "m": 0, "it": 0} for lh in loops}
    total = 0
    nwaves = 0
    for path in files:
        with open(path) as f:
            ins = json.load(f)["wave"]["instructions"]
        if len(ins) < 2:
            continue
        nwaves += 1
        total += ins[-1][0] - ins[0][0]
        for (c0, *_, i0), (c1, *_) in zip(ins, ins[1:]):
            for lo, hi in loops:
                if lo <= i0 <= hi:
                    s = acc[(lo, hi)]
                    s["t"] += c1 - c0
                    s["m"] += mfma_cyc.get(i0, 0)
                    if i0 == hi:
                        s["it"] += 1
    nested = {lh: any(o != lh and o[0] <= lh[0] and lh[1] <= o[1] for o in loops) for lh in loops}
    print(f"{nwaves} waves, {total / max(nwaves, 1):.0f} cycles per wave")
    print(
        f"{'loop (code idx)':<18}{'iters/wave':>11}{'cyc/iter':>10}{'share':>8}"
        f"{'MFMA eff/wave':>15}{'per SIMD':>10}"
    )
    for lh in loops:
        s = acc[lh]
        if s["t"] == 0:
            continue
        it = s["it"] / nwaves
        eff = s["m"] / s["t"]
        kind = (
            "inner"
            if not any(o != lh and lh[0] <= o[0] and o[1] <= lh[1] for o in loops)
            else "outer"
        )
        label = f"{lh[0]}-{lh[1]} {kind}{'' if nested[lh] else ' top'}"
        print(
            f"{label:<18}{it:>11.1f}{s['t'] / max(s['it'], 1):>10.0f}{s['t'] / total * 100:>7.1f}%"
            f"{eff * 100:>14.2f}%{eff * a.waves_per_simd * 100:>9.2f}%"
        )


if __name__ == "__main__":
    main()
