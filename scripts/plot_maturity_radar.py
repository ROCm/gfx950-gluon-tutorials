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
Draw the "optimization maturity" radar for each a16w16 kernel version.

Six dimensions, each scored 1 (naive / lowest) .. 5 (optimal):
  1. Codegen            — optimal count & choice of instructions
  2. Global latency     — global loads hidden (no s_waitcnt vmcnt stalls)
  3. LDS latency        — LDS reads hidden (no s_waitcnt lgkmcnt stalls)
  4. LDS bank conflict  — ds_read/ds_write are conflict-free
  5. Scheduling         — MFMA optimally interleaved with non-MFMA on the SIMD
  6. L2 locality        — workgroups on nearby/same tiles land on the same XCD

Each version's radar is written to
kernels/gemm/intra_wave/a16w16/<version>/images/maturity_radar.png. Edit SCORES
below as versions improve; run `python scripts/plot_maturity_radar.py`.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
A16W16 = os.path.join(REPO, "kernels", "gemm", "intra_wave", "a16w16")

DIMS = [
    "Codegen",
    "Global\nlatency",
    "LDS\nlatency",
    "LDS bank\nconflict",
    "Scheduling",
    "L2 locality",
]
LO, HI = 1, 5  # score scale: 1 = naive/lowest, 5 = optimal

# version dir -> six scores, in DIMS order:
#   [codegen, global-latency, lds-latency, lds-bank-conflict, scheduling, L2-locality]
# Roughly cumulative; a rough author judgment (e.g. v5's codegen dips before v6's
# unroll cleans it up). Edit as the rubric firms up.
SCORES = {
    "v0_naive": [1, 1, 1, 1, 1, 1],  # naive baseline
    "v1_buffer_load": [2, 1, 1, 1, 1, 1],  # buffer_load: hardware OOB, branch elimination
    "v2_async_copy": [3, 1, 1, 1, 1, 1],  # direct-to-LDS async copy (no register staging)
    "v3_lds": [3, 1, 1, 5, 1, 1],  # LDS layout: conflict-free ds_read/write
    "v4_global_prefetch": [3, 3, 1, 5, 1, 1],  # 2-stage pipeline: hide global latency
    "v5_local_prefetch": [2.5, 3, 5, 5, 4, 1],  # 3-stage + LLIR: LDS latency fully hidden
    "v6_loop_unroll": [4, 3, 5, 5, 4, 1],  # unroll: tighter codegen (keeps v5's LDS win)
    "v7_sliceN": [5, 4, 5, 5, 5, 1],  # N-slice: frees registers -> codegen/scheduling optimal
    "v8_sliceMN": [5, 5, 5, 5, 5, 1],  # M+N slice: buffer-load stall gone; loop optimal
    "v9_beyond_hotloop": [5, 5, 5, 5, 5, 5],  # XCD-aware PID remap: L2 locality
}


def radar(version, scores, out_png):
    n = len(DIMS)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([ang, ang[:1]])
    vals = np.array(scores + scores[:1], dtype=float)

    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=150, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)  # first axis at top
    ax.set_theta_direction(-1)  # clockwise

    # "optimal" reference envelope (all HI), dashed grey
    ax.plot(closed, np.full(n + 1, HI), color="#9aa4b0", lw=1.2, ls=(0, (4, 3)))

    # fill colour by mean maturity (red = immature -> green = mature); darken the
    # map because RdYlGn's mid-range is a pale yellow that washes out.
    mean = float(np.mean(scores))
    rgb = np.asarray(cm.RdYlGn((mean - LO) / (HI - LO)))[:3] * 0.78
    color = (*rgb, 1.0)
    ax.plot(closed, vals, color=color, lw=2.4)
    ax.fill(closed, vals, color=color, alpha=0.22)  # translucent so the grid shows through
    ax.plot(ang, scores, "o", color=color, ms=5)

    # A rough visual (polygon size vs the dashed optimal envelope) — no numeric
    # scale, per-vertex values, or legend.
    ax.set_xticks(ang)
    ax.set_xticklabels(DIMS, fontsize=13)
    ax.set_ylim(0, HI + 0.2)
    ax.set_yticks(range(LO, HI + 1))
    ax.set_yticklabels([])
    ax.tick_params(pad=9)
    ax.grid(color="#dcdcdc", lw=0.8)
    ax.spines["polar"].set_color("#cccccc")

    ax.set_title(
        f"Optimization maturity — {version}",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    # Fixed axes box (labels live in the margins) + a fixed canvas size — so every
    # version's PNG has identical dimensions and the before/after pair matches.
    fig.subplots_adjust(left=0.16, right=0.84, top=0.84, bottom=0.16)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(out_png, REPO))


def main():
    for version, scores in SCORES.items():
        assert len(scores) == len(DIMS), f"{version}: need {len(DIMS)} scores"
        radar(version, scores, os.path.join(A16W16, version, "images", "maturity_radar.png"))


if __name__ == "__main__":
    main()
