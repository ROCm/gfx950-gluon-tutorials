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
GEMM = os.path.join(REPO, "kernels", "gemm")

DIMS = [
    "Codegen",
    "Global\nlatency",
    "LDS\nlatency",
    "LDS bank\nconflict",
    "Scheduling",
    "L2 locality",
]
LO, HI = 1, 5  # score scale: 1 = naive/lowest, 5 = optimal

# path under kernels/gemm/ -> six scores, in DIMS order:
#   [codegen, global-latency, lds-latency, lds-bank-conflict, scheduling, L2-locality]
# A kernel may carry a 7th score for "Freq" (clock frequency) when throttling is
# part of its story — the inter_wave/a4w4 series does, to show v2 trading clock
# for occupancy. Roughly cumulative within a series; rough author judgment.
SCORES = {
    # intra_wave/a16w16 — the 4-wave FP16 v0->v9 optimization journey
    "intra_wave/a16w16/v0_naive": [1, 1, 1, 1, 1, 1],  # naive baseline
    "intra_wave/a16w16/v1_buffer_load": [2, 1, 1, 1, 1, 1],  # hardware OOB, branch elimination
    "intra_wave/a16w16/v2_async_copy": [3, 1, 1, 1, 1, 1],  # direct-to-LDS async copy
    "intra_wave/a16w16/v3_lds": [3, 1, 1, 5, 1, 1],  # LDS layout: conflict-free ds_read/write
    "intra_wave/a16w16/v4_global_prefetch": [3, 3, 1, 5, 1, 1],  # 2-stage: hide global latency
    "intra_wave/a16w16/v5_local_prefetch": [2.5, 3, 5, 5, 4, 1],  # 3-stage: LDS latency hidden
    "intra_wave/a16w16/v6_loop_unroll": [4, 3, 5, 5, 4, 1],  # unroll: tighter codegen
    "intra_wave/a16w16/v7_sliceN": [5, 4, 5, 5, 5, 1],  # N-slice: codegen/scheduling optimal
    "intra_wave/a16w16/v8_sliceMN": [5, 5, 5, 5, 5, 1],  # M+N slice: loop optimal
    "intra_wave/a16w16/v9_beyond_hotloop": [5, 5, 5, 5, 5, 5],  # XCD-aware PID remap: L2 locality
    # intra_wave a8w8 / a4w4 — the mature 4-wave design at BF8 / MXFP4
    "intra_wave/a8w8": [5, 5, 5, 5, 5, 5],  # a16w16 v8+v9 design at BF8 (~99.5% MFMA)
    "intra_wave/a4w4/v0_sliceN": [4, 5, 3, 4.5, 4, 5],  # +scale pipeline via LDS round-trip
    "intra_wave/a4w4/v1_sliceMN": [5, 5, 5, 5, 4.5, 5],  # M+N slice + direct-to-LDS scales (~94%)
    # inter_wave — the 8-wave warp-pipeline route
    "inter_wave/a16w16": [5, 5, 5, 5, 5, 5],  # ping-pong, ~99.8% loop MFMA
    "inter_wave/a8w8": [5, 5, 5, 5, 5, 5],  # ping-pong BF8, ~99.7%
    # a4w4 8-wave series carries the 7th "Freq" axis (clock frequency):
    "inter_wave/a4w4/v0_sliceMN": [2, 4, 3, 4, 3, 5, 5],  # byte-shuffle (~57%); full clock
    "inter_wave/a4w4/v1_combineBsc": [5, 5, 3, 4.5, 4, 5, 5],  # combined B-scale (~80%)
    # v2: 32x32x64 lifts scheduling/occupancy (~98%) but the wider MFMA throttles
    # the clock, so freq drops and end-to-end TFLOPS lands below v1. Bank conflict
    # matches v1 (4.5) — the MXFP4 scale-read conflict is inherent to both.
    "inter_wave/a4w4/v2_mfma32x32x64": [5, 5, 5, 4.5, 5, 5, 3],  # +sched, throttled clock
}


def title_of(key):
    """Radar title for a SCORES key: single-kernel dirs get a 4/8-wave tag;
    versioned dirs use just the version name (their README gives the context)."""
    parts = key.split("/")
    kernel, name = parts[1], parts[-1]
    if name == kernel:  # single-kernel dir (a8w8, a16w16, ...)
        wave = "8-wave" if parts[0] == "inter_wave" else "4-wave"
        return f"{name} ({wave})"
    return name


def radar(title, scores, out_png):
    # Most kernels use the 6 DIMS; a kernel may carry a 7th "Freq" (clock
    # frequency) score when clock throttling is part of its story (e.g. the
    # inter_wave/a4w4 series, where wider MFMA trades occupancy for clock).
    dims = DIMS + ["Freq"] if len(scores) > len(DIMS) else DIMS
    n = len(dims)
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
    # fuller polygons get a more transparent fill so the grid (spider web) is never
    # buried — the near-full v8/v9 end up lighter than the mid versions.
    n = len(scores)
    cover = sum(scores[i] * scores[(i + 1) % n] for i in range(n)) / (n * HI * HI)
    ax.plot(closed, vals, color=color, lw=2.4)
    ax.fill(closed, vals, color=color, alpha=0.13 if cover > 0.70 else 0.22)
    ax.plot(ang, scores, "o", color=color, ms=5)

    # A rough visual (polygon size vs the dashed optimal envelope) — no numeric
    # scale, per-vertex values, or legend.
    ax.set_xticks(ang)
    ax.set_xticklabels(dims, fontsize=13)
    ax.set_ylim(0, HI + 0.2)
    ax.set_yticks(range(LO, HI + 1))
    ax.set_yticklabels([])
    ax.tick_params(pad=9)
    ax.grid(color="#dcdcdc", lw=0.8)
    ax.spines["polar"].set_color("#cccccc")

    ax.set_title(
        f"Optimization maturity — {title}",
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
    for key, scores in SCORES.items():
        assert len(scores) in (len(DIMS), len(DIMS) + 1), f"{key}: need 6 or 7 scores"
        out = os.path.join(GEMM, key, "images", "maturity_radar.png")
        radar(title_of(key), scores, out)


if __name__ == "__main__":
    main()
