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
Regenerate kernels/gemm/images/perf_summary.png — the "Performance Summary"
bar chart in kernels/gemm/README.md.

Grouped bars of peak TFLOPS (4-wave vs 8-wave) at each precision's headline
shape, with the per-SIMD loop MFMA efficiency printed in red inside each bar.
The numbers mirror the README (MI355X, gfx950, Triton gfx950-tutorial-v1.0,
rocprof cold-rotating); edit the `rows` table below when they change.

    python scripts/plot_perf_summary.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

C4, C8 = "#4C72B0", "#DD8452"  # 4-wave, 8-wave bar colors
CMFMA = "#D62728"  # MFMA-efficiency labels (red)

# precision, K, 4-wave (TFLOPS, MFMA%), 8-wave (TFLOPS, MFMA%)
rows = [
    ("FP16", 8192, 1571, 97.89, 1480, 99.84),
    ("BF16", 8192, 1571, 97.89, 1540, 99.84),
    ("BF8", 16384, 3381, 99.24, 3142, 99.86),
    ("MXFP4", 32768, 5820, 93.75, 4948, 75.08),
]


def main():
    labels = [f"{p}\nK={k}" for (p, k, *_) in rows]
    four = [r[2] for r in rows]
    four_mfma = [r[3] for r in rows]
    eight = [r[4] for r in rows]
    eight_mfma = [r[5] for r in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    b4 = ax.bar(x - w / 2, four, w, color=C4)
    b8 = ax.bar(x + w / 2, eight, w, color=C8)

    def annotate(bars, mfma):
        for rect, m in zip(bars, mfma):
            h = rect.get_height()
            cx = rect.get_x() + rect.get_width() / 2
            # TFLOPS above the bar
            ax.text(
                cx, h + 55, f"{int(h)}", ha="center", va="bottom", fontsize=9.5, fontweight="bold"
            )
            # MFMA efficiency inside the bar, near the top (red on a white pill
            # so it stays legible on both the blue and orange fills)
            ax.text(
                cx,
                h - 90,
                f"{m:.1f}%",
                ha="center",
                va="top",
                fontsize=8.2,
                fontweight="bold",
                color=CMFMA,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
            )

    annotate(b4, four_mfma)
    annotate(b8, eight_mfma)

    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(
        "GEMM peak throughput: 4-wave vs 8-wave  (MI355X · gfx950 · 4096×4096)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 5900)
    legend = [
        Patch(facecolor=C4, label="4-wave  (LLIR + force-agpr + amdgcnas)"),
        Patch(facecolor=C8, label="8-wave  (warp-pipeline, no AGPRs)"),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color=CMFMA,
            markersize=8,
            label="red % = per-SIMD loop MFMA efficiency",
        ),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=9.5)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(
        os.path.join(here, "..", "kernels", "gemm", "images", "perf_summary.png")
    )
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
