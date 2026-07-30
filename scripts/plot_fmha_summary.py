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
Regenerate kernels/attention/images/results.png — the summary bar chart at the
top of kernels/attention/README.md.

Same shape as scripts/plot_perf_summary.py (the GEMM chart in
kernels/gemm/README.md section 3): grouped TFLOPS bars with the per-SIMD in-loop
MFMA efficiency printed in red inside each bar. Two groups, one per kernel, with
the stock-LLVM build leftmost in each so the left-to-right reading is "what the
scheduler adds". ROCm/FlyDSL sits with fmha_v4 as the external reference.

The numbers mirror section 9 of the attention README (MI355X, gfx950, rocprofv3
kernel time, prepared launch); edit the `groups` table below when they change.

    python scripts/plot_fmha_summary.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

C_STOCK, C_TUNED, C_REF = "#DD8452", "#4C72B0", "#55A868"
CMFMA = "#D62728"  # MFMA-efficiency labels (red)

# Colour follows the series, not the bar position: stock is always orange, the
# tuned build always blue, the external reference always green.
#   group label, sub-label, [(series, TFLOPS, MFMA%), ...]
groups = [
    (
        "fmha_v3",
        "eager rescale",
        [("stock", 1139, 67.3), ("tuned", 1249, 86.3)],
    ),
    (
        "fmha_v4",
        "lazy rescale",
        [("stock", 1204, 69.1), ("tuned", 1323, 94.2), ("ref", 1322, 84.7)],
    ),
]
COLOR = {"stock": C_STOCK, "tuned": C_TUNED, "ref": C_REF}


def main():
    w, pad = 0.34, 0.55  # bar width, gap between groups
    xs, centers, ticks = [], [], []
    cursor = 0.0
    for _, _, bars in groups:
        span = [cursor + i * w for i in range(len(bars))]
        xs.append(span)
        centers.append((span[0] + span[-1]) / 2)
        cursor = span[-1] + w + pad
    ticks = [f"{g}\n{sub}" for g, sub, _ in groups]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for span, (_, _, bars) in zip(xs, groups):
        for x, (series, tflops, mfma) in zip(span, bars):
            ax.bar(x, tflops, w, color=COLOR[series])
            # TFLOPS above the bar
            ax.text(
                x,
                tflops + 18,
                f"{tflops}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
            # MFMA efficiency inside the bar, near the top, red on a white pill
            # so it stays legible on every fill
            ax.text(
                x,
                tflops - 30,
                f"{mfma:.1f}%",
                ha="center",
                va="top",
                fontsize=8.2,
                fontweight="bold",
                color=CMFMA,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
            )

    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(
        "FMHA throughput: stock LLVM vs llirSched  "
        "(MI355X · gfx950 · B=32 S=8192 H=8 D=128 · bf16)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(centers)
    ax.set_xticklabels(ticks, fontsize=10)
    left, right = xs[0][0] - w / 2, xs[-1][-1] + w / 2
    ax.set_xlim(left - 0.36, right + 0.36)
    ax.set_ylim(0, 1900)
    ax.set_yticks([0, 500, 1000, 1500])
    legend = [
        Patch(facecolor=C_STOCK, label="stock LLVM  (no plugin, no env)"),
        Patch(facecolor=C_TUNED, label="llirSched, tuned  (SCALE_ON_Q=1, MEMNOP=2)"),
        Patch(facecolor=C_REF, label="ROCm/FlyDSL  (its own tuned config)"),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            color=CMFMA,
            markersize=8,
            label="red % = per-SIMD in-loop MFMA efficiency",
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
        os.path.join(here, "..", "kernels", "attention", "images", "results.png")
    )
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
