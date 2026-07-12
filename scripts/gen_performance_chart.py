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

"""Regenerate performance_chart.png for the a16w16 optimization journey.

Bars = TFLOPS (left axis), red line = MFMA efficiency (right axis), one bar per
(version, config).  Configs: base / llir / llir+force-agpr / llir+force-agpr+amdgcnas.

Data: MI355, 4096x4096x8192, FP16, rocprofv3 (1000 dispatches, last-100 avg),
collected with:
    python scripts/run_perf_table.py --kernel a16w16 --versions <v> \
        --configs base llir llir+force-agpr llir+force-agpr+amdgcnas \
        --K 8192 --dtype fp16 --rocprof --allow-unreported

Run:  python scripts/gen_performance_chart.py
      (writes kernels/gemm/intra_wave/a16w16/images/performance_chart.png)
"""

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Config styling, in cumulative order.
CONFIGS = {
    "base": dict(color="#4E95D9", label="Base kernel", tag=""),
    "llir": dict(color="#E8973A", label="+ llir", tag="+llir"),
    "llir+force-agpr": dict(color="#9467BD", label="+ llir + force-agpr", tag="+fa"),
    "llir+force-agpr+amdgcnas": dict(
        color="#5BA85B", label="+ llir + force-agpr + amdgcnas", tag="+fa+asm"
    ),
}

# (version, config, TFLOPS, MFMA%) in plotting order.
DATA = [
    (0, "base", 541, 25.47),
    (1, "base", 546, 26.42),
    (2, "base", 672, 32.81),
    (3, "base", 770, 41.35),
    (4, "base", 1123, 57.71),
    (5, "base", 1128, 58.35),
    (5, "llir", 1228, 79.91),
    (6, "base", 1055, 56.40),
    (6, "llir", 1167, 86.58),
    (7, "base", 1233, 64.99),
    (7, "llir", 1329, 84.91),
    (7, "llir+force-agpr", 1395, 97.29),
    (7, "llir+force-agpr+amdgcnas", 1384, 98.28),
    (8, "base", 1257, 69.03),
    (8, "llir", 1360, 88.65),
    (8, "llir+force-agpr", 1408, 97.79),
    (8, "llir+force-agpr+amdgcnas", 1391, 98.52),
    (9, "base", 1290, 69.19),
    (9, "llir", 1380, 88.44),
    (9, "llir+force-agpr", 1423, 97.53),
    (9, "llir+force-agpr+amdgcnas", 1421, 98.66),
]

x = list(range(len(DATA)))
tflops = [d[2] for d in DATA]
mfma = [d[3] for d in DATA]
colors = [CONFIGS[d[1]]["color"] for d in DATA]
labels = [f"v{d[0]}" + ("\n" + CONFIGS[d[1]]["tag"] if CONFIGS[d[1]]["tag"] else "") for d in DATA]

fig, ax1 = plt.subplots(figsize=(20, 9))

bars = ax1.bar(x, tflops, color=colors, width=0.8, zorder=2)
for xi, tf in zip(x, tflops):
    ax1.text(xi, tf + 12, f"{tf}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_ylabel("TFLOPS", fontsize=13)
ax1.set_xlabel("Kernel Version", fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0, max(tflops) * 1.18)
ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

ax2 = ax1.twinx()
ax2.plot(x, mfma, color="#D62728", marker="s", markersize=6, linewidth=2, zorder=3)
for xi, mf in zip(x, mfma):
    ax2.text(xi, mf - 4, f"{mf:.0f}%", ha="center", va="top", color="#D62728", fontsize=9)
ax2.set_ylabel("MFMA Efficiency (%)", color="#D62728", fontsize=13)
ax2.tick_params(axis="y", labelcolor="#D62728")
ax2.set_ylim(0, 140)

legend_handles = [Patch(facecolor=c["color"], label=c["label"]) for c in CONFIGS.values()]
legend_handles.append(Line2D([0], [0], color="#D62728", marker="s", label="MFMA Efficiency"))
ax1.legend(handles=legend_handles, loc="upper left", fontsize=11)

ax1.set_title("FP16 GEMM Performance on MI355 (4096×4096×8192)", fontsize=15, fontweight="bold")
fig.tight_layout()

out = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "kernels",
        "gemm",
        "intra_wave",
        "a16w16",
        "images",
        "performance_chart.png",
    )
)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
