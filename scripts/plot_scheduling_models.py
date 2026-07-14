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
Regenerate the three scheduling-model diagrams in kernels/gemm/README.md §2:

    images/sched_inter_wave.png   — two waves ping-pong (inter-wave)
    images/sched_intra_wave.png   — one wave, software-pipelined (intra-wave)
    images/sched_warp_spec.png    — dedicated producer/consumer waves (warp specialization)

The shared model: SIMD0 must run 2 regions of 4 MFMA (8 MFMA total). Each region
needs 2 ds_read (LDS -> registers) to prepare its operands and issues 2
buffer_load (global -> LDS prefetch for later regions). A row is one wave's issue
timeline; latencies are assumed hidden. The dependency arrow is the RAW edge from
a ds_read (producer) to the first MFMA that consumes it.

    python scripts/plot_scheduling_models.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

DS = "#EFA13C"  # ds_read       (orange square)
BL = "#F5E63D"  # buffer_load   (yellow square)
MF = "#4E7D2A"  # mfma          (green rectangle)
DEP = "#2E6E8E"  # dependency arrow (teal)
EDGE = "#2b2b2b"

MW, MH = 1.0, 0.62  # mfma rectangle (wide = long compute op)
MS = 0.36  # mem square side (small = quick issue)


def mfma(ax, x, y):
    ax.add_patch(Rectangle((x, y - MH / 2), MW, MH, fc=MF, ec=EDGE, lw=1.0, zorder=3))


def mem(ax, xc, yc, color):
    ax.add_patch(
        Rectangle((xc - MS / 2, yc - MS / 2), MS, MS, fc=color, ec=EDGE, lw=0.8, zorder=4)
    )


def dep(ax, xy_from, xy_to, rad):
    ax.add_patch(
        FancyArrowPatch(
            xy_from,
            xy_to,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=15,
            lw=2.0,
            color=DEP,
            zorder=6,
        )
    )


def wave_label(ax, y, text):
    ax.text(-0.35, y, text, ha="right", va="center", fontsize=10, fontweight="bold")


def finish(ax, title, xlim, ylim):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_aspect("equal")


LEGEND = [
    Line2D([], [], marker="s", ls="none", mfc=MF, mec=EDGE, ms=11, label="mfma"),
    Line2D([], [], marker="s", ls="none", mfc=DS, mec=EDGE, ms=9, label="ds_read"),
    Line2D([], [], marker="s", ls="none", mfc=BL, mec=EDGE, ms=9, label="buffer_load"),
    Line2D([], [], color=DEP, lw=2.0, marker=">", markevery=[-1], label="ds_read -> mfma dep"),
]


def legend(fig):
    fig.legend(
        handles=LEGEND, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
    )


def save(fig, name):
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "kernels", "gemm", "images", name))
    fig.savefig(out, facecolor="white", bbox_inches="tight", dpi=150)
    print("wrote", out)


# ------------------------------------------------------------------ inter-wave
def plot_inter():
    fig, ax = plt.subplots(figsize=(8.6, 2.5))
    fig.patch.set_facecolor("white")
    y0, y1 = 1.15, 0.0  # wave0 top, wave1 bottom
    wave_label(ax, y0, "wave0")
    wave_label(ax, y1, "wave1")
    # phase A (x 0-4): wave0 computes, wave1 preps memory
    for i in range(4):
        mfma(ax, i, y0)
    for xc, c in zip((0.5, 1.5, 2.5, 3.5), (DS, DS, BL, BL)):
        mem(ax, xc, y1, c)
    # phase B (x 4-8): wave1 computes, wave0 preps memory
    for i in range(4, 8):
        mfma(ax, i, y1)
    for xc, c in zip((4.5, 5.5, 6.5, 7.5), (DS, DS, BL, BL)):
        mem(ax, xc, y0, c)
    # dep: wave1's ds (phase A) -> wave1's first mfma (phase B)
    dep(ax, (1.5, y1 - MS / 2), (4.02, y1), rad=-0.5)
    ax.text(2.0, y0 + 0.6, "phase A", ha="center", fontsize=9, color="#555")
    ax.text(6.0, y0 + 0.6, "phase B", ha="center", fontsize=9, color="#555")
    finish(ax, "Inter-wave: two waves ping-pong (compute ↔ memory)", (-1.4, 8.3), (-1.0, 1.85))
    legend(fig)
    save(fig, "sched_inter_wave.png")


# ------------------------------------------------------------------ intra-wave
def plot_intra():
    fig, ax = plt.subplots(figsize=(8.6, 1.9))
    fig.patch.set_facecolor("white")
    y = 0.5
    wave_label(ax, y, "wave0")
    for i in range(8):  # 8 mfma, contiguous -> matrix pipe stays busy
        mfma(ax, i, y)
    seams = (0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9)
    cols = (DS, DS, BL, BL, DS, DS, BL, BL)
    ytop = y + MH / 2 + MS / 2 + 0.02
    for xc, c in zip(seams, cols):
        mem(ax, xc, ytop, c)
    # dep: a region-A ds prepares region-B operands (issued a full region early)
    dep(ax, (1.9, ytop + MS / 2), (4.02, y + MH / 2), rad=0.5)
    finish(ax, "Intra-wave: one wave, memory software-pipelined into the mfma stream",
           (-1.4, 8.3), (-0.2, 1.55))
    legend(fig)
    save(fig, "sched_intra_wave.png")


# ---------------------------------------------------------- warp specialization
def plot_ws():
    fig, ax = plt.subplots(figsize=(8.6, 2.5))
    fig.patch.set_facecolor("white")
    y0, y1 = 1.15, 0.0
    wave_label(ax, y0, "wave0")
    wave_label(ax, y1, "wave1")
    for i in range(8):  # wave0: only mfma, back to back, never stalls
        mfma(ax, i, y0)
    seams = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5)
    cols = (DS, DS, BL, BL, DS, DS, BL, BL)
    for xc, c in zip(seams, cols):  # wave1: only memory
        mem(ax, xc, y1, c)
    # dep crosses waves: producer (wave1 ds) -> consumer (wave0 mfma)
    dep(ax, (1.5, y1 + MS / 2), (4.0, y0 - MH / 2), rad=0.0)
    ax.text(4.0, y0 + 0.6, "compute wave (never issues memory)", ha="center", fontsize=8.5,
            color="#555")
    ax.text(4.0, y1 - 0.55, "producer wave (only memory)", ha="center", fontsize=8.5, color="#555")
    finish(ax, "Warp specialization: dedicated compute + producer waves (not on gfx950)",
           (-1.4, 8.3), (-1.0, 1.85))
    legend(fig)
    save(fig, "sched_warp_spec.png")


def main():
    plot_inter()
    plot_intra()
    plot_ws()


if __name__ == "__main__":
    main()
