#!/usr/bin/env python3
"""
Generate performance bar chart for FP16 GEMM kernel versions.

Usage:
    python generate_perf_chart.py

Requires:
    pip install matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    # Performance data from MI355, shape 4096x4096x8192, FP16
    # (version, tflops, variant, mfma_efficiency)
    # mfma_efficiency is None for v0-v2 where it wasn't measured
    data = [
        ("v0", 500, "base", None),
        ("v1", 482, "base", None),
        ("v2", 647, "base", None),
        ("v3", 700, "base", 43),
        ("v4", 984, "base", 57),
        ("v5", 983, "base", 58),
        ("v5", 1119, "llirSched", 76),
        ("v6", 1015, "base", 61),
        ("v6", 1122, "llirSched", 88),
        ("v7", 1128, "base", 65),
        ("v7", 1244, "llirSched", 79),
        ("v7", 1341, "llirSched+amdgcnas", 98),
        ("v8", 1137, "base", 67),
        ("v8", 1251, "llirSched", 77),
        ("v8", 1405, "llirSched+amdgcnas", 99),
    ]

    # Create labels for x-axis
    labels = []
    for version, _, variant, _ in data:
        if variant == "llirSched+amdgcnas":
            labels.append(f"{version}\n+sched\n+asm")
        elif variant == "llirSched":
            labels.append(f"{version}\n+sched")
        else:
            labels.append(version)

    tflops = [d[1] for d in data]
    variants = [d[2] for d in data]
    mfma_eff = [d[3] for d in data]

    # Colors by variant
    color_map = {
        "base": "#3498db",              # blue
        "llirSched": "#e67e22",         # orange
        "llirSched+amdgcnas": "#2ecc71",  # green
    }
    colors = [color_map[v] for v in variants]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Bar chart for TFLOPS
    bars = ax1.bar(range(len(tflops)), tflops, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)

    # Add TFLOPS value labels on bars
    for bar, val in zip(bars, tflops):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Line plot for MFMA efficiency (only where available)
    x_eff = [i for i, e in enumerate(mfma_eff) if e is not None]
    y_eff = [e for e in mfma_eff if e is not None]
    ax2.plot(x_eff, y_eff, "s-", color="#e74c3c", linewidth=2, markersize=8, label="MFMA Efficiency")

    # Add efficiency labels
    for x, y in zip(x_eff, y_eff):
        ax2.text(x, y + 3, f"{y}%", ha="center", va="bottom", fontsize=8, color="#e74c3c", fontweight="bold")

    # Configure axes
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("TFLOPS", fontsize=12)
    ax1.set_xlabel("Kernel Version", fontsize=12)
    ax1.set_title("FP16 GEMM Performance on MI355 (4096×4096×8192)", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 1600)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_ylabel("MFMA Efficiency (%)", fontsize=12, color="#e74c3c")
    ax2.set_ylim(0, 150)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    # Legend
    blue_patch = mpatches.Patch(color="#3498db", label="Base kernel")
    orange_patch = mpatches.Patch(color="#e67e22", label="+ llirSched")
    green_patch = mpatches.Patch(color="#2ecc71", label="+ llirSched + amdgcnas")
    eff_line = plt.Line2D([0], [0], color="#e74c3c", marker="s", linewidth=2, label="MFMA Efficiency")
    ax1.legend(handles=[blue_patch, orange_patch, green_patch, eff_line], loc="upper left")

    plt.tight_layout()

    output_path = "images/performance_chart.png"
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    main()
