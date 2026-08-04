#!/usr/bin/env python3
"""Publication theory schematic highlighting the fingerprint α parameter."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "theory_schematic.png"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    }
)

ELEM_COLOR = {"C": "#2166ac", "N": "#b2182b"}
ATOM_R = 0.18
CLUSTER_SCALE = 0.58
LINE_COLOR = "0.4"
LINE_LW = 1.1


def draw_cluster(ax, cx, cy, elements, *, shape="equilateral", scale=CLUSTER_SCALE):
    if shape == "equilateral":
        pos = np.array([[0.0, 0.58], [-0.52, -0.32], [0.52, -0.32]]) * scale
    else:
        pos = np.array([[0.0, 0.85], [-0.28, -0.40], [0.28, -0.40]]) * scale

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        ax.plot(
            [cx + pos[i, 0], cx + pos[j, 0]],
            [cy + pos[i, 1], cy + pos[j, 1]],
            color="0.45",
            lw=1.4,
            zorder=3,
            solid_capstyle="round",
        )
    for (x, y), el in zip(pos, elements):
        ax.add_patch(
            Circle(
                (cx + x, cy + y),
                ATOM_R,
                facecolor=ELEM_COLOR[el],
                edgecolor="0.15",
                linewidth=0.7,
                zorder=4,
            )
        )
        ax.text(
            cx + x,
            cy + y,
            el,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
            zorder=5,
        )


def distance_line(ax, p1, p2, label, *, offset=(0.0, 0.0)):
    """Draw a dashed line between cluster centers with a black distance label."""
    x1, y1 = p1
    x2, y2 = p2
    ax.plot([x1, x2], [y1, y2], color=LINE_COLOR, lw=LINE_LW, ls="--", zorder=1, alpha=0.85)
    mx = 0.5 * (x1 + x2) + offset[0]
    my = 0.5 * (y1 + y2) + offset[1]
    ax.text(
        mx,
        my,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color="black",
        zorder=6,
    )


def label_cluster(ax, p, name, *, above=False):
    dy = 0.90 if above else -0.68
    ax.text(p[0], p[1] + dy, name, ha="center", va="center", fontsize=10)


def main() -> None:
    # 6 × 3.5 in; equal aspect → data ylim ≈ 10 * 3.5/6 ≈ 5.83
    fig = plt.figure(figsize=(6.0, 3.5))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.83)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    C1, C2, C3 = ["C", "C", "N"], ["N", "C", "C"], ["C", "C", "C"]

    # Wider inter-cluster spacing within each panel
    # Boxes sit below α titles (titles drawn outside, above the grey frame)
    # Cluster content shifted up ~0.25 in (~0.42 data units); order text stays low
    dy = 0.42
    specs = [
        (
            0.20,
            0.25,
            3.10,
            4.75,
            r"$\boldsymbol{\alpha = 0}$",
            (0.80, 3.40 + dy),
            (2.70, 3.40 + dy),
            (1.75, 1.55 + dy),
            "short12",
            r"$D_{\mathcal{C}_1,\mathcal{C}_2} < D_{\mathcal{C}_1,\mathcal{C}_3} \approx D_{\mathcal{C}_2,\mathcal{C}_3}$",
        ),
        (
            3.45,
            0.25,
            3.10,
            4.75,
            r"$\boldsymbol{\alpha = 0.5}$",
            (4.05, 3.35 + dy),
            (5.95, 3.35 + dy),
            (5.00, 1.80 + dy),
            "balanced",
            r"$D_{\mathcal{C}_1,\mathcal{C}_2} \approx D_{\mathcal{C}_1,\mathcal{C}_3} \approx D_{\mathcal{C}_2,\mathcal{C}_3}$",
        ),
        (
            6.70,
            0.25,
            3.10,
            4.75,
            r"$\boldsymbol{\alpha = 1}$",
            (7.30, 3.40 + dy),
            (8.25, 1.55 + dy),
            (9.20, 3.40 + dy),
            "short13",
            r"$D_{\mathcal{C}_1,\mathcal{C}_3} < D_{\mathcal{C}_1,\mathcal{C}_2} \approx D_{\mathcal{C}_2,\mathcal{C}_3}$",
        ),
    ]

    for x, y, w, h, title, p1, p2, p3, mode, order in specs:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#f7f7f7",
                edgecolor="0.55",
                linewidth=0.9,
                zorder=0,
            )
        )
        ax.text(
            x + w / 2,
            y + h + 0.28,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            usetex=False,
        )

        if mode == "short12":
            distance_line(ax, p1, p2, r"$D_{\mathcal{C}_1,\mathcal{C}_2}$", offset=(0.0, 0.30))
            distance_line(ax, p1, p3, r"$D_{\mathcal{C}_1,\mathcal{C}_3}$", offset=(-0.38, 0.0))
            distance_line(ax, p2, p3, r"$D_{\mathcal{C}_2,\mathcal{C}_3}$", offset=(0.38, 0.0))
            draw_cluster(ax, *p1, C1, shape="equilateral")
            draw_cluster(ax, *p2, C2, shape="isosceles")
            draw_cluster(ax, *p3, C3, shape="equilateral")
            label_cluster(ax, p1, r"$\mathcal{C}_1$", above=True)
            label_cluster(ax, p2, r"$\mathcal{C}_2$", above=True)
            label_cluster(ax, p3, r"$\mathcal{C}_3$")

        elif mode == "balanced":
            distance_line(ax, p1, p2, r"$D_{\mathcal{C}_1,\mathcal{C}_2}$", offset=(0.0, 0.30))
            distance_line(ax, p1, p3, r"$D_{\mathcal{C}_1,\mathcal{C}_3}$", offset=(-0.38, 0.0))
            distance_line(ax, p2, p3, r"$D_{\mathcal{C}_2,\mathcal{C}_3}$", offset=(0.38, 0.0))
            draw_cluster(ax, *p1, C1, shape="equilateral")
            draw_cluster(ax, *p2, C2, shape="isosceles")
            draw_cluster(ax, *p3, C3, shape="equilateral")
            label_cluster(ax, p1, r"$\mathcal{C}_1$", above=True)
            label_cluster(ax, p2, r"$\mathcal{C}_2$", above=True)
            label_cluster(ax, p3, r"$\mathcal{C}_3$")

        else:  # short13
            c1, c2, c3 = p1, p2, p3
            distance_line(ax, c1, c3, r"$D_{\mathcal{C}_1,\mathcal{C}_3}$", offset=(0.0, 0.30))
            distance_line(ax, c1, c2, r"$D_{\mathcal{C}_1,\mathcal{C}_2}$", offset=(-0.38, 0.0))
            distance_line(ax, c3, c2, r"$D_{\mathcal{C}_2,\mathcal{C}_3}$", offset=(0.38, 0.0))
            draw_cluster(ax, *c1, C1, shape="equilateral")
            draw_cluster(ax, *c3, C3, shape="equilateral")
            draw_cluster(ax, *c2, C2, shape="isosceles")
            label_cluster(ax, c1, r"$\mathcal{C}_1$", above=True)
            label_cluster(ax, c3, r"$\mathcal{C}_3$", above=True)
            label_cluster(ax, c2, r"$\mathcal{C}_2$")

        ax.text(
            x + w / 2,
            y + 0.42,
            order,
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, facecolor="white")
    print(f"Wrote {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
