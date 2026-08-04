#!/usr/bin/env python3
"""
CN UMAP α sweep — single row, colored by thermodynamic statepoint (HEA-matched style).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models" / "workflows" / "fingerprints" / "umap" / "cache"
STATEPOINTS = ROOT / "models" / "statepoint_eval" / "statepoints.json"
FIG_OUT = Path(__file__).resolve().parents[1] / "cn_umap_degeneracy.png"  # figures/
SRC_OUT = ROOT / "models" / "workflows" / "fingerprints" / "umap" / "umap_degeneracy.png"

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]

# One distinct marker per statepoint, in increasing N%/density (legend) order.
STATEPOINT_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h"]

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
    }
)


def build_statepoint_labels(frame_ids: list[int]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Map frame IDs to the 10 evaluation statepoints (20 frames each).
    Frames outside those blocks are labeled 'Other'.
    Returns (label_index, legend_labels, colors).
    """
    sps = sorted(
        json.loads(STATEPOINTS.read_text())["statepoints"],
        key=lambda sp: sp["full_dft_frame"],
    )
    # Each block is 20 frames starting at full_dft_frame
    id_to_sp: dict[int, int] = {}
    legend_labels: list[str] = []
    for i, sp in enumerate(sps):
        start = int(sp["full_dft_frame"])
        end = start + 19
        for fid in range(start, end + 1):
            id_to_sp[fid] = i
        n_pct = sp["n_pct"]
        n_str = f"{n_pct:g}" if float(n_pct) != int(n_pct) else f"{int(n_pct)}"
        legend_labels.append(
            f"{n_str}% N, {sp['density_gcc']:g} g/cc, {int(sp['temperature_k'])} K"
        )

    other_idx = len(sps)
    labels = np.array([id_to_sp.get(fid, other_idx) for fid in frame_ids], dtype=int)
    if (labels == other_idx).any():
        legend_labels.append("Other")

    # Viridis colors ordered by increasing N% then density (i.e. legend order),
    # plus gray for Other.
    vir = mpl.colormaps["viridis"]
    base = [tuple(vir(x)) for x in np.linspace(0.0, 0.95, len(sps))]
    if (labels == other_idx).any():
        base.append((0.75, 0.75, 0.75))
    return labels, legend_labels, base


def spread_limits(emb: np.ndarray, pad: float = 0.14):
    x0, x1 = float(emb[:, 0].min()), float(emb[:, 0].max())
    y0, y1 = float(emb[:, 1].min()), float(emb[:, 1].max())
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    return (x0 - pad * dx, x1 + pad * dx), (y0 - pad * dy, y1 + pad * dy)


def plot_cn_statepoint_umap(
    *,
    emb: dict[float, np.ndarray],
    labels: np.ndarray,
    legend_labels: list[str],
    colors: list,
    outputs: list[Path],
) -> None:
    n_leg = len(legend_labels)
    fig = plt.figure(figsize=(6.5, 3.2))
    gs = GridSpec(
        2,
        5,
        figure=fig,
        height_ratios=[1.0, 0.55],
        left=0.10,
        right=0.98,
        top=0.88,
        bottom=0.04,
        wspace=0.14,
        hspace=0.35,
    )
    axes = [fig.add_subplot(gs[0, c]) for c in range(5)]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")

    col_titles = [
        r"$\alpha$ = 0",
        r"$\alpha$ = 0.25",
        r"$\alpha$ = 0.50",
        r"$\alpha$ = 0.75",
        r"$\alpha$ = 1",
    ]

    for col, alpha in enumerate(ALPHAS):
        ax = axes[col]
        e = emb[alpha]
        for lab_i, color in enumerate(colors):
            if legend_labels[lab_i] == "Other":
                continue
            idx = labels == lab_i
            if not np.any(idx):
                continue
            ax.scatter(
                e[idx, 0],
                e[idx, 1],
                s=26,
                color=color,
                marker=STATEPOINT_MARKERS[lab_i % len(STATEPOINT_MARKERS)],
                alpha=0.85,
                linewidths=0,
                rasterized=True,
            )
        # Limits from plotted statepoint frames only
        keep = labels < len([x for x in legend_labels if x != "Other"])
        xlim, ylim = spread_limits(e[keep], pad=0.14)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(col_titles[col], fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(0.04, 0.62, "UMAP 2", fontsize=10, rotation=90, va="center", ha="center")
    fig.text(0.54, 0.38, "UMAP 1", fontsize=10, ha="center", va="top")

    # Legend: evaluation statepoints only
    handles = [
        Line2D(
            [0],
            [0],
            marker=STATEPOINT_MARKERS[i % len(STATEPOINT_MARKERS)],
            color="w",
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            markersize=7,
            label=legend_labels[i],
        )
        for i in range(n_leg)
        if legend_labels[i] != "Other"
    ]
    legend_ax.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=8,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=1.4,
        labelspacing=0.6,
        title="Statepoint",
        title_fontsize=10,
    )

    for out in outputs:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        print(f"Saved → {out}")
    plt.close(fig)


def main() -> None:
    man = json.loads((CACHE / "manifest.json").read_text())
    emb_z = np.load(CACHE / "embeddings.npz")

    frame_ids = [
        int(re.search(r"frame_(\d+)", d).group(1)) for d in man["frame_dirs"]
    ]
    labels, legend_labels, colors = build_statepoint_labels(frame_ids)
    emb = {a: emb_z[f"emb_all_{a:.2f}"] for a in ALPHAS}

    counts = {lab: int((labels == i).sum()) for i, lab in enumerate(legend_labels)}
    print(f"CN frames={len(frame_ids)}; statepoint counts: {counts}")

    plot_cn_statepoint_umap(
        emb=emb,
        labels=labels,
        legend_labels=legend_labels,
        colors=colors,
        outputs=[FIG_OUT, SRC_OUT],
    )


if __name__ == "__main__":
    main()
