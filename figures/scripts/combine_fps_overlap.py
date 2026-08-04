#!/usr/bin/env python3
"""Combine the CN and HEA (mixed) FPS subset-overlap curves into one 1x2 figure.

Left panel  : CN            (output/fps_jaccard_summary.csv, system == CN)
Right panel : HEA mixed     (output/hea_mixed_fps_jaccard_summary.csv)

6 in wide x 3 in tall, shared y-axis, matching the styling of the two source
figures. Writes a NEW file; does not overwrite any existing figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
FIG_DIR = HERE.parent / "figures"

plt.rcParams.update({"font.size": 10})

SERIES = [
    (0.0, 1.0, r"$\alpha$ = 0 vs 1"),
    (0.25, 1.0, r"$\alpha$ = 0.25 vs 1"),
    (0.50, 1.0, r"$\alpha$ = 0.50 vs 1"),
    (0.75, 1.0, r"$\alpha$ = 0.75 vs 1"),
]
CMAP = mpl.colormaps["viridis"]
COLORS = [CMAP(x) for x in np.linspace(0.15, 0.85, len(SERIES))]
LINESTYLES = ["-", "--", "-.", ":"]


def panel(ax, df, title, tag, show_legend):
    for (aa, ab, lab), color, ls in zip(SERIES, COLORS, LINESTYLES):
        sub = df[(df.alpha_a == aa) & (df.alpha_b == ab)].sort_values("retention_pct")
        ax.errorbar(
            sub["retention_pct"],
            sub["shared_pct_mean"],
            yerr=sub["shared_pct_std"],
            color=color,
            ls=ls,
            marker="o",
            ms=3.5,
            lw=1.5,
            capsize=2,
            label=lab,
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("Dataset Retention (%)", fontsize=10)
    ax.set_title(title, fontsize=10)
    # Panel tag at the top-left for referencing in text.
    ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")
    if show_legend:
        ax.legend(frameon=False, loc="lower right", fontsize=9,
                  handlelength=1.6, borderaxespad=0.3)


def main():
    cn = pd.read_csv(OUT_DIR / "fps_jaccard_summary.csv")
    cn = cn[cn.system == "CN"]
    hea = pd.read_csv(OUT_DIR / "hea_mixed_fps_jaccard_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
    panel(axes[0], cn, "Carbon-Nitrogen (CN)", "(a)", show_legend=True)
    panel(axes[1], hea, "Y-Mg HEA (mixed)", "(b)", show_legend=False)
    axes[0].set_ylabel("% of Selected Frames Shared", fontsize=10)

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.18, top=0.88, wspace=0.12)
    out = FIG_DIR / "cn_hea_fps_overlap_combined"
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
