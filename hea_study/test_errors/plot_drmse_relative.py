#!/usr/bin/env python3
"""HEA pruning curve: deviation of each pruned model from the full-data (pct100)
model on the mixed holdout frames, vs dataset retention, one line per fingerprint
alpha -- the departure from the full model (full = Delta 0).

Styled to match deviation_force.png: viridis alpha ramp (alpha=0 dark purple ->
alpha=1 yellow), median + IQR band over reps, black dashed full-model reference.
Now includes the 37.5% / 62.5% midpoint retentions.  Pure re-plot of
deviation_permodel.csv -- no re-evaluation.
"""
import os, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

WORK = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(WORK, "deviation_permodel.csv")
OUT_BASE = os.path.join(WORK, "test_error_drmse_relative_hea")

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})
# viridis: alpha=0 dark purple -> alpha=1 yellow  (deviation_force.png scheme)
ALPHA_COLORS = {a: plt.cm.viridis(a) for a in (0.0, 0.25, 0.5, 0.75, 1.0)}
# retention labels present in deviation_permodel.csv (38=37.5%, 62=62.5%) -> true x
RETENTIONS = [25, 38, 50, 62, 75]
XMAP = {25: 25.0, 38: 37.5, 50: 50.0, 62: 62.5, 75: 75.0}


def load():
    rows = []
    for r in csv.DictReader(open(CSV)):
        rows.append(dict(alpha=float(r["alpha"]), retention=int(r["retention"]),
                         rep=int(r["rep"]), f=float(r["f_dev_eVA"])))
    return rows


def main():
    rows = load()
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["alpha"], r["retention"])].append(r["f"])
    alphas = sorted({a for a, _ in buckets})
    print(f"plotting {len(rows)} models; alphas={alphas}; "
          f"reps/cell~{len(next(iter(buckets.values())))}")

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for a in alphas:
        xs, med, p25, p75 = [], [], [], []
        for ret in RETENTIONS:
            vals = np.array(buckets.get((a, ret), []))
            if vals.size == 0:
                continue
            xs.append(XMAP[ret]); med.append(np.median(vals))
            p25.append(np.percentile(vals, 25)); p75.append(np.percentile(vals, 75))
        if not xs:
            continue
        med = np.array(med)
        yerr = np.vstack([med - np.array(p25), np.array(p75) - med])  # IQR band
        color = ALPHA_COLORS.get(a, "gray")
        ax.errorbar(xs, med, yerr=yerr, color=color, marker="o", markersize=4.5,
                    markeredgecolor=color, markerfacecolor=color, elinewidth=1.0,
                    capsize=2.5, linewidth=1.4, zorder=3)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlim(18, 82); ax.set_xticks(list(XMAP.values()))
    ax.set_xticklabels(["25", "37.5", "50", "62.5", "75"])
    ax.set_xlabel("Dataset retention (%)", fontsize=10)
    ax.set_ylabel(r"$\Delta$F RMSE vs Full (eV/$\AA$)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4, linewidth=0.6)
    ax.tick_params(labelsize=10)
    ax.margins(x=0.08)

    handles = [mlines.Line2D([], [], color=ALPHA_COLORS.get(a, "gray"), marker="o",
                             markersize=4.5, linewidth=1.4, label=f"α={a:g}")
               for a in alphas]
    handles.append(mlines.Line2D([], [], color="black", linestyle="--",
                                 linewidth=1.0, label="full model (ref)"))
    ax.legend(handles=handles, fontsize=10, framealpha=0.9, edgecolor="gray",
              loc="best", handlelength=1.3, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT_BASE + ".png", dpi=300)
    fig.savefig(OUT_BASE + ".pdf")
    plt.close(fig)
    print(f"Saved:\n  {OUT_BASE}.png\n  {OUT_BASE}.pdf")


if __name__ == "__main__":
    main()
