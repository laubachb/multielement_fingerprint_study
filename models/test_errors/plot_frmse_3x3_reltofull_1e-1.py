#!/usr/bin/env python3
"""Force-RMSE 3x3 (by state point) for the DLASSO lambda=1e-1 CN pruning grid,
plotted as the DIFFERENCE from the 100% (full-data) model AT THE SAME lambda=1e-1:

    y = (pruned-model F-RMSE) - (full@1e-1 F-RMSE)   per state point
    y = 0  is the full-data model (self)

Retentions: 25 / 37.5 / 50 / 62.5 / 75 %. Pure subtraction of already-computed
per-statepoint force RMSEs -- no re-evaluation.

Inputs (see ./data/, git-ignored):
    force_rmse_perSP_1e-1.csv          pruned per-SP RMSE, 25/50/75%
    force_rmse_perSP_mid_1e-1.csv      pruned per-SP RMSE, 37.5/62.5% midpoints
    force_rmse_result_full_1e-1.txt    full@1e-1 per-SP RMSE (parsed)

Style follows the HEA deviation figure: viridis alpha ramp (alpha=0 dark purple ->
alpha=1 yellow), median + IQR band over reps, black dashed full-model reference.
"""
import os, csv, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict

WORK = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(WORK, "data")
CSVS = [os.path.join(DATA, "force_rmse_perSP_1e-1.csv"),
        os.path.join(DATA, "force_rmse_perSP_mid_1e-1.csv")]
FULL_TXT = os.path.join(DATA, "force_rmse_result_full_1e-1.txt")
OUT_BASE = os.path.join(WORK, "test_error_frmse_3x3_relative_1e-1")

# viridis: alpha=0 dark purple -> alpha=1 yellow
ALPHA_COLORS = {a: plt.cm.viridis(a) for a in (0.0, 0.25, 0.5, 0.75, 1.0)}

SP_GRID = [
    ["20.3percN_2.0gcc", "20.3percN_2.7gcc", "20.3percN_3.5gcc"],
    ["50percN_1.0gcc",   "50percN_2.7gcc",   "50percN_3.5gcc"],
    ["75percN_1.0gcc",   "75percN_2.7gcc",   "75percN_3.0gcc"],
]
# pct labels present in the CSVs (37=37.5%, 62=62.5%) -> true x position
PCTS = [25, 37, 50, 62, 75]
XMAP = {25: 25.0, 37: 37.5, 50: 50.0, 62: 62.5, 75: 75.0}

COMPOSITION_TITLE = {"20pctN": "20% N", "50pctN": "50% N", "75pctN": "75% N"}


def sp_composition_key(sp):
    comp = sp.split("_", 1)[0]
    for p in ("20", "50", "75"):
        if comp.startswith(p):
            return f"{p}pctN"
    return comp


def short_sp_label(sp):
    return f"{sp.split('_', 1)[1].replace('gcc', '')} gcc"


def load_full_1e1():
    """Per-statepoint F-RMSE of the full@1e-1 model, parsed from its eval printout."""
    full = {}
    for line in open(FULL_TXT):
        m = re.match(r"\s*([0-9.]+percN_[0-9.]+gcc)\s+([0-9.]+)", line)
        if m:
            full[m.group(1)] = float(m.group(2))
    return full


def main():
    full1e1 = load_full_1e1()
    pruned = defaultdict(lambda: defaultdict(list))
    for csv_path in CSVS:
        for r in csv.DictReader(open(csv_path)):
            if r["statepoint"] == "OVERALL":
                continue
            a = int(r["alpha"]) / 100.0; pct = int(r["pct"])
            if pct not in PCTS:
                continue
            pruned[(a, pct)][r["statepoint"]].append(float(r["force_rmse_eVA"]))
    alpha_vals = sorted({a for a, _ in pruned})
    print(f"full@1e-1 statepoints: {len(full1e1)}; alphas={alpha_vals}")

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.0), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.16, "hspace": 0.30})
    for r in range(3):
        for c in range(3):
            ax = axes[r, c]; sp = SP_GRID[r][c]
            f0 = full1e1.get(sp)
            for alpha in alpha_vals:
                color = ALPHA_COLORS.get(alpha, "gray")
                xs, med, p25, p75 = [], [], [], []
                for pct in PCTS:
                    vals = pruned.get((alpha, pct), {}).get(sp, [])
                    if not vals or f0 is None:
                        continue
                    d = np.array(vals) - f0
                    xs.append(XMAP[pct]); med.append(np.median(d))
                    p25.append(np.percentile(d, 25)); p75.append(np.percentile(d, 75))
                if not xs:
                    continue
                med = np.array(med)
                yerr = np.vstack([med - np.array(p25), np.array(p75) - med])  # IQR band
                ax.errorbar(xs, med, yerr=yerr, color=color, marker="o", markersize=3.5,
                            markeredgecolor=color, markerfacecolor=color, elinewidth=0.9,
                            capsize=2.5, capthick=0.9, linewidth=1.4, zorder=3)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_xlim(15, 85); ax.set_xticks(list(XMAP.values()))
            ax.tick_params(axis="both", labelsize=6, pad=1, length=2)
            ax.set_ylim(-1.5, 3.0)
            ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.5)
            comp = COMPOSITION_TITLE.get(sp_composition_key(sp), sp_composition_key(sp))
            ax.set_title(f"{comp} — {short_sp_label(sp)}", fontsize=7, pad=3)
            if c == 0:
                ax.set_ylabel("ΔF RMSE vs Full@1e-1 (eV/Å)", fontsize=7)
            if r == 2:
                ax.set_xticklabels(["25", "37.5", "50", "62.5", "75"],
                                   fontsize=5, rotation=45)

    handles = [mlines.Line2D([], [], color=ALPHA_COLORS[a], marker="o",
                             linewidth=1.4, markersize=3.5, label=f"α={a:.2g}")
               for a in alpha_vals]
    handles.append(mlines.Line2D([], [], color="black", linestyle="--",
                                 linewidth=1.0, label="Full@1e-1 (Δ=0)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7,
               framealpha=0.92, edgecolor="gray", handlelength=1.5,
               columnspacing=0.9, handletextpad=0.3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.93)
    fig.supxlabel("Dataset Retention (%)", fontsize=9, y=0.05)
    fig.suptitle("Force RMSE relative to full-data model, both DLASSO λ=1e-1 "
                 "(high-density 20% N excluded)", fontsize=10, y=0.975)
    fig.savefig(OUT_BASE + ".png", dpi=150)
    fig.savefig(OUT_BASE + ".pdf")
    plt.close(fig)
    print(f"Saved:\n  {OUT_BASE}.png\n  {OUT_BASE}.pdf")


if __name__ == "__main__":
    main()
