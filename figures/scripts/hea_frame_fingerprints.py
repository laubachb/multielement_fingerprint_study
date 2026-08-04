#!/usr/bin/env python3
"""HEA cluster-graph fingerprints for two diverse frames, across the alpha sweep.

Analogous to cn_element_switching_graphite.png, but for the Y-Mg HEA dataset:
two frames (a low-energy, ordered one and a high-energy, disordered one) are
shown side by side, and within each panel the fingerprint P(d) is drawn for
alpha in {0, 0.25, 0.5, 0.75, 1}, colored viridis (purple = alpha 0, yellow =
alpha 1). Dashed lines separate the 2-, 3-, and 4-body sectors. No new
calculations: fingerprints are read from precomputed .hist files.

Writes a NEW file; does not overwrite any existing figure.
"""
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
HEA = STUDY / "hea_study"
TD = HEA / "transfer_to_local-Apr2026"
EXTXYZ = HEA / "data" / "DS_0bmdn3d792js_0.extxyz_MgY"
FIG = STUDY / "writing" / "figures" / "hea_frame_fingerprints"

ALPHA_DIRS = {
    0.00: "alpha_0-histograms", 0.25: "alpha_025-histograms",
    0.50: "alpha_050-histograms", 0.75: "alpha_075-histograms",
    1.00: "alpha_1-histograms",
}
HIST_FILES = ["0-0.2b_clu-s.hist", "0-0.3b_clu-s.hist", "0-0.4b_clu-s.hist"]
BODY_LABELS = ["2-body", "3-body", "4-body"]

# Two diverse, well-spread frames (lower / higher energy). Easy to swap.
FRAMES = ["frame_220", "frame_327"]


def load_fp(alpha, frame):
    fdir = TD / ALPHA_DIRS[alpha] / frame
    return np.concatenate([np.loadtxt(fdir / h)[:, 1] for h in HIST_FILES])


def seg_bounds(frame):
    lens = [len(np.loadtxt(TD / ALPHA_DIRS[0.0] / frame / h)) for h in HIST_FILES]
    b = [0] + list(np.cumsum(lens))
    return b


def energies_per_atom(extxyz):
    out = {}
    fid = 1
    with open(extxyz) as f:
        while True:
            line = f.readline()
            if not line:
                break
            nat = int(line.strip())
            comment = f.readline()
            m = re.search(r"energy_energy=([-\d.eE+]+)", comment)
            out[fid] = float(m.group(1)) / nat
            for _ in range(nat):
                f.readline()
            fid += 1
    return out


def main():
    E = energies_per_atom(EXTXYZ)
    alphas = sorted(ALPHA_DIRS)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0.0, vmax=1.0)

    bounds = seg_bounds(FRAMES[0])
    nbins = bounds[-1]
    x = np.arange(nbins)

    # Load and find shared y-limit.
    data = {}
    ymax = 0.0
    for fr in FRAMES:
        data[fr] = {a: load_fp(a, fr) for a in alphas}
        ymax = max(ymax, max(v.max() for v in data[fr].values()))

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.85, bottom=0.16, top=0.88, wspace=0.10)

    tags = ["(a)", "(b)"]
    titles = ["Low energy", "High energy"]
    for ax, fr, tag, ttl in zip(axes, FRAMES, tags, titles):
        for a in alphas:
            ax.plot(x, data[fr][a], color=cmap(norm(a)), lw=1.0, alpha=0.92)
        for b in bounds[1:-1]:
            ax.axvline(b, color="#aaaaaa", lw=0.7, ls="--", zorder=0)
        fid = int(fr.split("_")[1])
        ax.set_title(f"{ttl} ({E[fid]:.1f} eV/atom)", fontsize=10, pad=3)
        ax.text(0.02, 0.96, tag, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left")
        ax.set_xlim(0, nbins - 1)
        ax.set_ylim(0, 0.5)
        mids = [(bounds[j] + bounds[j + 1]) / 2 for j in range(3)]
        ax.set_xticks(mids)
        ax.set_xticklabels(BODY_LABELS, fontsize=10)
        ax.tick_params(labelsize=10, length=3)
    axes[0].set_ylabel(r"$P(d)$", fontsize=10)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.875, 0.16, 0.022, 0.72])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\alpha$", fontsize=10, labelpad=2)
    cbar.set_ticks(alphas)
    cbar.ax.tick_params(labelsize=9)

    for ext in (".png", ".pdf"):
        fig.savefig(FIG.with_suffix(ext), dpi=220)
    plt.close(fig)
    print(f"Saved {FIG}.png / .pdf")
    print("Frames:", {fr: round(E[int(fr.split('_')[1])], 3) for fr in FRAMES})


if __name__ == "__main__":
    main()
