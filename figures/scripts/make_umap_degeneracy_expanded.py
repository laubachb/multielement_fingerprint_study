#!/usr/bin/env python3
"""
2×5 HEA UMAP degeneracy figure (publication styling).

Row 0 – all frames colored by composition (Y / Mg / Mixed)
Row 1 – mixed frames colored by energy/atom + degeneracy annotations

Figure size: 6.5×5 in, 10 pt text. Legend between rows; energy colorbar below.
Unified UMAP 1 / UMAP 2 axis labels.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
# umap is only needed to RECOMPUTE embeddings; with the cached embeddings present
# it is imported lazily inside the recompute branch (see main()), so the figure can
# be regenerated without the umap package installed.
from sklearn.cluster import KMeans

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
HEA_DIR = SCRIPT_DIR.parent
DEFAULT_TRANSFER_DIR = HEA_DIR / "transfer_to_local-Apr2026"
DEFAULT_LABEL_DIR = HEA_DIR / "alpha_0-histograms"
DEFAULT_EXTXYZ = HEA_DIR / "data" / "DS_0bmdn3d792js_0.extxyz_MgY"
CACHE_DIR = SCRIPT_DIR / "cache"
WRITING_FIG = Path(__file__).resolve().parents[1] / "hea_umap_degeneracy.png"  # figures/

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
HIST_FILES = [
    "0-0.2b_clu-s.hist",
    "0-0.3b_clu-s.hist",
    "0-0.4b_clu-s.hist",
]
CLUSTER_FILE = "0.all-2b-clusters.txt"
FRAME_RE = re.compile(r"^frame_(\d+)$")

CAT_INFO = {
    0: dict(label="Y only", color="#2166ac", marker="o"),
    1: dict(label="Mg only", color="#d6604d", marker="s"),
    2: dict(label="Mixed", color="#4dac26", marker="^"),
}
DEGEN_COLOR = "#e6ac00"
CIRCLE_COLOR = "black"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-dir", type=Path, default=DEFAULT_TRANSFER_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--extxyz", type=Path, default=DEFAULT_EXTXYZ)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "umap_degeneracy_expanded.png",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def alpha_dirs_from_transfer(transfer_dir: Path) -> dict[float, Path]:
    return {
        0.00: transfer_dir / "alpha_0-histograms",
        0.25: transfer_dir / "alpha_025-histograms",
        0.50: transfer_dir / "alpha_050-histograms",
        0.75: transfer_dir / "alpha_075-histograms",
        1.00: transfer_dir / "alpha_1-histograms",
    }


def get_frame_list(base_dir: Path) -> list[str]:
    return sorted(
        [d.name for d in base_dir.iterdir() if d.is_dir() and FRAME_RE.match(d.name)],
        key=lambda x: int(FRAME_RE.match(x).group(1)),
    )


def frame_ok(frame: str, label_dir: Path, alpha_dirs: dict[float, Path]) -> bool:
    if not (label_dir / frame / CLUSTER_FILE).is_file():
        return False
    for alpha_dir in alpha_dirs.values():
        for hist_name in HIST_FILES:
            hist_path = alpha_dir / frame / hist_name
            if not hist_path.is_file():
                return False
            try:
                if np.isnan(np.loadtxt(hist_path)).any():
                    return False
            except Exception:
                return False
    return True


def frame_composition_label(frame: str, label_dir: Path) -> int:
    data = np.loadtxt(label_dir / frame / CLUSTER_FILE)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    unique = set(data[:, 1:].astype(int).flatten())
    if unique == {0}:
        return 0
    if unique == {1}:
        return 1
    return 2


def load_fp(alpha_dir: Path, frame: str) -> np.ndarray:
    fdir = alpha_dir / frame
    return np.concatenate(
        [np.loadtxt(fdir / hist_name)[:, 1] for hist_name in HIST_FILES]
    )


def drop_zero_cols(arr: np.ndarray) -> np.ndarray:
    return arr[:, arr.any(axis=0)]


def load_energy_per_atom(extxyz: Path) -> dict[int, float]:
    """Map 1-based frame index → energy/atom from Kolibri/extxyz energy_energy."""
    out: dict[int, float] = {}
    frame_id = 1
    with open(extxyz) as f:
        while True:
            line = f.readline()
            if not line:
                break
            nat = int(line.strip())
            comment = f.readline()
            m = re.search(r"energy_energy=([-\d.eE+]+)", comment)
            if not m:
                raise ValueError(f"No energy_energy at frame {frame_id}")
            out[frame_id] = float(m.group(1)) / nat
            for _ in range(nat):
                f.readline()
            frame_id += 1
    return out


def _spread_limits(emb: np.ndarray, pad: float = 0.12) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits with fractional padding so points look less cramped."""
    x0, x1 = float(emb[:, 0].min()), float(emb[:, 0].max())
    y0, y1 = float(emb[:, 1].min()), float(emb[:, 1].max())
    dx = (x1 - x0) or 1.0
    dy = (y1 - y0) or 1.0
    return (x0 - pad * dx, x1 + pad * dx), (y0 - pad * dy, y1 + pad * dy)


def plot_figure(
    output: Path,
    *,
    labels: np.ndarray,
    emb_all: dict[float, np.ndarray],
    emb_mix: dict[float, np.ndarray],
    energy_mix: np.ndarray,
    pts_degen: dict[float, np.ndarray],
    sub_centroids: np.ndarray,
    overall_centroid0: np.ndarray,
    centroid1: np.ndarray,
    radius1_x: float,
    radius1_y: float,
) -> None:
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.5, 5.0))
    # Rows: titles+top panels | legend | bottom panels | x-label+colorbar
    gs = GridSpec(
        4,
        5,
        figure=fig,
        height_ratios=[1.0, 0.14, 1.0, 0.26],
        left=0.10,
        right=0.90,
        top=0.93,
        bottom=0.07,
        wspace=0.14,
        hspace=0.20,
    )

    axes_top = [fig.add_subplot(gs[0, c]) for c in range(5)]
    axes_bot = [fig.add_subplot(gs[2, c]) for c in range(5)]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    footer_ax = fig.add_subplot(gs[3, :])
    footer_ax.axis("off")

    col_titles = [
        r"$\alpha$ = 0",
        r"$\alpha$ = 0.25",
        r"$\alpha$ = 0.50",
        r"$\alpha$ = 0.75",
        r"$\alpha$ = 1",
    ]

    lo, hi = np.percentile(energy_mix, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(energy_mix)), float(np.max(energy_mix))

    energy_mappable = None
    marker_all = 22
    marker_mix = 24
    marker_degen = 36

    # Top row: composition
    for col, alpha in enumerate(ALPHAS):
        ax = axes_top[col]
        emb = emb_all[alpha]
        for cat in (0, 1, 2):
            idx = labels == cat
            info = CAT_INFO[cat]
            ax.scatter(
                emb[idx, 0],
                emb[idx, 1],
                s=marker_all,
                color=info["color"],
                marker=info["marker"],
                alpha=0.80,
                linewidths=0,
                rasterized=True,
            )
        xlim, ylim = _spread_limits(emb, pad=0.14)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(col_titles[col], fontsize=10, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom row: energy
    for col, alpha in enumerate(ALPHAS):
        ax = axes_bot[col]
        emb = emb_mix[alpha]
        sc = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            s=marker_mix,
            c=energy_mix,
            cmap="viridis",
            vmin=lo,
            vmax=hi,
            marker="^",
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
        if energy_mappable is None:
            energy_mappable = sc

        pts = pts_degen[alpha]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=marker_degen,
            facecolors="none",
            edgecolors="black",
            marker="^",
            linewidths=1.0,
            zorder=5,
            rasterized=True,
        )
        xlim, ylim = _spread_limits(emb, pad=0.14)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    # Degeneracy callouts on mixed row
    ellipse = mpatches.Ellipse(
        centroid1,
        width=2 * radius1_x,
        height=2 * radius1_y,
        angle=0,
        fill=False,
        edgecolor=CIRCLE_COLOR,
        linewidth=1.1,
        linestyle="--",
        zorder=6,
    )
    axes_bot[4].add_patch(ellipse)
    for sc_pt in sub_centroids:
        axes_bot[0].annotate(
            "",
            xy=overall_centroid0,
            xytext=sc_pt,
            arrowprops=dict(
                arrowstyle="->",
                color=CIRCLE_COLOR,
                lw=1.0,
                mutation_scale=9,
                shrinkA=0,
                shrinkB=5.0,
            ),
        )

    # Shared UMAP 2 on the left; row labels on the right
    fig.text(0.04, 0.58, "UMAP 2", fontsize=10, rotation=90, va="center", ha="center")
    axes_top[-1].annotate(
        "All Frames",
        xy=(1.06, 0.5),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="center",
        rotation=270,
    )
    axes_bot[-1].annotate(
        "Mixed Only",
        xy=(1.06, 0.5),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="center",
        rotation=270,
    )

    # Composition + degeneracy legend in dedicated middle band
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=info["marker"],
            color="w",
            markerfacecolor=info["color"],
            markersize=8,
            label=info["label"],
        )
        for info in CAT_INFO.values()
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=8,
            label="Degenerate Cluster",
        )
    )
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=1.0,
    )

    # Footer: UMAP 1 + horizontal colorbar
    footer_ax.text(0.5, 0.92, "UMAP 1", fontsize=10, ha="center", va="top")
    if energy_mappable is not None:
        cax = footer_ax.inset_axes([0.18, 0.15, 0.64, 0.28])
        cbar = fig.colorbar(energy_mappable, cax=cax, orientation="horizontal")
        cbar.set_label("Energy/Atom", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=10)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Leave a little right pad for row labels when using bbox_inches=tight
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    print(f"Saved → {output}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb_cache = CACHE_DIR / "umap_embeddings_3row.npz"
    meta_cache = CACHE_DIR / "umap_meta_3row.npz"

    # Fast path: rebuild the figure from cached embeddings + per-frame metadata
    # (labels, energies) alone, without the raw histogram trees / 16 MB extxyz
    # (kept out of the repo for size). --recompute forces the full data path.
    if emb_cache.is_file() and meta_cache.is_file() and not args.recompute:
        mz = np.load(meta_cache)
        frame_ids = list(mz["frame_ids"])
        frame_dirs = [f"frame_{i}" for i in frame_ids]
        labels = mz["labels"]
        energy_all = mz["energy_all"]
        alpha_dirs = None
        print(f"Loaded cached metadata ({len(labels)} frames); "
              "skipping raw histogram trees.")
    else:
        alpha_dirs = alpha_dirs_from_transfer(args.transfer_dir)
        for path in alpha_dirs.values():
            if not path.is_dir():
                raise FileNotFoundError(f"Missing alpha directory: {path}")
        if not args.label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {args.label_dir}")
        if not args.extxyz.is_file():
            raise FileNotFoundError(f"Missing extxyz: {args.extxyz}")
        candidate_frames = get_frame_list(alpha_dirs[0.00])
        frame_dirs = [
            f for f in candidate_frames if frame_ok(f, args.label_dir, alpha_dirs)
        ]
        frame_ids = [int(FRAME_RE.match(f).group(1)) for f in frame_dirs]
        labels = np.array(
            [frame_composition_label(f, args.label_dir) for f in frame_dirs]
        )
        e_map = load_energy_per_atom(args.extxyz)
        energy_all = np.array([e_map[fid] for fid in frame_ids])
        np.savez_compressed(meta_cache, frame_ids=np.array(frame_ids),
                            labels=labels, energy_all=energy_all)

    n_frames = len(frame_dirs)
    mixed_mask = labels == 2
    energy_mix = energy_all[mixed_mask]
    print(f"Using {n_frames} frames. Composition — "
          f"Y-only: {(labels == 0).sum()}, Mg-only: {(labels == 1).sum()}, "
          f"Mixed: {(labels == 2).sum()}")
    print(f"Mixed energy/atom range: {energy_mix.min():.3f} … {energy_mix.max():.3f} "
          f"(colorbar uses 2–98%ile)")

    if emb_cache.is_file() and not args.recompute:
        z = np.load(emb_cache)
        emb_all = {a: z[f"all_{a:.2f}"] for a in ALPHAS}
        emb_mix = {a: z[f"mix_{a:.2f}"] for a in ALPHAS}
        print(f"Loaded cached embeddings from {emb_cache}")
    else:
        fp: dict[float, np.ndarray] = {}
        for alpha in ALPHAS:
            raw = np.array([load_fp(alpha_dirs[alpha], f) for f in frame_dirs])
            fp[alpha] = drop_zero_cols(raw)
            print(f"alpha={alpha:.2f}: fingerprint shape {fp[alpha].shape}")

        import umap  # lazy: only needed when recomputing embeddings
        reducer = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
        )
        emb_all = {}
        emb_mix = {}
        for alpha in ALPHAS:
            print(f"UMAP (all frames) alpha={alpha:.2f} ...")
            emb_all[alpha] = reducer.fit_transform(fp[alpha])
            print(f"UMAP (mixed only) alpha={alpha:.2f} ...")
            emb_mix[alpha] = reducer.fit_transform(
                drop_zero_cols(fp[alpha][mixed_mask])
            )
        save = {}
        for a in ALPHAS:
            save[f"all_{a:.2f}"] = emb_all[a]
            save[f"mix_{a:.2f}"] = emb_mix[a]
        np.savez_compressed(emb_cache, **save)
        print(f"Cached embeddings → {emb_cache}")

    km8 = KMeans(n_clusters=8, random_state=args.seed, n_init=10).fit(emb_mix[1.00])
    degen_mask = km8.labels_ == 1
    pts_degen = {alpha: emb_mix[alpha][degen_mask] for alpha in ALPHAS}

    pts0_degen = pts_degen[0.00]
    km_sub = KMeans(n_clusters=3, random_state=args.seed, n_init=10).fit(pts0_degen)
    sub_centroids = np.array(
        [pts0_degen[km_sub.labels_ == k].mean(axis=0) for k in range(3)]
    )
    overall_centroid0 = pts0_degen.mean(axis=0)

    pts1_degen = pts_degen[1.00]
    centroid1 = pts1_degen.mean(axis=0)
    radius1_x = pts1_degen[:, 0].std() * 2.8 + 0.4
    radius1_y = pts1_degen[:, 1].std() * 2.8 + 0.4

    plot_kwargs = dict(
        labels=labels,
        emb_all=emb_all,
        emb_mix=emb_mix,
        energy_mix=energy_mix,
        pts_degen=pts_degen,
        sub_centroids=sub_centroids,
        overall_centroid0=overall_centroid0,
        centroid1=centroid1,
        radius1_x=radius1_x,
        radius1_y=radius1_y,
    )
    plot_figure(args.output, **plot_kwargs)
    plot_figure(WRITING_FIG, **plot_kwargs)


if __name__ == "__main__":
    main()
