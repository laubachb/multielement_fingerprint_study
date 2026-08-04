#!/usr/bin/env python3
"""
2×5 extension of the CN umap_degeneracy figure for HEA fingerprints.

Histogram data are read from transfer_to_local-Apr2026/ (five α directories).
Composition labels (Y-only, Mg-only, Mixed) come from cluster files in
alpha_0-histograms/ (not included in the transfer bundle).

Layout
------
  Row 0 – all frames     (Y-only, Mg-only, Mixed colored)
  Row 1 – mixed frames only

  Col 0  alpha=0   (comp. only)
  Col 1  alpha=0.25
  Col 2  alpha=0.50
  Col 3  alpha=0.75
  Col 4  alpha=1   (struct. only)

Degeneracy annotations
  • Degenerate cluster defined by KMeans k=8 on the alpha=1 mixed UMAP.
  • Gold markers highlight those frames in ALL five bottom panels.
  • Black ellipse drawn only on the alpha=1 bottom panel (col 4).
  • Diverging arrows drawn only on the alpha=0 bottom panel (col 0).

Output
------
./umap_degeneracy_expanded.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
# umap is imported lazily inside main() only when recomputing embeddings; with the
# cached embeddings present the figure regenerates without the umap package.
from sklearn.cluster import KMeans

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SCRIPT_DIR = Path(__file__).resolve().parent
HEA_DIR = SCRIPT_DIR.parent
DEFAULT_TRANSFER_DIR = HEA_DIR / "transfer_to_local-Apr2026"
DEFAULT_LABEL_DIR = HEA_DIR / "alpha_0-histograms"

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
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=DEFAULT_TRANSFER_DIR,
        help="Directory containing alpha_*-histograms fingerprint data.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help="Directory with cluster files for composition labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "umap_degeneracy_expanded.png",
        help="Output figure path.",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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
    """Return 0=Y-only, 1=Mg-only, 2=Mixed."""
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
    return np.concatenate([
        np.loadtxt(fdir / hist_name)[:, 1]
        for hist_name in HIST_FILES
    ])


def drop_zero_cols(arr: np.ndarray) -> np.ndarray:
    return arr[:, arr.any(axis=0)]


def plot_figure(
    output: Path,
    *,
    labels: np.ndarray,
    emb_all: dict[float, np.ndarray],
    emb_mix: dict[float, np.ndarray],
    labels_mixed: np.ndarray,
    pts_degen: dict[float, np.ndarray],
    sub_centroids: np.ndarray,
    overall_centroid0: np.ndarray,
    centroid1: np.ndarray,
    radius1_x: float,
    radius1_y: float,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.subplots_adjust(left=0.06, right=0.95, top=0.91, bottom=0.16,
                        wspace=0.30, hspace=0.50)

    col_titles = [
        r"$\alpha = 0$" + "\n(comp. only)",
        r"$\alpha = 0.25$",
        r"$\alpha = 0.50$",
        r"$\alpha = 0.75$",
        r"$\alpha = 1$" + "\n(struct. only)",
    ]
    row_labels = ["All frames", "Mixed only"]

    for row, _row_ylabel in enumerate(row_labels):
        for col, alpha in enumerate(ALPHAS):
            ax = axes[row, col]
            if row == 0:
                emb = emb_all[alpha]
                row_labels_plot = labels
                cats = [0, 1, 2]
            else:
                emb = emb_mix[alpha]
                row_labels_plot = labels_mixed
                cats = [2]

            for cat in cats:
                idx = row_labels_plot == cat
                info = CAT_INFO[cat]
                ax.scatter(
                    emb[idx, 0], emb[idx, 1],
                    s=18, color=info["color"], marker=info["marker"],
                    alpha=0.75, linewidths=0,
                    rasterized=True,
                )

            if row == 0:
                ax.set_title(col_titles[col], fontsize=8.5, pad=3)
            ax.set_xlabel("UMAP 1", fontsize=8, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row, 0].set_ylabel("UMAP 2", fontsize=8, labelpad=2)

    for row, lbl in enumerate(row_labels):
        axes[row, 4].annotate(
            lbl, xy=(1.04, 0.5), xycoords="axes fraction",
            fontsize=9, ha="left", va="center", rotation=270,
        )

    for col, alpha in enumerate(ALPHAS):
        ax = axes[1, col]
        pts = pts_degen[alpha]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=30, color=DEGEN_COLOR, marker="^",
            linewidths=0.6, edgecolors="black",
            zorder=5, rasterized=True,
        )

    ellipse = mpatches.Ellipse(
        centroid1, width=2 * radius1_x, height=2 * radius1_y,
        angle=0, fill=False,
        edgecolor=CIRCLE_COLOR, linewidth=1.5, linestyle="--", zorder=6,
    )
    axes[1, 4].add_patch(ellipse)

    for sc in sub_centroids:
        axes[1, 0].annotate(
            "", xy=overall_centroid0, xytext=sc,
            arrowprops=dict(
                arrowstyle="->", color=CIRCLE_COLOR,
                lw=1.2, mutation_scale=10,
                shrinkA=0, shrinkB=6.4,
            ),
        )

    legend_handles = [
        plt.Line2D([0], [0], marker=info["marker"], color="w",
                   markerfacecolor=info["color"], markersize=7,
                   label=info["label"])
        for info in CAT_INFO.values()
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=DEGEN_COLOR, markeredgecolor="black",
                   markeredgewidth=0.6, markersize=7,
                   label="Degenerate cluster")
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.00),
        ncol=4,
        fontsize=9,
        framealpha=0.95,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output}")


def main() -> None:
    args = parse_args()
    cache_dir = SCRIPT_DIR / "cache"
    emb_cache = cache_dir / "umap_embeddings_3row.npz"
    meta_cache = cache_dir / "umap_meta_3row.npz"
    recompute = getattr(args, "recompute", False)

    # Fast path: rebuild the figure from cached embeddings + per-frame labels alone,
    # without the raw histogram trees or the umap package (kept out for size/deps).
    if emb_cache.is_file() and meta_cache.is_file() and not recompute:
        z = np.load(emb_cache)
        mz = np.load(meta_cache)
        emb_all = {a: z[f"all_{a:.2f}"] for a in ALPHAS}
        emb_mix = {a: z[f"mix_{a:.2f}"] for a in ALPHAS}
        labels = mz["labels"]
        print(f"Loaded cached embeddings + labels ({len(labels)} frames); "
              "skipping raw histogram trees.")
    else:
        alpha_dirs = alpha_dirs_from_transfer(args.transfer_dir)
        for path in alpha_dirs.values():
            if not path.is_dir():
                raise FileNotFoundError(f"Missing alpha directory: {path}")
        if not args.label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {args.label_dir}")
        candidate_frames = get_frame_list(alpha_dirs[0.00])
        frame_dirs = [f for f in candidate_frames if frame_ok(f, args.label_dir, alpha_dirs)]
        labels = np.array([frame_composition_label(f, args.label_dir) for f in frame_dirs])
        fp: dict[float, np.ndarray] = {}
        for alpha in ALPHAS:
            raw = np.array([load_fp(alpha_dirs[alpha], f) for f in frame_dirs])
            fp[alpha] = drop_zero_cols(raw)
        import umap  # lazy: only needed when recomputing embeddings
        reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                            random_state=args.seed)
        mm = labels == 2
        emb_all, emb_mix = {}, {}
        for alpha in ALPHAS:
            emb_all[alpha] = reducer.fit_transform(fp[alpha])
            emb_mix[alpha] = reducer.fit_transform(drop_zero_cols(fp[alpha][mm]))
        cache_dir.mkdir(parents=True, exist_ok=True)
        save = {}
        for a in ALPHAS:
            save[f"all_{a:.2f}"] = emb_all[a]; save[f"mix_{a:.2f}"] = emb_mix[a]
        np.savez_compressed(emb_cache, **save)
        np.savez_compressed(meta_cache, labels=labels)

    print("Composition counts — "
          f"Y-only: {(labels == 0).sum()}, Mg-only: {(labels == 1).sum()}, "
          f"Mixed: {(labels == 2).sum()}")
    mixed_mask = labels == 2
    labels_mixed = labels[mixed_mask]

    km8 = KMeans(n_clusters=8, random_state=args.seed, n_init=10).fit(emb_mix[1.00])
    degen_mask = km8.labels_ == 1
    pts_degen = {alpha: emb_mix[alpha][degen_mask] for alpha in ALPHAS}

    pts0_degen = pts_degen[0.00]
    km_sub = KMeans(n_clusters=3, random_state=args.seed, n_init=10).fit(pts0_degen)
    sub_centroids = np.array([
        pts0_degen[km_sub.labels_ == k].mean(axis=0) for k in range(3)
    ])
    overall_centroid0 = pts0_degen.mean(axis=0)

    pts1_degen = pts_degen[1.00]
    centroid1 = pts1_degen.mean(axis=0)
    radius1_x = pts1_degen[:, 0].std() * 2.8 + 0.4
    radius1_y = pts1_degen[:, 1].std() * 2.8 + 0.4

    print(
        f"Degenerate cluster: n={degen_mask.sum()}, "
        f"centroid_a1=({centroid1[0]:.2f}, {centroid1[1]:.2f})"
    )

    plot_figure(
        args.output,
        labels=labels,
        emb_all=emb_all,
        emb_mix=emb_mix,
        labels_mixed=labels_mixed,
        pts_degen=pts_degen,
        sub_centroids=sub_centroids,
        overall_centroid0=overall_centroid0,
        centroid1=centroid1,
        radius1_x=radius1_x,
        radius1_y=radius1_y,
    )


if __name__ == "__main__":
    main()
