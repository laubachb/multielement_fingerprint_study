#!/usr/bin/env python3
"""
1×5 UMAP degeneracy figure for C/N fingerprint alpha sweep.

Data are read from ../a*_fingerprints/ (alpha = 0, 0.25, 0.50, 0.75, 1).

Layout
------
  Single row – all C+N frames (the full valid frame set)

  Col 0  alpha=0   (comp. only)
  Col 1  alpha=0.25
  Col 2  alpha=0.50
  Col 3  alpha=0.75
  Col 4  alpha=1   (struct. only)

Degeneracy annotations
  • Degenerate cluster defined by KMeans k=8 on the alpha=1 UMAP.
  • Gold markers highlight those frames in all five panels.
  • Black ellipse drawn only on the alpha=1 panel (col 4).
  • Diverging arrows drawn only on the alpha=0 panel (col 0).

Caching
-------
UMAP embeddings and fingerprint matrices are written to ./cache/ and reused
on subsequent runs unless --recompute is passed or input histograms change.

Output
------
./umap_degeneracy.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SCRIPT_DIR = Path(__file__).resolve().parent
FINGERPRINTS_DIR = SCRIPT_DIR.parents[2] / "fingerprints"
CACHE_DIR = SCRIPT_DIR / "cache"

ALPHA_DIRS = {
    0.00: FINGERPRINTS_DIR / "a000_fingerprints",
    0.25: FINGERPRINTS_DIR / "a025_fingerprints",
    0.50: FINGERPRINTS_DIR / "a050_fingerprints",
    0.75: FINGERPRINTS_DIR / "a075_fingerprints",
    1.00: FINGERPRINTS_DIR / "a100_fingerprints",
}
ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]

HIST_FILES = [
    "0-0.2b_clu-s.hist",
    "0-0.3b_clu-s.hist",
    "0-0.4b_clu-s.hist",
]
CLUSTER_FILE = "0.all-2b-clusters.txt"
FRAME_RE = re.compile(r"^frame_(\d+)$")

CAT_INFO = {
    0: dict(label="C only", color="#2166ac", marker="o"),
    1: dict(label="N only", color="#d6604d", marker="s"),
    2: dict(label="C+N", color="#4dac26", marker="^"),
}
DEGEN_COLOR = "#e6ac00"
CIRCLE_COLOR = "black"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fingerprints-dir",
        type=Path,
        default=FINGERPRINTS_DIR,
        help="Directory containing a*_fingerprints folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "umap_degeneracy.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Directory for cached fingerprints and UMAP embeddings.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cache and recompute fingerprints and UMAP embeddings.",
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
    """Return 0=C-only, 1=N-only, 2=C+N from cluster file in label_dir."""
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


def fingerprint_digest(alpha_dirs: dict[float, Path], frame_dirs: list[str]) -> str:
    digest = hashlib.sha256()
    for alpha in ALPHAS:
        alpha_dir = alpha_dirs[alpha]
        for frame in frame_dirs:
            for hist_name in HIST_FILES:
                path = alpha_dir / frame / hist_name
                digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "manifest": cache_dir / "manifest.json",
        "fingerprints": cache_dir / "fingerprints.npz",
        "embeddings": cache_dir / "embeddings.npz",
        "degeneracy": cache_dir / "degeneracy.json",
    }


def load_cache(cache_dir: Path) -> dict | None:
    paths = cache_paths(cache_dir)
    if not all(p.exists() for p in paths.values()):
        return None

    with paths["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    fp_data = np.load(paths["fingerprints"])
    emb_data = np.load(paths["embeddings"])
    with paths["degeneracy"].open("r", encoding="utf-8") as handle:
        degen = json.load(handle)

    fp = {float(k): fp_data[f"alpha_{k}"] for k in manifest["alpha_keys"]}
    emb_all = {float(k): emb_data[f"emb_all_{k}"] for k in manifest["alpha_keys"]}
    emb_mix = {float(k): emb_data[f"emb_mix_{k}"] for k in manifest["alpha_keys"]}
    fp_mix = {float(k): fp_data[f"fp_mix_{k}"] for k in manifest["alpha_keys"]}

    return {
        "manifest": manifest,
        "frame_dirs": manifest["frame_dirs"],
        "labels": np.array(manifest["labels"], dtype=int),
        "fp": fp,
        "fp_mix": fp_mix,
        "emb_all": emb_all,
        "emb_mix": emb_mix,
        "degen_mask": np.array(degen["degen_mask"], dtype=bool),
        "sub_centroids": np.array(degen["sub_centroids"]),
        "overall_centroid0": np.array(degen["overall_centroid0"]),
        "centroid1": np.array(degen["centroid1"]),
        "radius1_x": degen["radius1_x"],
        "radius1_y": degen["radius1_y"],
    }


def save_cache(
    cache_dir: Path,
    *,
    frame_dirs: list[str],
    labels: np.ndarray,
    fp: dict[float, np.ndarray],
    fp_mix: dict[float, np.ndarray],
    emb_all: dict[float, np.ndarray],
    emb_mix: dict[float, np.ndarray],
    digest: str,
    umap_params: dict,
    degen_mask: np.ndarray,
    sub_centroids: np.ndarray,
    overall_centroid0: np.ndarray,
    centroid1: np.ndarray,
    radius1_x: float,
    radius1_y: float,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)

    alpha_keys = [f"{alpha:.2f}" for alpha in ALPHAS]
    manifest = {
        "frame_dirs": frame_dirs,
        "labels": labels.tolist(),
        "alpha_keys": alpha_keys,
        "digest": digest,
        "umap_params": umap_params,
        "n_frames": len(frame_dirs),
    }
    with paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    fp_payload = {}
    emb_payload = {}
    for alpha in ALPHAS:
        key = f"{alpha:.2f}"
        fp_payload[f"alpha_{key}"] = fp[alpha]
        fp_payload[f"fp_mix_{key}"] = fp_mix[alpha]
        emb_payload[f"emb_all_{key}"] = emb_all[alpha]
        emb_payload[f"emb_mix_{key}"] = emb_mix[alpha]
    np.savez_compressed(paths["fingerprints"], **fp_payload)
    np.savez_compressed(paths["embeddings"], **emb_payload)

    degen = {
        "degen_mask": degen_mask.tolist(),
        "sub_centroids": sub_centroids.tolist(),
        "overall_centroid0": overall_centroid0.tolist(),
        "centroid1": centroid1.tolist(),
        "radius1_x": float(radius1_x),
        "radius1_y": float(radius1_y),
    }
    with paths["degeneracy"].open("w", encoding="utf-8") as handle:
        json.dump(degen, handle, indent=2)


def compute_embeddings(
    fp: dict[float, np.ndarray],
    labels: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> tuple[dict, dict, dict, umap.UMAP]:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    mixed_mask = labels == 2

    emb_all: dict[float, np.ndarray] = {}
    emb_mix: dict[float, np.ndarray] = {}
    fp_mix: dict[float, np.ndarray] = {}

    for alpha in ALPHAS:
        print(f"UMAP (all frames) alpha={alpha:.2f} ...")
        emb_all[alpha] = reducer.fit_transform(fp[alpha])

        fp_mix_raw = fp[alpha][mixed_mask]
        fp_mix[alpha] = drop_zero_cols(fp_mix_raw)
        print(f"UMAP (C+N only) alpha={alpha:.2f} ...")
        emb_mix[alpha] = reducer.fit_transform(fp_mix[alpha])

    return emb_all, emb_mix, fp_mix, reducer


def compute_degeneracy(emb_mix: dict[float, np.ndarray]) -> dict:
    km8 = KMeans(n_clusters=8, random_state=42, n_init=10).fit(emb_mix[1.00])
    degen_mask = km8.labels_ == 1

    pts_degen = {alpha: emb_mix[alpha][degen_mask] for alpha in ALPHAS}
    pts0_degen = pts_degen[0.00]
    km_sub = KMeans(n_clusters=3, random_state=42, n_init=10).fit(pts0_degen)
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

    return {
        "degen_mask": degen_mask,
        "pts_degen": pts_degen,
        "sub_centroids": sub_centroids,
        "overall_centroid0": overall_centroid0,
        "centroid1": centroid1,
        "radius1_x": radius1_x,
        "radius1_y": radius1_y,
    }


def build_dataset(alpha_dirs: dict[float, Path]) -> dict:
    label_dir = alpha_dirs[0.00]
    candidate_frames = get_frame_list(label_dir)
    frame_dirs = [f for f in candidate_frames if frame_ok(f, label_dir, alpha_dirs)]
    n_frames = len(frame_dirs)
    print(f"Using {n_frames} frames (missing/NaN frames dropped).")

    labels = np.array([frame_composition_label(f, label_dir) for f in frame_dirs])
    print(
        "Composition counts — "
        f"C-only: {(labels == 0).sum()}, "
        f"N-only: {(labels == 1).sum()}, "
        f"C+N: {(labels == 2).sum()}"
    )

    fp: dict[float, np.ndarray] = {}
    for alpha in ALPHAS:
        raw = np.array([load_fp(alpha_dirs[alpha], f) for f in frame_dirs])
        fp[alpha] = drop_zero_cols(raw)
        print(f"alpha={alpha:.2f}: fingerprint shape {fp[alpha].shape}")

    pd.DataFrame(fp[0.00]).to_csv(SCRIPT_DIR / "fingerprints_alpha0.csv", index=False)
    pd.DataFrame(fp[1.00]).to_csv(SCRIPT_DIR / "fingerprints_alpha1.csv", index=False)

    return {
        "frame_dirs": frame_dirs,
        "labels": labels,
        "fp": fp,
    }


def plot_figure(
    output: Path,
    *,
    labels: np.ndarray,
    emb_all: dict[float, np.ndarray],
    emb_mix: dict[float, np.ndarray],
    pts_degen: dict[float, np.ndarray],
    sub_centroids: np.ndarray,
    overall_centroid0: np.ndarray,
    centroid1: np.ndarray,
    radius1_x: float,
    radius1_y: float,
) -> None:
    del labels, emb_all

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    fig.subplots_adjust(left=0.06, right=0.95, top=0.82, bottom=0.22,
                        wspace=0.30)

    col_titles = [
        r"$\alpha = 0$" + "\n(comp. only)",
        r"$\alpha = 0.25$",
        r"$\alpha = 0.50$",
        r"$\alpha = 0.75$",
        r"$\alpha = 1$" + "\n(struct. only)",
    ]
    cn_info = CAT_INFO[2]

    for col, alpha in enumerate(ALPHAS):
        ax = axes[col]
        emb = emb_mix[alpha]

        ax.scatter(
            emb[:, 0], emb[:, 1],
            s=18, color=cn_info["color"], marker=cn_info["marker"],
            alpha=0.75, linewidths=0,
            rasterized=True,
        )

        pts = pts_degen[alpha]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=30, color=DEGEN_COLOR, marker="^",
            linewidths=0.6, edgecolors="black",
            zorder=5, rasterized=True,
        )

        ax.set_title(col_titles[col], fontsize=8.5, pad=3)
        ax.set_xlabel("UMAP 1", fontsize=8, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_ylabel("UMAP 2", fontsize=8, labelpad=2)

    ellipse = mpatches.Ellipse(
        centroid1, width=2 * radius1_x, height=2 * radius1_y,
        angle=0, fill=False,
        edgecolor=CIRCLE_COLOR, linewidth=1.5, linestyle="--", zorder=6,
    )
    axes[4].add_patch(ellipse)

    for sc in sub_centroids:
        axes[0].annotate(
            "", xy=overall_centroid0, xytext=sc,
            arrowprops=dict(
                arrowstyle="->", color=CIRCLE_COLOR,
                lw=1.2, mutation_scale=10,
                shrinkA=0, shrinkB=6.4,
            ),
        )

    legend_handles = [
        plt.Line2D([0], [0], marker=cn_info["marker"], color="w",
                   markerfacecolor=cn_info["color"], markersize=7,
                   label=cn_info["label"]),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor=DEGEN_COLOR, markeredgecolor="black",
                   markeredgewidth=0.6, markersize=7,
                   label="Degenerate cluster"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.00),
        ncol=2,
        fontsize=9,
        framealpha=0.95,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved → {output}")


def discover_frames(alpha_dirs: dict[float, Path]) -> list[str]:
    label_dir = alpha_dirs[0.00]
    candidate_frames = get_frame_list(label_dir)
    return [f for f in candidate_frames if frame_ok(f, label_dir, alpha_dirs)]


def cache_is_valid(
    cached: dict,
    frame_dirs: list[str],
    digest: str,
    umap_params: dict,
) -> bool:
    manifest = cached["manifest"]
    return (
        manifest["digest"] == digest
        and manifest["umap_params"] == umap_params
        and manifest["frame_dirs"] == frame_dirs
    )


def main() -> None:
    args = parse_args()
    alpha_dirs = {
        alpha: (args.fingerprints_dir / path.name)
        for alpha, path in ALPHA_DIRS.items()
    }
    for alpha, path in alpha_dirs.items():
        if not path.is_dir():
            raise SystemExit(f"Missing alpha directory: {path}")

    umap_params = {
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "seed": args.seed,
    }

    frame_dirs = discover_frames(alpha_dirs)
    if not frame_dirs:
        raise SystemExit("No valid frames discovered.")
    digest = fingerprint_digest(alpha_dirs, frame_dirs)

    cached = None if args.recompute else load_cache(args.cache_dir)
    if cached is not None and cache_is_valid(cached, frame_dirs, digest, umap_params):
        print(f"Loading cached UMAP embeddings from {args.cache_dir}")
        labels = cached["labels"]
        emb_all = cached["emb_all"]
        emb_mix = cached["emb_mix"]
        degen_mask = cached["degen_mask"]
        sub_centroids = cached["sub_centroids"]
        overall_centroid0 = cached["overall_centroid0"]
        centroid1 = cached["centroid1"]
        radius1_x = cached["radius1_x"]
        radius1_y = cached["radius1_y"]
        pts_degen = {alpha: emb_mix[alpha][degen_mask] for alpha in ALPHAS}
    else:
        dataset = build_dataset(alpha_dirs)
        frame_dirs = dataset["frame_dirs"]
        labels = dataset["labels"]
        fp = dataset["fp"]
        digest = fingerprint_digest(alpha_dirs, frame_dirs)

        emb_all, emb_mix, fp_mix, _ = compute_embeddings(
            fp, labels, args.n_neighbors, args.min_dist, args.seed
        )
        degen = compute_degeneracy(emb_mix)
        degen_mask = degen["degen_mask"]
        pts_degen = degen["pts_degen"]
        sub_centroids = degen["sub_centroids"]
        overall_centroid0 = degen["overall_centroid0"]
        centroid1 = degen["centroid1"]
        radius1_x = degen["radius1_x"]
        radius1_y = degen["radius1_y"]

        save_cache(
            args.cache_dir,
            frame_dirs=frame_dirs,
            labels=labels,
            fp=fp,
            fp_mix=fp_mix,
            emb_all=emb_all,
            emb_mix=emb_mix,
            digest=digest,
            umap_params=umap_params,
            degen_mask=degen_mask,
            sub_centroids=sub_centroids,
            overall_centroid0=overall_centroid0,
            centroid1=centroid1,
            radius1_x=radius1_x,
            radius1_y=radius1_y,
        )
        print(f"Cached fingerprints and UMAP embeddings in {args.cache_dir}")

    plot_figure(
        args.output.resolve(),
        labels=labels,
        emb_all=emb_all,
        emb_mix=emb_mix,
        pts_degen=pts_degen,
        sub_centroids=sub_centroids,
        overall_centroid0=overall_centroid0,
        centroid1=centroid1,
        radius1_x=radius1_x,
        radius1_y=radius1_y,
    )


if __name__ == "__main__":
    main()
