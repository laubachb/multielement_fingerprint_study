"""Shared fingerprint / UMAP data loading for CN alpha-sweep analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from make_umap_degeneracy import (
    ALPHAS,
    ALPHA_DIRS,
    CACHE_DIR,
    discover_frames,
    build_dataset,
    cache_is_valid,
    compute_degeneracy,
    compute_embeddings,
    fingerprint_digest,
    load_cache,
    load_fp,
    save_cache,
)


def alpha_dirs_from_root(fingerprints_dir: Path) -> dict[float, Path]:
    return {
        alpha: fingerprints_dir / path.name
        for alpha, path in ALPHA_DIRS.items()
    }


def load_raw_fingerprints(
    alpha_dirs: dict[float, Path],
    frame_dirs: list[str],
) -> dict[float, np.ndarray]:
    """Full 300-bin histogram vectors (before zero-column dropping)."""
    return {
        alpha: np.array([load_fp(alpha_dirs[alpha], frame) for frame in frame_dirs])
        for alpha in ALPHAS
    }


def ensure_umap_data(
    *,
    fingerprints_dir: Path,
    cache_dir: Path = CACHE_DIR,
    recompute: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Return frame list, raw/dropped fingerprints, and per-alpha UMAP embeddings.

    Uses ./cache/ when valid; otherwise computes and writes cache via
    make_umap_degeneracy helpers.
    """
    alpha_dirs = alpha_dirs_from_root(fingerprints_dir)
    for path in alpha_dirs.values():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing alpha directory: {path}")

    umap_params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "seed": seed,
    }

    frame_dirs = discover_frames(alpha_dirs)
    if not frame_dirs:
        raise RuntimeError("No valid frames discovered.")

    digest = fingerprint_digest(alpha_dirs, frame_dirs)
    cached = None if recompute else load_cache(cache_dir)

    if cached is not None and cache_is_valid(cached, frame_dirs, digest, umap_params):
        print(f"Loading cached data from {cache_dir}")
        fp = cached["fp"]
        labels = cached["labels"]
        emb = cached["emb_mix"]
    else:
        dataset = build_dataset(alpha_dirs)
        frame_dirs = dataset["frame_dirs"]
        labels = dataset["labels"]
        fp = dataset["fp"]
        digest = fingerprint_digest(alpha_dirs, frame_dirs)
        emb_all, emb_mix, fp_mix, _ = compute_embeddings(
            fp, labels, n_neighbors, min_dist, seed
        )
        emb = emb_mix
        degen = compute_degeneracy(emb_mix)
        save_cache(
            cache_dir,
            frame_dirs=frame_dirs,
            labels=labels,
            fp=fp,
            fp_mix=fp_mix,
            emb_all=emb_all,
            emb_mix=emb_mix,
            digest=digest,
            umap_params=umap_params,
            degen_mask=degen["degen_mask"],
            sub_centroids=degen["sub_centroids"],
            overall_centroid0=degen["overall_centroid0"],
            centroid1=degen["centroid1"],
            radius1_x=degen["radius1_x"],
            radius1_y=degen["radius1_y"],
        )
        print(f"Wrote cache to {cache_dir}")

    frame_ids = np.array(
        [int(name.split("_")[1]) for name in frame_dirs],
        dtype=int,
    )
    fp_raw = load_raw_fingerprints(alpha_dirs, frame_dirs)

    return {
        "frame_dirs": frame_dirs,
        "frame_ids": frame_ids,
        "labels": labels,
        "fp": fp,
        "fp_raw": fp_raw,
        "emb": emb,
        "umap_params": umap_params,
    }
