#!/usr/bin/env python3
"""
Farthest-point sampling (FPS) on HEA CN-fingerprint vectors across ChIMES α spaces.

Uses histogram trees under hea_study/alpha_*-histograms/ (~535 valid frames:
Y-only, Mg-only, and mixed MgY). Composition labels come from alpha_0-histograms
cluster files.

Usage
-----
  cd hea_study/sampling
  python run_fps_sampling.py
  python run_fps_sampling.py --retentions 0.05 0.10 0.15 --replicates 10
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
HEA_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results_convergence_5pct"

ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
ALPHA_DIR_NAMES = {
    0.00: "alpha_0-histograms",
    0.25: "alpha_025-histograms",
    0.50: "alpha_050-histograms",
    0.75: "alpha_075-histograms",
    1.00: "alpha_1-histograms",
}
HIST_FILES = [
    "0-0.2b_clu-s.hist",
    "0-0.3b_clu-s.hist",
    "0-0.4b_clu-s.hist",
]
CLUSTER_FILE = "0.all-2b-clusters.txt"
FRAME_RE = re.compile(r"^frame_(\d+)$")

DEFAULT_RETENTIONS = tuple(i / 100 for i in range(5, 100, 5))
DEFAULT_REPLICATES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hea-root",
        type=Path,
        default=HEA_DIR,
        help="hea_study root (contains alpha_*-histograms/).",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--retentions",
        type=float,
        nargs="+",
        default=list(DEFAULT_RETENTIONS),
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frames-file",
        type=Path,
        default=None,
        help="Optional text file of frame_N names (one per line) restricting the FPS pool.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional split_manifest.json (records train/test provenance in manifest).",
    )
    return parser.parse_args()


def alpha_dirs_from_root(hea_root: Path) -> dict[float, Path]:
    return {a: hea_root / name for a, name in ALPHA_DIR_NAMES.items()}


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


def discover_frames(label_dir: Path, alpha_dirs: dict[float, Path]) -> list[str]:
    return [f for f in get_frame_list(label_dir) if frame_ok(f, label_dir, alpha_dirs)]


def load_fp(alpha_dir: Path, frame: str) -> np.ndarray:
    fdir = alpha_dir / frame
    return np.concatenate([np.loadtxt(fdir / h)[:, 1] for h in HIST_FILES])


def drop_zero_cols(arr: np.ndarray) -> np.ndarray:
    return arr[:, arr.any(axis=0)]


def retention_count(n_frames: int, fraction: float) -> int:
    return max(1, int(round(n_frames * fraction)))


def pct_label(fraction: float) -> str:
    return f"pct_{int(round(fraction * 100)):03d}"


def dir_alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:.2f}"


def farthest_point_sampling(
    vectors: np.ndarray, k: int, rng: np.random.Generator
) -> list[int]:
    n = vectors.shape[0]
    k = min(k, n)
    if k == 1:
        return [int(rng.integers(0, n))]
    start = int(rng.integers(0, n))
    selected = [start]
    min_dists = np.linalg.norm(vectors - vectors[start], axis=1)
    for _ in range(1, k):
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        min_dists = np.minimum(
            min_dists, np.linalg.norm(vectors - vectors[farthest], axis=1)
        )
    return selected


def write_replicate(
    out_dir: Path,
    *,
    frame_dirs: list[str],
    selected_indices: list[int],
    alpha: float,
    retention: float,
    replicate: int,
    seed: int,
    fingerprint_shape: tuple[int, int],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_frames = [frame_dirs[i] for i in selected_indices]
    (out_dir / "selected_frames.txt").write_text(
        "\n".join(selected_frames) + "\n", encoding="utf-8"
    )
    metadata = {
        "alpha": alpha,
        "retention_fraction": retention,
        "n_total_frames": len(frame_dirs),
        "n_selected": len(selected_frames),
        "replicate": replicate,
        "seed": seed,
        "fingerprint_shape": list(fingerprint_shape),
        "selected_frame_ids": [int(f.split("_")[1]) for f in selected_frames],
        "selected_indices": selected_indices,
        "system": "hea",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "retention_pct": int(round(retention * 100)),
        "n_selected": len(selected_frames),
    }


def main() -> None:
    args = parse_args()
    alpha_dirs = alpha_dirs_from_root(args.hea_root)
    label_dir = alpha_dirs[0.00]

    for path in alpha_dirs.values():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing: {path}")

    if args.frames_file is not None:
        allowed = {
            line.strip()
            for line in args.frames_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        frame_dirs = sorted(
            [f for f in discover_frames(label_dir, alpha_dirs) if f in allowed],
            key=lambda x: int(FRAME_RE.match(x).group(1)),
        )
        if len(frame_dirs) != len(allowed):
            missing = sorted(allowed - set(frame_dirs))
            raise RuntimeError(
                f"frames-file has {len(allowed)} names but only {len(frame_dirs)} "
                f"are valid; missing e.g. {missing[:5]}"
            )
        print(f"Restricted FPS pool to {len(frame_dirs)} frames from {args.frames_file}")
    else:
        frame_dirs = discover_frames(label_dir, alpha_dirs)
    if not frame_dirs:
        raise RuntimeError("No valid HEA frames discovered.")
    n_frames = len(frame_dirs)
    print(f"Using {n_frames} HEA frames (Y / Mg / mixed).")

    fingerprints: dict[float, np.ndarray] = {}
    for alpha in ALPHAS:
        raw = np.array([load_fp(alpha_dirs[alpha], f) for f in frame_dirs])
        fingerprints[alpha] = drop_zero_cols(raw)
        print(f"alpha={alpha:.2f}: fingerprint shape {fingerprints[alpha].shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    manifest = {
        "system": "hea",
        "n_frames": n_frames,
        "frame_dirs": frame_dirs,
        "alphas": ALPHAS,
        "retentions": args.retentions,
        "replicates": args.replicates,
        "base_seed": args.seed,
        "runs": [],
    }
    if args.frames_file is not None:
        manifest["frames_file"] = str(args.frames_file)
    if args.split_manifest is not None:
        manifest["split_manifest"] = str(args.split_manifest)
        split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        manifest["holdout_fraction"] = split.get("holdout_fraction")
        manifest["n_train_pool"] = split.get("n_train")
        manifest["n_test_holdout"] = split.get("n_test")

    print(f"\nHEA FPS: {n_frames} frames")
    print(f"Retentions: {[f'{r * 100:g}%' for r in args.retentions]}")
    print(f"Replicates: {args.replicates}\n")

    for alpha in ALPHAS:
        vectors = fingerprints[alpha]
        alpha_out = args.output_dir / dir_alpha_label(alpha)
        for retention in args.retentions:
            k = retention_count(n_frames, retention)
            pct_dir = alpha_out / pct_label(retention)
            print(f"alpha={alpha:.2f}  retention={retention * 100:g}%  k={k}")
            for replicate in range(args.replicates):
                seed = args.seed + replicate
                rng = np.random.default_rng(seed)
                selected_indices = farthest_point_sampling(vectors, k, rng)
                rep_dir = pct_dir / f"replicate_{replicate:02d}"
                row = write_replicate(
                    rep_dir,
                    frame_dirs=frame_dirs,
                    selected_indices=selected_indices,
                    alpha=alpha,
                    retention=retention,
                    replicate=replicate,
                    seed=seed,
                    fingerprint_shape=vectors.shape,
                )
                summary_rows.append(
                    {
                        "alpha": alpha,
                        "retention_fraction": retention,
                        "retention_pct": row["retention_pct"],
                        "replicate": replicate,
                        "seed": seed,
                        "n_total_frames": n_frames,
                        "n_selected": k,
                        "output_dir": str(rep_dir.relative_to(args.output_dir)),
                    }
                )
                manifest["runs"].append(
                    {
                        "alpha": alpha,
                        "retention_fraction": retention,
                        "replicate": replicate,
                        "seed": seed,
                        "n_selected": k,
                        "path": str(rep_dir.relative_to(args.output_dir)),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(summary_rows)} replicate sets to {args.output_dir}")


if __name__ == "__main__":
    main()
