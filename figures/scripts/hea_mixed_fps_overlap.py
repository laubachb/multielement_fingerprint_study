#!/usr/bin/env python3
"""
HEA FPS subset overlap vs retention, restricted to Mixed-composition frames only.

Runs FPS in each α fingerprint space on the mixed subset (~126 frames), then
plots Jaccard overlap for α ∈ {0, 0.25, 0.50, 0.75} vs α = 1 (same style as
the full-corpus HEA / CN figures).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HEA_DIR = ROOT / "hea_study"
FIG_DIR = ROOT / "writing" / "figures"
OUT_DIR = Path(__file__).resolve().parent / "output"
RESULTS_DIR = HEA_DIR / "sampling" / "results_convergence_5pct_mixed"

sys.path.insert(0, str(HEA_DIR / "fps_alpha_probe"))
from probe_data import (  # noqa: E402
    ALPHAS,
    build_frame_composition_table,
    load_fingerprint_matrices,
)

sys.path.insert(0, str(HEA_DIR / "sampling"))
from run_fps_sampling import farthest_point_sampling, pct_label, retention_count  # noqa: E402

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def run_mixed_fps(
    *,
    results_dir: Path = RESULTS_DIR,
    replicates: int = 10,
    seed: int = 42,
    retentions: tuple[float, ...] = tuple(i / 100 for i in range(5, 100, 5)),
) -> None:
    hea_fp_cache = OUT_DIR / "hea_fingerprints.npz"
    if hea_fp_cache.is_file():
        z = np.load(hea_fp_cache, allow_pickle=True)
        frame_ids = np.asarray(z["frame_ids"])
        matrices = {float(a): z[f"alpha_{a:.2f}"] for a in ALPHAS}
        print(f"Loaded fingerprint cache ({len(frame_ids)} frames)")
    else:
        frame_ids, matrices = load_fingerprint_matrices(HEA_DIR)
        frame_ids = np.asarray(frame_ids)

    comp = build_frame_composition_table()
    comp = comp[comp["frame_id"].isin(frame_ids)].copy()
    order = {int(fid): i for i, fid in enumerate(frame_ids.tolist())}
    comp = comp.sort_values("frame_id")
    idx_all = np.array([order[int(fid)] for fid in comp["frame_id"].tolist()])
    mixed_mask = comp["composition_id"].to_numpy() == 2
    mixed_local = idx_all[mixed_mask]
    mixed_frame_ids = frame_ids[mixed_local]
    n = len(mixed_frame_ids)
    print(f"Mixed-only FPS corpus: {n} frames")

    fps_matrices = {a: matrices[a][mixed_local] for a in ALPHAS}
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for alpha in ALPHAS:
        X = fps_matrices[alpha]
        for retention in retentions:
            k = retention_count(n, retention)
            pct = pct_label(retention)
            for rep in range(replicates):
                run_seed = seed + 1000 * rep + int(round(alpha * 100)) + int(round(retention * 10000))
                rng = np.random.default_rng(run_seed)
                selected_local = farthest_point_sampling(X, k, rng)
                selected_ids = [int(mixed_frame_ids[i]) for i in selected_local]
                out = results_dir / f"alpha_{alpha:.2f}" / pct / f"replicate_{rep:02d}"
                out.mkdir(parents=True, exist_ok=True)
                (out / "selected_frames.txt").write_text(
                    "\n".join(f"frame_{fid}" for fid in selected_ids) + "\n"
                )
                meta = {
                    "alpha": alpha,
                    "retention_fraction": retention,
                    "n_total_frames": n,
                    "n_selected": len(selected_ids),
                    "replicate": rep,
                    "seed": run_seed,
                    "fingerprint_shape": list(X.shape),
                    "selected_frame_ids": selected_ids,
                    "selected_indices": selected_local,
                    "system": "hea_mixed",
                    "composition_filter": "mixed",
                }
                (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
                summary_rows.append(
                    {
                        "alpha": alpha,
                        "retention_pct": int(round(retention * 100)),
                        "replicate": rep,
                        "n_selected": len(selected_ids),
                        "n_total": n,
                    }
                )
        print(f"  finished α={alpha:.2f}")

    pd.DataFrame(summary_rows).to_csv(results_dir / "summary.csv", index=False)
    print(f"Wrote FPS grid → {results_dir}")


def load_jaccard_long(results_dir: Path) -> pd.DataFrame:
    rows = []
    for pct in range(5, 100, 5):
        by_rep: dict[int, dict[float, set[int]]] = {}
        for alpha in ALPHAS:
            for meta_path in sorted(
                (results_dir / f"alpha_{alpha:.2f}" / f"pct_{pct:03d}").glob(
                    "replicate_*/metadata.json"
                )
            ):
                meta = json.loads(meta_path.read_text())
                rep = int(meta["replicate"])
                by_rep.setdefault(rep, {})[float(meta["alpha"])] = set(
                    int(x) for x in meta["selected_frame_ids"]
                )
        for rep, sets in by_rep.items():
            for aa in ALPHAS:
                for ab in ALPHAS:
                    if ab <= aa or aa not in sets or ab not in sets:
                        continue
                    rows.append(
                        {
                            "retention_pct": pct,
                            "replicate": rep,
                            "alpha_a": aa,
                            "alpha_b": ab,
                            "jaccard": jaccard(sets[aa], sets[ab]),
                        }
                    )
    return pd.DataFrame(rows)


def plot_overlap(long: pd.DataFrame, out: Path) -> pd.DataFrame:
    series_pairs = [
        (0.0, 1.0, r"$\alpha$ = 0 vs 1"),
        (0.25, 1.0, r"$\alpha$ = 0.25 vs 1"),
        (0.50, 1.0, r"$\alpha$ = 0.50 vs 1"),
        (0.75, 1.0, r"$\alpha$ = 0.75 vs 1"),
    ]
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(x) for x in np.linspace(0.15, 0.85, len(series_pairs))]
    linestyles = ["-", "--", "-.", ":"]

    summary_rows = []
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for (aa, ab, lab), color, ls in zip(series_pairs, colors, linestyles):
        sub = long[(long.alpha_a == aa) & (long.alpha_b == ab)]
        g = sub.groupby("retention_pct")["jaccard"].agg(["mean", "std"]).reset_index()
        g["std"] = g["std"].fillna(0.0)
        ax.errorbar(
            g["retention_pct"],
            100.0 * g["mean"],
            yerr=100.0 * g["std"],
            color=color,
            ls=ls,
            marker="o",
            ms=3.5,
            lw=1.5,
            label=lab,
            capsize=2,
        )
        for _, row in g.iterrows():
            summary_rows.append(
                {
                    "system": "HEA_mixed",
                    "retention_pct": int(row["retention_pct"]),
                    "alpha_a": aa,
                    "alpha_b": ab,
                    "shared_pct_mean": float(100.0 * row["mean"]),
                    "shared_pct_std": float(100.0 * row["std"]),
                }
            )

    ax.set_xlabel("(Mixed Only) Dataset Retention (%)", fontsize=10)
    ax.set_ylabel("% of Selected Frames Shared", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse fingerprints cache from probe_data path via writing analysis cache if present
    hea_fp_cache = OUT_DIR / "hea_fingerprints.npz"
    if not RESULTS_DIR.exists() or not any(RESULTS_DIR.glob("alpha_*/pct_*/replicate_*/metadata.json")):
        print("Running mixed-only FPS grid...")
        run_mixed_fps()
    else:
        print(f"Using existing mixed FPS results at {RESULTS_DIR}")

    long = load_jaccard_long(RESULTS_DIR)
    out = FIG_DIR / "hea_mixed_fps_subset_overlap_vs_retention.png"
    summary = plot_overlap(long, out)
    summary.to_csv(OUT_DIR / "hea_mixed_fps_jaccard_summary.csv", index=False)
    print(f"Wrote {out}")
    print(summary.groupby(["alpha_a", "alpha_b"])["retention_pct"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
