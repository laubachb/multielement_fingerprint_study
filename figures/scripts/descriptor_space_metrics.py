#!/usr/bin/env python3
"""
Descriptor-space metrics beyond UMAP:

  1. Separability / degeneracy scores vs fingerprint α (CN + HEA)
  2. FPS subset Jaccard overlap across α at fixed retention

Outputs land in writing/figures/ and writing/analysis/output/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "output"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

CN_CACHE = ROOT / "models" / "workflows" / "fingerprints" / "umap" / "cache"
CN_STATEPOINTS = ROOT / "models" / "statepoint_eval" / "statepoints.json"
CN_JACCARD = ROOT / "models" / "fps_alpha_probe" / "output" / "overlap"
CN_FPS_RESULTS = ROOT / "models" / "sampling" / "results"
CN_FPS_CONV = ROOT / "models" / "sampling" / "results_convergence_5pct"
HEA_DIR = ROOT / "hea_study"
HEA_CONV = HEA_DIR / "sampling" / "results_convergence_5pct"
HEA_FP_CACHE = OUT_DIR / "hea_fingerprints.npz"

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_COLORS = {
    0.0: "#2166ac",
    0.25: "#4393c3",
    0.5: "#92c5de",
    0.75: "#f4a582",
    1.0: "#b2182b",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def alpha_label(a: float) -> str:
    return f"{a:.2f}".rstrip("0").rstrip(".") if a % 1 else f"{int(a)}"


def knn_purity(X: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Fraction of k nearest neighbors (excluding self) sharing the query label."""
    n = len(labels)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean").fit(X)
    idx = nn.kneighbors(return_distance=False)[:, 1:]  # drop self
    hits = labels[idx] == labels[:, None]
    return float(hits.mean())


def degeneracy_ratio(X: np.ndarray, n_pct: np.ndarray, statepoint: np.ndarray) -> float:
    """
    Mean distance for (same n%, different statepoint) pairs divided by mean
    distance for (different n%) pairs. Smaller ⇒ stronger compositional collapse.
    """
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(X, metric="euclidean"))
    n = len(X)
    same_comp_diff_sp = []
    diff_comp = []
    for i in range(n):
        for j in range(i + 1, n):
            if n_pct[i] == n_pct[j] and statepoint[i] != statepoint[j]:
                same_comp_diff_sp.append(D[i, j])
            elif n_pct[i] != n_pct[j]:
                diff_comp.append(D[i, j])
    if not same_comp_diff_sp or not diff_comp:
        return float("nan")
    return float(np.mean(same_comp_diff_sp) / np.mean(diff_comp))


def load_cn_statepoint_labels(frame_ids: list[int]) -> pd.DataFrame:
    data = json.loads(CN_STATEPOINTS.read_text(encoding="utf-8"))
    sps = sorted(data["statepoints"], key=lambda sp: sp["full_dft_frame"])
    rows = []
    for i, sp in enumerate(sps):
        start = int(sp["full_dft_frame"])
        end = int(sps[i + 1]["full_dft_frame"]) - 1 if i + 1 < len(sps) else start + 19
        for fid in range(start, end + 1):
            rows.append(
                {
                    "frame_id": fid,
                    "statepoint_id": sp["id"],
                    "n_pct": float(sp["n_pct"]),
                    "case": int(sp["case"]),
                }
            )
    table = pd.DataFrame(rows)
    # Restrict to the 10 eval blocks (20 frames each); ignore unlabeled extras.
    return table[table["frame_id"].isin(frame_ids)].copy()


def compute_cn_separability() -> pd.DataFrame:
    fp = np.load(CN_CACHE / "fingerprints.npz")
    man = json.loads((CN_CACHE / "manifest.json").read_text())
    frame_ids = [
        int(re.search(r"frame_(\d+)", d).group(1)) for d in man["frame_dirs"]
    ]
    labels = load_cn_statepoint_labels(frame_ids)
    id_to_row = {fid: i for i, fid in enumerate(frame_ids)}
    keep = [id_to_row[fid] for fid in labels["frame_id"].tolist() if fid in id_to_row]
    labels = labels[labels["frame_id"].isin([frame_ids[i] for i in keep])].reset_index(
        drop=True
    )
    keep = [id_to_row[fid] for fid in labels["frame_id"].tolist()]

    sp_codes = LabelEncoder().fit_transform(labels["statepoint_id"].astype(str))
    n_pct = labels["n_pct"].to_numpy()
    n_pct_codes = LabelEncoder().fit_transform(n_pct)

    rows = []
    for a in ALPHAS:
        key = f"alpha_{a:.2f}"
        X = fp[key][keep]
        rows.append(
            {
                "system": "CN",
                "alpha": a,
                "n_frames": len(keep),
                "silhouette_statepoint": float(
                    silhouette_score(X, sp_codes, metric="euclidean")
                ),
                "silhouette_composition": float(
                    silhouette_score(X, n_pct_codes, metric="euclidean")
                ),
                "knn_purity_statepoint": knn_purity(X, sp_codes, k=10),
                "knn_purity_composition": knn_purity(X, n_pct_codes, k=10),
                "degeneracy_ratio": degeneracy_ratio(
                    X, n_pct, labels["statepoint_id"].to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows)


def load_hea_fingerprints() -> tuple[np.ndarray, dict[float, np.ndarray]]:
    sys.path.insert(0, str(HEA_DIR / "fps_alpha_probe"))
    from probe_data import build_frame_composition_table, load_fingerprint_matrices

    if HEA_FP_CACHE.is_file():
        z = np.load(HEA_FP_CACHE, allow_pickle=True)
        frame_ids = z["frame_ids"]
        matrices = {float(a): z[f"alpha_{a:.2f}"] for a in ALPHAS}
        return frame_ids, matrices

    print("Loading HEA fingerprints (first run; caching to disk)...")
    frame_ids, matrices = load_fingerprint_matrices(HEA_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save = {"frame_ids": np.asarray(frame_ids)}
    for a, X in matrices.items():
        save[f"alpha_{a:.2f}"] = X
    np.savez_compressed(HEA_FP_CACHE, **save)
    return np.asarray(frame_ids), matrices


def compute_hea_separability() -> pd.DataFrame:
    frame_ids, matrices = load_hea_fingerprints()
    sys.path.insert(0, str(HEA_DIR / "fps_alpha_probe"))
    from probe_data import build_frame_composition_table

    comp = build_frame_composition_table()
    comp = comp[comp["frame_id"].isin(frame_ids)].copy()
    # Align to fingerprint order
    order = {fid: i for i, fid in enumerate(frame_ids.tolist())}
    comp = comp.sort_values("frame_id")
    idx = [order[fid] for fid in comp["frame_id"].tolist()]
    y = comp["composition_id"].to_numpy()

    rows = []
    for a in ALPHAS:
        X = matrices[a][idx]
        rows.append(
            {
                "system": "HEA",
                "alpha": a,
                "n_frames": len(idx),
                "silhouette_composition": float(
                    silhouette_score(X, y, metric="euclidean")
                ),
                "knn_purity_composition": knn_purity(X, y, k=10),
                "silhouette_statepoint": np.nan,
                "knn_purity_statepoint": np.nan,
                "degeneracy_ratio": np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_separability(cn: pd.DataFrame, hea: pd.DataFrame, out: Path) -> None:
    """
    Two plain-language panels:
      CN  — fraction of neighbors that share the same thermodynamic statepoint
      HEA — fraction of neighbors that share the same composition class
    Higher = local environments of the same physical label stay together.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))

    ax = axes[0]
    ax.plot(
        cn["alpha"],
        100.0 * cn["knn_purity_statepoint"],
        "o-",
        color="#2166ac",
        lw=1.8,
        ms=7,
    )
    ax.set_xlabel(r"fingerprint $\alpha$" + "\n(0 = composition only, 1 = structure only)")
    ax.set_ylabel("% of 10 nearest neighbors\nfrom the same statepoint")
    ax.set_title("CN: do same-statepoint frames cluster together?")
    ax.set_xticks(list(ALPHAS))
    ax.set_ylim(0, 105)
    ax.annotate(
        "composition-only collapses\ndifferent T/ρ at same N%",
        xy=(0.0, 100.0 * cn.loc[cn.alpha == 0.0, "knn_purity_statepoint"].iloc[0]),
        xytext=(0.35, 45),
        fontsize=8,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )
    ax.annotate(
        "adding structure\nrecovers statepoint identity",
        xy=(0.25, 100.0 * cn.loc[cn.alpha == 0.25, "knn_purity_statepoint"].iloc[0]),
        xytext=(0.45, 75),
        fontsize=8,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )

    ax = axes[1]
    ax.plot(
        hea["alpha"],
        100.0 * hea["knn_purity_composition"],
        "o-",
        color="#762a83",
        lw=1.8,
        ms=7,
    )
    ax.set_xlabel(r"fingerprint $\alpha$" + "\n(0 = composition only, 1 = structure only)")
    ax.set_ylabel("% of 10 nearest neighbors\nwith the same composition\n(Y only / Mg only / Mixed)")
    ax.set_title("HEA: do same-composition frames cluster together?")
    ax.set_xticks(list(ALPHAS))
    ax.set_ylim(0, 105)
    ax.annotate(
        "composition alone\npoorly separates Y vs Mg",
        xy=(0.0, 100.0 * hea.loc[hea.alpha == 0.0, "knn_purity_composition"].iloc[0]),
        xytext=(0.15, 25),
        fontsize=8,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )
    ax.annotate(
        "best local separation\nnear α = 0.25",
        xy=(0.25, 100.0 * hea.loc[hea.alpha == 0.25, "knn_purity_composition"].iloc[0]),
        xytext=(0.45, 55),
        fontsize=8,
        color="0.25",
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )

    for ax, letter in zip(axes, "AB"):
        ax.text(
            -0.08,
            1.04,
            letter,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            va="bottom",
        )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def mean_jaccard_matrix(
    results_root: Path,
    retention: float,
    alphas: tuple[float, ...] = ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and std Jaccard matrices over replicates at fixed retention."""
    mats = []
    for meta_paths in _group_metas(results_root, retention):
        sets = {}
        for a, path in meta_paths.items():
            meta = json.loads(path.read_text())
            sets[a] = set(int(x) for x in meta["selected_frame_ids"])
        if len(sets) < 2:
            continue
        m = np.zeros((len(alphas), len(alphas)))
        for i, ai in enumerate(alphas):
            for j, aj in enumerate(alphas):
                m[i, j] = jaccard(sets.get(ai, set()), sets.get(aj, set()))
        mats.append(m)
    labels = [alpha_label(a) for a in alphas]
    if not mats:
        empty = pd.DataFrame(np.nan, index=labels, columns=labels)
        return empty, empty
    stack = np.stack(mats, axis=0)
    mean = pd.DataFrame(stack.mean(axis=0), index=labels, columns=labels)
    std = pd.DataFrame(stack.std(axis=0), index=labels, columns=labels)
    return mean, std


def _group_metas(results_root: Path, retention: float) -> list[dict[float, Path]]:
    """Group metadata.json paths by replicate for a given retention."""
    pct = int(round(retention * 100))
    # Support pct_001 and pct_010 naming
    candidates = []
    for pat in (f"pct_{pct:03d}", f"pct_{pct:02d}", f"pct_{pct}"):
        candidates.extend(results_root.glob(f"alpha_*/{pat}/replicate_*/metadata.json"))
    by_rep: dict[int, dict[float, Path]] = {}
    for path in candidates:
        meta = json.loads(path.read_text())
        if abs(float(meta["retention_fraction"]) - retention) > 1e-9:
            continue
        rep = int(meta["replicate"])
        a = float(meta["alpha"])
        by_rep.setdefault(rep, {})[a] = path
    return list(by_rep.values())


def _jaccard_long(
    results_root: Path,
    *,
    retention_pcts: range | None = None,
) -> pd.DataFrame:
    """Pairwise FPS Jaccard vs retention from a convergence-style results root."""
    if retention_pcts is None:
        retention_pcts = range(5, 100, 5)
    rows: list[dict] = []

    for pct in retention_pcts:
        for group in _group_metas(results_root, pct / 100.0):
            for aa in ALPHAS:
                for ab in ALPHAS:
                    if ab <= aa:
                        continue
                    if aa not in group or ab not in group:
                        continue
                    sa = set(
                        int(x)
                        for x in json.loads(group[aa].read_text())["selected_frame_ids"]
                    )
                    sb = set(
                        int(x)
                        for x in json.loads(group[ab].read_text())["selected_frame_ids"]
                    )
                    rows.append(
                        {
                            "retention_pct": pct,
                            "alpha_a": aa,
                            "alpha_b": ab,
                            "jaccard": jaccard(sa, sb),
                        }
                    )

    return pd.DataFrame(rows)


def plot_fps_overlap(
    out: Path,
    *,
    results_root: Path,
    system: str,
    legend_loc: str = "lower right",
) -> pd.DataFrame:
    """
    Line plot only (no heatmaps), data points at 5–95% retention.
    X-axis kept at 0–100 for readability. Style matches the CN publication figure.
    """
    long = _jaccard_long(results_root)
    if long.empty:
        raise RuntimeError(f"No FPS Jaccard data found under {results_root}")
    summary_rows: list[dict] = []

    series_pairs = [
        (0.0, 1.0, r"$\alpha$ = 0 vs 1"),
        (0.25, 1.0, r"$\alpha$ = 0.25 vs 1"),
        (0.50, 1.0, r"$\alpha$ = 0.50 vs 1"),
        (0.75, 1.0, r"$\alpha$ = 0.75 vs 1"),
    ]
    cmap = mpl.colormaps["viridis"]
    colors = [cmap(x) for x in np.linspace(0.15, 0.85, len(series_pairs))]
    linestyles = ["-", "--", "-.", ":"]

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
                    "system": system,
                    "retention_pct": int(row["retention_pct"]),
                    "alpha_a": aa,
                    "alpha_b": ab,
                    "shared_pct_mean": float(100.0 * row["mean"]),
                    "shared_pct_std": float(100.0 * row["std"]),
                }
            )

    ax.set_xlabel("Dataset Retention (%)", fontsize=10)
    ax.set_ylabel("% of Selected Frames Shared", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=False, loc=legend_loc, fontsize=10)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing CN separability...")
    cn = compute_cn_separability()
    print(cn.to_string(index=False))

    print("Computing HEA separability...")
    hea = compute_hea_separability()
    print(hea.to_string(index=False))

    sep = pd.concat([cn, hea], ignore_index=True)
    sep.to_csv(OUT_DIR / "separability_vs_alpha.csv", index=False)

    sep_fig = FIG_DIR / "separability_vs_alpha.png"
    plot_separability(cn, hea, sep_fig)
    print(f"Wrote {sep_fig}")

    print("Plotting FPS overlap (CN + HEA)...")
    cn_fps = FIG_DIR / "fps_subset_overlap_vs_retention.png"
    hea_fps = FIG_DIR / "hea_fps_subset_overlap_vs_retention.png"
    jac_cn = plot_fps_overlap(cn_fps, results_root=CN_FPS_CONV, system="CN")
    jac_hea = plot_fps_overlap(hea_fps, results_root=HEA_CONV, system="HEA",
                               legend_loc="upper left")
    jac = pd.concat([jac_cn, jac_hea], ignore_index=True)
    jac.to_csv(OUT_DIR / "fps_jaccard_summary.csv", index=False)
    print(f"Wrote {cn_fps}")
    print(f"Wrote {hea_fps}")

    # Remove superseded figures if present
    for stale in (
        FIG_DIR / "fps_jaccard_heatmaps.png",
        FIG_DIR / "fps_jaccard_cn_vs_retention.png",
    ):
        if stale.is_file():
            stale.unlink()
            print(f"Removed {stale.name}")


if __name__ == "__main__":
    main()
