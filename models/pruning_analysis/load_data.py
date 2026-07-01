"""Load completed CN pruned-model fits and statepoint RDF outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
PRUNED_RUNS = MODELS_DIR / "pruned_models" / "runs"
MD_RUNS = MODELS_DIR / "statepoint_eval" / "runs"
STATEPOINTS_JSON = MODELS_DIR / "statepoint_eval" / "statepoints.json"
REFERENCE_MODEL = "full"

RUN_RE = re.compile(
    r"^a(?P<alpha>\d{3})_pct(?P<pct>\d{3})_rep(?P<rep>\d{2})$"
)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RETENTIONS = (0.01, 0.10)
PAIR_NAMES = ("CC", "CN", "NN")
G_COLS = (2, 4, 6)  # g(r) columns in LAMMPS rdf.dat rows (after bin, r)


def parse_model_name(name: str) -> dict | None:
    m = RUN_RE.match(name)
    if not m:
        return None
    return {
        "model": name,
        "fps_alpha": int(m.group("alpha")) / 100.0,
        "retention_fraction": int(m.group("pct")) / 100.0,
        "retention_pct": int(m.group("pct")),
        "replicate": int(m.group("rep")),
    }


def retention_pct_label(fraction: float) -> int:
    return int(round(fraction * 100))


def params_complete(run_dir: Path) -> bool:
    p = run_dir / "params.txt"
    return p.is_file() and "ENDFILE" in p.read_text(encoding="utf-8", errors="ignore")


def training_rmse(run_dir: Path) -> float | None:
    ax_path = run_dir / "Ax.txt"
    b_path = run_dir / "b.txt"
    if not ax_path.is_file() or not b_path.is_file():
        return None
    ax = np.loadtxt(ax_path)
    b = np.loadtxt(b_path)
    if ax.shape != b.shape:
        return None
    return float(np.sqrt(np.mean((ax - b) ** 2)))


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_rdf_block(path: Path) -> np.ndarray | None:
    """Return (n_bins, 4) array: r, g_CC, g_CN, g_NN from final ave/time block."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    headers: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            headers.append((i, int(parts[1])))
    if not headers:
        return None
    start, nrows = headers[-1]
    rows: list[list[float]] = []
    for line in lines[start + 1 : start + 1 + nrows]:
        parts = line.split()
        if len(parts) < 8:
            continue
        r = float(parts[1])
        g_vals = [float(parts[c]) for c in G_COLS]
        rows.append([r, *g_vals])
    if not rows:
        return None
    return np.array(rows)


def rdf_pair_rmse_pct(model_gr: np.ndarray, ref_gr: np.ndarray, pair_idx: int) -> float:
    """RMS relative deviation (%) for one g(r) curve vs reference on overlapping r."""
    r_m, g_m = model_gr[:, 0], model_gr[:, pair_idx + 1]
    r_r, g_r = ref_gr[:, 0], ref_gr[:, pair_idx + 1]
    if len(r_m) != len(r_r) or not np.allclose(r_m, r_r, atol=1e-4):
        g_m = np.interp(r_r, r_m, g_m)
        r = r_r
        g_ref = g_r
        g_mod = g_m
    else:
        r = r_r
        g_ref = g_r
        g_mod = g_m
    mask = g_ref > 0.05
    if mask.sum() < 5:
        mask = np.ones_like(g_ref, dtype=bool)
    diff = g_mod[mask] - g_ref[mask]
    denom = np.sqrt(np.mean(g_ref[mask] ** 2))
    if denom <= 0:
        return float("nan")
    return float(100.0 * np.sqrt(np.mean(diff**2)) / denom)


def load_statepoints() -> list[dict]:
    data = json.loads(STATEPOINTS_JSON.read_text(encoding="utf-8"))
    return data["statepoints"]


def discover_completed_training() -> list[dict]:
    rows: list[dict] = []
    if not PRUNED_RUNS.is_dir():
        return rows
    for run_dir in sorted(PRUNED_RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_model_name(run_dir.name)
        if meta is None:
            continue
        if meta["retention_fraction"] not in RETENTIONS:
            continue
        if not params_complete(run_dir):
            continue
        manifest = load_manifest(run_dir)
        rmse = training_rmse(run_dir)
        rows.append(
            {
                **meta,
                "retention_pct": retention_pct_label(meta["retention_fraction"]),
                "n_frames": manifest.get("n_frames"),
                "training_rmse": rmse,
            }
        )
    return rows


def discover_rdf_metrics() -> list[dict]:
    rows: list[dict] = []
    statepoints = load_statepoints()
    ref_root = MD_RUNS / REFERENCE_MODEL
    if not ref_root.is_dir():
        return rows

    ref_curves: dict[str, np.ndarray] = {}
    for sp in statepoints:
        sid = sp["id"]
        curve = load_rdf_block(ref_root / sid / "rdf.dat")
        if curve is not None:
            ref_curves[sid] = curve

    for model_dir in sorted(MD_RUNS.iterdir()):
        if not model_dir.is_dir() or model_dir.name == REFERENCE_MODEL:
            continue
        meta = parse_model_name(model_dir.name)
        if meta is None or meta["retention_fraction"] not in RETENTIONS:
            continue
        if not params_complete(PRUNED_RUNS / model_dir.name):
            continue
        for sp in statepoints:
            sid = sp["id"]
            if sid not in ref_curves:
                continue
            model_rdf = model_dir / sid / "rdf.dat"
            curve = load_rdf_block(model_rdf)
            if curve is None:
                continue
            pair_dev = {
                name: rdf_pair_rmse_pct(curve, ref_curves[sid], i)
                for i, name in enumerate(PAIR_NAMES)
            }
            mean_dev = float(np.nanmean(list(pair_dev.values())))
            rows.append(
                {
                    **meta,
                    "retention_pct": retention_pct_label(meta["retention_fraction"]),
                    "statepoint_id": sid,
                    "case": sp["case"],
                    "temperature_k": sp["temperature_k"],
                    "density_gcc": sp["density_gcc"],
                    "n_pct": sp["n_pct"],
                    "rdf_deviation_pct": mean_dev,
                    "rdf_deviation_cn_pct": pair_dev["CN"],
                    **{f"rdf_deviation_{k}_pct": v for k, v in pair_dev.items()},
                }
            )
    return rows


def training_dataframe() -> pd.DataFrame:
    return pd.DataFrame(discover_completed_training())


def rdf_dataframe() -> pd.DataFrame:
    return pd.DataFrame(discover_rdf_metrics())


def md_force_dataframe() -> pd.DataFrame:
    path = MODELS_DIR / "holdout" / "metrics" / "md_force_deviation_long.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def discover_md_force_metrics() -> list[dict]:
    df = md_force_dataframe()
    if df.empty:
        return []
    return df.to_dict(orient="records")
