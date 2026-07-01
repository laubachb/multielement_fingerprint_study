#!/usr/bin/env python3
"""Merge per-model partial MD hold-out metrics and write aggregated tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PARTIAL_DIR = SCRIPT_DIR / "metrics" / "partial_rows"
LEGACY_PARTIAL = SCRIPT_DIR / "metrics" / "partial"
METRICS_DIR = SCRIPT_DIR / "metrics"


def main() -> None:
    long_path = METRICS_DIR / "md_force_deviation_long.csv"
    if long_path.is_file() and len(list(PARTIAL_DIR.glob("*.csv"))) == 0:
        print(f"Metrics already at {long_path}; nothing to merge.")
        return

    files = sorted(PARTIAL_DIR.glob("*.csv"))
    if not files and LEGACY_PARTIAL.is_dir():
        files = sorted(LEGACY_PARTIAL.glob("*.csv"))
    if not files:
        raise SystemExit(f"No partial metrics under {PARTIAL_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    long_path = METRICS_DIR / "md_force_deviation_long.csv"
    df.to_csv(long_path, index=False)

    agg = (
        df.groupby(["model", "fps_alpha", "retention_fraction", "retention_pct", "replicate"])[
            "force_deviation_ev_a"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "force_deviation_ev_a", "std": "force_deviation_std"})
    )
    agg.to_csv(METRICS_DIR / "md_force_deviation_by_model.csv", index=False)

    by_case = (
        df.groupby(["model", "fps_alpha", "retention_pct", "case", "statepoint_id"])[
            "force_deviation_ev_a"
        ]
        .mean()
        .reset_index()
    )
    by_case.to_csv(METRICS_DIR / "md_force_deviation_by_case.csv", index=False)
    print(f"Merged {len(files)} partial files → {long_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
