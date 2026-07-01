#!/usr/bin/env python3
"""
Extract hold-out snapshots from full-model (100%) statepoint MD trajectories.

Randomly samples frames after an equilibration window from runs/full/*/traj.lammpstrj
and writes per-snapshot LAMMPS data.in files plus a manifest for force evaluation.

Usage
-----
  cd models/holdout
  python extract_md_holdout.py
  python extract_md_holdout.py --frames-per-statepoint 25 --seed 42 --clean
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from lammpstrj import available_steps, iter_frames

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
STATEPOINTS_JSON = MODELS_DIR / "statepoint_eval" / "statepoints.json"
FULL_RUNS = MODELS_DIR / "statepoint_eval" / "runs" / "full"
SNAPSHOTS_DIR = SCRIPT_DIR / "snapshots"
MANIFEST_PATH = SCRIPT_DIR / "md_holdout_manifest.json"

# Case 4 (3.20.3percN_4.0gcc) — full-model MD lost atoms; no usable hold-out.
EXCLUDE_STATEPOINTS = {"3.20.3percN_4.0gcc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-before", type=int, default=1000, help="Skip MD steps before this.")
    parser.add_argument(
        "--frames-per-statepoint",
        type=int,
        default=25,
        help="Random hold-out frames drawn per statepoint trajectory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible frame selection.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing snapshots/ and forces/ before extracting.",
    )
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-combined-xyzf",
        action="store_true",
        help="Also write holdout_frames.xyzf + holdout_frames_index.json.",
    )
    return parser.parse_args()


def read_data_header(data_in: Path) -> str:
    """Return the LAMMPS data file header through the Atoms section title."""
    lines = data_in.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        out.append(line)
        if line.strip() == "Atoms":
            out.append("")
            return "\n".join(out) + "\n"
    raise ValueError(f"Could not parse header from {data_in}")


def write_snapshot_data(
    header: str,
    frame_atoms: list[tuple[int, int, str, float, float, float]],
    out_path: Path,
) -> None:
    atoms_sorted = sorted(frame_atoms, key=lambda a: a[0])
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for atom_id, atom_type, _element, x, y, z in atoms_sorted:
            handle.write(f"{atom_id} {atom_type} {x} {y} {z}\n")


def select_random_steps(
    steps: list[int],
    *,
    skip_before: int,
    n_frames: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample up to n_frames unique production steps without replacement."""
    prod = sorted(s for s in steps if s >= skip_before)
    if not prod:
        return []
    k = min(n_frames, len(prod))
    idx = rng.choice(len(prod), size=k, replace=False)
    return sorted(prod[i] for i in idx)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    statepoints = json.loads(STATEPOINTS_JSON.read_text(encoding="utf-8"))["statepoints"]

    if args.clean and not args.dry_run:
        for path in (args.snapshots_dir, SCRIPT_DIR / "forces", SCRIPT_DIR / "metrics"):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed {path}")

    manifest: list[dict] = []
    selection_log: list[dict] = []
    for sp in statepoints:
        sid = sp["id"]
        if sid in EXCLUDE_STATEPOINTS:
            print(f"Skipping {sid} (unstable / incomplete full-model MD)")
            continue
        run_dir = FULL_RUNS / sid
        traj_path = run_dir / "traj.lammpstrj"
        data_template = run_dir / "data.in"
        if not traj_path.is_file():
            print(f"Skipping {sid}: missing {traj_path}")
            continue
        if not data_template.is_file():
            print(f"Skipping {sid}: missing {data_template}")
            continue

        steps = available_steps(traj_path)
        if not steps:
            print(f"Skipping {sid}: empty trajectory")
            continue
        last_step = max(steps)
        if last_step < args.skip_before:
            print(f"Skipping {sid}: trajectory ends at step {last_step}")
            continue

        selected_set = set(
            select_random_steps(
                steps,
                skip_before=args.skip_before,
                n_frames=args.frames_per_statepoint,
                rng=rng,
            )
        )
        header = read_data_header(data_template)
        n_prod = sum(1 for s in steps if s >= args.skip_before)
        print(
            f"{sid}: {len(selected_set)} snapshots "
            f"(requested {args.frames_per_statepoint}, pool {n_prod})"
        )
        selection_log.append(
            {
                "statepoint_id": sid,
                "n_production_frames": n_prod,
                "n_selected": len(selected_set),
                "md_steps": sorted(selected_set),
            }
        )

        for frame in iter_frames(traj_path):
            if frame.step not in selected_set:
                continue
            snap_id = f"{sid}_step{frame.step:05d}"
            snap_dir = args.snapshots_dir / snap_id
            if not args.dry_run:
                snap_dir.mkdir(parents=True, exist_ok=True)
                write_snapshot_data(header, frame.atoms, snap_dir / "data.in")
            manifest.append(
                {
                    "snapshot_id": snap_id,
                    "statepoint_id": sid,
                    "case": sp["case"],
                    "temperature_k": sp["temperature_k"],
                    "density_gcc": sp["density_gcc"],
                    "n_pct": sp["n_pct"],
                    "md_step": frame.step,
                    "traj_path": str(traj_path),
                }
            )

    if args.dry_run:
        print(f"[dry-run] would write {len(manifest)} snapshots")
        return

    args.snapshots_dir.mkdir(parents=True, exist_ok=True)
    manifest_doc = {
        "seed": args.seed,
        "skip_before": args.skip_before,
        "frames_per_statepoint": args.frames_per_statepoint,
        "n_snapshots": len(manifest),
        "selections": selection_log,
        "snapshots": manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest_doc, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(manifest)} snapshots to {args.snapshots_dir}")
    print(f"Manifest: {MANIFEST_PATH}")

    if args.write_combined_xyzf:
        from holdout_xyzf import write_combined_xyzf

        xyzf_path = SCRIPT_DIR / "holdout_frames.xyzf"
        index_path = SCRIPT_DIR / "holdout_frames_index.json"
        n = write_combined_xyzf(manifest, args.snapshots_dir, xyzf_path, index_path)
        print(f"Combined xyzf: {xyzf_path} ({n} frames)")
        print(f"Frame index: {index_path}")


if __name__ == "__main__":
    main()
