#!/usr/bin/env python3
"""
Evaluate ChIMES forces on MD hold-out snapshots and compare pruned vs full model.

Runs on a compute node (MPI LAMMPS). Writes per-(snapshot, model) force arrays and
aggregated RMSE metrics relative to the full (100%) model.

Usage
-----
  cd models/holdout
  python extract_md_holdout.py
  bash submit_eval_batched.sh              # full cache + pruned array
  bash submit_eval_pruned.sh               # pruned only (diverse-first)
  python merge_metrics.py

  # Legacy per-snapshot mode (slow):
  python evaluate_md_forces.py --ntasks 1
  sbatch run_eval_all.cmd
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
MANIFEST_PATH = SCRIPT_DIR / "md_holdout_manifest.json"
SNAPSHOTS_DIR = SCRIPT_DIR / "snapshots"
FORCES_DIR = SCRIPT_DIR / "forces"
METRICS_DIR = SCRIPT_DIR / "metrics"
FULL_PARAMS = MODELS_DIR / "statepoint_eval" / "chimes_params" / "full" / "params.txt"
FALLBACK_FULL_PARAMS = MODELS_DIR.parent / "element_switching" / "model" / "params.txt"
PRUNED_RUNS = MODELS_DIR / "pruned_models" / "runs"
LAMMPS_TEMPLATE = SCRIPT_DIR / "templates" / "eval_forces.lammps"
LAMMPS_BATCHED_TEMPLATE = SCRIPT_DIR / "templates" / "eval_forces_batched.lammps"
BATCHED_TASK_LIST = SCRIPT_DIR / "batched_task_list.json"
COMBINED_XYZF = SCRIPT_DIR / "holdout_frames.xyzf"
COMBINED_XYZF_INDEX = SCRIPT_DIR / "holdout_frames_index.json"

RUN_RE = re.compile(r"^a(?P<alpha>\d{3})_pct(?P<pct>\d{3})_rep(?P<rep>\d{2})$")
RETENTIONS = (0.01, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR)
    parser.add_argument("--forces-dir", type=Path, default=FORCES_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--models", nargs="*", default=None, help="Limit to model names.")
    parser.add_argument(
        "--cache-full-only",
        action="store_true",
        help="Only evaluate/cache full-model forces for all snapshots.",
    )
    parser.add_argument(
        "--cache-snapshot-index",
        type=int,
        default=None,
        help="Cache full-model forces for one manifest index (debug array task).",
    )
    parser.add_argument(
        "--pruned-task-index",
        type=int,
        default=None,
        help="Evaluate one (snapshot, model) pair from pruned_task_list.json.",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process snapshots and models in reverse order (parallel with forward run).",
    )
    parser.add_argument(
        "--work-tag",
        default="",
        help="Subdirectory under forces/_work/ to avoid collisions between parallel jobs.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Only write per-task outputs; do not overwrite combined metrics CSVs.",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Evaluate many snapshots in one LAMMPS process (minimize startup overhead).",
    )
    parser.add_argument(
        "--array-task-id",
        type=int,
        default=None,
        help="SLURM array task id selecting one model from batched_task_list.json.",
    )
    parser.add_argument(
        "--write-combined-xyzf",
        action="store_true",
        help="Write holdout_frames.xyzf + index JSON from manifest snapshots.",
    )
    return parser.parse_args()


def params_complete(run_dir: Path) -> bool:
    p = run_dir / "params.txt"
    return p.is_file() and "ENDFILE" in p.read_text(encoding="utf-8", errors="ignore")


def resolve_full_params() -> Path:
    if FULL_PARAMS.is_file():
        return FULL_PARAMS
    if FALLBACK_FULL_PARAMS.is_file():
        return FALLBACK_FULL_PARAMS
    raise FileNotFoundError("Full model params.txt not found")


def discover_models(limit: list[str] | None) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(PRUNED_RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        retention = int(m.group("pct")) / 100.0
        if retention not in RETENTIONS:
            continue
        if not params_complete(run_dir):
            continue
        if limit and run_dir.name not in limit:
            continue
        rows.append(
            {
                "model": run_dir.name,
                "fps_alpha": int(m.group("alpha")) / 100.0,
                "retention_fraction": retention,
                "retention_pct": int(m.group("pct")),
                "replicate": int(m.group("rep")),
                "params_path": run_dir / "params.txt",
            }
        )
    return rows


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "snapshots" in data:
        return data["snapshots"]
    raise ValueError(f"Unrecognized manifest format in {path}")


def read_forces_dump(path: Path) -> np.ndarray:
    """Return (n_atoms, 3) forces sorted by atom id."""
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("ITEM:"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            atom_id = int(parts[0])
            fx, fy, fz = map(float, parts[-3:])
        except ValueError:
            continue
        rows.append([atom_id, fx, fy, fz])
    if not rows:
        raise ValueError(f"No forces parsed from {path}")
    arr = np.array(rows)
    order = np.argsort(arr[:, 0])
    return arr[order, 1:4]


def force_rmse(ref: np.ndarray, model: np.ndarray) -> float:
    if ref.shape != model.shape:
        raise ValueError(f"Shape mismatch {ref.shape} vs {model.shape}")
    diff = model - ref
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def batched_work_dir(args: argparse.Namespace, model_key: str) -> Path:
    base = args.forces_dir / "_work" / "batch"
    if args.work_tag:
        base = base / args.work_tag
    return base / model_key


def snapshots_to_evaluate(
    manifest: list[dict],
    *,
    model_key: str,
    forces_dir: Path,
    skip_existing: bool,
) -> list[dict]:
    pending: list[dict] = []
    for entry in manifest:
        snap_id = entry["snapshot_id"]
        out_npy = forces_dir / f"{snap_id}__{model_key}.npy"
        if skip_existing and out_npy.is_file():
            continue
        pending.append(entry)
    return pending


def write_batched_lammps_input(work_dir: Path, nframes: int) -> None:
    template = LAMMPS_BATCHED_TEMPLATE.read_text(encoding="utf-8")
    work_dir.joinpath("in.lammps").write_text(
        template.replace("{{NFRAMES}}", str(nframes)),
        encoding="utf-8",
    )


def prepare_batched_data_links(
    work_dir: Path,
    pending: list[dict],
    snapshots_dir: Path,
) -> dict[int, dict]:
    data_dir = work_dir / "data_files"
    dumps_dir = work_dir / "dumps"
    data_dir.mkdir(parents=True, exist_ok=True)
    dumps_dir.mkdir(parents=True, exist_ok=True)
    frame_map: dict[int, dict] = {}
    for frame_idx, entry in enumerate(pending, start=1):
        snap_id = entry["snapshot_id"]
        src = snapshots_dir / snap_id / "data.in"
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src}")
        link = data_dir / f"data_{frame_idx}.in"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())
        frame_map[frame_idx] = entry
    mapping_path = work_dir / "frame_map.json"
    mapping_path.write_text(
        json.dumps(
            {str(k): v for k, v in frame_map.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame_map


def run_batched_lammps(
    work_dir: Path,
    params_src: Path,
    pending: list[dict],
    snapshots_dir: Path,
    lammps_exe: str,
    ntasks: int,
) -> dict[str, np.ndarray]:
    if not pending:
        return {}
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(params_src, work_dir / "params.txt")
    frame_map = prepare_batched_data_links(work_dir, pending, snapshots_dir)
    write_batched_lammps_input(work_dir, len(pending))
    dumps_dir = work_dir / "dumps"
    for path in dumps_dir.glob("frame_*.dump"):
        path.unlink()

    cmd = ["ibrun", "-n", str(ntasks), lammps_exe, "-in", "in.lammps"]
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-4000:]
        raise RuntimeError(f"Batched LAMMPS failed in {work_dir}:\n{tail}")

    forces_by_snap: dict[str, np.ndarray] = {}
    for frame_idx, entry in frame_map.items():
        dump_path = dumps_dir / f"frame_{frame_idx}.dump"
        if not dump_path.is_file():
            raise FileNotFoundError(f"Missing batched dump {dump_path}")
        forces_by_snap[entry["snapshot_id"]] = read_forces_dump(dump_path)
    return forces_by_snap


def diverse_model_order(models: list[dict]) -> list[dict]:
    """Round-robin across (fps_alpha, retention_pct) before extra replicates."""
    from collections import defaultdict

    groups: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for model in models:
        key = (model["fps_alpha"], model["retention_pct"])
        groups[key].append(model)
    for group in groups.values():
        group.sort(key=lambda m: m["replicate"])

    ordered: list[dict] = []
    keys = sorted(groups)
    max_reps = max(len(groups[k]) for k in keys)
    for rep_round in range(max_reps):
        for key in keys:
            if rep_round < len(groups[key]):
                ordered.append(groups[key][rep_round])
    return ordered


def model_force_cache_complete(
    model_key: str,
    manifest: list[dict],
    forces_dir: Path,
) -> bool:
    return all(
        (forces_dir / f"{entry['snapshot_id']}__{model_key}.npy").is_file()
        for entry in manifest
    )


def write_batched_task_list(
    models: list[dict],
    full_params: Path,
    path: Path = BATCHED_TASK_LIST,
    *,
    diverse_first: bool = True,
) -> list[dict]:
    if diverse_first:
        models = diverse_model_order(models)
    tasks: list[dict] = [
        {
            "array_task_id": 0,
            "model_key": "full",
            "model": "full",
            "params_path": str(full_params),
            "fps_alpha": 0.0,
            "retention_fraction": 1.0,
            "retention_pct": 100,
            "replicate": 0,
        }
    ]
    for idx, model in enumerate(models, start=1):
        tasks.append(
            {
                "array_task_id": idx,
                "model_key": model["model"],
                **model,
                "params_path": str(model["params_path"]),
            }
        )
    path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    return tasks


def eval_batched_model(
    task: dict,
    manifest: list[dict],
    *,
    args: argparse.Namespace,
    lammps_exe: str,
) -> int:
    model_key = task["model_key"]
    params_path = Path(task["params_path"])
    pending = snapshots_to_evaluate(
        manifest,
        model_key=model_key,
        forces_dir=args.forces_dir,
        skip_existing=args.skip_existing,
    )
    if not pending:
        print(f"[{model_key}] all {len(manifest)} snapshots already cached")
        return 0

    if model_key != "full":
        missing_full = [
            e["snapshot_id"]
            for e in pending
            if not (args.forces_dir / f"{e['snapshot_id']}__full.npy").is_file()
        ]
        if missing_full:
            raise RuntimeError(
                f"[{model_key}] missing full reference for {len(missing_full)} snapshots; "
                "run array task 0 first"
            )

    print(f"[{model_key}] batched LAMMPS for {len(pending)} / {len(manifest)} snapshots")
    work = batched_work_dir(args, model_key)
    forces_by_snap = run_batched_lammps(
        work,
        params_path,
        pending,
        args.snapshots_dir,
        lammps_exe,
        args.ntasks,
    )

    n_saved = 0
    for entry in pending:
        snap_id = entry["snapshot_id"]
        forces = forces_by_snap[snap_id]
        out_npy = args.forces_dir / f"{snap_id}__{model_key}.npy"
        np.save(out_npy, forces)
        n_saved += 1

        if model_key == "full":
            continue

        ref_forces = np.load(args.forces_dir / f"{snap_id}__full.npy")
        rmse = force_rmse(ref_forces, forces)
        row = {
            **{k: entry[k] for k in ("snapshot_id", "statepoint_id", "case", "md_step")},
            **{k: task[k] for k in ("model", "fps_alpha", "retention_fraction", "retention_pct", "replicate")},
            "force_deviation_ev_a": rmse,
        }
        partial = args.metrics_dir / "partial_rows" / f"{snap_id}__{task['model']}.csv"
        partial.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(partial, index=False)

    print(f"[{model_key}] wrote {n_saved} force arrays")
    return n_saved


def eval_batched(args: argparse.Namespace) -> pd.DataFrame:
    manifest = load_manifest(args.manifest)
    if not manifest:
        raise RuntimeError(f"Empty manifest: {args.manifest}")

    multielement_root = MODELS_DIR.parent
    env = os_environ(multielement_root)
    lammps_exe = env.get("LAMMPS_EXE")
    if not lammps_exe or not Path(lammps_exe).exists():
        raise RuntimeError("LAMMPS_EXE not set or missing; source setup/env.sh on compute node")

    full_params = resolve_full_params()
    models = discover_models(args.models)
    args.forces_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.write_combined_xyzf:
        from holdout_xyzf import write_combined_xyzf

        n = write_combined_xyzf(
            manifest,
            args.snapshots_dir,
            COMBINED_XYZF,
            COMBINED_XYZF_INDEX,
        )
        print(f"Wrote {n} frames to {COMBINED_XYZF}")

    if not BATCHED_TASK_LIST.is_file():
        write_batched_task_list(models, full_params)
    tasks = json.loads(BATCHED_TASK_LIST.read_text(encoding="utf-8"))

    if args.array_task_id is None:
        raise ValueError("--batched requires --array-task-id (or run via submit_eval_batched.sh)")

    task = next((t for t in tasks if t["array_task_id"] == args.array_task_id), None)
    if task is None:
        print(f"No batched task for array id {args.array_task_id}; skipping.")
        return pd.DataFrame()

    eval_batched_model(task, manifest, args=args, lammps_exe=lammps_exe)

    if args.array_task_id != 0 or args.skip_aggregate:
        return pd.DataFrame()

    # Task 0 only: if everything is cached, still allow downstream merge from partial_rows.
    return pd.DataFrame()


def run_lammps_forces(
    work_dir: Path,
    params_src: Path,
    lammps_exe: str,
    ntasks: int,
) -> np.ndarray:
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(params_src, work_dir / "params.txt")
    shutil.copy2(LAMMPS_TEMPLATE, work_dir / "in.lammps")
    dump_path = work_dir / "forces.dump"
    if dump_path.exists():
        dump_path.unlink()

    cmd = ["ibrun", "-n", str(ntasks), lammps_exe, "-in", "in.lammps"]
    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-2000:]
        raise RuntimeError(f"LAMMPS failed in {work_dir}:\n{tail}")
    if not dump_path.is_file():
        raise FileNotFoundError(f"Missing {dump_path}")
    return read_forces_dump(dump_path)


def model_label(model_name: str) -> str:
    return "full" if model_name == "full" else model_name


def work_dir(args: argparse.Namespace, snap_id: str, model_name: str) -> Path:
    base = args.forces_dir / "_work"
    if args.work_tag:
        base = base / args.work_tag
    return base / snap_id / model_name


def cache_full_snapshot(
    entry: dict,
    *,
    args: argparse.Namespace,
    full_params: Path,
    lammps_exe: str,
) -> None:
    snap_id = entry["snapshot_id"]
    snap_dir = args.snapshots_dir / snap_id
    data_in = snap_dir / "data.in"
    if not data_in.is_file():
        raise FileNotFoundError(f"Missing {data_in}")
    ref_npy = args.forces_dir / f"{snap_id}__full.npy"
    if args.skip_existing and ref_npy.is_file():
        print(f"Already cached: {snap_id}")
        return
    ref_work = work_dir(args, snap_id, "full")
    ref_work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_in, ref_work / "data.in")
    forces = run_lammps_forces(ref_work, full_params, lammps_exe, args.ntasks)
    np.save(ref_npy, forces)
    print(f"Cached full forces: {snap_id}")


def eval_pruned_task(
    entry: dict,
    model: dict,
    *,
    args: argparse.Namespace,
    lammps_exe: str,
) -> dict:
    snap_id = entry["snapshot_id"]
    model_name = model["model"]
    snap_dir = args.snapshots_dir / snap_id
    data_in = snap_dir / "data.in"
    if not data_in.is_file():
        raise FileNotFoundError(f"Missing {data_in}")

    ref_npy = args.forces_dir / f"{snap_id}__full.npy"
    if not ref_npy.is_file():
        raise FileNotFoundError(f"Missing full reference: {ref_npy}")
    ref_forces = np.load(ref_npy)

    out_npy = args.forces_dir / f"{snap_id}__{model_name}.npy"
    if args.skip_existing and out_npy.is_file():
        model_forces = np.load(out_npy)
    else:
        work = work_dir(args, snap_id, model_name)
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_in, work / "data.in")
        model_forces = run_lammps_forces(work, model["params_path"], lammps_exe, args.ntasks)
        np.save(out_npy, model_forces)

    rmse = force_rmse(ref_forces, model_forces)
    row = {
        **{k: entry[k] for k in ("snapshot_id", "statepoint_id", "case", "md_step")},
        **model,
        "force_deviation_ev_a": rmse,
    }
    partial = args.metrics_dir / "partial_rows" / f"{snap_id}__{model_name}.csv"
    partial.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(partial, index=False)
    print(f"{snap_id} / {model_name}: RMSE={rmse:.6f} eV/Å")
    return row


def write_pruned_task_list(manifest: list[dict], models: list[dict], path: Path) -> int:
    tasks: list[dict] = []
    for si, snap in enumerate(manifest):
        for mi, model in enumerate(models):
            tasks.append(
                {
                    "snapshot_index": si,
                    "model_index": mi,
                    **snap,
                    **model,
                    "params_path": str(model["params_path"]),
                }
            )
    path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    return len(tasks)


def eval_all(args: argparse.Namespace) -> pd.DataFrame:
    manifest = load_manifest(args.manifest)
    if not manifest:
        raise RuntimeError(f"Empty manifest: {args.manifest}")

    if args.reverse:
        if not args.work_tag:
            args.work_tag = "rev"
        args.skip_aggregate = True
        manifest = list(reversed(manifest))

    multielement_root = MODELS_DIR.parent
    env = os_environ(multielement_root)
    lammps_exe = env.get("LAMMPS_EXE")
    if not lammps_exe or not Path(lammps_exe).exists():
        raise RuntimeError("LAMMPS_EXE not set or missing; source setup/env.sh on compute node")

    full_params = resolve_full_params()
    models = discover_models(args.models)
    if args.reverse:
        models = list(reversed(models))
        print("Reverse order: last snapshot/model first")

    if args.cache_snapshot_index is not None:
        manifest = load_manifest(args.manifest)
        idx = args.cache_snapshot_index
        if idx < 0 or idx >= len(manifest):
            raise IndexError(f"cache-snapshot-index {idx} out of range (n={len(manifest)})")
        args.forces_dir.mkdir(parents=True, exist_ok=True)
        cache_full_snapshot(
            manifest[idx], args=args, full_params=full_params, lammps_exe=lammps_exe
        )
        return pd.DataFrame()

    if args.pruned_task_index is not None:
        task_path = SCRIPT_DIR / "pruned_task_list.json"
        if not task_path.is_file():
            raise FileNotFoundError(f"Run submit_eval_pruned.sh first; missing {task_path}")
        tasks = json.loads(task_path.read_text(encoding="utf-8"))
        idx = args.pruned_task_index
        if idx < 0 or idx >= len(tasks):
            print(f"No task for index {idx}; skipping.")
            return pd.DataFrame()
        task = tasks[idx]
        entry = {k: task[k] for k in ("snapshot_id", "statepoint_id", "case", "md_step")}
        model = {
            k: task[k]
            for k in (
                "model",
                "fps_alpha",
                "retention_fraction",
                "retention_pct",
                "replicate",
                "params_path",
            )
        }
        model["params_path"] = Path(model["params_path"])
        args.forces_dir.mkdir(parents=True, exist_ok=True)
        args.metrics_dir.mkdir(parents=True, exist_ok=True)
        eval_pruned_task(entry, model, args=args, lammps_exe=lammps_exe)
        return pd.DataFrame([task])

    print(f"Hold-out snapshots: {len(manifest)}")
    print(f"Pruned models: {len(models)}")

    args.forces_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []

    # Cache full-model reference forces for every snapshot.
    for entry in manifest:
        snap_id = entry["snapshot_id"]
        snap_dir = args.snapshots_dir / snap_id
        data_in = snap_dir / "data.in"
        if not data_in.is_file():
            print(f"Skipping {snap_id}: missing data.in")
            continue
        ref_npy = args.forces_dir / f"{snap_id}__full.npy"
        if args.skip_existing and ref_npy.is_file():
            continue
        cache_full_snapshot(
            entry, args=args, full_params=full_params, lammps_exe=lammps_exe
        )

    if args.cache_full_only:
        print("Full-model force cache complete.")
        return pd.DataFrame()

    for entry in manifest:
        snap_id = entry["snapshot_id"]
        snap_dir = args.snapshots_dir / snap_id
        data_in = snap_dir / "data.in"
        if not data_in.is_file():
            print(f"Skipping {snap_id}: missing data.in")
            continue

        ref_npy = args.forces_dir / f"{snap_id}__full.npy"
        if not ref_npy.is_file():
            print(f"Skipping {snap_id}: missing full reference forces")
            continue
        ref_forces = np.load(ref_npy)

        for model in models:
            model_name = model["model"]
            out_npy = args.forces_dir / f"{snap_id}__{model_name}.npy"
            partial_row = args.metrics_dir / "partial_rows" / f"{snap_id}__{model_name}.csv"
            if args.skip_existing and out_npy.is_file():
                if partial_row.is_file() or args.skip_aggregate:
                    continue
                model_forces = np.load(out_npy)
            else:
                work = work_dir(args, snap_id, model_name)
                work.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_in, work / "data.in")
                model_forces = run_lammps_forces(
                    work, model["params_path"], lammps_exe, args.ntasks
                )
                np.save(out_npy, model_forces)
            rmse = force_rmse(ref_forces, model_forces)
            row = {
                **{k: entry[k] for k in ("snapshot_id", "statepoint_id", "case", "md_step")},
                **model,
                "force_deviation_ev_a": rmse,
            }
            partial_row.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(partial_row, index=False)
            if not args.skip_aggregate:
                metric_rows.append(row)

    df = pd.DataFrame(metric_rows)
    if df.empty:
        if args.skip_aggregate:
            print("Incremental partial_rows written (skip-aggregate mode).")
        return df

    if args.skip_aggregate:
        return df

    if args.models and len(args.models) == 1:
        partial = args.metrics_dir / "partial" / f"{args.models[0]}.csv"
        partial.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(partial, index=False)
        print(f"Wrote {partial} ({len(df)} rows)")
        return df

    long_path = args.metrics_dir / "md_force_deviation_long.csv"
    df.to_csv(long_path, index=False)

    agg = (
        df.groupby(
            ["model", "fps_alpha", "retention_fraction", "retention_pct", "replicate"]
        )["force_deviation_ev_a"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "force_deviation_ev_a", "std": "force_deviation_std"})
    )
    agg.to_csv(args.metrics_dir / "md_force_deviation_by_model.csv", index=False)

    by_case = (
        df.groupby(
            ["model", "fps_alpha", "retention_pct", "case", "statepoint_id"]
        )["force_deviation_ev_a"]
        .mean()
        .reset_index()
    )
    by_case.to_csv(args.metrics_dir / "md_force_deviation_by_case.csv", index=False)
    print(f"\nWrote {long_path} ({len(df)} rows)")
    return df


def os_environ(multielement_root: Path) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env.setdefault("MULTIELEMENT_ROOT", str(multielement_root))
    setup = multielement_root / "setup" / "env.sh"
    if setup.is_file():
        result = subprocess.run(
            ["bash", "-c", f"source {setup} && env"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                env[key] = val
    return env


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Run extract_md_holdout.py first; missing {args.manifest}")
    if args.batched:
        eval_batched(args)
        return
    df = eval_all(args)
    if args.cache_full_only or args.cache_snapshot_index is not None:
        return
    if args.pruned_task_index is not None:
        return
    if args.skip_aggregate:
        return
    if df.empty:
        print("No metrics produced.", file=sys.stderr)
        sys.exit(1)
    if "force_deviation_ev_a" in df.columns:
        print(
            df.groupby(["retention_pct", "fps_alpha"])["force_deviation_ev_a"]
            .mean()
            .unstack()
            .to_string()
        )


if __name__ == "__main__":
    main()
