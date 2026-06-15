#!/usr/bin/env python3
"""
Prepare LAMMPS run directories for CN statepoint evaluation.

Creates runs/{model_id}/{statepoint_id}/ with data.in, in.lammps, params.txt,
and run_lammps.cmd for each (model, statepoint) pair.

Usage
-----
  cd models/statepoint_eval
  python prepare_runs.py --sync-params
  python prepare_runs.py --models full
  python prepare_runs.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MULTIELEMENT_ROOT = SCRIPT_DIR.parents[1]
STATEPOINTS_JSON = SCRIPT_DIR / "statepoints.json"
CHIMES_PARAMS_DIR = SCRIPT_DIR / "chimes_params"
PRUNED_RUNS_DIR = SCRIPT_DIR.parent / "pruned_models" / "runs"
LEGACY_STATEPOINTS = SCRIPT_DIR.parent / "pruned_simulations"
TEMPLATE_DIR = SCRIPT_DIR / "templates"
RUNS_DIR = SCRIPT_DIR / "runs"
XYZF2DATA = SCRIPT_DIR / "xyzf2data.py"

FULL_MODEL_SRC = MULTIELEMENT_ROOT / "element_switching" / "model" / "params.txt"
PRUNED_RUN_RE = re.compile(r"^a(?P<alpha>\d{3})_pct(?P<pct>\d{3})_rep(?P<rep>\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statepoints-json",
        type=Path,
        default=STATEPOINTS_JSON,
        help="Manifest of evaluation statepoints.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Output root for per-run directories.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model IDs to prepare (default: all available under chimes_params/).",
    )
    parser.add_argument(
        "--statepoints",
        nargs="*",
        default=None,
        help="Statepoint IDs to prepare (default: all in manifest).",
    )
    parser.add_argument(
        "--sync-params",
        action="store_true",
        help="Copy params.txt from element_switching (full) and completed pruned_models runs.",
    )
    parser.add_argument(
        "--sync-statepoints",
        action="store_true",
        help="Refresh statepoints/*/starting_frame.xyzf from models/pruned_simulations/.",
    )
    parser.add_argument(
        "--md-steps",
        type=int,
        default=6_000,
        help="NVT MD steps (1 fs timestep by default → 6 ps).",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=1.0,
        help="LAMMPS timestep in fs (real units); 1.0 fs matches metal units 0.001 ps.",
    )
    parser.add_argument(
        "--thermo-freq",
        type=int,
        default=5,
        help="Thermo output interval (steps).",
    )
    parser.add_argument(
        "--dump-freq",
        type=int,
        default=1,
        help="Trajectory dump interval (steps).",
    )
    parser.add_argument(
        "--nvt-tau-fs",
        type=float,
        default=50.0,
        help="NVT temperature damping (fs); 50 fs matches metal units 0.05 ps.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="SLURM partition for run_lammps.cmd (default: skx, or skx-dev with --debug-queue).",
    )
    parser.add_argument(
        "--walltime",
        default=None,
        help="SLURM walltime for run_lammps.cmd.",
    )
    parser.add_argument(
        "--ntasks",
        type=int,
        default=None,
        help="MPI ranks for ibrun (default: 48 production, 4 debug).",
    )
    parser.add_argument(
        "--debug-queue",
        action="store_true",
        help="Use skx-dev with shorter MD and fewer MPI ranks for validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["statepoints"]


def seed_for_run(model_id: str, statepoint_id: str) -> int:
    digest = hashlib.sha256(f"{model_id}:{statepoint_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 900_000 + 10_000


def render_template(path: Path, mapping: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for key, value in mapping.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def sync_statepoints(dry_run: bool) -> int:
    if not LEGACY_STATEPOINTS.is_dir():
        print(f"Warning: legacy source missing: {LEGACY_STATEPOINTS}", file=sys.stderr)
        return 0

    copied = 0
    for src_dir in sorted(LEGACY_STATEPOINTS.iterdir()):
        if not src_dir.is_dir():
            continue
        src_xyzf = src_dir / "starting_frame.xyzf"
        if not src_xyzf.is_file():
            continue
        dst_xyzf = SCRIPT_DIR / "statepoints" / src_dir.name / "starting_frame.xyzf"
        if dry_run:
            print(f"[dry-run] would copy {src_xyzf} -> {dst_xyzf}")
            copied += 1
            continue
        dst_xyzf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_xyzf, dst_xyzf)
        copied += 1
    return copied


def params_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return "ENDFILE" in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def sync_params(dry_run: bool) -> list[str]:
    synced: list[str] = []

    full_dst = CHIMES_PARAMS_DIR / "full" / "params.txt"
    if FULL_MODEL_SRC.is_file():
        if dry_run:
            print(f"[dry-run] would copy {FULL_MODEL_SRC} -> {full_dst}")
        else:
            full_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(FULL_MODEL_SRC, full_dst)
        synced.append("full")
    else:
        print(f"Warning: full model missing: {FULL_MODEL_SRC}", file=sys.stderr)

    if PRUNED_RUNS_DIR.is_dir():
        for run_dir in sorted(PRUNED_RUNS_DIR.iterdir()):
            if not run_dir.is_dir():
                continue
            match = PRUNED_RUN_RE.match(run_dir.name)
            if not match:
                continue
            src = run_dir / "params.txt"
            if not params_complete(src):
                continue
            model_id = run_dir.name
            dst = CHIMES_PARAMS_DIR / model_id / "params.txt"
            if dry_run:
                print(f"[dry-run] would copy {src} -> {dst}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            synced.append(model_id)

    return synced


def discover_models(requested: list[str] | None) -> list[str]:
    available = []
    if CHIMES_PARAMS_DIR.is_dir():
        for model_dir in sorted(CHIMES_PARAMS_DIR.iterdir()):
            if model_dir.is_dir() and params_complete(model_dir / "params.txt"):
                available.append(model_dir.name)
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise FileNotFoundError(
                "Requested models missing or incomplete params.txt: "
                + ", ".join(missing)
            )
        return requested
    return available


def convert_xyzf(xyzf_path: Path, data_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would convert {xyzf_path} -> {data_path}")
        return
    subprocess.run(
        [sys.executable, str(XYZF2DATA), str(xyzf_path), str(data_path)],
        check=True,
    )


def prepare_run(
    model_id: str,
    statepoint: dict,
    args: argparse.Namespace,
) -> Path:
    statepoint_id = statepoint["id"]
    run_dir = args.runs_dir / model_id / statepoint_id
    xyzf_path = SCRIPT_DIR / statepoint["starting_frame"]
    params_src = CHIMES_PARAMS_DIR / model_id / "params.txt"

    if not xyzf_path.is_file():
        raise FileNotFoundError(f"Missing starting frame: {xyzf_path}")
    if not params_complete(params_src):
        raise FileNotFoundError(f"Missing or incomplete params: {params_src}")

    if args.dry_run:
        print(f"[dry-run] would prepare {run_dir}")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    convert_xyzf(xyzf_path, run_dir / "data.in", dry_run=False)
    shutil.copy2(params_src, run_dir / "params.txt")

    if args.debug_queue:
        rdf_nevery, rdf_nrepeat = 10, 2
        rdf_periods = rdf_nevery * rdf_nrepeat
    else:
        # Six RDF samples over a 6000-step run (Nfreq must equal Nevery * Nrepeat).
        rdf_nevery, rdf_nrepeat, rdf_periods = 100, 10, 1000

    in_lammps = render_template(
        TEMPLATE_DIR / "in.lammps",
        {
            "TEMPERATURE_K": str(int(statepoint["temperature_k"])),
            "SEED": str(seed_for_run(model_id, statepoint_id)),
            "TIMESTEP_FS": str(args.timestep_fs),
            "MD_STEPS": str(args.md_steps),
            "THERMO_FREQ": str(args.thermo_freq),
            "DUMP_FREQ": str(args.dump_freq),
            "NVT_TAU_FS": str(args.nvt_tau_fs),
            "RDF_NEVERY": str(rdf_nevery),
            "RDF_NREPEAT": str(rdf_nrepeat),
            "RDF_PERIODS": str(rdf_periods),
        },
    )
    (run_dir / "in.lammps").write_text(in_lammps, encoding="utf-8")

    job_name = f"cn_{model_id}_{statepoint_id}"[:64]
    ntasks = args.ntasks
    run_cmd = render_template(
        TEMPLATE_DIR / "run_lammps.cmd",
        {
            "JOB_NAME": job_name,
            "NCORES": str(ntasks),
            "NTASKS": str(ntasks),
            "WALLTIME": args.walltime,
            "PARTITION": args.partition,
            "MULTIELEMENT_ROOT": str(MULTIELEMENT_ROOT),
        },
    )
    run_script = run_dir / "run_lammps.cmd"
    run_script.write_text(run_cmd, encoding="utf-8")
    run_script.chmod(0o755)

    manifest = {
        "model_id": model_id,
        "statepoint_id": statepoint_id,
        "n_pct": statepoint["n_pct"],
        "density_gcc": statepoint["density_gcc"],
        "temperature_k": statepoint["temperature_k"],
        "full_dft_frame": statepoint.get("full_dft_frame"),
        "params_source": str(params_src),
        "starting_frame": str(xyzf_path),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def apply_run_defaults(args: argparse.Namespace) -> None:
    if args.debug_queue:
        if args.partition is None:
            args.partition = "skx-dev"
        if args.walltime is None:
            args.walltime = "01:00:00"
        if args.ntasks is None:
            args.ntasks = 48
        if args.md_steps == 6_000:
            args.md_steps = 300
    else:
        if args.partition is None:
            args.partition = "skx"
        if args.walltime is None:
            args.walltime = "02:00:00"
        if args.ntasks is None:
            args.ntasks = 48


def main() -> int:
    args = parse_args()
    apply_run_defaults(args)

    if args.sync_statepoints:
        n = sync_statepoints(args.dry_run)
        print(f"Synced {n} starting_frame.xyzf file(s) into statepoints/")

    if args.sync_params:
        synced = sync_params(args.dry_run)
        print(f"Synced params for: {', '.join(synced) if synced else '(none)'}")

    manifest = load_manifest(args.statepoints_json)
    if args.statepoints:
        wanted = set(args.statepoints)
        manifest = [sp for sp in manifest if sp["id"] in wanted]
        if not manifest:
            raise SystemExit("No matching statepoints in manifest.")

    models = discover_models(args.models)
    if not models:
        raise SystemExit(
            "No models with complete params.txt under chimes_params/. "
            "Run: python prepare_runs.py --sync-params"
        )

    prepared = 0
    for model_id in models:
        for statepoint in manifest:
            prepare_run(model_id, statepoint, args)
            prepared += 1

    print(
        f"Prepared {prepared} run(s) under {args.runs_dir} "
        f"({len(models)} model(s) × {len(manifest)} statepoint(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
