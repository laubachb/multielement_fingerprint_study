#!/usr/bin/env python3
"""Build the HEA pruning-midpoint run directories (37.5% and 62.5% retention).

Extends the existing 25/50/75% mixed-composition FPS-pruned grid with the two
intermediate retentions. The FPS selections themselves are produced first by the
HEA sampler (same methodology as the rest of the grid):

    python ../../sampling/run_fps_sampling.py \
        --retentions 0.375 0.625 --replicates 10 --seed 42 \
        --frames-file ../../splits/holdout20_mixed_clean/train_frames.txt \
        --output-dir $WORK/fps_selections

This script then, for each (alpha) x (pct_038=37.5%, pct_062=62.5%) x (rep 0..9):
  * reads the FPS selected_frames.txt
  * writes training.xyzf (frames extracted from the rotated HEA trajectory)
  * writes fm_setup.in (CHEBYSHEV 12 5 2, Y/Mg, forces-only; NFRAMES filled in)
  * writes run_manifest.json (provenance + SLURM resources for gen/solve)

Then generate A/b and DLASSO-solve at lambda=1e-5 via run_gen_solve.cmd.

Inputs (override via environment):
  HEA_FPS_DIR    root of FPS selections (default: $WORK/fps_selections)
  HEA_SRC_XYZF   rotated HEA trajectory (hea_chimes_format_rotated.xyzf)
  HEA_FM_TEMPLATE fm_setup.in template with {{NFRAMES}} (default: alongside this script)
  WORK           output root for the run directories
"""
import os, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]          # <repo>/hea_study/pruned_models/midpoints_1e-5

WORK = Path(os.environ.get("WORK", SCRIPT_DIR / "runs"))
FPS_DIR = Path(os.environ.get("HEA_FPS_DIR", WORK / "fps_selections"))
SRC_XYZF = Path(os.environ.get(
    "HEA_SRC_XYZF", REPO_ROOT / "hea_study/chimes_model/hea_chimes_format_rotated.xyzf"))
FM_TEMPLATE = Path(os.environ.get("HEA_FM_TEMPLATE", SCRIPT_DIR / "fm_setup.in.template"))

BASE_SEED = 42
N_REPS = 10
CORES_PER_NODE = 112
# fingerprint-alpha dirs (from run_fps_sampling.py) -> run-name tag
ALPHA_MAP = {"alpha_0.00": "a000", "alpha_0.25": "a025", "alpha_0.50": "a050",
             "alpha_0.75": "a075", "alpha_1.00": "a100"}
ALPHA_VAL = {"alpha_0.00": 0.0, "alpha_0.25": 0.25, "alpha_0.50": 0.5,
             "alpha_0.75": 0.75, "alpha_1.00": 1.0}
# FPS pct dir -> retention fraction  (round(0.375*100)=38, round(0.625*100)=62)
FRAC_MAP = {"pct_038": 0.375, "pct_062": 0.625}


def parse_xyzf(path):
    blocks = []
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            na = int(line.strip())
            box = fh.readline()
            atoms = [fh.readline() for _ in range(na)]
            blocks.append((na, box, atoms))
    return blocks


def frame_index(name):
    return int(name.split("_")[1]) - 1     # frame_N (1-indexed) -> 0-indexed traj row


def set_nframes(template, k):
    return template.replace("{{NFRAMES}}", str(k))


def main():
    blocks = parse_xyzf(SRC_XYZF)
    template = FM_TEMPLATE.read_text()
    WORK.mkdir(parents=True, exist_ok=True)

    n = 0
    for adir, tag in ALPHA_MAP.items():
        for pdir, frac in FRAC_MAP.items():
            plab = pdir.replace("pct_", "")
            for rep in range(N_REPS):
                repdir = FPS_DIR / adir / pdir / f"replicate_{rep:02d}"
                names = (repdir / "selected_frames.txt").read_text().split()
                k = len(names)
                rd = WORK / f"{tag}_pct{plab}_rep{rep:02d}"
                rd.mkdir(exist_ok=True)
                with open(rd / "training.xyzf", "w") as fh:
                    for nm in names:
                        na, box, atoms = blocks[frame_index(nm)]
                        fh.write(f"{na}\n"); fh.write(box); fh.writelines(atoms)
                (rd / "fm_setup.in").write_text(set_nframes(template, k))
                nnodes = max(1, -(-k // CORES_PER_NODE))
                (rd / "run_manifest.json").write_text(json.dumps(
                    {"name": rd.name, "alpha": ALPHA_VAL[adir],
                     "retention_fraction": frac, "replicate": rep, "seed": BASE_SEED + rep,
                     "n_frames": k, "selected_frames": names,
                     "frame_indices": [frame_index(x) for x in names],
                     "resources": {"partition": "pdebug", "nnodes": nnodes, "ntasks": k,
                                   "gen_walltime": "01:00:00", "solve_walltime": "01:00:00"},
                     "lambda": 1e-5}, indent=2))
                n += 1
    print(f"built {n} HEA midpoint run dirs under {WORK}")


if __name__ == "__main__":
    main()
