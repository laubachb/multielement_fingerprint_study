#!/usr/bin/env python3
"""Build the CN pruning-midpoint run directories (37.5% and 62.5% retention).

Extends the existing 25/50/75% FPS-pruned grid with the two intermediate
retentions, using the SAME farthest-point-sampling methodology (fingerprint-space
Euclidean FPS, per-replicate seed = base_seed + rep) so the new points are
consistent with the rest of the grid.

For each (fingerprint alpha in {0,.25,.5,.75,1}) x (retention in {0.375, 0.625})
x (replicate 0..N-1):
  * FPS-select k = round(297 * retention) frames in that alpha's fingerprint space
  * write training.xyzf (selected frames extracted from the full CN trajectory)
  * write fm_setup.in (CHEBYSHEV 25 10 4 template with NFRAMES filled in)
  * write fps_meta.json (provenance: selected frame ids + seed)

Then generate A/b and DLASSO-solve at lambda=1e-1 via run_gen_solve_eval.cmd.

Inputs (override via environment):
  CN_FINGERPRINTS_NPZ  per-alpha fingerprint matrices (keys alpha_0.00 .. alpha_1.00)
  CN_FULL_XYZF         full CN training trajectory (297 frames)
  CN_FM_TEMPLATE       fm_setup.in template with {{TRJFILE}} / {{NFRAMES}}
  WORK                 output root for the run directories
"""
import os, json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]          # <repo>/models/pruned_models/midpoints_1e-1

FINGERPRINTS_NPZ = Path(os.environ.get(
    "CN_FINGERPRINTS_NPZ",
    REPO_ROOT / "models/workflows/fingerprints/umap/cache/fingerprints.npz"))
FULL_XYZF = Path(os.environ.get("CN_FULL_XYZF", REPO_ROOT / "data/full_dft.xyzf"))
FM_TEMPLATE = Path(os.environ.get(
    "CN_FM_TEMPLATE", REPO_ROOT / "models/pruned_models/templates/fm_setup.in.template"))
WORK = Path(os.environ.get("WORK", SCRIPT_DIR / "runs"))

N_FRAMES = 297
BASE_SEED = 42
N_REPS = 5
# (fingerprint alpha, run-name tag, fingerprints.npz key)
ALPHAS = [(0.00, "a000", "alpha_0.00"), (0.25, "a025", "alpha_0.25"),
          (0.50, "a050", "alpha_0.50"), (0.75, "a075", "alpha_0.75"),
          (1.00, "a100", "alpha_1.00")]
# (retention fraction, k frames, pct label)  -- true midpoints of 25/50 and 50/75
MIDPOINTS = [(0.375, round(N_FRAMES * 0.375), "037"),
             (0.625, round(N_FRAMES * 0.625), "062")]


def parse_xyzf(path):
    """Yield (natoms_line_int, box_line, [atom_lines]) blocks in file order."""
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


def farthest_point_sampling(vectors, k, seed):
    """Greedy FPS in Euclidean space; returns selected indices in selection order."""
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    k = min(k, n)
    if k == 1:
        return [int(rng.integers(0, n))]
    start = int(rng.integers(0, n))
    selected = [start]
    min_d = np.linalg.norm(vectors - vectors[start], axis=1)
    for _ in range(1, k):
        far = int(np.argmax(min_d))
        selected.append(far)
        min_d = np.minimum(min_d, np.linalg.norm(vectors - vectors[far], axis=1))
    return selected


def main():
    fp = np.load(FINGERPRINTS_NPZ)
    template = FM_TEMPLATE.read_text()
    blocks = parse_xyzf(FULL_XYZF)
    assert len(blocks) == N_FRAMES, f"expected {N_FRAMES} frames, got {len(blocks)}"
    WORK.mkdir(parents=True, exist_ok=True)

    n = 0
    for _, tag, key in ALPHAS:
        vectors = fp[key]
        for frac, k, plab in MIDPOINTS:
            for rep in range(N_REPS):
                seed = BASE_SEED + rep
                sel = farthest_point_sampling(vectors, k, seed)
                rd = WORK / f"{tag}_pct{plab}_rep{rep:02d}"
                rd.mkdir(exist_ok=True)
                with open(rd / "training.xyzf", "w") as fh:
                    for i in sel:                 # FPS index i -> full-traj frame i
                        na, box, atoms = blocks[i]
                        fh.write(f"{na}\n"); fh.write(box); fh.writelines(atoms)
                fm = template.replace("{{TRJFILE}}", "training.xyzf").replace(
                    "{{NFRAMES}}", str(k))
                (rd / "fm_setup.in").write_text(fm)
                (rd / "fps_meta.json").write_text(json.dumps(
                    {"alpha_fingerprint": key, "retention_fraction": frac,
                     "pct_label": plab, "n_selected": k, "replicate": rep,
                     "seed": seed, "selected_indices": sel,
                     "selected_frame_ids": [i + 1 for i in sel]}, indent=2))
                n += 1
    print(f"built {n} CN midpoint run dirs under {WORK}")


if __name__ == "__main__":
    main()
