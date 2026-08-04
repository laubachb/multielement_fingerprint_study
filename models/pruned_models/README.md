# Pruned ChIMES model fitting

Build ChIMES parameter sets on FPS-pruned CN training subsets from
`models/sampling/results/`.

## Workflow

```bash
cd models/pruned_models

# 1. Create run dirs (subset xyzf + fm_setup.in + SLURM scripts)
python prepare_runs.py

# 2. Submit gen_Amat → solve_Amat job chains (re-run until all 45 are queued)
bash submit_all.sh --batch-size 5
```

Each run directory under `runs/` corresponds to one FPS replicate:

```
runs/a000_pct001_rep00/
  training.xyzf       # pruned subset of full_dft.xyzf
  fm_setup.in         # CN topology (from full_model template)
  run_gen_Amat.cmd    # chimes_lsq matrix generation
  run_solve_Amat.cmd  # dlasso solve → params.txt
  run_manifest.json
```

## Defaults

| Setting | Value |
|---------|-------|
| Source trajectory | `models/full_model/full_dft.xyzf` |
| FPS inputs | `models/sampling/results/` |
| Retention levels | 1%, 10%, 50% × 5 α × 3 replicates = **45 runs** |
| Solver | `chimes_lsq` + `chimes_lsq.py --algorithm dlasso` |

Walltime / partition scale with subset size (skx-dev for ≤10 frames, skx for larger).

## Outputs (gitignored)

`runs/` fitting artifacts: `A.txt`, `b.txt`, `params.txt`, `fm_setup.log`, `solve.log`, etc.

## Retention midpoints

[`midpoints_1e-1/`](midpoints_1e-1/) adds the 37.5 % and 62.5 % retention tiers
(midpoints of 25/50/75 %) re-solved at DLASSO **λ = 1e-1**, and self-contains its
FPS → gen → solve → eval pipeline (`build_midpoint_runs.py`,
`run_gen_solve_eval.cmd`, `eval_shard_perSP.py`). The resulting per-statepoint force
RMSE feeds [`models/test_errors/`](../test_errors/).
