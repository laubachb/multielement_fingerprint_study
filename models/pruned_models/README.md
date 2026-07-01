# Pruned ChIMES model fitting

Build ChIMES parameter sets on FPS-pruned CN training subsets from
`models/sampling/results/`.

## Workflow

```bash
cd models/pruned_models

# 1. Create run dirs (subset xyzf + fm_setup.in + SLURM scripts)
python prepare_runs.py

# 2. Debug queue: 1% × rep00 for all five α (recommended first)
bash submit_debug_pct001.sh
bash submit_debug_pct001.sh --batch-size 2   # if QOS submit limit hit

# 2b. Debug queue: all retentions × rep00 — 15 runs on skx-dev
bash submit_debug_all.sh
bash submit_debug_all.sh --batch-size 3   # if QOS submit limit hit

# 3. (Optional) Full production batch: all 3 replicates × 45 runs
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
| Retention levels | 1%, 10%, 50% × 5 α × 3 replicates = **45 runs** (debug: **15** rep00 only) |
| Solver | `chimes_lsq` + `chimes_lsq.py --algorithm dlasso` |

Walltime / partition scale with subset size (skx-dev for ≤10 frames, skx for larger).

## Validation downstream

| Stage | Directory | Metric |
|-------|-----------|--------|
| In-sample forces | (this step) | `Ax.txt` vs `b.txt` RMSE |
| Statepoint RDF | `models/statepoint_eval/` | g(r) vs full model |
| MD hold-out forces | `models/holdout/` | Forces on full-model MD frames vs full |

Impact figures: `models/pruning_analysis/make_plots.py`.

## Outputs (gitignored)

`runs/` fitting artifacts: `A.txt`, `b.txt`, `params.txt`, `fm_setup.log`, `solve.log`, etc.
