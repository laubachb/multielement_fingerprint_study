# CN pruning midpoints (λ = 1e-1)

Extends the CN FPS-pruned retention grid with the two **intermediate** retentions —
**37.5 %** (111 frames) and **62.5 %** (186 frames), the exact frame-count midpoints
of the existing 25/50/75 % tiers (74/148/223 frames). Same design of experiments
(5 fingerprint-α × 2 retentions × 5 replicates = **50 models**), same DLASSO
regularization **λ = 1e-1** as the rest of the 1e-1 CN grid — only the prune rate
differs.

## Pipeline

```bash
cd models/pruned_models/midpoints_1e-1

# 1. FPS-select frames + write training.xyzf / fm_setup.in per (α, retention, rep)
python build_midpoint_runs.py          # -> $WORK/<a*_pct{037,062}_rep*>/

# 2. gen A/b, DLASSO-solve (λ=1e-1), and per-statepoint eval (one pdebug job)
sbatch run_gen_solve_eval.cmd
```

| File | Role |
|------|------|
| `build_midpoint_runs.py` | Farthest-point sampling (seed 42 + rep) in each α's fingerprint space; extracts frames from the full CN trajectory; writes `training.xyzf`, `fm_setup.in` (CHEBYSHEV 25 10 4), `fps_meta.json`. |
| `run_gen_solve_eval.cmd` | SLURM (pdebug) — for each run: build the design matrix, `chimes_lsq.py --algorithm dlasso --alpha 0.1`, then shard the per-statepoint force-RMSE eval. Self-skips converged models. |
| `eval_shard_perSP.py` | Per-shard force-RMSE-vs-DFT on the reduced CN test set → `force_rmse_perSP_mid_1e-1.csv`. |

## Reproducibility inputs (override via environment)

| Var | Default | Contents |
|-----|---------|----------|
| `CN_FINGERPRINTS_NPZ` | `models/workflows/fingerprints/umap/cache/fingerprints.npz` | per-α fingerprint matrices for FPS |
| `CN_FULL_XYZF` | `data/full_dft.xyzf` | full 297-frame CN trajectory |
| `CN_FM_TEMPLATE` | `models/pruned_models/templates/fm_setup.in.template` | fm_setup with `{{TRJFILE}}`/`{{NFRAMES}}` |
| `CHIMES_CALC_LIB` | — | `libchimescalc_dl.so` (eval) |
| `WORK` | `./runs` | output root for run dirs |

The pruned per-statepoint RMSE feeds the relative test-error figure in
[`models/test_errors/`](../../test_errors/). Placeholders (`ACCOUNT`, `REPO_ROOT`,
`WORK`) in the SLURM script are set for the target cluster before submitting.
