# MD hold-out force validation

Sample configurations from **full (100%) ChIMES model** statepoint MD trajectories
and measure per-atom force RMSE of pruned models relative to the full model on those
frames. Metrics feed `models/pruning_analysis/` for publication figures.

## Prerequisites

1. **Full-model statepoint MD** completed under `models/statepoint_eval/runs/full/`
   (`traj.lammpstrj` per statepoint). See `models/statepoint_eval/README.md`.
2. **Pruned fits** with complete `params.txt` (`ENDFILE` present) under
   `models/pruned_models/runs/`.

## Workflow (batched — recommended)

Batched evaluation runs **one LAMMPS process per model** over all hold-out snapshots,
avoiding per-frame MPI/ChIMES startup overhead (~60× faster than the legacy loop).

```bash
cd models/holdout

# 1. Extract snapshots (9 statepoints × 25 random frames; case 4 excluded)
python extract_md_holdout.py
# Optional: also write combined trajectory + frame index
python extract_md_holdout.py --write-combined-xyzf

# 2. Cache full-model reference forces (one batched LAMMPS run)
bash submit_eval_batched.sh
# Or, if full cache already complete, submit pruned models only:
bash submit_eval_pruned.sh              # up to 2 jobs (skx-dev QOS)
bash submit_eval_pruned.sh --limit 4      # when submit slots free

# 3. Merge partial metrics and regenerate figures
python merge_metrics.py
cd ../pruning_analysis && python make_plots.py
```

Re-run `submit_eval_pruned.sh` after jobs finish to queue the next diverse models.
`--skip-existing` resumes from cached `forces/*.npy`.

### Task ordering (diverse-first)

Pruned jobs prioritize **one replicate per (FPS α, retention %)** before extra
replicates — e.g. `a000_pct001_rep00`, `a000_pct010_rep00`, `a025_pct001_rep00`, …
then `a000_pct001_rep01`, etc. See `diverse_model_order()` in `evaluate_md_forces.py`.

### SLURM layout

| Script | Purpose |
|--------|---------|
| `submit_eval_batched.sh` | Task 0: full-model cache; tasks 1–N: pruned (array, max 48 concurrent on skx) |
| `submit_eval_pruned.sh` | Pruned only; respects skx-dev QOS (`MaxJobsPU=2`, `MaxSubmitPU=4`) |
| `run_eval_batched.cmd` | Array worker: `python evaluate_md_forces.py --batched --array-task-id $SLURM_ARRAY_TASK_ID` |

Legacy per-snapshot launches (`submit_eval.sh`, `run_eval_all.cmd`) remain for
debugging but are not recommended for production.

## Hold-out definition

| Setting | Default |
|---------|---------|
| Source trajectories | `statepoint_eval/runs/full/{statepoint}/traj.lammpstrj` |
| Excluded | Case 4 (`3.20.3percN_4.0gcc`) — full-model MD lost atoms |
| Equilibration skip | MD steps &lt; 1000 |
| Sampling | **25 random** production frames per statepoint (`seed=42`) |
| Reference forces | Full-model `params.txt` (`statepoint_eval/chimes_params/full/` or fallback) |
| Pruned forces | `pruned_models/runs/{model}/params.txt` |
| Metric | RMS \(\|\mathbf{F}_\text{pruned} - \mathbf{F}_\text{full}\|\) per atom (eV/Å) |

## Layout

```
holdout/
  extract_md_holdout.py       # sample frames → snapshots/ + manifest
  evaluate_md_forces.py       # LAMMPS force eval (single-frame or --batched)
  holdout_xyzf.py             # data.in ↔ xyzf; combined trajectory writer
  lammpstrj.py                # trajectory parser
  merge_metrics.py            # aggregate partial_rows/ → long CSV
  templates/
    eval_forces.lammps        # single-snapshot template
    eval_forces_batched.lammps # loop over snapshots in one process
  submit_eval_batched.sh
  submit_eval_pruned.sh
```

## Outputs (gitignored)

| Path | Description |
|------|-------------|
| `snapshots/{id}_step{#####}/data.in` | Hold-out structures |
| `md_holdout_manifest.json` | Frame metadata (reproducible via `seed=42`) |
| `holdout_frames.xyzf` | Optional combined trajectory |
| `holdout_frames_index.json` | Frame index → snapshot metadata |
| `batched_task_list.json` | Auto-generated SLURM task map |
| `forces/*.npy` | Cached force arrays per (snapshot, model) |
| `forces/_work/` | LAMMPS scratch (dumps, logs) |
| `metrics/partial_rows/*.csv` | Per-(snapshot, model) RMSE during eval |
| `metrics/md_force_deviation_long.csv` | Merged table for plotting |
| `*.out`, `*.err` | SLURM logs |

## Figures (via pruning_analysis)

- `md_force_deviation_vs_retention.png`
- `md_force_deviation_by_case.png`

## Tracked in git

Scripts (`*.py`, `*.sh`, `*.cmd`), `templates/`, and this README. All generated
data above are gitignored (see root `.gitignore`).
