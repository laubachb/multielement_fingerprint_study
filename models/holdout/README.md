# MD hold-out hold-out pipeline

Sample configurations from **full (100%) ChIMES model** statepoint MD trajectories
and measure force RMSE of pruned models relative to the full model on those frames.

## Workflow

```bash
cd models/holdout

# 1. Extract snapshots (9 statepoints × up to 25 random frames; case 4 excluded)
python extract_md_holdout.py --clean

# 2. One skx-dev debug job (full cache + all pruned evals; re-run to resume)
bash submit_eval.sh

# 3. Regenerate pruning figures (includes MD force panels when metrics exist)
cd ../pruning_analysis && python make_plots.py
```

## Hold-out definition

| Setting | Default |
|---------|---------|
| Source trajectories | `statepoint_eval/runs/full/{statepoint}/traj.lammpstrj` |
| Excluded | Case 4 (`3.20.3percN_4.0gcc`) — lost atoms |
| Equilibration skip | MD steps &lt; 1000 |
| Sampling | **25 random** production frames per statepoint (`seed=42`) |
| Reference forces | Full-model `params.txt` |
| Metric | RMS \(\|\mathbf{F}_\text{pruned} - \mathbf{F}_\text{full}\|\) per atom (eV/Å) |

## Outputs (gitignored)

- `snapshots/{statepoint}_step{#####}/data.in`
- `md_holdout_manifest.json`
- `forces/*.npy` — cached force arrays
- `metrics/md_force_deviation_long.csv` — consumed by `pruning_analysis/`

## Figures (via pruning_analysis)

- `md_force_deviation_vs_retention.png`
- `md_force_deviation_by_case.png`
