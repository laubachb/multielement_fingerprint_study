# CN pruning impact analysis

Plots how FPS training-set pruning (1% / 10% retention) affects ChIMES model
quality across three validation channels:

1. **In-sample training RMSE** — force error on the pruned training set
2. **Statepoint RDF deviation** — MD g(r) vs full (100%) model at 10 thermodynamic points
3. **MD hold-out forces** — force RMSE on random frames from full-model statepoint MD

## Run

```bash
cd models/pruning_analysis
python make_plots.py
```

### Data requirements

| Metric | Source |
|--------|--------|
| `training_rmse` | `models/pruned_models/runs/*/Ax.txt`, `b.txt` |
| `rdf_deviation_*` | `models/statepoint_eval/runs/{model}/{statepoint}/rdf.dat` (reference: `full/`) |
| `force_deviation_ev_a` | `models/holdout/metrics/md_force_deviation_long.csv` |

Pruned fits must have complete `params.txt` (`ENDFILE`). RDF and hold-out panels
appear only when the corresponding outputs exist.

### Hold-out pipeline

```bash
cd models/holdout
python extract_md_holdout.py
bash submit_eval_batched.sh    # full cache
bash submit_eval_pruned.sh     # pruned models (re-run to queue more)
python merge_metrics.py
cd ../pruning_analysis && python make_plots.py
```

See `models/holdout/README.md` for batched LAMMPS details and SLURM notes.

## Metrics

| Metric | Definition |
|--------|------------|
| `training_rmse` | In-sample force RMSE from `Ax.txt` vs `b.txt` (eV/Å) |
| `rdf_deviation_pct` | RMS relative error of g(r) vs full model, mean over C–C, C–N, N–N |
| `rdf_deviation_cn_pct` | Same, C–N pair only |
| `force_deviation_ev_a` | RMS \(\|\mathbf{F}_\text{pruned} - \mathbf{F}_\text{full}\|\) per atom on MD hold-out frames (eV/Å) |

Aggregated over available replicates per (FPS α, retention %).

## Outputs (`output/`, gitignored)

**Tables:** `output/metrics/training_rmse_long.csv`, `rdf_deviation_long.csv`, aggregated summaries; hold-out CSV copied from `holdout/metrics/` when present.

**Figures:**
- `training_rmse_vs_retention.png` — force error vs retention, lines per α
- `training_rmse_heatmap.png` — α × retention heatmap with replicate counts
- `rdf_deviation_vs_retention.png` — mean RDF deviation vs retention
- `rdf_deviation_by_case.png` — 10 statepoint panels
- `rdf_deviation_heatmap_by_case.png` — case × retention per α
- `rdf_cn_deviation_by_case.png` — C–N RDF by case
- `md_force_deviation_vs_retention.png` — MD hold-out force error vs retention (when hold-out metrics exist)
- `md_force_deviation_by_case.png` — hold-out force error by statepoint case

## Tracked in git

`*.py`, this README. All `output/` artifacts are gitignored.
