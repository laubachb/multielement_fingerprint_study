# CN pruning convergence analysis

Plots training-set pruning (1% / 10% retention) vs ChIMES fit quality and
statepoint RDF agreement with the full (100%) model.

## Run

```bash
cd models/pruning_analysis
python make_plots.py
```

Requires completed fits (`params.txt` with `ENDFILE`) under `models/pruned_models/runs/`
and RDF outputs under `models/statepoint_eval/runs/` (reference: `full/`).

## Metrics

| Metric | Definition |
|--------|------------|
| `training_rmse` | In-sample force RMSE from `Ax.txt` vs `b.txt` (eV/Å) |
| `rdf_deviation_pct` | RMS relative error of g(r) vs full model, mean over C–C, C–N, N–N |
| `rdf_deviation_cn_pct` | Same, C–N pair only |

Aggregated over all available replicates per (FPS α, retention %).

## Outputs (`output/`, gitignored)

**Tables:** `metrics/training_rmse_long.csv`, `metrics/rdf_deviation_long.csv`, aggregated summaries

**Figures:**
- `training_rmse_vs_retention.png` — force error vs retention, lines per α
- `training_rmse_heatmap.png` — α × retention heatmap with replicate counts
- `rdf_deviation_vs_retention.png` — mean RDF deviation vs retention
- `rdf_deviation_by_case.png` — 10 statepoint panels
- `rdf_deviation_heatmap_by_case.png` — case × retention per α
- `rdf_cn_deviation_by_case.png` — C–N RDF by case
