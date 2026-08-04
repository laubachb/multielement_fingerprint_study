# CN test-error evaluation (λ = 1e-1)

Force-error of the FPS-pruned CN models **relative to the full-data model**, by state
point, at DLASSO λ = 1e-1. For each pruned model the per-statepoint force RMSE vs DFT
is differenced against the full-data model at the same λ:

```
Δ = (pruned-model F-RMSE) − (full@1e-1 F-RMSE)     per state point
Δ = 0  is the full-data model (dashed reference)
```

Covers all five retentions — 25 / 37.5 / 50 / 62.5 / 75 % (the 37.5/62.5 %
[midpoints](../pruned_models/midpoints_1e-1/) fold in seamlessly).

## Regenerate

```bash
cd models/test_errors
python plot_frmse_3x3_reltofull_1e-1.py     # -> test_error_frmse_3x3_relative_1e-1.png
```

| Tracked | Description |
|---------|-------------|
| `plot_frmse_3x3_reltofull_1e-1.py` | 3×3 per-statepoint figure (viridis α ramp, median + IQR band over reps) |
| `test_error_frmse_3x3_relative_1e-1.png` | Output figure |

## Input data (`data/`, gitignored)

| File | Contents |
|------|----------|
| `force_rmse_perSP_1e-1.csv` | pruned per-statepoint F-RMSE, 25/50/75 % |
| `force_rmse_perSP_mid_1e-1.csv` | pruned per-statepoint F-RMSE, 37.5/62.5 % midpoints |
| `force_rmse_result_full_1e-1.txt` | full-data model per-statepoint F-RMSE (Δ = 0 reference) |

The CSVs are produced by the pruned-model eval (`eval_shard_perSP.py`) under
[`../pruned_models/midpoints_1e-1/`](../pruned_models/midpoints_1e-1/) and the
corresponding 25/50/75 % grid.
