# HEA test-error evaluation (deviation from full model)

The HEA metric is **ChIMES-vs-ChIMES**: each pruned model's departure from the
full-data (`pct100`) model on the mixed holdout frames — which is, by construction,
the force error relative to the full model (full = Δ 0). No DFT reference is used
(the HEA DFT comparison is handled separately; see the study memo). Covers all five
retentions, 25 / 37.5 / 50 / 62.5 / 75 % (the 37.5/62.5 %
[midpoints](../pruned_models/midpoints_1e-5/) included).

## Pipeline

```bash
cd hea_study/test_errors

# 1. Predict forces/energies on the 23 mixed holdout frames for every model
#    (per-frame checkpointed; reads params from ../pruned_models/runs_holdout20_mixed_clean/)
python eval_predict.py <model_name>        # e.g. a000_pct038_rep00, pct100

# 2. Deviation of each model from pct100 -> deviation_permodel.csv / deviation_summary.csv
python deviation.py

# 3. Pruning curve (viridis α ramp, median + IQR)
python plot_drmse_relative.py              # -> test_error_drmse_relative_hea.png
```

| Tracked | Description |
|---------|-------------|
| `eval_predict.py` | ChIMES force/energy prediction on the holdout frames (set `CHIMES_CALC_LIB`) |
| `deviation.py` | Force/energy deviation of each model from `pct100`, aggregated over reps |
| `plot_drmse_relative.py` | Deviation-vs-retention figure, one line per α |
| `test_error_drmse_relative_hea.png` | Output figure |

## Data (gitignored)

| File | Contents |
|------|----------|
| `deviation_permodel.csv` | per-model force/energy deviation (input to the plot) |
| `deviation_summary.csv` | mean ± std aggregated by (α, retention) |
| `results_pred/<model>/` | cached per-frame predictions |
| `test_frames_mixed.npz` | the 23 mixed holdout frames |

Midpoint models are evaluated by symlinking their run dirs into
`../pruned_models/runs_holdout20_mixed_clean/` so `eval_predict.py`/`deviation.py`
pick them up alongside the 25/50/75 % grid.
