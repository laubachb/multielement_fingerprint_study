# FPS sampling

Farthest-point sampling (FPS) on CN cluster-graph fingerprints at each ChIMES
α value, for training-set construction.

## Run

```bash
cd models/sampling
python run_fps_sampling.py
```

Defaults:

| Setting | Value |
|---------|-------|
| α spaces | 0.00, 0.25, 0.50, 0.75, 1.00 |
| Retention | 1%, 10%, 50% |
| Replicates | 3 (seeds 42, 43, 44) |
| Distance | Euclidean in zero-dropped histogram space |

## Outputs (`results/`, gitignored)

```
results/
  manifest.json
  summary.csv
  alpha_0.00/pct_001/replicate_00/selected_frames.txt
  ...
```

`selected_frames.txt` lists frames in FPS selection order (first line = random
seed frame, each subsequent line maximizes distance to the set so far).

Downstream: `models/pruned_models/` fits ChIMES on each subset;
`models/fps_alpha_probe/` analyzes overlap/coverage; `models/holdout/` and
`models/statepoint_eval/` validate fitted models.
