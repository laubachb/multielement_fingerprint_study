# FPS α probe — training-set construction diagnostics

Analyzes whether **which fingerprint α you FPS in** materially changes the
training subset, and how uniformly each subset covers the CN thermodynamic grid.

Secondary probes for: *which α is best for training-set construction?*
**MLIP quality** validation uses `models/statepoint_eval/` (RDF) and
`models/holdout/` (MD hold-out forces) once fits complete; summary figures in
`models/pruning_analysis/`.

## Analyses

| Script | Probe | Question answered |
|--------|-------|-------------------|
| `overlap_across_alpha.py` | Frame-set overlap | Do α=0 and α=1 pick the same frames at fixed retention? |
| `statepoint_coverage.py` | Statepoint coverage | Which FPS α fills (T, ρ, N%) bins most uniformly? |

## Run

```bash
cd models/fps_alpha_probe
python run_probe.py
python run_probe.py --retentions 0.10 0.20
```

Requires `models/sampling/results/` and `models/statepoint_eval/statepoints.json`.

## Outputs (`output/`, gitignored)

- `overlap/jaccard_summary.csv` — pairwise Jaccard for every (retention, rep, αₐ, αᵦ)
- `overlap/jaccard_mean_{pct}pct.csv` — mean 5×5 matrix over replicates
- `coverage/counts_long.csv`, `coverage/uniformity_metrics.csv`
- `overlap/jaccard_mean_{pct}pct.csv` (tables only; figures are macro panels below)

### Interpreting overlap

- **Jaccard ≈ 1** → nearly identical frame sets; α may not matter much.
- **Jaccard ≈ 0** → disjoint selections; α strongly steers diversity.

### Interpreting coverage

- **`coverage_fraction`**: fraction of 10 statepoint cases with ≥1 training frame.
- **`entropy_norm`**: 1 = uniform across cases; 0 = concentrated in one case.
- **`cv_count`**: CV of per-case counts (lower = more uniform).

## System: carbon–nitrogen (CN)

All fingerprints come from the CN training corpus (`models/fingerprints/a*_fingerprints/`,
~297 mixed C+N frames spanning 10 thermodynamic statepoints in `full_dft.xyzf`).
This is **not** the HEA study.

## 5% convergence grid (FPS only, no training)

Probe retention convergence every 5% without fitting ChIMES models:

```bash
cd models/fps_alpha_probe
python run_convergence_fps.py              # FPS + convergence plots
python run_convergence_fps.py --skip-fps   # re-plot from existing results
```

Writes FPS selections to `models/sampling/results_convergence_5pct/` (separate
from the 1/10/20% training subsets in `models/sampling/results/`).

Outputs: `output/convergence/*.csv`

### Publication figures (macro only)

```bash
python make_probe_figures.py
```

`output/figures/` contains exactly four macro panels:

| Figure | Description |
|--------|-------------|
| `panel_convergence_summary.png` | 2×2: Jaccard 0v1, coverage, entropy, cases covered |
| `uniformity_coverage_heatmap.png` | FPS α × retention coverage grid |
| `jaccard_heatmap_evolution.png` | Jaccard matrices at 10–70% retention |
| `coverage_statepoint_x_retention.png` | Statepoint × retention heatmaps per α |

## Frame → statepoint mapping

Anchor frames in `statepoints.json` (1, 21, 41, …) define contiguous blocks in
`full_dft.xyzf`. Each `frame_N` is assigned to the case whose block contains N.
