# HEA study

High-entropy alloy fingerprint workflows across ChIMES α values.

## Tracked in git

| Path | Contents |
|------|----------|
| `alpha_*-histograms/*.sh`, `*.py` | Top-level workflow scripts only |
| `chimes_model/fm_setup.in` | Fitting setup |
| `chimes_model/run*.cmd` | ChIMES LSQ launch commands |
| `sampling/run_fps_sampling.py` | HEA FPS frame selector |
| `pruned_models/midpoints_1e-5/` | 37.5 % / 62.5 % retention generation (λ = 1e-5) |
| `test_errors/` | deviation-from-full-model eval + pruning-curve figure |
| `latent_space_visuals_expanded/` | HEA UMAP degeneracy figure + generation script |

CN shared parameters: `element_switching/model/params.txt` (not `chimes_model/params.txt`).

## Latent-space visualization

```bash
cd latent_space_visuals_expanded
python make_umap_degeneracy_expanded.py
```

Produces `umap_degeneracy_expanded.png` — a 2×5 UMAP panel (all frames + mixed-only
rows) across α = 0, 0.25, 0.50, 0.75, 1. Histograms are read from
`transfer_to_local-Apr2026/`; composition labels from `alpha_0-histograms/` cluster
files.

## Pruning-retention study

The mixed-composition FPS-pruned grid (25/50/75 %) is extended with 37.5 % / 62.5 %
midpoints under [`pruned_models/midpoints_1e-5/`](pruned_models/midpoints_1e-5/)
(DLASSO λ = 1e-5). Models are scored as **deviation from the full-data (`pct100`)
model** on the mixed holdout frames — see [`test_errors/`](test_errors/) for the
prediction/deviation pipeline and the pruning-curve figure.

## Local data (gitignored)

- `alpha_*-histograms/frame_*/` — per-frame LAMMPS / histogram outputs
- `chimes_model/` — all fitting outputs except files above
- `frame_clusters/`, `lmp_setup/`, `transfer_to_local-Apr2026/`, `data/archives/`
