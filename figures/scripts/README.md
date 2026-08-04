# Figure-generation scripts

Each script here regenerates the corresponding figure in `../` from data already in
the repo working tree. Some inputs are **git-ignored** (kept local, not pushed) — the
figures are self-contained within the checkout, per that convention. Run a script
from this directory:

```
cd figures/scripts && python3 <script>.py
```

## Figure → script → data inputs

| Figure (`../`) | Script | Input data (git-ignored) |
|---|---|---|
| `theory_schematic.png` | `make_theory_schematic.py` | none (drawn programmatically) |
| `cn_hea_fps_overlap_combined.png` | `combine_fps_overlap.py` | `../data/fps_jaccard_summary.csv`, `../data/hea_mixed_fps_jaccard_summary.csv` |
| `cn_umap_degeneracy.png` | `cn_umap_energy.py` | `../../models/workflows/fingerprints/umap/cache/{manifest.json,embeddings.npz}`, `../../models/statepoint_eval/statepoints.json` (tracked) |
| `hea_frame_fingerprints.png` | `hea_frame_fingerprints.py` | `../data/hea_histograms/` (2 frames × 5 α) |
| `hea_umap_degeneracy.png` | `make_umap_degeneracy_expanded.py` | `cache/umap_embeddings_3row.npz`, `cache/umap_meta_3row.npz` |

Companion data scripts (also here, for provenance): `descriptor_space_metrics.py`,
`hea_mixed_fps_overlap.py` (compute the Jaccard CSVs), `make_umap_degeneracy.py`,
`umap_data.py` (compute the UMAP embeddings from raw histograms).

## Notes

- **No `umap` package needed.** `make_umap_degeneracy_expanded.py` rebuilds the
  figure from cached embeddings + a small metadata cache (labels/energies); `umap`
  is imported lazily only under `--recompute`, which also needs the full histogram
  trees + source extxyz (not shipped — regenerate them with the fingerprint pipeline).
- `hea_frame_fingerprints.py` ships only the two plotted frames' histograms; the two
  DFT energies used in the panel titles are inlined (the 16 MB source extxyz is not shipped).

## Not included

`cn_fps_selection_density.png` and `cn_element_switching_graphite.png` are shipped as
images only — no script in the source tree writes those exact filenames.
