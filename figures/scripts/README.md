# Figure-generation scripts

Scripts that produce the paper figures in `../` (one level up). They were authored
against the original study tree, so their paths are **relative to that layout**
(`Path(__file__).parents[N]`, `writing/analysis/output/`, the fingerprint/UMAP
caches, etc.), not to `figures/scripts/`. The input data they read is mirrored into
`../data/` (git-ignored); point each script at those files (or restore the original
tree) before running.

## Figure → script → data inputs

| Figure (`../`) | Plot script | Data script(s) | Input data (`../data/`) |
|---|---|---|---|
| `cn_hea_fps_overlap_combined.png` | `combine_fps_overlap.py` | `descriptor_space_metrics.py` (CN Jaccard), `hea_mixed_fps_overlap.py` (HEA Jaccard) | `fps_jaccard_summary.csv`, `hea_mixed_fps_jaccard_summary.csv`, `cn_fingerprints.npz`, `hea_fingerprints.npz` |
| `cn_umap_degeneracy.png` | `cn_umap_energy.py` | `make_umap_degeneracy.py`, `umap_data.py` (UMAP embedding) | `cn_fingerprints.npz`, `cn_umap_embeddings.npz` |
| `hea_umap_degeneracy.png` | `make_umap_degeneracy_expanded.py` | (self-contained UMAP) | `umap_embeddings_3row.npz`, HEA histogram trees* |
| `hea_frame_fingerprints.png` | `hea_frame_fingerprints.py` | — | HEA histogram trees* |
| `theory_schematic.png` | `make_theory_schematic.py` | — | none (drawn programmatically) |

\* The raw per-frame histogram trees (`alpha_*-histograms/frame_*/*.hist`) are the
fingerprint pipeline's output and are **not** shipped here (large; git-ignored
repo-wide). Regenerate them with the fingerprint workflow, or use the cached
`*_fingerprints.npz` / `*_embeddings.npz` where a script supports it.

## Not included

`cn_fps_selection_density.png` and `cn_element_switching_graphite.png` are shipped
as images only — no script in the source tree writes those exact filenames
(likely renamed from another output or produced interactively).
