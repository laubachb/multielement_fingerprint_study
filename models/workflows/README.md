# models/workflows

Tracked scripts for CN fingerprint generation and full-corpus ChIMES fitting.

| Subdir | Purpose |
|--------|---------|
| `fingerprints/` | CN α-sweep histogram batch scripts + UMAP analysis |
| `full_model/` | ChIMES Amat generation / solve scripts |

Fingerprint **data** (`a*_fingerprints/frame_*`) lives in `../fingerprints/` (gitignored).

Other tracked workflow directories under `models/` (not in this tree):

| Directory | Purpose |
|-----------|---------|
| `sampling/` | FPS frame selection |
| `pruned_models/` | ChIMES fits on pruned subsets |
| `statepoint_eval/` | Statepoint NVT MD + RDF |
| `holdout/` | MD hold-out force validation |
| `pruning_analysis/` | Pruning impact figures |
| `fps_alpha_probe/` | FPS α construction diagnostics |
