# Element-Aware Cluster-Graph Fingerprints for Multi-Element ChIMES

Computational workflows supporting development and validation of **element-aware
cluster-graph fingerprints** for machine-learned interatomic potential (MLIP)
development with [ChIMES](https://github.com/LindseyLab-umich/chimes_calculator-LLfork).

This repository tracks **scripts and configuration only**. Large simulation outputs
(fingerprints, trajectories, fitted models) remain on disk at the paths below and
are gitignored.

## Scientific context

Multi-element systems—alloys, compounds, interfaces—require descriptors that capture
both **local geometry** and **chemical composition**. Existing MLIP descriptors
(SOAP, symmetry functions, GNN embeddings) typically embed composition implicitly
within structural features, with no tunable control over how much structure versus
composition drives configurational similarity.

This study implements a **hybrid cluster dissimilarity metric** integrated with
ChIMES fingerprinting:

\[
D_{\text{total}} = \alpha \, \tilde{D}_{\text{struct}} + (1 - \alpha) \, \tilde{D}_{\text{comp}}
\]

| α value | Interpretation |
|---------|----------------|
| **α = 0** | Pure **composition** dissimilarity (position-aware for 3b/4b clusters via centrality ranking; 1-Wasserstein for 2b) |
| **α = 1** | Pure **structural** dissimilarity (Morse-transformed interatomic distances) |
| **0 < α < 1** | Convex blend — tunable structure–composition tradeoff |

Key design choices:

- **Structural component:** sorted Morse-transformed edge lengths; Euclidean distance in normalized \([-1,1]\) space.
- **Compositional component (2-body):** position-agnostic 1-Wasserstein distance on atomic descriptors.
- **Compositional component (3-/4-body):** atoms ranked by geometric centrality (edge-sum); Euclidean distance on centrality-ordered element vectors — distinguishes e.g. B at the apex vs. base of an isosceles triangle at fixed stoichiometry.
- **Fingerprints:** normalized pairwise distances binned into histograms (300 bins over 2b + 3b + 4b clusters per frame).

The workflows here support two application areas from the associated manuscript:

1. **Carbon–nitrogen (CN)** — element-switching validation and α-sweep latent-space
   analysis across diverse thermodynamic state points.
2. **High-entropy alloys (HEA)** — extension to multi-principal-element compositions
   and ChIMES model fitting.

Shared ChIMES parameters for CN fingerprinting:
`element_switching/model/params.txt`.

---

## Repository layout

```
multielement_study/
├── setup/                   environment variables + ChIMES install
├── data/                    repo-root archives + layout docs
├── scripts/                 reorganize / cleanup helpers
├── element_switching/       CN element-switching validation (graphite + liquid)
│   ├── cn_switching_image/  element-switching figure + script (tracked)
│   └── data/                local outputs (gitignored)
├── models/
│   ├── workflows/           tracked scripts (fingerprint + full_model)
│   ├── sampling/            FPS pruned-frame selection (tracked scripts)
│   ├── pruned_models/       ChIMES fitting on FPS subsets (tracked scripts)
│   ├── statepoint_eval/     NVT MD + RDF at CN thermodynamic statepoints
│   ├── holdout/             MD hold-out force validation (pruned vs full)
│   ├── pruning_analysis/    training RMSE + RDF + hold-out impact figures
│   ├── fps_alpha_probe/     FPS α overlap / coverage diagnostics
│   └── fingerprints/        CN α-sweep data trees (gitignored)
├── hea_study/
│   ├── alpha_*-histograms/  HEA fingerprint workflows (scripts tracked)
│   ├── chimes_model/        HEA fitting setup (fm_setup.in, run*.cmd tracked)
│   ├── latent_space_visuals_expanded/  HEA UMAP figure + script (tracked)
│   └── data/                archives + frame outputs (gitignored)
└── external/                cloned ChIMES forks (gitignored)
```

---

## `element_switching/` — CN element-switching validation

**Purpose:** Demonstrate that the α parameter decouples structural and compositional
contributions to cluster fingerprints. A fixed atomic configuration is held constant
while C atoms are progressively replaced with N; fingerprint evolution is measured
as α decreases from 1 (structure-only, invariant to element swaps) toward 0
(composition-dominated).

**Systems:**

| Subdirectory | System | Conditions |
|--------------|--------|------------|
| `graphite/` | Crystalline graphite | 1500 K reference frame; ordered lattice scaffold |
| `liquid/` | Disordered C–N liquid | 2000 K, 1.0 g/cc |

**Workflow (per system):**

1. `create_swap_and_alpha_dirs.py` — read a reference `.xyzf`, build substitution
   series (0–100% N in increments), and create `alpha_000` … `alpha_100` subdirs.
2. `xyzf2data.py` — convert structures to LAMMPS data files.
3. `run_all_alphas.sh` — run LAMMPS equilibration and histogram generation at each
   α value for each substitution percentage.
4. `plot_histograms.py` — visualize fingerprint evolution along the C→N pathway.

**Expected outputs** (gitignored under `element_switching/data/` or local
`fingerprints/`):

```
fingerprints/
  graphite_1500K_0262_*pct/     # one dir per N-substitution level
    alpha_000/ … alpha_100/     # one dir per α value
      *.hist                      # 300-bin cluster dissimilarity histogram
      log.lammps, traj.lammpstrj  # LAMMPS outputs (if run)
  *-fingerprints_combined.csv     # aggregated histogram data for plotting
  test.png                        # fingerprint evolution figures
```

**Tracked in git:** `graphite/*.py`, `graphite/*.sh`, `graphite/in.lammps`,
`liquid/*.py`, `liquid/*.sh`, `liquid/in.lammps`, `model/params.txt`,
`cn_switching_image/` (figure + script).

### `cn_switching_image/` (tracked script + figure)

Publication-style stacked fingerprint plot for graphite C→N substitution (0–100% N)
across α = 0–1.

```bash
cd element_switching/cn_switching_image
python make_cn_switching_figure.py
```

| Tracked | Description |
|---------|-------------|
| `make_cn_switching_figure.py` | Generation script |
| `cn_switching_fingerprints.png` | Output figure |

Fingerprint data: `element_switching/data/graphite/fingerprints/` (gitignored).

---

## `models/` — CN fingerprint generation, analysis, and MLIP training

**Purpose:** Generate full-frame cluster fingerprints for the CN training corpus
across the α sweep; analyze latent-space degeneracy and frame displacement; build
pruned training sets via farthest-point sampling (FPS); fit ChIMES models.

### `models/fingerprints/` (data, gitignored)

Per-α data trees holding ~300 MD/DFT frames each, spanning CN state points
(temperature, density, composition):

| Directory | α | Role |
|-----------|---|------|
| `a000_fingerprints/` | 0.00 | Pure composition |
| `a025_fingerprints/` | 0.25 | |
| `a050_fingerprints/` | 0.50 | |
| `a075_fingerprints/` | 0.75 | |
| `a100_fingerprints/` | 1.00 | Pure structure |

**Per-frame structure:**

```
a025_fingerprints/
  frame_0/
    frame_0.hist          # 300-bin histogram (column 1 = combined 2b+3b+4b)
    lammps.in, data.lammps
  frame_1/
  …
```

Frames valid across all five α values are used for cross-α UMAP analysis (~276
frames after excluding incomplete/missing entries).

### `models/workflows/` (tracked)

| Subdirectory | Scripts | Purpose |
|--------------|---------|---------|
| `fingerprints/` | `gen_hists_a*.sh` | SLURM batch jobs to generate `*.hist` per frame |
| `fingerprints/` | `gen_missing_hists_a025.sh` | Backfill incomplete α=0.25 frames |
| `fingerprints/` | `clusters_*.sh`, `clusters_xyzf2data.py` | Cluster extraction from xyzf |
| `fingerprints/umap/` | `analyze_alpha_umap.py`, `make_umap_degeneracy.py`, `umap_data.py` | UMAP / PCA / displacement analysis |
| `full_model/` | `gen_Amat.sh`, `solve_Amat.sh`, `rotate_frames.py` | ChIMES LSQ matrix generation and fitting |

**Run fingerprint generation** from the data directory:

```bash
source setup/env.sh
cd models/fingerprints/a025_fingerprints
sbatch ../workflows/fingerprints/gen_hists_a025.sh   # or symlink gen_hists.sh
```

**Run UMAP analysis:**

```bash
cd models/workflows/fingerprints/umap
python analyze_alpha_umap.py
python analyze_alpha_umap.py --recompute   # rebuild cache from *.hist files
```

**Expected analysis outputs** (gitignored under `umap/`):

| Output | Description |
|--------|-------------|
| `umap/cache/` | Cached embeddings, distance matrices, manifest |
| `umap/figures/aligned_umap_panels.png` | Procrustes-aligned UMAP panels (α = 0, 0.25, …, 1) |
| `umap/figures/fingerprint_displacement_vs_alpha.png` | L2 distance in histogram space vs α |
| `umap/figures/alpha_trajectories.png` | Per-frame paths through aligned latent space |
| `umap/figures/pca_fingerprint_sweep.png` | PCA of stacked fingerprints colored by α |
| `umap/figures/frame_metrics.csv` | Per-frame scalar metrics for FPS ranking |

**MLIP training workflow** (`full_model/`): full-corpus ChIMES fitting on all 298
frames (gitignored outputs under `models/full_model/`).

### `models/sampling/` (tracked scripts, gitignored results)

FPS on CN fingerprint vectors at each α (1%, 10%, 50% retention, 3 replicates).

```bash
cd models/sampling && python run_fps_sampling.py
```

Outputs: `results/alpha_*/pct_*/replicate_*/selected_frames.txt`

### `models/pruned_models/` (tracked scripts, gitignored runs)

ChIMES LSQ fitting on each FPS subset (45 runs: 5 α × 3 retentions × 3 replicates).

```bash
cd models/pruned_models
python prepare_runs.py    # extract training.xyzf + fm_setup.in per subset
bash submit_all.sh        # submit gen_Amat → solve_Amat chains
```

Each run under `runs/a*_pct*_rep*/` produces `params.txt`, `A.txt`, `b.txt`, etc.

### `models/statepoint_eval/` (tracked scripts, gitignored runs)

NVT production MD and RDF accumulation at 10 CN thermodynamic statepoints for the
full model and pruned fits. Starting frames and workflow scripts are tracked;
`chimes_params/` (synced copies) and `runs/` are gitignored.

```bash
cd models/statepoint_eval
python prepare_runs.py --sync-params --models full
bash submit_full_model.sh
```

See `models/statepoint_eval/README.md`.

### `models/holdout/` (tracked scripts, gitignored data)

MD hold-out force validation: sample random frames from **full-model** statepoint
trajectories and compare pruned vs full ChIMES forces. Uses batched LAMMPS (one
process per model) for throughput.

```bash
cd models/holdout
python extract_md_holdout.py
bash submit_eval_batched.sh
bash submit_eval_pruned.sh     # diverse-first; re-run to queue more
python merge_metrics.py
```

See `models/holdout/README.md`.

### `models/pruning_analysis/` (tracked scripts, gitignored figures)

Publication figures for pruning impact: training RMSE, RDF deviation, and MD
hold-out force error vs retention and FPS α.

```bash
cd models/pruning_analysis && python make_plots.py
```

See `models/pruning_analysis/README.md`.

### `models/fps_alpha_probe/` (tracked scripts, gitignored output)

Diagnostics for FPS training-set construction (Jaccard overlap, statepoint coverage).
Does not replace hold-out / RDF validation — see `models/holdout/` and
`models/statepoint_eval/`.

---

## `hea_study/` — high-entropy alloy fingerprints and model fitting

**Purpose:** Extend element-aware fingerprinting to HEA compositions; generate
per-frame histograms at multiple α values; fit a HEA ChIMES model from selected
training frames.

### `alpha_*-histograms/` (scripts tracked, frame data gitignored)

Parallel workflow trees, one per α:

| Directory | α |
|-----------|---|
| `alpha_0-histograms/` | 0.00 |
| `alpha_025-histograms/` | 0.25 |
| `alpha_050-histograms/` | 0.50 |
| `alpha_075-histograms/` | 0.75 |
| `alpha_1-histograms/` | 1.00 |

**Typical pipeline** (top-level scripts in each α directory):

1. `split_and_convert_xyzf.sh` / `convert_xyzf_to_data.sh` — prepare per-frame
   LAMMPS inputs from `hea_chimes_format.xyzf`.
2. `submit_lammps_jobs.sh` — run MD / cluster extraction.
3. `gen_hist_all_frames.sh` — batch histogram generation with fixed α.
4. `post_process_lammpsin_files.py` / `run_post_processclusters_each_directory.sh`
   — post-process cluster files.

**Expected outputs** (gitignored):

```
alpha_025-histograms/
  frame_0/
    frame_0.hist
    lammps.in, data files, cluster outputs
  frame_1/
  …
```

### `chimes_model/` (partially tracked)

HEA ChIMES least-squares fitting setup.

| Tracked | Purpose |
|---------|---------|
| `fm_setup.in` | Fitting configuration (frames, basis, constraints) |
| `run_chimeslsq.cmd`, `run_lsqpy.cmd` | Launch ChIMES LSQ on HPC |

Fitting outputs (`A.txt`, `b.txt`, `restart.*.txt`, fitted `params.txt`) are
gitignored. For CN studies, use `element_switching/model/params.txt`; the copy under
`hea_study/chimes_model/params.txt` is a local fitting artifact.

### `latent_space_visuals_expanded/` (tracked script + figure)

2×5 UMAP degeneracy figure for the HEA α sweep (Y-only / Mg-only / Mixed composition
in row 0; mixed-only with degeneracy annotations in row 1).

```bash
cd hea_study/latent_space_visuals_expanded
python make_umap_degeneracy_expanded.py
```

| Tracked | Description |
|---------|-------------|
| `make_umap_degeneracy_expanded.py` | Generation script |
| `umap_degeneracy_expanded.png` | Output figure |

Histogram data: `transfer_to_local-Apr2026/` (gitignored). Composition labels:
`alpha_0-histograms/` cluster files.

---

## Quick start

```bash
# 1. Install ChIMES dependencies
bash setup/install_chimes.sh
source setup/env.sh

# 2. CN fingerprint generation (example: α = 0.25)
cd models/fingerprints/a025_fingerprints
sbatch ../../workflows/fingerprints/gen_hists_a025.sh

# 3. UMAP / α-sweep analysis
cd ../../workflows/fingerprints/umap
python analyze_alpha_umap.py

# 4. Element-switching validation (graphite)
cd element_switching/graphite
python create_swap_and_alpha_dirs.py
sbatch run_all_alphas.sh

# 5. Pruning impact (after pruned fits + statepoint MD + hold-out eval)
cd models/pruning_analysis && python make_plots.py
```

## Dependencies

- [chimes_calculator-LLfork](https://github.com/LindseyLab-umich/chimes_calculator-LLfork) — LAMMPS interface, histogram executable
- [chimes_lsq-LLfork](https://github.com/LindseyLab-umich/chimes_lsq-LLfork) — ChIMES parameter fitting
- Python: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `umap-learn`, `scipy`

Install ChIMES into `external/` via `setup/install_chimes.sh`. Legacy local forks
(`chimes_calculator-myLLfork/`, etc.) may exist on disk but are gitignored.

## Citation

If you use these workflows, please cite the associated work on element-aware
cluster-graph fingerprints for multi-component MLIP development (Laubach et al.,
*J. Chem. Theory Comput.*, in preparation).
