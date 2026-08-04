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
├── scripts/                 repo-level helpers + cross-study eval job
├── figures/                 paper figures + generation scripts (figures/scripts/)
├── element_switching/       CN element-switching validation (graphite + liquid)
│   ├── cn_switching_image/  element-switching figure + script (tracked)
│   └── data/                local outputs (gitignored)
├── models/
│   ├── workflows/           tracked scripts (fingerprint + full_model)
│   ├── sampling/            FPS pruned-frame selection (tracked scripts)
│   ├── pruned_models/       ChIMES fitting on FPS subsets (tracked scripts)
│   │   └── midpoints_1e-1/  37.5 % / 62.5 % retention generation (λ=1e-1)
│   ├── test_errors/         CN relative-to-full test-error figure (λ=1e-1)
│   └── fingerprints/        CN α-sweep data trees (gitignored)
├── hea_study/
│   ├── alpha_*-histograms/  HEA fingerprint workflows (scripts tracked)
│   ├── sampling/            HEA FPS sampler (tracked)
│   ├── chimes_model/        HEA fitting setup (fm_setup.in, run*.cmd tracked)
│   ├── pruned_models/
│   │   └── midpoints_1e-5/  37.5 % / 62.5 % retention generation (λ=1e-5)
│   ├── test_errors/         HEA deviation-from-full eval + figure
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

## Pruning-retention study & test-error evaluation

FPS data pruning is evaluated across dataset-retention fractions for both systems.
The base grids retain 25 / 50 / 75 % of the training frames; the **midpoint**
directories add 37.5 % and 62.5 % (the exact frame-count midpoints) so the pruning
curves are well resolved.

| System | Generation | Retentions | Regularization | Test-error metric |
|--------|-----------|-----------|----------------|-------------------|
| CN | `models/pruned_models/` (+ `midpoints_1e-1/`) | 25 / 37.5 / 50 / 62.5 / 75 % | DLASSO **λ = 1e-1** | per-statepoint force RMSE vs DFT, **relative to the full-data model** |
| HEA | `hea_study/pruned_models/` (+ `midpoints_1e-5/`) | 25 / 37.5 / 50 / 62.5 / 75 % | DLASSO **λ = 1e-5** | force **deviation from the full-data (`pct100`) model** on mixed holdout frames |

Each midpoint directory is self-documenting (FPS → gen → solve → eval): see
[`models/pruned_models/midpoints_1e-1/`](models/pruned_models/midpoints_1e-1/) and
[`hea_study/pruned_models/midpoints_1e-5/`](hea_study/pruned_models/midpoints_1e-5/).
Test-error figures and their data live in
[`models/test_errors/`](models/test_errors/) and
[`hea_study/test_errors/`](hea_study/test_errors/);
[`scripts/eval_both_midpoints.cmd`](scripts/) evaluates both systems in one job.

> **Note on regularization.** The two systems use different DLASSO λ: CN's larger
> λ = 1e-1 converges cleanly on the small, correlated pruned CN fits, while the HEA
> clean grid uses λ = 1e-5. Keep them distinct when comparing.

## `figures/` — paper figures

Manuscript figures with their generation scripts. Every figure (except two that are
image-only) regenerates from a script in
[`figures/scripts/`](figures/scripts/) against data already in the working tree
(some git-ignored). Notably, the UMAP figures rebuild from **cached embeddings**, so
the `umap` package is not required to regenerate them.

```bash
cd figures/scripts && python3 make_theory_schematic.py   # -> ../theory_schematic.png
```

See [`figures/README.md`](figures/) for the full figure → script → data map.

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
```

## Dependencies

- [chimes_calculator-LLfork](https://github.com/LindseyLab-umich/chimes_calculator-LLfork) — LAMMPS interface, histogram executable
- [chimes_lsq-LLfork](https://github.com/LindseyLab-umich/chimes_lsq-LLfork) — ChIMES parameter fitting
- Python: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, and `umap-learn`
  (only to *recompute* UMAP embeddings — the paper UMAP figures regenerate from
  cached embeddings without it)

Install ChIMES into `external/` via `setup/install_chimes.sh`. Legacy local forks
(`chimes_calculator-myLLfork/`, etc.) may exist on disk but are gitignored.

## Citation

If you use these workflows, please cite the associated work on element-aware
cluster-graph fingerprints for multi-component MLIP development (Laubach et al.,
*J. Chem. Theory Comput.*, in preparation).
