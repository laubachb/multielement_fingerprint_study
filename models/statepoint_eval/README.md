# CN statepoint MD evaluation

NVT production LAMMPS runs at fixed thermodynamic statepoints for comparing ChIMES
models (full corpus and FPS-pruned fits). Each run starts from a DFT reference frame,
equilibrates, then accumulates partial RDFs (`rdf.dat`).

Source statepoints were copied from `models/pruned_simulations/` (gitignored legacy
tree). This directory is the **tracked** evaluation workflow.

## Layout

```
statepoint_eval/
  statepoints.json           # metadata (T, ρ, N%, full_dft frame index)
  statepoints/               # starting_frame.xyzf per statepoint (tracked)
  chimes_params/
    full/params.txt          # 100% model (from element_switching/model/)
    a000_pct001_rep00/       # added when pruned fits complete
  templates/                 # in.lammps + SLURM cmd
  prepare_runs.py
  runs/                      # generated per (model, statepoint); gitignored
```

## Quick start (full model)

```bash
cd models/statepoint_eval

# Refresh params + build run dirs for the 100% model
python prepare_runs.py --sync-params --models full

# Submit all 10 statepoints on skx (production queue)
bash submit_full_model.sh
```

## Workflow

| Step | Command |
|------|---------|
| Refresh starting frames from legacy tree | `python prepare_runs.py --sync-statepoints` |
| Copy params (full + any finished pruned runs) | `python prepare_runs.py --sync-params` |
| Generate `runs/` directories | `python prepare_runs.py` |
| Submit full model only | `bash submit_full_model.sh` |
| Submit all available models | `bash submit_all.sh` |

## Simulation protocol

Each `runs/{model}/{statepoint}/` directory contains:

- `data.in` — LAMMPS data from `starting_frame.xyzf`
- `params.txt` — ChIMES parameters for that model
- `in.lammps` — NVT production (6000 steps, 1 fs timestep, ~6 ps) with `compute rdf`
- `run_lammps.cmd` — SLURM launcher (48 MPI ranks, `skx` partition by default)

**Outputs** (gitignored): `rdf.dat`, `traj.lammpstrj`, `log.lammps`, `output.txt`.

RDF pairs: C–C (1–1), C–N (1–2), N–N (2–2). Atom types follow sorted element
order from `xyzf2data.py` (C = 1, N = 2).

## Statepoints

| Case | ID | N% | ρ (g/cc) | T (K) | full_dft frame |
|------|-----|-----|----------|-------|----------------|
| 1 | `0.20.3percN_2.0gcc` | 20.3 | 2.0 | 300 | 1 |
| 2 | `1.20.3percN_2.7gcc` | 20.3 | 2.7 | 300 | 21 |
| 3 | `2.20.3percN_3.5gcc` | 20.3 | 3.5 | 6000 | 41 |
| 4 | `3.20.3percN_4.0gcc` | 20.3 | 4.0 | 9000 | 61 |
| 5 | `4.50percN_1.0gcc` | 50 | 1.0 | 1500 | 81 |
| 6 | `5.50percN_2.7gcc` | 50 | 2.7 | 3000 | 101 |
| 7 | `6.50percN_3.5gcc` | 50 | 3.5 | 6000 | 121 |
| 8 | `7.75percN_1.0gcc` | 75 | 1.0 | 300 | 141 |
| 9 | `8.75percN_2.7gcc` | 75 | 2.7 | 3000 | 161 |
| 10 | `9.75percN_3.0gcc` | 75 | 3.0 | 9000 | 181 |

## Adding pruned models

When `models/pruned_models/runs/*/params.txt` finishes (contains `ENDFILE`):

```bash
python prepare_runs.py --sync-params
python prepare_runs.py
bash submit_all.sh --batch-size 5
```
