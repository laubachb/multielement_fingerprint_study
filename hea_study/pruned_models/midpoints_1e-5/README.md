# HEA pruning midpoints (λ = 1e-5)

HEA analog of the CN pruning midpoints. Extends the mixed-composition FPS-pruned
grid with **37.5 %** (35 frames) and **62.5 %** (59 frames) retention — midpoints of
the existing 25/50/75 % tiers (24/47/70 frames of the 94-frame mixed train pool).
Same HEA design of experiments (5 fingerprint-α × 2 retentions × **10 replicates** =
**100 models**) and the same regularization **λ = 1e-5** as the other HEA clean runs
(NB: different from the CN grid's 1e-1).

## Pipeline

```bash
# 1. FPS-select frames from the mixed-clean train pool (same sampler as the grid)
python ../../sampling/run_fps_sampling.py \
    --retentions 0.375 0.625 --replicates 10 --seed 42 \
    --frames-file ../../splits/holdout20_mixed_clean/train_frames.txt \
    --output-dir $WORK/fps_selections

# 2. Build run dirs (training.xyzf + fm_setup.in + run_manifest.json)
python build_midpoint_runs.py

# 3. gen A/b + DLASSO-solve (λ=1e-5), one pdebug job
sbatch run_gen_solve.cmd
```

| File | Role |
|------|------|
| `build_midpoint_runs.py` | Reads the FPS `selected_frames.txt`, extracts frames from `hea_chimes_format_rotated.xyzf`, writes `training.xyzf`, `fm_setup.in`, `run_manifest.json`. |
| `fm_setup.in.template` | HEA basis (`CHEBYSHEV 12 5 2`, Y/Mg, **forces-only**) with `{{NFRAMES}}` placeholder. |
| `run_gen_solve.cmd` | SLURM (pdebug) — build the design matrix then `chimes_lsq.py --algorithm dlasso --alpha 1.0E-5`. Self-skips converged models. |

## Reproducibility inputs (override via environment)

| Var | Default | Contents |
|-----|---------|----------|
| `HEA_FPS_DIR` | `$WORK/fps_selections` | FPS selections from `run_fps_sampling.py` |
| `HEA_SRC_XYZF` | `hea_study/chimes_model/hea_chimes_format_rotated.xyzf` | source trajectory |
| `HEA_FM_TEMPLATE` | `./fm_setup.in.template` | HEA fm_setup template |
| `WORK` | `./runs` | output root for run dirs |

The converged models are evaluated as **deviation-from-full-model** on the mixed
holdout frames — see [`hea_study/test_errors/`](../../test_errors/).
