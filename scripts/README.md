# Repo-level scripts

| Script | Purpose |
|--------|---------|
| `eval_both_midpoints.cmd` | SLURM (pdebug) job that evaluates the CN **and** HEA pruning-midpoint models in one allocation: CN per-statepoint force-RMSE (`models/pruned_models/midpoints_1e-1/eval_shard_perSP.py`) + HEA deviation-from-`pct100` (`hea_study/test_errors/{eval_predict,deviation}.py`). |
| `reorganize_repo.sh` | One-off layout reorganization helper. |
| `cleanup_repo_layout.sh` | One-off cleanup of stale paths after reorganization. |
| `patch_workflow_paths.sh` | One-off path patching across workflow scripts. |

`eval_both_midpoints.cmd` uses placeholder env vars (`ACCOUNT`, `WORK`,
`CHIMES_CALC_LIB`, …) set for the target cluster before submitting. The
`reorganize_*`/`cleanup_*`/`patch_*` helpers are historical and are not needed to
use the repo.
