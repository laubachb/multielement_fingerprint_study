#!/usr/bin/env bash
# Submit batched MD hold-out force evaluation:
#   - task 0: one LAMMPS run for all full-model snapshots
#   - tasks 1..N: one LAMMPS run per pruned model (up to 48 concurrent on skx-dev)
#
#   cd models/holdout
#   bash submit_eval_batched.sh

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f md_holdout_manifest.json ]]; then
    echo "Missing md_holdout_manifest.json — run: python extract_md_holdout.py" >&2
    exit 1
fi

read -r NSNAPS NMODELS NTASKS <<EOF
$(python3 - <<'PY'
from pathlib import Path
from evaluate_md_forces import discover_models, load_manifest, write_batched_task_list, resolve_full_params

manifest = load_manifest(Path("md_holdout_manifest.json"))
models = discover_models(None)
tasks = write_batched_task_list(models, resolve_full_params())
print(len(manifest), len(models), len(tasks))
PY
)
EOF

if [[ "${NMODELS}" -eq 0 ]]; then
    echo "No completed pruned models found." >&2
    exit 1
fi

echo "Cancelling any legacy md_holdout_all / md_holdout_rev jobs..."
scancel -u "${USER}" --name=md_holdout_all 2>/dev/null || true
scancel -u "${USER}" --name=md_holdout_rev 2>/dev/null || true

echo "Snapshots: ${NSNAPS}  Models: ${NMODELS}  Batched LAMMPS runs: ${NTASKS}"
echo "  (was ${NSNAPS} + $((NSNAPS * NMODELS)) per-snapshot launches)"

FULL_JOB=$(sbatch --parsable --array=0 run_eval_batched.cmd)
echo "Submitted full-model batched job: ${FULL_JOB} (array task 0)"

PRUNED_MAX=48
PRUNED_JOB=$(sbatch --parsable --array="1-${NMODELS}%${PRUNED_MAX}" --dependency="afterok:${FULL_JOB}" run_eval_batched.cmd)
echo "Submitted pruned batched array: ${PRUNED_JOB} (tasks 1-${NMODELS}, max ${PRUNED_MAX} concurrent)"

echo ""
echo "Monitor: squeue -u ${USER} -n md_holdout_batched"
echo "When complete: python merge_metrics.py && cd ../pruning_analysis && python make_plots.py"
