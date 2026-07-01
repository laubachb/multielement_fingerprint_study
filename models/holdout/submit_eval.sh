#!/usr/bin/env bash
# Submit legacy per-snapshot MD hold-out force evaluation (slow; use submit_eval_batched.sh).
#
#   cd models/holdout
#   python extract_md_holdout.py
#   bash submit_eval_batched.sh     # recommended
#   bash submit_eval_pruned.sh      # pruned only, diverse-first
#   bash submit_eval.sh             # legacy

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f md_holdout_manifest.json ]]; then
    echo "Missing md_holdout_manifest.json — run: python extract_md_holdout.py" >&2
    exit 1
fi

read -r NSNAPS NMODELS NTASKS <<EOF
$(python3 - <<'PY'
from pathlib import Path
from evaluate_md_forces import discover_models, load_manifest

manifest = load_manifest(Path("md_holdout_manifest.json"))
models = discover_models(None)
print(len(manifest), len(models), len(manifest) * len(models))
PY
)
EOF

if [[ "${NMODELS}" -eq 0 ]]; then
    echo "No completed pruned models found." >&2
    exit 1
fi

echo "Snapshots: ${NSNAPS}  Models: ${NMODELS}  LAMMPS evals: $((NSNAPS + NTASKS))"
echo "Submitting single skx-dev job (run_eval_all.cmd)..."

JOB_ID=$(sbatch --parsable run_eval_all.cmd 2>&1 | awk '/^[0-9]+$/{print; exit}')
if [[ -z "${JOB_ID}" ]]; then
    echo "sbatch failed" >&2
    exit 1
fi

echo "Submitted job=${JOB_ID}"
echo "Monitor: squeue -j ${JOB_ID}"
echo "Re-run bash submit_eval.sh to resume (--skip-existing on cached forces)."
