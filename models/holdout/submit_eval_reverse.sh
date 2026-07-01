#!/usr/bin/env bash
# Submit reverse-order skx-dev job (parallel with forward md_holdout_all).
#
#   bash submit_eval_reverse.sh

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f md_holdout_manifest.json ]]; then
    echo "Missing md_holdout_manifest.json" >&2
    exit 1
fi

JOB_ID=$(sbatch --parsable run_eval_all_reverse.cmd 2>&1 | awk '/^[0-9]+$/{print; exit}')
if [[ -z "${JOB_ID}" ]]; then
    echo "sbatch failed" >&2
    exit 1
fi

echo "Submitted reverse debug job=${JOB_ID}"
echo "Monitor: squeue -j ${JOB_ID}"
