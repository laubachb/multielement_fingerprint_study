#!/bin/bash
#SBATCH -J md_holdout_arr
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 01:00:00
#SBATCH -p skx-dev
#SBATCH -A TG-CHM250118
#SBATCH --array=0-59%15
#SBATCH -o md_holdout_%A_%a.out
#SBATCH -e md_holdout_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export MULTIELEMENT_ROOT="/work2/09982/blaubach/stampede3/multielement_study"
source "${MULTIELEMENT_ROOT}/setup/env.sh"
module load intel/24.0 impi/21.11 python

mapfile -t MODELS < <(python3 - <<'PY'
from evaluate_md_forces import discover_models
for m in discover_models(None):
    print(m["model"])
PY
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]:-}"
if [[ -z "${MODEL}" ]]; then
  echo "No model for array task ${SLURM_ARRAY_TASK_ID}; skipping."
  exit 0
fi

echo "Evaluating hold-out forces for ${MODEL}"
python3 evaluate_md_forces.py --ntasks 1 --models "${MODEL}"
