#!/bin/bash
#SBATCH -J md_holdout_batched
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 02:00:00
#SBATCH -p skx-dev
#SBATCH -A TG-CHM250118
#SBATCH -o md_holdout_batched_%A_%a.out
#SBATCH -e md_holdout_batched_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export MULTIELEMENT_ROOT="/work2/09982/blaubach/stampede3/multielement_study"
source "${MULTIELEMENT_ROOT}/setup/env.sh"
module load intel/24.0 impi/21.11 python

TASK_ID="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID via --array=}"

EXTRA=()
if [[ "${TASK_ID}" -eq 0 ]]; then
  EXTRA+=(--write-combined-xyzf)
fi

python3 evaluate_md_forces.py --batched --array-task-id "${TASK_ID}" --ntasks 1 "${EXTRA[@]}"
