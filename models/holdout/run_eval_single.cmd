#!/bin/bash
#SBATCH -J md_ho_eval
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 00:15:00
#SBATCH -p skx-dev
#SBATCH -A TG-CHM250118
#SBATCH -o md_ho_eval_%A_%a.out
#SBATCH -e md_ho_eval_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export MULTIELEMENT_ROOT="/work2/09982/blaubach/stampede3/multielement_study"
source "${MULTIELEMENT_ROOT}/setup/env.sh"
module load intel/24.0 impi/21.11 python

python3 evaluate_md_forces.py --ntasks 1 --pruned-task-index "${SLURM_ARRAY_TASK_ID}"
