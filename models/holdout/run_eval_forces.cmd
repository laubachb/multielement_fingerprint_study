#!/bin/bash
#SBATCH -J md_holdout_forces
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 04:00:00
#SBATCH -p skx-dev
#SBATCH -A TG-CHM250118
#SBATCH -o md_holdout_forces_%j.out
#SBATCH -e md_holdout_forces_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export MULTIELEMENT_ROOT="/work2/09982/blaubach/stampede3/multielement_study"
source "${MULTIELEMENT_ROOT}/setup/env.sh"
module load intel/24.0 impi/21.11 python

python3 evaluate_md_forces.py --ntasks 1
