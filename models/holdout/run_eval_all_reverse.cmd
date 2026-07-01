#!/bin/bash
#SBATCH -J md_holdout_rev
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH -t 02:00:00
#SBATCH -p skx-dev
#SBATCH -A TG-CHM250118
#SBATCH -o md_holdout_rev_%j.out
#SBATCH -e md_holdout_rev_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export MULTIELEMENT_ROOT="/work2/09982/blaubach/stampede3/multielement_study"
source "${MULTIELEMENT_ROOT}/setup/env.sh"
module load intel/24.0 impi/21.11 python

# Reverse order: last snapshot/model first; skips existing *.npy (parallel with forward job).
python3 evaluate_md_forces.py --ntasks 1 --reverse
