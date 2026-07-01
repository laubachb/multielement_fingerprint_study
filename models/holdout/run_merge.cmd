#!/bin/bash
#SBATCH -J md_holdout_merge
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p skx-dev
#SBATCH -A TG-CHM250118
#SBATCH -o md_holdout_merge_%j.out
#SBATCH -e md_holdout_merge_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load python3
python3 merge_metrics.py
