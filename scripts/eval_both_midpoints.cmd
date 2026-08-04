#!/bin/bash
#SBATCH -J eval_both_mid
#SBATCH -A ACCOUNT
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -p pdebug
#SBATCH --mem=0
#SBATCH -o ${WORK}/eval_both_stdout_%j
#SBATCH -e ${WORK}/eval_both_stderr_%j
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=python3
NNODES=4

# ============ CN midpoints: per-statepoint force RMSE vs DFT (lambda=1e-1) ============
CN=${WORK}/pruned_midpoints_1e-1
echo "=== CN eval start $(date) ==="
rm -f "${CN}"/results/perSP_shard_*.csv
for i in $(seq 0 $((NNODES-1))); do
    srun -N1 -n1 -c 112 --exclusive "${PYTHON}" "${CN}/eval_shard_perSP.py" "$i" "${NNODES}" &
done
wait
"${PYTHON}" - <<'PYEOF'
import glob, csv
CN="${WORK}/pruned_midpoints_1e-1"
rows=[]
for f in sorted(glob.glob(CN+"/results/perSP_shard_*.csv")): rows+=list(csv.DictReader(open(f)))
out=CN+"/force_rmse_perSP_mid_1e-1.csv"
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=["model","alpha","pct","rep","statepoint","force_rmse_eVA"])
    w.writeheader(); w.writerows(rows)
print(f"CN: merged {len(rows)} rows ({len({r['model'] for r in rows})} models) -> {out}")
PYEOF
echo "=== CN eval done $(date) ==="

# ============ HEA midpoints: predict 23 holdout frames, then deviation-from-pct100 ============
TE=${REPO_ROOT}/hea_study/test_errors
MID=${WORK}/hea_midpoints_1e-5
echo "=== HEA predict start $(date) ==="
cd "${TE}"
# one eval_predict.py per midpoint model, parallel across this node's cores
ls -d ${MID}/a*_pct*_rep*/ | xargs -n1 basename | \
    xargs -P 100 -I{} "${PYTHON}" "${TE}/eval_predict.py" {} > "${MID}/hea_predict.log" 2>&1
ndone=$(ls -d ${MID}/a*_pct*_rep*/ | xargs -n1 basename | while read m; do [ -f "${TE}/results_pred/$m/DONE" ] && echo x; done | wc -l)
echo "HEA predictions DONE for ${ndone}/100 midpoint models"
echo "=== HEA deviation (regenerate incl. midpoints) $(date) ==="
"${PYTHON}" "${TE}/deviation.py"
echo "=== ALL DONE $(date) ==="
