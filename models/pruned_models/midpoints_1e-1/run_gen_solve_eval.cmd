#!/bin/bash
#SBATCH -J mid_1e-1
#SBATCH -A ACCOUNT
#SBATCH -N 6
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -p pdebug
#SBATCH --mem=0
#SBATCH -o gse_stdout_%j
#SBATCH -e gse_stderr_%j
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

BASE=${WORK}/pruned_midpoints_1e-1
GEN_EXE=${REPO_ROOT}/chimes_lsq-LLfork/build/chimes_lsq
LSQ_PY=${REPO_ROOT}/chimes_lsq-LLfork/build/chimes_lsq.py
PYTHON=python3
NNODES=6

# gen A/b (if missing) then DLASSO solve at lambda=1e-1 -- identical to the other
# CN 1e-1 runs, only the prune rate (37.5% / 62.5%) differs.
do_one() {
    d="$1"; cd "$d" || return 1
    b=$(basename "$d")
    grep -q ENDFILE params.txt 2>/dev/null && { echo "SKIP(done) $b"; return 0; }
    nf=$("${PYTHON}" -c "import json;print(json.load(open('fps_meta.json'))['n_selected'])")
    ntask=$(( nf < 112 ? nf : 112 ))
    if [ ! -s A.txt ]; then
        srun -N1 -n ${ntask} --exclusive "${GEN_EXE}" fm_setup.in > fm_setup.log 2> gen.err
        [ -s A.txt ] || { echo "GENFAIL $b"; return 1; }
    fi
    "${PYTHON}" "${LSQ_PY}" --algorithm dlasso --alpha 0.1 \
        --nodes 1 --cores ${ntask} --mpistyle srun > params.txt 2> solve.err
    grep -q ENDFILE params.txt && echo "SOLVED $b" || echo "INCOMPLETE $b"
}
export -f do_one; export BASE GEN_EXE LSQ_PY PYTHON

echo "PHASE1 gen+solve start $(date)"
for d in "${BASE}"/a*_pct*_rep*/; do
    grep -q ENDFILE "$d/params.txt" 2>/dev/null && continue
    while (( $(jobs -rp | wc -l) >= NNODES )); do wait -n; done
    do_one "$d" &
done
wait
nconv=$(for d in "${BASE}"/a*_pct*_rep*/; do grep -q ENDFILE "$d/params.txt" 2>/dev/null && echo x; done | wc -l)
echo "PHASE1 done $(date)  converged: ${nconv}/50"

echo "PHASE2 per-SP eval start $(date)"
rm -f "${BASE}"/results/perSP_shard_*.csv
for i in $(seq 0 $((NNODES-1))); do
    srun -N1 -n1 -c 112 --exclusive "${PYTHON}" "${BASE}/eval_shard_perSP.py" "$i" "${NNODES}" &
done
wait

"${PYTHON}" - <<'PYEOF'
import glob, csv
BASE="${WORK}/pruned_midpoints_1e-1"
rows=[]
for f in sorted(glob.glob(BASE+"/results/perSP_shard_*.csv")):
    rows += list(csv.DictReader(open(f)))
out=BASE+"/force_rmse_perSP_mid_1e-1.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["model","alpha","pct","rep","statepoint","force_rmse_eVA"])
    w.writeheader(); w.writerows(rows)
print(f"merged {len(rows)} rows ({len({r['model'] for r in rows})} models) -> {out}")
PYEOF
echo "ALL DONE $(date)"
