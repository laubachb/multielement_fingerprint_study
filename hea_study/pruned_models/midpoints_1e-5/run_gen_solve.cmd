#!/bin/bash
#SBATCH -J hea_mid_1e-5
#SBATCH -A ACCOUNT
#SBATCH -N 6
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -p pdebug
#SBATCH --mem=0
#SBATCH -o gs_stdout_%j
#SBATCH -e gs_stderr_%j
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

BASE=${WORK}/hea_midpoints_1e-5
GEN_EXE=${REPO_ROOT}/chimes_lsq-LLfork/build/chimes_lsq
LSQ_PY=${REPO_ROOT}/chimes_lsq-LLfork/build/chimes_lsq.py
PYTHON=python3
NNODES=6

# HEA basis (CHEBYSHEV 12 5 2, Y/Mg, forces-only). Regularization lambda=1.0E-5,
# matching the other HEA clean runs (NOT the CN 1e-1). gen A/b then DLASSO solve.
do_one() {
    d="$1"; cd "$d" || return 1
    b=$(basename "$d")
    grep -q ENDFILE params.txt 2>/dev/null && { echo "SKIP(done) $b"; return 0; }
    nt=$("${PYTHON}" -c "import json;print(json.load(open('run_manifest.json'))['resources']['ntasks'])")
    if [ ! -s A.txt ]; then
        srun -N1 -n ${nt} --exclusive "${GEN_EXE}" fm_setup.in > fm_setup.log 2> gen.err
        [ -s A.txt ] || { echo "GENFAIL $b"; return 1; }
    fi
    "${PYTHON}" "${LSQ_PY}" --algorithm dlasso --alpha 1.0E-5 \
        --nodes 1 --cores ${nt} --mpistyle srun > params.txt 2> solve.err
    grep -q ENDFILE params.txt && echo "SOLVED $b" || echo "INCOMPLETE $b"
}
export -f do_one; export BASE GEN_EXE LSQ_PY PYTHON

echo "gen+solve start $(date)  (lambda=1e-5, HEA basis)"
for d in "${BASE}"/a*_pct*_rep*/; do
    grep -q ENDFILE "$d/params.txt" 2>/dev/null && continue
    while (( $(jobs -rp | wc -l) >= NNODES )); do wait -n; done
    do_one "$d" &
done
wait
nconv=$(for d in "${BASE}"/a*_pct*_rep*/; do grep -q ENDFILE "$d/params.txt" 2>/dev/null && echo x; done | wc -l)
echo "done $(date)  converged: ${nconv}/100"
grep -h -E "GENFAIL|INCOMPLETE" "${BASE}"/gs_stdout_* 2>/dev/null | sort -u | head
