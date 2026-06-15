#!/bin/bash
#SBATCH -J pct001_train_bundle
#SBATCH -N {{NNODES}}
#SBATCH --ntasks-per-node {{NCORES}}
#SBATCH -t {{WALLTIME}}
#SBATCH -p {{PARTITION}}
#SBATCH -A TG-CHM250118
#SBATCH -o {{PRUNED_MODELS_ROOT}}/pct001_bundle_%j.out
#SBATCH -e {{PRUNED_MODELS_ROOT}}/pct001_bundle_%j.err

set -uo pipefail

export MULTIELEMENT_ROOT="{{MULTIELEMENT_ROOT}}"
source "${MULTIELEMENT_ROOT}/setup/env.sh"
module load intel/24.0 impi/21.11 python

PRUNED_MODELS_ROOT="{{PRUNED_MODELS_ROOT}}"
RUNS_DIR="${PRUNED_MODELS_ROOT}/runs"
NTASKS={{NTASKS}}
NNODES={{NNODES}}

if [[ -x "${CHIMES_LSQ}/build/chimes_lsq" ]]; then
    LSQ_EXE="${CHIMES_LSQ}/build/chimes_lsq"
else
    LSQ_EXE="${MULTIELEMENT_ROOT}/chimes_lsq-LLfork/build/chimes_lsq"
fi

if [[ -f "${CHIMES_LSQ}/build/chimes_lsq.py" ]]; then
    LSQ_PY="${CHIMES_LSQ}/build/chimes_lsq.py"
else
    LSQ_PY="${MULTIELEMENT_ROOT}/chimes_lsq-LLfork/build/chimes_lsq.py"
fi

RUNS=(
{{RUN_LIST}}
)

bundle_exit=0
completed=0
skipped=0

for run_name in "${RUNS[@]}"; do
    run_dir="${RUNS_DIR}/${run_name}"
    if [[ ! -d "${run_dir}" ]]; then
        echo "ERROR: missing run directory ${run_dir}" >&2
        bundle_exit=1
        break
    fi

    cd "${run_dir}"
    echo "========== ${run_name} ($(date -Is)) =========="

    if [[ -f params.txt ]] && grep -q ENDFILE params.txt; then
        echo "SKIP ${run_name}: params.txt already complete"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ ! -f A.txt || ! -f fm_setup.log ]]; then
        echo "GEN ${run_name}"
        gen_exit=0
        ibrun -n "${NTASKS}" "${LSQ_EXE}" fm_setup.in | tee fm_setup.log || gen_exit=$?
        "${MULTIELEMENT_ROOT}/scripts/log_compute_event.sh" chimes_gen "${run_dir}" "${gen_exit}" || true
        if [[ "${gen_exit}" -ne 0 ]]; then
            echo "FAILED gen ${run_name} (exit ${gen_exit})" >&2
            bundle_exit="${gen_exit}"
            break
        fi
    else
        echo "SKIP gen ${run_name}: A.txt present"
    fi

    echo "SOLVE ${run_name}"
    rm -f params.txt solve.err
    solve_exit=0
    python3 "${LSQ_PY}" \
        --algorithm dlasso \
        --alpha 1.0E-5 \
        --nodes "${NNODES}" \
        --cores "${NTASKS}" \
        --mpistyle ibrun > params.txt 2> solve.err || solve_exit=$?

    if [[ "${solve_exit}" -eq 0 ]] && ! grep -q ENDFILE params.txt; then
        echo "ERROR: params.txt missing ENDFILE marker for ${run_name}" >&2
        solve_exit=1
    fi
    if [[ "${solve_exit}" -eq 0 ]]; then
        echo "Wrote ${run_name}/params.txt ($(wc -l < params.txt) lines)"
        completed=$((completed + 1))
    else
        echo "FAILED solve ${run_name} (exit ${solve_exit})" >&2
    fi
    "${MULTIELEMENT_ROOT}/scripts/log_compute_event.sh" chimes_solve "${run_dir}" "${solve_exit}" || true

    if [[ "${solve_exit}" -ne 0 ]]; then
        bundle_exit="${solve_exit}"
        break
    fi
done

echo "========== pct001 train bundle done ($(date -Is)) =========="
echo "completed=${completed} skipped=${skipped} exit=${bundle_exit}"
exit "${bundle_exit}"
