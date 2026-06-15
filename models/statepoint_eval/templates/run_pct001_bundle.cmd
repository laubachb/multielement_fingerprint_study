#!/bin/bash
#SBATCH -J pct001_md_bundle
#SBATCH -N {{NNODES}}
#SBATCH --ntasks-per-node {{NCORES}}
#SBATCH -t {{WALLTIME}}
#SBATCH -p {{PARTITION}}
#SBATCH -A TG-CHM250118
#SBATCH -o {{STATEPOINT_EVAL_ROOT}}/pct001_bundle_%j.out
#SBATCH -e {{STATEPOINT_EVAL_ROOT}}/pct001_bundle_%j.err

set -uo pipefail

export MULTIELEMENT_ROOT="{{MULTIELEMENT_ROOT}}"
source "${MULTIELEMENT_ROOT}/setup/env.sh"
module load intel/24.0 impi/21.11 python

STATEPOINT_EVAL_ROOT="{{STATEPOINT_EVAL_ROOT}}"
NTASKS={{NTASKS}}

if [[ ! -x "${LAMMPS_EXE}" ]]; then
    echo "ERROR: LAMMPS_EXE not found or not executable: ${LAMMPS_EXE}" >&2
    exit 127
fi

RUNS=(
{{RUN_LIST}}
)

bundle_exit=0
completed=0
skipped=0
failed=0

for run_dir in "${RUNS[@]}"; do
    if [[ ! -d "${run_dir}" ]]; then
        echo "ERROR: missing run directory ${run_dir}" >&2
        bundle_exit=1
        break
    fi

    label="${run_dir#${STATEPOINT_EVAL_ROOT}/runs/}"
    label="${label//\//_}"
    cd "${run_dir}"
    echo "========== ${label} ($(date -Is)) =========="

    if [[ -f rdf.dat ]] && [[ -s rdf.dat ]]; then
        echo "SKIP ${label}: rdf.dat already complete"
        skipped=$((skipped + 1))
        continue
    fi

    rm -f output.txt rdf.dat traj.lammpstrj
    run_exit=0
    ibrun -n "${NTASKS}" "${LAMMPS_EXE}" -i in.lammps | tee output.txt || run_exit=$?
    "${MULTIELEMENT_ROOT}/scripts/log_compute_event.sh" statepoint_md "${run_dir}" "${run_exit}" || true

    if [[ "${run_exit}" -eq 0 ]] && [[ -s rdf.dat ]]; then
        echo "OK ${label}: rdf.dat ($(wc -c < rdf.dat) bytes)"
        completed=$((completed + 1))
    else
        echo "FAILED ${label} (exit ${run_exit})" >&2
        failed=$((failed + 1))
        bundle_exit="${run_exit}"
        if [[ ! -s rdf.dat ]]; then
            echo "ERROR: rdf.dat missing or empty for ${label}" >&2
        fi
        break
    fi
done

echo "========== pct001 MD bundle done ($(date -Is)) =========="
echo "completed=${completed} skipped=${skipped} failed=${failed} exit=${bundle_exit}"
exit "${bundle_exit}"
