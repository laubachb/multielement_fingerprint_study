#!/usr/bin/env bash
# Submit NVT+RDF jobs for every model with a complete params.txt under chimes_params/.
#
#   cd models/statepoint_eval
#   python prepare_runs.py --sync-params
#   bash submit_all.sh --batch-size 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
BATCH_SIZE=9999
PARTITION="skx"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

echo "Refreshing all run directories (partition=${PARTITION})..."
python3 "${SCRIPT_DIR}/prepare_runs.py" --sync-params --partition "${PARTITION}"

submitted=0
for model_dir in "${RUNS_DIR}"/*/; do
    [[ -d "${model_dir}" ]] || continue
    model="$(basename "${model_dir}")"

    for run_dir in "${model_dir}"/*/; do
        [[ -d "${run_dir}" ]] || continue
        name="${model}/$(basename "${run_dir}")"

        if [[ -f "${run_dir}/rdf.dat" ]]; then
            echo "Skipping ${name} (rdf.dat exists)"
            continue
        fi

        if [[ -f "${run_dir}/log.lammps" ]] \
            && grep -qE "^(run |Loop time)" "${run_dir}/log.lammps" 2>/dev/null; then
            echo "Skipping ${name} (MD already attempted)"
            continue
        fi

        if [[ "${submitted}" -ge "${BATCH_SIZE}" ]]; then
            echo "Batch limit (${BATCH_SIZE}) reached; re-run to submit more."
            exit 0
        fi

        cd "${run_dir}"
        if ! out="$(sbatch run_lammps.cmd 2>&1)"; then
            echo "FAILED ${name}: ${out}" >&2
            continue
        fi
        job_id="$(echo "${out}" | awk '/Submitted batch job/ {print $NF}')"
        echo "Submitted ${name}: job=${job_id}"
        submitted=$((submitted + 1))
    done
done

echo "Submitted ${submitted} job(s)."
