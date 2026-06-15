#!/usr/bin/env bash
# Submit NVT+RDF LAMMPS jobs for all trained 1% and 10% pruned models.
#
#   cd models/statepoint_eval
#   bash submit_pct001_pct010.sh
#   bash submit_pct001_pct010.sh --batch-size 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
CHIMES_PARAMS="${SCRIPT_DIR}/chimes_params"
PRUNED_RUNS="${SCRIPT_DIR}/../pruned_models/runs"

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

mapfile -t MODELS < <(
    for pct in pct001 pct010; do
        for d in "${PRUNED_RUNS}"/a*_"${pct}"_*/; do
            [[ -d "${d}" ]] || continue
            model="$(basename "${d}")"
            if grep -q ENDFILE "${d}/params.txt" 2>/dev/null; then
                echo "${model}"
            fi
        done
    done | sort -u
)

if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "No complete pct001/pct010 models found under pruned_models/runs/." >&2
    exit 1
fi

echo "Syncing params and preparing ${#MODELS[@]} model(s) on ${PARTITION}..."
python3 "${SCRIPT_DIR}/prepare_runs.py" \
    --sync-params \
    --models "${MODELS[@]}" \
    --partition "${PARTITION}"

submitted=0
skipped=0
failed=0

for model in "${MODELS[@]}"; do
    [[ -d "${RUNS_DIR}/${model}" ]] || continue
    for run_dir in "${RUNS_DIR}/${model}"/*/; do
        [[ -d "${run_dir}" ]] || continue
        name="${model}/$(basename "${run_dir}")"

        if [[ -s "${run_dir}/rdf.dat" ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        if [[ "${submitted}" -ge "${BATCH_SIZE}" ]]; then
            echo "Batch limit (${BATCH_SIZE}) reached; re-run to submit more."
            echo "Submitted ${submitted}, skipped ${skipped}, failed ${failed}"
            exit 0
        fi

        cd "${run_dir}"
        if ! out="$(sbatch run_lammps.cmd 2>&1)"; then
            echo "FAILED ${name}: ${out}" >&2
            failed=$((failed + 1))
            if echo "${out}" | grep -q QOSMaxSubmitJobPerUserLimit; then
                echo "Hit QOS submit limit; re-run later to continue."
                exit 0
            fi
            continue
        fi
        job_id="$(echo "${out}" | awk '/Submitted batch job/ {print $NF}')"
        echo "Submitted ${name}: job=${job_id}"
        submitted=$((submitted + 1))
    done
done

echo "Submitted ${submitted} job(s), skipped ${skipped} (rdf.dat present), failed ${failed}."

if [[ -n "${RESEARCH_NOTES_ROOT:-}" ]]; then
    python3 "${SCRIPT_DIR}/../../scripts/sync_proj_c_log.py" scan || true
fi
