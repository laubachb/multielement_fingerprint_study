#!/usr/bin/env bash
# Run all 1% ChIMES models × all statepoints sequentially in one SLURM job.
#
# Uses production MD (6000 steps, 1 fs timestep, ~6 ps NVT per run).
# skx-dev is limited to 2 hr and cannot finish production-length MD; default is skx.
#
#   cd models/statepoint_eval
#   bash submit_pct001_bundle.sh
#   bash submit_pct001_bundle.sh --partition skx --walltime 48:00:00  # if skx-dev walltime too short
#   bash submit_pct001_bundle.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
TEMPLATE="${SCRIPT_DIR}/templates/run_pct001_bundle.cmd"
BUNDLE_CMD="${SCRIPT_DIR}/run_pct001_bundle.cmd"
MULTIELEMENT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PARTITION="skx"
WALLTIME="48:00:00"
NNODES=1
NCORES=48
NTASKS=48
DRY_RUN=0
PRODUCTION_MD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --short-md) PRODUCTION_MD=0 ;;
        --partition) PARTITION="$2"; shift ;;
        --walltime) WALLTIME="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ "${PRODUCTION_MD}" -eq 1 ]]; then
    PREPARE_EXTRA=(--partition "${PARTITION}" --walltime "${WALLTIME}")
    echo "1% bundle: production MD (10k/50k) on ${PARTITION} (${WALLTIME})"
else
    PREPARE_EXTRA=(--debug-queue --partition "${PARTITION}" --walltime "${WALLTIME}")
    echo "1% bundle: validated short MD on ${PARTITION} (${WALLTIME})"
fi

job_active() {
    local jid="$1"
    [[ -n "${jid}" && "${jid}" != "None" && "${jid}" != "null" ]] \
        && squeue -j "${jid}" -h 2>/dev/null | grep -q .
}

rdf_complete() {
    [[ -f "$1/rdf.dat" ]] && [[ -s "$1/rdf.dat" ]]
}

mapfile -t PCT001_MODELS < <(
    for d in "${SCRIPT_DIR}"/chimes_params/a*_pct001_*/; do
        [[ -d "${d}" ]] || continue
        model="$(basename "${d}")"
        if grep -q ENDFILE "${d}/params.txt" 2>/dev/null; then
            echo "${model}"
        fi
    done | sort
)

if [[ "${#PCT001_MODELS[@]}" -eq 0 ]]; then
    echo "No 1% models with complete params under chimes_params/." >&2
    exit 1
fi

echo "Syncing params and preparing run dirs for: ${PCT001_MODELS[*]}"
python3 "${SCRIPT_DIR}/prepare_runs.py" \
    --sync-params \
    --models "${PCT001_MODELS[@]}" \
    "${PREPARE_EXTRA[@]}"

pending=()
for model in "${PCT001_MODELS[@]}"; do
    for run_dir in "${RUNS_DIR}/${model}"/*/; do
        [[ -d "${run_dir}" ]] || continue
        if rdf_complete "${run_dir}"; then
            continue
        fi
        pending+=("${run_dir%/}")
    done
done

if [[ "${#pending[@]}" -eq 0 ]]; then
    echo "All 1% statepoint runs already have rdf.dat; nothing to submit."
    exit 0
fi

if [[ -f "${SCRIPT_DIR}/pct001_bundle_submitted.json" ]]; then
    bundle_id="$(python3 -c "import json; print(json.load(open('${SCRIPT_DIR}/pct001_bundle_submitted.json')).get('bundle_job',''))" 2>/dev/null || true)"
    if job_active "${bundle_id}"; then
        echo "Bundle job ${bundle_id} still queued/running; not resubmitting."
        exit 0
    fi
fi

run_list=""
for run_dir in "${pending[@]}"; do
    run_list+="    ${run_dir}"$'\n'
done
run_list="${run_list%$'\n'}"

mapping=(
    "MULTIELEMENT_ROOT=${MULTIELEMENT_ROOT}"
    "STATEPOINT_EVAL_ROOT=${SCRIPT_DIR}"
    "PARTITION=${PARTITION}"
    "WALLTIME=${WALLTIME}"
    "NNODES=${NNODES}"
    "NCORES=${NCORES}"
    "NTASKS=${NTASKS}"
)

text="$(<"${TEMPLATE}")"
text="${text//\{\{RUN_LIST\}\}/${run_list}}"
for pair in "${mapping[@]}"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    text="${text//\{\{${key}\}\}/${value}}"
done
printf '%s\n' "${text}" > "${BUNDLE_CMD}"
chmod +x "${BUNDLE_CMD}"

echo ""
echo "Bundle: ${#pending[@]} run(s) on ${PARTITION} (${WALLTIME}):"
for run_dir in "${pending[@]}"; do
    echo "  ${run_dir#${RUNS_DIR}/}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run: wrote ${BUNDLE_CMD}, not submitting."
    exit 0
fi

cd "${SCRIPT_DIR}"
if ! submit_out="$(sbatch "${BUNDLE_CMD}" 2>&1)"; then
    echo "FAILED submit: ${submit_out}" >&2
    exit 1
fi
bundle_id="$(echo "${submit_out}" | awk '/Submitted batch job/ {print $NF}')"
if [[ -z "${bundle_id}" ]]; then
    echo "FAILED submit: ${submit_out}" >&2
    exit 1
fi

PENDING_JSON="$(printf '%s\n' "${pending[@]}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')"
python3 - <<PY
import json
from pathlib import Path

out = Path("${SCRIPT_DIR}") / "pct001_bundle_submitted.json"
out.write_text(json.dumps({
    "bundle_job": "${bundle_id}",
    "partition": "${PARTITION}",
    "walltime": "${WALLTIME}",
    "production_md": bool(int("${PRODUCTION_MD}")),
    "pending": json.loads("""${PENDING_JSON}"""),
}, indent=2) + "\n", encoding="utf-8")
PY

echo "Submitted pct001 MD bundle: job=${bundle_id}"
echo "Monitor: squeue -j ${bundle_id}"
echo "Log:     ${SCRIPT_DIR}/pct001_bundle_${bundle_id}.out"

if [[ -n "${RESEARCH_NOTES_ROOT:-}" ]]; then
    python3 "${SCRIPT_DIR}/../../scripts/sync_proj_c_log.py" scan || true
fi
