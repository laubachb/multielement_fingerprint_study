#!/usr/bin/env bash
# Submit one skx-dev job that sequentially trains pending 1% ChIMES models (gen→solve).
#
#   cd models/pruned_models
#   bash submit_debug_pct001_bundle.sh
#   bash submit_debug_pct001_bundle.sh --replicate 3
#   bash submit_debug_pct001_bundle.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
TEMPLATE="${SCRIPT_DIR}/templates/run_debug_pct001_bundle.cmd"
BUNDLE_CMD="${SCRIPT_DIR}/run_debug_pct001_bundle.cmd"
MULTIELEMENT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PARTITION="skx-dev"
WALLTIME="02:00:00"
NNODES=1
NCORES=48
NTASKS=48
REPLICATE=2
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --replicate) REPLICATE="$2"; shift ;;
        --partition) PARTITION="$2"; shift ;;
        --walltime) WALLTIME="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

job_active() {
    local jid="$1"
    [[ -n "${jid}" && "${jid}" != "None" && "${jid}" != "null" ]] \
        && squeue -j "${jid}" -h 2>/dev/null | grep -q .
}

params_complete() {
    [[ -f "$1/params.txt" ]] && grep -q ENDFILE "$1/params.txt" 2>/dev/null
}

rep_tag="$(printf 'rep%02d' "${REPLICATE}")"
echo "Preparing pct001 ${rep_tag} on ${PARTITION}..."
python3 "${SCRIPT_DIR}/prepare_runs.py" \
    --retention-fractions 0.01 \
    --replicate "${REPLICATE}" \
    --debug-queue

pending=()
for run_dir in "${RUNS_DIR}"/a*_pct001_"${rep_tag}"/; do
    [[ -d "${run_dir}" ]] || continue
    name="$(basename "${run_dir}")"
    if params_complete "${run_dir}"; then
        echo "  ${name}: complete"
    else
        echo "  ${name}: pending"
        pending+=("${name}")
    fi
done

if [[ "${#pending[@]}" -eq 0 ]]; then
    echo "All pct001 ${rep_tag} models complete; nothing to submit."
    exit 0
fi

if [[ -f "${SCRIPT_DIR}/pct001_train_bundle_submitted.json" ]]; then
    bundle_id="$(python3 -c "import json; print(json.load(open('${SCRIPT_DIR}/pct001_train_bundle_submitted.json')).get('bundle_job',''))" 2>/dev/null || true)"
    if job_active "${bundle_id}"; then
        echo "Bundle job ${bundle_id} still queued/running; not resubmitting."
        exit 0
    fi
fi

run_list=""
for name in "${pending[@]}"; do
    run_list+="    ${name}"$'\n'
done
run_list="${run_list%$'\n'}"

mapping=(
    "MULTIELEMENT_ROOT=${MULTIELEMENT_ROOT}"
    "PRUNED_MODELS_ROOT=${SCRIPT_DIR}"
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
echo "Bundle: ${#pending[@]} model(s) on ${PARTITION} (${WALLTIME}):"
printf '  %s\n' "${pending[@]}"

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

out = Path("${SCRIPT_DIR}") / "pct001_train_bundle_submitted.json"
out.write_text(json.dumps({
    "bundle_job": "${bundle_id}",
    "partition": "${PARTITION}",
    "walltime": "${WALLTIME}",
    "replicate": ${REPLICATE},
    "pending": json.loads("""${PENDING_JSON}"""),
    "debug": True,
}, indent=2) + "\n", encoding="utf-8")
PY

echo "Submitted pct001 train bundle: job=${bundle_id}"
echo "Monitor: squeue -j ${bundle_id}"
echo "Log:     ${SCRIPT_DIR}/pct001_bundle_${bundle_id}.out"

if [[ -n "${RESEARCH_NOTES_ROOT:-}" ]]; then
    python3 "${SCRIPT_DIR}/../../scripts/sync_proj_c_log.py" scan || true
fi
