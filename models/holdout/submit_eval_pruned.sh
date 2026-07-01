#!/usr/bin/env bash
# Submit batched pruned-model hold-out jobs on skx-dev.
# Task order favors diverse (alpha, retention) combos before extra replicates.
#
#   cd models/holdout
#   bash submit_eval_pruned.sh           # submit up to QOS limit (default 2)
#   bash submit_eval_pruned.sh --limit 4 # submit first 4 pending diverse models

set -euo pipefail
cd "$(dirname "$0")"

LIMIT=2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [[ ! -f md_holdout_manifest.json ]]; then
    echo "Missing md_holdout_manifest.json" >&2
    exit 1
fi

read -r PENDING_IDS PENDING_NAMES <<EOF
$(python3 - <<'PY'
import json
from pathlib import Path
from evaluate_md_forces import (
    discover_models,
    diverse_model_order,
    load_manifest,
    model_force_cache_complete,
    resolve_full_params,
    write_batched_task_list,
)

manifest = load_manifest(Path("md_holdout_manifest.json"))
models = diverse_model_order(discover_models(None))
tasks = write_batched_task_list(models, resolve_full_params(), diverse_first=False)
forces = Path("forces")

pending = []
for task in tasks:
    if task["array_task_id"] == 0:
        continue
    key = task["model_key"]
    if model_force_cache_complete(key, manifest, forces):
        continue
    pending.append(task)

if not pending:
    print("NONE", "NONE")
else:
    ids = ",".join(str(t["array_task_id"]) for t in pending)
    names = ",".join(t["model"] for t in pending)
    print(ids, names)
PY
)
EOF

if [[ "${PENDING_IDS}" == "NONE" ]]; then
    echo "All pruned models already cached."
    exit 0
fi

IFS=',' read -r -a ALL_IDS <<< "${PENDING_IDS}"
IFS=',' read -r -a ALL_NAMES <<< "${PENDING_NAMES}"

# Drop array tasks already running or queued for this job name.
ACTIVE_TASKS=$(
    squeue -u "${USER}" -h -n md_holdout_batched -o "%i" 2>/dev/null \
        | awk -F_ '{print $NF}' \
        | sort -u
)
FILTERED_IDS=()
FILTERED_NAMES=()
for i in "${!ALL_IDS[@]}"; do
    id="${ALL_IDS[$i]}"
    skip=0
    for active in ${ACTIVE_TASKS}; do
        [[ "${id}" == "${active}" ]] && skip=1 && break
    done
    [[ "${skip}" -eq 1 ]] && continue
    FILTERED_IDS+=("${id}")
    FILTERED_NAMES+=("${ALL_NAMES[$i]}")
done
ALL_IDS=("${FILTERED_IDS[@]}")
ALL_NAMES=("${FILTERED_NAMES[@]}")

if [[ "${#ALL_IDS[@]}" -eq 0 ]]; then
    echo "No pending pruned models outside the queue."
    exit 0
fi

# skx-dev (qdevelopment): MaxJobsPU=2, MaxSubmitPU=4 — stay under submit cap.
RUNNING=$(squeue -u "${USER}" -h -n md_holdout_batched 2>/dev/null | wc -l)
SUBMIT_CAP=$((4 - RUNNING))
if [[ "${SUBMIT_CAP}" -lt 1 ]]; then
    echo "At skx-dev submit cap (running=${RUNNING}). Wait and re-run."
    exit 0
fi
if [[ "${LIMIT}" -gt "${SUBMIT_CAP}" ]]; then
    LIMIT="${SUBMIT_CAP}"
fi

SUBMIT_IDS=()
SUBMIT_NAMES=()
for i in "${!ALL_IDS[@]}"; do
    [[ "${#SUBMIT_IDS[@]}" -ge "${LIMIT}" ]] && break
    SUBMIT_IDS+=("${ALL_IDS[$i]}")
    SUBMIT_NAMES+=("${ALL_NAMES[$i]}")
done

ARRAY_SPEC=$(IFS=,; echo "${SUBMIT_IDS[*]}")
echo "Pending pruned models (not in queue): ${#ALL_IDS[@]}  submitting: ${#SUBMIT_IDS[@]}"
echo "Next models (diverse-first): ${SUBMIT_NAMES[*]}"

if [[ "${#SUBMIT_IDS[@]}" -eq 0 ]]; then
    echo "Nothing new to submit."
    exit 0
fi

JOB_ID=$(sbatch --parsable --array="${ARRAY_SPEC}" run_eval_batched.cmd)
echo "Submitted job ${JOB_ID} array tasks ${ARRAY_SPEC}"
echo "Monitor: squeue -u ${USER} -n md_holdout_batched"
echo "Re-run this script to queue more after slots free."
