#!/usr/bin/env bash
# Submit only the runs needed to reach 8 completed models for
# alpha={0.00,0.25,1.00} x pct={001,010}.
#
#   cd models/pruned_models
#   bash submit_to8.sh [--dry-run]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SCRIPT_DIR}/runs"
TARGET=8
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

submitted=0
skipped=0

for alpha in a000 a025 a100; do
  for pct in pct001 pct010; do
    # Count already-done models for this group
    done_count=0
    for d in "${RUNS_DIR}/${alpha}_${pct}_rep"*/; do
      [[ -f "${d}/params.txt" ]] && grep -q ENDFILE "${d}/params.txt" 2>/dev/null && done_count=$((done_count+1))
    done
    need=$((TARGET - done_count))
    echo "=== ${alpha}_${pct}: ${done_count} done, need ${need} more ==="
    [[ $need -le 0 ]] && echo "  Already at target." && continue

    queued=0
    for d in "${RUNS_DIR}/${alpha}_${pct}_rep"*/; do
      [[ $queued -ge $need ]] && break
      name="$(basename "${d}")"
      # Skip if already complete
      if [[ -f "${d}/params.txt" ]] && grep -q ENDFILE "${d}/params.txt" 2>/dev/null; then
        continue
      fi
      # Skip if already submitted
      if [[ -f "${d}/submitted.json" ]]; then
        echo "  SKIP ${name} (submitted.json exists)"
        skipped=$((skipped+1))
        continue
      fi
      if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry-run] would submit ${name}"
        queued=$((queued+1))
        continue
      fi
      cd "${d}"
      gen_id="$(sbatch run_gen_Amat.cmd | awk '/Submitted batch job/ {print $NF}')"
      solve_id="$(sbatch --dependency=afterok:"${gen_id}" run_solve_Amat.cmd | awk '/Submitted batch job/ {print $NF}')"
      echo "  Submitted ${name}: gen=${gen_id} solve=${solve_id}"
      printf '{"gen_job": %s, "solve_job": %s}\n' "${gen_id}" "${solve_id}" > submitted.json
      queued=$((queued+1))
      submitted=$((submitted+1))
    done
  done
done

echo ""
echo "Done. Submitted ${submitted} gen→solve chains. Skipped ${skipped} (already queued)."
