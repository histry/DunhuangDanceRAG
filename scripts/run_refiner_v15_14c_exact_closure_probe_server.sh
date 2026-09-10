#!/usr/bin/env bash
# Server-only V15.14c exact-closure-consistent tangent probe; never training.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

if [[ $# -ge 1 ]]; then
  SOURCE_DIAGNOSTIC="$1"
else
  SOURCE_TAG=$(cat outputs/LATEST_REFINER_V15_13_FILM_PROBE_TAG)
  SOURCE_DIAGNOSTIC="$OUT_ROOT/checkpoints/$SOURCE_TAG/bridge_diagnostic"
fi
test -s "$SOURCE_DIAGNOSTIC/diagnostic_report.json"
test -s "$SOURCE_DIAGNOSTIC/diagnostic_state.pt"
test -s "$SOURCE_DIAGNOSTIC/fit_bank.pt"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_14c_exact_closure_tangent_probe_${STAMP}"
PROBE_DIR="$OUT_ROOT/checkpoints/$TAG/exact_closure_tangent_probe"
LOG="logs/refiner_v15_14c_exact_closure_probe_${STAMP}.log"
STATUS="outputs/refiner_v15_14c_exact_closure_probe_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_14c_exact_closure_probe_${STAMP}.scientific_status.txt"

mkdir -p "$PROBE_DIR" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_14C_EXACT_CLOSURE_PROBE_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_14C_EXACT_CLOSURE_PROBE_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_14C_EXACT_CLOSURE_PROBE_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" \
  > outputs/LATEST_REFINER_V15_14C_EXACT_CLOSURE_PROBE_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "source_diagnostic=$SOURCE_DIAGNOSTIC"
echo "started_at=$(date --iso-8601=seconds)"

ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export PROJECT_ROOT="$ROOT_DIR" EXPERIMENT_PROFILE=research
source configs/experiment.env
export MOTION_DEVICE=cuda
export MOTION_GPU_PREPROCESSING=1
export MOTION_PRODUCT_REFINER_FILM_CONDITIONING=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1

set +e
"$PY" -u -m training.refiner_exact_closure_tangent_probe \
  --config configs/motion_model.json \
  --source-diagnostic-dir "$SOURCE_DIAGNOSTIC" \
  --output-dir "$PROBE_DIR"
PROBE_STATUS=$?
set -e
printf '%s\n' "$PROBE_STATUS" > "$SCIENTIFIC_STATUS"

if [[ "$PROBE_STATUS" -ne 0 && "$PROBE_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.14c probe execution failed with status $PROBE_STATUS"
  exit "$PROBE_STATUS"
fi

REPORT="$PROBE_DIR/exact_closure_tangent.report.json"
test -s "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
rows = report.get("candidates", [])
print(json.dumps({
    "schema": report.get("schema"),
    "protocol": report.get("protocol"),
    "baseline_fixed_exact_guard_passed": report.get(
        "baseline_fixed_exact_guard_passed"
    ),
    "autograd_vs_exact_fd_sign_agreement": report.get(
        "autograd_vs_exact_fd_sign_agreement"
    ),
    "direction_replaced_with_exact_closure_mgda": report.get(
        "direction_replaced_with_exact_closure_mgda"
    ),
    "effective_direction_source": report.get("effective_direction_source"),
    "effective_common_descent_exists": report.get(
        "effective_common_descent_exists"
    ),
    "exact_fd_directional_derivatives": report.get(
        "exact_fd_directional_derivatives"
    ),
    "candidate_count": report.get("candidate_count"),
    "effective_projected_candidate_count": report.get(
        "effective_projected_candidate_count"
    ),
    "projected_direction_exists": report.get("projected_direction_exists"),
    "scope_safe": report.get("scope_safe"),
    "numeric_audit_complete": report.get("numeric_audit_complete"),
    "candidate_summary": [
        {
            "target_output_tangent_rms": row.get(
                "target_output_tangent_rms"
            ),
            "raw_fixed_exact_guard_passed": row.get(
                "raw_fixed_exact_guard_passed"
            ),
            "raw_exact_common_descent": row.get(
                "raw_exact_common_descent"
            ),
            "raw_candidate_admitted_to_projector": row.get(
                "raw_candidate_admitted_to_projector"
            ),
            "projected_scientific_nonregression": row.get(
                "projected_scientific_nonregression"
            ),
            "projection_backtracking_factor": row.get(
                "projection_backtracking_factor"
            ),
            "resolution_limited_under_exact_closure": row.get(
                "resolution_limited_under_exact_closure"
            ),
            "effective_projected_candidate": row.get(
                "effective_projected_candidate"
            ),
            "subobjective_deltas": row.get("subobjective_deltas"),
        }
        for row in rows
    ],
    "report": str(Path(sys.argv[1]).resolve()),
}, ensure_ascii=False, indent=2))
PY

echo "V15.14c exact-closure tangent probe completed; scientific_status=$PROBE_STATUS"
echo "No training, pilot, promotion, full replay, or generation was launched."
