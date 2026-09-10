#!/usr/bin/env bash
# Server-only V15.14h feasibility oracle; never training.
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
TAG="refiner_v15_14h_full_tangent_oracle_${STAMP}"
ORACLE_DIR="$OUT_ROOT/checkpoints/$TAG/full_tangent_oracle"
LOG="logs/refiner_v15_14h_full_tangent_oracle_${STAMP}.log"
STATUS="outputs/refiner_v15_14h_full_tangent_oracle_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_14h_full_tangent_oracle_${STAMP}.scientific_status.txt"

mkdir -p "$ORACLE_DIR" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_14H_ORACLE_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_14H_ORACLE_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_14H_ORACLE_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" \
  > outputs/LATEST_REFINER_V15_14H_ORACLE_SCIENTIFIC_STATUS
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
"$PY" -u -m training.refiner_case_local_full_tangent_oracle \
  --config configs/motion_model.json \
  --source-diagnostic-dir "$SOURCE_DIAGNOSTIC" \
  --output-dir "$ORACLE_DIR" \
  --target-rms 1e-4 \
  --iterations 60 \
  --learning-rate 2e-2 \
  --initial-penalty 10
ORACLE_STATUS=$?
set -e
printf '%s\n' "$ORACLE_STATUS" > "$SCIENTIFIC_STATUS"

if [[ "$ORACLE_STATUS" -ne 0 && "$ORACLE_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.14h oracle failed with status $ORACLE_STATUS"
  exit "$ORACLE_STATUS"
fi

REPORT="$ORACLE_DIR/case_local_full_tangent_oracle.report.json"
test -s "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(json.dumps({
    "schema": report.get("schema"),
    "raw_exact_candidate_count_by_group": report.get(
        "raw_exact_candidate_count_by_group"
    ),
    "effective_projected_candidate_count_by_group": report.get(
        "effective_projected_candidate_count_by_group"
    ),
    "cross_short_effective_projected_candidate_count": report.get(
        "cross_short_effective_projected_candidate_count"
    ),
    "cross_long_effective_projected_candidate_count": report.get(
        "cross_long_effective_projected_candidate_count"
    ),
    "primary_cross_raw_feasible": report.get("primary_cross_raw_feasible"),
    "routing_architecture_pivot_supported": report.get(
        "routing_architecture_pivot_supported"
    ),
    "outside_case_group_or_ownership_abs_max": report.get(
        "outside_case_group_or_ownership_abs_max"
    ),
    "feasibility_status": report.get("feasibility_status"),
    "scope_safe": report.get("scope_safe"),
    "numeric_audit_complete": report.get("numeric_audit_complete"),
    "report": str(Path(sys.argv[1]).resolve()),
}, ensure_ascii=False, indent=2))
PY

echo "V15.14h full-tangent oracle completed; scientific_status=$ORACLE_STATUS"
echo "No training, pilot, promotion, full replay, or generation was launched."
