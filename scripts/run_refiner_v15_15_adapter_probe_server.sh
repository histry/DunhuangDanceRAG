#!/usr/bin/env bash
# Server-only teacher expansion and 50-step V15.15 Adapter probe.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
MAX_CASES_PER_GROUP="${MAX_CASES_PER_GROUP:-4}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

BASE_ORACLE_TAG=$(cat outputs/LATEST_REFINER_V15_14H_ORACLE_TAG)
BASE_ORACLE_DIR="$OUT_ROOT/checkpoints/$BASE_ORACLE_TAG/full_tangent_oracle"
BASE_ORACLE_REPORT="$BASE_ORACLE_DIR/case_local_full_tangent_oracle.report.json"
test -s "$BASE_ORACLE_REPORT"
SOURCE_DIAGNOSTIC=$("$PY" - "$BASE_ORACLE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(report["source_diagnostic"])
PY
)
test -s "$SOURCE_DIAGNOSTIC/diagnostic_report.json"
test -s "$SOURCE_DIAGNOSTIC/diagnostic_state.pt"
test -s "$SOURCE_DIAGNOSTIC/fit_bank.pt"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15_observable_adapter_probe_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
EXPANDED_ORACLE_DIR="$ROOT/expanded_full_tangent_oracle"
TEACHER_DIR="$ROOT/teacher_bank"
PROBE_DIR="$ROOT/adapter_probe"
LOG="logs/refiner_v15_15_observable_adapter_probe_${STAMP}.log"
STATUS="outputs/refiner_v15_15_observable_adapter_probe_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15_observable_adapter_probe_${STAMP}.scientific_status.txt"

mkdir -p "$EXPANDED_ORACLE_DIR" "$TEACHER_DIR" "$PROBE_DIR" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15_ADAPTER_PROBE_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15_ADAPTER_PROBE_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15_ADAPTER_PROBE_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" \
  > outputs/LATEST_REFINER_V15_15_ADAPTER_PROBE_SCIENTIFIC_STATUS
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
export MOTION_PRODUCT_REFINER_OBSERVABLE_ADAPTER=0
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1

set +e
"$PY" -u -m training.refiner_case_local_full_tangent_oracle \
  --config configs/motion_model.json \
  --source-diagnostic-dir "$SOURCE_DIAGNOSTIC" \
  --output-dir "$EXPANDED_ORACLE_DIR" \
  --all-cross-cases \
  --max-cases-per-group "$MAX_CASES_PER_GROUP" \
  --target-rms 1e-4 \
  --iterations 60 \
  --learning-rate 2e-2 \
  --initial-penalty 10
EXPANSION_STATUS=$?
set -e
if [[ "$EXPANSION_STATUS" -ne 0 && "$EXPANSION_STATUS" -ne 2 ]]; then
  echo "[FATAL] expanded oracle failed with status $EXPANSION_STATUS"
  exit "$EXPANSION_STATUS"
fi

EXPANDED_REPORT="$EXPANDED_ORACLE_DIR/case_local_full_tangent_oracle.report.json"
test -s "$EXPANDED_REPORT"
"$PY" -u -m training.refiner_v15_15_teacher_bank \
  --config configs/motion_model.json \
  --oracle-report "$BASE_ORACLE_REPORT" \
  --oracle-report "$EXPANDED_REPORT" \
  --output-dir "$TEACHER_DIR"

TEACHER_BANK="$TEACHER_DIR/observable_adapter_teacher_bank.pt"
test -s "$TEACHER_BANK"
export MOTION_PRODUCT_REFINER_OBSERVABLE_ADAPTER=1

set +e
"$PY" -u -m training.refiner_observable_adapter_probe \
  --config configs/motion_model.json \
  --teacher-bank "$TEACHER_BANK" \
  --output-dir "$PROBE_DIR" \
  --steps 50 \
  --eval-every 10 \
  --learning-rate 1e-3 \
  --gradient-clip 1 \
  --target-rms 1e-4
PROBE_STATUS=$?
set -e
printf '%s\n' "$PROBE_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$PROBE_STATUS" -ne 0 && "$PROBE_STATUS" -ne 2 ]]; then
  echo "[FATAL] Adapter probe failed with status $PROBE_STATUS"
  exit "$PROBE_STATUS"
fi

REPORT="$PROBE_DIR/observable_adapter_probe.report.json"
test -s "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(json.dumps({
    "schema": r.get("schema"),
    "steps": r.get("steps"),
    "ready_for_expanded_adapter_training": r.get(
        "ready_for_expanded_adapter_training"
    ),
    "effective_projected_candidate_count_by_group": r.get(
        "effective_projected_candidate_count_by_group"
    ),
    "role_label_consumed_at_inference": r.get(
        "role_label_consumed_at_inference"
    ),
    "fixed_guard_thresholds_changed": r.get(
        "fixed_guard_thresholds_changed"
    ),
    "report": str(Path(sys.argv[1]).resolve()),
}, ensure_ascii=False, indent=2))
PY

echo "V15.15 Adapter probe completed; scientific_status=$PROBE_STATUS"
echo "No formal training, promotion, replay, or generation was launched."
