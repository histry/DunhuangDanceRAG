#!/usr/bin/env bash
# Server-only V15.15c Oracle teacher expansion. No Adapter training is run.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
MAX_CASES_PER_GROUP="${MAX_CASES_PER_GROUP:-16}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

BASE_ORACLE_TAG=$(cat outputs/LATEST_REFINER_V15_14H_ORACLE_TAG)
BASE_ORACLE_REPORT="$OUT_ROOT/checkpoints/$BASE_ORACLE_TAG/full_tangent_oracle/case_local_full_tangent_oracle.report.json"
test -s "$BASE_ORACLE_REPORT"
SOURCE_DIAGNOSTIC=$("$PY" - "$BASE_ORACLE_REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(r["source_diagnostic"])
PY
)
test -s "$SOURCE_DIAGNOSTIC/diagnostic_report.json"
test -s "$SOURCE_DIAGNOSTIC/diagnostic_state.pt"
test -s "$SOURCE_DIAGNOSTIC/fit_bank.pt"

SOURCE_ADAPTER_TAG="${SOURCE_ADAPTER_TAG:-$(cat outputs/LATEST_REFINER_V15_15_ADAPTER_PROBE_TAG)}"
SOURCE_EXPANDED_REPORT="$OUT_ROOT/checkpoints/$SOURCE_ADAPTER_TAG/expanded_full_tangent_oracle/case_local_full_tangent_oracle.report.json"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15c_teacher_expansion_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
ORACLE_DIR="$ROOT/expanded_full_tangent_oracle"
TEACHER_DIR="$ROOT/teacher_bank"
LOG="logs/refiner_v15_15c_teacher_expansion_${STAMP}.log"
STATUS="outputs/refiner_v15_15c_teacher_expansion_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15c_teacher_expansion_${STAMP}.scientific_status.txt"

mkdir -p "$ORACLE_DIR" "$TEACHER_DIR" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15C_TEACHER_EXPANSION_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15C_TEACHER_EXPANSION_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15C_TEACHER_EXPANSION_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15C_TEACHER_EXPANSION_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "max_cases_per_group=$MAX_CASES_PER_GROUP"
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
  --output-dir "$ORACLE_DIR" \
  --all-cross-cases \
  --max-cases-per-group "$MAX_CASES_PER_GROUP" \
  --target-rms 1e-4 \
  --iterations 60 \
  --learning-rate 2e-2 \
  --initial-penalty 10
ORACLE_STATUS=$?
set -e
printf '%s\n' "$ORACLE_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$ORACLE_STATUS" -ne 0 && "$ORACLE_STATUS" -ne 2 ]]; then
  echo "[FATAL] expanded Oracle failed with status $ORACLE_STATUS"
  exit "$ORACLE_STATUS"
fi

EXPANDED_REPORT="$ORACLE_DIR/case_local_full_tangent_oracle.report.json"
test -s "$EXPANDED_REPORT"
REPORT_ARGS=(--oracle-report "$BASE_ORACLE_REPORT")
if test -s "$SOURCE_EXPANDED_REPORT"; then
  REPORT_ARGS+=(--oracle-report "$SOURCE_EXPANDED_REPORT")
fi
REPORT_ARGS+=(--oracle-report "$EXPANDED_REPORT")

"$PY" -u -m training.refiner_v15_15_teacher_bank \
  --config configs/motion_model.json \
  "${REPORT_ARGS[@]}" \
  --output-dir "$TEACHER_DIR"

TEACHER_BANK="$TEACHER_DIR/observable_adapter_teacher_bank.pt"
TEACHER_REPORT="$TEACHER_DIR/observable_adapter_teacher_bank.report.json"
test -s "$TEACHER_BANK"
test -s "$TEACHER_REPORT"
cat "$TEACHER_REPORT"

echo "V15.15c teacher expansion completed; oracle_scientific_status=$ORACLE_STATUS"
echo "No Adapter training, formal training, promotion, replay, or generation was launched."
