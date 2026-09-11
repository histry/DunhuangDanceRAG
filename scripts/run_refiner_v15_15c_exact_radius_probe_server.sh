#!/usr/bin/env bash
# Server-only 200-step exact-radius constraint-aware Adapter probe.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
STEPS="${STEPS:-200}"
EVAL_EVERY="${EVAL_EVERY:-20}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

TEACHER_TAG="${TEACHER_TAG:-$(cat outputs/LATEST_REFINER_V15_15C_TEACHER_EXPANSION_TAG)}"
TEACHER_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank/observable_adapter_teacher_bank.pt"
SOURCE_ADAPTER_TAG="${SOURCE_ADAPTER_TAG:-$(cat outputs/LATEST_REFINER_V15_15_ADAPTER_PROBE_TAG)}"
ADAPTER_STATE="$OUT_ROOT/checkpoints/$SOURCE_ADAPTER_TAG/adapter_probe/observable_adapter_probe_state.pt"
test -s "$TEACHER_BANK"
test -s "$ADAPTER_STATE"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15c_exact_radius_adapter_probe_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
PROBE_DIR="$ROOT/adapter_probe"
LOG="logs/refiner_v15_15c_exact_radius_adapter_probe_${STAMP}.log"
STATUS="outputs/refiner_v15_15c_exact_radius_adapter_probe_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15c_exact_radius_adapter_probe_${STAMP}.scientific_status.txt"

mkdir -p "$PROBE_DIR" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15C_EXACT_RADIUS_PROBE_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15C_EXACT_RADIUS_PROBE_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15C_EXACT_RADIUS_PROBE_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15C_EXACT_RADIUS_PROBE_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "teacher_tag=$TEACHER_TAG"
echo "source_adapter_tag=$SOURCE_ADAPTER_TAG"
echo "steps=$STEPS"
echo "eval_every=$EVAL_EVERY"
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
export MOTION_PRODUCT_REFINER_OBSERVABLE_ADAPTER=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1

set +e
"$PY" -u -m training.refiner_observable_adapter_probe \
  --config configs/motion_model.json \
  --teacher-bank "$TEACHER_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --exact-radius-training \
  --output-dir "$PROBE_DIR" \
  --steps "$STEPS" \
  --eval-every "$EVAL_EVERY" \
  --learning-rate 1e-3 \
  --gradient-clip 1 \
  --target-rms 1e-4 \
  --normalization-eps 1e-8 \
  --nonregression-weight 25 \
  --case20-temporal-weight 25 \
  --violating-direction-weight 0.1
PROBE_STATUS=$?
set -e
printf '%s\n' "$PROBE_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$PROBE_STATUS" -ne 0 && "$PROBE_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15c Adapter probe failed with status $PROBE_STATUS"
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
    "exact_radius_training": r.get("exact_radius_training"),
    "required_case_status": r.get("required_case_status"),
    "formal_readiness_criteria": r.get("formal_readiness_criteria"),
    "effective_projected_candidate_count_by_group": r.get(
        "effective_projected_candidate_count_by_group"
    ),
    "ready_for_formal_adapter_training": r.get(
        "ready_for_formal_adapter_training"
    ),
    "scope_safe": r.get("scope_safe"),
    "numeric_audit_complete": r.get("numeric_audit_complete"),
    "fixed_guard_thresholds_changed": r.get(
        "fixed_guard_thresholds_changed"
    ),
    "report": str(Path(sys.argv[1]).resolve()),
}, ensure_ascii=False, indent=2))
PY

echo "V15.15c exact-radius Adapter probe completed; scientific_status=$PROBE_STATUS"
echo "No formal training, promotion, replay, or generation was launched."
