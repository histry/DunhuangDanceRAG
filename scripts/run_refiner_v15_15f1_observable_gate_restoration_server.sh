#!/usr/bin/env bash
# Server-only V15.15f1 observable wake-score restoration probe.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
STEPS="${STEPS:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test "$STEPS" -eq 100
test "$EVAL_EVERY" -eq 10

TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TEACHER_ROOT="$OUT_ROOT/checkpoints/$TEACHER_TAG"
TRAIN_BANK="$TEACHER_ROOT/teacher_bank_train/observable_adapter_teacher_bank.pt"
VALIDATION_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
SOURCE_TAG=$(cat outputs/LATEST_REFINER_V15_15D_GUARD_PROBE_TAG)
SOURCE_STATE="$OUT_ROOT/checkpoints/$SOURCE_TAG/adapter_probe/observable_adapter_probe_state.pt"
test -s "$TRAIN_BANK"
test -s "$VALIDATION_BANK"
test -s "$SOURCE_STATE"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15f1_observable_gate_restoration_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
TRAIN_DIR="$ROOT/train_probe"
VALIDATION_DIR="$ROOT/validation_audit"
LOG="logs/refiner_v15_15f1_observable_gate_restoration_${STAMP}.log"
STATUS="outputs/refiner_v15_15f1_observable_gate_restoration_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15f1_observable_gate_restoration_${STAMP}.scientific_status.txt"

mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ARGS=(
  --config configs/motion_model.json
  --exact-radius-training
  --case-isolated-guard-restoration
  --observable-gate-restoration
  --target-rms 1e-4
  --normalization-eps 1e-8
  --nonregression-weight 25
  --case20-temporal-weight 25
  --guard-safety-fraction 0.25
  --guard-restoration-weight 1
  --guard-direction-floor 0.1
  --guard-direction-decay 1
  --gate-restoration-margin 0.02
  --gate-restoration-weight 1
  --learning-rate 5e-4
  --gradient-clip 1
)

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "train_bank=$TRAIN_BANK"
echo "validation_bank=$VALIDATION_BANK"
echo "source_state=$SOURCE_STATE"
echo "started_at=$(date --iso-8601=seconds)"

set +e
"$PY" -u -m training.refiner_observable_adapter_probe \
  --teacher-bank "$TRAIN_BANK" \
  --adapter-state "$SOURCE_STATE" \
  --output-dir "$TRAIN_DIR" \
  --steps "$STEPS" \
  --eval-every "$EVAL_EVERY" \
  "${COMMON_ARGS[@]}"
TRAIN_STATUS=$?
set -e
if [[ "$TRAIN_STATUS" -ne 0 && "$TRAIN_STATUS" -ne 2 ]]; then
  echo "[FATAL] gate-restoration train probe failed with status $TRAIN_STATUS"
  exit "$TRAIN_STATUS"
fi

TRAIN_REPORT="$TRAIN_DIR/observable_adapter_probe.report.json"
TRAIN_STATE="$TRAIN_DIR/observable_adapter_probe_state.pt"
test -s "$TRAIN_REPORT"
test -s "$TRAIN_STATE"

set +e
"$PY" -u -m training.refiner_observable_adapter_probe \
  --teacher-bank "$VALIDATION_BANK" \
  --adapter-state "$TRAIN_STATE" \
  --preserve-adapter-gate-floor \
  --audit-only \
  --output-dir "$VALIDATION_DIR" \
  --steps 0 \
  --eval-every 1 \
  "${COMMON_ARGS[@]}"
VALIDATION_STATUS=$?
set -e
if [[ "$VALIDATION_STATUS" -ne 0 && "$VALIDATION_STATUS" -ne 2 ]]; then
  echo "[FATAL] gate-restoration validation audit failed with status $VALIDATION_STATUS"
  exit "$VALIDATION_STATUS"
fi

VALIDATION_REPORT="$VALIDATION_DIR/observable_adapter_probe.report.json"
test -s "$VALIDATION_REPORT"
SUMMARY="$ROOT/observable_gate_restoration.report.json"
"$PY" - \
  "$TRAIN_REPORT" \
  "$VALIDATION_REPORT" \
  "$SUMMARY" \
  "$TRAIN_STATUS" \
  "$VALIDATION_STATUS" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
validation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
summary = {
    "schema": "refiner_v15_15f1_observable_gate_restoration_probe_v1",
    "development_only": True,
    "formal_checkpoint": False,
    "promotion_allowed": False,
    "pseudo_teachers_generated": False,
    "inference_role_label_consumed": False,
    "observable_adapter_gate_mode": train.get("observable_adapter_gate_mode"),
    "train_execution_status": int(sys.argv[4]),
    "validation_execution_status": int(sys.argv[5]),
    "train_report": str(Path(sys.argv[1]).resolve()),
    "validation_report": str(Path(sys.argv[2]).resolve()),
    "train_ready": train.get("ready_for_formal_adapter_training"),
    "validation_ready": validation.get("ready_for_formal_adapter_training"),
    "train_projected": train.get("effective_projected_candidate_count_by_group"),
    "validation_projected": validation.get("effective_projected_candidate_count_by_group"),
    "train_single_gate_max": train.get("single_control_gate_max"),
    "validation_single_gate_max": validation.get("single_control_gate_max"),
    "train_scope_safe": train.get("scope_safe"),
    "validation_scope_safe": validation.get("scope_safe"),
    "train_numeric_audit_complete": train.get("numeric_audit_complete"),
    "validation_numeric_audit_complete": validation.get("numeric_audit_complete"),
    "fixed_guard_thresholds_changed": bool(
        train.get("fixed_guard_thresholds_changed")
        or validation.get("fixed_guard_thresholds_changed")
    ),
}
summary["passed"] = bool(
    summary["train_execution_status"] == 0
    and summary["validation_execution_status"] == 0
    and summary["train_ready"]
    and summary["validation_ready"]
    and summary["train_single_gate_max"] == 0.0
    and summary["validation_single_gate_max"] == 0.0
    and summary["train_scope_safe"]
    and summary["validation_scope_safe"]
    and summary["train_numeric_audit_complete"]
    and summary["validation_numeric_audit_complete"]
    and not summary["fixed_guard_thresholds_changed"]
)
Path(sys.argv[3]).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

printf '%s\n' "$SUMMARY" > outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_REPORT
printf '%s\n' "$TRAIN_STATE" > outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE

OVERALL_STATUS=$(
  "$PY" - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(0 if r.get("passed") else 2)
PY
)
printf '%s\n' "$OVERALL_STATUS" > "$SCIENTIFIC_STATUS"
echo "V15.15f1 observable gate restoration completed; scientific_status=$OVERALL_STATUS"
echo "No formal training, pseudo-teacher generation, promotion, replay, or video generation was launched."
