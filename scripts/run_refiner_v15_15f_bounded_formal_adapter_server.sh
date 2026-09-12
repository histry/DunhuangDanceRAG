#!/usr/bin/env bash
# Server-only V15.15f bounded formal Adapter training trial.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
AUDIT_EVERY="${AUDIT_EVERY:-20}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

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
TAG="refiner_v15_15f_bounded_formal_adapter_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
TRAIN_DIR="$ROOT/bounded_training"
LOG="logs/refiner_v15_15f_bounded_formal_adapter_${STAMP}.log"
STATUS="outputs/refiner_v15_15f_bounded_formal_adapter_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15f_bounded_formal_adapter_${STAMP}.scientific_status.txt"

mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15F_BOUNDED_TRAINING_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15F_BOUNDED_TRAINING_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15F_BOUNDED_TRAINING_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15F_BOUNDED_TRAINING_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "train_bank=$TRAIN_BANK"
echo "validation_bank=$VALIDATION_BANK"
echo "source_state=$SOURCE_STATE"
echo "total_steps=$TOTAL_STEPS"
echo "audit_every=$AUDIT_EVERY"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

set +e
"$PY" -u -m training.refiner_v15_15f_bounded_formal_adapter \
  --config configs/motion_model.json \
  --train-teacher-bank "$TRAIN_BANK" \
  --validation-teacher-bank "$VALIDATION_BANK" \
  --adapter-state "$SOURCE_STATE" \
  --output-dir "$TRAIN_DIR" \
  --total-steps "$TOTAL_STEPS" \
  --audit-every "$AUDIT_EVERY" \
  --learning-rate 5e-4 \
  --gradient-clip 1
RUN_STATUS=$?
set -e
printf '%s\n' "$RUN_STATUS" > "$SCIENTIFIC_STATUS"
REPORT="$TRAIN_DIR/bounded_formal_adapter.report.json"
test -s "$REPORT"
printf '%s\n' "$REPORT" > outputs/LATEST_REFINER_V15_15F_BOUNDED_TRAINING_REPORT
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(json.dumps({
    "schema": r.get("schema"),
    "total_steps_requested": r.get("total_steps_requested"),
    "total_steps_completed": r.get("total_steps_completed"),
    "bounded_training_passed": r.get("bounded_training_passed"),
    "fail_closed_reason": r.get("fail_closed_reason"),
    "fatal_execution_status": r.get("fatal_execution_status"),
    "split_contract": r.get("split_contract"),
    "formal_checkpoint": r.get("formal_checkpoint"),
}, ensure_ascii=False, indent=2))
PY

if [[ "$RUN_STATUS" -ne 0 && "$RUN_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15f execution failed with status $RUN_STATUS"
  exit "$RUN_STATUS"
fi

if [[ "$RUN_STATUS" -eq 0 ]]; then
  CHECKPOINT="$TRAIN_DIR/bounded_formal_adapter_state.pt"
  test -s "$CHECKPOINT"
  printf '%s\n' "$CHECKPOINT" > outputs/LATEST_REFINER_V15_15F_BOUNDED_TRAINING_STATE
fi
echo "V15.15f bounded formal Adapter trial completed; scientific_status=$RUN_STATUS"
echo "No promotion, replay, pseudo-teacher generation, or video generation was launched."
