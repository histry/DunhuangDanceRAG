#!/usr/bin/env bash
# Resumable low-compute paper-2 mechanism/closure evaluation on the 4090.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT=<full main SHA>}"
: "${PAPER2_PHASE:?mechanism|development|formal|sealed}"
: "${PAPER2_CASE_MANIFEST:?path to immutable paper2 case manifest}"
: "${PAPER2_PROTOCOL:?path to protocol frozen to EXPECTED_COMMIT}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -x "$PY"
test -s "$PAPER2_PROTOCOL"
test -s "$PAPER2_CASE_MANIFEST"

ADAPTER_STATE="${ADAPTER_STATE:-$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)}"
TEACHER_TAG="${TEACHER_TAG:-$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)}"
TRAIN_BANK="${TRAIN_BANK:-$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt}"
if test "$PAPER2_PHASE" = mechanism; then
  EVALUATION_BANK="${EVALUATION_BANK:-$TRAIN_BANK}"
else
  : "${EVALUATION_BANK:?set the frozen development or sealed bank}"
  : "${FROZEN_SEVERITY_ENVELOPE:?set frozen train conformal envelope}"
  : "${FROZEN_REPAIR_CONTRACT:?set frozen train repair contract}"
  test -s "$FROZEN_SEVERITY_ENVELOPE"
  test -s "$FROZEN_REPAIR_CONTRACT"
fi
for path in "$ADAPTER_STATE" "$TRAIN_BANK" "$EVALUATION_BANK"; do
  test -s "$path"
done

STAMP="${PAPER2_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${PAPER2_RUN_ROOT:-$OUT_ROOT/checkpoints/paper2_${PAPER2_PHASE}_${STAMP}}"
LOG="logs/paper2_${PAPER2_PHASE}_${STAMP}.log"
mkdir -p "$RUN_ROOT" logs outputs
printf '%s\n' "$RUN_ROOT" > "outputs/LATEST_PAPER2_${PAPER2_PHASE^^}_ROOT"
printf '%s\n' "$LOG" > "outputs/LATEST_PAPER2_${PAPER2_PHASE^^}_LOG"
exec > >(tee -a "$LOG") 2>&1
echo "Paper-2 protocol: $PAPER2_PROTOCOL"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMAND=(
  "$PY" -u -m experiments.paper2.run_paper2_eval
  --protocol "$PAPER2_PROTOCOL"
  --case-manifest "$PAPER2_CASE_MANIFEST"
  --phase "$PAPER2_PHASE"
  --python "$PY"
  --train-bank "$TRAIN_BANK"
  --evaluation-bank "$EVALUATION_BANK"
  --adapter-state "$ADAPTER_STATE"
  --output-root "$RUN_ROOT"
)
if test "$PAPER2_PHASE" = sealed; then
  PAPER2_SEALED_CONSUMPTION_ROOT="${PAPER2_SEALED_CONSUMPTION_ROOT:-$ROOT_DIR/outputs/paper2_sealed_receipts}"
  COMMAND+=(
    --sealed-consumption-root "$PAPER2_SEALED_CONSUMPTION_ROOT"
  )
fi
if test "$PAPER2_PHASE" != mechanism; then
  COMMAND+=(
    --frozen-severity-envelope "$FROZEN_SEVERITY_ENVELOPE"
    --frozen-repair-contract "$FROZEN_REPAIR_CONTRACT"
  )
fi
"${COMMAND[@]}"
