#!/usr/bin/env bash
# Consume the sealed g1f3 held-out exactly once after independent Oracle proof.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
: "${FINAL_HELD_OUT_ORACLE_REPORT:?Set the independent Oracle evidence report}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

ROOT=$(cat outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_ROOT)
SEALED_BANK=$(cat outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_BANK)
MANIFEST=$(cat outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_MANIFEST)
FROZEN_CONTRACT=$(cat outputs/LATEST_REFINER_V15_15G1F3_FROZEN_CONTRACT)
FROZEN_ENVELOPE=$(cat outputs/LATEST_REFINER_V15_15G1F3_FROZEN_CONFORMAL)
FROZEN_REPAIR=$(cat outputs/LATEST_REFINER_V15_15G1F3_FROZEN_REPAIR_CONTRACT)
TRAIN_REPORT=$(cat outputs/LATEST_REFINER_V15_15G1F3_TRAIN_REPORT)
DEV_REPORT=$(cat outputs/LATEST_REFINER_V15_15G1F3_DEVELOPMENT_REPORT)
ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
for path in \
  "$SEALED_BANK" "$MANIFEST" "$FROZEN_CONTRACT" "$FROZEN_ENVELOPE" \
  "$FROZEN_REPAIR" "$TRAIN_REPORT" "$DEV_REPORT" "$ADAPTER_STATE" \
  "$TRAIN_BANK" "$FINAL_HELD_OUT_ORACLE_REPORT"; do test -s "$path"; done

ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
ORACLE_RECEIPT="$ROOT/final_held_out.oracle_receipt.json"
ONE_SHOT_RECEIPT="$ROOT/final_held_out.one_shot_receipt.json"
"$PY" -m training.refiner_v15_15g1f3_contract record-oracle \
  --manifest "$MANIFEST" \
  --oracle-report "$FINAL_HELD_OUT_ORACLE_REPORT" \
  --output "$ORACLE_RECEIPT"
"$PY" -m training.refiner_v15_15g1f3_contract consume-held-out \
  --manifest "$MANIFEST" \
  --oracle-receipt "$ORACLE_RECEIPT" \
  --frozen-contract "$FROZEN_CONTRACT" \
  --output "$ONE_SHOT_RECEIPT"

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

PROBE="$ROOT/final_held_out_probe"
set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  --config configs/motion_model.json \
  --train-teacher-bank "$TRAIN_BANK" \
  --validation-teacher-bank "$SEALED_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --activation-aware-g1f3 \
  --evaluation-role final_held_out \
  --frozen-severity-envelope "$FROZEN_ENVELOPE" \
  --frozen-full-shadow-repair-contract "$FROZEN_REPAIR" \
  --full-shadow-line-search-backtracks 12 \
  --full-shadow-line-search-decay 0.5 \
  --geodesic-angular-max-radians 0.7853981633974483 \
  --joint-svd-relative-cutoff 1e-6 \
  --joint-direction-norm-floor 1e-8 \
  --joint-directional-margin 1e-6 \
  --temporal-fd-epsilon-radians 1e-4 \
  --temporal-fd-relative-error-tolerance 0.1 \
  --temporal-fd-absolute-floor 1e-8 \
  --temporal-fd-near-zero-threshold 1e-5 \
  --second-order-basis-dimension 5 \
  --second-order-grid-levels 9 \
  --second-order-feasibility-tolerance 1e-12 \
  --steps 2 3 5 \
  --target-rms 1e-4 \
  --output-dir "$PROBE"
RUN_STATUS=$?
set -e

REPORT="$PROBE/fixed_budget_correction.report.json"
test -s "$REPORT"
if [[ "$RUN_STATUS" -ne 0 ]]; then
  printf '%s\n' "$MANIFEST" > "$ROOT/FAILED_HELD_OUT_BECOMES_DEVELOPMENT_EVIDENCE"
  echo "[REJECTED] One-shot held-out failed and is now development evidence. Seal a new unseen transaction." >&2
  exit 2
fi

ACCEPTANCE="$ROOT/final_held_out.acceptance.json"
"$PY" -m training.refiner_v15_15g1f3_contract verify-held-out \
  --report "$REPORT" \
  --manifest "$MANIFEST" \
  --one-shot-receipt "$ONE_SHOT_RECEIPT" \
  --output "$ACCEPTANCE"
printf '%s\n' "$REPORT" > outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_REPORT
printf '%s\n' "$ACCEPTANCE" > outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_ACCEPTANCE
printf '%s\n' "$ONE_SHOT_RECEIPT" > outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_ONE_SHOT_RECEIPT
echo "g1f3 one-shot final held-out passed; no training or teacher recycling was launched"
