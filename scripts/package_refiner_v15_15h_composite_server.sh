#!/usr/bin/env bash
# Package only after g1f3 train, reused-development, and one-shot held-out pass.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
: "${BASE_REFINER_CHECKPOINT:?Set the frozen base Refiner diagnostic_state.pt}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
FROZEN_ENVELOPE=$(cat outputs/LATEST_REFINER_V15_15G1F3_FROZEN_CONFORMAL)
FROZEN_CONTRACT=$(cat outputs/LATEST_REFINER_V15_15G1F3_FROZEN_CONTRACT)
TRAIN_REPORT=$(cat outputs/LATEST_REFINER_V15_15G1F3_TRAIN_REPORT)
DEV_REPORT=$(cat outputs/LATEST_REFINER_V15_15G1F3_DEVELOPMENT_REPORT)
HELD_OUT_REPORT=$(cat outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_REPORT)
HELD_OUT_ACCEPTANCE=$(cat outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_ACCEPTANCE)
ONE_SHOT_RECEIPT=$(cat outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_ONE_SHOT_RECEIPT)
HELD_OUT_MANIFEST=$(cat outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_MANIFEST)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
DEV_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
for path in \
  "$BASE_REFINER_CHECKPOINT" "$ADAPTER_STATE" "$FROZEN_ENVELOPE" \
  "$FROZEN_CONTRACT" "$TRAIN_REPORT" "$DEV_REPORT" "$HELD_OUT_REPORT" \
  "$HELD_OUT_ACCEPTANCE" "$ONE_SHOT_RECEIPT" "$HELD_OUT_MANIFEST" \
  "$TRAIN_BANK" "$DEV_BANK"; do test -s "$path"; done

ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
readarray -t MANIFESTS < <("$PY" - "$TRAIN_BANK" "$DEV_BANK" <<'PY'
import sys
import torch
for path in sys.argv[1:]:
    bank = torch.load(path, map_location="cpu", weights_only=False)
    print(bank["split_manifest"])
PY
)
TRAIN_MANIFEST="${MANIFESTS[0]}"
DEV_MANIFEST="${MANIFESTS[1]}"
test -s "$TRAIN_MANIFEST"
test -s "$DEV_MANIFEST"

STAMP=$(date +%Y%m%d_%H%M%S)
DESTINATION="$OUT_ROOT/checkpoints/refiner_v15_15h_composite_${STAMP}"
"$PY" -m training.package_refiner_v15_15h_composite \
  --base-refiner-checkpoint "$BASE_REFINER_CHECKPOINT" \
  --adapter-state "$ADAPTER_STATE" \
  --conformal-envelope "$FROZEN_ENVELOPE" \
  --g1f3-frozen-contract "$FROZEN_CONTRACT" \
  --train-manifest "$TRAIN_MANIFEST" \
  --dev-manifest "$DEV_MANIFEST" \
  --held-out-manifest "$HELD_OUT_MANIFEST" \
  --train-report "$TRAIN_REPORT" \
  --dev-report "$DEV_REPORT" \
  --held-out-report "$HELD_OUT_REPORT" \
  --held-out-acceptance "$HELD_OUT_ACCEPTANCE" \
  --held-out-one-shot-receipt "$ONE_SHOT_RECEIPT" \
  --implementation-commit "$EXPECTED_COMMIT" \
  --output-dir "$DESTINATION"

MODEL="$DESTINATION/v15_15h_adapter_second_order_composite.pt"
CONTRACT="$DESTINATION/v15_15h_adapter_second_order_composite.contract.json"
test -s "$MODEL"
test -s "$CONTRACT"
printf '%s\n' "$MODEL" > outputs/LATEST_REFINER_V15_15H_COMPOSITE_MODEL
printf '%s\n' "$CONTRACT" > outputs/LATEST_REFINER_V15_15H_COMPOSITE_CONTRACT
echo "Packaged Adapter + second-order repair composite at $DESTINATION"
