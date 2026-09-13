#!/usr/bin/env bash
# Seal one deterministic, previously unseen transaction before Oracle access.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
: "${FINAL_HELD_OUT_CANDIDATE_BANK:?Set the untouched candidate transaction bank}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
SELECTION_SEED="${FINAL_HELD_OUT_SELECTION_SEED:-v15.15g1f3-heldout-v1}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -s "$FINAL_HELD_OUT_CANDIDATE_BANK"

TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
DEV_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
test -s "$TRAIN_BANK"
test -s "$DEV_BANK"

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT="$OUT_ROOT/checkpoints/refiner_v15_15g1f3_held_out_${STAMP}"
mkdir -p "$ROOT" outputs
SEALED_BANK="$ROOT/final_held_out.sealed.pt"
MANIFEST="$ROOT/final_held_out.manifest.json"
EXCLUDE_ARGS=()
for prior in "$OUT_ROOT"/checkpoints/refiner_v15_15g1f3_held_out_*/final_held_out.manifest.json; do
  [[ -e "$prior" ]] && EXCLUDE_ARGS+=(--exclude-manifest "$prior")
done

ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m training.refiner_v15_15g1f3_contract seal-held-out \
  --candidate-bank "$FINAL_HELD_OUT_CANDIDATE_BANK" \
  --train-bank "$TRAIN_BANK" \
  --dev-bank "$DEV_BANK" \
  --selection-seed "$SELECTION_SEED" \
  "${EXCLUDE_ARGS[@]}" \
  --output-bank "$SEALED_BANK" \
  --output-manifest "$MANIFEST"

printf '%s\n' "$ROOT" > outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_ROOT
printf '%s\n' "$SEALED_BANK" > outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_BANK
printf '%s\n' "$MANIFEST" > outputs/LATEST_REFINER_V15_15G1F3_HELD_OUT_MANIFEST
echo "Held-out transaction is sealed. Run Oracle independently against only $SEALED_BANK."
echo "Do not run g1f3 until the Oracle report has raw, Projector, scope, and complete-Guard evidence."
