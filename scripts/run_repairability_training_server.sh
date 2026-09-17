#!/usr/bin/env bash
# Validate a completed Outcome Bank, then train matched linear/MLP predictors.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?export EXPECTED_COMMIT=<full main SHA>}"
: "${REPAIRABILITY_BANK:?set the append-only Outcome Bank JSONL path}"
: "${REPAIRABILITY_SEEDS:?set the exact comma-separated capture seeds}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${REPAIRABILITY_OUT_ROOT:-outputs/repairability_$(date +%Y%m%d_%H%M%S)}"
LOG="${REPAIRABILITY_LOG:-logs/repairability_$(date +%Y%m%d_%H%M%S).log}"
SPLIT_ISOLATION="${REPAIRABILITY_SPLIT_ISOLATION:-all}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -x "$PY"
test -s "$REPAIRABILITY_BANK"

mkdir -p "$OUT_ROOT" logs outputs
printf '%s\n' "$OUT_ROOT" > outputs/LATEST_REPAIRABILITY_ROOT
printf '%s\n' "$LOG" > outputs/LATEST_REPAIRABILITY_LOG
exec > >(tee -a "$LOG") 2>&1

ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MOTION_DEVICE=cuda MOTION_GPU_PREPROCESSING=1

"$PY" -m pytest -q \
  tests/test_repairability_predictor.py \
  tests/test_gar_evaluation_readiness.py
"$PY" -m evaluation.repairability_outcome_bank \
  --bank "$REPAIRABILITY_BANK" \
  --seeds "$REPAIRABILITY_SEEDS" \
  --require-complete \
  --output "$OUT_ROOT/outcome_bank_summary.json"
"$PY" -u -m training.repairability_predictor \
  --bank "$REPAIRABILITY_BANK" \
  --output-dir "$OUT_ROOT" \
  --device cuda \
  --split-isolation "$SPLIT_ISOLATION"

echo "REPAIRABILITY_SERVER_TRAINING_OK"
