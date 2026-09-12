#!/usr/bin/env bash
# Server-only V15.15g fixed-budget correction ablation; no training.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

FORMAL_TAG=$(cat outputs/LATEST_REFINER_V15_15F_BOUNDED_TRAINING_TAG)
ADAPTER_STATE="$OUT_ROOT/checkpoints/$FORMAL_TAG/bounded_training/bounded_formal_adapter_state.pt"
VALIDATION_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
test -s "$ADAPTER_STATE"
test -s "$VALIDATION_BANK"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g_fixed_budget_correction_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
ABLATION_DIR="$ROOT/correction_ablation"
LOG="logs/refiner_v15_15g_fixed_budget_correction_${STAMP}.log"
STATUS="outputs/refiner_v15_15g_fixed_budget_correction_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15g_fixed_budget_correction_${STAMP}.scientific_status.txt"

mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G_CORRECTION_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G_CORRECTION_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15G_CORRECTION_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15G_CORRECTION_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "formal_tag=$FORMAL_TAG"
echo "adapter_state=$ADAPTER_STATE"
echo "validation_bank=$VALIDATION_BANK"
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
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  --config configs/motion_model.json \
  --validation-teacher-bank "$VALIDATION_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --output-dir "$ABLATION_DIR" \
  --steps 2 3 5 \
  --target-rms 1e-4
RUN_STATUS=$?
set -e
printf '%s\n' "$RUN_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$RUN_STATUS" -ne 0 && "$RUN_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15g execution failed with status $RUN_STATUS"
  exit "$RUN_STATUS"
fi

REPORT="$ABLATION_DIR/fixed_budget_correction.report.json"
test -s "$REPORT"
printf '%s\n' "$REPORT" > outputs/LATEST_REFINER_V15_15G_CORRECTION_REPORT
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(json.dumps({
    "schema": r.get("schema"),
    "correction_gradient_protocol": r.get(
        "correction_gradient_protocol"
    ),
    "future_end_to_end_gradient_protocol": r.get(
        "future_end_to_end_gradient_protocol"
    ),
    "riemannian_correction_supported": r.get(
        "riemannian_correction_supported"
    ),
    "numeric_audit_complete": r.get("numeric_audit_complete"),
    "variants": {
        key: {
            "raw": value.get("raw_pass_count_by_group"),
            "projected": value.get("projected_pass_count_by_group"),
            "numeric_failures": value.get("numeric_failure_count"),
            "elapsed_seconds": value.get("elapsed_seconds"),
        }
        for key, value in r.get("variants", {}).items()
    },
}, ensure_ascii=False, indent=2))
PY
echo "V15.15g fixed-budget correction ablation completed; scientific_status=$RUN_STATUS"
echo "No Adapter training, pseudo-teacher recycling, promotion, replay, or generation was launched."
