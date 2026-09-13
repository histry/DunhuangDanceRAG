#!/usr/bin/env bash
# Server-only V15.15g1f1 train calibration.
# Case 53, final held-out, training, recycling and generation remain disabled.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
test -s "$ADAPTER_STATE"
test -s "$TRAIN_BANK"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g1f1_temporal_directional_consistency_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
TRAIN_PROBE="$ROOT/train_directional_consistency_calibration"
LOG="logs/refiner_v15_15g1f1_temporal_directional_consistency_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1f1_temporal_directional_consistency_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15g1f1_temporal_directional_consistency_${STAMP}.scientific_status.txt"

mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1F1_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1F1_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15G1F1_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15G1F1_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "adapter_state=$ADAPTER_STATE"
echo "train_bank=$TRAIN_BANK"
echo "evaluation_role=train_calibration_only"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  --config configs/motion_model.json \
  --train-teacher-bank "$TRAIN_BANK" \
  --validation-teacher-bank "$TRAIN_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --activation-aware-g1f1 \
  --evaluation-role train_calibration \
  --output-dir "$TRAIN_PROBE" \
  --severity-envelope-absolute-margin 1e-6 \
  --severity-scale-floor 1e-6 \
  --severity-conformal-shrinkage 0.1 \
  --severity-conformal-uncertainty-fraction 0.05 \
  --guard-proxy-nonregression-tolerance 1e-6 \
  --guard-smooth-max-temperature 1e-3 \
  --correction-temporal-smoothness-weight 0.05 \
  --guard-minimum-reduction 1e-12 \
  --guard-safe-interior-margin 1e-6 \
  --full-shadow-lse-allowance-fraction 0.01 \
  --full-shadow-lse-temperature-floor 1e-8 \
  --full-shadow-projection-damping 1e-12 \
  --full-shadow-line-search-backtracks 12 \
  --full-shadow-line-search-decay 0.5 \
  --science-restoration-damping 1e-8 \
  --science-restoration-safety-fraction 0.25 \
  --geodesic-angular-max-radians 0.7853981633974483 \
  --joint-svd-relative-cutoff 1e-6 \
  --joint-direction-norm-floor 1e-8 \
  --joint-directional-margin 1e-6 \
  --temporal-fd-epsilon-radians 1e-4 \
  --temporal-fd-relative-error-tolerance 0.1 \
  --temporal-fd-absolute-floor 1e-8 \
  --steps 2 3 5 \
  --target-rms 1e-4
TRAIN_STATUS=$?
set -e
if [[ "$TRAIN_STATUS" -ne 0 && "$TRAIN_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15g1f1 train calibration failed with status $TRAIN_STATUS"
  exit "$TRAIN_STATUS"
fi

TRAIN_REPORT="$TRAIN_PROBE/fixed_budget_correction.report.json"
FROZEN_ENVELOPE="$TRAIN_PROBE/discriminative_transaction_conformal_severity.json"
FROZEN_REPAIR="$TRAIN_PROBE/train_frozen_full_shadow_repair_contract.json"
test -s "$TRAIN_REPORT"
test -s "$FROZEN_ENVELOPE"
test -s "$FROZEN_REPAIR"
printf '%s\n' "$TRAIN_REPORT" > outputs/LATEST_REFINER_V15_15G1F1_TRAIN_REPORT
printf '%s\n' "$FROZEN_ENVELOPE" > outputs/LATEST_REFINER_V15_15G1F1_FROZEN_CONFORMAL
printf '%s\n' "$FROZEN_REPAIR" > outputs/LATEST_REFINER_V15_15G1F1_FROZEN_REPAIR_CONTRACT
printf '%s\n' "$TRAIN_STATUS" > "$SCIENTIFIC_STATUS"

"$PY" - "$TRAIN_REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(json.dumps({
    "stage": "v15_15g1f1_train_directional_consistency_receipt",
    "scientific_status": 0 if r.get(
        "local_feasible_intersection_summary", {}
    ).get("local_feasible_intersection_complete") else 2,
    "schema": r.get("schema"),
    "optimization_coordinate": r.get("optimization_coordinate"),
    "jacobian_coordinate": r.get("jacobian_coordinate"),
    "temporal_fd_epsilon_radians": r.get("temporal_fd_epsilon_radians"),
    "local_feasible_intersection_summary": r.get(
        "local_feasible_intersection_summary"
    ),
    "case_53_launched": False,
    "development_validation_launched": False,
    "final_held_out_launched": False,
}, ensure_ascii=False, indent=2))
PY

echo "V15.15g1f1 train-only directional-consistency calibration completed; scientific_status=$TRAIN_STATUS"
echo "No case 53, development validation, final held-out, training, pseudo-teacher recycling, promotion, replay, or generation was launched."
