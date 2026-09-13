#!/usr/bin/env bash
# Server-only V15.15g1f train calibration and reused-development probe.
# It launches no training and never presents case 53 as pristine held-out data.
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
DEV_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
test -s "$ADAPTER_STATE"
test -s "$TRAIN_BANK"
test -s "$DEV_BANK"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g1f_exact_radius_geodesic_joint_sqp_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
TRAIN_PROBE="$ROOT/train_feasible_intersection_calibration"
DEV_PROBE="$ROOT/development_validation_case53"
LOG="logs/refiner_v15_15g1f_exact_radius_geodesic_joint_sqp_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1f_exact_radius_geodesic_joint_sqp_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15g1f_exact_radius_geodesic_joint_sqp_${STAMP}.scientific_status.txt"

mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1F_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1F_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15G1F_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15G1F_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "adapter_state=$ADAPTER_STATE"
echo "train_bank=$TRAIN_BANK"
echo "development_bank=$DEV_BANK"
echo "case_53_evidence_role=development_validation_reused"
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

COMMON_ARGS=(
  --config configs/motion_model.json
  --train-teacher-bank "$TRAIN_BANK"
  --adapter-state "$ADAPTER_STATE"
  --activation-aware-g1f
  --severity-envelope-absolute-margin 1e-6
  --severity-scale-floor 1e-6
  --severity-conformal-shrinkage 0.1
  --severity-conformal-uncertainty-fraction 0.05
  --guard-proxy-nonregression-tolerance 1e-6
  --guard-smooth-max-temperature 1e-3
  --correction-temporal-smoothness-weight 0.05
  --guard-minimum-reduction 1e-12
  --guard-safe-interior-margin 1e-6
  --full-shadow-lse-allowance-fraction 0.01
  --full-shadow-lse-temperature-floor 1e-8
  --full-shadow-projection-damping 1e-12
  --full-shadow-line-search-backtracks 12
  --full-shadow-line-search-decay 0.5
  --science-restoration-damping 1e-8
  --science-restoration-safety-fraction 0.25
  --geodesic-angular-max-radians 0.7853981633974483
  --joint-svd-relative-cutoff 1e-6
  --joint-direction-norm-floor 1e-8
  --joint-directional-margin 1e-6
  --steps 2 3 5
  --target-rms 1e-4
)

set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --validation-teacher-bank "$TRAIN_BANK" \
  --evaluation-role train_calibration \
  --output-dir "$TRAIN_PROBE"
TRAIN_STATUS=$?
set -e
if [[ "$TRAIN_STATUS" -ne 0 && "$TRAIN_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15g1f train calibration failed with status $TRAIN_STATUS"
  exit "$TRAIN_STATUS"
fi

TRAIN_REPORT="$TRAIN_PROBE/fixed_budget_correction.report.json"
FROZEN_ENVELOPE="$TRAIN_PROBE/discriminative_transaction_conformal_severity.json"
FROZEN_REPAIR="$TRAIN_PROBE/train_frozen_full_shadow_repair_contract.json"
test -s "$TRAIN_REPORT"
test -s "$FROZEN_ENVELOPE"
test -s "$FROZEN_REPAIR"
printf '%s\n' "$TRAIN_REPORT" > outputs/LATEST_REFINER_V15_15G1F_TRAIN_REPORT
printf '%s\n' "$FROZEN_ENVELOPE" > outputs/LATEST_REFINER_V15_15G1F_FROZEN_CONFORMAL
printf '%s\n' "$FROZEN_REPAIR" > outputs/LATEST_REFINER_V15_15G1F_FROZEN_REPAIR_CONTRACT

if [[ "$TRAIN_STATUS" -eq 2 ]]; then
  printf '%s\n' 2 > "$SCIENTIFIC_STATUS"
  "$PY" - "$TRAIN_REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(json.dumps({
    "stage": "v15_15g1f_train_feasible_intersection_rejected",
    "scientific_status": 2,
    "local_feasible_intersection_summary": r.get(
        "local_feasible_intersection_summary"
    ),
    "development_validation_launched": False,
    "final_held_out_launched": False,
}, ensure_ascii=False, indent=2))
PY
  echo "No development validation, final held-out evaluation, training, promotion, replay, or generation was launched."
  exit 0
fi

set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --validation-teacher-bank "$DEV_BANK" \
  --evaluation-role development_validation \
  --frozen-severity-envelope "$FROZEN_ENVELOPE" \
  --frozen-full-shadow-repair-contract "$FROZEN_REPAIR" \
  --output-dir "$DEV_PROBE"
DEV_STATUS=$?
set -e
printf '%s\n' "$DEV_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$DEV_STATUS" -ne 0 && "$DEV_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15g1f development validation failed with status $DEV_STATUS"
  exit "$DEV_STATUS"
fi

DEV_REPORT="$DEV_PROBE/fixed_budget_correction.report.json"
test -s "$DEV_REPORT"
printf '%s\n' "$DEV_REPORT" > outputs/LATEST_REFINER_V15_15G1F_DEVELOPMENT_REPORT
"$PY" - "$TRAIN_REPORT" "$DEV_REPORT" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
dev = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
a = dev.get("activation_aware_summary") or {}
decisions = a.get("decisions") or {}
case_53 = decisions.get("txn_0000_94bfdf553811:53")
print(json.dumps({
    "schema": dev.get("schema"),
    "scientific_status": 0 if dev.get("activation_aware_supported") else 2,
    "train_local_feasible_intersection": train.get(
        "local_feasible_intersection_summary"
    ),
    "case_53_evidence_role": "development_validation_reused",
    "case_53_pristine_held_out": False,
    "case_53_selection": case_53,
    "activation_aware_supported_on_development": dev.get(
        "activation_aware_supported"
    ),
    "final_held_out_required": True,
    "final_held_out_launched": False,
    "fixed_guard_thresholds_changed": dev.get(
        "fixed_guard_thresholds_changed"
    ),
    "observable_0p03_gate_changed": dev.get(
        "observable_0p03_gate_changed"
    ),
}, ensure_ascii=False, indent=2))
PY
echo "V15.15g1f train calibration and reused-development probe completed; scientific_status=$DEV_STATUS"
echo "No final held-out evaluation, training, pseudo-teacher recycling, promotion, replay, or generation was launched."
