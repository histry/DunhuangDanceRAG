#!/usr/bin/env bash
# Server-only g1f3 train calibration (three cold starts) and reused development.
# This script performs no Adapter training and never launches final held-out.
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
for path in "$ADAPTER_STATE" "$TRAIN_BANK" "$DEV_BANK"; do test -s "$path"; done

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
TAG="refiner_v15_15g1f3_train_dev_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
LOG="logs/refiner_v15_15g1f3_train_dev_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1f3_train_dev_${STAMP}.exit_status.txt"
mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1F3_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1F3_LOG
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"' EXIT
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --config configs/motion_model.json
  --train-teacher-bank "$TRAIN_BANK"
  --adapter-state "$ADAPTER_STATE"
  --activation-aware-g1f3
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
  --temporal-fd-epsilon-radians 1e-4
  --temporal-fd-relative-error-tolerance 0.1
  --temporal-fd-absolute-floor 1e-8
  --temporal-fd-near-zero-threshold 1e-5
  --second-order-basis-dimension 5
  --second-order-grid-levels 9
  --second-order-feasibility-tolerance 1e-12
  --second-order-guard-transition-band 1e-5
  --steps 2 3 5
  --target-rms 1e-4
)

TRAIN_REPORTS=()
for replicate in 1 2 3; do
  destination="$ROOT/train_cold_start_${replicate}"
  "$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
    "${COMMON_ARGS[@]}" \
    --validation-teacher-bank "$TRAIN_BANK" \
    --evaluation-role train_calibration \
    --output-dir "$destination"
  TRAIN_REPORTS+=("$destination/fixed_budget_correction.report.json")
done

COLD_ACCEPTANCE="$ROOT/three_cold_start.acceptance.json"
"$PY" -m training.refiner_v15_15g1f3_contract compare-replicates \
  --report "${TRAIN_REPORTS[0]}" \
  --report "${TRAIN_REPORTS[1]}" \
  --report "${TRAIN_REPORTS[2]}" \
  --output "$COLD_ACCEPTANCE"

FROZEN_ENVELOPE="$ROOT/train_cold_start_1/discriminative_transaction_conformal_severity.json"
FROZEN_REPAIR="$ROOT/train_cold_start_1/train_frozen_full_shadow_repair_contract.json"
DEV_DIR="$ROOT/development_validation_case53"
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --validation-teacher-bank "$DEV_BANK" \
  --evaluation-role development_validation \
  --frozen-severity-envelope "$FROZEN_ENVELOPE" \
  --frozen-full-shadow-repair-contract "$FROZEN_REPAIR" \
  --output-dir "$DEV_DIR"
DEV_REPORT="$DEV_DIR/fixed_budget_correction.report.json"

FROZEN_CONTRACT="$ROOT/g1f3_frozen_contract.json"
"$PY" -m training.refiner_v15_15g1f3_contract freeze \
  --train-report "${TRAIN_REPORTS[0]}" \
  --dev-report "$DEV_REPORT" \
  --cold-start-acceptance "$COLD_ACCEPTANCE" \
  --repair-contract "$FROZEN_REPAIR" \
  --conformal-envelope "$FROZEN_ENVELOPE" \
  --train-manifest "$TRAIN_MANIFEST" \
  --dev-manifest "$DEV_MANIFEST" \
  --implementation-commit "$EXPECTED_COMMIT" \
  --output "$FROZEN_CONTRACT"

printf '%s\n' "${TRAIN_REPORTS[0]}" > outputs/LATEST_REFINER_V15_15G1F3_TRAIN_REPORT
printf '%s\n' "$DEV_REPORT" > outputs/LATEST_REFINER_V15_15G1F3_DEVELOPMENT_REPORT
printf '%s\n' "$FROZEN_ENVELOPE" > outputs/LATEST_REFINER_V15_15G1F3_FROZEN_CONFORMAL
printf '%s\n' "$FROZEN_REPAIR" > outputs/LATEST_REFINER_V15_15G1F3_FROZEN_REPAIR_CONTRACT
printf '%s\n' "$FROZEN_CONTRACT" > outputs/LATEST_REFINER_V15_15G1F3_FROZEN_CONTRACT
echo "g1f3 train calibration and reused development passed; final held-out was not launched"
