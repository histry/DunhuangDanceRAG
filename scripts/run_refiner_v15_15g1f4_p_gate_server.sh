#!/usr/bin/env bash
# Server-only Gate 0 parity followed by one P-only train calibration run.
# M is intentionally unavailable until train-only equivalent-radius calibration.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
: "${V9_REFERENCE_REPORT:?Set V9_REFERENCE_REPORT to the completed be71ae12 v9 report}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -s "$V9_REFERENCE_REPORT"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
for path in "$ADAPTER_STATE" "$TRAIN_BANK"; do test -s "$path"; done

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g1f4_p_gate_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
LOG="logs/refiner_v15_15g1f4_p_gate_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1f4_p_gate_${STAMP}.exit_status.txt"
mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1F4_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1F4_LOG
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
  --validation-teacher-bank "$TRAIN_BANK"
  --evaluation-role train_calibration
  --metric-mode identity
)

BASELINE_DIR="$ROOT/current_equal_share_identity"
set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --progress-mode current_equal_share \
  --output-dir "$BASELINE_DIR"
BASELINE_STATUS=$?
set -e
if test "$BASELINE_STATUS" -ne 0 && test "$BASELINE_STATUS" -ne 2; then
  echo "baseline execution failed unexpectedly: $BASELINE_STATUS" >&2
  exit "$BASELINE_STATUS"
fi

BASELINE_REPORT="$BASELINE_DIR/fixed_budget_correction.report.json"
PARITY_REPORT="$ROOT/v9_canonical_parity_gate.json"
test -s "$BASELINE_REPORT"
"$PY" -m training.refiner_v15_15g1f4_parity \
  --v9-reference-report "$V9_REFERENCE_REPORT" \
  --candidate-report "$BASELINE_REPORT" \
  --output "$PARITY_REPORT"
printf '%s\n' "$PARITY_REPORT" > outputs/LATEST_REFINER_V15_15G1F4_PARITY_REPORT

P_DIR="$ROOT/weighted_debt_filter_identity"
set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --progress-mode weighted_debt_filter \
  --output-dir "$P_DIR"
P_STATUS=$?
set -e
P_REPORT="$P_DIR/fixed_budget_correction.report.json"
test -s "$P_REPORT"
printf '%s\n' "$P_REPORT" > outputs/LATEST_REFINER_V15_15G1F4_P_REPORT

echo "g1f4 Gate 0 passed; P-only train report: $P_REPORT"
exit "$P_STATUS"
