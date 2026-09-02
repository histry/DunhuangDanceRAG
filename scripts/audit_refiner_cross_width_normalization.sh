#!/usr/bin/env bash
set -euo pipefail

# Read-only Phase 2. No optimizer, no parameter update, no Pilot.
REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
ROOT_DIR="${ROOT_DIR:?set ROOT_DIR to outputs/run_smpl14_formal_20260822_163915}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the full SHA of the checked-out Phase 2 audit commit}"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/checkpoints/refiner_v15_4_1_lazy_reservoir_foundation_20260831_235832/bridge_diagnostic}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-$ROOT_DIR/audits/zero_start_trajectory_20260901_112920_vpO8Lh/trajectory}"
RCSP_DIR="${RCSP_DIR:-$ROOT_DIR/audits/role_conditioned_support_projection_20260902_132948_qu1hYg/result}"
PARAMETER_ATTRIBUTION_REPORT="${PARAMETER_ATTRIBUTION_REPORT:-$ROOT_DIR/audits/rcsp_single_direction_attribution_20260902_145442_VWA1LQ/result/report.json}"
PHASE1_REPORT="${PHASE1_REPORT:-$ROOT_DIR/audits/single_direction_decomposition_20260902_213356_uH9fqu/result/report.json}"
SINGLE_DECOMPOSITION_REPORT="${SINGLE_DECOMPOSITION_REPORT:-$PHASE1_REPORT}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/audits/cross_width_normalization_$(date +%Y%m%d_%H%M%S)_${RANDOM}/result}"
LEGACY_CORE_STRENGTH="${LEGACY_CORE_STRENGTH:?set the recorded legacy core decoder strength}"
LEGACY_TRANSITION_STRENGTH="${LEGACY_TRANSITION_STRENGTH:?set the recorded legacy transition decoder strength}"

cd "$REPO_DIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"

"$PYTHON" -m training.refiner_cross_width_normalization_audit \
  --root-dir "$ROOT_DIR" \
  --state-dir "$STATE_DIR" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --rcsp-dir "$RCSP_DIR" \
  --parameter-attribution-report "$PARAMETER_ATTRIBUTION_REPORT" \
  --phase1-report "$PHASE1_REPORT" \
  --single-decomposition-report "$SINGLE_DECOMPOSITION_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-main-commit "$EXPECTED_MAIN_COMMIT" \
  --legacy-core-strength "$LEGACY_CORE_STRENGTH" \
  --legacy-transition-strength "$LEGACY_TRANSITION_STRENGTH" \
  --device "${DEVICE:-cuda}"

# Pilot remains forbidden after Phase 2; stop after evidence output.
