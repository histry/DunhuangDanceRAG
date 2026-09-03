#!/usr/bin/env bash
set -Eeuo pipefail

# Fixed-budget SECDR intervention.  Phase 2.1, BCTR, and the BCTR correction
# are explicit frozen inputs; no latest-artifact discovery is permitted.
REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the final main SHA}"
PHASE21_REPORT="${1:?usage: run_refiner_support_extent_direction_rotation_intervention.sh PHASE21_REPORT BCTR_REPORT BCTR_CORRECTION_REPORT RUN_DIR}"
BCTR_REPORT="${2:?usage: run_refiner_support_extent_direction_rotation_intervention.sh PHASE21_REPORT BCTR_REPORT BCTR_CORRECTION_REPORT RUN_DIR}"
BCTR_CORRECTION_REPORT="${3:?usage: run_refiner_support_extent_direction_rotation_intervention.sh PHASE21_REPORT BCTR_REPORT BCTR_CORRECTION_REPORT RUN_DIR}"
OUTPUT_DIR="${4:?usage: run_refiner_support_extent_direction_rotation_intervention.sh PHASE21_REPORT BCTR_REPORT BCTR_CORRECTION_REPORT RUN_DIR}"
ROOT_DIR="${ROOT_DIR:?set ROOT_DIR to outputs/run_smpl14_formal_20260822_163915}"
LEGACY_CORE_STRENGTH="${LEGACY_CORE_STRENGTH:?set the recorded legacy core decoder strength}"
LEGACY_TRANSITION_STRENGTH="${LEGACY_TRANSITION_STRENGTH:?set the recorded legacy transition decoder strength}"

cd "$REPO_DIR"
test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"
test -f "$PHASE21_REPORT"
test -f "$BCTR_REPORT"
test -f "$BCTR_CORRECTION_REPORT"

"$PYTHON" -m training.refiner_support_extent_direction_rotation_intervention \
  --phase21-report "$PHASE21_REPORT" \
  --bctr-report "$BCTR_REPORT" \
  --bctr-correction-report "$BCTR_CORRECTION_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-main-commit "$EXPECTED_MAIN_COMMIT" \
  --legacy-core-strength "$LEGACY_CORE_STRENGTH" \
  --legacy-transition-strength "$LEGACY_TRANSITION_STRENGTH" \
  --device "${DEVICE:-cuda}"

# Stop after the fixed 400-step diagnostic and report.  No production model,
# default configuration, formal inference, publication, or Pilot is changed.
