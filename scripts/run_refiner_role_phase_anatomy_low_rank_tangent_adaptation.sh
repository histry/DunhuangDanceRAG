#!/usr/bin/env bash
set -Eeuo pipefail

# RPA-LRTA formal research-method candidate.
# This script refuses dirty/uncommitted source so the report has one exact SHA.

REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the committed RPA-LRTA SHA}"
ROOT_DIR="${ROOT_DIR:?set ROOT_DIR to the formal output root}"
PHASE21_REPORT="${1:?usage: run_refiner_role_phase_anatomy_low_rank_tangent_adaptation.sh PHASE21_REPORT BCTR_REPORT RUN_DIR}"
BCTR_REPORT="${2:?usage: run_refiner_role_phase_anatomy_low_rank_tangent_adaptation.sh PHASE21_REPORT BCTR_REPORT RUN_DIR}"
OUTPUT_DIR="${3:?usage: run_refiner_role_phase_anatomy_low_rank_tangent_adaptation.sh PHASE21_REPORT BCTR_REPORT RUN_DIR}"

LEGACY_CORE_STRENGTH="${LEGACY_CORE_STRENGTH:-0.02}"
LEGACY_TRANSITION_STRENGTH="${LEGACY_TRANSITION_STRENGTH:-1.0}"
DEVICE="${DEVICE:-cuda}"

cd "$REPO_DIR"

test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"
test -f "$PHASE21_REPORT"
test -f "$BCTR_REPORT"

"$PYTHON" -m training.refiner_role_phase_anatomy_low_rank_tangent_adaptation \
  --phase21-report "$PHASE21_REPORT" \
  --bctr-report "$BCTR_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-main-commit "$EXPECTED_MAIN_COMMIT" \
  --legacy-core-strength "$LEGACY_CORE_STRENGTH" \
  --legacy-transition-strength "$LEGACY_TRANSITION_STRENGTH" \
  --device "$DEVICE"

# Formal stop: no production integration or Pilot is authorized by this script.
