#!/usr/bin/env bash
set -Eeuo pipefail

# Read-only Phase 2.1. The explicit Phase 2 report is the only upstream
# selector; the Python audit follows the paths and hashes recorded in it.
REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the full Phase 2.1 commit SHA}"
PHASE2_REPORT="${1:?usage: audit_refiner_width_mechanism_adjudication.sh PHASE2_REPORT RUN_DIR}"
OUTPUT_DIR="${2:?usage: audit_refiner_width_mechanism_adjudication.sh PHASE2_REPORT RUN_DIR}"

cd "$REPO_DIR"
test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"
test -f "$PHASE2_REPORT"

"$PYTHON" -m training.refiner_width_mechanism_adjudication_audit \
  --phase2-report "$PHASE2_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-main-commit "$EXPECTED_MAIN_COMMIT" \
  --device "${DEVICE:-cuda}"

# Stop after the fixed-state report. No intervention, training, or Pilot.
