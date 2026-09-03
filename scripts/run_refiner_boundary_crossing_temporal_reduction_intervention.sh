#!/usr/bin/env bash
set -Eeuo pipefail

# Read-only Phase 3 BCTR intervention.  The explicit Phase 2.1 report is the
# only upstream selector; the Python audit follows its recorded paths/hashes.
REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the full Phase 3 commit SHA}"
PHASE21_REPORT="${1:?usage: run_refiner_boundary_crossing_temporal_reduction_intervention.sh PHASE21_REPORT RUN_DIR}"
OUTPUT_DIR="${2:?usage: run_refiner_boundary_crossing_temporal_reduction_intervention.sh PHASE21_REPORT RUN_DIR}"

cd "$REPO_DIR"
test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"
test -f "$PHASE21_REPORT"

"$PYTHON" -m training.refiner_boundary_crossing_temporal_reduction_intervention \
  --phase21-report "$PHASE21_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-main-commit "$EXPECTED_MAIN_COMMIT" \
  --device "${DEVICE:-cuda}"

# Stop after the fixed-state BCTR report.  No training, direction intervention,
# checkpoint search, production modification, or Pilot is performed here.
