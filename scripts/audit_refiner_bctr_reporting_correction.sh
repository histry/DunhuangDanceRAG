#!/usr/bin/env bash
set -Eeuo pipefail

# Create-only correction of the frozen BCTR reporting artifact.  The source
# report is immutable and is never regenerated or overwritten.
REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the checked-out main SHA}"
BCTR_REPORT="${1:?usage: audit_refiner_bctr_reporting_correction.sh BCTR_REPORT RUN_DIR}"
OUTPUT_DIR="${2:?usage: audit_refiner_bctr_reporting_correction.sh BCTR_REPORT RUN_DIR}"

cd "$REPO_DIR"
test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"
test -f "$BCTR_REPORT"

"$PYTHON" -m training.refiner_bctr_reporting_correction \
  --bctr-report "$BCTR_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-main-commit "$EXPECTED_MAIN_COMMIT"

# This audit only repairs reporting fields in a separate artifact.  It does
# not load a model, run inference, recompute metrics, or authorize Pilot.
