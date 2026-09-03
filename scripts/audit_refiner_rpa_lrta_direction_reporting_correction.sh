#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the committed correction SHA}"

RPA_REPORT="${1:?usage: audit_refiner_rpa_lrta_direction_reporting_correction.sh RPA_REPORT RPA_ADAPTER FREEZE_MANIFEST OUTPUT_DIR}"
RPA_ADAPTER="${2:?usage: audit_refiner_rpa_lrta_direction_reporting_correction.sh RPA_REPORT RPA_ADAPTER FREEZE_MANIFEST OUTPUT_DIR}"
FREEZE_MANIFEST="${3:?usage: audit_refiner_rpa_lrta_direction_reporting_correction.sh RPA_REPORT RPA_ADAPTER FREEZE_MANIFEST OUTPUT_DIR}"
OUTPUT_DIR="${4:?usage: audit_refiner_rpa_lrta_direction_reporting_correction.sh RPA_REPORT RPA_ADAPTER FREEZE_MANIFEST OUTPUT_DIR}"

DEVICE="${DEVICE:-cuda}"

cd "$REPO_DIR"

test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"

test -f "$RPA_REPORT"
test -f "$RPA_ADAPTER"
test -f "$FREEZE_MANIFEST"

"$PYTHON" -m training.refiner_rpa_lrta_direction_reporting_correction \
  --rpa-report "$RPA_REPORT" \
  --rpa-adapter "$RPA_ADAPTER" \
  --freeze-manifest "$FREEZE_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --expected-main-commit "$EXPECTED_MAIN_COMMIT" \
  --device "$DEVICE"
