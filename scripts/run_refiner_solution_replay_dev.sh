#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 BUNDLE SOURCE_REPORT FEASIBILITY_RUN OUTPUT_DIR [extra replay args...]" >&2
  exit 2
fi

BUNDLE=$1
SOURCE_REPORT=$2
FEASIBILITY_RUN=$3
OUTPUT_DIR=$4
shift 4

PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -m training.generation_stage_diagnostics replay \
  --bundle "$BUNDLE" \
  --source-report "$SOURCE_REPORT" \
  --feasibility-run "$FEASIBILITY_RUN" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
