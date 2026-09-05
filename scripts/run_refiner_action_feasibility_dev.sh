#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CASE_MANIFEST OUTPUT_DIR [additional evaluator arguments...]" >&2
  exit 2
fi

CASE_MANIFEST=$1
OUTPUT_DIR=$2
shift 2

PYTHON_BIN=${PYTHON_BIN:-python}
exec "$PYTHON_BIN" -m training.refiner_action_feasibility_evaluation \
  --case-manifest "$CASE_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
