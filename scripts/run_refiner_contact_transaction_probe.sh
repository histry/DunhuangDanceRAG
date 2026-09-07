#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 MOTION CONFIG SLIDING_SUPPORT_ELIGIBLE OUTPUT_DIR [--start N --end N | --max-windows N]" >&2
  exit 2
fi

MOTION=$1
CONFIG=$2
ELIGIBLE=$3
OUTPUT_DIR=$4
shift 4
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" -m training.contact_transaction_probe \
  --motion "$MOTION" \
  --config "$CONFIG" \
  --eligible "$ELIGIBLE" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
