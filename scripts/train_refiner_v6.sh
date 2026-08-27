#!/usr/bin/env bash
set -Eeuo pipefail
echo "[RETIRED] V6 synthetic-noise experiments cannot publish V7 models." >&2
echo "Use scripts/train_refiner_v7.sh diagnose EXISTING_OUT_ROOT NEW_TAG; see docs/refiner_v7_training.md." >&2
exit 2
