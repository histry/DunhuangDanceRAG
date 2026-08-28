#!/usr/bin/env bash
set -Eeuo pipefail
echo "[RETIRED] V6 synthetic-noise experiments cannot publish V8 models." >&2
echo "Use scripts/train_refiner_v8.sh foundation EXISTING_OUT_ROOT NEW_TAG; see docs/refiner_v8_training.md." >&2
exit 2
