#!/usr/bin/env bash
# V5 is a historical rejected pilot, not a launcher for the V6 objective.
set -Eeuo pipefail
echo "[RETIRED] V5 training is archived. Do not resume V5 snapshots under V6." >&2
echo "Use: bash scripts/train_refiner_v6.sh diagnose EXISTING_OUT_ROOT NEW_TAG" >&2
echo "See docs/refiner_v6_training.md for the diagnostic/pilot/resume protocol." >&2
exit 2
