#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/review_refiner_role_conditioned_support_projection.sh RCSP_REPORT_JSON NEW_REVIEW_JSON" >&2
  exit 2
fi

: "${PY:?Set PY to the validated environment Python executable}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the reviewed main commit}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT to the RCSP experiment commit}"

REPORT=$1
OUTPUT=$2

test -x "$PY"
test -f "$REPORT"
test ! -e "$OUTPUT"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain)"
git fetch origin main
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"

echo "[REVIEW ONLY] Recompute stored BASE/RCSP summaries and correct headline reporting logic."
echo "[IMMUTABLE] No checkpoint load, model forward, threshold edit, training, selection, production edit or Pilot."

"$PY" -m training.refiner_role_conditioned_support_projection_review \
  --report "$REPORT" \
  --output "$OUTPUT" \
  --expected-main-commit "$EXPECTED_COMMIT" \
  --expected-source-commit "$EXPECTED_SOURCE_COMMIT"

echo "[STOP] Inspect the direction/support summaries. Pilot remains forbidden."
