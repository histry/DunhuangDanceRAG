#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 STATE_DIR TRAJECTORY_DIR RCSP_RESULT_DIR PARAMETER_ATTRIBUTION_REPORT RUN_DIR" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PY:?Set PY to the exact Python interpreter}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the reviewed runtime commit}"

STATE_DIR="$(realpath "$1")"
TRAJECTORY_DIR="$(realpath "$2")"
RCSP_DIR="$(realpath "$3")"
PARAMETER_REPORT="$(realpath "$4")"
RUN_DIR="$(realpath "$5")"
RESULT_DIR="$RUN_DIR/result"

test -x "$PY" || { echo "[FATAL] PY is not executable: $PY" >&2; exit 2; }
test -d "$RUN_DIR" || { echo "[FATAL] RUN_DIR must already exist: $RUN_DIR" >&2; exit 2; }
test ! -e "$RESULT_DIR" || { echo "[FATAL] Refusing to overwrite: $RESULT_DIR" >&2; exit 2; }
test "$(git branch --show-current)" = main || { echo "[FATAL] Runtime branch must be main" >&2; exit 2; }
test -z "$(git status --porcelain)" || { echo "[FATAL] Repository must be clean" >&2; exit 2; }
git fetch origin main
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || { echo "[FATAL] Wrong HEAD" >&2; exit 2; }
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT" || { echo "[FATAL] origin/main mismatch" >&2; exit 2; }

for name in diagnostic_report.json diagnostic_state.pt fit_bank.pt probe_bank.pt; do
  test -s "$STATE_DIR/$name" || { echo "[FATAL] Missing frozen source artifact: $name" >&2; exit 2; }
done
for name in report.json experiment.json diagnostic_latest.pt updates.jsonl; do
  test -s "$TRAJECTORY_DIR/$name" || { echo "[FATAL] Missing trajectory artifact: $name" >&2; exit 2; }
done
for name in report.json reporting_logic_review_v1.json; do
  test -s "$RCSP_DIR/$name" || { echo "[FATAL] Missing RCSP artifact: $name" >&2; exit 2; }
done
test -s "$PARAMETER_REPORT" || { echo "[FATAL] Missing parameter attribution report" >&2; exit 2; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[AUDIT ONLY] Decompose all 64 frozen RCSP final actions by authoritative anatomy and active-frame thirds."
echo "[GRADIENT] Production temporal scientific deficit with respect to detached raw 75D geometry only."
echo "[FORBIDDEN] No optimizer, update, retraining, selection, new head, width normalization, production edit or Pilot."

"$PY" -m training.refiner_single_direction_decomposition_audit \
  --state-dir "$STATE_DIR" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --rcsp-dir "$RCSP_DIR" \
  --parameter-attribution-report "$PARAMETER_REPORT" \
  --output-dir "$RESULT_DIR" \
  --device cuda \
  --expected-main-commit "$EXPECTED_COMMIT" \
  --expected-trajectory-commit b2d71e1fa92cb2a6723810060722c0edea7a3a99 \
  --legacy-core-strength 0.02 \
  --legacy-transition-strength 1.0 \
  2>&1 | tee "$RUN_DIR/console.log"

echo "[STOP] Inspect decomposition summaries only. Pilot remains forbidden."
