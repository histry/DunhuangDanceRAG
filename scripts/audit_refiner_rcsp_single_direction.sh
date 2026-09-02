#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 STATE_DIR TRAJECTORY_DIR RCSP_RESULT_DIR RUN_DIR" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PY:?Set PY to the exact Python interpreter}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the reviewed runtime commit}"

STATE_DIR="$(realpath "$1")"
TRAJECTORY_DIR="$(realpath "$2")"
RCSP_DIR="$(realpath "$3")"
RUN_DIR="$(realpath "$4")"
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
for name in report.json adapter_final.pt reporting_logic_review_v1.json; do
  test -s "$RCSP_DIR/$name" || { echo "[FATAL] Missing completed RCSP artifact: $name" >&2; exit 2; }
done

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[AUDIT ONLY] Attribute single/cross role-head temporal parameter gradients at the frozen RCSP final state."
echo "[FIXED] TRAIN transaction 0 plus fixed seen/new-position groups; no case, checkpoint or scale selection."
echo "[FORBIDDEN] No optimizer, update, gradient surgery, width head, production edit, publishing or Pilot."

"$PY" -m training.refiner_rcsp_single_direction_attribution \
  --state-dir "$STATE_DIR" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --rcsp-dir "$RCSP_DIR" \
  --output-dir "$RESULT_DIR" \
  --device cuda \
  --expected-main-commit "$EXPECTED_COMMIT" \
  --expected-rcsp-commit 5a344f2950183ceb4c8e938a3c26fa5d76a78c3f \
  --expected-trajectory-commit b2d71e1fa92cb2a6723810060722c0edea7a3a99 \
  --legacy-core-strength 0.02 \
  --legacy-transition-strength 1.0 \
  2>&1 | tee "$RUN_DIR/console.log"

echo "[STOP] Interpret parameter-gradient signs and train/final alignment. Pilot remains forbidden."
