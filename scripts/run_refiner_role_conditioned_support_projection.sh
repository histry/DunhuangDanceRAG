#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 STATE_DIR TRAJECTORY_DIR RUN_DIR" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PY:?Set PY to the exact Python interpreter}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the reviewed runtime commit}"

STATE_DIR="$(realpath "$1")"
TRAJECTORY_DIR="$(realpath "$2")"
RUN_DIR="$(realpath "$3")"
RESULT_DIR="$RUN_DIR/result"

test -x "$PY" || { echo "[FATAL] PY is not executable: $PY" >&2; exit 2; }
test -d "$RUN_DIR" || { echo "[FATAL] RUN_DIR must already exist: $RUN_DIR" >&2; exit 2; }
test ! -e "$RESULT_DIR" || { echo "[FATAL] Refusing to overwrite: $RESULT_DIR" >&2; exit 2; }
test "$(git branch --show-current)" = main || { echo "[FATAL] Runtime branch must be main" >&2; exit 2; }
test -z "$(git status --porcelain)" || { echo "[FATAL] Repository must be clean" >&2; exit 2; }
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || {
  echo "[FATAL] HEAD does not match EXPECTED_COMMIT" >&2
  exit 2
}
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT" || {
  echo "[FATAL] origin/main does not match EXPECTED_COMMIT" >&2
  exit 2
}

for name in diagnostic_report.json diagnostic_state.pt fit_bank.pt probe_bank.pt; do
  test -s "$STATE_DIR/$name" || { echo "[FATAL] Missing frozen source artifact: $STATE_DIR/$name" >&2; exit 2; }
done
for name in report.json experiment.json diagnostic_latest.pt updates.jsonl; do
  test -s "$TRAJECTORY_DIR/$name" || { echo "[FATAL] Missing fixed A0 trajectory artifact: $TRAJECTORY_DIR/$name" >&2; exit 2; }
done

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[DIAGNOSTIC ONLY] Frozen step-400 A0 base; train two zero-init role adapters for exactly 400 checked steps."
echo "[SUPPORT] Binary projection from production root/joint support; production soft confidence is applied once."
echo "[FIXED] Same TRAIN reservoir, transaction schedule, optimizer, objective and final 64 cases; alpha=1 only."
echo "[FORBIDDEN] No production edit, width conditioning, alpha sweep, checkpoint selection, publishing or Pilot."

"$PY" -m training.refiner_role_conditioned_support_projection_experiment \
  --state-dir "$STATE_DIR" \
  --trajectory-dir "$TRAJECTORY_DIR" \
  --output-dir "$RESULT_DIR" \
  --device cuda \
  --expected-main-commit "$EXPECTED_COMMIT" \
  --expected-trajectory-commit b2d71e1fa92cb2a6723810060722c0edea7a3a99 \
  --legacy-core-strength 0.02 \
  --legacy-transition-strength 1.0 \
  2>&1 | tee "$RUN_DIR/console.log"

echo "[STOP] Inspect result/report.json. Scientific acceptance and Pilot remain false."
