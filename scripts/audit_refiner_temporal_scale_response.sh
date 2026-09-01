#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 STATE_DIR TRAJECTORY_DIR ALIGNMENT_REPORT OUTPUT_DIR" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PY:?Set PY to the exact Python interpreter}"
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the reviewed runtime commit}"

test -x "$PY" || { echo "[FATAL] PY is not executable: $PY" >&2; exit 2; }
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
  test -s "$1/$name" || { echo "[FATAL] Missing frozen source artifact: $1/$name" >&2; exit 2; }
done
for name in report.json experiment.json diagnostic_latest.pt updates.jsonl; do
  test -s "$2/$name" || { echo "[FATAL] Missing fixed trajectory artifact: $2/$name" >&2; exit 2; }
done
test -s "$3" || { echo "[FATAL] Missing alignment report: $3" >&2; exit 2; }
test ! -e "$4" || { echo "[FATAL] Refusing to overwrite output: $4" >&2; exit 2; }
mkdir -p "$4"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[AUDIT ONLY] Fixed model/action and preregistered alpha grid: 0,.5,.75,1,1.25,1.5,2."
echo "[DECODER] Scale raw 75D geometry only; contact stays unchanged; use the production decoder."
echo "[LINEAGE] Source, trajectory, probe and reviewed alignment report are verified fail closed."
echo "[FORBIDDEN] No optimizer, update, scale selection, decoder change, training or Pilot."

"$PY" -m training.refiner_temporal_scale_response_audit \
  --state-dir "$1" \
  --trajectory-dir "$2" \
  --alignment-report "$3" \
  --output "$4/report.json" \
  --device cuda \
  --expected-main-commit "$EXPECTED_COMMIT" \
  --expected-trajectory-commit b2d71e1fa92cb2a6723810060722c0edea7a3a99 \
  --legacy-core-strength 0.02 \
  --legacy-transition-strength 1.0 \
  2>&1 | tee "$4/console.log"

echo "[STOP] Fixed-grid response evidence only. No alpha is selected and Pilot remains forbidden."
