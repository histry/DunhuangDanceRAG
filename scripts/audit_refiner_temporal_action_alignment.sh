#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 STATE_DIR TRAJECTORY_DIR OUTPUT_DIR" >&2
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

for name in diagnostic_report.json diagnostic_state.pt fit_bank.pt probe_bank.pt; do
  test -s "$1/$name" || { echo "[FATAL] Missing frozen source artifact: $1/$name" >&2; exit 2; }
done
for name in report.json experiment.json diagnostic_latest.pt updates.jsonl; do
  test -s "$2/$name" || { echo "[FATAL] Missing fixed trajectory artifact: $2/$name" >&2; exit 2; }
done
test ! -e "$3" || { echo "[FATAL] Refusing to overwrite output: $3" >&2; exit 2; }
mkdir -p "$3"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[AUDIT ONLY] Fixed A0 checkpoint; one full 192-case TRAIN transaction and fixed 64-case final."
echo "[GRADIENT POINTS] Exact zero origin and current model output through the identical production decoder."
echo "[PROVENANCE] Source, probe, trajectory, checkpoint and model-state hashes are verified fail closed."
echo "[FORBIDDEN] No optimizer, update, checkpoint selection, scale search, architecture change or Pilot."

"$PY" -m training.refiner_temporal_action_alignment_audit \
  --state-dir "$1" \
  --trajectory-dir "$2" \
  --output "$3/report.json" \
  --device cuda \
  --expected-main-commit "$EXPECTED_COMMIT" \
  --expected-trajectory-commit b2d71e1fa92cb2a6723810060722c0edea7a3a99 \
  --legacy-core-strength 0.02 \
  --legacy-transition-strength 1.0 \
  2>&1 | tee "$3/console.log"

echo "[STOP] Alignment is attribution evidence only. Pilot remains forbidden."
