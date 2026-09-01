#!/usr/bin/env bash
# Read-only fixed-final failure attribution and contact connectivity. Never Pilot.
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/audit_refiner_final_failure.sh FROZEN_V15_4_1_DIR ZERO_START_TRAJECTORY_DIR NEW_OUTPUT_DIR" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PY:-python}"

if [[ -z "${EXPECTED_COMMIT:-}" || "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "[FATAL] EXPECTED_COMMIT must match the reviewed runtime commit" >&2
  exit 2
fi
test "$(git branch --show-current)" = main || { echo "[FATAL] Runtime branch must be main" >&2; exit 2; }
test -z "$(git status --porcelain)" || { echo "[FATAL] Repository must be clean" >&2; exit 2; }
for name in diagnostic_report.json diagnostic_state.pt fit_bank.pt probe_bank.pt; do
  test -s "$1/$name" || { echo "[FATAL] Missing frozen artifact: $1/$name" >&2; exit 2; }
done
for name in report.json experiment.json diagnostic_latest.pt updates.jsonl; do
  test -s "$2/$name" || { echo "[FATAL] Missing trajectory artifact: $2/$name" >&2; exit 2; }
done
test ! -e "$3" || { echo "[FATAL] Refusing to overwrite output: $3" >&2; exit 2; }
mkdir -p "$3"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "[AUDIT ONLY] Fixed final A0 state, TRAIN transaction 0 and immutable final probe; no optimizer."
echo "[PROVENANCE] Trajectory commit/hash, experiment, source and probe are verified fail closed."
echo "[CONTACT] Measure masks, true objective gradients and actual decoder VJPs; no mask/objective change."
echo "[LEGACY LIMIT] core=0.02 transition=1.0 are explicit values, not recovered source metadata."

"$PY" -m training.refiner_final_failure_audit \
  --state-dir "$1" --trajectory-dir "$2" --output "$3/report.json" \
  --device cuda --expected-trajectory-commit \
  b2d71e1fa92cb2a6723810060722c0edea7a3a99 \
  --legacy-core-strength 0.02 --legacy-transition-strength 1.0 \
  2>&1 | tee "$3/console.log"

echo "[STOP] This is attribution evidence only. No training, checkpoint promotion or Pilot is authorized."
