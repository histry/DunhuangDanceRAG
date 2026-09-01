#!/usr/bin/env bash
# Fresh A0 exact-zero trajectory, 400 checked TRAIN updates. Never Pilot.
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/diagnose_refiner_zero_start_trajectory.sh FROZEN_V15_4_1_DIR NEW_OUTPUT_DIR" >&2
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
test ! -e "$2" || { echo "[FATAL] Refusing to overwrite output: $2" >&2; exit 2; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "[DIAGNOSTIC TRAINING ONLY] Fresh exact-zero A0; historical weights are provenance only."
echo "[PREFLIGHT] All 2976 TRAIN checks must pass before optimizer construction."
echo "[BUDGET] Exactly 400 checked updates, 192 cases/update, 48/group; no resume or selection."
echo "[PROBE ISOLATION] probe_bank.pt stays unloaded until final step 400 state/hash are fixed."
echo "[LEGACY LIMIT] core=0.02 transition=1.0 are explicit values, not recovered source metadata."

"$PY" -m training.refiner_zero_start_trajectory \
  --state-dir "$1" --out-dir "$2" --device cuda \
  --legacy-core-strength 0.02 --legacy-transition-strength 1.0

echo "[STOP] Completion is descriptive trajectory evidence, not Scientific PASS. Pilot remains forbidden."
