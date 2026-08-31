#!/usr/bin/env bash
# A0 zero versus A1 Gaussian (std=1e-5), 400 TRAIN updates each. Never Pilot.
set -Eeuo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/diagnose_refiner_safe_start.sh FROZEN_V15_4_1_DIR NEW_OUTPUT_DIR" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PY:-python}"
if [[ -z "${EXPECTED_COMMIT:-}" || "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "[FATAL] EXPECTED_COMMIT must match the reviewed runtime commit" >&2
  exit 2
fi
test -z "$(git status --porcelain)" || { echo "[FATAL] Repository must be clean" >&2; exit 2; }
for name in diagnostic_report.json diagnostic_state.pt fit_bank.pt probe_bank.pt; do
  test -s "$1/$name" || { echo "[FATAL] Missing frozen artifact: $1/$name" >&2; exit 2; }
done
test ! -e "$2" || { echo "[FATAL] Refusing to overwrite output: $2" >&2; exit 2; }
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "[DIAGNOSTIC ONLY] Fresh paired trunks; sigma=0 vs 1e-5. No architecture, loss or LR changes."
echo "[PREFLIGHT] All TRAIN banks must pass initial physical/fidelity/clean safety before either arm trains."
echo "[BUDGET] 400 updates per arm, 192 TRAIN cases/update. Probe opens only after BOTH arms finish."
echo "[LEGACY LIMIT] Explicit decoder strengths core=0.02, transition=1.0; not recorded by old source."
"$PY" -m training.refiner_safe_start_diagnostics \
  --state-dir "$1" --out-dir "$2" --device cuda \
  --legacy-core-strength 0.02 --legacy-transition-strength 1.0
echo "[STOP] Execution completion is not Scientific PASS. Pilot and publication remain forbidden."
