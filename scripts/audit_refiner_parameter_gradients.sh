#!/usr/bin/env bash
# Frozen V15.4.1 layer audit, never training or an initialization intervention.
set -Eeuo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/audit_refiner_parameter_gradients.sh FROZEN_V15_4_1_DIAGNOSTIC_DIR NEW_REPORT_JSON" >&2
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
for name in diagnostic_report.json diagnostic_state.pt fit_bank.pt; do
  test -s "$1/$name" || { echo "[FATAL] Missing frozen artifact: $1/$name" >&2; exit 2; }
done
test ! -e "$2" || { echo "[FATAL] Refusing to overwrite audit output: $2" >&2; exit 2; }
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "[AUDIT ONLY] Transaction 0, 192 TRAIN cases. One shared forward; no optimizer or probe."
echo "[LEGACY LIMIT] Explicit decoder strengths: core=0.02 transition=1.0; not recorded by old source."
"$PY" -m training.refiner_parameter_gradient_audit \
  --state-dir "$1" \
  --expected-source-commit 6e73e0eda9f349d3a611864f4719b22807ee5952 \
  --transaction-index 0 --device cuda \
  --legacy-core-strength 0.02 --legacy-transition-strength 1.0 \
  --output "$2"
echo "[STOP] Pilot remains forbidden. Read the layer summary above and the full JSON."
