#!/usr/bin/env bash
# Run complete, isolated leave-one-confirmed-theme-out experiments.
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export EXPERIMENT_PROFILE=research
# shellcheck disable=SC1091
source configs/experiment.env

AUDIO="${1:-${AUDIO:-$TEST_AUDIO}}"
PY="${GENERATION_PYTHON:-${PYTHON_BIN:-python}}"
BASE_OUT="${LOTO_OUT_ROOT:-$ROOT_DIR/outputs/leave_one_theme_out}"
shift || true

if [[ $# -gt 0 ]]; then
  THEMES=("$@")
else
  mapfile -t THEMES < <(
    "$PY" - "$CHANG_E_OFFICIAL_SMPL_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("\n".join(sorted({
    str(row["dance_category"])
    for row in payload["sources"]
    if row.get("theme_label_status") == "confirmed"
    and bool(row.get("solo_compatible", False))
    and row.get("dance_category") not in {None, "", "unknown"}
})))
PY
  )
fi

[[ -s "$AUDIO" ]] || { echo "[FATAL] Input audio missing: $AUDIO" >&2; exit 2; }
[[ ${#THEMES[@]} -gt 0 ]] || { echo "[FATAL] No confirmed themes found" >&2; exit 2; }

for theme in "${THEMES[@]}"; do
  echo "========== LOTO FULL EXPERIMENT: $theme =========="
  RUN_TAG="loto_${theme}_$(date +%Y%m%d_%H%M%S)" \
  OUT_ROOT="$BASE_OUT/$theme" \
  GENERATION_SPLIT_PROTOCOL="leave_one_theme_out" \
  GENERATION_HELDOUT_THEME="$theme" \
  EXPERIMENT_CONFIG_LOADED=1 \
    bash scripts/run_experiment.sh "$AUDIO"
done
