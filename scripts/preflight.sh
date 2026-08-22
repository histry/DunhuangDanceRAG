#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

cd "$ROOT"

if [[ "${EXPERIMENT_CONFIG_LOADED:-0}" != "1" ]]; then
  source configs/experiment.env
fi

PY="${GENERATION_PYTHON:-${PYTHON_BIN:-python}}"

if [[ "$PY" != */* ]]; then
  PY="$(command -v "$PY")"
fi

SOURCE_MODE="${
  RETARGET_CLEAN_SOURCE_MODE:-chang_e_official_smpl
}"

SOURCE_DIR="${1:-}"
AUDIO_ARG="${2:-${AUDIO:-}}"
MUSIC_ARG="${3:-${MUSIC_DIRS:-}}"

OUT="${PREFLIGHT_OUT:-$ROOT/output/preflight/report.json}"

mkdir -p "$(dirname "$OUT")"

[[ "$SOURCE_MODE" == "chang_e_official_smpl" ]] || {
  echo "[FATAL] main supports only chang_e_official_smpl" >&2; exit 2;
}
SOURCE_DIR="${SOURCE_DIR:-$CHANG_E_OFFICIAL_SMPL_DIR}"
"$PY" evaluation/preflight_official_smpl.py \
  --root "$ROOT" \
  --audio "$AUDIO_ARG" \
  --music_dir "$MUSIC_ARG" \
  --smpl_dir "$SOURCE_DIR" \
  --smpl_manifest "$CHANG_E_OFFICIAL_SMPL_MANIFEST" \
  --out "$OUT"
