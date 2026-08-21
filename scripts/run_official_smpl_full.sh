#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"

REPO="${REPO:-$ROOT_DIR}"

PY="${
  PY:-${GENERATION_PYTHON:-${PYTHON_BIN:-python}}
}"

if [[ "$PY" != */* ]]; then
  PY="$(command -v "$PY")"
fi

SMPL_DIR="${
  1:-${CHANG_E_OFFICIAL_SMPL_DIR:-$REPO/assets/motion/smpl_official_12}
}"

AUDIO_ARG="${
  2:-${AUDIO:-}
}"

[[ -d "$REPO" ]] || {
  echo "[FATAL] repo missing: $REPO" >&2
  exit 2
}

[[ -x "$PY" ]] || {
  echo "[FATAL] python missing: $PY" >&2
  exit 2
}

[[ -d "$SMPL_DIR" ]] || {
  echo "[FATAL] Official SMPL directory missing: $SMPL_DIR" >&2
  exit 2
}

SMPL_DIR="$(realpath "$SMPL_DIR")"

SMPL_MANIFEST="${
  CHANG_E_OFFICIAL_SMPL_MANIFEST:-$SMPL_DIR/sources.json
}"

[[ -s "$SMPL_MANIFEST" ]] || {
  echo "[FATAL] Official SMPL manifest missing: $SMPL_MANIFEST" >&2
  echo "Build it first with:" >&2
  echo "  python scripts/build_official_smpl_manifest.py \\" >&2
  echo "    --smpl_dir \"$SMPL_DIR\" \\" >&2
  echo "    --out \"$SMPL_DIR/sources.json\" \\" >&2
  echo "    --source_fps 30 --overwrite" >&2
  exit 2
}

cd "$REPO"

export ROOT_DIR="$REPO"
export GENERATION_PYTHON="$PY"
export PYTHON_BIN="$PY"

export RETARGET_CLEAN_SOURCE_MODE="chang_e_official_smpl"

export CHANG_E_OFFICIAL_SMPL_DIR="$SMPL_DIR"

export CHANG_E_OFFICIAL_SMPL_MANIFEST="$(
  realpath "$SMPL_MANIFEST"
)"

# Never allow a stale interactive-shell activity override
# to contaminate the formal validation path.
unset MOTION_ACTIVITY_FINAL_MIN_JOINT_RADPS || true

if [[ -n "$AUDIO_ARG" ]]; then
  exec bash run.sh "$(realpath "$AUDIO_ARG")"
else
  exec bash run.sh
fi
