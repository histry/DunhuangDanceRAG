#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
SMPL_DIR="${1:-${CHANG_E_OFFICIAL_SMPL_DIR:-}}"
AUDIO_ARG="${2:-${AUDIO:-}}"

[[ -d "$REPO" ]] || { echo "[FATAL] repo missing: $REPO" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "[FATAL] python missing: $PY" >&2; exit 2; }
[[ -n "$SMPL_DIR" && -d "$SMPL_DIR" ]] || {
  echo "Usage: $0 /path/to/Chang-E-official-SMPL [audio.wav]" >&2
  exit 2
}

cd "$REPO"
export GENERATION_PYTHON="$PY"
export PYTHON_BIN="$PY"
export RETARGET_CLEAN_SOURCE_MODE="chang_e_official_smpl"
export CHANG_E_OFFICIAL_SMPL_DIR="$(realpath "$SMPL_DIR")"
export CHANG_E_SOURCE_MANIFEST="${CHANG_E_SOURCE_MANIFEST:-$REPO/assets/motion/bvh/sources.json}"

# This shell override was previously shown to break one activity unit test.
# It is not part of the formal final-generation contract.
unset MOTION_ACTIVITY_FINAL_MIN_JOINT_RADPS || true

if [[ -n "$AUDIO_ARG" ]]; then
  exec bash run.sh "$(realpath "$AUDIO_ARG")"
else
  exec bash run.sh
fi
