#!/usr/bin/env bash
# One-command Retarget Clean full rebuild, retraining and current-WAV whole-song generation.
set -Eeuo pipefail

# Ensure top-level research packages are importable.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"
if [[ "${EXPERIMENT_CONFIG_LOADED:-0}" != "1" ]]; then
  # shellcheck disable=SC1091
  source configs/experiment.env
fi

export ROOT_DIR
export GENERATION_PYTHON="${GENERATION_PYTHON:-${PYTHON_BIN:-python}}"
export MUSIC_DIRS="${MUSIC_DIRS:-$ROOT_DIR/data/music_router_music_999/splits/train}"
export AUDIO="${1:-${AUDIO:-$TEST_AUDIO}}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/output/retarget_clean_research_${RUN_TAG}}"
export REFINER_CKPT="${REFINER_CKPT:-$OUT_ROOT/checkpoints/boundary_refiner.pt}"
export MOTION_CKPT="${MOTION_CKPT:-$OUT_ROOT/checkpoints/local_diffusion.pt}"
export FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/results/motion.npy}"
export FINAL_REPORT="${FINAL_REPORT:-$OUT_ROOT/results/report.json}"
export FINAL_MP4="${FINAL_MP4:-$OUT_ROOT/results/video.mp4}"
mkdir -p "$OUT_ROOT/checkpoints" "$OUT_ROOT/results"

[[ -s "$AUDIO" ]] || { echo "[FATAL] Input audio missing: $AUDIO" >&2; exit 2; }
[[ -d "$MUSIC_DIRS" ]] || { echo "[FATAL] Training music directory missing: $MUSIC_DIRS" >&2; exit 2; }
[[ "$MUSIC_DIRS" != *test_music_bank* ]] || { echo "[FATAL] test_music_bank cannot enter training" >&2; exit 2; }

mkdir -p "$ROOT_DIR/logs"
LOG="$ROOT_DIR/logs/pipeline_${RUN_TAG}.log"
echo "[RUN] AUDIO=$AUDIO"
echo "[RUN] OUT_ROOT=$OUT_ROOT"
echo "[RUN] LOG=$LOG"

bash scripts/research_pipeline.sh "$AUDIO" 2>&1 | tee "$LOG"
