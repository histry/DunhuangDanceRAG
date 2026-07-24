#!/usr/bin/env bash
# One-command V46.53.1 full rebuild, retraining and current-WAV whole-song generation.
set -Eeuo pipefail

# Ensure top-level research packages are importable.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
[[ -f "$ROOT_DIR/configs/performer_policy.env" ]] && \
  source "$ROOT_DIR/configs/performer_policy.env"
[[ -f "$ROOT_DIR/configs/research_feasibility.env" ]] && source "$ROOT_DIR/configs/research_feasibility.env"

cd "$ROOT_DIR"

export ROOT_DIR
export V46_51_PYTHON="${V46_51_PYTHON:-${PYTHON_BIN:-python}}"
export CHANGE_BVH_DIR="${CHANGE_BVH_DIR:-$ROOT_DIR/change}"
export MUSIC_DIRS="${MUSIC_DIRS:-$ROOT_DIR/data/v21_router_music_999/splits/train}"
export AUDIO="${1:-${AUDIO:-$ROOT_DIR/test_music_bank/dunhuangwu2.wav}}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/output/v46_53_1_research_${RUN_TAG}}"
export V44_CKPT="${V44_CKPT:-$OUT_ROOT/checkpoints/semantic_retriever.pt}"
export V45_CKPT="${V45_CKPT:-$OUT_ROOT/checkpoints/boundary_refiner.pt}"
export V46_CKPT="${V46_CKPT:-$OUT_ROOT/checkpoints/local_diffusion.pt}"
export V46_53_GROUNDER_CKPT="${V46_53_GROUNDER_CKPT:-$OUT_ROOT/checkpoints/grounder.pt}"
# Historical music-domain knowledge is transferred only into the formal
# Router's music branch.  The Event-DB-specific motion branch is retrained.
export V46_54_MUSIC_ENCODER_PRIOR_CKPT="${V46_54_MUSIC_ENCODER_PRIOR_CKPT:-${MUSIC_ROUTER_WEIGHT:-$ROOT_DIR/assets/weights/music/router.pt}}"
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
