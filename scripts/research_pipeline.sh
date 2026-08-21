#!/usr/bin/env bash
# Retarget Clean direct replacement: repaired retarget/source-split/event contracts,
# followed by the preserved Fresh-Audio Generation/Anatomy-Heading/Geometry-Aware Routing training and generation stack.
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

[[ -f configs/experiment.env ]] || {
  echo "[FATAL] Missing configs/experiment.env" >&2; exit 2;
}
[[ -f scripts/pipeline.sh ]] || {
  echo "[FATAL] Missing preserved Fresh-Audio Generation base launcher" >&2; exit 2;
}

# experiment.env is the only public configuration entry. Nested launchers skip
# reloading it through EXPERIMENT_CONFIG_LOADED.
if [[ "${EXPERIMENT_CONFIG_LOADED:-0}" != "1" ]]; then
  # shellcheck disable=SC1091
  source configs/experiment.env
fi

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  export AUDIO="$(realpath "$1")"
fi
: "${AUDIO:?Set AUDIO or pass the current music file as argument 1}"
: "${MUSIC_DIRS:?Set MUSIC_DIRS to non-test training music directories}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/output/retarget_clean_research_${RUN_TAG}}"
export OUT_ROOT
export ROOT_DIR
export CHANGE_BVH_DIR="${CHANGE_BVH_DIR:-$ROOT_DIR/change}"
: "${CHANG_E_OFFICIAL_SMPL_DIR:?configs/experiment.env must define the official SMPL directory}"
: "${CHANG_E_OFFICIAL_SMPL_MANIFEST:?configs/experiment.env must define the official SMPL manifest}"
export RETARGET_CLEAN_SOURCE_MODE="${RETARGET_CLEAN_SOURCE_MODE:-chang_e_official_smpl}"
export GROUNDING_GROUNDER_CKPT="${GROUNDING_GROUNDER_CKPT:-$OUT_ROOT/event_geometry_dual_branch_grounder.pt}"
PY="${GENERATION_PYTHON:-${PYTHON_BIN:-python}}"
export GENERATION_PYTHON="$PY"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PY" ]] || { echo "[FATAL] Python not executable: $PY" >&2; exit 2; }
mkdir -p "$OUT_ROOT/preflight"

cat <<EOF
========== Retarget Clean RESEARCH CONTRACT REPAIR ==========
ROOT_DIR=$ROOT_DIR
OUT_ROOT=$OUT_ROOT
AUDIO=$AUDIO
MUSIC_DIRS=$MUSIC_DIRS
SOURCE_MODE=$RETARGET_CLEAN_SOURCE_MODE
CHANGE_BVH_DIR=$CHANGE_BVH_DIR
CHANG_E_OFFICIAL_SMPL_DIR=$CHANG_E_OFFICIAL_SMPL_DIR
MIN_OK_SOURCES=$RETARGET_MIN_OK_SOURCES
SPLIT=$GENERATION_TRAIN_RATIO/$GENERATION_VAL_RATIO/$GENERATION_TEST_RATIO
REBUILD_RETARGET=$GENERATION_REBUILD_RETARGET_CACHE
REBUILD_DB=$GENERATION_REBUILD_EVENT_DB
RETRAIN_ROUTER/DURATION/PLANNER=$GENERATION_RETRAIN_ROUTER/$GENERATION_RETRAIN_DURATION/$GENERATION_RETRAIN_PLANNER
RETRAIN_CONTRASTIVE/REFINER/DIFFUSION=$GENERATION_RETRAIN_CONTRASTIVE/$GENERATION_RETRAIN_REFINER/$GENERATION_RETRAIN_DIFFUSION
GROUNDER_CKPT=$GROUNDING_GROUNDER_CKPT
========================================================
EOF

echo "========== 0A. REAL-DATA PREFLIGHT =========="
case "$RETARGET_CLEAN_SOURCE_MODE" in
  chang_e_official_smpl)
    "$PY" evaluation/preflight_official_smpl.py \
      --root "$ROOT_DIR" \
      --audio "$AUDIO" \
      --music_dir "$MUSIC_DIRS" \
      --smpl_dir "$CHANG_E_OFFICIAL_SMPL_DIR" \
      --smpl_manifest "$CHANG_E_OFFICIAL_SMPL_MANIFEST" \
      --out "$OUT_ROOT/preflight/preflight.json"
    ;;
  bvh_retarget)
    "$PY" evaluation/preflight.py \
      --root "$ROOT_DIR" \
      --audio "$AUDIO" \
      --music_dir "$MUSIC_DIRS" \
      --change_dir "$CHANGE_BVH_DIR" \
      --out "$OUT_ROOT/preflight/preflight.json"
    ;;
  *)
    echo "[FATAL] Unknown RETARGET_CLEAN_SOURCE_MODE=$RETARGET_CLEAN_SOURCE_MODE" >&2
    exit 2
    ;;
esac

echo "========== 0B. Geometry-Aware Routing + Retarget Clean CONTRACT TESTS =========="
"$PY" -m unittest discover -s tests -p 'test_*.py' -v

echo "========== 1. PRESERVED TRAINING/GENERATION PIPELINE =========="
bash scripts/pipeline.sh

FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/fresh_audio_final.npy}"
[[ -s "$FINAL_NPY" ]] || { echo "[FATAL] Final motion missing: $FINAL_NPY" >&2; exit 2; }

echo "========== 2. FINAL POSTURE-AWARE ANATOMY AUDIT =========="
"$PY" evaluation/audit_motion.py \
  --input "$FINAL_NPY" \
  --fps "${GENERATION_FPS:-30}" \
  --out "$OUT_ROOT/final.retarget_clean_anatomy.json" \
  --csv "$OUT_ROOT/final.retarget_clean_anatomy.csv"

echo "========== 3. FINAL INTRINSIC MOTION AUDIT =========="
"$PY" contracts/boundary.py \
  --input "$FINAL_NPY" \
  --out "$OUT_ROOT/final.event_geometry_intrinsic.json" \
  --fps "${GENERATION_FPS:-30}"

cat <<EOF
========== Retarget Clean COMPLETE ==========
FINAL_NPY=$FINAL_NPY
ANATOMY_AUDIT=$OUT_ROOT/final.retarget_clean_anatomy.json
INTRINSIC_AUDIT=$OUT_ROOT/final.event_geometry_intrinsic.json
DURATION_AUDIT=$FINAL_NPY.event_geometry_duration.json
GROUNDER_CKPT=$GROUNDING_GROUNDER_CKPT
========================================
EOF
