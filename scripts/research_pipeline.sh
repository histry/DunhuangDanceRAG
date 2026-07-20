#!/usr/bin/env bash
# V46.53.1 direct replacement: repaired retarget/source-split/event contracts,
# followed by the preserved V46.51/V46.52/V46.53 training and generation stack.
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT_DIR"

[[ -f configs/scheduler.env ]] || {
  echo "[FATAL] Missing configs/scheduler.env" >&2; exit 2;
}
[[ -f configs/research.env ]] || {
  echo "[FATAL] Missing configs/research.env" >&2; exit 2;
}
[[ -f scripts/pipeline.sh ]] || {
  echo "[FATAL] Missing preserved V46.51 base launcher" >&2; exit 2;
}

# Load legacy profiles first when present, then apply the authoritative V46.53.1
# profile. Repeated sourcing in the preserved base becomes harmless because the
# repaired code reads V46.53.1 values and this profile is exported to child shells.
# shellcheck disable=SC1091
source configs/scheduler.env
[[ -f configs/anatomy.env ]] && source configs/anatomy.env
[[ -f configs/geometry.env ]] && source configs/geometry.env
# shellcheck disable=SC1091
source configs/research.env
[[ -f "$ROOT_DIR/configs/research_feasibility.env" ]] && source "$ROOT_DIR/configs/research_feasibility.env"

[[ -f "$ROOT_DIR/configs/performer_policy.env" ]] && source "$ROOT_DIR/configs/performer_policy.env"

if [[ $# -ge 1 && -n "${1:-}" ]]; then
  export AUDIO="$(realpath "$1")"
fi
: "${AUDIO:?Set AUDIO or pass the current music file as argument 1}"
: "${MUSIC_DIRS:?Set MUSIC_DIRS to non-test training music directories}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/output/v46_53_1_research_${RUN_TAG}}"
export OUT_ROOT
export ROOT_DIR
export CHANGE_BVH_DIR="${CHANGE_BVH_DIR:-$ROOT_DIR/change}"
export V46_53_GROUNDER_CKPT="${V46_53_GROUNDER_CKPT:-$OUT_ROOT/v46_53_dual_branch_grounder.pt}"
PY="${V46_51_PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
export V46_51_PYTHON="$PY"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
[[ -x "$PY" ]] || { echo "[FATAL] Python not executable: $PY" >&2; exit 2; }
mkdir -p "$OUT_ROOT/preflight"

cat <<EOF
========== V46.53.1 RESEARCH CONTRACT REPAIR ==========
ROOT_DIR=$ROOT_DIR
OUT_ROOT=$OUT_ROOT
AUDIO=$AUDIO
MUSIC_DIRS=$MUSIC_DIRS
CHANGE_BVH_DIR=$CHANGE_BVH_DIR
MIN_OK_SOURCES=$V46_52_MIN_OK_SOURCES
SPLIT=$V46_51_TRAIN_RATIO/$V46_51_VAL_RATIO/$V46_51_TEST_RATIO
REBUILD_RETARGET=$V46_51_REBUILD_RETARGET_CACHE
REBUILD_DB=$V46_51_REBUILD_EVENT_DB
RETRAIN_V44/V45/V46=$V46_51_RETRAIN_V44/$V46_51_RETRAIN_V45/$V46_51_RETRAIN_V46
GROUNDER_CKPT=$V46_53_GROUNDER_CKPT
========================================================
EOF

echo "========== 0A. REAL-DATA PREFLIGHT =========="
"$PY" evaluation/preflight.py \
  --root "$ROOT_DIR" \
  --audio "$AUDIO" \
  --music_dir "$MUSIC_DIRS" \
  --change_dir "$CHANGE_BVH_DIR" \
  --out "$OUT_ROOT/preflight/preflight.json"

echo "========== 0B. V46.53 + V46.53.1 CONTRACT TESTS =========="
"$PY" -m unittest discover -s tests -p 'test_*.py' -v

echo "========== 1. PRESERVED TRAINING/GENERATION PIPELINE =========="
bash scripts/pipeline.sh

FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/v46_51_final.npy}"
[[ -s "$FINAL_NPY" ]] || { echo "[FATAL] Final motion missing: $FINAL_NPY" >&2; exit 2; }

echo "========== 2. FINAL POSTURE-AWARE ANATOMY AUDIT =========="
"$PY" evaluation/audit_motion.py \
  --input "$FINAL_NPY" \
  --out "$OUT_ROOT/final.v46_53_1_anatomy.json" \
  --csv "$OUT_ROOT/final.v46_53_1_anatomy.csv"

echo "========== 3. FINAL INTRINSIC MOTION AUDIT =========="
"$PY" contracts/boundary.py \
  --input "$FINAL_NPY" \
  --out "$OUT_ROOT/final.v46_53_intrinsic.json" \
  --fps "${V46_51_FPS:-30}"

cat <<EOF
========== V46.53.1 COMPLETE ==========
FINAL_NPY=$FINAL_NPY
ANATOMY_AUDIT=$OUT_ROOT/final.v46_53_1_anatomy.json
INTRINSIC_AUDIT=$OUT_ROOT/final.v46_53_intrinsic.json
DURATION_AUDIT=$FINAL_NPY.v46_53_duration.json
GROUNDER_CKPT=$V46_53_GROUNDER_CKPT
========================================
EOF
