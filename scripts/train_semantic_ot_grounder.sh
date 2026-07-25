#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

source configs/scheduler.env
source configs/anatomy.env
source configs/semantic_ot.env

PY="${PY:-${V46_51_PYTHON:-python}}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT must identify the current audited run}"
DB_SPLIT_ROOT="${DB_SPLIT_ROOT:?DB_SPLIT_ROOT must contain train/val/test events.npz}"
MUSIC_ENCODER_PRIOR_CKPT="${MUSIC_ENCODER_PRIOR_CKPT:-${V46_54_MUSIC_ENCODER_PRIOR_CKPT:-${V46_51_ROUTER_CKPT:-}}}"
if [[ "$PY" == */* ]]; then
  [[ -x "$PY" ]] || { echo "[FATAL] Python is not executable: $PY" >&2; exit 2; }
else
  command -v "$PY" >/dev/null 2>&1 || { echo "[FATAL] Python command not found: $PY" >&2; exit 2; }
fi
[[ -s "$MUSIC_ENCODER_PRIOR_CKPT" ]] || {
  echo "[FATAL] Historical Router music prior is missing: $MUSIC_ENCODER_PRIOR_CKPT" >&2
  exit 2
}

read -r -a MUSIC_DIR_ARRAY <<< "${MUSIC_DIRS:?MUSIC_DIRS is required}"
for directory in "${MUSIC_DIR_ARRAY[@]}"; do
  if [[ "$directory" == *"classical_eval"* || "$directory" == *"test_music_bank"* || "$directory" == *"assets/music/test"* ]]; then
    echo "[FATAL] Evaluation/test music cannot enter teacher or OT training: $directory" >&2
    exit 2
  fi
done

SEM_ROOT="$OUT_ROOT/semantic_ot"
MUSIC_SPLIT_ROOT="$SEM_ROOT/music_splits"
TEACHER_ROOT="$SEM_ROOT/music_teacher"
MSSD_ROOT="$SEM_ROOT/mssd"
OT_ROOT="$SEM_ROOT/datasets"
CHECKPOINT_ROOT="$OUT_ROOT/checkpoints"
AESD_ROOT="$SEM_ROOT/aesd_event_db_split"
EMBED_ROOT="$SEM_ROOT/embedded_event_db_split"
CACHE_ROOT="$SEM_ROOT/cache"
mkdir -p "$MUSIC_SPLIT_ROOT" "$TEACHER_ROOT" "$MSSD_ROOT" "$OT_ROOT" "$CHECKPOINT_ROOT" "$AESD_ROOT" "$EMBED_ROOT" "$CACHE_ROOT"

TEACHER_DATA="$TEACHER_ROOT/music_semantic_teacher.npz"
TEACHER_CKPT="$CHECKPOINT_ROOT/music_semantic_teacher.pt"
GROUNDER_CKPT="$CHECKPOINT_ROOT/semantic_ot_mixed_grounder.pt"

printf 'ROOT_DIR=%s\nOUT_ROOT=%s\nDB_SPLIT_ROOT=%s\nMUSIC_PRIOR=%s\n' \
  "$ROOT_DIR" "$OUT_ROOT" "$DB_SPLIT_ROOT" "$MUSIC_ENCODER_PRIOR_CKPT"

# 1. Calibrated AESD in an isolated copy of each source-disjoint motion split.
#    Authoritative DB_SPLIT_ROOT files are never overwritten.
for split in train val test; do
  input="$DB_SPLIT_ROOT/$split/events.npz"
  copy="$AESD_ROOT/$split/events.npz"
  output="$AESD_ROOT/$split/events_aesd.npz"
  [[ -s "$input" ]] || { echo "[FATAL] Missing $input" >&2; exit 2; }
  mkdir -p "$AESD_ROOT/$split"
  cp -a "$input" "$copy"
  "$PY" events/build_semantics.py \
    --db "$copy" \
    --out "$output" \
    --json "$SEM_ROOT/${split}.aesd.json" \
    --prior_alpha "$SEMANTIC_OT_PRIOR_ALPHA" \
    --ambiguity_margin "$SEMANTIC_OT_AMBIGUITY_MARGIN" \
    --intrinsic_low_threshold "$SEMANTIC_OT_INTRINSIC_LOW" \
    --intrinsic_high_threshold "$SEMANTIC_OT_INTRINSIC_HIGH"
done

# 2. Deterministic whole-song split before teacher inference and before OT.
"$PY" data_pipeline/split_music_corpus.py \
  --music_dirs "${MUSIC_DIR_ARRAY[@]}" \
  --out_dir "$MUSIC_SPLIT_ROOT" \
  --train_ratio "$SEMANTIC_OT_MUSIC_TRAIN_RATIO" \
  --validation_ratio "$SEMANTIC_OT_MUSIC_VALIDATION_RATIO" \
  --seed "$SEMANTIC_OT_SEED"

# 3. Import only the historical music_encoder and train an 8-class weak teacher.
"$PY" training/music_semantic_teacher.py build-dataset \
  --music_manifest "$MUSIC_SPLIT_ROOT/music_train.json" \
  --cache_dir "$CACHE_ROOT/teacher_features" \
  --out "$TEACHER_DATA" \
  --num_frames "$SEMANTIC_OT_TEACHER_FRAMES" \
  --min_phrase_seconds "$SEMANTIC_OT_MIN_PHRASE_SECONDS" \
  --max_phrase_seconds "$SEMANTIC_OT_MAX_PHRASE_SECONDS" \
  --boundary_quantile "$SEMANTIC_OT_BOUNDARY_QUANTILE" \
  --beat_snap_seconds "$SEMANTIC_OT_BEAT_SNAP_SECONDS" \
  --weak_prior_alpha "$SEMANTIC_OT_MUSIC_PRIOR_ALPHA"

"$PY" training/music_semantic_teacher.py train \
  --data "$TEACHER_DATA" \
  --music_prior_ckpt "$MUSIC_ENCODER_PRIOR_CKPT" \
  --out "$TEACHER_CKPT" \
  --epochs "$SEMANTIC_OT_TEACHER_EPOCHS" \
  --batch_size "$SEMANTIC_OT_TEACHER_BATCH" \
  --seed "$SEMANTIC_OT_SEED" \
  --freeze_music_encoder "$SEMANTIC_OT_FREEZE_MUSIC_ENCODER"

# 4. Export train/validation/test MSSD in separate directories.
for split in train validation test; do
  mkdir -p "$MSSD_ROOT/$split"
  "$PY" training/music_semantic_teacher.py infer \
    --checkpoint "$TEACHER_CKPT" \
    --music_manifest "$MUSIC_SPLIT_ROOT/music_${split}.json" \
    --cache_dir "$CACHE_ROOT/teacher_features" \
    --out_dir "$MSSD_ROOT/$split" \
    --num_frames "$SEMANTIC_OT_TEACHER_FRAMES" \
    --min_phrase_seconds "$SEMANTIC_OT_MIN_PHRASE_SECONDS" \
    --max_phrase_seconds "$SEMANTIC_OT_MAX_PHRASE_SECONDS" \
    --boundary_quantile "$SEMANTIC_OT_BOUNDARY_QUANTILE" \
    --beat_snap_seconds "$SEMANTIC_OT_BEAT_SNAP_SECONDS" \
    --fps "$V46_51_FPS"
done

# 5. Build OT only after both music and motion splits. Validation maps to val DB.
declare -A MOTION_SPLIT=( [train]=train [validation]=val [test]=test )
for music_split in train validation test; do
  motion_split="${MOTION_SPLIT[$music_split]}"
  "$PY" -m grounding.semantic_optimal_transport \
    --event_db "$AESD_ROOT/$motion_split/events_aesd.npz" \
    --mssd_dirs "$MSSD_ROOT/$music_split" \
    --out "$OT_ROOT/semantic_ot_${music_split}.npz" \
    --model_name clap \
    --cache_dir "$CACHE_ROOT/clap" \
    --temporal_frames 64 \
    --temporal_source_frames 2048 \
    --phrase_fps "$V46_51_FPS" \
    --top_k "$SEMANTIC_OT_TOP_K" \
    --preselect_k "$SEMANTIC_OT_PRESELECT_K" \
    --preselect_per_source "$SEMANTIC_OT_PRESELECT_PER_SOURCE" \
    --sinkhorn_epsilon "$SEMANTIC_OT_SINKHORN_EPSILON" \
    --sinkhorn_iterations "$SEMANTIC_OT_SINKHORN_ITERATIONS"
done

# 6. Train with explicit separate train/validation datasets; no identity-graph split.
"$PY" -m grounding.semantic_ot_grounder \
  --train_data "$OT_ROOT/semantic_ot_train.npz" \
  --validation_data "$OT_ROOT/semantic_ot_validation.npz" \
  --out "$GROUNDER_CKPT" \
  --epochs "$SEMANTIC_OT_GROUNDER_EPOCHS" \
  --batch_phrases "$SEMANTIC_OT_BATCH_PHRASES" \
  --seed "$SEMANTIC_OT_SEED" \
  --patience "$SEMANTIC_OT_GROUNDER_PATIENCE"

# 7. Embed only copies, preserving the authoritative Event-DB files.
for split in train val test; do
  mkdir -p "$EMBED_ROOT/$split"
  cp -a "$AESD_ROOT/$split/events_aesd.npz" "$EMBED_ROOT/$split/events.npz"
  "$PY" -m grounding.mixed_curvature embed \
    --db "$EMBED_ROOT/$split/events.npz" \
    --checkpoint "$GROUNDER_CKPT" \
    --batch_size 256
done

# 8. Optional physical threshold calibration on validation motion only.
RISK_LOW="$SEMANTIC_OT_REPAIR_LOW"
RISK_HIGH="$SEMANTIC_OT_REPAIR_HIGH"
if [[ "$SEMANTIC_OT_CALIBRATE_RISK" == "1" ]]; then
  "$PY" evaluation/calibrate_transition_risk.py \
    --db "$AESD_ROOT/val/events_aesd.npz" \
    --out "$SEM_ROOT/transition_risk_calibration.json" \
    --fps "$V46_51_FPS"
  read -r RISK_LOW RISK_HIGH < <(
    "$PY" - "$SEM_ROOT/transition_risk_calibration.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
thresholds = report["global_thresholds"]
print(float(thresholds["low"]), float(thresholds["high"]))
PY
  )
fi

# 9. Audit semantic alignment without presenting teacher agreement as ground truth.
declare -A AUDIT_MOTION_SPLIT=( [train]=train [validation]=val [test]=test )
for split in train validation test; do
  motion_split="${AUDIT_MOTION_SPLIT[$split]}"
  "$PY" evaluation/audit_semantic_alignment.py \
    --dataset "$OT_ROOT/semantic_ot_${split}.npz" \
    --aesd "$AESD_ROOT/$motion_split/events_aesd.npz" \
    --out "$SEM_ROOT/${split}.semantic_alignment.audit.json"
done

cat > "$SEM_ROOT/activate_semantic_ot.env" <<ENV
export SEMANTIC_OT_ENABLE=1
export SEMANTIC_OT_GENERATION_DB="$EMBED_ROOT/train/events.npz"
export SEMANTIC_OT_GROUNDER_CKPT="$GROUNDER_CKPT"
export V46_53_GROUNDER_ARCHITECTURE=mixed
export V46_53_GROUNDER_CKPT="$GROUNDER_CKPT"
export V46_53_MIXED_REQUIRE_RUNTIME_AUDIO=1
export SEMANTIC_OT_REPAIR_LOW="$RISK_LOW"
export SEMANTIC_OT_REPAIR_HIGH="$RISK_HIGH"
export SEMANTIC_OT_RISK_CALIBRATION="$SEM_ROOT/transition_risk_calibration.json"
ENV

cat > "$SEM_ROOT/SCIENTIFIC_USE.json" <<JSON
{
  "schema": "dunhuang_semantic_ot_run_contract_v1",
  "supervision": "semantic_optimal_transport",
  "is_ground_truth_pair": false,
  "historical_router_branch_reused": "music_encoder",
  "historical_motion_encoder_reused": false,
  "music_split_before_ot": true,
  "motion_source_split_before_ot": true,
  "grounder_checkpoint": "$GROUNDER_CKPT",
  "authoritative_event_db_root": "$DB_SPLIT_ROOT",
  "calibrated_aesd_copy_root": "$AESD_ROOT",
  "embedded_db_root": "$EMBED_ROOT"
}
JSON

echo "Semantic-OT research pipeline completed: $SEM_ROOT"
