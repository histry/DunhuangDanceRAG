#!/usr/bin/env bash
set -euo pipefail

# Ensure top-level research packages are importable.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"


cd "$ROOT_DIR"

source configs/scheduler.env
source configs/anatomy.env
[[ -f "$ROOT_DIR/configs/research_feasibility.env" ]] && source "$ROOT_DIR/configs/research_feasibility.env"

PY="${V46_51_PYTHON}"
[[ -x "$PY" ]] || {
  echo "[FATAL] V46.51 Python is not executable: $PY" >&2
  exit 2
}

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-output/v46_51_fresh_wav_${RUN_TAG}}"
RETARGET_CACHE="${RETARGET_CACHE:-$OUT_ROOT/retarget_cache}"
CACHE_SPLIT_ROOT="${CACHE_SPLIT_ROOT:-$OUT_ROOT/retarget_cache_split}"
DB_SPLIT_ROOT="${DB_SPLIT_ROOT:-$OUT_ROOT/event_db_split}"
ALL_DB_DIR="${ALL_DB_DIR:-$OUT_ROOT/all_change_demo_db}"

TRAIN_DB="$DB_SPLIT_ROOT/train/events.npz"
VAL_DB="$DB_SPLIT_ROOT/val/events.npz"
TEST_DB="$DB_SPLIT_ROOT/test/events.npz"
TRAIN_AESD="$DB_SPLIT_ROOT/train/events_aesd.npz"
VAL_AESD="$DB_SPLIT_ROOT/val/events_aesd.npz"
TEST_AESD="$DB_SPLIT_ROOT/test/events_aesd.npz"
ALL_DB="$ALL_DB_DIR/events.npz"
ALL_AESD="$ALL_DB_DIR/events_aesd.npz"

V44_CKPT="${V44_CKPT:-$OUT_ROOT/v44_train_only_contrastive.pt}"
V45_CKPT="${V45_CKPT:-$OUT_ROOT/v45_train_only_refiner.pt}"
V46_CKPT="${V46_CKPT:-$OUT_ROOT/v46_train_only_diffusion.pt}"

SCHEDULER_CHECKPOINT_DIR="${SCHEDULER_CHECKPOINT_DIR:-$OUT_ROOT/checkpoints}"
SCHEDULER_TRAIN_DIR="${SCHEDULER_TRAIN_DIR:-$OUT_ROOT/scheduler_training}"
FORMAL_ROUTER_CKPT="${FORMAL_ROUTER_CKPT:-$SCHEDULER_CHECKPOINT_DIR/music_motion_router.pt}"
FORMAL_DURATION_CKPT="${FORMAL_DURATION_CKPT:-$SCHEDULER_CHECKPOINT_DIR/duration_predictor.pt}"
FORMAL_PLANNER_CKPT="${FORMAL_PLANNER_CKPT:-$SCHEDULER_CHECKPOINT_DIR/whole_song_planner.pt}"
# The historical 985-song Router contributes only its music encoder.  Its
# motion encoder is tied to the old Event-DB and is never reused.
MUSIC_ENCODER_PRIOR_CKPT="${V46_54_MUSIC_ENCODER_PRIOR_CKPT:-${V46_51_ROUTER_CKPT:-}}"

SCHEDULE_ROOT="${SCHEDULE_ROOT:-$OUT_ROOT/fresh_schedule}"
FRESH_MSSD="${FRESH_MSSD:-$SCHEDULE_ROOT/current_wav.final.mssd.json}"
FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/v46_51_final.npy}"
FINAL_REPORT="${FINAL_REPORT:-$OUT_ROOT/v46_51_final.report.json}"
FINAL_MP4="${FINAL_MP4:-$OUT_ROOT/v46_51_final.scientific_fixed.mp4}"

mkdir -p "$OUT_ROOT" "$SCHEDULER_CHECKPOINT_DIR" "$SCHEDULER_TRAIN_DIR"

require_file() {
  local p="$1"
  local label="$2"
  [[ -s "$p" ]] || {
    echo "[FATAL] Missing $label: $p" >&2
    exit 2
  }
}

echo "========== V46.51 FORMAL PATHS =========="
printf "PY=%s\nOUT_ROOT=%s\nAUDIO=%s\nCHANGE_BVH_DIR=%s\nCONFIG=%s\nDB_MODE=%s\n" \
  "$PY" "$OUT_ROOT" "$AUDIO" "$CHANGE_BVH_DIR" "$CONFIG" "$V46_51_DB_MODE"

require_file "$AUDIO" "current WAV"
require_file "$CONFIG" "V46 config"

echo "========== 1. STRICT V46.49.4 RETARGET CACHE =========="
if [[ "$V46_51_REBUILD_RETARGET_CACHE" == "1" ]]; then
  "$PY" retargeting/build_retarget_cache.py \
    --in_dir "$CHANGE_BVH_DIR" \
    --out_dir "$RETARGET_CACHE" \
    --overwrite
else
  require_file "$RETARGET_CACHE/v46_50_retarget_cache_report.json" \
    "existing retarget cache report"
fi

echo "========== 2. RETARGET GRAVITY AUDIT =========="
"$PY" evaluation/audit_gravity.py \
  --motion_dir "$RETARGET_CACHE" \
  --fps "$V46_51_FPS" \
  --out "$OUT_ROOT/retarget_cache.gravity.json" \
  --csv "$OUT_ROOT/retarget_cache.gravity.csv"

echo "========== 3. SOURCE SPLIT BEFORE EVENT SLICING =========="
"$PY" data_pipeline/split_sources.py \
  --cache_root "$RETARGET_CACHE" \
  --out_root "$CACHE_SPLIT_ROOT" \
  --seed "$V46_51_SPLIT_SEED" \
  --train_ratio "$V46_51_TRAIN_RATIO" \
  --val_ratio "$V46_51_VAL_RATIO" \
  --test_ratio "$V46_51_TEST_RATIO" \
  --mode copy \
  --overwrite

echo "========== 4. BUILD SPLIT-SPECIFIC HEADING EVENT DATABASES =========="
if [[ "$V46_51_REBUILD_EVENT_DB" == "1" ]]; then
  for split in train val test; do
    cache_dir="$CACHE_SPLIT_ROOT/$split"
    db_dir="$DB_SPLIT_ROOT/$split"
    "$PY" events/build_database_entry.py \
      --config "$CONFIG" \
      --motion_dirs "$cache_dir" \
      --out_db "$db_dir" \
      --overwrite
  done
else
  require_file "$TRAIN_DB" "train event DB"
  require_file "$VAL_DB" "val event DB"
  require_file "$TEST_DB" "test event DB"
fi

echo "========== 5. SPLIT EVENT-DB HARD AUDITS =========="
for split in train val test; do
  db="$DB_SPLIT_ROOT/$split/events.npz"
  "$PY" evaluation/audit_event_database.py \
    --db "$db" \
    --out "$OUT_ROOT/${split}.event_heading.audit.json" \
    --csv "$OUT_ROOT/${split}.event_heading.audit.csv"
done

echo "========== 6. AESD ENRICHMENT PER SPLIT =========="
for split in train val test; do
  db="$DB_SPLIT_ROOT/$split/events.npz"
  aesd="$DB_SPLIT_ROOT/$split/events_aesd.npz"
  "$PY" events/build_semantics.py \
    --db "$db" \
    --out "$aesd" \
    --json "$OUT_ROOT/${split}.aesd_build.json"
done

if [[ "$V46_51_DB_MODE" == "qualitative_all_change" ]]; then
  echo "========== 6B. BUILD ALL-CHANGE QUALITATIVE UPPER-BOUND DB =========="
  "$PY" events/build_database_entry.py \
    --config "$CONFIG" \
    --motion_dirs "$RETARGET_CACHE" \
    --out_db "$ALL_DB_DIR" \
    --overwrite
  "$PY" evaluation/audit_event_database.py \
    --db "$ALL_DB" \
    --out "$OUT_ROOT/all_change.event_heading.audit.json" \
    --csv "$OUT_ROOT/all_change.event_heading.audit.csv"
  "$PY" events/build_semantics.py \
    --db "$ALL_DB" \
    --out "$ALL_AESD" \
    --json "$OUT_ROOT/all_change.aesd_build.json"
  GENERATION_DB="$ALL_AESD"
else
  GENERATION_DB="$TRAIN_AESD"
fi

echo "========== 6C. BUILD GENERATION-ALIGNED SCHEDULER INDEX =========="
ALIGNED_SCHEDULER_DIR="$OUT_ROOT/scheduler_generation_assets"
ALIGNED_INDEX_JSON="$ALIGNED_SCHEDULER_DIR/event_index.json"
ALIGNED_INDEX_NPZ="$ALIGNED_SCHEDULER_DIR/duration_index.npz"
mkdir -p "$ALIGNED_SCHEDULER_DIR"
"$PY" scheduling/build_generation_index.py \
  --db "$GENERATION_DB" \
  --out_json "$ALIGNED_INDEX_JSON" \
  --out_npz "$ALIGNED_INDEX_NPZ" \
  --report "$ALIGNED_SCHEDULER_DIR/build_report.json"
export V46_51_INDEX_JSON="$ALIGNED_INDEX_JSON"
export V46_51_DURATION_INDEX_NPZ="$ALIGNED_INDEX_NPZ"
# A hierarchy built for the old 4225-event snapshot is never reused with the
# Generation DB. It may be rebuilt separately from this aligned index later.
export V46_51_HIERARCHY_INDEX_NPZ=""

read -r -a MUSIC_DIR_ARRAY <<< "$MUSIC_DIRS"
for d in "${MUSIC_DIR_ARRAY[@]}"; do
  if [[ "$d" == *"test_music_bank"* || "$d" == *"classical_eval"* ]]; then
    echo "[FATAL] evaluation music must not enter training: $d" >&2
    exit 2
  fi
done

ROUTER_DATA="$SCHEDULER_TRAIN_DIR/router_training.npz"
DURATION_DATA="$SCHEDULER_TRAIN_DIR/duration_training.npz"
PLANNER_DATA="$SCHEDULER_TRAIN_DIR/planner_training.npz"

echo "========== 7. BUILD + TRAIN FORMAL MUSIC-MOTION ROUTER =========="
if [[ "$V46_51_RETRAIN_ROUTER" == "1" ]]; then
  require_file "$MUSIC_ENCODER_PRIOR_CKPT" \
    "historical music-semantic Router prior"
  "$PY" training/music_router.py build-dataset \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --music_dirs "${MUSIC_DIR_ARRAY[@]}" \
    --cache_dir "$SCHEDULER_TRAIN_DIR/music_feature_cache" \
    --out "$ROUTER_DATA" \
    --fps "$V46_51_FPS" \
    --phrases "$V46_54_ROUTER_PHRASES_PER_SONG" \
    --positives_per_phrase "$V46_54_ROUTER_POSITIVES_PER_PHRASE" \
    --negatives_per_positive "$V46_54_ROUTER_NEGATIVES_PER_POSITIVE"
  "$PY" training/music_router.py train \
    --data "$ROUTER_DATA" \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --music_prior_ckpt "$MUSIC_ENCODER_PRIOR_CKPT" \
    --freeze_music_encoder "$V46_54_FREEZE_MUSIC_ENCODER" \
    --out "$FORMAL_ROUTER_CKPT" \
    --fps "$V46_51_FPS" \
    --epochs "$V46_54_ROUTER_EPOCHS" \
    --batch_size "$V46_54_ROUTER_BATCH"
else
  require_file "$FORMAL_ROUTER_CKPT" "formal Router checkpoint"
fi

echo "========== 7B. BUILD + TRAIN FORMAL DURATION MODEL =========="
if [[ "$V46_51_RETRAIN_DURATION" == "1" ]]; then
  "$PY" training/duration_model.py build-dataset \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --out "$DURATION_DATA" \
    --fps "$V46_51_FPS" \
    --window_len "${V46_54_DURATION_WINDOW_FRAMES:-0}" \
    --augmentations_per_event "$V46_54_DURATION_AUGMENTATIONS"
  "$PY" training/duration_model.py train \
    --data "$DURATION_DATA" \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --out "$FORMAL_DURATION_CKPT" \
    --fps "$V46_51_FPS" \
    --epochs "$V46_54_DURATION_EPOCHS" \
    --batch_size "$V46_54_DURATION_BATCH"
else
  require_file "$FORMAL_DURATION_CKPT" "formal Duration checkpoint"
fi

echo "========== 7C. BUILD + TRAIN FORMAL WHOLE-SONG PLANNER =========="
if [[ "$V46_51_RETRAIN_PLANNER" == "1" ]]; then
  "$PY" training/whole_song_planner.py build-dataset \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --router_ckpt "$FORMAL_ROUTER_CKPT" \
    --duration_ckpt "$FORMAL_DURATION_CKPT" \
    --music_dirs "${MUSIC_DIR_ARRAY[@]}" \
    --cache_dir "$SCHEDULER_TRAIN_DIR/whole_song_feature_cache" \
    --out "$PLANNER_DATA" \
    --fps "$V46_51_FPS" \
    --cooldown_slots "$V46_54_EVENT_COOLDOWN_SLOTS"
  "$PY" training/whole_song_planner.py train \
    --data "$PLANNER_DATA" \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --out "$FORMAL_PLANNER_CKPT" \
    --fps "$V46_51_FPS" \
    --epochs "$V46_54_PLANNER_EPOCHS" \
    --batch_size "$V46_54_PLANNER_BATCH"
else
  require_file "$FORMAL_PLANNER_CKPT" "formal Planner checkpoint"
fi

export V46_51_ROUTER_CKPT="$FORMAL_ROUTER_CKPT"
export V46_51_PLANNER_CKPT="$FORMAL_PLANNER_CKPT"
export V46_51_V23_CKPT="$FORMAL_DURATION_CKPT"
export V46_51_RESOLVED_INDEX_JSON="$ALIGNED_INDEX_JSON"
export V46_51_RESOLVED_DURATION_INDEX_NPZ="$ALIGNED_INDEX_NPZ"
export V46_51_RESOLVED_ROUTER_CKPT="$FORMAL_ROUTER_CKPT"
export V46_51_RESOLVED_PLANNER_CKPT="$FORMAL_PLANNER_CKPT"
export V46_51_RESOLVED_V23_CKPT="$FORMAL_DURATION_CKPT"

echo "========== 7D. VALIDATE FORMAL SCHEDULER ASSET BUNDLE =========="
"$PY" scheduling/build_asset_bundle.py \
  --index_json "$ALIGNED_INDEX_JSON" \
  --index_npz "$ALIGNED_INDEX_NPZ" \
  --router_ckpt "$FORMAL_ROUTER_CKPT" \
  --planner_ckpt "$FORMAL_PLANNER_CKPT" \
  --duration_ckpt "$FORMAL_DURATION_CKPT" \
  --fps "$V46_51_FPS" \
  --out "$ALIGNED_SCHEDULER_DIR/scheduler_asset_bundle.json"

if [[ "$V46_54_RUN_PRETRAIN_REGRESSION" == "1" ]]; then
  echo "========== 7E. SAME-WAV NO-TRAINING ROUTE/ACTION REGRESSION =========="
  PRETRAIN_REGRESSION_DIR="$OUT_ROOT/pretrain_same_wav_regression_${RUN_TAG}"
  "$PY" scripts/run_no_training_regression.py \
    --audio "$AUDIO" \
    --index_json "$V46_51_RESOLVED_INDEX_JSON" \
    --index_npz "$V46_51_RESOLVED_DURATION_INDEX_NPZ" \
    --router_ckpt "$V46_51_RESOLVED_ROUTER_CKPT" \
    --planner_ckpt "$V46_51_RESOLVED_PLANNER_CKPT" \
    --duration_ckpt "$V46_51_RESOLVED_V23_CKPT" \
    --config "$CONFIG" \
    --out_dir "$PRETRAIN_REGRESSION_DIR" \
    --fps "$V46_51_FPS" \
    --max_source_share "$V46_54_MAX_SOURCE_SHARE" \
    --max_transition_fraction "$V46_51_MAX_TRANSITION_FRACTION"
  require_file "$PRETRAIN_REGRESSION_DIR/regression_gate.json" \
    "same-WAV regression gate"
fi

echo "========== 7F. TRAIN V44 ON TRAIN SOURCES + NON-TEST MUSIC =========="
if [[ "$V46_51_RETRAIN_V44" == "1" ]]; then
  "$PY" training/motion_models.py \
    --config "$CONFIG" \
    train-contrastive \
    --db "$TRAIN_AESD" \
    --out "$V44_CKPT" \
    --unpaired_audio_dirs "${MUSIC_DIR_ARRAY[@]}" \
    --epochs "$V44_EPOCHS"
else
  require_file "$V44_CKPT" "V44 checkpoint"
fi

echo "========== 8. TRAIN V45 ON TRAIN-SOURCE CANONICAL EVENTS =========="
if [[ "$V46_51_RETRAIN_V45" == "1" ]]; then
  "$PY" training/motion_models.py \
    --config "$CONFIG" \
    train-refiner \
    --db "$TRAIN_AESD" \
    --val_db "$VAL_AESD" \
    --out "$V45_CKPT" \
    --steps "$V45_STEPS"
else
  require_file "$V45_CKPT" "V45 checkpoint"
fi

echo "========== 9. TRAIN V46 ON TRAIN-SOURCE CANONICAL EVENTS =========="
if [[ "$V46_51_RETRAIN_V46" == "1" ]]; then
  "$PY" training/motion_models.py \
    --config "$CONFIG" \
    train-diffusion \
    --db "$TRAIN_AESD" \
    --val_db "$VAL_AESD" \
    --out "$V46_CKPT" \
    --steps "$V46_STEPS" \
    --diffusion_steps "$V46_DIFFUSION_STEPS"
else
  require_file "$V46_CKPT" "V46 checkpoint"
fi

echo "========== 10. USE VALIDATED GENERATION-ALIGNED SCHEDULER ASSETS =========="
require_file "$ALIGNED_SCHEDULER_DIR/scheduler_asset_bundle.json" \
  "Router/Planner/Duration asset bundle"

echo "========== 11. REBUILD SCHEDULE FROM CURRENT WAV =========="
AUDIO_SHA="$(sha256sum "$AUDIO" | awk '{print $1}')"
export V46_51_SCHEDULE_RUN_ID="${RUN_TAG}_${AUDIO_SHA:0:12}"
FRESH_RUN_DIR="$SCHEDULE_ROOT/$V46_51_SCHEDULE_RUN_ID"

FRESH_ARGS=(
  --audio "$AUDIO"
  --out_json "$FRESH_MSSD"
  --run_dir "$FRESH_RUN_DIR"
  --run_id "$V46_51_SCHEDULE_RUN_ID"
  --router_ckpt "$V46_51_RESOLVED_ROUTER_CKPT"
  --planner_ckpt "$V46_51_RESOLVED_PLANNER_CKPT"
  --v23_ckpt "$V46_51_RESOLVED_V23_CKPT"
  --index_json "$V46_51_RESOLVED_INDEX_JSON"
  --duration_index_npz "$V46_51_RESOLVED_DURATION_INDEX_NPZ"
  --fps "$V46_51_FPS"
  --min_phrase_seconds "$V46_51_MIN_PHRASE_SECONDS"
  --max_phrase_seconds "$V46_51_MAX_PHRASE_SECONDS"
  --max_phrases "$V46_51_MAX_PHRASES"
  --boundary_quantile "$V46_51_BOUNDARY_QUANTILE"
  --beat_snap_seconds "$V46_51_BEAT_SNAP_SECONDS"
  --max_single_event_seconds "$V46_51_MAX_SINGLE_EVENT_SECONDS"
  --calm_max_single_event_seconds "$V46_51_CALM_MAX_SINGLE_EVENT_SECONDS"
  --min_subphrase_seconds "$V46_51_MIN_SUBPHRASE_SECONDS"
  --max_events_per_phrase "$V46_51_MAX_EVENTS_PER_PHRASE"
  --transition_min_frames "$V46_51_TRANSITION_MIN_FRAMES"
  --transition_max_frames "$V46_51_TRANSITION_MAX_FRAMES"
  --max_transition_fraction "$V46_51_MAX_TRANSITION_FRACTION"
  --transition_budget_min_frames "$V46_51_TRANSITION_BUDGET_MIN_FRAMES"
  --slot_beat_snap_seconds "$V46_51_SLOT_BEAT_SNAP_SECONDS"
  --beam_size "$V46_51_BEAM_SIZE"
  --candidate_top_k "$V46_51_CANDIDATE_TOP_K"
  --graph_node_top_k "$V46_51_GRAPH_NODE_TOP_K"
  --physical_edge_weight "$V46_54_PHYSICAL_EDGE_WEIGHT"
  --physical_edge_reset_accent "$V46_54_PHYSICAL_EDGE_RESET_ACCENT"
  --root_height_gap_reference_m "$V46_54_ROOT_HEIGHT_GAP_REFERENCE_M"
  --root_height_gap_hard_m "$V46_54_ROOT_HEIGHT_GAP_HARD_M"
  --posture_state_gap_hard "$V46_54_POSTURE_STATE_GAP_HARD"
  --floor_gap_reference_m "$V46_54_FLOOR_GAP_REFERENCE_M"
  --floor_gap_hard_m "$V46_54_FLOOR_GAP_HARD_M"
  --root_velocity_jump_reference_mps "$V46_54_ROOT_VELOCITY_JUMP_REFERENCE_MPS"
  --root_velocity_jump_hard_mps "$V46_54_ROOT_VELOCITY_JUMP_HARD_MPS"
  --contact_gap_hard "$V46_54_CONTACT_GAP_HARD"
  --stage_floor_y "$V46_54_STAGE_FLOOR_Y"
  --event_floor_quantile "$V46_54_EVENT_FLOOR_QUANTILE"
  --event_max_floor_penetration_m "$V46_54_EVENT_MAX_FLOOR_PENETRATION_M"
  --transition_angular_speed_cap_radps "$V46_54_TRANSITION_ANGULAR_SPEED_CAP_RADPS"
  --transition_root_horizontal_speed_cap_mps "$V46_54_TRANSITION_ROOT_XZ_SPEED_CAP_MPS"
  --transition_root_vertical_speed_cap_mps "$V46_54_TRANSITION_ROOT_Y_SPEED_CAP_MPS"
  --transition_root_tangent_margin_m "$V46_54_TRANSITION_ROOT_TANGENT_MARGIN_M"
  --transition_floor_clearance_m "$V46_54_TRANSITION_FLOOR_CLEARANCE_M"
  --transition_floor_smoothing_seconds "$V46_54_TRANSITION_FLOOR_SMOOTH_SECONDS"
  --transition_contact_ramp_seconds "$V46_54_TRANSITION_CONTACT_RAMP_SECONDS"
  --max_frame_error "$V46_51_MAX_FRAME_ERROR"
  --max_seconds_error "$V46_51_MAX_SECONDS_ERROR"
)

if [[ "$V46_54_PHYSICAL_EDGE_HARD_PRUNE" == "1" ]]; then
  FRESH_ARGS+=(--physical_edge_hard_prune)
else
  FRESH_ARGS+=(--no-physical_edge_hard_prune)
fi

[[ -n "$V46_51_RESOLVED_HIERARCHY_INDEX_NPZ" ]] && \
  FRESH_ARGS+=(--hierarchy_index_npz "$V46_51_RESOLVED_HIERARCHY_INDEX_NPZ")
[[ -n "$V46_51_RESOLVED_START_POSE" ]] && \
  FRESH_ARGS+=(--start_pose "$V46_51_RESOLVED_START_POSE")
[[ "$V46_51_DEEP_MUSIC_FEATURES" == "1" ]] && \
  FRESH_ARGS+=(--deep_music_features)
[[ "$V46_51_REQUIRE_DEEP_MUSIC" == "1" ]] && \
  FRESH_ARGS+=(--require_deep_music)
FRESH_ARGS+=(--deep_music_model "$V46_51_DEEP_MUSIC_MODEL")
FRESH_ARGS+=(--deep_music_min_success "$V46_51_DEEP_MUSIC_MIN_SUCCESS")

if [[ "$V46_51_TRANSITION_DIFFUSION" == "1" ]]; then
  require_file "$V46_51_TRANSITION_DIFFUSION_CKPT" \
    "V26 transition diffusion checkpoint"
  FRESH_ARGS+=(
    --transition_diffusion
    --transition_diffusion_ckpt "$V46_51_TRANSITION_DIFFUSION_CKPT"
    --transition_diffusion_blend "$V46_51_TRANSITION_DIFFUSION_BLEND"
    --transition_diffusion_steps "$V46_51_TRANSITION_DIFFUSION_STEPS"
  )
fi

"$PY" scheduling/build_schedule.py "${FRESH_ARGS[@]}"

echo "========== 12. FRESH-WAV CONTRACT RECHECK =========="
"$PY" scheduling/validate_schedule.py \
  --audio "$AUDIO" \
  --schedule "$FRESH_MSSD" \
  --required_run_id "$V46_51_SCHEDULE_RUN_ID" \
  --fps "$V46_51_FPS" \
  --max_frame_error "$V46_51_MAX_FRAME_ERROR" \
  --max_seconds_error "$V46_51_MAX_SECONDS_ERROR" \
  --out "$OUT_ROOT/fresh_schedule.contract.json" \
  --csv "$OUT_ROOT/fresh_schedule.contract.csv"

ROUTING_MSSD="$FRESH_MSSD"
if [[ "${V46_53_GROUNDER_ARCHITECTURE:-legacy}" == "mixed" ]]; then
  echo "========== 12B. MIXED-GROUNDER REAL-AUDIO SLOT FEATURES =========="
  MIXED_GROUNDER_CKPT="${V46_53_GROUNDER_CKPT:-}"
  require_file "$MIXED_GROUNDER_CKPT" \
    "mixed-curvature Grounder checkpoint"
  MIXED_MSSD="$SCHEDULE_ROOT/current_wav.final.mixed_grounding.json"
  "$PY" -m grounding.audio_query \
    --audio "$AUDIO" \
    --schedule "$FRESH_MSSD" \
    --out "$MIXED_MSSD" \
    --checkpoint "$MIXED_GROUNDER_CKPT" \
    --model_name "$V46_51_DEEP_MUSIC_MODEL" \
    --cache_dir "$OUT_ROOT/mixed_audio_cache" \
    --temporal_frames "${V46_53_MIXED_TEMPORAL_FRAMES:-64}" \
    --temporal_source_frames "${V46_53_MIXED_TEMPORAL_SOURCE_FRAMES:-2048}" \
    --phrase_fps "$V46_51_FPS"
  require_file "$MIXED_MSSD" "mixed-grounding enriched schedule"
  "$PY" scheduling/validate_schedule.py \
    --audio "$AUDIO" \
    --schedule "$MIXED_MSSD" \
    --required_run_id "$V46_51_SCHEDULE_RUN_ID" \
    --fps "$V46_51_FPS" \
    --max_frame_error "$V46_51_MAX_FRAME_ERROR" \
    --max_seconds_error "$V46_51_MAX_SECONDS_ERROR" \
    --out "$OUT_ROOT/fresh_schedule.mixed_grounding.contract.json" \
    --csv "$OUT_ROOT/fresh_schedule.mixed_grounding.contract.csv"
  ROUTING_MSSD="$MIXED_MSSD"
fi

echo "========== 13. V46.51 HEADING/BOUNDARY CLOSED-LOOP GENERATION =========="
"$PY" routing/closed_loop.py \
  generate \
  --config "$CONFIG" \
  --audio "$AUDIO" \
  --slots_json "$ROUTING_MSSD" \
  --db "$GENERATION_DB" \
  --contrastive "$V44_CKPT" \
  --refiner "$V45_CKPT" \
  --diffusion "$V46_CKPT" \
  --out "$FINAL_NPY" \
  --json "$FINAL_REPORT"

echo "========== 14. FINAL GRAVITY AUDIT =========="
"$PY" evaluation/audit_gravity.py \
  --input "$FINAL_NPY" \
  --fps "$V46_51_FPS" \
  --out "$OUT_ROOT/final.gravity.json" \
  --csv "$OUT_ROOT/final.gravity.csv"

echo "========== 15. FINAL HEADING-PLAN AUDIT =========="
"$PY" evaluation/audit_heading.py \
  --motion "$FINAL_NPY" \
  --report "$FINAL_REPORT" \
  --db "$GENERATION_DB" \
  --fps "$V46_51_FPS" \
  --out "$OUT_ROOT/final.heading.json" \
  --csv "$OUT_ROOT/final.heading.csv"

echo "========== 16. EXACT FINAL FRAME CONTRACT =========="
"$PY" - "$FINAL_NPY" "$OUT_ROOT/fresh_schedule.contract.json" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

motion_path = Path(sys.argv[1])
contract_path = Path(sys.argv[2])
x = np.load(motion_path, allow_pickle=True)
frames = int(x.shape[-2])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
scheduled = int(contract["total_target_frames"])
audio_expected = int(contract["expected_audio_target_frames"])
if frames != scheduled:
    raise SystemExit(
        f"[FATAL] final motion frames={frames}, scheduled={scheduled}"
    )
print(json.dumps({
    "ok": True,
    "motion": str(motion_path),
    "frames": frames,
    "scheduled_frames": scheduled,
    "audio_expected_frames": audio_expected,
    "audio_frame_error": scheduled - audio_expected,
}, indent=2))
PY

echo "========== 17. SCIENTIFIC FIXED-CAMERA RENDER =========="
"$PY" rendering/render_motion.py \
  --motion "$FINAL_NPY" \
  --audio "$AUDIO" \
  --output "$FINAL_MP4" \
  --fps "$V46_51_FPS" \
  --camera_mode fixed \
  --render_smooth_window 1 \
  --gravity_audit_json "$OUT_ROOT/final.render_gravity.json"

echo "========== V46.51 COMPLETE =========="
printf "FRESH_MSSD=%s\nROUTING_MSSD=%s\nGENERATION_DB=%s\nFINAL_NPY=%s\nFINAL_REPORT=%s\nFINAL_MP4=%s\n" \
  "$FRESH_MSSD" "$ROUTING_MSSD" "$GENERATION_DB" "$FINAL_NPY" "$FINAL_REPORT" "$FINAL_MP4"
ls -lh \
  "$TRAIN_AESD" \
  "$VAL_AESD" \
  "$TEST_AESD" \
  "$V44_CKPT" \
  "$V45_CKPT" \
  "$V46_CKPT" \
  "$FRESH_MSSD" \
  "$ROUTING_MSSD" \
  "$FINAL_NPY" \
  "$FINAL_REPORT" \
  "$FINAL_MP4"
