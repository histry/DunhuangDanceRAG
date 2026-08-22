#!/usr/bin/env bash
set -euo pipefail

# Ensure top-level research packages are importable.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"


cd "$ROOT_DIR"

if [[ "${EXPERIMENT_CONFIG_LOADED:-0}" != "1" ]]; then
  # shellcheck disable=SC1091
  source configs/experiment.env
fi

PY="${GENERATION_PYTHON}"
[[ -x "$PY" ]] || {
  echo "[FATAL] Fresh-Audio Generation Python is not executable: $PY" >&2
  exit 2
}

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-output/fresh_audio_fresh_wav_${RUN_TAG}}"
RETARGET_CACHE="${RETARGET_CACHE:-$OUT_ROOT/retarget_cache}"
SOURCE_MODE="${RETARGET_CLEAN_SOURCE_MODE:-chang_e_official_smpl}"
if [[ "$SOURCE_MODE" != "chang_e_official_smpl" ]]; then
  echo "[FATAL] main supports only Chang-E official SMPL14. Historical BVH code is archived in Git." >&2
  exit 2
fi
OFFICIAL_SMPL_DIR="${CHANG_E_OFFICIAL_SMPL_DIR:?configs/experiment.env must define the official SMPL directory}"
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

REFINER_CKPT="${REFINER_CKPT:-$OUT_ROOT/motion_refiner_train_only_refiner.pt}"
MOTION_CKPT="${MOTION_CKPT:-$OUT_ROOT/motion_train_only_diffusion.pt}"

SCHEDULER_CHECKPOINT_DIR="${SCHEDULER_CHECKPOINT_DIR:-$OUT_ROOT/checkpoints}"
SCHEDULER_TRAIN_DIR="${SCHEDULER_TRAIN_DIR:-$OUT_ROOT/scheduler_training}"
FORMAL_ROUTER_CKPT="${FORMAL_ROUTER_CKPT:-$SCHEDULER_CHECKPOINT_DIR/ctsr_weak_temporal_router.pt}"
BASELINE_DIR="${BASELINE_DIR:-$OUT_ROOT/baselines}"
CURRENT_ROUTER_BASELINE_CKPT="${CURRENT_ROUTER_BASELINE_CKPT:-$BASELINE_DIR/ctsr_mean_pool_mlp.pt}"
FORMAL_DURATION_CKPT="${FORMAL_DURATION_CKPT:-$SCHEDULER_CHECKPOINT_DIR/duration_predictor.pt}"
FORMAL_PLANNER_CKPT="${FORMAL_PLANNER_CKPT:-$SCHEDULER_CHECKPOINT_DIR/whole_song_planner.pt}"
SCHEDULE_ROOT="${SCHEDULE_ROOT:-$OUT_ROOT/fresh_schedule}"
FRESH_MSSD="${FRESH_MSSD:-$SCHEDULE_ROOT/current_wav.final.mssd.json}"
FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/fresh_audio_final.npy}"
FINAL_REPORT="${FINAL_REPORT:-$OUT_ROOT/fresh_audio_final.report.json}"
FINAL_MP4="${FINAL_MP4:-$OUT_ROOT/fresh_audio_final.scientific_fixed.mp4}"

mkdir -p "$OUT_ROOT" "$SCHEDULER_CHECKPOINT_DIR" "$SCHEDULER_TRAIN_DIR" "$BASELINE_DIR"

require_file() {
  local p="$1"
  local label="$2"
  [[ -s "$p" ]] || {
    echo "[FATAL] Missing $label: $p" >&2
    exit 2
  }
}

# Formal music semantics fail closed on the only supported feature backend.
if [[ "${REQUIRE_LIBROSA_BACKEND:-0}" != "1" ]]; then
  echo "[FATAL] Formal music semantics require Librosa 12D." >&2
  exit 2
fi

if [[ "${ROUTING_FORMAL_ROUTER_ARCHITECTURE:-}" != "ctsr_weak_temporal_v1" \
   || "${ROUTING_FORMAL_SUPERVISION_SOURCE:-}" != "semantic_ot_teacher" \
   || "${ROUTING_SAFETY_FREEZE_MUSIC_ENCODER:-1}" != "0" ]]; then
  echo "[FATAL] Formal routing must use scratch-trained CTSR-Weak only." >&2
  exit 2
fi

if [[ "${FORMAL_REQUIRE_CLEAN_GIT:-1}" != "1" ]]; then
  echo "[FATAL] Formal training requires FORMAL_REQUIRE_CLEAN_GIT=1." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "[FATAL] Formal training requires a clean Git worktree so every checkpoint has reproducible code provenance." >&2
  git status --short >&2
  exit 2
fi

if [[ "${GRAPH_ROUTE_SOLVER:-}" != "fisher_rao_graph_sb" ]]; then
  echo "[FATAL] Formal routing requires solver=fisher_rao_graph_sb." >&2
  exit 2
fi

if [[ "${PERFORMER_REQUIRE_SOLO_COMPATIBLE:-0}" != "1" ]]; then
  echo "[FATAL] Formal one-body generation must exclude unreviewed multi-performer recordings." >&2
  exit 2
fi

echo "========== Fresh-Audio Generation FORMAL PATHS =========="
printf "PY=%s\nOUT_ROOT=%s\nAUDIO=%s\nSOURCE_MODE=%s\nOFFICIAL_SMPL_DIR=%s\nCONFIG=%s\nDB_MODE=%s\n" \
  "$PY" "$OUT_ROOT" "$AUDIO" "$SOURCE_MODE" "$OFFICIAL_SMPL_DIR" "$CONFIG" "$GENERATION_DB_MODE"

require_file "$AUDIO" "current WAV"
require_file "$CONFIG" "Motion Generation config"

echo "========== 1. SOURCE-AWARE SOURCE CACHE =========="
if [[ "$GENERATION_REBUILD_RETARGET_CACHE" == "1" ]]; then
  "$PY" retargeting/official_smpl_source_preprocess.py \
    --in_dir "$OFFICIAL_SMPL_DIR" \
    --out_dir "$RETARGET_CACHE" \
    --smpl_manifest "$CHANG_E_OFFICIAL_SMPL_MANIFEST" \
    --target_fps "$GENERATION_FPS" \
    --min_ok_sources "$RETARGET_MIN_OK_SOURCES" \
    --overwrite
else
  require_file "$RETARGET_CACHE/event_heading_retarget_cache_report.json" \
    "existing source cache report"
fi

echo "========== 2. SOURCE GRAVITY DIAGNOSTIC =========="
# Direct official SMPL is not rejected at whole-source level for a
# posture-style gravity statistic. Event-level anatomy remains strict.
"$PY" evaluation/audit_gravity.py \
  --motion_dir "$RETARGET_CACHE" \
  --profile source \
  --allow_failed \
  --fps "$GENERATION_FPS" \
  --out "$OUT_ROOT/retarget_cache.gravity.json" \
  --csv "$OUT_ROOT/retarget_cache.gravity.csv"

echo "========== 3. SOURCE SPLIT BEFORE EVENT SLICING =========="
"$PY" data_pipeline/split_sources.py \
  --cache_root "$RETARGET_CACHE" \
  --out_root "$CACHE_SPLIT_ROOT" \
  --seed "$GENERATION_SPLIT_SEED" \
  --train_ratio "$GENERATION_TRAIN_RATIO" \
  --val_ratio "$GENERATION_VAL_RATIO" \
  --test_ratio "$GENERATION_TEST_RATIO" \
  --protocol "$GENERATION_SPLIT_PROTOCOL" \
  --heldout_theme "$GENERATION_HELDOUT_THEME" \
  --mode copy \
  --overwrite

echo "========== 4. BUILD SPLIT-SPECIFIC HEADING EVENT DATABASES =========="
if [[ "$GENERATION_REBUILD_EVENT_DB" == "1" ]]; then
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
SPLIT_MANIFEST="$CACHE_SPLIT_ROOT/source_split_manifest.json"
require_file "$SPLIT_MANIFEST" "source split manifest"
for split in train val test; do
  db="$DB_SPLIT_ROOT/$split/events.npz"
  "$PY" evaluation/audit_event_database.py \
    --db "$db" \
    --split_manifest "$SPLIT_MANIFEST" \
    --split "$split" \
    --out "$OUT_ROOT/${split}.event_heading.audit.json" \
    --csv "$OUT_ROOT/${split}.event_heading.audit.csv"
  "$PY" evaluation/audit_formal_single_person_db.py \
    --db "$db" \
    --out "$OUT_ROOT/${split}.single_person.audit.json"
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

if [[ "$GENERATION_DB_MODE" == "qualitative_all_change" ]]; then
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

echo "========== 6A. BUILD GENERATION-ALIGNED SCHEDULER INDEX =========="
ALIGNED_SCHEDULER_DIR="$OUT_ROOT/scheduler_generation_assets"
ALIGNED_INDEX_JSON="$ALIGNED_SCHEDULER_DIR/event_index.json"
ALIGNED_INDEX_NPZ="$ALIGNED_SCHEDULER_DIR/duration_index.npz"
mkdir -p "$ALIGNED_SCHEDULER_DIR"
"$PY" scheduling/build_generation_index.py \
  --db "$GENERATION_DB" \
  --out_json "$ALIGNED_INDEX_JSON" \
  --out_npz "$ALIGNED_INDEX_NPZ" \
  --report "$ALIGNED_SCHEDULER_DIR/build_report.json"
export GENERATION_INDEX_JSON="$ALIGNED_INDEX_JSON"
export GENERATION_DURATION_INDEX_NPZ="$ALIGNED_INDEX_NPZ"
# A hierarchy built for the old 4225-event snapshot is never reused with the
# Generation DB. It may be rebuilt separately from this aligned index later.
export GENERATION_HIERARCHY_INDEX_NPZ=""

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

echo "========== 7. BUILD + TRAIN ZERO-LABEL CTSR-WEAK TEMPORAL ROUTER =========="
if [[ "$GENERATION_RETRAIN_ROUTER" == "1" ]]; then
  "$PY" training/temporal_music_router.py build-dataset \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --music_dirs "${MUSIC_DIR_ARRAY[@]}" \
    --heldout_audio "$AUDIO" \
    --cache_dir "$SCHEDULER_TRAIN_DIR/music_feature_cache" \
    --out "$ROUTER_DATA" \
    --fps "$GENERATION_FPS" \
    --expected_num_songs "$RETARGET_CLEAN_EXPECTED_TRAIN_MUSIC" \
    --min_phrase_seconds "$GENERATION_MIN_PHRASE_SECONDS" \
    --max_phrase_seconds "$GENERATION_MAX_PHRASE_SECONDS" \
    --boundary_quantile "$GENERATION_BOUNDARY_QUANTILE" \
    --beat_snap_seconds "$GENERATION_BEAT_SNAP_SECONDS" \
    --max_slot_seconds "$GENERATION_MAX_SINGLE_EVENT_SECONDS" \
    --calm_max_slot_seconds "$GENERATION_CALM_MAX_SINGLE_EVENT_SECONDS" \
    --min_slot_seconds "$GENERATION_MIN_SUBPHRASE_SECONDS" \
    --max_events_per_phrase "$GENERATION_MAX_EVENTS_PER_PHRASE" \
    --slot_beat_snap_seconds "$GENERATION_SLOT_BEAT_SNAP_SECONDS" \
    --sequence_frames "$ROUTING_TEMPORAL_SEQUENCE_FRAMES" \
    --teacher_top_k "$ROUTING_WEAK_OT_TOP_K" \
    --teacher_epsilon "$ROUTING_WEAK_OT_EPSILON" \
    --teacher_max_iterations "$ROUTING_WEAK_OT_MAX_ITER" \
    --teacher_tolerance "$ROUTING_WEAK_OT_TOLERANCE" \
    --teacher_max_marginal_error "$ROUTING_WEAK_OT_MAX_MARGINAL_ERROR" \
    --teacher_balance_key "$ROUTING_WEAK_OT_BALANCE_KEY" \
    --require_librosa_backend "$REQUIRE_LIBROSA_BACKEND"
  "$PY" training/temporal_music_router.py train \
    --data "$ROUTER_DATA" \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --out "$FORMAL_ROUTER_CKPT" \
    --fps "$GENERATION_FPS" \
    --sequence_frames "$ROUTING_TEMPORAL_SEQUENCE_FRAMES" \
    --hidden_dim "$ROUTING_TEMPORAL_HIDDEN_DIM" \
    --latent_dim "$ROUTING_TEMPORAL_LATENT_DIM" \
    --transformer_layers "$ROUTING_TEMPORAL_TRANSFORMER_LAYERS" \
    --transformer_heads "$ROUTING_TEMPORAL_TRANSFORMER_HEADS" \
    --mask_ratio "$ROUTING_TEMPORAL_MASK_RATIO" \
    --epochs "$ROUTING_SAFETY_ROUTER_EPOCHS" \
    --batch_size "$ROUTING_SAFETY_ROUTER_BATCH"
else
  require_file "$FORMAL_ROUTER_CKPT" "formal Router checkpoint"
fi

if [[ "$CURRENT_PROTOCOL_BASELINES_ENABLE" == "1" ]]; then
  echo "========== 7A-B. TRAIN CURRENT-PROTOCOL NON-TEMPORAL ROUTER BASELINE =========="
  if [[ "$GENERATION_RETRAIN_ROUTER" == "1" ]]; then
    "$PY" training/current_protocol_router_baseline.py \
      --data "$ROUTER_DATA" \
      --index_json "$ALIGNED_INDEX_JSON" \
      --index_npz "$ALIGNED_INDEX_NPZ" \
      --out "$CURRENT_ROUTER_BASELINE_CKPT" \
      --fps "$GENERATION_FPS" \
      --epochs "$CURRENT_PROTOCOL_BASELINE_EPOCHS" \
      --batch_size "$ROUTING_SAFETY_ROUTER_BATCH"
  else
    require_file "$CURRENT_ROUTER_BASELINE_CKPT" \
      "current-protocol non-temporal Router baseline checkpoint"
  fi
fi

echo "========== 7B. BUILD + TRAIN FORMAL DURATION MODEL =========="
if [[ "$GENERATION_RETRAIN_DURATION" == "1" ]]; then
  "$PY" training/duration_model.py build-dataset \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --out "$DURATION_DATA" \
    --fps "$GENERATION_FPS" \
    --window_len "${ROUTING_SAFETY_DURATION_WINDOW_FRAMES:-0}" \
    --augmentations_per_event "$ROUTING_SAFETY_DURATION_AUGMENTATIONS"
  "$PY" training/duration_model.py train \
    --data "$DURATION_DATA" \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --out "$FORMAL_DURATION_CKPT" \
    --fps "$GENERATION_FPS" \
    --epochs "$ROUTING_SAFETY_DURATION_EPOCHS" \
    --batch_size "$ROUTING_SAFETY_DURATION_BATCH"
else
  require_file "$FORMAL_DURATION_CKPT" "formal Duration checkpoint"
fi

echo "========== 7C. BUILD + TRAIN FORMAL WHOLE-SONG PLANNER =========="
if [[ "$GENERATION_RETRAIN_PLANNER" == "1" ]]; then
  "$PY" training/whole_song_planner.py build-dataset \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --router_ckpt "$FORMAL_ROUTER_CKPT" \
    --duration_ckpt "$FORMAL_DURATION_CKPT" \
    --music_dirs "${MUSIC_DIR_ARRAY[@]}" \
    --heldout_audio "$AUDIO" \
    --expected_num_songs "$RETARGET_CLEAN_EXPECTED_TRAIN_MUSIC" \
    --cache_dir "$SCHEDULER_TRAIN_DIR/whole_song_feature_cache" \
    --out "$PLANNER_DATA" \
    --fps "$GENERATION_FPS" \
    --min_phrase_seconds "$GENERATION_MIN_PHRASE_SECONDS" \
    --max_phrase_seconds "$GENERATION_MAX_PHRASE_SECONDS" \
    --boundary_quantile "$GENERATION_BOUNDARY_QUANTILE" \
    --beat_snap_seconds "$GENERATION_BEAT_SNAP_SECONDS" \
    --max_slot_seconds "$GENERATION_MAX_SINGLE_EVENT_SECONDS" \
    --calm_max_slot_seconds "$GENERATION_CALM_MAX_SINGLE_EVENT_SECONDS" \
    --min_slot_seconds "$GENERATION_MIN_SUBPHRASE_SECONDS" \
    --max_events_per_phrase "$GENERATION_MAX_EVENTS_PER_PHRASE" \
    --slot_beat_snap_seconds "$GENERATION_SLOT_BEAT_SNAP_SECONDS" \
    --require_rhythm_features \
    --cooldown_slots "$ROUTING_SAFETY_EVENT_COOLDOWN_SLOTS"
  "$PY" training/whole_song_planner.py train \
    --data "$PLANNER_DATA" \
    --index_json "$ALIGNED_INDEX_JSON" \
    --index_npz "$ALIGNED_INDEX_NPZ" \
    --out "$FORMAL_PLANNER_CKPT" \
    --fps "$GENERATION_FPS" \
    --epochs "$ROUTING_SAFETY_PLANNER_EPOCHS" \
    --batch_size "$ROUTING_SAFETY_PLANNER_BATCH"
else
  require_file "$FORMAL_PLANNER_CKPT" "formal Planner checkpoint"
fi

export GENERATION_ROUTER_CKPT="$FORMAL_ROUTER_CKPT"
export GENERATION_PLANNER_CKPT="$FORMAL_PLANNER_CKPT"
export GENERATION_DURATION_CKPT="$FORMAL_DURATION_CKPT"
export GENERATION_RESOLVED_INDEX_JSON="$ALIGNED_INDEX_JSON"
export GENERATION_RESOLVED_DURATION_INDEX_NPZ="$ALIGNED_INDEX_NPZ"
export GENERATION_RESOLVED_ROUTER_CKPT="$FORMAL_ROUTER_CKPT"
export GENERATION_RESOLVED_PLANNER_CKPT="$FORMAL_PLANNER_CKPT"
export GENERATION_RESOLVED_DURATION_CKPT="$FORMAL_DURATION_CKPT"

echo "========== 7D. VALIDATE FORMAL SCHEDULER ASSET BUNDLE =========="
"$PY" scheduling/build_asset_bundle.py \
  --index_json "$ALIGNED_INDEX_JSON" \
  --index_npz "$ALIGNED_INDEX_NPZ" \
  --router_ckpt "$FORMAL_ROUTER_CKPT" \
  --planner_ckpt "$FORMAL_PLANNER_CKPT" \
  --duration_ckpt "$FORMAL_DURATION_CKPT" \
  --fps "$GENERATION_FPS" \
  --out "$ALIGNED_SCHEDULER_DIR/scheduler_asset_bundle.json"

if [[ "$ROUTING_SAFETY_RUN_PRETRAIN_REGRESSION" == "1" ]]; then
  echo "========== 7E. SAME-WAV NO-TRAINING ROUTE/ACTION REGRESSION =========="
  PRETRAIN_REGRESSION_DIR="$OUT_ROOT/pretrain_same_wav_regression_${RUN_TAG}"
  "$PY" scripts/run_no_training_regression.py \
    --audio "$AUDIO" \
    --index_json "$GENERATION_RESOLVED_INDEX_JSON" \
    --index_npz "$GENERATION_RESOLVED_DURATION_INDEX_NPZ" \
    --router_ckpt "$GENERATION_RESOLVED_ROUTER_CKPT" \
    --planner_ckpt "$GENERATION_RESOLVED_PLANNER_CKPT" \
    --duration_ckpt "$GENERATION_RESOLVED_DURATION_CKPT" \
    --config "$CONFIG" \
    --out_dir "$PRETRAIN_REGRESSION_DIR" \
    --fps "$GENERATION_FPS" \
    --max_source_share "$ROUTING_SAFETY_MAX_SOURCE_SHARE" \
    --max_transition_fraction "$GENERATION_MAX_TRANSITION_FRACTION"
  require_file "$PRETRAIN_REGRESSION_DIR/regression_gate.json" \
    "same-WAV regression gate"
fi

echo "========== 8. TRAIN Motion Refiner ON TRAIN-SOURCE CANONICAL EVENTS =========="
if [[ "$GENERATION_RETRAIN_REFINER" == "1" ]]; then
  "$PY" training/motion_models.py \
    --config "$CONFIG" \
    train-refiner \
    --db "$TRAIN_AESD" \
    --val_db "$VAL_AESD" \
    --out "$REFINER_CKPT" \
    --steps "$REFINER_STEPS"
else
  require_file "$REFINER_CKPT" "Motion Refiner checkpoint"
fi

echo "========== 9. TRAIN Motion Generation ON TRAIN-SOURCE CANONICAL EVENTS =========="
if [[ "$GENERATION_RETRAIN_DIFFUSION" == "1" ]]; then
  "$PY" training/motion_models.py \
    --config "$CONFIG" \
    train-diffusion \
    --db "$TRAIN_AESD" \
    --val_db "$VAL_AESD" \
    --out "$MOTION_CKPT" \
    --steps "$MOTION_STEPS" \
    --diffusion_steps "$MOTION_DIFFUSION_STEPS"
else
  require_file "$MOTION_CKPT" "Motion Generation checkpoint"
fi

echo "========== 10. USE VALIDATED GENERATION-ALIGNED SCHEDULER ASSETS =========="
require_file "$ALIGNED_SCHEDULER_DIR/scheduler_asset_bundle.json" \
  "Router/Planner/Duration asset bundle"

echo "========== 11. REBUILD SCHEDULE FROM CURRENT WAV =========="
AUDIO_SHA="$(sha256sum "$AUDIO" | awk '{print $1}')"
export GENERATION_SCHEDULE_RUN_ID="${RUN_TAG}_${AUDIO_SHA:0:12}"
FRESH_RUN_DIR="$SCHEDULE_ROOT/$GENERATION_SCHEDULE_RUN_ID"

FRESH_ARGS=(
  --audio "$AUDIO"
  --out_json "$FRESH_MSSD"
  --run_dir "$FRESH_RUN_DIR"
  --run_id "$GENERATION_SCHEDULE_RUN_ID"
  --router_ckpt "$GENERATION_RESOLVED_ROUTER_CKPT"
  --planner_ckpt "$GENERATION_RESOLVED_PLANNER_CKPT"
  --duration_model_ckpt "$GENERATION_RESOLVED_DURATION_CKPT"
  --index_json "$GENERATION_RESOLVED_INDEX_JSON"
  --duration_index_npz "$GENERATION_RESOLVED_DURATION_INDEX_NPZ"
  --fps "$GENERATION_FPS"
  --min_phrase_seconds "$GENERATION_MIN_PHRASE_SECONDS"
  --max_phrase_seconds "$GENERATION_MAX_PHRASE_SECONDS"
  --max_phrases "$GENERATION_MAX_PHRASES"
  --boundary_quantile "$GENERATION_BOUNDARY_QUANTILE"
  --beat_snap_seconds "$GENERATION_BEAT_SNAP_SECONDS"
  --max_single_event_seconds "$GENERATION_MAX_SINGLE_EVENT_SECONDS"
  --calm_max_single_event_seconds "$GENERATION_CALM_MAX_SINGLE_EVENT_SECONDS"
  --min_subphrase_seconds "$GENERATION_MIN_SUBPHRASE_SECONDS"
  --max_events_per_phrase "$GENERATION_MAX_EVENTS_PER_PHRASE"
  --transition_min_frames "$GENERATION_TRANSITION_MIN_FRAMES"
  --transition_max_frames "$GENERATION_TRANSITION_MAX_FRAMES"
  --max_transition_fraction "$GENERATION_MAX_TRANSITION_FRACTION"
  --transition_budget_min_frames "$GENERATION_TRANSITION_BUDGET_MIN_FRAMES"
  --slot_beat_snap_seconds "$GENERATION_SLOT_BEAT_SNAP_SECONDS"
  --beam_size "$GENERATION_BEAM_SIZE"
  --candidate_top_k "$GENERATION_CANDIDATE_TOP_K"
  --graph_node_top_k "$GENERATION_GRAPH_NODE_TOP_K"
  --max_source_share "$ROUTING_SAFETY_MAX_SOURCE_SHARE"
  --max_recording_share "$ROUTING_SAFETY_MAX_RECORDING_SHARE"
  --max_pose_hold_ratio "$GENERATION_MAX_POSE_HOLD_RATIO"
  --min_unique_events "$GENERATION_MIN_UNIQUE_EVENTS"
  --min_core_frame_ratio "$GENERATION_MIN_CORE_FRAME_RATIO"
  --physical_edge_weight "$ROUTING_SAFETY_PHYSICAL_EDGE_WEIGHT"
  --physical_edge_reset_accent "$ROUTING_SAFETY_PHYSICAL_EDGE_RESET_ACCENT"
  --root_height_gap_reference_m "$ROUTING_SAFETY_ROOT_HEIGHT_GAP_REFERENCE_M"
  --root_height_gap_hard_m "$ROUTING_SAFETY_ROOT_HEIGHT_GAP_HARD_M"
  --posture_state_gap_hard "$ROUTING_SAFETY_POSTURE_STATE_GAP_HARD"
  --floor_gap_reference_m "$ROUTING_SAFETY_FLOOR_GAP_REFERENCE_M"
  --floor_gap_hard_m "$ROUTING_SAFETY_FLOOR_GAP_HARD_M"
  --root_velocity_jump_reference_mps "$ROUTING_SAFETY_ROOT_VELOCITY_JUMP_REFERENCE_MPS"
  --root_velocity_jump_hard_mps "$ROUTING_SAFETY_ROOT_VELOCITY_JUMP_HARD_MPS"
  --contact_gap_hard "$ROUTING_SAFETY_CONTACT_GAP_HARD"
  --stage_floor_y "$ROUTING_SAFETY_STAGE_FLOOR_Y"
  --event_floor_quantile "$ROUTING_SAFETY_EVENT_FLOOR_QUANTILE"
  --event_max_floor_penetration_m "$ROUTING_SAFETY_EVENT_MAX_FLOOR_PENETRATION_M"
  --transition_angular_speed_cap_radps "$ROUTING_SAFETY_TRANSITION_ANGULAR_SPEED_CAP_RADPS"
  --transition_root_horizontal_speed_cap_mps "$ROUTING_SAFETY_TRANSITION_ROOT_XZ_SPEED_CAP_MPS"
  --transition_root_vertical_speed_cap_mps "$ROUTING_SAFETY_TRANSITION_ROOT_Y_SPEED_CAP_MPS"
  --transition_root_tangent_margin_m "$ROUTING_SAFETY_TRANSITION_ROOT_TANGENT_MARGIN_M"
  --transition_floor_clearance_m "$ROUTING_SAFETY_TRANSITION_FLOOR_CLEARANCE_M"
  --transition_floor_smoothing_seconds "$ROUTING_SAFETY_TRANSITION_FLOOR_SMOOTH_SECONDS"
  --transition_contact_ramp_seconds "$ROUTING_SAFETY_TRANSITION_CONTACT_RAMP_SECONDS"
  --max_frame_error "$GENERATION_MAX_FRAME_ERROR"
  --max_seconds_error "$GENERATION_MAX_SECONDS_ERROR"
)

if [[ "$ROUTING_SAFETY_PHYSICAL_EDGE_HARD_PRUNE" == "1" ]]; then
  FRESH_ARGS+=(--physical_edge_hard_prune)
else
  FRESH_ARGS+=(--no-physical_edge_hard_prune)
fi

[[ -n "${GENERATION_RESOLVED_HIERARCHY_INDEX_NPZ:-}" ]] && \
  FRESH_ARGS+=(--hierarchy_index_npz "$GENERATION_RESOLVED_HIERARCHY_INDEX_NPZ")
[[ -n "${GENERATION_RESOLVED_START_POSE:-}" ]] && \
  FRESH_ARGS+=(--start_pose "$GENERATION_RESOLVED_START_POSE")
[[ "$GENERATION_REQUIRE_RHYTHM_FEATURES" == "1" ]] && \
  FRESH_ARGS+=(--require_rhythm_features)

"$PY" scheduling/build_schedule.py "${FRESH_ARGS[@]}"

echo "========== 12. FRESH-WAV CONTRACT RECHECK =========="
"$PY" scheduling/validate_schedule.py \
  --audio "$AUDIO" \
  --schedule "$FRESH_MSSD" \
  --required_run_id "$GENERATION_SCHEDULE_RUN_ID" \
  --fps "$GENERATION_FPS" \
  --max_frame_error "$GENERATION_MAX_FRAME_ERROR" \
  --max_seconds_error "$GENERATION_MAX_SECONDS_ERROR" \
  --max_pose_hold_ratio "$GENERATION_MAX_POSE_HOLD_RATIO" \
  --max_single_source_ratio "$ROUTING_SAFETY_MAX_SOURCE_SHARE" \
  --max_single_recording_ratio "$ROUTING_SAFETY_MAX_RECORDING_SHARE" \
  --min_unique_events "$GENERATION_MIN_UNIQUE_EVENTS" \
  --min_core_frame_ratio "$GENERATION_MIN_CORE_FRAME_RATIO" \
  --out "$OUT_ROOT/fresh_schedule.contract.json" \
  --csv "$OUT_ROOT/fresh_schedule.contract.csv"

if [[ "$CURRENT_PROTOCOL_BASELINES_ENABLE" == "1" ]]; then
  echo "========== 12B. EVALUATE CURRENT-PROTOCOL GREEDY + BEAM BASELINES =========="
  "$PY" scripts/evaluate_current_protocol_baselines.py \
    --schedule "$FRESH_MSSD" \
    --db "$GENERATION_DB" \
    --out "$BASELINE_DIR/current_protocol_routes.json" \
    --candidate_top_k "$BOUNDARY_RESELECT_TOPK" \
    --beam_size "$CURRENT_PROTOCOL_BASELINE_BEAM_SIZE"
fi

ROUTING_MSSD="$FRESH_MSSD"
echo "========== 13. Fresh-Audio Generation HEADING/BOUNDARY CLOSED-LOOP GENERATION =========="
"$PY" routing/boundary_closed_loop.py \
  generate \
  --config "$CONFIG" \
  --audio "$AUDIO" \
  --slots_json "$ROUTING_MSSD" \
  --db "$GENERATION_DB" \
  --refiner "$REFINER_CKPT" \
  --diffusion "$MOTION_CKPT" \
  --out "$FINAL_NPY" \
  --json "$FINAL_REPORT"

echo "========== 13B. FORMAL GRAPH-SB FAIL-CLOSED ACCEPTANCE =========="
"$PY" evaluation/validate_formal_route.py \
  --report "$FINAL_REPORT" \
  --out "$OUT_ROOT/final.graph_sb.acceptance.json"

echo "========== 14. FINAL GRAVITY AUDIT =========="
"$PY" evaluation/audit_gravity.py \
  --input "$FINAL_NPY" \
  --fps "$GENERATION_FPS" \
  --out "$OUT_ROOT/final.gravity.json" \
  --csv "$OUT_ROOT/final.gravity.csv"

echo "========== 15. FINAL HEADING-PLAN AUDIT =========="
"$PY" evaluation/audit_heading.py \
  --motion "$FINAL_NPY" \
  --report "$FINAL_REPORT" \
  --db "$GENERATION_DB" \
  --fps "$GENERATION_FPS" \
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
  --fps "$GENERATION_FPS" \
  --camera_mode fixed \
  --render_smooth_window 1 \
  --gravity_audit_json "$OUT_ROOT/final.render_gravity.json"

echo "========== Fresh-Audio Generation COMPLETE =========="
printf "FRESH_MSSD=%s\nROUTING_MSSD=%s\nGENERATION_DB=%s\nFINAL_NPY=%s\nFINAL_REPORT=%s\nFINAL_MP4=%s\n" \
  "$FRESH_MSSD" "$ROUTING_MSSD" "$GENERATION_DB" "$FINAL_NPY" "$FINAL_REPORT" "$FINAL_MP4"
ls -lh \
  "$TRAIN_AESD" \
  "$VAL_AESD" \
  "$TEST_AESD" \
  "$REFINER_CKPT" \
  "$MOTION_CKPT" \
  "$FRESH_MSSD" \
  "$ROUTING_MSSD" \
  "$FINAL_NPY" \
  "$FINAL_REPORT" \
  "$FINAL_MP4"
