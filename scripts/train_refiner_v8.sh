#!/usr/bin/env bash
# Foundation + current V10 local Refiner. Keep this entrypoint for existing runbooks.
# Existing source and scheduler assets are read-only.
set -Eeuo pipefail
if [[ $# -ne 3 || ! "$1" =~ ^(foundation|diagnose|pilot|resume|generate)$ || ! "$3" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Usage: bash scripts/train_refiner_v8.sh foundation|diagnose|pilot|resume|generate EXISTING_OUT_ROOT NEW_TAG" >&2
  exit 2
fi
MODE=$1
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$(cd "$2" && pwd)"
TAG=$3
cd "$ROOT_DIR"
PY="${PY:-python}"
export PY PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -z "$(git status --porcelain)" || { echo "[FATAL] Repository must be clean" >&2; exit 2; }
if [[ -z "${EXPECTED_COMMIT:-}" || "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "[FATAL] Set EXPECTED_COMMIT to the reviewed commit" >&2
  exit 2
fi
if pgrep -af '[t]raining/motion_models.py|[t]raining.motion_models|[t]raining.refiner_.*diagnostics|[r]un_official_smpl_full.sh|[r]esearch_pipeline.sh|[s]cripts/pipeline.sh'; then
  echo "[FATAL] Another training/pipeline process is active" >&2
  exit 2
fi
unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export PROJECT_ROOT="$ROOT_DIR" EXPERIMENT_PROFILE=research
source configs/experiment.env
export MOTION_DEVICE=cuda MOTION_GPU_PREPROCESSING=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1
export MOTION_DIFFUSION_NOISE_BATCH_MAX_MIB=256
export REQUIRE_LIBROSA_BACKEND=1 GENERATION_DEEP_MUSIC_FEATURES=0
export GRAPH_ROUTE_SB_ALLOW_LEGACY_FALLBACK=0
TRAIN_DB="$OUT_ROOT/event_db_split/train/events_aesd.npz"
VAL_DB="$OUT_ROOT/event_db_split/val/events_aesd.npz"
CANDIDATES="$OUT_ROOT/checkpoints/$TAG"
REFINER="$CANDIDATES/boundary_refiner.pt"
DIFFUSION="$CANDIDATES/local_diffusion.pt"
SNAPSHOT="$CANDIDATES/boundary_refiner.training_snapshot.pt"
DIFF_SNAPSHOT="$CANDIDATES/local_diffusion.training_snapshot.pt"
test -s "$TRAIN_DB"
test -s "$VAL_DB"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
mkdir -p "$CANDIDATES"
exec 9>"$CANDIDATES/.training.lock"
flock -n 9 || { echo "[FATAL] Candidate already running" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
FIT_DIR="$CANDIDATES/bridge_diagnostic"
FIT_ARGS=(--config configs/motion_model.json --db "$TRAIN_DB" --val_db "$VAL_DB")
FOUNDATION="$CANDIDATES/foundation_diagnostic/foundation_report.json"
if [[ "$MODE" == foundation ]]; then
  "$PY" -u -m training.refiner_bridge_diagnostics "${FIT_ARGS[@]}" \
    --out_dir "$CANDIDATES/foundation_diagnostic" --windows 8 --baseline_only --direct_steps 200 \
    2>&1 | tee "$CANDIDATES/console.foundation_$STAMP.log"
  echo "[CONTROL ONLY] Review $FOUNDATION. This is not generalization or formal acceptance."
  exit 0
fi
if [[ "$MODE" == diagnose ]]; then
  echo "[DIAGNOSTIC] Up to 400 neural fitting steps, not an Internet/download check."
  echo "[OPTIMIZER] Complete 32-case TRAIN bank, curvature-aware sufficient decrease; no probe fitting."
  echo "[PROBE] New cuts also change local motion context; they are not a pure position-shift test."
  echo "[STOP] A fixed-bank search stall saves reports and blocks all later stages."
  echo "[REPORT] $FIT_DIR/summary.json and diagnostic_report.json (also saved on gate rejection)."
  echo "[REPLAY] Exact TRAIN inputs in fit_bank.pt, held-out inputs in probe_bank.pt, retained state in diagnostic_state.pt; diagnostic only."
  "$PY" -u -m training.refiner_bridge_diagnostics "${FIT_ARGS[@]}" \
    --out_dir "$FIT_DIR" --windows 8 --steps 400 --eval_every 200 --foundation_report "$FOUNDATION" \
    2>&1 | tee "$CANDIDATES/console.bridge_diagnostic_$STAMP.log"
  echo "[DIAGNOSTIC ONLY] Review $FIT_DIR/diagnostic_report.json before a fresh pilot."
  exit 0
fi
"$PY" -m training.refiner_bridge_diagnostics "${FIT_ARGS[@]}" \
  --check_report "$FIT_DIR/diagnostic_report.json"
if [[ "$MODE" == generate ]]; then
  "$PY" - "$OUT_ROOT/checkpoints/boundary_refiner.pt" "$OUT_ROOT/checkpoints/local_diffusion.pt" "$REFINER" "$DIFFUSION" <<'PY'
import hashlib,subprocess,sys
from pathlib import Path
from training.motion_models import _trusted_torch_load, REFINER_MODEL_VERSION, DIFFUSION_MODEL_VERSION, BOUNDARY_PROTOCOL
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for path,candidate,version in zip(sys.argv[1:3],sys.argv[3:5],(REFINER_MODEL_VERSION,DIFFUSION_MODEL_VERSION)):
    p=_trusted_torch_load(path,map_location="cpu")
    assert p['version']==version and p['training_resume']['runtime_code_revision']==revision, path
    d=p['validation']['checkpoint_decision']
    assert d['scientific_acceptance'] and d['publish_allowed'] and d['boundary_protocol']==BOUNDARY_PROTOCOL, d
    assert hashlib.sha256(Path(path).read_bytes()).digest()==hashlib.sha256(Path(candidate).read_bytes()).digest(), 'Formal model differs from this run candidate'
    print('CURRENT_CANDIDATE_CONFIRMED:',path,flush=True)
PY
  export OUT_ROOT RUN_TAG="${OUT_ROOT##*/run_}"
  export MUSIC_DIRS="${MUSIC_DIRS:-$ROOT_DIR/assets/music/train}"
  export RETARGET_CLEAN_REBUILD_RETARGET_CACHE=0 RETARGET_CLEAN_REBUILD_EVENT_DB=0
  export RETARGET_CLEAN_RETRAIN_ROUTER=0 RETARGET_CLEAN_RETRAIN_DURATION=0 RETARGET_CLEAN_RETRAIN_PLANNER=0
  export RETARGET_CLEAN_RETRAIN_REFINER=0 RETARGET_CLEAN_RETRAIN_DIFFUSION=0
  export FORMAL_REQUIRE_CLEAN_GIT=1
  bash scripts/run_official_smpl_full.sh \
    "${SMPL_DIR:-$ROOT_DIR/assets/motion/smpl_official_14}" \
    "${AUDIO:-$ROOT_DIR/assets/music/test/audio/dunhuangwu2.wav}" \
    2>&1 | tee "$CANDIDATES/console.generate_$STAMP.log"
  exit 0
fi
REF_ARGS=(--config configs/motion_model.json train-refiner --db "$TRAIN_DB" --val_db "$VAL_DB"
  --out "$REFINER" --steps 8000 --snapshot_path "$SNAPSHOT" --snapshot_every 200
  --validation_every 1000 --train_probe_windows 8)
if [[ "$MODE" == pilot ]]; then
  if [[ -e "$SNAPSHOT" || -e "$REFINER" || -e "$CANDIDATES/boundary_refiner.best_validation.pt" ]]; then
    echo "[FATAL] Pilot exists; do not overwrite. Use resume or a new tag." >&2
    exit 2
  fi
  "$PY" -u -m training.motion_models "${REF_ARGS[@]}" --stop_after_steps 1000 \
    2>&1 | tee "$CANDIDATES/console.pilot_$STAMP.log"
  echo "[PAUSED] Review $CANDIDATES/boundary_refiner.validation_step_001000.json. No formal model published."
  exit 0
fi
# A source-disjoint pilot must pass before spending the full training budget.
"$PY" - "$CANDIDATES/boundary_refiner.validation_step_001000.json" <<'PY'
import json,sys,subprocess
from training.motion_models import MotionGenerationConfig,_checkpoint_validation_decision
r=json.load(open(sys.argv[1],encoding='utf8'))
assert r['code_revision']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
d=_checkpoint_validation_decision(r['validation'],MotionGenerationConfig.from_json('configs/motion_model.json').apply_env(),stage='refiner')
assert d['scientific_acceptance'], d
print('SOURCE_DISJOINT_PILOT_ACCEPTED',flush=True)
PY
"$PY" -u -m training.motion_models "${REF_ARGS[@]}" --resume_snapshot "$SNAPSHOT" \
  2>&1 | tee "$CANDIDATES/console.refiner_resume_$STAMP.log"
test -s "$REFINER"
DIFF_ARGS=(--config configs/motion_model.json train-diffusion --db "$TRAIN_DB" --val_db "$VAL_DB"
  --out "$DIFFUSION" --steps 15000 --diffusion_steps 50 --snapshot_path "$DIFF_SNAPSHOT" --snapshot_every 200)
if [[ -s "$DIFF_SNAPSHOT" ]]; then DIFF_ARGS+=(--resume_snapshot "$DIFF_SNAPSHOT"); fi
"$PY" -u -m training.motion_models "${DIFF_ARGS[@]}" \
  2>&1 | tee "$CANDIDATES/console.diffusion_$STAMP.log"
"$PY" - "$REFINER" "$DIFFUSION" <<'PY'
import subprocess,sys
from training.motion_models import _trusted_torch_load,REFINER_MODEL_VERSION,DIFFUSION_MODEL_VERSION,BOUNDARY_PROTOCOL
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
for path,version in zip(sys.argv[1:],(REFINER_MODEL_VERSION,DIFFUSION_MODEL_VERSION)):
    p=_trusted_torch_load(path,map_location='cpu')
    assert p['version']==version and p['training_resume']['runtime_code_revision']==revision, path
    d=p['validation']['checkpoint_decision']
    assert d['scientific_acceptance'] and d['publish_allowed'] and not d['reasons'], d
    assert d['boundary_protocol']==BOUNDARY_PROTOCOL, d
    print('ACCEPTED:',path,flush=True)
PY
# Both candidates must pass before either replaces an existing formal model.
for NAME in boundary_refiner.pt local_diffusion.pt; do
  DEST="$OUT_ROOT/checkpoints/$NAME"
  if [[ -e "$DEST" ]]; then cp -p "$DEST" "$CANDIDATES/$NAME.before_promotion_$STAMP"; fi
  cp -p "$CANDIDATES/$NAME" "$DEST.candidate_$STAMP"
done
for NAME in boundary_refiner.pt local_diffusion.pt; do
  mv -f "$OUT_ROOT/checkpoints/$NAME.candidate_$STAMP" "$OUT_ROOT/checkpoints/$NAME"
done
echo "[ACCEPTED] New models promoted; final closed-loop/IK/video acceptance is still pending."
