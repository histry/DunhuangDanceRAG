#!/usr/bin/env bash
# Foreground-only, isolated-candidate training. Never rebuild source/scheduler assets.
set -Eeuo pipefail

if [[ $# -ne 3 || ! "$1" =~ ^(diagnose|pilot|resume)$ || ! "$3" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Usage: bash scripts/train_refiner_v6.sh diagnose|pilot|resume EXISTING_OUT_ROOT CANDIDATE_TAG" >&2
  exit 2
fi
MODE=$1
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$(cd "$2" && pwd)"
TAG=$3
cd "$ROOT_DIR"
PY="${PY:-python}"
export PY
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

test -z "$(git status --porcelain)" || { echo "[FATAL] Repository must be clean" >&2; exit 2; }
if [[ -n "${EXPECTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "[FATAL] Unexpected commit" >&2
  exit 2
fi
if pgrep -af '[t]raining/motion_models.py|[t]raining.motion_models|[t]raining.refiner_diagnostics|[r]un_official_smpl_full.sh|[r]esearch_pipeline.sh|[s]cripts/pipeline.sh'; then
  echo "[FATAL] Another training/pipeline process is active" >&2
  exit 2
fi

unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export PROJECT_ROOT="$ROOT_DIR"
export EXPERIMENT_PROFILE=research
source configs/experiment.env
export MOTION_DEVICE=cuda
export MOTION_GPU_PREPROCESSING=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1
export MOTION_DIFFUSION_NOISE_BATCH_MAX_MIB=256
export REQUIRE_LIBROSA_BACKEND=1
export GENERATION_DEEP_MUSIC_FEATURES=0
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
flock -n 9 || { echo "[FATAL] This candidate is already running" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"

# Training-only diagnosis is a prerequisite, never a warm-start checkpoint.
FIT_DIR="$CANDIDATES/fixed_fit"
FIT_REPORT="$FIT_DIR/diagnostic_report.json"
FIT_ARGS=(--config configs/motion_model.json --db "$TRAIN_DB" --val_db "$VAL_DB")
if [[ "$MODE" == diagnose ]]; then
  "$PY" -u -m training.refiner_diagnostics "${FIT_ARGS[@]}" \
    --out_dir "$FIT_DIR" --windows 8 --steps 400 --eval_every 50 --gradient_every 25 \
    2>&1 | tee "$CANDIDATES/console.fixed_fit_$STAMP.log"
  echo "[DIAGNOSTIC ONLY] Inspect $FIT_REPORT before starting a fresh pilot."
  exit 0
fi
test -s "$FIT_REPORT" || { echo "[FATAL] Run diagnose first with this tag" >&2; exit 2; }
"$PY" -m training.refiner_diagnostics "${FIT_ARGS[@]}" --check_report "$FIT_REPORT"

REFINER_ARGS=(--config configs/motion_model.json train-refiner
  --db "$TRAIN_DB" --val_db "$VAL_DB" --out "$REFINER" --steps 8000
  --snapshot_path "$SNAPSHOT" --snapshot_every 200 --validation_every 1000
  --train_probe_windows 8)
if [[ "$MODE" == pilot ]]; then
  if [[ -e "$SNAPSHOT" || -e "$REFINER" || -e "$CANDIDATES/boundary_refiner.best_validation.pt" ]]; then
    echo "[FATAL] Candidate already exists; use a new tag or resume the matching V6 snapshot" >&2
    exit 2
  fi
  "$PY" -u -m training.motion_models "${REFINER_ARGS[@]}" --stop_after_steps 1000 \
    2>&1 | tee "$CANDIDATES/console.pilot_$STAMP.log"
  echo "[PILOT ONLY] Paused at 1000/8000; no formal model was published."
  echo "Report: $CANDIDATES/boundary_refiner.validation_step_001000.json"
  echo "Inspect train-fit and source-disjoint validation before using resume with this same tag."
  exit 0
fi

test -s "$SNAPSHOT" || { echo "[FATAL] V6 pilot snapshot missing" >&2; exit 2; }
"$PY" -u -m training.motion_models "${REFINER_ARGS[@]}" --resume_snapshot "$SNAPSHOT" \
  2>&1 | tee "$CANDIDATES/console.refiner_resume_$STAMP.log"
test -s "$REFINER"

DIFF_ARGS=(--config configs/motion_model.json train-diffusion
  --db "$TRAIN_DB" --val_db "$VAL_DB" --out "$DIFFUSION" --steps 15000
  --diffusion_steps 50 --snapshot_path "$DIFF_SNAPSHOT" --snapshot_every 200)
if [[ -s "$DIFF_SNAPSHOT" ]]; then
  DIFF_ARGS+=(--resume_snapshot "$DIFF_SNAPSHOT")
fi
"$PY" -u -m training.motion_models "${DIFF_ARGS[@]}" \
  2>&1 | tee "$CANDIDATES/console.diffusion_$STAMP.log"

# Reject stale, nonformal, or scientifically rejected files before promotion.
"$PY" - "$REFINER" "$DIFFUSION" <<'PY'
import subprocess
import sys
from training.motion_models import _trusted_torch_load, REFINER_MODEL_VERSION

revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
for path, version in zip(sys.argv[1:], [REFINER_MODEL_VERSION, "reference_tangent_motion_diffusion_v2"]):
    payload = _trusted_torch_load(path, map_location="cpu")
    assert payload["version"] == version, path
    assert payload["training_resume"]["runtime_code_revision"] == revision, path
    decision = payload["validation"]["checkpoint_decision"]
    assert decision["scientific_acceptance"] and decision["publish_allowed"], decision
    assert not decision["reasons"], decision
    print("ACCEPTED:", path, flush=True)
PY

# Preserve both historical formal files until BOTH new candidates pass.
for NAME in boundary_refiner.pt local_diffusion.pt; do
  DEST="$OUT_ROOT/checkpoints/$NAME"
  if [[ -e "$DEST" ]]; then
    cp -p "$DEST" "$CANDIDATES/$NAME.before_promotion_$STAMP"
  fi
  cp -p "$CANDIDATES/$NAME" "$DEST.candidate_$STAMP"
  mv -f "$DEST.candidate_$STAMP" "$DEST"
done
echo "[ACCEPTED] Refiner and Diffusion published. This does not mean final video gates have passed."
echo "Candidate logs, validation reports and previous-model backups: $CANDIDATES"
