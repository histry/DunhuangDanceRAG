#!/usr/bin/env bash
# Four bounded, foreground diagnostics. Never resumes/publishes formal models.
set -Eeuo pipefail
if [[ $# -ne 4 ]]; then
  echo "Usage: bash scripts/diagnose_refiner_noise_refresh.sh EXISTING_OUT_ROOT COHORT_TAG FACTOR_TAG NEW_TAG" >&2
  exit 2
fi
for TAG in "$2" "$3" "$4"; do
  [[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "[FATAL] Invalid tag" >&2; exit 2; }
done
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$(cd "$1" && pwd)"
COHORT="$OUT_ROOT/checkpoints/$2/fixed_fit"
FACTOR="$OUT_ROOT/refiner_factor_diagnostics/$3"
EXPERIMENT="$OUT_ROOT/refiner_noise_refresh/$4"
cd "$ROOT_DIR"
test -z "$(git status --porcelain)" || { echo "[FATAL] Repository is dirty" >&2; exit 2; }
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ -n "${EXPECTED_COMMIT:-}" && "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "[FATAL] Commit mismatch: actual=$ACTUAL_COMMIT expected=$EXPECTED_COMMIT" >&2
  exit 2
fi
if pgrep -af '[t]raining/motion_models.py|[t]raining.motion_models|[t]raining.refiner_.*diagnostics|[s]cripts/pipeline.sh|[r]esearch_pipeline.sh'; then
  echo "[FATAL] Another training/diagnostic process is running" >&2
  exit 2
fi
test ! -e "$EXPERIMENT" || { echo "[FATAL] Experiment exists; use a new tag" >&2; exit 2; }
PY="${PY:-python}"
export PY PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export PROJECT_ROOT="$ROOT_DIR" EXPERIMENT_PROFILE=research
source configs/experiment.env
export MOTION_DEVICE=cuda MOTION_GPU_PREPROCESSING=1
export GENERATION_DEEP_MUSIC_FEATURES=0 REQUIRE_LIBROSA_BACKEND=1
export GRAPH_ROUTE_SB_ALLOW_LEGACY_FALLBACK=0 MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1
TRAIN_DB="$OUT_ROOT/event_db_split/train/events_aesd.npz"
VAL_DB="$OUT_ROOT/event_db_split/val/events_aesd.npz"
for INPUT in "$TRAIN_DB" "$VAL_DB" "$COHORT/diagnostic_report.json" "$COHORT/fixed_training_batch.npz" \
  "$FACTOR/factor_report.json" "$FACTOR/clean_cohort.npz" "$FACTOR/recipes.npz" "$FACTOR/diagnostic_initial_weights.pt"; do
  test -s "$INPUT" || { echo "[FATAL] Missing: $INPUT" >&2; exit 2; }
done
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name(0), flush=True)'
mkdir -p "$OUT_ROOT/refiner_noise_refresh"
# Same lock as the earlier factor runner: do not race two diagnostic jobs.
exec 9>"$OUT_ROOT/refiner_factor_diagnostics/.diagnostic.lock"
flock -n 9 || { echo "[FATAL] Another diagnosis owns the lock" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/refiner_noise_refresh/console.$4.$STAMP.log"
echo "Code: $ACTUAL_COMMIT"
echo "Four fresh models, 400 updates each; no formal model promotion or Diffusion."
echo "Log: $LOG"
"$PY" -u -m training.refiner_noise_refresh_diagnostics \
  --config configs/motion_model.json --db "$TRAIN_DB" --val_db "$VAL_DB" \
  --fixed_fit_dir "$COHORT" --factor_dir "$FACTOR" --out_dir "$EXPERIMENT" \
  --windows 8 --steps 400 --eval_every 100 2>&1 | tee "$LOG"
echo "[DIAGNOSTIC COMPLETE, NOT MODEL ACCEPTANCE] $EXPERIMENT/comparison.json"
