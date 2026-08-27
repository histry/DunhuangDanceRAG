#!/usr/bin/env bash
# Foreground, diagnostic-only. No cache rebuild, formal resume, or promotion.
set -Eeuo pipefail
if [[ $# -ne 3 || ! "$2" =~ ^[a-zA-Z0-9_-]+$ || ! "$3" =~ ^[a-zA-Z0-9_-]+$ || "$2" == "$3" ]]; then
  echo "Usage: bash scripts/diagnose_refiner_factors.sh EXISTING_OUT_ROOT COHORT_TAG NEW_EXPERIMENT_TAG" >&2
  exit 2
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$(cd "$1" && pwd)"
COHORT="$OUT_ROOT/checkpoints/$2"
EXPERIMENT="$OUT_ROOT/refiner_factor_diagnostics/$3"
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
for INPUT in "$TRAIN_DB" "$VAL_DB" "$COHORT/fixed_fit/diagnostic_report.json" \
  "$COHORT/fixed_fit/fixed_training_batch.npz" "$COHORT/boundary_refiner.training_snapshot.pt"; do
  test -s "$INPUT" || { echo "[FATAL] Missing: $INPUT" >&2; exit 2; }
done
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name(0), flush=True)'
mkdir -p "$OUT_ROOT/refiner_factor_diagnostics"
exec 9>"$OUT_ROOT/refiner_factor_diagnostics/.diagnostic.lock"
flock -n 9 || { echo "[FATAL] A factor diagnosis is already running" >&2; exit 2; }
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/refiner_factor_diagnostics/console.$3.$STAMP.log"
echo "Code: $ACTUAL_COMMIT"
echo "Reference snapshot is read-only: $COHORT/boundary_refiner.training_snapshot.pt"
echo "Log: $LOG"
"$PY" -u -m training.refiner_factor_diagnostics \
  --config configs/motion_model.json --db "$TRAIN_DB" --val_db "$VAL_DB" \
  --fixed_fit_dir "$COHORT/fixed_fit" \
  --reference_snapshot "$COHORT/boundary_refiner.training_snapshot.pt" \
  --out_dir "$EXPERIMENT" --windows 8 --steps 400 --eval_every 100 \
  --counterfactual_windows 4 2>&1 | tee "$LOG"
echo "[DIAGNOSTIC COMPLETE] $EXPERIMENT/factor_report.json"
echo "No formal model was published. Review the factors before any full training."
