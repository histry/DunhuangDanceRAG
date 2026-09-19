#!/usr/bin/env bash
# Fresh V14 2,000-step pilot.  Never publishes, promotes or starts generation.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the reviewed V14 commit}"
RUN_TAG="${V14_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TAG="refiner_v14_joint_closure_pilot_${RUN_TAG}"
RUN_ROOT="$OUT_ROOT/checkpoints/$TAG"
TRAIN_DB="$OUT_ROOT/event_db_split/train/events_aesd.npz"
VAL_DB="$OUT_ROOT/event_db_split/val/events_aesd.npz"
OUT="$RUN_ROOT/boundary_refiner.pt"
SNAPSHOT="$RUN_ROOT/boundary_refiner.training_snapshot.pt"
LOG="logs/${TAG}.log"

export PY PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MOTION_DEVICE=cuda MOTION_GPU_PREPROCESSING=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -s "$TRAIN_DB"
test -s "$VAL_DB"

if pgrep -af '[t]raining/motion_models.py.*train-refiner|[t]raining.motion_models.*train-refiner'; then
  echo "[FATAL] Another Refiner training process is active" >&2
  exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "[FATAL] V14 pilot output already exists: $RUN_ROOT" >&2
  exit 2
fi

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
mkdir -p "$RUN_ROOT" logs outputs
printf '%s\n' "$RUN_ROOT" > outputs/LATEST_REFINER_V14_PILOT_ROOT
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V14_PILOT_LOG

"$PY" -u -m training.motion_models \
  --config configs/motion_model.json \
  train-refiner \
  --db "$TRAIN_DB" \
  --val_db "$VAL_DB" \
  --out "$OUT" \
  --steps 8000 \
  --stop_after_steps 2000 \
  --snapshot_path "$SNAPSHOT" \
  --snapshot_every 200 \
  --validation_every 500 \
  --train_probe_windows 8 \
  2>&1 | tee "$LOG"

test -s "$SNAPSHOT"
test -s "$RUN_ROOT/boundary_refiner.validation_step_002000.json"
test ! -e "$OUT"

"$PY" - "$RUN_ROOT/boundary_refiner.validation_step_002000.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
decision = report["checkpoint_decision"]
observed = decision["observed"]
print(json.dumps({
    "stage": "refiner_v14_joint_closure_pilot_paused",
    "completed_steps": report["completed_steps"],
    "scientific_acceptance": decision["scientific_acceptance"],
    "observable_joint_closure_count": observed.get(
        "observable_joint_closure_count"
    ),
    "observable_joint_closure_rate": observed.get(
        "observable_joint_closure_rate"
    ),
    "cross_event_joint_closure_count": observed.get(
        "cross_event_joint_closure_count"
    ),
    "cross_event_joint_closure_rate": observed.get(
        "cross_event_joint_closure_rate"
    ),
    "reasons": decision["reasons"],
    "published": False,
    "full_run_authorized": False,
}, ensure_ascii=False))
PY

echo "[PAUSED] Review 500/1000/1500/2000 reports in $RUN_ROOT"
echo "[NO PROMOTION] V14 pilot did not publish a Refiner or start generation."
