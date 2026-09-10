#!/usr/bin/env bash
# Server-only 50-step V15.13-FiLM learnability probe; never pilot or promotion.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${1:-outputs/run_smpl14_formal_20260822_163915}"
PROBE_STEPS=50

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -s "$OUT_ROOT/event_db_split/train/events_aesd.npz"
test -s "$OUT_ROOT/event_db_split/val/events_aesd.npz"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_13_observable_film_probe_${STAMP}"
CANDIDATES="$OUT_ROOT/checkpoints/$TAG"
LOG="logs/refiner_v15_13_film_probe_${STAMP}.log"
STATUS="outputs/refiner_v15_13_film_probe_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_13_film_probe_${STAMP}.diagnostic_status.txt"

mkdir -p logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_13_FILM_PROBE_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_13_FILM_PROBE_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_13_FILM_PROBE_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" \
  > outputs/LATEST_REFINER_V15_13_FILM_PROBE_DIAGNOSTIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "started_at=$(date --iso-8601=seconds)"

export MOTION_DEVICE=cuda
export MOTION_GPU_PREPROCESSING=1
export MOTION_PRODUCT_REFINER_FILM_CONDITIONING=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1

"$PY" -m pytest -q \
  tests/test_refiner_full_bank_diagnostic.py \
  tests/test_refiner_optimizer.py \
  tests/test_refiner_local_context.py \
  tests/test_zero_edit_retraction.py
"$PY" -m ruff check --select E9,F63,F7,F82 \
  training/motion_models.py \
  training/refiner_bridge_diagnostics.py \
  training/refiner_optimizer.py \
  tests/test_refiner_full_bank_diagnostic.py \
  tests/test_refiner_local_context.py

PY="$PY" EXPECTED_COMMIT="$EXPECTED_COMMIT" \
  bash scripts/train_refiner_v8.sh foundation "$OUT_ROOT" "$TAG"

ROOT_DIR=$(pwd)
export PY PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export PROJECT_ROOT="$ROOT_DIR" EXPERIMENT_PROFILE=research
source configs/experiment.env
export MOTION_DEVICE=cuda
export MOTION_GPU_PREPROCESSING=1
export MOTION_PRODUCT_REFINER_FILM_CONDITIONING=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1

TRAIN_DB="$OUT_ROOT/event_db_split/train/events_aesd.npz"
VAL_DB="$OUT_ROOT/event_db_split/val/events_aesd.npz"
FOUNDATION="$CANDIDATES/foundation_diagnostic/foundation_report.json"
FIT_DIR="$CANDIDATES/bridge_diagnostic"

exec 9>"$CANDIDATES/.training.lock"
flock -n 9
set +e
"$PY" -u -m training.refiner_bridge_diagnostics \
  --config configs/motion_model.json \
  --db "$TRAIN_DB" \
  --val_db "$VAL_DB" \
  --out_dir "$FIT_DIR" \
  --windows 8 \
  --steps "$PROBE_STEPS" \
  --eval_every "$PROBE_STEPS" \
  --foundation_report "$FOUNDATION"
DIAG_STATUS=$?
set -e
printf '%s\n' "$DIAG_STATUS" > "$SCIENTIFIC_STATUS"

if [[ "$DIAG_STATUS" -ne 0 && "$DIAG_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.13-FiLM probe execution failed with status $DIAG_STATUS"
  exit "$DIAG_STATUS"
fi

REPORT="$FIT_DIR/diagnostic_report.json"
test -s "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
history = report.get("history", [])
latest = history[-1] if history else {}
fit = latest.get("fit_context_readiness", {})
updates = report.get("optimizer_updates", {})
multiobjective = report.get("multiobjective_diagnostics", {})
last_mgda = multiobjective.get("last") or {}
amplitude = report.get("output_amplitude_diagnostics", {})
print(json.dumps({
    "schema": report.get("schema"),
    "completed_steps": report.get("completed_steps"),
    "diagnostic_ready": report.get("diagnostic_ready"),
    "accepted_steps": updates.get("accepted_steps"),
    "attempted_steps": updates.get("attempted_steps"),
    "group_guard_rejection_reasons": updates.get(
        "group_guard_rejection_reasons"
    ),
    "feasibility_blockers_by_category": updates.get(
        "group_guard_rejection_categories"
    ),
    "resolution_limited_steps": updates.get(
        "resolution_limited_steps"
    ),
    "accepted_step_scale_distribution": updates.get(
        "accepted_step_scale_distribution"
    ),
    "minimum_audited_scale": updates.get("minimum_audited_scale"),
    "minimum_accepted_scale": updates.get("minimum_accepted_scale"),
    "common_descent_steps": multiobjective.get(
        "common_descent_steps"
    ),
    "pareto_stationary_steps": multiobjective.get(
        "pareto_stationary_steps"
    ),
    "constrained_common_descent_steps": multiobjective.get(
        "constrained_common_descent_steps"
    ),
    "no_constrained_common_descent_steps": multiobjective.get(
        "no_constrained_common_descent_steps"
    ),
    "active_constraint_counts": multiobjective.get(
        "active_constraint_counts"
    ),
    "last_mgda_min_norm": last_mgda.get("mgda_min_norm"),
    "last_common_descent_exists": last_mgda.get(
        "common_descent_exists"
    ),
    "last_mgda_weights": last_mgda.get("mgda_weights"),
    "last_directional_derivatives": last_mgda.get(
        "directional_derivatives"
    ),
    "last_constraint_directional_derivatives": last_mgda.get(
        "constraint_directional_derivatives"
    ),
    "last_constraint_metrics": last_mgda.get("constraint_metrics"),
    "last_output_amplitude_by_group": amplitude.get("by_group"),
    "last_film_modulation": amplitude.get("film_modulation"),
    "conditional_modulation": report.get("conditional_modulation"),
    "fit_contexts_passed": fit.get("contexts_passed"),
    "fit_contexts_evaluated": fit.get("contexts_evaluated"),
    "fit_context_pass_rate": fit.get("pass_rate"),
    "learning_scope_diagnosis": latest.get("learning_scope_diagnosis"),
    "candidate_audit_jsonl": str(
        Path(sys.argv[1]).with_name("optimizer_updates.jsonl")
    ),
}, ensure_ascii=False, indent=2))
PY

echo "V15.13-FiLM 50-step probe completed; diagnostic_status=$DIAG_STATUS"
echo "No pilot, formal training, promotion, or generation was launched."

