#!/usr/bin/env bash
# Server-only activation-aware V15.15g probe; no training or teacher recycling.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
VALIDATION_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
test -s "$ADAPTER_STATE"
test -s "$VALIDATION_BANK"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g_activation_aware_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
PROBE_DIR="$ROOT/activation_aware_probe"
LOG="logs/refiner_v15_15g_activation_aware_${STAMP}.log"
STATUS="outputs/refiner_v15_15g_activation_aware_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15g_activation_aware_${STAMP}.scientific_status.txt"

mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G_ACTIVATION_AWARE_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G_ACTIVATION_AWARE_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15G_ACTIVATION_AWARE_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15G_ACTIVATION_AWARE_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "adapter_state=$ADAPTER_STATE"
echo "validation_bank=$VALIDATION_BANK"
echo "started_at=$(date --iso-8601=seconds)"

ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export PROJECT_ROOT="$ROOT_DIR" EXPERIMENT_PROFILE=research
source configs/experiment.env
export MOTION_DEVICE=cuda
export MOTION_GPU_PREPROCESSING=1
export MOTION_PRODUCT_REFINER_FILM_CONDITIONING=1
export MOTION_PRODUCT_REFINER_OBSERVABLE_ADAPTER=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  --config configs/motion_model.json \
  --validation-teacher-bank "$VALIDATION_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --output-dir "$PROBE_DIR" \
  --activation-aware \
  --activation-proxy-tolerance 1e-6 \
  --activation-relative-improvement 1e-3 \
  --steps 2 3 5 \
  --target-rms 1e-4
RUN_STATUS=$?
set -e
printf '%s\n' "$RUN_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$RUN_STATUS" -ne 0 && "$RUN_STATUS" -ne 2 ]]; then
  echo "[FATAL] activation-aware V15.15g failed with status $RUN_STATUS"
  exit "$RUN_STATUS"
fi

REPORT="$PROBE_DIR/fixed_budget_correction.report.json"
test -s "$REPORT"
printf '%s\n' "$REPORT" > outputs/LATEST_REFINER_V15_15G_ACTIVATION_AWARE_REPORT
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
a = r.get("activation_aware_summary") or {}
cross_long = {
    uid: row
    for uid, row in (a.get("decisions") or {}).items()
    if row.get("evaluation_only", {}).get("audit_group") == "cross_long"
}
print(json.dumps({
    "schema": r.get("schema"),
    "activation_aware_supported": r.get(
        "activation_aware_supported"
    ),
    "selected_method_counts": a.get("selected_method_counts"),
    "required_projected_count_by_group": a.get(
        "required_projected_count_by_group"
    ),
    "selected_projected_count_by_group": a.get(
        "selected_projected_count_by_group"
    ),
    "false_activation_case_uids": a.get(
        "false_activation_case_uids"
    ),
    "missed_cross_case_uids": a.get("missed_cross_case_uids"),
    "cross_long_decisions": {
        uid: {
            "selected_method": row.get("selected_method"),
            "selected_nonzero": row.get("selected_nonzero"),
        }
        for uid, row in cross_long.items()
    },
    "single_control_applied_tangent_rms_max": a.get(
        "single_control_applied_tangent_rms_max"
    ),
    "cross_exact_closure_complete": a.get(
        "cross_exact_closure_complete"
    ),
    "scope_safe": a.get("scope_safe"),
    "numeric_audit_complete": a.get("numeric_audit_complete"),
    "fixed_guard_thresholds_changed": r.get(
        "fixed_guard_thresholds_changed"
    ),
}, ensure_ascii=False, indent=2))
PY
echo "Activation-aware V15.15g completed; scientific_status=$RUN_STATUS"
echo "No training, pseudo-teacher recycling, promotion, replay, or generation was launched."
