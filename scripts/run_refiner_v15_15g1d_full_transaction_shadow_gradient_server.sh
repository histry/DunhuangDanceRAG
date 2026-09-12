#!/usr/bin/env bash
# Server-only V15.15g1d full-transaction shadow-gradient repair probe.
# It preserves the 2/3/5 budget, starts no training, and recycles no validation.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
VALIDATION_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
test -s "$ADAPTER_STATE"
test -s "$TRAIN_BANK"
test -s "$VALIDATION_BANK"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g1d_full_transaction_shadow_gradient_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
PROBE_DIR="$ROOT/full_transaction_shadow_gradient_probe"
LOG="logs/refiner_v15_15g1d_full_transaction_shadow_gradient_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1d_full_transaction_shadow_gradient_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15g1d_full_transaction_shadow_gradient_${STAMP}.scientific_status.txt"

mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1D_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1D_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15G1D_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15G1D_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "adapter_state=$ADAPTER_STATE"
echo "train_bank=$TRAIN_BANK"
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
  --train-teacher-bank "$TRAIN_BANK" \
  --validation-teacher-bank "$VALIDATION_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --output-dir "$PROBE_DIR" \
  --activation-aware-g1d \
  --severity-envelope-absolute-margin 1e-6 \
  --severity-scale-floor 1e-6 \
  --severity-conformal-shrinkage 0.1 \
  --severity-conformal-uncertainty-fraction 0.05 \
  --guard-proxy-nonregression-tolerance 1e-6 \
  --guard-smooth-max-temperature 1e-3 \
  --correction-temporal-smoothness-weight 0.05 \
  --guard-minimum-reduction 1e-12 \
  --guard-safe-interior-margin 1e-6 \
  --full-shadow-lse-allowance-fraction 0.01 \
  --full-shadow-lse-temperature-floor 1e-8 \
  --full-shadow-projection-damping 1e-12 \
  --steps 2 3 5 \
  --target-rms 1e-4
RUN_STATUS=$?
set -e
printf '%s\n' "$RUN_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$RUN_STATUS" -ne 0 && "$RUN_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15g1d failed with status $RUN_STATUS"
  exit "$RUN_STATUS"
fi

REPORT="$PROBE_DIR/fixed_budget_correction.report.json"
ENVELOPE="$PROBE_DIR/discriminative_transaction_conformal_severity.json"
TRAIN_SHADOW_CONTRACT="$PROBE_DIR/train_frozen_full_shadow_repair_contract.json"
test -s "$REPORT"
test -s "$ENVELOPE"
test -s "$TRAIN_SHADOW_CONTRACT"
printf '%s\n' "$REPORT" > outputs/LATEST_REFINER_V15_15G1D_REPORT
printf '%s\n' "$ENVELOPE" > outputs/LATEST_REFINER_V15_15G1D_DISCRIMINATIVE_CONFORMAL
printf '%s\n' "$TRAIN_SHADOW_CONTRACT" > outputs/LATEST_REFINER_V15_15G1D_TRAIN_SHADOW_CONTRACT
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
a = r.get("activation_aware_summary") or {}
decisions = a.get("decisions") or {}
case_53_uids = sorted(uid for uid in decisions if uid.endswith(":53"))
case_53_uid = case_53_uids[0] if len(case_53_uids) == 1 else None
print(json.dumps({
    "schema": r.get("schema"),
    "activation_aware_supported": r.get("activation_aware_supported"),
    "selection_protocol": a.get("selection_protocol"),
    "single_conformal_distance_threshold": a.get(
        "single_conformal_distance_threshold"
    ),
    "cross_conformal_distance_threshold": a.get(
        "cross_conformal_distance_threshold"
    ),
    "activation_discriminant_threshold": a.get(
        "activation_discriminant_threshold"
    ),
    "conformal_class_overlap_in_calibration": a.get(
        "conformal_class_overlap_in_calibration"
    ),
    "conformal_fallback_case_uids": a.get(
        "conformal_fallback_case_uids"
    ),
    "selected_method_counts": a.get("selected_method_counts"),
    "adapter_incumbent_locked_case_uids": sorted(
        uid for uid, decision in decisions.items()
        if decision.get("adapter_incumbent_locked")
    ),
    "selected_projected_count_by_group": a.get(
        "selected_projected_count_by_group"
    ),
    "false_activation_case_uids": a.get("false_activation_case_uids"),
    "missed_cross_case_uids": a.get("missed_cross_case_uids"),
    "cross_exact_closure_complete": a.get("cross_exact_closure_complete"),
    "single_identity_safe": a.get("single_identity_safe"),
    "selected_fixed_guard_shadow_complete": a.get(
        "selected_fixed_guard_shadow_complete"
    ),
    "train_frozen_full_shadow_repair_contract": r.get(
        "train_frozen_full_shadow_repair_contract"
    ),
    "held_out_validation_evaluation_passes": r.get(
        "held_out_validation_evaluation_passes"
    ),
    "case_53_selection": (
        decisions.get(case_53_uid) if case_53_uid is not None else None
    ),
    "case_53_correction_by_variant": (
        {
            name: (variant.get("correction_by_case") or {}).get(case_53_uid)
            for name, variant in (r.get("variants") or {}).items()
            if name != "adapter"
        }
        if case_53_uid is not None else None
    ),
    "scope_safe": a.get("scope_safe"),
    "numeric_audit_complete": a.get("numeric_audit_complete"),
    "fixed_guard_thresholds_changed": r.get(
        "fixed_guard_thresholds_changed"
    ),
    "observable_0p03_gate_changed": r.get("observable_0p03_gate_changed"),
}, ensure_ascii=False, indent=2))
PY
echo "V15.15g1d full-transaction shadow-gradient probe completed; scientific_status=$RUN_STATUS"
echo "No training, pseudo-teacher recycling, promotion, replay, or generation was launched."
