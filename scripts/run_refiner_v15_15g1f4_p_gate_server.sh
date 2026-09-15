#!/usr/bin/env bash
# Server-only Gate 0 parity followed by one P-only train calibration run.
# M is intentionally unavailable until train-only equivalent-radius calibration.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
: "${V9_REFERENCE_REPORT:?Set V9_REFERENCE_REPORT to the completed be71ae12 v9 report}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -s "$V9_REFERENCE_REPORT"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
for path in "$ADAPTER_STATE" "$TRAIN_BANK"; do test -s "$path"; done

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g1f4_p_gate_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
LOG="logs/refiner_v15_15g1f4_p_gate_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1f4_p_gate_${STAMP}.exit_status.txt"
mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1F4_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1F4_LOG
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"' EXIT
exec > >(tee -a "$LOG") 2>&1

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

COMMON_ARGS=(
  --config configs/motion_model.json
  --train-teacher-bank "$TRAIN_BANK"
  --adapter-state "$ADAPTER_STATE"
  --activation-aware-g1f3
  --severity-envelope-absolute-margin 1e-6
  --severity-scale-floor 1e-6
  --severity-conformal-shrinkage 0.1
  --severity-conformal-uncertainty-fraction 0.05
  --guard-proxy-nonregression-tolerance 1e-6
  --guard-smooth-max-temperature 1e-3
  --correction-temporal-smoothness-weight 0.05
  --guard-minimum-reduction 1e-12
  --guard-safe-interior-margin 1e-6
  --full-shadow-lse-allowance-fraction 0.01
  --full-shadow-lse-temperature-floor 1e-8
  --full-shadow-projection-damping 1e-12
  --full-shadow-line-search-backtracks 12
  --full-shadow-line-search-decay 0.5
  --science-restoration-damping 1e-8
  --science-restoration-safety-fraction 0.25
  --geodesic-angular-max-radians 0.7853981633974483
  --joint-svd-relative-cutoff 1e-6
  --joint-direction-norm-floor 1e-8
  --joint-directional-margin 1e-6
  --temporal-fd-epsilon-radians 1e-4
  --temporal-fd-relative-error-tolerance 0.1
  --temporal-fd-absolute-floor 1e-8
  --temporal-fd-near-zero-threshold 1e-5
  --second-order-basis-dimension 5
  --second-order-grid-levels 9
  --second-order-feasibility-tolerance 1e-12
  --second-order-guard-transition-band 1e-5
  --target-rms 1e-4
  --validation-teacher-bank "$TRAIN_BANK"
  --evaluation-role train_calibration
  --metric-mode identity
)

BASELINE_DIR="$ROOT/current_equal_share_identity"
set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --steps 2 3 5 \
  --progress-mode current_equal_share \
  --output-dir "$BASELINE_DIR"
BASELINE_STATUS=$?
set -e
if test "$BASELINE_STATUS" -ne 0 && test "$BASELINE_STATUS" -ne 2; then
  echo "baseline execution failed unexpectedly: $BASELINE_STATUS" >&2
  exit "$BASELINE_STATUS"
fi

BASELINE_REPORT="$BASELINE_DIR/fixed_budget_correction.report.json"
PARITY_REPORT="$ROOT/v9_canonical_parity_gate.json"
test -s "$BASELINE_REPORT"
"$PY" -m training.refiner_v15_15g1f4_parity \
  --v9-reference-report "$V9_REFERENCE_REPORT" \
  --candidate-report "$BASELINE_REPORT" \
  --output "$PARITY_REPORT"
printf '%s\n' "$PARITY_REPORT" > outputs/LATEST_REFINER_V15_15G1F4_PARITY_REPORT

P_DIR="$ROOT/weighted_debt_filter_identity"
set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --steps 5 \
  --progress-mode weighted_debt_filter \
  --diagnostic-case-uid txn_0001_97ecf5fd6e62:169 \
  --diagnostic-case-uid txn_0005_a6fbd294b71c:169 \
  --diagnostic-case-uid txn_0007_0d8eea4df4f1:137 \
  --output-dir "$P_DIR"
P_STATUS=$?
set -e
P_REPORT="$P_DIR/fixed_budget_correction.report.json"
test -s "$P_REPORT"
printf '%s\n' "$P_REPORT" > outputs/LATEST_REFINER_V15_15G1F4_P_REPORT

P_SUMMARY="$ROOT/p_only_three_target_k5.summary.json"
"$PY" - "$V9_REFERENCE_REPORT" "$P_REPORT" "$P_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

reference = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
candidate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
output = Path(sys.argv[3])
uids = (
    "txn_0001_97ecf5fd6e62:169",
    "txn_0005_a6fbd294b71c:169",
    "txn_0007_0d8eea4df4f1:137",
)

def cases(report):
    return report["variants"]["geodesic_joint_sqp_k5"]["correction_by_case"]

def shadow(row):
    history = row.get("history") or []
    return history[-1].get("full_shadow_after") if history else None

rows = {}
for uid in uids:
    old = cases(reference)[uid]
    new = cases(candidate)[uid]
    rows[uid] = {
        "v9_accepted_steps": old.get("accepted_steps"),
        "p_accepted_steps": new.get("accepted_steps"),
        "v9_second_order_state": old.get("second_order_state"),
        "p_second_order_state": new.get("second_order_state"),
        "v9_final_shadow": shadow(old),
        "p_final_shadow": shadow(new),
        "accepted_additional_step": (
            int(new.get("accepted_steps", 0)) > int(old.get("accepted_steps", 0))
        ),
    }

value = {
    "schema": "refiner_v15_15g1f4_p_three_target_k5_diagnostic_v1",
    "diagnostic_only": True,
    "train_acceptance_claimed": False,
    "numeric_audit_complete": candidate.get("numeric_audit_complete"),
    "guard_debt_definition": candidate.get("guard_debt_definition"),
    "guard_debt_scale_sha256": candidate.get("guard_debt_scale_sha256"),
    "debt_weight_schema": candidate.get("debt_weight_schema"),
    "debt_weight_sha256": candidate.get("debt_weight_sha256"),
    "by_case": rows,
    "additional_step_case_count": sum(
        int(row["accepted_additional_step"]) for row in rows.values()
    ),
}
output.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "stage": "g1f4_p_three_target_k5_diagnostic",
    "report": str(output),
    "numeric_audit_complete": value["numeric_audit_complete"],
    "additional_step_case_count": value["additional_step_case_count"],
}), flush=True)
PY
printf '%s\n' "$P_SUMMARY" > outputs/LATEST_REFINER_V15_15G1F4_P_SUMMARY

echo "g1f4 Gate 0 passed; P-only three-target k5 diagnostic: $P_REPORT"
exit "$P_STATUS"
