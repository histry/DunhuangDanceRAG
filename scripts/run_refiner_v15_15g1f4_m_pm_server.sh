#!/usr/bin/env bash
# Train-only M calibration, followed by the preregistered M-only and PM cells.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
: "${V9_REFERENCE_REPORT:?Set V9_REFERENCE_REPORT to the canonical be71ae12 report}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
M_PREREG_CONTRACT="${M_PREREG_CONTRACT:-$(cat outputs/LATEST_REFINER_V15_15G1F4_M_V1_PREREG_CONTRACT)}"
M_PROTOCOL_AMENDMENT="${M_PROTOCOL_AMENDMENT:-docs/refiner_v15_15g1f4_m_protocol_amendment_v1.md}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -x "$PY"
test -s "$M_PREREG_CONTRACT"
test -s "$M_PROTOCOL_AMENDMENT"
test -s "$V9_REFERENCE_REPORT"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
for path in "$ADAPTER_STATE" "$TRAIN_BANK"; do test -s "$path"; done

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g1f4_m_pm_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
LOG="logs/refiner_v15_15g1f4_m_pm_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1f4_m_pm_${STAMP}.exit_status.txt"
mkdir -p "$ROOT" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1F4_M_PM_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1F4_M_PM_LOG
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"' EXIT
exec > >(tee -a "$LOG") 2>&1

ROOT_DIR=$(pwd)
M_PREREG_SHA=$(sha256sum "$M_PREREG_CONTRACT" | awk '{print $1}')
M_AMENDMENT_SHA=$(sha256sum "$M_PROTOCOL_AMENDMENT" | awk '{print $1}')
M_BINDING="$ROOT/m_protocol_amendment_binding.json"
"$PY" - "$M_BINDING" "$EXPECTED_COMMIT" "$M_PROTOCOL_AMENDMENT" \
  "$M_AMENDMENT_SHA" "$M_PREREG_CONTRACT" "$M_PREREG_SHA" <<'PY'
import json, sys
from pathlib import Path
out, commit, amendment, amendment_sha, prereg, prereg_sha = sys.argv[1:]
payload = {
    "schema": "g1f4_m_protocol_amendment_binding_v1",
    "implementation_commit": commit,
    "protocol_amendment_path": str(Path(amendment).resolve()),
    "protocol_amendment_sha256": amendment_sha,
    "base_preregistered_contract_path": str(Path(prereg).resolve()),
    "base_preregistered_contract_sha256": prereg_sha,
    "bound_before_m_calibration": True,
}
Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
sha256sum "$M_BINDING" > "$ROOT/m_protocol_amendment_binding.sha256"
echo "M preregistration SHA256: $M_PREREG_SHA"
echo "M protocol amendment SHA256: $M_AMENDMENT_SHA"
echo "M protocol amendment binding: $M_BINDING"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset EXPERIMENT_CONFIG_LOADED EXPERIMENT_ACTIVE_PROFILE
export PROJECT_ROOT="$ROOT_DIR" EXPERIMENT_PROFILE=research
source configs/experiment.env
export MOTION_DEVICE=cuda MOTION_GPU_PREPROCESSING=1
export MOTION_PRODUCT_REFINER_FILM_CONDITIONING=1
export MOTION_PRODUCT_REFINER_OBSERVABLE_ADAPTER=1
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --config configs/motion_model.json
  --train-teacher-bank "$TRAIN_BANK"
  --validation-teacher-bank "$TRAIN_BANK"
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
  --steps 2 3 5
  --evaluation-role train_calibration
)

CALIBRATION="$ROOT/anchor_kinematic_metric.calibration.json"
CALIBRATION_DIR="$ROOT/metric_calibration_identity"
set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  "${COMMON_ARGS[@]}" \
  --progress-mode current_equal_share \
  --metric-mode identity \
  --metric-preregistered-contract "$M_PREREG_CONTRACT" \
  --calibrate-anchor-metric-output "$CALIBRATION" \
  --output-dir "$CALIBRATION_DIR"
CALIBRATION_NATIVE_RC=$?
set -e
if test "$CALIBRATION_NATIVE_RC" -ne 0 \
    && test "$CALIBRATION_NATIVE_RC" -ne 2; then
  exit "$CALIBRATION_NATIVE_RC"
fi
test -s "$CALIBRATION"
printf '%s\n' "$CALIBRATION" > outputs/LATEST_REFINER_V15_15G1F4_M_CALIBRATION
CALIBRATION_SHA=$(sha256sum "$CALIBRATION" | awk '{print $1}')
printf '%s\n' "$CALIBRATION_SHA" > "$ROOT/anchor_kinematic_metric.calibration.sha256"
printf '%s\n' "$CALIBRATION_NATIVE_RC" \
  > "$ROOT/metric_calibration_identity.native_rc.txt"

CALIBRATION_REPORT="$CALIBRATION_DIR/fixed_budget_correction.report.json"
PARITY_REPORT="$ROOT/v9_canonical_parity_gate.json"
test -s "$CALIBRATION_REPORT"
"$PY" - "$CALIBRATION_REPORT" "$CALIBRATION_NATIVE_RC" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8-sig"))
if not report.get("numeric_audit_complete"):
    raise SystemExit("identity metric calibration numeric audit failed")
print(json.dumps({
    "stage": "g1f4_metric_calibration_complete",
    "native_exit_status": int(sys.argv[2]),
    "activation_aware_supported": report.get("activation_aware_supported"),
    "numeric_audit_complete": report.get("numeric_audit_complete"),
    "report": sys.argv[1],
}, ensure_ascii=False), flush=True)
PY
"$PY" -m training.refiner_v15_15g1f4_parity \
  --v9-reference-report "$V9_REFERENCE_REPORT" \
  --candidate-report "$CALIBRATION_REPORT" \
  --output "$PARITY_REPORT"
printf '%s\n' "$PARITY_REPORT" > outputs/LATEST_REFINER_V15_15G1F4_PARITY_REPORT

run_cell() {
  local name=$1 progress=$2 output_dir="$ROOT/$1"
  set +e
  "$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
    "${COMMON_ARGS[@]}" \
    --progress-mode "$progress" \
    --metric-mode anchor_kinematic \
    --metric-radius-calibration "$CALIBRATION" \
    --metric-radius-calibration-sha256 "$CALIBRATION_SHA" \
    --metric-preregistered-contract "$M_PREREG_CONTRACT" \
    --output-dir "$output_dir"
  local rc=$?
  set -e
  if test "$rc" -ne 0 && test "$rc" -ne 2; then return "$rc"; fi
  LAST_NATIVE_RC="$rc"
  local report="$output_dir/fixed_budget_correction.report.json"
  test -s "$report"
  "$PY" - "$report" "$name" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
if not r.get("numeric_audit_complete"):
    raise SystemExit(f"{sys.argv[2]} numeric audit failed")
if r.get("g1f4_mode") not in {"m_only", "pm"}:
    raise SystemExit(f"unexpected factorial cell: {r.get('g1f4_mode')}")
print(json.dumps({
    "stage": "g1f4_factorial_cell_complete",
    "cell": r["g1f4_mode"],
    "report": sys.argv[1],
    "activation_aware_supported": r.get("activation_aware_supported"),
    "numeric_audit_complete": r.get("numeric_audit_complete"),
}, ensure_ascii=False), flush=True)
PY
  LAST_REPORT="$report"
}

run_cell current_equal_share_anchor_kinematic current_equal_share
M_REPORT="$LAST_REPORT"
M_NATIVE_RC="$LAST_NATIVE_RC"
printf '%s\n' "$M_REPORT" > outputs/LATEST_REFINER_V15_15G1F4_M_ONLY_REPORT
printf '%s\n' "$M_NATIVE_RC" > outputs/LATEST_REFINER_V15_15G1F4_M_ONLY_NATIVE_RC

run_cell weighted_debt_filter_anchor_kinematic weighted_debt_filter
PM_REPORT="$LAST_REPORT"
PM_NATIVE_RC="$LAST_NATIVE_RC"
printf '%s\n' "$PM_REPORT" > outputs/LATEST_REFINER_V15_15G1F4_PM_REPORT
printf '%s\n' "$PM_NATIVE_RC" > outputs/LATEST_REFINER_V15_15G1F4_PM_NATIVE_RC

FACTORIAL_STATUS="$ROOT/g1f4_factorial_status.json"
"$PY" - "$FACTORIAL_STATUS" "$M_REPORT" "$M_NATIVE_RC" "$PM_REPORT" "$PM_NATIVE_RC" <<'PY'
import json, sys
from pathlib import Path
out, m_path, m_rc, pm_path, pm_rc = sys.argv[1:]
def cell(path, rc):
    report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {
        "native_exit_status": int(rc),
        "activation_aware_supported": report.get("activation_aware_supported"),
        "numeric_audit_complete": report.get("numeric_audit_complete"),
        "scientific_acceptance": bool(report.get("activation_aware_supported")),
        "report": str(Path(path).resolve()),
    }
payload = {
    "schema": "refiner_v15_15g1f4_factorial_status_v1",
    "pipeline_completed": True,
    "m_only": cell(m_path, m_rc),
    "pm": cell(pm_path, pm_rc),
}
Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False), flush=True)
PY

echo "M calibration: $CALIBRATION"
echo "M-only report: $M_REPORT"
echo "PM report: $PM_REPORT"
echo "M-only native exit status: $M_NATIVE_RC"
echo "M-only activation-aware supported: $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["activation_aware_supported"])' "$M_REPORT")"
echo "PM native exit status: $PM_NATIVE_RC"
echo "PM activation-aware supported: $("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["activation_aware_supported"])' "$PM_REPORT")"
echo "pipeline_exit_status=0 (pipeline completion; not scientific acceptance)"
