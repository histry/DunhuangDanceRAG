#!/usr/bin/env bash
# One-shot V15.15g1f evaluation on a new cross-long transaction bank.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
: "${FINAL_HELD_OUT_BANK:?Set FINAL_HELD_OUT_BANK to the new untouched bank}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

ADAPTER_STATE=$(cat outputs/LATEST_REFINER_V15_15F1_GATE_RESTORATION_STATE)
TEACHER_TAG=$(cat outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG)
TRAIN_BANK="$OUT_ROOT/checkpoints/$TEACHER_TAG/teacher_bank_train/observable_adapter_teacher_bank.pt"
DEV_BANK=$(cat outputs/LATEST_REFINER_V15_15E_VALIDATION_BANK)
FROZEN_ENVELOPE=$(cat outputs/LATEST_REFINER_V15_15G1F_FROZEN_CONFORMAL)
FROZEN_REPAIR=$(cat outputs/LATEST_REFINER_V15_15G1F_FROZEN_REPAIR_CONTRACT)
TRAIN_REPORT=$(cat outputs/LATEST_REFINER_V15_15G1F_TRAIN_REPORT)
DEV_REPORT=$(cat outputs/LATEST_REFINER_V15_15G1F_DEVELOPMENT_REPORT)
for path in \
  "$ADAPTER_STATE" "$TRAIN_BANK" "$DEV_BANK" "$FINAL_HELD_OUT_BANK" \
  "$FROZEN_ENVELOPE" "$FROZEN_REPAIR" "$TRAIN_REPORT" "$DEV_REPORT"; do
  test -s "$path"
done

FINAL_SHA=$(sha256sum "$FINAL_HELD_OUT_BANK" | cut -d ' ' -f 1)
RECEIPT="outputs/V15_15G1F_FINAL_HELD_OUT_CONSUMED_${FINAL_SHA}.json"
if test -e "$RECEIPT"; then
  echo "[FATAL] final held-out bank was already consumed: $RECEIPT"
  exit 3
fi

"$PY" - \
  "$TRAIN_BANK" "$DEV_BANK" "$FINAL_HELD_OUT_BANK" \
  "$TRAIN_REPORT" "$DEV_REPORT" <<'PY'
import json
import sys
from pathlib import Path

import torch

train_path, dev_path, final_path, train_report_path, dev_report_path = map(
    Path, sys.argv[1:]
)
train = torch.load(train_path, map_location="cpu", weights_only=False)
dev = torch.load(dev_path, map_location="cpu", weights_only=False)
final = torch.load(final_path, map_location="cpu", weights_only=False)
train_report = json.loads(train_report_path.read_text(encoding="utf-8-sig"))
dev_report = json.loads(dev_report_path.read_text(encoding="utf-8-sig"))
expected_schema = (
    "refiner_v15_15g1f_exact_radius_geodesic_joint_active_set_sqp_v1"
)
if train_report.get("schema") != expected_schema:
    raise SystemExit("train calibration report schema mismatch")
if dev_report.get("schema") != expected_schema:
    raise SystemExit("development report schema mismatch")
intersection = train_report.get("local_feasible_intersection_summary") or {}
if not intersection.get("local_feasible_intersection_complete"):
    raise SystemExit("train geodesic joint feasible intersection is incomplete")
if not train_report.get("numeric_audit_complete"):
    raise SystemExit("train numeric audit is incomplete")
if not dev_report.get("activation_aware_supported"):
    raise SystemExit("development activation-aware gate did not pass")
if final.get("split") != "validation":
    raise SystemExit("final bank must declare split=validation")
if not final.get("teacher_bank_ready"):
    raise SystemExit("final bank must contain complete cross routes")
if final.get("train_validation_case_overlap"):
    raise SystemExit("final bank reports train case overlap")
if final.get("train_validation_source_case_overlap"):
    raise SystemExit("final bank reports train source-case overlap")
for key in ("split_manifest_file_sha256", "split_manifest_content_sha256"):
    if not train.get(key):
        raise SystemExit(f"train bank lacks sealed manifest field {key}")
    if not final.get(key):
        raise SystemExit(f"final bank lacks sealed manifest field {key}")

def transaction_ids(bank):
    return {str(sample["transaction_id"]) for sample in bank["samples"]}

def source_case_uids(bank):
    return {
        str(sample["source_case_uid"])
        for sample in bank["samples"]
        if sample.get("source_case_uid") is not None
    }

train_ids = transaction_ids(train)
dev_ids = transaction_ids(dev)
final_ids = transaction_ids(final)
if train_ids & final_ids:
    raise SystemExit("final held-out transaction overlaps train")
if dev_ids & final_ids:
    raise SystemExit("final held-out transaction overlaps reused development")
if source_case_uids(train) & source_case_uids(final):
    raise SystemExit("final held-out source case overlaps train")
if source_case_uids(dev) & source_case_uids(final):
    raise SystemExit("final held-out source case overlaps reused development")
uids = {str(sample["case_uid"]) for sample in final["samples"]}
if "txn_0000_94bfdf553811:53" in uids:
    raise SystemExit("reused development case 53 entered final held-out")
if not any(
    sample.get("audit_group") == "cross_long"
    and sample.get("teacher_kind") == "exact_projected_direction"
    for sample in final["samples"]
):
    raise SystemExit("final bank lacks an effective cross_long teacher")
print("final_held_out_manifest_is_separately_sealed=true")
PY

"$PY" - \
  "$RECEIPT" "$FINAL_HELD_OUT_BANK" "$FINAL_SHA" \
  "$FROZEN_REPAIR" "$FROZEN_ENVELOPE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, bank, digest, repair, envelope = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema": "refiner_v15_15g1f_final_held_out_one_shot_receipt_v1",
    "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
    "final_held_out_bank": bank,
    "final_held_out_bank_sha256": digest,
    "frozen_repair_contract": repair,
    "frozen_severity_envelope": envelope,
    "rerun_allowed": False,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15g1f_final_held_out_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
PROBE_DIR="$ROOT/final_held_out_probe"
LOG="logs/refiner_v15_15g1f_final_held_out_${STAMP}.log"
STATUS="outputs/refiner_v15_15g1f_final_held_out_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15g1f_final_held_out_${STAMP}.scientific_status.txt"
mkdir -p "$ROOT" logs outputs

printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15G1F_FINAL_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15G1F_FINAL_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15G1F_FINAL_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15G1F_FINAL_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
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

set +e
"$PY" -u -m training.refiner_v15_15g_fixed_budget_correction \
  --config configs/motion_model.json \
  --train-teacher-bank "$TRAIN_BANK" \
  --validation-teacher-bank "$FINAL_HELD_OUT_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --output-dir "$PROBE_DIR" \
  --activation-aware-g1f \
  --evaluation-role final_held_out \
  --frozen-severity-envelope "$FROZEN_ENVELOPE" \
  --frozen-full-shadow-repair-contract "$FROZEN_REPAIR" \
  --full-shadow-line-search-backtracks 12 \
  --full-shadow-line-search-decay 0.5 \
  --science-restoration-damping 1e-8 \
  --science-restoration-safety-fraction 0.25 \
  --geodesic-angular-max-radians 0.7853981633974483 \
  --joint-svd-relative-cutoff 1e-6 \
  --joint-direction-norm-floor 1e-8 \
  --joint-directional-margin 1e-6 \
  --steps 2 3 5 \
  --target-rms 1e-4
RUN_STATUS=$?
set -e
printf '%s\n' "$RUN_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$RUN_STATUS" -ne 0 && "$RUN_STATUS" -ne 2 ]]; then
  echo "[FATAL] V15.15g1f final held-out failed with status $RUN_STATUS"
  exit "$RUN_STATUS"
fi

REPORT="$PROBE_DIR/fixed_budget_correction.report.json"
test -s "$REPORT"
printf '%s\n' "$REPORT" > outputs/LATEST_REFINER_V15_15G1F_FINAL_REPORT
echo "V15.15g1f one-shot final held-out completed; scientific_status=$RUN_STATUS"
echo "No training, pseudo-teacher recycling, promotion, replay, or generation was launched."
