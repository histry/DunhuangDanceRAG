#!/usr/bin/env bash
# Server-only audit of the saved V15.15 Adapter with exact ownership scope.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

SOURCE_TAG="${SOURCE_ADAPTER_TAG:-$(cat outputs/LATEST_REFINER_V15_15_ADAPTER_PROBE_TAG)}"
SOURCE_ROOT="$OUT_ROOT/checkpoints/$SOURCE_TAG"
TEACHER_BANK="$SOURCE_ROOT/teacher_bank/observable_adapter_teacher_bank.pt"
ADAPTER_STATE="$SOURCE_ROOT/adapter_probe/observable_adapter_probe_state.pt"
SOURCE_REPORT="$SOURCE_ROOT/adapter_probe/observable_adapter_probe.report.json"
test -s "$TEACHER_BANK"
test -s "$ADAPTER_STATE"
test -s "$SOURCE_REPORT"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15b_exact_scope_adapter_audit_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
AUDIT_DIR="$ROOT/adapter_probe"
LOG="logs/refiner_v15_15b_exact_scope_adapter_audit_${STAMP}.log"
STATUS="outputs/refiner_v15_15b_exact_scope_adapter_audit_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15b_exact_scope_adapter_audit_${STAMP}.scientific_status.txt"

mkdir -p "$AUDIT_DIR" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15B_EXACT_SCOPE_AUDIT_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15B_EXACT_SCOPE_AUDIT_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15B_EXACT_SCOPE_AUDIT_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" \
  > outputs/LATEST_REFINER_V15_15B_EXACT_SCOPE_AUDIT_SCIENTIFIC_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "source_adapter_tag=$SOURCE_TAG"
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

set +e
"$PY" -u -m training.refiner_observable_adapter_probe \
  --config configs/motion_model.json \
  --teacher-bank "$TEACHER_BANK" \
  --adapter-state "$ADAPTER_STATE" \
  --audit-only \
  --output-dir "$AUDIT_DIR" \
  --steps 0 \
  --eval-every 1 \
  --target-rms 1e-4
AUDIT_STATUS=$?
set -e
printf '%s\n' "$AUDIT_STATUS" > "$SCIENTIFIC_STATUS"
if [[ "$AUDIT_STATUS" -ne 0 && "$AUDIT_STATUS" -ne 2 ]]; then
  echo "[FATAL] exact-scope Adapter audit failed with status $AUDIT_STATUS"
  exit "$AUDIT_STATUS"
fi

REPORT="$AUDIT_DIR/observable_adapter_probe.report.json"
test -s "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
final = r["history"][-1]["exact_audits"]
print(json.dumps({
    "audit_only": r.get("audit_only"),
    "scope_safe": r.get("scope_safe"),
    "numeric_audit_complete": r.get("numeric_audit_complete"),
    "outside_scope_abs_max_after_final_mask": r.get(
        "outside_scope_abs_max_after_final_mask"
    ),
    "effective_projected_candidate_count_by_group": r.get(
        "effective_projected_candidate_count_by_group"
    ),
    "ready_for_expanded_adapter_training": r.get(
        "ready_for_expanded_adapter_training"
    ),
    "cases": [{
        "case_index": row["case_index"],
        "group": row["group"],
        "raw_passed": row["raw_audit"]["passed"],
        "scope_safe": row["raw_audit"]["scope"]["scope_safe"],
        "projector_invoked": row["projector_result"] is not None,
        "effective_projected_candidate": row[
            "effective_projected_candidate"
        ],
    } for row in final],
    "report": str(Path(sys.argv[1]).resolve()),
}, ensure_ascii=False, indent=2))
PY

echo "V15.15b exact-scope Adapter audit completed; scientific_status=$AUDIT_STATUS"
echo "No training, oracle expansion, promotion, replay, or generation was launched."
