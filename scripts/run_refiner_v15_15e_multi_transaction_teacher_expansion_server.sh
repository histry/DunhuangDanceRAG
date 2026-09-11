#!/usr/bin/env bash
# Server-only V15.15e multi-transaction Oracle teacher expansion.
# This command freezes the split before Oracle search and starts no training.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/run_smpl14_formal_20260822_163915}"
MAX_TRANSACTIONS="${MAX_TRANSACTIONS:-8}"
MAX_CASES_PER_GROUP="${MAX_CASES_PER_GROUP:-4}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.25}"
ORACLE_ITERATIONS="${ORACLE_ITERATIONS:-60}"

test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

SOURCE_TEACHER_TAG="${SOURCE_TEACHER_TAG:-$(cat outputs/LATEST_REFINER_V15_15C_TEACHER_EXPANSION_TAG)}"
SOURCE_TEACHER_REPORT="$OUT_ROOT/checkpoints/$SOURCE_TEACHER_TAG/teacher_bank/observable_adapter_teacher_bank.report.json"
test -s "$SOURCE_TEACHER_REPORT"
SOURCE_DIAGNOSTIC=$(
  "$PY" - "$SOURCE_TEACHER_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(report["source_diagnostic"])
PY
)
test -s "$SOURCE_DIAGNOSTIC/diagnostic_report.json"
test -s "$SOURCE_DIAGNOSTIC/diagnostic_state.pt"
test -s "$SOURCE_DIAGNOSTIC/fit_bank.pt"

STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_15e_multi_transaction_teacher_expansion_${STAMP}"
ROOT="$OUT_ROOT/checkpoints/$TAG"
MANIFEST="$ROOT/split_manifest/v15_15e_teacher_split_manifest.json"
ORACLE_ROOT="$ROOT/oracles"
TRAIN_BANK_DIR="$ROOT/teacher_bank_train"
VALIDATION_BANK_DIR="$ROOT/teacher_bank_validation"
WORKLIST="$ROOT/split_manifest/oracle_worklist.tsv"
LOG="logs/refiner_v15_15e_multi_transaction_teacher_expansion_${STAMP}.log"
STATUS="outputs/refiner_v15_15e_multi_transaction_teacher_expansion_${STAMP}.exit_status.txt"
SCIENTIFIC_STATUS="outputs/refiner_v15_15e_multi_transaction_teacher_expansion_${STAMP}.scientific_status.txt"

mkdir -p "$(dirname "$MANIFEST")" "$ORACLE_ROOT" \
  "$TRAIN_BANK_DIR" "$VALIDATION_BANK_DIR" logs outputs
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_STATUS
printf '%s\n' "$SCIENTIFIC_STATUS" > outputs/LATEST_REFINER_V15_15E_TEACHER_EXPANSION_SCIENTIFIC_STATUS
printf '%s\n' "$MANIFEST" > outputs/LATEST_REFINER_V15_15E_TEACHER_MANIFEST
printf '%s\n' "$TRAIN_BANK_DIR/observable_adapter_teacher_bank.pt" \
  > outputs/LATEST_REFINER_V15_15E_TRAIN_TEACHER_BANK
printf '%s\n' "$VALIDATION_BANK_DIR/observable_adapter_teacher_bank.pt" \
  > outputs/LATEST_REFINER_V15_15E_VALIDATION_TEACHER_BANK
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1

echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "source_diagnostic=$SOURCE_DIAGNOSTIC"
echo "max_transactions=$MAX_TRANSACTIONS"
echo "max_cases_per_group=$MAX_CASES_PER_GROUP"
echo "validation_fraction=$VALIDATION_FRACTION"
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
export MOTION_PRODUCT_REFINER_OBSERVABLE_ADAPTER=0
export MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED=1

# This immutable manifest and its SHA256 sidecar are written before the first
# Oracle process starts.
"$PY" -u -m training.refiner_v15_15e_transaction_manifest \
  --source-diagnostic-dir "$SOURCE_DIAGNOSTIC" \
  --output "$MANIFEST" \
  --max-transactions "$MAX_TRANSACTIONS" \
  --max-cases-per-group "$MAX_CASES_PER_GROUP" \
  --validation-fraction "$VALIDATION_FRACTION"
MANIFEST_SHA256=$(cut -d ' ' -f 1 "$MANIFEST.sha256")
test "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)" = "$MANIFEST_SHA256"

"$PY" - "$MANIFEST" "$WORKLIST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
rows = []
for transaction in manifest["transactions"]:
    indices = ",".join(
        str(row["case_index"]) for row in transaction["oracle_cases"]
    )
    if not indices:
        raise SystemExit("manifest retained a transaction without Oracle cases")
    rows.append("\t".join((
        transaction["split"],
        str(transaction["transaction_index"]),
        transaction["transaction_id"],
        indices,
    )))
Path(sys.argv[2]).write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

TRAIN_REPORT_ARGS=()
VALIDATION_REPORT_ARGS=()
ORACLE_SCIENTIFIC_FAILURES=0
while IFS=$'\t' read -r SPLIT TRANSACTION_INDEX TRANSACTION_ID CASE_CSV; do
  ORACLE_DIR="$ORACLE_ROOT/$TRANSACTION_ID"
  mkdir -p "$ORACLE_DIR"
  IFS=',' read -r -a CASE_INDICES <<< "$CASE_CSV"
  CASE_ARGS=()
  for CASE_INDEX in "${CASE_INDICES[@]}"; do
    CASE_ARGS+=(--case-index "$CASE_INDEX")
  done
  echo "oracle_transaction=$TRANSACTION_ID split=$SPLIT cases=$CASE_CSV"
  test "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)" = "$MANIFEST_SHA256"
  set +e
  "$PY" -u -m training.refiner_case_local_full_tangent_oracle \
    --config configs/motion_model.json \
    --source-diagnostic-dir "$SOURCE_DIAGNOSTIC" \
    --output-dir "$ORACLE_DIR" \
    --transaction-index "$TRANSACTION_INDEX" \
    --transaction-fixed-guard \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --teacher-split "$SPLIT" \
    "${CASE_ARGS[@]}" \
    --target-rms 1e-4 \
    --iterations "$ORACLE_ITERATIONS" \
    --learning-rate 2e-2 \
    --initial-penalty 10
  ORACLE_STATUS=$?
  set -e
  if [[ "$ORACLE_STATUS" -ne 0 && "$ORACLE_STATUS" -ne 2 ]]; then
    echo "[FATAL] Oracle $TRANSACTION_ID failed with status $ORACLE_STATUS"
    exit "$ORACLE_STATUS"
  fi
  if [[ "$ORACLE_STATUS" -eq 2 ]]; then
    ORACLE_SCIENTIFIC_FAILURES=$((ORACLE_SCIENTIFIC_FAILURES + 1))
  fi
  REPORT="$ORACLE_DIR/case_local_full_tangent_oracle.report.json"
  test -s "$REPORT"
  if [[ "$SPLIT" == "train" ]]; then
    TRAIN_REPORT_ARGS+=(--oracle-report "$REPORT")
  else
    VALIDATION_REPORT_ARGS+=(--oracle-report "$REPORT")
  fi
done < "$WORKLIST"
test "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)" = "$MANIFEST_SHA256"

if [[ "${#TRAIN_REPORT_ARGS[@]}" -eq 0 || "${#VALIDATION_REPORT_ARGS[@]}" -eq 0 ]]; then
  echo "[FATAL] frozen manifest did not produce both split report sets"
  exit 4
fi

set +e
"$PY" -u -m training.refiner_v15_15_teacher_bank \
  --config configs/motion_model.json \
  --split-manifest "$MANIFEST" \
  --split train \
  "${TRAIN_REPORT_ARGS[@]}" \
  --output-dir "$TRAIN_BANK_DIR"
TRAIN_BANK_STATUS=$?
"$PY" -u -m training.refiner_v15_15_teacher_bank \
  --config configs/motion_model.json \
  --split-manifest "$MANIFEST" \
  --split validation \
  "${VALIDATION_REPORT_ARGS[@]}" \
  --output-dir "$VALIDATION_BANK_DIR"
VALIDATION_BANK_STATUS=$?
set -e
for BUILD_STATUS in "$TRAIN_BANK_STATUS" "$VALIDATION_BANK_STATUS"; do
  if [[ "$BUILD_STATUS" -ne 0 && "$BUILD_STATUS" -ne 2 ]]; then
    echo "[FATAL] teacher-bank construction failed with status $BUILD_STATUS"
    exit "$BUILD_STATUS"
  fi
done

TRAIN_REPORT="$TRAIN_BANK_DIR/observable_adapter_teacher_bank.report.json"
VALIDATION_REPORT="$VALIDATION_BANK_DIR/observable_adapter_teacher_bank.report.json"
test -s "$TRAIN_REPORT"
test -s "$VALIDATION_REPORT"
"$PY" - "$MANIFEST" "$TRAIN_REPORT" "$VALIDATION_REPORT" <<'PY'
import json
import sys
from pathlib import Path

manifest, train, validation = [
    json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for path in sys.argv[1:]
]
train_case_uids = {row["case_uid"] for row in train["evidence"]}
validation_case_uids = {
    row["case_uid"] for row in validation["evidence"]
}
train_source_uids = {row["source_case_uid"] for row in train["evidence"]}
validation_source_uids = {
    row["source_case_uid"] for row in validation["evidence"]
}
case_overlap = sorted(train_case_uids & validation_case_uids)
source_overlap = sorted(train_source_uids & validation_source_uids)
if case_overlap or source_overlap:
    raise SystemExit({
        "teacher_case_overlap": case_overlap,
        "teacher_source_case_overlap": source_overlap,
    })
print(json.dumps({
    "manifest": str(Path(sys.argv[1]).resolve()),
    "manifest_file_sha256": train["split_manifest_file_sha256"],
    "manifest_content_sha256": manifest["manifest_content_sha256"],
    "train_projected_teacher_count_by_group": (
        train["projected_teacher_count_by_group"]
    ),
    "validation_projected_teacher_count_by_group": (
        validation["projected_teacher_count_by_group"]
    ),
    "train_teacher_bank_ready": train["teacher_bank_ready"],
    "validation_teacher_bank_ready": validation["teacher_bank_ready"],
    "train_validation_case_overlap": case_overlap,
    "train_validation_source_case_overlap": source_overlap,
    "fixed_guard_thresholds_changed": False,
}, ensure_ascii=False, indent=2))
PY

SCIENTIFIC_RESULT=0
if [[ "$TRAIN_BANK_STATUS" -eq 2 || "$VALIDATION_BANK_STATUS" -eq 2 ]]; then
  SCIENTIFIC_RESULT=2
fi
printf '%s\n' "$SCIENTIFIC_RESULT" > "$SCIENTIFIC_STATUS"
echo "V15.15e multi-transaction teacher expansion completed; scientific_status=$SCIENTIFIC_RESULT"
echo "oracle_transactions_without_complete_route=$ORACLE_SCIENTIFIC_FAILURES"
echo "No Adapter training, formal training, promotion, replay, or generation was launched."
