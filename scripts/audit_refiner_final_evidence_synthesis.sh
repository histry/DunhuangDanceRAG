#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_MAIN_COMMIT="${EXPECTED_MAIN_COMMIT:?set EXPECTED_MAIN_COMMIT to the committed Final Synthesis SHA}"

JOINT_REPORT="${1:?usage: audit_refiner_final_evidence_synthesis.sh JOINT_REPORT RPA_REPORT RPA_FREEZE DIRECTION_REPORT DIRECTION_FREEZE OUTPUT_DIR}"
RPA_REPORT="${2:?usage: audit_refiner_final_evidence_synthesis.sh JOINT_REPORT RPA_REPORT RPA_FREEZE DIRECTION_REPORT DIRECTION_FREEZE OUTPUT_DIR}"
RPA_FREEZE="${3:?usage: audit_refiner_final_evidence_synthesis.sh JOINT_REPORT RPA_REPORT RPA_FREEZE DIRECTION_REPORT DIRECTION_FREEZE OUTPUT_DIR}"
DIRECTION_REPORT="${4:?usage: audit_refiner_final_evidence_synthesis.sh JOINT_REPORT RPA_REPORT RPA_FREEZE DIRECTION_REPORT DIRECTION_FREEZE OUTPUT_DIR}"
DIRECTION_FREEZE="${5:?usage: audit_refiner_final_evidence_synthesis.sh JOINT_REPORT RPA_REPORT RPA_FREEZE DIRECTION_REPORT DIRECTION_FREEZE OUTPUT_DIR}"
OUTPUT_DIR="${6:?usage: audit_refiner_final_evidence_synthesis.sh JOINT_REPORT RPA_REPORT RPA_FREEZE DIRECTION_REPORT DIRECTION_FREEZE OUTPUT_DIR}"

cd "$REPO_DIR"

test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -z "$(git status --porcelain)"

for path in \
  "$JOINT_REPORT" \
  "$RPA_REPORT" \
  "$RPA_FREEZE" \
  "$DIRECTION_REPORT" \
  "$DIRECTION_FREEZE"
do
  test -f "$path"
done

"$PYTHON" -m training.refiner_final_evidence_synthesis \
  --joint-report "$JOINT_REPORT" \
  --rpa-report "$RPA_REPORT" \
  --rpa-freeze "$RPA_FREEZE" \
  --direction-report "$DIRECTION_REPORT" \
  --direction-freeze "$DIRECTION_FREEZE" \
  --output-dir "$OUTPUT_DIR" \
  --expected-commit "$EXPECTED_MAIN_COMMIT"
