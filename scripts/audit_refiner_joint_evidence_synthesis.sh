#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 11 ]]; then
  echo "usage: $0 RCSP_REPORT RCSP_REVIEW PARAMETER_REPORT SINGLE_REPORT PHASE2_REPORT PHASE21_REPORT BCTR_REPORT BCTR_CORRECTION SECDR_REPORT DEFECTIVE_SECDR_REPORT OUTPUT_DIR" >&2
  exit 2
fi

REPO_DIR="${REPO_DIR:-/home/disk/lsm/storage/DunhuangDanceRAG}"
PYTHON="${PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must name the current synthesis commit}"

RCSP_REPORT="$1"
RCSP_REVIEW="$2"
PARAMETER_REPORT="$3"
SINGLE_REPORT="$4"
PHASE2_REPORT="$5"
PHASE21_REPORT="$6"
BCTR_REPORT="$7"
BCTR_CORRECTION="$8"
SECDR_REPORT="$9"
DEFECTIVE_SECDR_REPORT="${10}"
OUTPUT_DIR="${11}"

cd "$REPO_DIR"
test "$(git branch --show-current)" = "main"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"

for path in \
  "$RCSP_REPORT" \
  "$RCSP_REVIEW" \
  "$PARAMETER_REPORT" \
  "$SINGLE_REPORT" \
  "$PHASE2_REPORT" \
  "$PHASE21_REPORT" \
  "$BCTR_REPORT" \
  "$BCTR_CORRECTION" \
  "$SECDR_REPORT" \
  "$DEFECTIVE_SECDR_REPORT"; do
  test -s "$path"
done

"$PYTHON" -m training.refiner_joint_evidence_synthesis \
  --rcsp-report "$RCSP_REPORT" \
  --rcsp-review "$RCSP_REVIEW" \
  --parameter-report "$PARAMETER_REPORT" \
  --single-report "$SINGLE_REPORT" \
  --phase2-report "$PHASE2_REPORT" \
  --phase21-report "$PHASE21_REPORT" \
  --bctr-report "$BCTR_REPORT" \
  --bctr-correction "$BCTR_CORRECTION" \
  --secdr-report "$SECDR_REPORT" \
  --defective-secdr-report "$DEFECTIVE_SECDR_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --expected-commit "$EXPECTED_COMMIT"
