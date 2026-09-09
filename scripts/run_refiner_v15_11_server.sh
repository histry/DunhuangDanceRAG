#!/usr/bin/env bash
# Server-only validation and fresh diagnostic; never pilot or promotion.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
: "${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the supplied full commit SHA}"
export PY="${PY:-/home/disk/lsm/conda_envs/edge/bin/python}"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
OUT_ROOT="${1:-outputs/run_smpl14_formal_20260822_163915}"
STAMP=$(date +%Y%m%d_%H%M%S)
TAG="refiner_v15_11_feasibility_slack_guard_${STAMP}"
mkdir -p outputs logs
LOG="logs/refiner_v15_11_${STAMP}.log"
STATUS="outputs/refiner_v15_11_${STAMP}.exit_status.txt"
printf '%s\n' "$TAG" > outputs/LATEST_REFINER_V15_11_TAG
printf '%s\n' "$LOG" > outputs/LATEST_REFINER_V15_11_LOG
printf '%s\n' "$STATUS" > outputs/LATEST_REFINER_V15_11_STATUS
trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "exit_status=$rc"; date --iso-8601=seconds' EXIT
exec > >(tee -a "$LOG") 2>&1
echo "commit=$EXPECTED_COMMIT"
echo "tag=$TAG"
echo "started_at=$(date --iso-8601=seconds)"
"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.cuda.get_device_name(0))'
"$PY" -m pytest -q \
  tests/test_refiner_local_context.py \
  tests/test_refiner_v15_2_smooth_bottleneck.py \
  tests/test_refiner_v15_3_tail_objective.py \
  tests/test_refiner_v15_4_context_reservoir.py \
  tests/test_refiner_v15_feasibility_guard.py \
  tests/test_refiner_optimizer.py \
  tests/test_zero_edit_retraction.py
"$PY" -m ruff check --select E9,F63,F7,F82 \
  training/motion_models.py training/refiner_bridge_diagnostics.py \
  tests/test_refiner_v15_feasibility_guard.py
bash scripts/train_refiner_v8.sh foundation "$OUT_ROOT" "$TAG"
bash scripts/train_refiner_v8.sh diagnose "$OUT_ROOT" "$TAG"
echo "DIAGNOSTIC PASSED; no pilot, training resume, or promotion was launched."
