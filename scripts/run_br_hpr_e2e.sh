#!/usr/bin/env bash
set -euo pipefail

: "${ROOT:?Set ROOT to the DunhuangDanceRAG repository}"
: "${PY:?Set PY to the Python interpreter}"
: "${CONFIG:?Set CONFIG}"
: "${SMOKE_AUDIO:?Set SMOKE_AUDIO}"
: "${FROZEN_MIXED_SCHEDULE:?Set FROZEN_MIXED_SCHEDULE}"
: "${SEM_DB:?Set SEM_DB}"
: "${V44_CKPT:?Set V44_CKPT}"
: "${V45_CKPT:?Set V45_CKPT}"
: "${V46_CKPT:?Set V46_CKPT}"
: "${BR_HPR_OUT_ROOT:?Set BR_HPR_OUT_ROOT}"

export EXPERIMENT_PROFILE=br_hpr
# shellcheck disable=SC1090
source "$ROOT/configs/experiment.env"

mkdir -p "$BR_HPR_OUT_ROOT"
MOTION="$BR_HPR_OUT_ROOT/br_hpr.motion.npy"
REPORT="$BR_HPR_OUT_ROOT/br_hpr.report.json"
LOG="$BR_HPR_OUT_ROOT/br_hpr.generate.log"

rm -f "$MOTION" "$REPORT" "$LOG"
git -C "$ROOT" rev-parse HEAD > "$BR_HPR_OUT_ROOT/source_commit.txt"
git -C "$ROOT" status --porcelain=v1 > "$BR_HPR_OUT_ROOT/source_worktree_status.txt"
git -C "$ROOT" diff > "$BR_HPR_OUT_ROOT/source_worktree.diff"
(
  cd "$ROOT"
  git ls-files --others --exclude-standard -z \
    | tar --null -T - -czf "$BR_HPR_OUT_ROOT/source_untracked_files.tar.gz" \
      2>/dev/null || true
)
env | grep -E '^(BR_HPR_|V46_|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS)' \
  | sort > "$BR_HPR_OUT_ROOT/runtime_environment.txt"

set -o pipefail
set +e
timeout --signal=INT --kill-after=60s "${BR_HPR_TIMEOUT:-90m}" \
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
      PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/dunhuang_python_cache}" \
  "$PY" -B -u "$ROOT/routing/closed_loop.py" generate \
    --config "$CONFIG" \
    --audio "$SMOKE_AUDIO" \
    --slots_json "$FROZEN_MIXED_SCHEDULE" \
    --db "$SEM_DB" \
    --contrastive "$V44_CKPT" \
    --refiner "$V45_CKPT" \
    --diffusion "$V46_CKPT" \
    --out "$MOTION" \
    --json "$REPORT" \
  2>&1 | tee "$LOG"
status="${PIPESTATUS[0]}"
set -e
printf 'br_hpr_e2e_exit=%s\n' "$status" | tee "$BR_HPR_OUT_ROOT/exit_status.txt"
exit "$status"
