#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure top-level research packages are importable.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "${EXPERIMENT_CONFIG_LOADED:-0}" != "1" ]]; then
  # shellcheck disable=SC1091
  source configs/experiment.env
fi
: "${OUT_ROOT:?Set OUT_ROOT to an existing run directory with a valid retarget cache}"
export GENERATION_REBUILD_RETARGET_CACHE=0
export GENERATION_REBUILD_EVENT_DB=1
export GENERATION_RETRAIN_CONTRASTIVE=1
export GENERATION_RETRAIN_REFINER=1
export GENERATION_RETRAIN_DIFFUSION=1
exec bash scripts/run_experiment.sh "${1:-$TEST_AUDIO}"
