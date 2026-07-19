#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure top-level research packages are importable.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
source configs/paths.env
source configs/experiment.env
: "${OUT_ROOT:?Set OUT_ROOT to an existing run directory with a valid retarget cache}"
export V46_51_REBUILD_RETARGET_CACHE=0
export V46_51_REBUILD_EVENT_DB=1
export V46_51_RETRAIN_V44=1
export V46_51_RETRAIN_V45=1
export V46_51_RETRAIN_V46=1
exec bash scripts/run_experiment.sh "${1:-$TEST_AUDIO}"
