#!/usr/bin/env bash
set -Eeuo pipefail

# Ensure top-level research packages are importable.
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
source configs/paths.env
source configs/experiment.env
[[ -f "$ROOT_DIR/configs/research_feasibility.env" ]] && source "$ROOT_DIR/configs/research_feasibility.env"

[[ -f "$ROOT_DIR/configs/performer_policy.env" ]] && source "$ROOT_DIR/configs/performer_policy.env"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/outputs/run_${RUN_TAG}}"
exec bash scripts/run_experiment.sh "${1:-$TEST_AUDIO}"
