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
mkdir -p outputs/preflight
exec "$PYTHON_BIN" evaluation/preflight.py   --root "$PROJECT_ROOT"   --audio "$TEST_AUDIO"   --music_dir "$TRAIN_MUSIC_DIR"   --change_dir "$BVH_DATASET_DIR"   --out outputs/preflight/report.json
