#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

TRAINED_RUN="${1:-${OUT_ROOT:-}}"
GROUP="${2:-${PERFORMER_GROUP:-auto}}"

[[ -n "$TRAINED_RUN" ]] || {
  echo "Usage: generate_test_set.sh TRAINED_RUN_ROOT [auto|female|male|mixed]" >&2
  exit 2
}

for audio in "$ROOT_DIR"/assets/music/test/audio/*.wav; do
  echo "========== $(basename "$audio") | performer=$GROUP =========="
  bash scripts/generate_only.sh "$audio" "$TRAINED_RUN" "$GROUP"
done
