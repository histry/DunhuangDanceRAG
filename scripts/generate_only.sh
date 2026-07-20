#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT_DIR
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

AUDIO="${1:?Usage: generate_only.sh AUDIO [TRAINED_RUN_ROOT] [PERFORMER_GROUP]}"
TRAINED_RUN="${2:-${TRAINED_RUN_ROOT:-}}"
export PERFORMER_GROUP="${3:-${PERFORMER_GROUP:-auto}}"

if [[ -z "$TRAINED_RUN" ]]; then
  TRAINED_RUN="$(
    find "$ROOT_DIR/outputs" -maxdepth 1 -type d -name 'run_*' \
      -exec test -s '{}/checkpoints/grounder.pt' ';' -print 2>/dev/null \
      | sort | tail -1
  )"
fi

[[ -n "$TRAINED_RUN" && -d "$TRAINED_RUN" ]] || {
  echo "[FATAL] No trained run found. Train once with run.sh before test-set generation." >&2
  exit 2
}

TRAINED_RUN="$(realpath "$TRAINED_RUN")"
export OUT_ROOT="$TRAINED_RUN"
export AUDIO="$(realpath "$AUDIO")"

# Reuse the trained database and checkpoints. Keep compatibility variable names
# because the preserved internal pipeline still reads them.
export REBUILD_RETARGET_CACHE=0
export REBUILD_EVENT_DB=0
export RETRAIN_CONTRASTIVE=0
export RETRAIN_REFINER=0
export RETRAIN_DIFFUSION=0
export V46_51_REBUILD_RETARGET_CACHE=0
export V46_51_REBUILD_EVENT_DB=0
export V46_51_RETRAIN_V44=0
export V46_51_RETRAIN_V45=0
export V46_51_RETRAIN_V46=0
export V46_53_1_FULL_REBUILD=0
export V46_53_1_REBUILD_RETARGET_CACHE=0
export V46_53_1_REBUILD_EVENT_DB=0
export V46_53_1_RETRAIN_V44=0
export V46_53_1_RETRAIN_V45=0
export V46_53_1_RETRAIN_V46=0
export V46_53_GROUNDER_CKPT="${V46_53_GROUNDER_CKPT:-$TRAINED_RUN/checkpoints/grounder.pt}"
export V44_CKPT="${V44_CKPT:-$TRAINED_RUN/checkpoints/semantic_retriever.pt}"
export V45_CKPT="${V45_CKPT:-$TRAINED_RUN/checkpoints/boundary_refiner.pt}"
export V46_CKPT="${V46_CKPT:-$TRAINED_RUN/checkpoints/local_diffusion.pt}"

TRACK_ID="$(basename "${AUDIO%.*}")"
RESULT_ROOT="$TRAINED_RUN/test_results/$TRACK_ID/$PERFORMER_GROUP"
mkdir -p "$RESULT_ROOT"

MARKER="$(mktemp)"
touch "$MARKER"

echo "[GENERATE] AUDIO=$AUDIO"
echo "[GENERATE] TRAINED_RUN=$TRAINED_RUN"
echo "[GENERATE] PERFORMER_GROUP=$PERFORMER_GROUP"
echo "[GENERATE] RESULT_ROOT=$RESULT_ROOT"

bash scripts/research_pipeline.sh "$AUDIO"

# The preserved pipeline writes final products under OUT_ROOT. Archive only files
# created or modified by this generation call so the next test track cannot
# overwrite the previous result.
while IFS= read -r -d '' file; do
  rel="${file#$TRAINED_RUN/}"
  safe_name="${rel//\//__}"
  cp -f "$file" "$RESULT_ROOT/$safe_name"
done < <(
  find "$TRAINED_RUN" -type f -newer "$MARKER" \
    \( -name '*.npy' -o -name '*.json' -o -name '*.csv' -o -name '*.mp4' \) \
    -print0
)

cat > "$RESULT_ROOT/generation_manifest.json" <<JSON
{
  "track_id": "$TRACK_ID",
  "audio": "$AUDIO",
  "trained_run": "$TRAINED_RUN",
  "performer_group": "$PERFORMER_GROUP"
}
JSON

rm -f "$MARKER"
echo "[DONE] archived generation outputs to $RESULT_ROOT"
