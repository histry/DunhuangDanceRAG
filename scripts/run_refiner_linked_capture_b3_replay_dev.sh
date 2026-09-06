#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_COMMIT="${1:-$(git rev-parse HEAD)}"
MAX_ITERATIONS="${2:-32}"
PRIOR_CAPTURE="${PRIOR_CAPTURE:-outputs/generation_stage_capture_20260905_205755}"
LEGACY_REPORT="${LEGACY_REPORT:-outputs/diagnostic_refiner_only/fresh_audio_final.report.json}"
SCHEDULE="${SCHEDULE:-$PRIOR_CAPTURE/schedule.from_failure_run.mssd.json}"
OLD_CASES="${OLD_CASES:-$PRIOR_CAPTURE/refiner_failure_cases_selected_round2/cases.json}"
PRIOR_BUNDLE="${PRIOR_BUNDLE:-$(cat "$PRIOR_CAPTURE/selected_bundle.path" 2>/dev/null || true)}"
FROZEN_DIFFUSION="${FROZEN_DIFFUSION:-outputs/motion_v12_v4_direct_20260903_184738/checkpoints/local_diffusion.rejected_validation.pt}"
DIFFUSION_CHECKPOINT_STATUS="${DIFFUSION_CHECKPOINT_STATUS:-rejected_validation_diagnostic_only}"

STAMP=$(date +%Y%m%d_%H%M%S)
CAPTURE_DIR="outputs/generation_stage_capture_linked_${STAMP}"
CAPTURE_OUT="$CAPTURE_DIR/fresh_audio_final.npy"
CAPTURE_REPORT="$CAPTURE_DIR/fresh_audio_final.report.json"
PROVENANCE="$CAPTURE_DIR/verified_development_provenance.json"
ENV_FILE="$CAPTURE_DIR/captured_runtime.env"
LOG_PATH="logs/refiner_linked_capture_b3_replay_${STAMP}.log"
STATUS_PATH="$CAPTURE_DIR/full_chain.exit_status.txt"

mkdir -p "$CAPTURE_DIR" logs
printf '%s\n' "$CAPTURE_DIR" > outputs/LATEST_REFINER_LINKED_CAPTURE
printf '%s\n' "$LOG_PATH" > outputs/LATEST_REFINER_LINKED_FULL_CHAIN_LOG
printf '%s\n' "$STATUS_PATH" > outputs/LATEST_REFINER_LINKED_FULL_CHAIN_STATUS

exec > >(tee -a "$LOG_PATH") 2>&1
trap '
status=$?
printf "%s\n" "$status" > "$STATUS_PATH"
echo "exit_status=$status"
echo "finished_at=$(date --iso-8601=seconds)"
' EXIT

echo "===== LINKED CAPTURE + FROZEN B3 + REPLAY START ====="
echo "started_at=$(date --iso-8601=seconds)"
echo "commit=$EXPECTED_COMMIT"
echo "capture_dir=$CAPTURE_DIR"

[[ "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[FATAL] MAX_ITERATIONS must be a positive integer" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || {
  echo "[FATAL] commit mismatch" >&2
  exit 2
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "[FATAL] Git worktree is dirty" >&2
  git status --short >&2
  exit 2
}
for file in "$LEGACY_REPORT" "$SCHEDULE" "$OLD_CASES" "$PRIOR_BUNDLE" "$FROZEN_DIFFUSION"; do
  [[ -s "$file" ]] || {
    echo "[FATAL] missing input: $file" >&2
    exit 2
  }
done

INPUTS_JSON="$CAPTURE_DIR/inputs.json"
"$PYTHON_BIN" - "$LEGACY_REPORT" "$OLD_CASES" "$PRIOR_BUNDLE" "$FROZEN_DIFFUSION" "$INPUTS_JSON" "$ENV_FILE" "$DIFFUSION_CHECKPOINT_STATUS" <<'PY'
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


legacy_path, cases_path, bundle_path, diffusion_path, output_path, env_path = map(
    lambda value: Path(value).resolve(), sys.argv[1:7]
)
diffusion_status = sys.argv[7]
legacy = json.loads(legacy_path.read_text(encoding="utf-8-sig"))
cases = json.loads(cases_path.read_text(encoding="utf-8-sig"))
bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))

case_hashes = {
    str(row.get("metadata", {}).get("bundle_sha256", "")).lower()
    for row in cases.get("cases", [])
}
if case_hashes != {sha256(bundle_path)}:
    raise SystemExit("[FATAL] prior cases do not select the prior bundle")
if bundle.get("schema") != "generation_round_bundle_v1":
    raise SystemExit("[FATAL] invalid prior bundle schema")

config_path = Path(bundle["config_path"]).resolve()
if not config_path.is_file() or sha256(config_path) != bundle.get("config_sha256"):
    raise SystemExit("[FATAL] captured config SHA256 mismatch")
refiner = bundle.get("checkpoints", {}).get("refiner", {})
refiner_path = Path(refiner.get("path", "")).resolve()
if refiner.get("active") is not True or not refiner_path.is_file():
    raise SystemExit("[FATAL] captured Refiner is unavailable")
if sha256(refiner_path) != refiner.get("sha256"):
    raise SystemExit("[FATAL] captured Refiner SHA256 mismatch")

audio_path = Path(legacy.get("audio", "")).resolve()
db_path = Path(legacy.get("db", "")).resolve()
for label, path in (("audio", audio_path), ("db", db_path), ("diffusion", diffusion_path)):
    if not path.is_file():
        raise SystemExit(f"[FATAL] {label} missing: {path}")

environment = {}
environment.update(legacy.get("closed_loop", {}).get("env", {}))
environment.update(legacy.get("closed_loop", {}).get("diversity_env", {}))
environment.update(bundle.get("runtime_environment", {}))
environment.update({
    "BOUNDARY_STAGE_DIAGNOSTICS": "1",
    "BOUNDARY_DIAGNOSTIC_SLOTS": "25,26",
    "MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS": "1",
    "MOTION_ENABLE_REFINER": "1",
    "BOUNDARY_USE_REFINER": "1",
    "MOTION_ENABLE_DIFFUSION": "1",
    "BOUNDARY_USE_DIFFUSION": "1",
    "MOTION_ENABLE_TRUE_IK": "1",
    "BOUNDARY_USE_IK": "1",
    "MOTION_DIFFUSION_CHECKPOINT_STATUS": diffusion_status,
})
for key in environment:
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", str(key)):
        raise SystemExit(f"[FATAL] invalid environment key: {key!r}")
env_path.write_text(
    "".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in sorted(environment.items())),
    encoding="utf-8",
)

result = {
    "legacy_report": {"path": str(legacy_path), "sha256": sha256(legacy_path)},
    "prior_bundle": {"path": str(bundle_path), "sha256": sha256(bundle_path)},
    "config": {"path": str(config_path), "sha256": sha256(config_path)},
    "audio": {"path": str(audio_path), "sha256": sha256(audio_path)},
    "db": {"path": str(db_path), "sha256": sha256(db_path)},
    "refiner": {"path": str(refiner_path), "sha256": sha256(refiner_path)},
    "diffusion": {
        "path": str(diffusion_path),
        "sha256": sha256(diffusion_path),
        "selection_status": diffusion_status,
        "production_eligible": False,
    },
}
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

value() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["path"])' "$INPUTS_JSON" "$1"
}

CONFIG=$(value config)
AUDIO=$(value audio)
DB=$(value db)
REFINER=$(value refiner)
DIFFUSION=$(value diffusion)
REFINER_SHA=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["refiner"]["sha256"])' "$INPUTS_JSON")
DIFFUSION_SHA=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["diffusion"]["sha256"])' "$INPUTS_JSON")

# Restore the captured thresholds and switches, then enable the requested
# frozen Diffusion stage without changing any acceptance threshold.
source "$ENV_FILE"

echo "===== CONTROLLED LINKED CAPTURE ====="
set +e
time "$PYTHON_BIN" -m routing.boundary_closed_loop generate \
  --config "$CONFIG" \
  --audio "$AUDIO" \
  --slots_json "$SCHEDULE" \
  --db "$DB" \
  --refiner "$REFINER" \
  --diffusion "$DIFFUSION" \
  --out "$CAPTURE_OUT" \
  --json "$CAPTURE_REPORT"
CAPTURE_STATUS=$?
set -e
echo "capture_status=$CAPTURE_STATUS"
if [[ "$CAPTURE_STATUS" -ne 0 && "$CAPTURE_STATUS" -ne 2 ]]; then
  echo "[FATAL] controlled capture crashed" >&2
  exit "$CAPTURE_STATUS"
fi
[[ "$(sha256sum "$REFINER" | awk '{print $1}')" == "$REFINER_SHA" ]] || {
  echo "[FATAL] frozen Refiner changed during capture" >&2
  exit 2
}
[[ "$(sha256sum "$DIFFUSION" | awk '{print $1}')" == "$DIFFUSION_SHA" ]] || {
  echo "[FATAL] frozen Diffusion changed during capture" >&2
  exit 2
}
[[ -s "$CAPTURE_REPORT" ]] || {
  echo "[FATAL] controlled capture did not write its report" >&2
  exit 2
}

NEW_BUNDLE=$("$PYTHON_BIN" - "$CAPTURE_REPORT" "$REFINER_SHA" "$DIFFUSION_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
entry = report.get("stage_reports", {}).get("generation_stage_diagnostics")
if not isinstance(entry, dict):
    raise SystemExit("[FATAL] linked capture report has no selected bundle")
path = Path(entry.get("path", "")).resolve()
if not path.is_file() or sha256(path) != entry.get("sha256"):
    raise SystemExit("[FATAL] selected linked bundle SHA256 mismatch")
bundle = json.loads(path.read_text(encoding="utf-8-sig"))
for name, expected in (("refiner", sys.argv[2]), ("diffusion", sys.argv[3])):
    checkpoint = bundle.get("checkpoints", {}).get(name, {})
    if checkpoint.get("active") is not True or checkpoint.get("sha256") != expected:
        raise SystemExit(f"[FATAL] linked bundle does not bind active frozen {name}")
print(path)
PY
)
printf '%s\n' "$NEW_BUNDLE" > "$CAPTURE_DIR/selected_bundle.path"
echo "selected_bundle=$NEW_BUNDLE"

echo "===== BUILD NEW BUNDLE-BOUND TRAIN PROVENANCE ====="
"$PYTHON_BIN" - "$NEW_BUNDLE" "$DB" "$PROVENANCE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


bundle_path, db_path, output_path = map(Path, sys.argv[1:])
bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
with np.load(db_path, allow_pickle=True) as payload:
    paths = np.asarray(payload["paths"], dtype=object)
    source_uids = np.asarray(payload["source_uids"], dtype=object)
    recording_uids = np.asarray(payload["recording_uids"], dtype=object)
lookup = {str(Path(str(path)).resolve()): index for index, path in enumerate(paths)}
assembly = bundle["assembly"]
slot_index = {int(row["slot"]): index for index, row in enumerate(assembly)}
selected = set()
for slot in (25, 26):
    index = slot_index[slot]
    if index < 1:
        raise SystemExit(f"[FATAL] slot {slot} has no preceding event")
    selected.update(str(Path(assembly[item]["event_path"]).resolve()) for item in (index - 1, index))
events = {}
for event_path in sorted(selected):
    if event_path not in lookup:
        raise SystemExit(f"[FATAL] captured event absent from train DB: {event_path}")
    index = lookup[event_path]
    path = Path(event_path)
    events[event_path] = {
        "event_sha256": sha256(path),
        "source_uid": str(source_uids[index]),
        "recording_uid": str(recording_uids[index]),
        "split": "train",
        "position_stratum": "captured_train_route_failure",
    }
result = {
    "schema": "refiner_action_feasibility_development_provenance_v1",
    "development_only": True,
    "bundle": {"path": str(bundle_path.resolve()), "sha256": sha256(bundle_path)},
    "event_db": {"path": str(db_path.resolve()), "sha256": sha256(db_path)},
    "events": events,
}
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"provenance": str(output_path.resolve()), "events": len(events)}, ensure_ascii=False))
PY

echo "===== FROZEN B3 + DIFFUSION/IK/GATE REPLAY ====="
PYTHON_BIN="$PYTHON_BIN" bash scripts/run_refiner_frozen_b3_pipeline_dev.sh \
  "$CAPTURE_DIR" \
  "$OLD_CASES" \
  "$EXPECTED_COMMIT" \
  "$MAX_ITERATIONS" \
  "$CAPTURE_REPORT" \
  "$PROVENANCE"

echo "===== LINKED CAPTURE + FROZEN B3 + REPLAY COMPLETE ====="
