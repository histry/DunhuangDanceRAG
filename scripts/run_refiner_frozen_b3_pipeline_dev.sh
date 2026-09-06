#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CAPTURE_ROOT="${1:-outputs/generation_stage_capture_20260905_205755}"
OLD_CASES="${2:-$CAPTURE_ROOT/refiner_failure_cases_selected_round2/cases.json}"
EXPECTED_COMMIT="${3:-$(git rev-parse HEAD)}"
MAX_ITERATIONS="${4:-32}"
SOURCE_REPORT="$CAPTURE_ROOT/fresh_audio_final.report.json"

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
[[ -s "$SOURCE_REPORT" ]] || {
  echo "[FATAL] source report missing: $SOURCE_REPORT" >&2
  exit 2
}
[[ -s "$OLD_CASES" ]] || {
  echo "[FATAL] old case manifest missing: $OLD_CASES" >&2
  exit 2
}

STAMP=$(date +%Y%m%d_%H%M%S)
CASE_DIR="outputs/refiner_failure_cases_frozen_v1_${STAMP}"
FEAS_DIR="outputs/refiner_feasibility_dev/frozen_v1_b3_${STAMP}"
REPLAY_DIR="outputs/refiner_solution_replay/frozen_v1_b3_${STAMP}"
LOG_PATH="logs/refiner_frozen_b3_replay_${STAMP}.log"
STATUS_PATH="outputs/refiner_frozen_b3_replay_${STAMP}.exit_status.txt"

mkdir -p outputs/refiner_feasibility_dev outputs/refiner_solution_replay logs

printf '%s\n' "$CASE_DIR" > outputs/LATEST_REFINER_FROZEN_CASES
printf '%s\n' "$FEAS_DIR" > outputs/LATEST_REFINER_FROZEN_B3_RUN
printf '%s\n' "$REPLAY_DIR" > outputs/LATEST_REFINER_SOLUTION_REPLAY
printf '%s\n' "$LOG_PATH" > outputs/LATEST_REFINER_FROZEN_B3_LOG
printf '%s\n' "$STATUS_PATH" > outputs/LATEST_REFINER_FROZEN_B3_STATUS

exec > >(tee -a "$LOG_PATH") 2>&1

trap '
status=$?
printf "%s\n" "$status" > "$STATUS_PATH"
echo "exit_status=$status"
echo "finished_at=$(date --iso-8601=seconds)"
' EXIT

echo "===== FROZEN B3 DEVELOPMENT PIPELINE START ====="
echo "started_at=$(date --iso-8601=seconds)"
echo "commit=$EXPECTED_COMMIT"
echo "source_report=$SOURCE_REPORT"

BUNDLE=$("$PYTHON_BIN" - "$SOURCE_REPORT" <<'PY'
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
entry = report["stage_reports"]["generation_stage_diagnostics"]
path = Path(entry["path"]).resolve()
if not path.is_file():
    raise SystemExit(f"[FATAL] selected bundle missing: {path}")
if sha256(path) != entry["sha256"]:
    raise SystemExit("[FATAL] selected bundle SHA256 mismatch")
print(path)
PY
)

REFINER=$("$PYTHON_BIN" - "$BUNDLE" <<'PY'
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


bundle = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
entry = bundle["checkpoints"]["refiner"]
path = Path(entry["path"]).resolve()
if entry.get("active") is not True:
    raise SystemExit("[FATAL] captured Refiner was inactive")
if not path.is_file():
    raise SystemExit(f"[FATAL] captured Refiner missing: {path}")
if sha256(path) != entry["sha256"]:
    raise SystemExit("[FATAL] captured Refiner SHA256 mismatch")
print(path)
PY
)

PROVENANCE=$("$PYTHON_BIN" - "$OLD_CASES" <<'PY'
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


old = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
wanted = str(old.get("provenance_sha256", "")).lower()
if not wanted:
    raise SystemExit("[FATAL] old cases have no provenance_sha256")
for path in Path("outputs").rglob("*.json"):
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            continue
        if sha256(path) != wanted:
            continue
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if value.get("development_only") is True and isinstance(value.get("events"), dict):
            print(path.resolve())
            raise SystemExit(0)
    except (OSError, UnicodeError, json.JSONDecodeError):
        continue
raise SystemExit("[FATAL] matching development provenance was not found")
PY
)

printf '%s\n' "$SOURCE_REPORT" > outputs/LATEST_REFINER_SOURCE_REPORT
printf '%s\n' "$BUNDLE" > outputs/LATEST_REFINER_CAPTURE_BUNDLE
printf '%s\n' "$PROVENANCE" > outputs/LATEST_REFINER_PROVENANCE
printf '%s\n' "$REFINER" > outputs/LATEST_REFINER_FROZEN_CHECKPOINT

echo "bundle=$BUNDLE"
echo "provenance=$PROVENANCE"
echo "refiner=$REFINER"
echo "case_dir=$CASE_DIR"
echo "feasibility_dir=$FEAS_DIR"
echo "replay_dir=$REPLAY_DIR"

echo "===== EXPORT HASH-BOUND FROZEN PROPOSALS ====="
"$PYTHON_BIN" -m training.generation_stage_diagnostics export \
  --bundle "$BUNDLE" \
  --provenance "$PROVENANCE" \
  --slots 25 26 \
  --context-frames 16 \
  --proposal-stage refiner \
  --output-dir "$CASE_DIR"

[[ -s "$CASE_DIR/cases.json" ]]
[[ -s "$CASE_DIR/config.json" ]]

echo "===== RUN B0/B1/B2/B3 FEASIBILITY ====="
time bash scripts/run_refiner_action_feasibility_dev.sh \
  "$CASE_DIR/cases.json" \
  "$FEAS_DIR" \
  --config "$CASE_DIR/config.json" \
  --v1-checkpoint "$REFINER" \
  --source-commit "$EXPECTED_COMMIT" \
  --dirty-state clean \
  --seed 42 \
  --max-iterations "$MAX_ITERATIONS" \
  --iteration-detail full

echo "===== B3 ACCEPTANCE ====="
"$PYTHON_BIN" - "$FEAS_DIR/report.json" <<'PY'
import json
import sys
from pathlib import Path


report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
b2 = report["baselines"]["B2_frozen_v1_proposal"]
b3 = report["baselines"]["B3_frozen_v1_proposal_plus_action_solver"]
print(json.dumps({
    "B2": b2,
    "B3": b3,
    "training_started": report.get("training_started"),
    "production_model_modified": report.get("production_model_modified"),
    "next_action": report.get("decision", {}).get("next_action"),
}, ensure_ascii=False, indent=2))
accepted = (
    b2.get("cases") == 2
    and b2.get("valid_cases") == 2
    and b3.get("cases") == 2
    and b3.get("valid_cases") == 2
    and b3.get("verified_feasible") == 2
    and b3.get("joint_pass_rate") == 1.0
    and b3.get("rollback_rate") == 0.0
    and report.get("training_started") is False
    and report.get("production_model_modified") is False
)
if not accepted:
    raise SystemExit("[STOP] B3 acceptance failed; replay is forbidden")
print("B3_ACCEPTED=true")
PY

echo "===== DIFFUSION + IK + FINAL-GATE REPLAY ====="
set +e
time bash scripts/run_refiner_solution_replay_dev.sh \
  "$BUNDLE" \
  "$SOURCE_REPORT" \
  "$FEAS_DIR" \
  "$REPLAY_DIR"
REPLAY_STATUS=$?
set -e
echo "replay_status=$REPLAY_STATUS"
if [[ "$REPLAY_STATUS" -ne 0 ]]; then
  echo "[STOP] replay final gate failed; replay.report.json was preserved"
  exit "$REPLAY_STATUS"
fi

echo "===== FROZEN B3 DEVELOPMENT PIPELINE COMPLETE ====="
