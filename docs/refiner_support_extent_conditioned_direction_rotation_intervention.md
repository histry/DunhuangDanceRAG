# Support-Extent Conditioned Direction Rotation (SECDR)

SECDR is the second and final Phase 3 intervention after the frozen Phase 2.1
mechanism adjudication and the negative BCTR result.  It tests the
single-recording direction/alignment bottleneck while preserving the current
production metric and gate.

## Frozen inputs and model boundary

The command consumes one explicit Phase 2.1 report, one explicit frozen BCTR
report, and the separately generated BCTR reporting-correction artifact.  It
follows every source, trajectory, RCSP, Phase 1, decomposition, parameter
attribution, and adapter-checkpoint path recorded in the Phase 2.1 lineage.
No latest-artifact search is performed.

The A0 step-400 `ProductManifoldTemporalRefiner` and the completed RCSP adapter
are loaded and frozen.  SECDR adds only
`TangentDirectionRotator`: a zero-initialized, bias-free root 3x3 map and joint
72x72 map, exactly 5193 parameters.  It rotates the already binary-support-
projected RCSP geometric correction between the RCSP projection and the
unchanged production decoder.  Contact channels, the decoder, smoothing,
taper, caps, retraction, objective, thresholds, and production inference are
unchanged.  Single-recording controls bypass the rotation with effective
`q=0` and must equal RCSP exactly.

The support condition is authoritative geometric support from production
`_refiner_decode_masks`: `weight > 0`, with active frame defined by any root or
joint geometric coordinate being active.  For each case
`s=N_active/T`.  `s_min` and `s_max` are calibrated only on the cross-event
cases of frozen TRAIN transaction 0; `q=clip((s-s_min)/(s_max-s_min),0,1)`.
Width is evaluation/reporting metadata only and is never passed to the model.

For each root/joint block, the rotator uses

```text
u = W a
u_perp = u - a (a^T u)/(||a||^2 + 1e-12)
v = a + q(s) u_perp
a' = v ||a|| / max(||v||, 1e-12)
```

Zero blocks return unchanged.  The binary support is reapplied after the
rotation.  Temporal alignment uses the existing authoritative
`refiner_temporal_action_alignment_audit` gradient; no new direction,
cosine, width, or rank loss is introduced.

## Fixed training and decision

Only the 96 cross-event cases in fixed TRAIN transaction 0 are used, for
exactly 400 checked AdamW/Armijo/rollback attempts.  The existing RCSP
`training_total`, observable, endpoint, physical, safety, and clean-identity
components are reused without changing weights or thresholds.  The fixed
64-case final bank is held out for evaluation and includes 32 primary
cross-event cases (four 8-case groups) plus 32 single-recording controls.

The report includes case-level BASE/RCSP/SECDR metrics and gates, support
extent/q, action norms, temporal alignment, block-norm preservation, support
and contact parity, safety non-regression, summaries for the required scopes,
width gaps, efficacy, mechanism, decision, and immutable state hashes.

The decision is fixed to FULL, PARTIAL, MECHANISM ONLY, or NOT SUPPORTED.  All
scientific acceptance, publish, and Pilot flags remain false, and no further
intervention search is authorized after SECDR.

## Server execution

Run on the RTX 4090 server after the final `main` commit has been pushed.  The
correction is created in a fresh directory and the original BCTR report is not
overwritten.

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail

git fetch origin main
git merge --ff-only origin/main

export PY=/home/disk/lsm/conda_envs/edge/bin/python
export ROOT_DIR=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915
export EXPECTED_MAIN_COMMIT=<FINAL_MAIN_SHA>
export LEGACY_CORE_STRENGTH=0.02
export LEGACY_TRANSITION_STRENGTH=1.0
export PHASE21_REPORT="$ROOT_DIR/audits/width_mechanism_adjudication_20260903_074314_6Vs6w5/result/report.json"
export BCTR_REPORT="$ROOT_DIR/interventions/bctr_temporal_reduction_20260903_092435_wPdK3U/result/report.json"

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -f "$PHASE21_REPORT"
test -f "$BCTR_REPORT"

"$PY" -m pytest -q \
  tests/test_refiner_bctr_reporting_correction.py \
  tests/test_refiner_support_extent_direction_rotation_intervention.py \
  tests/test_refiner_boundary_crossing_temporal_reduction_intervention.py \
  tests/test_refiner_width_mechanism_adjudication_audit.py \
  tests/test_refiner_cross_width_normalization_audit.py \
  tests/test_refiner_role_conditioned_support_projection.py

CORRECTION_DIR="$(mktemp -d "$ROOT_DIR/audits/bctr_reporting_correction_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
bash scripts/audit_refiner_bctr_reporting_correction.sh \
  "$BCTR_REPORT" "$CORRECTION_DIR"
export BCTR_CORRECTION_REPORT="$CORRECTION_DIR/result/report.json"

"$PY" - "$BCTR_REPORT" "$BCTR_CORRECTION_REPORT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

bctr_path, correction_path = map(Path, sys.argv[1:])
bctr_before = sha256(bctr_path)
r = json.loads(correction_path.read_text(encoding="utf-8"))
assert r["schema"] == "refiner_bctr_reporting_correction_v1"
assert r["completed"] is True
assert r["provenance"]["source_bctr_report_sha256"] == bctr_before
assert r["source_report_modified"] is False
assert r["measurements_changed"] is False
assert r["decision_inputs_changed"] is False
assert r["scientific_classification_changed"] is False
assert r["decision"]["result"] == "METRIC_SUPPORT_TIME_INTERVENTION_NOT_SUPPORTED"
assert r["decision"]["publish_allowed"] is False
assert r["decision"]["pilot_allowed"] is False
for scope in ("overall", "seen", "new"):
    s = r["corrected_summaries"][scope]
    assert set(s["width10_newly_rescued_cases"]) | set(s["width28_newly_rescued_cases"]) == set(s["newly_rescued_cases"])
assert sha256(bctr_path) == bctr_before
print("BCTR_CORRECTION_VERIFICATION_OK")
PY

RUN_DIR="$(mktemp -d "$ROOT_DIR/interventions/secdr_direction_rotation_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
bash scripts/run_refiner_support_extent_direction_rotation_intervention.sh \
  "$PHASE21_REPORT" "$BCTR_REPORT" "$BCTR_CORRECTION_REPORT" "$RUN_DIR"

REPORT="$RUN_DIR/result/report.json"
test -s "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert r["schema"] == "refiner_support_extent_conditioned_direction_rotation_intervention_v1"
assert r["completed"] is True
assert r["cohort"]["primary_cases"] == 32
assert r["training"]["transaction_index"] == 0
assert r["training"]["cross_event_cases"] == 96
assert r["training"]["accepted_plus_rollback_equals_steps"] is True
assert r["training"]["steps"] == 400
assert r["initial_parity"]["train_cross_event_transaction_0"]["verified"] is True
assert r["initial_parity"]["fixed_final_64"]["verified"] is True
assert r["control"]["exact_rcsp_parity"] is True
assert r["intervention"]["bctr_recomputed"] is False
assert r["intervention"]["bctr_used_for_candidate_evaluation"] is False
assert r["intervention"]["production_temporal_metric_changed"] is False
assert r["intervention"]["gate_threshold_changed"] is False
assert r["state_integrity"]["production_model_modified"] is False
assert r["state_integrity"]["scientific_acceptance"] is False
assert r["scientific_acceptance"] is False
assert r["publish_allowed"] is False
assert r["pilot_allowed"] is False
assert r["no_further_intervention_search"] is True
for scope in ("overall", "seen", "new"):
    assert scope in r["summaries"]
for group in (
    "seen/cross_event/10", "seen/cross_event/28",
    "new_position/cross_event/10", "new_position/cross_event/28",
):
    assert r["summaries"][group]["cases"] == 8
print("SECDR_REPORT_VERIFICATION_OK")
print("DECISION =", r["decision"]["result"])
print("NEXT_ACTION =", r["decision"]["next_action"])
print("REPORT =", sys.argv[1])
PY

test -z "$(git status --porcelain)"
echo SECDR_SERVER_AUDIT_OK
echo RUN_DIR="$RUN_DIR"
echo REPORT="$REPORT"
```
