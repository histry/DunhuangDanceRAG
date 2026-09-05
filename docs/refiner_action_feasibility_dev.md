# Refiner action-feasibility development protocol

This document describes an isolated **DEVELOPMENT / NOT A FORMAL
PREREGISTRATION** diagnostic. It is not a replacement for the frozen Refiner
protocol, does not modify a production model, and does not authorize pilot or
blind-test generation.

## Scope

The diagnostic evaluates a fixed 75D geometric action through the existing
product-manifold decoder. Contact residuals are fixed to exactly zero. The
existing confidence masks, supported smoothing, taper, caps and product
retraction remain active. A candidate is accepted only when all of the
following independent checks pass for that case:

- observable endpoint and temporal gates, including jerk non-regression;
- stage-relative physical acceptance;
- fixed-reference support acceptance;
- reference-fidelity acceptance;
- finite values, zero contact residual and no edit outside the fixed support.

The whole-motion absolute physical audit is retained as a separate diagnostic;
it is not substituted for the local stage-relative contract. No hidden clean
interior is passed to the repair evaluator or solver.

The solver starts at the supplied initial action (zero for B1), uses a fixed
trust-region restoration budget, re-evaluates every candidate with the true
decoder and independent audits, then runs a fixed minimum-edit schedule after a
feasible point is found. A rejected or unsuccessful case rolls back to the
observed reference. A rollback is never counted as a rescue.

## v7 constrained restoration solver

`refiner_action_feasibility_dev_v7` merges physical/fixed-support source
metrics by canonical root before searching. Each root retains its source
metrics and uses the worst normalized residual and minimum signed margin;
the original layer-specific gates still decide acceptance. Diagnostic layer
summaries remain available, but are not independent search directions.
`dominant_failed_constraint` requires a positive/nonfinite residual;
zero-residual low-margin metrics appear only as `dominant_guard_constraint`.
The legacy `dominant_hard_constraint` now means failures only.
Zero-limit, nonnegative Rot6D nonfinite/degenerate ratios are recorded as
`boundary_invariant`: zero remains valid, positive residual still fails, and
their unattainable positive margin is excluded from guards and cone fitting.

The solver measures true decoded margins at probe radii `h`, `h/2`, and
`h/4` near a boundary. Its deterministic search basis includes observable
gradients; left/center/right and symmetric/antisymmetric seam profiles; root
x/y/z blocks; three joint groups by tangent axis; and fixed-seed block-sparse
directions. It fits and retains separate forward/backward margin gradients at
each scale instead of averaging them. Candidate directions are projected into
all active hard-margin halfspaces. A zero-margin joint-jerk or penetration
restoration requires a strictly positive derivative for every such metric.
Temporal search also protects the actual endpoint margin. Missing active
models, a zero projected direction, or an unconverged projection disable that
direction. The linear step bound uses the most adverse retained derivative
and 90% of the remaining signed margin before backtracking; every resulting
trial still passes through the original audits. These are local models, not
guarantees of nonlinear or global feasibility.

When no sampled probe passes observable and hard gates together, the solver
can accept a `hard_margin_restoration` preparation step if ordinary stage
search found no better candidate. All hard gates must pass, both observable
excesses must not increase, previously passed observable gates must remain
passed, and the minimum available margin among joint jerk, foot penetration,
and fidelity seam jerk must increase by at least 5% of its current gap to the
safety floor (and more than `comparison_tolerance`). An ordinary observable
step that spends the binding physical margin is inadmissible when no joint
feasible probe exists, unless that step itself reaches `joint_pass`.
Preparation never sets `joint_pass`; the original final conjunction and
rollback behavior remain authoritative.

Iteration output records `observable_stage`, `accepted_phase`, canonical
metric changes, one-sided probe fits, cone constraints, step bounds, and each
trial's preparation eligibility. Failed finite probes only describe the
sampled local span. The configured `0.03` observable thresholds are unchanged.
The development runner still fixes `pilot_allowed=false`; even a fully
feasible B1 development result does not itself authorize network training.

## Case manifest format

`training/refiner_action_feasibility_evaluation.py` consumes an explicit JSON
manifest with schema `refiner_action_feasibility_case_manifest_v1`:

```json
{
  "schema": "refiner_action_feasibility_case_manifest_v1",
  "cases": [
    {
      "case_id": "dev-0001",
      "role": "cross_event",
      "width": 10,
      "position_stratum": "seen",
      "split": "dev",
      "source_uid": "source-a",
      "recording_uid": "recording-a",
      "left_source_uid": "source-a",
      "right_source_uid": "source-b",
      "left_recording_uid": "recording-a",
      "right_recording_uid": "recording-b",
      "left_split": "dev",
      "right_split": "dev",
      "reference_path": "arrays/dev-0001-reference.npy",
      "seam_path": "arrays/dev-0001-seam.npy",
      "joint_mask_path": "arrays/dev-0001-joint-mask.npy",
      "root_mask_path": "arrays/dev-0001-root-mask.npy",
      "contact_mask_path": "arrays/dev-0001-contact-mask.npy",
      "condition_path": "arrays/dev-0001-condition.npy",
      "proposal_action_path": "arrays/dev-0001-v1-action.npy"
    }
  ]
}
```

The motion reference is `[T,151]`, the raw action is `[T,75]`, the joint mask
is `[T,24]`, the root mask is `[T]` or `[T,1]`, and the contact mask must be
exactly zero. Only `train`, `dev`/`development`, and `validation` splits are
accepted; `test` and `blind` are rejected. Cross-event endpoints must declare
the same split. Recording identities are checked before event/window slicing
and may not occur in multiple splits.

## Baselines and outputs

- B0: zero action, Bridge only;
- B1: zero initial action plus the bounded action solver;
- B2: frozen V1 proposal without the solver;
- B3: the frozen V1 proposal plus the same solver budget as B1.

B2/B3 are marked unavailable unless both explicit proposal actions and a
verified V1 checkpoint are supplied. Random initialization is never used as a
V1 proposal. Every run writes a fresh directory containing
`manifest.json`, `report.json`, `case_level.jsonl`, `solver_iterations.jsonl`,
and `evidence_summary.md`.

## Server execution template

After copying this branch to the server, prepare a development-only case
manifest from confirmed TRAIN/DEV data and run:

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail
export PY=/home/disk/lsm/conda_envs/edge/bin/python
export CASE_MANIFEST=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/refiner_feasibility_dev/case_manifest.json
export RUN_DIR=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/refiner_feasibility_dev/$(date +%Y%m%d_%H%M%S)

mkdir -p "$(dirname "$RUN_DIR")"
bash scripts/run_refiner_action_feasibility_dev.sh \
  "$CASE_MANIFEST" "$RUN_DIR" \
  --source-commit "$(git rev-parse HEAD)" \
  --dirty-state "$(test -z "$(git status --porcelain)" && echo clean || echo dirty)"
```

The command is intentionally separate from formal reports and does not start
network training. If the confirmed development arrays or V1 checkpoint are not
available, implementation remains complete but the corresponding diagnosis is
not an executed result.

## Scheduler diagnostic bypass

The current checkout retains `DEFAULT_MIN_CORE_FRAME_RATIO = 0.70`. The
historical frozen parent recorded `0.80`; this task does not change that
threshold. `DISABLE_FINAL_CORE_FRAME_ASSERT=1` can now return only a report
whose sole failure is the exact core-frame-ratio reason. It preserves
`ok=false` and adds `formal_pass=false`, `diagnostic_bypass_used=true`, and
`bypassed_reasons`. Empty schedules, zero core frames, source/recording
concentration, insufficient events, malformed/unknown reasons, and mixed
failures still raise `ScheduleHardConstraintError`.
