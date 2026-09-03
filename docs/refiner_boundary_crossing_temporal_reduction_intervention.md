# Phase 3 — Boundary-Crossing Temporal Reduction (BCTR)

Phase 3 is the first minimal intervention after the frozen Phase 2.1
adjudication.  It consumes exactly one explicit Phase 2.1 report and follows
the source, trajectory, RCSP adapter and SHA-256 lineage recorded there.  It
does not search for a latest artifact.

The frozen Phase 2.1 report must be the report produced from
`c461ba44689103cd0690488267e3bd42507ad7ab`, with mechanism
`MIXED_WIDTH_MECHANISM`.  Its conclusion fixes the intervention order:
metric/support-time first, direction second.  This experiment tests only the
first item.  It does not change direction, the decoder, the model, a
checkpoint, a threshold, or production inference.

## Hypothesis and intervention

The hypothesis is that the width-conditioned gate gap is partly caused by the
current temporal reduction including all derivative stencils that touch the
seam core.  BCTR retains only a stencil that touches both sides:

```text
core(t)       = seam(t) >= 0.5
touches_core  = any(core over the k+1 frames of a stencil)
touches_outside = any(not core over the same stencil)
BCTR_support_k = touches_core AND touches_outside
```

In report shorthand, the same core rule is written as `seam >= 0.5`.

The derivative values are exactly the production decomposition: FK joints in
`float64`, `diff(J, n=k) * fps**k`, the L2 norm over coordinates and the mean
over joints.  Acceleration is the mean over the BCTR order-2 support divided by
10; jerk is the mean over the BCTR order-3 support divided by 1000:

```text
BCTR_acc = mean_crossing(||diff(J,2) * fps^2||_2) / 10
BCTR_jerk = mean_crossing(||diff(J,3) * fps^3||_2) / 1000
BCTR_temporal = BCTR_acc + BCTR_jerk
```

The required support count is used only as implementation sanity evidence.  A
zero acceleration or jerk crossing count is invalid/null; BCTR never clamps a
zero denominator to one.  There is no width-specific support rule, tuned
band, `±5`/`±14` convention, top-k, percentile, clipping, or learned weight.

## Gate and anti-gaming contract

Only temporal energy is replaced for the candidate gate.  The production
relative-gain floor semantics remain exact:

```text
G = (before-after)/before,                         if before > 1e-6
    1,                                             if before <= 1e-6 and after <= 1e-6
    -1,                                            otherwise
```

The pass threshold is read from
`cfg.checkpoint_validation_min_temporal_repair_gain`; no `0.03` is embedded
in the intervention.  Endpoint values and endpoint acceptance are original
production values.  The jerk non-regression guard uses the original
full-support `seam_jerk_mps3`, never BCTR jerk.  Candidate overall acceptance
is original endpoint acceptance AND candidate temporal acceptance.

BASE and RCSP outputs are the same frozen outputs used for the current metric.
The model and adapter are frozen, all gradients remain absent, and no update
path or checkpoint selection exists.  The report records current metric parity
against the Phase 2.1 actual-width values for all 32 primary cases within the
frozen `2e-6` tolerance.  The observed width comparison is unpaired.  The 32
`single_recording` cases are excluded controls; no new same-boundary
counterfactual is made.

## Cohort and pre-registered decision

The scientific primary cohort is exactly:

- `seen/cross_event/10`: 8
- `seen/cross_event/28`: 8
- `new_position/cross_event/10`: 8
- `new_position/cross_event/28`: 8

For `overall`, `seen` and `new`, the report gives current and BCTR median
RCSP gate gains at widths 10 and 28, each width gap, gap shrink fraction,
current/BCTR pass counts, newly rescued width-28 cases and lost width-10
cases.  The split is supported only if all of the following hold:

1. `abs(BCTR_gap) <= 0.50 * abs(current_gap)`;
2. median width-28 BCTR gain is strictly greater than current width-28 gain;
3. median width-10 BCTR gain is not lower than current width-10 gain;
4. every primary case has valid BCTR support;
5. endpoint and original-jerk semantics, outputs and state are unchanged.

Both splits supported gives
`METRIC_SUPPORT_TIME_INTERVENTION_SUPPORTED`.  Overall support or only one
split supported gives `PARTIAL_METRIC_SUPPORT_TIME_INTERVENTION`.  Otherwise,
or when width 10 degrades, the result is
`METRIC_SUPPORT_TIME_INTERVENTION_NOT_SUPPORTED`.  The next actions are fixed
in the JSON report.  There is no further metric search after this phase;
direction intervention is the next scientific branch.

All acceptance, publish and Pilot flags remain false.  A positive result is a
frozen candidate/evidence result, not scientific acceptance or a production
promotion.

## Server execution

There is no local validation in this workflow.  Execute the
following on the RTX 4090 server after pushing the new commit:

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail

git fetch origin main
git merge --ff-only origin/main

export PY=/home/disk/lsm/conda_envs/edge/bin/python
export ROOT_DIR=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915
export EXPECTED_MAIN_COMMIT=<NEW_PHASE_3_COMMIT_SHA>
export PHASE21_REPORT="$ROOT_DIR/audits/width_mechanism_adjudication_20260903_074314_6Vs6w5/result/report.json"

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -f "$PHASE21_REPORT"

"$PY" -m pytest -q \
  tests/test_refiner_boundary_crossing_temporal_reduction_intervention.py \
  tests/test_refiner_width_mechanism_adjudication_audit.py \
  tests/test_refiner_cross_width_normalization_audit.py \
  tests/test_refiner_role_conditioned_support_projection.py

RUN_DIR="$(mktemp -d \
  "$ROOT_DIR/interventions/bctr_temporal_reduction_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

bash scripts/run_refiner_boundary_crossing_temporal_reduction_intervention.sh \
  "$PHASE21_REPORT" "$RUN_DIR"

REPORT="$RUN_DIR/result/report.json"
test -s "$REPORT"
```

The execution gate is the generated `result/report.json`.  Inspect
`decision`, `summaries`, `parity`, `case_level`, `state_integrity` and the
false acceptance/pilot flags before any separate direction intervention.
