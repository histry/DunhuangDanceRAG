# Paper 2: bounded matched-trial evaluation

This protocol adds evidence collection around the existing g1f2/g1f3 solver.
It does not change witness selection, Projector, Guard acceptance, ownership,
the `1e-4` production radius, fixed angles, or correction budgets.

## Evidence separation

- `matched_candidate_curvature_audit` compares first-order prediction,
  execution-path second-order prediction and the raw trial on the identical
  anchor, ownership support, physical direction, angle and frozen scalar row.
- `authoritative_hard_guard_change_by_term` is recorded separately.  A hard
  max/P95 support transition is not silently compared with a frozen row.
- Projector and final full Guard remain the only source of end-to-end closure.
  A smaller model error is never reported as a safety or closure certificate.

## Low-compute mechanism pool

Freeze a 12--20 case train manifest before running.  The mechanism runner
collects at most the configured number of directions per case, budget and
source from:

1. g1f2 selected directions;
2. g1f3 selected directions;
3. g1f3 verified basis direction zero as a deterministic reference.

For each direction, the owned Euclidean radial unit vector and tangent unit
vector are frozen.  Jets and real trials are evaluated at `3e-5`, `1e-4`, and
`3e-4`.  No reduced Hessian, grid search, or witness rebuild search is rerun for
the extra radii.  The optional tangent-line jet removes sphere-geodesic
acceleration for explanation only and is never an accepted trial.

The JSONL artifact is append-only, bound to code/protocol/input hashes and can
resume without repeating completed candidate audits.  A caller-owned jet cache
requires an explicit anchor/direction/witness key, preventing reuse across a
witness or basis change.

## Closure schedule

1. Development: Adapter/g1f2/g1f3 at `Closure@5` and radius `1e-4`.
2. After protocol freeze: add independent `k2` and `k3` runs.  A `k5` prefix
   cannot substitute because the remaining-gap quotas differ by budget.
3. Sealed held-out: launch the frozen comparison once.  A failed or interrupted
   sealed case cannot be rerun under the same held-out claim.

Every case/method/budget job writes a completion record.  Normal development
jobs may create a new immutable attempt directory after interruption; sealed
jobs fail closed instead.

## Cost accounting

g1f3 reports separate time and call counts for curvature construction, angle
subproblems, directional checks, authoritative raw trials and mechanism audit.
The result summarizer emits:

- normalized `E1`, `E2`, `E1-E2`, log error ratio and paired win rate;
- stable-witness versus witness-transition strata;
- full-path versus no-geodesic-acceleration second-order error;
- Closure@budget, numeric failure, runtime, and function-call tables.

All radius-dependent benefit and closure benefit remain hypotheses until the
sealed evidence is complete.
