# V15.15g1f3 fail-closed compute budget

This change bounds the two observed sources of unbounded wall time without
changing the ProgressPolicy, witness selection rule, exact-radius update,
Projector, or authoritative Guard.

## Constraint-generation budget

`--second-order-max-constraint-generation-depth` limits same-expansion active
witness rebuilds. `-1` preserves the original finite-universe reference path.
When the configured non-negative depth is exhausted, a newly exposed witness
still rejects the trial, but the solver does not build another curvature
model. It records `constraint_generation_budget_exhausted`, returns no partial
repair, and the composite path falls back to identity. Budget exhaustion is an
abstention, not a numeric failure and not a certificate of infeasibility.

## Adaptive reduced search

The initial tier is controlled by:

- `--second-order-initial-max-coarse-directions`
- `--second-order-initial-refinement-starts`
- `--second-order-initial-refinement-iterations`

With `--second-order-adaptive-full-search`, the full tier is executed only
when the initial tier finds no predicted-feasible candidate. The full tier is
controlled by the corresponding `--second-order-full-*` options. The trigger,
selected tier, candidate counts, and both budgets are included in each angle
solver audit.

The initial and full tiers share the same reduced quadratic model. Search
budgeting therefore does not remove curvature construction, change any Guard
row, or weaken authoritative acceptance. It can change candidate selection
and is consequently frozen in the train repair contract and packaged runtime
contract.

## Paper-2 fast protocol

`experiments/paper2/protocol_fast_budget_v1.json` freezes the first bounded
profile:

```text
constraint-generation depth: 3
initial search: 1024 / 64 / 32
full search: 8192 / 768 / 64
full-search trigger: no predicted-feasible initial candidate
progress policy: current_equal_share
witness policy: existing reactive monotone union
```

The original `protocol.json` remains unchanged so earlier full-reference
outputs retain their exact protocol hash. A fast run must use a new output
root because the run binding includes the protocol SHA256 and implementation
commit.

## Numeric-failure boundary

Finite, explicitly verified execution rejections are not numerical failures.
A zero geodesic direction/radius or a finite sphere-tangency violation remains
a fail-closed model/execution abstention. NaN/Inf in the direction, projected
direction, temporal probe, or generated trial remains a hard numerical
failure. Paper-2 jobs persist a `paper2_job_failure.json` receipt with the
exact numeric status before terminating.
