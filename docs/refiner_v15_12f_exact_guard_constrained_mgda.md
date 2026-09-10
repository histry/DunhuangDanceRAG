# Refiner V15.12f exact-Guard-constrained subgroup MGDA

V15.12e found a nonzero eight-objective MGDA direction at every probe step,
but the fixed Guard rejected most real candidates.  The dominant coarse label
`cross_long.feasibility` combined endpoint and temporal deficits and did not
identify the physical or observable constraint that closed the feasible set.

V15.12f removes that aggregate from transaction acceptance.  The immutable
seen-train Guard bank now exposes separate differentiable components for:

- joint and extremity jerk;
- foot skate and support drift;
- penetration and fixed support;
- boundary jerk;
- geometry, contact, temporal and support fidelity;
- endpoint and temporal observable `0.03` requirements.

Every component keeps the same initial anchor and production threshold source.
No reference rolls forward and no tolerance accumulates.  Candidate audit rows
record the fixed anchor, current value, candidate value, absolute Guard limit,
remaining margins, first-order directional derivative, linear predicted delta
and real closure delta.

The eight single/cross, short/long endpoint/temporal gradients remain RMS
normalized and use deterministic Frank-Wolfe MGDA.  Guard components at or near
their immutable limits are differentiated on the exact same complete seen
training bank used by the real closure.  A deterministic halfspace projection
then requires both:

```text
task_gradient[i] dot update_direction < 0 for at least one task
task_gradient[i] dot update_direction <= 0 for every task
guard_gradient[j] dot update_direction <= 0 for every active Guard constraint
```

The physical part of the training gradient is admitted only while these task
and Guard inequalities remain satisfied.  When the unconstrained task cone is
nonempty but the exact Guard closes it, the step records
`no_exact_guard_constrained_common_descent`.  Candidates that pass only at or
below a `1e-7` backtracking scale, or whose real observable reduction is below
the audit resolution, record `resolution_limited_under_exact_guard` and do not
count as learned updates.

`optimizer_updates.jsonl` contains every real candidate audit.  The diagnostic
report summarizes concrete blocker names and categories, active constraint
gradients, constrained-common-descent counts, accepted scale distribution,
per-group decoder amplitude and independent fit-context pass counts.

Run only `scripts/run_refiner_v15_12f_probe_server.sh` first.  It is fixed to 50
development steps and cannot launch pilot training, checkpoint promotion or
video generation.  Physical, fixed-support, fidelity, boundary and observable
`0.03` final gates are unchanged.
