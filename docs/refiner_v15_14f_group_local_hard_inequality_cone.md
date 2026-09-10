# V15.14f group-local one-sided hard-inequality cone

V15.14f is a development-only fixed-bank probe. It does not train, promote,
replay, generate motion, or produce video.

The probe retains the V15.14e group-local `[case, frame, 79]` product actions,
ownership mask, C2 seam taper, endpoint and temporal scientific directions,
and structured hard-constraint correction directions. The four contact action
channels remain zero; only the 75-dimensional manifold tangent is retracted.

V15.14e imposed equality against every active hard-constraint Jacobian row.
Its remaining nullspaces had no resolved endpoint/temporal component. V15.14f
instead solves the actual first-order Guard condition. For every nonnegative
within-group endpoint/temporal mixture, deterministic active-set enumeration
finds the minimum-L2 signed hard-correction coefficients satisfying:

```text
J_hard c <= 0
D_endpoint c <= 0
D_temporal c <= 0
```

At least one scientific derivative must be strictly negative. Every matrix
entry comes from the same fixed-bank one-sided exact-closure finite difference.
Rows are scaled independently for numerical conditioning, so jerk units cannot
dominate endpoint or temporal signs.

A feasible linearized direction is normalized only after the inequality solve.
The resulting candidate must reach an actual output tangent RMS of at least
`1e-4`, keep all eight observable residuals nonregressing with one resolved
strict decrease, and pass the complete immutable physical, fixed-support,
fidelity, boundary, and observable `0.03` Guard. Only then may it enter the
scientific projector and its `1`, `1/2`, `1/4`, `1/8`, `1/16` backtracking.

If no candidate survives at the required radius, the report records
`no_group_local_hard_inequality_descent_at_required_radius`. The report marks a
routing architecture pivot as supported only when both cross-short and
cross-long have an effective projected candidate, group/ownership leakage is
exactly zero, and the full audit is complete and scope-safe.

The report separates linear inequality feasibility from resolved exact-closure
evaluation. `linear_feasible_direction_by_group` counts nonzero, scope-safe
directions before radius normalization; `resolved_candidate_audit_by_group`
counts those that actually reached `1e-4` and received the full fixed audit.
