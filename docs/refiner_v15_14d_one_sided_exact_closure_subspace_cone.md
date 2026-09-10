# V15.14d one-sided exact-closure subspace cone

V15.14d is a development-only probe. It does not train, promote a checkpoint,
run a full replay, or generate a video.

The probe reconstructs the same fixed Anchor bank and immutable Guard used by
the failed Refiner diagnostic. It keeps the eight `group × endpoint/temporal`
negative gradients as separate RMS-normalized parameter-space directions.
For every direction, a small one-sided finite difference measures the actual
change of all eight signed `observable_*_0p03` residuals. This small radius is
diagnostic only and cannot count as learning.

The probe enumerates all one-, two-, and three-direction subsets. Within each
subset, a deterministic simplex search chooses nonnegative coefficients whose
sum is one. Every resulting direction is materialized at an output tangent RMS
of at least `1e-4` and evaluated by the real fixed-bank closure. Linear finite
difference predictions do not decide acceptance. A raw combination passes only
when all eight residuals are nonregressing within their numeric audit tolerance,
at least one residual has a resolved strict decrease, and every physical,
fixed-support, fidelity, boundary, and observable `0.03` Guard component passes.

Only a passing raw combination enters the identity-preserving scientific
contact projector. Projector strengths `1`, `1/2`, `1/4`, `1/8`, and `1/16`
are each re-audited through full exact closure. If no raw combination passes at
the required radius, the report records
`no_resolved_exact_closure_common_descent_at_required_radius`.

The fixed Anchor and all final thresholds remain unchanged. The report marks
all threshold-change fields false and records the independent finite
differences, sparse coefficients, predicted and actual per-objective deltas,
Guard blockers, projector trials, scope safety, and numeric audit completion.
