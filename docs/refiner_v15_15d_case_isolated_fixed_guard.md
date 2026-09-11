# V15.15d case-isolated fixed-Guard restoration

Development-only follow-up to the V15.15c 200-step exact-radius probe.
V15.15c established endpoint and temporal non-regression for required cases,
but case 16 exceeded only `cross_short.penetration` and case 29 exceeded only
`cross_long.support_drift_max`.  Both failures occurred before Projector
invocation.

V15.15d resumes the V15.15c Adapter state and preserves the frozen teacher
bank, shared 1D-CNN, observable gate, ownership scope, exact `1e-4` radius,
Retraction, Projector, and every final fixed-Guard threshold.

For each projected teacher case, training constructs a candidate that changes
only that case and its ownership window.  It evaluates the same differentiable
fixed-Guard values used by exact audit.  Each metric uses the immutable fixed
Anchor allowance as its normalization scale.  The differentiable safety target
is one quarter of that existing allowance:

```text
allowance_j = max(abs(anchor_j) * relative_j, absolute_j)
training_limit_j = anchor_j + 0.25 * allowance_j
excess_j = relu((value_j - training_limit_j) / allowance_j)
case_guard_loss = sum_j excess_j
```

The sum gives every active metric a gradient.  It does not modify the final
limit `anchor_j + allowance_j`.  Direction distillation decays continuously
toward `0.1` as case-isolated Guard or endpoint/temporal pressure rises, then
returns toward one inside the safety target.  Projected-teacher objectives are
averaged inside `cross_short` and `cross_long` first, then averaged across the
two groups so the six short teachers cannot drown the single long teacher.

The server runner performs a 100-step probe with exact closure every 10 steps.
It does not authorize formal training, checkpoint promotion, replay, or video
generation.  Formal Adapter training remains gated on required cases 16, 18,
20, and 29 all passing raw fixed Guard and entering Projector, with effective
projected candidates in both cross groups and the unchanged single-control,
scope, numeric-audit, and fixed-threshold criteria.
