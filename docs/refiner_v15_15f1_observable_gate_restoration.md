# Refiner V15.15f1: observable gate restoration

V15.15f1 addresses the multi-transaction gate-starvation result. The frozen
training bank calibrated a scalar physical dead-zone at the maximum single
difficulty. Twenty-seven of twenty-nine projected cross teachers fell below
that scalar floor and therefore received exactly zero Adapter output and zero
direction gradient.

The repair adds a learned residual to the existing physical difficulty before
the unchanged exact dead-zone:

```text
wake_score = physical_difficulty + observable_residual(features)
support = 1 - exp(-relu(wake_score - frozen_single_floor)^2)
```

The residual consumes only endpoint/temporal/root/yaw/support observables.
Offline teacher/control metadata supplies a margin loss during development,
but is not serialized into or consumed by inference. Exact projected teachers
are pushed above the frozen floor and identity controls below it. Transaction
and cross-group balancing remain active.

The 100-step server probe audits the frozen train bank every 10 steps and then
performs an audit-only pass on the disjoint validation bank. It preserves the
exact `1e-4` radius, ownership mask, Retraction, Projector, fixed Guard and
`0.03` observability contract. It emits only a development state and cannot
authorize formal training, pseudo-teacher generation, replay or video output.
