# V15.10: resolution-aware component transaction guard

V15.9 completed all 400 optimizer transactions safely and retained the exact
physical checks, but it did not pass scientific readiness. It improved the
short cross-event temporal subgroup to 5/8 on both splits. Both 28-frame
temporal subgroups remained at 0/8, and the single-recording groups remained
weak.

The final V15.9 transaction identified a concrete optimizer obstruction. A
larger scalar descent step was rejected because the already small
`cross_short.endpoint` tail risk moved from about `3.648e-4` to `3.686e-4`.
The accepted step was roughly four times smaller. Both values are below the
existing `1e-3` smooth-CVaR resolution, while the single/long deficits are one
to two orders of magnitude larger.

V15.10 guards endpoint and temporal subgroup excess above a fixed `1e-3`
training-resolution deadband. Movement wholly inside that deadband is ignored
by the optimizer transaction guard. Crossing the deadband or increasing any
unresolved component still rejects the transaction. Complete subgroup repair
and joint-feasibility guards remain unchanged.

The endpoint and temporal acceptance thresholds remain exactly `0.03` and are
still evaluated per case. Physical, fixed-support, fidelity, boundary and final
gates are unchanged. The diagnostic remains fresh-only, nonpublishing and
cannot launch pilot training or checkpoint promotion.
