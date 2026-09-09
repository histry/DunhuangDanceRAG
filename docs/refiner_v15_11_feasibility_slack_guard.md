# V15.11: resolution-aware joint feasibility guard

V15.10 retained all exact acceptance checks and completed 400 safe optimizer
transactions, but it still did not pass scientific readiness. The seen
endpoint rate reached 0.1875 and the short cross-event temporal rate reached
5/8 on both evaluation splits. New-position endpoint repair fell to zero,
and both 28-frame temporal subgroups remained at 0/8.

The final V15.10 transaction exposed the remaining duplicated optimizer
obstruction. Endpoint and temporal component guards already ignored movement
inside their individual `1e-3` training-resolution deadbands, but their joint
feasibility sum was still compared without a deadband. A valid larger scalar
descent step was rejected when the already resolved short cross-event joint
deficit moved from about `1.053e-3` to `1.108e-3`; the transaction accepted a
step roughly eight times smaller instead.

V15.11 applies a `2e-3` training-resolution deadband to the joint feasibility
guard, exactly equal to the sum of the two component deadbands. Movement that
remains below this resolution is ignored by the optimizer transaction guard.
Crossing the joint deadband, crossing either component deadband, or increasing
any unresolved component still rejects the transaction. Complete subgroup
repair totals remain protected.

Endpoint and temporal acceptance thresholds remain exactly `0.03` and are
evaluated independently for every case. Physical, fixed-support, fidelity,
boundary and final gates are unchanged. The server runner creates a fresh
foundation and diagnostic only; it cannot launch the pilot or promote a
checkpoint.
