# V15.15 observable-conditioned cross Adapter

V15.15 is a development-only response to the V15.14h result. Cases 16 and 29
both admitted exact-closure raw and projected directions at a fixed `1e-4`
radius, while the former shared and hand-built action bases did not. This
supports testing a learned output-space Adapter; it does not establish held-out
generalization or authorize formal training.

The existing temporal 1D-CNN, shared 79D projection and conservative single
path remain intact. The optional Adapter reads shared features plus 19
inference-visible condition channels: anchor FK gap, endpoint velocity gap,
root position and velocity mismatch, wrapped root yaw mismatch, eight anchor
support values and six relative seam-phase channels. It consumes no
single/cross role label, source identity or hidden clean interior.

The Adapter produces only the 75 geometry-tangent channels. Its last layer is
zero initialized. A continuous learned gate is multiplied by a calibrated
observable dead-zone, so difficulties at or below the frozen single-control
ceiling produce exact zero Adapter output. Ownership and the same inward
quintic C2 taper used by the decoder are applied inside the Adapter before its
output is added to the shared path. The four contact-logit channels are always
zero in this branch.

Teacher construction accepts one or more frozen V15.14h reports, verifies
complete numeric audits, exact scope and unchanged hard gates, and preserves
the accepted Projector factor and audit lineage. Projected cross directions are
stored as 75D tangent teachers; every single case in the frozen transaction is
stored only as a zero-output and gate-calibration example. Audit labels remain
metadata and never enter the model forward call. Frozen clean tensors are
retained only because the unchanged fixed Guard audits its existing
fidelity/identity branch; they never enter the Adapter repair forward call or
its observable condition vector.

The 50-step probe freezes every shared and FiLM parameter. It trains only the
Adapter with directional cosine distillation, a relaxed upper amplitude bound,
direct endpoint/temporal objectives, single-control identity loss and the
differentiable form of the fixed Guard. Gradient norm is clipped. The DLS
Projector is stop-gradient during this initial probe and is used only after a
raw fixed-radius candidate passes. Every reported candidate is still judged by
the unchanged physical, fixed-support, fidelity, boundary and observable
`0.03` closure.

The server runner first expands the oracle over a deterministic prefix of each
cross group, aggregates the existing successful cases with new results, and
then runs the short Adapter probe. Its saved state is explicitly diagnostic
and cannot be promoted. Formal Adapter training remains forbidden until the
expanded teacher report and the short probe both pass their independent gates.
