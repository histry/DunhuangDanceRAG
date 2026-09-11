# V15.15b Exact-Scope Observable Adapter Audit

V15.15b fixes the development Adapter probe's candidate scope.  The decoder's
temporal smoothing may spread an ownership-local raw Adapter proposal beyond
the seam.  The probe now isolates the Adapter contribution by decoding the
same frozen shared output with and without the Adapter, maps that difference
to the 79D product tangent, hard-masks it with the canonical ownership mask,
and applies the resulting patch to the fixed teacher-bank Anchor.

The exact fixed-radius contract remains defined on the applied, tapered,
decoder-consistent tangent.  Pre-taper, post-taper and decoder-consistent RMS
values are reported separately so taper-related amplitude inflation is
visible without changing the established `1e-4` radius or any physical,
fixed-support, fidelity, boundary, Projector or observable `0.03` gate.

`scope_audit_space` is the 79D product tangent action.  FK workspace movement
continues to be reported as an observability check and is not used to decide
whether frames outside the ownership window changed.

The server runner reuses the saved V15.15 teacher bank and step-50 Adapter
state.  It performs an audit-only pass and does not rerun the nonlinear
Oracle, update weights, publish a checkpoint, launch formal training, replay,
or generate video.
