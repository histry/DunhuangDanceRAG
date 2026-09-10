# V15.14e group-local boundary-nullspace exact-closure cone

V15.14e is a development-only fixed-bank probe. It cannot train or promote a
checkpoint and cannot launch a pilot, replay, generation stage, or video.

The probe replaces shared parameter-space actions with local product actions
of shape `[case, frame, 79]`. The four contact channels remain exactly zero;
the remaining 75 channels form the product-manifold tangent used by
`product_exp_torch`. Every endpoint or temporal direction is masked to exactly
one of `single_short`, `single_long`, `cross_short`, or `cross_long`, then
restricted to its ownership window with the existing C2 seam activity.

For each group, the probe measures a one-sided finite-difference Jacobian of
the unchanged fixed Guard in a structured output subspace. That subspace
contains the group endpoint and temporal descent directions plus group-local
correction directions for active boundary, joint/extremity jerk, support,
penetration, and fidelity constraints. The endpoint and temporal directions
are projected into the measured hard-constraint row nullspace. All Jacobian
entries that control this projection come from the real fixed-bank closure,
not a proxy loss.

The two projected directions are evaluated individually and on a deterministic
nonnegative within-group simplex grid. Each raw candidate is materialized at
an output tangent RMS of at least `1e-4`. It enters the scientific contact
projector only if all eight observable residuals are nonregressing, at least
one has a numerically resolved strict decrease, and the complete fixed Guard
passes. Projector strengths `1`, `1/2`, `1/4`, `1/8`, and `1/16` are each
re-audited by the same closure.

The fixed Anchor and all physical, fixed-support, fidelity, boundary, and
observable `0.03` limits remain unchanged. If the resolved-radius group-local
cone is empty, the report records
`no_group_local_boundary_nullspace_descent_at_required_radius`.
