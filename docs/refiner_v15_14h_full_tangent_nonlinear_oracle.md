# V15.14h case-local full-tangent nonlinear feasibility oracle

V15.14h is a development-only feasibility oracle over the frozen failed
V15.13 diagnostic. It evaluates cases 16 (`cross_short`) and 29 (`cross_long`),
with cases 10 and 14 (`single_short`) as controls. It cannot train, promote,
replay, generate motion, or publish a checkpoint.

Each search variable spans every one of the 75 product-manifold tangent
channels at every owned frame of exactly one case. Other cases and frames
outside ownership remain exactly zero. The same decoder-consistent inward
quintic C2 taper used by V15.14g is applied before retraction.

Eight starts are evaluated independently: endpoint, temporal, their equal
mixture, a Top-K/epsilon-active physical witness correction and four
deterministic orthogonal directions. Optimization uses an augmented Lagrangian
over the exact differentiable case objectives and the same fixed-bank Guard
components used by closure auditing. Endpoint and temporal are both objectives
and explicit non-regression inequalities. A small tangent velocity plus
acceleration Tikhonov prior selects smoother solutions among redundant 75D
directions.

Every candidate is mapped to an actual case-local output tangent RMS of
`1e-4`. The unit-direction sphere equality is also included in the augmented
Lagrangian, so the solver cannot improve its score by shrinking the edit.
Smaller-radius candidates are never evaluated as evidence.

Every iteration performs the complete unchanged physical, fixed-support,
fidelity, boundary and observable `0.03` closure. A raw candidate must make
endpoint and temporal non-regressing, strictly improve at least one beyond
audit resolution, pass the fixed Guard, remain exactly case/ownership local,
and produce FK24 workspace displacement above the declared numerical floor.
The requested `workspace_displacement_mean` and
`workspace_displacement_max` therefore measure FK24 joint positions in metres;
they are not presented as unavailable SMPL mesh-vertex measurements.

Only a passing raw candidate enters the existing scientific DLS Projector and
the unchanged `1`, `1/2`, `1/4`, `1/8`, `1/16` backtracking audit. A routing
pivot is supported only when both primary cross cases produce effective
projected candidates with zero scope leakage and complete numeric audits.
Each projected trial is rematerialized on the same fixed `1e-4` case-local
sphere before its closure audit, so Projector interpolation cannot pass by
silently shrinking the edit radius.

If both cross cases are raw-feasible but projection fails, the report assigns
the failure to the Projector boundary. If the full-tangent, eight-start oracle
finds no raw candidate at the fixed radius, it records
`full_tangent_multistart_no_raw_feasible_at_required_radius`; this is strong
empirical evidence about the current representation and Guard, not a global
mathematical infeasibility proof.
