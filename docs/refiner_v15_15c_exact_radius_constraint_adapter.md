# V15.15c Exact-Radius Constraint-Aware Adapter Probe

V15.15c resumes the saved V15.15 Adapter and trains only the observable
Adapter and its continuous gate.  Every cross-case proposal is projected onto
the actual decoder-consistent ownership tangent sphere at RMS `1e-4` before
the differentiable endpoint, temporal and fixed-Guard objectives are measured.

The normalization uses `torch.linalg.vector_norm` above a per-coordinate
`eps` floor and a detached constant denominator below that floor. This keeps
the derivative finite at the zero initialized Adapter while placing every
resolved nonzero proposal exactly on the requested RMS sphere. PyTorch's
`torch.linalg.vector_norm` does not provide an `eps` argument.

Endpoint and temporal residuals are independently compared with the fixed
teacher-bank Anchor and its exact numeric tolerance.  A non-regression hinge
is applied to every cross case.  Case 20 receives an additional temporal
signed-residual term.  While either hinge is active, that case's directional
distillation weight is reduced, allowing the non-regression constraint to
take priority over imitating an infeasible local direction.

The single controls, label-free continuous gate, exact ownership scope,
Projector and all physical, fixed-support, fidelity, boundary and observable
`0.03` thresholds remain unchanged.  Full closure audits run every 20 steps.
The runner is development-only and cannot launch formal training or video
generation.

Teacher expansion is a separate server job.  It runs the existing V15.14h
Oracle over a larger deterministic cross-case set, merges the original and
V15.15 Oracle reports, and writes a larger teacher bank without updating the
Adapter.
