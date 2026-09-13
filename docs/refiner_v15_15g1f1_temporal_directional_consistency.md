# V15.15g1f1 temporal directional-consistency repair

V15.15g1f1 is a train-only diagnostic repair of the g1f SQP coordinate path.
It does not train an Adapter, recycle pseudo-teachers, evaluate reused case 53,
consume a final held-out transaction, promote a checkpoint, replay motion, or
generate video.

## Coordinate contract

The free optimization coordinate is `z`. The physical trial direction is

```text
p = c2_taper * scope_null_projection(z)
```

Endpoint, temporal and smooth full-transaction shadow Jacobians are computed
with respect to `z`. The C2 taper is part of that forward graph and is not
applied again to those Jacobians. Sphere tangency is imposed on the physical
path by solving `dot(p, d) = 0`, where `d` is the current exact-radius tangent.
The exact geodesic update uses the same `p`.

The temporal observable remains the repository's authoritative differentiable
path: full local transaction reassembly, product-manifold retraction, float64
FK, cross-frame acceleration and jerk differences, and the observable temporal
metric. No neighbouring frame is detached from the candidate path.

## Directional consistency gate

Before the 12-angle authoritative line search, g1f1 evaluates the temporal
angular derivative in two ways:

1. Autograd with respect to `z`, converted to the derivative with respect to
   geodesic angle.
2. A symmetric finite difference through the real geodesic, retraction, FK and
   temporal metric path.

The epsilon (`1e-4` radians), relative-error tolerance (`0.1`) and absolute
comparison floor (`1e-8`) are frozen in the train repair contract. A sign
conflict reports `temporal_directional_derivative_mismatch`. A matching sign
with excessive relative error reports
`temporal_directional_relative_error_exceeded`. A rejected probe serializes
the complete `z` and physical `p` directions in its correction report.

## Failure semantics

- `empty_geodesic_joint_feasible_intersection`: the active-set solver found no
  nonzero direction satisfying its linearized joint constraints.
- `temporal_directional_derivative_mismatch`: Autograd and authoritative
  finite difference disagree in sign.
- `temporal_directional_relative_error_exceeded`: the train-frozen relative
  error bound is exceeded.
- `geodesic_authoritative_line_search_exhausted`: the solver and FD gate found
  a direction, but all 12 authoritative nonlinear trials failed.

These states remain ineligible for teacher construction.

## Frozen invariants

The correction budgets remain exactly 2/3/5 and the angular ladder remains the
same 12 train-frozen values. The `1e-4` RMS radius, ownership, C2 taper,
Projector, complete fixed Guard, physical thresholds and observable `0.03`
gate are unchanged.

## Server entrypoint

After synchronizing to the supplied clean commit, run only the train phase:

```bash
EXPECTED_COMMIT=<full-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1f1_temporal_directional_consistency_server.sh
```

The runner stops after train calibration regardless of scientific status. It
does not launch development case 53 or the final held-out transaction.
