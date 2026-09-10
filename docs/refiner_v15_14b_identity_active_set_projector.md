# V15.14b identity-preserving active-set projector

V15.14 showed that absolute segment-anchor IK reduces its own foot residual but
dominates finite MGDA candidates and reverses every endpoint/temporal objective.
V15.14b replaces that target with an identity-preserving active set.

The projector has the following contract:

- `projector(baseline) == baseline` at numerical precision. A failed zero-edit
  control is reported as `projector_not_identity_at_anchor`.
- Static support is inferred only from the observed reference. Hidden clean
  motion is not consumed by the projector.
- Foot skate, support drift, and penetration are activated and counted
  separately. Only a candidate-induced regression relative to the current
  baseline contributes an IK residual row.
- Horizontal skate/drift use X/Z Jacobian rows; penetration uses the vertical
  row. The target removes the newly introduced violation instead of forcing the
  complete support trajectory onto a segment-start anchor.
- Ownership and C2 taper stiffness are inside the DLS normal equation. D2 and
  D3 temporal regularization are solved in the same ownership-window system.
  No solved update is multiplied by a taper afterward.
- Every linear system uses `torch.linalg.solve`; pseudoinverse and explicit
  inverse operations are forbidden.

Raw and projected candidates receive separate exact-Guard audits. A candidate
is effective only when the raw MGDA candidate has a real observable decrease,
the projected candidate retains a real decrease, the immutable exact Guard
passes, the identity control passes, and ownership scope is safe.

Run the offline server probe with:

```bash
EXPECTED_COMMIT=<full-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_14b_identity_projector_probe_server.sh
```

The runner starts no training, pilot, promotion, full replay, or generation.
Physical, fixed-support, fidelity, boundary, and observable 0.03 thresholds are
inherited unchanged from the source diagnostic's immutable Guard contract.
