# V15.15g1f2 finite-gap angular feasibility SQP

V15.15g1f2 is a train-calibration-only diagnostic. It keeps the frozen 2/3/5
correction budgets, 12-angle ladder, exact `1e-4` radius, ownership mask,
Projector, fixed Guard contract and observable `0.03` gate. It does not run case
53, development validation, final held-out evaluation, training, teacher
recycling or generation.

For each frozen angle `theta_k`, the implementation constructs a separate
three-constraint active-set subproblem. Endpoint and temporal bounds contain the
current finite distance to their strict pass lines. Shadow uses the train-frozen
minimum hard-shadow reduction. Jacobians are taken with respect to the free
coordinate `z`, while the solver converts them to the true physical angular
covector for `p = c2_taper * scope_null_projection(z)`. The recorded derivative
therefore includes the exact factor `||d|| / ||p||` and is not row-normalized.

The report distinguishes two outcomes:

- `insufficient_linearized_science_progress`: no frozen angle has a first-order
  active set that predicts crossing the endpoint, temporal and shadow gaps.
- `finite_radius_model_mismatch`: an angle-specific first-order model predicts
  feasibility, but the authoritative geodesic, Retraction, FK and hard Guard
  trial fails.

`zero_science_gradient` remains the cause code for a missing science direction;
its overall case status is `empty_geodesic_joint_feasible_intersection`. A
near-zero temporal mismatch triggers a float64 symmetric finite-difference
ladder at `1e-5`, `3e-5`, `1e-4` and `3e-4` radians. The implementation does not
construct or use a Hessian.

Server entry point:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1f2_finite_gap_angular_feasibility_server.sh
```

The runner requires a clean checkout whose `HEAD` and `origin/main` both equal
`EXPECTED_COMMIT`, and writes the train report path to
`outputs/LATEST_REFINER_V15_15G1F2_TRAIN_REPORT`.
