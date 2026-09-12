# Refiner V15.15g1d: full-transaction shadow-gradient repair

V15.15g1d keeps the V15.15g1c discriminative conformal activation,
Adapter incumbent lock, `2/3/5` correction budgets, existing starts, exact
`1e-4` owned-tangent radius, Projector, fixed Guard, and observable `0.03`
gate. It changes the gradient seen by Euclidean and Riemannian correction.

Each local 75D tangent is inserted into an otherwise zero full-transaction
tangent before the motion is evaluated. The authoritative forward pass calls
the same `_refiner_batch_objectives` and `_diagnostic_group_guard_values`
aggregation as the fixed Guard. Exact hard group shadows determine the active
violations and every line-search acceptance decision.

For gradient direction only, cross-case hard maxima are replaced by a
LogSumExp. Its temperature is derived from fixed Guard allowances
in train transactions and serialized before held-out evaluation. Hard `p95`
indices inside each case are fixed by the one autograd graph used for an
iteration; line search does not rebuild or switch that gradient active set.
No smoothed value can pass or relax the real Guard.

The gradient is masked by the current case ownership and C2 taper. When a full
shadow is positive, alternating damped projections remove the radius-radial
component and any endpoint/temporal increasing component. Every trial is then
renormalized to the exact sphere and accepted only when all four conditions
hold:

1. the maximum exact full-transaction shadow strictly decreases by the
   train-frozen numerical amount;
2. endpoint and temporal strict descent still pass;
3. owned tangent RMS equals `1e-4`;
4. outside-scope absolute maximum equals `0.0`.

The correction report records `full_shadow_gradient_norm`,
`full_shadow_reduction`, and `step_rejection_reason`, plus per-backtrack
rejections. Rejection reasons distinguish shadow non-descent, science-cone
breakage after radius normalization, radius failure, scope leakage, exhausted
line search, and unusable gradients.

All conformal and repair numerical choices are frozen from the train bank.
The validation bank is not passed to either calibration function. The server
runner performs one held-out validation evaluation; case 53 remains an
evaluation-only cross-long case and is never recycled, promoted, or used to
change parameters.

Run the development probe on the server with:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1d_full_transaction_shadow_gradient_server.sh
```

The script starts no Adapter training, pseudo-teacher recycling, promotion,
replay, or video generation.
