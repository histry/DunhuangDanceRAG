# Refiner V15.15g1c: fixed-Guard shadow and discriminative conformal activation

V15.15g1c keeps the V15.15g correction budgets (`2/3/5`), starts, exact
`1e-4` owned-tangent radius, Projector, fixed Guard, and observable `0.03`
gate unchanged. It changes activation calibration and candidate selection.

Activation reads only the observable severity vector. Train labels are used
offline to form separate `single` and `cross` populations. Each transaction
is held out in turn, shrinkage Mahalanobis models are fit on the remaining
transactions, and same-class distances plus the single/cross discriminant
freeze a two-sided conformal decision rule. Runtime activation does not read
role, teacher kind, `single/cross`, validation labels, or hidden clean motion.
Samples not confidently assigned to the cross region return identity.

The per-case physical margins remain correction-generation diagnostics. They
are reported as `case_physical_signed_margin_by_term` and are not treated as
the fixed Guard result. After every candidate is generated, V15.15g1c places
that case back into its complete frozen transaction and runs the same exact
Guard evaluator used by the Oracle and Projector path. For every Guard term it
records the following shadow calculation and all of its operands:

```text
group_guard_value
- (fixed_anchor + max(abs(fixed_anchor) * relative_tolerance,
                      absolute_tolerance))
- numeric_tolerance
```

These values are reported under
`full_transaction_fixed_guard_shadow_margin_by_term`. A consistency check
requires the reconstructed absolute limit and pass/fail result to match the
authoritative fixed Guard audit.

If the Adapter passes the exact raw closure in the complete transaction, it is
locked as the incumbent and no Euclidean or Riemannian correction can replace
it. Otherwise, a correction is eligible only when the full fixed-Guard shadow,
endpoint/temporal descent, exact radius, observable displacement, numerical
checks, and zero scope leakage all pass. If no correction qualifies, the
selection returns identity. The selected result then runs through the existing
Projector and exact closure audit before the probe can report support.

Run the development probe on the server with:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1c_fixed_guard_shadow_server.sh
```

The script starts no Adapter training, pseudo-teacher recycling, promotion,
replay, or video generation.
