# V15.15g1f exact-radius geodesic joint active-set SQP

V15.15g1f replaces the g1e add/restore/renormalize sequence with a single
geodesic update on the owned tangent sphere. It is a bounded correction probe;
it does not train the Adapter, recycle pseudo-teachers, promote checkpoints, or
generate motion or video.

## Frozen protocol

Train calibration freezes the discriminative transaction-conformal envelope,
the full-transaction fixed-Guard shadow contract, the 2/3/5 correction budgets,
and a 12-angle search ladder. The maximum angle is `pi/4`; each following angle
is half the previous angle. The joint active-set solver uses a relative SVD
cutoff of `1e-6`, a direction norm floor of `1e-8`, and a strict directional
margin of `1e-6`.

Every gradient is first multiplied by the exact boolean ownership mask and
projected onto the tangent space orthogonal to the current direction. The
shadow, endpoint, and temporal rows are then solved jointly by active-set
enumeration with a truncated-SVD pseudo-inverse. The update is

```text
d' = cos(theta) d + sin(theta) ||d|| q / ||q||
```

so the radius is preserved by construction. No post-update normalization or
science restoration step is allowed.

## Fail-closed acceptance

A trial is accepted only after the real retraction when all of these hold:

- the authoritative hard full-transaction fixed-Guard shadow strictly drops;
- endpoint and temporal both strictly improve;
- the owned tangent RMS remains exactly `1e-4` within numeric tolerance;
- the value outside ownership is exactly `0.0`.

The report separates `nonfinite_joint_jacobian`,
`rank_deficient_joint_jacobian`, and `zero_science_gradient`. A missing joint
direction or an exhausted angular ladder is reported as
`empty_geodesic_joint_feasible_intersection`; it is never teacher-eligible.

## Evidence roles

All parameters are frozen from train transactions. Case 53 remains reused
development evidence. The development runner stops if train calibration fails
and never launches the final held-out evaluation. A separately sealed,
previously untouched cross-long transaction bank may be consumed exactly once
by the final runner after the train and development gates pass.

## Server entry points

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1f_exact_radius_geodesic_joint_sqp_server.sh
```

Only after that run succeeds should the one-shot final command be used:

```bash
EXPECTED_COMMIT=<same-full-main-sha> \
FINAL_HELD_OUT_BANK=<new-sealed-validation-bank.pt> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1f_final_held_out_server.sh
```
