# Refiner V15.15g: fixed-budget correction ablation

V15.15g evaluates whether a frozen V15.15f Adapter can serve as a warm start
for a very small constrained correction budget. It runs on the frozen
validation split and compares the following candidates from the same Adapter
checkpoint, cases and exact `target_rms=1e-4` contract:

1. uncorrected Adapter output;
2. Euclidean tangent correction with exact ownership masking, projection back
   to the Euclidean RMS sphere, and product-manifold Retraction after every
   accepted step;
3. moving-tangent product-manifold correction with Retraction at the current
   point, pullback to the immutable Anchor, and projection back to the same
   fixed-radius sphere.

Both correction branches use budgets 2, 3 and 5, identical differentiable
endpoint, temporal and fixed-Guard residuals, identical step bounds, and the
same final raw/Projector/Guard audit. This prevents the Euclidean baseline from
becoming a strawman that leaves the sphere or skips Retraction.

The Adapter is a learner-provided warm start. All correction results are
stop-gradient in this stage. Validation candidates are never recycled into a
teacher bank. Guard-rejected variants are recorded only as non-training hard
negative metadata. A later train-only online-teacher experiment may use the
defined contrastive penalty
`relu(cos(predicted, rejected) - margin)`, but rejected directions can never
be direction teachers.

If end-to-end differentiation is later justified, the declared path is an
implicit derivative of the converged KKT system and a linear adjoint solve.
Unrolling solver iterations is explicitly outside the V15.15g contract.
Because max/p95, contact switching and penetration witnesses remain
non-smooth, the complete fixed Guard remains the final acceptance authority.

Server execution, after V15.15f passes, uses:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g_fixed_budget_correction_server.sh
```

V15.15g is an offline ablation. It starts no Adapter training, online teacher
recycling, promotion, replay or generation.
