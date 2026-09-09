# Refiner V15.12d: gate-metric trust region

V15.12c proved that symmetric PCGrad can construct endpoint/temporal common
descent directions, but its hard guard constrained the aggregate clean
identity objective to `1e-6`. That objective includes a no-op regularizer for
every nonzero edit, so the guard converted a soft preference into a near-zero
output constraint. At step 200, all 92 fitted contexts failed and the applied
tangent RMS remained around `1e-6`.

V15.12d keeps the no-op term in the training loss but removes it from hard
candidate rejection. The fixed, nonaccumulating guard now audits maximum clean
product-log error against
`checkpoint_validation_max_clean_identity_product_log_l1` and maximum clean
contact error against
`checkpoint_validation_max_clean_identity_contact_l1`. Clean temporal/support
and repair physical quantities remain zero-excess guards with only the existing
configured numerical tolerance. Endpoint and temporal components remain
separate fixed-anchor constraints; total and joint feasibility remain
best-so-far constraints.

The previous hand-written single/cross allowances are removed. No physical,
fixed-support, fidelity, boundary, observable `0.03`, or final acceptance
threshold changes.

Run only `scripts/run_refiner_v15_12d_probe_server.sh` first. It performs a
50-step server probe and records execution status separately from diagnostic
readiness. It never starts a pilot, formal training, promotion, or generation.
