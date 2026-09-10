# Refiner V15.12e subgroup-aware gradient probe

V15.12d removed the aggregate clean-fidelity loss from the hard guard, but its
50-step server probe still accepted only 18 updates and passed 0/92 fitted
contexts.  The two-objective endpoint/temporal PCGrad hid conflicts between
single/cross and short/long subgroups.

V15.12e separates the scientific objective into eight gradients:

- `single_short.endpoint` and `single_short.temporal`
- `single_long.endpoint` and `single_long.temporal`
- `cross_short.endpoint` and `cross_short.temporal`
- `cross_long.endpoint` and `cross_long.temporal`

Each gradient is normalized by its parameter RMS.  A deterministic
Frank-Wolfe solve finds the minimum-norm convex combination on the eight-task
simplex.  The resulting direction is admitted only when every true subgroup
directional derivative is non-positive and at least one is strictly negative.
A near-zero minimum norm records
`pareto_stationary_or_no_common_descent` and skips backtracking.

The non-scientific gradient remainder is scaled only as far as the same eight
directional constraints permit.  Candidate acceptance continues to use the
existing deterministic closure and immutable physical, fixed-support,
fidelity, boundary and observable `0.03` gates.  No production threshold is
changed.

Detailed step evidence is written to `gradients.jsonl` and
`optimizer_updates.jsonl`.  It includes the 8x8 cosine matrix, pre-normalized
RMS norms, MGDA weights and minimum norm, all directional derivatives, every
candidate's real guard residual deltas, blocking metric names and smallest
audited/accepted scales.

Run only `scripts/run_refiner_v15_12e_probe_server.sh` first.  It executes a
50-step development diagnostic and cannot start pilot training, checkpoint
promotion or video generation.
