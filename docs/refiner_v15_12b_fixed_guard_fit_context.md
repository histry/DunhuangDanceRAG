# V15.12b: fixed-anchor guard and independent fit-context evaluation

V15.12 completed 400 accepted diagnostic updates but failed readiness. All
temporal subgroups finished at 0/8. Seen and new-position cross-event endpoint
rates both fell to 0.50, despite every accepted step satisfying the previous
per-step subgroup tolerance. The evidence rejects RMS aggregation as a
sufficient repair and exposes cumulative forgetting through the rolling guard.

V15.12b keeps the V15.12 network, RMS group aggregation, 400-step schedule,
training banks and exact scientific objectives unchanged. It changes only the
development optimizer acceptance and diagnostic reporting:

- Armijo loss is still evaluated on the same rotating 192-case C5 TRAIN
  transaction used for the gradient.
- Subgroup protection is evaluated on one immutable complete `seen` TRAIN
  anchor containing all four role/width groups.
- Each protected anchor metric is compared with its componentwise best-so-far
  value. The 0.5% relative allowance is always measured from that persistent
  value, never from the previous accepted step, so it cannot accumulate.
- Every materialized `fit_context_*` TRAIN cut is evaluated independently at
  diagnostic checkpoints. The report distinguishes
  `fit_context_learning_failed` from
  `fit_context_passed_probe_generalization_failed`.
- Held-out `new_position` tensors remain update-forbidden.

The diagnostic remains fail-closed. All fit contexts and both held-out probe
splits must pass their role/width decisions before readiness can be true.
Endpoint and temporal acceptance remain exactly 0.03. Physical,
fixed-support, fidelity, boundary and final gates are unchanged.

Relative Seam Phase is deliberately absent from V15.12b. It becomes the single
new architectural variable in V15.13-P only if all fitted contexts pass while
the held-out temporal-cut probe still fails.

`scripts/run_refiner_v15_12b_server.sh` performs server tests, Ruff, a fresh
foundation control and the 400-step diagnostic. It cannot launch a pilot,
resume formal training, generate video, or promote a checkpoint.
