# Refiner V14.1: normalized activation-aware physical Guard

The retained V14 strict-row pilot is a negative baseline. It completed 1,200
steps, with preserved 500/1,000 validation reports and a 1,200-step snapshot.
Endpoint, temporal and joint closure remained zero at both validation points.
The snapshot is not a formal checkpoint and must not be resumed.

V14.1 changes only the optimizer's differentiable training Guard:

1. Each guarded physical row uses a dimensionless signed residual, normalized
   by the existing stage margin. Zero remains the unchanged physical threshold.
2. Active or near-threshold rows use a fixed `1e-3` training-resolution
   deadband. The near-threshold band is `5e-2` in normalized units.
3. Rows farther inside the safe region may vary up to the unchanged zero
   boundary and therefore do not force backtracking on harmless movement.
4. Any previously safe row that becomes positive is rejected, including a
   violation smaller than the training deadband.

The V14.1a execution path reduces wasted work without changing acceptance:

- A rejected trial uses its measured Guard headroom and residual change to
  propose the next scale near the strict boundary. The next full closure still
  decides acceptance. This is recorded as optimizer protocol
  `exact_guard_constrained_fixed_anchor_armijo_v10`.
- Formal Refiner training permits four trials per direction, at most eight
  full trial closures per step.
- Reference and clean FK/boundary quantities that are fixed within one
  optimizer transaction are computed once and reused by its trial closures.
- Lightweight progress prints every 20 steps. Full component-gradient
  diagnostics remain every 200 steps and no longer distort the first-step ETA.
- The pilot fails closed with exit status 2 after 50 consecutive rolled-back
  updates. It saves a snapshot and validation report for diagnosis and must not
  be resumed after this early stop.

The independent validation audit, checkpoint thresholds, authoritative
whole-sequence physical Guard, model architecture, data split and generation
thresholds are unchanged.

Run `scripts/run_refiner_v14_1_pilot_server.sh` from a clean checkout at the
reviewed commit. The script creates a new output directory, accepts no resume
snapshot, stops at 2,000 steps, and never publishes or starts generation.
