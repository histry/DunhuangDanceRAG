# Whole-song physical restoration V12

This change separates three concepts that were previously conflated:

1. a frame is observable as physically risky;
2. a frame is editable by a repair stage;
3. a candidate is safe to commit.

The final physical Guard, KBO, protected-frame contract and atomic rollback
remain authoritative and unchanged.

## Repair support

The Refiner/Diffusion support is now the union of seam, world-FK peak jerk,
observable planted-foot skate/drift and SO(3) rotation-step risk.  Adding a
frame to this mask never accepts an edit; it only allows the frozen repair
model to propose one.

## Multi-objective local transactions

The full-sequence IK transaction selects objectives from the positive physical
residuals in the current ownership window.  Contact, jerk and rotation rows can
therefore become repair objectives.  All other physical rows remain strict
non-regression constraints, followed by fixed-support, boundary/fidelity,
scope, KBO and full-sequence audits.

This fixes the V11 false-infeasibility mode in which a jerk/rotation window
containing a static support frame was required to improve a contact residual
that was already zero.

## Long-horizon root drift

Root drift is owned by a separate full-sequence transaction because a short
window recentres its local root median and cannot faithfully observe the
whole-song drift metric.  The candidate uniformly scales root-XZ displacement
about the first-frame anchor.  It is committed only if the normal full physical
stage audit accepts it; otherwise the exact input snapshot is restored.

## Required server evidence

- V12 targeted tests pass on the 4090 environment.
- `root_drift_restoration_transaction` either commits with full audit evidence
  or rolls back without changing the selected hash.
- each IK transaction reports `objective_metric_keys` and exact-audit results;
  `local_infeasible_under_current_action_basis` is only emitted after the
  actual selected objective basis has no accepted direction.
- the final whole-song physical, boundary-continuity and activity gates pass
  before an MP4 is accepted as a formal output.
