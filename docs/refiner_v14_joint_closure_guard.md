# Refiner V14: same-window closure and physical row Guard

V13 completed 8,000 steps without numeric failure but was rejected by the
source-disjoint checkpoint gate.  Its final validation closed endpoint,
temporal and physical non-regression on the same candidate in only 3/16
ordinary windows and 2/8 cross-event windows.  The dominant failures were
temporal repair and overlapping jerk/penetration/support-drift regressions.

V14 is a minimal contract correction.  It does not change the network,
sampling distribution, endpoint/temporal gains, temporal weight, physical
limits, final Guard, IK, diffusion or generation thresholds.

## Changes

1. Validation reports explicit same-window closure counts and rates:
   `endpoint_accepted && temporal_accepted && physical_non_regression_accepted`.
   This is separate from the historical hidden-clean `stage_repair` metric.
2. Checkpoint candidates are ranked lexicographically by scientific
   acceptance, the weakest ordinary/cross joint-closure rate, the weakest of
   the six marginal closure rates, their sum, and reference fidelity.
3. The transactional training Guard independently protects the six physical
   rows observed to block V13: joint jerk max, joint jerk window P95,
   extremity jerk P95, extremity jerk window P95, foot penetration and foot
   support-drift P95.  It consumes existing differentiable signed residuals;
   the independent NumPy stage audit remains authoritative.

Physical failure reasons can overlap.  Counts by reason must not be summed and
interpreted as a partition of failed windows.

## Pilot protocol

Run a fresh 2,000-step V14 pilot with deterministic validation every 500
steps.  The 500/1,500 reports describe the V14 trajectory; strict matched V13
comparisons use the retained V13 1,000/2,000 reports because no historical
500/1,500 checkpoints were preserved.

The pilot does not publish or promote a checkpoint.  A full run is considered
only when optimizer accounting is complete, all numeric trials are finite,
ordinary and cross-event joint closure improve at matched steps, jerk failure
windows do not increase, cross-event temporal repair trends upward, and
endpoint repair does not systematically regress.  The final two V14
validation points should not show a decline in the weaker joint-closure rate;
because cross-event validation contains eight windows, this stability rule is
supporting evidence rather than an isolated veto.
