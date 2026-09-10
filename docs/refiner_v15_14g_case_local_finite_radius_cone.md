# V15.14g case-local finite-radius sequential inequality cone

V15.14g is a development-only fixed-bank probe. It never trains, promotes,
replays, generates motion, or produces video.

The probe moves the action basis from four group-wide edits to independent
`[case, frame, 79]` edits. Every action is exactly zero for all other cases,
groups and frames outside the ownership window. The existing C2 seam taper is
applied before any direction is normalized. Its inward quintic distance uses
the same one-based edge level as the decoder, leaving a small nonzero weight on
the first and last owned frames so endpoint derivatives remain observable.
Contact action channels remain zero and only the 75-dimensional
product-manifold tangent is retracted.

For every case the probe constructs independent endpoint and temporal descent
directions. Hard correction atoms come from concrete Top-K and epsilon-active
witnesses: joint/frame jerk peaks, left and right seam jerk, foot/frame
penetration, static-support foot skate and static-support anchor drift. New
witnesses discovered after a finite-radius trial are added as cuts, preventing
the active maximum from moving silently to an adjacent frame or joint.

The `1e-6` one-sided finite difference initializes the constraint Jacobian and
is diagnostic only. A candidate is always rematerialized at an actual output
tangent RMS of at least `1e-4`, measured only over that case's owned seam rather
than diluted across the fixed bank. Up to six sequential iterations audit that
real candidate, add violated witness cuts, refresh the one-sided Jacobian around
the current finite-radius point, and solve for signed hard-correction increments.
Smaller radii cannot be accepted as learning evidence.

Jerk witness cuts may consume only the remaining margin beneath the unchanged
fixed Guard limit. Penetration, support/fixed-support and boundary cuts remain
hard. Endpoint and temporal are independently nonregressing and at least one
must decrease beyond audit resolution. Every trial still receives the complete
fixed physical, fixed-support, fidelity, boundary and observable `0.03` audit.
Only a passing raw group composition may enter the unchanged scientific
projector and its `1`, `1/2`, `1/4`, `1/8`, `1/16` correction backtracking.

Every refreshed witness Jacobian reports singular values, effective rank and
condition number. Iterations report linear predicted deltas, exact deltas after
baseline rematerialization and their curvature ratios. The final report also
distinguishes a true group-composition failure from a case improvement hidden
by the group `max/p95` aggregation. It reports unique passing cases separately
from passing endpoint/temporal seed trials.

If no case survives at the required radius, the scientific status is
`no_case_local_exact_closure_descent_at_required_radius`. A routing pivot is
supported only when cross-short and cross-long both produce effective projected
candidates, scope leakage is exactly zero and all numeric audits complete.
