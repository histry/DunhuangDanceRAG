# Refiner V15.7 confidence-preconditioned observable learning

V15.6 completed 400/400 accepted optimizer updates and materially improved
cross-event endpoint readiness, but the exact diagnostic still failed.  The
remaining deficit was concentrated in low-confidence single-recording cases
and in temporal repair, especially for 28-frame windows.  Their decoded edits
and parameter gradients were attenuated by root and joint soft confidence
masks even though the four role/width groups had equal objective weight.

V15.7 changes only aggregation of the development Refiner scientific loss:

- Each case receives a detached inverse-confidence weight computed from its
  seam-average root and joint confidence.
- The inverse weight is capped at 5 and normalized to mean 1 inside the full
  optimizer transaction.
- Endpoint and temporal deficits are preconditioned separately before the
  existing group-balanced smooth CVaR aggregation.
- Exact group guards continue to use the raw, unweighted scientific deficits.

This is a scalar forward objective evaluated identically by the Armijo line
search.  Autograd still follows the true decoder and its soft masks; no custom
backward derivative or binary support mask is introduced.  The decoder,
physical and fidelity constraints, clean identity branch, and exact 0.03
endpoint/temporal gate are unchanged.

The protocol, checkpoint fingerprint, and diagnostic schema are incremented.
Old V15.6 snapshots cannot be resumed as V15.7 evidence.  The implementation
remains unverified until a fresh 400-step diagnostic reports exact per-group
endpoint, temporal, jerk, physical, fidelity, and clean decisions.  Pilot and
full generation remain forbidden while readiness is false.
