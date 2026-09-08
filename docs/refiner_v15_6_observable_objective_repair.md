# Refiner V15.6 observable objective repair

V15.5.1 completed 400/400 accepted optimizer updates, but its final exact
validation still reported zero temporal repair in every role/width group and
very low endpoint repair.  The optimizer and decoder were numerically active;
the remaining failure was a mismatch between the training objective and the
exact endpoint/temporal acceptance contract.

V15.6 changes only the development Refiner fitting objective:

- The scientific endpoint and temporal deficits now use the same `1e-6`
  uninformative-baseline branch as the exact observable auditor.
- A training-only gain buffer of `0.005` targets 3.5% improvement while the
  authoritative validation threshold remains exactly 3%.
- A linear gradient floor is added to the one-sided Huber deficit so the
  repair gradient does not vanish immediately below the buffered target.
- Endpoint and temporal tail risks are computed independently in every
  role/width group and then summed.  Difficult endpoint cases therefore cannot
  be hidden by the temporal component, or vice versa.
- Minimum-edit regularization is inactive until both buffered observable
  targets are met.  It remains a feasible-set selection preference rather
  than opposing the repair needed to enter that set.

The product-manifold decoder, soft risk masks, true chain-rule gradient,
physical and fidelity constraints, clean identity branch, and exact 0.03
validation gate are unchanged.  The protocol and diagnostic schema were
incremented so V15.5.1 snapshots and reports cannot be mistaken for V15.6
evidence.

This code change is not evidence that the scientific gate now passes.  A fresh
foundation run and 400-step bridge diagnostic must produce the exact per-group
endpoint, temporal, jerk, physical, and clean decisions before any pilot or
full generation is allowed.
