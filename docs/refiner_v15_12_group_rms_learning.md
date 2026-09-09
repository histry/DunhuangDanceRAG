# V15.12: unresolved-group RMS learning

V15.11 passed 115 server tests and completed all 400 diagnostic steps, but it
did not pass scientific readiness. The joint feasibility deadband removed the
duplicated subresolution blocker, yet seen endpoint repair fell to 2/16,
new-position endpoint repair reached only 1/16, and both 28-frame temporal
subgroups remained at 0/8. The direct foundation report passed all eight
role/width subgroups, so the current evidence supports an optimization and
function-fitting failure rather than infeasibility of the exact gates.

The network objective previously used an arithmetic mean of the four
single/cross by short/long group tail risks. This continued to allocate equal
objective weight to an already near-resolved short cross-event group while the
single-recording and long groups remained much farther from acceptance.

V15.12 replaces only this across-group arithmetic mean with a symmetric
root-mean-square. RMS equals the original value when all group risks are equal
and is invariant to group ordering, but its derivative gives proportionally
more weight to a larger unresolved risk. It adds no semantic label, threshold,
temperature or held-out input. Within-group mean/CVaR, the fixed temporal
weight, full-cycle training bank and per-group transaction guards are retained.

Endpoint and temporal acceptance thresholds remain exactly `0.03`. Physical,
fixed-support, fidelity, boundary and final gates are unchanged. The server
runner performs fresh foundation and diagnostic execution only; it cannot
launch pilot training or checkpoint promotion.
