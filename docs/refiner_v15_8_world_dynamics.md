# V15.8: world FK conditioning and objective-consistent diagnostics

The submitted V15.7 server run completed 400 steps but failed readiness.
Cross-event temporal passes fell from 3/16 and 4/16 in V15.6 to 0/16 in both
seen and new-position banks. Confidence reweighting is therefore withdrawn;
its helper remains only for historical reproduction. This comparison does not
establish confidence attenuation as the cause of the learning failure.

The conditioning path had removed horizontal root motion before FK and never
added its derivatives back. The objective measures world-space derivatives.
V15.8 computes body FK in float64, then adds root first/second/third differences
to the corresponding body differences. Constant translation disappears while
actual root motion remains. Features are cast to network dtype only after
differencing. Existing support masks, feature units and feature count stay the
same. This fixes an input/metric mismatch; it does not prove final readiness.

Endpoint and temporal loss return to equal case weights with separate smooth
CVaR aggregation. Component-gradient logging now uses those actual aggregate
objectives rather than raw case means. Raw deficits remain available for the
existing subgroup guards and reporting.

No hidden clean target or foundation probe solution is used as supervision.
The decoder, 0.03 endpoint/temporal thresholds, physical/fixed-support/fidelity
audits and final gates are unchanged. Input and objective protocols invalidate
old snapshots. A fresh foundation and 400-step diagnostic are required.

Local validation is deliberately not executed at the user's request. Server
tests must pass before fitting. Inspect all eight role/width groups and the
final diagnostic_ready flag; an output checkpoint alone is not acceptance.
