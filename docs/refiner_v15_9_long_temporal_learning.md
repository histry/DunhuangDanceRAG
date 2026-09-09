# V15.9: long-window temporal learning repair

V15.8 completed every optimizer transaction safely and restored cross-event
temporal passes, but it did not pass readiness. Both seen and new-position
width-28 groups remained at zero temporal passes. At the final step the logged
endpoint gradient norm was about three times the temporal gradient norm, and a
joint subgroup guard still allowed endpoint and temporal slack to trade.

V15.9 adds four continuous features computed only from the supplied descriptor
path and edit support: local condition velocity, local condition acceleration,
whole-support path length and endpoint condition gap. These expose the
procedural single-recording versus cross-event distinction and long-transition
extent without semantic labels, categorical width heads, hidden clean frames or
target corrections.

The network objective applies a fixed factor of 3 to the unchanged temporal
scientific deficit. This factor comes from the V15.8 final-step component
gradient ratio; the endpoint and temporal pass thresholds remain 0.03. Checked
transactions now guard endpoint and temporal tail risks separately in every
role/width subgroup, in addition to total repair and joint feasibility.

Physical, fixed-support, fidelity, boundary and final gates are unchanged. Old
snapshots are incompatible. This is an unvalidated implementation until a fresh
server foundation and 400-step diagnostic pass all eight subgroups.
