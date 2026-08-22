# Fisher-Rao Graph-SB formal route

Graph-SB consumes only Scheduler-issued CTSR sibling candidates. Its unary
measure is the Router candidate probability; Event quality and anatomy remain
physical feasibility evidence, not an alternative semantic model.

Formal invariants:

- `GRAPH_ROUTE_SOLVER=fisher_rao_graph_sb`;
- every slot declares `router_architecture=ctsr_weak_temporal_v1`;
- every slot carries non-empty, aligned candidate UIDs and probabilities;
- immutable node and pairwise geometry gates are applied before IPF;
- IPF non-convergence raises an error;
- the accepted report has `solver=fisher_rao_graph_sb` and
  `fallback_used=false`.

Validate a final report with:

```bash
$PY evaluation/validate_formal_route.py \
  --report "$FINAL_REPORT" \
  --out "$OUT_ROOT/final.graph_sb.acceptance.json"
```

Current-protocol greedy and finite-beam comparisons are documented in
`docs/BASELINES.md`; they share the same SMPL14 data, split, candidates, and
evaluation code.
