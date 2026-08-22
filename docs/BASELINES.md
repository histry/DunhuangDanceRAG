# Current-protocol baselines

Historical 72/BVH experiments are not baselines for the formal SMPL14 result.
They use different source files, semantics, splits, and music-routing contracts
and are preserved only by the Git tag
`archive/legacy-motion-pipeline-e8217222`.

The supported comparison set shares the formal Event-DB, source split,
Librosa 12D phrase features, Semantic-OT weak teacher, CTSR candidate sets, and
evaluation code:

- `ctsr_mean_pool_mlp_baseline_v1`: removes temporal ordering while retaining
  the same weak-teacher dataset and song-disjoint validation policy;
- `ctsr_independent_greedy_v1`: selects each slot independently from the same
  CTSR sibling set;
- `ctsr_finite_beam_v1`: uses the same candidates and event geometry but no
  Fisher--Rao marginals or Schrödinger bridge.

These implementations are `IMPLEMENTED`. They become `EXECUTED` only after
checkpoints and route reports from the current clean Git revision exist, and
`VERIFIED` only after metrics are reconciled with the paper tables.

The formal pipeline trains the non-temporal Router immediately after the CTSR
Router and evaluates both route baselines immediately after schedule creation.
The outputs are:

- `$OUT_ROOT/baselines/ctsr_mean_pool_mlp.pt`
- `$OUT_ROOT/baselines/ctsr_mean_pool_mlp.history.json`
- `$OUT_ROOT/baselines/current_protocol_routes.json`

The equivalent standalone training command is:

```bash
$PY training/current_protocol_router_baseline.py \
  --data "$OUT_ROOT/scheduler_training/router_training.npz" \
  --index_json "$OUT_ROOT/scheduler_generation_assets/event_index.json" \
  --index_npz "$OUT_ROOT/scheduler_generation_assets/duration_index.npz" \
  --out "$OUT_ROOT/baselines/ctsr_mean_pool_mlp.pt" \
  --fps "$GENERATION_FPS"
```

Evaluate route-only greedy and beam baselines on a freshly generated formal
schedule:

```bash
$PY scripts/evaluate_current_protocol_baselines.py \
  --schedule "$FRESH_MSSD" \
  --db "$GENERATION_DB" \
  --out "$OUT_ROOT/baselines/current_protocol_routes.json"
```
