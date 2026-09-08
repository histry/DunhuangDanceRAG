# Three isolated baseline runners

Implementation only: these new runners have not been executed or validated.
No checkpoint was trained, no benchmark result was produced, and no production
pipeline or gate was changed. Invoke modules from the repository root.

All NPY/config/checkpoint entries use `{ "path": "relative/or/absolute/path",
"sha256": "actual file SHA256" }`. Relative paths resolve against the manifest,
not the shell working directory. Output directories must not already exist.
Never substitute a missing learned model with an analytic fallback.

## Paper 1: fixed boundary windows

Future invocation (not run during implementation):

```bash
python -m baselines.paper1_boundary --manifest paper1.json --output-dir outputs/paper1_new
```

Manifest schema `paper1_boundary_benchmark_v1`:

```json
{
  "schema": "paper1_boundary_benchmark_v1",
  "config": {"path": "config.json", "sha256": "REPLACE_WITH_HASH"},
  "seed": 42,
  "cases": [{
    "case_id": "fixed_pair_001",
    "reference": {"path": "observed_reference.npy", "sha256": "REPLACE_WITH_HASH"},
    "seam": {"path": "seam.npy", "sha256": "REPLACE_WITH_HASH"},
    "eligible": {"path": "eligible.npy", "sha256": "REPLACE_WITH_HASH"},
    "edit_span": [20, 48]
  }]
}
```

Reference shape is `[T,151]`, column-concatenated rot6d, contact channels 0:4,
root 4:7, rotations 7:151. The half-open edit span needs observed frames on both
sides. Do not pass hidden clean motion as reference. A reference may already be
physically invalid; its failures remain visible.

Built-ins: `reference`, `linear_rot6d` (intentionally unprojected naive baseline),
`slerp`, `so3_bridge`, `so3_bridge_ik`, `refiner`, `diffusion`.
Learned methods require a case `condition` NPY entry and
`checkpoints.refiner` / `checkpoints.diffusion` hashed entries. They run the
existing guarded inference functions, so label them as guarded baselines.
IK requires the frozen `eligible` array and receives the same ownership window.
Each method restarts from the identical reference and seed. No method receives
another method's repaired result.

`context_transformer` is a concrete repository baseline architecture with frozen
inference, not a reproduction of a named external paper. Supply the case
`recording_uid` and hashed `checkpoints.context_transformer`. Checkpoint schema
is `paper1_frozen_context_transformer_v1`, with `architecture` (hidden_dim,
heads, layers), `state_dict` matching ContextTransformer, and
`training_recording_uids`. Heldout recording overlap is rejected. This baseline
predicts geometry residuals from an analytic bridge plus observed context,
mask and normalized time. It projects rotations before the independent audit.
Weights do not exist merely because this architecture has been implemented.

For external Transformer, published inbetweening or complete-method outputs,
provide `cases[].frozen_outputs.METHOD` with the hashed motion entry plus
`checkpoint` (hashed entry) and `reference_sha256`. Request that METHOD through
`--methods`. This is a frozen-output adapter, not a reimplementation/trained
reproduction of the external paper. Missing outputs fail rather than skip cases.

Every candidate receives the same existing observable, physical, fixed-support,
fidelity and absolute physical audits; observable gain configuration must remain
0.03. No projection is applied after audit. Report both per-method failures and
passes; the reference baseline need not pass. Complete diagnostics are gzipped
JSONL, while report.json contains compact case rows and saved motion hashes.
Return code 2 means at least one audited candidate failed; this is expected for
weak baselines, not evidence that the experiment failed to execute.

## Paper 2: no-label offline grounding

```bash
python -m baselines.paper2_grounding --manifest paper2.json --output-dir outputs/paper2_new
```

Schema `paper2_grounding_benchmark_v1` requires hashed `music_targets` `[Q,12]`
and `motion_descriptors` `[E,12]`, aligned with `query_uids`, `query_song_uids`,
`event_uids`, `event_source_uids`. Use existing `observable_music_target` and
the event database's observable descriptor convention. Declare
`training_song_uids` (empty for fully analytic methods), `temperature` and
`top_k`. OT is solved per song, never by pooling independent evaluation songs.

Built-ins: weighted observable cosine, control-distance softmax, dense uniform
Sinkhorn, sparse uniform OT, sparse source-balanced OT. The cosine baseline uses
the same zero body-region weights as the OT control cost. No action categories
or fictional paired labels are introduced.

`contrastive_encoder` accepts `checkpoints.contrastive_encoder` as a hashed
checkpoint. Its tensor dictionary must contain:

- schema `paper2_frozen_dual_encoder_v1`;
- `training_objective: "contrastive"`, `training_song_uids`;
- `architecture`: music_dim, motion_dim, hidden_dim, embedding_dim;
- `state_dict` matching `baselines.frozen_grounding_model.DualEncoder`.

This provides frozen inference only. Producing contrastive weights requires a
separately justified positive-pair/augmentation protocol and future training.
No unpaired random assignment is treated as a positive pair.

Other frozen baselines, including the existing mean-pool or temporal Router,
can supply `frozen_scores.METHOD` with a hashed `[Q,E]` score NPY, a hashed
`checkpoint`, matching query/event UID lists, and `training_song_uids`.
The runner applies the same temperature to these logits/similarities.

Default metrics: entropy, source mass/HHI, top-K coverage, top-1 diversity and
expected observable cost **proxy**. The latter is not independent evidence for
OT quality. Retrieval metrics are null unless a hashed `relevance` `[Q,E]`
entry declares origin `human_evaluation` or `heldout_paired_dataset`. Teacher
probabilities are forbidden as evaluation truth. NDCG uses supplied nonnegative
relevance directly as gain; Recall@K uses positive relevance as membership.
Report raw transport convergence and column error after sparsification separately.
Music beat extraction and dynamic interpolation are outside this runner.

## Paper 3: frozen graph routing

```bash
python -m baselines.export_routing_fixture --schedule schedule.json --db events.npz --output-dir outputs/route_fixture_new
python -m baselines.paper3_routing --manifest outputs/route_fixture_new/manifest.json --output-dir outputs/paper3_new
```

The exporter reuses current Graph-SB layer preparation and geometry, saving
unaries, edge costs, masks and hashes once for every method. Schema
`paper3_routing_benchmark_v1` may also be supplied directly: `event_uids` is a
list of layers; `probabilities` lists hashed vectors; `edge_costs` and
`feasible_masks` list hashed adjacent-layer matrices. Mask entries are 0/1.
Zero unary probability excludes a candidate for every method.

Methods: independent greedy, finite beam, Viterbi, equivalent DAG shortest
path, entropy-regularized chain, existing multi-marginal Graph-SB. Same candidates,
unary probabilities, edge costs and forbidden edges are used throughout.
Greedy paths that violate masks are reported INFEASIBLE and never repaired.
Graph-SB nonconvergence is an explicit failure, with no greedy/beam fallback.

Viterbi and shortest path are the same objective/algorithmic recurrence here;
do not count them as independent scientific comparisons. Entropy-chain and
Viterbi can have identical MAP paths, despite different distributional outputs.
The existing Graph-SB solver reports Fisher-Rao residual; this runner does not
invent a separate 'no Fisher-Rao' algorithm by renaming the same IPF solver.

This isolates the layered-graph optimization. Nonlocal cooldown/source quotas
must be encoded in expanded states/masks before claiming those constraints are
controlled. Results are route-only; they do not certify final video gates.
The production generation entrypoint is deliberately not modified.

## Next execution gate

First review these new interfaces and provide frozen fixtures/checkpoints.
Then, when authorized, validate the new code and run small baseline comparisons
before large datasets. No tests, lint, compile, inference, training or benchmark
execution were performed while writing this implementation.
