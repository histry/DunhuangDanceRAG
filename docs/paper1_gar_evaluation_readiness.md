# Paper 1 / GAR Evaluation-Readiness Contract

## Scope

This change adds a passive primitive-data interface for a future Geometry-Aware
Routing (GAR) evaluation. It does not run or implement the Paper 1 comparison.
The current retrieval order, candidate simulation, boundary decision, transition
builder, neural generation, post-generation audit, and reselection behavior remain
authoritative and unchanged.

The trace schema is `gar_selection_trace_v1`. It is implemented in
`evaluation/gar_evaluation_readiness.py` and is emitted only after all production
selection rounds and final quality gates have completed. The schema accepts
operator and generator metadata from the caller; it does not hard-code a bridge,
model, selector, threshold, or candidate ranking rule.

## Configuration and output

The existing `MotionGenerationConfig` has two metadata-only fields:

- `gar_evaluation_trace_enable: false` enables the sidecar writer when explicitly
  set to `true`. The default therefore preserves current runtime and output motion.
- `gar_evaluation_method_variant_id: "current_boundary_closed_loop"` labels the
  current real method in a paired dataset. It is never read by candidate selection.

When enabled, the formal closed-loop generator writes
`<requested-motion-stem>.gar_selection_trace.json` and records its path in the
normal generation report under `gar_evaluation_readiness.trace_path`. The normal
report always states the readiness-only status and whether sidecar recording was
enabled.

## Stable identity and provenance

Every trace records the runtime Git commit, behavior config fingerprint, complete
candidate-pool fingerprint, retrieval-index fingerprint, generator and repair
metadata, selection policy ID, method variant ID, and random seed. Each boundary
record repeats the complete sequence-level provenance tuple so it remains auditable
after tabular flattening; schema validation requires the two copies to agree.

- `candidate_id` is the existing portable `event_uid`; a Generation-DB row index
  is never treated as identity.
- `sequence_id` hashes the audio content when the file is available, otherwise its
  portable file name, plus slot timing and event count. Selected candidates are
  excluded so different methods retain the same sequence ID.
- `boundary_id` and `evaluation_case_id` are independently namespaced hashes of
  `sequence_id + slot_index`. They remain equal across methods evaluating the same
  boundary, even when a method selects a different candidate.
- Candidate ranks in the readiness trace are strictly 1-based and refer to the
  original ordered pool. Existing internal 0-based ranks remain untouched. This is
  significant after a failed candidate is banned, because production internally
  renumbers the remaining list.
- Every slot, including slot 0, has a `candidate_pool_manifest`. Boundary records
  reference the pool for the current slot.
- A pool fingerprint hashes only ordered pairs of `candidate_id` and 1-based
  `retrieval_rank`. Retrieval score, runtime, temporary paths, metadata, and JSON
  key order are excluded. The same ordered Top-K pool therefore has the same
  fingerprint under different selection methods.
- `retrieval_index_fingerprint` reuses the Generation Event-DB contract's
  `ordered_event_uid_sha256`.
- `config_fingerprint` hashes the resolved motion config and the active Boundary,
  Routing-Safety, Graph-SB, Grounding, Event-Heading, Routing-Budget, Generation,
  and Motion-Activity environment while excluding the two `gar_evaluation_*`
  trace controls. Run IDs, executable/checkpoint/index paths, and rebuild/retrain
  orchestration switches are also excluded because they are provenance or launch
  state rather than final decision behavior. Enabling logging or moving identical
  artifacts cannot create a different behavior fingerprint.

`selection_policy_id` is resolved from the policy that actually produced the final
assembly. The base formal path records its first-safe/minimum-risk plus post-audit
reselection policy. When the Geometry-Aware Routing patch stack is active, the
trace instead identifies the Fisher--Rao Graph-SB preorder plus viability-aware
boundary-reselection path. Thus a wrapper cannot be mislabeled as the base policy.

The current production integration reports
`repair_operator_id=so3_endpoint_velocity_bridge`, obtained at the actual
`make_geodesic_transition` bridge call site. Its fingerprint covers the active
transition configuration and boundary environment. The schema itself accepts any
future stable repair-operator ID.

The current generator is reported as `edge151_motion_generation_pipeline`, with a
version describing the Refiner/Diffusion/IK stack and a fingerprint of the active
stage configuration. Existing Refiner and Diffusion checkpoints are content-hashed
when their stages are active and the files exist. The checkpoint fingerprint stays
`null` when no applicable artifact is available; no path string is presented as a
checkpoint identity.

## Primitive trace fields

The formal CTSR candidate list and router probabilities supply the complete
candidate-pool manifest:

- `candidate_id`
- `retrieval_rank`
- `retrieval_score`
- `source_event_id`
- `source_recording_id`
- `candidate_metadata_fingerprint`

If a patched selector evaluates an event outside the declared shared pool, trace
construction fails closed instead of silently expanding the manifest after seeing
the method's decision. A future paired runner must declare the complete common pool
before selection; otherwise it cannot claim a same-pool comparison.

Each actually simulated candidate attempt records:

- stable candidate ID and original 1-based pool rank
- evaluation order and reselection attempt index
- independent `pre_risk` and `pre_risk_components`
- `pre_safe`
- whether the candidate received full downstream generation
- independent `post_risk`, `post_risk_components`, and `post_safe` when generated
- authoritative post-audit failure reason codes
- initial/final selection and post-audit rejection flags
- repair and generator IDs

`pre_risk` comes from the current simulated bridge `risk_score_predicted`; its
components are the already computed transition-risk dictionary. `post_risk` comes
from the same score function applied by `audit_boundaries` after the full downstream
generator. Its components and failure reasons come from the authoritative final
boundary audit. A simulated but non-generated candidate has `post_risk=null` and
`post_safe=null`; no missing observation is serialized as zero or false.

The current safety contract has multiple independent limits in
`BoundaryContinuityLimits.from_environment`, not one authoritative scalar risk
threshold. The trace therefore records that source and the complete resolved limit
mapping, while `risk_threshold_value` and candidate-level `false_safe` remain
`null`. A future frozen protocol may derive `false_safe` only when pre-risk,
post-risk, and one authoritative scalar threshold all exist. This readiness change
does not create a Paper 1 threshold.

Each boundary summary records pool size, evaluated attempts, initial/final
candidate IDs and ranks, reselection count, initial/final post-risk and safety,
recovery status, hard-failure status, and authoritative failure reasons.
`recovered_after_reselection` is defined only as:

```text
initial_post_safe == false
and final_post_safe == true
and reselection_count > 0
```

It is `null` when either safety observation is unavailable. The current production
path does contain post-audit reselection: it bans the selected candidate at the
worst unsafe slot and reruns closed-loop assembly, generation, and audit within the
existing configured round limit. This code records those existing attempts and does
not add or enable reselection.

## Boundary motion metrics

`boundary_metrics` maps only existing authoritative values with matching semantics:

| Stable field | Current source | Unit / meaning |
|---|---|---|
| `fk_position_jump` | max of entry/exit FK jump | m |
| `so3_rotation_geodesic_jump` | max of entry/exit SO(3) rotation step | rad/frame |
| `velocity_jump` | max of entry/exit velocity jump | current transition-risk SI convention |
| `acceleration_jump` | max of entry/exit acceleration jump | current transition-risk SI convention |
| `foot_skate` | supported-foot slip p95 | m/s |
| `penetration_max` | maximum penetration depth | m |
| `contact_discontinuity` | contact-switch value | dimensionless |

The following fields currently remain `null`: `boundary_jerk_mean`,
`boundary_jerk_p95`, `penetration_mean`, `root_velocity_discontinuity`, and
`heading_discontinuity`. Production currently exposes boundary jerk maximum rather
than mean/p95, and its aggregate penetration value is mean squared depth rather
than mean depth. Those values are retained in raw risk components but are not
mislabelled in the standard metric fields.

## Runtime measurements

When tracing is enabled, sequence-level instrumentation measures real elapsed time
for retrieval, candidate simulation/assembly, downstream generation, post-audit,
reselection bookkeeping, and the complete sequence. Timing is read only after each
stage and never changes Top-K, stopping, ranking, or selection.

Current code has no isolated per-candidate or per-boundary timers. Their
`runtime_ms` and boundary runtime breakdown fields therefore remain `null` rather
than receiving an allocated or placeholder time. A future Paper 1 runner may add
those measurements without changing the schema.

## Offline Paper 1 metric mapping

The trace preserves primitive observations needed by a future, separate evaluator:

| Future analysis | Primitive fields |
|---|---|
| Unsafe Boundary Rate | `boundary_id`, `final_post_safe` |
| Failure Recovery Rate | `initial_post_safe`, `final_post_safe`, `reselection_count` |
| False Safe Rate | candidate `pre_risk`, `post_risk`, frozen authoritative threshold |
| Final Candidate Rank | `final_candidate_rank`, `candidate_pool_size` |
| Pre/post risk correlation | candidate-level paired `pre_risk`, `post_risk` |
| Sequence Success Rate | `sequence_id`, boundary hard failures, `completed` |
| Efficiency | evaluated attempts, reselection count, sequence runtime breakdown |
| Repair-agnostic comparison | repair ID and repair config fingerprint |
| Generator-controlled comparison | generator ID, config and checkpoint fingerprints |
| Long-horizon reliability | sequence event count, boundary IDs, hard-failure count |
| Paired significance tests | `evaluation_case_id`, `method_variant_id` |
| Future exhaustive Top-K analysis | all pool manifests plus candidate post-risk fields |

No aggregate function for these analyses exists in the production path. Planned
horizons `[5, 10, 20, 40]` are descriptive metadata only and are not read by
generation.

## Scientific boundary

The sidecar explicitly declares:

```text
Paper 1 experiments implemented = false
Oracle implemented = false
Statistical tests implemented = false
Long-horizon benchmark implemented = false
Production selection behavior changed = false
```

The contract does not implement Top-1/Greedy/GAR comparisons, exhaustive
generation, a new candidate selector, a new repair operator, a Delta-Interpolator,
Paper 1 aggregate metrics, tables, figures, significance tests, or a pilot. Passing
the readiness tests proves only that future experiments can record auditable
primitive data; it does not show that GAR is effective.
