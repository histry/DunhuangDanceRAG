# Refiner Single-Direction Decomposition Audit

## Scientific question

The completed RCSP diagnostic rescued 5 of 64 temporal gates, all in
`cross_event/10`, while `single_recording` remained 0 of 32. Its median
projected-adapter cosine to the negative temporal gradient was about `0.0013`,
compared with about `0.1288` for cross event. The preceding parameter audit
rejected gradient starvation but found source- and width-dependent parameter
gradient relations. This audit asks where the nearly orthogonal single action
appears in motion space: anatomy, active-time region, or their interaction.

This is a frozen-state decomposition. It does not test width normalization,
select an architecture, train another adapter, or authorize Pilot.

## Frozen evaluation and lineage

The audit evaluates all fixed-final 64 cases: 32 `single_recording`, 32
`cross_event`, and eight cases in every split/role/width group. The eight
`new_position/single_recording/28` cases are retained and also emitted as a
dedicated table. The parameter-attribution report is used only as immutable
provenance and side-by-side scientific context. It never selects a case,
anatomy block, temporal partition, objective, or gradient.

The loader fails closed unless the runtime, trajectory, completed RCSP report,
reporting-logic review, diagnostic adapter checkpoint, and parameter audit have
the expected lineage and read-only contracts. The adapter checkpoint path is
read from `report.json` at
`parameter_update_scope.adapter_checkpoint.path`; no checkpoint filename is
assumed.

## Authoritative action gradient

The objective is the existing temporal scientific deficit returned by
`training.refiner_temporal_action_alignment_audit._scientific_terms`. After a
normal frozen RCSP forward, the 75 geometric channels of `raw_adapted` are
detached and copied with `requires_grad=True`. The unchanged production decoder
maps that copy to the prediction. `torch.autograd.grad` then computes

```text
g_temporal = d L_temporal / d raw_geometric_action_75d
```

No parameter gradient substitutes for this action gradient. All base and
adapter parameters remain frozen, and every parameter `.grad` must remain
`None` before and after the audit. Contact channels `raw[..., 0:4]` are held
fixed and excluded from every decomposition.

The three decomposed actions are:

- base: `raw_base[..., 4:]`;
- adapter: `last_details["adapter_projected"]`;
- total: `raw_adapted[..., 4:]`.

The whole adapter and total cosines must reproduce the corresponding completed
RCSP case rows within the fixed numerical tolerance, including case identity
and defined/null status.

## Anatomy and active-time partitions

The anatomy contract reuses
`motion_geometry.physical.EXTREMITY_JOINTS` and cross-checks it against
`training.refiner_temporal_action_alignment_audit.GEOMETRY_BLOCKS`:

- ROOT coordinates: `0..2`, the root-translation tangent;
- BODY joints: `0,1,2,3,4,5,6,9,12,13,14,15,16,17,18,19`;
- BODY coordinates: `3..23`, `30..32`, and `39..62`;
- EXTREMITY joints: `7,8,10,11,20,21,22,23`;
- EXTREMITY coordinates: `24..29`, `33..38`, and `63..74`.

The audit checks that BODY and EXTREMITY are disjoint and cover all 24 joints,
and that ROOT, BODY, and EXTREMITY cover each of the 75 geometric coordinates
exactly once.

A frame is active when any of its 75 coordinates has positive binary
production support from the current root/joint decoder masks. The ordered
active indices are split with `numpy.array_split(active_indices, 3)` into
EARLY, CENTER, and LATE. This rule uses the actual support indices even when
they are discontinuous; width never chooses or expands the frame interval. An
empty active set is a hard failure.

## Block statistics and interpretation

For whole action, three anatomy blocks, three temporal blocks, and all nine
anatomy-by-time blocks, the audit records gradient norm, action norm, cosine to
the negative gradient, signed contribution, positive and negative contribution
sums, absolute contribution, and cancellation ratio for base, adapter, and
total actions. For scalar coordinate `i`:

```text
c_i = action_i * (-gradient_i)
positive_sum = sum(max(c_i, 0))
negative_sum = sum(min(c_i, 0))
signed_sum = sum(c_i)
absolute_sum = sum(abs(c_i))
cancellation_ratio = 1 - abs(signed_sum) / absolute_sum
```

Cosine is `null` when either norm is zero. Cancellation is `null` when the
absolute contribution is zero. The nine signed block contributions must
reconstruct the whole signed contribution.

Scientific labels use contribution sign, relative single-versus-cross
structure, strict majority case consistency, and cancellation structure. They
do not use an invented cosine threshold. Source and width comparisons are
unpaired group summaries; no nonexistent case pairing or paired correlation is
reported. Results can localize where mismatch appears, but cannot prove a
causal architectural root cause, select a new model, or authorize Pilot.

## Read-only contract

The report schema is `refiner_single_direction_decomposition_audit_v1` and
always records:

```text
optimizer_steps = 0
parameter_update_performed = false
checkpoint_selection_performed = false
scale_selection_performed = false
architecture_selection_performed = false
width_conditioning_added = false
production_model_modified = false
production_inference_modified = false
scientific_acceptance = false
publish_allowed = false
pilot_allowed = false
```

Before completion, the audit re-hashes the base and adapter states plus every
source, trajectory, RCSP, review, adapter-checkpoint, and parameter-attribution
artifact. Output uses a fresh create-only directory.

## Server execution

After updating the server to the reviewed commit, set the fixed artifact paths
and create a fresh run directory:

```bash
ROOT=outputs/run_smpl14_formal_20260822_163915
SOURCE="$ROOT/checkpoints/refiner_v15_4_1_lazy_reservoir_foundation_20260831_235832/bridge_diagnostic"
TRAJECTORY="$ROOT/audits/zero_start_trajectory_20260901_112920_vpO8Lh/trajectory"
RCSP_RESULT="$ROOT/audits/role_conditioned_support_projection_20260902_132948_qu1hYg/result"
PARAMETER_REPORT="$ROOT/audits/rcsp_single_direction_attribution_20260902_145442_VWA1LQ/result/report.json"
RUN_DIR="$(mktemp -d "$ROOT/audits/single_direction_decomposition_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

bash scripts/audit_refiner_single_direction_decomposition.sh \
  "$SOURCE" "$TRAJECTORY" "$RCSP_RESULT" "$PARAMETER_REPORT" "$RUN_DIR"
```

Inspect `result/report.json` and stop. Do not launch normalization work,
architecture changes, retraining, or Pilot from this audit automatically.
