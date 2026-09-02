# Phase 2 — Cross-event width normalization audit

This is a read-only, fixed-state audit. It evaluates the step-400 frozen base
and the completed RCSP adapter at alpha=1. It does not train, select a
checkpoint or scale, modify the Refiner/decoder/support/gate, run an alpha
sweep, or authorize a Pilot.

## Primary cohort

The scientific cohort is exactly 32 `cross_event` cases:

- `seen/cross_event/10`: 8
- `seen/cross_event/28`: 8
- `new_position/cross_event/10`: 8
- `new_position/cross_event/28`: 8

The 32 `single_recording` cases are retained for frozen parity and integrity
checks, but are explicitly excluded from normalization medians, ratios,
classification, and dominant-mechanism selection. Width-10 and width-28 are
unpaired group comparisons unless authoritative metadata proves an identical
boundary identity; this audit therefore records `pair_key: null` and
`fake_case_pairing_performed: false`.

## Authoritative temporal contract

The temporal metric is `motion_geometry.boundary_observables.boundary_metrics_torch`:

```text
A = sum(v2 * S2) / max(sum(S2), 1)
K = sum(v3 * S3) / max(sum(S3), 1)
M_temporal = A / 10 + K / 1000
```

`v2` and `v3` are the actual FK joint-vector norms after second and third
finite differences, scaled by `fps**2` and `fps**3`. `S2` and `S3` are the
production seam-touching derivative supports. Thus the denominator is the
valid seam-touching derivative-stencil count, not duration, total frame count,
active-frame count, joint count, coordinate count, or soft-weight mass.

The scientific deficit is the production
`training.motion_models._observable_refiner_objective` path:

```text
D = one_sided_huber(
      relu(M_candidate - (1-gain) * M_degraded)
      / max(abs(M_degraded), TRAIN_reference_scale_floor),
      shoulder=gain)
```

The original observable temporal gate uses relative metric gain
`(M_before-M_after)/M_before`, with threshold
`cfg.checkpoint_validation_min_temporal_repair_gain`, plus the unchanged jerk
non-regression condition. A positive reported gate margin means the relative
gain is at or above that original threshold; a negative margin means it is
below the threshold.

There is no explicit width dependency in the objective, feature normalization,
decoder, support, or gate. Any width contrast is therefore measured from the
resulting derivative-support counts, soft-weight mass, temporal distribution,
finite-action efficiency, gate margin, and frozen direction covariates.

## Decoder stages recorded

For both BASE and RCSP, the audit reads the production decoder trace:

`raw_action -> soft_weighted_action -> smoothed_action -> tapered_action ->
capped_action/final_tangent -> final_decoded_geometric_displacement`.

For RCSP it additionally records `raw_adapter` and
`binary_projected_adapter`. Effective weight mass is read from the production
`root_weight` and `joint_weight`; effective count is `(sum(w)^2)/sum(w^2)`.
No synthetic weight or proxy temporal error is introduced.

## Server command

After the commit is on `main`, synchronize the server and run:

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
git fetch origin main
test "$(git rev-parse HEAD)" = "a9fbff524e46b0e13ab5e902f09c608e43cfb40f"
test "$(git rev-parse origin/main)" = "a9fbff524e46b0e13ab5e902f09c608e43cfb40f"
test -z "$(git status --porcelain)"

export ROOT_DIR=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915
export EXPECTED_MAIN_COMMIT=a9fbff524e46b0e13ab5e902f09c608e43cfb40f
export PYTHON=/home/disk/lsm/conda_envs/edge/bin/python
export LEGACY_CORE_STRENGTH='<recorded source core strength>'
export LEGACY_TRANSITION_STRENGTH='<recorded source transition strength>'
bash scripts/audit_refiner_cross_width_normalization.sh 2>&1 | tee "$ROOT_DIR/audits/phase2_console.log"
```

The placeholders must be replaced by the values recorded in the frozen source
contract. Do not invent them. The audit creates a fresh
`audits/cross_width_normalization_<timestamp>_<random>/result/report.json`.
After report acceptance checks, stop. Do not automatically intervene, train,
run a Pilot, or claim a causal root cause.
