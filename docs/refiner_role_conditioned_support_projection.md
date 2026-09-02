# RCSP Adapter Diagnostic Experiment

## Scope

`refiner_role_conditioned_support_projection_experiment_v2` is a diagnostic
rescue experiment on the immutable, completed 400-step A0 Refiner trajectory.
It does not modify `ProductManifoldTemporalRefiner`, the production decoder,
formal inference, GAR, contact, confidence masks, smoothing, taper, caps,
thresholds, losses, or the 3% gate.

The experiment asks one narrow question: can a small role-conditioned
geometric direction adapter improve the fixed final temporal result while the
existing Refiner and decoder remain frozen?

## Architecture and routing

The input to the frozen Refiner's final `out` convolution is captured by a
temporary forward hook. The production forward is not reimplemented. Two
independent zero-initialized heads consume this feature:

```text
single_adapter = Conv1d(hidden_dim, 75, kernel_size=1)
cross_adapter  = Conv1d(hidden_dim, 75, kernel_size=1)
```

The fixed role mapping is:

```text
0 = single_recording
1 = cross_event
```

TRAIN role ids come from the existing four-group contract. Fixed-final role
ids come from the explicit split/role bank metadata. Width and case position
never determine role. There is no width embedding, width-specific head,
attention, bottleneck, or architecture search.

## Geometry and support contract

The adapter produces only the 75 geometric tangent coordinates: three root
translation coordinates and 24 times three joint-rotation coordinates.
Frozen base contact channels `raw[..., :4]` remain bit-identical.

Production `_refiner_decode_masks` supplies the authoritative effective
`root_weight` and `joint_weight`. RCSP converts these values to binary support
with `weight > 0` and projects the adapter correction onto that support. It
does not multiply the correction by soft confidence. The resulting adapted
raw output is passed to `_decode_product_refiner_output`, where the unchanged
production decoder applies soft confidence once, followed by the existing
smoothing, taper, caps, and retraction. Support is not expanded.

## Optimization and data

All base Refiner parameters have `requires_grad=False`; the base runs in eval
mode and its forward is detached. Only the two adapter heads are passed to the
same AdamW, gradient clipping, checked Armijo/rollback step, subgroup guard,
objective, frozen TRAIN reservoir, and transaction schedule used by the A0
trajectory. Optimizer hyperparameters are not searched. Exactly 400 checked
steps are attempted, and only fixed step 400 is evaluated.

The held-out `new_position` probe is used only for mandatory step-zero parity
and fixed-final evaluation. It is never supplied to an optimizer closure and
does not select a checkpoint or architecture.

## Fail-closed parity

Before training, the zero adapters must reproduce the frozen A0 model for the
anchor, every context-reservoir bank, the full TRAIN transaction 0, and fixed
final 64 cases. The audit checks raw output,
decoded repair and clean motion, temporal and endpoint objective values, and
the complete fixed-final physical, geometry, clean-identity, and observable
case records. The recomputed A0 final also has to match the historical
trajectory report for all 64 cases. Any mismatch aborts before step 1.

## Fixed-final report

The final report compares `BASE` and `RCSP` at production alpha 1 for all
eight split/role/width cells. It records gate counts, physical/geometry/clean
counts, scientific deficits, repair gains, role-adapter updates and outputs,
support retention, and final temporal raw-action direction cosines. The
direction VJP runs once after step 400 and is never used for an update.

The output is diagnostic evidence only. Every report and adapter artifact
fixes the following flags:

```text
production_model_modified = false
checkpoint_selection_performed = false
scale_selection_performed = false
scientific_acceptance = false
publish_allowed = false
pilot_allowed = false
```

## Server execution

Create a fresh parent directory and pass it to the shell wrapper. The wrapper
creates `result/` through the Python module and writes `console.log` beside it.
The exact runtime commit must be supplied through `EXPECTED_COMMIT`.

```bash
RUN_DIR="$(mktemp -d outputs/run_smpl14_formal_20260822_163915/audits/role_conditioned_support_projection_$(date +%Y%m%d_%H%M%S)_XXXXXX)"

bash scripts/run_refiner_role_conditioned_support_projection.sh \
  "$SOURCE" "$TRAJECTORY" "$RUN_DIR"
```

Do not run Pilot after this experiment. Review `result/report.json` first.

## Reporting-logic correction for completed v1 artifacts

The first completed server artifact used the v1 report schema. Its case-level
measurements, BASE/RCSP summaries, direction VJP, and support statistics are
valid. The v1 headline classifier nevertheless mixed two different events:

- any decrease in the continuous temporal scientific deficit;
- an actual increase in the fixed 3% temporal gate pass count.

That made small deficit decreases in both widths hide a gate-level width
asymmetry. Schema v2 separates descriptive deficit improvement from gate
rescue and records gate deltas by group, role, and width. A completed v1 run
must not be retrained merely to obtain the corrected classification. Review it
directly with the read-only reviewer:

```bash
bash scripts/review_refiner_role_conditioned_support_projection.sh \
  "$REPORT" "$REVIEW_JSON"
```

The reviewer hashes the source report, recomputes all stored summaries from
the 64 BASE and 64 RCSP case rows, verifies the diagnostic-only flags and zero
support escape, and prints the already-recorded direction/support summaries.
It creates no optimizer, does not load the adapter checkpoint, changes no
measurement or threshold, and cannot authorize publication or Pilot. Because
the reporting rule was corrected after the v1 result existed, the review
artifact labels itself `post_run_reporting_logic_correction`.
