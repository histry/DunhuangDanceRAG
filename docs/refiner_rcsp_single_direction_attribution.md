# RCSP Single-Direction Attribution

## Scientific question

The completed RCSP fixed-final diagnostic produced all five temporal gate
rescues in `cross_event/10`. No `single_recording` case crossed the temporal
gate. Its recorded projected-action direction cosine was about `0.0013` for
single recording and `0.1288` for cross event. Binary projection retained
nonzero single action and had exactly zero support escape; width 28 retained
more action than width 10 despite having no temporal gate rescue.

The next question is therefore narrower than another adapter experiment:

> Does one role head receive conflicting temporal parameter gradients across
> widths, or did its learned displacement from zero become non-descent for the
> fixed final single-recording groups?

## Fixed read-only protocol

`refiner_rcsp_single_direction_attribution_v1` loads the immutable step-400 A0
base and the completed step-400 RCSP adapter. It evaluates exactly:

- frozen TRAIN transaction 0, with 48 cases in each role/width group;
- the fixed 32 seen final cases;
- the fixed 32 held-out `new_position` final cases.

For each source and role/width group, it computes the mean temporal scientific
deficit and its true unclipped gradient with respect to that role head's own
weight and bias. It records:

- parameter-gradient norm;
- cosine between the learned parameter displacement from exact zero and the
  current negative temporal gradient;
- within-role pairwise gradient cosines;
- width-10 versus width-28 gradient cosine at TRAIN, seen, and new position;
- TRAIN-to-final same-width gradient cosine.

Single and cross heads occupy disjoint parameter spaces, so the audit does not
report a meaningless cross-role parameter cosine. A negative within-role
cosine is reported as a descriptive sign conflict. A nonnegative cosine does
not prove adequate capacity, generalization, or finite-step effectiveness.

## Immutability and claim boundary

The audit uses `torch.autograd.grad` only. It constructs no optimizer, calls no
backward update, performs no gradient surgery, and changes no parameter. It
checks base/adapter state hashes before and after, hashes the frozen source,
trajectory, RCSP report/checkpoint/review, and verifies the probe remains
unchanged. The held-out probe is used only for read-only final attribution.

The output always fixes:

```text
optimizer_steps = 0
checkpoint_selection_performed = false
scale_selection_performed = false
width_conditioning_added = false
production_model_modified = false
scientific_acceptance = false
publish_allowed = false
pilot_allowed = false
```

No new head, normalization, training run, production edit, or Pilot is
authorized by this audit.

## Server execution

Use a fresh audit directory outside the immutable source, trajectory, and RCSP
run:

```bash
RUN_DIR="$(mktemp -d "$ROOT/audits/rcsp_single_direction_attribution_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

bash scripts/audit_refiner_rcsp_single_direction.sh \
  "$SOURCE" "$TRAJECTORY" "$RCSP_RESULT" "$RUN_DIR"
```

Review `result/report.json` before deciding whether the next diagnostic should
target within-single width conflict, TRAIN-to-final generalization, or finite
action efficiency. Pilot remains forbidden.
