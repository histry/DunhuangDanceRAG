# Refiner fixed-final failure and contact connectivity audit

## Question and boundary

The fresh exact-zero A0 trajectory accepted all 400 checked updates. Its output
head became nonzero at step 1, true trunk gradient and retained trunk movement
began at step 2, and the trunk continued to move. The fixed step-400 diagnostic
still failed. Meanwhile all logged contact-head parameter and retained-update
norms were exactly zero.

This audit explains those two observations without another intervention. It is
strictly read-only: no optimizer is constructed, no parameter is changed, the
probe is used only to reproduce the already fixed final evaluation, and no
checkpoint is selected. `scientific_acceptance`, `publish_allowed` and
`pilot_allowed` remain false unconditionally.

## Immutable inputs

The runner consumes:

- the complete A0 trajectory produced by commit
  `b2d71e1fa92cb2a6723810060722c0edea7a3a99`;
- its `report.json`, `experiment.json`, `diagnostic_latest.pt` and
  `updates.jsonl`;
- the original frozen V15.4.1 `diagnostic_report.json`,
  `diagnostic_state.pt`, `fit_bank.pt` and `probe_bank.pt`.

It verifies the trajectory schema, 400-step completion, nonpublishing flags,
canonical experiment hash, final checkpoint file hash, model-state hash,
runtime commit, source hashes and probe hash. All input hashes are checked again
before the create-only report is written. Any mismatch fails closed.

The V15.4.1 source omitted decoder strengths. Core `0.02` and transition `1.0`
remain explicit legacy inputs, not reconstructed metadata.

## Failure attribution

The report reproduces the fixed final `seen` and `new_position` evaluation and
requires four groups of eight cases in each split. It records a group table and
all 64 cases separately. Each case contains:

- observable endpoint before/after, absolute and relative gain, and threshold;
- temporal energy before/after, gain, threshold, seam jerk and its 2% budget;
- input-relative FK/product reference fidelity and its absolute limits;
- physical before/after joint/extremity jerk, support drift and penetration;
- clean-branch product-log and contact L1 identity metrics;
- exact categorized failure reasons.

Hidden-clean repair geometry is reported only for `single_recording`. It is
explicitly unavailable for `cross_event`, whose scientific protocol forbids
using hidden clean content. Failure counts are kept by split and by full
split/role/width group, so averages cannot hide case failures.

The group `passed` field is the unchanged authoritative group decision.
Additional geometry and clean columns expose attribution; they do not define a
second acceptance rule.

## Contact connectivity

For frozen TRAIN transaction 0 and both final splits, the audit reports source
and effective contact-mask count, fraction, mean, RMS and maximum for repair and
clean branches in every role/width group. Root and joint effective-mask stats
are included as controls.

On the fixed final model, ordinary `torch.autograd.grad` measures the current
repair objective, clean identity objective and weighted training total. For
each objective the report slices true `out.weight` and `out.bias` gradients into
contact rows 0:4, root rows 4:7 and joint rows 7:79. It also records gradients
with respect to raw contact/root/joint outputs in repair and clean branches.
No surrogate gradient is constructed and `.grad` is restored exactly.

Actual decoder VJPs measure
`d(sum(decoded_contact))/d(raw_output)` on TRAIN and final probe masks. A zero
mask control and a nonzero synthetic-mask control use the real decoder. The
nonzero control is also compared with central finite difference. These controls
test connectivity only; they do not alter the model, artifact, mask policy or
objective.

`updates.jsonl` is streamed in exact step order. It proves whether the contact
parameter and retained update stayed exactly zero for all 400 accepted steps.
That historical artifact did not save contact-row gradients separately, so the
report does not invent them: it names this evidence limit and reports a direct
true-gradient measurement at the fixed final state.

The origin classifier uses four measured alternatives:

- `mask_zero`: actual effective masks and decoder derivatives are zero while
  the real decoder has a nonzero synthetic-mask derivative;
- `objective_zero_gradient`: actual decoder connectivity is nonzero but current
  objective gradients are zero;
- `decoder_zero_jacobian`: even the nonzero-mask control has zero derivative;
- `mixed_or_not_identified`: evidence does not isolate one of the above.

Rollback is reported independently. Since the completed A0 trajectory retained
all 400 steps, rollback cannot be silently assigned as the cause.

## Objective/gate alignment and historical boundary

The report places step-1, step-400 and fixed-final TRAIN objective values beside
the held-out discrete scientific gates. A checked scalar descent is not itself
a held-out endpoint/temporal pass. Case and group tables show which required
quantity remained unmet.

The historical V15.4.1 final groups may be displayed for context, but
`historical_comparison_is_descriptive_only=true` is mandatory because the runs
do not have matched initialization. The audit cannot claim that A0
statistically beats V15.4.1.

## Server execution

Run only after updating a clean server `main` to the reviewed audit commit:

```bash
export PY=/home/disk/lsm/conda_envs/edge/bin/python
export EXPECTED_COMMIT=<reviewed-audit-commit-sha>

bash scripts/audit_refiner_final_failure.sh \
  "$SOURCE" \
  "$ROOT/audits/zero_start_trajectory_20260901_112920_vpO8Lh/trajectory" \
  "$RUN_DIR"
```

The new directory contains `report.json` and `console.log`. Completion means
the fixed state was attributed. It does not authorize training or Pilot.

## Validation boundary

Local regression passed the new audit plus trajectory tests (`32 passed`), the
broader Refiner/bridge/product-manifold/training-contract set (`335 passed`),
and the complete repository suite (`788 passed`). The full suite emitted one
existing PyTorch Transformer nested-tensor warning. Python compilation, CLI
help, shell `bash -n` and `git diff --check` also passed. Synthetic tests prove
the audit contracts and derivative controls; only the server command above can
produce attribution for the fixed GPU artifact.
