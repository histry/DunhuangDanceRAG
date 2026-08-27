# Refiner corruption-factor diagnosis

This is a training-only experiment, not an independent-validation result and
not permission to resume a rejected Refiner or start Diffusion. Successful
process completion only means the experiment reports were produced.

## Frozen paired design

Reuse the eight clean training windows and descriptor coordinates from the
previous fixed-fit archive. Verify database hashes, selected event contents,
source IDs, configuration and the archive hash before use. Validation metadata
is loaded solely for source/recording separation; validation motion is not read.

Four recipes per window yield 32 fit cases. Private seam/noise RNGs generate
recipes once; actual tangent arrays and hashes are saved. All three factors use
the same seam and tangent recipe:

- `bridge_only`: the existing 0.5 interpolation mixture, without tangent noise.
- `tangent_only`: perturb authentic clean motion without interpolation mixing.
- `mixed`: interpolation mixture followed by the same tangent perturbation.

All modes use identical projection/contact preprocessing. Their baseline
corruption magnitudes differ: compare gains alongside original errors, not
raw final loss alone.

The 64 `probe_unseen_noise` cases change only noise (two new seeds per recipe).
They are **not applicable** to deterministic `bridge_only`, not duplicate wins.
The 16 `probe_unseen_position` cases move the shortest/longest fit recipes to
new positions, preserving each recipe's length and exact tangent array.
Fit and position probes stay within the formal generator's allowed seam
positions; they do not introduce out-of-distribution window-edge seams.
These are unseen corruptions of TRAINING motions, never held-out dancers/sources.

## Models and decisions

First evaluate the existing V6 training snapshot read-only on every bank.
Validate its model/configuration/Event-DB contracts and record its original
revision and checksum. It never initializes the experiments.

Three new full-size Refiner models share initial weights, optimizer settings,
batch order, batch size 8 and 400 updates each. Evaluate at 0/100/200/300/400;
the fixed final update is the primary comparison. Probe scores do not select
checkpoints. Record both per-window raw gates and informative-target rates:
near-zero baseline errors are marked trivial, not repair successes.
Temporal-gain and endpoint-gain component rates are also reported separately;
neither replaces the existing combined temporal/endpoint/jerk criterion.

The original geometry/temporal/endpoint/clean gates are unchanged. Formal
validation and all final video/IK gates remain separate. Outputs use the
`refiner_factor_diagnostic_only_v1` model version; no formal loader accepts them.

## Decoder observations and offline counterfactuals

At evaluation record raw tangent, mask weighting, smoothing, inward taper,
real vector cap, and applied product-log correction. Separate root metres and
joint radians, core/halo/outside, and body groups. Record target alignment,
amplitude ratios, zero/low-mask target-error coverage, and cap fractions whose
denominator is editable core vectors. The cap is measured AFTER smoothing,
not on raw network logits.
The original risk mask dilates beyond the declared seam (default three frames
at 30 fps). Thus `outside` is not necessarily locked: inspect `mask_by_scope`
and `applied_where_mask_zero` to distinguish allowed dilation from a real
zero-mask protection failure. The experiment does not change that policy.

Record task gradients and global gradient-clip scale every 25 updates. Trace
collection is detached and does not change default outputs or gradients.

At reference/final evaluation, four preselected cohort windows per applicable
bank receive frozen-output counterfactuals: full confidence only within the
original support, no smoothing (taper retained), and no cap. Full clean/physical
diagnostics accompany their gains. Unsafe counterfactuals are never published
or used for videos, and no extra optimizer step is taken.

## Server command (after pulling the supplied commit)

Run in the foreground, with old training stopped. Use a NEW experiment tag:

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/diagnose_refiner_factors.sh \
  /home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915 \
  refiner_v6_trial1 refiner_factors_trial1
```

The script preserves the old snapshot and all cached assets. It stops after
diagnosis; it does not resume the 8,000-step run or train Diffusion.

Inside the run's `refiner_factor_diagnostics/refiner_factors_trial1/`:

- `factor_report.json`: compact cross-factor history and links to all reports.
- `clean_cohort.npz`, `recipes.npz`, `bank_*.npz`, `training_order.npy`: replay inputs.
- `reference/<mode>/*.json`: read-only old snapshot evaluations.
- `<mode>/000400_*.json`: final per-window gates and decoder/counterfactual traces.
- `<mode>/gradients.jsonl`: gradient norms/conflicts and clipping.
- `<mode>/diagnostic_final.pt`: isolated diagnostic weights, not formal assets.

Return the factor report first. Do not lower acceptance thresholds based on a
failed diagnosis. A fixed-seed result needs replication before a scientific
claim. No validation/test performance or final video success is implied.
