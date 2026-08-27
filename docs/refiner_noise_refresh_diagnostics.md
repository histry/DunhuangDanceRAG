# Fixed versus refreshed Refiner noise

This bounded TRAIN-only diagnostic follows the corruption-factor experiment.
It tests whether reusing fixed corruptions explains the correction-direction
generalization gap. It does not establish that noise refresh will solve it.
The formal trainer already draws new corruptions per batch; this experiment
isolates that variable on the same eight authentic training windows.

## Controlled comparison

There are four fresh full-size models: `tangent_only` and `mixed`, each with
`fixed_noise` and `refreshed_noise`. Deterministic `bridge_only` has no noise to
refresh and is explicitly not applicable, rather than run as a duplicate.

Both arms reuse the earlier factor run's exact **untrained initial state**,
verified against the seed-generated fresh model. They use the same optimizer,
batch size 8, window/recipe order, four fixed seams per window, intensity 0.06,
400 updates, clean constraints, decoder masks/smoothing/caps and acceptance
metrics. No model/objective/configuration change is part of this comparison.
The fixed arm repeatedly uses the old 32 fit recipes. The refresh arm gets a
new tangent array at each sample presentation, including revisits to the same
window/position. Both motion modes share that exact refresh schedule.

Private streams pre-materialize noise arrays, without touching model or sample
RNG state. Actual arrays, offsets, seeds, hashes and the draw schedule are saved
for replay. Seed **and array-hash** collisions with old/probe/other refresh
noise are rejected. Noise amplitude, smoothing and boundary taper do not change.
Formal preprocessing is shared by fixed banks and on-demand refresh batches;
risk masks are recomputed from the new corrupted input, not copied from old noise.

## Evaluation and isolation

Use three frozen banks per motion mode:

- `anchor_fixed_noise`: the original 32 fit cases. These are training cases for
  the fixed arm but unseen noise at training positions for the refresh arm.
  It is intentionally **not** described as training fit for both arms.
- `probe_unseen_noise`: 64 newly generated seeds at the same training positions,
  unseen by both arms and distinct from the previously inspected factor probes.
- `probe_unseen_position`: 16 new legal seam positions not used in training or
  the earlier position probes (both starts and centers must be new, across all
  seam widths). Each uses the **same new tangent array** as a
  linked case in `probe_unseen_noise`, with the same window and seam length.
  Thus the linked pair changes position only, not noise and position together.

The 32/64/16 recipes are correlated observations on **eight training windows**,
not 112 independent motions or unseen dancers. Validation DB metadata is used
only for source/recording separation; its motion is never read or fitted.
TRAIN/validation DB hashes, fixed cohort contents, descriptors, sources,
configuration, mask environment, acceptance policy and formal model code are
checked against the earlier experiment. Changed contracts fail closed.

Evaluate at steps 0/100/200/300/400. The fixed final step is primary; no best
checkpoint is chosen using probes. Every bank also has an **exact no-edit**
baseline: degraded input stays degraded and clean input stays clean, including
contact channels. Zero model logits are not used as a substitute for identity.
No-edit direction cosine is undefined, not a fabricated zero or repair win.

Reports retain per-case geometry/combined temporal/clean gates, decomposed
temporal and endpoint gains, raw decoder stages, direction cosines, mask/cap
observations and near-zero-target exclusions. Gradient norms/conflicts and
global gradient clipping are logged every 25 updates. Final `comparison.json`
includes paired pass/fail counts and links to every detailed case report.
`*_position_pairs.json` compares precisely noise-matched positions.

## Server run

After pulling the supplied commit, stop old training and use a fresh output tag:

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/diagnose_refiner_noise_refresh.sh \
  /home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915 \
  refiner_v6_trial1 refiner_factors_trial1 refiner_noise_refresh_trial1
```

This foreground run reuses existing assets, checks CUDA, streams output and
saves a timestamped console log. It never rebuilds Event-DB/cache, modifies the
Router/Duration/Planner or trains Diffusion. An existing destination is refused.

Under `refiner_noise_refresh/refiner_noise_refresh_trial1/`:

- `noise_refresh_report.json`: provenance, isolation, progress and all four runs.
- `comparison.json`: final no-edit/fixed/refreshed comparison; printed compactly
  to the console as well.
- `<mode>/<arm>/000400_*.json`: final detailed cases and paired-position reports.
- `<mode>/<arm>/gradients.jsonl`: losses, gradient diagnostics and elapsed time.
- `<mode>/<arm>/diagnostic_latest.pt`: latest fixed-step weights, saved before
  evaluation. This is **not a resumable formal training snapshot**.
- `recipes.json`, `evaluation_noise.npz`, `refresh_noise.npz`, `training_order.npy`
  and `clean_cohort.npz`: replay inputs and sampling order.

All diagnostic weights have a non-formal model version and are rejected by
formal inference. Exit code zero means diagnostic execution completed, **not**
model acceptance. Review held-out noise, paired positions and clean protection
together. Do not lower thresholds or start an 8000-step run/Diffusion merely
because fixed recipes can be fitted. Replication and independent validation
remain necessary even if this small controlled comparison improves.
