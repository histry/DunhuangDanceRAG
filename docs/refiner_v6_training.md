# Refiner V6: tolerance constraints and fixed real-window diagnosis

V5's 1,000-step report at `d9b3c74` had 0/16 geometry and temporal validation
passes, despite 16/16 clean-identity passes. Its 8 training probes also failed
repair. That evidence established weak repair, not defective SMPL source data.
The loss conflict was a hypothesis, not something the old loss log proved.

## Changes and unchanged acceptance criteria

- Clean geometry/contact penalties are per-window dead-band constraints using
  the existing 0.005 product-log and 0.05 contact tolerances. Inside a tolerance,
  the penalty and gradient are zero; another sample cannot dilute a violation.
- High-frequency protection compares global/local-window and extremity FK
  jerk p95, plus global jerk max, against
  `max(clean * ratio, clean + margin)`, using the same `StageAcceptancePolicy`
  values as the clean audit. The differentiable surrogate is NOT the full
  clean gate: foot/support/rotation and other clean checks still run unchanged.
- Clean planted-foot speed also uses the existing p95/max ratio/margin budget.
  Its support mask is frozen from clean input; changing contact predictions or
  making a foot faster cannot hide a violation. Safe foot motion is not driven
  to zero. The full audit still independently checks actual support and drift.
- Repair and weighted clean gradients are measured before global clipping.
  Reports include norms, norm ratio, cosine, conflict flag and combined/summed
  norm ratio. Zero clean gradient has `gradient_cosine=null`, not a conflict.
  Measurements do not mutate optimizer gradients or take an extra update.
- Random formal training and fixed-fit diagnosis share the batch preprocessing,
  risk masks, full-size model, decoder, loss and optimizer settings.
- Geometry, temporal and endpoint improvement requirements remain 3%; stage
  coverage and clean-identity coverage remain 75%. Final absolute whole-song
  gates still belong after Diffusion, boundary repair and IK.

V6 rejects V1-V5 Refiner models and resume snapshots. Do not rename an old model,
a diagnostic model, or a rejected candidate into `boundary_refiner.pt`.

## Reuse existing assets

Keep the current retarget cache, train/val/test Event-DB, Router, Duration,
Planner and generation index. None of the following commands rebuilds them.
The test split is never used for fitting. Validation metadata is read only for
source/recording separation; fixed-fit diagnosis never reads validation motion.

First pull the supplied commit and run tests in the existing `edge` environment.
Use a new candidate tag. All commands run in the foreground, with flushed logs.

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
export PY=/home/disk/lsm/conda_envs/edge/bin/python
bash scripts/train_refiner_v6.sh diagnose \
  /home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915 \
  refiner_v6_trial1
```

This runs 400 updates on 8 deterministic, source-balanced **training** windows,
with fixed corruption prepared once. It uses the full 256-channel Refiner,
not the small synthetic root-only test network. Evaluation runs every 50 updates
and gradient diagnosis every 25. This is deliberately an overfit/learnability
test, not a generalization estimate or a formal checkpoint selection stage.

Artifacts inside the existing run:

```text
checkpoints/refiner_v6_trial1/
  console.fixed_fit_*.log
  fixed_fit/
    fixed_training_batch.npz
    diagnostic_report.json
    fit_step_*.json
    gradient_diagnostics.jsonl
    diagnostic_weights.pt
```

Inspect `history`, `best.metrics.windows`, `last`, `fit_gate` and `gradient_history`.
`fit_gate` describes the best small-fit checkpoint, not necessarily the last
update. Every evaluation's full window details are also saved separately.
The report records event paths/hashes, fixed seams, seed, configuration, database
hashes, code revision and implementation hashes. If the fit fails, exit code 2
means **do not start formal training**. The failed report is still saved.
`published=false` and `scientific_acceptance=false` remain true even when
`fit_gate.fit_passed=true`: train-fit evidence cannot accept a formal model.

## Fresh source-disjoint pilot, only after diagnosis passes

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
export PY=/home/disk/lsm/conda_envs/edge/bin/python
bash scripts/train_refiner_v6.sh pilot \
  /home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915 \
  refiner_v6_trial1
```

The launcher rejects missing/failed/stale diagnostics. It checks the exact
revision, code/configuration, databases and selected event contents. It starts
a **fresh** model/optimizer/RNG sequence, not the overfit diagnostic weights.
The pilot pauses at 1,000/8,000 and publishes nothing. Examine its independent
validation report before continuing. Formal training logs weighted gradient
conflicts every 200 updates in `boundary_refiner.gradient_diagnostics.jsonl`.

## Explicit continuation after reviewing the pilot

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
export PY=/home/disk/lsm/conda_envs/edge/bin/python
bash scripts/train_refiner_v6.sh resume \
  /home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915 \
  refiner_v6_trial1
```

This restores the exact V6 pilot state to 8,000 total updates. Only an accepted
Refiner proceeds to 15,000 Diffusion updates. Both new models must pass before
promotion; old formal models are backed up. The script does not generate video.
An interrupted V6 run can repeat `resume` with the same tag/config/revision.
Changed objectives require a fresh tag, diagnosis and pilot.

Local real-data fitting can verify learnability on local training windows. It
does not verify the server's 328-event training split, held-out performance,
end-to-end musical alignment, IK acceptance or final rendered video quality.
