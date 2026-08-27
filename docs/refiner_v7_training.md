# V7: observable full-bridge repair

Original Chang-E SMPL14 files, retarget cache, Event-DB, Router, Duration,
Planner and generation index are preserved. This is a NEW repair protocol, not
a claim that the original motion data were defective or that training is now
guaranteed to succeed. No external pretrained model is introduced.

## What changed

- Formal reference corruption replaces the entire missing span with the same
  root-Hermite / SO(3)-SLERP bridge used by generation. No retained clean
  interior, no random tangent target. Explicit historical factor recipes remain
  diagnostics, not the formal default.
- Refiner training uses both single-recording occlusion and cross-source event
  joins. Cross joins have no hidden ground-truth interior. Their repair loss
  uses observed boundaries, not an invented clean target. The separate clean
  identity branch still receives authentic original motion and its own descriptor.
- Both models receive phase, relative endpoint poses and endpoint velocities
  derived only from the observed input. The decoder mask, smoothing and caps
  remain; halo strength is now identical in training, validation and inference.
- Refiner repair loss uses actual world-space boundary velocity jumps,
  acceleration/jerk and reference-relative safety. Clean protection still uses
  the existing dead-band/support/jerk tolerances. Geometry trust bounds are
  relative to the input bridge, not a required unique missing trajectory.
- Diffusion retains its conditional generative noise-prediction objective;
  it is not a deterministic inverse-noise Refiner. Its physical regularizer
  uses the observable contract. Publication validation runs the FULL reverse
  process from noise, not teacher-forced reconstruction containing clean data.
- Publication requires single-recording AND cross-event boundary validation.
  Endpoint and temporal pass-rate thresholds remain 0.75; per-window minimum
  gains remain 0.03; actual seam jerk and stage physical safety cannot regress.
  Original clean reconstruction/derivative-error rates are separately reported
  diagnostics, not the new repair-rate definition. Do not compare old and new
  rates as if they measured the same thing.
- V6 Refiner / V2 Diffusion models and snapshots are incompatible. New versions
  are `product_manifold_boundary_refiner_v7` and
  `reference_tangent_motion_diffusion_v3`. Use a NEW candidate tag.

## Foreground server workflow

Use Bash on the existing server. Every command below assumes the four variables
are initialized in the current shell. The release response supplies the exact
`EXPECTED_COMMIT`; do not paste a `<NEW_COMMIT>` shell placeholder.

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
git checkout main
git pull --ff-only origin main
export PY=/home/disk/lsm/conda_envs/edge/bin/python
export OUT_ROOT=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915
export TAG=refiner_v7_bridge_trial1
# Compare this SHA with the reviewed release before running:
git rev-parse HEAD
export EXPECTED_COMMIT="$(git rev-parse HEAD)"
```

### 1. Controlled training-window diagnosis

```bash
bash scripts/train_refiner_v7.sh diagnose "$OUT_ROOT" "$TAG"
```

Eight source-balanced TRAIN windows, two seam widths, fixed training joins and
new held-out seam positions. Both single-recording and cross-event cases are
evaluated, with an exact no-edit baseline. Only TRAIN motion is used; validation
metadata is read solely to verify source separation. Fixed 400 updates,
evaluation at 200/400, final-step readiness (not best probe selection).

Report:
`$OUT_ROOT/checkpoints/$TAG/bridge_diagnostic/diagnostic_report.json`

`published=false` is intentional. `diagnostic_ready=false` returns nonzero and
blocks the pilot. Do not reuse diagnostic weights or weaken thresholds. This
experiment is not independent-validation evidence.

### 2. Fresh independent-validation pilot

Only after diagnosis passes:

```bash
bash scripts/train_refiner_v7.sh pilot "$OUT_ROOT" "$TAG"
```

Fresh initialization, 1000/8000 updates, independent source-disjoint validation
including cross-event joins. Saves a recovery snapshot and best candidate;
does not replace the formal model. Inspect:
`$OUT_ROOT/checkpoints/$TAG/boundary_refiner.validation_step_001000.json`.

### 3. Resume Refiner, then train Diffusion

Only after the pilot's actual validation metrics pass:

```bash
bash scripts/train_refiner_v7.sh resume "$OUT_ROOT" "$TAG"
```

The wrapper recomputes the pilot decision and refuses failed/stale reports.
It resumes the matching V7 snapshot to 8000, evaluates every 1000 and publishes
the best accepted Refiner candidate. Only then does Diffusion train for 15000
updates. Snapshots are saved every 200 updates. Both candidates must pass
before either replaces the formal files. Prior models are backed up in the
candidate directory. Rejected files never replace formal models.

The Diffusion training references are single-recording complete bridges; cross
joins are evaluated without a unique clean label. A passing isolated model is
still not proof that Refiner + Diffusion + IK composition passes the final gate.

### 4. Reuse assets for full generation

After both models are accepted/promoted:

```bash
bash scripts/train_refiner_v7.sh generate "$OUT_ROOT" "$TAG"
```

This disables rebuilding cache/Event-DB and retraining all five models, while
retaining asset compatibility, source-split, routing and final physical gates.
It uses `assets/music/test/audio/dunhuangwu2.wav` by default. Override `AUDIO`
or `SMPL_DIR` explicitly if needed. Do not suppress a provenance mismatch.

Expected final video: `$OUT_ROOT/results/video.mp4`. File existence alone is
not acceptance: check this run's final gate, file time, decode and playback.

## Inspect without rerunning

Each candidate directory contains foreground logs, snapshot paths and per-step
validation. The new diagnostic reports include `baseline`, `final.seen`,
`final.new_position`, cross-event rows, separate observable endpoint/temporal
rates, actual before/after dynamics, physical failure reasons and gradient logs.
Raw single-recording clean reconstruction metrics remain visible for comparison.
