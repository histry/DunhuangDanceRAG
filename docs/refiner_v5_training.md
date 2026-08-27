# Refiner V5: objective alignment and staged training

Historical protocol. V5 did not pass the supplied 1,000-step real-data pilot.
The V5 launcher is retired; use [Refiner V6 diagnosis](refiner_v6_training.md).
Do not resume V5 snapshots under the V6 objective.

## Why V4 was rejected

The supplied server run at `4d33cf4` completed 8,000 updates but its best
source-disjoint candidate (step 3,000) passed geometry / temporal / clean identity
on only 5 / 5 / 7 of 16 windows. This was a rejected experiment, not a final video.

Code-level failures reproduced by regression tests:

1. Disabling exact tangent repair supervision also disabled the clean branch's
   temporal supervision. Equal-amplitude smooth and oscillating translations
   had the same clean identity loss.
2. Geometry training weighted risk confidence and the soft halo, but validation
   measured unweighted product-log error on the seam core. The unnormalized
   repair margin also gave mild corruptions proportionally less influence.
3. Hard support edges could turn a constant residual into an impulsive jerk.
4. Zero contact logits changed observed contacts toward 0.5 rather than acting
   as an identity operation.

V5 adds independent clean FK velocity/acceleration/jerk protection; per-window
relative geometry and temporal margins; support-preserving filtering/tapering
of corrections only; and identity-initialized residual contact logits. Contact
BCE retains gradients for incorrect binary observations. Original motion and
uneditable support are not smoothed. Caps and formal acceptance thresholds
remain in force. Old Refiner models/snapshots are deliberately incompatible.

These fixes do not prove source-disjoint generalization. The real server must
still pass the unchanged checkpoint gates, then the full generation/IK gates.

## Reuse policy

Reuse the existing SMPL retarget cache, train/val/test Event-DB, Router, Duration,
Planner and generation index. Train a new Refiner and Diffusion in an isolated
candidate directory. Do not rename a rejected/best/snapshot file into a formal
model. Historical files are preserved; promotion happens only after both new
models pass, and previous formal files are backed up in the candidate directory.

## Foreground pilot (no tmux)

First pull the supplied commit and run `python -m pytest -q` in the server's
existing `edge` environment. Then:

```bash
export PY=/home/disk/lsm/conda_envs/edge/bin/python
cd /home/disk/lsm/storage/DunhuangDanceRAG
bash scripts/train_refiner_v5.sh pilot \
  /home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915 \
  refiner_v5_trial1
```

This stops at 1,000/8,000 updates. It is a pause, **not acceptance or publication**.
The snapshot still records an 8,000-step target, so later continuation restores
the exact model, optimizer and random states. Use the same tag and commit.

Inspect `checkpoints/refiner_v5_trial1/boundary_refiner.validation_step_001000.json`
inside the run directory:

- `checkpoint_decision.observed`: source-disjoint validation, used for selection;
- `training_probe.decision.observed`: fixed training-window fit, diagnostic only;
- `validation.windows`: source/path, seam, per-window geometry/temporal/identity
  reasons and actual metric values.

If train fit and validation both remain poor, investigate the objective/capacity
before a long run. If train fit improves but held-out validation does not,
investigate generalization. Do not lower thresholds or use test data for tuning.

## Continue the same pilot, then train Diffusion

After reviewing the pilot:

```bash
export PY=/home/disk/lsm/conda_envs/edge/bin/python
cd /home/disk/lsm/storage/DunhuangDanceRAG
bash scripts/train_refiner_v5.sh resume \
  /home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915 \
  refiner_v5_trial1
```

Refiner continues to 8,000 total updates, preserving source-disjoint best-model
selection. Only accepted Refiner completion proceeds to 15,000 Diffusion updates.
Rejection returns nonzero and stops. After interruption, repeat `resume` with the
same tag/config/commit; matching snapshots are reused. A changed commit or config
requires a new tag and fresh training, not reuse of V4 state.

Logs stream to the console and `console.*.log` in the candidate directory.
Both accepted checkpoints are promoted to the original run's `checkpoints/`
directory with backups. This script does **not** generate a video or claim that
final motion quality gates have passed.
