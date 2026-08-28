# V8: validate the interpolation foundation before learning

This change repairs artificial joins, not original SMPL14 assets. Existing
retarget cache, train/val/test Event-DB, Router/Duration/Planner and generation
index stay read-only. No external pretrained model is introduced.

## Geometry and scientific boundaries

- A nonzero bridge has `(frames + 1) / fps` seconds between its anchors. Incoming
  XZ placement uses average observed endpoint velocity over that interval,
  with the existing speed cap. It no longer forces both roots to the same point.
  Zero-length joins retain hard-concatenation semantics. Height is not ramped.
- Root interpolation is quintic Hermite; joint interpolation is a quintic SO(3)
  log-chart curve with body angular velocity/acceleration boundary conditions.
  Derivatives use observed context, not hidden clean motion. This is a proposal,
  not a guarantee of low jerk or a uniquely correct trajectory. Incompatible
  endpoints can still be rejected.
- Analytic two-bone IK constrains only feet with low-speed, near-floor support
  at both anchors and compatible placements. Swing feet are untouched. It
  preserves root, upper body and external context and reports unreachable targets.
  IK is transactional: unreachable targets or increased endpoint/temporal error
  reject its candidate. The original bridge and explicit rollback reasons remain;
  a rollback is NOT a successful foot repair. Inspect each recipe's bridge report.
  This local bridge IK does NOT replace whole-song final IK or physical checks.
- Corruption, cross-event diagnostics and production generation use one bridge
  implementation. Formal failures propagate instead of silently falling back.
  Scheduler preflight also uses it; production seams cannot exceed training's
  supported length. Duplicate legacy interpolation implementations are removed.
- Quiet/active windows use relative loss normalization with a numerical `1e-6`
  floor (not the former `0.05` physical floor). Sampling balances single/cross
  events and short/long seams. Weighted endpoint/temporal/support gradients and
  pairwise cosines are logged separately from repair/clean conflicts.
- Relative event safety freezes support frames and segment anchors from the
  reference. Loss and audit share support statistics. Independent support audits
  stay in reports; final whole-song checks stay unchanged. A no-edit roundtrip
  experiment measures numerical error without widening tolerances. Nonzero
  natural acceleration is NOT proof of a defect.
- Direct-output optimization uses the exact decoder masks, smoothing and caps,
  observable objective and safety checks, but independent parameters per case.
  It fits probe cases too: an optimistic feasibility CONTROL, not generalization.
  A finite optimizer failing is not proof of impossibility. No control weights
  can be published or resumed as a learned model.
- Gains (3%), jerk guard (2%) and pass rates (75%) are not relaxed. Diagnostic
  short/long subgroups must pass individually, not just as a pooled mean. There
  is no blanket "already smooth => repaired" rule. If controls do not establish
  repair headroom, stop and inspect instead of running 8000 steps blindly.

Refiner: `product_manifold_boundary_refiner_v8`; Diffusion:
`reference_tangent_motion_diffusion_v4`; boundary protocol:
`observable_duration_c2_bridge_v2`. V7 reports/models/snapshots are incompatible.

## Zero-edit numerical contract fix

The retraction protocol is now `zero_centered_rot6d_action_v1`. Both NumPy and
Torch rotate the original 6D columns by an increment, rather than re-encoding
the reference even when the requested correction is zero. Zero output keeps
all motion values exactly equal while retaining tangent and reference gradients;
nonzero output still implements the same body-frame SO(3) action. This does not
sanitize invalid inputs or relax physical tolerances. The observable temporal
objective and its CPU audit now both use float64 from FK onwards; promoting
only already-rounded float32 joint positions left CPU/CUDA discrepancies at
the 2% jerk boundary. Gradients still flow to the float32 model on GPU.
Independent physical/fidelity audits and all acceptance thresholds are unchanged.

Foundation schema v4 records exact identity counts and per-case changed values
as well as FK errors. The 64-case no-edit check runs before direct optimization
and fails closed immediately if identity or physical non-regression fails.
Retraction and physical-auditor source hashes are included in the fingerprint.
Reports from the previous commit must not have their acceptance flags edited:
start a NEW tag and rerun the 200-step foundation control under the new code.
The nonzero decoder changed numerically too, so old direct-control pass counts
cannot simply be copied into a fresh report. Source/scheduler assets are reused.

## Peak-safe repair and direct-control contract

The repair branch now adds five per-window physical-registry constraints:
joint jerk P95, maximum, maximum sliding-window P95, and the corresponding
extremity P95/window P95. Mean seam jerk cannot hide a single-joint spike.
The budget is exactly `min(absolute_limit, reference * ratio + margin)` when
the reference is within the limit, otherwise the unchanged reference. The
existing numerical epsilon is retained. Penalties and gradients are zero
inside these budgets; natural motion is not continuously pushed toward zero.
No hidden clean trajectory or new pretrained model is used. GPU float64 FK
and differences supply the differentiable statistics, while the existing
independent physical auditor remains the final authority for candidate edits.

Direct optimization now requires BOTH non-increasing loss and the full
input-relative fixed-support physical/fidelity audit before accepting each
trial. Rejected trials retain the last safe candidate (initially no edit).
The reference never moves with the optimizer. Reference audits and bounded,
exact-content decisions are cached; retained outputs are freshly audited at
the end. Already-computed GPU jerk budgets prefilter known violations, and
physical rejections skip unnecessary geometry SVDs. Every surviving proposal
still receives the full audit. No-edit retention is NOT counted as successful repair. Thresholds,
decoder masks, smoothing and caps remain unchanged.

Reports include per-case safety rejection reasons, retained-no-edit flags,
and `observable.jerk_peak_diagnostic` with joint names and the four-frame
jerk stencil before/after editing. Frame indices are local to the diagnostic
window, zero-based, not absolute SMPL recording frames. Separate jerk-safety
gradient norms/conflicts are logged alongside endpoint/temporal/support terms.

Both foundation and neural diagnostic schemas are v4. Old reports/snapshots
must not authorize a new run: use a fresh tag and rerun foundation. Passing
safe direct optimization still does not demonstrate neural generalization.
Generation verifies the current code revision and byte-identical promotion
from this tag's accepted Refiner/Diffusion candidates; an older accepted V8
model with the same architecture name cannot silently substitute for them.
"Network diagnostic" below means a locally trained NEURAL network test, not
an Internet connectivity check or model download.

## Per-case descent and smooth target margins

The `4d9f255` server control kept all 64 candidates safe, but the new-position
10-frame cross-event group only passed temporal repair in 4/8 cases. Three of
its failures stopped at approximately 10% endpoint gain with no unsafe trials;
one retained the exact input with no unsafe trials. This is different from a
peak-jerk violation and does not establish a defect in the original recordings.

The optimizer and shared repair surrogate now address these concrete risks:

- Differentiate a SUM of independent case losses before per-case clipping and
  Adam, while logging the mean. A batch mean made the clipping/Adam epsilon
  depend on the number of unrelated cases in this direct control.
- Use a one-sided Huber shoulder for the same 10% relative training targets.
  The derivative tends continuously to zero at the target instead of changing
  abruptly at a linear-hinge cusp. The loss remains exactly zero after reaching
  the target and keeps a bounded linear slope for large violations. The 3%
  evaluation thresholds, 2% jerk guard and 75% pass rates are UNCHANGED.
- Check that Adam's proposed direction is downhill; retry exhausted searches
  with the current steepest-descent direction. Up to 24 backtracks evaluate
  only pending cases, reuse the previous accepted step scale, and distinguish
  loss rejection from storage-resolution no-ops. Only strictly lower loss AND
  the unchanged full physical/fidelity audit can accept an actual motion edit.
- Stop updating a case only after BOTH original 10% training targets and the
  differentiable safety/trust constraints are satisfied at an already-audited
  state. Other cases retain their full budget. The 200 steps are a per-case
  maximum, not a requirement to waste trials after reaching the targets. The
  report records actual `attempted_optimizer_steps` and `target_satisfied`;
  evaluation pass/fail labels are not used for optimizer stopping.
- Three consecutive iterations with no accepted update after both available
  search directions mark a case `search_stalled` and stop that finite search.
  This is NOT convergence or repair acceptance. Its retained candidate still
  has to pass every unchanged metric, and a failed subgroup still blocks fitting.
- Match REPAIR support-loss budgets to the auditor's ratio PLUS margin, ceiling and
  numerical epsilon. The old max(ratio, margin) surrogate was stricter than the
  repair acceptance rule. The clean-identity branch keeps its separate original
  budget; it must not inherit the repair-stage allowance. Neither audit changes.

`decoder` records `loss_rejected_trial_count`, `resolution_limited_trial_count`,
`non_descent_adam_steps`, `gradient_fallback_attempts/updates`, `search_stalled`, and last pre-update gradient
norm alongside the safety counters. Training logs keep the smooth losses and
the raw `endpoint_relative_gap`/`temporal_relative_gap` separately. A zero edit
or equal objective value is no longer counted as an accepted optimization step.
New objective/optimizer protocol identifiers and source fingerprints prevent
v3 reports from authorizing fitting under the changed optimization protocol.

The direct control is still finite, independent per case, and non-neural. Its
failure is not a proof of impossibility; its success is not generalization.
Start a new tag; do not reuse the peak-safe trial report or an older snapshot.

## Foreground server workflow (no tmux)

After pulling the reviewed release on main, initialize the current shell:

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
export PY=/home/disk/lsm/conda_envs/edge/bin/python
export OUT_ROOT=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915
export TAG=refiner_v8_descent_trial1
git rev-parse HEAD
# Set EXPECTED_COMMIT to the literal reviewed SHA from the release message.
```

All stages stream progress and save logs. Each requires clean Git, an explicit
matching EXPECTED_COMMIT, CUDA, existing train/val databases, no competing
process, and a candidate-directory lock. Never reuse an old tag.

### 1. Foundation control, no neural training

```bash
bash scripts/train_refiner_v8.sh foundation "$OUT_ROOT" "$TAG"
```

Eight fixed TRAIN windows, widths 10/28, seen/new positions. Compare pure
interpolation with contact IK, measure no-edit roundtrip, then optimize each
case directly for at most 200 steps. Inspect `interpolation_vs_ik`, `roundtrip`, `direct`,
`decoder`, and `decision` in:
`$OUT_ROOT/checkpoints/$TAG/foundation_diagnostic/foundation_report.json`.

Exit 2 means completed but not ready, not automatically a crash. Check the
first traceback if present. Do not continue after a failed gate.
The first `bridge_zero_edit_preflight` line must show `cases=64`,
`exact_identity_count=64`, `rejected_count=0`, `max_fk_roundtrip_m=0`.
That line alone does NOT authorize fitting: the final decision must still have
`ready_for_network_diagnostic=true` after the direct control.

### 2. Only after foundation passes: 400-step neural-network diagnostic

```bash
bash scripts/train_refiner_v8.sh diagnose "$OUT_ROOT" "$TAG"
```

Fits only seen TRAIN positions; new positions are held out from fitting. Fresh
initialization, balanced four-group batches, no direct-control warm start.
Inspect `bridge_diagnostic/diagnostic_report.json` and `gradients.jsonl` in the
candidate directory. Fixed final step decides readiness, not best probes.

### 3. Only after diagnosis passes: fresh source-disjoint pilot

```bash
bash scripts/train_refiner_v8.sh pilot "$OUT_ROOT" "$TAG"
```

1000/8000 updates. Inspect `boundary_refiner.validation_step_001000.json`.
The independent val set is not used by the two TRAIN-window diagnostics.
Best accepted candidate and recovery snapshots stay isolated. No formal
model is overwritten at this stage.

### 4. Only after pilot passes: Refiner resume, then Diffusion

```bash
bash scripts/train_refiner_v8.sh resume "$OUT_ROOT" "$TAG"
```

Refiner to 8000, Diffusion to 15000; snapshots every 200 steps. Refiner validation
every 1000 with best accepted checkpoint selection. Diffusion cannot follow a
rejected Refiner. Both candidates must pass before either replaces formal
models; previous files are backed up. Rejected files never replace formal ones.

### 5. Only after model acceptance: full generation using existing assets

```bash
bash scripts/train_refiner_v8.sh generate "$OUT_ROOT" "$TAG"
```

Default audio: `assets/music/test/audio/dunhuangwu2.wav`. All five retrain flags
and both cache/Event-DB rebuild flags are 0. Full closed-loop, final IK, strict
physical gate and video checks still run. Expected output:
`$OUT_ROOT/results/video.mp4`. Verify current timestamp, final gate and playback.

Stop at any failed stage and preserve its reports. No diagnostic alone establishes
final dance quality or justifies bypassing a gate.
