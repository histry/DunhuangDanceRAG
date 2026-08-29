# V8 foundation / V9 local Refiner: validate before learning

This change repairs artificial joins, not original SMPL14 assets. Existing
retarget cache, train/val/test Event-DB, Router/Duration/Planner and generation
index stay read-only. No external pretrained model is introduced.

## Latest repair: smooth safety boundaries and fresh search curvature

The uploaded `02e449a` / `refiner_v9_armijo_20260828_223139` run passed all
eight foundation groups, then stopped at attempt 382: 381 accepted updates,
671 trial evaluations and no accepted non-descent update. Fitted positions
passed 30/32 temporal cases; unfitted local contexts passed only 10/32. This
is an optimization stall plus unverified generalization, not a damaged-SMPL
finding and not permission to train a pilot or Diffusion.

The uploaded `fit_bank.pt` and `diagnostic_state.pt` enabled an exact TRAIN-input
replay. The repair support surrogate still used a linear hinge divided by 1e-4
on already-over-limit references. Its loss was about 2.19e-6 but the total
gradient norm was 25.43. A zero-slope quadratic shoulder reduced that contribution
to about 3.08e-10 and the total norm to 0.388; a same-bank directional probe then
decreased the objective. A separate controlled seam-bank probe exposed the same
problem in the jerk-tail hinge: a roughly 6.25e-6 safety loss produced a total
gradient norm around 2200. Merely adjusting the search floor did not resolve it.

Corrections in this release:

- Support, jerk peak/window quantiles and root-vertical safety now share one
  one-sided quadratic shoulder. The metric budget, absolute ceiling, ratio,
  margin and numerical epsilon are UNCHANGED. Inside the permitted region,
  both loss and gradient are zero; larger violations still incur a bounded
  linear penalty. The existing stage margin sets normalization, not tolerance.
  Parenthesized subtraction preserves tiny quadratic values without cancellation.
- Every deterministic optimizer update re-estimates curvature from the full Adam
  proposal. The previous accepted scale is provenance, not a permanent upper
  bound for the next step. At most 12 trials per direction, strict Armijo decrease,
  finite checks and full parameter/optimizer rollback remain enforced. Useful
  decrease is `max(1e-8, 8*loss-dtype-epsilon)` relative to loss magnitude; a
  representable 2e-8 relative decrease is no longer rejected, while a 1e-13
  roundoff-sized decrease cannot count as progress. This is NOT a physical gate.
- The report explicitly identifies `new_position` as an unfitted LOCAL MOTION
  CONTEXT within the same TRAIN windows: moving the cut changes its content as
  well as its position. It is neither a pure translation-equivariance test nor
  independent source-disjoint validation. The original 32-case fit bank and
  all per-role/per-width acceptance gates remain unchanged.

Two-position fitting and an input variant excluding absolute poses/music
conditions were tested locally but did not establish generalization. A 400-step
bracketed-position variant completed with 400 descending updates, yet passed
only 8/32 new-context temporal cases (single-recording 0/16). Those experimental
architecture/sampling changes were therefore NOT adopted. Model architecture
remains V9; do not attribute the variant's numbers to the released model.

A fresh local V9 control using the uploaded exact 32-case TRAIN tensor bank
completed 400/400 descending updates (880 trial evaluations, zero nonfinite
trials), reducing loss from 0.0625 to 0.00155109. Fitted temporal cases passed
28/32; reconstructed unfitted local contexts passed 9/32: single/10 0/8,
single/28 0/8, cross/10 7/8, cross/28 2/8. Clean identity passed 16/16 in both
splits. The fit inputs/masks were exact uploaded tensors; the probe was rebuilt
from those uploaded clean TRAIN windows on the local GPU, not claimed to be
bitwise identical to the server's unsaved probe tensors. This verifies removal
of the observed optimization stall, NOT improved generalization or scientific
acceptance. The new-context gate correctly remains closed.

Diagnostic schema is v9, update protocol `same_batch_fresh_curvature_armijo_v4`,
repair safety `stage_registry_smooth_tail_support_root_v4`. Use a NEW
`refiner_v9_smooth_safety_*` tag, regenerate foundation, then rerun the gated
400-step diagnostic. Old reports/snapshots cannot authorize current training.
Source/scheduler assets remain reusable; no external pretrained model is added.

## Previous repair: curvature-aware search, vertical safety, portable diagnostics

The full `055d85e` server report is now available. Foundation passed all eight
groups, but the network stopped at attempt 19 (18 accepted, 84 trial forwards).
All 64 temporal cases failed. This is NOT a completed 400-step experiment or
evidence of damaged original SMPL. Its loss dropped from 0.0625 to 0.0347613,
while the temporal loss increased from 0.0500 to 0.0558140. At step 1 the
weighted endpoint/temporal gradient norms were 139.22/4.59 and cosine -0.421;
step 19 lacked component gradients. One sampled conflict cannot establish the
complete cause of the neural generalization failure.

Three concrete issues are corrected without relaxing acceptance:

- The preceding 2^-11 step floor was not scale-invariant. A high-curvature
  quadratic regression test shows it can forbid representable steps that
  remove nearly all loss. Replace blind halving/floor exclusion with safeguarded
  quadratic interpolation and an Armijo sufficient-decrease check. There are
  still at most 12 trials per direction (24 total), finite-state checks and
  transactional rollback. Clipping is explicitly undone for the directional
  derivative calculation, not for the Adam proposal. Loss decrease must exceed
  both the Armijo amount and max(1e-7, 8*loss-dtype-epsilon) relative to initial
  loss magnitude. A 1e-13 numerical gain cannot count as useful progress.
  This is an optimizer tolerance, NOT a change to physical quality thresholds.
- Repair loss previously guarded jerk tails and horizontal foot support but
  omitted root vertical range/speed. The uploaded cross-event case 12 has
  root-speed P95 1.464076 -> 1.466682 m/s (seen) and 1.465710 -> 1.468365 (new).
  Both references exceed the 1.25 m/s absolute ceiling, so the unchanged stage
  policy allows identity but no additional budget beyond numerical tolerance.
  Add per-case root robust range, vertical P95 and maximum excess losses using
  the same stage registry, ratio-plus-margin, ceiling and epsilon. These losses
  and gradients are zero inside the permitted region. Their statistics match
  the independent float32 root-difference auditor. Final audits are unchanged.
  A linear penalty divided by 1e-4 on zero-headroom inputs was rejected after
  a local control stalled at step 5. The adopted one-sided quadratic shoulder
  has zero slope at the boundary; the existing policy margin sets only its
  normalization scale and NEVER enlarges the permitted metric value.
- At an unlogged stall, recompute endpoint/temporal/support/jerk/root gradient
  diagnostics on the retained model and SAME TRAIN bank. Save every update's
  trials in `optimizer_updates.jsonl`. Save `fit_bank.pt` (only the exact seen
  TRAIN inputs, masks, descriptors and provenance) plus rolling
  `diagnostic_state.pt` (model + optimizer state, including search scale).
  Local Event-DB files did not match the eight uploaded window hashes; same
  filenames cannot justify replay claims. These artifacts are explicitly
  diagnostic-only and cannot be used as formal training/resume checkpoints.

The optimizer-only local control (before adding the vertical-loss term) reached
392 attempts/391 accepted, loss 0.00302907, versus the preceding local control's
143/142 and 0.00401173. Both use eight local TRAIN windows, NOT the exact server
events. At 392, seen single-recording temporal passed 16/16 but new-position
single-recording remained 0/16; new cross-event passed 4/16. This supports an
optimization correction, NOT repaired generalization or authorization to train.
No old model, safety threshold or original asset is overwritten by this audit.

The final integrated local smoke control (80-step budget, new root constraint)
stopped at attempt 77/76 accepted, loss 0.01308330. Seen single/short temporal
passed 5/8; all other temporal groups failed, including new-position 0/32.
Vertical-speed rejections were absent in this smoke result, but foot safety
and learned repair still failed. It is neither a successful 400-step diagnostic
nor an independent validation result. The extra root protection is an objective
alignment fix, NOT evidence of increased overall repair quality.

Diagnostic schema is v8, update protocol `same_batch_armijo_quadratic_v3`,
repair safety `stage_registry_jerk_root_constraints_v2`. Model architecture
remains V9. Use a new `refiner_v9_armijo_*` tag and regenerate foundation under
the current objective. An early stop or any failed subgroup still blocks pilot,
full training and Diffusion. No external pretrained model is downloaded.
Refiner resume fingerprints now include input, objective and safety protocols
as well as optimizer protocol; changing a loss cannot silently resume old
training moments simply because the model tensor shapes remain V9-compatible.

## Previous diagnostic: full-bank fitting and bounded step scale

The server's `5fad394` run completed the control and neural diagnostic normally.
All 64 zero-edit cases were exact identities; the eight foundation groups passed.
All 400 neural updates decreased their CURRENT minibatch loss (491 forward
trials), but temporal repair was only 3/16 for seen single-recording cases and
0/16 for each other split/role. Clean identity and physical non-regression
passed. Neither the previous Adam-overshoot fix nor those safety passes prove
learned temporal repair. Original SMPL files are not implicated by this report.

The following diagnostic/optimization issues are addressed:

- The small fixed-bank diagnostic previously drew only 8 of its 32 seen cases
  each step. A decrease on that subset does not bound the loss on the other 24.
  It now uses ALL 32 seen TRAIN cases for both gradients and trial evaluation,
  with equal role/width coverage. New-position probes and independent validation
  remain excluded. The formal random-window training sampler is unchanged.
  This is a changed diagnostic protocol: 400 full-bank steps expose four times
  as many cases as the old 400 minibatch steps, not an equal-compute comparison.
- The persistent line-search scale previously could halve across successive
  steps without a global lower bound. A local full-bank control reached scale
  9.31e-10 and counted an approximately 1e-13 loss decrease as progress at step
  150. The 12-trial grid is now bounded below by 2^-11. Hitting this floor tries
  the current-gradient direction; failure restores the prior model/optimizer.
  It does not loosen any motion-quality threshold or force a harmful update.
  The unbounded local control was stopped after observing this plateau; it
  did not produce a completed 400-step result.
- If BOTH directions fail on the fixed bank (or its gradient is zero), diagnosis
  stops, evaluates and saves the retained model, and returns nonzero before the
  400-step budget. No new sample or optimizer state would justify repeatedly
  trying the same failed search. Actual `completed_steps`, `stopped_early` and
  `termination_reason` are reported; an early stop cannot authorize pilot.
  Formal random-minibatch training does NOT stop on a single retained update.

The bounded local full-bank control stopped at attempt 143 (142 accepted
updates), with loss 0.0625 -> 0.00401173. On its eight local TRAIN windows,
seen single-recording temporal repair passed 16/16, but new-position
single-recording repair passed 0/16; cross-event temporal repair passed 4/16
seen and 3/16 new-position. Some cross-event physical checks also failed.
This is a local optimization control, NOT the exact server case set, a completed
400-step diagnostic, independent validation, or permission to train a pilot.
It supports improved fixed-bank fitting, not repaired generalization.

The model remains V9. A separate local FK-derivative-input control improved
new-position short cross-event temporal repair from 0/8 to 3/8, but failed the
long-seam groups; it is NOT adopted here as a validated architectural solution.
Neural generalization remains unverified. Do not infer successful training from
loss descent or run 8000 steps directly after pulling this change.

That revision used neural diagnostic schema v7, fitting protocol
`complete_seen_bank_descent_v1`, and update protocol
`same_batch_descent_backtracking_v2_bounded_scale` (superseded above).
Reports/optimizer snapshots from the earlier protocol cannot authorize or resume
a fresh run. Use a new `refiner_v9_fullbank_*` tag. All source/scheduler assets
remain reusable and no external model is downloaded.

## Previous repair: checked Refiner parameter updates

The `a12f877` server run passed the foundation control but failed the 400-step
neural gate: seen single-recording temporal repair was 4/16, and the other three
split/role temporal totals were 0/16. All 64 physical non-regression checks
passed, clean identity passed, and no decoder caps were active. This is a
learning/generalization failure, not a program crash or evidence of corrupt SMPL.

A local eight-TRAIN-window audit confirmed a separate optimization problem:
with the same initialization/minibatches, the first unchecked Adam step raised
the actual same-batch objective from 0.0625 to 5.6164264 despite a downhill
first-order direction and gradient clipping. Ten of the first 25 updates raised
the same-batch loss. This local database is NOT the server's exact event set,
and this mechanism is NOT established as the sole cause of its failed gate.

That revision introduced `same_batch_descent_backtracking_v1` for both the
400-step diagnostic and formal Refiner (superseded by bounded-scale v2 above):

- Form the Adam proposal, then evaluate the SAME fixed minibatch after the
  proposed update. The trial never regenerates corruption or reads probe/val
  data. Only a finite, strictly smaller total objective accepts an update.
- Try at most 12 step scales per direction. If Adam momentum is uphill or its
  search fails, try the current gradient; accepted rescue clears stale moments.
  If both searches fail, retain parameters AND the complete optimizer state.
  Exceptions also roll back. Zero gradients do not silently apply weight decay.
- Persist the adaptive trial scale in the optimizer snapshot. The training
  configuration fingerprint rejects snapshots from the old unchecked protocol.
  Model V9, inputs, targets, masks, caps, safety thresholds and Diffusion updates
  are unchanged. Use a new tag, not an old training snapshot.
- Log `optimizer_update.loss_before/loss_after`, `step_scale`, `direction` and
  `trial_evaluations`. `optimizer_updates` counts EVERY step, including those
  between printed samples. `accepted_non_descent_steps` must remain zero;
  `retained_steps` are attempted steps with no accepted parameter update.
  Formal training counters explicitly cover the current invocation from
  `start_step`, not unrecorded updates from before a resume.

Backtracking costs extra forward evaluations (at most 24 per attempted step);
it is bounded, not a speed-up guarantee. In the local 400-step mechanism check,
the paired unchecked run increased same-batch loss in 42/400 steps; all 400
checked updates reduced it with 499 trial evaluations. Both runs used the same
eight local TRAIN windows, initialization, minibatches and fixed 400-step budget.
New-position temporal repair was 0/32 in BOTH arms; seen single/10 passed 2/8
unchecked versus 3/8 checked, with the other temporal subgroups at zero.
Neither arm meets the diagnostic gate. Thus this fixes blind
overshooting, NOT the unresolved generalization problem. A decreasing weighted
minibatch loss does not guarantee each window, temporal component or physical
metric improves. Independent validation and the original role/width gates are
still required; never use this optimizer acceptance as scientific acceptance.

Foundation schema remains v4; that neural diagnostic schema was v6 and recorded
the optimizer source hash/protocol. Old reports cannot authorize a fresh pilot.
Rerun the bounded foundation/400-step checks before considering more training.

## Previous V9 input-locality repair

The server's `bbb2aaf` run passed all eight foundation groups, then completed
400 neural steps and correctly STOPPED. Single-recording temporal repair was
6/16 at seen positions and 0/16 at new positions; cross-event temporal repair
was 0/16 in both splits. All 17 sampled clean-loss gradients were zero. This is
neither an interpolation-control failure nor evidence of damaged SMPL assets.
It also does not establish one unique cause of the neural generalization gap.

Two reproducible architecture defects are corrected in V9:

- `GroupNorm` on `[B,C,T]` used statistics over the entire time window, so remote
  frames changed the same seam's output. Per-frame channel LayerNorm removes
  this dependence. The convolutional field is 33 frames; the local horizontal
  difference reads one preceding frame, and boundary conditioning explicitly
  reads the supplied external anchors. No claim of a purely 33-frame total
  dependency is made.
- Raw world X/Z made correction depend on arbitrary route placement. The neural
  input now contains local horizontal displacement in metres/frame instead.
  Height, joint rotations and observed boundary features remain available;
  the original motion is unchanged as the geometric decoder reference.

CPU/CUDA tests use NONZERO heads, translations, remote-frame perturbations,
consistent crops and backpropagation. A zero head alone is not an invariance
test. Model version and input contract reject V8 checkpoints/snapshots.
The physical objectives, interpolation, masks, smoothing, caps, clean tolerance,
3% improvement gates and 75% pass rates are unchanged. These structural fixes
are NOT a claim that a new 400-step experiment will pass.

A local 400-step two-TRAIN-window probe (eight fitted role/width cases, fixed
initialization and objectives) still failed to improve temporal energy on the
four held-out single-recording seam positions, for both V8 and V9. It is a
mechanism check, not the server's eight-window experiment or an independent
validation result. Thus generalization remains unresolved; do not proceed
directly to pilot, 8000-step training or Diffusion after pulling this repair.

`bridge_diagnostic/summary.json` and `bridge_failure_breakdown` console rows now
separate temporal-energy improvement from jerk non-regression, for every
role/width/split, and include raw/applied tangent RMS, mask and cap statistics.
The complete report retains individual cases and all original safety checks.
If diagnosis fails, supply this new summary together with the console log;
do not substitute a historical same-name JSON from another revision.

The script name `scripts/train_refiner_v8.sh` is intentionally retained, with
the dependency order unchanged. Use a new `refiner_v9_armijo_*` tag. Source
assets are reusable; old foundation/diagnostic authorizations and trained
Refiner weights are not silently migrated across source fingerprints.

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

Refiner: `product_manifold_boundary_refiner_v9`; Diffusion:
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

Foundation is v4; the latest neural diagnostic is v8. Old reports/snapshots
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
export TAG=refiner_v9_smooth_safety_trial1
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

Fits the complete seen TRAIN bank; new local contexts are held out from fitting.
The report key `new_position` retains its historical name, but moving the cut
also changes local motion content. It is not a pure tensor-shift test.
Fresh initialization, all four groups in each update, no direct-control warm start.
Inspect `bridge_diagnostic/diagnostic_report.json` and `gradients.jsonl` in the
candidate directory, plus `summary.json` for all-step optimizer accounting and
role/width failure counts. Fixed final step decides readiness, not best probes.
An early fixed-bank search stall saves the actual step and rejects continuation.

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
