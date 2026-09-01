# Fresh exact-zero Refiner trajectory diagnostic

## Scientific question

The frozen V15.4.1 parameter audit shows a nonzero output head at step 400,
nonzero trunk gradients, and a head-dominated parameter-gradient snapshot. It
does not reconstruct accepted updates over training and cannot establish
permanent trunk starvation. The paired safe-start preflight adds a separate
result: A0 passed all 2,976 TRAIN checks, while Gaussian `sigma=1e-5`, seed 42
failed 217 checks across all 93 banks. Its reasons were joint/extremity jerk,
support drift and penetration, concentrated in `cross_short`.

This change does not shrink sigma, tune a seed or relax safety. It asks one
descriptive question: after a fresh, exact-zero output head blocks the true
step-1 trunk gradient, how do true gradients, checked AdamW updates, cumulative
movement and final displacement propagate during 400 unchanged TRAIN steps?

This is a training diagnostic. It is not a read-only audit, Pilot, formal
training run, production change, matched replication of historical V15.4.1 or
checkpoint promotion.

## Fixed contract

- One arm: `A0_zero`; `output_init_std=0.0` exactly.
- Fresh CPU initialization from the recorded config seed; there is no CLI seed
  override. Historical model
  weights are loaded for artifact provenance but never copied into A0.
- Existing frozen anchor, all 92 unique context banks and deterministic C5
  schedule; 192 cases/step and 48 cases/group.
- Exhaustive initial safety over 2,976 checks before optimizer construction.
  Any rejection returns status 2 with zero optimizer steps and no probe load.
- Exactly 400 checked steps, no public step-count option, resume, early stop,
  checkpoint selection, setting retry or budget extension.
- Unchanged decoder, confidence, masks, taper, smoothing, caps, retraction,
  objective, CVaR, clean weight, AdamW LR/decay, clipping, Armijo search,
  subgroup guard and `soft_confidence_true_chain_rule_v2` autograd.
- The original probe is loaded only after the step-400 state and checkpoint
  hashes are fixed. Its descriptor, hash, config, fingerprint, roles, widths
  and `updates_forbidden` contract use the existing fail-closed loader.
- Only the fixed step-400 state receives the existing `seen` and
  `new_position` four-group evaluation, eight cases/group.

The V15.4.1 artifact omitted decoder strengths. The runner supplies core=0.02
and transition=1.0 as explicit legacy values. They are not described as
independently recovered source metadata.

## Measurements

Every line of `updates.jsonl` records:

- pre-update repair, clean and exact weighted total objectives plus all named
  group terms;
- accepted/rolled-back result, Armijo and subgroup guard fields, trial counts,
  rescue status and losses;
- unclipped trunk, head and total gradient norms/RMS and their ratio;
- actual post-check/rollback updates, displacement from initialization, and
  true pre-update gradient dot actual update;
- post-step output-head weight/bias norms, RMS, maximum, exact-zero flag and
  spectral norm;
- contact/root/joint head block parameter and actual update norms;
- parameter, pre-clipping gradient, actual update and displacement statistics
  for `in_proj`, convolution layers `net.0/3/6` and normalization layers
  `net.1/4/7`.

AdamW decay is part of the measured update. A nonzero actual trunk movement
with zero task gradient must not be called gradient propagation; the logged dot
product and raw gradient distinguish these cases.

Detailed create-only JSON snapshots are written at steps
`0,1,2,3,5,10,25,50,100,200,300,400`. Step 0 is initialization. Later details
are measured read-only at the post-checked-step state on that step's named TRAIN
transaction. They include every parameter's norm, ordinary autograd gradient,
actual retained update, displacement, head VJP identity and decoder tangent RMS
at raw/mask/smoothing/taper/applied stages. This checkpoint-state gradient is
explicitly separate from the pre-update gradient that generated the step.

The final report computes, by exact floating-point comparison without an
epsilon threshold:

- first accepted step where `out.weight` is nonzero;
- first nonzero trunk gradient and actual trunk update steps;
- early trunk gradients and fixed-step head/trunk gradient, update and
  displacement ratios;
- cumulative retained trunk/head update path lengths;
- final trunk/head displacement from initialization;
- accepted, retained and rolled-back counts/rates. The older optimizer helper's
  `retained_steps` counter means rejected steps restored by rollback; the new
  report exposes unambiguous names.

Path length and final displacement are intentionally separate. Neither one is
automatically a scientific success metric.

## Artifact and interpretation boundary

The output directory is create-only and source hashes are checked before and
after training. It contains `experiment.json`, `report.json`,
`diagnostic_initial.pt`, `diagnostic_latest.pt`, `updates.jsonl`, and the fixed
`snapshots/step_*.json` set. Exceptions produce `completed=false` with type and
message; existing partial logs remain, but there is no resume path.

Both diagnostic checkpoints use a unique schema/version and set
`formal_checkpoint=false`, `resume_allowed=false`, `publish_allowed=false`, and
`pilot_allowed=false`. Formal inference rejects their version; formal resume
rejects their schema.

Top-level `scientific_acceptance`, `publish_allowed` and `pilot_allowed` stay
false even if final diagnostic gates pass. The report also fixes
`historical_comparison_is_descriptive_only=true`. Results must not be phrased as
“A0 beats historical V15.4.1,” “zero initialization solved/failed,” “starvation
proved/eliminated,” or “production Refiner accepted.”

## Server execution

Run the reviewed commit from a clean `main` in the existing GPU environment.
The wrapper performs the complete bounded diagnostic and does not start Pilot:

```bash
export PY=/home/disk/lsm/conda_envs/edge/bin/python
export EXPECTED_COMMIT=<reviewed-full-commit-sha>
bash scripts/diagnose_refiner_zero_start_trajectory.sh \
  "$SOURCE" "$RUN_DIR/trajectory"
```

Inspect `trajectory/report.json`, `updates.jsonl`, fixed detailed snapshots and
the console log together. Process exit 0 means execution completed; it does not
mean the scientific hypothesis passed.

## Validation boundary

Regression tests cover exact fresh initialization and RNG restoration, the
fixed schedule/case counts, step-0 true VJP, real checked CPU/CUDA updates,
rollback identity, path/displacement separation, preflight and source mutation
stops, probe ordering, provenance-only source weights, final artifact isolation,
probe mismatch, and a finite-difference true-gradient check. Local synthetic
tests do not establish the 400-step server trajectory or final motion result.

The reviewed local checkout passed **14 trajectory tests** (including available
CUDA), **109 related Refiner tests**, and the complete **770-test suite**. The
full suite emitted one existing PyTorch Transformer nested-tensor warning. The
CLI help check, Python compilation, shell `bash -n`, and `git diff --check` also
passed. Logs remain outside tracked source under
`output/audits/zero_start_trajectory/`.
