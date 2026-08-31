# True-gradient safe-start: paired initialization diagnostic

## Decision

Implement a **single-variable initialization experiment**, not four heads, a
new loss, gradient surgery or a production-default replacement:

| Arm | Output weights | Bias | Trunk | Optimizer, data, budget |
|---|---|---|---|---|
| A0_zero | exact zero | zero | same fresh tensors | unchanged, 400 steps |
| A1_gaussian | Normal(0, 1e-5 squared) | zero | same fresh tensors | unchanged, 400 steps |

The standard deviation `1e-5` is a preregistered conservative candidate, **not an
optimum established by the frozen audit**. There is no sigma sweep, automatic
shrinking, seed search, LR adjustment or longer retry in the runner. Both arms
use the source configuration's seed unless an explicit seed is recorded in the
CLI. A fresh paired A0 is necessary: the failed historical 400-step model is not
an appropriate initial state or a matched random-initialization control. We do
not claim to reconstruct the historical run's initialization RNG sequence.

The question is whether safe initialization affects upstream adaptation and
observable boundary repair under a fixed small-data compute budget. Initialization
alone is not presented as a new paper contribution. Single-seed TRAIN-window
evidence will not establish generalization, causality over training histories,
or a publishable result; multiple paired seeds and source-disjoint validation
remain later requirements.

## Why this intervention fits this code

The refiner is a randomly initialized three-convolution temporal stack with
framewise normalization and one residual addition. It is not a pretrained
diffusion backbone or a thousand-layer unnormalized residual network. The
supplied final-state audit shows head-dominated **parameter** gradients, mostly
nonnegative group cosines, and no measured history of accepted layer updates.
It motivates an initialization test, not an established starvation mechanism.

For `z = W h + b`, zero W blocks the initial `dL/dh = W^T dL/dz`. Small nonzero
W allows true gradients into the trunk without altering the decoder, objectives
or backward rules. It does **not guarantee** a nonzero derivative for every
objective/input, nor that every decoded motion is safe. Nonzero initialization
is approximately, not exactly, identity; the explicit preflight below is required.

Alternative choices were checked against primary literature:

- [ControlNet](https://arxiv.org/abs/2302.05543) couples zero convolutions with a
  large pretrained backbone. Our randomly initialized trunk lacks that starting
  representation, so directly borrowing its training dynamics is unwarranted.
- [ReZero](https://arxiv.org/abs/2003.04887) uses zero-initialized residual gates.
  For an output gate `alpha*P(h)`, alpha=0 still zeros the initial trunk gradient.
  It adds another experimental variable without directly removing that barrier.
- [Fixup](https://arxiv.org/abs/1901.09321) designs initialization for residual
  networks without normalization. Replacing our normalization/scale system would
  change several factors simultaneously and obscure this first comparison.
- [AdaLN-Gaussian](https://arxiv.org/abs/2608.09438) studies Gaussian initialization
  of DiT conditioning; the August 2026 arXiv version reports image-generation
  experiments. It supports studying initialization as a factor, not transplanting
  its result or its scale into this motion refiner as a proven remedy.

Layer-wise learning rates are deferred until **actual accepted updates** are
measured. PCGrad is not intrinsically incompatible with a true-gradient descent
test if proposal and derivative are kept separate; there is simply insufficient
conflict evidence to prioritize it here. Auxiliary supervision would change the
scientific objective and the role of unavailable cross-event clean targets.

## Implementation and safety contract

`ProductManifoldTemporalRefiner(output_init_std=0.0)` preserves the existing
default, state-dictionary layout and RNG draws. The candidate is used only by
`training.refiner_safe_start_diagnostics`. No formal-training environment flag
is added. Existing train/inference paths continue using the zero default;
loading an existing checkpoint still replaces all its parameters as before.

The paired diagnostic:

1. Verifies the completed, unpublished V15.4.1 source from `6e73e0e`, its
   report/state/TRAIN fingerprints, checksums and C5 reservoir schedule. Loads
   source model bytes for provenance only, never as initialization or optimizer
   state. Restores recorded physical policies; does not apply caller overrides.
2. Constructs fresh CPU initial states with exactly identical trunk tensors and
   zero biases. Records config, seed, code/dependency hashes, source hashes,
   schedule and runtime in `experiment.json`; tensor hashes go into `report.json`
   and exact initial states into each arm's `diagnostic_initial.pt`.
3. Evaluates **every case of every unique TRAIN bank** for each initial model.
   Uses the existing fixed-support physical non-regression, reference geometry
   budget and clean-identity gate. No endpoint/temporal improvement is demanded
   at initialization. If any case fails, neither arm trains and probe stays closed.
4. Fits each arm for exactly 400 checked steps, 192 cases/step, 48/group. Both
   use the same anchor plus the same rotating five contexts, same AdamW LR and
   decay, same true autograd, clipping/unscale, Armijo test and subgroup guards.
   Only the current transaction is materialized; closure and gradient consume
   that exact batch. There is no checkpoint resume or early-stop selection.
5. Logs every pre-clipping parameter gradient, actual post-acceptance/rollback
   update, displacement from initialization and true gradient dot actual update.
   Statistics include weight decay: nonzero trunk movement is not alone proof
   of task learning. Losses and group terms are explicitly **pre-update**.
   The initial state is preserved separately; `diagnostic_latest.pt` is updated
   every 25 steps (also step 1 and the final step).
6. Only after both arms finish does it open the original `probe_bank.pt`, verify
   its descriptor/hash/config/roles/widths, and evaluate the two fixed final
   states on seen and held-out local positions using the existing gates. All
   four eight-case role/width groups must be present. Final checkpoint hashes,
   arm, experiment identity and step are rechecked. No probe affects updates,
   sigma, seed, stopping, or a within-run best-checkpoint choice.
7. Retains `pilot_allowed=false`, `publish_allowed=false`, and top-level
   `scientific_acceptance=false` even if an arm passes TRAIN diagnostic gates.
   New snapshot versions cannot load through formal inference or resume loaders.
   Source files and existing outputs are never overwritten.

The legacy source omitted regional decoder strengths. The wrapper explicitly
uses core=0.02 and transition=1.0 as in preceding audits and labels that limitation.
These are not independently recovered historical overrides. A mismatch in the
actual original settings requires correcting the explicit inputs, not ignoring it.

## Server command

Set `EXPECTED_COMMIT` to the full reviewed commit after a clean fast-forward of
main. Run in the existing edge environment, preferably within a tmux session.
This performs a bounded **paired diagnostic: up to 800 optimizer attempts total**,
not a read-only audit and not a Pilot. Initial safety failure returns status 2;
do not automatically lower sigma, remove checks or extend the budget.

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail
export PY=/home/disk/lsm/conda_envs/edge/bin/python
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
ROOT=outputs/run_smpl14_formal_20260822_163915
SOURCE="$ROOT/checkpoints/refiner_v15_4_1_lazy_reservoir_foundation_20260831_235832/bridge_diagnostic"
mkdir -p "$ROOT/audits"
RUN_DIR="$(mktemp -d "$ROOT/audits/safe_start_pair_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
bash scripts/diagnose_refiner_safe_start.sh "$SOURCE" "$RUN_DIR/paired" \
  2>&1 | tee "$RUN_DIR/console.log"
```

Use module flag `--preflight-only` for an explicitly zero-update safety check;
the default wrapper performs the full paired diagnostic if preflight passes.
The 400-step budget and sigma are fixed in code, not CLI tuning knobs.

Return `paired/report.json` and the console result rows. For optimization
history, each arm has `updates.jsonl`, `diagnostic_initial.pt` and
`diagnostic_latest.pt`. If interrupted, the report is marked incomplete and logs
are preserved; there is deliberately no resume path that silently changes the
preregistered comparison. Do not confuse process exit 0 with scientific success.

Promising evidence requires initial safety, improvement in endpoint **and**
temporal gates without damage to clean/physical safety, and consistent actual
update histories. More raw gradient, lower TRAIN loss or more accepted optimizer
steps alone is insufficient. A0 and A1 are both reported; no arm is promoted
automatically. Local synthetic regression tests cannot establish the server's
initial safety or repair outcome, which remain pending execution.

## Local verification

The full suite passed **754 tests, 1 existing PyTorch Transformer warning**.
After adding the final source-change and preflight-only protections, the related
suite passed **47 tests**, including all **18** safe-start tests. Together these
runs cover the current 756 distinct tests. CPU/CUDA rollback, finite-difference
trunk derivatives, actual final scoring, probe isolation, frozen provenance,
and formal-inference rejection are covered. CLI help, `bash -n` and
`git diff --check` also passed.

A separate CPU synthetic smoke test used hidden=4, 60 frames, the actual
192-case objective and unchanged checked optimizer for two steps per arm.
All four updates were accepted. A0's trunk gradient norm was 0 at step 1 and
0.00715115 at step 2; A1's was 0.01390797 at step 1. The corresponding first-step
actual trunk update norms were 0 and 0.000141289. Thus even this small example
does **not** support a claim that zero initialization permanently blocks the
trunk. It validates execution and measurement, not candidate superiority.

Local logs are retained outside tracked source in
`output/audits/safe_start_recovery/pytest_full.log`,
`pytest_final_regression.log`, and `real_update_smoke.log`. The real frozen
server banks, full hidden=256 paired run and scientific outcome remain untested
locally; no Pilot was run and no diagnostic state was promoted.
