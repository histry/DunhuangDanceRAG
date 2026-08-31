V15.5 is a rejected diagnostic experiment. Restoring correct code does not turn
V15.4.1 or V15.5 into a scientifically accepted refiner, and no pilot is authorized.

The user-supplied server report is from commit
`ee5976a9dbfa90522d9253ee8299d2014acb3299`, run
`outputs/run_smpl14_formal_20260822_163915/checkpoints/refiner_v15_5_support_gradient_20260901_013439/bridge_diagnostic`.
It reports 400 completed steps, no early stop, `diagnostic_ready=false`, and
`published=false`. The local repair does not overwrite that run or rewrite Git history.

| Split | single/10 E/T | single/28 E/T | cross/10 E/T | cross/28 E/T |
|---|---|---|---|---|
| seen | 0/8, 0/8 | 1/8, 0/8 | 1/8, 0/8 | 0/8, 0/8 |
| new_position | 1/8, 0/8 | 0/8, 0/8 | 1/8, 0/8 | 0/8, 0/8 |

The report records 358 accepted and 42 retained updates, 106 gradient rescues,
4728 trial evaluations and no nonfinite trials. These are supplied execution
records; the original server state was not independently rerun locally.
The failed intervention does not establish shared-head interference as a fact.
Neither larger gradients nor more accepted optimizer updates imply scientific success.

Confirmed code defects and fixes:

1. V15.5 forwards `raw * confidence` but supplies a binary-support surrogate
   derivative to an optimizer which treats `.grad` as the true derivative.
   A regression using the actual decoder and a shared parameter gives finite-
   difference derivative `-0.64` versus the old backward's `+1.0`. It makes the
   checked optimizer search uphill and reject every trial in that example.
   Restore ordinary multiplication and its true chain rule. Keep forward values,
   smoothing, taper, caps, support masks and objective/gate thresholds unchanged.
   The existing real-loss closure still rejected increasing losses; this is not
   a claim that V15.5 necessarily accepted ascending steps.
2. `MOTION_REFINER_CORE_STRENGTH` and `MOTION_REFINER_TRANSITION_STRENGTH` formerly
   changed decoding without entering the serialized configuration or resume hash.
   Record resolved strengths in the configuration, diagnostic fingerprint and
   refiner resume hash; reject nonfinite/out-of-range strengths. Inference and
   training use the same strength resolver.

The recovery is protocol `soft_confidence_true_chain_rule_v2`, diagnostic schema
`refiner_observable_bridge_diagnostic_v15_5_1`. Old V15.5 optimizer snapshots and
diagnostic authorizations cannot silently carry over. Its source commit remains
in history. No alternative backward gain, loss reweighting, larger network or
extra training budget is introduced.

The next operation is a read-only group-gradient audit. Use the frozen **V15.4.1**
state from `6e73e0eda9f349d3a611864f4719b22807ee5952`, not the rejected V15.5 state.
The audit verifies state/report/bank fingerprints, the fit-bank SHA-256, all
protocols, 400-step completion, the 8-window reservoir contract and deterministic
schedule. It reads `fit_bank.pt` and `diagnostic_state.pt`; it never opens
`probe_bank.pt`. Transaction index 0 is fixed before examining gradients, contains
192 TRAIN cases and must contain 48 examples of each role/width group.

The output gives four-by-four cosine matrices and norms for all parameters,
shared trunk and output head, for:

- Actual repair objective, including the established group scientific CVaR.
- Endpoint and temporal deficit means, explicitly diagnostic quantities rather
  than an additive decomposition of the CVaR objective.
- Clean-identity loss and total weighted training objective.

All group objectives are extracted from one full 192-case forward computation.
Do not run four independent 48-case forwards: that would recompute the
TRAIN-reference quantile floors separately and change the objectives being
compared. Regression tests check that the mean of the four total gradients
reconstructs the full-transaction gradient on heterogeneous reference severities.

Zero-gradient cosine is JSON `null`, never an invented zero/one. The audit restores
physical environment settings from the source fingerprint and loads the stored
configuration without `apply_env()`. It does not step an optimizer, change model
weights or write into the frozen source directory. Its output always says
`scientific_acceptance=false`, `publish_allowed=false`, `pilot_allowed=false`.

Legacy limitation: V15.4.1 did not record regional decoder strengths. The wrapper
explicitly supplies core `0.02` and transition `1.0`, matching the repository's
documented default run settings, and labels these as legacy supplied values.
They are not silently promoted to independently verified source metadata. If the
original server used different overrides, do not run this wrapper unchanged.
New recovery artifacts record the resolved values directly.

Server command, after updating to the exact reviewed recovery commit:

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail
export PY=/home/disk/lsm/conda_envs/edge/bin/python
# Set EXPECTED_COMMIT to the full recovery SHA supplied with the code review.
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
ROOT=outputs/run_smpl14_formal_20260822_163915
SOURCE="$ROOT/checkpoints/refiner_v15_4_1_lazy_reservoir_foundation_20260831_235832/bridge_diagnostic"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$ROOT/audits/group_gradients_${STAMP}"
mkdir -p "$ROOT/audits"
mkdir "$DEST"  # refuse a timestamp collision instead of replacing an old log
bash scripts/audit_refiner_group_gradients.sh "$SOURCE" "$DEST/report.json" \
  2>&1 | tee "$DEST/console.log"
```

Completion means the JSON exists, `completed=true`, `optimizer_steps=0`,
`probe_loaded=false` and `pilot_allowed=false`. Inspect all group matrices and
component norms; cosine conflict is local evidence, not proof that a particular
head architecture will fix the scientific task. Missing artifacts or mismatched
contracts are hard errors: do not substitute a different checkpoint, rerun
V15.5, increase steps, or start pilot to work around them.
