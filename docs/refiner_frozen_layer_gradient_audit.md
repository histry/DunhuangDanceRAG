# Frozen layer-gradient follow-up (Pilot forbidden)

The user supplied two console exports of
`outputs/run_smpl14_formal_20260822_163915/audits/group_gradients_20260901_025324_8yKxdF/report.json`.
They identify runtime `315c2f84cb04610935df9cfa5ad11eb22fa6aad2`, source
`6e73e0eda9f349d3a611864f4719b22807ee5952`, transaction 0, 192 TRAIN cases,
48 cases/group, 400 completed source steps, no optimizer/probe/pilot.
The original server checkpoint and full JSON are not available locally.
The console evidence is not an independently rerun server experiment.

## Evidence and limits

| Group | Trunk gradient norm | Head gradient norm | Trunk/head | Head squared-norm share |
|---|---:|---:|---:|---:|
| single_short | 0.744027 | 81.25944 | 0.916% | 99.9916% |
| single_long | 0.853043 | 106.0174 | 0.805% | 99.9935% |
| cross_short | 0.244722 | 17.73872 | 1.380% | 99.9810% |
| cross_long | 1.509686 | 119.8691 | 1.259% | 99.9841% |

These are **unclipped parameter-gradient norms**, not effective optimizer
updates, learning progress, or comparable activation gradients. Five of six
head `training_total` group-pair cosines are positive; SS/CL is about -0.14047.
The minimum trunk cosine is about -0.07472. There is local conflict on this
transaction, not evidence of pervasive four-group conflict or its absence at
other steps/transactions. Full-training-gradient norm is 52.082115.

The zero-initialized output projection guarantees zero raw network output and
safe identity decoding at initialization. For `z = W h + b`, `dL/dh = W^T dL/dz`,
so `W=0` blocks trunk gradients at the **initial** step. A 400-step gradient
snapshot does not establish how far the trunk moved, whether small weights
caused task failure, or whether changing initialization would improve repair.
In particular, changing coordinates from `(h,W)` to `(c*h,W/c)` preserves the
function while changing hidden/head gradient norms. A regression demonstrates
the ratio can change by 10,000 with identical forward output.

Clean-to-clean cosine near +1 does not measure clean-vs-repair cosine. With the
recorded clean weight 0.5, however, the supplied norms bound weighted clean norm
at about 0.05%-0.24% of repair norm in this snapshot. This limits aggregate norm
cancellation locally; it does not establish the direction or every parameter's
behavior. The new audit measures the cross-objective cosine directly.

As a consistency check, the identity `g_total = g_repair + 0.5*g_clean`
also permits estimating that cosine from the three supplied rounded norms:
`(||g_total||^2 - ||g_repair||^2 - 0.25*||g_clean||^2) / (||g_repair||*||g_clean||)`.
In group order SS, SL, CS, CL it gives approximately +0.2285, +0.1916,
+0.0933, +0.1510 for all parameters, versus +0.3326, -0.0349, -0.1556,
+0.0573 for the trunk. These are estimates from rounded console values, not
new directly measured dot products; the layer report records direct cosines.

SS/CL endpoint-deficit mean cosine is about -0.248, but endpoint/temporal means
are **not additive contributions** to the smooth-CVaR repair objective. Their
cosines cannot attribute the final repair conflict to endpoint alone.

## Changes

1. Fix frozen TRAIN-bank concatenation: require identical tensor keys across
   the anchor and selected contexts. Previously, an anchor without `clean_cond`
   silently dropped that field from contexts that had it; the reverse case
   failed later with an uninformative KeyError. Both now fail before evaluation.
   Reproducing the old loader from commit `315c2f8` drops conditioning for all
   160 context cases in a deliberately inconsistent test bank. There is no
   evidence that the user's existing frozen source contains this inconsistency.
2. Add weighted clean-vs-repair cosine, norm ratio and cancellation factor to
   the group audit, including the full transaction and each parameter scope.
3. Add `training.refiner_parameter_gradient_audit` and its server wrapper. It
   reuses strict source checks and the exact full-transaction objective. Only
   detached decoder tracing is added to the objective API; training mathematics,
   model state layout, zero initialization, decoder, backward, optimizer and
   scientific thresholds remain unchanged.

The new JSON contains the old group geometry plus `layer_details`:

- Every trainable parameter: count, shape, norm, RMS, zero fraction, gradient
  norm/RMS, connectivity and gradient/parameter ratio (null for zero denominator).
- `in_proj`, all nine sequential trunk modules, head input and head output:
  activation RMS and gradient RMS for each role/width group and repair/clean
  branch; also activation-times-gradient RMS. The trunk uses one residual
  addition around the sequential stack, not nine separate residual blocks.
- Head weight/bias norms, contact/root/joint channel blocks, spectral norm,
  and the actual head VJP compared to `W^T dL/dz` in matching coordinates.
- All five group objectives, plus a directly differentiated full training total.
  The latter must match the four-group mean within stated floating-point
  tolerance. RMS includes all frames of the named cases, not only edit support.
- Decoder raw, masked, smoothed, tapered and applied tangent norms/RMS from the
  same forward. Trace entries may be absent when a decoder stage is disabled.

Observation uses one model forward on 192 repair + 192 clean examples, true
`autograd.grad`, and no optimizer construction/step. Hooks and module modes are
restored even on failure. Source hashes are checked again before writing a new
output outside the frozen source directory. Existing outputs are never replaced.
All outputs keep `pilot_allowed=false`, `scientific_acceptance=false`,
`publish_allowed=false`. No initial-state reconstruction, head rescaling, PCGrad,
four-head routing, clipping changes or clean-loss reweighting is performed.

## Server run

After a clean fast-forward to the reviewed commit, set `EXPECTED_COMMIT` to the
full SHA provided with that review. Keep the existing conda environment.

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail
export PY=/home/disk/lsm/conda_envs/edge/bin/python
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
ROOT=outputs/run_smpl14_formal_20260822_163915
SOURCE="$ROOT/checkpoints/refiner_v15_4_1_lazy_reservoir_foundation_20260831_235832/bridge_diagnostic"
mkdir -p "$ROOT/audits"
RUN_DIR="$(mktemp -d "$ROOT/audits/parameter_gradients_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
bash scripts/audit_refiner_parameter_gradients.sh "$SOURCE" "$RUN_DIR/report.json" \
  2>&1 | tee "$RUN_DIR/console.log"
```

This is a new read-only audit, not another 400-step diagnostic. It automatically
prints a layer summary at completion; return that console log or the full JSON.
It still explicitly supplies legacy decoder strengths core=0.02, transition=1.0;
the old source fingerprint omitted them. Do not use these values if the original
run actually used different overrides. No audit result automatically opens Pilot.

Local regression checks use synthetic data, including CPU/CUDA head VJP checks,
zero vs nonzero test heads, unchanged plain/detailed gradients, full-objective
reconstruction, hook/mode/gradient preservation, invalid inputs and a coordinate
rescaling counterexample. These tests do not validate the server's scientific
result. The actual frozen-state layer measurements remain pending execution.

Validation for this change: **738 tests passed, 1 existing PyTorch Transformer
warning**, including CPU and CUDA checks; both audit shell wrappers passed
`bash -n`, and `git diff --check` was clean. The full local test log is retained
outside tracked source at `output/audits/layer_gradient_recovery/pytest_full.log`.

A later explicit implementation request adds an optional paired initialization
candidate; see [refiner_safe_start_plan.md](refiner_safe_start_plan.md). It does
not retroactively prove starvation or replace the production zero default.
