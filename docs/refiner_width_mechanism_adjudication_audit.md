# Phase 2.1 — Width Mechanism Adjudication

This is the final read-only mechanism adjudication after the frozen Phase 2
cross-width normalization audit. It consumes one explicit Phase 2
`refiner_cross_width_normalization_audit_v1` report and follows its recorded
paths and SHA-256 hashes. It does not search for a latest artifact.

The scientific primary cohort remains exactly 32 `cross_event` cases:

- `seen/cross_event/10`: 8
- `seen/cross_event/28`: 8
- `new_position/cross_event/10`: 8
- `new_position/cross_event/28`: 8

The 32 `single_recording` cases are retained as excluded controls only. The
observed width comparison is unpaired. Only the counterfactual pair within one
case is paired by construction.

## Three quantities

1. Normalized temporal spread fraction (NTSF) uses the Phase 2 aligned
   `seam_acceleration` and `seam_jerk` stencil-start contributions. For a
   nonnegative contribution vector on authoritative active support `A`:

   ```text
   effective_support(c) = (sum(c))^2 / sum(c^2)
   NTSF(c) = effective_support(c) / |A|
   ```

   A zero square sum is `null`, never zero. The report includes RCSP temporal
   error spread and positive repair spread, where
   `positive_repair = max(BASE_error - RCSP_error, 0)`. Signed repair is only a
   descriptive field and is not used in effective support.

2. Relative-gate gain per applied action norm uses the production gate:

   ```text
   G_base = (M_before - M_base) / M_before
   G_rcsp = (M_before - M_rcsp) / M_before
   delta_G_rcsp = G_rcsp - G_base
   E_gate = delta_G_rcsp / ||final_tangent_RCSP - final_tangent_BASE||_2
   ```

   The action is the production decoder `after_cap` / `final_tangent` delta
   before manifold retraction, restricted to geometric 75D: 3 root tangent
   coordinates plus 24×3 joint tangent coordinates. The four contact logits
   are excluded. A zero action norm gives `null` efficiency. The frozen
   adapter direction cosine is an upstream covariate; no new gradient audit is
   performed.

3. Same-boundary counterfactual metrics use the same degraded motion, BASE
   output, RCSP output, FPS, model, adapter, support/repair output and alpha.
   Only the seam supplied to `boundary_metrics_torch` changes between `cf10`
   and `cf28`. The primary counterfactual quantity is:

   ```text
   delta_G_counterfactual = G_rcsp_cf28 - G_rcsp_cf10
   ```

   The metric ratios for before, BASE and RCSP are components of this metric
   family, not extra scientific branches.

## Seam reconstruction and parity

The reconstruction follows the authoritative frozen construction in
`training.motion_models.degrade_for_refiner` and
`make_cross_event_boundary_np`: halo value `0.35`, core value `1.0`, and the
metric core is `seam >= 0.5`. The actual core interval is read from each
frozen seam; the counterfactual interval is centered at the same half-frame
center and has width 10 or 28. No `center ± 5` or `center ± 14` convention is
assumed without first deriving the center from the actual core.

The audit requires exact full-value and boolean-core parity for the actual
width on all 64 frozen cases. It fails closed if reconstruction parity is not
verified. For the 32 primary cases, the actual-width counterfactual values for
`M_before`, `M_base`, `M_rcsp`, `G_base`, `G_rcsp` and both gate margins must
match the Phase 2/RCSP authoritative values within the Phase 2 tolerance
(`2e-6`).

Pure common scalar normalization cannot change a relative gate:

```text
(cB - cA) / (cB) = (B - A) / B
```

Therefore `pure_scalar_normalization_can_change_relative_gate` is explicitly
`false`; a width-dependent gate shift requires a distribution/support effect
or unequal before/after effects.

## Pre-registered adjudication

The major-gap threshold is fixed at `0.50` of the observed median gate gap.

- If the counterfactual explains at least half the gap in both `seen` and
  `new_position`, and normalized RCSP spread rises in both, classify
  `TEMPORAL_SPREADING_PRIMARY`.
- If that counterfactual condition holds but spread does not rise in both,
  classify `WIDTH_NORMALIZATION_PRIMARY`, worded as a width-dependent
  temporal evaluation/reduction effect rather than a denominator-only causal
  claim.
- If the counterfactual does not explain both gaps, but both observed groups
  have lower `E_gate` and lower frozen adapter cosine at width 28, classify
  `WIDTH_CONDITIONED_DIRECTION_PRIMARY`.
- If counterfactual and direction conditions both hold, classify
  `MIXED_WIDTH_MECHANISM` and order metric/support-time intervention before
  direction intervention.
- Otherwise classify `MIXED_OR_UNRESOLVED_WITH_THREE_METRICS` and proceed to a
  minimal controlled intervention under the strongest cross-source evidence.

All outcomes remain fixed-state evidence only:
`causal_root_cause_proven`, `scientific_acceptance`, `publish_allowed` and
`pilot_allowed` are false. Phase 2.1 is the stopping point for read-only width
mechanism audits; do not add Phase 2.2 or automatically intervene/train/run a
Pilot.

## Server command

Run on the RTX 4090 server after pushing the new commit. The local workflow
does not run validation; the server command is the execution/acceptance gate.

```bash
cd /home/disk/lsm/storage/DunhuangDanceRAG
set -Eeuo pipefail

git fetch origin main
git merge --ff-only origin/main

export PY=/home/disk/lsm/conda_envs/edge/bin/python
export ROOT_DIR=/home/disk/lsm/storage/DunhuangDanceRAG/outputs/run_smpl14_formal_20260822_163915
export EXPECTED_MAIN_COMMIT=<NEW_PHASE_2_1_COMMIT_SHA>
export PHASE2_REPORT="$ROOT_DIR/audits/cross_width_normalization_20260903_003257_14288/result/report.json"

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_MAIN_COMMIT"
test "$(git rev-parse origin/main)" = "$EXPECTED_MAIN_COMMIT"
test -f "$PHASE2_REPORT"

"$PY" -m pytest -q \
  tests/test_refiner_width_mechanism_adjudication_audit.py \
  tests/test_refiner_cross_width_normalization_audit.py

RUN_DIR="$(mktemp -d \
  "$ROOT_DIR/audits/width_mechanism_adjudication_$(date +%Y%m%d_%H%M%S)_XXXXXX")"

bash scripts/audit_refiner_width_mechanism_adjudication.sh \
  "$PHASE2_REPORT" "$RUN_DIR"

REPORT="$RUN_DIR/result/report.json"
test -s "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert r["schema"] == "refiner_width_mechanism_adjudication_audit_v1"
assert r["completed"] is True
assert r["primary_cohort"]["cases"] == 32
c = r["counterfactual_contract"]
assert c["same_case"] is True
assert c["same_motion"] is True
assert c["same_output"] is True
assert c["same_seam_center"] is True
assert c["evaluation_width_only_changed"] is True
assert c["mask_reconstruction_parity_cases"] == 64
assert c["mask_reconstruction_parity_verified"] is True
assert c["fake_case_pairing_performed"] is False
assert r["optimizer_constructed"] is False
assert r["optimizer_steps"] == 0
assert r["parameter_update_performed"] is False
assert r["production_model_modified"] is False
assert r["production_inference_modified"] is False
assert r["scientific_acceptance"] is False
assert r["publish_allowed"] is False
assert r["pilot_allowed"] is False
print(json.dumps(r["adjudication"], ensure_ascii=False, indent=2))
print(json.dumps(r["summaries"], ensure_ascii=False, indent=2))
print("REPORT =", Path(sys.argv[1]))
PY

test -z "$(git status --porcelain)"
echo PHASE_2_1_WIDTH_MECHANISM_ADJUDICATION_OK
echo RUN_DIR=$RUN_DIR
echo REPORT=$REPORT
```
