# V15.15g1f4 progress-policy / metric interfaces

g1f4 factors the repair experiment into two independent choices on one
authoritative g1f3 execution path:

| Mode | `--progress-mode` | `--metric-mode` | Current availability |
| --- | --- | --- | --- |
| baseline | `current_equal_share` | `identity` | executable; Gate 0 only |
| P-only | `weighted_debt_filter` | `identity` | executable |
| M-only | `current_equal_share` | `anchor_kinematic` | executable after train-only calibration |
| PM | `weighted_debt_filter` | `anchor_kinematic` | executable after M-only |

No mode changes ownership, taper semantics, the fixed `1e-4` Euclidean
baseline radius, the 2/3/5 correction budgets, the 12 frozen angles,
Projector, or the authoritative full-transaction Guard.

## Gate 0: canonical v9 parity

Before P-only is executed, `current_equal_share + identity` is rerun and
compared with a completed report from canonical commit
`be71ae12a637073714d602525166c620f41bd243`. The comparison covers the full
operational variant payload, including case selection, second-order states,
accepted steps and angles, radius, science diagnostics, Guard margins, and
witness-generation sequences. Only elapsed time and newly added interface
audit fields are removed from the canonical payload.

The server entry point refuses to continue to P-only if this gate fails:

```bash
export V9_REFERENCE_REPORT=/absolute/path/to/v9/fixed_budget_correction.report.json
bash scripts/run_refiner_v15_15g1f4_p_gate_server.sh
```

After Gate 0, that entry point runs only the three frozen v9 failures at k5.
The resulting report is marked `diagnostic_only=true`; its process success
means the numeric audit completed, not that train acceptance passed. A full
2/3/5 train run is a later gate and may start only after this diagnostic shows
useful additional accepted steps or closure.

## P kernel: weighted debt filter

The P kernel changes only intermediate-step acceptance. For every non-final
step it requires:

- exact radius and scope;
- no newly positive top-level Guard term;
- no new internal Guard witness;
- strict endpoint and temporal improvement;
- strict authoritative hard-shadow decrease;
- every already-safe modeled row remains safe; and
- strict decrease of Guard-only normalized positive debt
  `sum_i w_i [max(g_i / s_i, 0)]^2`. Endpoint and temporal terms are
  excluded because they have their own strict-progress checks.

The first-version weights are immutably `w_i=1`. Each `s_i` is frozen from
train as the median fixed-Guard allowance or numeric tolerance (whichever is
larger) for the row's base Guard term. Scale and weight maps carry independent
canonical SHA256 values. A missing, non-positive, non-finite, non-unit, or
hash-mismatched contract fails closed; scales are never inferred from the
current iteration gap.

It deliberately removes the v9 requirement that every modeled row repay an
equal fraction of its remaining gap at every step. The last correction step
is unchanged and accepts only real full closure. Candidate projection and the
final composite full-Guard audit remain authoritative.

## M kernel and calibration contract

The metric interface is defined in the owned physical tangent coordinate. It
does not reinterpret the correction budget in the free `z` coordinate.
`anchor_kinematic` remains fail-closed without a completed train-only
calibration.  The kernel is frozen once per case at the immutable Adapter
physical-tangent anchor.  Boundary, jerk and science Jacobian rows form
independent, train-scale-normalized outer-product groups.  The resulting
`beta I + U.T U` metric is trace-normalized on owned active coordinates;
matrix-vector products are low-rank and inverse products use the exact
Woodbury solve.  The ambient metric and Hessian are never materialized.

Metric inverse-gradient directions, tangent projection, Gram-Schmidt,
direction normalization, differentiable curvature geodesics, finite-angle
trials and radius audits all use the same frozen metric.  The closed-form
update remains on one fixed metric shell for all 2/3/5 repair steps.

Before any M-only or PM experiment, train-only equivalent-radius calibration
must freeze:

1. trace-normalized anchor metric scale on active physical coordinates;
2. Euclidean RMS, metric RMS, and task-space `||Jd||_W` for successful train
   corrections;
3. the train-only scale `rho_G = alpha * rho_E`; and
4. the calibration manifest and SHA256 in the frozen contract.

Development and held-out data are forbidden during that calibration.

The server entry point executes the preregistered order without mixing the
factorial cells:

```bash
export M_PREREG_CONTRACT=$(cat outputs/LATEST_REFINER_V15_15G1F4_M_V1_PREREG_CONTRACT)
bash scripts/run_refiner_v15_15g1f4_m_pm_server.sh
```

It first runs `current_equal_share + identity` only to collect the first
successful train correction per case and freezes `alpha` as the deterministic
median metric/Euclidean RMS ratio.  It then runs M-only followed by PM.  An
uncalibrated preregistration, a calibration that consumed non-train evidence,
or a calibration hash mismatch fails closed.

For M candidates the unchanged Projector kernel and backtracking factors are
used, but each projected trial is rematerialized on the frozen metric shell.
There is no post-hoc Euclidean normalization.  Every such trial reports metric
RMS, Euclidean RMS, `||Jd||_W` and relative metric-radius error before the
authoritative full-transaction Guard decision.  Adapter incumbents retain the
original Euclidean-radius contract.

## Experimental order

1. Run Gate 0 and require exact canonical parity.
2. Run P-only on the three frozen failures at k5; do not treat this diagnostic
   as train acceptance.
3. If promising, require all five targets, full group coverage, numeric audit,
   scope, single identity, and Adapter incumbent preservation.
4. Run cold starts 2 and 3, then the frozen development bank.
5. Implement/calibrate M only if P has removed the progress-policy bottleneck
   but a stable k5 geometry bottleneck remains.

This revision does not start development, held-out, packaging, whole-song
generation, or Adapter retraining.
