# V15.15g1f4 progress-policy / metric interfaces

g1f4 factors the repair experiment into two independent choices on one
authoritative g1f3 execution path:

| Mode | `--progress-mode` | `--metric-mode` | Current availability |
| --- | --- | --- | --- |
| baseline | `current_equal_share` | `identity` | executable; Gate 0 only |
| P-only | `weighted_debt_filter` | `identity` | executable |
| M-only | `current_equal_share` | `anchor_kinematic` | interface only; fail-closed |
| PM | `weighted_debt_filter` | `anchor_kinematic` | interface only; fail-closed |

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

## P kernel: weighted debt filter

The P kernel changes only intermediate-step acceptance. For every non-final
step it requires:

- exact radius and scope;
- no newly positive top-level Guard term;
- no new internal Guard witness;
- strict endpoint and temporal improvement;
- strict authoritative hard-shadow decrease;
- every already-safe modeled row remains safe; and
- strict decrease of normalized positive debt
  `sum_i [max(g_i / s_i, 0)]^2`, with first-version row weights `w_i=1`.

It deliberately removes the v9 requirement that every modeled row repay an
equal fraction of its remaining gap at every step. The last correction step
is unchanged and accepts only real full closure. Candidate projection and the
final composite full-Guard audit remain authoritative.

## M staging contract

The metric interface is defined in the owned physical tangent coordinate. It
does not reinterpret the correction budget in the free `z` coordinate.
`anchor_kinematic` is intentionally fail-closed in this revision: supplying no
calibration rejects the request, and supplying a calibration reports that the
kernel is not yet implemented. It may be implemented only after P evidence
shows additional accepted intermediate steps followed by stable positive hard
shadow at k5.

Before any M-only or PM experiment, train-only equivalent-radius calibration
must freeze:

1. trace-normalized anchor metric scale on active physical coordinates;
2. Euclidean RMS, metric RMS, and task-space `||Jd||_W` for successful train
   corrections;
3. the train-only scale `rho_G = alpha * rho_E`; and
4. the calibration manifest and SHA256 in the frozen contract.

Development and held-out data are forbidden during that calibration.

## Experimental order

1. Run Gate 0 and require exact canonical parity.
2. Run P-only on train; inspect the three known failed targets at k5.
3. If promising, require all five targets, full group coverage, numeric audit,
   scope, single identity, and Adapter incumbent preservation.
4. Run cold starts 2 and 3, then the frozen development bank.
5. Implement/calibrate M only if P has removed the progress-policy bottleneck
   but a stable k5 geometry bottleneck remains.

This revision does not start development, held-out, packaging, whole-song
generation, or Adapter retraining.
