# Refiner V15.15g1b: contract-margin and transaction-conformal repair

V15.15g1b keeps the V15.15g fixed correction budgets (`2/3/5`), existing
starts, and the exact `1e-4` owned-tangent radius. It changes the proxy contract and the
activation calibration identified as false blockers by the V15.15g1 audit.

The five Guard-aligned proxy fields are already signed margins against their
fixed contract limits. A candidate therefore passes when every signed margin
is at most the numeric tolerance. V15.15g1b does not subtract the Anchor
signed margin a second time. The complete fixed Guard, Projector and `0.03`
gate remain unchanged and continue to provide the final acceptance result.

When a proxy margin is positive, correction uses a Guard-first active set. A
smooth maximum of the positive contract margins becomes the primary objective.
The Guard descent direction is projected onto the first-order endpoint and
temporal non-increase cones and onto the exact-radius sphere tangent. A trial
is accepted only if endpoint and temporal descent still pass, the largest
positive Guard margin falls, the exact radius is restored, and scope leakage
is exactly zero. Owned-tangent second- and third-difference penalties suppress
local high-frequency corrections that inflate jerk. These operations do not
add iterations or starts.

An eligible Adapter output is the incumbent. Outside the Guard's safe interior,
a correction must Pareto-dominate the Adapter in every signed margin and both
scientific deltas. When both lie at least `guard_safe_interior_margin` inside
all proxy limits, their Guard states form a safe equivalence class and the
normalized endpoint/temporal descent score decides between them.

Activation uses the full eight-channel observable severity vector. The
calibration bank is restricted to train-split identity controls. Each train
transaction is held out in turn, a shrinkage Mahalanobis model is fit on the
remaining transactions, and the maximum held-out nonconformity score freezes
the conformal threshold. Validation samples never select this threshold. A
small uncertainty band separates the control envelope from activation; cases
inside that band return identity and record `conformal_fallback=true`.

The report records candidate signed margins, largest positive margin, accepted
margin reduction and accepted correction steps, together with conformal score,
threshold, abstention and fallback diagnostics. Role and group labels remain
evaluation-only.

Run the development probe on the server with:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1b_contract_margin_conformal_server.sh
```

The script starts no Adapter training, pseudo-teacher recycling, promotion,
replay, or video generation.
