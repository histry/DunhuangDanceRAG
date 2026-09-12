# Refiner V15.15g: activation-aware manifold selection

V15.15g adds a development-only activation probe after V15.15f1 showed that
the learned observable dead-zone still wakes single controls and suppresses
some cross cases. The probe treats activation as candidate selection rather
than as a scalar classifier.

For every enumerated validation case it constructs the following candidates
from the same immutable Anchor:

1. exact identity with a zero tangent;
2. the frozen Adapter direction at exact `target_rms=1e-4`;
3. Euclidean projected corrections with budgets 2, 3 and 5;
4. product-manifold Retraction corrections with budgets 2, 3 and 5.

The correction and selection objective uses only Anchor-relative endpoint,
temporal and differentiable physical proxy terms. It does not consume the
offline role, `teacher_kind`, `audit_group`, validation label, hidden clean
motion, or complete fixed Guard. A nonidentity candidate must satisfy the
exact radius, strict endpoint/temporal scientific descent, a fixed observable
objective improvement and the physical proxy tolerance. The lowest eligible
objective wins; when none is eligible the output is exact identity.

Role and group metadata are attached only after selection. They are used to
measure false activation on single controls and to run the unchanged raw,
Projector, scope and complete fixed-Guard audit on held-out cross teachers.
Validation results are never converted into teachers or training examples.

The probe passes only when every held-out cross teacher completes raw and
projected closure, both cross groups retain complete coverage, every single
control selects exact identity, ownership leakage is exactly zero, and all
correction/selection diagnostics remain finite. This is stricter than merely
improving the learned gate score.

Server execution uses the rejected V15.15f1 development state as a frozen
input:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g_activation_aware_server.sh
```

This command starts no Adapter training, pseudo-teacher recycling, promotion,
replay, or video generation.
