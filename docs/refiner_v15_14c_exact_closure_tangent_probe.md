# V15.14c exact-closure-consistent tangent probe

V15.14b established an exact identity control but exposed a domain mismatch:
the MGDA direction was differentiated on the rotating training transaction,
while finite candidates were accepted on the immutable fixed Anchor bank.
All eight training directional derivatives were negative, yet the smallest
finite raw candidate regressed five of the eight fixed-bank observables.

V15.14c is an offline development probe with this contract:

- It evaluates positive and negative output-RMS perturbations of the original
  MGDA direction and computes central finite differences of the exact fixed-bank
  `observable_*_0p03` signed residuals. Diagnostic scales cannot count as
  learning steps.
- It records the original autograd derivative, the exact-closure finite-
  difference derivative, and finite raw/projected residual deltas separately.
- A sign mismatch replaces the training-transaction direction with deterministic
  RMS-normalized MGDA built directly from the eight differentiable fixed-bank
  observable Guard tensors.
- A raw candidate can enter the projector only when all eight exact residual
  deltas are nonpositive within their recorded numerical tolerance, at least
  one decrease exceeds numerical resolution, and the complete immutable Guard
  passes.
- Each contact IK update is projected into the eight fixed-bank observable
  tangent halfspaces before retraction. The complete projector correction is
  then retried at factors `1, 1/2, 1/4, 1/8, 1/16`; every factor receives a new
  exact closure audit.
- Resolution-limited changes are reported and cannot become effective
  candidates.

The source diagnostic, fixed Anchor, physical, fixed-support, fidelity,
boundary, and observable `0.03` thresholds remain unchanged. The probe starts
no training, pilot, promotion, full replay, or generation.

Run on the server with:

```bash
EXPECTED_COMMIT=<full-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_14c_exact_closure_probe_server.sh
```
