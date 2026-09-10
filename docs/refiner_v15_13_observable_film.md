# V15.13 observable FiLM probe

This is a development-only, nonpublishing 50-step learnability probe.  It does
not authorize a pilot, formal training, checkpoint promotion, generation, or a
paper result.

## Change under test

The existing 33-frame temporal 1D-CNN remains the shared backbone.  When
`product_refiner_film_conditioning` is enabled, the final shared feature map is
modulated before the unchanged 79D output projection:

`F' = gamma(c) * F + beta(c)`

`c` contains two continuous inference-time observables computed from the clean
frames immediately outside each declared edit region:

1. mean SMPL24 FK endpoint distance, transformed with `log1p`;
2. absolute wrapped root-yaw difference divided by pi.

No `single_recording` / `cross_event` role label, source identity, hidden clean
interior, target correction, or validation signal enters the model.  FiLM is
identity-initialized (`gamma=1`, `beta=0`), and the established output layer is
still zero-initialized, preserving the exact safe start.

The modulation is bounded to `exp([-2,2])` for gamma and `[-0.1,0.1]` for beta.
It is active only on the declared seam support.  Retraction, FK, physical,
fixed-support, fidelity, boundary, and observable 0.03 gates are unchanged.

## Isolation and compatibility

The feature is opt-in through
`MOTION_PRODUCT_REFINER_FILM_CONDITIONING=1`.  With the default disabled value,
the module is absent and legacy state dictionaries retain their exact layout.
An enabled checkpoint records the FiLM protocol in its motion contract and is
rejected if loaded with a runtime that has FiLM disabled or mismatched.

## Required first run

Run only `scripts/run_refiner_v15_13_film_probe_server.sh`.  The runner executes
the focused server tests, creates a fresh foundation report under the same
configuration fingerprint, and then runs 50 diagnostic steps.  It records
condition, gamma, beta, output-amplitude, exact-Guard, and fit-context evidence.

Continue beyond the probe only if the report shows measurable accepted updates,
nontrivial output amplitude, improvement on the fitted contexts, and unchanged
hard-gate compliance.  An execution status of zero alone is not scientific
acceptance.
