# RPA-LRTA: Role–Phase–Anatomy Conditioned Low-Rank Tangent Adaptation

## Scope

RPA-LRTA is a research-method candidate layered on the frozen A0
`ProductManifoldTemporalRefiner` and frozen RCSP adapter. It is not a motion
generator and it does not modify production inference.

The candidate changes only how the already available Refiner hidden state is
mapped to an additional local geometric tangent correction.

## Why this is not "add duration"

The production Refiner already includes a continuous duration channel in
`_refiner_fk_dynamics_features`. RPA-LRTA therefore reuses that information
rather than claiming duration was previously absent.

## Why this is not "add phase"

The production boundary feature contract already computes normalized transition
phase in `boundary_features_torch`. RPA-LRTA reads exactly that phase channel.

## Why not width-specific heads

`width=10/28` is evaluation metadata, not a model condition. The model receives
continuous duration seconds and normalized phase, so the same mapping can
generalize between and beyond the observed widths.

## Why low rank

The Dunhuang motion bank is small and specialized. A low-rank residual keeps the
method in the minimum-change repair regime and avoids a new dense motion
generator.

Fixed ranks:

- ROOT translation: 2
- BODY rotations: 8
- EXTREMITY rotations: 4

For current hidden dimension 256 the adapter has exactly 4692 parameters.

## Why anatomy factorization

Previous single-direction decomposition localized direction mismatch in
anatomy×time rather than showing one global reversal. RPA-LRTA therefore maps
ROOT, BODY, and EXTREMITY through separate low-rank branches.

The authoritative extremity joints are imported from
`motion_geometry.physical.EXTREMITY_JOINTS`:

`(7, 8, 10, 11, 20, 21, 22, 23)`.

The complement is BODY:

`(0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)`.

ROOT is only the 3D root translation tangent. Skeleton joint 0 rotation belongs
to BODY.

## Method

For anatomy branch `a`:

```text
z_a(t) = V_a h_t
delta_a(t) = U_a [ g_a(role, phase_t, duration) * z_a(t) ]
```

The shared conditioner is:

```text
(role one-hot 2 + phase 1 + duration 1)
4 -> 32 -> 14
```

The 14 outputs split as `2 + 8 + 4` gates.

The new residual is projected by the existing RCSP binary geometric support.

Endpoint envelope:

```text
E(p) = 64 p^3 (1-p)^3
```

Composition:

```text
delta_total = delta_RCSP + E(p) * delta_RPA_projected
```

The envelope is applied only to the new RPA residual. The frozen RCSP
correction is not rescaled.

## Zero-start contract

- Down projections: deterministic Kaiming.
- Up projections: exactly zero.
- Conditioner final weight: exactly zero.
- Conditioner final bias: exactly one.

Therefore the initial RPA residual is exactly zero and the complete candidate
is initially identical to frozen RCSP.

A preflight backward pass must show a nonzero gradient into at least one
zero-initialized up-projection before AdamW is constructed.

## Frozen components

RPA-LRTA does not change:

- `training/motion_models.py`
- `motion_geometry/boundary_observables.py`
- `motion_geometry/product_manifold.py`
- `motion_geometry/rotations.py`
- `motion_geometry/physical.py`
- temporal metric/reduction
- endpoint metric
- repair thresholds
- jerk non-regression
- decoder support/confidence
- smoothing
- taper
- caps
- SO(3) retraction
- contact path
- Bridge/GAR/Event-RAG

## Training

- frozen TRAIN transaction 0 only
- 192 cases, 48/group
- both roles trained
- fixed 400 checked optimizer attempts
- same authoritative training objective and group guard
- optimizer contains RPA parameters only
- no rank/LR/alpha/checkpoint/architecture search

## Evaluation

Fixed final64 only:

- seen/single/10: 8
- seen/single/28: 8
- new/single/10: 8
- new/single/28: 8
- seen/cross/10: 8
- seen/cross/28: 8
- new/cross/10: 8
- new/cross/28: 8

BASE, RCSP, and RPA-LRTA are all recomputed with the current authoritative
observable metric and gate.

## Decision

Advance review requires all fixed A-I conditions implemented in the experiment:

A. seen single temporal rescue
B. new-position single temporal rescue
C. seen cross/28 median temporal deficit improves
D. new-position cross/28 median temporal deficit improves
E. no RCSP temporal pass regresses
F. no RCSP endpoint pass regresses
G. no physical/geometry/clean/support/contact regression
H. single total-action temporal alignment improves
I. cross/28 total-action temporal alignment improves

Even `RPA_LRTA_CANDIDATE_ADVANCE_REVIEW` does **not** authorize Pilot or
production integration.
