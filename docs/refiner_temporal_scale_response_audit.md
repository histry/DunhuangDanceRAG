# Refiner Temporal Scale-Response and Decoder Attenuation Audit

The fixed server alignment audit found a present temporal gradient, 61 of 64
final actions aligned with the zero-origin negative temporal gradient, and 54 of
64 current actions with a locally descending positive scale direction. Gradient
outside decoder support was exactly zero. These observations do not support a
global directional mismatch as the primary explanation for the universal 64 of
64 temporal gate failure. They also do not justify V15.6 or separate endpoint
and temporal heads. The next question is how the fixed action behaves over a
finite, preregistered scale range.

The audit uses exactly this immutable grid:

`[0.00, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]`

It never extends or refines the grid. It does not search between points or use a
probe result to alter the grid. The values are counterfactual response points,
not scale tuning and not candidates for inference or deployment.

The model executes once for the complete 192-case TRAIN transaction and once for
the combined fixed 64-case final set. Both repair and clean raw outputs are then
detached. At every alpha, contact channels 0:4 remain exactly equal to the model
output, while raw geometric channels 4:79 are multiplied by alpha and passed
through the unchanged production decoder. Decoded motion is never scaled
directly. The clean branch follows the same geometric rule so the existing clean
identity gate remains observable.

Alpha zero is the exact zero-geometric-edit baseline. The audit verifies that
the geometric applied tangent is zero while preserving the original contact
channels. Alpha one must reproduce the production repair and clean predictions
within a strict device tolerance. Its endpoint, temporal, physical, reference
fidelity, and clean identity results must also match the fixed trajectory final
evaluation within declared numeric tolerances. No gate is redefined.

The real decoder trace order is:

`raw -> after_mask -> after_smoothing -> after_taper -> after_cap -> applied`

The root and joint vector caps therefore operate after confidence scaling,
smoothing, and taper. Saturation is measured directly from the pre-cap vector
norm and the production root/rotation cap values. Statistics are reported for
root translation, canonical body joints, canonical extremity joints, frames,
and cases. A plateau in applied norm alone is not labeled cap saturation.

For each point, the audit records continuous temporal and endpoint scientific
deficits, authoritative boundary metrics and repair gains, unchanged 3 percent
gate decisions, scalar derivatives with respect to alpha, decoder stage norms,
attenuation ratios, cap saturation, physical safety, reference fidelity, and
clean identity. TRAIN objective values retain one complete 192-case batch so the
batch-derived scale floor is not recomputed independently by group. Model
parameters are never differentiation targets.

Response-shape fields are descriptive. Monotonicity uses only the seven fixed
points, turning records a derivative sign change on that grid, and gate crossing
lists every fixed point with at least one pass. Correlations use only seven
points and carry no significance claim. No first crossing, selected scale,
deployment scale, or formal checkpoint decision is produced.

The four mechanism classifications mean only `supported_by_fixed_response_audit`
or `not_supported_by_fixed_response_audit`. They cannot prove a root cause. In
particular, a favorable result at alpha 1.5 or 2 cannot change formal inference,
authorize a new training run, or authorize Pilot. The evidence must be reviewed
before deciding whether later work concerns amplitude-aware training, decoder
parameterization, single-recording nonlinearity, objective/gate design, or a
future architectural experiment.
