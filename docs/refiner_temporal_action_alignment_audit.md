# Refiner Temporal Action Alignment Audit

This audit addresses the fixed A0 result in which all 64 final cases fail the
temporal scientific gate, while 50 also fail endpoint continuity and none fail
physical safety, reference fidelity, or clean identity. The 64/64 temporal
failure is the primary evidence because it is universal across split, role, and
seam width. Endpoint is retained as a control so a temporal-specific pattern is
not inferred from a direction shared by both observable deficits.

The measured action is the 75-dimensional geometric part of the 79-dimensional
Refiner output. Contact channels 0:4 are excluded. Geometry contains root
translation followed by 24 three-dimensional joint tangent blocks. The report
also partitions geometry into root translation, canonical body joints, and the
actual canonical extremity joints `(7, 8, 10, 11, 20, 21, 22, 23)`.

Two gradient points answer different local questions. At the **zero origin**, the
raw geometric output is exactly zero and is decoded by the unchanged production
decoder. If `g0 = dL/da | a=0`, the cosine between the learned action and `-g0`
asks whether the model points toward the local descent direction available from
identity. At the **current output**, `g1 = dL/da | a=a_model`; the scalar
`g1 dot a_model` is the derivative obtained by increasing the current raw action
scale. A negative value is local descent, a positive value is local ascent, and
zero is locally flat. A zero action or gradient has no defined cosine and is
serialized as `null`.

The report retains three coordinate views: all raw geometry, raw geometry on
nonzero decoder support, and a soft-confidence-weighted supported view. The raw
all-geometry dot product is the true derivative for a common raw scale. The
soft-weighted view is only a diagnostic comparison that gives more weight to
coordinates with higher effective decoder confidence. It must not be interpreted
as a replacement backward rule. Raw gradient outside effective decoder support
is reported separately.

TRAIN transaction 0 is materialized as one immutable 192-case transaction with
48 cases in each role/width group. The model executes once on the entire
repair-plus-clean production batch; groups are sliced only for reporting. This
preserves the batch-derived objective scale floors. The fixed final evaluation
combines seen/new-position, single/cross, widths 10/28, and eight cases per cell
into one 64-case forward. The probe is used only to materialize that already
fixed final set and never selects a checkpoint, direction, or finite-difference
step.

The production prediction is compared with a manual call to the official decoder
using the captured raw model output. The audit fails closed if parity exceeds the
declared device tolerance, if any number is nonfinite, if provenance changes, or
if model state, gradients, modes, or hooks are not restored. It creates only a
new JSON report and console log. It constructs no optimizer and performs no
parameter update.

These are first-order measurements at one fixed checkpoint. Negative zero-origin
cosines can support a local directional mismatch, but cannot prove that separate
endpoint and temporal heads are superior. Positive alignment alongside failed
gates may instead reflect insufficient magnitude, decoder attenuation,
nonlinearity, or an objective-to-gate scale mismatch. Exact-zero gradients can
indicate a local dead zone or conditioning issue. The audit does not distinguish
these mechanisms globally, does not tune an action scale, and does not authorize
V15.6, architecture changes, checkpoint promotion, publication, or Pilot.
