# Refiner V15.15g1e: post-retraction science-feasibility restoration

V15.15g1e addresses the V15.15g1d failure mode in which a shadow descent
direction is feasible in the anchor tangent approximation but violates the
endpoint/temporal cone after exact `1e-4` radius normalization and product
retraction. It does not change the Adapter, correction budgets, starts,
Projector, fixed Guard, ownership contract, or observable `0.03` gate.

For an active full-transaction Guard shadow, the existing direction is first
projected onto the exact-radius sphere tangent and the linearized endpoint and
temporal halfspaces. The candidate is then normalized to the exact radius. If
either scientific term is not strictly improved, one damped, two-row active-set
QP is solved in the endpoint/temporal gradient span. Its displacement is C2
tapered, restricted to current-case ownership, projected onto the sphere
tangent, and normalized again. The candidate is accepted only when the
authoritative hard full-transaction shadow strictly decreases, both scientific
terms strictly improve, the exact radius is recovered, and scope leakage is
exactly zero.

Every line-search candidate records endpoint and temporal deltas, signed margin
to each immutable strict-descent boundary, and the directional derivative
before and after radius normalization. Restoration diagnostics include the
science Jacobian condition number, endpoint/temporal gradient cosine, the
linearized right-hand side, predicted change, and restoration norm.

The search ladder is a train-frozen geometric sequence. The server runner uses
12 scales `1, 1/2, ..., 1/2048`; this is a more resolved feasibility search and
does not add correction iterations beyond `2/3/5`. Damping, the restoration
safety fraction, LogSumExp temperatures, and minimum hard-shadow reductions are
serialized before any development or final evaluation. Exhausting this bounded
ladder without any method producing a joint shadow/science step records
`empty_local_feasible_intersection`. This status is scoped to the frozen local
model and search budget and is not a proof that the global nonlinear feasible
set is empty.

Case `txn_0000_94bfdf553811:53` has influenced g1b through g1e development and
is therefore labeled `development_validation_reused`. The g1e workflow first
runs the intersection check on the train bank. It runs the reused development
bank only if the train check succeeds. It never calls the final held-out runner
automatically.

After the implementation and all numerical choices are frozen, construct a new
validation bank containing at least one effective `cross_long` teacher from a
transaction absent from both train and reused development. The one-shot final
runner rejects transaction overlap, rejects reused case 53, requires the frozen
repair and conformal artifacts, requires both train and final banks to carry
cryptographically sealed manifest hashes, and writes a consumption receipt
before opening the new bank for evaluation. The final bank may use a distinct
append-only held-out manifest because its transaction was created after the
train contract was frozen. Both manifest lineages are recorded in the report;
the final transaction and source-case disjointness checks remain fail-closed.
Reusing that bank is fail-closed.

Server development command:

```bash
EXPECTED_COMMIT=<full-main-commit> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1e_post_retraction_science_restoration_server.sh
```

Final held-out evaluation is a separate, explicitly invoked command after a new
bank exists:

```bash
EXPECTED_COMMIT=<same-full-main-commit> \
FINAL_HELD_OUT_BANK=<new-untouched-validation-bank.pt> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1e_final_held_out_server.sh
```

Neither runner trains an Adapter, recycles pseudo-teachers, promotes a
checkpoint, launches replay, or generates motion/video.
