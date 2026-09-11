# V15.15e multi-transaction teacher expansion

V15.15e expands the development teacher evidence across the frozen rotating-C5
transactions already stored in `fit_bank.pt`. It does not train or publish a
Refiner.

The split manifest is materialized and SHA256-locked before any Oracle search.
Each transaction has a deterministic identifier derived from its schedule, and
each case uses `transaction_id:local_case_index` as its external key. Because
rotating transactions can contain the same underlying reservoir case, the
manifest also records a source-case identifier and assigns every source case to
one split and one canonical transaction occurrence. Both composite-key overlap
and source-case overlap between train and validation are fail-closed errors.

Each manifest Oracle case is evaluated at the unchanged `target_rms=1e-4` by
the V15.14h full-tangent Oracle on its complete transaction. Before search, the
Oracle freezes that transaction's baseline Guard values as its immutable
anchor; it keeps the source contract's relative and absolute tolerances
unchanged. This avoids comparing transaction metrics against the unrelated
seen-only aggregate anchor. A projected direction enters a teacher bank
only when the raw candidate, Projector result, exact ownership scope, complete
fixed Guard, and numeric audit all pass. Failed Oracle cases remain visible in
the report as `oracle_case_uids_without_teacher`; they are never converted into
teachers.

The train and validation banks contain concatenated frozen transaction tensors.
Their local case indices are mapped to collision-free global tensor offsets,
while logs and evidence retain the composite case identity. Projected samples
also carry an inverse `group x transaction` density weight. When a V15.15
Adapter probe consumes this bank in case-isolated restoration mode, the loss
gives equal mass to `cross_short` and `cross_long`, and equal mass to each
represented transaction within a group.

Server execution uses:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15e_multi_transaction_teacher_expansion_server.sh
```

Optional environment controls are `MAX_TRANSACTIONS`,
`MAX_CASES_PER_GROUP`, `VALIDATION_FRACTION`, and `ORACLE_ITERATIONS`. The
defaults are 8, 4, 0.25, and 60. A process exit status of zero means execution
completed. The separate scientific status is 0 only when both frozen splits
contain at least one passing projected teacher from each cross group; status 2
preserves a complete negative or incomplete-coverage result without starting
training.
