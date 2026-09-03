# BCTR reporting correction

This is a create-only reporting correction for the frozen Phase 3 BCTR
artifact.  The original BCTR report is read and hashed before and after the
operation and is never overwritten.

The historical bug wrote the all-width `newly_rescued_cases` list into
`width28_newly_rescued_cases`.  The correction derives three lists directly
from the 32 frozen `case_level` rows:

- `newly_rescued_cases`;
- `width10_newly_rescued_cases`, restricted to identities containing `/10/`;
- `width28_newly_rescued_cases`, restricted to identities containing `/28/`.

The correction does not load a model, run inference, recompute measurements,
or change BCTR decision inputs.  It writes
`result/report.json` under a fresh directory with schema
`refiner_bctr_reporting_correction_v1`.  The artifact records the source
SHA, corrected lists for `overall`, `seen`, and `new`, unchanged decision
inputs/classification, and false acceptance, publish, and Pilot flags.

Server usage:

```bash
RUN_DIR="$(mktemp -d "$ROOT_DIR/audits/bctr_reporting_correction_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
bash scripts/audit_refiner_bctr_reporting_correction.sh \
  "$BCTR_REPORT" "$RUN_DIR"
```
