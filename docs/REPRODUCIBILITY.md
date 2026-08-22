# Reproducibility

1. Use `configs/experiment.env` as the only profile entry.
2. Keep the worktree clean; checkpoint provenance records the Git revision.
3. Run `bash scripts/preflight.sh`.
4. Run `bash scripts/run_official_smpl_full.sh "$CHANG_E_OFFICIAL_SMPL_DIR" <audio.wav>`.
5. Preserve the source manifest hash, music-corpus report, split report,
   Event-DB fingerprints, checkpoint contracts, final MSSD, route acceptance,
   physical audits, and rendered MP4.

Code presence and passing tests mean `IMPLEMENTED`, not `EXECUTED`.
Formal metrics become `VERIFIED` only after a complete run and reconciliation
against the paper tables.
