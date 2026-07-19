# Reproducibility

1. Create the environment: `conda env create -f environment.yml`.
2. Edit `configs/paths.env` only when Python or assets are stored elsewhere.
3. Run `bash scripts/preflight.sh`.
4. Run `bash run.sh` for a full rebuild, retraining and whole-song generation.
5. Use `bash scripts/resume_after_retarget.sh` only after a successful retarget cache.
6. Use `bash scripts/generate_only.sh <audio.wav>` with a trained run directory.

Every release contains `PROJECT_MANIFEST.json`, `ASSET_MANIFEST.json`,
`PATH_MIGRATION.json` and `SHA256SUMS`.
