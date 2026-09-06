# Generation failure stage diagnostics

Development tooling; not a formal V2 experiment or a replacement for any
physical gate. Implemented locally at user request. No local tests, lint,
compilation, generation or solver validation were performed for this change.
Server validation is required before relying on results.

## Analyze the existing failure without regeneration

From the server checkout with its installed environment:

```bash
python -m training.generation_stage_diagnostics summarize \
  --report outputs/diagnostic_refiner_only/fresh_audio_final.report.json \
  --output outputs/diagnostic_refiner_only/stage_diagnosis_v1.json
```

The output must not already exist. Missing transactions are unknown, not a
pass. Candidate metrics and final selected metrics are separate. Equal report
metrics are not proof of byte-identical motion. No checkpoint is loaded.

## Capture a new controlled diagnostic run

Keep the original generator command, checkpoints, seed, thresholds and input
configuration. Use a new output path, and add:

```bash
export BOUNDARY_STAGE_DIAGNOSTICS=1
export BOUNDARY_DIAGNOSTIC_SLOTS=25,26
export MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS=1
```

The new opt-in capture does not change candidate selection, loss weights or
acceptance. Default operation remains unchanged. It adds disk and audit cost.
Each invocation creates `<output-stem>.diagnostics.<id>/round_NNN/`.
The final report's `stage_reports.generation_stage_diagnostics` identifies the
selected round's bundle and SHA256. Do not select the last directory by time:
the final selected round can differ from the last attempted round.

For selected diagnostic slots, `slot_NNN/stages.json` records hashes, arrays
and descriptive audits for source file motion, loaded/contract-normalized
event, resampled core, aligned core, bridge, and assembled piece. Empty bridges
and clips shorter than four frames are preserved with unavailable audit status.
Local audits use source-observation policy; local floor, duration and sample
counts vary. They are not directly interchangeable with final full-song
acceptance. Use boundary metrics and common support samples before claiming
which transform caused an increase.

Each round captures the assembled full reference, final motion, condition,
seam, semantic sliding eligibility, resolved configuration, environment and
checkpoint hashes. Selected stage snapshots live inside the same round.
Refiner/diffusion guarded candidates are captured before the outer transaction;
they are already downstream of internal runtime guards and must not be called
raw neural proposals. No new checkpoint publication is authorized.

## Export fixed development cases to the existing solver

Supply a provenance JSON with `development_only: true` and `events` keyed by
exact event paths from the selected round. Each entry requires:

```json
{
  "development_only": true,
  "events": {
    "/absolute/server/path/event.npy": {
      "event_sha256": "actual SHA256 of this event file",
      "source_uid": "verified source ID",
      "recording_uid": "verified recording ID",
      "split": "dev",
      "position_stratum": "explicit development stratum"
    }
  }
}
```

Include both preceding and current event for every selected slot. Derive IDs
and split from the data manifest, not filenames. Never relabel blind/test data.

```bash
python -m training.generation_stage_diagnostics export \
  --bundle /absolute/path/to/selected/round_002/bundle.json \
  --provenance /absolute/path/to/verified_development_provenance.json \
  --slots 25 26 --context-frames 16 \
  --proposal-stage refiner \
  --output-dir outputs/refiner_failure_cases_run01

bash scripts/run_refiner_action_feasibility_dev.sh \
  outputs/refiner_failure_cases_run01/cases.json \
  outputs/refiner_feasibility_dev/failure_run01 \
  --config outputs/refiner_failure_cases_run01/config.json \
  --v1-checkpoint /absolute/path/to/the/captured/refiner.pt \
  --max-iterations 32 \
  --iteration-detail full
```

Export alone does not run the solver. The second command is a separate server
diagnostic, not formal training. Use fresh output directories. Preserve the
recorded environment; the existing evaluator applies environment overrides.
The exported masks are recomputed for each local development boundary, contacts
are fixed to zero, and the crop is NOT an exact replay of whole-song inference.
The original whole-song seam and condition remain in the bundle for comparison.
No frozen V1 baseline is claimed unless independently verified proposal and
checkpoint provenance are supplied to the existing evaluator.

With `--proposal-stage refiner`, export verifies the captured full-motion
candidate and checkpoint hashes, crops the guarded Refiner candidate to each
case window, and records `proposal_motion_path`, `proposal_motion_sha256`, and
`proposal_checkpoint_sha256`. The candidate is after the runtime's internal
guards and before the outer stage transaction; it is not relabelled as a raw
network tensor. B2/B3 remain invalid unless the evaluator receives the exact
checkpoint whose SHA256 is recorded by every proposal.

## Replay verified B3 solutions

Successful B1/B3 solves write immutable reference, returned action, and
returned motion arrays under the feasibility run's `solutions/` directory.
The following development-only replay accepts only a complete B3 set:

```bash
python -m training.generation_stage_diagnostics replay \
  --bundle /absolute/path/to/selected/round_002/bundle.json \
  --source-report /absolute/path/to/fresh_audio_final.report.json \
  --feasibility-run outputs/refiner_feasibility_dev/frozen_v1_run01 \
  --output-dir outputs/refiner_solution_replay/run01
```

Replay verifies the source-report-to-bundle link, the frozen Refiner
checkpoint, every solution and reference hash, the original full-sequence
frame spans, and the absence of overlapping actual edit frames. It starts from
the captured reference, commits only changed frames, restores the captured
runtime environment, and then runs the configured diffusion stage, IK,
boundary audit, physical gate, and motion-activity gate. A failed final gate
returns status 2 after preserving `replay.report.json` and all motion artifacts.
The command never trains or modifies a checkpoint and always reports
`pilot_allowed=false` and `production_model_modified=false`.

Legacy reports without condition arrays and a round bundle can be summarized,
but cannot be silently exported with fabricated zero conditions. Regenerate a
captured diagnostic to obtain those missing artifacts.

## Server verification (not run locally)

Run the project's appropriate lint and the scheduling, feasibility and
`tests/test_generation_stage_diagnostics.py` tests in the server environment.
Then perform one controlled capture, confirm selected-round linkage and hashes,
export a known development boundary and check its seam, context and identities.
Only then run the bounded feasibility comparison. Passing unit tests alone is
not evidence that foot contact or the full generation pipeline is repaired.
