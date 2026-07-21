# Scheduler Runtime Migration

The temporary `vendor/edge_scheduler` recovery snapshot has been removed from
the production runtime. The whole-song scheduler now runs entirely from the
project package with:

```bash
python -m scheduling.whole_song_scheduler
python -m scheduling.music_slot_descriptor
```

## Runtime ownership

- `scheduling/whole_song_scheduler.py`: whole-song route and exact frame allocation
- `scheduling/index_io.py`: aligned event-index loading
- `scheduling/retrieval.py`: music/event candidate similarity
- `scheduling/transition_builder.py`: deterministic and learned transition helpers
- `scheduling/event_resampling.py`: event-local SO(3) resampling
- `scheduling/duration_features.py`: inference-safe duration features
- `scheduling/duration_alignment.py`: exact global duration allocation
- `scheduling/transition_diffusion.py`: lazy optional sampler facade
- `motion_geometry/heading.py`: heading and turn state
- `motion_geometry/rotations.py`: canonical column-concatenated Rot6D and SO(3) contract
- `support/scheduler_common.py`: scheduler facade over the EDGE151 contract

Historical checkpoint state keys and serialized descriptor schema identifiers
remain unchanged. Runtime motion arrays use the canonical column-concatenated
Rot6D layout. Historical EDGE duration and transition checkpoints keep their
native PyTorch3D row-concatenated layout and are adapted explicitly at the
model boundary; missing layout metadata defaults to that historical layout.
The event index declares its layout and loading fails closed on a mismatch.

The migration also centralizes event-motion path resolution, including
project-relative and index-metadata roots, so scheduler behavior no longer
depends on the process working directory. SO(3) logarithms use a dedicated
near-pi branch to avoid collapsing valid half turns to a zero tangent vector.

The migration preserved the 94 whole-song CLI switches and 15 descriptor CLI
switches from the recovery runtime. Large event assets and trained weights are
not modified by this refactor.
