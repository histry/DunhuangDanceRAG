# Data and asset contract

## Motion

Formal input is exactly the 14 manifest-authorized Chang-E aligned SMPL NPZ
sequences in `assets/motion/smpl_official_14/`. Each record declares
`coordinate_system`, `translation_units`, and `pose_layout`.

The adapter explicitly maps `poses[T,165]` to 22 body joints, expands to
SMPL24, and zero-fills unavailable hand joints. The canonical downstream tensor
is `151D = 4 contacts + root XYZ + 24 x Rot6D`.

Splits are recording-group-disjoint and category-covered when feasible.
Verified dancer-disjoint claims are forbidden unless authoritative dancer IDs
exist. Single-recording themes belong in leave-one-theme-out evaluation, not
ordinary category-internal metrics.

After the formal solo filter, the ordinary SMPL14 protocol has eight eligible
recording groups and uses an exact `4/2/2` train/validation/test split. Unique
confirmed themes remain in training. Validation and test each contain two
recording groups and at least one confirmed theme that also occurs in training;
pending ribbon themes cannot form an entire held-out split. Event-DB audits
fail closed against the exact source and recording membership recorded in
`source_split_manifest.json`. A one-recording held-out split is accepted only
when that manifest declares the separate leave-one-theme-out protocol.

## Event semantics

The Event-DB separates source identity, dance theme, multi-label local action,
source-only cultural context, and weak music compatibility. Local actions are:
`pose_hold`, `locomotion`, `turn_spin`, `jump_aerial`, `floorwork`,
`upper_body_gesture`, `rhythmic_accent`, `transition`, and `unknown`.

## Music

The formal feature contract is strict Librosa 12D. Any extraction fallback
terminates corpus preparation. Chang-E has no paired audio, so Router
supervision is an explicitly non-ground-truth local semantic-OT teacher.
`paired_audio_motion=false`, `human_training_labels=0`, and
`external_pretrained_model=false` are embedded in checkpoints.

No trained checkpoint or generated index is bundled as a formal asset. Every
formal run rebuilds them from the current Event-DB and music corpus.
