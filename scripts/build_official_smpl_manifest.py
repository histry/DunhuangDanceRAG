#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.chang_e_smpl_manifest import (
    CANONICAL_SKELETON,
    COORDINATE_SYSTEM,
    HAND_ROTATION_POLICY,
    MANIFEST_SCHEMA,
    OFFICIAL_RELEASE_ID,
    POSE_LAYOUT,
    SOURCE_FORMAT,
    TRANSLATION_UNITS,
    inspect_smpl_source,
    load_manifest,
)


SOURCE_METADATA = {
    "female_36pose_1": {
        "recording_uid": "female_36pose_sequence",
        "sequence_id": "female_36pose_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "thirty_six_postures",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "female_36pose_2": {
        "recording_uid": "female_36pose_sequence",
        "sequence_id": "female_36pose_sequence",
        "performer_track_id": 2,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "thirty_six_postures",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "female_FeiTian": {
        "recording_uid": "female_feitian_sequence",
        "sequence_id": "female_feitian_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "flying_apsaras",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "female_lotus": {
        "recording_uid": "female_lotus_sequence",
        "sequence_id": "female_lotus_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "lotus_steps",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "female_meditation": {
        "recording_uid": "female_meditation_sequence",
        "sequence_id": "female_meditation_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "revelation_meditation",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "male_36pose_1": {
        "recording_uid": "male_36pose_sequence",
        "sequence_id": "male_36pose_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "thirty_six_postures",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "male_36pose_2": {
        "recording_uid": "male_36pose_sequence",
        "sequence_id": "male_36pose_sequence",
        "performer_track_id": 2,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "thirty_six_postures",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "male_drum_1": {
        "recording_uid": "male_drum_sequence",
        "sequence_id": "male_drum_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "lei_gong_drum",
        "theme_label_status": "confirmed",
        "source_context": ["drum"],
        "take_id": None,
    },
    "male_drum_2": {
        "recording_uid": "male_drum_sequence",
        "sequence_id": "male_drum_sequence",
        "performer_track_id": 2,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "lei_gong_drum",
        "theme_label_status": "confirmed",
        "source_context": ["drum"],
        "take_id": None,
    },
    "male_meditation": {
        "recording_uid": "male_meditation_sequence",
        "sequence_id": "male_meditation_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "revelation_meditation",
        "theme_label_status": "confirmed",
        "source_context": [],
        "take_id": None,
    },
    "male_pipa_1": {
        "recording_uid": "male_pipa_sequence_1",
        "sequence_id": "male_pipa_sequence_1",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "pipa_behind_back",
        "theme_label_status": "confirmed",
        "source_context": ["pipa"],
        "take_id": 1,
    },
    "male_pipa_2": {
        "recording_uid": "male_pipa_sequence_2",
        "sequence_id": "male_pipa_sequence_2",
        "performer_track_id": 1,
        "sequence_index": 2,
        "performer_group": "male",
        "dance_category": "pipa_behind_back",
        "theme_label_status": "confirmed",
        "source_context": ["pipa"],
        "take_id": 2,
    },
    "male_ribbon": {
        "recording_uid": "male_ribbon_sequence_1",
        "sequence_id": "male_ribbon_sequence_1",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "unknown",
        "candidate_dance_category": "sogdian_whirl",
        "theme_label_status": "pending_official_confirmation",
        "source_context": ["ribbon"],
        "take_id": None,
    },
    "male_ribbon_FenHe": {
        "recording_uid": "male_ribbon_sequence_2",
        "sequence_id": "male_ribbon_sequence_2",
        "performer_track_id": 1,
        "sequence_index": 2,
        "performer_group": "male",
        "dance_category": "unknown",
        "candidate_dance_category": "sogdian_whirl",
        "theme_label_status": "pending_official_confirmation",
        "source_context": ["ribbon"],
        "take_id": None,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smpl_dir",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--source_fps",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    smpl_dir = Path(
        args.smpl_dir
    ).expanduser().resolve()

    out = Path(
        args.out
    ).expanduser().resolve()

    if not smpl_dir.is_dir():
        raise FileNotFoundError(smpl_dir)

    if out.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out} exists; pass --overwrite"
        )

    files = {
        path.stem: path
        for path in sorted(
            smpl_dir.glob("*.npz")
        )
    }

    expected = set(SOURCE_METADATA)
    actual = set(files)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing or extra:
        raise RuntimeError(
            "Official SMPL file-set mismatch: "
            f"missing={missing}, extra={extra}"
        )

    rows = []

    for source_id in sorted(SOURCE_METADATA):
        path = files[source_id]

        inspected = inspect_smpl_source(
            path,
            source_fps=float(args.source_fps),
        )

        metadata = SOURCE_METADATA[source_id]

        rows.append({
            "source_id": source_id,
            "file": path.name,
            "sha256": inspected["sha256"],
            "frames": inspected["frames"],
            "source_fps": inspected["source_fps"],
            "duration_seconds": (
                inspected["duration_seconds"]
            ),
            "embedded_fps": (
                inspected["embedded_fps"]
            ),
            "embedded_fps_key": (
                inspected["embedded_fps_key"]
            ),
            "pose_key": inspected["pose_key"],
            "recording_uid": (
                metadata["recording_uid"]
            ),
            "sequence_id": metadata["sequence_id"],
            # The release states four dancers, but the filenames do not expose
            # stable cross-sequence identities.  Do not fabricate dancer IDs.
            "dancer_id": None,
            "dancer_id_status": "unverified",
            "performer_track_id": (
                metadata["performer_track_id"]
            ),
            "sequence_index": (
                metadata["sequence_index"]
            ),
            "performer_group": (
                metadata["performer_group"]
            ),
            "dance_category": (
                metadata["dance_category"]
            ),
            "candidate_dance_category": metadata.get(
                "candidate_dance_category"
            ),
            "theme_label_status": metadata["theme_label_status"],
            "source_context": list(metadata["source_context"]),
            "take_id": metadata["take_id"],
            "skeleton_id": (
                "chang_e_official_smpl"
            ),
            "role": "formal_motion_source",
            "enters_recording_disjoint_split": True,
            "coordinate_system": COORDINATE_SYSTEM,
            "translation_units": TRANSLATION_UNITS,
            "pose_layout": POSE_LAYOUT,
        })

    recording_groups = {
        row["recording_uid"]
        for row in rows
    }
    unique_recording_duration_seconds = sum(
        max(
            row["duration_seconds"]
            for row in rows
            if row["recording_uid"] == recording_uid
        )
        for recording_uid in recording_groups
    )

    payload = {
        "schema": MANIFEST_SCHEMA,
        "dataset_name": (
            "Chang-E: A High-Quality Motion Capture "
            "Dataset of Chinese Classical Dunhuang Dance"
        ),
        "dataset_release_id": OFFICIAL_RELEASE_ID,
        "source_format": SOURCE_FORMAT,
        "formal_motion_source": True,
        "coordinate_system": COORDINATE_SYSTEM,
        "translation_units": TRANSLATION_UNITS,
        "pose_layout": POSE_LAYOUT,
        "canonical_skeleton": CANONICAL_SKELETON,
        "hand_rotation_policy": HAND_ROTATION_POLICY,
        "dancer_identity_status": "unverified_in_release_filenames",
        "num_dancers_declared_by_dataset": 4,
        "timebase_authority": (
            "manifest_source_fps"
        ),
        "source_fps": float(args.source_fps),
        "num_sources": len(rows),
        "num_recording_groups": (
            len(recording_groups)
        ),
        "unique_recording_duration_seconds": float(
            unique_recording_duration_seconds
        ),
        "unique_recording_duration_minutes": float(
            unique_recording_duration_seconds / 60.0
        ),
        "sources": rows,
    }

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Self-validation.
    load_manifest(
        out,
        required=True,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(out),
                "num_sources": len(rows),
                "num_recording_groups": (
                    len(recording_groups)
                ),
                "source_fps": float(
                    args.source_fps
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
