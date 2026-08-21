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
    MANIFEST_SCHEMA,
    inspect_smpl_source,
    load_manifest,
)


SOURCE_METADATA = {
    "female_36pose_1": {
        "recording_uid": "female_36pose_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "thirty_six_postures",
        "take_id": None,
    },
    "female_36pose_2": {
        "recording_uid": "female_36pose_sequence",
        "performer_track_id": 2,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "thirty_six_postures",
        "take_id": None,
    },
    "female_lotus": {
        "recording_uid": "female_lotus_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "lotus_steps",
        "take_id": None,
    },
    "female_mediation": {
        "recording_uid": "female_meditation_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "female",
        "dance_category": "revelation_meditation",
        "take_id": None,
    },
    "male_36pose_1": {
        "recording_uid": "male_36pose_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "thirty_six_postures",
        "take_id": None,
    },
    "male_36pose_2": {
        "recording_uid": "male_36pose_sequence",
        "performer_track_id": 2,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "thirty_six_postures",
        "take_id": None,
    },
    "male_drum_1": {
        "recording_uid": "male_drum_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "lei_gong_drum",
        "take_id": None,
    },
    "male_drum_2": {
        "recording_uid": "male_drum_sequence",
        "performer_track_id": 2,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "lei_gong_drum",
        "take_id": None,
    },
    "male_mediation": {
        "recording_uid": "male_meditation_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "revelation_meditation",
        "take_id": None,
    },
    "male_pipa_1": {
        "recording_uid": "male_pipa_sequence_1",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "pipa_behind_back",
        "take_id": 1,
    },
    "male_pipa_2": {
        "recording_uid": "male_pipa_sequence_2",
        "performer_track_id": 1,
        "sequence_index": 2,
        "performer_group": "male",
        "dance_category": "pipa_behind_back",
        "take_id": 2,
    },
    "male_ribbon": {
        "recording_uid": "male_sogdian_whirl_sequence",
        "performer_track_id": 1,
        "sequence_index": 1,
        "performer_group": "male",
        "dance_category": "sogdian_whirl",
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
            "take_id": metadata["take_id"],
            "skeleton_id": (
                "chang_e_official_smpl"
            ),
            "role": "formal_motion_source",
            "enters_recording_disjoint_split": True,
        })

    recording_groups = {
        row["recording_uid"]
        for row in rows
    }

    payload = {
        "schema": MANIFEST_SCHEMA,
        "dataset_name": (
            "Chang-E: A High-Quality Motion Capture "
            "Dataset of Chinese Classical Dunhuang Dance"
        ),
        "source_format": "official_smpl_npz",
        "formal_motion_source": True,
        "timebase_authority": (
            "manifest_source_fps"
        ),
        "source_fps": float(args.source_fps),
        "num_sources": len(rows),
        "num_recording_groups": (
            len(recording_groups)
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
