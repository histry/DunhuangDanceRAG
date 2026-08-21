#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_pipeline.chang_e_smpl_manifest import (
    CANONICAL_SKELETON,
    COORDINATE_SYSTEM,
    HAND_ROTATION_POLICY,
    MANIFEST_SCHEMA,
    POSE_LAYOUT,
    TRANSLATION_UNITS,
    TEST_RELEASE_ID,
    file_sha256,
    load_manifest,
    validate_source,
)
from scripts.build_official_smpl_manifest import SOURCE_METADATA


class OfficialSmplManifestTests(unittest.TestCase):

    def test_authoritative_release_file_set_has_fourteen_sources(self):
        self.assertEqual(len(SOURCE_METADATA), 14)
        self.assertIn("female_FeiTian", SOURCE_METADATA)
        self.assertIn("male_ribbon_FenHe", SOURCE_METADATA)
        self.assertIn("female_meditation", SOURCE_METADATA)
        self.assertIn("male_meditation", SOURCE_METADATA)
        self.assertNotIn("female_mediation", SOURCE_METADATA)
        self.assertEqual(
            SOURCE_METADATA["male_ribbon"]["theme_label_status"],
            "pending_official_confirmation",
        )
        self.assertEqual(
            SOURCE_METADATA["male_ribbon_FenHe"]["dance_category"],
            "unknown",
        )

    def build_fixture(self, root: Path):
        source = root / "female_lotus.npz"

        np.savez(
            source,
            poses=np.zeros(
                (61, 165),
                dtype=np.float32,
            ),
            trans=np.zeros(
                (61, 3),
                dtype=np.float32,
            ),
            smpl_scaling=np.asarray(
                [1.0],
                dtype=np.float32,
            ),
        )

        manifest_path = (
            root / "sources.json"
        )

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "dataset_name": "test",
            "dataset_release_id": TEST_RELEASE_ID,
            "source_format": (
                "official_smpl_npz"
            ),
            "formal_motion_source": True,
            "coordinate_system": COORDINATE_SYSTEM,
            "translation_units": TRANSLATION_UNITS,
            "pose_layout": POSE_LAYOUT,
            "canonical_skeleton": CANONICAL_SKELETON,
            "hand_rotation_policy": HAND_ROTATION_POLICY,
            "timebase_authority": (
                "manifest_source_fps"
            ),
            "source_fps": 30.0,
            "num_sources": 1,
            "num_recording_groups": 1,
            "sources": [
                {
                    "source_id": (
                        "female_lotus"
                    ),
                    "file": source.name,
                    "sha256": file_sha256(
                        source
                    ),
                    "frames": 61,
                    "source_fps": 30.0,
                    "duration_seconds": 2.0,
                    "recording_uid": (
                        "female_lotus_sequence"
                    ),
                    "sequence_id": "female_lotus_sequence",
                    "dancer_id": None,
                    "dancer_id_status": "unverified",
                    "performer_track_id": 1,
                    "sequence_index": 1,
                    "performer_group": (
                        "female"
                    ),
                    "dance_category": (
                        "lotus_steps"
                    ),
                    "theme_label_status": "confirmed",
                    "source_context": [],
                    "coordinate_system": COORDINATE_SYSTEM,
                    "translation_units": TRANSLATION_UNITS,
                    "pose_layout": POSE_LAYOUT,
                    "take_id": None,
                }
            ],
        }

        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        return source, manifest_path

    def test_valid_official_smpl_contract(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            source, manifest_path = (
                self.build_fixture(root)
            )

            manifest = load_manifest(
                manifest_path
            )

            contract = validate_source(
                source,
                manifest=manifest,
                manifest_file=manifest_path,
            )

            self.assertEqual(
                contract["frames"],
                61,
            )

            self.assertEqual(
                contract["source_fps"],
                30.0,
            )

            self.assertAlmostEqual(
                contract[
                    "duration_seconds"
                ],
                2.0,
            )

            self.assertEqual(
                contract[
                    "recording_uid"
                ],
                "female_lotus_sequence",
            )

    def test_hash_mismatch_fails_closed(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            source, manifest_path = (
                self.build_fixture(root)
            )

            manifest = load_manifest(
                manifest_path
            )

            with source.open("ab") as handle:
                handle.write(b"corruption")

            with self.assertRaises(
                ValueError
            ):
                validate_source(
                    source,
                    manifest=manifest,
                    manifest_file=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
