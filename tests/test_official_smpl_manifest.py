#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_pipeline.chang_e_smpl_manifest import (
    MANIFEST_SCHEMA,
    file_sha256,
    load_manifest,
    validate_source,
)


class OfficialSmplManifestTests(unittest.TestCase):

    def build_fixture(self, root: Path):
        source = root / "female_lotus.npz"

        np.savez(
            source,
            smpl_poses=np.zeros(
                (61, 72),
                dtype=np.float32,
            ),
            smpl_trans=np.zeros(
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
            "source_format": (
                "official_smpl_npz"
            ),
            "formal_motion_source": True,
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
                    "performer_track_id": 1,
                    "sequence_index": 1,
                    "performer_group": (
                        "female"
                    ),
                    "dance_category": (
                        "lotus_steps"
                    ),
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
