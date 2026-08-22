#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from events.build_database import main as build_database_main
from events.build_database import validate_retarget_contract
from training.motion_models import identity6d_np


class SourceAwareEventContractTest(unittest.TestCase):
    def _report(self):
        return {
            "schema": "chang_e_official_smpl_source_aware_preprocess_v1",
            "version": "chang_e_official_smpl_source_aware_preprocess_event_geometry_3_solo_aware",
            "ok": True,
            "source_gate_ok": True,
            "source_preprocess_ok": True,
            "retargeting_applied": False,
            "source_preprocess_contract": {
                "schema": "chang_e_official_smpl_source_aware_preprocess_v1",
                "direct_official_smpl": True,
                "retargeting_applied": False,
                "event_quality_gate_deferred": True,
            },
            "preprocess_segment": {
                "clean": True,
                "segment_index": 2,
                "source_start_seconds": 10.0,
                "source_end_seconds": 15.0,
            },
        }

    def test_official_source_aware_contract_is_accepted(self):
        ok, reasons = validate_retarget_contract(None, self._report())
        self.assertTrue(ok, reasons)

    def test_failed_source_gate_is_rejected(self):
        report = self._report()
        report["source_gate_ok"] = False
        ok, reasons = validate_retarget_contract(None, report)
        self.assertFalse(ok)
        self.assertIn("source_aware_source_gate_not_ok", reasons)

    def test_formal_database_entry_uses_explicit_event_duration_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "event_db"
            cache.mkdir()
            motion = np.zeros((150, 151), dtype=np.float32)
            motion[:, 7:151] = np.tile(identity6d_np(), 24)[None]
            motion_path = cache / "segment_000.npy"
            np.save(motion_path, motion)

            report = self._report()
            report.update(
                {
                    "source_format": "chang_e_official_smpl",
                    "source": str(root / "female_lotus.npz"),
                    "source_duration_seconds": 5.0,
                    "source_manifest_sha256": "fixture_manifest",
                    "source_metadata": {
                        "source_format": "chang_e_official_smpl",
                        "source_id": "female_lotus",
                        "recording_uid": "female_lotus_sequence",
                        "sequence_id": "female_lotus_sequence",
                        "dancer_id": None,
                        "dancer_id_status": "unverified",
                        "performer_group": "female",
                        "recording_performer_count": 1,
                        "solo_compatibility": "single_track_recording",
                        "solo_compatible": True,
                        "solo_review_status": "not_required_single_track",
                        "dance_category": "lotus_steps",
                        "candidate_dance_category": None,
                        "theme_label_status": "confirmed",
                        "source_context": [],
                        "coordinate_system": "y_up",
                        "translation_units": "m",
                        "pose_layout": (
                            "smplx55_axis_angle_body22_to_smpl24_hands_zero_v1"
                        ),
                    },
                }
            )
            motion_path.with_suffix(".retarget.json").write_text(
                json.dumps(report),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"PERFORMER_REQUIRE_SOLO_COMPATIBLE": "1"},
                clear=False,
            ):
                result = build_database_main(
                    [
                        "--motion_dirs",
                        str(cache),
                        "--out_db",
                        str(output),
                        "--config",
                        "configs/motion_model.json",
                        "--overwrite",
                    ]
                )

            self.assertEqual(result, 0)
            with np.load(output / "events.npz", allow_pickle=True) as database:
                lengths = np.asarray(database["frames"], dtype=np.int64)
            self.assertGreater(len(lengths), 0)
            self.assertTrue(np.all(lengths >= 45))
            self.assertTrue(np.all(lengths <= 120))


if __name__ == "__main__":
    unittest.main()
