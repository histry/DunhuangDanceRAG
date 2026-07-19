#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from contracts.duration import audit_dynamic_duration


class DynamicDurationContractTest(unittest.TestCase):
    def test_audio_derived_frames(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "motion.npy"
            np.save(path, np.zeros((300, 151), dtype=np.float32))
            contract = {
                "audio": "music.wav",
                "schedule_path": "fresh.mssd.json",
                "total_target_frames": 300,
                "expected_audio_target_frames": 300,
            }
            report = audit_dynamic_duration(path, contract, fps=30.0)
            self.assertTrue(report["ok"])
            self.assertEqual(report["actual_output_frames"], 300)
            self.assertAlmostEqual(report["actual_output_seconds"], 10.0)
            self.assertIsNone(report["fixed_duration_seconds"])

    def test_detects_fixed_length_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "motion.npy"
            np.save(path, np.zeros((3600, 151), dtype=np.float32))
            contract = {
                "total_target_frames": 5400,
                "expected_audio_target_frames": 5400,
            }
            report = audit_dynamic_duration(path, contract, fps=30.0)
            self.assertFalse(report["ok"])
            self.assertEqual(report["actual_output_seconds"], 120.0)
            self.assertEqual(report["expected_audio_seconds"], 180.0)


if __name__ == "__main__":
    unittest.main()
