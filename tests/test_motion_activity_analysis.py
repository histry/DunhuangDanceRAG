#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.motion_activity_analysis import (
    ActivityThresholds,
    candidate_activity_assessment,
    evaluate_final_motion_activity,
    motion_activity_metrics,
    motion_density_alignment,
    per_slot_activity,
    save_stage_snapshot,
    slot_activity_target,
)


def _matrix_to_canonical_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def _motion(frames: int, dynamic: bool) -> np.ndarray:
    output = np.zeros((frames, 151), dtype=np.float32)
    output[:, 3] = 1.0
    matrices = np.zeros((frames, 24, 3, 3), dtype=np.float32)
    for frame in range(frames):
        angle = 0.0 if not dynamic else 0.024 * frame
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.asarray(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ],
            dtype=np.float32,
        )
        matrices[frame, :, :, :] = rotation
        if dynamic:
            output[frame, 4] = 0.0025 * frame
            output[frame, 6] = 0.0007 * frame
    output[:, 7:151] = _matrix_to_canonical_rot6d(matrices).reshape(frames, 144)
    return output


class MotionActivityAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = ActivityThresholds()

    def test_static_and_dynamic_sequences_are_separated(self) -> None:
        static = motion_activity_metrics(
            _motion(180, dynamic=False), fps=30.0, thresholds=self.thresholds
        )
        dynamic = motion_activity_metrics(
            _motion(180, dynamic=True), fps=30.0, thresholds=self.thresholds
        )
        self.assertGreater(static["static_frame_ratio"], 0.95)
        self.assertLess(dynamic["static_frame_ratio"], 0.10)
        self.assertGreater(dynamic["joint_speed_mean_rad_s"], 0.40)
        self.assertGreater(
            dynamic["motion_density_mean"], static["motion_density_mean"] + 0.50
        )

    def test_calm_semantic_distribution_produces_low_activity_target(
        self,
    ) -> None:
        target = slot_activity_target(
            {
                "role": "calm",
                "music_semantic_probs": {
                    "calm_meditative": 1.0,
                    "pose_hold": 4.924252737341372e-13,
                    "lyrical_flow": 4.924252737341372e-13,
                    "instrument_phrase": 4.924252737341372e-13,
                    "percussive_accent": 4.924252737341372e-13,
                    "turning_climax": 4.924252737341372e-13,
                    "footwork_flow": 4.924252737341372e-13,
                    "aerial_curve": 4.924252737341372e-13,
                },
            }
        )
        self.assertIsNotNone(target)
        self.assertAlmostEqual(float(target), 0.20, places=5)

    def test_turning_climax_produces_high_activity_target(self) -> None:
        target = slot_activity_target(
            {
                "music_semantic_probs": {
                    "calm_meditative": 0.02,
                    "turning_climax": 0.98,
                }
            }
        )
        self.assertIsNotNone(target)
        self.assertGreater(float(target), 0.85)

    def test_probability_dust_does_not_imply_maximum_activity(self) -> None:
        target = slot_activity_target(
            {
                "music_semantic_probs": {
                    "pose_hold": 4.924252737341372e-13,
                }
            }
        )
        self.assertIsNone(target)


    def test_active_slot_rejects_static_candidate(self) -> None:
        metrics = motion_activity_metrics(_motion(120, dynamic=False), fps=30.0)
        assessment = candidate_activity_assessment(metrics, target_activity=0.90)
        self.assertTrue(assessment["active_motion_required"])
        self.assertTrue(assessment["hard_reject"])
        self.assertGreaterEqual(len(assessment["reasons"]), 2)
        self.assertFalse(
            assessment["immutable_physical_anatomy_heading_gates_relaxed"]
        )

    def test_final_gate_rejects_static_but_accepts_dynamic(self) -> None:
        slots = [
            {
                "target_frames": 90,
                "music_semantic_probs": {"pose_hold": 0.0, "climax": 1.0},
            },
            {
                "target_frames": 90,
                "music_semantic_probs": {"pose_hold": 0.0, "build_up": 1.0},
            },
        ]
        static = evaluate_final_motion_activity(
            _motion(180, dynamic=False), slots=slots, fps=30.0
        )
        dynamic = evaluate_final_motion_activity(
            _motion(180, dynamic=True), slots=slots, fps=30.0
        )
        self.assertFalse(static["ok"])
        self.assertTrue(static["collapse_detected"])
        self.assertTrue(dynamic["ok"])
        self.assertFalse(dynamic["collapse_detected"])

    def test_per_slot_alignment_is_reported(self) -> None:
        motion = np.concatenate(
            [_motion(90, dynamic=False), _motion(90, dynamic=True)], axis=0
        )
        slots = [
            {
                "target_frames": 90,
                "music_semantic_probs": {"pose_hold": 1.0},
            },
            {
                "target_frames": 90,
                "music_semantic_probs": {"pose_hold": 0.0, "climax": 1.0},
            },
        ]
        rows = per_slot_activity(motion, slots, fps=30.0)
        alignment = motion_density_alignment(rows)
        self.assertEqual(len(rows), 2)
        self.assertTrue(alignment["available"])
        self.assertEqual(alignment["slot_count"], 2)
        self.assertLess(rows[0]["measured_activity"], rows[1]["measured_activity"])

    def test_slot_failure_is_an_independent_rejection_signal(
        self,
    ) -> None:
        slots = [
            {
                "target_frames": 100,
                "target_motion_density": 0.90,
            },
            {
                "target_frames": 100,
                "target_motion_density": 0.90,
            },
            {
                "target_frames": 100,
                "target_motion_density": 0.90,
            },
        ]

        thresholds = ActivityThresholds(
            # Disable global rejection to isolate the slot-level decision.
            final_max_static_ratio=1.0,
            final_max_static_seconds=999.0,
            final_min_joint_speed_rad_s=0.0,
            final_min_root_travel_per_second_m_s=0.0,
            high_target_slot_threshold=0.55,
            high_target_slot_max_static_ratio=0.75,
            high_target_slot_max_density_gap=0.50,
            high_target_failed_slot_fraction=0.34,
        )

        report = evaluate_final_motion_activity(
            _motion(300, dynamic=False),
            slots=slots,
            fps=30.0,
            thresholds=thresholds,
        )

        self.assertEqual(report["high_activity_target_slots"], 3)
        self.assertEqual(report["failed_high_activity_target_slots"], 3)
        self.assertEqual(
            report["failed_high_activity_target_fraction"],
            1.0,
        )
        self.assertTrue(report["collapse_detected"])
        self.assertFalse(report["ok"])
        self.assertIn(
            "high_activity_slot_failure_fraction",
            report["reasons"],
        )

    def test_stage_snapshot_uses_stable_research_name(self) -> None:
        previous = os.environ.get("MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS")
        os.environ["MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS"] = "1"
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "motion.npy"
                report = save_stage_snapshot(
                    output, "retrieval", _motion(30, dynamic=True), fps=30.0
                )
                self.assertTrue(report["snapshot_saved"])
                self.assertTrue(Path(report["snapshot_path"]).is_file())
                self.assertTrue(
                    str(report["snapshot_path"]).endswith("motion.stage_retrieval.npy")
                )
        finally:
            if previous is None:
                os.environ.pop("MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS", None)
            else:
                os.environ["MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS"] = previous


if __name__ == "__main__":
    unittest.main()
