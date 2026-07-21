import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from routing.boundary_closed_loop import make_seam_mask, physical_quality_gate
try:
    from scheduling.whole_song_scheduler import cap_transition_budget
except ModuleNotFoundError as exc:  # lightweight audit runtime may omit torch
    if exc.name != "torch":
        raise
    cap_transition_budget = None


class TransitionBudgetTests(unittest.TestCase):
    @unittest.skipIf(cap_transition_budget is None, "PyTorch scheduler runtime is unavailable")
    def test_scheduler_transition_fraction_is_capped(self):
        values, report = cap_transition_budget(
            [0, 48, 48, 48],
            total_frames=300,
            max_fraction=0.20,
            minimum_nonzero=6,
        )
        self.assertLessEqual(sum(values), 60)
        self.assertLessEqual(report["actual_fraction"], 0.20)
        self.assertTrue(report["capped"])

    def test_seam_mask_coverage_is_capped(self):
        class FakeV46:
            @staticmethod
            def make_transition_budget_mask(T, spans, cfg):
                return np.ones((T, 1), dtype=np.float32)

        with patch.dict(os.environ, {"V46_54_MAX_TRANSITION_MASK_RATIO": "0.25"}, clear=False):
            mask, _centers, policy = make_seam_mask(
                FakeV46(), 100, [[10, 30], [60, 80]], SimpleNamespace()
            )
        self.assertLessEqual(int((mask[:, 0] > 0).sum()), 25)
        self.assertIn("coverage_cap", policy)

    def test_physical_gate_rejects_skate(self):
        result = physical_quality_gate(
            {
                "foot_skate_mps_p95": 0.60,
                "foot_skate_mps_max": 1.20,
                "foot_penetration_min_m": -0.01,
                "joint_jerk_mps3_p95": 270.0,
                "joint_jerk_mps3_max": 540.0,
                "root_y_range_m": 0.20,
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("foot_skate_mps_p95_too_high", result["reasons"])

    def test_physical_gate_rejects_single_frame_jerk_spike(self):
        result = physical_quality_gate(
            {
                "foot_skate_mps_p95": 0.06,
                "foot_skate_mps_max": 0.30,
                "foot_penetration_min_m": -0.01,
                "joint_jerk_mps3_p95": 270.0,
                "joint_jerk_mps3_max": 3240.0,
                "root_y_range_m": 0.20,
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("joint_jerk_mps3_max_too_high", result["reasons"])

    def test_physical_gate_accepts_stable_locked_feet(self):
        result = physical_quality_gate(
            {
                "foot_skate_mps_p95": 0.096,
                "foot_skate_mps_max": 0.54,
                "foot_penetration_min_m": -0.046,
                "joint_jerk_mps3_p95": 145.8,
                "joint_jerk_mps3_max": 972.0,
                "root_y_range_m": 0.35,
            }
        )
        self.assertTrue(result["ok"], result["reasons"])


if __name__ == "__main__":
    unittest.main()
