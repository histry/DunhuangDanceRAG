from __future__ import annotations

import unittest

import numpy as np

from contracts.heading import adaptive_event_segments
from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
    so3_exp_np,
    so3_geodesic_np,
)
from support.common import (
    apply_start_anchor,
    make_geodesic_transition,
    make_linear_transition,
    motion_boundary_metrics,
    transition_cost_from_arrays,
)


def frame(rotation: np.ndarray | None = None) -> np.ndarray:
    value = np.zeros((151,), dtype=np.float32)
    matrix = np.eye(3, dtype=np.float32) if rotation is None else rotation
    rotations = np.broadcast_to(matrix, (24, 3, 3)).copy()
    value[7:151] = matrix_to_rot6d_np(rotations).reshape(-1)
    value[5] = 0.95
    value[:4] = 1.0
    return value


class SchedulerGeometryTests(unittest.TestCase):
    def test_historical_linear_transition_api_uses_so3_geodesic(self):
        start = frame()[None]
        end_rotation = so3_exp_np(np.asarray([0.0, np.pi - 1.0e-5, 0.0], np.float32))
        end = frame(end_rotation)[None]
        start[0, 4:7] = np.asarray([1.0, 0.8, -2.0], dtype=np.float32)
        end[0, 4:7] = np.asarray([3.0, 1.2, 4.0], dtype=np.float32)
        middle = make_linear_transition(start, end, 1)
        canonical = make_geodesic_transition(start, end, 1)
        np.testing.assert_allclose(middle, canonical, atol=1.0e-6)
        np.testing.assert_allclose(
            middle[0, 4:7],
            np.asarray([2.0, 1.0, 1.0], dtype=np.float32),
            atol=1.0e-6,
        )
        rotation = rot6d_to_matrix_np(middle[0, 7:151].reshape(24, 6))
        distance = so3_geodesic_np(np.eye(3, dtype=np.float32), rotation[0])
        self.assertAlmostEqual(float(distance), (np.pi - 1.0e-5) / 2.0, places=4)
        self.assertAlmostEqual(float(np.linalg.det(rotation[0])), 1.0, places=5)

    def test_geodesic_transition_switches_contact_at_midpoint(self):
        start = frame()[None]
        end = frame()[None]
        start[0, :4] = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        end[0, :4] = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        bridge = make_geodesic_transition(start, end, 3)
        np.testing.assert_array_equal(bridge[0, :4], start[0, :4])
        np.testing.assert_array_equal(bridge[-1, :4], end[0, :4])
        self.assertTrue(set(np.unique(bridge[:, :4])).issubset({0.0, 1.0}))

    def test_transition_cost_is_invariant_to_equivalent_rot6d_scaling(self):
        left = frame()
        right = frame()
        scaled = right.copy()
        six = scaled[7:151].reshape(24, 6)
        six[:, :3] *= 3.0
        six[:, 3:] *= 0.25
        scaled[7:151] = six.reshape(-1)
        zeros = np.zeros((151,), dtype=np.float32)
        cost = transition_cost_from_arrays(left, zeros, scaled, zeros)
        self.assertLess(cost, 1.0e-5)

        metrics = motion_boundary_metrics(
            np.stack([left, left], axis=0),
            np.stack([scaled, scaled], axis=0),
        )
        self.assertLess(metrics["pose_jump"], 1.0e-5)

    def test_start_anchor_keeps_rotations_valid(self):
        rotations = [
            so3_exp_np(np.asarray([0.0, angle, 0.0], np.float32))
            for angle in np.linspace(0.0, 2.8, 12)
        ]
        motion = np.stack([frame(value) for value in rotations], axis=0)
        anchored = apply_start_anchor(motion, frame(rotations[-1]), blend_frames=6)
        matrix = rot6d_to_matrix_np(anchored[:, 7:151].reshape(-1, 24, 6))
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_allclose(
            np.linalg.det(matrix),
            np.ones((len(anchored), 24), dtype=np.float32),
            atol=1.0e-5,
        )

    def test_adaptive_segments_obey_configured_frame_bounds(self):
        motion = np.stack([frame() for _ in range(407)], axis=0)
        segments, report = adaptive_event_segments(
            motion,
            {"natural_duration_range_sec": [1.0, 6.0]},
            fps=30.0,
            min_event_frames=45,
            max_event_frames=120,
        )
        lengths = [end - start for start, end in segments]
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], len(motion))
        self.assertTrue(all(45 <= value <= 120 for value in lengths), lengths)
        self.assertEqual(report["min_frames"], 45)
        self.assertEqual(report["max_frames"], 120)


if __name__ == "__main__":
    unittest.main()
