from __future__ import annotations

import unittest

import numpy as np

from contracts.heading import adaptive_event_segments
from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    relative_rotvec_np,
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
from support.motion_geometry import (
    canonicalize_event_root_np,
    compose_event_root_xz_np,
    make_so3_transition,
    project_transition_floor_np,
    recompute_transition_contacts_np,
)
from contracts.gravity import fk24_np
from motion_geometry.smpl24 import FOOT_JOINTS


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

    def test_start_anchor_translates_but_does_not_delete_root_path(self):
        motion = np.stack([frame() for _ in range(8)], axis=0)
        motion[:, 4] = np.linspace(2.0, 2.7, len(motion))
        motion[:, 6] = np.linspace(-3.0, -2.6, len(motion))
        anchor = frame()
        anchor[[4, 6]] = np.asarray([10.0, 20.0], dtype=np.float32)
        before_delta = motion[-1, [4, 6]] - motion[0, [4, 6]]
        before_non_xz = motion[:, [0, 1, 2, 3, 5]].copy()
        before_rotations = motion[:, 7:151].copy()
        anchored = apply_start_anchor(motion, anchor, blend_frames=4)
        np.testing.assert_allclose(
            anchored[0, [4, 6]],
            anchor[[4, 6]],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            anchored[-1, [4, 6]] - anchored[0, [4, 6]],
            before_delta,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(
            anchored[:, [0, 1, 2, 3, 5]],
            before_non_xz,
        )
        np.testing.assert_array_equal(
            anchored[:, 7:151],
            before_rotations,
        )

    def test_event_root_contract_preserves_relative_motion_and_height_shape(self):
        motion = np.stack([frame() for _ in range(12)], axis=0)
        motion[:, 4] = 5.0 + np.linspace(0.0, 0.8, len(motion))
        motion[:, 6] = -4.0 + np.linspace(0.0, -0.3, len(motion))
        motion[:, 5] += 0.37 + 0.08 * np.sin(
            np.linspace(0.0, np.pi, len(motion))
        )
        relative_xz = motion[:, [4, 6]] - motion[0, [4, 6]]
        relative_y = motion[:, 5] - motion[0, 5]
        canonical, report = canonicalize_event_root_np(motion)
        composed, stage = compose_event_root_xz_np(
            canonical,
            np.asarray([3.0, -2.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            composed[:, [4, 6]] - composed[0, [4, 6]],
            relative_xz,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            composed[:, 5] - composed[0, 5],
            relative_y,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            composed[0, [4, 6]],
            np.asarray([3.0, -2.0], dtype=np.float32),
            atol=1.0e-6,
        )
        self.assertAlmostEqual(report["target_floor_y_m"], 0.0)
        self.assertEqual(stage["stage_start_xz_m"], [3.0, -2.0])

    def test_velocity_aware_transition_matches_root_endpoint_velocity(self):
        angles_a = [0.0, 0.04, 0.08]
        angles_b = [0.22, 0.26, 0.30]
        previous = np.stack(
            [
                frame(
                    so3_exp_np(
                        np.asarray([0.0, angle, 0.0], dtype=np.float32)
                    )
                )
                for angle in angles_a
            ],
            axis=0,
        )
        following = np.stack(
            [
                frame(
                    so3_exp_np(
                        np.asarray([0.0, angle, 0.0], dtype=np.float32)
                    )
                )
                for angle in angles_b
            ],
            axis=0,
        )
        previous[:, 4] = np.asarray([0.0, 0.02, 0.04], dtype=np.float32)
        following[:, 4] = np.asarray([0.18, 0.20, 0.22], dtype=np.float32)
        bridge = make_so3_transition(previous, following, 5, fps=30.0)
        joined = np.concatenate([previous, bridge, following], axis=0)
        root_velocity = np.diff(joined[:, 4])
        left_join = len(previous) - 1
        right_join = len(previous) + len(bridge)
        self.assertLess(
            abs(root_velocity[left_join] - root_velocity[left_join - 1]),
            0.01,
        )
        self.assertLess(
            abs(root_velocity[right_join] - root_velocity[right_join - 1]),
            0.01,
        )
        root_rotation = rot6d_to_matrix_np(joined[:, 7:13])
        angular_step = relative_rotvec_np(
            root_rotation[:-1],
            root_rotation[1:],
        )
        self.assertLess(
            float(
                np.linalg.norm(
                    angular_step[left_join] - angular_step[left_join - 1]
                )
            ),
            0.03,
        )
        self.assertLess(
            float(
                np.linalg.norm(
                    angular_step[right_join] - angular_step[right_join - 1]
                )
            ),
            0.03,
        )

    def test_transition_floor_projection_and_contact_ramp(self):
        transition = np.stack([frame() for _ in range(9)], axis=0)
        transition[:, 5] -= 0.18
        corrected, floor_report = project_transition_floor_np(
            transition,
            target_floor_y=0.0,
            clearance_m=0.002,
            smoothing_frames=5,
        )
        foot_y = fk24_np(corrected)[:, list(FOOT_JOINTS), 1]
        self.assertGreaterEqual(float(foot_y.min()), 0.0019)
        self.assertTrue(floor_report["applied"])
        rebuilt, contact_report = recompute_transition_contacts_np(
            corrected,
            fps=30.0,
            floor_y=0.0,
            left_contact=np.zeros((4,), dtype=np.float32),
            right_contact=np.zeros((4,), dtype=np.float32),
            ramp_seconds=4.0 / 30.0,
        )
        self.assertTrue(np.all((rebuilt[:, :4] >= 0.0) & (rebuilt[:, :4] <= 1.0)))
        self.assertEqual(contact_report["ramp_frames"], 4)

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
