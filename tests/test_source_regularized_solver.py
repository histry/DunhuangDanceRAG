import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from retargeting import source_regularized_solver as solver


class SourceRegularizedSolverTests(unittest.TestCase):
    def test_default_temporal_regularizer_is_opt_in(self):
        keys = {
            "SOURCE_RETARGET_BONE_DIR_VEL_W": "",
            "SOURCE_RETARGET_BONE_DIR_ACC_W": "",
        }
        clean = {
            key: value for key, value in os.environ.items()
            if key not in keys
        }
        with patch.dict(os.environ, clean, clear=True):
            settings = solver.SourceTemporalRegularization.from_environment()
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.bone_direction_velocity_weight, 0.0)
        self.assertEqual(settings.bone_direction_acceleration_weight, 0.0)


    def test_disabled_regularizer_uses_exact_legacy_fast_path(self):
        fake_bvh = SimpleNamespace(joints=[])
        cfg = solver.RetargetConfig()
        expected_motion = np.zeros((2, 151), dtype=np.float32)
        expected_report = {"ok": True, "fit": {}, "gravity": {}}
        clean = {
            key: value for key, value in os.environ.items()
            if key not in {
                "SOURCE_RETARGET_BONE_DIR_VEL_W",
                "SOURCE_RETARGET_BONE_DIR_ACC_W",
            }
        }
        with (
            patch.dict(os.environ, clean, clear=True),
            patch.object(solver, "parse_bvh", return_value=fake_bvh),
            patch.object(solver, "build_joint_mapping", return_value={}),
            patch.object(
                solver,
                "common_direct_mapped_bone_children",
                return_value=(7, 8),
            ),
            patch.object(
                solver._legacy,
                "retarget_bvh",
                return_value=(expected_motion, expected_report),
            ) as legacy_call,
        ):
            motion, report = solver.retarget_bvh("unused.bvh", cfg)
        legacy_call.assert_called_once_with("unused.bvh", cfg)
        self.assertIs(motion, expected_motion)
        self.assertTrue(
            report["source_temporal_regularization"]["baseline_fast_path"]
        )
        self.assertFalse(
            report["source_temporal_regularization"]["enabled"]
        )

    def test_common_direct_mapping_excludes_proxy_and_hierarchy_mismatch(self):
        joints = [
            SimpleNamespace(parent=int(solver.PARENTS[i]))
            for i in range(int(solver.NUM_JOINTS))
        ]
        bvh = SimpleNamespace(joints=joints)
        mapping = {i: i for i in range(int(solver.NUM_JOINTS))}
        mapping.pop(3)
        mapping[10] = 7
        joints[20].parent = 17

        bones = solver.common_direct_mapped_bone_children(bvh, mapping)
        self.assertNotIn(3, bones)
        self.assertNotIn(6, bones)
        self.assertNotIn(10, bones)
        self.assertNotIn(20, bones)
        self.assertIn(7, bones)
        self.assertIn(9, bones)
        self.assertIn(21, bones)

    def test_direction_matching_is_zero_for_identical_trajectory(self):
        torch = solver.torch
        frames = 20
        target = torch.zeros((frames, 24, 3), dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, frames)
        target[:, 18, 0] = 0.1 * torch.sin(t)
        target[:, 20, 0] = target[:, 18, 0] + torch.cos(t)
        target[:, 20, 1] = torch.sin(t)

        settings = solver.SourceTemporalRegularization(
            bone_direction_velocity_weight=0.1,
            bone_direction_acceleration_weight=0.025,
        )
        vel, acc = solver._bone_direction_derivative_losses(
            target,
            target,
            children=[20],
            target_fps=30.0,
            settings=settings,
        )
        self.assertAlmostEqual(float(vel), 0.0, places=7)
        self.assertAlmostEqual(float(acc), 0.0, places=7)

    def test_direction_matching_detects_high_frequency_candidate_wobble(self):
        torch = solver.torch
        frames = 40
        source = torch.zeros((frames, 24, 3), dtype=torch.float32)
        candidate = source.clone()
        t = torch.linspace(0.0, 1.0, frames)
        source[:, 20, 0] = 1.0
        candidate[:, 20, 0] = torch.cos(0.15 * torch.sin(24.0 * t))
        candidate[:, 20, 1] = torch.sin(0.15 * torch.sin(24.0 * t))

        settings = solver.SourceTemporalRegularization(
            bone_direction_velocity_weight=0.1,
            bone_direction_acceleration_weight=0.025,
        )
        vel, acc = solver._bone_direction_derivative_losses(
            candidate,
            source,
            children=[20],
            target_fps=30.0,
            settings=settings,
        )
        self.assertGreater(float(vel), 0.0)
        self.assertGreater(float(acc), 0.0)


if __name__ == "__main__":
    unittest.main()
