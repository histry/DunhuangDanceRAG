import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from retargeting import anatomy_retarget as research
from retargeting import source_regularized_solver as compat


class SourceRegularizedSolverTests(unittest.TestCase):
    def test_default_direction_regularizer_is_opt_in_on_research_solver(self):
        blocked = {
            "RETARGET_CLEAN_BONE_DIR_VEL_W",
            "RETARGET_CLEAN_BONE_DIR_ACC_W",
            "SOURCE_RETARGET_BONE_DIR_VEL_W",
            "SOURCE_RETARGET_BONE_DIR_ACC_W",
        }
        clean = {key: value for key, value in os.environ.items() if key not in blocked}
        with patch.dict(os.environ, clean, clear=True):
            settings = research.SourceDirectionRegularization.from_environment()
        self.assertFalse(settings.enabled)
        self.assertEqual(settings.bone_direction_velocity_weight, 0.0)
        self.assertEqual(settings.bone_direction_acceleration_weight, 0.0)

    def test_compatibility_wrapper_delegates_to_research_solver(self):
        cfg = compat.RetargetConfig()
        expected_motion = np.zeros((2, 151), dtype=np.float32)
        expected_report = {
            "ok": True,
            "source_direction_regularization": {
                "solver": "anatomy_retarget_research_so3"
            },
        }
        with patch.object(
            compat._research,
            "retarget_bvh_research",
            return_value=(expected_motion, expected_report),
        ) as call:
            motion, report = compat.retarget_bvh("unused.bvh", cfg)
        call.assert_called_once_with("unused.bvh", cfg)
        self.assertIs(motion, expected_motion)
        self.assertEqual(
            report["source_regularized_solver_compatibility"]["delegated_solver"],
            "retargeting.anatomy_retarget.retarget_bvh_research",
        )
        self.assertFalse(
            report["source_regularized_solver_compatibility"]["owns_optimizer"]
        )

    def test_legacy_weight_environment_aliases_only_bridge_when_new_absent(self):
        clean = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "RETARGET_CLEAN_BONE_DIR_VEL_W",
                "RETARGET_CLEAN_BONE_DIR_ACC_W",
                "SOURCE_RETARGET_BONE_DIR_VEL_W",
                "SOURCE_RETARGET_BONE_DIR_ACC_W",
            }
        }
        clean["SOURCE_RETARGET_BONE_DIR_VEL_W"] = "0.123"
        clean["SOURCE_RETARGET_BONE_DIR_ACC_W"] = "0.004"
        with patch.dict(os.environ, clean, clear=True):
            bridged = compat._bridge_legacy_environment()
            self.assertIn(
                "SOURCE_RETARGET_BONE_DIR_VEL_W->RETARGET_CLEAN_BONE_DIR_VEL_W",
                bridged,
            )
            self.assertEqual(os.environ["RETARGET_CLEAN_BONE_DIR_VEL_W"], "0.123")
            self.assertEqual(os.environ["RETARGET_CLEAN_BONE_DIR_ACC_W"], "0.004")

    def test_common_direct_mapping_excludes_virtual_proxy_and_hierarchy_mismatch(self):
        joints = [
            SimpleNamespace(parent=int(research.legacy.PARENTS[i]))
            for i in range(int(research.NUM_JOINTS))
        ]
        bvh = SimpleNamespace(joints=joints)
        mapping = {i: i for i in range(int(research.NUM_JOINTS))}
        mapping.pop(3)
        mapping[10] = 7
        joints[20].parent = 17

        bones = research.common_direct_mapped_bone_children(bvh, mapping)
        self.assertNotIn(3, bones)
        self.assertNotIn(6, bones)
        self.assertNotIn(10, bones)
        self.assertNotIn(20, bones)
        self.assertIn(7, bones)
        self.assertIn(9, bones)
        self.assertIn(21, bones)

    def test_direction_matching_is_zero_for_identical_trajectory(self):
        torch = research.torch
        frames = 20
        target = torch.zeros((frames, 24, 3), dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, frames)
        target[:, 18, 0] = 0.1 * torch.sin(t)
        target[:, 20, 0] = target[:, 18, 0] + torch.cos(t)
        target[:, 20, 1] = torch.sin(t)

        settings = research.SourceDirectionRegularization(
            bone_direction_velocity_weight=0.01,
            bone_direction_acceleration_weight=0.001,
        )
        vel, acc = research._bone_direction_derivative_losses(
            target,
            target,
            children=[20],
            target_fps=30.0,
            settings=settings,
        )
        self.assertAlmostEqual(float(vel), 0.0, places=7)
        self.assertAlmostEqual(float(acc), 0.0, places=7)

    def test_direction_matching_detects_high_frequency_candidate_wobble(self):
        torch = research.torch
        frames = 40
        source = torch.zeros((frames, 24, 3), dtype=torch.float32)
        candidate = source.clone()
        t = torch.linspace(0.0, 1.0, frames)
        source[:, 20, 0] = 1.0
        candidate[:, 20, 0] = torch.cos(0.15 * torch.sin(24.0 * t))
        candidate[:, 20, 1] = torch.sin(0.15 * torch.sin(24.0 * t))

        settings = research.SourceDirectionRegularization(
            bone_direction_velocity_weight=0.01,
            bone_direction_acceleration_weight=0.001,
        )
        vel, acc = research._bone_direction_derivative_losses(
            candidate,
            source,
            children=[20],
            target_fps=30.0,
            settings=settings,
        )
        self.assertGreater(float(vel), 0.0)
        self.assertGreater(float(acc), 0.0)

    def test_build_cache_no_longer_imports_experimental_wrapper(self):
        text = (
            Path(__file__).resolve().parents[1]
            / "retargeting"
            / "build_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn("import retargeting.bvh_solver as legacy", text)
        self.assertNotIn("import retargeting.source_regularized_solver as legacy", text)
        self.assertIn("retarget_bvh_research", text)


if __name__ == "__main__":
    unittest.main()
