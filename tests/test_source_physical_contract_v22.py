import unittest
from types import SimpleNamespace

import numpy as np

from contracts.physical_quality import (
    PhysicalQualityLimits,
    _append_required_low_relative_reference_aware_floor,
    evaluate_source_physical_clean_audit,
    physical_metric_specs,
)
from motion_geometry.physical import (
    PHYSICAL_METRICS_SCHEMA,
    SOURCE_REFERENCE_KINEMATICS_SCHEMA,
    SUPPORT_POLICY_SOURCE,
    _unit_bone_direction_jerk_metrics,
)
from motion_geometry.smpl24 import NUM_JOINTS, PARENTS
from retargeting.build_cache import _common_valid_mapped_bone_contract


class SourcePhysicalContractV22Tests(unittest.TestCase):
    def test_unit_bone_jerk_is_invariant_to_length_and_upstream_translation(self):
        fps = 30.0
        frames = 120
        t = np.arange(frames, dtype=np.float64) / fps
        parent, child = 18, 20

        source = np.zeros((frames, NUM_JOINTS, 3), dtype=np.float64)
        candidate = np.zeros_like(source)
        parent_path = np.stack(
            [
                0.2 * np.sin(0.7 * t),
                0.1 * np.cos(0.4 * t),
                0.15 * np.sin(0.3 * t),
            ],
            axis=1,
        )
        angle = 0.5 * np.sin(1.3 * t) + 0.15 * np.sin(3.1 * t)
        direction = np.stack(
            [np.cos(angle), np.sin(angle), 0.2 * np.sin(0.6 * angle)],
            axis=1,
        )
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)

        source[:, parent] = parent_path
        source[:, child] = parent_path + 0.25 * direction

        translated_parent = parent_path + np.stack(
            [
                0.4 * np.sin(0.2 * t),
                0.3 * np.cos(0.1 * t),
                0.2 * np.sin(0.15 * t),
            ],
            axis=1,
        )
        candidate[:, parent] = translated_parent
        candidate[:, child] = translated_parent + 0.55 * direction

        ref = _unit_bone_direction_jerk_metrics(
            source,
            fps=fps,
            source_comparison_bones=[child],
        )
        cand = _unit_bone_direction_jerk_metrics(
            candidate,
            fps=fps,
            source_comparison_bones=[child],
        )
        for key in (
            "unit_bone_joint_jerk_s3_p95",
            "unit_bone_joint_jerk_s3_p99",
            "unit_bone_joint_jerk_window_p95_max_s3",
        ):
            self.assertAlmostEqual(ref[key], cand[key], places=8)
        self.assertAlmostEqual(ref["unit_bone_median_lengths_m"][0], 0.25, places=8)
        self.assertAlmostEqual(cand["unit_bone_median_lengths_m"][0], 0.55, places=8)

    def test_common_bone_filter_excludes_virtual_proxy_and_hierarchy_mismatch(self):
        joints = [SimpleNamespace(parent=int(parent)) for parent in PARENTS]
        bvh = SimpleNamespace(joints=joints)

        rest = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
        for joint in range(1, NUM_JOINTS):
            rest[joint] = rest[int(PARENTS[joint])] + np.asarray(
                [0.02 * ((joint % 3) - 1), 0.1 + 0.001 * joint, 0.03],
                dtype=np.float64,
            )
        aligned = np.repeat(rest[None], 10, axis=0)

        mapping = {index: index for index in range(NUM_JOINTS)}
        mapping.pop(3)  # virtual belly: not a direct source observation
        mapping[10] = 7  # toe proxy copied from ankle
        bvh.joints[20].parent = 17  # wrong direct parent for target bone 18->20

        valid, contract = _common_valid_mapped_bone_contract(
            bvh,
            mapping,
            aligned,
            rest,
        )
        self.assertNotIn(3, valid)
        self.assertNotIn(6, valid)  # its target parent 3 is not directly mapped
        self.assertNotIn(10, valid)
        self.assertNotIn(20, valid)
        self.assertIn(7, valid)
        self.assertIn(9, valid)
        self.assertIn(21, valid)

        reasons = {
            int(row["child"]): row["reason"] for row in contract["excluded"]
        }
        self.assertEqual(reasons[3], "child_not_directly_mapped")
        self.assertEqual(reasons[6], "parent_not_directly_mapped")
        self.assertEqual(reasons[10], "child_parent_share_source_proxy")
        self.assertEqual(reasons[20], "source_hierarchy_not_direct_parent_match")

    def test_p001_floor_is_reference_aware(self):
        reasons = []
        detail = {}
        _append_required_low_relative_reference_aware_floor(
            reasons,
            detail,
            {"p001": -0.165},
            {"p001": -0.288},
            key="p001",
            margin=0.06,
            absolute_floor=-0.14,
            reason="regressed",
        )
        self.assertEqual(reasons, [])
        self.assertAlmostEqual(detail["p001"]["allowed_minimum"], -0.348)
        self.assertEqual(
            detail["p001"]["semantics"],
            "reference_below_floor_relative_non_regression_only",
        )

        reasons = []
        detail = {}
        _append_required_low_relative_reference_aware_floor(
            reasons,
            detail,
            {"p001": -0.132},
            {"p001": -0.056},
            key="p001",
            margin=0.06,
            absolute_floor=-0.14,
            reason="regressed",
        )
        self.assertEqual(reasons, ["regressed"])
        self.assertAlmostEqual(detail["p001"]["allowed_minimum"], -0.116)

    @staticmethod
    def _reference(p001=-0.288):
        payload = {
            "schema": SOURCE_REFERENCE_KINEMATICS_SCHEMA,
            "frames": 100,
            "fps": 30.0,
            "unit_bone_comparison_bones": [20],
            "unit_bone_comparison_parents": [18],
            "unit_bone_extremity_comparison_bones": [20],
            "foot_support_drift_m_p95": 0.0,
            "foot_support_drift_m_max": 0.0,
            "foot_penetration_p01_m": 0.0,
            "foot_penetration_p001_m": float(p001),
            "root_y_robust_range_m": 0.0,
            "root_vertical_speed_mps_p95": 0.0,
            "root_vertical_speed_mps_max": 0.0,
        }
        for key in (
            "unit_bone_joint_jerk_s3_p95",
            "unit_bone_joint_jerk_s3_p99",
            "unit_bone_joint_jerk_window_p95_max_s3",
            "unit_bone_extremity_jerk_s3_p95",
            "unit_bone_extremity_jerk_s3_p99",
            "unit_bone_extremity_jerk_window_p95_max_s3",
        ):
            payload[key] = 100.0
        return payload

    @staticmethod
    def _candidate(p001=-0.165, run_seconds=0.0, rotation_max=0.0):
        payload = {
            "schema": PHYSICAL_METRICS_SCHEMA,
            "frames": 100,
            "fps": 30.0,
            "support_state_contract": {"policy": SUPPORT_POLICY_SOURCE},
            "unit_bone_comparison_bones": [20],
            "unit_bone_comparison_parents": [18],
            "unit_bone_extremity_comparison_bones": [20],
            "foot_support_drift_m_p95": 0.0,
            "foot_support_drift_m_max": 0.0,
            "foot_contact_height_m_max": 0.0,
            "foot_penetration_p01_m": 0.0,
            "foot_penetration_p001_m": float(p001),
            "foot_penetration_catastrophic_threshold_m": -0.18,
            "foot_penetration_catastrophic_run_max_seconds": float(run_seconds),
            "foot_penetration_min_m": -0.22,
            "root_y_robust_range_m": 0.0,
            "root_vertical_speed_mps_p95": 0.0,
            "root_vertical_speed_mps_max": 0.0,
        }
        for key in (
            "unit_bone_joint_jerk_s3_p95",
            "unit_bone_joint_jerk_s3_p99",
            "unit_bone_joint_jerk_window_p95_max_s3",
            "unit_bone_extremity_jerk_s3_p95",
            "unit_bone_extremity_jerk_s3_p99",
            "unit_bone_extremity_jerk_window_p95_max_s3",
        ):
            payload[key] = 100.0
        for spec in physical_metric_specs(PhysicalQualityLimits()):
            if spec.layer == "rotation_quality":
                payload[spec.key] = 0.0
        payload["joint_rotation_step_rad_max"] = float(rotation_max)
        return payload

    def test_sustained_penetration_is_independent_hard_gate(self):
        gate = evaluate_source_physical_clean_audit(
            self._candidate(run_seconds=0.20),
            source_reference_audit=self._reference(),
        )
        self.assertFalse(gate["ok"])
        self.assertIn("foot_penetration_sustained_catastrophic", gate["reasons"])
        self.assertTrue(
            gate["relative_checks"]["foot_penetration_sustained_catastrophic"][
                "independent_hard_gate"
            ]
        )

    def test_rotation_step_limit_remains_strict(self):
        self.assertEqual(PhysicalQualityLimits().joint_rotation_step_rad_max, 1.20)
        gate = evaluate_source_physical_clean_audit(
            self._candidate(rotation_max=1.2788559198379517),
            source_reference_audit=self._reference(),
        )
        self.assertFalse(gate["ok"])
        self.assertIn("joint_rotation_step_rad_max_too_high", gate["reasons"])


if __name__ == "__main__":
    unittest.main()
