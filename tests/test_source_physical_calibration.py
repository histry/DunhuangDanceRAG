import unittest

from contracts.physical_quality import (
    PhysicalQualityLimits,
    SourcePhysicalQualityPolicy,
    evaluate_source_physical_clean_audit,
    physical_metric_specs,
)
from motion_geometry.physical import (
    PHYSICAL_METRICS_SCHEMA,
    SOURCE_REFERENCE_KINEMATICS_SCHEMA,
    SUPPORT_POLICY_SOURCE,
)


class SourcePhysicalCalibrationTests(unittest.TestCase):
    @staticmethod
    def _reference():
        payload = {
            "schema": SOURCE_REFERENCE_KINEMATICS_SCHEMA,
            "frames": 100,
            "fps": 30.0,
            "unit_bone_comparison_bones": [20],
            "unit_bone_comparison_parents": [18],
            "unit_bone_extremity_comparison_bones": [20],
            "foot_support_drift_m_p95": 0.0,
            "foot_support_drift_m_max": 0.0,
            "foot_penetration_p01_m": -0.02,
            "foot_penetration_p001_m": -0.043868622899055486,
            "root_y_robust_range_m": 0.0,
            "root_vertical_speed_mps_p95": 0.0,
            "root_vertical_speed_mps_max": 0.0,
            "unit_bone_joint_jerk_s3_p95": 382.0,
            "unit_bone_joint_jerk_s3_p99": 1338.0,
            "unit_bone_joint_jerk_window_p95_max_s3": 4462.0,
            "unit_bone_extremity_jerk_s3_p95": 537.0,
            "unit_bone_extremity_jerk_s3_p99": 2174.0,
            "unit_bone_extremity_jerk_window_p95_max_s3": 11409.0,
        }
        return payload

    @staticmethod
    def _candidate():
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
            "foot_penetration_p01_m": -0.02,
            "foot_penetration_p001_m": -0.10515625424683094,
            "foot_penetration_catastrophic_threshold_m": -0.18,
            "foot_penetration_catastrophic_run_max_seconds": 0.0,
            "foot_penetration_min_m": -0.12,
            "root_y_robust_range_m": 0.0,
            "root_vertical_speed_mps_p95": 0.0,
            "root_vertical_speed_mps_max": 0.0,
            "unit_bone_joint_jerk_s3_p95": 1724.0,
            "unit_bone_joint_jerk_s3_p99": 3870.0,
            "unit_bone_joint_jerk_window_p95_max_s3": 6317.0,
            "unit_bone_extremity_jerk_s3_p95": 2569.0,
            "unit_bone_extremity_jerk_s3_p99": 6516.0,
            "unit_bone_extremity_jerk_window_p95_max_s3": 16122.0,
        }
        for spec in physical_metric_specs(PhysicalQualityLimits()):
            if spec.layer == "rotation_quality":
                payload[spec.key] = 0.0
        return payload

    def test_source_noise_floor_accepts_low_dynamic_reference(self):
        gate = evaluate_source_physical_clean_audit(
            self._candidate(),
            source_reference_audit=self._reference(),
        )
        jerk_reasons = [
            reason for reason in gate["reasons"] if "unit_bone" in reason
        ]
        self.assertEqual(jerk_reasons, [])
        detail = gate["relative_checks"]["unit_bone_joint_jerk_s3_p95"]
        self.assertEqual(detail["source_only_noise_floor_s3"], 1900.0)
        self.assertEqual(
            detail["semantics"],
            "source_relative_plus_source_only_noise_floor",
        )

    def test_noise_floor_does_not_hide_large_window_burst(self):
        candidate = self._candidate()
        candidate["unit_bone_extremity_jerk_window_p95_max_s3"] = 30430.0
        gate = evaluate_source_physical_clean_audit(
            candidate,
            source_reference_audit=self._reference(),
        )
        self.assertIn(
            "unit_bone_extremity_jerk_window_regressed_vs_source",
            gate["reasons"],
        )

    def test_p001_two_mm_epsilon_accepts_quantile_boundary(self):
        reference = self._reference()
        candidate = self._candidate()
        reference["foot_penetration_p001_m"] = -0.043868622899055486
        candidate["foot_penetration_p001_m"] = -0.10515625424683094
        gate = evaluate_source_physical_clean_audit(
            candidate,
            source_reference_audit=reference,
        )
        self.assertNotIn(
            "foot_penetration_p001_regressed_vs_source",
            gate["reasons"],
        )
        detail = gate["relative_checks"]["foot_penetration_p001_m"]
        self.assertAlmostEqual(detail["comparison_epsilon_m"], 0.002)
        self.assertAlmostEqual(
            detail["effective_allowed_minimum"],
            -0.10586862289905549,
        )

    def test_p001_epsilon_does_not_hide_real_regression(self):
        reference = self._reference()
        candidate = self._candidate()
        reference["foot_penetration_p001_m"] = -0.07414967691898344
        candidate["foot_penetration_p001_m"] = -0.15103378051519395
        gate = evaluate_source_physical_clean_audit(
            candidate,
            source_reference_audit=reference,
        )
        self.assertIn(
            "foot_penetration_p001_regressed_vs_source",
            gate["reasons"],
        )

    def test_final_limits_are_bitwise_default_constants(self):
        limits = PhysicalQualityLimits()
        self.assertEqual(limits.joint_jerk_mps3_p95, 810.0)
        self.assertEqual(limits.joint_jerk_mps3_max, 1620.0)
        self.assertEqual(limits.joint_jerk_window_p95_max_mps3, 1080.0)
        self.assertEqual(limits.foot_penetration_min_m, -0.05)
        self.assertEqual(limits.joint_rotation_step_rad_max, 1.20)

    def test_calibration_is_source_policy_only(self):
        policy = SourcePhysicalQualityPolicy()
        self.assertEqual(policy.unit_bone_jerk_p95_floor_s3, 1900.0)
        self.assertEqual(policy.unit_bone_jerk_p99_floor_s3, 4500.0)
        self.assertEqual(policy.unit_bone_jerk_window_floor_s3, 11000.0)
        self.assertEqual(policy.unit_bone_extremity_jerk_p95_floor_s3, 2800.0)
        self.assertEqual(policy.unit_bone_extremity_jerk_p99_floor_s3, 7500.0)
        self.assertEqual(
            policy.unit_bone_extremity_jerk_window_floor_s3,
            22000.0,
        )
        self.assertEqual(
            policy.foot_penetration_p001_comparison_epsilon_m,
            0.002,
        )


if __name__ == "__main__":
    unittest.main()
