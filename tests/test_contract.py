import unittest

from contracts.anatomy import (
    AnatomyThresholds,
    SourceAnatomyThresholds,
    evaluate_anatomy_contract_detailed,
    evaluate_source_anatomy_contract,
)


BASE = {
    "nonfinite_count": 0,
    "rot_orthogonality_p95": 0.0,
    "rot_det_abs_error_p95": 0.0,
    "local_angle_violation_ratio": 0.07,
    "local_angle_severe_ratio": 0.001,
    "spine_cumulative_angle_p95_rad": 1.0,
    "torso_compression_ratio_p01": 0.7,
    "torso_compression_ratio_p05": 0.7,
    "neck_compression_ratio_p01": 0.7,
    "neck_compression_ratio_p05": 0.7,
    "self_collision_severe_ratio": 0.0,
    "knee_collapse_ratio": 0.0,
    "elbow_collapse_ratio": 0.0,
    "foot_penetration_p01_m": 0.0,
    "bone_length_drift_max": 0.0,
    "anatomy_quality": 0.8,
}


class ContractSeparationTests(unittest.TestCase):
    def test_mild_event_limit_is_soft_not_catastrophic(self):
        detail = evaluate_anatomy_contract_detailed(BASE, AnatomyThresholds())
        self.assertTrue(detail["hard_ok"])
        self.assertFalse(detail["soft_ok"])

    def test_source_gate_ignores_mild_limit_ratio(self):
        ok, reasons = evaluate_source_anatomy_contract(BASE, SourceAnatomyThresholds())
        self.assertTrue(ok, reasons)

    def test_source_gate_rejects_severe_rotation(self):
        bad = dict(BASE)
        bad["local_angle_severe_ratio"] = 0.05
        ok, reasons = evaluate_source_anatomy_contract(bad, SourceAnatomyThresholds())
        self.assertFalse(ok)
        self.assertTrue(any("local_angle_severe_ratio" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
