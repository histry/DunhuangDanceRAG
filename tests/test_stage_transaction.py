import numpy as np

from contracts.physical_quality import (
    PhysicalQualityLimits,
    StageAcceptancePolicy,
    evaluate_stage_candidate,
    run_stage_transaction,
)


def _audit_for_level(level: float):
    return {
        "foot_skate_mps_p95": 0.05,
        "foot_skate_mps_max": 0.10,
        "foot_penetration_min_m": -0.001,
        "joint_jerk_mps3_p95": 235.0 if level < 2.0 else 509.0,
        "joint_jerk_mps3_max": 1702.0 if level < 2.0 else 5179.0,
        "root_y_robust_range_m": 0.1,
        "root_vertical_speed_mps_p95": 0.1,
        "root_vertical_speed_mps_max": 0.2,
    }


def test_refiner_regression_is_rejected():
    before = _audit_for_level(1.0)
    candidate = _audit_for_level(3.0)

    decision = evaluate_stage_candidate(
        before,
        candidate,
        limits=PhysicalQualityLimits(),
        policy=StageAcceptancePolicy(),
    )

    assert decision["accepted"] is False
    assert "joint_jerk_max_regressed" in decision["reasons"]


def test_transaction_rolls_back_to_snapshot():
    snapshot = np.ones((4, 3), dtype=np.float32)

    def apply_fn(value):
        return value * 3.0

    def audit_fn(value):
        return _audit_for_level(float(value.mean()))

    selected, report = run_stage_transaction(
        stage_name="refiner",
        motion=snapshot,
        apply_fn=apply_fn,
        audit_fn=audit_fn,
        limits=PhysicalQualityLimits(),
        policy=StageAcceptancePolicy(),
    )

    assert report["accepted"] is False
    assert report["rolled_back"] is True
    np.testing.assert_allclose(selected, snapshot)


def test_diffusion_requires_meaningful_gain_when_input_fails():
    before = _audit_for_level(1.0)
    candidate = dict(before)
    candidate["joint_jerk_mps3_max"] = 1702.1

    decision = evaluate_stage_candidate(
        before,
        candidate,
        limits=PhysicalQualityLimits(),
        policy=StageAcceptancePolicy(minimum_repair_gain=0.03),
        require_repair_gain=True,
    )

    assert decision["accepted"] is False
    assert "no_meaningful_repair_gain" in decision["reasons"]
