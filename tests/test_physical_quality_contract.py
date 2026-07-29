import numpy as np

from contracts.physical_quality import (
    PhysicalQualityLimits,
    compute_joint_kinematic_metrics,
    evaluate_physical_audit,
)


def test_true_joint_max_is_not_frame_mean_max():
    joints = np.zeros((8, 24, 3), dtype=np.float32)
    joints[4, 20, 0] = 0.10

    metrics = compute_joint_kinematic_metrics(joints, fps=30.0)

    assert metrics["joint_jerk_mps3_max"] > 1620.0
    assert (
        metrics["joint_jerk_mps3_max"]
        > metrics["frame_mean_jerk_mps3_max"]
    )


def test_final_gate_uses_true_si_jerk_limits():
    limits = PhysicalQualityLimits()
    audit = {
        "foot_skate_mps_p95": 0.01,
        "foot_skate_mps_max": 0.02,
        "foot_penetration_min_m": -0.001,
        "joint_jerk_mps3_p95": 200.0,
        "joint_jerk_mps3_max": 1700.0,
        "root_y_robust_range_m": 0.1,
        "root_vertical_speed_mps_p95": 0.1,
        "root_vertical_speed_mps_max": 0.2,
    }

    result = evaluate_physical_audit(audit, limits)

    assert result["ok"] is False
    assert "joint_jerk_mps3_max_too_high" in result["reasons"]
    assert result["limits"]["joint_jerk_mps3_max"] == 1620.0
