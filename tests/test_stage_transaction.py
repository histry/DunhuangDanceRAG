import numpy as np

from contracts.physical_quality import (
    PhysicalQualityLimits,
    StageAcceptancePolicy,
    evaluate_stage_candidate,
    evaluate_stage_reference_fidelity,
    run_stage_transaction,
)
from motion_geometry.physical import motion_physical_metrics_np
from motion_geometry.rotations import matrix_to_rot6d_np


def _audit_for_level(level: float):
    motion = np.zeros((8, 151), dtype=np.float32)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (8, 24, 3, 3),
    )
    motion[:, 7:151] = matrix_to_rot6d_np(rotations).reshape(8, -1)
    audit = motion_physical_metrics_np(motion, fps=30.0)
    audit.update({
        "foot_skate_mps_p95": 0.05,
        "foot_skate_mps_max": 0.10,
        "foot_penetration_min_m": -0.001,
        "joint_jerk_mps3_p95": 235.0 if level < 2.0 else 509.0,
        "joint_jerk_mps3_max": 1702.0 if level < 2.0 else 5179.0,
        "root_y_robust_range_m": 0.1,
        "root_vertical_speed_mps_p95": 0.1,
        "root_vertical_speed_mps_max": 0.2,
    })
    return audit


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


def test_stage_cannot_worsen_an_already_failing_upper_bound():
    before = _audit_for_level(1.0)
    before["foot_skate_mps_max"] = 0.70
    candidate = dict(before)
    candidate["foot_skate_mps_max"] = 0.71

    decision = evaluate_stage_candidate(before, candidate)

    assert decision["accepted"] is False
    assert "foot_skate_max_regressed" in decision["reasons"]


def test_stage_can_incrementally_repair_an_already_failing_lower_bound():
    before = _audit_for_level(1.0)
    before["foot_penetration_min_m"] = -0.08
    candidate = dict(before)
    candidate["foot_penetration_min_m"] = -0.06

    decision = evaluate_stage_candidate(before, candidate)

    assert decision["accepted"] is True, decision["reasons"]


def test_stage_policy_covers_window_extremity_support_and_root_drift_metrics():
    before = _audit_for_level(1.0)
    candidate = dict(before)
    candidate["joint_jerk_window_p95_max_mps3"] = 5000.0
    candidate["extremity_jerk_window_p95_max_mps3"] = 5000.0
    candidate["foot_support_drift_m_max"] = 1.0
    candidate["root_horizontal_drift_speed_mps"] = 1.0
    candidate["joint_rotation_step_rad_max"] = 2.0

    decision = evaluate_stage_candidate(before, candidate)

    assert decision["accepted"] is False
    assert "joint_jerk_window_p95_max_mps3_regressed" in decision["reasons"]
    assert "extremity_jerk_window_p95_max_mps3_regressed" in decision["reasons"]
    assert "foot_support_drift_m_max_regressed" in decision["reasons"]
    assert "root_horizontal_drift_speed_mps_regressed" in decision["reasons"]
    assert "joint_rotation_step_rad_max_regressed" in decision["reasons"]


def test_checkpoint_stage_can_explicitly_ignore_short_event_root_travel():
    before = _audit_for_level(1.0)
    before["root_horizontal_radius_p95_m"] = 3.0
    candidate = dict(before)
    candidate["root_horizontal_radius_p95_m"] = 3.01

    strict = evaluate_stage_candidate(before, candidate)
    checkpoint = evaluate_stage_candidate(
        before,
        candidate,
        ignored_layers=("long_horizon_root_drift",),
    )

    assert strict["accepted"] is False
    assert "root_horizontal_radius_p95_m_regressed" in strict["reasons"]
    assert checkpoint["accepted"] is True, checkpoint["reasons"]
    assert checkpoint["ignored_layers"] == ["long_horizon_root_drift"]


def test_clean_reference_fidelity_is_relative_not_an_absolute_final_gate():
    reference = _audit_for_level(1.0)
    reference["root_horizontal_radius_p95_m"] = 3.0

    faithful = evaluate_stage_reference_fidelity(reference, dict(reference))
    regressed = dict(reference)
    regressed["root_horizontal_radius_p95_m"] = 3.5
    rejected = evaluate_stage_reference_fidelity(reference, regressed)

    assert faithful["accepted"] is True, faithful["reasons"]
    assert faithful["absolute_final_gate_used"] is False
    assert rejected["accepted"] is False
    assert (
        "reference_fidelity_root_horizontal_radius_p95_m_regressed"
        in rejected["reasons"]
    )


def test_stage_candidate_tolerates_only_float_round_trip_noise():
    before = _audit_for_level(1.0)
    candidate = dict(before)
    candidate["joint_jerk_mps3_max"] = (
        before["joint_jerk_mps3_max"] * (1.0 + 5.0e-7)
    )
    rounding = evaluate_stage_candidate(before, candidate)
    assert rounding["accepted"] is True, rounding["reasons"]

    candidate["joint_jerk_mps3_max"] = before["joint_jerk_mps3_max"] + 1.0
    regression = evaluate_stage_candidate(before, candidate)
    assert regression["accepted"] is False
    assert "joint_jerk_max_regressed" in regression["reasons"]
