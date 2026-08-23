import numpy as np

from contracts.physical_quality import (
    evaluate_physical_audit,
    evaluate_pretraining_route_audit,
)
from motion_geometry.physical import motion_physical_metrics_np
from motion_geometry.rotations import matrix_to_rot6d_np
from motion_geometry.smpl24 import MOTION_DIM
from scripts.run_no_training_regression import evaluate_scheduler_motion_tensor


def _identity_motion(frames: int = 180) -> np.ndarray:
    motion = np.zeros((frames, MOTION_DIM), dtype=np.float32)
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (frames, 24, 3, 3),
    ).copy()
    motion[:, 7:151] = matrix_to_rot6d_np(identity).reshape(frames, -1)
    return motion


def _audit() -> dict:
    return motion_physical_metrics_np(_identity_motion(), fps=30.0)


def test_unrefined_jitter_skate_and_horizontal_drift_are_diagnostic_only():
    audit = _audit()
    audit.update(
        {
            "joint_jerk_mps3_max": 52000.0,
            "joint_jerk_window_p95_max_mps3": 24000.0,
            "extremity_jerk_window_p95_max_mps3": 24000.0,
            "foot_skate_mps_p95": 0.62,
            "foot_skate_mps_max": 29.0,
            "foot_support_drift_m_p95": 0.74,
            "foot_support_drift_m_max": 1.08,
            "root_horizontal_radius_p95_m": 2.48,
            "root_horizontal_radius_max_m": 2.50,
            "root_horizontal_window_displacement_max_m": 1.58,
        }
    )

    final_gate = evaluate_physical_audit(audit)
    pretraining_gate = evaluate_pretraining_route_audit(audit)

    assert final_gate["ok"] is False
    assert pretraining_gate["ok"] is True, pretraining_gate["reasons"]
    assert "joint_jerk_mps3_max_too_high" in pretraining_gate[
        "diagnostic_only_reasons"
    ]
    assert pretraining_gate[
        "final_generation_gate_required_after_motion_repair"
    ] is True


def test_rotation_integrity_remains_blocking_before_training():
    audit = _audit()
    audit["rot6d_degenerate_ratio"] = 0.01

    gate = evaluate_pretraining_route_audit(audit)

    assert gate["ok"] is False
    assert "rot6d_degenerate_ratio_too_high" in gate["reasons"]


def test_root_vertical_safety_remains_blocking_before_training():
    audit = _audit()
    audit["root_vertical_speed_mps_max"] = 8.0

    gate = evaluate_pretraining_route_audit(audit)

    assert gate["ok"] is False
    assert "root_vertical_speed_mps_max_too_high" in gate["reasons"]


def test_sustained_catastrophic_penetration_remains_blocking():
    audit = _audit()
    audit["foot_penetration_catastrophic_run_max_seconds"] = 0.20

    gate = evaluate_pretraining_route_audit(audit)

    assert gate["ok"] is False
    assert "foot_penetration_sustained_catastrophic" in gate["reasons"]


def test_missing_pretraining_contract_metric_fails_closed():
    audit = _audit()
    del audit["foot_penetration_catastrophic_run_max_seconds"]

    gate = evaluate_pretraining_route_audit(audit)

    assert gate["ok"] is False
    assert (
        "missing_or_nonfinite:foot_penetration_catastrophic_run_max_seconds"
        in gate["reasons"]
    )


def test_scheduler_motion_tensor_contract_is_explicit_and_fail_closed():
    valid = _identity_motion(30)
    assert evaluate_scheduler_motion_tensor(valid)["ok"] is True

    nonfinite = valid.copy()
    nonfinite[4, 20] = np.nan
    decision = evaluate_scheduler_motion_tensor(nonfinite)
    assert decision["ok"] is False
    assert "motion_nonfinite_count:1" in decision["reasons"]

    batched = valid[None, ...]
    decision = evaluate_scheduler_motion_tensor(batched)
    assert decision["ok"] is False
    assert "motion_rank_mismatch:3!=2" in decision["reasons"]
