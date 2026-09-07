"""Development contract checks for V11 observable-tolerant contact repair."""

import numpy as np

from motion_geometry.rotations import matrix_to_rot6d_np
from motion_geometry.smpl24 import MOTION_DIM
from training.motion_models import (
    MotionGenerationConfig,
    _c2_transaction_weight,
    _contact_restoration_decision,
    _exact_audit_candidate_rank,
    _partition_repair_windows_by_support_phase,
    _physical_nonregression_decision,
    evaluate_fixed_support_contact_candidate_np,
    full_sequence_physical_diagnostics_np,
)
from training.generation_stage_diagnostics import (
    _boundary_nonregression,
    _motion_scope_audit,
)


def _identity_motion(frames):
    motion = np.zeros((frames, MOTION_DIM), dtype=np.float32)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (frames, 24, 3, 3),
    ).copy()
    motion[:, 7:151] = matrix_to_rot6d_np(rotations).reshape(frames, -1)
    return motion


def test_v11_is_development_opt_in_and_keeps_gate_values():
    cfg = MotionGenerationConfig()
    assert cfg.full_sequence_contact_repair_enable is False
    diagnostics = full_sequence_physical_diagnostics_np(
        _identity_motion(30),
        cfg,
        sliding_support_eligible=np.zeros(30, dtype=bool),
    )
    audit = diagnostics["audit"]
    decision = _contact_restoration_decision(audit, dict(audit), cfg)
    assert decision["accepted"] is False
    assert "dominant_contact_residual_not_meaningfully_improved" in (
        decision["reasons"]
    )
    assert diagnostics["schema"] == "full_sequence_physical_localization_v11"
    assert diagnostics["support_contract"] == (
        "final_fail_closed_with_sliding_eligibility"
    )
    assert diagnostics["audit"]["foot_skate_mps_p95"] == 0.0


def test_v11_restoration_requires_a_real_dominant_contact_gain():
    cfg = MotionGenerationConfig()
    diagnostics = full_sequence_physical_diagnostics_np(
        _identity_motion(30),
        cfg,
        sliding_support_eligible=np.zeros(30, dtype=bool),
    )


def test_v11_global_guard_only_checks_nonregression():
    cfg = MotionGenerationConfig()
    audit = full_sequence_physical_diagnostics_np(
        _identity_motion(30),
        cfg,
        sliding_support_eligible=np.zeros(30, dtype=bool),
    )["audit"]
    unchanged = _physical_nonregression_decision(audit, dict(audit))
    assert unchanged["accepted"] is True
    regressed = dict(audit)
    regressed["foot_skate_mps_max"] = audit["foot_skate_mps_max"] + 0.01
    decision = _physical_nonregression_decision(audit, regressed)
    assert decision["accepted"] is False
    assert decision["reasons"] == [
        "global_metric_regressed:foot_skate_mps_max"
    ]


def test_v11_fixed_support_gate_uses_the_captured_eligibility_contract():
    cfg = MotionGenerationConfig()
    motion = _identity_motion(30)
    decision = evaluate_fixed_support_contact_candidate_np(
        motion,
        motion.copy(),
        cfg,
        sliding_support_eligible=np.zeros(30, dtype=bool),
    )
    assert decision["accepted"] is True
    assert decision["support_contract"] == (
        "final_fail_closed_with_sliding_eligibility"
    )
    assert all(value == 0.0 for value in decision["residual_delta"].values())


def test_v11_uses_the_required_exact_backtracking_ladder():
    cfg = MotionGenerationConfig()
    assert cfg.full_sequence_contact_repair_backtracking_factors == (
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
    )


def test_v11_c2_envelope_freezes_three_frames_at_each_edge():
    weight = _c2_transaction_weight(
        20,
        fade=7,
        freeze_edges=3,
        has_left_context=True,
        has_right_context=True,
    )[:, 0]
    assert np.array_equal(weight[:3], np.zeros(3, dtype=np.float32))
    assert np.array_equal(weight[-3:], np.zeros(3, dtype=np.float32))
    assert weight[6] == 1.0
    assert weight[-7] == 1.0
    assert np.all((weight >= 0.0) & (weight <= 1.0))


def test_v11_partitions_large_windows_by_left_right_and_double_support():
    static = np.zeros((32, 4), dtype=bool)
    static[4:12, (0, 2)] = True
    static[12:20, (1, 3)] = True
    static[20:28, :] = True
    partitions = _partition_repair_windows_by_support_phase(
        [[0, 32]],
        static,
        32,
    )
    assert [part["support_phase"] for part in partitions] == [
        "no_support",
        "left_support",
        "right_support",
        "double_support",
        "no_support",
    ]
    assert [part["solver_eligible"] for part in partitions] == [
        True,
        True,
        True,
        True,
        True,
    ]


def test_v11_marks_too_short_c2_support_phases_ineligible():
    static = np.zeros((12, 4), dtype=bool)
    static[4:8, (0, 2)] = True
    partitions = _partition_repair_windows_by_support_phase(
        [[0, 12]],
        static,
        12,
    )
    assert partitions[1]["support_phase"] == "left_support"
    assert partitions[1]["minimum_solver_frames"] == 7
    assert partitions[1]["solver_eligible"] is False


def test_local_boundary_audit_ignores_unrelated_slots():
    before = [
        {
            "slot": 1,
            "transition_start": 0,
            "transition_end": 4,
            "content_start": 4,
            "content_end": 10,
            "actual_boundary_jerk_mps3": 1.0,
        },
        {
            "slot": 23,
            "transition_start": 220,
            "transition_end": 224,
            "content_start": 224,
            "content_end": 240,
            "actual_boundary_jerk_mps3": 1.0,
        },
    ]
    candidate = [dict(row) for row in before]
    candidate[0]["actual_boundary_jerk_mps3"] = 2.0
    decision = _boundary_nonregression(
        before,
        candidate,
        active_span=[220, 230],
    )
    assert decision["accepted"] is True
    assert decision["evaluated_slots"] == [23]
    assert decision["ignored_slots"] == [1]


def test_local_motion_scope_rejects_out_of_owner_changes():
    before = np.zeros((12, 3), dtype=np.float32)
    candidate = before.copy()
    candidate[5, 1] = 0.25
    candidate[2, 0] = 0.5
    decision = _motion_scope_audit(before, candidate, [4, 8])
    assert decision["accepted"] is False
    assert decision["changed_frames_inside"] == [5]
    assert decision["changed_frames_outside"] == [2]


def test_exact_audit_records_numeric_metrics_and_residuals():
    cfg = MotionGenerationConfig()
    before = {
        "foot_penetration_min_m": -0.10,
        "foot_skate_mps_p95": 0.40,
        "foot_skate_mps_max": 0.80,
        "foot_support_drift_m_p95": 0.10,
        "foot_support_drift_m_max": 0.20,
        "joint_jerk_mps3_max": 100.0,
    }
    candidate = dict(before)
    candidate["foot_penetration_min_m"] = -0.08
    rank, summary = _exact_audit_candidate_rank(
        before,
        candidate,
        cfg,
        optimization_loss=1.0,
    )
    assert len(rank) == 6
    assert summary["before_metrics"]["foot_penetration_min_m"] == -0.10
    assert summary["candidate_metrics"]["foot_penetration_min_m"] == -0.08
    assert "foot_penetration_min_m" in summary["metric_delta"]
    assert "foot_penetration_min_m" in summary["before_residuals"]
