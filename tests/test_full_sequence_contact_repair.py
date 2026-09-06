"""Development contract checks for V10 exact-audit contact repair."""

import numpy as np

from motion_geometry.rotations import matrix_to_rot6d_np
from motion_geometry.smpl24 import MOTION_DIM
from training.motion_models import (
    MotionGenerationConfig,
    _c2_transaction_weight,
    _contact_restoration_decision,
    _partition_repair_windows_by_support_phase,
    evaluate_fixed_support_contact_candidate_np,
    full_sequence_physical_diagnostics_np,
)


def _identity_motion(frames):
    motion = np.zeros((frames, MOTION_DIM), dtype=np.float32)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (frames, 24, 3, 3),
    ).copy()
    motion[:, 7:151] = matrix_to_rot6d_np(rotations).reshape(frames, -1)
    return motion


def test_v10_is_development_opt_in_and_keeps_gate_values():
    cfg = MotionGenerationConfig()
    assert cfg.full_sequence_contact_repair_enable is False
    diagnostics = full_sequence_physical_diagnostics_np(
        _identity_motion(30),
        cfg,
        sliding_support_eligible=np.zeros(30, dtype=bool),
    )
    assert diagnostics["schema"] == "full_sequence_physical_localization_v10"
    assert diagnostics["support_contract"] == (
        "final_fail_closed_with_sliding_eligibility"
    )
    assert diagnostics["audit"]["foot_skate_mps_p95"] == 0.0


def test_v10_restoration_requires_a_real_dominant_contact_gain():
    cfg = MotionGenerationConfig()
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


def test_v10_fixed_support_gate_uses_the_captured_eligibility_contract():
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


def test_v10_uses_the_required_exact_backtracking_ladder():
    cfg = MotionGenerationConfig()
    assert cfg.full_sequence_contact_repair_backtracking_factors == (
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
    )


def test_v10_c2_envelope_freezes_three_frames_at_each_edge():
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


def test_v10_partitions_large_windows_by_left_right_and_double_support():
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


def test_v10_marks_too_short_c2_support_phases_ineligible():
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
