"""Development contract checks for V9 full-sequence contact repair."""

import numpy as np

from motion_geometry.rotations import matrix_to_rot6d_np
from motion_geometry.smpl24 import MOTION_DIM
from training.motion_models import (
    MotionGenerationConfig,
    _contact_restoration_decision,
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


def test_v9_is_development_opt_in_and_keeps_gate_values():
    cfg = MotionGenerationConfig()
    assert cfg.full_sequence_contact_repair_enable is False
    diagnostics = full_sequence_physical_diagnostics_np(
        _identity_motion(30),
        cfg,
        sliding_support_eligible=np.zeros(30, dtype=bool),
    )
    assert diagnostics["schema"] == "full_sequence_physical_localization_v9"
    assert diagnostics["support_contract"] == (
        "final_fail_closed_with_sliding_eligibility"
    )
    assert diagnostics["audit"]["foot_skate_mps_p95"] == 0.0


def test_v9_restoration_requires_a_real_dominant_contact_gain():
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


def test_v9_fixed_support_gate_uses_the_captured_eligibility_contract():
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
