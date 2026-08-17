from contracts.boundary_continuity import BoundaryContinuityLimits
from support.transition_quality import accept_candidate


def _safe_risk():
    return {
        "total": 0.1,
        "entry_velocity": 0.01,
        "exit_velocity": 0.01,
        "joint_jerk": 1.0,
        "angular_jerk": 1.0,
        "foot_slip": 0.01,
        "foot_slip_p95": 0.02,
        "foot_slip_max": 0.03,
        "foot_penetration": 0.0,
        "foot_penetration_max_m": 0.0,
        "boundary_joint_jerk_max": 10.0,
        "entry_fk_jump": 0.005,
        "exit_fk_jump": 0.005,
        "entry_fk_jump_max_m": 0.01,
        "exit_fk_jump_max_m": 0.01,
        "entry_rotation_step_rad": 0.01,
        "exit_rotation_step_rad": 0.01,
    }


def test_transition_candidate_uses_final_fail_closed_absolute_limits():
    baseline = _safe_risk()
    candidate = dict(baseline)
    candidate["entry_fk_jump"] = 0.020

    accepted, report = accept_candidate(
        baseline,
        candidate,
        boundary_limits=BoundaryContinuityLimits(),
    )

    assert accepted is False
    assert report["checks"]["absolute_boundary_continuity"] is False
    assert any("entry_fk_jump_m_too_high" in reason for reason in report["absolute_reasons"])
