import numpy as np

from contracts.boundary_continuity import evaluate_boundary_continuity
from contracts.physical_quality import evaluate_physical_audit
from evaluation.motion_activity_analysis import (
    ActivityThresholds,
    evaluate_final_motion_activity,
)
from contracts.gravity import fk24_np
from motion_geometry.physical import motion_physical_metrics_np
from motion_geometry.rotations import matrix_to_rot6d_np, so3_exp_np
from motion_geometry.smpl24 import MOTION_DIM
from support.transition_quality import _boundary_jerk_regions, transition_risk


def _identity_motion(frames: int) -> np.ndarray:
    motion = np.zeros((frames, MOTION_DIM), dtype=np.float32)
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (frames, 24, 3, 3),
    ).copy()
    motion[:, 7:151] = matrix_to_rot6d_np(identity).reshape(frames, -1)
    return motion


def _alternating_shoulder_jitter(frames: int, angle: float = 0.015) -> np.ndarray:
    motion = _identity_motion(frames)
    matrices = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (frames, 24, 3, 3),
    ).copy()
    signed = np.where(np.arange(frames) % 2 == 0, angle, -angle).astype(
        np.float32
    )
    rotation = so3_exp_np(
        np.stack(
            [np.zeros_like(signed), np.zeros_like(signed), signed],
            axis=-1,
        )
    )
    matrices[:, 16] = rotation
    matrices[:, 17] = rotation
    motion[:, 7:151] = matrix_to_rot6d_np(matrices).reshape(frames, -1)
    return motion


def _safe_boundary_risk() -> dict[str, float]:
    return {
        "boundary_joint_jerk_max": 100.0,
        "entry_fk_jump": 0.005,
        "exit_fk_jump": 0.005,
        "entry_fk_jump_max_m": 0.010,
        "exit_fk_jump_max_m": 0.010,
        "entry_rotation_step_rad": 0.02,
        "exit_rotation_step_rad": 0.02,
        "foot_slip": 0.01,
        "foot_slip_p95": 0.02,
        "foot_slip_max": 0.03,
        "foot_penetration": 0.0001,
        "foot_penetration_max_m": 0.005,
    }


def test_alternating_jitter_is_not_accepted_as_sustained_motion():
    motion = _alternating_shoulder_jitter(180)
    thresholds = ActivityThresholds(
        final_max_static_ratio=1.0,
        final_max_static_seconds=999.0,
        final_min_joint_speed_rad_s=0.0,
        final_min_root_travel_per_second_m_s=0.0,
        final_min_fk_visible_joint_speed_m_s=0.0,
        final_max_low_amplitude_window_ratio=0.60,
        sustained_motion_filter_seconds=0.20,
    )

    report = evaluate_final_motion_activity(
        motion,
        fps=30.0,
        thresholds=thresholds,
    )

    assert report["whole_sequence"]["fk_visible_joint_raw_speed_top4_mean_m_s"] > 0.05
    assert report["window_motion_amplitude_gate_failed"] is True
    assert report["ok"] is False


def test_local_extremity_jitter_trips_windowed_anti_jitter_gate():
    report = motion_physical_metrics_np(
        _alternating_shoulder_jitter(180, angle=0.03),
        fps=30.0,
    )
    decision = evaluate_physical_audit(report)

    assert report["joint_jerk_window_p95_max_mps3"] > 1080.0
    assert decision["layers"]["anti_jitter"]["ok"] is False


def test_stable_identity_motion_passes_all_physical_layers():
    report = motion_physical_metrics_np(_identity_motion(180), fps=30.0)
    decision = evaluate_physical_audit(report)

    assert decision["ok"] is True, decision["reasons"]
    assert all(layer["ok"] for layer in decision["layers"].values())


def test_precomputed_fk_preserves_every_physical_metric():
    motion = _alternating_shoulder_jitter(90, angle=0.01)
    expected = motion_physical_metrics_np(motion, fps=30.0)
    actual = motion_physical_metrics_np(
        motion,
        fps=30.0,
        precomputed_joints=fk24_np(motion),
    )

    assert actual == expected


def test_raw_rot6d_degeneracy_is_not_hidden_by_so3_projection():
    motion = _identity_motion(60)
    motion[20, 7:13] = 0.0

    report = motion_physical_metrics_np(motion, fps=30.0)
    decision = evaluate_physical_audit(report)

    assert report["rot6d_degenerate_ratio"] > 0.0
    assert decision["layers"]["rotation_quality"]["ok"] is False
    assert "rot6d_degenerate_ratio_too_high" in decision["reasons"]


def test_whole_sequence_so3_step_gate_catches_non_boundary_snap():
    motion = _identity_motion(90)
    matrices = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (90, 24, 3, 3),
    ).copy()
    matrices[45:, 20] = so3_exp_np(
        np.asarray([[0.0, 0.0, 1.8]], dtype=np.float32)
    )[0]
    motion[:, 7:151] = matrix_to_rot6d_np(matrices).reshape(90, -1)

    report = motion_physical_metrics_np(motion, fps=30.0)
    decision = evaluate_physical_audit(report)

    assert report["joint_rotation_step_rad_max"] > 1.2
    assert decision["layers"]["rotation_quality"]["ok"] is False


def test_root_vertical_metrics_are_required_fail_closed():
    report = motion_physical_metrics_np(_identity_motion(60), fps=30.0)
    del report["root_vertical_speed_mps_max"]

    decision = evaluate_physical_audit(report)

    assert decision["ok"] is False
    assert "missing_or_nonfinite:root_vertical_speed_mps_max" in decision["reasons"]


def test_fast_low_foot_slide_cannot_hide_by_clearing_contact_labels():
    fps = 30.0
    motion = _identity_motion(61)
    motion[:, :4] = 0.0
    motion[:, 4] = 0.70 * np.arange(len(motion), dtype=np.float32) / fps

    report = motion_physical_metrics_np(motion, fps=fps)
    decision = evaluate_physical_audit(report)

    assert report["contact_ratio"] == 0.0
    assert report["foot_support_ratio"] > 0.0
    assert report["foot_skate_mps_p95"] > 0.60
    assert report["foot_support_drift_m_max"] > 0.12
    assert decision["layers"]["foot_contact"]["ok"] is False


def test_explicit_chang_e_slide_requires_semantics_and_root_consistency():
    fps = 30.0
    motion = _identity_motion(61)
    motion[:, :4] = 1.0
    motion[:, 4] = 0.20 * np.arange(len(motion), dtype=np.float32) / fps

    conservative = motion_physical_metrics_np(motion, fps=fps)
    semantic = motion_physical_metrics_np(
        motion,
        fps=fps,
        sliding_support_eligible=np.ones(len(motion), dtype=bool),
    )

    assert conservative["sliding_support_ratio"] == 0.0
    assert conservative["foot_skate_mps_p95"] > 0.18
    assert semantic["sliding_support_ratio"] > 0.0
    assert semantic["foot_skate_mps_p95"] == 0.0
    assert semantic["support_state_contract"][
        "sliding_requires_explicit_semantic_eligibility"
    ] is True


def test_boundary_foot_slip_proxy_is_also_speed_independent():
    fps = 30.0
    motion = _identity_motion(16)
    motion[:, :4] = 0.0
    motion[:, 4] = 0.70 * np.arange(len(motion), dtype=np.float32) / fps

    risk = transition_risk(
        motion[:4],
        motion[4:12],
        motion[12:],
        fps=fps,
    )

    assert risk["foot_slip"] > 0.60


def test_boundary_foot_slip_includes_bridge_entry_and_exit_steps():
    fps = 30.0
    previous = _identity_motion(4)
    transition = _identity_motion(3)
    following = _identity_motion(4)
    transition[:, 4] = 0.10
    following[:, 4] = 0.10

    risk = transition_risk(previous, transition, following, fps=fps)

    assert risk["foot_slip_p95"] > 2.0
    assert risk["foot_slip_max"] > 2.0


def test_direct_join_is_measured_instead_of_returning_a_sentinel():
    motion = _identity_motion(8)

    risk = transition_risk(
        motion[:4],
        np.zeros((0, MOTION_DIM), dtype=np.float32),
        motion[4:],
        fps=30.0,
    )

    assert np.isfinite(risk["total"])
    assert risk["total"] < 1.0e8
    assert risk["entry_fk_jump_max_m"] == 0.0
    assert risk["foot_slip_max"] == 0.0


def test_boundary_peak_metrics_cannot_hide_behind_safe_means():
    risk = _safe_boundary_risk()
    risk["entry_fk_jump"] = 0.005
    risk["entry_fk_jump_max_m"] = 0.080
    risk["foot_slip"] = 0.01
    risk["foot_slip_p95"] = 0.05
    risk["foot_slip_max"] = 0.50

    decision = evaluate_boundary_continuity(
        [{"slot": 1, "risk": risk}],
        expected_boundaries=1,
    )

    assert decision["ok"] is False
    assert any("entry_fk_joint_jump_max_m_too_high" in x for x in decision["reasons"])
    assert any("foot_slip_peak_mps_too_high" in x for x in decision["reasons"])


def test_boundary_jerk_uses_true_joint_max_not_joint_average():
    jerk = np.zeros((5, 24, 3), dtype=np.float32)
    jerk[2, 20, 0] = 700.0

    entry, exit_, maximum = _boundary_jerk_regions(
        jerk,
        left_count=3,
        transition_count=1,
    )

    assert maximum == 700.0
    assert max(entry, exit_) == 700.0


def test_long_horizon_horizontal_root_drift_is_rejected():
    fps = 30.0
    motion = _identity_motion(3601)
    motion[:, 4] = np.linspace(0.0, 4.0, len(motion), dtype=np.float32)

    report = motion_physical_metrics_np(motion, fps=fps)
    decision = evaluate_physical_audit(report)

    assert report["root_horizontal_net_displacement_m"] == 4.0
    assert decision["layers"]["long_horizon_root_drift"]["ok"] is False
    assert "root_horizontal_net_displacement_m_too_high" in decision["reasons"]


def test_boundary_gate_requires_every_metric_on_every_seam():
    safe = evaluate_boundary_continuity(
        [{"slot": 1, "risk": _safe_boundary_risk()}],
        expected_boundaries=1,
    )
    assert safe["ok"] is True

    unsafe_risk = _safe_boundary_risk()
    unsafe_risk["entry_fk_jump"] = 0.03
    unsafe = evaluate_boundary_continuity(
        [{"slot": 1, "risk": unsafe_risk}],
        expected_boundaries=1,
    )
    assert unsafe["ok"] is False
    assert unsafe["unsafe_boundaries"] == 1

    missing = evaluate_boundary_continuity(
        [{"slot": 1, "risk": {"boundary_joint_jerk_max": 1.0}}],
        expected_boundaries=1,
    )
    assert missing["ok"] is False
    assert any("missing_or_nonfinite" in reason for reason in missing["reasons"])


def test_boundary_gate_rejects_incomplete_audit_coverage():
    report = evaluate_boundary_continuity([], expected_boundaries=2)
    assert report["ok"] is False
    assert report["reasons"] == ["boundary_audit_count_mismatch:0!=2"]
