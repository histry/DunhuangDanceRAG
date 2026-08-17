import numpy as np

from evaluation.motion_activity import (
    ActivityThresholds,
    candidate_activity_assessment,
    diagnose_motion,
    final_activity_gate,
    motion_activity_metrics,
    slot_activity_target,
)
from motion_geometry.rotations import matrix_to_rot6d_np


def _identity_motion(frames=90):
    motion = np.zeros((frames, 151), dtype=np.float32)
    matrices = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frames, 24, 3, 3)
    ).copy()
    motion[:, 7:151] = matrix_to_rot6d_np(matrices).reshape(frames, 144)
    return motion


def _active_motion(frames=90, fps=30.0):
    motion = _identity_motion(frames)
    angles = np.arange(frames, dtype=np.float32) * (0.45 / fps)
    c = np.cos(angles)
    s = np.sin(angles)
    matrices = np.zeros((frames, 24, 3, 3), dtype=np.float32)
    matrices[..., 1, 1] = 1.0
    matrices[..., 0, 0] = c[:, None]
    matrices[..., 0, 2] = s[:, None]
    matrices[..., 2, 0] = -s[:, None]
    matrices[..., 2, 2] = c[:, None]
    motion[:, 7:151] = matrix_to_rot6d_np(matrices).reshape(frames, 144)
    motion[:, 4] = np.arange(frames, dtype=np.float32) * (0.04 / fps)
    return motion


def test_static_and_active_motion_are_separated():
    static = motion_activity_metrics(_identity_motion(), fps=30.0)
    active = motion_activity_metrics(_active_motion(), fps=30.0)
    assert static["static_frame_ratio"] > 0.95
    assert active["static_frame_ratio"] < 0.10
    assert active["joint_speed_mean_rad_s"] > 0.35
    assert active["motion_density_mean"] > static["motion_density_mean"]


def test_pose_hold_probability_implies_activity_target():
    assert slot_activity_target(
        {"music_semantic_probs": {"pose_hold": 0.0}}
    ) == 1.0
    assert slot_activity_target(
        {"music_semantic_probs": {"pose_hold": 1.0}}
    ) < 0.1


def test_static_candidate_is_rejected_for_active_slot():
    metrics = motion_activity_metrics(_identity_motion(), fps=30.0)
    assessment = candidate_activity_assessment(
        metrics,
        target_activity=1.0,
        thresholds=ActivityThresholds(),
    )
    assert assessment["active_motion_required"]
    assert assessment["hard_reject"]
    assert len(assessment["reasons"]) >= 2


def test_final_gate_rejects_nearly_static_whole_song():
    metrics = {
        "static_frame_ratio": 0.73,
        "longest_static_streak_seconds": 5.0,
        "joint_speed_mean_rad_s": 0.065,
    }
    per_slot = [
        {
            "slot": index,
            "target_activity": 1.0,
            "static_frame_ratio": 0.85,
        }
        for index in range(8)
    ]
    gate = final_activity_gate(metrics, per_slot=per_slot)
    assert not gate["ok"]
    assert "high_activity_slots_collapsed" in gate["reasons"]


def test_diagnose_motion_reports_per_slot_alignment():
    motion = np.concatenate([_identity_motion(45), _active_motion(45)], axis=0)
    slots = [
        {"target_frames": 45, "music_semantic_probs": {"pose_hold": 1.0}},
        {"target_frames": 45, "music_semantic_probs": {"pose_hold": 0.0}},
    ]
    result = diagnose_motion(motion, fps=30.0, slots=slots)
    assert len(result["per_slot"]) == 2
    assert result["motion_density_alignment"]["available"]
    assert (
        result["per_slot"][1]["motion_density_mean"]
        > result["per_slot"][0]["motion_density_mean"]
    )
