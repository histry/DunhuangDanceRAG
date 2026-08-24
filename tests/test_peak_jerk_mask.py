import numpy as np

from motion_geometry.rotations import matrix_to_rot6d_np, so3_exp_np
from contracts.physical_quality import (
    PeakJerkMaskConfig,
    _expand_peak_jerk_pairs,
    build_peak_jerk_risk_mask,
    build_repair_mask,
)
from motion_geometry.smpl24 import NUM_JOINTS, PARENTS


def _identity_motion(frames: int = 20) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 7:151] = np.tile(identity, 24)
    return motion


def test_core_peak_jerk_extends_zero_seam_mask():
    motion = _identity_motion()
    motion[10, 4] = 0.10
    seam = np.zeros((len(motion), 1), dtype=np.float32)
    config = PeakJerkMaskConfig(
        enabled=True,
        absolute_threshold_mps3=1000.0,
        percentile=99.0,
        radius_frames_at_30fps=2,
        parent_depth=2,
    )

    peak = build_peak_jerk_risk_mask(
        motion,
        fps=30.0,
        config=config,
    )
    repair, report = build_repair_mask(
        motion,
        seam,
        fps=30.0,
        config=config,
    )

    assert peak["report"]["peak_count"] > 0
    assert peak["report"]["masked_frames"] > 0
    assert np.count_nonzero(repair[:, 0]) > 0
    assert report["seam_active_frames"] == 0
    assert report["peak_active_frames"] > 0


def test_no_peak_keeps_zero_mask():
    motion = _identity_motion()
    seam = np.zeros((len(motion), 1), dtype=np.float32)
    config = PeakJerkMaskConfig(
        enabled=True,
        absolute_threshold_mps3=1000.0,
        percentile=100.0,
        radius_frames_at_30fps=2,
        parent_depth=2,
    )

    repair, report = build_repair_mask(
        motion,
        seam,
        fps=30.0,
        config=config,
    )

    assert np.count_nonzero(repair) == 0
    assert report["peak_active_frames"] == 0


def test_foot_peak_jerk_also_owns_contact_channel():
    motion = _identity_motion()
    rotation = so3_exp_np(np.asarray([[0.0, 0.0, 1.2]], dtype=np.float32))[0]
    motion[10, 7 + 7 * 6 : 7 + 8 * 6] = matrix_to_rot6d_np(rotation)
    config = PeakJerkMaskConfig(
        enabled=True,
        absolute_threshold_mps3=100.0,
        percentile=95.0,
        radius_frames_at_30fps=2,
        parent_depth=2,
    )

    peak = build_peak_jerk_risk_mask(motion, fps=30.0, config=config)

    assert np.count_nonzero(peak["joint"][:, 7]) > 0
    assert np.count_nonzero(peak["contact"]) > 0


def test_vectorized_peak_expansion_matches_reference_loop():
    frames = 37
    rng = np.random.default_rng(20260824)
    risky = rng.random((frames - 3, NUM_JOINTS)) < 0.12
    for radius, parent_depth in ((0, 0), (2, 2), (4, 3)):
        expected_joint = np.zeros((frames, NUM_JOINTS), dtype=np.float32)
        expected_frame = np.zeros(frames, dtype=np.float32)
        for derivative_frame, joint_id in np.argwhere(risky):
            start = max(0, int(derivative_frame) - radius)
            end = min(frames, int(derivative_frame) + 4 + radius)
            chain_joint = int(joint_id)
            for _ in range(parent_depth + 1):
                if chain_joint < 0 or chain_joint >= NUM_JOINTS:
                    break
                expected_joint[start:end, chain_joint] = 1.0
                chain_joint = int(PARENTS[chain_joint])
            expected_frame[start:end] = 1.0

        actual_joint, actual_frame = _expand_peak_jerk_pairs(
            risky,
            frames=frames,
            radius=radius,
            parent_depth=parent_depth,
            parents=PARENTS,
        )
        assert np.array_equal(actual_joint, expected_joint)
        assert np.array_equal(actual_frame, expected_frame)
