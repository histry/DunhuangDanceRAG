import numpy as np

from contracts.anatomy import anatomy_metrics_np, event_anatomy_features, posture_labels_np
from contracts.gravity import EDGE_DIM, NUM_JOINTS, ROT6D_END, ROT6D_START, fk24_from_local_np
from motion_geometry.rotations import rot6d_to_matrix_np


def _identity_motion(frames=24):
    x = np.zeros((frames, EDGE_DIM), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    x[:, ROT6D_START:ROT6D_END] = np.tile(identity, NUM_JOINTS)
    return x


def test_precomputed_geometry_matches_direct_metrics():
    motion = _identity_motion()
    local = rot6d_to_matrix_np(
        motion[:, ROT6D_START:ROT6D_END].reshape(len(motion), NUM_JOINTS, 6)
    )
    joints = fk24_from_local_np(motion, local)
    labels = posture_labels_np(motion, joints)

    direct = anatomy_metrics_np(motion)
    reused = anatomy_metrics_np(
        motion,
        local_matrices=local,
        joints=joints,
        labels=labels,
    )
    for key in (
        "anatomy_quality",
        "local_angle_violation_ratio",
        "self_collision_severe_ratio",
        "bone_length_drift_max",
    ):
        assert np.isclose(direct[key], reused[key], rtol=1e-6, atol=1e-7)

    direct_feature = event_anatomy_features(motion)
    feature = event_anatomy_features(
        motion,
        local_matrices=local,
        joints=joints,
        labels=labels,
        metrics=reused,
    )

    assert (
        feature["anatomy_valid"]
        == direct_feature["anatomy_valid"]
    )
    assert (
        feature["anatomy_hard_valid"]
        == direct_feature["anatomy_hard_valid"]
    )
    assert (
        feature["anatomy_soft_valid"]
        == direct_feature["anatomy_soft_valid"]
    )
    assert (
        feature["anatomy_reasons"]
        == direct_feature["anatomy_reasons"]
    )
    assert np.isclose(
        feature["anatomy_quality"],
        direct_feature["anatomy_quality"],
        rtol=1e-6,
        atol=1e-7,
    )
