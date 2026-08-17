import numpy as np

from retargeting.bvh_solver import stabilize_source_heading_positions


def _turning_positions(frames: int = 90) -> tuple[np.ndarray, dict[int, int]]:
    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    yaw = np.linspace(0.0, 2.0 * np.pi, frames, dtype=np.float32)
    positions = np.empty((frames, 4, 3), dtype=np.float32)
    for index, angle in enumerate(yaw):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            dtype=np.float32,
        )
        positions[index] = base @ rotation.T
    return positions, {0: 0, 1: 1, 2: 2, 12: 3}


def test_known_whirl_semantics_cannot_be_heading_stabilized(monkeypatch):
    positions, mapping = _turning_positions()
    monkeypatch.setenv("SOURCE_HEADING_MODE", "stabilize")
    monkeypatch.setenv("SOURCE_HEADING_MIN_DRIFT_DEG_S", "1")
    monkeypatch.setenv("SOURCE_HEADING_MIN_PERSIST_SECONDS", "0.1")
    monkeypatch.setenv("SOURCE_HEADING_DIRECTION_CONSISTENCY", "0.5")

    corrected, report = stabilize_source_heading_positions(
        positions,
        mapping,
        30.0,
        {"dance_category": "sogdian_whirl", "motion_stage_role": "climax"},
    )

    assert report["semantic_turn_preserved"] is True
    assert report["persistent_drift_ratio"] == 0.0
    assert report["removed_turns"] == 0.0
    assert np.allclose(corrected, positions, atol=1.0e-6)
