import numpy as np

import routing.anatomy_feature_cache as module
from contracts.gravity import EDGE_DIM, NUM_JOINTS, ROT6D_END, ROT6D_START


def _motion(frames=30):
    x = np.zeros((frames, EDGE_DIM), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    x[:, ROT6D_START:ROT6D_END] = np.tile(identity, NUM_JOINTS)
    return x


def _db():
    fields = {
        "anatomy_valid": True,
        "anatomy_hard_valid": True,
        "anatomy_soft_valid": True,
        "anatomy_quality": 0.9,
        "posture_entry": "standing",
        "posture_exit": "standing",
        "posture_mode": "standing",
        "pelvis_height_entry_norm": 1.0,
        "pelvis_height_exit_norm": 1.0,
        "pelvis_height_median_norm": 1.0,
        "body_height_entry_norm": 1.0,
        "body_height_exit_norm": 1.0,
        "body_height_median_norm": 1.0,
        "entry_floor_offset_m": 0.0,
        "exit_floor_offset_m": 0.0,
        "torso_compression_ratio_p05": 1.0,
        "local_angle_violation_ratio": 0.0,
        "raw_local_angle_violation_ratio": 0.0,
        "local_angle_severe_ratio": 0.0,
        "self_collision_severe_ratio": 0.0,
        "spine_cumulative_angle_p95_deg": 0.0,
    }
    return {key: np.asarray([value], dtype=object) for key, value in fields.items()}


def test_static_database_path_skips_runtime(monkeypatch):
    cache = module.CandidateAnatomyCache(maximum_entries=8)
    monkeypatch.setattr(
        module,
        "event_anatomy_features",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runtime called")),
    )
    feature, report = module.evaluate_candidate_anatomy(
        db=_db(),
        event_id=0,
        core_motion=_motion(30),
        fps=30.0,
        source_frames=30,
        cache=cache,
    )
    assert feature["anatomy_valid"] is True
    assert report["mode"] == "static_event_db"


def test_runtime_result_is_cached(monkeypatch):
    cache = module.CandidateAnatomyCache(maximum_entries=8)
    calls = {"count": 0}

    def fake(_motion, fps=30.0):
        calls["count"] += 1
        return {"anatomy_valid": True, "anatomy_quality": 0.8}

    monkeypatch.setattr(module, "event_anatomy_features", fake)
    kwargs = dict(
        db={},
        event_id=7,
        core_motion=_motion(42),
        fps=30.0,
        source_frames=30,
        cache=cache,
    )
    _, first = module.evaluate_candidate_anatomy(**kwargs)
    _, second = module.evaluate_candidate_anatomy(**kwargs)
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert calls["count"] == 1
