import numpy as np

from routing.safe_source_coverage import (
    SafeSourceCoverageConfig,
    build_source_reservoir_layers,
    select_state_source_expansion_candidates,
)


def _db():
    return {
        "paths": np.asarray([f"{i}.npy" for i in range(6)], dtype=object),
        "dance_keys": np.asarray(["d"] * 6, dtype=object),
        "source_uids": np.asarray(["s0", "s0", "s1", "s1", "s2", "s2"], dtype=object),
        "event_families": np.asarray(["f0", "f0", "f0", "f1", "f0", "f2"], dtype=object),
        "event_uids": np.asarray([f"e{i}" for i in range(6)], dtype=object),
        "performer_groups": np.asarray(["female"] * 6, dtype=object),
        "anatomy_valid": np.asarray([True] * 6),
        "anatomy_quality": np.asarray([0.9] * 6, dtype=np.float32),
        "event_heading_valid": np.asarray([True] * 6),
        "event_heading_quality": np.asarray([0.9] * 6, dtype=np.float32),
        "event_quality_scores": np.asarray([0.8] * 6, dtype=np.float32),
        "event_frames": np.asarray([80, 82, 81, 83, 79, 84]),
    }


def test_source_reservoir_targets_uncovered_performer_compatible_sources(monkeypatch):
    monkeypatch.setenv("ROUTING_BUDGET_SOURCE_COVERAGE_ENABLE", "1")
    monkeypatch.setenv("ROUTING_BUDGET_SOURCE_RESERVOIR_MAXIMUM_PER_SLOT", "6")
    config = SafeSourceCoverageConfig.from_environment()
    layers, report = build_source_reservoir_layers(
        slots=[{"role": "build_up"}],
        target_lengths=[82],
        candidate_lists=[[0, 1]],
        db=_db(),
        fps=30.0,
        config=config,
    )
    sources = {_db()["source_uids"][event_id] for event_id in layers[0]}
    assert "s1" in sources
    assert "s2" in sources
    selected, expansion = select_state_source_expansion_candidates(
        reservoir_event_ids=layers[0],
        attempted_event_ids=[0, 1],
        hard_safe_event_ids=[0],
        selected_event_ids=(0,),
        previous_event_id=0,
        db=_db(),
        config=config,
    )
    assert selected
    assert all(_db()["source_uids"][event_id] != "s0" for event_id in selected)
    assert expansion["physical_constraints_relaxed"] is False
    assert report["schema"] == "safe_source_candidate_reservoir"
