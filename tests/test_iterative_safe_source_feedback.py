import numpy as np

from routing.safe_source_coverage import (
    SafeSourceCoverageConfig,
    build_state_source_expansion_batches,
    select_bottleneck_layer_expansion_candidates,
)


def _db():
    return {
        "paths": np.asarray([f"{i}.npy" for i in range(7)], dtype=object),
        "dance_keys": np.asarray(["d"] * 7, dtype=object),
        "source_uids": np.asarray(["s0", "s0", "s1", "s1", "s2", "s2", "s3"], dtype=object),
        "event_families": np.asarray(["f0", "f0", "f1", "f1", "f2", "f2", "f3"], dtype=object),
        "event_uids": np.asarray([f"e{i}" for i in range(7)], dtype=object),
    }


def test_expansion_plan_keeps_multiple_unverified_sources(monkeypatch):
    monkeypatch.setenv("ROUTING_BUDGET_SOURCE_EXPANSION_MAXIMUM_EXACT", "4")
    monkeypatch.setenv("ROUTING_BUDGET_SOURCE_EXPANSION_PER_SOURCE", "2")
    config = SafeSourceCoverageConfig.from_environment()
    batches, report = build_state_source_expansion_batches(
        reservoir_event_ids=[2, 3, 4, 5, 6],
        attempted_event_ids=[0, 1],
        hard_safe_event_ids=[0],
        selected_event_ids=(0,),
        previous_event_id=0,
        db=_db(),
        config=config,
    )
    batch_sources = [_db()["source_uids"][batch[0]] for batch in batches]
    assert len(batches) >= 2
    assert len(set(batch_sources)) >= 2
    assert report["additional_exact_simulation_budget"] == 4


def test_bottleneck_selection_prefers_new_source_and_family(monkeypatch):
    monkeypatch.setenv("ROUTING_BUDGET_BOTTLENECK_EXPANSION_MAXIMUM", "2")
    config = SafeSourceCoverageConfig.from_environment()
    selected, report = select_bottleneck_layer_expansion_candidates(
        reservoir_event_ids=[2, 4, 6],
        active_event_ids=[0, 1],
        selected_event_ids=(0, 1),
        db=_db(),
        config=config,
    )
    assert selected
    assert 6 in selected or 4 in selected
    assert report["triggered"] is True
