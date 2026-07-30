import numpy as np

from routing.hierarchical_constraint_model import (
    ConstraintBudgetConfig,
    assess_candidate_constraints,
    build_source_scarcity_context,
    event_hyperbolic_distance,
)


def _db():
    return {
        "dance_keys": np.asarray(["dance", "dance", "dance", "other"], dtype=object),
        "source_uids": np.asarray(["s0", "s1", "s0", "s2"], dtype=object),
        "event_families": np.asarray(["f0", "f0", "f1", "f2"], dtype=object),
        "event_uids": np.asarray(["e0", "e1", "e2", "e3"], dtype=object),
    }


def test_hyperbolic_hierarchy_preserves_identity_and_separation():
    db = _db()
    assert event_hyperbolic_distance(db, 0, 0) == 0.0
    sibling = event_hyperbolic_distance(db, 0, 2)
    different_dance = event_hyperbolic_distance(db, 0, 3)
    assert sibling > 0.0
    assert different_dance > 0.0
    assert not np.isclose(sibling, different_dance)


def test_safe_source_scarcity_scales_only_non_safety_source_preferences(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_SOURCE_SCARCITY_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_MINIMUM_SAFE_SOURCE_COUNT", "2")
    monkeypatch.setenv("BR_HPR_SOURCE_SCARCITY_MINIMUM_SCALE", "0.10")
    monkeypatch.setenv("BR_HPR_MAX_SOURCE_RUN", "1")
    config = ConstraintBudgetConfig.from_environment(total_slots=5)
    context = build_source_scarcity_context(
        db=_db(),
        hard_safe_event_ids=[0],
        all_event_ids=[0, 1],
        config=config,
    )
    usage, duals = config.initial_state()
    result = assess_candidate_constraints(
        db=_db(),
        event_id=0,
        selected_event_ids=(0,),
        observability=0.8,
        future_reachability_probability=0.9,
        slot_index=1,
        constraint_usage=usage,
        dual_variables=duals,
        config=config,
        scarcity_context=context,
    )
    assert context.source_scarcity_exemption is True
    assert context.alternative_safe_source_exists is False
    assert result["raw_violations"]["source_run"] > 0.0
    assert result["violations"]["source_run"] < result["raw_violations"]["source_run"]
    assert result["diversity"]["hard_valid"] is True
    assert result["source_scarcity"]["source_scarcity_exemption"] is True
