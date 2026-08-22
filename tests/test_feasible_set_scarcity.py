import numpy as np

from routing.hierarchical_constraint_model import (
    ConstraintBudgetConfig,
    assess_candidate_constraints,
    build_feasible_set_scarcity_context,
)


def _db():
    return {
        "dance_keys": np.asarray(["d", "d", "d"], dtype=object),
        "source_uids": np.asarray(["s0", "s1", "s2"], dtype=object),
        "event_families": np.asarray(["f0", "f1", "f2"], dtype=object),
        "event_uids": np.asarray(["e0", "e1", "e2"], dtype=object),
    }


def test_family_scarcity_scales_family_and_hierarchy_preferences(monkeypatch):
    monkeypatch.setenv("ROUTING_BUDGET_SOURCE_SCARCITY_ENABLE", "1")
    monkeypatch.setenv("ROUTING_BUDGET_FAMILY_SCARCITY_ENABLE", "1")
    monkeypatch.setenv("ROUTING_BUDGET_MINIMUM_SAFE_SOURCE_COUNT", "2")
    monkeypatch.setenv("ROUTING_BUDGET_MINIMUM_SAFE_FAMILY_COUNT", "2")
    monkeypatch.setenv("ROUTING_BUDGET_FAMILY_SCARCITY_MINIMUM_SCALE", "0.10")
    monkeypatch.setenv("ROUTING_BUDGET_MIN_SHARE_HISTORY", "1")
    monkeypatch.setenv("ROUTING_BUDGET_MAX_FAMILY_SHARE", "0.20")
    config = ConstraintBudgetConfig.from_environment(total_slots=5)
    context = build_feasible_set_scarcity_context(
        db=_db(),
        hard_safe_event_ids=[0],
        all_event_ids=[0, 1, 2],
        config=config,
    )
    usage, duals = config.initial_state()
    result = assess_candidate_constraints(
        db=_db(),
        event_id=0,
        selected_event_ids=(0,),
        observability=0.8,
        future_reachability_probability=0.5,
        slot_index=1,
        constraint_usage=usage,
        dual_variables=duals,
        config=config,
        scarcity_context=context,
    )
    assert context.family_scarcity_exemption is True
    assert context.alternative_safe_family_exists is False
    assert result["raw_violations"]["family_share"] > 0.0
    assert result["violations"]["family_share"] < result["raw_violations"]["family_share"]
    assert result["feasible_set_scarcity"]["safe_family_count"] == 1
