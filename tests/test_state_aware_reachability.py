import numpy as np

from routing.bidirectional_reachability import BackwardReachabilityModel
from routing.hierarchical_constraint_model import ConstraintBudgetConfig


def _db():
    return {
        "paths": np.asarray(["a.npy", "b.npy", "c.npy"], dtype=object),
        "dance_keys": np.asarray(["d", "d", "d"], dtype=object),
        "source_uids": np.asarray(["s0", "s1", "s0"], dtype=object),
        "event_families": np.asarray(["f0", "f1", "f0"], dtype=object),
        "event_uids": np.asarray(["e0", "e1", "e2"], dtype=object),
        "anatomy_valid": np.asarray([True, True, True]),
        "anatomy_quality": np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        "event_heading_valid": np.asarray([True, True, True]),
    }


def test_state_history_can_invalidate_a_statically_reachable_node(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_SOURCE_SCARCITY_ENABLE", "0")
    monkeypatch.setenv("BR_HPR_EVENT_REPEAT_BUDGET", "0")
    monkeypatch.setenv("BR_HPR_RECOVERY_BUDGET_TOTAL", "0")
    monkeypatch.setenv("BR_HPR_STATE_REACHABILITY_HORIZON", "3")
    config = ConstraintBudgetConfig.from_environment(total_slots=3)
    model = BackwardReachabilityModel.build(
        [[0], [1], [2]],
        _db(),
        constraint_config=config,
    )
    usage, duals = config.initial_state()
    static = model.get(0, 0)
    clean = model.query(
        slot=0,
        event_id=0,
        selected_event_ids=(),
        constraint_usage=usage,
        dual_variables=duals,
        recovery_budget_used=0.0,
        observability=0.8,
    )
    repeated = model.query(
        slot=0,
        event_id=0,
        selected_event_ids=(0,),
        constraint_usage=usage,
        dual_variables=duals,
        recovery_budget_used=0.0,
        observability=0.8,
    )
    assert static["future_reachable"] is True
    assert clean["future_reachable"] is True
    assert repeated["future_reachable"] is False
    assert repeated["state_budget_feasible"] is False
