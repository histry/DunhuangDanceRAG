import numpy as np

from routing.dynamic_route import DynamicBeamState, prune_states


def test_pruning_preserves_distinct_constraint_state_signatures(monkeypatch):
    monkeypatch.setenv("ROUTING_BUDGET_BEAM_SHARE_BIN", "0.10")
    db = {
        "source_uids": np.asarray(["s0", "s0", "s1"], dtype=object),
        "event_families": np.asarray(["f0", "f0", "f1"], dtype=object),
    }
    states = [
        DynamicBeamState(
            motion=np.zeros((1, 151), dtype=np.float32),
            selected_event_ids=(0,),
            score=0.0,
            latest_future_viability_depth=2,
        ),
        DynamicBeamState(
            motion=np.zeros((1, 151), dtype=np.float32),
            selected_event_ids=(1,),
            score=0.1,
            latest_future_viability_depth=2,
        ),
        DynamicBeamState(
            motion=np.zeros((1, 151), dtype=np.float32),
            selected_event_ids=(2,),
            score=1.0,
            latest_future_viability_depth=3,
        ),
    ]
    kept = prune_states(states, db, width=2)
    kept_last = {state.selected_event_ids[-1] for state in kept}
    assert 2 in kept_last
    assert len(kept) == 2
