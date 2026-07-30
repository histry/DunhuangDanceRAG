from routing.constraint_audit import (
    controlled_recovery_metadata,
    summarize_constraint_trials,
)


def test_state_aware_constraint_diagnostics_are_explicit():
    trials = [
        {
            "event_id": 10,
            "safe": True,
            "preferred": False,
            "eligible": True,
            "recovery_triggered": True,
            "observability": 0.6,
            "future_reachability": {
                "future_reachable": True,
                "future_safe_successor_count": 3,
                "future_reachability_probability": 0.8,
            },
            "constraint_assessment": {
                "identity": {"source_uid": "s0", "family_id": "f0"},
                "recovery_charge": 0.2,
                "budget_overrun_reasons": ["event_repeat"],
                "diversity": {"soft_reasons": ["event_repeat"]},
            },
        },
        {
            "event_id": 11,
            "safe": True,
            "preferred": False,
            "eligible": False,
            "recovery_triggered": False,
            "observability": 0.5,
            "future_reachability": {
                "future_reachable": False,
                "future_safe_successor_count": 0,
                "future_reachability_probability": 0.0,
                "future_first_dead_end_slot": 5,
            },
            "constraint_assessment": {
                "identity": {"source_uid": "s0", "family_id": "f0"},
                "recovery_charge": 0.8,
                "budget_overrun_reasons": ["source_run"],
                "diversity": {"soft_reasons": ["source_run"]},
            },
        },
    ]
    summary = summarize_constraint_trials(
        trials,
        source_expansion={"triggered": True},
        scarcity_context={"source_scarcity_exemption": True},
    )
    assert summary["physically_safe"] == 2
    assert summary["state_future_reachable"] == 1
    assert summary["controlled_recovery"] == 1
    assert summary["safe_source_count"] == 1
    assert summary["future_first_dead_end_slot_counts"] == {"5": 1}

    metadata = controlled_recovery_metadata(
        {
            "constraint_usage_after": {"event_repeat": 1.2},
            "effective_budget": {"event_repeat": 1.0},
            "budget_overrun": {"event_repeat": 0.2},
            "budget_overrun_reasons": ["event_repeat"],
            "future_reachability_probability": 0.8,
            "recovery_charge": 0.2,
            "source_scarcity": {"source_scarcity_exemption": True},
        },
        triggered=True,
        recovery_count_after=1,
        recovery_budget_used_before=0.4,
        recovery_budget_used_after=0.6,
        recovery_budget_total=3.0,
    )
    assert metadata["physical_constraints_relaxed"] is False
    assert metadata["recovery_budget_used_after"] == 0.6
    assert metadata["source_scarcity_exemption"] is True
