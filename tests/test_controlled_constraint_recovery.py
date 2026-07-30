from routing.hierarchical_constraint_model import (
    ConstraintBudgetConfig,
    select_controlled_recovery_indices,
)


def test_continuous_recovery_uses_remaining_resource_not_fixed_count(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_CONTROLLED_RECOVERY_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_RECOVERY_BUDGET_TOTAL", "1.0")
    monkeypatch.setenv("BR_HPR_RECOVERY_TOPK", "3")
    monkeypatch.setenv("BR_HPR_RECOVERY_MAXIMUM_CHARGE_PER_SLOT", "1.0")
    config = ConstraintBudgetConfig.from_environment(total_slots=11)
    rows = [
        {
            "hard_safe": True,
            "preferred": False,
            "future_reachability": {"future_reachable": True},
            "constraint_assessment": {
                "recovery_score": 1.0,
                "recovery_charge": 0.30,
                "future_reachability_probability": 0.8,
            },
        },
        {
            "hard_safe": True,
            "preferred": False,
            "future_reachability": {"future_reachable": True},
            "constraint_assessment": {
                "recovery_score": 0.5,
                "recovery_charge": 0.75,
                "future_reachability_probability": 0.9,
            },
        },
        {
            "hard_safe": True,
            "preferred": False,
            "future_reachability": {"future_reachable": False},
            "constraint_assessment": {
                "recovery_score": 0.1,
                "recovery_charge": 0.10,
                "future_reachability_probability": 0.0,
            },
        },
    ]
    selected = select_controlled_recovery_indices(
        rows,
        current_recovery_budget_used=0.40,
        config=config,
    )
    assert selected == {0}
