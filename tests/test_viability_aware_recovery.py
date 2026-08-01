from routing.hierarchical_constraint_model import (
    ConstraintBudgetConfig,
    select_controlled_recovery_indices,
)


def _row(*, terminal, depth, until, successors, charge, score=1.0):
    return {
        "hard_safe": True,
        "preferred": False,
        "future_reachability": {
            "future_reachable": terminal,
            "terminal_reachable": terminal,
            "future_viability_depth": depth,
            "reachable_until_slot": until,
            "future_safe_successor_count": successors,
        },
        "constraint_assessment": {
            "recovery_score": score,
            "recovery_charge": charge,
            "future_reachability_probability": 0.2,
        },
    }


def test_nonterminal_but_viable_hard_safe_candidate_can_recover(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_CONTROLLED_RECOVERY_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_RECOVERY_BUDGET_TOTAL", "3.0")
    monkeypatch.setenv("BR_HPR_RECOVERY_MINIMUM_VIABILITY_DEPTH", "2")
    monkeypatch.setenv("BR_HPR_RECOVERY_TOPK", "1")
    config = ConstraintBudgetConfig.from_environment(total_slots=11)
    rows = [_row(terminal=False, depth=3, until=9, successors=6, charge=0.055)]
    assert select_controlled_recovery_indices(rows, config=config) == {0}


def test_immediate_dead_end_is_not_recovered(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_CONTROLLED_RECOVERY_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_RECOVERY_BUDGET_TOTAL", "3.0")
    config = ConstraintBudgetConfig.from_environment(total_slots=11)
    rows = [_row(terminal=False, depth=0, until=6, successors=0, charge=0.01)]
    assert select_controlled_recovery_indices(rows, config=config) == set()


def test_terminal_reachable_branch_has_lexicographic_priority(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_CONTROLLED_RECOVERY_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_RECOVERY_BUDGET_TOTAL", "3.0")
    monkeypatch.setenv("BR_HPR_RECOVERY_TOPK", "1")
    config = ConstraintBudgetConfig.from_environment(total_slots=11)
    rows = [
        _row(terminal=False, depth=5, until=10, successors=8, charge=0.01, score=0.1),
        _row(terminal=True, depth=2, until=10, successors=2, charge=0.2, score=2.0),
    ]
    assert select_controlled_recovery_indices(rows, config=config) == {1}
