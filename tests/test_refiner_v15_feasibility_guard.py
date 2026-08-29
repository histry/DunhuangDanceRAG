import pytest
import torch

from training import motion_models as m
from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL


def test_v15_optimizer_protocol():
    assert (
        REFINER_UPDATE_PROTOCOL
        == "full_cycle_feasibility_guard_armijo_v7"
    )


def test_v15_three_percent_scientific_margin():
    baseline = torch.ones(
        2,
        dtype=torch.float64,
    )

    proposed = torch.tensor(
        [
            0.96,  # 4% improvement -> feasible
            0.98,  # 2% improvement -> deficient
        ],
        dtype=torch.float64,
    )

    loss, gap = m._smooth_observable_margin(
        proposed,
        baseline,
        0.03,
        scale_floor=torch.tensor(
            1.0e-6,
            dtype=torch.float64,
        ),
    )

    assert float(gap[0]) == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert float(loss[0]) == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert float(gap[1]) > 0.0
    assert float(loss[1]) > 0.0


def test_v15_joint_deficit_is_worst_requirement():
    endpoint = torch.tensor(
        [0.00, 0.20, 0.10],
        dtype=torch.float64,
    )

    temporal = torch.tensor(
        [0.30, 0.10, 0.10],
        dtype=torch.float64,
    )

    result = m._joint_scientific_deficit(
        endpoint,
        temporal,
    )

    expected = torch.tensor(
        [0.30, 0.20, 0.10],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        result,
        expected,
    )


def test_v15_joint_deficit_allows_slack_trade():
    before = m._joint_scientific_deficit(
        torch.tensor([0.00]),
        torch.tensor([0.20]),
    )

    after = m._joint_scientific_deficit(
        torch.tensor([0.05]),
        torch.tensor([0.10]),
    )

    assert float(after) < float(before)


def _group_terms():
    terms = {}

    for index, label in enumerate(
        m.REFINER_GROUP_LABELS
    ):
        value = float(index + 1)

        terms[
            f"group_{label}_repair_total"
        ] = torch.tensor(value)

        terms[
            f"group_{label}_joint_scientific_deficit"
        ] = torch.tensor(value + 0.1)

        # Historical fields deliberately remain present but must not become
        # independent V15 monotonic guards.
        terms[
            f"group_{label}_endpoint_continuity"
        ] = torch.tensor(value + 0.2)

        terms[
            f"group_{label}_temporal_supervision_raw"
        ] = torch.tensor(value + 0.3)

    return terms


def test_v15_guard_has_eight_keys():
    guards = m._refiner_group_repair_losses(
        _group_terms(),
        require_all=True,
    )

    expected = set()

    for label in m.REFINER_GROUP_LABELS:
        expected.add(label)
        expected.add(
            f"{label}.feasibility"
        )

    assert set(guards) == expected
    assert len(guards) == 8

    assert not any(
        key.endswith(".endpoint")
        for key in guards
    )

    assert not any(
        key.endswith(".temporal")
        for key in guards
    )


def test_v15_guard_fails_closed_on_partial_group():
    terms = {
        "group_single_short_repair_total":
            torch.tensor(1.0)
    }

    with pytest.raises(
        RuntimeError,
        match="incomplete Refiner subgroup objectives",
    ):
        m._refiner_group_repair_losses(
            terms,
            require_all=False,
        )


def test_v15_objective_protocol():
    assert (
        m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL
        == "scientific_feasibility_balanced_observable_v7"
    )
