import torch

from training import motion_models as m
from training import refiner_bridge_diagnostics as d
from training import bridge_feasibility as b
from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL


def test_v15_3_is_batch_only_objective_change():
    # V15.2 per-case scientific formulation stays frozen.
    assert (
        m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL
        == "scientific_feasibility_smooth_bottleneck_observable_v8"
    )

    assert (
        m.REFINER_BATCH_AGGREGATION_PROTOCOL
        == "group_balanced_scientific_mean_cvar_v1"
    )

    assert (
        m.SCIENTIFIC_BOTTLENECK_SMOOTH_EPS
        == 1.0e-3
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_FRACTION
        == 0.25
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_MIX
        == 0.50
    )

    # Foundation direct optimizer stays unchanged.
    assert (
        b.DIRECT_OPTIMIZER_PROTOCOL
        == "per_case_scientific_feasibility_backtracking_v2"
    )

    # Network transactional optimizer stays unchanged.
    assert (
        REFINER_UPDATE_PROTOCOL
        == "full_cycle_feasibility_guard_armijo_v7"
    )

    assert (
        d.SCHEMA
        == "refiner_observable_bridge_diagnostic_v15_3"
    )

    assert d.FIT_CONTEXT_COUNT == 5


def test_tail_risk_matches_mean_cvar_definition():
    x = torch.tensor(
        [0.0, 1.0, 2.0, 3.0],
        dtype=torch.float64,
    )

    risk, mean, tail, k = (
        m._refiner_scientific_tail_risk(x)
    )

    assert k == 1

    torch.testing.assert_close(
        mean,
        torch.tensor(
            1.5,
            dtype=torch.float64,
        ),
    )

    torch.testing.assert_close(
        tail,
        torch.tensor(
            3.0,
            dtype=torch.float64,
        ),
    )

    # 0.5 * 1.5 + 0.5 * 3.0
    torch.testing.assert_close(
        risk,
        torch.tensor(
            2.25,
            dtype=torch.float64,
        ),
    )


def test_zero_scientific_deficit_stays_zero():
    x = torch.zeros(
        8,
        dtype=torch.float64,
    )

    risk, mean, tail, k = (
        m._refiner_scientific_tail_risk(x)
    )

    assert k == 2
    assert risk.item() == 0.0
    assert mean.item() == 0.0
    assert tail.item() == 0.0


def test_tail_gradient_prioritizes_hardest_case():
    x = torch.tensor(
        [0.0, 1.0, 2.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    risk, _, _, _ = (
        m._refiner_scientific_tail_risk(x)
    )

    risk.backward()

    # Ordinary mean contributes 0.5 / 4 = 0.125 to all cases.
    # Hardest 25% case receives another 0.5.
    expected = torch.tensor(
        [
            0.125,
            0.125,
            0.125,
            0.625,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        x.grad,
        expected,
        rtol=0,
        atol=1.0e-12,
    )


def test_each_role_width_group_receives_equal_weight():
    values = torch.tensor(
        [
            0, 0, 0, 1,
            0, 0, 0, 2,
            0, 0, 0, 3,
            0, 0, 0, 4,
        ],
        dtype=torch.float64,
    )

    groups = torch.tensor(
        [
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
        ],
        dtype=torch.long,
    )

    risk, stats = (
        m._refiner_group_balanced_scientific_tail(
            values,
            groups,
        )
    )

    # For [0,0,0,x]:
    # mean = x/4
    # tail25 = x
    # risk = .5*(x/4) + .5*x = .625*x
    expected = {
        "single_short": 0.625,
        "single_long": 1.250,
        "cross_short": 1.875,
        "cross_long": 2.500,
    }

    for label, value in expected.items():
        torch.testing.assert_close(
            stats[label]["risk"],
            torch.tensor(
                value,
                dtype=torch.float64,
            ),
        )

        assert stats[label]["tail_count"] == 1
        assert stats[label]["case_count"] == 4

    torch.testing.assert_close(
        risk,
        torch.tensor(
            sum(expected.values()) / 4.0,
            dtype=torch.float64,
        ),
    )


def test_hard_case_gets_more_gradient_in_every_group():
    values = torch.tensor(
        [
            .001, .002, .003, .020,
            .001, .002, .003, .030,
            .001, .002, .003, .040,
            .001, .002, .003, .050,
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    groups = torch.tensor(
        [
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
        ],
        dtype=torch.long,
    )

    risk, _ = (
        m._refiner_group_balanced_scientific_tail(
            values,
            groups,
        )
    )

    risk.backward()

    for group_index in range(4):
        start = 4 * group_index

        easy = values.grad[
            start:start + 3
        ]

        hard = values.grad[
            start + 3
        ]

        assert hard > easy.max()


def test_tail_contract_fails_closed():
    try:
        m._refiner_scientific_tail_risk(
            torch.zeros(
                2,
                2,
                dtype=torch.float64,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "rank-2 input must fail closed"
        )

    try:
        m._refiner_scientific_tail_risk(
            torch.tensor(
                [0.0, -0.1],
                dtype=torch.float64,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "negative scientific deficit must fail closed"
        )
