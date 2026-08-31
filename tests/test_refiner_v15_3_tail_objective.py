import torch

from training import motion_models as m
from training import refiner_bridge_diagnostics as d
from training import bridge_feasibility as b
from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL


def test_v15_3_1_contract():
    assert (
        m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL
        == "scientific_feasibility_smooth_bottleneck_observable_v8"
    )

    assert (
        m.REFINER_BATCH_AGGREGATION_PROTOCOL
        == "group_balanced_scientific_mean_smooth_cvar_v2"
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_FRACTION
        == 0.25
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_MIX
        == 0.50
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_TEMPERATURE
        == 1.0e-3
    )

    assert (
        b.DIRECT_OPTIMIZER_PROTOCOL
        == "per_case_scientific_feasibility_backtracking_v2"
    )

    assert (
        REFINER_UPDATE_PROTOCOL
        == "full_cycle_feasibility_guard_armijo_v7"
    )

    assert (
        d.SCHEMA
        == "refiner_observable_bridge_diagnostic_v15_3_1"
    )

    assert d.FIT_CONTEXT_COUNT == 5


def test_equal_deficits_are_value_preserving():
    x = torch.full(
        (48,),
        0.015,
        dtype=torch.float64,
    )

    risk, mean, tail, k = (
        m._refiner_scientific_tail_risk(x)
    )

    assert k == 12

    torch.testing.assert_close(
        tail,
        mean,
        rtol=0,
        atol=1.0e-10,
    )

    torch.testing.assert_close(
        risk,
        mean,
        rtol=0,
        atol=1.0e-10,
    )


def test_equal_deficits_have_uniform_gradient():
    x = torch.full(
        (48,),
        0.015,
        dtype=torch.float64,
        requires_grad=True,
    )

    risk, _, _, k = (
        m._refiner_scientific_tail_risk(x)
    )

    assert k == 12

    risk.backward()

    expected = torch.full(
        (48,),
        1.0 / 48.0,
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        x.grad,
        expected,
        rtol=0,
        atol=1.0e-9,
    )


def test_zero_scientific_group_remains_exact_zero():
    x = torch.zeros(
        48,
        dtype=torch.float64,
        requires_grad=True,
    )

    risk, mean, tail, k = (
        m._refiner_scientific_tail_risk(x)
    )

    assert k == 12

    assert risk.item() == 0.0
    assert mean.item() == 0.0
    assert tail.item() == 0.0

    risk.backward()

    assert torch.equal(
        x.grad,
        torch.zeros_like(x),
    )


def test_separated_hard_tail_gets_more_gradient():
    x = torch.tensor(
        [0.001] * 36
        + [0.010] * 12,
        dtype=torch.float64,
        requires_grad=True,
    )

    risk, mean, tail, k = (
        m._refiner_scientific_tail_risk(x)
    )

    assert k == 12

    assert tail > mean
    assert risk > mean

    risk.backward()

    easy = x.grad[:36].mean()
    hard = x.grad[36:].mean()

    assert hard > easy

    ratio = float(
        hard / easy
    )

    assert 3.0 < ratio < 5.1


def test_near_tie_weights_are_positive_and_continuous():
    x = torch.linspace(
        0.0148,
        0.0152,
        48,
        dtype=torch.float64,
        requires_grad=True,
    )

    risk, _, _, _ = (
        m._refiner_scientific_tail_risk(x)
    )

    risk.backward()

    assert torch.isfinite(
        x.grad
    ).all()

    assert (
        x.grad > 0
    ).all()

    # Larger deficits should receive smoothly non-decreasing weight.
    assert (
        x.grad[1:]
        >= x.grad[:-1] - 1.0e-12
    ).all()


def test_four_group_aggregation_is_symmetric():
    group_values = torch.tensor(
        [
            0.001,
            0.002,
            0.003,
            0.020,
        ],
        dtype=torch.float64,
    )

    values = group_values.repeat(4)

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

    rows = [
        stats[label]["risk"]
        for label in m.REFINER_SCIENTIFIC_GROUP_LABELS
    ]

    for row in rows[1:]:
        torch.testing.assert_close(
            row,
            rows[0],
            rtol=0,
            atol=1.0e-12,
        )

    torch.testing.assert_close(
        risk,
        rows[0],
        rtol=0,
        atol=1.0e-12,
    )


def test_smooth_cvar_fails_closed():
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
