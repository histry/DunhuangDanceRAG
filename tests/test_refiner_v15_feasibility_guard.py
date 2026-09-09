import pytest
import torch

from tests.test_bridge_feasibility import bank
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


def test_v15_6_gate_aligned_deficit_keeps_gradient_until_buffered_target():
    baseline = torch.ones(
        2,
        dtype=torch.float64,
    )
    proposed = torch.tensor(
        [
            0.969,  # exact 3% gate passes, buffered 3.5% target is active
            0.9650001,  # extremely close to the buffered target
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    loss, gap = m._gate_aligned_observable_deficit(
        proposed,
        baseline,
        0.03,
        training_buffer=0.005,
        gradient_floor=0.10,
    )

    assert torch.all(gap > 0.0)
    loss.sum().backward()
    assert torch.all(proposed.grad >= 0.10)


def test_v15_6_gate_aligned_deficit_matches_uninformative_audit_floor():
    baseline = torch.tensor(
        [0.0, 1.0e-6],
        dtype=torch.float64,
    )
    proposed = torch.tensor(
        [1.0e-6, 1.0e-6],
        dtype=torch.float64,
    )

    loss, gap = m._gate_aligned_observable_deficit(
        proposed,
        baseline,
        0.03,
        training_buffer=0.005,
        gradient_floor=0.10,
    )

    assert torch.equal(gap, torch.zeros_like(gap))
    assert torch.equal(loss, torch.zeros_like(loss))


def test_v15_6_minimum_edit_is_inactive_until_both_targets_pass():
    edit = torch.tensor(
        [0.4, 0.4, 0.4],
        dtype=torch.float64,
    )
    endpoint_gap = torch.tensor(
        [0.0, 0.1, 0.0],
        dtype=torch.float64,
    )
    temporal_gap = torch.tensor(
        [0.0, 0.0, 0.1],
        dtype=torch.float64,
    )

    penalty, active = m._feasible_minimum_edit_penalty(
        edit,
        endpoint_gap,
        temporal_gap,
    )

    torch.testing.assert_close(
        active,
        torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        penalty,
        torch.tensor(
            [0.4, 0.0, 0.0],
            dtype=torch.float64,
        ),
    )


def test_v15_6_objective_keeps_endpoint_and_temporal_components_active():
    batch, cfg = bank()

    _, terms = m._observable_refiner_objective(
        batch["bad"],
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )

    torch.testing.assert_close(
        terms["joint_scientific_deficit"],
        terms["endpoint_scientific_deficit"]
        + terms["temporal_scientific_deficit"],
    )
    torch.testing.assert_close(
        terms["scientific_observable"],
        terms["joint_scientific_deficit"],
    )
    assert torch.equal(
        terms["minimum_edit_active"],
        torch.zeros_like(terms["minimum_edit_active"]),
    )


def test_v15_2_joint_deficit_is_smooth_worst_requirement():
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

    hard = torch.maximum(
        endpoint,
        temporal,
    )

    eps = m.SCIENTIFIC_BOTTLENECK_SMOOTH_EPS

    # V15.2 deliberately smooths the hard max from below while
    # preserving the same zero-feasibility set.
    assert torch.all(
        result <= hard + 1.0e-14
    )

    assert torch.all(
        result >= hard - eps / 2.0 - 1.0e-14
    )

    # When the two scientific deficits are tied, the smooth
    # bottleneck equals the original hard worst requirement exactly.
    torch.testing.assert_close(
        result[2],
        hard[2],
        rtol=0,
        atol=1.0e-14,
    )

    # Unequal requirements are intentionally smoothed rather than
    # remaining exactly equal to torch.maximum(...).
    assert result[0] < hard[0]
    assert result[1] < hard[1]


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
        == "gate_aligned_component_tail_observable_v11"
    )
