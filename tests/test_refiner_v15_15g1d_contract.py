import pytest
import torch

from training import refiner_v15_15g_fixed_budget_correction as g


def _contract(anchor=2.0, relative=0.1, absolute=0.05):
    return {
        "initial_anchor": {"cross_long.boundary": anchor},
        "relative_tolerance": {"cross_long.boundary": relative},
        "absolute_tolerance": {"cross_long.boundary": absolute},
    }


def test_g1d_exact_limit_matches_fixed_guard_formula():
    limit = g._fixed_guard_limit(_contract(), "cross_long.boundary")

    assert limit["fixed_anchor_allowance"] == pytest.approx(0.2)
    assert limit["absolute_limit"] == pytest.approx(2.2)
    assert limit["numeric_tolerance"] == pytest.approx(2.0e-7)


def test_g1d_repair_contract_is_frozen_from_train_transactions_only():
    train = {
        "split": "train",
        "transaction_guard_contracts": {
            "txn_a": _contract(),
            "txn_b": _contract(anchor=4.0),
        },
    }

    frozen = g._freeze_train_full_shadow_repair_contract(
        train,
        lse_allowance_fraction=0.01,
        lse_temperature_floor=1.0e-8,
        projection_damping=1.0e-12,
        minimum_reduction_floor=1.0e-12,
    )

    assert frozen["calibration_split"] == "train"
    assert frozen["validation_consumed_for_calibration"] is False
    assert frozen["transaction_ids"] == ["txn_a", "txn_b"]
    assert frozen["lse_temperature_by_guard_term"][
        "cross_long.boundary"
    ] == pytest.approx(0.003)
    assert frozen["minimum_shadow_reduction_by_guard_term"][
        "cross_long.boundary"
    ] > 0.0

    with pytest.raises(RuntimeError, match="requires train split"):
        g._freeze_train_full_shadow_repair_contract(
            {**train, "split": "validation"},
            lse_allowance_fraction=0.01,
            lse_temperature_floor=1.0e-8,
            projection_damping=1.0e-12,
            minimum_reduction_floor=1.0e-12,
        )


def test_g1d_smooth_group_max_distributes_gradient_without_replacing_hard_max():
    values = torch.tensor([0.2, 0.1], dtype=torch.float64, requires_grad=True)
    case_terms = {
        g.FULL_GUARD_PHYSICAL_CASE_TERMS["boundary"]: values,
    }
    batch = {"group": torch.tensor([3, 3])}

    smooth = g._smooth_group_guard_value(
        "cross_long.boundary",
        exact_value=values.max(),
        case_terms=case_terms,
        batch=batch,
        temperature=0.05,
    )
    smooth.backward()

    assert values.grad is not None
    assert torch.all(values.grad > 0.0)
    assert values.max().item() == pytest.approx(0.2)
    assert smooth.item() >= values.max().item()


def test_g1d_full_shadow_uses_authoritative_group_value(monkeypatch):
    candidate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    group_value = candidate + 2.3

    monkeypatch.setattr(
        g.m,
        "_refiner_batch_objectives",
        lambda *args, **kwargs: (
            torch.tensor(0.0),
            torch.tensor(0.0),
            {"sentinel": group_value},
            {},
        ),
    )
    monkeypatch.setattr(
        g.diagnostic,
        "_diagnostic_group_guard_values",
        lambda terms, groups: {"cross_long.boundary": terms["sentinel"]},
    )

    shadows, values, limits = g._full_transaction_fixed_guard_shadows(
        object(), {}, object(), candidate, candidate, _contract()
    )

    expected = 2.3 - 2.2 - 2.0e-7
    assert shadows["cross_long.boundary"].item() == pytest.approx(expected)
    assert values["cross_long.boundary"] is group_value
    assert limits["cross_long.boundary"]["absolute_limit"] == pytest.approx(
        2.2
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((False, True, True, True), "exact_radius_normalization_failed"),
        ((True, False, True, True), "scope_leakage"),
        ((True, True, False, True), "radius_normalization_broke_science_cone"),
        ((True, True, True, False), "full_shadow_not_strictly_reduced"),
        ((True, True, True, True), None),
    ],
)
def test_g1d_step_rejection_reasons_are_disjoint(flags, expected):
    assert g._g1d_rejection_reason(
        radius_ok=flags[0],
        scope_ok=flags[1],
        science_ok=flags[2],
        shadow_ok=flags[3],
    ) == expected
