import inspect
from pathlib import Path

import torch

from training import refiner_v15_15g_fixed_budget_correction as g
from training import refiner_v15_15g1f3_second_order as second_order
from routing import refiner_v15_15h_composite as composite


def _guard_contract():
    return {
        "initial_anchor": {
            "cross_short.observable_temporal_0p03": 1.0,
            "cross_short.joint_jerk_max": 1.0,
        },
        "relative_tolerance": {
            "cross_short.observable_temporal_0p03": 0.0,
            "cross_short.joint_jerk_max": 0.0,
        },
        "absolute_tolerance": {
            "cross_short.observable_temporal_0p03": 1.0e-3,
            "cross_short.joint_jerk_max": 1.0e-3,
        },
    }


def test_g1f3_strict_bundle_excludes_safe_rows_below_frontier():
    margins = {
        "cross_short.observable_temporal_0p03": 9.86284e-6,
        "cross_short.joint_jerk_max": -1.0e-6,
        "cross_long.boundary": -2.0e-5,
    }

    selected = g._select_second_order_guard_rows(margins, 1.0e-5)

    assert selected == ["cross_short.observable_temporal_0p03"]


def test_g1f3_bundle_keeps_every_violated_guard_below_hard_frontier():
    margins = {
        "cross_short.joint_jerk_window_p95": 0.1735,
        "cross_short.boundary": 0.004,
        "cross_short.observable_temporal_0p03": 0.001,
        "cross_short.joint_jerk_max": -1.0e-6,
    }

    selected = g._select_second_order_guard_rows(margins, 1.0e-5)

    assert selected == [
        "cross_short.joint_jerk_window_p95",
        "cross_short.boundary",
        "cross_short.observable_temporal_0p03",
    ]


def test_g1f3_authoritative_full_shadow_quota_rejects_hidden_guard_worsening():
    required, passed = g._authoritative_full_shadow_quota(
        current_shadow=0.17350872408973847,
        trial_shadow=0.5345796410971798,
        remaining_steps=5,
        closure_margin=1.0e-8,
        tolerance=1.0e-12,
    )

    assert required > 0.0
    assert passed is False


def test_g1f3_filter_rejects_full_shadow_worsening_despite_row_progress():
    assert not g._authoritative_full_shadow_strictly_decreased(
        current_shadow=0.2864490385019271,
        trial_shadow=0.6918157517684123,
        tolerance=1.0e-12,
    )


def test_g1f3_constraint_generation_adds_only_new_trial_violations():
    rows = [
        {
            "authoritative_trial_executed": False,
            "authoritative_trial_active_terms": ["guard.not_executed"],
        },
        {
            "authoritative_trial_executed": True,
            "authoritative_trial_active_terms": [
                "cross_short.observable_temporal_0p03",
                "cross_short.extremity_jerk_p95",
            ],
        },
        {
            "authoritative_trial_executed": True,
            "authoritative_trial_active_terms": [
                "cross_short.joint_jerk_p95",
                "cross_short.extremity_jerk_p95",
            ],
        },
    ]

    added = g._newly_violated_guard_terms(
        rows, ("cross_short.observable_temporal_0p03",)
    )

    assert added == [
        "cross_short.extremity_jerk_p95",
        "cross_short.joint_jerk_p95",
    ]


def test_g1f3_second_order_physical_row_isolates_edited_case():
    cross_short = g.m.REFINER_GROUP_LABELS.index("cross_short")
    values = torch.tensor(
        [5.0, -0.25, 7.0], dtype=torch.float64, requires_grad=True
    )
    selected = g._case_isolated_second_order_guard_value(
        "cross_short.extremity_jerk_p95",
        exact_value=values.max(),
        case_terms={
            "repair_extremity_jerk_mps3_p95_signed_margin": values,
        },
        batch={"group": torch.tensor([cross_short] * 3)},
        local_case=1,
    )

    selected.backward()

    assert float(selected.detach()) == -0.25
    assert values.grad.tolist() == [0.0, 1.0, 0.0]


def test_v15h_runtime_models_complete_five_row_proxy_universe():
    margins = {
        "joint_jerk_p95": -3.0,
        "joint_jerk_window_p95": -2.0,
        "extremity_jerk_p95": -1.0,
        "extremity_jerk_window_p95": 0.0,
        "boundary": 1.0,
    }

    assert composite._runtime_proxy_guard_rows(margins) == tuple(
        sorted(margins)
    )


def test_g1f3_freezes_row_wise_qcqp_contract_from_train():
    frozen = g._freeze_train_full_shadow_repair_contract(
        {
            "split": "train",
            "transaction_guard_contracts": {"txn_a": _guard_contract()},
        },
        lse_allowance_fraction=0.01,
        lse_temperature_floor=1.0e-8,
        projection_damping=1.0e-12,
        minimum_reduction_floor=1.0e-12,
        second_order_joint_sqp=True,
        second_order_guard_transition_band=1.0e-5,
    )

    assert frozen["schema"] == g.G1F3_TRAIN_CONTRACT_SCHEMA
    assert frozen["second_order_guard_transition_bundle"] == (
        "all_violated_plus_strict_frontier_guard_rows"
    )
    assert frozen["second_order_guard_transition_band"] == 1.0e-5
    assert frozen["second_order_guard_transition_aggregation"] == (
        "independent_row_wise_qcqp"
    )
    assert frozen["second_order_guard_transition_threshold"] == (
        "hard_margin_positive_or_greater_equal_max_zero_and_hard_max_minus_band"
    )
    assert frozen["second_order_active_set_constraint_generation"] == (
        "authoritative_trial_new_positive_guard_rows_same_expansion_rebuild"
    )
    assert frozen["second_order_physical_guard_row_scope"] == (
        "edited_case_exact_signed_margin_no_cross_case_softmax"
    )


def test_second_order_solver_accepts_independent_guard_rows():
    names = (
        "guard::cross_short.observable_temporal_0p03",
        "guard::cross_short.boundary",
        "endpoint",
        "temporal",
    )
    models = {
        name: {
            "value": torch.tensor(0.0, dtype=torch.float64),
            "first": torch.tensor([1.0, 0.0], dtype=torch.float64),
            "hessian": torch.zeros((2, 2), dtype=torch.float64),
        }
        for name in names
    }

    coefficients, audit = second_order.solve_angle_subproblem(
        models=models,
        theta_radians=0.1,
        required_reduction={name: 0.01 for name in names},
        grid_levels=3,
        feasibility_tolerance=1.0e-12,
    )

    assert coefficients is not None
    assert audit["constraint_names"] == list(names)
    assert set(audit["predicted_change_by_term"]) == set(names)
    assert all(
        value <= -0.01 + 1.0e-12 for value in audit["predicted_change_by_term"].values()
    )


def test_g1f3_server_runner_pins_strict_guard_band():
    text = Path("scripts/run_refiner_v15_15g1f3_train_dev_server.sh").read_text(
        encoding="utf-8"
    )

    assert "--second-order-guard-transition-band 1e-5" in text


def test_g1f3_and_composite_do_not_aggregate_guard_rows_with_logsumexp():
    training_source = inspect.getsource(g._g1d_shadow_objective)
    isolated_source = inspect.getsource(
        g._case_isolated_second_order_guard_value
    )
    composite_source = inspect.getsource(composite._apply_one_transaction)

    assert "_case_isolated_second_order_guard_value" in training_source
    assert "_smooth_logsumexp" not in isolated_source
    assert "_smooth_logsumexp" not in composite_source
