from pathlib import Path

import pytest
import torch

from training import refiner_v15_15g_fixed_budget_correction as g


def _guard_contract(anchor=2.0):
    return {
        "initial_anchor": {"cross_long.boundary": anchor},
        "relative_tolerance": {"cross_long.boundary": 0.1},
        "absolute_tolerance": {"cross_long.boundary": 0.05},
    }


def _requirements(value=1.0e-3):
    return {
        name: {
            "current_delta": value,
            "strict_pass_limit": -1.0e-7,
            "gap_to_strict_pass_line": value,
            "safety_margin": 1.0e-12,
            "required_predicted_reduction": value,
        }
        for name in ("endpoint", "temporal")
    }


def test_g1f2_freezes_finite_gap_and_float64_contract_from_train():
    train = {
        "split": "train",
        "transaction_guard_contracts": {
            "txn_a": _guard_contract(),
            "txn_b": _guard_contract(anchor=4.0),
        },
    }

    frozen = g._freeze_train_full_shadow_repair_contract(
        train,
        lse_allowance_fraction=0.01,
        lse_temperature_floor=1.0e-8,
        projection_damping=1.0e-12,
        minimum_reduction_floor=1.0e-12,
        line_search_backtrack_count=12,
        line_search_decay=0.5,
        geodesic_joint_sqp=True,
        temporal_directional_consistency=True,
        finite_gap_angular_feasibility=True,
        temporal_fd_float64_epsilon_ladder=(
            1.0e-5,
            3.0e-5,
            1.0e-4,
            3.0e-4,
        ),
        temporal_fd_near_zero_threshold=1.0e-5,
    )

    assert frozen["schema"] == g.G1F2_TRAIN_CONTRACT_SCHEMA
    assert frozen["finite_gap_angular_feasibility_sqp"] is True
    assert frozen["per_angle_active_set"] is True
    assert frozen["finite_gap_constraint_coordinate"] == (
        "unnormalized_true_physical_angular_derivative"
    )
    assert frozen["temporal_fd_float64_epsilon_ladder"] == [
        1.0e-5,
        3.0e-5,
        1.0e-4,
        3.0e-4,
    ]
    assert frozen["second_order_hessian_used"] is False
    assert frozen["validation_consumed_for_calibration"] is False


def test_g1f2_solves_each_angle_with_true_radius_scaled_derivative():
    current = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)
    taper = torch.tensor([[[0.25, 0.5, 1.0, 1.0]]], dtype=torch.float64)
    gradients = {
        "shadow": torch.tensor([[[0.0, 0.5, 0.0, 0.0]]], dtype=torch.float64),
        "endpoint": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float64),
        "temporal": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64),
    }

    physical, audit = g._joint_geodesic_finite_gap_direction_for_angle(
        current=current,
        mask=mask,
        taper=taper,
        gradients=gradients,
        theta_radians=0.5,
        science_requirements=_requirements(),
        shadow_minimum_reduction=1.0e-3,
        svd_relative_cutoff=1.0e-6,
        direction_norm_floor=1.0e-8,
    )

    assert physical is not None
    z = audit.pop("_optimization_direction_z")
    assert torch.allclose(physical, z * taper, atol=1.0e-12, rtol=1.0e-12)
    assert torch.dot(physical[mask], current[mask]).item() == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert audit["solver_status"] == "finite_gap_angular_direction_found"
    assert audit["linearized_prediction_passed"] is True
    assert all(
        value < 0.0
        for value in audit["finite_angle_predicted_change_by_term"].values()
    )


def test_g1f2_reports_insufficient_linearized_progress_for_unreachable_gap():
    current = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)
    gradients = {
        "shadow": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=torch.float64),
        "endpoint": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float64),
        "temporal": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64),
    }

    physical, audit = g._joint_geodesic_finite_gap_direction_for_angle(
        current=current,
        mask=mask,
        taper=torch.ones_like(current),
        gradients=gradients,
        theta_radians=1.0e-3,
        science_requirements=_requirements(value=10.0),
        shadow_minimum_reduction=10.0,
        svd_relative_cutoff=1.0e-6,
        direction_norm_floor=1.0e-8,
    )

    assert physical is None
    assert audit["solver_status"] == "insufficient_linearized_science_progress"
    assert audit["second_order_hessian_used"] is False


def test_g1f2_summary_preserves_empty_reason_and_model_mismatch():
    reports = {
        "geodesic_joint_sqp_k2": {
            "txn_zero:131": {
                "history": [{
                    "constraints": {
                        "primary_objective": "full_transaction_fixed_guard_shadow"
                    },
                    "joint_solver": {
                        "solver_status": "empty_geodesic_joint_feasible_intersection",
                        "solver_failure_reason": "zero_science_gradient",
                    },
                }],
                "correction_accepted_steps": 0,
                "full_shadow_reduction": 0.0,
                "step_rejection_reason": "empty_geodesic_joint_feasible_intersection",
                "solver_failure_reason": "zero_science_gradient",
            },
            "txn_gap:170": {
                "history": [{
                    "constraints": {
                        "primary_objective": "full_transaction_fixed_guard_shadow"
                    },
                    "joint_solver": {
                        "solver_status": "finite_gap_angular_direction_found"
                    },
                }],
                "correction_accepted_steps": 0,
                "full_shadow_reduction": 0.0,
                "step_rejection_reason": "finite_radius_model_mismatch",
            },
        }
    }

    summary = g._local_feasible_intersection_summary(reports)

    assert summary["empty_geodesic_joint_feasible_intersection_case_uids"] == [
        "txn_zero:131"
    ]
    assert summary["zero_science_gradient_case_uids"] == ["txn_zero:131"]
    assert summary["finite_radius_model_mismatch_case_uids"] == ["txn_gap:170"]


def test_g1f2_runner_is_strictly_train_only():
    runner = Path(
        "scripts/run_refiner_v15_15g1f2_finite_gap_angular_feasibility_server.sh"
    ).read_text(encoding="utf-8")

    assert "--activation-aware-g1f2" in runner
    assert "--evaluation-role train_calibration" in runner
    assert "--evaluation-role development_validation" not in runner
    assert "--evaluation-role final_held_out" not in runner
    assert "--temporal-fd-near-zero-threshold 1e-5" in runner
    assert 'case_53_launched": False' in runner
    assert 'development_validation_launched": False' in runner
    assert 'final_held_out_launched": False' in runner
