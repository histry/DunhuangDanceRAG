from pathlib import Path

import pytest
import torch

from motion_geometry.boundary_observables import boundary_metrics_torch
from training import refiner_v15_15g_fixed_budget_correction as g


def _guard_contract(anchor=2.0):
    return {
        "initial_anchor": {"cross_long.boundary": anchor},
        "relative_tolerance": {"cross_long.boundary": 0.1},
        "absolute_tolerance": {"cross_long.boundary": 0.05},
    }


def test_g1f1_freezes_coordinate_and_fd_contract_from_train():
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
        temporal_fd_epsilon_radians=1.0e-4,
        temporal_fd_relative_error_tolerance=0.1,
        temporal_fd_absolute_floor=1.0e-8,
    )

    assert frozen["schema"] == g.G1F1_TRAIN_CONTRACT_SCHEMA
    assert frozen["temporal_directional_consistency_repair"] is True
    assert frozen["jacobian_coordinate"] == "free_z_exactly_once"
    assert frozen["temporal_fd_epsilon_radians"] == 1.0e-4
    assert frozen["temporal_fd_relative_error_tolerance"] == 0.1
    assert frozen["temporal_fd_absolute_floor"] == 1.0e-8
    assert frozen["validation_consumed_for_calibration"] is False
    assert frozen["development_case_53_consumed_for_calibration"] is False


def test_g1f1_maps_z_to_physical_path_with_exactly_one_taper():
    current = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)
    taper = torch.tensor([[[0.25, 1.0, 0.5, 1.0]]], dtype=torch.float64)
    gradients = {
        "shadow": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]], dtype=torch.float64),
        "endpoint": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64),
        "temporal": torch.tensor([[[1.0, -0.25, 0.0, 0.0]]], dtype=torch.float64),
    }

    physical, audit = g._joint_geodesic_active_set_direction(
        current=current,
        mask=mask,
        taper=taper,
        gradients=gradients,
        svd_relative_cutoff=1.0e-6,
        direction_norm_floor=1.0e-8,
        directional_margin=1.0e-6,
        coordinate_consistent=True,
    )

    assert physical is not None
    z = audit.pop("_optimization_direction_z")
    assert torch.allclose(physical, z * taper, atol=1.0e-12, rtol=1.0e-12)
    assert torch.dot(physical[mask], current[mask]).item() == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert audit["taper_applications_after_autograd"] == 0
    assert audit["optimization_coordinate"] == "z"


def _patch_linear_temporal(monkeypatch, sign=1.0):
    monkeypatch.setattr(
        g,
        "_case_isolated_transaction_tangent",
        lambda baseline, trial, local_case: trial,
    )
    monkeypatch.setattr(g, "product_exp_torch", lambda baseline, tangent: tangent)
    monkeypatch.setattr(
        g.oracle.case_probe,
        "_case_terms",
        lambda candidate, batch, cfg: {
            "temporal": sign * candidate[:, 0, 1]
        },
    )


def test_g1f1_temporal_fd_matches_autograd_angular_derivative(monkeypatch):
    _patch_linear_temporal(monkeypatch, sign=1.0)
    current = torch.tensor([[[2.0, 0.0]]], dtype=torch.float64)
    direction = torch.tensor([[[0.0, 0.5]]], dtype=torch.float64)
    direction_z = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    gradient_z = torch.tensor([[[0.0, 0.5]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)

    audit = g._temporal_directional_consistency_probe(
        current=current,
        physical_direction=direction,
        optimization_direction_z=direction_z,
        temporal_gradient_z=gradient_z,
        mask=mask,
        epsilon_radians=1.0e-4,
        relative_error_tolerance=0.1,
        absolute_floor=1.0e-8,
        direction_norm_floor=1.0e-8,
        baseline=current,
        batch={},
        cfg=None,
        local_case=0,
    )

    assert audit["passed"] is True
    assert audit["status"] == "temporal_directional_consistency_passed"
    assert audit["autograd_temporal_angular_derivative"] == pytest.approx(2.0)
    assert audit["finite_difference_temporal_angular_derivative"] == pytest.approx(
        2.0, rel=1.0e-7
    )
    assert audit["geodesic_update_by_side"]["plus"][
        "input_physical_direction_used_without_reprojection"
    ] is True


def test_g1f1_temporal_fd_reports_sign_mismatch(monkeypatch):
    _patch_linear_temporal(monkeypatch, sign=-1.0)
    current = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    direction = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)

    audit = g._temporal_directional_consistency_probe(
        current=current,
        physical_direction=direction,
        optimization_direction_z=direction,
        temporal_gradient_z=direction,
        mask=mask,
        epsilon_radians=1.0e-4,
        relative_error_tolerance=0.1,
        absolute_floor=1.0e-8,
        direction_norm_floor=1.0e-8,
        baseline=current,
        batch={},
        cfg=None,
        local_case=0,
    )

    assert audit["passed"] is False
    assert audit["status"] == "temporal_directional_derivative_mismatch"
    assert audit["sign_mismatch"] is True


def test_g1f1_authoritative_temporal_graph_reaches_cross_frame_stencil():
    torch.manual_seed(7)
    joints = torch.randn(
        1, 10, 3, 3, dtype=torch.float64, requires_grad=True
    )
    seam = torch.zeros(1, 10, dtype=torch.float64)
    seam[:, 3:7] = 1.0

    temporal = boundary_metrics_torch(joints, seam, fps=30.0)[
        "temporal_energy"
    ].sum()
    temporal.backward()

    per_frame = joints.grad.abs().sum(dim=(0, 2, 3))
    assert torch.isfinite(joints.grad).all()
    assert int((per_frame > 0.0).sum()) >= 7


def test_g1f1_summary_separates_solver_empty_from_line_search_exhaustion():
    reports = {
        "geodesic_joint_sqp_k2": {
            "txn_empty:1": {
                "history": [{
                    "constraints": {
                        "primary_objective": "full_transaction_fixed_guard_shadow"
                    },
                    "joint_solver": {
                        "solver_status": "empty_geodesic_joint_feasible_intersection"
                    },
                }],
                "correction_accepted_steps": 0,
                "full_shadow_reduction": 0.0,
                "step_rejection_reason": "empty_geodesic_joint_feasible_intersection",
            },
            "txn_search:2": {
                "history": [{
                    "constraints": {
                        "primary_objective": "full_transaction_fixed_guard_shadow"
                    },
                    "joint_solver": {
                        "solver_status": "geodesic_joint_direction_found"
                    },
                }],
                "correction_accepted_steps": 0,
                "full_shadow_reduction": 0.0,
                "step_rejection_reason": "geodesic_authoritative_line_search_exhausted",
            },
        }
    }

    summary = g._local_feasible_intersection_summary(reports)

    assert summary["empty_geodesic_joint_feasible_intersection_case_uids"] == [
        "txn_empty:1"
    ]
    assert summary[
        "geodesic_authoritative_line_search_exhausted_case_uids"
    ] == ["txn_search:2"]


def test_g1f1_runner_is_train_first_and_does_not_launch_final_held_out():
    runner = Path(
        "scripts/run_refiner_v15_15g1f1_temporal_directional_consistency_server.sh"
    ).read_text(encoding="utf-8")

    assert "--activation-aware-g1f1" in runner
    assert "--evaluation-role train_calibration" in runner
    assert "--evaluation-role development_validation" not in runner
    assert "--evaluation-role final_held_out" not in runner
    assert "temporal-fd-epsilon-radians 1e-4" in runner
    assert 'case_53_launched": False' in runner
    assert 'final_held_out_launched": False' in runner
