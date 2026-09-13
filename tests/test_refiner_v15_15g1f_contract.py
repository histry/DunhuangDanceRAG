import math
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


def test_g1f_freezes_exact_geodesic_train_contract():
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
        angular_max_radians=math.pi / 4.0,
        joint_svd_relative_cutoff=1.0e-6,
        joint_direction_norm_floor=1.0e-8,
        joint_directional_margin=1.0e-6,
    )

    assert frozen["schema"] == g.G1F_TRAIN_CONTRACT_SCHEMA
    assert frozen["calibration_split"] == "train"
    assert frozen["validation_consumed_for_calibration"] is False
    assert frozen["development_case_53_consumed_for_calibration"] is False
    assert frozen["exact_radius_geodesic_joint_active_set_sqp"] is True
    assert frozen["geodesic_update_requires_post_normalization"] is False
    assert frozen["angular_line_search_radians"] == pytest.approx(
        [(math.pi / 4.0) * 0.5**index for index in range(12)]
    )
    assert all(
        0.0 < value <= math.pi / 2.0
        for value in frozen["angular_line_search_radians"]
    )


def test_g1f_scope_projection_is_radially_orthogonal_and_exactly_masked():
    current = torch.tensor([[[1.0, 2.0, 0.0, 3.0]]], dtype=torch.float64)
    vector = torch.tensor([[[4.0, 5.0, 6.0, 7.0]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True, True, False]]])
    taper = torch.ones_like(current)

    projected = g._owned_sphere_tangent_projection(
        vector, current, mask, taper
    )

    assert projected.masked_select(~mask).abs().max().item() == 0.0
    assert torch.dot(projected[mask], current[mask]).item() == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_g1f_geodesic_update_preserves_radius_without_normalization():
    current = torch.tensor([[[3.0, 4.0, 0.0, 0.0]]], dtype=torch.float64)
    direction = torch.tensor([[[4.0, -3.0, 2.0, 9.0]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True, True, False]]])

    trial, ok, audit = g._exact_radius_geodesic_update(
        current, direction, mask, math.pi / 6.0, 1.0e-8
    )

    assert ok is True
    assert torch.linalg.vector_norm(trial[mask]).item() == pytest.approx(5.0)
    assert trial.masked_select(~mask).abs().max().item() == 0.0
    assert audit["post_update_normalization_applied"] is False


def test_g1f_joint_active_set_finds_three_term_descent_direction():
    current = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)
    taper = torch.ones_like(current)
    gradients = {
        "shadow": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]),
        "endpoint": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
        "temporal": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]),
    }

    direction, audit = g._joint_geodesic_active_set_direction(
        current=current,
        mask=mask,
        taper=taper,
        gradients=gradients,
        svd_relative_cutoff=1.0e-6,
        direction_norm_floor=1.0e-8,
        directional_margin=1.0e-6,
    )

    assert direction is not None
    assert audit["solver_status"] == "geodesic_joint_direction_found"
    assert audit["joint_jacobian_status"] == "full_rank_joint_jacobian"
    assert torch.dot(direction[mask], current[mask]).item() == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert all(
        value < 0.0
        for value in audit["directional_derivative_by_term"].values()
    )


@pytest.mark.parametrize(
    ("gradients", "expected"),
    [
        (
            {
                "shadow": torch.tensor([[[0.0, float("nan"), 0.0]]]),
                "endpoint": torch.tensor([[[0.0, 1.0, 0.0]]]),
                "temporal": torch.tensor([[[0.0, 0.0, 1.0]]]),
            },
            "nonfinite_joint_jacobian",
        ),
        (
            {
                "shadow": torch.tensor([[[0.0, 1.0, 0.0]]]),
                "endpoint": torch.zeros(1, 1, 3),
                "temporal": torch.tensor([[[0.0, 0.0, 1.0]]]),
            },
            "zero_science_gradient",
        ),
    ],
)
def test_g1f_distinguishes_nonfinite_and_zero_science_gradients(
    gradients, expected
):
    current = torch.tensor([[[1.0, 0.0, 0.0]]])
    mask = torch.ones_like(current, dtype=torch.bool)
    direction, audit = g._joint_geodesic_active_set_direction(
        current=current,
        mask=mask,
        taper=torch.ones_like(current),
        gradients=gradients,
        svd_relative_cutoff=1.0e-6,
        direction_norm_floor=1.0e-8,
        directional_margin=1.0e-6,
    )

    assert direction is None
    assert audit["joint_jacobian_status"] == expected


def test_g1f_reports_rank_deficiency_without_hiding_feasible_direction():
    current = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    mask = torch.ones_like(current, dtype=torch.bool)
    gradients = {
        "shadow": torch.tensor([[[0.0, 1.0, 0.0, 0.0]]]),
        "endpoint": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
        "temporal": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
    }

    direction, audit = g._joint_geodesic_active_set_direction(
        current=current,
        mask=mask,
        taper=torch.ones_like(current),
        gradients=gradients,
        svd_relative_cutoff=1.0e-6,
        direction_norm_floor=1.0e-8,
        directional_margin=1.0e-6,
    )

    assert direction is not None
    assert audit["joint_jacobian_status"] == "rank_deficient_joint_jacobian"


def test_g1f_reports_empty_opposed_joint_halfspaces():
    current = torch.tensor([[[1.0, 0.0, 0.0]]])
    mask = torch.ones_like(current, dtype=torch.bool)
    gradients = {
        "shadow": torch.tensor([[[0.0, 1.0, 0.0]]]),
        "endpoint": torch.tensor([[[0.0, -1.0, 0.0]]]),
        "temporal": torch.tensor([[[0.0, 0.0, 1.0]]]),
    }

    direction, audit = g._joint_geodesic_active_set_direction(
        current=current,
        mask=mask,
        taper=torch.ones_like(current),
        gradients=gradients,
        svd_relative_cutoff=1.0e-6,
        direction_norm_floor=1.0e-8,
        directional_margin=1.0e-6,
    )

    assert direction is None
    assert audit["joint_jacobian_status"] == "rank_deficient_joint_jacobian"
    assert audit["solver_status"] == (
        "empty_geodesic_joint_feasible_intersection"
    )


def test_g1f_runners_preserve_dev_and_one_shot_final_roles():
    development = Path(
        "scripts/run_refiner_v15_15g1f_exact_radius_geodesic_joint_sqp_server.sh"
    ).read_text(encoding="utf-8")
    final = Path(
        "scripts/run_refiner_v15_15g1f_final_held_out_server.sh"
    ).read_text(encoding="utf-8")

    assert "--activation-aware-g1f" in development
    assert "--evaluation-role train_calibration" in development
    assert "--evaluation-role development_validation" in development
    assert 'final_held_out_launched": False' in development
    assert "--activation-aware-g1f" in final
    assert "--evaluation-role final_held_out" in final
    assert "V15_15G1F_FINAL_HELD_OUT_CONSUMED_" in final
    assert "reused development case 53 entered final held-out" in final
