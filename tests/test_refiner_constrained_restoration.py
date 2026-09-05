"""Server-side regression coverage for the v7 development solver."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from training.refiner_action_feasibility import (
    ACTION_DIM,
    FeasibilitySolverConfig,
    _dominant_constraints,
    _gate_residual_details,
    _hard_residual_components,
    _margin_models,
    _ordinary_candidate_admissible,
    _project_margin_cone,
    _restoration_acceptance,
    _safe_margin_step,
    _structured_probe_basis,
    solve_action_feasibility,
)


def _evaluation():
    return {
        "physical_pass": True,
        "fixed_reference_support": {"accepted": True},
        "fidelity_pass": True,
        "finite_pass": True,
        "endpoint_pass": True,
        "temporal_pass": False,
        "joint_pass": False,
        "action": {"contact_residual_max": 0.0, "support_outside_edit_max": 0.0},
        "solver_observable_excess": {"endpoint": 0.0, "temporal": 0.2},
        "solver_observable_margin": {"endpoint": 0.1, "temporal": -0.2},
        "hard_residual_components": {"canonical_metrics": {
            "joint_jerk": {"residual": 0.0, "minimum_margin": 0.01},
            "foot_penetration": {"residual": 0.0, "minimum_margin": 0.03},
            "fidelity_seam_jerk": {"residual": 0.0, "minimum_margin": 0.04},
            "rot6d_validity": {
                "residual": 0.0,
                "minimum_margin": None,
                "boundary_invariant_metrics": ["rot6d_nonfinite_ratio"],
            },
        }},
    }


def _axis(index):
    value = np.zeros((1, ACTION_DIM))
    value[0, index] = 1.0
    return value


def test_zero_residual_guard_is_never_reported_as_a_failure():
    components = _evaluation()["hard_residual_components"]
    assert _dominant_constraints(components) == {
        "dominant_failed_constraint": None,
        "dominant_guard_constraint": "joint_jerk",
    }
    components["canonical_metrics"]["joint_jerk"]["residual"] = 0.002
    assert _dominant_constraints(components)["dominant_failed_constraint"] == "joint_jerk"


def test_duplicate_layers_retain_stricter_margin_under_one_root():
    def gate(allowed):
        return {"detail": {
            "before_joint_jerk_mps3_max": 8.0,
            "candidate_joint_jerk_mps3_max": 9.0,
            "allowed_joint_jerk_mps3_max": allowed,
        }}

    with patch("training.refiner_action_feasibility._physical_metric_directions", return_value={}):
        components = _hard_residual_components({
            "physical_stage": gate(10.0), "fixed_reference_support": gate(8.0)
        })
    canonical = components["canonical_metrics"]
    assert list(canonical) == ["joint_jerk"]
    assert canonical["joint_jerk"]["residual"] == 0.125
    assert canonical["joint_jerk"]["minimum_margin"] == -0.125
    assert len(canonical["joint_jerk"]["metrics"]) == 2


def test_zero_limit_validity_metric_is_a_boundary_invariant():
    gate = {"detail": {
        "before_rot6d_nonfinite_ratio": 0.0,
        "candidate_rot6d_nonfinite_ratio": 0.0,
        "allowed_rot6d_nonfinite_ratio": 0.0,
        "before_rot6d_collinearity_abs_p99": 0.0,
        "candidate_rot6d_collinearity_abs_p99": 0.001,
        "allowed_rot6d_collinearity_abs_p99": 0.01,
    }}
    detail = _gate_residual_details(gate)
    assert detail["rot6d_nonfinite_ratio"]["margin_kind"] == "boundary_invariant"
    assert not detail["rot6d_nonfinite_ratio"]["safety_margin_eligible"]
    components = _hard_residual_components({"reference_fidelity": gate})
    validity = components["canonical_metrics"]["fidelity_rot6d_validity"]
    assert validity["minimum_margin"] == 0.9
    assert validity["boundary_invariant_metrics"] == [
        "fidelity:rot6d_nonfinite_ratio"
    ]
    gate["detail"]["candidate_rot6d_nonfinite_ratio"] = 0.001
    violated = _gate_residual_details(gate)["rot6d_nonfinite_ratio"]
    assert violated["margin_kind"] == "boundary_invariant"
    assert violated["residual"] > 0.0


def test_preparation_improves_margin_without_claiming_joint_success():
    before = _evaluation()
    after = copy.deepcopy(before)
    after["hard_residual_components"]["canonical_metrics"]["joint_jerk"]["minimum_margin"] = 0.02
    accepted, detail = _restoration_acceptance(before, after, 1.0e-7)
    assert accepted and detail["minimum_margin_increased"]
    assert after["joint_pass"] is False
    for gate in ("physical_pass", "fidelity_pass", "finite_pass", "endpoint_pass"):
        rejected = copy.deepcopy(after)
        rejected[gate] = False
        assert not _restoration_acceptance(before, rejected, 1.0e-7)[0]
    rejected = copy.deepcopy(after)
    rejected["fixed_reference_support"]["accepted"] = False
    assert not _restoration_acceptance(before, rejected, 1.0e-7)[0]
    for name in ("endpoint", "temporal"):
        rejected = copy.deepcopy(after)
        rejected["solver_observable_excess"][name] += 1.0e-9
        assert not _restoration_acceptance(before, rejected, 1.0e-7)[0]
    assert not _restoration_acceptance(before, before, 1.0e-7)[0]
    noise = copy.deepcopy(before)
    noise["hard_residual_components"]["canonical_metrics"]["joint_jerk"][
        "minimum_margin"
    ] += 1.0e-6
    accepted, detail = _restoration_acceptance(before, noise, 1.0e-7)
    assert not accepted
    assert detail["margin_increase_required"] > 1.0e-6


def test_cone_protects_two_physical_normals_and_endpoint_together():
    cfg = SimpleNamespace(product_refiner_root_cap_m=1.0, product_refiner_rotation_cap_rad=1.0)
    models = {"joint_jerk": [_axis(0)], "foot_penetration": [_axis(1)],
              "observable_endpoint": [_axis(2)]}
    margins = {name: 0.01 for name in models}
    direction = -_axis(0) - _axis(1) - _axis(2) + _axis(3)
    projected, evidence = _project_margin_cone(
        direction, models, margins, cfg, preserve_observables=("observable_endpoint",)
    )
    assert evidence["available"]
    assert projected is not None
    assert all(float(np.sum(gradient[0] * projected)) >= -1.0e-8 for gradient in models.values())
    assert projected[0, 3] > 0.0
    assert _project_margin_cone(-_axis(0), models, margins, cfg)[0] is None


def test_zero_margin_restoration_requires_both_derivatives_strictly_positive():
    cfg = SimpleNamespace(
        product_refiner_root_cap_m=1.0,
        product_refiner_rotation_cap_rad=1.0,
    )
    models = {"joint_jerk": [_axis(0)], "foot_penetration": [_axis(1)]}
    margins = {"joint_jerk": 0.0, "foot_penetration": 0.0}
    projected, evidence = _project_margin_cone(
        _axis(2), models, margins, cfg,
        require_positive=("joint_jerk", "foot_penetration"),
    )
    assert evidence["available"]
    assert float(np.sum(models["joint_jerk"][0] * projected)) > 0.0
    assert float(np.sum(models["foot_penetration"][0] * projected)) > 0.0
    assert min(evidence["projected_directional_derivatives"]["joint_jerk"]) > 0.0
    assert min(
        evidence["projected_directional_derivatives"]["foot_penetration"]
    ) > 0.0


def test_ordinary_candidate_cannot_spend_binding_margin_without_joint_pass():
    evidence = {"minimum_margin_before": 0.03, "minimum_margin_after": 0.02}
    assert not _ordinary_candidate_admissible(
        evidence, feasible_probe_exists=False, joint_pass=False, tolerance=1.0e-7
    )[0]
    assert _ordinary_candidate_admissible(
        evidence, feasible_probe_exists=False, joint_pass=True, tolerance=1.0e-7
    )[0]


def test_structured_probe_basis_is_seeded_and_spans_required_blocks():
    frames = 9
    cfg = SimpleNamespace(
        product_refiner_root_cap_m=1.0,
        product_refiner_rotation_cap_rad=1.0,
    )
    seam = np.zeros((frames, 1))
    seam[1:8] = 1.0
    case = SimpleNamespace(
        frames=frames,
        case_id="structured-case",
        cfg=cfg,
        seam=seam,
        root_mask=seam.copy(),
        joint_mask=np.broadcast_to(seam, (frames, 24)).copy(),
    )
    observable = np.zeros((frames, ACTION_DIM))
    observable[:, 3] = 1.0
    controls = FeasibilitySolverConfig(
        structured_probe_seed=42,
        structured_sparse_directions=3,
    )
    first, _ = _structured_probe_basis(case, {"endpoint": observable}, controls)
    second, _ = _structured_probe_basis(case, {"endpoint": observable}, controls)
    assert {
        "structured_endpoint_left",
        "structured_endpoint_center",
        "structured_endpoint_right",
        "structured_endpoint_symmetric",
        "structured_endpoint_antisymmetric",
        "structured_root_x_symmetric",
        "structured_root_y_antisymmetric",
        "structured_joint_group2_axis2_symmetric",
        "structured_sparse_2",
    }.issubset(first)
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_safe_step_uses_worst_one_sided_slope_and_refuses_missing_model():
    step, _ = _safe_margin_step(
        _axis(0), {"joint_jerk": [-_axis(0), -2.0 * _axis(0)]},
        {"joint_jerk": 0.02}, 0.25,
    )
    assert abs(step - 0.009) < 1.0e-12
    assert _safe_margin_step(_axis(0), {}, {"joint_jerk": 0.02}, 0.25)[0] == 0.0


def test_margin_fit_handles_nonorthogonal_span_and_both_sides():
    before = _evaluation()
    directions = {"endpoint": _axis(0), "temporal": _axis(0) + _axis(1)}
    expected = 2.0 * _axis(0) + 3.0 * _axis(1)

    def probe(case, action, sampled, stage, radius):
        rows = []
        for name, direction in sampled.items():
            for sign in (-1, 1):
                margin = 0.01 + sign * radius * float(np.sum(expected * direction))
                rows.append({
                    "objective": name, "sign": sign,
                    "endpoint_margin": 0.1, "temporal_margin": -0.2,
                    "hard_residual_components": {"canonical_metrics": {
                        "joint_jerk": {"minimum_margin": margin}
                    }},
                })
        return rows

    with patch("training.refiner_action_feasibility._finite_difference_reachability", side_effect=probe):
        models, probes, diagnostics = _margin_models(
            None, np.zeros_like(expected), before, directions, "temporal", 0.03
        )
    assert len(models["joint_jerk"]) == 6
    for gradient in models["joint_jerk"]:
        np.testing.assert_allclose(gradient, expected, atol=1.0e-10)
    assert {row["probe_scale"] for row in probes} == {1.0, 0.5, 0.25}
    assert any(not row["available"] for row in diagnostics)


def test_solver_accepts_preparation_then_rolls_back_if_final_gate_still_fails():
    pytest.importorskip("torch")
    cfg = SimpleNamespace(
        device="cpu", product_refiner_root_cap_m=1.0,
        product_refiner_rotation_cap_rad=1.0,
    )
    case = SimpleNamespace(frames=1, cfg=cfg, reference=np.zeros((1, 151)))

    def evaluate(case, action, **kwargs):
        result = _evaluation()
        result["hard_residual_components"]["canonical_metrics"]["joint_jerk"]["minimum_margin"] += float(action[0, 0])
        return result

    def losses(case, action):
        loss = (action[0, 0] - 1.0) ** 2
        return {name: loss for name in ("endpoint", "temporal", "joint")}, {}

    models = {name: [_axis(0)] for name in (
        "foot_penetration", "fidelity_seam_jerk", "observable_endpoint",
        "observable_temporal",
    )}
    models["joint_jerk"] = [_axis(0)]
    module = "training.refiner_action_feasibility."
    with (
        patch(module + "evaluate_action_candidate", side_effect=evaluate),
        patch(module + "_proxy_loss_components", side_effect=losses),
        patch(module + "_structured_probe_basis", return_value=({}, [])),
        patch(module + "_margin_models", return_value=(models, [], [])),
        patch(module + "_finite_difference_reachability", return_value=[]),
    ):
        result = solve_action_feasibility(
            case, solver_config=FeasibilitySolverConfig(max_iterations=1)
        )
    assert result.iterations[0]["accepted_phase"] == "hard_margin_restoration"
    assert result.iterations[0]["accepted"]
    assert result.final_evaluation["joint_pass"] is False
    assert result.rollback
    np.testing.assert_array_equal(result.returned_motion, case.reference)
