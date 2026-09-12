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


def test_g1e_freezes_logarithmic_search_and_restoration_from_train_only():
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
        post_retraction_restoration=True,
        line_search_backtrack_count=12,
        line_search_decay=0.5,
        science_restoration_damping=1.0e-8,
        science_restoration_safety_fraction=0.25,
    )

    assert frozen["schema"] == g.G1E_TRAIN_CONTRACT_SCHEMA
    assert frozen["calibration_split"] == "train"
    assert frozen["validation_consumed_for_calibration"] is False
    assert frozen["development_case_53_consumed_for_calibration"] is False
    assert frozen["line_search_backtrack_count"] == 12
    assert frozen["line_search_scales"] == pytest.approx(
        [0.5**index for index in range(12)]
    )
    assert frozen["science_restoration_damping"] == pytest.approx(1.0e-8)


def test_g1e_science_margins_require_both_terms_to_strictly_improve():
    diagnostics = g._science_pass_margin_diagnostics(
        {
            "endpoint_delta": -0.1,
            "temporal_delta": 1.0e-5,
            "numeric_tolerance": {"endpoint": 1.0e-6, "temporal": 1.0e-6},
        }
    )

    assert diagnostics["trial_endpoint_margin_to_pass"] > 0.0
    assert diagnostics["trial_temporal_margin_to_pass"] < 0.0
    assert diagnostics["both_scientific_terms_strictly_improved"] is False


def test_g1e_restoration_solves_feasible_two_row_sphere_tangent_qp():
    current = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)
    taper = torch.ones_like(current)
    gradients = {
        "endpoint": torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64),
        "temporal": torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float64),
    }
    scientific = {
        "endpoint_delta": 0.1,
        "temporal_delta": 0.2,
        "numeric_tolerance": {"endpoint": 1.0e-6, "temporal": 1.0e-6},
    }

    restoration, diagnostics = g._minimum_norm_halfspace_restoration(
        current=current,
        mask=mask,
        taper=taper,
        gradients=gradients,
        scientific=scientific,
        damping=1.0e-12,
        safety_fraction=0.25,
    )

    assert restoration is not None
    assert diagnostics["restoration_status"] == (
        "linearized_science_feasible_step_found"
    )
    assert torch.dot(restoration[mask], current[mask]).item() == pytest.approx(0.0)
    assert restoration[0, 0, 1].item() < -0.1
    assert restoration[0, 0, 2].item() < -0.2


def test_g1e_reports_empty_opposed_science_halfspaces():
    current = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    mask = torch.ones_like(current, dtype=torch.bool)
    taper = torch.ones_like(current)
    gradients = {
        "endpoint": torch.tensor([[[0.0, 1.0]]], dtype=torch.float64),
        "temporal": torch.tensor([[[0.0, -1.0]]], dtype=torch.float64),
    }
    scientific = {
        "endpoint_delta": 0.1,
        "temporal_delta": 0.1,
        "numeric_tolerance": {"endpoint": 1.0e-6, "temporal": 1.0e-6},
    }

    restoration, diagnostics = g._minimum_norm_halfspace_restoration(
        current=current,
        mask=mask,
        taper=taper,
        gradients=gradients,
        scientific=scientific,
        damping=1.0e-12,
        safety_fraction=0.25,
    )

    assert restoration is None
    assert diagnostics["restoration_status"] == (
        "empty_linearized_science_feasible_intersection"
    )
    assert diagnostics["science_gradient_cosine_similarity"] == pytest.approx(-1.0)


def test_g1e_intersection_summary_requires_a_real_accepted_shadow_step():
    summary = g._local_feasible_intersection_summary(
        {
            "euclidean_projected_k2": {
                "txn:1": {
                    "history": [
                        {
                            "constraints": {
                                "primary_objective": (
                                    "full_transaction_fixed_guard_shadow"
                                )
                            }
                        }
                    ],
                    "correction_accepted_steps": 0,
                    "full_shadow_reduction": 0.0,
                    "step_rejection_reason": "empty_local_feasible_intersection",
                }
            },
            "riemannian_retraction_k5": {
                "txn:1": {
                    "history": [],
                    "correction_accepted_steps": 0,
                    "full_shadow_reduction": 0.0,
                }
            },
        }
    )

    assert summary["local_feasible_intersection_complete"] is False
    assert summary["empty_local_feasible_intersection_case_uids"] == ["txn:1"]


def test_g1e_source_declares_case_53_as_reused_development_evidence():
    assert g.REUSED_DEVELOPMENT_CASE_UID == "txn_0000_94bfdf553811:53"
    assert g.G1E_SCHEMA.endswith("science_feasibility_restoration_v1")


def test_g1e_runners_separate_reused_development_from_one_shot_final():
    development = (
        Path("scripts")
        / "run_refiner_v15_15g1e_post_retraction_science_restoration_server.sh"
    ).read_text(encoding="utf-8")
    final = (
        Path("scripts") / "run_refiner_v15_15g1e_final_held_out_server.sh"
    ).read_text(encoding="utf-8")

    assert "--evaluation-role train_calibration" in development
    assert "--evaluation-role development_validation" in development
    assert "development_validation_reused" in development
    assert 'final_held_out_launched": False' in development
    assert "FINAL_HELD_OUT_BANK" in final
    assert "--evaluation-role final_held_out" in final
    assert "V15_15G1E_FINAL_HELD_OUT_CONSUMED_" in final
    assert "reused development case 53 entered final held-out" in final
    assert "final_held_out_manifest_is_separately_sealed=true" in final
    assert "final bank lacks sealed manifest field" in final
    assert "final held-out source case overlaps reused development" in final


def test_g1e_core_enforces_final_transaction_and_source_case_disjointness():
    source = Path(
        "training/refiner_v15_15g_fixed_budget_correction.py"
    ).read_text(encoding="utf-8")

    assert "final held-out transaction overlaps train" in source
    assert "final held-out source case overlaps train" in source
    assert '"separate_final_held_out_manifest"' in source
    assert '"evaluation_split_manifest_content_sha256"' in source
