import pytest
import torch

from training import refiner_v15_15g_fixed_budget_correction as g


def test_fixed_guard_shadow_reconstructs_authoritative_formula():
    margins, details = g._fixed_guard_shadow_from_details(
        {
            "cross_short.boundary": {
                "fixed_anchor": 2.0,
                "candidate": 2.21,
                "absolute_limit": 2.2,
                "numeric_tolerance": 0.02,
            }
        },
        {"cross_short.boundary": 0.1},
        {"cross_short.boundary": 0.05},
    )

    assert margins["cross_short.boundary"] == pytest.approx(-0.01)
    row = details["cross_short.boundary"]
    assert row["fixed_anchor_allowance"] == pytest.approx(0.2)
    assert row["absolute_limit_matches_exact_guard"] is True
    assert row["passed"] is True


def test_discriminative_status_uses_observables_and_abstains(monkeypatch):
    dimension = len(g.G1_SEVERITY_CHANNELS)
    identity = torch.eye(dimension, dtype=torch.float64).tolist()
    envelope = {
        "final_models": {
            "single": {
                "center": [0.0] * dimension,
                "scale": [1.0] * dimension,
                "precision": identity,
            },
            "cross": {
                "center": [10.0] * dimension,
                "scale": [1.0] * dimension,
                "precision": identity,
            },
        },
        "single_distance_threshold": 2.0,
        "cross_distance_threshold": 2.0,
        "single_discriminant_upper": -1.0,
        "activation_discriminant_threshold": 1.0,
    }

    def status_for(value):
        monkeypatch.setattr(
            g,
            "_observable_severity",
            lambda sample: {
                key: value for key in g.G1_SEVERITY_CHANNELS
            },
        )
        return g._discriminative_conformal_status(
            {"teacher_kind": "must_not_be_read", "audit_group": "hidden"},
            envelope,
        )

    assert status_for(0.0)["activation_supported_by_observables"] is False
    assert status_for(0.0)["severity_abstained"] is False
    assert status_for(10.0)["activation_supported_by_observables"] is True
    assert status_for(5.0)["severity_abstained"] is True


def test_g1c_report_names_keep_physical_and_shadow_margins_distinct():
    renamed = g._g1c_name_case_physical_diagnostics({
        "candidate_guard_signed_margin_by_term": {"boundary": 0.1},
        "history": [{
            "maximum_positive_guard_signed_margin": 0.1,
            "accepted_guard_margin_reduction": 0.05,
        }],
    })

    serialized_names = repr(renamed)
    assert "guard_signed_margin" not in serialized_names
    assert renamed["candidate_case_physical_signed_margin_by_term"] == {
        "boundary": 0.1
    }
    assert renamed["history"][0][
        "accepted_case_physical_margin_reduction"
    ] == pytest.approx(0.05)


def test_g1c_locks_exact_adapter_incumbent(monkeypatch):
    variants = {
        "adapter": torch.tensor([[[1.0]]]),
        "riemannian_retraction_k5": torch.tensor([[[2.0]]]),
    }
    monkeypatch.setattr(
        g,
        "_discriminative_conformal_status",
        lambda sample, envelope: {
            "activation_supported_by_observables": True,
            "severity_abstained": False,
        },
    )

    def evidence(**kwargs):
        is_adapter = kwargs["tangent"] is variants["adapter"]
        return {
            "case_physical_signed_margin_by_term": {"boundary": -0.1},
            "maximum_positive_case_physical_signed_margin": 0.0,
            "full_transaction_fixed_guard_shadow_margin_by_term": {
                "boundary": -0.1 if is_adapter else -1.0
            },
            "full_transaction_fixed_guard_shadow_detail_by_term": {},
            "maximum_full_transaction_fixed_guard_shadow_margin": (
                -0.1 if is_adapter else -1.0
            ),
            "maximum_positive_full_transaction_fixed_guard_shadow_margin": 0.0,
            "fixed_guard_shadow_passed": True,
            "fixed_guard_shadow_matches_exact_guard": True,
            "exact_raw_closure_passed": True,
            "fixed_guard_passed": True,
            "fixed_guard_blockers": [],
            "case_scientific": {
                "endpoint_delta": -0.1 if is_adapter else -1.0,
                "temporal_delta": -0.1 if is_adapter else -1.0,
                "passed": True,
            },
            "radius_rms": 1.0e-4,
            "radius_equality_resolved": True,
            "workspace_observable_resolved": True,
            "scope_safe": True,
            "outside_scope_abs_max": 0.0,
        }

    monkeypatch.setattr(g, "_g1c_candidate_evidence", evidence)
    selected, decisions, counts = g._g1c_fixed_guard_shadow_selection(
        model=object(),
        variants=variants,
        correction_reports={},
        samples=[{
            "case_uid": "txn:0",
            "case_index": 0,
            "transaction_id": "txn",
            "local_case_index": 0,
            "teacher_kind": "exact_projected_direction",
            "audit_group": "cross_long",
        }],
        domains={},
        ownership=torch.ones_like(variants["adapter"], dtype=torch.bool),
        baseline_terms={
            "endpoint": torch.tensor([1.0]),
            "temporal": torch.tensor([1.0]),
        },
        cfg=object(),
        target_rms=1.0e-4,
        workspace_floor=1.0e-8,
        severity_envelope={},
    )

    assert torch.equal(selected, variants["adapter"])
    assert decisions["txn:0"]["selected_method"] == "adapter"
    assert decisions["txn:0"]["adapter_incumbent_locked"] is True
    assert counts == {"adapter": 1}
