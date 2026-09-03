from __future__ import annotations

from pathlib import Path

import pytest

from training import refiner_final_evidence_synthesis as final


def test_schema_and_parent_are_frozen():
    assert final.SCHEMA == "refiner_final_evidence_synthesis_v1"
    assert final.IMPLEMENTATION_PARENT_COMMIT == (
        "5a06a03b97e4e2d46344a88eb6a3bcc48d41b7d2"
    )


def test_final_classification_and_stop_rule_are_frozen():
    assert final.FINAL_CLASSIFICATION == (
        "MULTIPLE_MANIPULABLE_MECHANISMS_WITHOUT_SUFFICIENT_SAFE_REFINER_CANDIDATE"
    )
    assert final.FINAL_NEXT_ACTION == (
        "freeze_final_refiner_evidence_and_transition_to_manuscript_synthesis"
    )


def test_formal_frozen_hashes_are_explicit():
    assert final.EXPECTED_RPA_REPORT_SHA256 == (
        "08fd36d5bd504a16cb5f18348358e8e236008e0758481e7c5372dddca0c6808e"
    )
    assert final.EXPECTED_RPA_ADAPTER_SHA256 == (
        "2b6a7ae7d08721bcff5b174403a7137ec7494c2f871a21ffc5a63bdc7be70110"
    )
    assert final.EXPECTED_RPA_UPDATES_SHA256 == (
        "aedcf96068976ead5988d055af248b067e641849214365ba9fea3fdee35f0a86"
    )
    assert final.EXPECTED_DIRECTION_REPORT_SHA256 == (
        "e71397bb72afbf8f7e27b7ff141e2a04a49bcf7fcceeda980beb0b07d19afba6"
    )


def _joint():
    return {
        "schema": final.JOINT_SCHEMA,
        "completed": True,
        "read_only": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "model_loaded": False,
        "forward_pass_performed": False,
        "autograd_used": False,
        "parameter_update_performed": False,
        "checkpoint_selection_performed": False,
        "case_selection_performed": False,
        "metric_selection_performed": False,
        "architecture_selection_performed": False,
        "intervention_search_performed": False,
        "final_decision": {
            "result": final.PRE_RPA_DECISION,
            "formal_candidate_supported": False,
        },
        "no_further_intervention_search": True,
        "scientific_answers": {
            "direction_component_causally_manipulable": True,
            "direction_only_solution_sufficient": False,
            "unique_causal_root_cause_proven": False,
        },
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
    }


def test_validate_joint_accepts_frozen_terminal_ledger():
    info = final._validate_joint(_joint())
    assert info["decision"] == final.PRE_RPA_DECISION
    assert info["direction_causally_manipulable"] is True
    assert info["direction_only_sufficient"] is False


def test_validate_joint_fails_closed_if_candidate_was_supported():
    value = _joint()
    value["final_decision"]["formal_candidate_supported"] = True
    with pytest.raises(final.EvidenceIntegrityError):
        final._validate_joint(value)


def _direction():
    return {
        "schema": final.DIRECTION_SCHEMA,
        "completed": True,
        "read_only": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_update_performed": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "case_selection_performed": False,
        "metric_selection_performed": False,
        "architecture_selection_performed": False,
        "fixed_final_case_count": 64,
        "gradient_correction": {
            "primary_space": "raw_all_geometry",
            "comparison_gradient_shared_by_RCSP_and_RPA": True,
            "model_parameter_autograd_used": False,
        },
        "corrected_direction_conditions": {
            "H_single_direction_improved": True,
            "I_cross28_direction_improved": False,
        },
        "final_interpretation": {
            "classification": final.DIRECTION_CLASSIFICATION,
        },
        "formal_candidate_decision": final.RPA_DECISION,
        "decision_invariance_check": {
            "decision_invariant": True,
            "result": final.RPA_DECISION,
        },
        "state_integrity": {
            "base_unchanged": True,
            "rcsp_unchanged": True,
            "rpa_unchanged": True,
            "frozen_inputs_unchanged": True,
            "model_parameter_gradients_none": True,
        },
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
    }


def test_direction_correction_resolves_h_true_i_false():
    info = final._validate_direction(_direction())
    assert info["corrected_H_single_direction_improved"] is True
    assert info["corrected_I_cross28_direction_improved"] is False
    assert info["formal_candidate_decision"] == final.RPA_DECISION


def test_direction_decision_invariance_is_required():
    value = _direction()
    value["decision_invariance_check"]["decision_invariant"] = False
    with pytest.raises(final.EvidenceIntegrityError):
        final._validate_direction(value)


def test_mechanism_synthesis_closes_single_direction_as_insufficient():
    mechanisms = final._build_mechanism_synthesis()
    single = mechanisms["single_direction_alignment"]
    assert single["status"] == "CAUSALLY_SUPPORTED_BUT_INSUFFICIENT"
    assert single["corrected_H"] is True
    assert single["temporal_gate_rescue"] == "0/32 -> 0/32"
    assert single["sufficient_solution"] is False


def test_rpa_mechanism_never_becomes_method_support():
    mechanisms = final._build_mechanism_synthesis()
    rpa = mechanisms["rpa_lrta"]
    assert rpa["single_direction_improved"] is True
    assert rpa["cross28_direction_improved"] is False
    assert rpa["endpoint_regressions"] == 3
    assert rpa["physical_regressions"] == 1
    assert rpa["sufficient_safe_method"] is False


def test_paper_summary_preserves_claim_boundary():
    summary = final._paper_safe_summary()
    assert "genuine manipulable mechanism" in summary["single_recording"]
    assert "0/32" in summary["single_recording"]
    assert "0/16" in summary["cross_event_width28"]
    assert "three endpoint regressions" in summary["method_boundary"]
    assert "one physical regression" in summary["method_boundary"]


def test_source_has_no_ml_execution_dependencies():
    source = Path(final.__file__).read_text(encoding="utf-8")
    forbidden = (
        "import torch",
        "from torch",
        "torch.optim",
        "torch.load",
        ".backward()",
        "autograd.grad",
        "checked_refiner_step",
        "motion_models",
    )
    for token in forbidden:
        assert token not in source


def test_source_has_no_latest_artifact_search():
    source = Path(final.__file__).read_text(encoding="utf-8")
    assert "glob(" not in source
    assert "rglob(" not in source
    assert "os.walk" not in source
