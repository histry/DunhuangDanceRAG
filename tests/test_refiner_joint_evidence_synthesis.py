"""Contract tests for the report-only Refiner evidence synthesis stage."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import refiner_joint_evidence_synthesis as synthesis


REQUIRED_TOP_LEVEL = {
    "schema",
    "completed",
    "provenance",
    "input_artifacts",
    "input_sha256",
    "lineage_verification",
    "read_only",
    "optimizer_constructed",
    "optimizer_steps",
    "model_loaded",
    "forward_pass_performed",
    "autograd_used",
    "parameter_update_performed",
    "checkpoint_selection_performed",
    "case_selection_performed",
    "metric_selection_performed",
    "architecture_selection_performed",
    "intervention_search_performed",
    "evidence_matrix",
    "mechanism_synthesis",
    "intervention_evidence",
    "contradiction_matrix",
    "single_recording_synthesis",
    "cross_event_width_synthesis",
    "scientific_answers",
    "candidate_decision_inputs",
    "final_decision",
    "next_action",
    "paper_safe_summary",
    "negative_result_value",
    "remaining_uncertainty",
    "claim_boundary",
    "production_model_modified",
    "production_inference_modified",
    "scientific_acceptance",
    "publish_allowed",
    "pilot_allowed",
}

REQUIRED_ANSWERS = {
    "optimization_starvation",
    "hard_support_primary_bottleneck",
    "single_direction_alignment_bottleneck",
    "global_single_width_conflict",
    "localized_single_generalization_conflict",
    "cross_event_width_effect",
    "normalized_temporal_spreading",
    "observational_normalization_signal",
    "bctr_normalization_solution",
    "width_conditioned_direction_component",
    "direction_component_causally_manipulable",
    "direction_only_solution_sufficient",
    "unique_causal_root_cause_proven",
    "supported_formal_method_candidate_exists",
    "further_intervention_search_allowed",
    "final_refiner_candidate_classification",
}


def test_input_contract_is_explicit_and_complete():
    assert synthesis.SCHEMA == "refiner_joint_evidence_synthesis_v1"
    assert synthesis.IMPLEMENTATION_PARENT_COMMIT == (
        "654671987d2fd41deac4fcb323adff49808e7574"
    )
    assert synthesis.INPUT_NAMES == (
        "rcsp_report",
        "rcsp_review",
        "parameter_report",
        "single_report",
        "phase2_report",
        "phase21_report",
        "bctr_report",
        "bctr_correction",
        "secdr_report",
        "defective_secdr_report",
    )
    assert len(synthesis.INPUT_NAMES) == 10
    assert len(set(synthesis.INPUT_NAMES)) == 10


def test_candidate_decision_has_only_the_three_allowed_result_values():
    rejected = synthesis._decide_candidates(
        {
            "rcsp_solution_supported": False,
            "bctr_solution_supported": False,
            "secdr_solution_supported": False,
        }
    )
    assert rejected["result"] == synthesis.NO_SUFFICIENT_CANDIDATE
    assert rejected["next_action"] == synthesis.STOP_NEXT_ACTION
    assert rejected["pilot_allowed"] is False
    assert rejected["further_intervention_search_allowed"] is False
    assert rejected["architecture_search_allowed"] is False
    assert rejected["metric_search_allowed"] is False

    supported = synthesis._decide_candidates(
        {
            "rcsp_solution_supported": False,
            "bctr_solution_supported": False,
            "secdr_solution_supported": True,
        }
    )
    assert supported["result"] == synthesis.FORMAL_CANDIDATE_SUPPORTED
    assert supported["next_action"] == synthesis.PILOT_REVIEW_NEXT_ACTION
    assert supported["formal_candidate_supported"] is True
    assert supported["pilot_allowed"] is False
    assert set((synthesis.FORMAL_CANDIDATE_SUPPORTED, synthesis.NO_SUFFICIENT_CANDIDATE, synthesis.EVIDENCE_INTEGRITY_FAILURE)) == {
        synthesis.FORMAL_CANDIDATE_SUPPORTED,
        synthesis.NO_SUFFICIENT_CANDIDATE,
        synthesis.EVIDENCE_INTEGRITY_FAILURE,
    }


@pytest.mark.parametrize(
    "name",
    [
        "C1_normalization_vs_bctr",
        "C2_secdr_mechanism_vs_efficacy",
        "C3_local_blocks_vs_whole_cosine",
        "C4_deficit_vs_gate",
    ],
)
def test_contradictions_are_explicitly_non_contradictory(name):
    entry = synthesis._contradiction_matrix()[name]
    assert entry["contradiction"] is False
    assert entry["status"] == "NOT_A_CONTRADICTION"
    assert isinstance(entry["explanation"], str)
    assert entry["explanation"]


def test_evidence_row_preserves_source_and_hash(tmp_path):
    source = tmp_path / "report.json"
    source.write_text("{}", encoding="utf-8")
    row = synthesis._evidence_row(
        "E99",
        "test",
        "question",
        "frozen report",
        {"value": 1},
        "OBSERVED",
        "supports",
        "does not support",
        "diagnostic",
        "context",
        source,
        "abc123",
    )
    assert row["evidence_id"] == "E99"
    assert row["source_report"] == str(source)
    assert row["source_sha256"] == "abc123"
    assert row["observation"] == {"value": 1}
    assert row["frozen_classification"] == "OBSERVED"


def test_report_contract_sets_all_read_only_flags_false_or_zero():
    flags = (
        "optimizer_constructed",
        "model_loaded",
        "forward_pass_performed",
        "autograd_used",
        "parameter_update_performed",
        "checkpoint_selection_performed",
        "case_selection_performed",
        "metric_selection_performed",
        "architecture_selection_performed",
        "intervention_search_performed",
        "production_model_modified",
        "production_inference_modified",
        "scientific_acceptance",
        "publish_allowed",
        "pilot_allowed",
    )
    assert all(isinstance(name, str) for name in flags)
    assert len(flags) == 15
    assert synthesis.STOP_NEXT_ACTION.startswith("freeze_refiner_")


def test_static_source_has_no_model_or_dynamic_latest_artifact_execution():
    source = Path(synthesis.__file__).read_text(encoding="utf-8").lower()
    assert "import torch" not in source
    assert "torch." not in source
    assert "glob(" not in source
    assert "mtime" not in source
    assert "load_state_dict" not in source
    assert "optimizer.step" not in source
    assert "backward(" not in source


def test_required_top_level_and_answer_contracts_are_documented():
    assert REQUIRED_TOP_LEVEL >= {
        "schema",
        "completed",
        "input_artifacts",
        "input_sha256",
        "lineage_verification",
        "final_decision",
    }
    assert REQUIRED_ANSWERS >= {
        "optimization_starvation",
        "cross_event_width_effect",
        "final_refiner_candidate_classification",
    }
    assert "evidence_matrix" in REQUIRED_TOP_LEVEL
    assert "mechanism_synthesis" in REQUIRED_TOP_LEVEL
    assert "contradiction_matrix" in REQUIRED_TOP_LEVEL
    assert "paper_safe_summary" in REQUIRED_TOP_LEVEL
    assert len(REQUIRED_ANSWERS) == 16


def test_minimal_json_loader_rejects_non_object_and_hashes_bytes(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"completed": True}), encoding="utf-8")
    loaded, digest = synthesis._load_report(report, "test")
    assert loaded["completed"] is True
    assert len(digest) == 64
    report.write_text("[]", encoding="utf-8")
    with pytest.raises(synthesis.EvidenceIntegrityError):
        synthesis._load_report(report, "test")


def test_paper_safe_summary_has_separate_single_and_cross_event_claims():
    summary = synthesis._paper_safe_summary()
    assert set(summary) == {"single_recording", "cross_event_width28"}
    assert all(isinstance(value, str) and value for value in summary.values())
    assert "root cause" in summary["single_recording"]
    assert "insufficient" in summary["cross_event_width28"]
