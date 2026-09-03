"""Read-only joint synthesis of the frozen Refiner evidence ledger.

This stage consumes explicit JSON reports only.  It deliberately has no model,
tensor, checkpoint, optimizer, or metric execution dependency: all observations
are copied from and cross-checked against completed upstream reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "refiner_joint_evidence_synthesis_v1"
IMPLEMENTATION_PARENT_COMMIT = "654671987d2fd41deac4fcb323adff49808e7574"

RCSP_SCHEMA = "refiner_role_conditioned_support_projection_experiment_v1"
RCSP_REVIEW_SCHEMA = "refiner_role_conditioned_support_projection_result_review_v1"
PARAMETER_SCHEMA = "refiner_rcsp_single_direction_attribution_v1"
SINGLE_SCHEMA = "refiner_single_direction_decomposition_audit_v1"
PHASE2_SCHEMA = "refiner_cross_width_normalization_audit_v1"
PHASE21_SCHEMA = "refiner_width_mechanism_adjudication_audit_v1"
BCTR_SCHEMA = "refiner_boundary_crossing_temporal_reduction_intervention_v1"
BCTR_CORRECTION_SCHEMA = "refiner_bctr_reporting_correction_v1"
SECDR_SCHEMA = "refiner_support_extent_conditioned_direction_rotation_intervention_v1"

RCSP_CLASSIFICATION = "ROLE_CONDITIONING_USEFUL_BUT_WIDTH_DEPENDENT_MECHANISM_REMAINS"
PHASE21_CLASSIFICATION = "MIXED_WIDTH_MECHANISM"
BCTR_RESULT = "METRIC_SUPPORT_TIME_INTERVENTION_NOT_SUPPORTED"
SECDR_RESULT = "DIRECTION_MECHANISM_WITHOUT_SUFFICIENT_EFFICACY"
OLD_SECDR_RESULT = "WIDTH_CONDITIONED_DIRECTION_INTERVENTION_NOT_SUPPORTED"
OLD_SECDR_RUNTIME_COMMIT = "534da47fe5939a9b09eb998c68537f49f9adf70d"
CORRECTED_SECDR_RUNTIME_COMMIT = "654671987d2fd41deac4fcb323adff49808e7574"

FORMAL_CANDIDATE_SUPPORTED = "FORMAL_REFINER_METHOD_CANDIDATE_SUPPORTED"
NO_SUFFICIENT_CANDIDATE = "MECHANISMS_IDENTIFIED_BUT_NO_SUFFICIENT_METHOD_CANDIDATE"
EVIDENCE_INTEGRITY_FAILURE = "EVIDENCE_INTEGRITY_FAILURE"
STOP_NEXT_ACTION = "freeze_refiner_diagnostic_findings_and_stop_candidate_development"
PILOT_REVIEW_NEXT_ACTION = "freeze_supported_candidate_for_separate_pilot_authorization_review"
INTEGRITY_FAILURE_NEXT_ACTION = "fail_closed_on_evidence_integrity_failure"

PRIMARY_GROUPS = (
    "seen/cross_event/10",
    "seen/cross_event/28",
    "new_position/cross_event/10",
    "new_position/cross_event/28",
)
INPUT_NAMES = (
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


class EvidenceIntegrityError(ValueError):
    """Raised when a frozen input or cross-report contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _load_report(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise EvidenceIntegrityError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceIntegrityError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceIntegrityError(f"{label} must be a JSON object")
    return value, _sha256(path)


def _get(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _required(value: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    result = _get(value, *path)
    if result is None:
        raise EvidenceIntegrityError(f"{label} is missing")
    return result


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise EvidenceIntegrityError(f"{label}: expected {expected!r}, got {value!r}")


def _false_flags(report: Mapping[str, Any], label: str, fields: Sequence[str]) -> None:
    for field in fields:
        _exact(report.get(field), False, f"{label}.{field}")


def _canonical(path_value: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise EvidenceIntegrityError(f"{label} path is missing")
    return Path(path_value).resolve()


def _same_path(actual: Any, expected: Path, label: str) -> None:
    if _canonical(actual, label) != expected.resolve():
        raise EvidenceIntegrityError(f"{label} path mismatch")


def _hash_aliases(mapping: Any, names: Sequence[str], label: str) -> str:
    if not isinstance(mapping, Mapping):
        raise EvidenceIntegrityError(f"{label} hash mapping is missing")
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str):
            return value
    raise EvidenceIntegrityError(f"{label} hash is missing")


def _validate_read_only(report: Mapping[str, Any], label: str, *, optimizer_steps: int | None = None) -> None:
    _exact(report.get("completed"), True, f"{label}.completed")
    if optimizer_steps is not None:
        _exact(report.get("optimizer_steps"), optimizer_steps, f"{label}.optimizer_steps")
    _false_flags(
        report,
        label,
        (
            "production_model_modified",
            "production_inference_modified",
            "scientific_acceptance",
            "publish_allowed",
            "pilot_allowed",
        ),
    )


def _validate_rcsp(
    report: Mapping[str, Any], review: Mapping[str, Any], report_hash: str
) -> dict[str, Any]:
    _exact(report.get("schema"), RCSP_SCHEMA, "RCSP schema")
    _validate_read_only(report, "RCSP", optimizer_steps=400)
    _exact(report.get("base_model_frozen"), True, "RCSP base_model_frozen")
    _exact(report.get("adapter_only_training"), True, "RCSP adapter_only_training")
    _exact(review.get("schema"), RCSP_REVIEW_SCHEMA, "RCSP review schema")
    _exact(review.get("completed"), True, "RCSP review completed")
    _exact(_get(review, "source_report", "sha256"), report_hash, "RCSP review source hash")
    _exact(
        review.get("measurement_recomputation_verified"),
        True,
        "RCSP review measurement recomputation",
    )
    _exact(
        _get(review, "formal_conclusion", "classification"),
        RCSP_CLASSIFICATION,
        "RCSP review classification",
    )
    _exact(
        _get(review, "formal_conclusion", "role_conditioning_alone_sufficient"),
        False,
        "RCSP role-conditioning sufficiency",
    )
    _exact(
        _get(review, "corrected_scientific_answers", "physical_geometry_or_clean_regression"),
        False,
        "RCSP safety regression",
    )
    _false_flags(review, "RCSP review", ("production_model_modified", "scientific_acceptance", "pilot_allowed"))
    return {
        "classification": _get(review, "formal_conclusion", "classification"),
        "role_conditioning_alone_sufficient": False,
        "measurement_recomputation_verified": True,
    }


def _validate_parameter(
    report: Mapping[str, Any], report_hash: str, rcsp_hash: str
) -> dict[str, Any]:
    _exact(report.get("schema"), PARAMETER_SCHEMA, "parameter attribution schema")
    _validate_read_only(report, "parameter attribution", optimizer_steps=0)
    _exact(
        _get(report, "gradient_protocol", "parameter_update_performed"),
        False,
        "parameter attribution update flag",
    )
    _exact(
        _get(report, "provenance", "rcsp_sha256", "report.json"),
        rcsp_hash,
        "parameter attribution RCSP hash",
    )
    _exact(
        _get(report, "scientific_answers", "all_single_parameter_gradients_nonzero"),
        True,
        "parameter attribution nonzero gradients",
    )
    return {
        "hash": report_hash,
        "all_single_parameter_gradients_nonzero": True,
        "scientific_answers": report.get("scientific_answers", {}),
    }


def _validate_single(
    report: Mapping[str, Any], report_hash: str, parameter_path: Path, parameter_hash: str
) -> dict[str, Any]:
    _exact(report.get("schema"), SINGLE_SCHEMA, "single decomposition schema")
    _validate_read_only(report, "single decomposition", optimizer_steps=0)
    _exact(_get(report, "case_counts", "overall"), 64, "single decomposition overall cases")
    _exact(_get(report, "case_counts", "single_recording"), 32, "single decomposition single cases")
    _exact(_get(report, "case_counts", "cross_event"), 32, "single decomposition cross cases")
    _same_path(report.get("parameter_attribution"), parameter_path, "single decomposition parameter")
    _exact(
        _get(report, "provenance", "parameter_attribution", "sha256"),
        parameter_hash,
        "single decomposition parameter hash",
    )
    _exact(
        _get(report, "scientific_answers", "single_direction_decomposition"),
        "MULTIPLE_MECHANISMS_SUPPORTED",
        "single decomposition classification",
    )
    return {"hash": report_hash, "scientific_answers": report.get("scientific_answers", {})}


def _validate_phase2(
    report: Mapping[str, Any],
    report_hash: str,
    rcsp_directory: Path,
    parameter_path: Path,
    parameter_hash: str,
    single_path: Path,
    single_hash: str,
) -> dict[str, Any]:
    _exact(report.get("schema"), PHASE2_SCHEMA, "Phase 2 schema")
    _validate_read_only(report, "Phase 2", optimizer_steps=0)
    _exact(_get(report, "primary_cohort", "cases"), 32, "Phase 2 primary cases")
    _exact(_get(report, "primary_cohort", "role"), "cross_event", "Phase 2 primary role")
    _exact(report.get("fake_case_pairing_performed"), False, "Phase 2 fake pairing")
    _exact(_get(report, "provenance", "rcsp_directory"), str(rcsp_directory), "Phase 2 RCSP directory")
    _same_path(_get(report, "provenance", "parameter_attribution_report"), parameter_path, "Phase 2 parameter report")
    _same_path(_get(report, "provenance", "single_decomposition_report"), single_path, "Phase 2 single report")
    hashes = _required(report.get("provenance", {}), ("hashes",), "Phase 2 hashes")
    _exact(_hash_aliases(hashes, ("parameter_attribution_report",), "Phase 2 parameter"), parameter_hash, "Phase 2 parameter hash")
    _exact(_hash_aliases(hashes, ("single_decomposition_report",), "Phase 2 single"), single_hash, "Phase 2 single hash")
    return {
        "hash": report_hash,
        "width_audit_classification": _get(report, "scientific_answers", "width_audit_classification"),
        "report": report,
    }


def _validate_phase21(
    report: Mapping[str, Any], report_hash: str, phase2_path: Path, phase2_hash: str
) -> dict[str, Any]:
    _exact(report.get("schema"), PHASE21_SCHEMA, "Phase 2.1 schema")
    _validate_read_only(report, "Phase 2.1", optimizer_steps=0)
    _exact(_get(report, "primary_cohort", "cases"), 32, "Phase 2.1 primary cases")
    _exact(_get(report, "adjudication", "adjudicated_primary_mechanism"), PHASE21_CLASSIFICATION, "Phase 2.1 classification")
    _exact(
        _get(report, "adjudication", "counterfactual_mask_parity_verified"),
        True,
        "Phase 2.1 counterfactual mask parity",
    )
    evidence = _required(report.get("adjudication", {}), ("normalization_evidence",), "Phase 2.1 normalization evidence")
    spreading = _required(report.get("adjudication", {}), ("temporal_spreading_evidence",), "Phase 2.1 spreading evidence")
    direction = _required(report.get("adjudication", {}), ("width_conditioned_direction_evidence",), "Phase 2.1 direction evidence")
    for name, value, expected in (
        ("normalization", evidence, True),
        ("temporal spreading", spreading, False),
        ("direction", direction, True),
    ):
        _exact(_get(value, "seen"), expected, f"Phase 2.1 {name}.seen")
        _exact(_get(value, "new_position"), expected, f"Phase 2.1 {name}.new_position")
    _same_path(_get(report, "provenance", "phase2_report"), phase2_path, "Phase 2.1 Phase 2")
    _exact(_get(report, "provenance", "phase2_report_sha256"), phase2_hash, "Phase 2.1 Phase 2 hash")
    _exact(_get(report, "lineage", "no_latest_artifact_search"), True, "Phase 2.1 artifact search contract")
    return {
        "hash": report_hash,
        "normalization_observed": True,
        "temporal_spreading_supported": False,
        "direction_observed": True,
    }


def _validate_bctr(
    report: Mapping[str, Any], report_hash: str, phase21_path: Path, phase21_hash: str
) -> dict[str, Any]:
    _exact(report.get("schema"), BCTR_SCHEMA, "BCTR schema")
    _validate_read_only(report, "BCTR", optimizer_steps=0)
    _exact(_get(report, "primary_cohort", "cases"), 32, "BCTR primary cases")
    _same_path(_get(report, "provenance", "phase21_report"), phase21_path, "BCTR Phase 2.1")
    _exact(_get(report, "provenance", "phase21_report_sha256"), phase21_hash, "BCTR Phase 2.1 hash")
    _exact(_get(report, "decision", "result"), BCTR_RESULT, "BCTR decision")
    _exact(_get(report, "decision", "overall_supported"), False, "BCTR overall support")
    _exact(_get(report, "decision", "split_supported"), {"seen": False, "new": False}, "BCTR split support")
    _exact(report.get("no_further_metric_search"), True, "BCTR stopping rule")
    _exact(_get(report, "parity", "current_metric_parity_verified"), True, "BCTR current parity")
    _exact(_get(report, "parity", "model_output_unchanged"), True, "BCTR model parity")
    return {"hash": report_hash, "solution_supported": False}


def _validate_bctr_correction(
    report: Mapping[str, Any], report_hash: str, bctr_path: Path, bctr_hash: str
) -> dict[str, Any]:
    _exact(report.get("schema"), BCTR_CORRECTION_SCHEMA, "BCTR correction schema")
    _exact(report.get("completed"), True, "BCTR correction completed")
    _exact(report.get("optimizer_steps"), 0, "BCTR correction optimizer_steps")
    _exact(report.get("model_loaded"), False, "BCTR correction model_loaded")
    _exact(report.get("inference_performed"), False, "BCTR correction inference_performed")
    _exact(report.get("metric_recomputed"), False, "BCTR correction metric_recomputed")
    _same_path(_get(report, "provenance", "source_bctr_report"), bctr_path, "BCTR correction source")
    _exact(_get(report, "provenance", "source_bctr_report_sha256"), bctr_hash, "BCTR correction source hash")
    _exact(_get(report, "provenance", "source_decision"), BCTR_RESULT, "BCTR correction source decision")
    for field in ("source_report_modified", "measurements_changed", "decision_inputs_changed", "scientific_classification_changed"):
        _exact(_get(report, "correction", field), False, f"BCTR correction {field}")
        _exact(report.get(field), False, f"BCTR correction top-level {field}")
    _exact(_get(report, "correction", "recomputed_decision_same"), True, "BCTR correction decision identity")
    _exact(_get(report, "decision", "recomputed_same_as_source"), True, "BCTR correction recomputation")
    _exact(_get(report, "decision", "result"), BCTR_RESULT, "BCTR corrected decision")
    return {"hash": report_hash, "source_unchanged": True, "solution_supported": False}


def _validate_defective_secdr(report: Mapping[str, Any], report_hash: str) -> dict[str, Any]:
    _exact(report.get("schema"), SECDR_SCHEMA, "historical SECDR schema")
    _exact(report.get("completed"), True, "historical SECDR completed")
    _exact(_get(report, "provenance", "runtime_commit"), OLD_SECDR_RUNTIME_COMMIT, "historical SECDR runtime")
    _exact(_get(report, "decision", "result"), OLD_SECDR_RESULT, "historical SECDR decision")
    return {
        "hash": report_hash,
        "historical_artifact": True,
        "implementation_defective": True,
        "scientifically_reused": False,
    }


def _validate_secdr(
    report: Mapping[str, Any],
    report_hash: str,
    phase21_path: Path,
    phase21_hash: str,
    bctr_path: Path,
    bctr_hash: str,
    correction_path: Path,
    correction_hash: str,
    defective_path: Path,
    defective_hash: str,
    rcsp_directory: Path,
    parameter_path: Path,
    single_path: Path,
) -> dict[str, Any]:
    _exact(report.get("schema"), SECDR_SCHEMA, "corrected SECDR schema")
    _validate_read_only(report, "corrected SECDR")
    _exact(_get(report, "provenance", "runtime_commit"), CORRECTED_SECDR_RUNTIME_COMMIT, "corrected SECDR runtime")
    _same_path(_get(report, "provenance", "phase21_report"), phase21_path, "SECDR Phase 2.1")
    _exact(_get(report, "provenance", "phase21_report_sha256"), phase21_hash, "SECDR Phase 2.1 hash")
    _same_path(_get(report, "provenance", "bctr_report"), bctr_path, "SECDR BCTR")
    _exact(_get(report, "provenance", "bctr_report_sha256"), bctr_hash, "SECDR BCTR hash")
    _same_path(_get(report, "provenance", "bctr_correction_report"), correction_path, "SECDR BCTR correction")
    _exact(_get(report, "provenance", "bctr_correction_report_sha256"), correction_hash, "SECDR correction hash")
    _same_path(_get(report, "provenance", "previous_defective_secdr_report"), defective_path, "SECDR historical report")
    _exact(_get(report, "provenance", "previous_defective_secdr_report_sha256"), defective_hash, "SECDR historical hash")
    _same_path(_get(report, "provenance", "rcsp_directory"), rcsp_directory, "SECDR RCSP directory")
    _same_path(_get(report, "provenance", "parameter_attribution_report"), parameter_path, "SECDR parameter report")
    _same_path(_get(report, "provenance", "single_decomposition_report"), single_path, "SECDR single report")
    correction = _required(report, ("implementation_correction",), "SECDR implementation correction")
    _exact(correction.get("defect_corrected"), True, "SECDR defect correction")
    _exact(correction.get("type"), "ZERO_GRADIENT_DEADLOCK", "SECDR correction type")
    _exact(correction.get("previous_result_scientifically_reused"), False, "SECDR historical reuse")
    preflight = _required(report, ("zero_start_trainability_preflight",), "SECDR zero-start preflight")
    _exact(preflight.get("passed"), True, "SECDR preflight passed")
    _exact(preflight.get("any_gradient_nonzero"), True, "SECDR preflight gradient")
    _exact(preflight.get("parameters_unchanged_after_probe"), True, "SECDR preflight parameter immutability")
    _exact(preflight.get("gradients_cleared_after_probe"), True, "SECDR preflight gradient cleanup")
    training = _required(report, ("training",), "SECDR training")
    _exact(training.get("steps"), 400, "SECDR training steps")
    _exact(training.get("accepted_steps"), 400, "SECDR accepted steps")
    _exact(training.get("rollback_steps"), 0, "SECDR rollback steps")
    _exact(training.get("retained_parameter_update_performed"), True, "SECDR retained update")
    _exact(_get(report, "control", "exact_rcsp_parity"), True, "SECDR control parity")
    _exact(_get(report, "initial_parity", "train_cross_event_transaction_0", "verified"), True, "SECDR train parity")
    _exact(_get(report, "initial_parity", "fixed_final_64", "verified"), True, "SECDR final parity")
    _exact(_get(report, "decision", "result"), SECDR_RESULT, "SECDR decision")
    _exact(report.get("no_further_intervention_search"), True, "SECDR stopping rule")
    mechanism = _required(report, ("mechanism",), "SECDR mechanism")
    efficacy = _required(report, ("efficacy",), "SECDR efficacy")
    mechanism_supported = all(_get(mechanism, split, "supported") is True for split in ("seen", "new"))
    efficacy_supported = all(_get(efficacy, split, "supported") is True for split in ("seen", "new"))
    endpoint_non_degradation = all(
        _get(efficacy, split, "conditions", "width28_endpoint_non_decreased") is True
        for split in ("seen", "new")
    )
    safety_non_degradation = all(
        _get(efficacy, split, "conditions", "safety_non_regression") is True
        for split in ("seen", "new")
    )
    width28_rows = [
        row for row in report.get("case_level", [])
        if isinstance(row, Mapping) and row.get("role") == "cross_event" and row.get("width") == 28
    ]
    width28_temporal_pass = sum(bool(row.get("temporal_pass_secdr")) for row in width28_rows)
    return {
        "hash": report_hash,
        "mechanism_supported": mechanism_supported,
        "efficacy_supported": efficacy_supported,
        "seen_supported": _get(efficacy, "seen", "supported") is True,
        "new_supported": _get(efficacy, "new", "supported") is True,
        "endpoint_non_degradation": endpoint_non_degradation,
        "safety_non_degradation": safety_non_degradation,
        "width28_temporal_gate_pass": {"passed": width28_temporal_pass, "cases": len(width28_rows)},
        "direction_mechanism_decision": report.get("decision", {}).get("result"),
    }


def _rcsp_rescue_counts(report: Mapping[str, Any]) -> dict[str, Any]:
    groups = _required(report, ("baseline_comparison", "groups"), "RCSP baseline comparison groups")
    if not isinstance(groups, Mapping):
        raise EvidenceIntegrityError("RCSP baseline comparison groups are not an object")

    def delta(name: str) -> int:
        value = _get(groups, name, "delta_rcsp_minus_base", "temporal_gate_pass_cases")
        if not isinstance(value, int):
            raise EvidenceIntegrityError(f"RCSP rescue delta missing for {name}")
        return value

    result = {}
    for role in ("single_recording", "cross_event"):
        for width in (10, 28):
            values = [
                delta(f"seen/{role}/{width}"),
                delta(f"new_position/{role}/{width}"),
            ]
            result[f"{role}/{width}"] = {"rescued": sum(values), "cases": 16}
    if result["cross_event/10"]["rescued"] != 5:
        raise EvidenceIntegrityError("RCSP cross_event/10 rescue count is not frozen at 5/16")
    if result["cross_event/28"]["rescued"] != 0:
        raise EvidenceIntegrityError("RCSP cross_event/28 rescue count is not frozen at 0/16")
    if result["single_recording/10"]["rescued"] + result["single_recording/28"]["rescued"] != 0:
        raise EvidenceIntegrityError("RCSP single-recording rescue is not frozen at 0/32")
    return result


def _evidence_row(
    evidence_id: str,
    stage: str,
    question: str,
    evidence_type: str,
    observation: Any,
    classification: str,
    supports: Any,
    does_not_support: Any,
    causal_strength: str,
    solution_relevance: str,
    path: Path,
    digest: str,
) -> dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "stage": stage,
        "scientific_question": question,
        "evidence_type": evidence_type,
        "cohort": "frozen report-defined cohort",
        "observation": observation,
        "frozen_classification": classification,
        "supports": supports,
        "does_not_support": does_not_support,
        "causal_strength": causal_strength,
        "solution_relevance": solution_relevance,
        "source_report": str(path),
        "source_sha256": digest,
    }


def _contradiction_matrix() -> dict[str, Any]:
    return {
        "C1_normalization_vs_bctr": {
            "contradiction": False,
            "status": "NOT_A_CONTRADICTION",
            "explanation": "observational normalization signature is distinct from the tested BCTR intervention being sufficient",
        },
        "C2_secdr_mechanism_vs_efficacy": {
            "contradiction": False,
            "status": "NOT_A_CONTRADICTION",
            "explanation": "a causally manipulable direction component does not by itself close the efficacy gate",
        },
        "C3_local_blocks_vs_whole_cosine": {
            "contradiction": False,
            "status": "NOT_A_CONTRADICTION",
            "explanation": "local positive blocks can coexist with whole-action cancellation or low efficiency without proving global cancellation",
        },
        "C4_deficit_vs_gate": {
            "contradiction": False,
            "status": "NOT_A_CONTRADICTION",
            "explanation": "continuous deficit improvement is not equivalent to crossing the pre-registered temporal gate",
        },
    }


def _decide_candidates(inputs: Mapping[str, Any]) -> dict[str, Any]:
    integrity_failure = inputs.get("evidence_integrity_verified") is False
    formal = any(bool(inputs.get(name)) for name in (
        "rcsp_solution_supported",
        "bctr_solution_supported",
        "secdr_solution_supported",
    ))
    if integrity_failure:
        result = EVIDENCE_INTEGRITY_FAILURE
        action = INTEGRITY_FAILURE_NEXT_ACTION
    elif formal:
        result = FORMAL_CANDIDATE_SUPPORTED
        action = PILOT_REVIEW_NEXT_ACTION
    else:
        result = NO_SUFFICIENT_CANDIDATE
        action = STOP_NEXT_ACTION
    return {
        "result": result,
        "formal_candidate_supported": formal,
        "final_candidate_decision_performed": True,
        "evidence_integrity_verified": not integrity_failure,
        "next_action": action,
        "pilot_allowed": False,
        "further_intervention_search_allowed": False,
        "architecture_search_allowed": False,
        "metric_search_allowed": False,
    }


def _paper_safe_summary() -> dict[str, str]:
    return {
        "single_recording": (
            "Single-recording failures are associated with weak action-space and parameter-space "
            "alignment with the local temporal descent direction, with condition-dependent localized "
            "anatomy-time mismatches. The evidence rejects gradient starvation and does not support "
            "a global width-conflict explanation, but does not identify a unique causal architectural root cause."
        ),
        "cross_event_width28": (
            "Cross-event long-window degradation exhibits a genuine width-conditioned direction/effectiveness "
            "component. A controlled support-extent-conditioned rotation improves temporal descent alignment "
            "and gain per applied-action norm on both seen and new-position cases. However, the intervention "
            "does not rescue any width-28 temporal gate cases and does not satisfy the pre-registered efficacy "
            "and endpoint-preservation criteria. Direction inefficiency is therefore a real mechanism but is "
            "insufficient as a standalone solution."
        ),
    }


def _build_report(
    reports: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    expected_commit: str,
) -> dict[str, Any]:
    rcsp = reports["rcsp_report"]
    rcsp_review = reports["rcsp_review"]
    parameter = reports["parameter_report"]
    single = reports["single_report"]
    phase21 = reports["phase21_report"]
    bctr = reports["bctr_report"]
    correction = reports["bctr_correction"]
    secdr = reports["secdr_report"]
    defective = reports["defective_secdr_report"]

    rescue = _rcsp_rescue_counts(rcsp)
    secdr_state = _validate_secdr_state_for_synthesis(secdr)
    rcsp_solution = bool(
        _get(rcsp_review, "formal_conclusion", "role_conditioning_alone_sufficient") is True
        and rcsp.get("scientific_acceptance") is True
    )
    bctr_solution = bool(
        _get(bctr, "decision", "overall_supported") is True
        and _get(bctr, "decision", "split_supported") == {"seen": True, "new": True}
        and bctr.get("scientific_acceptance") is True
    )
    secdr_solution = bool(
        secdr_state["mechanism_supported"]
        and secdr_state["efficacy_supported"]
        and secdr_state["seen_supported"]
        and secdr_state["new_supported"]
        and secdr_state["endpoint_non_degradation"]
        and secdr_state["safety_non_degradation"]
        and secdr.get("decision", {}).get("result") == "WIDTH_CONDITIONED_DIRECTION_INTERVENTION_SUPPORTED"
    )
    candidate_inputs = {
        "evidence_integrity_verified": True,
        "rcsp_solution_supported": rcsp_solution,
        "bctr_solution_supported": bctr_solution,
        "secdr_solution_supported": secdr_solution,
        "formal_candidate_requires_seen_and_new": True,
        "formal_candidate_requires_endpoint_non_degradation": True,
        "formal_candidate_requires_safety_non_degradation": True,
        "formal_candidate_requires_own_frozen_intervention_support": True,
        "formal_candidate_supported": any((rcsp_solution, bctr_solution, secdr_solution)),
    }
    decision = _decide_candidates(candidate_inputs)

    evidence_matrix = [
        _evidence_row("E01", "RCSP", "role conditioning usefulness", "controlled diagnostic intervention", rescue, RCSP_CLASSIFICATION, "role conditioning is useful diagnostically", "accepted formal method and width-independent solution", "diagnostic", "partial diagnostic only", paths["rcsp_report"], hashes["rcsp_report"]),
        _evidence_row("E02", "Phase 1A", "single parameter gradients", "read-only parameter attribution", _get(parameter, "scientific_answers"), "GRADIENT_STARVATION_REJECTED", "single gradients are nonzero", "gradient starvation as root cause", "descriptive attribution", "diagnostic context only", paths["parameter_report"], hashes["parameter_report"]),
        _evidence_row("E03", "Phase 1B", "single action decomposition", "read-only action-space decomposition", _get(single, "scientific_answers"), "MULTIPLE_MECHANISMS_SUPPORTED", "localized and condition-dependent direction evidence", "unique architectural root cause", "descriptive decomposition", "diagnostic context only", paths["single_report"], hashes["single_report"]),
        _evidence_row("E04", "Phase 2", "cross-width effect", "observational width comparison", _get(reports["phase2_report"], "scientific_answers"), "WIDTH_EFFECT_OBSERVED", "cross-event width-conditioned evidence", "global causal normalization proof", "observational", "mechanism evidence", paths["phase2_report"], hashes["phase2_report"]),
        _evidence_row("E05", "Phase 2.1", "mixed width mechanism", "frozen adjudication", _get(phase21, "adjudication"), PHASE21_CLASSIFICATION, "normalization signal and direction component are distinct evidence", "normalization proved causal or all normalization effects disproved", "observational adjudication", "intervention ordering context", paths["phase21_report"], hashes["phase21_report"]),
        _evidence_row("E06", "BCTR", "metric/support-time intervention", "controlled negative intervention", _get(bctr, "decision"), BCTR_RESULT, "the tested BCTR intervention is insufficient", "all possible normalization mechanisms are false", "controlled intervention", "rejected as sufficient solution", paths["bctr_report"], hashes["bctr_report"]),
        _evidence_row("E07", "corrected SECDR", "direction mechanism", "controlled direction intervention", _get(secdr, "mechanism"), "DIRECTION_MECHANISM_SUPPORTED", "width-conditioned direction/effectiveness is causally manipulable", "direction alone is sufficient", "controlled intervention", "mechanism supported", paths["secdr_report"], hashes["secdr_report"]),
        _evidence_row("E08", "corrected SECDR", "solution efficacy", "controlled final adjudication", _get(secdr, "efficacy"), SECDR_RESULT, "no sufficient method candidate", "temporal gate rescue and endpoint-preserving solution", "controlled intervention", "insufficient solution efficacy", paths["secdr_report"], hashes["secdr_report"]),
    ]

    mechanism_synthesis = {
        "M1_optimization_starvation": {"status": "REJECTED", "evidence": ["E02"], "reason": "single parameter gradients are nonzero and effective movement evidence is present"},
        "M2_hard_support_projection_bottleneck": {"status": "NOT_SUPPORTED_AS_PRIMARY_EXPLANATION", "evidence": ["E01", "E03", "E04"], "reason": "support projection exists, but zero support effect is not established"},
        "M3_single_direction_alignment_bottleneck": {"status": "SUPPORTED_AS_DIAGNOSTIC_BOTTLENECK", "evidence": ["E02", "E03"], "causal_architecture_root_cause_proven": False},
        "M4_global_single_width_gradient_conflict": {"status": "NOT_SUPPORTED", "evidence": ["E02"], "localized_new_position_conflict": True},
        "M5_cross_event_width_effect": {"status": "CONFIRMED", "evidence": ["E01", "E04", "E05"]},
        "M6_normalized_temporal_spreading": {"status": "NOT_SUPPORTED", "evidence": ["E05"]},
        "M7_normalization_temporal_evaluation": {"observational_normalization_signal": "OBSERVED", "tested_BCTR_normalization_intervention": "NOT_SUPPORTED_AS_SUFFICIENT_SOLUTION", "evidence": ["E05", "E06"]},
        "M8_width_conditioned_direction_effectiveness": {"status": "SUPPORTED_AS_CAUSALLY_MANIPULABLE_COMPONENT", "evidence": ["E05", "E07"], "sufficient_root_cause": False},
    }

    intervention_evidence = {
        "RCSP": {
            "candidate": "RCSP",
            "changed_variable": "role-conditioned support-projected adapter",
            "mechanism_supported": _get(rcsp_review, "formal_conclusion", "classification") == RCSP_CLASSIFICATION,
            "efficacy_supported": rcsp_solution,
            "seen_supported": False,
            "new_supported": False,
            "temporal_gate_rescue": {"cross_event/10": "5/16", "cross_event/28": "0/16", "single_recording": "0/32"},
            "endpoint_non_degradation": True,
            "safety_non_degradation": True,
            "production_modified": False,
            "pilot_allowed": False,
            "solution_status": "PARTIAL_DIAGNOSTIC_ONLY",
        },
        "BCTR": {
            "candidate": "BCTR",
            "changed_variable": "temporal metric derivative-stencil support",
            "mechanism_supported": False,
            "efficacy_supported": bctr_solution,
            "seen_supported": False,
            "new_supported": False,
            "temporal_gate_rescue": "not sufficient according to frozen decision",
            "endpoint_non_degradation": False,
            "safety_non_degradation": True,
            "production_modified": False,
            "pilot_allowed": False,
            "solution_status": "REJECTED_AS_SUFFICIENT_INTERVENTION",
        },
        "corrected_SECDR": {
            "candidate": "corrected SECDR",
            "changed_variable": "support-extent-conditioned direction rotation",
            "mechanism_supported": secdr_state["mechanism_supported"],
            "efficacy_supported": secdr_solution,
            "seen_supported": secdr_state["seen_supported"],
            "new_supported": secdr_state["new_supported"],
            "temporal_gate_rescue": secdr_state["width28_temporal_gate_pass"],
            "endpoint_non_degradation": secdr_state["endpoint_non_degradation"],
            "safety_non_degradation": secdr_state["safety_non_degradation"],
            "production_modified": False,
            "pilot_allowed": False,
            "solution_status": "MECHANISM_SUPPORTED_BUT_SOLUTION_INSUFFICIENT",
        },
    }

    return {
        "schema": SCHEMA,
        "completed": True,
        "provenance": {
            "runtime_commit": expected_commit,
            "implementation_parent_commit": IMPLEMENTATION_PARENT_COMMIT,
            "execution_mode": "read-only JSON/report synthesis",
            "no_latest_artifact_search": True,
        },
        "input_artifacts": {name: str(paths[name]) for name in INPUT_NAMES},
        "input_sha256": dict(hashes),
        "lineage_verification": {
            "verified": True,
            "no_latest_artifact_search": True,
            "rcsp_review_to_report_hash": True,
            "parameter_to_rcsp_hash": True,
            "single_to_parameter_hash": True,
            "phase2_to_upstream_paths_and_hashes": True,
            "phase21_to_phase2_path_and_hash": True,
            "bctr_to_phase21_path_and_hash": True,
            "bctr_correction_to_bctr_path_and_hash": True,
            "corrected_secdr_to_all_required_lineage": True,
            "defective_secdr_excluded_from_scientific_evidence": True,
            "historical_artifact": defective.get("historical_artifact") is True,
            "implementation_defective": defective.get("implementation_defective") is True,
            "scientifically_reused": False,
        },
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
        "evidence_matrix": evidence_matrix,
        "mechanism_synthesis": mechanism_synthesis,
        "intervention_evidence": intervention_evidence,
        "contradiction_matrix": _contradiction_matrix(),
        "single_recording_synthesis": {
            "status": "DIAGNOSTIC_BOTTLENECK_WITH_UNIQUE_ROOT_CAUSE_UNPROVEN",
            "gradient_starvation": "REJECTED",
            "direction_alignment": "SUPPORTED_AS_DIAGNOSTIC_BOTTLENECK",
            "global_width_conflict": "NOT_SUPPORTED",
            "localized_generalization_conflict": True,
            "unique_causal_architectural_root_cause_proven": False,
            "whole_single_direction": _get(single, "single_summary", "whole"),
            "anatomy": single.get("single_anatomy"),
            "temporal": single.get("single_temporal"),
            "anatomy_time": single.get("single_anatomy_time"),
            "source_conditioned_shifts": single.get("source_conditioned_comparison"),
            "width_conditioned_shifts": single.get("width_conditioned_comparison"),
            "new_position_single_28_localized_ascent": single.get("new_position_single_28"),
            "parameter_to_action_bridge": single.get("parameter_to_action_bridge"),
            "parameter_gradient_answers": parameter.get("scientific_answers"),
            "parameter_gradient_geometry": parameter.get("within_role_parameter_gradient_geometry"),
            "parameter_gradient_rows": parameter.get("parameter_gradient_rows"),
        },
        "cross_event_width_synthesis": {
            "width_effect": "CONFIRMED",
            "observational_normalization_signal": "OBSERVED",
            "normalized_temporal_spreading": "NOT_SUPPORTED",
            "direction_component": "SUPPORTED_AS_CAUSALLY_MANIPULABLE_COMPONENT",
            "direction_only_solution_sufficient": False,
            "phase2_width_contrasts": _get(reports["phase2_report"], "width_contrasts"),
            "phase21_adjudication": phase21.get("adjudication"),
            "bctr_observation": bctr.get("decision"),
            "bctr_correction_observation": correction.get("decision"),
            "secdr_mechanism": secdr.get("mechanism"),
            "secdr_efficacy": secdr.get("efficacy"),
            "width28_temporal_gate_pass": secdr_state["width28_temporal_gate_pass"],
        },
        "scientific_answers": {
            "optimization_starvation": "REJECTED",
            "hard_support_primary_bottleneck": "NOT_SUPPORTED_AS_PRIMARY_EXPLANATION",
            "single_direction_alignment_bottleneck": "SUPPORTED_AS_DIAGNOSTIC_BOTTLENECK",
            "global_single_width_conflict": "NOT_SUPPORTED",
            "localized_single_generalization_conflict": True,
            "cross_event_width_effect": "CONFIRMED",
            "normalized_temporal_spreading": "NOT_SUPPORTED",
            "observational_normalization_signal": "OBSERVED",
            "bctr_normalization_solution": "NOT_SUPPORTED_AS_SUFFICIENT_SOLUTION",
            "width_conditioned_direction_component": "SUPPORTED_AS_CAUSALLY_MANIPULABLE_COMPONENT",
            "direction_component_causally_manipulable": True,
            "direction_only_solution_sufficient": False,
            "unique_causal_root_cause_proven": False,
            "supported_formal_method_candidate_exists": decision["formal_candidate_supported"],
            "further_intervention_search_allowed": False,
            "final_refiner_candidate_classification": decision["result"],
        },
        "candidate_decision_inputs": candidate_inputs,
        "final_decision": decision,
        "next_action": {
            "result": decision["next_action"],
            "pilot_allowed": False,
            "further_intervention_search_allowed": False,
        },
        "no_further_intervention_search": True,
        "paper_safe_summary": _paper_safe_summary(),
        "negative_result_value": {
            "bctr": "BCTR excludes the tested temporal evaluation/reduction intervention as a sufficient explanation/solution.",
            "secdr": "Corrected SECDR shows that direction is causally manipulable but insufficient alone.",
            "stopping": "Therefore further unconstrained search is not scientifically justified under the current preregistered workflow.",
        },
        "remaining_uncertainty": [
            "The evidence does not identify a unique causal architectural root cause.",
            "Observational normalization evidence is not a causal proof of denominator-only normalization.",
            "The direction component does not close the temporal-gate and endpoint-preservation criteria alone.",
        ],
        "claim_boundary": "Mechanisms are frozen diagnostic findings; no formal production or Pilot Refiner method candidate is supported.",
        "production_model_modified": False,
        "production_inference_modified": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
    }


def _validate_secdr_state_for_synthesis(report: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = report.get("mechanism", {})
    efficacy = report.get("efficacy", {})
    mechanism_supported = all(_get(mechanism, split, "supported") is True for split in ("seen", "new"))
    efficacy_supported = all(_get(efficacy, split, "supported") is True for split in ("seen", "new"))
    endpoint = all(_get(efficacy, split, "conditions", "width28_endpoint_non_decreased") is True for split in ("seen", "new"))
    safety = all(_get(efficacy, split, "conditions", "safety_non_regression") is True for split in ("seen", "new"))
    rows = [
        row for row in report.get("case_level", [])
        if isinstance(row, Mapping) and row.get("role") == "cross_event" and row.get("width") == 28
    ]
    return {
        "mechanism_supported": mechanism_supported,
        "efficacy_supported": efficacy_supported,
        "seen_supported": _get(efficacy, "seen", "supported") is True,
        "new_supported": _get(efficacy, "new", "supported") is True,
        "endpoint_non_degradation": endpoint,
        "safety_non_degradation": safety,
        "width28_temporal_gate_pass": {
            "passed": sum(bool(row.get("temporal_pass_secdr")) for row in rows),
            "cases": len(rows),
        },
    }


def synthesize(args: argparse.Namespace) -> dict[str, Any]:
    raw_paths = {
        "rcsp_report": args.rcsp_report,
        "rcsp_review": args.rcsp_review,
        "parameter_report": args.parameter_report,
        "single_report": args.single_report,
        "phase2_report": args.phase2_report,
        "phase21_report": args.phase21_report,
        "bctr_report": args.bctr_report,
        "bctr_correction": args.bctr_correction,
        "secdr_report": args.secdr_report,
        "defective_secdr_report": args.defective_secdr_report,
    }
    paths = {name: Path(value).resolve() for name, value in raw_paths.items()}
    if paths["rcsp_report"].parent.name != "result":
        raise EvidenceIntegrityError("RCSP report must be under an explicit result directory")
    rcsp_directory = paths["rcsp_report"].parent
    output = Path(args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Joint synthesis output must be a fresh empty directory")
    if any(output == path.parent or output.is_relative_to(path.parent) for path in paths.values()):
        raise FileExistsError("Joint synthesis output overlaps an immutable input directory")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir = output / "result"
    result_dir.mkdir(exist_ok=False)
    failure_path = result_dir / "failure.json"
    reports: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    try:
        for name in INPUT_NAMES:
            reports[name], hashes[name] = _load_report(paths[name], name)
        rcsp_info = _validate_rcsp(reports["rcsp_report"], reports["rcsp_review"], hashes["rcsp_report"])
        parameter_info = _validate_parameter(reports["parameter_report"], hashes["parameter_report"], hashes["rcsp_report"])
        single_info = _validate_single(reports["single_report"], hashes["single_report"], paths["parameter_report"], hashes["parameter_report"])
        phase2_info = _validate_phase2(reports["phase2_report"], hashes["phase2_report"], rcsp_directory, paths["parameter_report"], hashes["parameter_report"], paths["single_report"], hashes["single_report"])
        phase21_info = _validate_phase21(reports["phase21_report"], hashes["phase21_report"], paths["phase2_report"], hashes["phase2_report"])
        bctr_info = _validate_bctr(reports["bctr_report"], hashes["bctr_report"], paths["phase21_report"], hashes["phase21_report"])
        correction_info = _validate_bctr_correction(reports["bctr_correction"], hashes["bctr_correction"], paths["bctr_report"], hashes["bctr_report"])
        defective_info = _validate_defective_secdr(reports["defective_secdr_report"], hashes["defective_secdr_report"])
        secdr_info = _validate_secdr(reports["secdr_report"], hashes["secdr_report"], paths["phase21_report"], hashes["phase21_report"], paths["bctr_report"], hashes["bctr_report"], paths["bctr_correction"], hashes["bctr_correction"], paths["defective_secdr_report"], hashes["defective_secdr_report"], rcsp_directory, paths["parameter_report"], paths["single_report"])
        for name, before in hashes.items():
            if _sha256(paths[name]) != before:
                raise EvidenceIntegrityError(f"input report changed during synthesis: {name}")
        report = _build_report(reports, paths, hashes, args.expected_commit)
        report["input_unchanged_during_synthesis"] = True
        report["validated_components"] = {
            "rcsp": rcsp_info,
            "parameter_attribution": parameter_info,
            "single_decomposition": single_info,
            "phase2": phase2_info,
            "phase21": phase21_info,
            "bctr": bctr_info,
            "bctr_correction": correction_info,
            "defective_secdr": defective_info,
            "corrected_secdr": secdr_info,
        }
        _exclusive_json(result_dir / "report.json", report)
        print(json.dumps({"stage": "refiner_joint_evidence_synthesis_complete", "report": str(result_dir / "report.json"), "decision": report["final_decision"]["result"], "pilot_allowed": False}, ensure_ascii=False, allow_nan=False), flush=True)
        return report
    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "schema": SCHEMA,
                    "completed": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "final_candidate_decision_performed": False,
                    "read_only": True,
                    "optimizer_constructed": False,
                    "optimizer_steps": 0,
                    "model_loaded": False,
                    "forward_pass_performed": False,
                    "autograd_used": False,
                    "parameter_update_performed": False,
                    "production_model_modified": False,
                    "production_inference_modified": False,
                    "scientific_acceptance": False,
                    "publish_allowed": False,
                    "pilot_allowed": False,
                    "final_decision": {
                        "result": EVIDENCE_INTEGRITY_FAILURE,
                        "formal_candidate_supported": False,
                        "final_candidate_decision_performed": False,
                        "next_action": INTEGRITY_FAILURE_NEXT_ACTION,
                        "pilot_allowed": False,
                        "further_intervention_search_allowed": False,
                    },
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcsp-report", required=True)
    parser.add_argument("--rcsp-review", required=True)
    parser.add_argument("--parameter-report", required=True)
    parser.add_argument("--single-report", required=True)
    parser.add_argument("--phase2-report", required=True)
    parser.add_argument("--phase21-report", required=True)
    parser.add_argument("--bctr-report", required=True)
    parser.add_argument("--bctr-correction", required=True)
    parser.add_argument("--secdr-report", required=True)
    parser.add_argument("--defective-secdr-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    synthesize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
