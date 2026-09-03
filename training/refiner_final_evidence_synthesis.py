"""Final read-only synthesis of the frozen Refiner evidence ledger.

This stage is the terminal scientific evidence-composition step for the current
Refiner candidate-development protocol.  It consumes only explicit frozen JSON
reports/manifests.  It performs no model loading, tensor execution, checkpoint
loading, forward pass, autograd, optimizer step, metric recomputation, case
selection, intervention search, or architecture search.

Inputs:
  1. pre-RPA joint evidence synthesis v1
  2. frozen RPA-LRTA v2 formal report
  3. RPA-LRTA v2 freeze manifest
  4. frozen RPA direction-reporting correction
  5. direction-correction freeze manifest

Outputs:
  result/report.json
  result/evidence_summary.md
  result/freeze_manifest.json

The final method-level conclusion is deliberately conservative:
multiple diagnostic mechanisms are real and some are causally manipulable, but
no tested candidate satisfies the frozen efficacy + endpoint + safety criteria.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "refiner_final_evidence_synthesis_v1"
IMPLEMENTATION_PARENT_COMMIT = "5a06a03b97e4e2d46344a88eb6a3bcc48d41b7d2"

JOINT_SCHEMA = "refiner_joint_evidence_synthesis_v1"
RPA_SCHEMA = "refiner_role_phase_anatomy_low_rank_tangent_adaptation_experiment_v2"
RPA_FREEZE_SCHEMA = "refiner_rpa_lrta_v2_result_freeze_v1"
DIRECTION_SCHEMA = "refiner_rpa_lrta_direction_reporting_correction_v1"
DIRECTION_FREEZE_SCHEMA = "refiner_rpa_lrta_direction_correction_freeze_v1"

PRE_RPA_DECISION = "MECHANISMS_IDENTIFIED_BUT_NO_SUFFICIENT_METHOD_CANDIDATE"
RPA_DECISION = "RPA_LRTA_NOT_SUPPORTED"
RPA_NEXT_ACTION = "reject_rpa_lrta_candidate_without_additional_architecture_search"
DIRECTION_CLASSIFICATION = (
    "RPA_DIRECTION_MECHANISM_PRESENT_BUT_METHOD_REMAINS_UNSUPPORTED"
)

FINAL_CLASSIFICATION = (
    "MULTIPLE_MANIPULABLE_MECHANISMS_WITHOUT_SUFFICIENT_SAFE_REFINER_CANDIDATE"
)
FINAL_NEXT_ACTION = (
    "freeze_final_refiner_evidence_and_transition_to_manuscript_synthesis"
)

EXPECTED_RPA_REPORT_SHA256 = (
    "08fd36d5bd504a16cb5f18348358e8e236008e0758481e7c5372dddca0c6808e"
)
EXPECTED_RPA_UPDATES_SHA256 = (
    "aedcf96068976ead5988d055af248b067e641849214365ba9fea3fdee35f0a86"
)
EXPECTED_RPA_ADAPTER_SHA256 = (
    "2b6a7ae7d08721bcff5b174403a7137ec7494c2f871a21ffc5a63bdc7be70110"
)
EXPECTED_DIRECTION_REPORT_SHA256 = (
    "e71397bb72afbf8f7e27b7ff141e2a04a49bcf7fcceeda980beb0b07d19afba6"
)

INPUT_NAMES = (
    "joint_report",
    "rpa_report",
    "rpa_freeze",
    "direction_report",
    "direction_freeze",
)


class EvidenceIntegrityError(ValueError):
    """Raised when frozen evidence does not satisfy the final synthesis contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceIntegrityError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceIntegrityError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceIntegrityError(f"{label} must be a JSON object")
    return value


def _get(value: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceIntegrityError(
            f"{label}: expected {expected!r}, got {actual!r}"
        )


def _false(report: Mapping[str, Any], field: str, label: str) -> None:
    _exact(report.get(field), False, f"{label}.{field}")


def _true(report: Mapping[str, Any], field: str, label: str) -> None:
    _exact(report.get(field), True, f"{label}.{field}")


def _runtime_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise EvidenceIntegrityError("git HEAD is not a full SHA")
    return value


def _exclusive_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _validate_joint(report: Mapping[str, Any]) -> dict[str, Any]:
    _exact(report.get("schema"), JOINT_SCHEMA, "joint schema")
    _true(report, "completed", "joint")
    _true(report, "read_only", "joint")
    _false(report, "optimizer_constructed", "joint")
    _exact(report.get("optimizer_steps"), 0, "joint.optimizer_steps")
    _false(report, "model_loaded", "joint")
    _false(report, "forward_pass_performed", "joint")
    _false(report, "autograd_used", "joint")
    _false(report, "parameter_update_performed", "joint")
    _false(report, "checkpoint_selection_performed", "joint")
    _false(report, "case_selection_performed", "joint")
    _false(report, "metric_selection_performed", "joint")
    _false(report, "architecture_selection_performed", "joint")
    _false(report, "intervention_search_performed", "joint")

    _exact(
        _get(report, "final_decision", "result"),
        PRE_RPA_DECISION,
        "joint final decision",
    )
    _exact(
        _get(report, "final_decision", "formal_candidate_supported"),
        False,
        "joint formal candidate",
    )
    _exact(
        report.get("no_further_intervention_search"),
        True,
        "joint stop rule",
    )
    _exact(
        _get(report, "scientific_answers", "direction_component_causally_manipulable"),
        True,
        "joint direction manipulability",
    )
    _exact(
        _get(report, "scientific_answers", "direction_only_solution_sufficient"),
        False,
        "joint direction sufficiency",
    )
    _exact(
        _get(report, "scientific_answers", "unique_causal_root_cause_proven"),
        False,
        "joint root-cause boundary",
    )
    _false(report, "scientific_acceptance", "joint")
    _false(report, "publish_allowed", "joint")
    _false(report, "pilot_allowed", "joint")
    return {
        "decision": PRE_RPA_DECISION,
        "direction_causally_manipulable": True,
        "direction_only_sufficient": False,
        "unique_root_cause_proven": False,
    }


def _validate_rpa(report: Mapping[str, Any]) -> dict[str, Any]:
    _exact(report.get("schema"), RPA_SCHEMA, "RPA schema")
    _true(report, "completed", "RPA")
    _exact(report.get("fixed_final_case_count"), 64, "RPA final64")
    _exact(_get(report, "decision", "result"), RPA_DECISION, "RPA decision")
    _exact(
        _get(report, "decision", "next_action"),
        RPA_NEXT_ACTION,
        "RPA next action",
    )
    _false(report, "scientific_acceptance", "RPA")
    _false(report, "publish_allowed", "RPA")
    _false(report, "pilot_allowed", "RPA")
    _false(report, "production_model_modified", "RPA")
    _false(report, "production_inference_modified", "RPA")

    training = report.get("training")
    if not isinstance(training, Mapping):
        raise EvidenceIntegrityError("RPA training block missing")
    _exact(training.get("attempt_budget"), 400, "RPA attempt budget")
    _exact(training.get("attempted_steps"), 110, "RPA attempted steps")
    _exact(training.get("accepted_steps"), 108, "RPA accepted steps")
    _exact(training.get("rollback_steps"), 2, "RPA rollback steps")
    _exact(
        training.get("termination_reason"),
        "DETERMINISTIC_NO_DESCENT_FIXED_POINT",
        "RPA termination",
    )
    _exact(training.get("last_accepted_step"), 108, "RPA last accepted step")

    conditions = _get(report, "decision", "conditions")
    if not isinstance(conditions, Mapping):
        raise EvidenceIntegrityError("RPA A-I conditions missing")
    for name in (
        "A_single_seen_rescue",
        "B_single_new_rescue",
        "F_no_endpoint_regression",
        "G_no_safety_regression",
        "H_single_direction_improved",
        "I_cross28_direction_improved",
    ):
        _exact(conditions.get(name), False, f"RPA original {name}")
    _exact(
        _get(report, "decision", "total_temporal_newly_rescued_vs_RCSP"),
        1,
        "RPA total temporal rescues",
    )

    summaries = report.get("summaries")
    if not isinstance(summaries, Mapping):
        raise EvidenceIntegrityError("RPA summaries missing")
    overall = summaries.get("overall")
    single = summaries.get("single_recording")
    cross28 = summaries.get("cross_event/28")
    if not all(isinstance(value, Mapping) for value in (overall, single, cross28)):
        raise EvidenceIntegrityError("RPA key summaries missing")

    _exact(overall.get("RCSP_temporal_pass"), 5, "RPA overall RCSP temporal pass")
    _exact(overall.get("RPA_temporal_pass"), 6, "RPA overall RPA temporal pass")
    _exact(single.get("RCSP_temporal_pass"), 0, "RPA single RCSP temporal pass")
    _exact(single.get("RPA_temporal_pass"), 0, "RPA single RPA temporal pass")
    _exact(cross28.get("RCSP_temporal_pass"), 0, "RPA cross28 RCSP temporal pass")
    _exact(cross28.get("RPA_temporal_pass"), 0, "RPA cross28 RPA temporal pass")

    rows = report.get("case_level")
    if not isinstance(rows, list) or len(rows) != 64:
        raise EvidenceIntegrityError("RPA case_level is not final64")
    endpoint_regressions = sum(
        bool(row.get("endpoint_regression_vs_rcsp"))
        for row in rows
        if isinstance(row, Mapping)
    )
    physical_regressions = sum(
        bool(row.get("physical_regression_vs_rcsp"))
        for row in rows
        if isinstance(row, Mapping)
    )
    temporal_regressions = sum(
        bool(row.get("temporal_regression_vs_rcsp"))
        for row in rows
        if isinstance(row, Mapping)
    )
    _exact(endpoint_regressions, 3, "RPA endpoint regressions")
    _exact(physical_regressions, 1, "RPA physical regressions")
    _exact(temporal_regressions, 0, "RPA temporal regressions")

    return {
        "decision": RPA_DECISION,
        "attempt_budget": 400,
        "attempted_steps": 110,
        "accepted_steps": 108,
        "rollback_steps": 2,
        "temporal_pass_RCSP": 5,
        "temporal_pass_RPA": 6,
        "temporal_newly_rescued": 1,
        "single_temporal_gate": "0/32 -> 0/32",
        "cross28_temporal_gate": "0/16 -> 0/16",
        "endpoint_regressions": 3,
        "physical_regressions": 1,
        "temporal_regressions": 0,
        "overall_summary": dict(overall),
        "single_summary": dict(single),
        "cross28_summary": dict(cross28),
    }


def _validate_rpa_freeze(
    freeze: Mapping[str, Any],
    rpa_report_path: Path,
    rpa_report_hash: str,
) -> dict[str, Any]:
    _exact(freeze.get("schema"), RPA_FREEZE_SCHEMA, "RPA freeze schema")
    _true(freeze, "completed", "RPA freeze")
    _true(freeze, "read_only", "RPA freeze")
    _exact(freeze.get("source_decision"), RPA_DECISION, "RPA freeze decision")
    _exact(
        freeze.get("source_next_action"),
        RPA_NEXT_ACTION,
        "RPA freeze next action",
    )
    _exact(freeze.get("attempt_budget"), 400, "RPA freeze attempt budget")
    _exact(freeze.get("attempted_steps"), 110, "RPA freeze attempted steps")
    _exact(freeze.get("accepted_steps"), 108, "RPA freeze accepted steps")
    _exact(freeze.get("rollback_steps"), 2, "RPA freeze rollback steps")

    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvidenceIntegrityError("RPA freeze artifacts missing")
    report_artifact = artifacts.get("report")
    updates_artifact = artifacts.get("updates")
    adapter_artifact = artifacts.get("adapter")
    if not all(
        isinstance(value, Mapping)
        for value in (report_artifact, updates_artifact, adapter_artifact)
    ):
        raise EvidenceIntegrityError("RPA freeze artifact entries missing")

    _exact(
        str(Path(str(report_artifact.get("path"))).resolve()),
        str(rpa_report_path.resolve()),
        "RPA freeze report path",
    )
    _exact(
        report_artifact.get("sha256"),
        rpa_report_hash,
        "RPA freeze report hash",
    )
    _exact(
        report_artifact.get("sha256"),
        EXPECTED_RPA_REPORT_SHA256,
        "RPA formal report frozen hash",
    )
    _exact(
        updates_artifact.get("sha256"),
        EXPECTED_RPA_UPDATES_SHA256,
        "RPA updates frozen hash",
    )
    _exact(
        adapter_artifact.get("sha256"),
        EXPECTED_RPA_ADAPTER_SHA256,
        "RPA adapter frozen hash",
    )
    _false(freeze, "scientific_acceptance", "RPA freeze")
    _false(freeze, "publish_allowed", "RPA freeze")
    _false(freeze, "pilot_allowed", "RPA freeze")
    return {
        "report_sha256": rpa_report_hash,
        "updates_sha256": EXPECTED_RPA_UPDATES_SHA256,
        "adapter_sha256": EXPECTED_RPA_ADAPTER_SHA256,
    }


def _validate_direction(report: Mapping[str, Any]) -> dict[str, Any]:
    _exact(report.get("schema"), DIRECTION_SCHEMA, "direction schema")
    _true(report, "completed", "direction correction")
    _true(report, "read_only", "direction correction")
    _false(report, "optimizer_constructed", "direction correction")
    _exact(report.get("optimizer_steps"), 0, "direction optimizer steps")
    _false(report, "parameter_update_performed", "direction correction")
    _false(report, "training_performed", "direction correction")
    _false(report, "checkpoint_selection_performed", "direction correction")
    _false(report, "case_selection_performed", "direction correction")
    _false(report, "metric_selection_performed", "direction correction")
    _false(report, "architecture_selection_performed", "direction correction")
    _exact(report.get("fixed_final_case_count"), 64, "direction final64")
    _exact(
        _get(report, "gradient_correction", "primary_space"),
        "raw_all_geometry",
        "direction primary space",
    )
    _exact(
        _get(
            report,
            "gradient_correction",
            "comparison_gradient_shared_by_RCSP_and_RPA",
        ),
        True,
        "direction shared reference gradient",
    )
    _exact(
        _get(report, "gradient_correction", "model_parameter_autograd_used"),
        False,
        "direction model parameter autograd",
    )
    _exact(
        _get(report, "corrected_direction_conditions", "H_single_direction_improved"),
        True,
        "corrected H",
    )
    _exact(
        _get(report, "corrected_direction_conditions", "I_cross28_direction_improved"),
        False,
        "corrected I",
    )
    _exact(
        _get(report, "final_interpretation", "classification"),
        DIRECTION_CLASSIFICATION,
        "direction classification",
    )
    _exact(
        report.get("formal_candidate_decision"),
        RPA_DECISION,
        "direction preserved formal decision",
    )
    _exact(
        _get(report, "decision_invariance_check", "decision_invariant"),
        True,
        "direction decision invariance",
    )
    _exact(
        _get(report, "decision_invariance_check", "result"),
        RPA_DECISION,
        "direction invariant decision result",
    )
    _false(report, "scientific_acceptance", "direction correction")
    _false(report, "publish_allowed", "direction correction")
    _false(report, "pilot_allowed", "direction correction")
    _false(report, "production_model_modified", "direction correction")
    _false(report, "production_inference_modified", "direction correction")

    state = report.get("state_integrity")
    if not isinstance(state, Mapping):
        raise EvidenceIntegrityError("direction state integrity missing")
    for field in (
        "base_unchanged",
        "rcsp_unchanged",
        "rpa_unchanged",
        "frozen_inputs_unchanged",
        "model_parameter_gradients_none",
    ):
        _exact(state.get(field), True, f"direction state {field}")

    return {
        "corrected_H_single_direction_improved": True,
        "corrected_I_cross28_direction_improved": False,
        "classification": DIRECTION_CLASSIFICATION,
        "formal_candidate_decision": RPA_DECISION,
    }


def _validate_direction_freeze(
    freeze: Mapping[str, Any],
    direction_path: Path,
    direction_hash: str,
) -> dict[str, Any]:
    _exact(
        freeze.get("schema"),
        DIRECTION_FREEZE_SCHEMA,
        "direction freeze schema",
    )
    _true(freeze, "completed", "direction freeze")
    _true(freeze, "read_only", "direction freeze")
    _exact(
        str(Path(str(freeze.get("source_report"))).resolve()),
        str(direction_path.resolve()),
        "direction freeze source path",
    )
    _exact(
        freeze.get("source_report_sha256"),
        direction_hash,
        "direction freeze source hash",
    )
    _exact(
        direction_hash,
        EXPECTED_DIRECTION_REPORT_SHA256,
        "direction report frozen hash",
    )
    _exact(
        freeze.get("corrected_H_single_direction_improved"),
        True,
        "direction freeze H",
    )
    _exact(
        freeze.get("corrected_I_cross28_direction_improved"),
        False,
        "direction freeze I",
    )
    _exact(
        freeze.get("direction_classification"),
        DIRECTION_CLASSIFICATION,
        "direction freeze classification",
    )
    _exact(
        freeze.get("formal_candidate_decision"),
        RPA_DECISION,
        "direction freeze formal decision",
    )
    _false(freeze, "new_training_performed", "direction freeze")
    _false(freeze, "new_metric_selection_performed", "direction freeze")
    _false(freeze, "new_case_selection_performed", "direction freeze")
    _false(freeze, "new_architecture_search_performed", "direction freeze")
    _false(freeze, "pilot_allowed", "direction freeze")
    return {
        "report_sha256": direction_hash,
        "classification": DIRECTION_CLASSIFICATION,
    }


def _build_evidence_ledger(
    joint: Mapping[str, Any],
    rpa_info: Mapping[str, Any],
    direction_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "F01",
            "stage": "pre-RPA joint synthesis",
            "mechanism": "optimization / gradient starvation",
            "status": "REJECTED_AS_PRIMARY_CAUSE",
            "evidence_type": "frozen multi-stage evidence synthesis",
            "supports": "single parameter gradients are nonzero; useful action movement exists",
            "does_not_support": "gradient starvation as the primary Refiner failure explanation",
            "method_level_sufficiency": False,
        },
        {
            "evidence_id": "F02",
            "stage": "RCSP",
            "mechanism": "role conditioning",
            "status": "REAL_MECHANISM_BUT_INSUFFICIENT",
            "evidence_type": "controlled adapter intervention",
            "supports": "role conditioning can rescue a subset of temporal failures",
            "does_not_support": "role conditioning alone as a sufficient safe Refiner method",
            "method_level_sufficiency": False,
        },
        {
            "evidence_id": "F03",
            "stage": "Phase2 / Phase2.1",
            "mechanism": "width-conditioned effectiveness",
            "status": "CONDITIONING_EFFECT_CONFIRMED",
            "evidence_type": "frozen cross-width adjudication",
            "supports": "width conditions intervention effectiveness and direction behavior",
            "does_not_support": "width as a unique independent root cause",
            "method_level_sufficiency": False,
        },
        {
            "evidence_id": "F04",
            "stage": "BCTR",
            "mechanism": "metric/support-time explanation",
            "status": "REJECTED_AS_SUFFICIENT_SOLUTION",
            "evidence_type": "controlled negative intervention",
            "supports": "the tested metric/support-time intervention is insufficient",
            "does_not_support": "continued metric search under the frozen protocol",
            "method_level_sufficiency": False,
        },
        {
            "evidence_id": "F05",
            "stage": "SECDR",
            "mechanism": "width-conditioned direction/effectiveness",
            "status": "CAUSALLY_MANIPULABLE_BUT_INSUFFICIENT",
            "evidence_type": "controlled direction intervention",
            "supports": "direction can be causally manipulated",
            "does_not_support": "direction alone as a sufficient gate-rescuing solution",
            "method_level_sufficiency": False,
        },
        {
            "evidence_id": "F06",
            "stage": "RPA-LRTA v2",
            "mechanism": "role-phase-anatomy low-rank conditioned action",
            "status": "MECHANISTIC_EFFECT_PRESENT_METHOD_NOT_SUPPORTED",
            "evidence_type": "frozen trained intervention",
            "observation": {
                "temporal_pass": "5/64 -> 6/64",
                "single_temporal_gate": rpa_info["single_temporal_gate"],
                "cross28_temporal_gate": rpa_info["cross28_temporal_gate"],
                "endpoint_regressions": rpa_info["endpoint_regressions"],
                "physical_regressions": rpa_info["physical_regressions"],
            },
            "supports": "continuous deficit modulation and one additional temporal rescue",
            "does_not_support": "a sufficiently effective and safe RPA-LRTA method candidate",
            "method_level_sufficiency": False,
        },
        {
            "evidence_id": "F07",
            "stage": "RPA direction correction",
            "mechanism": "single-recording action-direction alignment",
            "status": "CAUSALLY_SUPPORTED_BUT_INSUFFICIENT",
            "evidence_type": "read-only corrected action-space gradient audit",
            "observation": {
                "H_single_direction_improved": True,
                "I_cross28_direction_improved": False,
                "classification": direction_info["classification"],
            },
            "supports": (
                "single-recording direction/alignment is a genuine manipulable mechanism"
            ),
            "does_not_support": (
                "single direction improvement as sufficient for temporal-gate rescue"
            ),
            "method_level_sufficiency": False,
        },
        {
            "evidence_id": "F08",
            "stage": "final synthesis",
            "mechanism": "single-factor explanation",
            "status": "NOT_SUPPORTED",
            "evidence_type": "cross-intervention synthesis",
            "supports": (
                "multiple interacting/conditioned mechanisms are needed to explain the "
                "observed failure pattern"
            ),
            "does_not_support": "one unique causal architectural root cause",
            "method_level_sufficiency": False,
        },
    ]


def _build_mechanism_synthesis() -> dict[str, Any]:
    return {
        "gradient_starvation": {
            "status": "REJECTED_AS_PRIMARY_CAUSE",
            "unique_root_cause": False,
        },
        "global_action_scale": {
            "status": "REJECTED_AS_SUFFICIENT_EXPLANATION",
            "unique_root_cause": False,
        },
        "hard_support_extent": {
            "status": "NOT_SUPPORTED_AS_PRIMARY_EXPLANATION",
            "unique_root_cause": False,
        },
        "role_conditioning": {
            "status": "REAL_MECHANISM_BUT_INSUFFICIENT",
            "causally_manipulable": True,
            "sufficient_solution": False,
        },
        "single_direction_alignment": {
            "status": "CAUSALLY_SUPPORTED_BUT_INSUFFICIENT",
            "causally_manipulable": True,
            "corrected_H": True,
            "temporal_gate_rescue": "0/32 -> 0/32",
            "sufficient_solution": False,
        },
        "width_conditioning": {
            "status": "CONDITIONING_EFFECT_CONFIRMED",
            "sufficient_independent_cause": False,
        },
        "normalized_temporal_spreading": {
            "status": "NOT_SUPPORTED",
        },
        "metric_support_time": {
            "status": "REJECTED_AS_SUFFICIENT_SOLUTION",
            "further_metric_search_authorized": False,
        },
        "conditioned_direction_rotation": {
            "status": "CAUSALLY_MANIPULABLE_BUT_INSUFFICIENT",
            "sufficient_solution": False,
        },
        "rpa_lrta": {
            "status": "MECHANISTIC_EFFECT_PRESENT_METHOD_NOT_SUPPORTED",
            "single_direction_improved": True,
            "cross28_direction_improved": False,
            "single_temporal_gate": "0/32 -> 0/32",
            "cross28_temporal_gate": "0/16 -> 0/16",
            "endpoint_regressions": 3,
            "physical_regressions": 1,
            "sufficient_safe_method": False,
        },
    }


def _paper_safe_summary() -> dict[str, str]:
    return {
        "core_conclusion": (
            "Multiple Refiner failure mechanisms are empirically identifiable and "
            "some are causally manipulable, particularly role conditioning and "
            "single-recording action-direction alignment. However, none of the "
            "tested isolated or conditioned interventions provides a sufficiently "
            "effective and safe Refiner candidate under the frozen temporal, "
            "endpoint, physical, geometry, support, contact, and clean-identity "
            "criteria."
        ),
        "single_recording": (
            "The corrected RPA-LRTA direction audit confirms that single-recording "
            "action-direction alignment is a genuine manipulable mechanism. This "
            "direction gain is nevertheless insufficient for temporal-gate rescue: "
            "single-recording remains 0/32 under the frozen final evaluation."
        ),
        "cross_event_width28": (
            "Cross-event width-28 remains 0/16 under RPA-LRTA. Continuous temporal "
            "deficit can improve without a corresponding corrected direction gain, "
            "so the remaining long-window failure cannot be reduced to the tested "
            "direction-alignment mechanism alone."
        ),
        "method_boundary": (
            "Mechanism success must not be conflated with method success. RPA-LRTA "
            "shows a single-direction mechanism signal while also introducing three "
            "endpoint regressions and one physical regression; the formal candidate "
            "therefore remains unsupported."
        ),
    }


def _markdown_summary(report: Mapping[str, Any]) -> str:
    rpa = report["rpa_lrta"]
    direction = report["direction_correction"]
    lines = [
        "# Final Refiner Evidence Synthesis",
        "",
        "## Final classification",
        "",
        f"`{report['final_classification']}`",
        "",
        "No formal Refiner method candidate is supported for Pilot or production.",
        "",
        "## Frozen evidence chain",
        "",
        "- Gradient starvation: rejected as the primary cause.",
        "- Global action scale: insufficient as a standalone explanation.",
        "- Role conditioning: real and useful, but insufficient.",
        "- Width: conditions effectiveness; not established as a unique root cause.",
        "- BCTR metric/support-time intervention: rejected as a sufficient solution.",
        "- SECDR: direction is causally manipulable but insufficient alone.",
        (
            "- RPA-LRTA: RCSP temporal pass 5/64 -> RPA 6/64; "
            f"single {rpa['single_temporal_gate']}; "
            f"cross/28 {rpa['cross28_temporal_gate']}."
        ),
        (
            "- RPA safety trade-off: "
            f"{rpa['endpoint_regressions']} endpoint regressions and "
            f"{rpa['physical_regressions']} physical regression."
        ),
        (
            "- Corrected RPA direction result: "
            f"H={direction['corrected_H_single_direction_improved']}, "
            f"I={direction['corrected_I_cross28_direction_improved']}."
        ),
        "",
        "## Scientific interpretation",
        "",
        (
            "Single-recording direction/alignment is now supported as a genuine "
            "manipulable mechanism, but the improvement does not rescue any of the "
            "32 single-recording temporal gates."
        ),
        "",
        (
            "Cross-event width-28 remains unresolved at method level: its continuous "
            "deficit can move, but the corrected target direction condition is not "
            "improved and temporal gate pass remains 0/16."
        ),
        "",
        (
            "The evidence therefore supports multiple conditioned/manipulable "
            "mechanisms without a sufficient safe method candidate. It does not "
            "prove one unique architectural root cause, nor does it prove that all "
            "possible Refiner architectures are impossible."
        ),
        "",
        "## Stop rule",
        "",
        "- Candidate development: CLOSED for the current protocol.",
        "- New architecture search: NOT AUTHORIZED.",
        "- New metric search: NOT AUTHORIZED.",
        "- New scale/width/direction sweep: NOT AUTHORIZED.",
        "- Pilot: NOT AUTHORIZED.",
        "- Production change: NOT AUTHORIZED.",
        "",
        "Next action:",
        "",
        f"`{report['next_action']}`",
        "",
    ]
    return "\n".join(lines)


def _build_report(
    *,
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    runtime_commit: str,
    joint: Mapping[str, Any],
    joint_info: Mapping[str, Any],
    rpa_info: Mapping[str, Any],
    rpa_freeze_info: Mapping[str, Any],
    direction_info: Mapping[str, Any],
    direction_freeze_info: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_ledger = _build_evidence_ledger(joint, rpa_info, direction_info)
    mechanism_synthesis = _build_mechanism_synthesis()

    return {
        "schema": SCHEMA,
        "completed": True,
        "provenance": {
            "runtime_commit": runtime_commit,
            "implementation_parent_commit": IMPLEMENTATION_PARENT_COMMIT,
            "execution_mode": "read-only frozen JSON evidence synthesis",
            "explicit_input_paths_only": True,
            "latest_artifact_search_performed": False,
        },
        "input_artifacts": {
            name: {
                "path": str(paths[name]),
                "sha256": hashes[name],
            }
            for name in INPUT_NAMES
        },
        "validated_inputs": {
            "pre_rpa_joint": dict(joint_info),
            "rpa_lrta": {
                key: value
                for key, value in rpa_info.items()
                if key not in ("overall_summary", "single_summary", "cross28_summary")
            },
            "rpa_freeze": dict(rpa_freeze_info),
            "direction_correction": dict(direction_info),
            "direction_freeze": dict(direction_freeze_info),
        },
        "read_only": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "model_loaded": False,
        "checkpoint_loaded": False,
        "forward_pass_performed": False,
        "autograd_used": False,
        "metric_recomputed": False,
        "case_recomputed": False,
        "parameter_update_performed": False,
        "checkpoint_selection_performed": False,
        "case_selection_performed": False,
        "metric_selection_performed": False,
        "architecture_selection_performed": False,
        "intervention_search_performed": False,
        "new_training_performed": False,
        "evidence_ledger": evidence_ledger,
        "mechanism_synthesis": mechanism_synthesis,
        "pre_rpa_joint_decision": PRE_RPA_DECISION,
        "rpa_lrta": {
            "decision": RPA_DECISION,
            "termination": "DETERMINISTIC_NO_DESCENT_FIXED_POINT",
            "attempt_budget": rpa_info["attempt_budget"],
            "attempted_steps": rpa_info["attempted_steps"],
            "accepted_steps": rpa_info["accepted_steps"],
            "rollback_steps": rpa_info["rollback_steps"],
            "temporal_pass": "5/64 -> 6/64",
            "temporal_newly_rescued": 1,
            "single_temporal_gate": rpa_info["single_temporal_gate"],
            "cross28_temporal_gate": rpa_info["cross28_temporal_gate"],
            "endpoint_regressions": rpa_info["endpoint_regressions"],
            "physical_regressions": rpa_info["physical_regressions"],
            "method_level_supported": False,
        },
        "direction_correction": {
            "corrected_H_single_direction_improved": True,
            "corrected_I_cross28_direction_improved": False,
            "classification": DIRECTION_CLASSIFICATION,
            "single_direction_mechanism_supported": True,
            "single_direction_sufficient_for_gate_rescue": False,
            "cross28_target_direction_improved": False,
            "formal_method_decision_changed": False,
        },
        "supported_findings": [
            "role conditioning matters diagnostically",
            "width conditions intervention effectiveness",
            "single-recording direction/alignment is a genuine manipulable mechanism",
            "direction is causally manipulable in controlled interventions",
            "cross-event width-28 continuous temporal deficit can be reduced",
            "RPA-LRTA produces a mechanistic signal and one additional temporal rescue",
        ],
        "rejected_as_sufficient_explanations_or_solutions": [
            "gradient starvation",
            "global action scale alone",
            "hard support extent alone",
            "role conditioning alone",
            "metric/support-time intervention alone",
            "conditioned direction rotation alone",
            "RPA-LRTA as a sufficiently effective and safe method candidate",
        ],
        "not_proven": [
            "one unique causal architectural root cause",
            "global width conflict as the cause of single-recording failure",
            "direction as a sufficient cause of Refiner failure",
            "RPA-LRTA as generally ineffective outside the frozen protocol",
            "all possible Refiner architectures are impossible",
        ],
        "scientific_answers": {
            "single_direction_alignment_real": True,
            "single_direction_alignment_causally_manipulable": True,
            "single_direction_alignment_sufficient": False,
            "cross28_corrected_direction_gain_supported": False,
            "multiple_mechanisms_identified": True,
            "unique_causal_root_cause_proven": False,
            "sufficient_safe_refiner_candidate_exists": False,
            "candidate_development_should_continue_under_current_protocol": False,
        },
        "final_classification": FINAL_CLASSIFICATION,
        "final_candidate_supported": False,
        "candidate_development_closed": True,
        "new_architecture_search_authorized": False,
        "new_metric_search_authorized": False,
        "new_scale_search_authorized": False,
        "new_width_search_authorized": False,
        "new_direction_search_authorized": False,
        "new_intervention_search_authorized": False,
        "pilot_authorized": False,
        "production_change_authorized": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
        "paper_safe_summary": _paper_safe_summary(),
        "claim_boundary": (
            "The frozen evidence supports multiple real/manipulable mechanisms but "
            "no sufficiently effective and safe Refiner method candidate. It does "
            "not identify one unique causal architectural root cause and does not "
            "generalize the negative method result beyond the frozen protocol."
        ),
        "next_action": FINAL_NEXT_ACTION,
    }


def synthesize(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "joint_report": Path(args.joint_report).resolve(),
        "rpa_report": Path(args.rpa_report).resolve(),
        "rpa_freeze": Path(args.rpa_freeze).resolve(),
        "direction_report": Path(args.direction_report).resolve(),
        "direction_freeze": Path(args.direction_freeze).resolve(),
    }
    output = Path(args.output_dir).resolve()

    for name, path in paths.items():
        if not path.is_file():
            raise EvidenceIntegrityError(f"{name} missing: {path}")

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise FileExistsError(
            "Final Refiner synthesis output must be a fresh empty directory"
        )
    if any(
        output == path.parent or output.is_relative_to(path.parent)
        for path in paths.values()
    ):
        raise FileExistsError(
            "Final Refiner synthesis output overlaps a frozen input directory"
        )

    runtime_commit = _runtime_commit()
    if runtime_commit != args.expected_commit:
        raise EvidenceIntegrityError(
            "runtime commit does not match --expected-commit"
        )

    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir = output / "result"
    result_dir.mkdir(exist_ok=False)
    failure_path = result_dir / "failure.json"

    hashes_before = {
        name: _sha256(path)
        for name, path in paths.items()
    }

    try:
        _exact(
            hashes_before["rpa_report"],
            EXPECTED_RPA_REPORT_SHA256,
            "RPA report SHA256",
        )
        _exact(
            hashes_before["direction_report"],
            EXPECTED_DIRECTION_REPORT_SHA256,
            "direction report SHA256",
        )

        joint = _load_json(paths["joint_report"], "joint_report")
        rpa_report = _load_json(paths["rpa_report"], "rpa_report")
        rpa_freeze = _load_json(paths["rpa_freeze"], "rpa_freeze")
        direction_report = _load_json(
            paths["direction_report"],
            "direction_report",
        )
        direction_freeze = _load_json(
            paths["direction_freeze"],
            "direction_freeze",
        )

        joint_info = _validate_joint(joint)
        rpa_info = _validate_rpa(rpa_report)
        rpa_freeze_info = _validate_rpa_freeze(
            rpa_freeze,
            paths["rpa_report"],
            hashes_before["rpa_report"],
        )
        direction_info = _validate_direction(direction_report)
        direction_freeze_info = _validate_direction_freeze(
            direction_freeze,
            paths["direction_report"],
            hashes_before["direction_report"],
        )

        hashes_after_validation = {
            name: _sha256(path)
            for name, path in paths.items()
        }
        if hashes_after_validation != hashes_before:
            raise EvidenceIntegrityError(
                "a frozen input changed during final evidence validation"
            )

        report = _build_report(
            paths=paths,
            hashes=hashes_before,
            runtime_commit=runtime_commit,
            joint=joint,
            joint_info=joint_info,
            rpa_info=rpa_info,
            rpa_freeze_info=rpa_freeze_info,
            direction_info=direction_info,
            direction_freeze_info=direction_freeze_info,
        )

        report_path = result_dir / "report.json"
        summary_path = result_dir / "evidence_summary.md"
        freeze_path = result_dir / "freeze_manifest.json"

        _exclusive_json(report_path, report)
        _exclusive_text(summary_path, _markdown_summary(report))

        hashes_before_freeze = {
            name: _sha256(path)
            for name, path in paths.items()
        }
        if hashes_before_freeze != hashes_before:
            raise EvidenceIntegrityError(
                "a frozen input changed during final evidence synthesis"
            )

        freeze_manifest = {
            "schema": "refiner_final_evidence_synthesis_freeze_v1",
            "completed": True,
            "runtime_commit": runtime_commit,
            "source_schema": SCHEMA,
            "source_classification": FINAL_CLASSIFICATION,
            "source_next_action": FINAL_NEXT_ACTION,
            "input_artifacts": {
                name: {
                    "path": str(paths[name]),
                    "sha256": hashes_before[name],
                }
                for name in INPUT_NAMES
            },
            "outputs": {
                "report": {
                    "path": str(report_path),
                    "sha256": _sha256(report_path),
                },
                "evidence_summary": {
                    "path": str(summary_path),
                    "sha256": _sha256(summary_path),
                },
            },
            "read_only": True,
            "candidate_development_closed": True,
            "new_training_performed": False,
            "new_metric_recomputation_performed": False,
            "new_case_recomputation_performed": False,
            "new_architecture_search_performed": False,
            "new_intervention_search_performed": False,
            "pilot_allowed": False,
            "production_change_authorized": False,
        }
        _exclusive_json(freeze_path, freeze_manifest)

        hashes_final = {
            name: _sha256(path)
            for name, path in paths.items()
        }
        if hashes_final != hashes_before:
            raise EvidenceIntegrityError(
                "a frozen input changed while writing final synthesis artifacts"
            )

        print(
            json.dumps(
                {
                    "stage": "refiner_final_evidence_synthesis_complete",
                    "report": str(report_path),
                    "evidence_summary": str(summary_path),
                    "freeze_manifest": str(freeze_path),
                    "classification": FINAL_CLASSIFICATION,
                    "candidate_development_closed": True,
                    "pilot_allowed": False,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )
        return report

    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "schema": SCHEMA,
                    "completed": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "read_only": True,
                    "optimizer_constructed": False,
                    "optimizer_steps": 0,
                    "model_loaded": False,
                    "checkpoint_loaded": False,
                    "forward_pass_performed": False,
                    "autograd_used": False,
                    "metric_recomputed": False,
                    "case_recomputed": False,
                    "parameter_update_performed": False,
                    "candidate_development_closed": True,
                    "new_architecture_search_authorized": False,
                    "new_intervention_search_authorized": False,
                    "pilot_allowed": False,
                    "production_model_modified": False,
                    "production_inference_modified": False,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-report", required=True)
    parser.add_argument("--rpa-report", required=True)
    parser.add_argument("--rpa-freeze", required=True)
    parser.add_argument("--direction-report", required=True)
    parser.add_argument("--direction-freeze", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    synthesize(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
