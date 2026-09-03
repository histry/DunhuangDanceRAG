"""Create-only correction of the frozen BCTR rescue-list reporting bug.

The frozen BCTR report contains the measurements and decision that were
already produced on the server.  Older reports accidentally wrote the full
newly-rescued list into the width-28 field.  This module reads the explicit
report once, recomputes only the three reporting lists from its case rows, and
writes a separate correction artifact.  It never loads a model, runs an
inference path, recomputes a metric, or overwrites the source report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "refiner_bctr_reporting_correction_v1"
BCTR_SCHEMA = "refiner_boundary_crossing_temporal_reduction_intervention_v1"
PRIMARY_CASES = 32
WIDTHS = (10, 28)
SCOPES = ("overall", "seen", "new")
EXPECTED_DECISION = "METRIC_SUPPORT_TIME_INTERVENTION_NOT_SUPPORTED"
EXPECTED_NEXT_ACTION = "reject_bctr_candidate_and_proceed_to_width_conditioned_direction_intervention"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _exact_bool(value: Any, expected: bool, label: str) -> None:
    if isinstance(expected, bool):
        matches = value is expected
    else:
        matches = value == expected
    if not matches:
        raise ValueError(f"{label} must be exactly {expected}")


def _identity(row: Mapping[str, Any]) -> str:
    value = row.get("identity")
    if not isinstance(value, str) or not value:
        raise ValueError("BCTR case row is missing identity")
    return value


def _validate_case_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("case_level")
    if not isinstance(rows, list) or len(rows) != PRIMARY_CASES:
        raise ValueError("BCTR correction requires exactly 32 primary case rows")
    result: list[dict[str, Any]] = []
    identities: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("BCTR primary case rows must be objects")
        identity = _identity(row)
        if identity in identities:
            raise ValueError(f"duplicate BCTR case identity: {identity}")
        identities.add(identity)
        if row.get("role") != "cross_event":
            raise ValueError("BCTR correction primary rows must be cross_event cases")
        if row.get("split") not in ("seen", "new_position"):
            raise ValueError("BCTR correction primary row has an unexpected split")
        if int(row.get("width")) not in WIDTHS:
            raise ValueError("BCTR correction primary row has an unexpected width")
        current = row.get("current", {})
        bctr = row.get("bctr", {})
        if not isinstance(current, Mapping) or not isinstance(bctr, Mapping):
            raise ValueError(f"BCTR row lacks current/bctr gate fields: {identity}")
        _exact_bool(current.get("temporal_pass_rcsp"), current.get("temporal_pass_rcsp") is True, f"{identity} current temporal pass")
        _exact_bool(bctr.get("candidate_temporal_pass_rcsp"), bctr.get("candidate_temporal_pass_rcsp") is True, f"{identity} BCTR temporal pass")
        result.append(row)
    return result


def _rows_for_scope(rows: list[Mapping[str, Any]], scope: str) -> list[Mapping[str, Any]]:
    if scope not in SCOPES:
        raise ValueError(f"unknown correction scope: {scope}")
    if scope == "overall":
        return list(rows)
    expected_split = "seen" if scope == "seen" else "new_position"
    return [row for row in rows if row.get("split") == expected_split]


def _rescue_lists(rows: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Recompute all/width10/width28 rescues from the frozen case rows."""
    rescued_rows = [
        row
        for row in rows
        if row["current"]["temporal_pass_rcsp"] is False
        and row["bctr"]["candidate_temporal_pass_rcsp"] is True
    ]
    newly_rescued = [_identity(row) for row in rescued_rows]
    width10 = [
        _identity(row) for row in rescued_rows if int(row["width"]) == 10
    ]
    width28 = [
        _identity(row) for row in rescued_rows if int(row["width"]) == 28
    ]
    if len(set(newly_rescued)) != len(newly_rescued):
        raise ValueError("newly rescued identities are not unique")
    if any("/10/" not in identity for identity in width10):
        raise ValueError("width-10 rescue list contains a non-width-10 identity")
    if any("/28/" not in identity for identity in width28):
        raise ValueError("width-28 rescue list contains a non-width-28 identity")
    if set(width10) | set(width28) != set(newly_rescued):
        raise ValueError("width-specific rescue lists do not partition all rescues")
    if set(width10) & set(width28):
        raise ValueError("width-specific rescue lists overlap")
    return {
        "newly_rescued_cases": newly_rescued,
        "width10_newly_rescued_cases": width10,
        "width28_newly_rescued_cases": width28,
    }


def corrected_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(rows) != PRIMARY_CASES:
        raise ValueError("BCTR correction summaries require 32 primary rows")
    return {
        scope: {
            "cases": len(scope_rows := _rows_for_scope(rows, scope)),
            **_rescue_lists(scope_rows),
        }
        for scope in SCOPES
    }


def _decision_inputs(report: Mapping[str, Any]) -> dict[str, Any]:
    summaries = report.get("summaries", {})
    result: dict[str, Any] = {}
    for scope in SCOPES:
        summary = summaries.get(scope)
        if not isinstance(summary, Mapping):
            raise ValueError(f"BCTR summary is missing: {scope}")
        fields = (
            "split_supported",
            "width10_degradation",
            "endpoint_semantics_unchanged",
            "jerk_semantics_unchanged",
            "outputs_unchanged",
            "state_unchanged",
        )
        if any(field not in summary for field in fields):
            raise ValueError(f"BCTR decision inputs are incomplete: {scope}")
        result[scope] = {field: summary[field] for field in fields}
    return result


def recompute_decision(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    seen = inputs["seen"]
    new = inputs["new"]
    overall = inputs["overall"]
    both_supported = bool(seen["split_supported"] and new["split_supported"])
    one_split_supported = bool(seen["split_supported"] != new["split_supported"])
    overall_supported = bool(overall["split_supported"])
    width10_degradation = bool(seen["width10_degradation"] or new["width10_degradation"])
    controls_ok = all(
        bool(scope[field])
        for scope in (seen, new)
        for field in (
            "endpoint_semantics_unchanged",
            "jerk_semantics_unchanged",
            "outputs_unchanged",
            "state_unchanged",
        )
    )
    if both_supported:
        result = "METRIC_SUPPORT_TIME_INTERVENTION_SUPPORTED"
        next_action = "freeze_candidate_and_design_separate_direction_intervention"
    elif (overall_supported or one_split_supported) and not width10_degradation and controls_ok:
        result = "PARTIAL_METRIC_SUPPORT_TIME_INTERVENTION"
        next_action = "retain_partial_evidence_and_proceed_to_width_conditioned_direction_intervention"
    else:
        result = EXPECTED_DECISION
        next_action = EXPECTED_NEXT_ACTION
    return {
        "result": result,
        "next_action": next_action,
        "split_supported": {
            "seen": bool(seen["split_supported"]),
            "new": bool(new["split_supported"]),
        },
        "overall_supported": overall_supported,
        "width10_degradation_observed": width10_degradation,
        "endpoint_semantics_unchanged": controls_ok,
        "no_further_metric_search": True,
        "causal_root_cause_proven": False,
    }


def _validate_source(report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if report.get("schema") != BCTR_SCHEMA:
        raise ValueError("source report schema mismatch")
    _exact_bool(report.get("completed"), True, "source report completed")
    rows = _validate_case_rows(report)
    decision_inputs = _decision_inputs(report)
    stored_decision = report.get("decision")
    if not isinstance(stored_decision, Mapping):
        raise ValueError("source BCTR decision is missing")
    recomputed = recompute_decision(decision_inputs)
    for field in (
        "result",
        "next_action",
        "split_supported",
        "overall_supported",
        "width10_degradation_observed",
        "endpoint_semantics_unchanged",
        "no_further_metric_search",
        "causal_root_cause_proven",
    ):
        if stored_decision.get(field) != recomputed[field]:
            raise ValueError(f"source BCTR decision input mismatch: {field}")
    if stored_decision.get("result") != EXPECTED_DECISION:
        raise ValueError("frozen BCTR decision is not the required NOT_SUPPORTED result")
    if stored_decision.get("next_action") != EXPECTED_NEXT_ACTION:
        raise ValueError("frozen BCTR next action mismatch")
    for field in (
        "optimizer_steps",
        "parameter_update_performed",
        "production_model_modified",
        "production_inference_modified",
        "scientific_acceptance",
        "publish_allowed",
        "pilot_allowed",
    ):
        expected = 0 if field == "optimizer_steps" else False
        _exact_bool(report.get(field), expected, f"source BCTR {field}")
    _exact_bool(report.get("no_further_metric_search"), True, "source BCTR no_further_metric_search")
    return rows, decision_inputs, dict(stored_decision)


def run(args: argparse.Namespace) -> int:
    source = Path(args.bctr_report).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"BCTR report does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("BCTR correction output must be a fresh empty directory")
    if output == source or output.is_relative_to(source.parent):
        raise FileExistsError("BCTR correction output overlaps the immutable source report")
    source_hash_before = _file_sha256(source)
    report = json.loads(source.read_text(encoding="utf-8"))
    rows, decision_inputs, source_decision = _validate_source(report)
    corrected = corrected_summaries(rows)
    source_hash_after = _file_sha256(source)
    if source_hash_before != source_hash_after:
        raise RuntimeError("source BCTR report changed during correction")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir = output / "result"
    result_dir.mkdir(exist_ok=False)
    runtime_commit = args.expected_main_commit
    report_out = {
        "schema": SCHEMA,
        "completed": True,
        "provenance": {
            "runtime_commit": runtime_commit,
            "source_bctr_report": str(source),
            "source_bctr_report_sha256": source_hash_before,
            "source_bctr_schema": BCTR_SCHEMA,
            "source_decision": source_decision["result"],
        },
        "correction": {
            "reason": "width-28 rescue list previously received the all-width newly-rescued list",
            "source_report_immutable": True,
            "source_report_sha256_before": source_hash_before,
            "source_report_sha256_after": source_hash_after,
            "source_report_modified": False,
            "measurements_changed": False,
            "decision_inputs_changed": False,
            "scientific_classification_changed": False,
            "recomputed_decision_same": True,
        },
        "decision_inputs": decision_inputs,
        "source_decision_record": source_decision,
        "corrected_summaries": corrected,
        "decision": {
            "result": source_decision["result"],
            "next_action": source_decision["next_action"],
            "recomputed_same_as_source": True,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
        },
        "optimizer_steps": 0,
        "model_loaded": False,
        "inference_performed": False,
        "metric_recomputed": False,
        "source_report_modified": False,
        "measurements_changed": False,
        "decision_inputs_changed": False,
        "scientific_classification_changed": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
    }
    _exclusive_json(result_dir / "report.json", report_out)
    print(json.dumps({
        "stage": "refiner_bctr_reporting_correction_complete",
        "report": str(result_dir / "report.json"),
        "source_report_sha256": source_hash_before,
        "source_report_modified": False,
        "decision": source_decision["result"],
        "scientific_acceptance": False,
        "pilot_allowed": False,
    }, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bctr-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-main-commit", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
