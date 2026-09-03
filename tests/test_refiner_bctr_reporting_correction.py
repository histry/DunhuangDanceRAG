"""Contract tests for the create-only BCTR reporting correction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import refiner_bctr_reporting_correction as correction


def _row(split, width, index, current=False, candidate=False):
    return {
        "split": split,
        "role": "cross_event",
        "width": width,
        "identity": f"{split}/cross_event/{width}/{index}",
        "current": {"temporal_pass_rcsp": current},
        "bctr": {"candidate_temporal_pass_rcsp": candidate},
    }


def _report(rows=None, *, result=None):
    rows = rows or [
        _row(split, width, index, current=False, candidate=(width == 28 and index == 0))
        for split in ("seen", "new_position")
        for width in (10, 28)
        for index in range(8)
    ]
    result = result or correction.EXPECTED_DECISION
    summary = {
        "split_supported": False,
        "width10_degradation": False,
        "endpoint_semantics_unchanged": True,
        "jerk_semantics_unchanged": True,
        "outputs_unchanged": True,
        "state_unchanged": True,
    }
    return {
        "schema": correction.BCTR_SCHEMA,
        "completed": True,
        "case_level": rows,
        "summaries": {scope: dict(summary) for scope in correction.SCOPES},
        "decision": {
            "result": result,
            "next_action": correction.EXPECTED_NEXT_ACTION,
            "split_supported": {"seen": False, "new": False},
            "overall_supported": False,
            "width10_degradation_observed": False,
            "endpoint_semantics_unchanged": True,
            "no_further_metric_search": True,
            "causal_root_cause_proven": False,
        },
        "optimizer_steps": 0,
        "parameter_update_performed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "no_further_metric_search": True,
    }


def test_schema_is_exact():
    assert correction.SCHEMA == "refiner_bctr_reporting_correction_v1"


def test_source_schema_is_exact():
    assert correction.BCTR_SCHEMA == "refiner_boundary_crossing_temporal_reduction_intervention_v1"


def test_primary_case_count_is_frozen():
    assert correction.PRIMARY_CASES == 32


def test_scope_names_are_frozen():
    assert correction.SCOPES == ("overall", "seen", "new")


def test_rescue_list_recomputes_from_current_and_candidate_flags():
    rows = [_row("seen", 10, 0, False, True), _row("seen", 28, 1, False, True)]
    lists = correction._rescue_lists(rows)
    assert lists["newly_rescued_cases"] == [rows[0]["identity"], rows[1]["identity"]]


def test_width10_list_is_restricted():
    rows = [_row("seen", 10, 0, False, True)]
    assert correction._rescue_lists(rows)["width10_newly_rescued_cases"] == [rows[0]["identity"]]


def test_width28_list_is_restricted():
    rows = [_row("seen", 28, 0, False, True)]
    assert correction._rescue_lists(rows)["width28_newly_rescued_cases"] == [rows[0]["identity"]]


def test_non_rescue_is_excluded():
    rows = [_row("seen", 10, 0, True, True), _row("seen", 28, 1, False, False)]
    assert correction._rescue_lists(rows)["newly_rescued_cases"] == []


def test_partition_union_equals_all_rescues():
    rows = [_row("seen", 10, 0, False, True), _row("seen", 28, 1, False, True)]
    result = correction._rescue_lists(rows)
    assert set(result["width10_newly_rescued_cases"]) | set(result["width28_newly_rescued_cases"]) == set(result["newly_rescued_cases"])


def test_partition_has_no_overlap():
    rows = [_row("seen", 10, 0, False, True), _row("seen", 28, 1, False, True)]
    result = correction._rescue_lists(rows)
    assert not (set(result["width10_newly_rescued_cases"]) & set(result["width28_newly_rescued_cases"]))


def test_duplicate_identity_fails_closed():
    rows = [_row("seen", 10, 0, False, True), _row("seen", 10, 0, False, True)]
    with pytest.raises(ValueError, match="not unique"):
        correction._rescue_lists(rows)


def test_scope_mapping_seen():
    rows = [_row("seen", 10, 0), _row("new_position", 28, 1)]
    assert len(correction._rows_for_scope(rows, "seen")) == 1


def test_scope_mapping_new():
    rows = [_row("seen", 10, 0), _row("new_position", 28, 1)]
    assert len(correction._rows_for_scope(rows, "new")) == 1


def test_scope_mapping_overall():
    rows = [_row("seen", 10, 0), _row("new_position", 28, 1)]
    assert correction._rows_for_scope(rows, "overall") == rows


def test_unknown_scope_fails_closed():
    with pytest.raises(ValueError, match="unknown correction scope"):
        correction._rows_for_scope([], "bad")


def test_summary_contains_all_three_lists_for_all_scopes():
    rows = [
        _row(split, width, index, False, width == 28 and index == 0)
        for split in ("seen", "new_position")
        for width in (10, 28)
        for index in range(8)
    ]
    summaries = correction.corrected_summaries(rows)
    for scope in correction.SCOPES:
        assert set(("newly_rescued_cases", "width10_newly_rescued_cases", "width28_newly_rescued_cases")) <= set(summaries[scope])


def test_recompute_not_supported_decision():
    inputs = {scope: {
        "split_supported": False,
        "width10_degradation": False,
        "endpoint_semantics_unchanged": True,
        "jerk_semantics_unchanged": True,
        "outputs_unchanged": True,
        "state_unchanged": True,
    } for scope in correction.SCOPES}
    result = correction.recompute_decision(inputs)
    assert result["result"] == correction.EXPECTED_DECISION
    assert result["next_action"] == correction.EXPECTED_NEXT_ACTION


def test_source_validation_accepts_frozen_contract():
    rows, inputs, decision = correction._validate_source(_report())
    assert len(rows) == 32
    assert inputs["seen"]["split_supported"] is False
    assert decision["result"] == correction.EXPECTED_DECISION


def test_source_validation_rejects_wrong_decision():
    with pytest.raises(ValueError, match="required NOT_SUPPORTED"):
        correction._validate_source(_report(result="PARTIAL_METRIC_SUPPORT_TIME_INTERVENTION"))


def test_source_validation_rejects_wrong_schema():
    report = _report()
    report["schema"] = "wrong"
    with pytest.raises(ValueError, match="schema mismatch"):
        correction._validate_source(report)


def test_source_validation_rejects_non_primary_case_count():
    report = _report()
    report["case_level"] = report["case_level"][:-1]
    with pytest.raises(ValueError, match="exactly 32"):
        correction._validate_source(report)


def test_module_has_no_model_or_optimizer_symbols():
    source = Path(correction.__file__).read_text(encoding="utf-8")
    assert "torch" not in source
    assert "optimizer.step" not in source
    assert "load_state_dict" not in source


def test_source_report_is_not_written_by_module():
    source = Path(correction.__file__).read_text(encoding="utf-8")
    assert 'source.open("w"' not in source
    assert "source.write_text" not in source
