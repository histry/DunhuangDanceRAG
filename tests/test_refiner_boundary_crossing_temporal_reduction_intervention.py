"""Deterministic contract tests for the frozen BCTR intervention."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training import refiner_boundary_crossing_temporal_reduction_intervention as audit
from training import refiner_width_mechanism_adjudication_audit as phase21


def _cfg(threshold: float = 0.03):
    return SimpleNamespace(
        checkpoint_validation_min_temporal_repair_gain=threshold,
        checkpoint_validation_min_endpoint_repair_gain=0.03,
    )


def _metric(energy: float, endpoint: float = 1.0, jerk: float = 1.0, valid: bool = True):
    return {
        "endpoint_velocity_jump_mps": endpoint,
        "seam_acceleration_mps2": energy * 10.0,
        "seam_jerk_mps3": jerk,
        "context_seam_acceleration_mps2": 0.0,
        "context_seam_jerk_mps3": 0.0,
        "temporal_energy": energy,
        "context_temporal_energy": 0.0,
        "valid": valid,
    }


def _bctr(energy: float | None, valid: bool = True):
    return {"temporal_energy": energy, "valid": valid}


def _summary(*, supported: bool, width10_degradation: bool = False):
    return {
        "split_supported": supported,
        "width10_degradation": width10_degradation,
        "endpoint_semantics_unchanged": True,
        "jerk_semantics_unchanged": True,
        "outputs_unchanged": True,
        "state_unchanged": True,
    }


def _decision_summaries(seen: bool, new: bool, *, overall: bool = False, degraded: bool = False):
    return {
        "overall": _summary(supported=overall, width10_degradation=degraded),
        "seen": _summary(supported=seen, width10_degradation=degraded),
        "new": _summary(supported=new, width10_degradation=degraded),
    }


def _row(split: str, width: int, index: int, current_gain: float, bctr_gain: float):
    current_pass = current_gain >= 0.03
    bctr_pass = bctr_gain >= 0.03
    return {
        "split": split,
        "role": "cross_event",
        "width": width,
        "identity": f"{split}/cross_event/{width}/{index}",
        "current": {
            "G_rcsp": current_gain,
            "temporal_pass_rcsp": current_pass,
        },
        "bctr": {
            "G_rcsp_BCTR": bctr_gain,
            "candidate_temporal_pass_rcsp": bctr_pass,
            "valid": True,
        },
        "anti_gaming": {
            "endpoint_semantics_unchanged": True,
            "jerk_semantics_unchanged": True,
            "outputs_unchanged": True,
            "state_unchanged": True,
        },
    }


def test_schema_and_frozen_phase21_commit_are_exact():
    assert audit.SCHEMA == "refiner_boundary_crossing_temporal_reduction_intervention_v1"
    assert audit.FROZEN_PHASE21_COMMIT == "c461ba44689103cd0690488267e3bd42507ad7ab"
    assert audit.MAJOR_GAP_FRACTION == 0.50
    assert audit.TEMPORAL_FLOOR == 1.0e-6


def test_primary_cohort_constants_are_frozen():
    assert audit.PRIMARY_CASES == 32
    assert audit.FINAL_CASES == 64
    assert audit.WIDTHS == (10, 28)
    assert audit.GROUP_ORDER == (
        "seen/cross_event/10", "seen/cross_event/28",
        "new_position/cross_event/10", "new_position/cross_event/28",
    )
    assert audit.PRIMARY_ROLE == "cross_event"
    assert audit.EXCLUDED_ROLE == "single_recording"


def test_crossing_support_has_no_width_parameter():
    assert set(inspect.signature(audit.boundary_crossing_support).parameters) == {"seam", "order"}
    assert "width" not in inspect.signature(audit.boundary_crossing_support).parameters


def test_crossing_support_selects_mixed_stencils_only():
    seam = torch.tensor([[[0.0], [0.0], [1.0], [1.0], [1.0], [0.0]]])
    assert audit.boundary_crossing_support(seam, 2).tolist() == [[True, True, False, True]]


def test_crossing_support_rejects_pure_core_and_pure_outside_stencils():
    all_core = torch.ones((1, 6, 1))
    all_outside = torch.zeros((1, 6, 1))
    assert not bool(audit.boundary_crossing_support(all_core, 2).any())
    assert not bool(audit.boundary_crossing_support(all_outside, 2).any())


def test_crossing_support_rejects_nonpositive_order():
    with pytest.raises(ValueError, match="positive"):
        audit.boundary_crossing_support(torch.zeros((1, 4, 1)), 0)


def test_crossing_support_returns_empty_when_window_is_shorter_than_order():
    result = audit.boundary_crossing_support(torch.ones((2, 3, 1)), 3)
    assert tuple(result.shape) == (2, 0)


def test_bctr_uses_float64_production_derivative_and_exact_scale():
    joints = torch.zeros((1, 8, 24, 3), dtype=torch.float32)
    joints[0, :, 0, 0] = torch.arange(8, dtype=torch.float32).square()
    seam = torch.zeros((1, 8, 1), dtype=torch.float32)
    seam[:, 2:4] = 1.0
    row = audit._bctr_temporal_rows(joints, seam, 30.0)[0]
    assert row["terms"]["seam_acceleration"]["order"] == 2
    assert row["terms"]["seam_acceleration"]["scale"] == 10.0
    assert row["terms"]["seam_acceleration"]["normalized_value"] == pytest.approx(7.5)
    assert row["terms"]["seam_jerk"]["scale"] == 1000.0
    assert row["valid"] is True


def test_bctr_temporal_energy_is_acceleration_over_10_plus_jerk_over_1000():
    joints = torch.zeros((1, 8, 24, 3), dtype=torch.float64)
    joints[0, :, 0, 0] = torch.arange(8, dtype=torch.float64).square()
    seam = torch.zeros((1, 8, 1))
    seam[:, 2:4] = 1.0
    row = audit._bctr_temporal_rows(joints, seam, 30.0)[0]
    expected = row["terms"]["seam_acceleration"]["normalized_value"] + row["terms"]["seam_jerk"]["normalized_value"]
    assert row["temporal_energy"] == pytest.approx(expected)


def test_zero_crossing_support_is_invalid_and_null():
    joints = torch.zeros((1, 8, 24, 3), dtype=torch.float64)
    seam = torch.ones((1, 8, 1))
    row = audit._bctr_temporal_rows(joints, seam, 30.0)[0]
    assert row["terms"]["seam_acceleration"]["crossing_support_count"] == 0
    assert row["terms"]["seam_jerk"]["crossing_support_count"] == 0
    assert row["temporal_energy"] is None
    assert row["valid"] is False


def test_bctr_support_counts_are_not_clamped_to_one():
    joints = torch.zeros((1, 8, 24, 3), dtype=torch.float64)
    seam = torch.zeros((1, 8, 1))
    row = audit._bctr_temporal_rows(joints, seam, 30.0)[0]
    assert row["terms"]["seam_acceleration"]["crossing_support_count"] == 0
    assert row["terms"]["seam_jerk"]["crossing_support_count"] == 0


def test_current_gate_uses_configured_temporal_threshold():
    before = _metric(10.0)
    after = _metric(9.6)
    assert audit._current_state(before, after, _cfg(0.03))["temporal_pass"] is True
    assert audit._current_state(before, after, _cfg(0.05))["temporal_pass"] is False


def test_gate_floor_semantics_match_production_for_zero_before():
    before = _metric(0.0)
    after = _metric(0.0)
    result = audit._current_state(before, after, _cfg())
    assert result["G"] == pytest.approx(1.0)
    after_positive = _metric(1.0)
    assert audit._current_state(before, after_positive, _cfg())["G"] == pytest.approx(-1.0)


def test_candidate_uses_original_full_support_jerk():
    before = _metric(10.0, jerk=1.0)
    after = _metric(9.0, jerk=2.0)
    candidate = audit._candidate_state(before, after, _bctr(10.0), _bctr(5.0), _cfg())
    assert candidate["jerk_non_regression"] is False
    assert candidate["temporal_pass"] is False


def test_candidate_endpoint_semantics_are_not_replaced():
    before = _metric(10.0, endpoint=2.0)
    after = _metric(9.0, endpoint=1.0)
    candidate = audit._candidate_state(before, after, _bctr(10.0), _bctr(5.0), _cfg())
    assert candidate["endpoint_acceptance"] is True


def test_candidate_overall_is_endpoint_and_temporal_acceptance():
    before = _metric(10.0, endpoint=2.0, jerk=1.0)
    after = _metric(9.0, endpoint=1.0, jerk=1.0)
    candidate = audit._candidate_state(before, after, _bctr(10.0), _bctr(9.8), _cfg())
    assert candidate["endpoint_acceptance"] is True
    assert candidate["temporal_pass"] is False


def test_scope_rows_maps_new_position_to_new_scope():
    rows = [{"split": "new_position"}, {"split": "seen"}]
    assert audit._scope_rows(rows, "new") == [rows[0]]
    assert audit._scope_rows(rows, "seen") == [rows[1]]
    assert audit._scope_rows(rows, "overall") == rows


def test_scope_summary_computes_current_and_bctr_gaps():
    rows = []
    for split in ("seen", "new_position"):
        for width in (10, 28):
            for index in range(8):
                rows.append(_row(split, width, index, 0.20 if width == 10 else 0.30, 0.21 if width == 10 else 0.40))
    summary = audit._scope_summary(audit._scope_rows(rows, "seen"), "seen")
    assert summary["current_gap_width28_minus_width10"] == pytest.approx(0.10)
    assert summary["bctr_gap_width28_minus_width10"] == pytest.approx(0.19)
    assert summary["delta_width28_gain"] == pytest.approx(0.10)
    assert summary["delta_width10_gain"] == pytest.approx(0.01)


def test_scope_summary_gap_shrink_is_null_when_current_gap_is_zero():
    rows = [_row("seen", width, index, 0.2, 0.2) for width in (10, 28) for index in range(8)]
    summary = audit._scope_summary(rows, "seen")
    assert summary["bctr_gap_shrink_fraction"] is None


def test_scope_summary_requires_strict_width28_improvement():
    rows = [_row("seen", width, index, 0.2 if width == 10 else 0.3, 0.2 if width == 10 else 0.3) for width in (10, 28) for index in range(8)]
    assert audit._scope_summary(rows, "seen")["split_supported"] is False


def test_scope_summary_allows_non_degraded_width10_median():
    rows = [_row("seen", width, index, 0.2 if width == 10 else 0.3, 0.2 if width == 10 else 0.4) for width in (10, 28) for index in range(8)]
    summary = audit._scope_summary(rows, "seen")
    assert summary["width10_median_gain_non_degraded"] is True


def test_scope_summary_reports_all_and_width_specific_rescue_lists():
    rows = [
        _row("seen", 10, 0, 0.01, 0.04),
        _row("seen", 28, 1, 0.01, 0.04),
        _row("seen", 28, 2, 0.04, 0.04),
    ]
    summary = audit._scope_summary(rows, "seen")
    assert summary["newly_rescued_cases"] == [
        "seen/cross_event/10/0", "seen/cross_event/28/1"
    ]
    assert summary["width10_newly_rescued_cases"] == ["seen/cross_event/10/0"]
    assert summary["width28_newly_rescued_cases"] == ["seen/cross_event/28/1"]


def test_make_summaries_has_overall_seen_and_new_only():
    rows = [_row(split, width, index, 0.2 if width == 10 else 0.3, 0.2 if width == 10 else 0.4)
            for split in ("seen", "new_position") for width in (10, 28) for index in range(8)]
    assert set(audit.make_summaries(rows)) == {"overall", "seen", "new"}
    assert audit.make_summaries(rows)["overall"]["cases"] == 32


def test_adjudication_support_requires_both_splits():
    result = audit.adjudicate(_decision_summaries(True, True))
    assert result["result"] == "METRIC_SUPPORT_TIME_INTERVENTION_SUPPORTED"
    assert result["next_action"] == "freeze_candidate_and_design_separate_direction_intervention"


def test_adjudication_one_split_is_partial():
    result = audit.adjudicate(_decision_summaries(True, False))
    assert result["result"] == "PARTIAL_METRIC_SUPPORT_TIME_INTERVENTION"
    assert result["next_action"] == "retain_partial_evidence_and_proceed_to_width_conditioned_direction_intervention"


def test_adjudication_both_splits_fail_closed():
    result = audit.adjudicate(_decision_summaries(False, False))
    assert result["result"] == "METRIC_SUPPORT_TIME_INTERVENTION_NOT_SUPPORTED"
    assert result["next_action"] == "reject_bctr_candidate_and_proceed_to_width_conditioned_direction_intervention"


def test_adjudication_width10_degradation_rejects_partial_claim():
    result = audit.adjudicate(_decision_summaries(True, False, degraded=True))
    assert result["result"] == "METRIC_SUPPORT_TIME_INTERVENTION_NOT_SUPPORTED"
    assert result["width10_degradation_observed"] is True


def test_adjudication_fixed_major_gap_fraction_is_half():
    result = audit.adjudicate(_decision_summaries(True, True))
    assert result["major_gap_fraction"] == 0.50
    assert result["no_further_metric_search"] is True


def test_validate_phase21_rejects_wrong_schema(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema": "wrong", "completed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        audit._validate_phase21_lineage(path)


def test_validate_phase21_rejects_uncompleted_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema": phase21.SCHEMA, "completed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        audit._validate_phase21_lineage(path)


def test_validate_phase21_rejects_wrong_frozen_runtime(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema": phase21.SCHEMA, "completed": True, "provenance": {"runtime_commit": "old"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen parent"):
        audit._validate_phase21_lineage(path)


def test_source_has_no_update_path():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "torch.optim" not in source


def test_production_modules_are_imported_read_only():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "from motion_geometry import boundary_observables" in source
    assert "from motion_geometry import product_manifold" in source
    assert "refiner_role_conditioned_support_projection_experiment" in source
    assert "load_state_dict" in source


def test_intervention_contract_has_no_direction_or_decoder_change():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert '"direction_changed": False' in source
    assert '"decoder_changed": False' in source
    assert '"gate_threshold_changed": False' in source


def test_report_false_flags_are_present_in_source():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    for field in (
        "optimizer_constructed", "optimizer_steps", "parameter_update_performed",
        "checkpoint_selection_performed", "scale_selection_performed",
        "architecture_selection_performed", "production_model_modified",
        "production_inference_modified", "scientific_acceptance", "publish_allowed", "pilot_allowed",
    ):
        assert f'"{field}"' in source


def test_script_uses_explicit_phase21_report_and_main_sha():
    script = Path(__file__).parents[1] / "scripts" / "run_refiner_boundary_crossing_temporal_reduction_intervention.sh"
    text = script.read_text(encoding="utf-8")
    assert "PHASE21_REPORT" in text
    assert "git rev-parse origin/main" in text
    assert "--expected-main-commit" in text
    assert "--device \"${DEVICE:-cuda}\"" in text


def test_docs_define_bctr_and_server_boundary():
    doc = Path(__file__).parents[1] / "docs" / "refiner_boundary_crossing_temporal_reduction_intervention.md"
    text = doc.read_text(encoding="utf-8")
    for phrase in ("BCTR", "seam >= 0.5", "crossing_support", "original full-support", "server", "no local validation"):
        assert phrase in text


def test_no_production_file_is_declared_as_an_edit_target():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    for name in ("boundary_observables.py", "motion_models.py", "refiner_role_conditioned_support_projection_experiment.py"):
        assert name not in source.split("implementation_paths", 1)[0]
