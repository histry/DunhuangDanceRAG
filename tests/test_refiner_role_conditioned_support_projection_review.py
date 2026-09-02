from pathlib import Path

import pytest

from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_role_conditioned_support_projection_review as review


def case_rows(*, adapted):
    rows = []
    for split, role in rcsp.FINAL_BLOCK_ORDER:
        for width in (10, 28):
            for case_index in range(8):
                rescued = adapted and role == "cross_event" and width == 10 and case_index < 2
                rows.append(
                    {
                        "split": split,
                        "role": role,
                        "width": width,
                        "case_index": case_index,
                        "bank_case_index": case_index,
                        "temporal_gate_pass": rescued,
                        "endpoint_gate_pass": rescued,
                        "physical_pass": True,
                        "geometry_pass": True,
                        "clean_pass": True,
                        "all_diagnostic_conditions": rescued,
                        "temporal_scientific_deficit": 0.8 if adapted else 1.0,
                        "endpoint_scientific_deficit": 0.9 if adapted else 1.0,
                        "temporal_repair_gain": 0.2 if adapted else 0.0,
                        "endpoint_repair_gain": 0.1 if adapted else 0.0,
                    }
                )
    return rows


def scoped_summaries(value):
    result = {"overall": value}
    for name in review.EXPECTED_GROUPS:
        result[f"group:{name}"] = value
    return result


def source_report():
    base_rows = case_rows(adapted=False)
    adapted_rows = case_rows(adapted=True)
    base = rcsp.fixed_final_summary(base_rows)
    adapted = rcsp.fixed_final_summary(adapted_rows)
    direction_row = {
        "projected_adapter_delta_vs_negative_temporal_gradient_cosine": 0.25,
        "adapted_total_action_vs_negative_temporal_gradient_cosine": 0.1,
    }
    support_row = {"projected_outside_support_max": 0.0}
    return {
        "schema": "refiner_role_conditioned_support_projection_experiment_v1",
        "completed": True,
        "provenance": {"runtime_commit": "source-commit"},
        "fixed_final_64": {
            "BASE": {"summary": base, "case_level": base_rows},
            "RCSP": {"summary": adapted, "case_level": adapted_rows},
        },
        "baseline_comparison": rcsp.baseline_comparison(base, adapted),
        "direction_alignment": {
            "case_level": [direction_row] * 64,
            "summary": scoped_summaries({"cases": 8}),
            "read_only_final_step_400": True,
            "used_for_optimizer_update": False,
        },
        "support_projection_stats": {
            "case_level": [support_row] * 64,
            "summary": scoped_summaries(
                {"cases": 8, "projected_outside_support_max": 0.0}
            ),
        },
        "scientific_answers": {
            "role_conditioned_direction_rescue": "ROLE_CONDITIONING_ALONE_INSUFFICIENT"
        },
        "checkpoint_selection_performed": False,
        "scale_selection_performed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
    }


def test_review_recomputes_measurements_and_fixes_cross_short_only_classification(tmp_path):
    result = review.review_report(
        source_report(),
        source_path=tmp_path / "report.json",
        source_sha256="abc",
        review_runtime_commit="review-commit",
    )
    assert result["measurement_recomputation_verified"]
    assert result["reporting_logic_issue"]["measurements_changed"] is False
    assert result["formal_conclusion"]["observed_temporal_gate_rescue"] == {
        "cross_event_width_10": 4,
        "cross_event_width_28": 0,
        "single_recording_all_widths": 0,
    }
    assert result["formal_conclusion"]["width_dependent_mechanism_remains"]
    assert result["formal_conclusion"]["single_recording_specific_mechanism_remains"]
    assert result["corrected_scientific_answers"][
        "temporal_gate_rescue_width_pattern"
    ] == "WIDTH_10_ONLY"
    assert result["scientific_acceptance"] is False
    assert result["pilot_allowed"] is False


def test_review_fails_closed_on_changed_measurements_or_permissions():
    report = source_report()
    report["fixed_final_64"]["RCSP"]["summary"]["overall"][
        "temporal_gate_pass_cases"
    ] = 99
    with pytest.raises(RuntimeError, match="stored summary"):
        review.review_report(
            report,
            source_path="report.json",
            source_sha256="abc",
            review_runtime_commit="review-commit",
        )

    report = source_report()
    report["pilot_allowed"] = True
    with pytest.raises(ValueError, match="diagnostic-only flags"):
        review.review_report(
            report,
            source_path="report.json",
            source_sha256="abc",
            review_runtime_commit="review-commit",
        )


def test_review_shell_is_read_only_fail_closed_and_forbids_pilot():
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "review_refiner_role_conditioned_support_projection.sh"
    ).read_text(encoding="utf-8")
    assert 'test ! -e "$OUTPUT"' in source
    assert 'git status --porcelain' in source
    assert 'git rev-parse origin/main' in source
    assert "--expected-source-commit" in source
    assert "No checkpoint load" in source
    assert "Pilot remains forbidden" in source
