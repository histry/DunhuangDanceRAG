"""Read-only reporting-logic review for a completed RCSP diagnostic.

The v1 RCSP report correctly stored every measurement, but its headline
classification conflated any continuous temporal-deficit decrease with a
temporal gate rescue.  This reviewer recomputes the recorded summaries from
the immutable case rows, separates those two signals, and emits a new review
artifact.  It never loads a checkpoint, runs a model, changes a threshold, or
authorizes Pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training import motion_models as m
from training import refiner_role_conditioned_support_projection_experiment as rcsp


SCHEMA = "refiner_role_conditioned_support_projection_result_review_v1"
SOURCE_SCHEMAS = {
    "refiner_role_conditioned_support_projection_experiment_v1",
    "refiner_role_conditioned_support_projection_experiment_v2",
}
EXPECTED_GROUPS = tuple(
    f"{split}/{role}/{width}"
    for split, role in rcsp.FINAL_BLOCK_ORDER
    for width in (10, 28)
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _assert_false_flags(report):
    fields = (
        "checkpoint_selection_performed",
        "scale_selection_performed",
        "production_model_modified",
        "production_inference_modified",
        "scientific_acceptance",
        "publish_allowed",
        "pilot_allowed",
    )
    bad = {field: report.get(field) for field in fields if report.get(field) is not False}
    if bad:
        raise ValueError(f"RCSP source report violates diagnostic-only flags: {bad}")


def _recompute_measurements(report):
    fixed = report.get("fixed_final_64")
    if not isinstance(fixed, dict):
        raise ValueError("RCSP source report is missing fixed_final_64")
    rows = {}
    summaries = {}
    for label in ("BASE", "RCSP"):
        section = fixed.get(label)
        if not isinstance(section, dict):
            raise ValueError(f"RCSP source report is missing fixed_final_64/{label}")
        rows[label] = section.get("case_level")
        if not isinstance(rows[label], list) or len(rows[label]) != 64:
            raise ValueError(f"{label} fixed-final case rows must contain exactly 64 cases")
        summaries[label] = rcsp.fixed_final_summary(rows[label])
        if summaries[label] != section.get("summary"):
            raise RuntimeError(f"{label} stored summary does not match its immutable case rows")
    comparison = rcsp.baseline_comparison(summaries["BASE"], summaries["RCSP"])
    if comparison != report.get("baseline_comparison"):
        raise RuntimeError("stored BASE/RCSP comparison does not match immutable case rows")
    return rows, summaries, comparison


def _validate_direction_and_support(report):
    direction = report.get("direction_alignment")
    support = report.get("support_projection_stats")
    if not isinstance(direction, dict) or len(direction.get("case_level", ())) != 64:
        raise ValueError("RCSP source report lacks the 64-case direction audit")
    if not direction.get("read_only_final_step_400"):
        raise ValueError("RCSP direction audit is not fixed at final step 400")
    if direction.get("used_for_optimizer_update") is not False:
        raise ValueError("RCSP direction audit was marked as an optimizer input")
    if not isinstance(support, dict) or len(support.get("case_level", ())) != 64:
        raise ValueError("RCSP source report lacks the 64-case support audit")
    summaries = support.get("summary", {})
    if summaries.get("overall", {}).get("projected_outside_support_max") != 0.0:
        raise RuntimeError("RCSP projected action escaped binary decoder support")
    required = {"overall"} | {f"group:{name}" for name in EXPECTED_GROUPS}
    if not required.issubset(direction.get("summary", {})):
        raise ValueError("RCSP direction summary is missing fixed group scopes")
    if not required.issubset(summaries):
        raise ValueError("RCSP support summary is missing fixed group scopes")
    return direction, support


def review_report(report, *, source_path, source_sha256, review_runtime_commit):
    if report.get("schema") not in SOURCE_SCHEMAS:
        raise ValueError(f"unsupported RCSP source schema: {report.get('schema')!r}")
    if report.get("completed") is not True:
        raise ValueError("RCSP source report is not complete")
    _assert_false_flags(report)
    rows, summaries, comparison = _recompute_measurements(report)
    direction, support = _validate_direction_and_support(report)
    corrected = rcsp.scientific_answers(
        {**summaries["BASE"], "case_level": rows["BASE"]},
        {**summaries["RCSP"], "case_level": rows["RCSP"]},
        comparison,
    )
    gate_delta = corrected["temporal_gate_pass_delta_by_group"]
    cross_short_delta = sum(
        value for name, value in gate_delta.items() if "/cross_event/10" in name
    )
    cross_long_delta = sum(
        value for name, value in gate_delta.items() if "/cross_event/28" in name
    )
    single_delta = sum(
        value for name, value in gate_delta.items() if "/single_recording/" in name
    )
    conclusion = {
        "classification": corrected["role_conditioned_direction_rescue"],
        "observed_temporal_gate_rescue": {
            "cross_event_width_10": cross_short_delta,
            "cross_event_width_28": cross_long_delta,
            "single_recording_all_widths": single_delta,
        },
        "role_conditioning_alone_sufficient": (
            corrected["role_conditioned_direction_rescue"]
            == "SUPPORTED_BY_DIAGNOSTIC_EXPERIMENT"
        ),
        "width_dependent_mechanism_remains": (
            corrected["temporal_gate_rescue_width_pattern"]
            in ("WIDTH_10_ONLY", "WIDTH_28_ONLY")
        ),
        "single_recording_specific_mechanism_remains": single_delta == 0,
        "next_evidence": (
            "Review the already-recorded direction and support summaries by role and width; "
            "do not add width heads or run another training experiment until those values are interpreted."
        ),
    }
    return {
        "schema": SCHEMA,
        "completed": True,
        "review_kind": "post_run_reporting_logic_correction",
        "review_runtime_commit": review_runtime_commit,
        "source_report": {
            "path": str(Path(source_path).resolve()),
            "sha256": source_sha256,
            "schema": report["schema"],
            "runtime_commit": report["provenance"]["runtime_commit"],
        },
        "reporting_logic_issue": {
            "description": (
                "The original headline used continuous deficit improvement as a width-rescue "
                "predicate, so small non-gating improvements in both widths hid gate-level width asymmetry."
            ),
            "measurements_changed": False,
            "thresholds_changed": False,
            "model_or_decoder_changed": False,
        },
        "measurement_recomputation_verified": True,
        "baseline_comparison": comparison,
        "original_scientific_answers": report.get("scientific_answers"),
        "corrected_scientific_answers": corrected,
        "formal_conclusion": conclusion,
        "direction_alignment_summary": direction["summary"],
        "support_projection_summary": support["summary"],
        "checkpoint_selection_performed": False,
        "scale_selection_performed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "next_action": "interpret_existing_direction_and_support_evidence_no_pilot",
    }


def run(args):
    report_path = Path(args.report).resolve()
    output_path = Path(args.output).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"RCSP report does not exist: {report_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite review output: {output_path}")
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise RuntimeError(
            f"review runtime commit {runtime_commit} != expected {args.expected_main_commit}"
        )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    source_commit = report.get("provenance", {}).get("runtime_commit")
    if source_commit != args.expected_source_commit:
        raise RuntimeError(
            f"RCSP source commit {source_commit} != expected {args.expected_source_commit}"
        )
    result = review_report(
        report,
        source_path=report_path,
        source_sha256=_file_sha256(report_path),
        review_runtime_commit=runtime_commit,
    )
    _exclusive_json(output_path, result)
    print(json.dumps({"stage": "rcsp_result_review", **result["formal_conclusion"]},
                     ensure_ascii=False, allow_nan=False), flush=True)
    print("DIRECTION ALIGNMENT SUMMARY", flush=True)
    print(json.dumps(result["direction_alignment_summary"], ensure_ascii=False,
                     allow_nan=False), flush=True)
    print("SUPPORT PROJECTION SUMMARY", flush=True)
    print(json.dumps(result["support_projection_summary"], ensure_ascii=False,
                     allow_nan=False), flush=True)
    print(json.dumps({
        "stage": "rcsp_result_review_complete",
        "output": str(output_path),
        "measurements_changed": False,
        "production_model_modified": False,
        "scientific_acceptance": False,
        "pilot_allowed": False,
    }, allow_nan=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
