"""Create paper tables from saved mechanism JSONL and phase indices."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _audit_for_case(variant, uid):
    for row in variant.get("exact_audits", []):
        if str(row.get("case_uid")) == str(uid):
            return row
    raise RuntimeError(f"missing exact audit for {uid}")


def _flatten_mechanism(path):
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = json.loads(line)
        for radius in candidate.get("radius_rows", []):
            exact = radius.get("exact_path") or {}
            transition = bool(exact.get("internal_witness_transition"))
            ablation = radius.get(
                "no_geodesic_acceleration_ablation"
            ) or {}
            for name, row in (exact.get("rows") or {}).items():
                ablation_row = (ablation.get("rows") or {}).get(name)
                result.append({
                    "candidate_id": candidate["candidate_id"],
                    "candidate_source": candidate["candidate_source"],
                    "case_uid": candidate["case_uid"],
                    "budget": candidate["budget"],
                    "theta_radians": candidate["theta_radians"],
                    "radius_rms": radius["radius_rms"],
                    "metric_row": name,
                    "witness_transition": transition,
                    "stable_witness": not transition,
                    "E1": row["first_order_absolute_error"],
                    "E2": row["second_order_absolute_error"],
                    "normalized_E1": row["first_order_normalized_error"],
                    "normalized_E2": row["second_order_normalized_error"],
                    "normalized_E1_minus_E2": row[
                        "normalized_error_improvement"
                    ],
                    "log_E2_over_E1": row[
                        "log_second_over_first_error"
                    ],
                    "E2_less_than_E1": row[
                        "second_order_more_accurate"
                    ],
                    "no_geodesic_acceleration_normalized_E2": (
                        ablation_row["second_order_normalized_error"]
                        if ablation_row else None
                    ),
                    "full_path_E2_improvement_over_no_acceleration": (
                        ablation_row["second_order_normalized_error"]
                        - row["second_order_normalized_error"]
                        if ablation_row else None
                    ),
                })
    return result


def _mechanism_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["radius_rms"], row["stable_witness"])].append(row)
    summary = []
    for (radius, stable), group in sorted(groups.items()):
        e1 = [float(row["normalized_E1"]) for row in group]
        e2 = [float(row["normalized_E2"]) for row in group]
        improvement = [
            float(row["normalized_E1_minus_E2"]) for row in group
        ]
        log_ratio = [float(row["log_E2_over_E1"]) for row in group]
        summary.append({
            "radius_rms": radius,
            "stable_witness": stable,
            "row_count": len(group),
            "candidate_count": len({row["candidate_id"] for row in group}),
            "median_normalized_E1": statistics.median(e1),
            "median_normalized_E2": statistics.median(e2),
            "median_normalized_E1_minus_E2": statistics.median(improvement),
            "median_log_E2_over_E1": statistics.median(log_ratio),
            "probability_E2_less_than_E1": sum(
                bool(row["E2_less_than_E1"]) for row in group
            ) / len(group),
        })
    return summary


def _closure_rows(indices):
    rows = []
    adapter_seen = set()
    for index_path in indices:
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        phase = index["binding"]["phase"]
        for job in index["jobs"]:
            report = json.loads(
                Path(job["report"]).read_text(encoding="utf-8")
            )
            uid = job["case_uid"]
            method = job["method"]
            budget = int(job["budget"])
            variant_name = f"geodesic_joint_sqp_k{budget}"
            variant = report["variants"][variant_name]
            audit = _audit_for_case(variant, uid)
            correction = variant["correction_by_case"][uid]
            timing = correction.get("stage_timing_seconds") or {}
            counts = correction.get("function_call_counts") or {}
            iteration_rows = correction.get("history") or []
            angle_rows = [
                angle
                for iteration in iteration_rows
                for angle in (
                    iteration.get("angle_specific_second_order_solvers") or []
                )
            ]
            rows.append({
                "phase": phase,
                "case_uid": uid,
                "method": method,
                "budget": budget,
                "raw_guard_passed": bool(audit["raw_audit"]["passed"]),
                "projected_closure": bool(
                    audit.get("effective_projected_candidate")
                ),
                "numeric_failure": bool(correction.get("numeric_failure")),
                "accepted_steps": correction.get("accepted_steps"),
                "second_order_state": correction.get("second_order_state"),
                "constraint_generation_budget_exhausted": bool(
                    correction.get("constraint_generation_budget_exhausted")
                ),
                "maximum_constraint_generation_depth_observed": max(
                    (
                        int(row.get("constraint_generation_depth", 0))
                        for row in iteration_rows
                    ),
                    default=0,
                ),
                "adaptive_full_search_execution_count": sum(
                    bool(row.get("adaptive_full_search_executed"))
                    for row in angle_rows
                ),
                "runtime_seconds": correction.get("elapsed_seconds"),
                **{f"time_{key}": value for key, value in timing.items()},
                **{f"calls_{key}": value for key, value in counts.items()},
            })
            adapter_key = (phase, uid)
            if adapter_key not in adapter_seen:
                adapter = _audit_for_case(report["variants"]["adapter"], uid)
                rows.append({
                    "phase": phase,
                    "case_uid": uid,
                    "method": "adapter",
                    "budget": 0,
                    "raw_guard_passed": bool(
                        adapter["raw_audit"]["passed"]
                    ),
                    "projected_closure": bool(
                        adapter.get("effective_projected_candidate")
                    ),
                    "numeric_failure": False,
                    "accepted_steps": 0,
                    "second_order_state": None,
                    "runtime_seconds": None,
                })
                adapter_seen.add(adapter_key)
    return rows


def _closure_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["method"], row["budget"])].append(row)
    result = []
    for (phase, method, budget), group in sorted(groups.items()):
        runtime = [
            float(row["runtime_seconds"])
            for row in group
            if row.get("runtime_seconds") is not None
            and math.isfinite(float(row["runtime_seconds"]))
        ]
        result.append({
            "phase": phase,
            "method": method,
            "budget": budget,
            "case_count": len(group),
            "closure_count": sum(row["projected_closure"] for row in group),
            "closure_rate": sum(
                row["projected_closure"] for row in group
            ) / len(group),
            "numeric_failure_count": sum(
                row["numeric_failure"] for row in group
            ),
            "constraint_generation_budget_exhausted_count": sum(
                bool(row.get("constraint_generation_budget_exhausted"))
                for row in group
            ),
            "adaptive_full_search_execution_count": sum(
                int(row.get("adaptive_full_search_execution_count", 0))
                for row in group
            ),
            "median_runtime_seconds": (
                statistics.median(runtime) if runtime else None
            ),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-index", action="append", default=[])
    parser.add_argument("--mechanism-jsonl", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not args.phase_index and not args.mechanism_jsonl:
        parser.error("provide a phase index and/or mechanism JSONL")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mechanism_rows = [
        row
        for path in args.mechanism_jsonl
        for row in _flatten_mechanism(path)
    ]
    mechanism_summary = _mechanism_summary(mechanism_rows)
    closure_rows = _closure_rows(args.phase_index)
    closure_summary = _closure_summary(closure_rows)
    if mechanism_rows:
        _write_csv(output / "matched_candidate_rows.csv", mechanism_rows)
        _write_csv(output / "matched_candidate_summary.csv", mechanism_summary)
    if closure_rows:
        _write_csv(output / "closure_case_rows.csv", closure_rows)
        _write_csv(output / "closure_summary.csv", closure_summary)
    summary = {
        "schema": "paper2_result_summary_v1",
        "mechanism": mechanism_summary,
        "closure": closure_summary,
        "claims_are_hypotheses_until_sealed_evidence": True,
        "raw_model_accuracy_is_not_authoritative_safety": True,
    }
    (output / "paper2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "paper2_summary_complete",
        "output": str(output / "paper2_summary.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
