"""Create paper tables from saved mechanism JSONL and phase indices."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        if candidate.get("schema") != (
            "paper2_matched_candidate_curvature_audit_v1"
        ):
            raise RuntimeError(f"unexpected mechanism schema in {path}")
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
                    "binding_sha256": candidate["binding_sha256"],
                    "candidate_source": candidate["candidate_source"],
                    "case_uid": candidate["case_uid"],
                    "budget": candidate["budget"],
                    "theta_radians": candidate["theta_radians"],
                    "constraint_generation_depth": candidate[
                        "constraint_generation_depth"
                    ],
                    "direction_sha256": candidate["direction_sha256"],
                    "support_sha256": candidate["support_sha256"],
                    "witness_sha256": candidate["witness_sha256"],
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


def _trial_mechanism_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["candidate_id"], row["radius_rms"])].append(row)
    result = []
    for (candidate_id, radius), group in sorted(grouped.items()):
        stable_values = {bool(row["stable_witness"]) for row in group}
        if len(stable_values) != 1:
            raise RuntimeError(f"mixed witness state in trial {candidate_id}")
        common = {
            "candidate_id": candidate_id,
            "candidate_source": group[0]["candidate_source"],
            "case_uid": group[0]["case_uid"],
            "budget": group[0]["budget"],
            "theta_radians": group[0]["theta_radians"],
            "constraint_generation_depth": group[0][
                "constraint_generation_depth"
            ],
            "direction_sha256": group[0]["direction_sha256"],
            "support_sha256": group[0]["support_sha256"],
            "witness_sha256": group[0]["witness_sha256"],
            "radius_rms": radius,
            "stable_witness": stable_values.pop(),
            "metric_row_count": len(group),
        }
        first = [float(row["normalized_E1"]) for row in group]
        second = [float(row["normalized_E2"]) for row in group]
        for aggregation, reducer in (
            ("median", statistics.median),
            ("worst", max),
        ):
            e1 = float(reducer(first))
            e2 = float(reducer(second))
            result.append({
                **common,
                "row_aggregation": aggregation,
                "normalized_E1": e1,
                "normalized_E2": e2,
                "normalized_E1_minus_E2": e1 - e2,
                "log_E2_over_E1": math.log(
                    (e2 + 1.0e-15) / (e1 + 1.0e-15)
                ),
                "E2_less_than_E1": bool(e2 < e1),
            })
    return result


def _bootstrap_mean_ci(values, key, replicates=2000):
    if not values:
        return None, None
    seed = int(
        hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16], 16
    )
    generator = random.Random(seed)
    count = len(values)
    samples = sorted(
        statistics.fmean(
            values[generator.randrange(count)] for _ in range(count)
        )
        for _ in range(int(replicates))
    )
    lower = samples[int(0.025 * (len(samples) - 1))]
    upper = samples[int(0.975 * (len(samples) - 1))]
    return float(lower), float(upper)


def _paired_inference(group, key):
    e1 = [float(row["normalized_E1"]) for row in group]
    e2 = [float(row["normalized_E2"]) for row in group]
    delta = [left - right for left, right in zip(e1, e2)]
    ci_low, ci_high = _bootstrap_mean_ci(delta, key)
    result = {
        "paired_trial_count": len(delta),
        "bootstrap_mean_delta_ci95_low": ci_low,
        "bootstrap_mean_delta_ci95_high": ci_high,
        "bootstrap_replicates": 2000,
        "normality_test": None,
        "normality_p_value": None,
        "paired_test": None,
        "paired_test_statistic": None,
        "paired_test_p_value": None,
        "effect_name": None,
        "effect_value": None,
    }
    if len(delta) < 2:
        return result
    normality_p = None
    if len(delta) >= 3 and len(set(delta)) > 1:
        normality_p = float(stats.shapiro(delta).pvalue)
        result["normality_test"] = "shapiro"
        result["normality_p_value"] = normality_p
    approximately_normal = bool(
        len(delta) >= 8
        and normality_p is not None
        and normality_p >= 0.05
    )
    if approximately_normal:
        test = stats.ttest_rel(e1, e2)
        deviation = statistics.stdev(delta)
        result.update({
            "paired_test": "paired_t",
            "paired_test_statistic": float(test.statistic),
            "paired_test_p_value": float(test.pvalue),
            "effect_name": "cohen_dz",
            "effect_value": (
                float(statistics.fmean(delta) / deviation)
                if deviation > 0.0 else 0.0
            ),
        })
        return result
    nonzero = [value for value in delta if value != 0.0]
    if not nonzero:
        statistic = 0.0
        p_value = 1.0
        effect = 0.0
    else:
        test = stats.wilcoxon(nonzero, alternative="two-sided", method="auto")
        ranks = stats.rankdata([abs(value) for value in nonzero])
        rank_total = float(sum(ranks))
        effect = float(sum(
            rank if value > 0.0 else -rank
            for value, rank in zip(nonzero, ranks)
        ) / rank_total)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    result.update({
        "paired_test": "wilcoxon_signed_rank",
        "paired_test_statistic": statistic,
        "paired_test_p_value": p_value,
        "effect_name": "rank_biserial_correlation",
        "effect_value": effect,
    })
    return result


def _holm_adjust(rows):
    families = defaultdict(list)
    for row in rows:
        if row["paired_test_p_value"] is not None:
            family_key = (
                row["row_aggregation"], row["stable_witness"]
            )
            families[family_key].append(row)
    for family in families.values():
        ordered = sorted(family, key=lambda row: row["paired_test_p_value"])
        running = 0.0
        total = len(ordered)
        for index, row in enumerate(ordered):
            running = max(
                running,
                (total - index) * float(row["paired_test_p_value"]),
            )
            row["holm_adjusted_p_value"] = min(1.0, running)
    return rows


def _mechanism_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(
            row["radius_rms"],
            row["stable_witness"],
            row["row_aggregation"],
        )].append(row)
    summary = []
    for (radius, stable, aggregation), group in sorted(groups.items()):
        e1 = [float(row["normalized_E1"]) for row in group]
        e2 = [float(row["normalized_E2"]) for row in group]
        improvement = [
            float(row["normalized_E1_minus_E2"]) for row in group
        ]
        log_ratio = [float(row["log_E2_over_E1"]) for row in group]
        summary.append({
            "radius_rms": radius,
            "stable_witness": stable,
            "row_aggregation": aggregation,
            "trial_count": len(group),
            "median_normalized_E1": statistics.median(e1),
            "median_normalized_E2": statistics.median(e2),
            "median_normalized_E1_minus_E2": statistics.median(improvement),
            "median_log_E2_over_E1": statistics.median(log_ratio),
            "probability_E2_less_than_E1": sum(
                bool(row["E2_less_than_E1"]) for row in group
            ) / len(group),
            **_paired_inference(
                group, (radius, stable, aggregation)
            ),
        })
    return _holm_adjust(summary)


def _closure_rows(indices):
    rows = []
    adapter_seen = set()
    expected_lineage = None
    for index_path in indices:
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if index.get("schema") != "paper2_phase_index_v1":
            raise RuntimeError(f"unexpected phase-index schema in {index_path}")
        binding = index["binding"]
        lineage = (
            binding["implementation_commit"],
            binding["protocol_sha256"],
        )
        if expected_lineage is None:
            expected_lineage = lineage
        elif lineage != expected_lineage:
            raise RuntimeError("phase indices do not share one frozen lineage")
        phase = binding["phase"]
        for job in index["jobs"]:
            if job.get("binding_sha256") != binding["binding_sha256"]:
                raise RuntimeError("phase-index job binding mismatch")
            report_path = Path(job["report"])
            if job.get("report_sha256") != _file_sha(report_path):
                raise RuntimeError("phase-index report SHA256 mismatch")
            report = json.loads(report_path.read_text(encoding="utf-8"))
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
    mechanism_bindings = {
        row["binding_sha256"] for row in mechanism_rows
    }
    if len(mechanism_bindings) > 1:
        raise RuntimeError("mechanism JSONL files do not share one binding")
    mechanism_trial_rows = _trial_mechanism_rows(mechanism_rows)
    mechanism_summary = _mechanism_summary(mechanism_trial_rows)
    closure_rows = _closure_rows(args.phase_index)
    closure_summary = _closure_summary(closure_rows)
    if mechanism_rows:
        _write_csv(output / "matched_candidate_rows.csv", mechanism_rows)
        _write_csv(output / "matched_trial_rows.csv", mechanism_trial_rows)
        _write_csv(output / "matched_candidate_summary.csv", mechanism_summary)
    if closure_rows:
        _write_csv(output / "closure_case_rows.csv", closure_rows)
        _write_csv(output / "closure_summary.csv", closure_summary)
    summary = {
        "schema": "paper2_result_summary_v1",
        "mechanism": mechanism_summary,
        "mechanism_primary_unit": "candidate_trial_after_row_aggregation",
        "mechanism_row_aggregations": ["median", "worst"],
        "multiple_comparison_correction": (
            "holm_within_aggregation_and_witness_stratum"
        ),
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
