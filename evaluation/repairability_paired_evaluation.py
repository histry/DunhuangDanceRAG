"""Paired same-pool evaluation for baseline versus repairability routing traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from evaluation.gar_evaluation_readiness import GARSelectionTrace


def load_traces(paths: Sequence[str]) -> list[GARSelectionTrace]:
    result: list[GARSelectionTrace] = []
    for value in paths:
        path = Path(value)
        candidates = sorted(path.rglob("*.json")) if path.is_dir() else [path]
        for candidate in candidates:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "gar_selection_trace_v1":
                continue
            result.append(GARSelectionTrace.from_dict(payload))
    if not result:
        raise ValueError("no GAR selection traces were found")
    return result


def boundary_rows(traces: Sequence[GARSelectionTrace]) -> Dict[tuple[str, int, int], Dict[str, Any]]:
    rows: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    for trace in traces:
        seed = -1 if trace.random_seed is None else int(trace.random_seed)
        for boundary in trace.boundaries:
            key = (trace.sequence.sequence_id, int(boundary.slot_index), seed)
            if key in rows:
                raise ValueError(f"duplicate paired evaluation key={key}")
            manifest = {
                entry.candidate_id: entry for entry in boundary.candidate_pool_manifest
            }
            final_id = boundary.summary.final_candidate_id
            selected_entry = manifest.get(final_id) if final_id is not None else None
            rows[key] = {
                "candidate_pool_fingerprint": boundary.candidate_pool_fingerprint,
                "generator": (
                    boundary.generator_id,
                    boundary.generator_version,
                    boundary.generator_checkpoint_fingerprint,
                    boundary.generator_config_fingerprint,
                ),
                "repair": (
                    boundary.repair_operator_id,
                    boundary.repair_operator_version,
                    boundary.repair_config_fingerprint,
                ),
                "initial_post_safe": boundary.summary.initial_post_safe,
                "final_post_safe": boundary.summary.final_post_safe,
                "reselection_count": int(boundary.summary.reselection_count),
                "final_candidate_rank": boundary.summary.final_candidate_rank,
                "selected_router_score": (
                    None if selected_entry is None else selected_entry.retrieval_score
                ),
                "method_variant_id": boundary.method_variant_id,
            }
    return rows


def finite_mean(values: Sequence[Any]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return None if not clean else float(np.mean(clean))


def aggregate(rows: Sequence[Mapping[str, Any]], traces: Sequence[GARSelectionTrace]) -> Dict[str, Any]:
    final_safe = [row["final_post_safe"] for row in rows if row["final_post_safe"] is not None]
    initial_safe = [row["initial_post_safe"] for row in rows if row["initial_post_safe"] is not None]
    sequence_runtime = [trace.sequence.sequence_runtime_ms for trace in traces]
    return {
        "boundaries": len(rows),
        "initial_post_safe_rate": finite_mean(initial_safe),
        "final_post_safe_rate": finite_mean(final_safe),
        "unsafe_boundary_rate": (
            None if not final_safe else 1.0 - float(np.mean(final_safe))
        ),
        "mean_reselection_count": finite_mean(
            [row["reselection_count"] for row in rows]
        ),
        "mean_final_candidate_rank": finite_mean(
            [row["final_candidate_rank"] for row in rows]
        ),
        "mean_selected_router_score": finite_mean(
            [row["selected_router_score"] for row in rows]
        ),
        "mean_sequence_runtime_ms": finite_mean(sequence_runtime),
    }


def paired_bootstrap(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> Dict[str, Any]:
    if baseline.shape != method.shape or baseline.ndim != 1:
        raise ValueError("paired bootstrap arrays must be aligned vectors")
    if len(baseline) == 0:
        return {"mean_delta": None, "ci95": None}
    rng = np.random.default_rng(int(seed))
    delta = method - baseline
    draws = np.empty((int(samples),), dtype=np.float64)
    for index in range(int(samples)):
        sampled = rng.integers(0, len(delta), size=len(delta))
        draws[index] = float(delta[sampled].mean())
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Paired repairability trace evaluation")
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--method", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ubr-increase", type=float, default=0.0)
    parser.add_argument("--max-router-score-drop", type=float, default=0.02)
    args = parser.parse_args(argv)
    baseline_traces = load_traces(args.baseline)
    method_traces = load_traces(args.method)
    baseline = boundary_rows(baseline_traces)
    method = boundary_rows(method_traces)
    if set(baseline) != set(method):
        raise ValueError("paired trace case/seed sets do not match")
    keys = sorted(baseline)
    for key in keys:
        left, right = baseline[key], method[key]
        if left["candidate_pool_fingerprint"] != right["candidate_pool_fingerprint"]:
            raise ValueError(f"candidate pool mismatch for {key}")
        if left["generator"] != right["generator"]:
            raise ValueError(f"generator mismatch for {key}")
        if left["repair"] != right["repair"]:
            raise ValueError(f"repair operator mismatch for {key}")
    baseline_rows = [baseline[key] for key in keys]
    method_rows = [method[key] for key in keys]
    baseline_metrics = aggregate(baseline_rows, baseline_traces)
    method_metrics = aggregate(method_rows, method_traces)
    required_metrics = (
        "initial_post_safe_rate",
        "final_post_safe_rate",
        "unsafe_boundary_rate",
        "mean_reselection_count",
        "mean_selected_router_score",
    )
    for metric_name in required_metrics:
        if baseline_metrics[metric_name] is None or method_metrics[metric_name] is None:
            raise ValueError(f"paired evaluation is missing required metric {metric_name}")
    paired: Dict[str, Any] = {}
    for name in ("initial_post_safe", "final_post_safe", "reselection_count"):
        sequence_pairs: Dict[str, list[tuple[float, float]]] = {}
        for key, left, right in zip(keys, baseline_rows, method_rows):
            if left[name] is None or right[name] is None:
                continue
            sequence_pairs.setdefault(key[0], []).append(
                (float(left[name]), float(right[name]))
            )
        pairs = [
            (
                float(np.mean([value[0] for value in values])),
                float(np.mean([value[1] for value in values])),
            )
            for values in sequence_pairs.values()
        ]
        paired[name] = paired_bootstrap(
            np.asarray([float(pair[0]) for pair in pairs]),
            np.asarray([float(pair[1]) for pair in pairs]),
            seed=args.seed,
            samples=args.bootstrap_samples,
        )
        paired[name]["bootstrap_unit"] = "sequence"
    ubr_increase = float(
        method_metrics["unsafe_boundary_rate"]
        - baseline_metrics["unsafe_boundary_rate"]
    )
    router_drop = float(
        baseline_metrics["mean_selected_router_score"]
        - method_metrics["mean_selected_router_score"]
    )
    report = {
        "schema": "repairability_paired_evaluation_v1",
        "same_pool_generator_guard": True,
        "baseline": baseline_metrics,
        "method": method_metrics,
        "paired_bootstrap": paired,
        "gate": {
            "unsafe_boundary_rate_increase": ubr_increase,
            "max_allowed_unsafe_boundary_rate_increase": args.max_ubr_increase,
            "selected_router_score_drop": router_drop,
            "max_allowed_router_score_drop": args.max_router_score_drop,
            "passed": bool(
                ubr_increase <= args.max_ubr_increase
                and router_drop <= args.max_router_score_drop
                and method_metrics["initial_post_safe_rate"]
                >= baseline_metrics["initial_post_safe_rate"]
                and method_metrics["mean_reselection_count"]
                <= baseline_metrics["mean_reselection_count"]
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
