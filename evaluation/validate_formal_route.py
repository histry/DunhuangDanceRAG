#!/usr/bin/env python3
"""Fail-closed acceptance gate for a formal Fisher-Rao Graph-SB run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def validate_formal_route_report(report: Mapping[str, Any]) -> dict[str, Any]:
    route = report.get("graph_route_graph_sb_route")
    if not isinstance(route, Mapping):
        route = report.get("event_geometry_global_route")
    errors: list[str] = []
    if not isinstance(route, Mapping):
        errors.append("missing Graph-SB route report")
        route = {}
    if route.get("solver") != "fisher_rao_graph_sb":
        errors.append(f"solver={route.get('solver')!r}")
    if bool(route.get("fallback_used", False)):
        errors.append("fallback_used=true")
    if route.get("unary_semantic_contract") != "ctsr_candidate_probability":
        errors.append(
            f"unary_semantic_contract={route.get('unary_semantic_contract')!r}"
        )
    trace = route.get("trace")
    if not isinstance(trace, list) or not trace:
        errors.append("missing Graph-SB unary trace")
    else:
        invalid_rows = []
        for slot_row in trace:
            candidates = (
                slot_row.get("candidates", [])
                if isinstance(slot_row, Mapping)
                else []
            )
            for candidate in candidates:
                if (
                    not isinstance(candidate, Mapping)
                    or candidate.get("association_source")
                    != "ctsr_candidate_probability"
                ):
                    invalid_rows.append(candidate)
        if invalid_rows:
            errors.append("Graph-SB unary trace contains non-CTSR association")
    schrodinger = route.get("schrodinger")
    if not isinstance(schrodinger, Mapping):
        errors.append("missing schrodinger convergence report")
    elif not bool(schrodinger.get("converged", False)):
        errors.append("schrodinger.converged=false")
    slots = report.get("slots")
    if not isinstance(slots, list) or not slots:
        errors.append("missing formal CTSR slots")
        slots = []
    for index, slot in enumerate(slots):
        if not isinstance(slot, Mapping):
            errors.append(f"slot[{index}] is not a mapping")
            continue
        if slot.get("router_architecture") != "ctsr_weak_temporal_v1":
            errors.append(f"slot[{index}].router_architecture={slot.get('router_architecture')!r}")
        if slot.get("formal_candidate_contract") != "ctsr_weak_scheduler_siblings_v1":
            errors.append(f"slot[{index}] missing formal candidate contract")
        if bool(slot.get("router_compatibility_is_ground_truth", True)):
            errors.append(f"slot[{index}] Router evidence is declared ground truth")
    stages = report.get("stage_reports")
    if not isinstance(stages, Mapping):
        errors.append("missing stage_reports")
        stages = {}
    retrieval = stages.get("retrieval")
    if not isinstance(retrieval, list) or len(retrieval) != len(slots):
        errors.append("formal retrieval report/slot count mismatch")
        retrieval = []
    selected = report.get("selected_event_indices_final")
    if not isinstance(selected, list) or len(selected) != len(slots):
        errors.append("final selected event/slot count mismatch")
        selected = []
    for index, row in enumerate(retrieval):
        if not isinstance(row, Mapping):
            errors.append(f"retrieval[{index}] is not a mapping")
            continue
        if row.get("routing_policy") != "formal_ctsr_scheduler_locked_candidates":
            errors.append(f"retrieval[{index}] used a non-formal routing policy")
        candidate_indices = row.get("candidate_event_indices")
        if (
            index < len(selected)
            and (
                not isinstance(candidate_indices, list)
                or int(selected[index]) not in {int(value) for value in candidate_indices}
            )
        ):
            errors.append(f"selected event {index} is outside the formal candidate set")
    conditioning = stages.get("neural_music_conditioning")
    if not isinstance(conditioning, Mapping):
        errors.append("missing neural conditioning report")
    elif (
        conditioning.get("conditioning_contract")
        != "selected_event_motion_descriptor_v1"
        or bool(conditioning.get("categorical_music_label_used_as_body_semantics", True))
    ):
        errors.append("formal neural repair used a non-event-descriptor condition")
    result = {
        "schema": "formal_graph_sb_acceptance_v1",
        "ok": not errors,
        "solver": route.get("solver"),
        "fallback_used": bool(route.get("fallback_used", False)),
        "converged": bool(
            isinstance(schrodinger, Mapping) and schrodinger.get("converged", False)
        ),
        "ctsr_slots": int(len(slots)),
        "candidate_contract": "ctsr_weak_scheduler_siblings_v1",
        "errors": errors,
    }
    if errors:
        raise RuntimeError(f"Formal Graph-SB acceptance failed: {errors}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        result = validate_formal_route_report(report)
    except RuntimeError as exc:
        result = {
            "schema": "formal_graph_sb_acceptance_v1",
            "ok": False,
            "report": str(report_path),
            "error": str(exc),
        }
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2
    result["report"] = str(report_path)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
