#!/usr/bin/env python3
"""Compare a preserved whole-song baseline with a no-training regression."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _as_rows(values: Sequence[Any]) -> list[str]:
    return [str(value) for value in values]


def _route_metrics(event_ids: Sequence[Any], source_ids: Sequence[Any]) -> dict[str, Any]:
    events = _as_rows(event_ids)
    sources = _as_rows(source_ids)
    counts = Counter(sources)
    return {
        "num_slots": len(events),
        "unique_events": len(set(events)),
        "unique_ratio": len(set(events)) / max(1, len(events)),
        "adjacent_repeats": sum(a == b for a, b in zip(events, events[1:])),
        "source_counts": dict(counts),
        "max_source_share": max(counts.values(), default=0) / max(1, len(sources)),
    }


def _baseline_sources(db_path: Path, event_indices: Sequence[int]) -> list[str]:
    if not db_path.is_file():
        return ["unknown"] * len(event_indices)
    with np.load(db_path, allow_pickle=True) as db:
        values = np.asarray(db["source_uids"], dtype=object)
        return [str(values[int(index)]) for index in event_indices]


def _improvement(old: Mapping[str, Any], new: Mapping[str, Any], key: str) -> float:
    return float(new[key]) - float(old[key])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_report", required=True)
    parser.add_argument("--baseline_db", required=True)
    parser.add_argument("--baseline_transition_mask")
    parser.add_argument("--candidate_gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline_report).resolve()
    candidate_path = Path(args.candidate_gate).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    event_indices = [int(value) for value in baseline["selected_event_indices_final"]]
    sources = _baseline_sources(Path(args.baseline_db).resolve(), event_indices)
    old_route = _route_metrics(event_indices, sources)
    new_route = dict(candidate["route"])

    mask_metrics: dict[str, float] = {}
    if args.baseline_transition_mask:
        mask_path = Path(args.baseline_transition_mask).resolve()
        if mask_path.is_file():
            mask = np.asarray(np.load(mask_path, allow_pickle=True), dtype=np.float32)
            mask_metrics = {
                "nonzero_frame_fraction": float(np.mean(mask > 0.0)),
                "strong_frame_fraction": float(np.mean(mask > 0.5)),
            }

    old_physical = dict(baseline.get("final_audit", {}))
    new_physical = dict(candidate.get("physical", {}).get("audit", {}))
    comparable_physical = {}
    for key in (
        "foot_skate_mps_mean",
        "foot_skate_mps_p95",
        "foot_skate_mps_max",
        "foot_penetration_min_m",
        "joint_jerk_mps3_p95",
        "joint_jerk_mps3_max",
        "root_y_range_m",
    ):
        if key in old_physical and key in new_physical:
            comparable_physical[key] = {
                "baseline": float(old_physical[key]),
                "candidate": float(new_physical[key]),
                "delta": _improvement(old_physical, new_physical, key),
            }

    result = {
        "schema": "same_wav_old_new_differential_v1",
        "audio": candidate.get("audio"),
        "baseline": {
            "report": str(baseline_path),
            "route": old_route,
            "transition_mask": mask_metrics,
            "physical": old_physical,
        },
        "candidate": {
            "gate": str(candidate_path),
            "ok": bool(candidate.get("ok", False)),
            "route": new_route,
            "physical": new_physical,
        },
        "differential": {
            "slot_count_delta": int(new_route["num_slots"]) - int(old_route["num_slots"]),
            "unique_event_delta": int(new_route["unique_events"]) - int(old_route["unique_events"]),
            "adjacent_repeat_delta": int(new_route["adjacent_repeats"]) - int(old_route["adjacent_repeats"]),
            "max_source_share_delta": float(new_route["max_source_share"]) - float(old_route["max_source_share"]),
            "physical": comparable_physical,
        },
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if candidate.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
