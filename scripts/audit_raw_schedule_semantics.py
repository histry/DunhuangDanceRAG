#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit raw schedule semantic and continuous-feature diversity."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def extract_slots(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]

    if isinstance(data, dict):
        for key in ("schedule", "slots", "segments"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [
                    row
                    for row in rows
                    if isinstance(row, dict)
                ]

    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail-on-collapse", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    data = json.loads(path.read_text(encoding="utf-8"))
    slots = extract_slots(data)

    events = [
        str(
            slot.get(
                "music_event",
                slot.get(
                    "music_semantic_top_label",
                    slot.get("music_alignment_label"),
                ),
            )
        )
        for slot in slots
    ]

    event_counts = Counter(events)
    unique_events = len(event_counts)

    continuous_fields = (
        "target_motion_density",
        "music_energy",
        "motion_energy",
        "arousal",
        "onset_density",
        "spectral_flux",
    )

    continuous_available = any(
        any(slot.get(field) is not None for field in continuous_fields)
        for slot in slots
    )

    collapsed_hard_labels = bool(
        len(slots) >= 3
        and unique_events <= 1
    )

    insufficient_semantics = bool(
        collapsed_hard_labels
        and not continuous_available
    )

    report = {
        "schema": "raw_schedule_semantic_audit",
        "input": str(path),
        "slot_count": len(slots),
        "event_counts": dict(event_counts),
        "unique_events": unique_events,
        "continuous_semantic_fields_available":
            continuous_available,
        "collapsed_hard_labels": collapsed_hard_labels,
        "insufficient_time_local_semantics":
            insufficient_semantics,
        "interpretation": (
            "All slots share one hard label and no continuous "
            "time-local motion target is available."
            if insufficient_semantics
            else
            "The schedule contains usable semantic variation "
            "or continuous motion targets."
        ),
    }

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")

    if args.fail_on_collapse and insufficient_semantics:
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
