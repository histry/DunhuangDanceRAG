#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace music-semantic fields through schedule construction stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


SEMANTIC_FIELDS = (
    "slot_id",
    "slot",
    "start",
    "end",
    "start_sec",
    "end_sec",
    "start_frame",
    "end_frame",
    "music_start",
    "music_end",
    "target_frames",
    "music_length",
    "role",
    "slot_role",
    "phrase_role",
    "label",
    "top_label",
    "music_event",
    "motion_event",
    "music_alignment_label",
    "music_semantic_top_label",
    "semantic_probability_source",
    "external_music_semantic_source",
    "slot_source",
    "slot_plan_source",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_slots(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [dict(row) for row in obj if isinstance(row, Mapping)]

    if not isinstance(obj, Mapping):
        return []

    for key in ("slots", "schedule", "segments", "descriptors"):
        rows = obj.get(key)
        if isinstance(rows, list):
            return [
                dict(row)
                for row in rows
                if isinstance(row, Mapping)
            ]

    stage_reports = obj.get("stage_reports")
    if isinstance(stage_reports, Mapping):
        retrieval = stage_reports.get("retrieval")
        if isinstance(retrieval, list):
            rows = []
            for index, row in enumerate(retrieval):
                if not isinstance(row, Mapping):
                    continue
                rows.append({
                    "slot": row.get("slot", index),
                    "music_alignment_label": row.get(
                        "slot_music_alignment_label",
                        row.get("music_alignment_label"),
                    ),
                    "music_semantic_top_label": row.get(
                        "slot_music_semantic_top_label",
                        row.get("music_semantic_top_label"),
                    ),
                    "music_semantic_probs": row.get(
                        "slot_music_semantic_probs",
                        row.get("music_semantic_probs", {}),
                    ),
                    "role": row.get(
                        "slot_role",
                        row.get("role"),
                    ),
                })
            if rows:
                return rows

    return []


def canonical_slot_index(slot: Mapping[str, Any], fallback: int) -> int:
    for key in ("slot_id", "slot", "index"):
        try:
            return int(slot[key])
        except Exception:
            continue
    return int(fallback)


def top_label(slot: Mapping[str, Any]) -> Optional[str]:
    for key in (
        "music_semantic_top_label",
        "music_alignment_label",
        "top_label",
        "label",
        "music_event",
        "motion_event",
    ):
        value = slot.get(key)
        if value is not None and str(value).strip():
            return str(value)

    probs = slot.get("music_semantic_probs")
    if isinstance(probs, Mapping) and probs:
        valid = {}
        for key, value in probs.items():
            try:
                valid[str(key)] = float(value)
            except Exception:
                continue
        if valid:
            return max(valid, key=valid.get)

    return None


def probability_summary(slot: Mapping[str, Any]) -> Dict[str, Any]:
    probs = slot.get("music_semantic_probs")
    if not isinstance(probs, Mapping):
        return {
            "support": 0,
            "entropy": None,
            "top_probability": None,
            "probabilities": None,
        }

    cleaned = {}
    for key, value in probs.items():
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number) and number >= 0.0:
            cleaned[str(key)] = number

    total = sum(cleaned.values())
    if total > 0.0:
        normalized = {
            key: value / total
            for key, value in cleaned.items()
        }
    else:
        normalized = cleaned

    support = sum(value > 1.0e-6 for value in normalized.values())
    entropy = (
        -sum(
            value * np.log(max(value, 1.0e-12))
            for value in normalized.values()
        )
        if normalized
        else None
    )
    top_probability = max(normalized.values()) if normalized else None

    return {
        "support": int(support),
        "entropy": None if entropy is None else float(entropy),
        "top_probability": (
            None
            if top_probability is None
            else float(top_probability)
        ),
        "probabilities": normalized or None,
    }


def stage_rows(name: str, path: Path) -> List[Dict[str, Any]]:
    obj = load_json(path)
    slots = extract_slots(obj)
    output = []

    for fallback, slot in enumerate(slots):
        probability = probability_summary(slot)
        output.append({
            "stage": name,
            "slot": canonical_slot_index(slot, fallback),
            "top_label": top_label(slot),
            "role": slot.get("role", slot.get("slot_role")),
            "source": slot.get(
                "semantic_probability_source",
                slot.get(
                    "external_music_semantic_source",
                    slot.get("slot_source"),
                ),
            ),
            "support": probability["support"],
            "entropy": probability["entropy"],
            "top_probability": probability["top_probability"],
            "probabilities": probability["probabilities"],
            "raw_fields": {
                key: slot.get(key)
                for key in SEMANTIC_FIELDS
                if key in slot
            },
        })

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--unstamped", required=True)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--final-report", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    stages = [
        ("raw_v26", Path(args.raw)),
        ("descriptor_unstamped", Path(args.unstamped)),
        ("mixed_grounding", Path(args.mixed)),
        ("final_report", Path(args.final_report)),
    ]

    all_rows = {}
    for name, path in stages:
        if not path.is_file():
            raise FileNotFoundError(path)
        all_rows[name] = stage_rows(name, path)

    maximum_slots = max(
        (len(rows) for rows in all_rows.values()),
        default=0,
    )

    print(
        f"{'slot':>4} "
        f"{'raw_v26':>20} "
        f"{'unstamped':>20} "
        f"{'mixed':>20} "
        f"{'final':>20}"
    )
    print("-" * 92)

    stage_names = [
        "raw_v26",
        "descriptor_unstamped",
        "mixed_grounding",
        "final_report",
    ]

    for slot_index in range(maximum_slots):
        labels = []

        for name in stage_names:
            rows = all_rows[name]
            match = next(
                (
                    row
                    for row in rows
                    if int(row["slot"]) == slot_index
                ),
                None,
            )
            labels.append(
                "-"
                if match is None
                else str(match.get("top_label") or "None")
            )

        print(
            f"{slot_index:4d} "
            f"{labels[0]:>20.20} "
            f"{labels[1]:>20.20} "
            f"{labels[2]:>20.20} "
            f"{labels[3]:>20.20}"
        )

    summary = {}

    for name, rows in all_rows.items():
        labels = [row.get("top_label") for row in rows]
        roles = [row.get("role") for row in rows]

        unique_labels = sorted({
            str(value)
            for value in labels
            if value is not None
        })
        unique_roles = sorted({
            str(value)
            for value in roles
            if value is not None
        })
        unique_vectors = {
            json.dumps(
                row.get("probabilities"),
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in rows
        }

        summary[name] = {
            "slot_count": len(rows),
            "unique_top_labels": unique_labels,
            "unique_roles": unique_roles,
            "unique_probability_vectors": len(unique_vectors),
            "rows": rows,
        }

    first_collapsed_stage = None
    for name in stage_names:
        stage = summary[name]
        if (
            stage["slot_count"] >= 3
            and len(stage["unique_top_labels"]) <= 1
            and stage["unique_probability_vectors"] <= 1
        ):
            first_collapsed_stage = name
            break

    report = {
        "schema": "music_semantic_provenance_trace",
        "first_collapsed_stage": first_collapsed_stage,
        "interpretation": (
            "No semantic collapse detected in the supplied stages."
            if first_collapsed_stage is None
            else
            "The first stage with identical time-local semantic "
            f"descriptions is {first_collapsed_stage}."
        ),
        "stages": summary,
    }

    print()
    print(json.dumps(
        {
            "first_collapsed_stage": first_collapsed_stage,
            "interpretation": report["interpretation"],
            "stage_summary": {
                key: {
                    "slot_count": value["slot_count"],
                    "unique_top_labels": value["unique_top_labels"],
                    "unique_roles": value["unique_roles"],
                    "unique_probability_vectors":
                        value["unique_probability_vectors"],
                }
                for key, value in summary.items()
            },
        },
        ensure_ascii=False,
        indent=2,
    ))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
