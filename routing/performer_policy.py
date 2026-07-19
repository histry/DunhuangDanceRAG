#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whole-song performer-group policy for Event-RAG candidate routing."""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

VALID_GROUPS = {"female", "male", "mixed", "auto"}


def _env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _db_groups(db: Mapping[str, Any]) -> np.ndarray:
    for key in ("performer_groups", "genders"):
        if key in db:
            values = np.asarray(db[key], dtype=object)
            return np.asarray([str(x).strip().lower() for x in values], dtype=object)
    return np.asarray(["unknown"] * len(np.asarray(db["paths"])), dtype=object)


def _quality(db: Mapping[str, Any], event_id: int) -> float:
    for key in ("v46_53_combined_quality", "event_quality_scores", "anatomy_quality"):
        try:
            return float(np.asarray(db[key])[int(event_id)])
        except Exception:
            pass
    return 0.5


def _auto_group(candidate_lists: Sequence[Sequence[int]], db: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    groups = _db_groups(db)
    scores: Dict[str, list[float]] = {"female": [], "male": []}
    missing: Dict[str, int] = {"female": 0, "male": 0}
    for candidates in candidate_lists:
        for group in ("female", "male"):
            ids = [int(e) for e in candidates if 0 <= int(e) < len(groups) and groups[int(e)] == group]
            if ids:
                scores[group].append(max(_quality(db, e) for e in ids))
            else:
                missing[group] += 1
    summary = {
        group: {
            "mean_best_quality": float(np.mean(scores[group])) if scores[group] else -1.0,
            "missing_slots": int(missing[group]),
        }
        for group in ("female", "male")
    }
    chosen = min(
        ("female", "male"),
        key=lambda g: (
            summary[g]["missing_slots"],
            -summary[g]["mean_best_quality"],
            g,
        ),
    )
    return chosen, summary


def resolve_candidate_policy(
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
):
    requested = str(os.environ.get("PERFORMER_GROUP", "auto")).strip().lower()
    if requested not in VALID_GROUPS:
        raise ValueError(f"PERFORMER_GROUP must be one of {sorted(VALID_GROUPS)}, got {requested!r}")

    if requested == "mixed":
        return [list(map(int, row)) for row in candidate_lists], {
            "requested": requested,
            "resolved": "mixed",
            "mode": "explicit_mixed",
        }

    if requested == "auto":
        resolved, auto_summary = _auto_group(candidate_lists, db)
    else:
        resolved, auto_summary = requested, None

    groups = _db_groups(db)
    allow_rescue = _env_bool("PERFORMER_ALLOW_CROSS_GROUP_RESCUE", False)
    filtered = []
    rescue_slots = []
    for slot_id, candidates in enumerate(candidate_lists):
        same = [
            int(e) for e in candidates
            if 0 <= int(e) < len(groups) and groups[int(e)] == resolved
        ]
        if same:
            filtered.append(same)
        elif allow_rescue:
            filtered.append(list(map(int, candidates)))
            rescue_slots.append(slot_id)
        else:
            raise RuntimeError(
                f"No {resolved} candidates for whole-song slot {slot_id}; "
                "cross-group rescue is disabled"
            )

    return filtered, {
        "requested": requested,
        "resolved": resolved,
        "mode": "whole_song_fixed_group",
        "cross_group_rescue_enabled": allow_rescue,
        "cross_group_rescue_slots": rescue_slots,
        "auto_summary": auto_summary,
    }


def performer_switch_penalty(
    db: Mapping[str, Any],
    previous_event: int,
    current_event: int,
    slot: Mapping[str, Any] | None = None,
) -> float:
    groups = _db_groups(db)
    a = groups[int(previous_event)] if 0 <= int(previous_event) < len(groups) else "unknown"
    b = groups[int(current_event)] if 0 <= int(current_event) < len(groups) else "unknown"
    if a == b:
        return 0.0
    mode = str(os.environ.get("PERFORMER_GROUP", "auto")).strip().lower()
    if mode != "mixed":
        return float(os.environ.get("PERFORMER_FIXED_GROUP_SWITCH_PENALTY", "1000000"))
    role = str((slot or {}).get("role", "")).lower()
    base = float(os.environ.get("PERFORMER_MIXED_SWITCH_PENALTY", "2.0"))
    if role in {"transition", "resolution", "intro"}:
        return 0.5 * base
    return base
