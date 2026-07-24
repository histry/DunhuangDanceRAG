"""Global diversity and cooldown policy for closed-loop event reselection."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _db_value(db: Mapping[str, Any], key: str, index: int, default: str) -> str:
    values = db.get(key)
    if values is None or index < 0 or index >= len(values):
        return default
    return str(np.asarray(values, dtype=object)[index])


def event_identity(db: Mapping[str, Any], event_id: int) -> dict[str, str]:
    return {
        "event_uid": _db_value(db, "event_uids", event_id, f"legacy_index_{event_id}"),
        "source_uid": _db_value(db, "source_uids", event_id, "unknown"),
        "family_id": _db_value(db, "event_families", event_id, "unknown"),
        "dance_key": _db_value(db, "dance_keys", event_id, "unknown"),
    }


def diversity_assessment(
    db: Mapping[str, Any],
    event_id: int,
    selected_event_ids: Sequence[int],
) -> dict[str, Any]:
    identity = event_identity(db, event_id)
    history = [event_identity(db, int(value)) for value in selected_event_ids]
    cooldown = max(1, _env_int("V46_54_EVENT_COOLDOWN_SLOTS", 8))
    max_source_run = max(1, _env_int("V46_54_MAX_SOURCE_RUN", 2))
    max_source_share = _env_float("V46_54_MAX_SOURCE_SHARE", 0.40)
    max_family_share = _env_float("V46_54_MAX_FAMILY_SHARE", 0.50)
    minimum_share_history = max(1, _env_int("V46_54_MIN_SHARE_HISTORY", 6))

    recent_uids = [row["event_uid"] for row in history[-cooldown:]]
    exact_cooldown_violation = identity["event_uid"] in recent_uids
    source_run = 0
    for row in reversed(history):
        if row["source_uid"] != identity["source_uid"]:
            break
        source_run += 1
    source_run_after = source_run + 1

    source_counts = Counter(row["source_uid"] for row in history)
    family_counts = Counter(row["family_id"] for row in history)
    total_after = len(history) + 1
    source_share = (source_counts[identity["source_uid"]] + 1) / max(1, total_after)
    family_share = (family_counts[identity["family_id"]] + 1) / max(1, total_after)
    share_active = len(history) >= minimum_share_history

    hard_reasons: list[str] = []
    if exact_cooldown_violation:
        hard_reasons.append("event_uid_cooldown")
    if source_run_after > max_source_run:
        hard_reasons.append("source_run")
    if share_active and source_share > max_source_share + 1.0e-9:
        hard_reasons.append("source_share")
    if share_active and family_share > max_family_share + 1.0e-9:
        hard_reasons.append("family_share")

    penalty = 0.0
    if share_active:
        penalty += _env_float("V46_54_SOURCE_SHARE_WEIGHT", 2.0) * max(
            0.0, source_share - max_source_share
        )
        penalty += _env_float("V46_54_FAMILY_SHARE_WEIGHT", 1.2) * max(
            0.0, family_share - max_family_share
        )
    penalty += _env_float("V46_54_SOURCE_REUSE_WEIGHT", 0.08) * source_counts[
        identity["source_uid"]
    ]
    penalty += _env_float("V46_54_FAMILY_REUSE_WEIGHT", 0.05) * family_counts[
        identity["family_id"]
    ]
    return {
        **identity,
        "hard_valid": not hard_reasons,
        "hard_reasons": hard_reasons,
        "penalty": float(penalty),
        "cooldown_slots": cooldown,
        "source_run_after": source_run_after,
        "source_share_after": float(source_share),
        "family_share_after": float(family_share),
    }


def select_safe_diverse_proposal(
    rows: Sequence[tuple[Any, dict[str, Any]]],
    *,
    db: Mapping[str, Any],
    selected_event_ids: Sequence[int],
    primary_event_id: int,
) -> tuple[Any, dict[str, Any], str]:
    """Preserve a safe primary; otherwise choose a globally diverse safe row."""
    enriched: list[tuple[Any, dict[str, Any]]] = []
    for proposal, extra0 in rows:
        extra = dict(extra0)
        assessment = diversity_assessment(db, int(proposal.event_id), selected_event_ids)
        extra["diversity"] = assessment
        extra["selection_score"] = float(proposal.risk_score) + float(assessment["penalty"])
        enriched.append((proposal, extra))

    primary = [
        row
        for row in enriched
        if int(row[0].event_id) == int(primary_event_id)
        and bool(row[0].safe)
        and bool(row[1]["diversity"]["hard_valid"])
    ]
    if primary:
        return primary[0][0], primary[0][1], "preserved_primary_safe"

    safe_valid = [
        row
        for row in enriched
        if bool(row[0].safe) and bool(row[1]["diversity"]["hard_valid"])
    ]
    if safe_valid:
        proposal, extra = min(safe_valid, key=lambda row: float(row[1]["selection_score"]))
        return proposal, extra, "reselected_heading_physics_diverse"

    safe_count = sum(bool(row[0].safe) for row in enriched)
    hard_reasons = sorted(
        {
            str(reason)
            for _proposal, extra in enriched
            for reason in extra["diversity"].get("hard_reasons", [])
        }
    )
    # Physical/anatomy safety and exact-event/source-run cooldown are immutable
    # contracts.  The outer feasibility controller may widen duration or
    # observability tiers and retry, but this selector must never silently
    # commit an unsafe or diversity-invalid proposal.
    raise RuntimeError(
        "Heading/diversity exhausted candidates: "
        f"proposals={len(enriched)}, physically_safe={safe_count}, "
        f"diversity_hard_reasons={hard_reasons}"
    )
