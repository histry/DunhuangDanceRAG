"""Global diversity and cooldown policy for closed-loop Event reselection."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


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
    """Evaluate history diversity without turning preferences into safety gates.

    Under BR-HPR, exact repetition and source/family concentration are continuous
    costs.  The legacy binary behavior remains available by setting
    ``BR_HPR_ENABLE=0``.
    """
    identity = event_identity(db, event_id)
    history = [event_identity(db, int(value)) for value in selected_event_ids]
    cooldown = max(
        1,
        _env_int(
            "BR_HPR_EVENT_COOLDOWN_SLOTS",
            _env_int("V46_54_EVENT_COOLDOWN_SLOTS", 8),
        ),
    )
    max_source_run = max(
        1,
        _env_int(
            "BR_HPR_MAX_SOURCE_RUN",
            _env_int("V46_54_MAX_SOURCE_RUN", 2),
        ),
    )
    max_source_share = _env_float(
        "BR_HPR_MAX_SOURCE_SHARE",
        _env_float("V46_54_MAX_SOURCE_SHARE", 0.40),
    )
    max_family_share = _env_float(
        "BR_HPR_MAX_FAMILY_SHARE",
        _env_float("V46_54_MAX_FAMILY_SHARE", 0.50),
    )
    minimum_share_history = max(
        1,
        _env_int(
            "BR_HPR_MIN_SHARE_HISTORY",
            _env_int("V46_54_MIN_SHARE_HISTORY", 6),
        ),
    )

    recent_uids = [row["event_uid"] for row in history[-cooldown:]]
    exact_cooldown_violation = identity["event_uid"] in recent_uids
    repeat_gap = None
    for gap, row in enumerate(reversed(history), start=1):
        if row["event_uid"] == identity["event_uid"]:
            repeat_gap = gap
            break
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

    legacy_reasons: list[str] = []
    if exact_cooldown_violation:
        legacy_reasons.append("event_uid_cooldown")
    if source_run_after > max_source_run:
        legacy_reasons.append("source_run")
    if share_active and source_share > max_source_share + 1.0e-9:
        legacy_reasons.append("source_share")
    if share_active and family_share > max_family_share + 1.0e-9:
        legacy_reasons.append("family_share")

    repeat_violation = 0.0
    if repeat_gap is not None and repeat_gap <= cooldown:
        repeat_violation = (cooldown - repeat_gap + 1) / max(1.0, float(cooldown))
    source_run_violation = max(
        0.0,
        (source_run_after - max_source_run) / max(1.0, float(max_source_run)),
    )
    source_share_violation = (
        max(0.0, source_share - max_source_share)
        / max(1.0 - max_source_share, 1.0e-9)
        if share_active
        else 0.0
    )
    family_share_violation = (
        max(0.0, family_share - max_family_share)
        / max(1.0 - max_family_share, 1.0e-9)
        if share_active
        else 0.0
    )

    penalty = 0.0
    penalty += _env_float("BR_HPR_EVENT_REPEAT_WEIGHT", 1.20) * repeat_violation
    penalty += _env_float("BR_HPR_SOURCE_RUN_WEIGHT", 1.00) * source_run_violation
    penalty += _env_float("BR_HPR_SOURCE_SHARE_WEIGHT", 0.80) * source_share_violation
    penalty += _env_float("BR_HPR_FAMILY_SHARE_WEIGHT", 0.65) * family_share_violation
    penalty += _env_float("V46_54_SOURCE_REUSE_WEIGHT", 0.08) * source_counts[
        identity["source_uid"]
    ]
    penalty += _env_float("V46_54_FAMILY_REUSE_WEIGHT", 0.05) * family_counts[
        identity["family_id"]
    ]

    probabilistic = _env_bool("BR_HPR_ENABLE", False)
    return {
        **identity,
        "hard_valid": True if probabilistic else not legacy_reasons,
        "hard_reasons": [] if probabilistic else list(legacy_reasons),
        "soft_reasons": list(legacy_reasons),
        "legacy_hard_reasons": list(legacy_reasons),
        "penalty": float(penalty),
        "cooldown_slots": int(cooldown),
        "event_repeat_gap": repeat_gap,
        "source_run_after": int(source_run_after),
        "source_share_after": float(source_share),
        "family_share_after": float(family_share),
        "probabilistic_preference_mode": bool(probabilistic),
        "preference_violations": {
            "event_repeat": float(repeat_violation),
            "source_run": float(source_run_violation),
            "source_share": float(source_share_violation),
            "family_share": float(family_share_violation),
        },
    }

def proposal_selection_score(
    proposal: Any,
    extra: Mapping[str, Any],
    *,
    primary_event_id: int,
) -> float:
    """Score one already-safe proposal using only soft preferences.

    The primary Event is a finite prior, not an unconditional commit.  Hard
    anatomy, heading, physical, cooldown, and source constraints are evaluated
    before this function is used.
    """

    diversity = extra.get("diversity", {})
    score = float(proposal.risk_score) + float(diversity.get("penalty", 0.0))
    score += _env_float("V46_54_CANDIDATE_RANK_WEIGHT", 0.01) * float(
        getattr(proposal, "rank", 0)
    )
    score += _env_float("V46_50_POSTERIOR_WEIGHT", 0.35) * float(
        extra.get("route_prior_cost", 0.0)
    )
    score += _env_float("V46_50_UNCERTAINTY_WEIGHT", 0.20) * float(
        extra.get("route_uncertainty", 0.0)
    )
    score += _env_float("V46_50_SOURCE_CALIBRATION_WEIGHT", 0.15) * float(
        extra.get("source_calibration_penalty", 0.0)
    )
    if int(proposal.event_id) == int(primary_event_id):
        score -= max(0.0, _env_float("V46_54_PRIMARY_EVENT_BONUS", 0.18))
    return float(score)


def select_safe_diverse_proposal(
    rows: Sequence[tuple[Any, dict[str, Any]]],
    *,
    db: Mapping[str, Any],
    selected_event_ids: Sequence[int],
    primary_event_id: int,
) -> tuple[Any, dict[str, Any], str]:
    """Choose the minimum-score safe row; the primary is only a soft prior."""

    enriched: list[tuple[Any, dict[str, Any]]] = []
    for proposal, extra0 in rows:
        extra = dict(extra0)
        assessment = diversity_assessment(db, int(proposal.event_id), selected_event_ids)
        extra["diversity"] = assessment
        extra["selection_score"] = proposal_selection_score(
            proposal,
            extra,
            primary_event_id=primary_event_id,
        )
        enriched.append((proposal, extra))

    safe_valid = [
        row
        for row in enriched
        if bool(row[0].safe) and bool(row[1]["diversity"]["hard_valid"])
    ]
    if safe_valid:
        proposal, extra = min(safe_valid, key=lambda row: float(row[1]["selection_score"]))
        decision = (
            "selected_primary_soft_prior"
            if int(proposal.event_id) == int(primary_event_id)
            else "reselected_heading_physics_diverse"
        )
        return proposal, extra, decision

    physically_safe = [row for row in enriched if bool(row[0].safe)]
    reason_counts = Counter(
        str(reason)
        for _proposal, extra in physically_safe
        for reason in extra["diversity"].get("hard_reasons", [])
    )
    # Physical/anatomy safety and exact-event/source history are immutable.  The
    # outer feasibility controller may expand search width or bounded duration
    # tiers, but this selector never commits an unsafe or history-invalid row.
    raise RuntimeError(
        "Heading/diversity exhausted candidates: "
        f"proposals={len(enriched)}, physically_safe={len(physically_safe)}, "
        f"safe_diversity_reason_counts={dict(reason_counts)}"
    )
