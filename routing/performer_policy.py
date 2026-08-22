#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whole-song performer-group policy for Event-RAG candidate routing."""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

VALID_GROUPS = {"female", "male", "mixed", "auto"}
VALID_IDENTITY_MODES = {"group", "fixed_source", "fixed_dancer"}


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


def _db_text(db: Mapping[str, Any], key: str, default: str = "") -> np.ndarray:
    count = len(np.asarray(db["paths"]))
    if key not in db:
        return np.asarray([default] * count, dtype=object)
    values = np.asarray(db[key], dtype=object)
    if len(values) != count:
        raise RuntimeError(f"Performer metadata {key!r} is not event-aligned")
    return np.asarray([str(value).strip() for value in values], dtype=object)


def _choose_fixed_identity(
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    identities: np.ndarray,
    *,
    label: str,
    valid: np.ndarray | None = None,
) -> Tuple[list[list[int]], Dict[str, Any]]:
    if valid is None:
        valid = np.asarray([bool(str(value).strip()) for value in identities], dtype=bool)
    universe = sorted(
        {
            str(identities[int(event)])
            for row in candidate_lists
            for event in row
            if 0 <= int(event) < len(identities) and bool(valid[int(event)])
        }
    )
    if not universe:
        raise RuntimeError(f"{label} routing requires verified non-empty identity metadata")
    summaries: dict[str, dict[str, Any]] = {}
    for identity in universe:
        best = []
        missing = 0
        for row in candidate_lists:
            matches = [
                int(event)
                for event in row
                if bool(valid[int(event)]) and str(identities[int(event)]) == identity
            ]
            if matches:
                best.append(max(_quality(db, event) for event in matches))
            else:
                missing += 1
        summaries[identity] = {
            "missing_slots": missing,
            "mean_best_quality": float(np.mean(best)) if best else -1.0,
        }
    chosen = min(
        universe,
        key=lambda value: (
            summaries[value]["missing_slots"],
            -summaries[value]["mean_best_quality"],
            value,
        ),
    )
    allow_rescue = _env_bool("PERFORMER_ALLOW_IDENTITY_RESCUE", False)
    filtered: list[list[int]] = []
    rescue_slots: list[int] = []
    for slot_id, row in enumerate(candidate_lists):
        matches = [
            int(event)
            for event in row
            if bool(valid[int(event)]) and str(identities[int(event)]) == chosen
        ]
        if matches:
            filtered.append(matches)
        elif allow_rescue:
            filtered.append(list(map(int, row)))
            rescue_slots.append(slot_id)
        else:
            raise RuntimeError(
                f"No {label}={chosen!r} candidate for whole-song slot {slot_id}; "
                "identity rescue is disabled"
            )
    return filtered, {
        "resolved_identity": chosen,
        "identity_summary": summaries,
        "identity_rescue_enabled": allow_rescue,
        "identity_rescue_slots": rescue_slots,
    }


def _quality(db: Mapping[str, Any], event_id: int) -> float:
    for key in ("event_geometry_combined_quality", "event_quality_scores", "anatomy_quality"):
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

    require_solo = _env_bool("PERFORMER_REQUIRE_SOLO_COMPATIBLE", False)
    solo_values = np.asarray(
        db.get("solo_compatible", np.zeros(len(np.asarray(db["paths"])), dtype=bool)),
        dtype=bool,
    )
    if len(solo_values) != len(np.asarray(db["paths"])):
        raise RuntimeError("solo_compatible metadata is not event-aligned")
    working: list[list[int]] = []
    excluded_pair_events = 0
    for slot_id, row in enumerate(candidate_lists):
        original = list(map(int, row))
        eligible = (
            [event for event in original if bool(solo_values[event])]
            if require_solo
            else original
        )
        excluded_pair_events += len(original) - len(eligible)
        if not eligible:
            raise RuntimeError(
                f"No solo-compatible candidates for whole-song slot {slot_id}; "
                "multi-performer recordings require manual review"
            )
        working.append(eligible)

    if requested == "auto":
        resolved, auto_summary = _auto_group(working, db)
    elif requested == "mixed":
        resolved, auto_summary = "mixed", None
    else:
        resolved, auto_summary = requested, None

    groups = _db_groups(db)
    allow_rescue = _env_bool("PERFORMER_ALLOW_CROSS_GROUP_RESCUE", False)
    filtered = []
    rescue_slots = []
    for slot_id, candidates in enumerate(working):
        if resolved == "mixed":
            filtered.append(list(map(int, candidates)))
            continue
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

    identity_mode = str(os.environ.get("PERFORMER_IDENTITY_MODE", "group")).strip().lower()
    if identity_mode not in VALID_IDENTITY_MODES:
        raise ValueError(
            f"PERFORMER_IDENTITY_MODE must be one of {sorted(VALID_IDENTITY_MODES)}, "
            f"got {identity_mode!r}"
        )
    identity_report: Dict[str, Any] = {}
    if identity_mode == "fixed_source":
        filtered, identity_report = _choose_fixed_identity(
            filtered,
            db,
            _db_text(db, "source_uids", ""),
            label="source_uid",
        )
    elif identity_mode == "fixed_dancer":
        dancer_ids = _db_text(db, "dancer_ids", "")
        statuses = _db_text(db, "dancer_id_statuses", "unverified")
        verified = np.asarray(
            [bool(value) and status == "verified" for value, status in zip(dancer_ids, statuses)],
            dtype=bool,
        )
        filtered, identity_report = _choose_fixed_identity(
            filtered,
            db,
            dancer_ids,
            label="dancer_id",
            valid=verified,
        )

    return filtered, {
        "requested": requested,
        "resolved": resolved,
        "mode": "explicit_mixed" if resolved == "mixed" else "whole_song_fixed_group",
        "cross_group_rescue_enabled": allow_rescue,
        "cross_group_rescue_slots": rescue_slots,
        "auto_summary": auto_summary,
        "single_body_output": True,
        "require_solo_compatible": require_solo,
        "excluded_unreviewed_pair_candidates": excluded_pair_events,
        "identity_mode": identity_mode,
        "same_source_track_guaranteed": identity_mode == "fixed_source" and not identity_report.get("identity_rescue_slots"),
        "same_dancer_claim_supported": identity_mode == "fixed_dancer" and not identity_report.get("identity_rescue_slots"),
        **identity_report,
    }


def performer_switch_penalty(
    db: Mapping[str, Any],
    previous_event: int,
    current_event: int,
    slot: Mapping[str, Any] | None = None,
) -> float:
    identity_mode = str(os.environ.get("PERFORMER_IDENTITY_MODE", "group")).strip().lower()
    if identity_mode == "fixed_source":
        identities = _db_text(db, "source_uids", "")
    elif identity_mode == "fixed_dancer":
        identities = _db_text(db, "dancer_ids", "")
    else:
        identities = _db_groups(db)
    groups = identities
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
