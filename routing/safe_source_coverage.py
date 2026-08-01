"""Source--Family-targeted candidate expansion for low-resource whole-song routing.

The module is deliberately a *candidate proposal* mechanism.  It never labels an
Event physically safe.  It adds statically plausible Events from uncovered sources and families
and delegates every final decision to the authoritative exact simulator.
"""
from __future__ import annotations

import dataclasses
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from routing.hierarchical_constraint_model import (
    event_hyperbolic_distance,
    event_identity,
)

_EPS = 1.0e-9


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


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


def _db_value(
    db: Mapping[str, Any],
    key: str,
    event_id: int,
    default: Any,
) -> Any:
    try:
        values = np.asarray(db[key], dtype=object)
        value = values[int(event_id)]
        return value.item() if isinstance(value, np.generic) else value
    except Exception:
        return default


def _db_float(
    db: Mapping[str, Any],
    keys: Sequence[str],
    event_id: int,
    default: float,
) -> float:
    for key in keys:
        try:
            value = float(np.asarray(db[key])[int(event_id)])
            if np.isfinite(value):
                return value
        except Exception:
            pass
    return float(default)


def _performer_group(db: Mapping[str, Any], event_id: int) -> str:
    for key in ("performer_groups", "genders"):
        if key in db:
            return str(_db_value(db, key, event_id, "unknown")).strip().lower()
    return "unknown"


def _static_valid(db: Mapping[str, Any], event_id: int, minimum_anatomy: float) -> bool:
    try:
        if not bool(np.asarray(db["anatomy_valid"], dtype=bool)[int(event_id)]):
            return False
    except Exception:
        return False
    try:
        if not bool(np.asarray(db["event_heading_valid"], dtype=bool)[int(event_id)]):
            return False
    except Exception:
        return False
    anatomy = _db_float(db, ("anatomy_quality",), event_id, 0.0)
    if anatomy < float(minimum_anatomy):
        return False
    path = str(_db_value(db, "paths", event_id, ""))
    return bool(path)


def _event_frames(db: Mapping[str, Any], event_id: int, fps: float) -> Optional[float]:
    for key in ("event_frames", "source_frames", "frame_counts"):
        try:
            value = float(np.asarray(db[key])[int(event_id)])
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
    for key in ("event_duration_seconds", "durations", "duration_seconds"):
        try:
            value = float(np.asarray(db[key])[int(event_id)])
            if np.isfinite(value) and value > 0.0:
                return value * float(fps)
        except Exception:
            pass
    return None


@dataclass(frozen=True)
class SafeSourceCoverageConfig:
    """Bounded source-reservoir and exact-expansion policy."""

    enabled: bool
    target_safe_sources: int
    target_safe_families: int
    reservoir_maximum_per_slot: int
    reservoir_per_source: int
    expansion_maximum_exact: int
    expansion_per_source: int
    minimum_anatomy_quality: float
    quality_weight: float
    anatomy_weight: float
    heading_weight: float
    duration_weight: float
    family_affinity_weight: float
    hierarchy_affinity_weight: float
    history_novelty_weight: float
    bottleneck_initial_reservoir_per_slot: int
    bottleneck_expansion_maximum: int

    @classmethod
    def from_environment(cls) -> "SafeSourceCoverageConfig":
        return cls(
            enabled=_env_bool("BR_HPR_SOURCE_COVERAGE_ENABLE", True),
            target_safe_sources=max(
                1, _env_int("BR_HPR_MINIMUM_SAFE_SOURCE_COUNT", 2)
            ),
            target_safe_families=max(
                1, _env_int("BR_HPR_MINIMUM_SAFE_FAMILY_COUNT", 2)
            ),
            reservoir_maximum_per_slot=max(
                1, _env_int("BR_HPR_SOURCE_RESERVOIR_MAXIMUM_PER_SLOT", 24)
            ),
            reservoir_per_source=max(
                1, _env_int("BR_HPR_SOURCE_RESERVOIR_PER_SOURCE", 3)
            ),
            expansion_maximum_exact=max(
                0, _env_int("BR_HPR_SOURCE_EXPANSION_MAXIMUM_EXACT", 8)
            ),
            expansion_per_source=max(
                1, _env_int("BR_HPR_SOURCE_EXPANSION_PER_SOURCE", 2)
            ),
            minimum_anatomy_quality=float(
                np.clip(
                    _env_float("BR_HPR_SOURCE_EXPANSION_ANATOMY_MIN", 0.30),
                    0.0,
                    1.0,
                )
            ),
            quality_weight=max(
                0.0, _env_float("BR_HPR_SOURCE_EXPANSION_QUALITY_WEIGHT", 0.55)
            ),
            anatomy_weight=max(
                0.0, _env_float("BR_HPR_SOURCE_EXPANSION_ANATOMY_WEIGHT", 0.45)
            ),
            heading_weight=max(
                0.0, _env_float("BR_HPR_SOURCE_EXPANSION_HEADING_WEIGHT", 0.25)
            ),
            duration_weight=max(
                0.0, _env_float("BR_HPR_SOURCE_EXPANSION_DURATION_WEIGHT", 0.35)
            ),
            family_affinity_weight=max(
                0.0,
                _env_float("BR_HPR_SOURCE_EXPANSION_FAMILY_AFFINITY_WEIGHT", 0.30),
            ),
            hierarchy_affinity_weight=max(
                0.0,
                _env_float("BR_HPR_SOURCE_EXPANSION_HIERARCHY_WEIGHT", 0.20),
            ),
            history_novelty_weight=max(
                0.0,
                _env_float("BR_HPR_SOURCE_EXPANSION_HISTORY_NOVELTY_WEIGHT", 0.20),
            ),
            bottleneck_initial_reservoir_per_slot=max(
                0, _env_int("BR_HPR_BOTTLENECK_INITIAL_RESERVOIR_PER_SLOT", 8)
            ),
            bottleneck_expansion_maximum=max(
                0, _env_int("BR_HPR_BOTTLENECK_EXPANSION_MAXIMUM", 12)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _allowed_performer_groups(
    candidate_ids: Sequence[int],
    db: Mapping[str, Any],
) -> set[str]:
    groups = {
        _performer_group(db, int(event_id))
        for event_id in candidate_ids
        if int(event_id) >= 0
    }
    groups.discard("unknown")
    return groups


def _reservoir_score(
    *,
    db: Mapping[str, Any],
    event_id: int,
    primary_event_id: int,
    target_frames: int,
    fps: float,
    selected_event_ids: Sequence[int],
    config: SafeSourceCoverageConfig,
) -> tuple[float, dict[str, Any]]:
    quality = _db_float(
        db,
        ("v46_53_combined_quality", "event_quality_scores"),
        event_id,
        0.5,
    )
    anatomy = _db_float(db, ("anatomy_quality",), event_id, 0.5)
    heading = _db_float(db, ("event_heading_quality",), event_id, 0.5)
    frames = _event_frames(db, event_id, fps)
    if frames is None:
        duration_affinity = 0.5
    else:
        ratio = max(float(frames), 1.0) / max(float(target_frames), 1.0)
        duration_affinity = math.exp(-abs(math.log(max(ratio, _EPS))))

    identity = event_identity(db, event_id)
    primary_identity = event_identity(db, primary_event_id)
    family_affinity = float(identity["family_id"] == primary_identity["family_id"])
    hierarchy_distance = event_hyperbolic_distance(
        db,
        event_id,
        primary_event_id,
    )
    hierarchy_affinity = math.exp(-hierarchy_distance)
    history_sources = Counter(
        event_identity(db, int(value))["source_uid"] for value in selected_event_ids
    )
    history_families = Counter(
        event_identity(db, int(value))["family_id"] for value in selected_event_ids
    )
    novelty = 1.0 / (
        1.0
        + history_sources[identity["source_uid"]]
        + 0.5 * history_families[identity["family_id"]]
    )
    score = (
        config.quality_weight * quality
        + config.anatomy_weight * anatomy
        + config.heading_weight * heading
        + config.duration_weight * duration_affinity
        + config.family_affinity_weight * family_affinity
        + config.hierarchy_affinity_weight * hierarchy_affinity
        + config.history_novelty_weight * novelty
    )
    return float(score), {
        "quality": float(quality),
        "anatomy_quality": float(anatomy),
        "heading_quality": float(heading),
        "duration_affinity": float(duration_affinity),
        "family_affinity": float(family_affinity),
        "hierarchy_affinity": float(hierarchy_affinity),
        "history_novelty": float(novelty),
        "reservoir_score": float(score),
    }


def build_source_reservoir_layers(
    *,
    slots: Sequence[Mapping[str, Any]],
    target_lengths: Sequence[int],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    fps: float,
    blocked: Optional[Mapping[int, set]] = None,
    config: Optional[SafeSourceCoverageConfig] = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Build statically valid source-diverse reservoirs for every slot."""
    cfg = config or SafeSourceCoverageConfig.from_environment()
    banned = blocked or {}
    total_events = len(np.asarray(db["paths"], dtype=object))
    layers: list[list[int]] = []
    reports: list[dict[str, Any]] = []
    for slot_index, candidates0 in enumerate(candidate_lists):
        candidates = [
            int(value)
            for value in candidates0
            if 0 <= int(value) < total_events
            and int(value) not in banned.get(slot_index, set())
        ]
        if not candidates or not cfg.enabled:
            layers.append([])
            reports.append(
                {
                    "slot": int(slot_index),
                    "enabled": bool(cfg.enabled),
                    "reservoir_event_ids": [],
                    "reservoir_sources": [],
                }
            )
            continue
        primary = int(candidates[0])
        allowed_groups = _allowed_performer_groups(candidates, db)
        existing = set(candidates)
        rows_by_source: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}
        for event_id in range(total_events):
            if event_id in existing or event_id in banned.get(slot_index, set()):
                continue
            if not _static_valid(db, event_id, cfg.minimum_anatomy_quality):
                continue
            group = _performer_group(db, event_id)
            if allowed_groups and group not in allowed_groups:
                continue
            score, detail = _reservoir_score(
                db=db,
                event_id=event_id,
                primary_event_id=primary,
                target_frames=int(target_lengths[slot_index]),
                fps=float(fps),
                selected_event_ids=(),
                config=cfg,
            )
            source = event_identity(db, event_id)["source_uid"]
            rows_by_source.setdefault(source, []).append((score, event_id, detail))

        source_order = sorted(
            rows_by_source,
            key=lambda source: (
                -max(row[0] for row in rows_by_source[source]),
                source,
            ),
        )
        selected: list[int] = []
        selected_details: list[dict[str, Any]] = []
        for source in source_order:
            rows = sorted(rows_by_source[source], key=lambda row: (-row[0], row[1]))
            for score, event_id, detail in rows[: cfg.reservoir_per_source]:
                if len(selected) >= cfg.reservoir_maximum_per_slot:
                    break
                selected.append(int(event_id))
                selected_details.append(
                    {
                        "event_id": int(event_id),
                        "source_uid": source,
                        **detail,
                    }
                )
            if len(selected) >= cfg.reservoir_maximum_per_slot:
                break
        layers.append(selected)
        reports.append(
            {
                "slot": int(slot_index),
                "enabled": True,
                "primary_event_id": int(primary),
                "allowed_performer_groups": sorted(allowed_groups),
                "reservoir_event_ids": selected,
                "reservoir_sources": sorted(
                    {event_identity(db, value)["source_uid"] for value in selected}
                ),
                "candidate_details": selected_details,
            }
        )
    return layers, {
        "schema": "safe_source_candidate_reservoir",
        "configuration": cfg.to_dict(),
        "slots": reports,
    }


def build_state_source_expansion_batches(
    *,
    reservoir_event_ids: Sequence[int],
    attempted_event_ids: Sequence[int],
    hard_safe_event_ids: Sequence[int],
    selected_event_ids: Sequence[int],
    previous_event_id: Optional[int],
    db: Mapping[str, Any],
    config: SafeSourceCoverageConfig,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Return source-grouped batches for exact-safety feedback expansion.

    The planner must simulate one batch, update the *actual* hard-safe set, then
    decide whether another batch is needed.  No source is counted as covered merely
    because its candidates were selected for simulation.
    """
    attempted = set(map(int, attempted_event_ids))
    safe_sources = {event_identity(db, int(v))["source_uid"] for v in hard_safe_event_ids}
    safe_families = {event_identity(db, int(v))["family_id"] for v in hard_safe_event_ids}
    previous_source = (
        event_identity(db, int(previous_event_id))["source_uid"]
        if previous_event_id is not None
        else None
    )
    history_sources = Counter(
        event_identity(db, int(value))["source_uid"] for value in selected_event_ids
    )
    history_families = Counter(
        event_identity(db, int(value))["family_id"] for value in selected_event_ids
    )
    rows_by_source: dict[str, list[tuple[tuple[Any, ...], int]]] = {}
    for value in reservoir_event_ids:
        event_id = int(value)
        if event_id in attempted:
            continue
        identity = event_identity(db, event_id)
        source = identity["source_uid"]
        family = identity["family_id"]
        if source in safe_sources and family in safe_families:
            continue
        key = (
            source in safe_sources,
            family in safe_families,
            source == previous_source,
            history_sources[source],
            history_families[family],
            event_id,
        )
        rows_by_source.setdefault(source, []).append((key, event_id))

    source_order = sorted(
        rows_by_source,
        key=lambda source: min(row[0] for row in rows_by_source[source]),
    )
    batches: list[list[int]] = []
    selected_event_ids: list[int] = []
    selected_sources: list[str] = []
    for source in source_order:
        rows = sorted(rows_by_source[source], key=lambda row: row[0])
        batch = [int(event_id) for _key, event_id in rows[: config.expansion_per_source]]
        if not batch:
            continue
        remaining = config.expansion_maximum_exact - len(selected_event_ids)
        if remaining <= 0:
            break
        batch = batch[:remaining]
        batches.append(batch)
        selected_event_ids.extend(batch)
        selected_sources.append(source)
    return batches, {
        "schema": "iterative_safe_source_family_expansion_plan",
        "triggered": bool(batches),
        "safe_source_count_before": int(len(safe_sources)),
        "safe_family_count_before": int(len(safe_families)),
        "safe_sources_before": sorted(safe_sources),
        "safe_families_before": sorted(safe_families),
        "reservoir_candidates_available": int(sum(len(v) for v in rows_by_source.values())),
        "selected_event_ids": selected_event_ids,
        "selected_source_uids": selected_sources,
        "planned_batches": [list(map(int, batch)) for batch in batches],
        "additional_exact_simulation_budget": int(config.expansion_maximum_exact),
        "physical_constraints_relaxed": False,
        "anatomy_constraints_relaxed": False,
        "severe_heading_constraints_relaxed": False,
    }


def select_state_source_expansion_candidates(
    *,
    reservoir_event_ids: Sequence[int],
    attempted_event_ids: Sequence[int],
    hard_safe_event_ids: Sequence[int],
    selected_event_ids: Sequence[int],
    previous_event_id: Optional[int],
    db: Mapping[str, Any],
    config: SafeSourceCoverageConfig,
) -> tuple[list[int], dict[str, Any]]:
    """Compatibility wrapper returning the flattened iterative expansion plan."""
    batches, report = build_state_source_expansion_batches(
        reservoir_event_ids=reservoir_event_ids,
        attempted_event_ids=attempted_event_ids,
        hard_safe_event_ids=hard_safe_event_ids,
        selected_event_ids=selected_event_ids,
        previous_event_id=previous_event_id,
        db=db,
        config=config,
    )
    return [event for batch in batches for event in batch], report


def select_bottleneck_layer_expansion_candidates(
    *,
    reservoir_event_ids: Sequence[int],
    active_event_ids: Sequence[int],
    selected_event_ids: Sequence[int],
    db: Mapping[str, Any],
    config: SafeSourceCoverageConfig,
) -> tuple[list[int], dict[str, Any]]:
    """Select Source--Family novel candidates for a predicted future bottleneck."""
    active = set(map(int, active_event_ids))
    history_sources = Counter(
        event_identity(db, int(value))["source_uid"] for value in selected_event_ids
    )
    history_families = Counter(
        event_identity(db, int(value))["family_id"] for value in selected_event_ids
    )
    active_sources = {event_identity(db, value)["source_uid"] for value in active}
    active_families = {event_identity(db, value)["family_id"] for value in active}
    rows: list[tuple[tuple[Any, ...], int]] = []
    for value in reservoir_event_ids:
        event_id = int(value)
        if event_id in active:
            continue
        identity = event_identity(db, event_id)
        source = identity["source_uid"]
        family = identity["family_id"]
        rows.append(
            (
                (
                    source in active_sources,
                    family in active_families,
                    history_sources[source],
                    history_families[family],
                    event_id,
                ),
                event_id,
            )
        )
    rows.sort(key=lambda row: row[0])
    selected = [
        int(event_id)
        for _key, event_id in rows[: config.bottleneck_expansion_maximum]
    ]
    return selected, {
        "schema": "predicted_bottleneck_source_family_expansion",
        "triggered": bool(selected),
        "selected_event_ids": selected,
        "active_source_count": len(active_sources),
        "active_family_count": len(active_families),
        "reservoir_candidates_available": len(rows),
    }
