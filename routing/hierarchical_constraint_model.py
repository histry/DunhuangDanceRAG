"""Hierarchical probabilistic preference constraints for whole-song motion routing.

The module keeps physical, anatomical and severe-heading decisions outside the
preference model.  It represents repetition, source/family concentration,
observability and hierarchy novelty as continuous song-level resources.  The
implementation additionally models joint source--family scarcity and allocates
controlled recovery as a continuous resource guided by future viability depth.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_EPS = 1.0e-9
CONSTRAINT_NAMES = (
    "event_repeat",
    "source_run",
    "source_share",
    "family_share",
    "observability",
    "hierarchy_repetition",
)


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


def event_identity(db: Mapping[str, Any], event_id: int) -> dict[str, str]:
    """Return the Event hierarchy used by diversity and budget models."""
    return {
        "dance_key": str(_db_value(db, "dance_keys", event_id, "unknown_dance")),
        "source_uid": str(_db_value(db, "source_uids", event_id, "unknown_source")),
        "family_id": str(_db_value(db, "event_families", event_id, "unknown_family")),
        "event_uid": str(
            _db_value(db, "event_uids", event_id, f"event_index_{int(event_id)}")
        ),
    }


def _stable_direction(token: str, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=max(2, int(dimension))).astype(np.float64)
    return vector / max(float(np.linalg.norm(vector)), _EPS)


@lru_cache(maxsize=65536)
def _hierarchy_embedding_cached(
    dance_key: str,
    source_uid: str,
    family_id: str,
    event_uid: str,
    dimension: int,
    maximum_radius: float,
) -> tuple[float, ...]:
    """Construct a deterministic Poincare-ball embedding of one hierarchy path."""
    tokens = (
        f"dance:{dance_key}",
        f"source:{source_uid}",
        f"family:{family_id}",
        f"event:{event_uid}",
    )
    direction = np.zeros(max(2, int(dimension)), dtype=np.float64)
    point = direction.copy()
    path = ""
    radii = np.linspace(0.24, float(maximum_radius), len(tokens))
    for depth, token in enumerate(tokens):
        path = f"{path}/{token}"
        local = _stable_direction(path, len(direction))
        if depth == 0:
            direction = local
        else:
            direction = 0.78 * direction + 0.22 * local
            direction /= max(float(np.linalg.norm(direction)), _EPS)
        point = float(radii[depth]) * direction
    return tuple(map(float, point))


def event_hyperbolic_embedding(
    db: Mapping[str, Any],
    event_id: int,
    *,
    dimension: int = 8,
    maximum_radius: float = 0.82,
) -> np.ndarray:
    """Read learned Event embeddings or use a deterministic hierarchy fallback."""
    learned = db.get("event_hyperbolic_embeddings")
    if learned is not None:
        try:
            vector = np.asarray(learned, dtype=np.float64)[int(event_id)].reshape(-1)
            if len(vector) >= 2 and np.isfinite(vector).all():
                norm = float(np.linalg.norm(vector))
                if norm >= 1.0:
                    vector = vector / max(norm, _EPS) * min(0.95, maximum_radius)
                return vector.astype(np.float64)
        except Exception:
            pass
    identity = event_identity(db, int(event_id))
    return np.asarray(
        _hierarchy_embedding_cached(
            identity["dance_key"],
            identity["source_uid"],
            identity["family_id"],
            identity["event_uid"],
            max(2, int(dimension)),
            float(np.clip(maximum_radius, 0.20, 0.95)),
        ),
        dtype=np.float64,
    )


def poincare_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the curvature-one Poincare-ball geodesic distance."""
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Poincare vectors must share shape, got {x.shape} and {y.shape}")
    x2 = min(float(np.dot(x, x)), 1.0 - 1.0e-7)
    y2 = min(float(np.dot(y, y)), 1.0 - 1.0e-7)
    delta2 = float(np.dot(x - y, x - y))
    argument = 1.0 + 2.0 * delta2 / max((1.0 - x2) * (1.0 - y2), _EPS)
    return float(np.arccosh(max(1.0, argument)))


def event_hyperbolic_distance(
    db: Mapping[str, Any],
    left_event_id: int,
    right_event_id: int,
    *,
    dimension: int = 8,
    maximum_radius: float = 0.82,
) -> float:
    if int(left_event_id) == int(right_event_id):
        return 0.0
    return poincare_distance(
        event_hyperbolic_embedding(
            db,
            int(left_event_id),
            dimension=dimension,
            maximum_radius=maximum_radius,
        ),
        event_hyperbolic_embedding(
            db,
            int(right_event_id),
            dimension=dimension,
            maximum_radius=maximum_radius,
        ),
    )


def _entropy_from_counts(counts: Counter) -> float:
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if values.size <= 1 or float(values.sum()) <= 0.0:
        return 0.0
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, _EPS))))


def _normalized_entropy(counts: Counter) -> float:
    if len(counts) <= 1:
        return 0.0
    return float(_entropy_from_counts(counts) / max(math.log(len(counts)), _EPS))


@dataclass(frozen=True)
class FeasibleSetScarcityContext:
    """Observed Source--Family diversity of the exact hard-safe candidate set."""

    enabled: bool = False
    hard_safe_event_ids: tuple[int, ...] = ()
    all_event_ids: tuple[int, ...] = ()
    safe_source_count: int = 0
    all_source_count: int = 0
    safe_family_count: int = 0
    all_family_count: int = 0
    safe_source_entropy: float = 0.0
    all_source_entropy: float = 0.0
    safe_family_entropy: float = 0.0
    all_family_entropy: float = 0.0
    safe_source_entropy_normalized: float = 0.0
    all_source_entropy_normalized: float = 0.0
    safe_family_entropy_normalized: float = 0.0
    all_family_entropy_normalized: float = 0.0
    source_penalty_scale: float = 1.0
    family_penalty_scale: float = 1.0
    alternative_safe_source_exists: bool = True
    alternative_safe_family_exists: bool = True
    source_scarcity_exemption: bool = False
    family_scarcity_exemption: bool = False
    safe_source_counts: tuple[tuple[str, int], ...] = ()
    all_source_counts: tuple[tuple[str, int], ...] = ()
    safe_family_counts: tuple[tuple[str, int], ...] = ()
    all_family_counts: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hard_safe_feasible_set_scarcity_context",
            "enabled": bool(self.enabled),
            "hard_safe_event_ids": list(map(int, self.hard_safe_event_ids)),
            "all_event_ids": list(map(int, self.all_event_ids)),
            "safe_source_count": int(self.safe_source_count),
            "all_source_count": int(self.all_source_count),
            "safe_family_count": int(self.safe_family_count),
            "all_family_count": int(self.all_family_count),
            "safe_source_entropy": float(self.safe_source_entropy),
            "all_source_entropy": float(self.all_source_entropy),
            "safe_family_entropy": float(self.safe_family_entropy),
            "all_family_entropy": float(self.all_family_entropy),
            "safe_source_entropy_normalized": float(self.safe_source_entropy_normalized),
            "all_source_entropy_normalized": float(self.all_source_entropy_normalized),
            "safe_family_entropy_normalized": float(self.safe_family_entropy_normalized),
            "all_family_entropy_normalized": float(self.all_family_entropy_normalized),
            "source_penalty_scale": float(self.source_penalty_scale),
            "family_penalty_scale": float(self.family_penalty_scale),
            "alternative_safe_source_exists": bool(self.alternative_safe_source_exists),
            "alternative_safe_family_exists": bool(self.alternative_safe_family_exists),
            "source_scarcity_exemption": bool(self.source_scarcity_exemption),
            "family_scarcity_exemption": bool(self.family_scarcity_exemption),
            "safe_source_counts": dict(self.safe_source_counts),
            "all_source_counts": dict(self.all_source_counts),
            "safe_family_counts": dict(self.safe_family_counts),
            "all_family_counts": dict(self.all_family_counts),
        }


# Compatibility alias retained for existing callers and historical reports.
SourceScarcityContext = FeasibleSetScarcityContext


@dataclass(frozen=True)
class ConstraintBudgetConfig:
    """Song-level preference budgets, scarcity policy and recovery resources."""

    enabled: bool
    total_slots: int
    event_cooldown_slots: int
    maximum_source_run: int
    maximum_source_share: float
    maximum_family_share: float
    minimum_share_history: int
    observability_target: float
    hierarchy_similarity_target: float
    hierarchy_temperature: float
    hierarchy_recency_decay: float
    hyperbolic_dimension: int
    hyperbolic_maximum_radius: float
    event_repeat_budget: float
    source_run_budget: float
    source_share_budget: float
    family_share_budget: float
    observability_budget: float
    hierarchy_repetition_budget: float
    event_repeat_weight: float
    source_run_weight: float
    source_share_weight: float
    family_share_weight: float
    observability_weight: float
    hierarchy_repetition_weight: float
    future_reachability_weight: float
    dual_learning_rate: float
    recovery_penalty: float
    recovery_topk: int
    budget_tolerance: float
    controlled_recovery_enabled: bool
    recovery_budget_total: float
    recovery_minimum_charge: float
    recovery_maximum_charge_per_slot: float
    recovery_event_repeat_weight: float
    recovery_source_run_weight: float
    recovery_source_share_weight: float
    recovery_family_share_weight: float
    recovery_observability_weight: float
    recovery_hierarchy_weight: float
    source_scarcity_enabled: bool
    minimum_safe_source_count: int
    source_scarcity_minimum_scale: float
    source_scarcity_budget_credit: float
    family_scarcity_enabled: bool
    minimum_safe_family_count: int
    family_scarcity_minimum_scale: float
    family_scarcity_budget_credit: float
    recovery_minimum_viability_depth: int
    recovery_require_safe_successor: bool

    @classmethod
    def from_environment(cls, total_slots: int) -> "ConstraintBudgetConfig":
        slots = max(1, int(total_slots))
        return cls(
            enabled=True,
            total_slots=slots,
            event_cooldown_slots=max(1, _env_int("ROUTING_BUDGET_EVENT_COOLDOWN_SLOTS", 8)),
            maximum_source_run=max(1, _env_int("ROUTING_BUDGET_MAX_SOURCE_RUN", 2)),
            maximum_source_share=float(
                np.clip(_env_float("ROUTING_BUDGET_MAX_SOURCE_SHARE", 0.40), 0.0, 1.0)
            ),
            maximum_family_share=float(
                np.clip(_env_float("ROUTING_BUDGET_MAX_FAMILY_SHARE", 0.50), 0.0, 1.0)
            ),
            minimum_share_history=max(1, _env_int("ROUTING_BUDGET_MIN_SHARE_HISTORY", 6)),
            observability_target=float(
                np.clip(_env_float("ROUTING_BUDGET_OBSERVABILITY_TARGET", 0.45), 0.05, 1.0)
            ),
            hierarchy_similarity_target=float(
                np.clip(
                    _env_float("ROUTING_BUDGET_HIERARCHY_SIMILARITY_TARGET", 0.55),
                    0.0,
                    0.99,
                )
            ),
            hierarchy_temperature=max(
                1.0e-3, _env_float("ROUTING_BUDGET_HIERARCHY_TEMPERATURE", 1.25)
            ),
            hierarchy_recency_decay=max(
                0.25, _env_float("ROUTING_BUDGET_HIERARCHY_RECENCY_DECAY", 3.0)
            ),
            hyperbolic_dimension=max(
                2, _env_int("ROUTING_BUDGET_HYPERBOLIC_DIMENSION", 8)
            ),
            hyperbolic_maximum_radius=float(
                np.clip(
                    _env_float("ROUTING_BUDGET_HYPERBOLIC_MAXIMUM_RADIUS", 0.82),
                    0.20,
                    0.95,
                )
            ),
            event_repeat_budget=max(
                0.0,
                _env_float("ROUTING_BUDGET_EVENT_REPEAT_BUDGET", max(1.0, 0.10 * slots)),
            ),
            source_run_budget=max(
                0.0,
                _env_float("ROUTING_BUDGET_SOURCE_RUN_BUDGET", max(0.75, 0.08 * slots)),
            ),
            source_share_budget=max(
                0.0,
                _env_float("ROUTING_BUDGET_SOURCE_SHARE_BUDGET", max(0.75, 0.08 * slots)),
            ),
            family_share_budget=max(
                0.0,
                _env_float("ROUTING_BUDGET_FAMILY_SHARE_BUDGET", max(0.90, 0.10 * slots)),
            ),
            observability_budget=max(
                0.0,
                _env_float("ROUTING_BUDGET_OBSERVABILITY_BUDGET", max(0.75, 0.08 * slots)),
            ),
            hierarchy_repetition_budget=max(
                0.0,
                _env_float(
                    "ROUTING_BUDGET_HIERARCHY_REPETITION_BUDGET", max(1.50, 0.15 * slots)
                ),
            ),
            event_repeat_weight=max(
                0.0, _env_float("ROUTING_BUDGET_EVENT_REPEAT_WEIGHT", 1.20)
            ),
            source_run_weight=max(
                0.0, _env_float("ROUTING_BUDGET_SOURCE_RUN_WEIGHT", 1.00)
            ),
            source_share_weight=max(
                0.0, _env_float("ROUTING_BUDGET_SOURCE_SHARE_WEIGHT", 0.80)
            ),
            family_share_weight=max(
                0.0, _env_float("ROUTING_BUDGET_FAMILY_SHARE_WEIGHT", 0.65)
            ),
            observability_weight=max(
                0.0, _env_float("ROUTING_BUDGET_OBSERVABILITY_WEIGHT", 0.60)
            ),
            hierarchy_repetition_weight=max(
                0.0, _env_float("ROUTING_BUDGET_HIERARCHY_REPETITION_WEIGHT", 0.90)
            ),
            future_reachability_weight=max(
                0.0, _env_float("ROUTING_BUDGET_FUTURE_REACHABILITY_WEIGHT", 0.75)
            ),
            dual_learning_rate=max(
                0.0, _env_float("ROUTING_BUDGET_DUAL_LEARNING_RATE", 0.25)
            ),
            recovery_penalty=max(0.0, _env_float("ROUTING_BUDGET_RECOVERY_PENALTY", 4.0)),
            recovery_topk=max(1, _env_int("ROUTING_BUDGET_RECOVERY_TOPK", 2)),
            budget_tolerance=max(
                0.0, _env_float("ROUTING_BUDGET_BUDGET_TOLERANCE", 1.0e-6)
            ),
            controlled_recovery_enabled=_env_bool(
                "ROUTING_BUDGET_CONTROLLED_RECOVERY_ENABLE", True
            ),
            recovery_budget_total=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_BUDGET_TOTAL", 3.0)
            ),
            recovery_minimum_charge=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_MINIMUM_CHARGE", 0.05)
            ),
            recovery_maximum_charge_per_slot=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_MAXIMUM_CHARGE_PER_SLOT", 0.90)
            ),
            recovery_event_repeat_weight=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_EVENT_REPEAT_WEIGHT", 1.00)
            ),
            recovery_source_run_weight=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_SOURCE_RUN_WEIGHT", 0.65)
            ),
            recovery_source_share_weight=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_SOURCE_SHARE_WEIGHT", 0.45)
            ),
            recovery_family_share_weight=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_FAMILY_SHARE_WEIGHT", 0.55)
            ),
            recovery_observability_weight=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_OBSERVABILITY_WEIGHT", 0.50)
            ),
            recovery_hierarchy_weight=max(
                0.0, _env_float("ROUTING_BUDGET_RECOVERY_HIERARCHY_WEIGHT", 0.55)
            ),
            source_scarcity_enabled=_env_bool(
                "ROUTING_BUDGET_SOURCE_SCARCITY_ENABLE", True
            ),
            minimum_safe_source_count=max(
                1, _env_int("ROUTING_BUDGET_MINIMUM_SAFE_SOURCE_COUNT", 2)
            ),
            source_scarcity_minimum_scale=float(
                np.clip(
                    _env_float("ROUTING_BUDGET_SOURCE_SCARCITY_MINIMUM_SCALE", 0.12),
                    0.0,
                    1.0,
                )
            ),
            source_scarcity_budget_credit=max(
                0.0, _env_float("ROUTING_BUDGET_SOURCE_SCARCITY_BUDGET_CREDIT", 0.25)
            ),
            family_scarcity_enabled=_env_bool(
                "ROUTING_BUDGET_FAMILY_SCARCITY_ENABLE", True
            ),
            minimum_safe_family_count=max(
                1, _env_int("ROUTING_BUDGET_MINIMUM_SAFE_FAMILY_COUNT", 2)
            ),
            family_scarcity_minimum_scale=float(
                np.clip(
                    _env_float("ROUTING_BUDGET_FAMILY_SCARCITY_MINIMUM_SCALE", 0.12),
                    0.0,
                    1.0,
                )
            ),
            family_scarcity_budget_credit=max(
                0.0, _env_float("ROUTING_BUDGET_FAMILY_SCARCITY_BUDGET_CREDIT", 0.25)
            ),
            recovery_minimum_viability_depth=max(
                1, _env_int("ROUTING_BUDGET_RECOVERY_MINIMUM_VIABILITY_DEPTH", 2)
            ),
            recovery_require_safe_successor=_env_bool(
                "ROUTING_BUDGET_RECOVERY_REQUIRE_SAFE_SUCCESSOR", True
            ),
        )

    @property
    def budgets(self) -> tuple[float, ...]:
        return (
            self.event_repeat_budget,
            self.source_run_budget,
            self.source_share_budget,
            self.family_share_budget,
            self.observability_budget,
            self.hierarchy_repetition_budget,
        )

    @property
    def weights(self) -> tuple[float, ...]:
        return (
            self.event_repeat_weight,
            self.source_run_weight,
            self.source_share_weight,
            self.family_share_weight,
            self.observability_weight,
            self.hierarchy_repetition_weight,
        )

    @property
    def recovery_weights(self) -> tuple[float, ...]:
        return (
            self.recovery_event_repeat_weight,
            self.recovery_source_run_weight,
            self.recovery_source_share_weight,
            self.recovery_family_share_weight,
            self.recovery_observability_weight,
            self.recovery_hierarchy_weight,
        )

    def initial_state(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        zeros = tuple(0.0 for _ in CONSTRAINT_NAMES)
        return zeros, zeros

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["constraint_names"] = list(CONSTRAINT_NAMES)
        result["budgets"] = dict(zip(CONSTRAINT_NAMES, self.budgets))
        result["weights"] = dict(zip(CONSTRAINT_NAMES, self.weights))
        result["recovery_weights"] = dict(
            zip(CONSTRAINT_NAMES, self.recovery_weights)
        )
        return result


def _scarcity_scale(
    safe_counts: Counter,
    all_counts: Counter,
    minimum_count: int,
    minimum_scale: float,
    enabled: bool,
) -> tuple[bool, bool, float, float, float]:
    safe_count = len(safe_counts)
    safe_norm = _normalized_entropy(safe_counts)
    all_norm = _normalized_entropy(all_counts)
    alternative = safe_count >= int(minimum_count)
    scarcity = bool(enabled and bool(safe_counts) and not alternative)
    if scarcity:
        ratio = safe_norm / max(all_norm, _EPS) if all_norm > 0.0 else 0.0
        scale = max(float(minimum_scale), min(1.0, ratio))
    else:
        scale = 1.0
    return alternative, scarcity, float(scale), float(safe_norm), float(all_norm)


def build_feasible_set_scarcity_context(
    *,
    db: Mapping[str, Any],
    hard_safe_event_ids: Sequence[int],
    all_event_ids: Sequence[int],
    config: ConstraintBudgetConfig,
) -> FeasibleSetScarcityContext:
    """Build a joint Source--Family scarcity context from exact-safe candidates."""
    safe_ids = tuple(dict.fromkeys(map(int, hard_safe_event_ids)))
    all_ids = tuple(dict.fromkeys(map(int, all_event_ids)))
    safe_sources = Counter(event_identity(db, value)["source_uid"] for value in safe_ids)
    all_sources = Counter(event_identity(db, value)["source_uid"] for value in all_ids)
    safe_families = Counter(event_identity(db, value)["family_id"] for value in safe_ids)
    all_families = Counter(event_identity(db, value)["family_id"] for value in all_ids)

    source_alt, source_scarcity, source_scale, safe_source_norm, all_source_norm = _scarcity_scale(
        safe_sources,
        all_sources,
        config.minimum_safe_source_count,
        config.source_scarcity_minimum_scale,
        config.source_scarcity_enabled,
    )
    family_alt, family_scarcity, family_scale, safe_family_norm, all_family_norm = _scarcity_scale(
        safe_families,
        all_families,
        config.minimum_safe_family_count,
        config.family_scarcity_minimum_scale,
        config.family_scarcity_enabled,
    )
    return FeasibleSetScarcityContext(
        enabled=bool(config.source_scarcity_enabled or config.family_scarcity_enabled),
        hard_safe_event_ids=safe_ids,
        all_event_ids=all_ids,
        safe_source_count=len(safe_sources),
        all_source_count=len(all_sources),
        safe_family_count=len(safe_families),
        all_family_count=len(all_families),
        safe_source_entropy=_entropy_from_counts(safe_sources),
        all_source_entropy=_entropy_from_counts(all_sources),
        safe_family_entropy=_entropy_from_counts(safe_families),
        all_family_entropy=_entropy_from_counts(all_families),
        safe_source_entropy_normalized=safe_source_norm,
        all_source_entropy_normalized=all_source_norm,
        safe_family_entropy_normalized=safe_family_norm,
        all_family_entropy_normalized=all_family_norm,
        source_penalty_scale=source_scale,
        family_penalty_scale=family_scale,
        alternative_safe_source_exists=source_alt,
        alternative_safe_family_exists=family_alt,
        source_scarcity_exemption=source_scarcity,
        family_scarcity_exemption=family_scarcity,
        safe_source_counts=tuple(sorted(safe_sources.items())),
        all_source_counts=tuple(sorted(all_sources.items())),
        safe_family_counts=tuple(sorted(safe_families.items())),
        all_family_counts=tuple(sorted(all_families.items())),
    )


def build_source_scarcity_context(
    *,
    db: Mapping[str, Any],
    hard_safe_event_ids: Sequence[int],
    all_event_ids: Sequence[int],
    config: ConstraintBudgetConfig,
) -> FeasibleSetScarcityContext:
    """Compatibility wrapper for the former source-only API."""
    return build_feasible_set_scarcity_context(
        db=db,
        hard_safe_event_ids=hard_safe_event_ids,
        all_event_ids=all_event_ids,
        config=config,
    )


def _normalize_state(values: Sequence[float], length: int) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
    if len(array) == length:
        return array
    output = np.zeros(length, dtype=np.float64)
    output[: min(length, len(array))] = array[: min(length, len(array))]
    return output


def _hierarchical_repetition(
    db: Mapping[str, Any],
    event_id: int,
    selected_event_ids: Sequence[int],
    config: ConstraintBudgetConfig,
) -> tuple[float, float, list[dict[str, float]]]:
    if not selected_event_ids:
        return 0.0, float("inf"), []
    rows: list[dict[str, float]] = []
    similarities: list[float] = []
    weights: list[float] = []
    minimum_distance = float("inf")
    for age, previous in enumerate(reversed(tuple(selected_event_ids)), start=1):
        distance = event_hyperbolic_distance(
            db,
            int(event_id),
            int(previous),
            dimension=config.hyperbolic_dimension,
            maximum_radius=config.hyperbolic_maximum_radius,
        )
        similarity = math.exp(-distance / config.hierarchy_temperature)
        recency = math.exp(-(age - 1) / config.hierarchy_recency_decay)
        rows.append(
            {
                "event_id": float(int(previous)),
                "age": float(age),
                "distance": float(distance),
                "similarity": float(similarity),
                "recency_weight": float(recency),
            }
        )
        similarities.append(float(similarity))
        weights.append(float(recency))
        minimum_distance = min(minimum_distance, float(distance))
    weighted = float(np.average(similarities, weights=np.maximum(weights, _EPS)))
    return weighted, minimum_distance, rows


def assess_candidate_constraints(
    *,
    db: Mapping[str, Any],
    event_id: int,
    selected_event_ids: Sequence[int],
    observability: float,
    future_reachability_probability: float,
    slot_index: int,
    constraint_usage: Sequence[float],
    dual_variables: Sequence[float],
    config: ConstraintBudgetConfig,
    scarcity_context: Optional[FeasibleSetScarcityContext] = None,
) -> dict[str, Any]:
    """Evaluate one hard-safe candidate under stateful probabilistic budgets."""
    count = len(CONSTRAINT_NAMES)
    usage = _normalize_state(constraint_usage, count)
    duals = _normalize_state(dual_variables, count)
    identity = event_identity(db, int(event_id))
    history = [event_identity(db, int(value)) for value in selected_event_ids]

    repeat_gap = None
    for gap, row in enumerate(reversed(history), start=1):
        if row["event_uid"] == identity["event_uid"]:
            repeat_gap = gap
            break
    event_repeat = 0.0
    if repeat_gap is not None and repeat_gap <= config.event_cooldown_slots:
        event_repeat = (
            config.event_cooldown_slots - repeat_gap + 1
        ) / max(1.0, float(config.event_cooldown_slots))

    source_run = 0
    for row in reversed(history):
        if row["source_uid"] != identity["source_uid"]:
            break
        source_run += 1
    source_run_after = source_run + 1
    source_run_violation = max(
        0.0,
        (source_run_after - config.maximum_source_run)
        / max(1.0, float(config.maximum_source_run)),
    )

    source_counts = Counter(row["source_uid"] for row in history)
    family_counts = Counter(row["family_id"] for row in history)
    total_after = len(history) + 1
    source_share = (source_counts[identity["source_uid"]] + 1) / max(1, total_after)
    family_share = (family_counts[identity["family_id"]] + 1) / max(1, total_after)
    share_active = len(history) >= config.minimum_share_history
    source_share_violation = (
        max(0.0, source_share - config.maximum_source_share)
        / max(1.0 - config.maximum_source_share, _EPS)
        if share_active
        else 0.0
    )
    family_share_violation = (
        max(0.0, family_share - config.maximum_family_share)
        / max(1.0 - config.maximum_family_share, _EPS)
        if share_active
        else 0.0
    )

    obs = float(np.clip(observability, 0.0, 1.0))
    observability_violation = max(0.0, config.observability_target - obs) / max(
        config.observability_target, _EPS
    )
    hierarchy_similarity, minimum_distance, hierarchy_rows = _hierarchical_repetition(
        db,
        int(event_id),
        selected_event_ids,
        config,
    )
    hierarchy_violation = max(
        0.0, hierarchy_similarity - config.hierarchy_similarity_target
    ) / max(1.0 - config.hierarchy_similarity_target, _EPS)

    raw_violations = np.asarray(
        (
            event_repeat,
            source_run_violation,
            source_share_violation,
            family_share_violation,
            observability_violation,
            hierarchy_violation,
        ),
        dtype=np.float64,
    )
    context = scarcity_context or FeasibleSetScarcityContext()
    effective_violations = raw_violations.copy()
    source_scale = float(np.clip(context.source_penalty_scale, 0.0, 1.0))
    family_scale = float(np.clip(context.family_penalty_scale, 0.0, 1.0))
    if context.source_scarcity_exemption:
        effective_violations[1] *= source_scale
        effective_violations[2] *= source_scale
    if context.family_scarcity_exemption:
        effective_violations[3] *= family_scale
        effective_violations[5] *= family_scale

    usage_after = usage + effective_violations
    budgets = np.asarray(config.budgets, dtype=np.float64)
    effective_budgets = budgets.copy()
    if context.source_scarcity_exemption:
        credit = float(config.source_scarcity_budget_credit) * (1.0 - source_scale)
        effective_budgets[1] += credit
        effective_budgets[2] += credit
    if context.family_scarcity_exemption:
        credit = float(config.family_scarcity_budget_credit) * (1.0 - family_scale)
        effective_budgets[3] += credit
        effective_budgets[5] += credit
    weights = np.asarray(config.weights, dtype=np.float64)
    overrun = np.maximum(0.0, usage_after - effective_budgets)
    normalized_overrun = overrun / np.maximum(effective_budgets, 1.0)
    within_budget = bool(
        np.all(overrun <= float(config.budget_tolerance)) or not config.enabled
    )

    progress = float(np.clip((int(slot_index) + 1) / config.total_slots, 0.0, 1.0))
    progress_budget = effective_budgets * progress
    duals_after = np.maximum(
        0.0,
        duals + config.dual_learning_rate * (usage_after - progress_budget),
    )

    preference_probabilities = np.exp(-effective_violations)
    future_probability = float(np.clip(future_reachability_probability, _EPS, 1.0))
    diversity_penalty = float(
        np.dot(weights[[0, 1, 2, 3, 5]], effective_violations[[0, 1, 2, 3, 5]])
    )
    probabilistic_auxiliary_cost = float(
        weights[4] * effective_violations[4]
        + np.dot(duals, effective_violations)
        - config.future_reachability_weight * math.log(future_probability)
    )

    recovery_weights = np.asarray(config.recovery_weights, dtype=np.float64)
    recovery_charge_raw = float(np.dot(recovery_weights, normalized_overrun))
    recovery_charge = 0.0
    if np.any(overrun > float(config.budget_tolerance)):
        recovery_charge = max(config.recovery_minimum_charge, recovery_charge_raw)
        recovery_charge = min(config.recovery_maximum_charge_per_slot, recovery_charge)
    recovery_score = float(
        probabilistic_auxiliary_cost
        + diversity_penalty
        + config.recovery_penalty * recovery_charge
    )

    soft_reasons = [
        name
        for name, value in zip(CONSTRAINT_NAMES, raw_violations)
        if float(value) > 0.0
    ]
    budget_overrun_reasons = [
        name for name, value in zip(CONSTRAINT_NAMES, overrun) if float(value) > 0.0
    ]
    diversity = {
        **identity,
        "hard_valid": True,
        "hard_reasons": [],
        "soft_reasons": list(soft_reasons),
        "penalty": diversity_penalty,
        "cooldown_slots": int(config.event_cooldown_slots),
        "event_repeat_gap": repeat_gap,
        "source_run_after": int(source_run_after),
        "source_share_after": float(source_share),
        "family_share_after": float(family_share),
        "hierarchical_similarity": float(hierarchy_similarity),
        "event_hyperbolic_distance_min": (
            None if not np.isfinite(minimum_distance) else float(minimum_distance)
        ),
    }
    return {
        "schema": "stateful_hierarchical_probabilistic_constraint_assessment",
        "enabled": bool(config.enabled),
        "identity": identity,
        "diversity": diversity,
        "raw_violations": dict(zip(CONSTRAINT_NAMES, map(float, raw_violations))),
        "violations": dict(
            zip(CONSTRAINT_NAMES, map(float, effective_violations))
        ),
        "preference_probabilities": dict(
            zip(CONSTRAINT_NAMES, map(float, preference_probabilities))
        ),
        "source_scarcity": context.to_dict(),
        "feasible_set_scarcity": context.to_dict(),
        "future_reachability_probability": future_probability,
        "within_budget": within_budget,
        "constraint_usage_before": dict(zip(CONSTRAINT_NAMES, map(float, usage))),
        "constraint_usage_after": dict(
            zip(CONSTRAINT_NAMES, map(float, usage_after))
        ),
        "budget": dict(zip(CONSTRAINT_NAMES, map(float, budgets))),
        "effective_budget": dict(
            zip(CONSTRAINT_NAMES, map(float, effective_budgets))
        ),
        "budget_overrun": dict(zip(CONSTRAINT_NAMES, map(float, overrun))),
        "budget_overrun_reasons": budget_overrun_reasons,
        "dual_variables_before": dict(zip(CONSTRAINT_NAMES, map(float, duals))),
        "dual_variables_after": dict(
            zip(CONSTRAINT_NAMES, map(float, duals_after))
        ),
        "probabilistic_auxiliary_cost": probabilistic_auxiliary_cost,
        "diversity_penalty": diversity_penalty,
        "recovery_score": recovery_score,
        "recovery_charge": float(recovery_charge),
        "hierarchy_comparisons": hierarchy_rows,
        "usage_after_tuple": tuple(map(float, usage_after)),
        "duals_after_tuple": tuple(map(float, duals_after)),
    }


def select_controlled_recovery_indices(
    evaluated_rows: Sequence[Mapping[str, Any]],
    *,
    current_recovery_budget_used: float = 0.0,
    config: ConstraintBudgetConfig,
    current_recoveries: Optional[int] = None,
) -> set[int]:
    """Select hard-safe recovery branches using graded future viability.

    Terminal reachability is preferred, but a candidate with safe successors and
    sufficient viability depth may consume bounded recovery resource.  Immediate
    dead ends remain ineligible, and immutable safety gates are never relaxed.
    """
    del current_recoveries
    if not config.enabled or not config.controlled_recovery_enabled:
        return set()
    if any(
        bool(row.get("hard_safe", False))
        and bool(row.get("preferred", False))
        for row in evaluated_rows
    ):
        return set()
    remaining = max(0.0, config.recovery_budget_total - float(current_recovery_budget_used))
    if remaining <= config.budget_tolerance:
        return set()

    candidates: list[tuple[int, int, int, float, float, int]] = []
    for index, row in enumerate(evaluated_rows):
        if not bool(row.get("hard_safe", False)) or bool(row.get("preferred", False)):
            continue
        reachability = row.get("future_reachability", {})
        terminal = bool(
            reachability.get(
                "terminal_reachable",
                reachability.get("future_reachable", False),
            )
        )
        viability_depth = int(reachability.get("future_viability_depth", 0) or 0)
        reachable_until = int(
            reachability.get("reachable_until_slot", -1)
            if reachability.get("reachable_until_slot") is not None
            else -1
        )
        successor_count = int(
            reachability.get("future_safe_successor_count", 0) or 0
        )
        if not terminal:
            if viability_depth < int(config.recovery_minimum_viability_depth):
                continue
            if config.recovery_require_safe_successor and successor_count <= 0:
                continue
        assessment = row.get("constraint_assessment", {})
        charge = float(assessment.get("recovery_charge", float("inf")))
        if not np.isfinite(charge) or charge > remaining + config.budget_tolerance:
            continue
        if charge > config.recovery_maximum_charge_per_slot + config.budget_tolerance:
            continue
        score = float(assessment.get("recovery_score", float("inf")))
        candidates.append(
            (
                0 if terminal else 1,
                -viability_depth,
                -reachable_until,
                charge,
                score,
                int(index),
            )
        )
    candidates.sort()
    return {row[5] for row in candidates[: config.recovery_topk]}
