"""Posterior-guided dynamic route utilities for whole-song Event assembly.

This module is intentionally small and dependency-light.  It connects the
static whole-song Graph-SB prior with the authoritative closed-loop simulator
without moving geometry, grounding, or physical contracts out of their current
modules.

The Graph-SB solver registers node/transition posteriors here.  The heading
assembler then uses those probabilities as a *soft prior* while retaining
multiple exact-simulation branches.  Anatomy, heading, history diversity, and
severe physical gates remain hard constraints.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_EPS = 1.0e-12
_ROUTE_PRIOR: Optional["RoutePosterior"] = None


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
        values = np.asarray(db[key])
        value = values[int(event_id)]
        return value.item() if isinstance(value, np.generic) else value
    except Exception:
        return default


@dataclass(frozen=True)
class RoutePosterior:
    """Process-local Graph-SB posterior aligned to Event IDs."""

    layers: tuple[tuple[int, ...], ...]
    node_marginals: tuple[np.ndarray, ...] = ()
    transition_marginals: tuple[np.ndarray, ...] = ()
    chosen_path: tuple[int, ...] = ()
    source: str = "candidate_rank"
    path_entropy: float = 0.0
    _local_index: tuple[dict[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_local_index",
            tuple(
                {int(event_id): int(index) for index, event_id in enumerate(layer)}
                for layer in self.layers
            ),
        )

    def node_probability(self, slot: int, event_id: int) -> Optional[float]:
        if not 0 <= int(slot) < len(self.node_marginals):
            return None
        local = self._local_index[int(slot)].get(int(event_id))
        if local is None:
            return None
        values = np.asarray(self.node_marginals[int(slot)], dtype=np.float64)
        if not 0 <= local < len(values):
            return None
        value = float(values[local])
        return value if np.isfinite(value) and value > 0.0 else None

    def transition_probability(
        self,
        slot: int,
        previous_event_id: Optional[int],
        event_id: int,
    ) -> Optional[float]:
        if previous_event_id is None or int(slot) <= 0:
            return None
        edge_slot = int(slot) - 1
        if not 0 <= edge_slot < len(self.transition_marginals):
            return None
        left = self._local_index[edge_slot].get(int(previous_event_id))
        right = self._local_index[int(slot)].get(int(event_id))
        if left is None or right is None:
            return None
        values = np.asarray(
            self.transition_marginals[edge_slot], dtype=np.float64
        )
        if values.ndim != 2 or left >= values.shape[0] or right >= values.shape[1]:
            return None
        value = float(values[left, right])
        return value if np.isfinite(value) and value > 0.0 else None


@dataclass(frozen=True)
class DynamicSearchConfig:
    """Runtime search policy; hard contracts are deliberately not represented."""

    beam_width: int = 6
    maximum_beam_width: int = 12
    branch_topk: int = 16
    candidates_per_source: int = 2
    primary_bonus: float = 0.18
    posterior_weight: float = 0.35
    uncertainty_weight: float = 0.20
    source_calibration_weight: float = 0.15
    minimum_observability: float = 0.0

    @classmethod
    def from_environment(cls) -> "DynamicSearchConfig":
        beam = max(1, _env_int("V46_50_DYNAMIC_BEAM_WIDTH", 6))
        maximum = max(beam, _env_int("V46_50_DYNAMIC_BEAM_MAX", 12))
        return cls(
            beam_width=beam,
            maximum_beam_width=maximum,
            branch_topk=max(1, _env_int("V46_50_DYNAMIC_BRANCH_TOPK", 16)),
            candidates_per_source=max(
                1, _env_int("V46_50_DYNAMIC_MIN_PER_SOURCE", 2)
            ),
            primary_bonus=max(
                0.0, _env_float("V46_54_PRIMARY_EVENT_BONUS", 0.18)
            ),
            posterior_weight=max(
                0.0, _env_float("V46_50_POSTERIOR_WEIGHT", 0.35)
            ),
            uncertainty_weight=max(
                0.0, _env_float("V46_50_UNCERTAINTY_WEIGHT", 0.20)
            ),
            source_calibration_weight=max(
                0.0, _env_float("V46_50_SOURCE_CALIBRATION_WEIGHT", 0.15)
            ),
            minimum_observability=float(
                np.clip(
                    _env_float("V46_50_DYNAMIC_OBSERVABILITY_MIN", 0.0),
                    0.0,
                    1.0,
                )
            ),
        )


@dataclass(frozen=True)
class DynamicBeamState:
    """One exact-simulation prefix retained by the dynamic assembler."""

    motion: np.ndarray
    selected_event_ids: tuple[int, ...] = ()
    selected_ranks: tuple[int, ...] = ()
    report: tuple[dict[str, Any], ...] = ()
    state_trace: tuple[dict[str, Any], ...] = ()
    stage_heading: float = 0.0
    recent_turn_count: int = 0
    cumulative_abs_yaw: float = 0.0
    score: float = 0.0
    observability: float = 1.0


class DynamicRouteDeadEnd(RuntimeError):
    """Raised when every retained exact-simulation state is blocked."""

    def __init__(self, slot: int, diagnostics: Mapping[str, Any]):
        self.slot = int(slot)
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Dynamic route dead-end at slot "
            f"{self.slot}: {self.diagnostics}"
        )


def clear_route_prior() -> None:
    global _ROUTE_PRIOR
    _ROUTE_PRIOR = None


def register_route_prior(
    layers: Sequence[Sequence[int]],
    *,
    node_marginals: Optional[Sequence[np.ndarray]] = None,
    transition_marginals: Optional[Sequence[np.ndarray]] = None,
    chosen_path: Optional[Sequence[int]] = None,
    source: str,
    path_entropy: float = 0.0,
) -> None:
    """Register one song's route posterior for the downstream exact simulator."""

    global _ROUTE_PRIOR
    normalized_layers = tuple(tuple(map(int, layer)) for layer in layers)
    nodes = tuple(
        np.asarray(value, dtype=np.float64).copy()
        for value in (node_marginals or ())
    )
    transitions = tuple(
        np.asarray(value, dtype=np.float64).copy()
        for value in (transition_marginals or ())
    )
    _ROUTE_PRIOR = RoutePosterior(
        layers=normalized_layers,
        node_marginals=nodes,
        transition_marginals=transitions,
        chosen_path=tuple(map(int, chosen_path or ())),
        source=str(source),
        path_entropy=float(path_entropy),
    )


def route_prior_summary() -> dict[str, Any]:
    prior = _ROUTE_PRIOR
    if prior is None:
        return {
            "available": False,
            "source": "candidate_rank",
            "path_entropy": None,
        }
    return {
        "available": True,
        "source": prior.source,
        "slots": len(prior.layers),
        "has_node_marginals": bool(prior.node_marginals),
        "has_transition_marginals": bool(prior.transition_marginals),
        "path_entropy": float(prior.path_entropy),
    }


def route_prior_cost(
    slot: int,
    event_id: int,
    *,
    previous_event_id: Optional[int],
    fallback_rank: int,
    candidate_count: int,
) -> tuple[float, dict[str, Any]]:
    """Return a finite negative-log prior; rank is used only as fallback."""

    prior = _ROUTE_PRIOR
    node_probability: Optional[float] = None
    transition_probability: Optional[float] = None
    chosen_match = False
    source = "candidate_rank"

    if prior is not None:
        source = prior.source
        node_probability = prior.node_probability(slot, event_id)
        transition_probability = prior.transition_probability(
            slot, previous_event_id, event_id
        )
        chosen_match = bool(
            0 <= int(slot) < len(prior.chosen_path)
            and int(prior.chosen_path[int(slot)]) == int(event_id)
        )

    probabilities = [
        value
        for value in (node_probability, transition_probability)
        if value is not None
    ]
    if probabilities:
        probability = float(np.exp(np.mean(np.log(np.maximum(probabilities, _EPS)))))
        cost = -math.log(max(probability, _EPS))
    else:
        count = max(1, int(candidate_count))
        probability = math.exp(-int(fallback_rank) / max(1.0, count / 4.0))
        cost = -math.log(max(probability, _EPS))

    return float(cost), {
        "source": source,
        "node_probability": node_probability,
        "transition_probability": transition_probability,
        "fallback_rank": int(fallback_rank),
        "chosen_path_match": chosen_match,
        "negative_log_cost": float(cost),
    }


def candidate_subset(
    candidates: Sequence[int],
    db: Mapping[str, Any],
    *,
    limit: int,
    minimum_per_source: int,
    primary_event_id: Optional[int],
) -> list[int]:
    """Take a rank-preserving but source-covered candidate subset.

    A pure top-k can remove every alternative source before the exact simulator
    runs.  This function keeps the ranking prior while reserving a small number
    of entries for each source represented in the pool.
    """

    ordered: list[int] = []
    seen: set[int] = set()
    for raw in candidates:
        event_id = int(raw)
        if event_id not in seen:
            ordered.append(event_id)
            seen.add(event_id)
    if not ordered:
        return []

    cap = max(1, min(int(limit), len(ordered)))
    result: list[int] = []
    selected: set[int] = set()

    selected_per_source: dict[str, int] = {}
    if primary_event_id is not None and int(primary_event_id) in seen:
        primary = int(primary_event_id)
        result.append(primary)
        selected.add(primary)
        primary_source = str(_db_value(db, "source_uids", primary, "unknown"))
        selected_per_source[primary_source] = 1

    source_rows: dict[str, list[int]] = {}
    for event_id in ordered:
        source = str(_db_value(db, "source_uids", event_id, "unknown"))
        source_rows.setdefault(source, []).append(event_id)

    quota = max(1, int(minimum_per_source))
    for source in source_rows:
        taken = selected_per_source.get(source, 0)
        for event_id in source_rows[source]:
            if event_id in selected:
                continue
            if taken >= quota or len(result) >= cap:
                break
            result.append(event_id)
            selected.add(event_id)
            taken += 1
        selected_per_source[source] = taken
        if len(result) >= cap:
            break

    for event_id in ordered:
        if len(result) >= cap:
            break
        if event_id not in selected:
            result.append(event_id)
            selected.add(event_id)
    return result


def observability_from_extra(
    extra: Mapping[str, Any],
    *,
    safe_ratio: float,
) -> float:
    grounding = extra.get("v46_53_grounding", {})
    value = grounding.get("observability") if isinstance(grounding, Mapping) else None
    if value is None:
        value = 0.5 + 0.5 * float(np.clip(safe_ratio, 0.0, 1.0))
    return float(np.clip(0.75 * float(value) + 0.25 * safe_ratio, 0.0, 1.0))


def source_calibration_penalty(
    db: Mapping[str, Any],
    previous_event_id: Optional[int],
    event_id: int,
) -> tuple[float, dict[str, Any]]:
    """Use optional source-to-source safety calibration without inventing it.

    If the Event-DB has no calibrated source transition matrix, this term is
    exactly zero.  The authoritative exact simulator still decides safety.
    """

    if previous_event_id is None:
        return 0.0, {"available": False, "reason": "first_slot"}
    previous_source = str(
        _db_value(db, "source_uids", int(previous_event_id), "unknown")
    )
    current_source = str(_db_value(db, "source_uids", int(event_id), "unknown"))
    if previous_source == current_source:
        return 0.0, {
            "available": False,
            "reason": "same_source",
            "from": previous_source,
            "to": current_source,
        }

    labels = db.get("source_transition_labels")
    matrix = db.get("source_transition_safe_rate")
    if labels is None or matrix is None:
        return 0.0, {
            "available": False,
            "reason": "missing_calibration",
            "from": previous_source,
            "to": current_source,
        }
    labels_list = [str(value) for value in np.asarray(labels, dtype=object).tolist()]
    if previous_source not in labels_list or current_source not in labels_list:
        return 0.0, {
            "available": False,
            "reason": "source_not_calibrated",
            "from": previous_source,
            "to": current_source,
        }
    values = np.asarray(matrix, dtype=np.float64)
    left = labels_list.index(previous_source)
    right = labels_list.index(current_source)
    if values.ndim != 2 or left >= values.shape[0] or right >= values.shape[1]:
        return 0.0, {"available": False, "reason": "invalid_calibration_shape"}
    safe_rate = float(np.clip(values[left, right], _EPS, 1.0))
    penalty = -math.log(safe_rate)
    return float(penalty), {
        "available": True,
        "from": previous_source,
        "to": current_source,
        "safe_rate": safe_rate,
        "negative_log_penalty": float(penalty),
    }


def adaptive_beam_width(
    config: DynamicSearchConfig,
    observabilities: Sequence[float],
) -> int:
    if not observabilities:
        return config.maximum_beam_width
    mean_observability = float(
        np.clip(np.mean(np.asarray(observabilities, dtype=np.float64)), 0.0, 1.0)
    )
    width = int(
        round(
            config.beam_width
            + (config.maximum_beam_width - config.beam_width)
            * (1.0 - mean_observability)
        )
    )
    return max(config.beam_width, min(config.maximum_beam_width, width))


def prune_states(
    states: Sequence[DynamicBeamState],
    db: Mapping[str, Any],
    *,
    width: int,
) -> list[DynamicBeamState]:
    """Score-prune while preserving at least one current-source branch."""

    ordered = sorted(states, key=lambda state: float(state.score))
    cap = max(1, min(int(width), len(ordered)))
    selected: list[DynamicBeamState] = []
    selected_ids: set[int] = set()
    best_by_source: dict[str, DynamicBeamState] = {}
    for state in ordered:
        if not state.selected_event_ids:
            source = "initial"
        else:
            source = str(
                _db_value(db, "source_uids", state.selected_event_ids[-1], "unknown")
            )
        best_by_source.setdefault(source, state)

    for state in sorted(best_by_source.values(), key=lambda item: float(item.score)):
        if len(selected) >= cap:
            break
        selected.append(state)
        selected_ids.add(id(state))
    for state in ordered:
        if len(selected) >= cap:
            break
        if id(state) not in selected_ids:
            selected.append(state)
            selected_ids.add(id(state))
    return selected
