"""State-aware backward reachability for whole-song Event planning.

The base candidate graph provides a static suffix prior.  Query-time reachability
then augments each node with the retained route history, accumulated constraint
usage, dual variables and remaining continuous recovery resource.  The model is a
bounded surrogate; exact transition simulation remains authoritative.
"""
from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from routing.hierarchical_constraint_model import (
    ConstraintBudgetConfig,
    SourceScarcityContext,
    assess_candidate_constraints,
    build_source_scarcity_context,
    event_hyperbolic_distance,
    event_identity,
)

_EPS = 1.0e-12


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


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


def _logsumexp(values: Sequence[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("-inf")
    maximum = float(finite.max())
    return float(maximum + np.log(np.exp(finite - maximum).sum()))


def _quantized(values: Sequence[float], step: float) -> tuple[int, ...]:
    scale = max(float(step), 1.0e-6)
    array = np.asarray(tuple(values), dtype=np.float64)
    return tuple(map(int, np.rint(array / scale)))


@dataclass(frozen=True)
class ReachabilityConfig:
    hierarchy_weight: float
    same_event_penalty: float
    same_source_penalty: float
    same_family_penalty: float
    source_transition_weight: float
    candidate_rank_weight: float
    maximum_edge_cost: float
    anatomy_quality_minimum: float
    successor_probability_floor: float
    state_horizon: int
    state_branch_topk: int
    state_history_window: int
    state_usage_quantization: float
    state_recovery_quantization: float
    state_surrogate_observability: float
    state_cache_maximum_entries: int

    @classmethod
    def from_environment(cls) -> "ReachabilityConfig":
        return cls(
            hierarchy_weight=max(
                0.0, _env_float("BR_HPR_REACHABILITY_HIERARCHY_WEIGHT", 0.55)
            ),
            same_event_penalty=max(
                0.0, _env_float("BR_HPR_REACHABILITY_SAME_EVENT_PENALTY", 1.20)
            ),
            same_source_penalty=max(
                0.0, _env_float("BR_HPR_REACHABILITY_SAME_SOURCE_PENALTY", 0.30)
            ),
            same_family_penalty=max(
                0.0, _env_float("BR_HPR_REACHABILITY_SAME_FAMILY_PENALTY", 0.20)
            ),
            source_transition_weight=max(
                0.0,
                _env_float("BR_HPR_REACHABILITY_SOURCE_TRANSITION_WEIGHT", 0.35),
            ),
            candidate_rank_weight=max(
                0.0, _env_float("BR_HPR_REACHABILITY_RANK_WEIGHT", 0.02)
            ),
            maximum_edge_cost=max(
                0.1, _env_float("BR_HPR_REACHABILITY_MAXIMUM_EDGE_COST", 8.0)
            ),
            anatomy_quality_minimum=float(
                np.clip(
                    _env_float("BR_HPR_REACHABILITY_ANATOMY_QUALITY_MIN", 0.30),
                    0.0,
                    1.0,
                )
            ),
            successor_probability_floor=float(
                np.clip(
                    _env_float("BR_HPR_REACHABILITY_PROBABILITY_FLOOR", 1.0e-6),
                    1.0e-12,
                    1.0,
                )
            ),
            state_horizon=max(
                1, _env_int("BR_HPR_STATE_REACHABILITY_HORIZON", 4)
            ),
            state_branch_topk=max(
                1, _env_int("BR_HPR_STATE_REACHABILITY_BRANCH_TOPK", 6)
            ),
            state_history_window=max(
                1, _env_int("BR_HPR_STATE_REACHABILITY_HISTORY_WINDOW", 8)
            ),
            state_usage_quantization=max(
                1.0e-4,
                _env_float("BR_HPR_STATE_REACHABILITY_USAGE_QUANTIZATION", 0.05),
            ),
            state_recovery_quantization=max(
                1.0e-4,
                _env_float("BR_HPR_STATE_REACHABILITY_RECOVERY_QUANTIZATION", 0.05),
            ),
            state_surrogate_observability=float(
                np.clip(
                    _env_float("BR_HPR_STATE_REACHABILITY_OBSERVABILITY", 0.70),
                    0.0,
                    1.0,
                )
            ),
            state_cache_maximum_entries=max(
                128,
                _env_int("BR_HPR_STATE_REACHABILITY_CACHE_MAXIMUM_ENTRIES", 20000),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class BackwardReachabilityModel:
    """Static suffix graph plus bounded state-conditioned suffix queries."""

    def __init__(
        self,
        *,
        layers: tuple[tuple[int, ...], ...],
        db: Mapping[str, Any],
        constraint_config: ConstraintBudgetConfig,
        config: ReachabilityConfig,
        records: Mapping[tuple[int, int], Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> None:
        self.layers = layers
        self.db = db
        self.constraint_config = constraint_config
        self.config = config
        self.records = {key: dict(value) for key, value in records.items()}
        self.summary = dict(summary)
        self._edge_cache: dict[tuple[int, int], float] = {}
        self._state_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._state_queries = 0
        self._state_cache_lookups = 0
        self._state_cache_hits = 0

    @staticmethod
    def _static_valid(
        db: Mapping[str, Any],
        event_id: int,
        config: ReachabilityConfig,
    ) -> tuple[bool, dict[str, Any]]:
        try:
            anatomy_valid = bool(
                np.asarray(db["anatomy_valid"], dtype=bool)[int(event_id)]
            )
        except Exception:
            anatomy_valid = False
        try:
            heading_valid = bool(
                np.asarray(db["event_heading_valid"], dtype=bool)[int(event_id)]
            )
        except Exception:
            heading_valid = False
        try:
            anatomy_quality = float(
                np.asarray(db["anatomy_quality"], dtype=np.float64)[int(event_id)]
            )
        except Exception:
            anatomy_quality = 0.0
        path = str(_db_value(db, "paths", event_id, ""))
        valid = bool(
            anatomy_valid
            and heading_valid
            and anatomy_quality >= config.anatomy_quality_minimum
            and path
        )
        return valid, {
            "anatomy_valid": anatomy_valid,
            "heading_valid": heading_valid,
            "anatomy_quality": float(anatomy_quality),
            "path_available": bool(path),
        }

    @staticmethod
    def _source_transition_penalty(
        db: Mapping[str, Any],
        left_event: int,
        right_event: int,
    ) -> tuple[float, dict[str, Any]]:
        left_source = event_identity(db, left_event)["source_uid"]
        right_source = event_identity(db, right_event)["source_uid"]
        if left_source == right_source:
            return 0.0, {"available": False, "reason": "same_source"}
        labels = db.get("source_transition_labels")
        matrix = db.get("source_transition_safe_rate")
        if labels is None or matrix is None:
            return 0.0, {"available": False, "reason": "missing_calibration"}
        label_list = [str(value) for value in np.asarray(labels, dtype=object).tolist()]
        if left_source not in label_list or right_source not in label_list:
            return 0.0, {"available": False, "reason": "source_not_calibrated"}
        values = np.asarray(matrix, dtype=np.float64)
        left = label_list.index(left_source)
        right = label_list.index(right_source)
        if values.ndim != 2 or left >= values.shape[0] or right >= values.shape[1]:
            return 0.0, {"available": False, "reason": "invalid_shape"}
        safe_rate = float(np.clip(values[left, right], _EPS, 1.0))
        return -math.log(safe_rate), {
            "available": True,
            "safe_rate": safe_rate,
            "from": left_source,
            "to": right_source,
        }

    @classmethod
    def build(
        cls,
        candidate_lists: Sequence[Sequence[int]],
        db: Mapping[str, Any],
        *,
        constraint_config: ConstraintBudgetConfig,
        additional_candidate_layers: Optional[Sequence[Sequence[int]]] = None,
    ) -> "BackwardReachabilityModel":
        config = ReachabilityConfig.from_environment()
        total_events = len(np.asarray(db["paths"], dtype=object))
        layers_list: list[tuple[int, ...]] = []
        for slot, candidates0 in enumerate(candidate_lists):
            merged = list(map(int, candidates0))
            if additional_candidate_layers is not None and slot < len(additional_candidate_layers):
                merged.extend(map(int, additional_candidate_layers[slot]))
            layer = tuple(
                dict.fromkeys(
                    value for value in merged if 0 <= int(value) < total_events
                )
            )
            layers_list.append(layer)
        layers = tuple(layers_list)
        static: dict[tuple[int, int], tuple[bool, dict[str, Any]]] = {}
        for slot, layer in enumerate(layers):
            for event_id in layer:
                static[(slot, event_id)] = cls._static_valid(db, event_id, config)

        log_mass: dict[tuple[int, int], float] = {}
        successor_count: dict[tuple[int, int], int] = {}
        best_successor: dict[tuple[int, int], Optional[int]] = {}
        edge_minimum: dict[tuple[int, int], Optional[float]] = {}
        if layers:
            final_slot = len(layers) - 1
            for event_id in layers[final_slot]:
                valid, _detail = static[(final_slot, event_id)]
                log_mass[(final_slot, event_id)] = 0.0 if valid else float("-inf")
                successor_count[(final_slot, event_id)] = 0
                best_successor[(final_slot, event_id)] = None
                edge_minimum[(final_slot, event_id)] = None

        for slot in range(len(layers) - 2, -1, -1):
            next_layer = layers[slot + 1]
            rank_lookup = {event_id: rank for rank, event_id in enumerate(next_layer)}
            for event_id in layers[slot]:
                valid, _detail = static[(slot, event_id)]
                if not valid:
                    log_mass[(slot, event_id)] = float("-inf")
                    successor_count[(slot, event_id)] = 0
                    best_successor[(slot, event_id)] = None
                    edge_minimum[(slot, event_id)] = None
                    continue
                left_identity = event_identity(db, event_id)
                terms: list[tuple[float, int, float]] = []
                for next_event in next_layer:
                    next_valid, _next_detail = static[(slot + 1, next_event)]
                    next_mass = log_mass.get((slot + 1, next_event), float("-inf"))
                    if not next_valid or not np.isfinite(next_mass):
                        continue
                    right_identity = event_identity(db, next_event)
                    distance = event_hyperbolic_distance(
                        db,
                        event_id,
                        next_event,
                        dimension=constraint_config.hyperbolic_dimension,
                        maximum_radius=constraint_config.hyperbolic_maximum_radius,
                    )
                    calibration, _detail = cls._source_transition_penalty(
                        db, event_id, next_event
                    )
                    edge_cost = config.hierarchy_weight * distance
                    if left_identity["event_uid"] == right_identity["event_uid"]:
                        edge_cost += config.same_event_penalty
                    if left_identity["source_uid"] == right_identity["source_uid"]:
                        edge_cost += config.same_source_penalty
                    if left_identity["family_id"] == right_identity["family_id"]:
                        edge_cost += config.same_family_penalty
                    edge_cost += config.source_transition_weight * calibration
                    edge_cost += config.candidate_rank_weight * rank_lookup[next_event]
                    if edge_cost > config.maximum_edge_cost:
                        continue
                    terms.append((-float(edge_cost) + float(next_mass), next_event, edge_cost))
                log_mass[(slot, event_id)] = _logsumexp([row[0] for row in terms])
                successor_count[(slot, event_id)] = len(terms)
                if terms:
                    best = max(terms, key=lambda row: row[0])
                    best_successor[(slot, event_id)] = int(best[1])
                    edge_minimum[(slot, event_id)] = float(min(row[2] for row in terms))
                else:
                    best_successor[(slot, event_id)] = None
                    edge_minimum[(slot, event_id)] = None

        records: dict[tuple[int, int], dict[str, Any]] = {}
        valid_counts: list[int] = []
        reachable_counts: list[int] = []
        for slot, layer in enumerate(layers):
            finite_values = [
                log_mass.get((slot, event_id), float("-inf"))
                for event_id in layer
                if np.isfinite(log_mass.get((slot, event_id), float("-inf")))
            ]
            maximum = max(finite_values) if finite_values else float("-inf")
            valid_count = 0
            reachable_count = 0
            for event_id in layer:
                valid, validity_detail = static[(slot, event_id)]
                valid_count += int(valid)
                value = log_mass.get((slot, event_id), float("-inf"))
                reachable = bool(np.isfinite(value))
                reachable_count += int(reachable)
                probability = (
                    float(np.exp(value - maximum))
                    if reachable and np.isfinite(maximum)
                    else 0.0
                )
                if reachable:
                    probability = max(config.successor_probability_floor, probability)
                records[(slot, event_id)] = {
                    "schema": "backward_candidate_graph_reachability",
                    "slot": int(slot),
                    "event_id": int(event_id),
                    "static_valid": bool(valid),
                    "static_validity": validity_detail,
                    "future_reachable": reachable,
                    "future_log_mass": None if not reachable else float(value),
                    "future_reachability_probability": float(
                        np.clip(probability, 0.0, 1.0)
                    ),
                    "future_safe_successor_count": int(
                        successor_count.get((slot, event_id), 0)
                    ),
                    "best_future_successor_event_id": best_successor.get(
                        (slot, event_id)
                    ),
                    "minimum_surrogate_edge_cost": edge_minimum.get(
                        (slot, event_id)
                    ),
                }
            valid_counts.append(valid_count)
            reachable_counts.append(reachable_count)
        summary = {
            "schema": "state_aware_backward_candidate_graph_summary",
            "slots": int(len(layers)),
            "layer_sizes": [int(len(layer)) for layer in layers],
            "static_valid_counts": valid_counts,
            "future_reachable_counts": reachable_counts,
            "configuration": config.to_dict(),
            "state_conditioning": {
                "history": True,
                "constraint_usage": True,
                "dual_variables": True,
                "continuous_recovery_budget": True,
            },
        }
        return cls(
            layers=layers,
            db=db,
            constraint_config=constraint_config,
            config=config,
            records=records,
            summary=summary,
        )

    def get(self, slot: int, event_id: int) -> dict[str, Any]:
        record = self.records.get((int(slot), int(event_id)))
        if record is not None:
            return dict(record)
        valid, detail = self._static_valid(self.db, int(event_id), self.config)
        return {
            "schema": "backward_candidate_graph_reachability",
            "slot": int(slot),
            "event_id": int(event_id),
            "static_valid": bool(valid),
            "static_validity": detail,
            "future_reachable": bool(valid and int(slot) >= len(self.layers) - 1),
            "future_log_mass": 0.0 if valid else None,
            "future_reachability_probability": 1.0 if valid else 0.0,
            "future_safe_successor_count": 0,
            "best_future_successor_event_id": None,
            "minimum_surrogate_edge_cost": None,
        }

    def _edge_cost(self, left_event: int, right_event: int, rank: int) -> float:
        key = (int(left_event), int(right_event))
        base = self._edge_cache.get(key)
        if base is None:
            left = event_identity(self.db, int(left_event))
            right = event_identity(self.db, int(right_event))
            distance = event_hyperbolic_distance(
                self.db,
                int(left_event),
                int(right_event),
                dimension=self.constraint_config.hyperbolic_dimension,
                maximum_radius=self.constraint_config.hyperbolic_maximum_radius,
            )
            calibration, _detail = self._source_transition_penalty(
                self.db, int(left_event), int(right_event)
            )
            base = self.config.hierarchy_weight * distance
            if left["event_uid"] == right["event_uid"]:
                base += self.config.same_event_penalty
            if left["source_uid"] == right["source_uid"]:
                base += self.config.same_source_penalty
            if left["family_id"] == right["family_id"]:
                base += self.config.same_family_penalty
            base += self.config.source_transition_weight * calibration
            self._edge_cache[key] = float(base)
        return float(base + self.config.candidate_rank_weight * int(rank))

    def _cache_key(
        self,
        *,
        slot: int,
        previous_event_id: int,
        selected_event_ids: Sequence[int],
        usage: Sequence[float],
        duals: Sequence[float],
        recovery_budget_used: float,
        depth: int,
    ) -> tuple[Any, ...]:
        history = tuple(map(int, selected_event_ids))[-self.config.state_history_window :]
        return (
            int(slot),
            int(previous_event_id),
            history,
            _quantized(usage, self.config.state_usage_quantization),
            _quantized(duals, self.config.state_usage_quantization),
            int(round(float(recovery_budget_used) / self.config.state_recovery_quantization)),
            int(depth),
        )

    def _layer_scarcity(self, slot: int) -> SourceScarcityContext:
        layer = self.layers[int(slot)] if 0 <= int(slot) < len(self.layers) else ()
        valid = [
            event_id
            for event_id in layer
            if bool(self.get(slot, event_id).get("static_valid", False))
        ]
        return build_source_scarcity_context(
            db=self.db,
            hard_safe_event_ids=valid,
            all_event_ids=layer,
            config=self.constraint_config,
        )

    def _suffix(
        self,
        *,
        slot: int,
        previous_event_id: int,
        selected_event_ids: tuple[int, ...],
        usage: tuple[float, ...],
        duals: tuple[float, ...],
        recovery_budget_used: float,
        depth: int,
    ) -> dict[str, Any]:
        if slot >= len(self.layers):
            return {
                "reachable": True,
                "best_cost": 0.0,
                "path_count": 1,
                "immediate_successor_count": 0,
                "first_dead_end_slot": None,
                "best_successor": None,
                "explored_steps": 0,
            }
        if depth >= self.config.state_horizon:
            static_candidates = [
                event_id
                for event_id in self.layers[slot]
                if bool(self.get(slot, event_id).get("future_reachable", False))
            ]
            return {
                "reachable": bool(static_candidates),
                "best_cost": 0.0 if static_candidates else float("inf"),
                "path_count": max(1, len(static_candidates)) if static_candidates else 0,
                "immediate_successor_count": len(static_candidates),
                "first_dead_end_slot": None if static_candidates else int(slot),
                "best_successor": int(static_candidates[0]) if static_candidates else None,
                "explored_steps": 0,
            }

        key = self._cache_key(
            slot=slot,
            previous_event_id=previous_event_id,
            selected_event_ids=selected_event_ids,
            usage=usage,
            duals=duals,
            recovery_budget_used=recovery_budget_used,
            depth=depth,
        )
        self._state_cache_lookups += 1
        cached = self._state_cache.get(key)
        if cached is not None:
            self._state_cache_hits += 1
            return dict(cached)
        if len(self._state_cache) >= self.config.state_cache_maximum_entries:
            self._state_cache.clear()

        indexed_layer = list(enumerate(self.layers[slot]))
        indexed_layer.sort(
            key=lambda row: (
                -float(
                    self.get(slot, row[1]).get(
                        "future_reachability_probability", 0.0
                    )
                ),
                int(row[0]),
            )
        )
        layer = [
            int(event_id)
            for _rank, event_id in indexed_layer[: self.config.state_branch_topk]
        ]
        scarcity = self._layer_scarcity(slot)
        rows: list[tuple[float, int, dict[str, Any], dict[str, Any], float]] = []
        immediate = 0
        first_dead_end: Optional[int] = None
        for rank, event_id in enumerate(layer):
            static = self.get(slot, event_id)
            if not bool(static.get("static_valid", False)):
                continue
            assessment = assess_candidate_constraints(
                db=self.db,
                event_id=int(event_id),
                selected_event_ids=selected_event_ids,
                observability=self.config.state_surrogate_observability,
                future_reachability_probability=float(
                    static.get("future_reachability_probability", 1.0)
                ),
                slot_index=slot,
                constraint_usage=usage,
                dual_variables=duals,
                config=self.constraint_config,
                scarcity_context=scarcity,
            )
            charge = 0.0 if assessment["within_budget"] else float(
                assessment.get("recovery_charge", float("inf"))
            )
            allowed = bool(assessment["within_budget"])
            if not allowed:
                allowed = bool(
                    self.constraint_config.controlled_recovery_enabled
                    and np.isfinite(charge)
                    and recovery_budget_used + charge
                    <= self.constraint_config.recovery_budget_total
                    + self.constraint_config.budget_tolerance
                )
            if not allowed:
                continue
            immediate += 1
            edge = self._edge_cost(previous_event_id, event_id, rank)
            if edge > self.config.maximum_edge_cost:
                continue
            child = self._suffix(
                slot=slot + 1,
                previous_event_id=int(event_id),
                selected_event_ids=selected_event_ids + (int(event_id),),
                usage=tuple(assessment["usage_after_tuple"]),
                duals=tuple(assessment["duals_after_tuple"]),
                recovery_budget_used=float(recovery_budget_used + charge),
                depth=depth + 1,
            )
            if not child["reachable"]:
                dead = child.get("first_dead_end_slot")
                if dead is not None:
                    first_dead_end = dead if first_dead_end is None else min(first_dead_end, dead)
                continue
            local_cost = float(
                edge
                + assessment["probabilistic_auxiliary_cost"]
                + assessment["diversity_penalty"]
                + self.constraint_config.recovery_penalty * charge
            )
            total_cost = local_cost + float(child["best_cost"])
            rows.append((total_cost, int(event_id), assessment, child, charge))

        if rows:
            rows.sort(key=lambda row: (row[0], row[1]))
            best = rows[0]
            result = {
                "reachable": True,
                "best_cost": float(best[0]),
                "path_count": int(min(1000000, sum(int(row[3]["path_count"]) for row in rows))),
                "immediate_successor_count": int(immediate),
                "first_dead_end_slot": None,
                "best_successor": int(best[1]),
                "explored_steps": int(1 + best[3].get("explored_steps", 0)),
            }
        else:
            result = {
                "reachable": False,
                "best_cost": float("inf"),
                "path_count": 0,
                "immediate_successor_count": int(immediate),
                "first_dead_end_slot": int(first_dead_end if first_dead_end is not None else slot),
                "best_successor": None,
                "explored_steps": 0,
            }
        self._state_cache[key] = dict(result)
        return result

    def query(
        self,
        *,
        slot: int,
        event_id: int,
        selected_event_ids: Sequence[int],
        constraint_usage: Sequence[float],
        dual_variables: Sequence[float],
        recovery_budget_used: float,
        observability: float,
        scarcity_context: Optional[SourceScarcityContext] = None,
    ) -> dict[str, Any]:
        """Evaluate future capacity conditioned on the retained Beam state."""
        self._state_queries += 1
        static = self.get(slot, event_id)
        if not bool(static.get("static_valid", False)):
            return {
                **static,
                "schema": "state_aware_backward_reachability",
                "future_reachable": False,
                "state_budget_feasible": False,
                "predicted_recovery_charge": None,
                "future_first_dead_end_slot": int(slot),
                "state_cache_hit_rate": self.state_cache_hit_rate,
            }
        assessment = assess_candidate_constraints(
            db=self.db,
            event_id=int(event_id),
            selected_event_ids=selected_event_ids,
            observability=float(observability),
            future_reachability_probability=float(
                static.get("future_reachability_probability", 1.0)
            ),
            slot_index=int(slot),
            constraint_usage=constraint_usage,
            dual_variables=dual_variables,
            config=self.constraint_config,
            scarcity_context=scarcity_context,
        )
        charge = 0.0 if assessment["within_budget"] else float(
            assessment.get("recovery_charge", float("inf"))
        )
        state_budget_feasible = bool(assessment["within_budget"])
        if not state_budget_feasible:
            state_budget_feasible = bool(
                self.constraint_config.controlled_recovery_enabled
                and np.isfinite(charge)
                and recovery_budget_used + charge
                <= self.constraint_config.recovery_budget_total
                + self.constraint_config.budget_tolerance
            )
        if not state_budget_feasible:
            return {
                **static,
                "schema": "state_aware_backward_reachability",
                "future_reachable": False,
                "future_reachability_probability": 0.0,
                "future_safe_successor_count": 0,
                "state_budget_feasible": False,
                "predicted_recovery_charge": None if not np.isfinite(charge) else float(charge),
                "recovery_budget_used_after_prediction": float(recovery_budget_used),
                "future_first_dead_end_slot": int(slot),
                "state_cache_hit_rate": self.state_cache_hit_rate,
            }

        if int(slot) >= len(self.layers) - 1:
            suffix = {
                "reachable": True,
                "best_cost": 0.0,
                "path_count": 1,
                "immediate_successor_count": 0,
                "first_dead_end_slot": None,
                "best_successor": None,
                "explored_steps": 0,
            }
        else:
            suffix = self._suffix(
                slot=int(slot) + 1,
                previous_event_id=int(event_id),
                selected_event_ids=tuple(map(int, selected_event_ids)) + (int(event_id),),
                usage=tuple(assessment["usage_after_tuple"]),
                duals=tuple(assessment["duals_after_tuple"]),
                recovery_budget_used=float(recovery_budget_used + charge),
                depth=0,
            )
        reachable = bool(suffix["reachable"])
        if reachable:
            steps = max(1, int(suffix.get("explored_steps", 0)))
            probability = math.exp(-float(suffix["best_cost"]) / steps)
            probability = max(self.config.successor_probability_floor, probability)
        else:
            probability = 0.0
        return {
            **static,
            "schema": "state_aware_backward_reachability",
            "future_reachable": reachable,
            "future_reachability_probability": float(np.clip(probability, 0.0, 1.0)),
            "future_safe_successor_count": int(suffix["immediate_successor_count"]),
            "future_state_path_count": int(suffix["path_count"]),
            "best_future_successor_event_id": suffix["best_successor"],
            "minimum_state_conditioned_cost": (
                None if not reachable else float(suffix["best_cost"])
            ),
            "state_budget_feasible": True,
            "predicted_recovery_charge": float(charge),
            "recovery_budget_used_after_prediction": float(recovery_budget_used + charge),
            "future_first_dead_end_slot": suffix["first_dead_end_slot"],
            "state_history_length": int(len(selected_event_ids)),
            "state_cache_hit_rate": self.state_cache_hit_rate,
        }

    @property
    def state_cache_hit_rate(self) -> float:
        return float(self._state_cache_hits / max(1, self._state_cache_lookups))

    def runtime_summary(self) -> dict[str, Any]:
        return {
            **dict(self.summary),
            "state_queries": int(self._state_queries),
            "state_cache_lookups": int(self._state_cache_lookups),
            "state_cache_hits": int(self._state_cache_hits),
            "state_cache_entries": int(len(self._state_cache)),
            "state_cache_hit_rate": self.state_cache_hit_rate,
        }
