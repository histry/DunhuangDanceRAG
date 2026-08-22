"""Routing baselines constrained to the formal SMPL14/CTSR candidate contract."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from routing.event_graph_geometry import manifold_edge_cost
from support.event_identity import event_uids_from_generation_db


BASELINE_SCHEMA = "smpl14_ctsr_current_protocol_route_baseline_v1"


def _db_value(db: Mapping[str, Any], key: str, index: int, default: Any) -> Any:
    try:
        value = np.asarray(db[key])[int(index)]
        return value.item() if isinstance(value, np.generic) else value
    except Exception:
        return default


def _candidate_probability(
    slot: Mapping[str, Any], event_uid: str
) -> float:
    if str(slot.get("router_architecture", "")) != "ctsr_weak_temporal_v1":
        raise RuntimeError("Current-protocol route baselines require CTSR-Weak slots")
    if str(slot.get("formal_candidate_contract", "")) != "ctsr_weak_scheduler_siblings_v1":
        raise RuntimeError("Current-protocol baseline received an unaudited candidate set")
    uids = [str(value) for value in slot.get("formal_candidate_event_uids", [])]
    values = np.asarray(
        slot.get("formal_candidate_router_probabilities", []), dtype=np.float64
    )
    if len(uids) != len(values) or not len(uids):
        raise RuntimeError("Malformed CTSR candidate probabilities")
    try:
        position = uids.index(str(event_uid))
    except ValueError as exc:
        raise RuntimeError(f"Event {event_uid!r} is outside the CTSR sibling set") from exc
    probability = float(values[position])
    if not np.isfinite(probability) or probability < 0.0:
        raise RuntimeError("Invalid CTSR candidate probability")
    return probability


def _validate_inputs(
    slots: Sequence[Mapping[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
) -> list[str]:
    if not slots or len(slots) != len(candidate_lists):
        raise RuntimeError("Slots and candidate layers must be non-empty and aligned")
    event_uids = [str(value) for value in event_uids_from_generation_db(db)]
    for slot_index, (slot, candidates) in enumerate(zip(slots, candidate_lists)):
        if not candidates:
            raise RuntimeError(f"Baseline candidate layer {slot_index} is empty")
        for event_id in candidates:
            if not 0 <= int(event_id) < len(event_uids):
                raise RuntimeError(f"Baseline event index out of range: {event_id}")
            _candidate_probability(slot, event_uids[int(event_id)])
    return event_uids


def greedy_route(
    slots: Sequence[Mapping[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
) -> dict[str, Any]:
    """Independent slot-wise CTSR argmax; no global transition model."""

    event_uids = _validate_inputs(slots, candidate_lists, db)
    chosen = [
        max(
            map(int, candidates),
            key=lambda event_id: _candidate_probability(
                slot, event_uids[event_id]
            ),
        )
        for slot, candidates in zip(slots, candidate_lists)
    ]
    return {
        "schema": BASELINE_SCHEMA,
        "baseline": "ctsr_independent_greedy_v1",
        "same_smpl14_event_db": True,
        "same_ctsr_candidate_sets": True,
        "formal_fallback": False,
        "chosen_event_path": chosen,
        "chosen_event_uids": [event_uids[index] for index in chosen],
    }
def beam_route(
    slots: Sequence[Mapping[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    *,
    beam_size: int = 32,
    transition_weight: float = 1.0,
) -> dict[str, Any]:
    """Finite beam baseline with CTSR unary and identical event geometry."""

    event_uids = _validate_inputs(slots, candidate_lists, db)
    beams: list[tuple[float, list[int]]] = [(0.0, [])]
    trace: list[dict[str, Any]] = []
    for slot_index, (slot, candidates) in enumerate(zip(slots, candidate_lists)):
        expanded: list[tuple[float, list[int]]] = []
        rows: list[dict[str, Any]] = []
        for event_id in map(int, candidates):
            probability = _candidate_probability(slot, event_uids[event_id])
            quality = float(
                _db_value(db, "event_geometry_combined_quality", event_id, 0.5)
            )
            unary = math.log(max(probability, 1.0e-8)) + 0.20 * quality
            rows.append(
                {
                    "event_id": event_id,
                    "event_uid": event_uids[event_id],
                    "ctsr_probability": probability,
                    "unary": unary,
                }
            )
            for score, prefix in beams:
                edge = 0.0
                if prefix:
                    edge = float(
                        manifold_edge_cost(db, prefix[-1], event_id)["total"]
                    )
                expanded.append(
                    (score + unary - float(transition_weight) * edge, prefix + [event_id])
                )
        if not expanded:
            raise RuntimeError(f"Beam baseline exhausted at slot {slot_index}")
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = expanded[: max(1, int(beam_size))]
        trace.append(
            {
                "slot": slot_index,
                "candidates": rows,
                "best_prefix_score": float(beams[0][0]),
            }
        )
    score, chosen = beams[0]
    return {
        "schema": BASELINE_SCHEMA,
        "baseline": "ctsr_finite_beam_v1",
        "same_smpl14_event_db": True,
        "same_ctsr_candidate_sets": True,
        "formal_fallback": False,
        "beam_size": int(beam_size),
        "transition_weight": float(transition_weight),
        "score": float(score),
        "chosen_event_path": chosen,
        "chosen_event_uids": [event_uids[index] for index in chosen],
        "trace": trace,
    }
