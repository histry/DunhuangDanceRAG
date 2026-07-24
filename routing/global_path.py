#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.53 whole-song closed loop.

This module keeps the public V46.52 Fresh-WAV/Heading/Anatomy transaction and
adds the strongest low-resource-safe parts of the research reconstruction:

- dual-branch semantic + intrinsic-geometry candidate grounding;
- opt-in Fisher--Rao categorical marginals and a finite discrete
  multi-marginal Schrödinger path solver on the hard-feasible Event graph;
- legacy entropy-inspired beam routing retained as an auditable fallback;
- bidirectional tangent-space transition risk;
- observability-aware hard rejection;
- frame x joint risk masks;
- tangent-space masked merge after V45/V46/IK;
- final anatomy rollback inherited from V46.52.

No Event core is globally redrawn.  All neural edits remain bounded by the
existing seam mask and the new joint-level risk mask.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from routing.performer_policy import (
    performer_switch_penalty,
    resolve_candidate_policy,
)
from routing.diversity import diversity_assessment, event_identity
from routing.event_graph_geometry import (
    EventGraphGeometryConfig,
    event_node_feasibility,
    manifold_edge_cost,
)
from routing.fisher_rao import (
    categorical_entropy,
    simplex_softmax,
)
from routing.graph_schrodinger import multi_marginal_schrodinger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import routing.heading_closed_loop_impl2 as v52
from contracts.boundary import (
    audit_motion,
    build_frame_joint_risk_mask,
    tangent_masked_merge,
    transition_multiscale_risk,
)
from grounding.model import GroundingRuntime
from contracts.duration import audit_dynamic_duration, save_duration_report
from motion_geometry.product_manifold import (
    riemannian_trust_region_refine_np,
)

SCHEMA = "v46_53_geometry_probabilistic_eventrag_closed_loop"
# Cross-module edge contract.  The numerical implementation lives in
# routing.event_graph_geometry; keeping the field list here makes the final
# closed-loop dependency explicit and auditable.
ROUTE_EDGE_CONTRACT_FIELDS = (
    "v46_53_entry_root_velocity_mps",
    "v46_53_exit_root_velocity_mps",
    "entry_floor_offset_m",
    "exit_floor_offset_m",
    "v46_55_entry_rotation_matrix",
    "v46_55_exit_rotation_matrix",
)
_INSTALLED = False
_RUNTIME: Optional[GroundingRuntime] = None
_RUNTIME_DB_ID: Optional[int] = None
_GLOBAL_ROUTE_REPORT: Dict[str, Any] = {}


def _env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


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


def _db_value(db: Mapping[str, Any], key: str, event_id: int, default: Any) -> Any:
    try:
        arr = np.asarray(db[key])
        value = arr[int(event_id)]
        return value.item() if isinstance(value, np.generic) else value
    except Exception:
        return default


def _runtime(db: Mapping[str, Any]) -> GroundingRuntime:
    global _RUNTIME, _RUNTIME_DB_ID
    identity = id(db)
    if _RUNTIME is None or _RUNTIME_DB_ID != identity:
        ckpt = str(os.environ.get("V46_53_GROUNDER_CKPT", "")).strip()
        if not ckpt:
            out_root = str(os.environ.get("OUT_ROOT", "")).strip()
            if out_root:
                architecture = str(
                    os.environ.get(
                        "V46_53_GROUNDER_ARCHITECTURE", "legacy"
                    )
                ).strip().lower()
                name = (
                    "v46_53_mixed_curvature_grounder.pt"
                    if architecture == "mixed"
                    else "v46_53_dual_branch_grounder.pt"
                )
                ckpt = str(Path(out_root) / name)
        _RUNTIME = GroundingRuntime(db, ckpt)
        _RUNTIME_DB_ID = identity
    return _RUNTIME


def _posture_gap(db: Mapping[str, Any], a: int, b: int) -> float:
    order = {"floor_pose": 0, "kneeling": 1, "deep_squat": 2, "half_squat": 3, "standing": 4, "aerial": 5}
    pa = str(_db_value(db, "posture_exit", a, "standing"))
    pb = str(_db_value(db, "posture_entry", b, "standing"))
    return float(abs(order.get(pa, 4) - order.get(pb, 4)))


def _vector_gap(db: Mapping[str, Any], key_a: str, key_b: str, a: int, b: int) -> float:
    try:
        va = np.asarray(db[key_a], dtype=np.float32)[int(a)]
        vb = np.asarray(db[key_b], dtype=np.float32)[int(b)]
        return float(np.mean(np.linalg.norm(va - vb, axis=-1)))
    except Exception:
        return 0.0


def _global_transition_energy(db: Mapping[str, Any], a: int, b: int) -> float:
    return float(manifold_edge_cost(db, a, b)["total"])


def _legacy_global_route_preorder(
    slots: Sequence[Mapping[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    banned: Optional[Dict[int, set]] = None,
) -> List[List[int]]:
    """Entropy-regularised global beam path used only to pre-order candidates.

    The existing V46.52 simulator still performs the authoritative physical and
    anatomy check.  This layer prevents local top-1 choices from creating an
    obviously poor long-range family/posture path.
    """
    global _GLOBAL_ROUTE_REPORT
    if not _env_bool("V46_53_GLOBAL_ROUTE_ENABLE", True):
        return [list(map(int, x)) for x in candidate_lists]
    banned = banned or {}
    candidate_lists, performer_policy = resolve_candidate_policy(candidate_lists, db)
    runtime = _runtime(db)
    beam_size = max(1, _env_int("V46_53_GLOBAL_ROUTE_BEAM", 32))
    topk = max(1, _env_int("V46_53_GLOBAL_ROUTE_TOPK", 20))
    entropy_eps = _env_float("V46_53_GLOBAL_ROUTE_ENTROPY", 0.08)
    repeat_w = _env_float("V46_53_GLOBAL_REPEAT_W", 0.16)

    beams: List[Tuple[float, List[int], Dict[str, int]]] = [(0.0, [], {})]
    trace: List[Dict[str, Any]] = []
    for i, slot in enumerate(slots):
        candidates = [
            int(e) for e in candidate_lists[i]
            if int(e) not in banned.get(i, set()) and 0 <= int(e) < len(np.asarray(db["paths"]))
        ][:topk]
        if not candidates:
            raise RuntimeError(f"V46.53 global route has no candidates for slot {i}")
        new: List[Tuple[float, List[int], Dict[str, int]]] = []
        unary_rows = []
        for rank, event_id in enumerate(candidates):
            association = runtime.score(slot, event_id)
            quality = float(_db_value(db, "v46_53_combined_quality", event_id, _db_value(db, "event_quality_scores", event_id, 0.5)))
            anatomy = float(_db_value(db, "anatomy_quality", event_id, 0.5))
            prior = math.exp(-rank / max(1.0, topk / 4.0))
            unary = (
                _env_float("V46_53_GLOBAL_GROUND_W", 1.05) * association
                + _env_float("V46_53_GLOBAL_QUALITY_W", 0.35) * quality
                + _env_float("V46_53_GLOBAL_ANATOMY_W", 0.25) * anatomy
                + entropy_eps * math.log(max(prior, 1e-8))
            )
            unary_rows.append({"event_id": event_id, "rank": rank, "association": association, "unary": unary})
            family = str(_db_value(db, "event_families", event_id, "unknown"))
            source = str(_db_value(db, "source_uids", event_id, "unknown"))
            for score, path, usage in beams:
                diversity = diversity_assessment(db, event_id, path)
                if not bool(diversity["hard_valid"]):
                    continue
                step_score = unary
                if path:
                    step_score -= _global_transition_energy(db, path[-1], event_id)
                    if family == str(_db_value(db, "event_families", path[-1], "unknown")):
                        step_score -= repeat_w
                    if source == str(_db_value(db, "source_uids", path[-1], "unknown")):
                        step_score -= 0.5 * repeat_w
                    step_score -= performer_switch_penalty(db, path[-1], event_id, slot)
                # Capped run-local diversity rather than an unbounded global ban.
                step_score -= min(0.30, 0.04 * usage.get("family::" + family, 0))
                step_score -= float(diversity["penalty"])
                ns = dict(usage)
                ns["family::" + family] = ns.get("family::" + family, 0) + 1
                ns["source::" + source] = ns.get("source::" + source, 0) + 1
                new.append((score + step_score, path + [event_id], ns))
        if not new:
            raise RuntimeError(
                "V46.53 global route diversity/cooldown contract exhausted "
                f"all candidates for slot {i}"
            )
        new.sort(key=lambda row: row[0], reverse=True)
        beams = new[:beam_size]
        trace.append({"slot": i, "candidates": unary_rows, "best_prefix_score": float(beams[0][0])})

    chosen = beams[0][1]
    reordered: List[List[int]] = []
    for i, candidates in enumerate(candidate_lists):
        ordered = [chosen[i]] + [int(x) for x in candidates if int(x) != chosen[i]]
        reordered.append(ordered)
    _GLOBAL_ROUTE_REPORT = {
        "schema": "v46_53_entropy_regularised_global_event_path",
        "solver": "legacy_beam",
        "exact_solver_claim": False,
        "description": "Schroedinger-inspired entropic discrete path prior followed by V46.52 simulated physical reselection",
        "beam_size": beam_size,
        "candidate_topk": topk,
        "chosen_event_path": chosen,
        "chosen_event_uids": [event_identity(db, event_id)["event_uid"] for event_id in chosen],
        "chosen_source_uids": [event_identity(db, event_id)["source_uid"] for event_id in chosen],
        "performer_policy": performer_policy,
        "best_score": float(beams[0][0]),
        "trace": trace,
    }
    return reordered


def _route_unary(
    runtime: GroundingRuntime,
    db: Mapping[str, Any],
    slot: Mapping[str, Any],
    event_id: int,
    rank: int,
    topk: int,
) -> tuple[float, Dict[str, Any]]:
    association = float(runtime.score(slot, int(event_id)))
    quality = float(
        _db_value(
            db,
            "v46_53_combined_quality",
            event_id,
            _db_value(db, "event_quality_scores", event_id, 0.5),
        )
    )
    anatomy = float(_db_value(db, "anatomy_quality", event_id, 0.5))
    rank_prior = math.exp(-int(rank) / max(1.0, int(topk) / 4.0))
    unary = (
        _env_float("V46_53_GLOBAL_GROUND_W", 1.05) * association
        + _env_float("V46_53_GLOBAL_QUALITY_W", 0.35) * quality
        + _env_float("V46_53_GLOBAL_ANATOMY_W", 0.25) * anatomy
        + _env_float("V46_53_GLOBAL_ROUTE_ENTROPY", 0.08)
        * math.log(max(rank_prior, 1.0e-8))
    )
    if not np.isfinite(unary):
        raise RuntimeError(
            f"non-finite global route unary for Event {event_id}"
        )
    return float(unary), {
        "event_id": int(event_id),
        "event_uid": event_identity(db, int(event_id))["event_uid"],
        "rank": int(rank),
        "association": float(association),
        "quality": float(quality),
        "anatomy_quality": float(anatomy),
        "rank_prior": float(rank_prior),
        "unary": float(unary),
    }


def _prepare_graph_layers(
    slots: Sequence[Mapping[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    *,
    banned: Dict[int, set],
    topk: int,
) -> tuple[
    List[List[int]],
    List[np.ndarray],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    filtered_lists, performer_policy = resolve_candidate_policy(
        candidate_lists, db
    )
    runtime = _runtime(db)
    count = len(np.asarray(db["paths"]))
    layers: List[List[int]] = []
    target_marginals: List[np.ndarray] = []
    traces: List[Dict[str, Any]] = []
    temperature = _env_float("V46_55_FR_TEMPERATURE", 0.65)
    strength = float(
        np.clip(_env_float("V46_55_FR_MARGINAL_STRENGTH", 0.90), 0.0, 1.0)
    )

    for slot_id, slot in enumerate(slots):
        rejected: List[Dict[str, Any]] = []
        candidates: List[int] = []
        for raw_event in filtered_lists[slot_id]:
            event_id = int(raw_event)
            if event_id in banned.get(slot_id, set()):
                rejected.append(
                    {"event_id": event_id, "reasons": ["runtime_ban"]}
                )
                continue
            if not 0 <= event_id < count:
                rejected.append(
                    {"event_id": event_id, "reasons": ["event_index"]}
                )
                continue
            valid, reasons = event_node_feasibility(db, event_id)
            if not valid:
                rejected.append(
                    {"event_id": event_id, "reasons": list(reasons)}
                )
                continue
            candidates.append(event_id)
            if len(candidates) >= topk:
                break
        if not candidates:
            raise RuntimeError(
                "V46.55 time-expanded graph has no immutable-valid candidates "
                f"for slot {slot_id}; rejected={rejected[:12]}"
            )

        unary_rows: List[Dict[str, Any]] = []
        unary_values: List[float] = []
        for rank, event_id in enumerate(candidates):
            unary, detail = _route_unary(
                runtime, db, slot, event_id, rank, topk
            )
            unary_values.append(unary)
            unary_rows.append(detail)
        learned = simplex_softmax(
            np.asarray(unary_values, dtype=np.float64),
            temperature=temperature,
        )
        uniform = np.full(learned.shape, 1.0 / learned.size)
        target = strength * learned + (1.0 - strength) * uniform
        target = target / target.sum()
        for row, probability in zip(unary_rows, target):
            row["fisher_rao_target_probability"] = float(probability)
        layers.append(candidates)
        target_marginals.append(target.astype(np.float64))
        traces.append(
            {
                "slot": int(slot_id),
                "candidates": unary_rows,
                "rejected_immutable": rejected,
                "target_entropy": float(categorical_entropy(target)),
            }
        )
    return layers, target_marginals, traces, performer_policy


def _build_graph_edges(
    slots: Sequence[Mapping[str, Any]],
    layers: Sequence[Sequence[int]],
    db: Mapping[str, Any],
) -> tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    costs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    reports: List[Dict[str, Any]] = []
    repeat_weight = _env_float("V46_53_GLOBAL_REPEAT_W", 0.16)
    geometry_config = EventGraphGeometryConfig.from_environment()
    cooldown = max(1, _env_int("V46_54_EVENT_COOLDOWN_SLOTS", 8))

    for time in range(len(layers) - 1):
        previous_layer = list(map(int, layers[time]))
        current_layer = list(map(int, layers[time + 1]))
        matrix = np.zeros(
            (len(previous_layer), len(current_layer)), dtype=np.float64
        )
        support = np.ones(matrix.shape, dtype=bool)
        hard_reason_counts: Dict[str, int] = {}
        so3_available = 0
        lorentz_available = 0
        boundary_strength = float(
            slots[time + 1].get(
                "boundary_accent_strength",
                slots[time + 1].get("boundary_strength", 0.0),
            )
        )
        boundary_reset = boundary_strength >= _env_float(
            "V46_55_GRAPH_RESET_ACCENT", 0.82
        )
        for left_index, previous_event in enumerate(previous_layer):
            previous_family = str(
                _db_value(db, "event_families", previous_event, "unknown")
            )
            previous_source = str(
                _db_value(db, "source_uids", previous_event, "unknown")
            )
            for right_index, current_event in enumerate(current_layer):
                detail = manifold_edge_cost(
                    db,
                    previous_event,
                    current_event,
                    config=geometry_config,
                    boundary_reset=boundary_reset,
                )
                edge_cost = float(detail["total"])
                current_family = str(
                    _db_value(db, "event_families", current_event, "unknown")
                )
                current_source = str(
                    _db_value(db, "source_uids", current_event, "unknown")
                )
                if previous_family == current_family:
                    edge_cost += repeat_weight
                if previous_source == current_source:
                    edge_cost += 0.5 * repeat_weight
                edge_cost += performer_switch_penalty(
                    db,
                    previous_event,
                    current_event,
                    slots[time + 1],
                )
                hard_reasons = list(detail["hard_reasons"])
                if cooldown > 0 and previous_event == current_event:
                    hard_reasons.append("event_uid_cooldown")
                feasible = bool(detail["hard_feasible"] and not hard_reasons)
                support[left_index, right_index] = feasible
                matrix[left_index, right_index] = edge_cost
                so3_available += int(bool(detail["so3_available"]))
                lorentz_available += int(bool(detail["lorentz_available"]))
                for reason in hard_reasons:
                    token = str(reason)
                    hard_reason_counts[token] = (
                        hard_reason_counts.get(token, 0) + 1
                    )

        active_cost = matrix[support]
        reports.append(
            {
                "from_slot": int(time),
                "to_slot": int(time + 1),
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "feasible_edges": int(support.sum()),
                "total_edges": int(support.size),
                "feasible_ratio": float(support.mean()),
                "hard_reason_counts": hard_reason_counts,
                "so3_available_edges": int(so3_available),
                "lorentz_available_edges": int(lorentz_available),
                "cost_min": (
                    float(active_cost.min()) if active_cost.size else None
                ),
                "cost_median": (
                    float(np.median(active_cost))
                    if active_cost.size
                    else None
                ),
                "cost_max": (
                    float(active_cost.max()) if active_cost.size else None
                ),
                "boundary_reset": bool(boundary_reset),
            }
        )
        costs.append(matrix)
        masks.append(support)
    return costs, masks, reports


def _validate_history_constraints(
    db: Mapping[str, Any],
    path: Sequence[int],
) -> tuple[bool, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    selected: List[int] = []
    for slot, event_id in enumerate(path):
        assessment = diversity_assessment(db, int(event_id), selected)
        if not bool(assessment["hard_valid"]):
            failures.append(
                {
                    "slot": int(slot),
                    "event_id": int(event_id),
                    "hard_reasons": list(assessment["hard_reasons"]),
                }
            )
        selected.append(int(event_id))
    return not failures, failures


def _posterior_constrained_decode(
    layers: Sequence[Sequence[int]],
    posterior_nodes: Sequence[np.ndarray],
    posterior_transitions: Sequence[np.ndarray],
    db: Mapping[str, Any],
) -> tuple[List[int], Dict[str, Any]]:
    """Decode the SB posterior while preserving history-dependent contracts."""

    beam_size = max(1, _env_int("V46_55_SB_DECODE_BEAM", 128))
    beams: List[Tuple[float, List[int], List[int]]] = []
    for local_index, event_id in enumerate(layers[0]):
        assessment = diversity_assessment(db, int(event_id), [])
        if not bool(assessment["hard_valid"]):
            continue
        score = math.log(max(float(posterior_nodes[0][local_index]), 1.0e-15))
        score -= float(assessment["penalty"])
        beams.append((score, [int(event_id)], [int(local_index)]))
    beams.sort(key=lambda row: row[0], reverse=True)
    beams = beams[:beam_size]
    if not beams:
        raise RuntimeError("SB constrained decoder has no valid first-slot state")

    expanded_counts = [len(beams)]
    for time in range(1, len(layers)):
        expanded: List[Tuple[float, List[int], List[int]]] = []
        transition = np.asarray(
            posterior_transitions[time - 1], dtype=np.float64
        )
        for score, event_path, local_path in beams:
            previous_local = local_path[-1]
            for local_index, event_id in enumerate(layers[time]):
                probability = float(transition[previous_local, local_index])
                if probability <= 0.0:
                    continue
                assessment = diversity_assessment(
                    db, int(event_id), event_path
                )
                if not bool(assessment["hard_valid"]):
                    continue
                expanded.append(
                    (
                        score
                        + math.log(max(probability, 1.0e-15))
                        - float(assessment["penalty"]),
                        event_path + [int(event_id)],
                        local_path + [int(local_index)],
                    )
                )
        if not expanded:
            raise RuntimeError(
                "SB constrained decoder exhausted history-valid paths at "
                f"slot {time}"
            )
        expanded.sort(key=lambda row: row[0], reverse=True)
        beams = expanded[:beam_size]
        expanded_counts.append(len(expanded))
    best = beams[0]
    return best[1], {
        "decoder": "posterior_history_constrained_beam",
        "beam_size": int(beam_size),
        "score": float(best[0]),
        "expanded_counts": expanded_counts,
    }


def _graph_sb_global_route_preorder(
    slots: Sequence[Mapping[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    banned: Optional[Dict[int, set]] = None,
) -> List[List[int]]:
    global _GLOBAL_ROUTE_REPORT
    blocked = banned or {}
    topk = max(1, _env_int("V46_53_GLOBAL_ROUTE_TOPK", 20))
    (
        layers,
        targets,
        traces,
        performer_policy,
    ) = _prepare_graph_layers(
        slots,
        candidate_lists,
        db,
        banned=blocked,
        topk=topk,
    )
    transition_costs, feasible_masks, edge_reports = _build_graph_edges(
        slots, layers, db
    )
    total_edges = int(sum(row["total_edges"] for row in edge_reports))
    so3_edges = int(
        sum(row["so3_available_edges"] for row in edge_reports)
    )
    lorentz_edges = int(
        sum(row["lorentz_available_edges"] for row in edge_reports)
    )
    if (
        total_edges > 0
        and _env_bool("V46_55_REQUIRE_SO3_EDGE", False)
        and so3_edges != total_edges
    ):
        raise RuntimeError(
            "V46.55 strict route requires SO(3) endpoint geometry for every "
            f"edge, available={so3_edges}/{total_edges}; rebuild Event-DB"
        )
    if (
        total_edges > 0
        and _env_bool("V46_55_REQUIRE_LORENTZ_EDGE", False)
        and lorentz_edges != total_edges
    ):
        raise RuntimeError(
            "V46.55 strict route requires paper-one Lorentz factors for every "
            f"edge, available={lorentz_edges}/{total_edges}; embed Event-DB "
            "with the mixed-curvature Grounder"
        )
    result = multi_marginal_schrodinger(
        targets,
        transition_costs,
        feasible_masks=feasible_masks,
        epsilon=_env_float("V46_55_SB_EPSILON", 0.35),
        maximum_iterations=max(1, _env_int("V46_55_SB_MAX_ITER", 300)),
        tolerance=_env_float("V46_55_SB_TOLERANCE", 1.0e-7),
        damping=_env_float("V46_55_SB_DAMPING", 1.0),
    )
    if not result.converged:
        raise RuntimeError(
            "Graph-SB IPF did not converge: "
            f"iterations={result.iterations}, "
            f"L1={result.maximum_l1_residual:.6g}, "
            f"FR={result.maximum_fisher_rao_residual:.6g}"
        )

    map_path = [
        int(layers[slot][local])
        for slot, local in enumerate(result.map_path)
    ]
    history_valid, history_failures = _validate_history_constraints(
        db, map_path
    )
    if history_valid:
        chosen = map_path
        decoder_report = {
            "decoder": "exact_viterbi_on_fitted_markov_measure",
            "history_constraints_valid": True,
        }
    else:
        chosen, decoder_report = _posterior_constrained_decode(
            layers,
            result.node_marginals,
            result.posterior_transitions,
            db,
        )
        valid, failures = _validate_history_constraints(db, chosen)
        if not valid:
            raise RuntimeError(
                "posterior constrained decoder emitted an invalid path: "
                f"{failures}"
            )
        decoder_report["viterbi_history_failures"] = history_failures
        decoder_report["history_constraints_valid"] = True

    reordered: List[List[int]] = []
    for slot, original in enumerate(candidate_lists):
        event = int(chosen[slot])
        reordered.append(
            [event] + [int(value) for value in original if int(value) != event]
        )
        traces[slot]["posterior_marginal"] = (
            result.node_marginals[slot].tolist()
        )

    _GLOBAL_ROUTE_REPORT = {
        "schema": "v46_55_fisher_rao_discrete_graph_schrodinger_route_v1",
        "solver": "fisher_rao_graph_sb",
        "formal_path_measure": True,
        "continuous_sde_bridge_claim": False,
        "multi_marginal_constraints": "all music slots",
        "reference_process": "time-inhomogeneous hard-feasible Markov chain",
        "fisher_rao": {
            "temperature": _env_float("V46_55_FR_TEMPERATURE", 0.65),
            "marginal_strength": _env_float(
                "V46_55_FR_MARGINAL_STRENGTH", 0.90
            ),
        },
        "schrodinger": {
            "epsilon": _env_float("V46_55_SB_EPSILON", 0.35),
            "iterations": int(result.iterations),
            "converged": bool(result.converged),
            "maximum_l1_residual": float(result.maximum_l1_residual),
            "maximum_fisher_rao_residual": float(
                result.maximum_fisher_rao_residual
            ),
            "map_log_probability": float(result.map_log_probability),
            "path_entropy": float(result.path_entropy),
        },
        "decoder": decoder_report,
        "hard_contracts": {
            "immutable_node_gates": True,
            "pairwise_edge_support": True,
            "history_dependent_diversity": True,
            "downstream_heading_anatomy_physics_authoritative": True,
        },
        "manifold_edge_coverage": {
            "total_edges": int(total_edges),
            "so3_edges": int(so3_edges),
            "lorentz_edges": int(lorentz_edges),
            "require_so3": _env_bool("V46_55_REQUIRE_SO3_EDGE", False),
            "require_lorentz": _env_bool(
                "V46_55_REQUIRE_LORENTZ_EDGE", False
            ),
        },
        "candidate_topk": int(topk),
        "chosen_event_path": chosen,
        "chosen_event_uids": [
            event_identity(db, event_id)["event_uid"] for event_id in chosen
        ],
        "chosen_source_uids": [
            event_identity(db, event_id)["source_uid"] for event_id in chosen
        ],
        "performer_policy": performer_policy,
        "trace": traces,
        "edge_layers": edge_reports,
    }
    return reordered


def _global_route_preorder(
    slots: Sequence[Mapping[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    banned: Optional[Dict[int, set]] = None,
) -> List[List[int]]:
    """Choose and pre-order one whole-song Event path with safe fallback."""

    global _GLOBAL_ROUTE_REPORT
    # One process may generate more than one song.  Never allow a disabled or
    # failed current route to inherit the previous song's audit payload.
    _GLOBAL_ROUTE_REPORT = {}
    if not _env_bool("V46_53_GLOBAL_ROUTE_ENABLE", True):
        return [list(map(int, values)) for values in candidate_lists]
    solver = str(
        os.environ.get("V46_55_ROUTE_SOLVER", "legacy_beam")
    ).strip().lower()
    if solver in {"legacy", "beam", "legacy_beam"}:
        return _legacy_global_route_preorder(
            slots, candidate_lists, db, banned=banned
        )
    if solver not in {
        "graph_sb",
        "fisher_rao_graph_sb",
        "fisher-rao-graph-sb",
    }:
        raise ValueError(
            "V46_55_ROUTE_SOLVER must be legacy_beam or "
            f"fisher_rao_graph_sb, got {solver!r}"
        )
    try:
        return _graph_sb_global_route_preorder(
            slots, candidate_lists, db, banned=banned
        )
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        if not _env_bool("V46_55_SB_ALLOW_LEGACY_FALLBACK", True):
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        graph_attempt_report = dict(_GLOBAL_ROUTE_REPORT)
        reordered = _legacy_global_route_preorder(
            slots, candidate_lists, db, banned=banned
        )
        legacy_report = dict(_GLOBAL_ROUTE_REPORT)
        _GLOBAL_ROUTE_REPORT = {
            **legacy_report,
            "schema": "v46_55_fisher_rao_graph_sb_fallback_v1",
            "solver": "legacy_beam",
            "requested_solver": "fisher_rao_graph_sb",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "fallback_is_auditable": True,
            "graph_sb_attempt": {
                "error": fallback_reason,
                "partial_report": graph_attempt_report or None,
            },
            "fallback_route": legacy_report,
        }
        return reordered


def _install_v53_patches() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    # Install all V46.52 policies first, then capture the resulting functions.
    v52._install_patches()
    original_proposal = v52.v4650._build_heading_proposal
    original_apply = v52.v4650.apply_generators_with_heading_guard
    original_assemble = v52.v4650.assemble_event_heading_reference

    def proposal_v53(*args, **kwargs):
        proposal, extra = original_proposal(*args, **kwargs)
        db = kwargs.get("db")
        slot = kwargs.get("slot", {})
        cfg = kwargs.get("cfg")
        event_id = int(kwargs.get("event_id", proposal.event_id))
        prev = kwargs.get("prev_motion")
        if db is None:
            return proposal, extra

        association = _runtime(db).score(slot, event_id)
        quality = float(_db_value(db, "v46_53_combined_quality", event_id, _db_value(db, "event_quality_scores", event_id, 0.5)))
        structure_q = float(_db_value(db, "v46_53_structure_quality", event_id, 0.5))
        boundary = None
        if prev is not None and len(prev) and len(proposal.core):
            boundary = transition_multiscale_risk(
                np.asarray(prev)[-max(8, _env_int("V46_53_TANGENT_WINDOW", 8)):],
                proposal.bridge,
                proposal.core[:max(8, _env_int("V46_53_TANGENT_WINDOW", 8))],
                fps=float(getattr(cfg, "fps", 30.0)),
            )

        observability = float(np.clip(
            0.45 * association + 0.25 * quality + 0.20 * structure_q
            + 0.10 * float(_db_value(db, "semantic_confidence", event_id, 0.5)),
            0.0, 1.0,
        ))
        reward = (
            _env_float("V46_53_ASSOCIATION_REWARD_W", 0.75) * association
            + _env_float("V46_53_STRUCTURE_REWARD_W", 0.25) * structure_q
        )
        penalty = 0.0 if boundary is None else _env_float("V46_53_TANGENT_RISK_W", 0.55) * float(boundary["score"])
        hard = bool(
            (boundary is not None and boundary.get("hard_reject", False))
            or observability < _env_float("V46_53_OBSERVABILITY_HARD_MIN", 0.22)
        )
        proposal.risk_score = float(proposal.risk_score + penalty - reward + (1e6 if hard else 0.0))
        proposal.safe = bool(proposal.safe and not hard)
        proposal.risk["v46_53_grounding"] = {
            "association": association,
            "quality": quality,
            "structure_quality": structure_q,
            "observability": observability,
            "reward": reward,
            "penalty": penalty,
            "hard_reject": hard,
        }
        proposal.risk["v46_53_tangent_boundary"] = boundary
        extra = dict(extra)
        extra["v46_53_grounding"] = proposal.risk["v46_53_grounding"]
        extra["v46_53_tangent_boundary"] = boundary
        extra.setdefault("heading_detail", {})["hard_reject"] = bool(
            extra.get("heading_detail", {}).get("hard_reject", False) or hard
        )
        return proposal, extra

    def assemble_v53(v46: Any, slots: Sequence[Dict[str, Any]], candidate_lists: Sequence[Sequence[int]], db: Dict[str, Any], cfg: Any, banned: Optional[Dict[int, set]] = None):
        reordered = _global_route_preorder(slots, candidate_lists, db, banned=banned)
        return original_assemble(v46, slots, reordered, db, cfg, banned=banned)

    def apply_v53(v46: Any, motion_ref: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, args: Any, cfg: Any):
        proposal_motion, stage = original_apply(v46, motion_ref, cond, seam_mask, args, cfg)
        if not _env_bool("V46_53_BODY_PART_MASK_ENABLE", True):
            return proposal_motion, stage
        masks = build_frame_joint_risk_mask(
            motion_ref,
            seam_mask,
            fps=float(getattr(cfg, "fps", 30.0)),
        )
        trust_region_report: Dict[str, Any]
        if _env_bool(
            "V46_53_RIEMANNIAN_TRUST_REGION_ENABLE",
            bool(
                getattr(
                    cfg, "riemannian_trust_region_enable", True
                )
            ),
        ):
            try:
                merged, trust_region_report = (
                    riemannian_trust_region_refine_np(
                        motion_ref,
                        proposal_motion,
                        joint_mask=masks["joint"],
                        root_mask=masks["root"],
                        contact_mask=masks["contact"],
                        steps=_env_int(
                            "V46_53_RIEMANNIAN_TRUST_REGION_STEPS",
                            int(
                                getattr(
                                    cfg,
                                    "riemannian_trust_region_steps",
                                    5,
                                )
                            ),
                        ),
                        initial_radius=_env_float(
                            "V46_53_RIEMANNIAN_TRUST_REGION_INITIAL_RADIUS",
                            float(
                                getattr(
                                    cfg,
                                    "riemannian_trust_region_initial_radius",
                                    1.0,
                                )
                            ),
                        ),
                        min_radius=_env_float(
                            "V46_53_RIEMANNIAN_TRUST_REGION_MIN_RADIUS",
                            float(
                                getattr(
                                    cfg,
                                    "riemannian_trust_region_min_radius",
                                    0.0625,
                                )
                            ),
                        ),
                        max_rotation_rad=_env_float(
                            "V46_53_RIEMANNIAN_TRUST_REGION_ROTATION_CAP_RAD",
                            float(
                                getattr(
                                    cfg,
                                    "product_refiner_rotation_cap_rad",
                                    0.35,
                                )
                            ),
                        ),
                        max_root_m=_env_float(
                            "V46_53_RIEMANNIAN_TRUST_REGION_ROOT_CAP_M",
                            float(
                                getattr(
                                    cfg,
                                    "product_refiner_root_cap_m",
                                    0.08,
                                )
                            ),
                        ),
                        fidelity_weight=_env_float(
                            "V46_53_RIEMANNIAN_TRUST_REGION_FIDELITY_WEIGHT",
                            0.35,
                        ),
                        smoothness_weight=_env_float(
                            "V46_53_RIEMANNIAN_TRUST_REGION_SMOOTHNESS_WEIGHT",
                            0.65,
                        ),
                    )
                )
            except Exception as exc:
                # Geometry refinement is a repair stage, so a numerical failure
                # falls back to the established tangent-masked transaction.
                merged = tangent_masked_merge(
                    motion_ref, proposal_motion, masks
                )
                trust_region_report = {
                    "algorithm": "masked_product_manifold_adaptive_trust_region",
                    "enabled": True,
                    "fallback": "legacy_tangent_masked_merge",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            merged = tangent_masked_merge(
                motion_ref, proposal_motion, masks
            )
            trust_region_report = {
                "algorithm": "disabled",
                "enabled": False,
                "fallback": "legacy_tangent_masked_merge",
            }
        merged = v52.base.enforce_contract(
            v46,
            merged,
            cfg,
            source_hint="v46_53_tangent_masked_merge",
        )
        # V46.52 has already guarded proposal_motion; the second check ensures
        # the tangent projection itself did not introduce a contract regression.
        metrics = v52.anatomy_metrics_np(merged, fps=float(getattr(cfg, "fps", 30.0)))
        ok, reasons = v52.evaluate_anatomy_contract(metrics, v52.AnatomyThresholds.from_env())
        fallback = False
        if not ok:
            if _env_bool("V46_53_FULL_ROLLBACK_ON_FAIL", True):
                merged = np.asarray(motion_ref, dtype=np.float32).copy()
                fallback = True
            else:
                raise RuntimeError("V46.53 tangent-masked merge failed anatomy contract: " + " | ".join(reasons))
        stage["v46_53_bodypart_tangent_mask"] = {
            **dict(masks["report"]),
            "riemannian_trust_region": trust_region_report,
            "anatomy_ok": bool(ok),
            "anatomy_reasons": reasons,
            "full_rollback": fallback,
        }
        stage["v46_53_motion_audit"] = audit_motion(merged, fps=float(getattr(cfg, "fps", 30.0)))
        return merged.astype(np.float32), stage

    v52.v4650._build_heading_proposal = proposal_v53
    v52.v4650.assemble_event_heading_reference = assemble_v53
    v52.v4650.apply_generators_with_heading_guard = apply_v53
    _INSTALLED = True


def _dynamic_duration_guard(
    output_path: Path,
    contract: Mapping[str, Any],
    fps: float,
) -> Dict[str, Any]:
    """Enforce audio-derived output duration; no fixed video length is allowed."""
    report = audit_dynamic_duration(
        output_path=output_path,
        contract=contract,
        fps=fps,
        output_frame_tolerance=_env_int(
            "V46_53_OUTPUT_FRAME_TOLERANCE",
            _env_int("V46_51_MAX_FRAME_ERROR", 2),
        ),
        schedule_audio_tolerance=_env_int("V46_51_MAX_FRAME_ERROR", 2),
    )
    save_duration_report(
        report,
        output_path.with_suffix(output_path.suffix + ".v46_53_duration.json"),
    )
    if _env_bool("V46_53_ENFORCE_DYNAMIC_DURATION", True) and not report["ok"]:
        raise RuntimeError(
            "V46.53 dynamic-duration contract failed: "
            f"actual={report['actual_output_frames']}, "
            f"schedule={report['schedule_target_frames']}, "
            f"audio={report['expected_audio_frames']}"
        )
    return report


def _patch_report(
    report_path: Path,
    duration_guard: Optional[Mapping[str, Any]] = None,
    motion_path: Optional[Path] = None,
    fps: float = 30.0,
) -> None:
    if not report_path.is_file():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return
    report["version"] = SCHEMA
    if _GLOBAL_ROUTE_REPORT:
        report["v46_53_global_route"] = _GLOBAL_ROUTE_REPORT
        if str(_GLOBAL_ROUTE_REPORT.get("schema", "")).startswith("v46_55_"):
            report["v46_55_graph_sb_route"] = _GLOBAL_ROUTE_REPORT
    report["v46_53_env"] = {k: v for k, v in os.environ.items() if k.startswith("V46_53_")}
    report["v46_55_env"] = {
        k: v for k, v in os.environ.items() if k.startswith("V46_55_")
    }
    if duration_guard is not None:
        report["v46_53_dynamic_duration"] = dict(duration_guard)
    resolved_motion = v52._resolve_motion_path(
        report_path,
        report,
        explicit_motion_path=motion_path,
    )

    if resolved_motion is not None:
        x = np.load(resolved_motion, allow_pickle=True)
        report["v46_53_final_intrinsic_audit"] = audit_motion(
            x,
            fps=float(fps),
        )
        report["v46_53_final_motion_path"] = str(resolved_motion)
    v52.save_json(report, report_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    audio = v52._arg_value(args, "--audio")
    schedule = v52._arg_value(args, "--slots_json")
    report_json = v52._arg_value(args, "--json")
    output = v52._arg_value(args, "--out")
    fps = v52._runtime_fps(args)
    if not audio or not schedule or not output:
        raise RuntimeError("V46.53 requires --audio, a fresh --slots_json, and --out")
    required_run_id = os.environ.get("V46_51_SCHEDULE_RUN_ID")
    if not required_run_id:
        raise RuntimeError("V46_51_SCHEDULE_RUN_ID is required")
    contract = v52.audit_contract(
        audio=audio,
        schedule=schedule,
        fps=fps,
        required_run_id=required_run_id,
        require_fresh=True,
        max_frame_error=int(float(os.environ.get("V46_51_MAX_FRAME_ERROR", "2"))),
        max_seconds_error=float(os.environ.get("V46_51_MAX_SECONDS_ERROR", "0.10")),
        require_raw_report=True,
    )
    v52.save_json(contract, Path(schedule).with_suffix(Path(schedule).suffix + ".pre_generate_contract.json"))
    if not contract["ok"]:
        raise RuntimeError("Fresh-WAV contract failed: " + "; ".join(contract["reasons"]))

    _install_v53_patches()
    rc = int(v52.v4650.main(args))
    duration_guard: Optional[Dict[str, Any]] = None
    if rc == 0:
        duration_guard = _dynamic_duration_guard(
            Path(output),
            contract,
            fps=fps,
        )
    if report_json:
        resolved_output = Path(output)

        v52._patch_report(
            Path(report_json),
            contract,
            motion_path=resolved_output,
            fps=fps,
        )

        _patch_report(
            Path(report_json),
            duration_guard=duration_guard,
            motion_path=resolved_output,
            fps=fps,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
