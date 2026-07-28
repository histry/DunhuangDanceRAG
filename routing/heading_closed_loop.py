#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.50 Event-Heading Closed-Loop Planner.

This is an additive replacement entry point for
routing/boundary_closed_loop.py.  It reuses the latest retrieval,
transition simulation, refiner, diffusion, IK and rollback machinery, but
replaces two policies:

1. candidate assembly uses a planner-owned stage heading state rather than
   blindly inheriting the previous root yaw;
2. refiner/diffusion/IK output is guarded against changing the planned root
   heading.

Run with the same CLI as V46.46:
    python routing/heading_closed_loop.py generate ...
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from routing.diversity import (
    diversity_assessment,
    event_identity,
    proposal_selection_score,
)
from routing.dynamic_route import (
    DynamicBeamState,
    DynamicRouteDeadEnd,
    DynamicSearchConfig,
    adaptive_beam_width,
    candidate_subset,
    observability_from_extra,
    prune_states,
    route_prior_cost,
    route_prior_summary,
    source_calibration_penalty,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import routing.boundary_closed_loop as base  # noqa: E402
from contracts.heading import (  # noqa: E402
    EDGE_DIM,
    ROOT_X_IDX,
    ROOT_Z_IDX,
    angle_diff,
    candidate_heading_penalty,
    event_meta_from_db,
    heading_metrics_np,
    restore_planned_root_heading_np,
    root_yaw_np,
    rotate_motion_constant_yaw_np,
    slot_turn_policy,
    wrap_angle,
)

_ORIG_APPLY_GENERATORS = base.apply_generators
_LAST_HEADING_PLAN: Dict[str, Any] = {}


def env_bool(name: str, default: bool) -> bool:
    return base.env_bool(name, default)


def env_float(name: str, default: float) -> float:
    return base.env_float(name, default)


def _event_delta(db: Dict[str, Any], event_id: int) -> float:
    for key in ("event_stage_delta_yaw_rad", "event_net_yaw_rad"):
        try:
            return float(np.asarray(db[key], dtype=np.float32)[int(event_id)])
        except Exception:
            pass
    return 0.0


def _heading_valid(db: Dict[str, Any], event_id: int) -> bool:
    try:
        return bool(np.asarray(db["event_heading_valid"], dtype=bool)[int(event_id)])
    except Exception:
        return False


def _heading_schema_guard(db: Dict[str, Any]) -> None:
    required = [
        "event_turn_intents",
        "event_stage_delta_yaw_rad",
        "event_yaw_budget_rad",
        "event_heading_quality",
        "event_heading_valid",
    ]
    missing = [k for k in required if k not in db]
    if missing:
        raise RuntimeError(
            "V46.50 requires a heading-aware DB. Missing arrays: "
            + ", ".join(missing)
            + ". Rebuild with events/build_database_entry.py"
        )


def _align_core_to_stage_heading(
    v46: Any,
    prev: Optional[np.ndarray],
    core: np.ndarray,
    stage_heading_rad: float,
    cfg: Any,
    event_id: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = np.asarray(core, dtype=np.float32).copy()
    if len(out) == 0:
        return out, {"mode": "empty"}

    entry_before = float(root_yaw_np(out[:1])[0])
    dyaw = angle_diff(float(stage_heading_rad), entry_before)
    pivot = out[0, [ROOT_X_IDX, ROOT_Z_IDX]].copy()

    if hasattr(v46, "rotate_motion_around_y_np"):
        out = v46.rotate_motion_around_y_np(out, dyaw, pivot_xz=pivot)
    else:
        out = rotate_motion_constant_yaw_np(out, dyaw, pivot_xz=pivot)

    delta_xz = np.zeros(2, dtype=np.float32)
    if prev is not None and len(prev):
        delta_xz = (
            np.asarray(prev[-1, [ROOT_X_IDX, ROOT_Z_IDX]], dtype=np.float32)
            - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
        )
        out[:, ROOT_X_IDX] += float(delta_xz[0])
        out[:, ROOT_Z_IDX] += float(delta_xz[1])

    out = base.enforce_contract(
        v46,
        out,
        cfg,
        source_hint=f"v46_50_stage_heading_align:{event_id}",
    )
    entry_after = float(root_yaw_np(out[:1])[0])
    return out.astype(np.float32), {
        "schema": "v46_50_stage_heading_alignment",
        "mode": "planner_absolute_stage_heading_plus_xz_continuity",
        "event_id": int(event_id),
        "stage_heading_target_rad": float(stage_heading_rad),
        "stage_heading_target_deg": float(np.degrees(stage_heading_rad)),
        "entry_heading_before_rad": entry_before,
        "entry_heading_after_rad": entry_after,
        "dyaw_applied_rad": float(dyaw),
        "dyaw_applied_deg": float(np.degrees(dyaw)),
        "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
        "root_y_ramp_applied": False,
    }


def _build_heading_proposal(
    v46: Any,
    prev_motion: Optional[np.ndarray],
    event_id: int,
    event_path: str,
    slot: Dict[str, Any],
    slot_idx: int,
    candidate_rank: int,
    target_len: int,
    cfg: Any,
    db: Dict[str, Any],
    stage_heading_rad: float,
    recent_turn_count: int,
) -> Tuple[base.CandidateProposal, Dict[str, Any]]:
    raw = base.load_event_motion(
        v46,
        event_path,
        cfg,
        source_hint=f"v46_50_load_event:{event_id}",
    )
    has_prev = prev_motion is not None and len(prev_motion) > 0
    core_len, trans_len, length_info = base.choose_transition_lengths(
        v46,
        prev_motion,
        raw.shape[0],
        target_len,
        raw,
        slot,
        cfg,
    )
    core = base.resample_motion(v46, raw, core_len)
    core = base.enforce_contract(
        v46,
        core,
        cfg,
        source_hint=f"v46_50_core_resample:{event_id}",
    )

    event_meta = event_meta_from_db(db, event_id)
    heading_penalty, heading_detail = candidate_heading_penalty(
        event_meta,
        slot,
        stage_heading_rad,
        recent_turn_count=recent_turn_count,
    )

    core, align_report = _align_core_to_stage_heading(
        v46,
        prev_motion,
        core,
        stage_heading_rad,
        cfg,
        event_id,
    )
    bridge = np.zeros((0, EDGE_DIM), dtype=np.float32)
    if has_prev:
        bridge = base.build_bridge(v46, prev_motion, core, trans_len, cfg)
        risk = base.transition_risk(
            v46,
            prev_motion[-4:],
            bridge,
            core[:4],
            fps=float(getattr(cfg, "fps", 30.0)),
        )
    else:
        risk = {
            "total": 0.0,
            "boundary_joint_jerk_max": 0.0,
            "exit_fk_jump": 0.0,
            "exit_rotation_step_rad": 0.0,
            "foot_slip": 0.0,
            "foot_penetration": 0.0,
            "contact_switch": 0.0,
        }

    piece = np.concatenate([bridge, core], axis=0).astype(np.float32)
    if piece.shape[0] != int(target_len):
        raise RuntimeError(
            "Closed-loop slot frame contract mismatch: "
            f"event_id={event_id}, bridge={len(bridge)}, core={len(core)}, "
            f"assembled={len(piece)}, target={int(target_len)}. "
            "Do not resample a floor/contact-audited composite; repair the "
            "upstream transition budget instead."
        )

    physical_risk = float(base.risk_score(risk))
    combined = (
        physical_risk
        + env_float("V46_50_HEADING_PLANNER_WEIGHT", 0.85)
        * float(heading_penalty)
    )
    hard_reject = bool(heading_detail.get("hard_reject", False))
    safe = bool((not hard_reject) and (base.risk_safe(risk) if has_prev else True))

    length_info = dict(length_info)
    length_info["v46_50_heading"] = heading_detail
    proposal = base.CandidateProposal(
        slot=int(slot_idx),
        event_id=int(event_id),
        rank=int(candidate_rank),
        event_path=str(event_path),
        motion_piece=piece.astype(np.float32),
        bridge=bridge.astype(np.float32),
        core=core.astype(np.float32),
        transition_span_local=[0, int(len(bridge))] if has_prev and len(bridge) else None,
        core_span_local=[int(len(bridge)), int(len(bridge) + len(core))],
        risk=risk,
        risk_score=float(combined),
        safe=safe,
        length_info=length_info,
        align_report=align_report,
        decision="candidate",
    )
    return proposal, {
        "physical_risk_score": physical_risk,
        "heading_penalty": float(heading_penalty),
        "combined_score": float(combined),
        "heading_detail": heading_detail,
        "event_meta": event_meta,
    }


def assemble_event_heading_reference(
    v46: Any,
    slots: Sequence[Dict[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Dict[str, Any],
    cfg: Any,
    banned: Optional[Dict[int, set]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[List[int]]]:
    """Assemble an exact-simulation route with posterior-guided dynamic beam.

    Graph-SB probabilities remain soft priors.  Every retained branch is built
    by the authoritative heading/bridge/physics simulator, and anatomy,
    heading, cooldown, source, family, and severe physical gates remain hard.
    """

    global _LAST_HEADING_PLAN
    _heading_schema_guard(db)

    paths = np.asarray(db["paths"], dtype=object)
    blocked = banned or {}
    search = DynamicSearchConfig.from_environment()
    initial_heading = float(
        np.radians(env_float("V46_50_STAGE_INITIAL_HEADING_DEG", 0.0))
    )
    beam: List[DynamicBeamState] = [
        DynamicBeamState(
            motion=np.zeros((0, EDGE_DIM), dtype=np.float32),
            stage_heading=initial_heading,
        )
    ]
    layer_trace: List[Dict[str, Any]] = []

    for slot_idx, slot in enumerate(slots):
        target_len = base.slot_target_frames(slot, cfg)
        candidates = [
            int(value)
            for value in candidate_lists[slot_idx]
            if int(value) not in blocked.get(slot_idx, set())
            and 0 <= int(value) < len(paths)
        ]
        if not candidates:
            raise RuntimeError(f"No candidates remain for slot {slot_idx}")

        primary_event_id = int(candidates[0])
        expanded: List[DynamicBeamState] = []
        state_diagnostics: List[Dict[str, Any]] = []

        for state_index, state in enumerate(beam):
            previous_event_id = (
                int(state.selected_event_ids[-1])
                if state.selected_event_ids
                else None
            )
            subset = candidate_subset(
                candidates,
                db,
                limit=search.branch_topk,
                minimum_per_source=search.candidates_per_source,
                primary_event_id=primary_event_id,
            )

            preordered: List[
                Tuple[float, int, int, Dict[str, Any], Dict[str, Any]]
            ] = []
            rank_lookup = {int(event_id): rank for rank, event_id in enumerate(candidates)}
            for event_id in subset:
                original_rank = int(rank_lookup[int(event_id)])
                meta = event_meta_from_db(db, int(event_id))
                heading_penalty, heading_detail = candidate_heading_penalty(
                    meta,
                    slot,
                    state.stage_heading,
                    recent_turn_count=state.recent_turn_count,
                )
                if not _heading_valid(db, int(event_id)):
                    heading_detail = dict(heading_detail)
                    heading_detail["hard_reject"] = True
                    heading_penalty += 1.0e6
                prior_cost, prior_detail = route_prior_cost(
                    slot_idx,
                    int(event_id),
                    previous_event_id=previous_event_id,
                    fallback_rank=original_rank,
                    candidate_count=len(candidates),
                )
                order_score = float(heading_penalty) + search.posterior_weight * float(
                    prior_cost
                )
                preordered.append(
                    (
                        order_score,
                        original_rank,
                        int(event_id),
                        heading_detail,
                        prior_detail,
                    )
                )
            preordered.sort(key=lambda row: (row[0], row[1], row[2]))

            proposals: List[Tuple[base.CandidateProposal, Dict[str, Any]]] = []
            for _, original_rank, event_id, _heading_detail, prior_detail in preordered:
                proposal, extra0 = _build_heading_proposal(
                    v46=v46,
                    prev_motion=(state.motion if len(state.motion) else None),
                    event_id=event_id,
                    event_path=str(paths[event_id]),
                    slot=dict(slot),
                    slot_idx=slot_idx,
                    candidate_rank=original_rank,
                    target_len=target_len,
                    cfg=cfg,
                    db=db,
                    stage_heading_rad=state.stage_heading,
                    recent_turn_count=state.recent_turn_count,
                )
                extra = dict(extra0)
                diversity = diversity_assessment(
                    db,
                    int(event_id),
                    state.selected_event_ids,
                )
                calibration_penalty, calibration_detail = source_calibration_penalty(
                    db,
                    previous_event_id,
                    int(event_id),
                )
                extra["diversity"] = diversity
                extra["route_prior"] = prior_detail
                extra["route_prior_cost"] = float(prior_detail["negative_log_cost"])
                extra["source_calibration"] = calibration_detail
                extra["source_calibration_penalty"] = float(calibration_penalty)
                proposals.append((proposal, extra))

            safe_count = sum(bool(proposal.safe) for proposal, _extra in proposals)
            safe_ratio = safe_count / max(1, len(proposals))
            eligible_count = 0
            trials: List[Dict[str, Any]] = []

            for proposal, extra in proposals:
                observability = observability_from_extra(
                    extra,
                    safe_ratio=safe_ratio,
                )
                extra["observability"] = observability
                extra["route_uncertainty"] = 1.0 - observability
                extra["selection_score"] = proposal_selection_score(
                    proposal,
                    extra,
                    primary_event_id=primary_event_id,
                )
                diversity = extra["diversity"]
                eligible = bool(
                    proposal.safe
                    and diversity["hard_valid"]
                    and observability >= search.minimum_observability
                )
                eligible_count += int(eligible)
                trials.append(
                    {
                        "event_id": int(proposal.event_id),
                        "rank": int(proposal.rank),
                        "safe": bool(proposal.safe),
                        "eligible": eligible,
                        "combined_score": float(proposal.risk_score),
                        "selection_score": float(extra["selection_score"]),
                        "physical_risk_score": float(
                            extra.get("physical_risk_score", proposal.risk_score)
                        ),
                        "heading_penalty": float(extra.get("heading_penalty", 0.0)),
                        "hard_reject": bool(
                            extra.get("heading_detail", {}).get("hard_reject", False)
                        ),
                        "observability": float(observability),
                        "route_prior": dict(extra.get("route_prior", {})),
                        "source_calibration": dict(
                            extra.get("source_calibration", {})
                        ),
                        "diversity": dict(diversity),
                    }
                )
                if not eligible:
                    continue

                event_meta = extra["event_meta"]
                event_delta = float(
                    event_meta.get(
                        "event_stage_delta_yaw_rad",
                        _event_delta(db, int(proposal.event_id)),
                    )
                )
                stage_before = float(state.stage_heading)
                stage_after = float(wrap_angle(stage_before + event_delta))
                intent = str(event_meta.get("event_turn_intent", "none"))
                recent_turn_count = (
                    state.recent_turn_count + 1
                    if intent in {"turn", "explicit_spin", "uncertain_turn"}
                    else 0
                )

                cursor = int(len(state.motion))
                piece = np.asarray(proposal.motion_piece, dtype=np.float32)
                motion = (
                    np.concatenate([state.motion, piece], axis=0).astype(np.float32)
                    if len(state.motion)
                    else piece.copy()
                )
                transition_span = None
                if proposal.transition_span_local is not None:
                    transition_span = [
                        cursor + int(proposal.transition_span_local[0]),
                        cursor + int(proposal.transition_span_local[1]),
                    ]
                core_span = [
                    cursor + int(proposal.core_span_local[0]),
                    cursor + int(proposal.core_span_local[1]),
                ]
                decision = (
                    "selected_primary_soft_prior"
                    if int(proposal.event_id) == primary_event_id
                    else "selected_dynamic_beam"
                )
                proposal.decision = decision
                source_frames = base.load_event_motion(
                    v46,
                    proposal.event_path,
                    cfg,
                    "v46_50_warp_probe",
                ).shape[0]
                row = {
                    "slot": int(slot_idx),
                    "event_id": int(proposal.event_id),
                    "candidate_rank": int(proposal.rank),
                    "event_path": proposal.event_path,
                    "target_frames": int(target_len),
                    "piece_frames": int(piece.shape[0]),
                    "transition_span": transition_span,
                    "transition_spans": [transition_span] if transition_span else [],
                    "core_span": core_span,
                    "transition_in_frames": int(len(proposal.bridge)),
                    "core_frames": int(len(proposal.core)),
                    "core_warp": float(len(proposal.core) / max(1, source_frames)),
                    "risk_predicted": proposal.risk,
                    "risk_score_predicted": float(
                        extra.get("physical_risk_score", proposal.risk_score)
                    ),
                    "heading_penalty": float(extra.get("heading_penalty", 0.0)),
                    "combined_candidate_score": float(proposal.risk_score),
                    "dynamic_selection_score": float(extra["selection_score"]),
                    "safe_predicted": bool(proposal.safe),
                    "decision": decision,
                    "primary_event_id": primary_event_id,
                    "planned_event_diverged": bool(
                        int(proposal.event_id) != primary_event_id
                    ),
                    "observability": float(observability),
                    "route_prior": dict(extra.get("route_prior", {})),
                    "source_calibration": dict(
                        extra.get("source_calibration", {})
                    ),
                    "dynamic_hard_mask": {
                        "physical_safe": bool(proposal.safe),
                        "history_safe": bool(diversity["hard_valid"]),
                        "hard_reasons": list(diversity["hard_reasons"]),
                    },
                    "diversity": dict(diversity),
                    "length_policy": proposal.length_info,
                    "contract_after_align": proposal.align_report,
                    "event_turn_intent": intent,
                    "event_turn_confidence": float(
                        event_meta.get("event_turn_confidence", 0.0)
                    ),
                    "event_heading_quality": float(
                        event_meta.get("event_heading_quality", 0.0)
                    ),
                    "event_stage_delta_yaw_rad": event_delta,
                    "event_stage_delta_yaw_deg": float(np.degrees(event_delta)),
                    "stage_heading_before_rad": stage_before,
                    "stage_heading_before_deg": float(np.degrees(stage_before)),
                    "stage_heading_after_rad": stage_after,
                    "stage_heading_after_deg": float(np.degrees(stage_after)),
                    "slot_turn_policy": slot_turn_policy(slot),
                    "candidate_trials": trials,
                    "version": "mode_sb_dynamic_heading_reference",
                }
                state_row = {
                    "slot": int(slot_idx),
                    "event_id": int(proposal.event_id),
                    "intent": intent,
                    "stage_heading_before_rad": stage_before,
                    "event_delta_rad": event_delta,
                    "stage_heading_after_rad": stage_after,
                    "cumulative_abs_yaw_rad": float(
                        state.cumulative_abs_yaw + abs(event_delta)
                    ),
                    "observability": float(observability),
                    "prefix_score": float(
                        state.score + float(extra["selection_score"])
                    ),
                }
                expanded.append(
                    DynamicBeamState(
                        motion=motion,
                        selected_event_ids=state.selected_event_ids
                        + (int(proposal.event_id),),
                        selected_ranks=state.selected_ranks + (int(proposal.rank),),
                        report=state.report + (row,),
                        state_trace=state.state_trace + (state_row,),
                        stage_heading=stage_after,
                        recent_turn_count=recent_turn_count,
                        cumulative_abs_yaw=state.cumulative_abs_yaw
                        + abs(event_delta),
                        score=state.score + float(extra["selection_score"]),
                        observability=float(observability),
                    )
                )

            state_diagnostics.append(
                {
                    "state_index": int(state_index),
                    "prefix_event_ids": list(map(int, state.selected_event_ids)),
                    "candidate_subset": list(map(int, subset)),
                    "proposals": int(len(proposals)),
                    "physically_safe": int(safe_count),
                    "eligible": int(eligible_count),
                }
            )

        if not expanded:
            raise DynamicRouteDeadEnd(
                slot_idx,
                {
                    "retained_prefixes": int(len(beam)),
                    "candidate_count": int(len(candidates)),
                    "state_diagnostics": state_diagnostics,
                    "hard_contracts_relaxed": False,
                },
            )

        width = adaptive_beam_width(
            search,
            [state.observability for state in expanded],
        )
        beam = prune_states(expanded, db, width=width)
        layer_trace.append(
            {
                "slot": int(slot_idx),
                "input_states": int(len(state_diagnostics)),
                "expanded_states": int(len(expanded)),
                "retained_states": int(len(beam)),
                "adaptive_beam_width": int(width),
                "best_prefix_score": float(min(state.score for state in beam)),
                "retained_sources": [
                    event_identity(db, state.selected_event_ids[-1])["source_uid"]
                    for state in beam
                ],
                "state_diagnostics": state_diagnostics,
            }
        )

    best = min(beam, key=lambda state: float(state.score))
    final = base.enforce_contract(
        v46,
        np.asarray(best.motion, dtype=np.float32),
        cfg,
        source_hint="mode_sb_dynamic_heading_reference_final",
    )
    _LAST_HEADING_PLAN = {
        "schema": "mode_sb_manifold_observable_dynamic_route",
        "search": {
            "beam_width": int(search.beam_width),
            "maximum_beam_width": int(search.maximum_beam_width),
            "branch_topk": int(search.branch_topk),
            "candidates_per_source": int(search.candidates_per_source),
            "primary_is_soft_prior": True,
            "exact_simulation_authoritative": True,
            "implicit_bounded_backtracking": True,
        },
        "graph_route_prior": route_prior_summary(),
        "initial_stage_heading_rad": float(initial_heading),
        "final_stage_heading_rad": float(best.stage_heading),
        "cumulative_abs_event_yaw_rad": float(best.cumulative_abs_yaw),
        "cumulative_abs_event_yaw_deg": float(
            np.degrees(best.cumulative_abs_yaw)
        ),
        "final_route_score": float(best.score),
        "state_trace": list(best.state_trace),
        "layer_trace": layer_trace,
    }
    selected = [
        [int(event_id), int(rank)]
        for event_id, rank in zip(best.selected_event_ids, best.selected_ranks)
    ]
    return final, list(best.report), selected


def apply_generators_with_heading_guard(
    v46: Any,
    motion_ref: np.ndarray,
    cond: np.ndarray,
    seam_mask: np.ndarray,
    args: Any,
    cfg: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refiner/diffusion edit seams; planner root heading remains authoritative."""
    stage: Dict[str, Any] = {}
    motion = np.asarray(motion_ref, dtype=np.float32).copy()
    stage["pre_refine_audit"] = (
        v46.audit_motion_np(motion, cfg)
        if hasattr(v46, "audit_motion_np")
        else {}
    )

    if bool(getattr(cfg, "refiner_enable", False)) and base.env_bool(
        "V46_46_USE_REFINER", True
    ):
        motion = v46.apply_refiner_model(
            motion,
            cond,
            seam_mask,
            getattr(args, "refiner", None),
            cfg,
        )
        stage["v45_refiner_audit"] = (
            v46.audit_motion_np(motion, cfg)
            if hasattr(v46, "audit_motion_np")
            else {}
        )

    if bool(getattr(cfg, "diffusion_enable", False)) and base.env_bool(
        "V46_46_USE_DIFFUSION", True
    ):
        motion = v46.apply_diffusion_model(
            motion,
            cond,
            seam_mask,
            getattr(args, "diffusion", None),
            cfg,
        )
        stage["v46_diffusion_audit"] = (
            v46.audit_motion_np(motion, cfg)
            if hasattr(v46, "audit_motion_np")
            else {}
        )

    if env_bool("V46_50_PROTECT_PLANNED_ROOT_HEADING", True):
        motion, heading_guard_pre_ik = restore_planned_root_heading_np(
            motion,
            motion_ref,
        )
        motion = base.enforce_contract(
            v46,
            motion,
            cfg,
            source_hint="v46_50_heading_guard_pre_ik",
        )
    else:
        heading_guard_pre_ik = {"enabled": False}
    stage["v46_50_heading_guard_pre_ik"] = heading_guard_pre_ik

    ik_report = {"enabled": False}
    if bool(getattr(cfg, "ik_enable", False)) and base.env_bool(
        "V46_46_USE_IK", True
    ):
        motion, ik_report = v46.true_lower_body_ik(motion, cfg)
    stage["v43_true_ik"] = ik_report

    if env_bool("V46_50_PROTECT_PLANNED_ROOT_HEADING", True):
        motion, heading_guard_post_ik = restore_planned_root_heading_np(
            motion,
            motion_ref,
        )
        motion = base.enforce_contract(
            v46,
            motion,
            cfg,
            source_hint="v46_50_heading_guard_post_ik",
        )
    else:
        heading_guard_post_ik = {"enabled": False}
    stage["v46_50_heading_guard_post_ik"] = heading_guard_post_ik
    stage["v46_50_final_heading_metrics"] = heading_metrics_np(
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )
    stage["final_audit"] = (
        v46.audit_motion_np(motion, cfg)
        if hasattr(v46, "audit_motion_np")
        else {}
    )
    stage["final_physical_gate"] = base.physical_quality_gate(stage["final_audit"])
    return motion.astype(np.float32), stage


def _patch_final_report(args: Any) -> None:
    path = Path(
        args.json
        or str(
            Path(args.out).with_name(
                Path(args.out).stem + ".v46_46_closed_loop_report.json"
            )
        )
    )
    if not path.is_file():
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    report["version"] = "v46_50_event_heading_closed_loop_scheduler"
    report["event_heading_planner"] = _LAST_HEADING_PLAN
    report["v46_50_env"] = {
        k: v for k, v in os.environ.items() if k.startswith("V46_50_")
    }
    motion_path = Path(args.out)
    if motion_path.is_file():
        x = np.load(motion_path, allow_pickle=True).astype(np.float32)
        if x.ndim == 3:
            x = x[0]
        report["v46_50_final_heading_metrics"] = heading_metrics_np(
            x,
            fps=float(getattr(args, "fps", os.environ.get("V46_51_FPS", 30.0))),
        )
    path.write_text(
        json.dumps(base.jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = base.parse_args(argv)
    if args.cmd != "generate":
        raise RuntimeError(args.cmd)

    # Monkey-patch only the policies owned by V46.50. All other current code,
    # including V46.38 routing and V46.46 boundary reselection, remains latest.
    base.assemble_closed_loop_reference = assemble_event_heading_reference
    base.apply_generators = apply_generators_with_heading_guard

    rc = base.generate_closed_loop(args)
    _patch_final_report(args)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
