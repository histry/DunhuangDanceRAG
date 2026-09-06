#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event-Heading Event-Heading Closed-Loop Planner.

This is an additive replacement entry point for
routing/boundary_closed_loop.py.  It reuses the latest retrieval,
transition simulation, refiner, diffusion, IK and rollback machinery, but
replaces two policies:

1. candidate assembly uses a planner-owned stage heading state rather than
   blindly inheriting the previous root yaw;
2. refiner/diffusion/IK output is guarded against changing the planned root
   heading.

Run with the same CLI as Boundary Closed-Loop:
    python routing/heading_closed_loop.py generate ...
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from routing.diversity import (
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

from routing.bidirectional_reachability import BackwardReachabilityModel
from routing.constraint_audit import (
    controlled_recovery_metadata,
    summarize_constraint_trials,
)
from routing.hierarchical_constraint_model import (
    CONSTRAINT_NAMES,
    ConstraintBudgetConfig,
    assess_candidate_constraints,
    build_feasible_set_scarcity_context,
    select_controlled_recovery_indices,
)
from routing.safe_source_coverage import (
    SafeSourceCoverageConfig,
    build_source_reservoir_layers,
    build_state_source_expansion_batches,
    select_bottleneck_layer_expansion_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# MOTION_ACTIVITY_INTEGRATION_BEGIN
from evaluation.motion_activity_analysis import (  # noqa: E402
    candidate_activity_assessment,
    motion_activity_metrics,
    save_stage_snapshot,
    slot_activity_target,
)
# MOTION_ACTIVITY_INTEGRATION_END

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
            "Event-Heading requires a heading-aware DB. Missing arrays: "
            + ", ".join(missing)
            + ". Rebuild with events/build_database_entry.py"
        )


def _align_core_to_stage_heading(
    motion_runtime: Any,
    prev: Optional[np.ndarray],
    core: np.ndarray,
    stage_heading_rad: float,
    cfg: Any,
    event_id: int,
    *, transition_frames: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = np.asarray(core, dtype=np.float32).copy()
    if len(out) == 0:
        return out, {"mode": "empty"}

    entry_before = float(root_yaw_np(out[:1])[0])
    dyaw = angle_diff(float(stage_heading_rad), entry_before)
    pivot = out[0, [ROOT_X_IDX, ROOT_Z_IDX]].copy()

    if hasattr(motion_runtime, "rotate_motion_around_y_np"):
        out = motion_runtime.rotate_motion_around_y_np(out, dyaw, pivot_xz=pivot)
    else:
        out = rotate_motion_constant_yaw_np(out, dyaw, pivot_xz=pivot)

    delta_xz = np.zeros(2, dtype=np.float32)
    landing_report = {}
    if prev is not None and len(prev):
        from motion_geometry.inbetween import duration_displacement
        landing,landing_report = duration_displacement(prev,out,transition_frames,cfg.fps,
            float(getattr(cfg,"transition_root_tangent_max_mps",1.35)))
        delta_xz = (
            np.asarray(prev[-1, [ROOT_X_IDX, ROOT_Z_IDX]], dtype=np.float32)
            + landing
            - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
        )
        out[:, ROOT_X_IDX] += float(delta_xz[0])
        out[:, ROOT_Z_IDX] += float(delta_xz[1])

    out = base.enforce_contract(
        motion_runtime,
        out,
        cfg,
        source_hint=f"event_heading_stage_heading_align:{event_id}",
    )
    entry_after = float(root_yaw_np(out[:1])[0])
    return out.astype(np.float32), {
        "schema": "event_heading_stage_heading_alignment",
        "mode": "planner_absolute_stage_heading_plus_xz_continuity",
        "event_id": int(event_id),
        "stage_heading_target_rad": float(stage_heading_rad),
        "stage_heading_target_deg": float(np.degrees(stage_heading_rad)),
        "entry_heading_before_rad": entry_before,
        "entry_heading_after_rad": entry_after,
        "dyaw_applied_rad": float(dyaw),
        "dyaw_applied_deg": float(np.degrees(dyaw)),
        "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
        "landing":landing_report,
        "root_y_ramp_applied": False,
    }


def _build_heading_proposal(
    motion_runtime: Any,
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
        motion_runtime,
        event_path,
        cfg,
        source_hint=f"event_heading_load_event:{event_id}",
    )
    has_prev = prev_motion is not None and len(prev_motion) > 0
    core_len, trans_len, length_info = base.choose_transition_lengths(
        motion_runtime,
        prev_motion,
        raw.shape[0],
        target_len,
        raw,
        slot,
        cfg,
    )
    core = base.resample_motion(motion_runtime, raw, core_len)
    core = base.enforce_contract(
        motion_runtime,
        core,
        cfg,
        source_hint=f"event_heading_core_resample:{event_id}",
    )

    event_meta = event_meta_from_db(db, event_id)
    heading_penalty, heading_detail = candidate_heading_penalty(
        event_meta,
        slot,
        stage_heading_rad,
        recent_turn_count=recent_turn_count,
    )

    core, align_report = _align_core_to_stage_heading(
        motion_runtime,
        prev_motion,
        core,
        stage_heading_rad,
        cfg,
        event_id,
        transition_frames=trans_len,
    )

    # Candidate activity is measured after duration resampling and heading
    # alignment, because this is the exact core that enters the route.
    activity_fps = float(getattr(cfg, "fps", 30.0))
    core_activity_metrics = motion_activity_metrics(core, fps=activity_fps)
    core_activity_target = slot_activity_target(slot)
    core_activity_assessment = candidate_activity_assessment(
        core_activity_metrics,
        core_activity_target,
    )

    bridge = np.zeros((0, EDGE_DIM), dtype=np.float32)
    if has_prev:
        bridge = base.build_bridge(motion_runtime, prev_motion, core, trans_len, cfg)
        risk = base.transition_risk(
            motion_runtime,
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

    risk = dict(risk)
    risk["motion_activity"] = {
        "metrics": core_activity_metrics,
        "assessment": core_activity_assessment,
    }
    physical_risk = float(base.risk_score(risk))
    combined = (
        physical_risk
        + env_float("EVENT_HEADING_PLANNER_WEIGHT", 0.85)
        * float(heading_penalty)
        + float(core_activity_assessment["penalty"])
    )
    heading_hard_reject = bool(heading_detail.get("hard_reject", False))
    activity_hard_reject = bool(core_activity_assessment["hard_reject"])
    hard_reject = bool(heading_hard_reject or activity_hard_reject)
    safe = bool((not hard_reject) and (base.risk_safe(risk) if has_prev else True))
    heading_detail = dict(heading_detail)
    heading_detail["motion_activity_hard_reject"] = activity_hard_reject

    length_info = dict(length_info)
    length_info["event_heading_heading"] = heading_detail
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
        "motion_activity": {
            "metrics": core_activity_metrics,
            "assessment": core_activity_assessment,
        },
        "source_frames": int(raw.shape[0]),
        "core_frames": int(len(core)),
    }


def assemble_event_heading_reference(
    motion_runtime: Any,
    slots: Sequence[Dict[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Dict[str, Any],
    cfg: Any,
    banned: Optional[Dict[int, set]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[List[int]]]:
    """Assemble a hard-safe route with viability-aware probabilistic constraints.

    Immutable physical, anatomical and severe-heading gates are evaluated by the
    existing exact simulator.  The route planner adds four bounded mechanisms:
    state-conditioned suffix reachability, source-targeted candidate expansion,
    scarcity-aware source budgets and continuous safe-set recovery resources.
    """
    global _LAST_HEADING_PLAN
    _heading_schema_guard(db)

    paths = np.asarray(db["paths"], dtype=object)
    blocked = banned or {}
    search = DynamicSearchConfig.from_environment()
    constraint_config = ConstraintBudgetConfig.from_environment(len(slots))
    source_coverage_config = SafeSourceCoverageConfig.from_environment()
    target_lengths = [base.slot_target_frames(slot, cfg) for slot in slots]
    reservoir_layers, reservoir_report = build_source_reservoir_layers(
        slots=slots,
        target_lengths=target_lengths,
        candidate_lists=candidate_lists,
        db=db,
        fps=float(getattr(cfg, "fps", 30.0)),
        blocked=blocked,
        config=source_coverage_config,
    )
    initial_reachability_reservoir_layers = [
        list(layer[: source_coverage_config.bottleneck_initial_reservoir_per_slot])
        for layer in reservoir_layers
    ]
    reachability_model = BackwardReachabilityModel.build(
        candidate_lists,
        db,
        constraint_config=constraint_config,
        additional_candidate_layers=initial_reachability_reservoir_layers,
    )

    route_progress = globals().get("ROUTE_PROGRESS")

    def progress_call(name: str, *args: Any, **kwargs: Any) -> Any:
        if route_progress is None or not hasattr(route_progress, name):
            return None
        try:
            return getattr(route_progress, name)(*args, **kwargs)
        except Exception:
            return None

    progress_call(
        "start",
        len(slots),
        {
            "beam_width": int(search.beam_width),
            "maximum_beam_width": int(search.maximum_beam_width),
            "branch_topk": int(search.branch_topk),
            "candidates_per_source": int(search.candidates_per_source),
            "method": "viability-aware Routing Budget",
            "source_targeted_expansion": bool(source_coverage_config.enabled),
            "continuous_recovery_budget": float(
                constraint_config.recovery_budget_total
            ),
        },
    )

    initial_heading = float(
        np.radians(env_float("EVENT_HEADING_STAGE_INITIAL_HEADING_DEG", 0.0))
    )
    initial_usage, initial_duals = constraint_config.initial_state()
    beam: List[DynamicBeamState] = [
        DynamicBeamState(
            motion=np.zeros((0, EDGE_DIM), dtype=np.float32),
            stage_heading=initial_heading,
            constraint_usage=initial_usage,
            constraint_duals=initial_duals,
            recovery_count=0,
            recovery_budget_used=0.0,
            minimum_future_reachability=1.0,
            minimum_future_viability_depth=len(slots),
            latest_future_viability_depth=len(slots),
            source_scarcity_exemption_count=0,
            family_scarcity_exemption_count=0,
            source_expansion_count=0,
            bottleneck_expansion_count=0,
        )
    ]
    layer_trace: List[Dict[str, Any]] = []
    collapse_events: List[Dict[str, Any]] = []
    bottleneck_expansion_events: List[Dict[str, Any]] = []

    for slot_idx, slot in enumerate(slots):
        target_len = int(target_lengths[slot_idx])
        candidates = [
            int(value)
            for value in candidate_lists[slot_idx]
            if int(value) not in blocked.get(slot_idx, set())
            and 0 <= int(value) < len(paths)
        ]
        if not candidates:
            raise RuntimeError(f"No candidates remain for slot {slot_idx}")
        progress_call(
            "slot_start", slot_idx, target_len, len(beam), len(candidates)
        )

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
            rank_lookup = {
                int(event_id): rank for rank, event_id in enumerate(candidates)
            }

            preordered: List[
                Tuple[float, int, int, Dict[str, Any], Dict[str, Any], str]
            ] = []
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
                static_reachability = reachability_model.get(
                    slot_idx, int(event_id)
                )
                future_probability = float(
                    static_reachability.get(
                        "future_reachability_probability", 0.0
                    )
                )
                order_score = (
                    float(heading_penalty)
                    + search.posterior_weight * float(prior_cost)
                    - constraint_config.future_reachability_weight
                    * math.log(max(future_probability, 1.0e-9))
                )
                preordered.append(
                    (
                        order_score,
                        original_rank,
                        int(event_id),
                        heading_detail,
                        prior_detail,
                        "retrieval",
                    )
                )
            preordered.sort(key=lambda row: (row[0], row[1], row[2]))

            proposals: List[Tuple[base.CandidateProposal, Dict[str, Any]]] = []

            def simulate_candidate(
                *,
                event_id: int,
                original_rank: int,
                prior_detail: Mapping[str, Any],
                origin: str,
            ) -> None:
                skip_pairs_raw = str(
                    os.environ.get(
                        "EVENT_HEADING_EXACT_SKIP_PAIRS",
                        "",
                    )
                ).strip()

                skip_pairs = set()
                if skip_pairs_raw:
                    for item in skip_pairs_raw.split(","):
                        item = item.strip()
                        if not item:
                            continue
                        try:
                            slot_text, event_text = item.split(":", 1)
                            skip_pairs.add(
                                (
                                    int(slot_text),
                                    int(event_text),
                                )
                            )
                        except (TypeError, ValueError) as exc:
                            raise RuntimeError(
                                "Invalid EVENT_HEADING_EXACT_SKIP_PAIRS "
                                f"item: {item!r}; expected slot:event"
                            ) from exc

                if (
                    int(slot_idx),
                    int(event_id),
                ) in skip_pairs:
                    print(
                        "[Routing Budget-CANDIDATE-SKIP] "
                        + json.dumps(
                            {
                                "slot": int(slot_idx),
                                "state_index": int(state_index),
                                "event_id": int(event_id),
                                "candidate_rank": int(original_rank),
                                "target_frames": int(target_len),
                                "candidate_origin": str(origin),
                                "decision": (
                                    "skip_known_nonreturning_"
                                    "exact_candidate"
                                ),
                                "safety_gates_relaxed": False,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    return

                progress_token = progress_call(
                    "candidate_start",
                    slot=slot_idx,
                    state_index=state_index,
                    event_id=event_id,
                    candidate_rank=original_rank,
                    target_frames=target_len,
                )
                proposal, extra0 = _build_heading_proposal(
                    motion_runtime=motion_runtime,
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
                progress_call(
                    "candidate_finish", progress_token, safe=bool(proposal.safe)
                )
                calibration_penalty, calibration_detail = source_calibration_penalty(
                    db,
                    previous_event_id,
                    int(event_id),
                )
                extra = dict(extra0)
                extra["candidate_origin"] = str(origin)
                extra["route_prior"] = dict(prior_detail)
                extra["route_prior_cost"] = float(
                    prior_detail.get("negative_log_cost", 0.0)
                )
                extra["source_calibration"] = calibration_detail
                extra["source_calibration_penalty"] = float(calibration_penalty)
                proposals.append((proposal, extra))

            for (
                _order_score,
                original_rank,
                event_id,
                _heading_detail,
                prior_detail,
                origin,
            ) in preordered:
                simulate_candidate(
                    event_id=event_id,
                    original_rank=original_rank,
                    prior_detail=prior_detail,
                    origin=origin,
                )

            initial_safe_event_ids = [
                int(proposal.event_id)
                for proposal, _extra in proposals
                if bool(proposal.safe)
            ]
            expansion_ids: List[int] = []
            expansion_batches, expansion_report = build_state_source_expansion_batches(
                reservoir_event_ids=reservoir_layers[slot_idx],
                attempted_event_ids=[
                    int(proposal.event_id) for proposal, _extra in proposals
                ],
                hard_safe_event_ids=initial_safe_event_ids,
                selected_event_ids=state.selected_event_ids,
                previous_event_id=previous_event_id,
                db=db,
                config=source_coverage_config,
            )
            reservoir_rank_lookup = {
                int(event_id): index
                for index, event_id in enumerate(reservoir_layers[slot_idx])
            }
            executed_batches: List[Dict[str, Any]] = []
            exact_budget_used = 0
            if source_coverage_config.enabled:
                for batch_index, batch in enumerate(expansion_batches):
                    batch_attempted: List[int] = []
                    batch_safe: List[int] = []
                    for event_id in batch:
                        if exact_budget_used >= source_coverage_config.expansion_maximum_exact:
                            break
                        original_rank = int(
                            len(candidates)
                            + reservoir_rank_lookup.get(int(event_id), 0)
                        )
                        _prior_cost, prior_detail = route_prior_cost(
                            slot_idx,
                            int(event_id),
                            previous_event_id=previous_event_id,
                            fallback_rank=original_rank,
                            candidate_count=len(candidates)
                            + len(reservoir_layers[slot_idx]),
                        )
                        before = len(proposals)
                        simulate_candidate(
                            event_id=int(event_id),
                            original_rank=original_rank,
                            prior_detail=prior_detail,
                            origin="safe_source_expansion",
                        )
                        exact_budget_used += 1
                        expansion_ids.append(int(event_id))
                        batch_attempted.append(int(event_id))
                        if len(proposals) > before and bool(proposals[-1][0].safe):
                            batch_safe.append(int(event_id))
                    current_safe = [
                        int(proposal.event_id)
                        for proposal, _extra in proposals
                        if bool(proposal.safe)
                    ]
                    current_sources = {
                        event_identity(db, value)["source_uid"] for value in current_safe
                    }
                    current_families = {
                        event_identity(db, value)["family_id"] for value in current_safe
                    }
                    executed_batches.append(
                        {
                            "batch_index": int(batch_index),
                            "attempted_event_ids": batch_attempted,
                            "hard_safe_event_ids": batch_safe,
                            "safe_source_count_after_batch": len(current_sources),
                            "safe_family_count_after_batch": len(current_families),
                        }
                    )
                    if (
                        len(current_sources) >= source_coverage_config.target_safe_sources
                        and len(current_families) >= source_coverage_config.target_safe_families
                    ):
                        break
                    if exact_budget_used >= source_coverage_config.expansion_maximum_exact:
                        break

            hard_safe_event_ids = [
                int(proposal.event_id)
                for proposal, _extra in proposals
                if bool(proposal.safe)
            ]
            safe_count = len(hard_safe_event_ids)
            safe_ratio = safe_count / max(1, len(proposals))
            scarcity_context = build_feasible_set_scarcity_context(
                db=db,
                hard_safe_event_ids=hard_safe_event_ids,
                all_event_ids=[
                    int(proposal.event_id) for proposal, _extra in proposals
                ],
                config=constraint_config,
            )
            expansion_report = dict(expansion_report)
            expansion_report["executed_batches"] = executed_batches
            expansion_report["additional_exact_simulations"] = int(exact_budget_used)
            expansion_report["safe_source_count_after"] = int(
                scarcity_context.safe_source_count
            )
            expansion_report["safe_family_count_after"] = int(
                scarcity_context.safe_family_count
            )
            expansion_report["safe_sources_after"] = sorted(
                dict(scarcity_context.safe_source_counts)
            )
            expansion_report["safe_families_after"] = sorted(
                dict(scarcity_context.safe_family_counts)
            )

            def evaluate_current_proposals() -> List[Dict[str, Any]]:
                rows: List[Dict[str, Any]] = []
                for proposal, extra in proposals:
                    observability = observability_from_extra(
                        extra,
                        safe_ratio=safe_ratio,
                    )
                    if bool(proposal.safe):
                        reachability = reachability_model.query(
                            slot=slot_idx,
                            event_id=int(proposal.event_id),
                            selected_event_ids=state.selected_event_ids,
                            constraint_usage=state.constraint_usage,
                            dual_variables=state.constraint_duals,
                            recovery_budget_used=state.recovery_budget_used,
                            observability=observability,
                            scarcity_context=scarcity_context,
                        )
                    else:
                        reachability = reachability_model.get(
                            slot_idx, int(proposal.event_id)
                        )
                        reachability["future_reachable"] = False
                        reachability["terminal_reachable"] = False
                        reachability["future_viability_depth"] = 0
                        reachability["reachable_until_slot"] = int(slot_idx)
                        reachability["future_reachability_probability"] = 0.0
                    assessment = assess_candidate_constraints(
                        db=db,
                        event_id=int(proposal.event_id),
                        selected_event_ids=state.selected_event_ids,
                        observability=observability,
                        future_reachability_probability=float(
                            reachability.get("future_reachability_probability", 0.0)
                        ),
                        slot_index=slot_idx,
                        constraint_usage=state.constraint_usage,
                        dual_variables=state.constraint_duals,
                        config=constraint_config,
                        scarcity_context=scarcity_context,
                    )
                    extra["observability"] = float(observability)
                    extra["route_uncertainty"] = 1.0 - float(observability)
                    extra["future_reachability"] = reachability
                    extra["constraint_assessment"] = assessment
                    extra["diversity"] = dict(assessment["diversity"])
                    base_selection_score = proposal_selection_score(
                        proposal,
                        extra,
                        primary_event_id=primary_event_id,
                    )
                    hard_safe = bool(proposal.safe)
                    terminal_reachable = bool(
                        reachability.get(
                            "terminal_reachable",
                            reachability.get("future_reachable", False),
                        )
                    )
                    preferred = bool(
                        hard_safe and assessment["within_budget"] and terminal_reachable
                    )
                    rows.append(
                        {
                            "proposal": proposal,
                            "extra": extra,
                            "hard_safe": hard_safe,
                            "future_reachable": terminal_reachable,
                            "terminal_reachable": terminal_reachable,
                            "future_viability_depth": int(
                                reachability.get("future_viability_depth", 0) or 0
                            ),
                            "preferred": preferred,
                            "base_selection_score": float(base_selection_score),
                            "constraint_assessment": assessment,
                            "future_reachability": reachability,
                            "observability": float(observability),
                        }
                    )
                return rows

            evaluated = evaluate_current_proposals()
            predicted_dead_ends = [
                int(row["future_reachability"].get("future_first_dead_end_slot"))
                for row in evaluated
                if row["hard_safe"]
                and row["future_reachability"].get("future_first_dead_end_slot")
                is not None
                and int(row["future_reachability"].get("future_first_dead_end_slot"))
                > slot_idx
            ]
            bottleneck_report: Dict[str, Any] = {
                "schema": "predicted_bottleneck_source_family_expansion",
                "triggered": False,
            }
            if predicted_dead_ends:
                bottleneck_slot = min(predicted_dead_ends)
                bottleneck_candidates, selection_report = (
                    select_bottleneck_layer_expansion_candidates(
                        reservoir_event_ids=reservoir_layers[bottleneck_slot],
                        active_event_ids=reachability_model.layers[bottleneck_slot],
                        selected_event_ids=state.selected_event_ids,
                        db=db,
                        config=source_coverage_config,
                    )
                )
                activation_report = reachability_model.activate_candidates(
                    slot=bottleneck_slot,
                    event_ids=bottleneck_candidates,
                    reason=f"predicted_from_slot_{slot_idx}",
                )
                bottleneck_report = {
                    **dict(selection_report),
                    **dict(activation_report),
                    "predicted_from_slot": int(slot_idx),
                    "predicted_first_dead_end_slot": int(bottleneck_slot),
                }
                if bool(activation_report.get("triggered", False)):
                    bottleneck_expansion_events.append(dict(bottleneck_report))
                    evaluated = evaluate_current_proposals()

            recovery_indices = select_controlled_recovery_indices(
                evaluated,
                current_recovery_budget_used=state.recovery_budget_used,
                config=constraint_config,
            )
            trials: List[Dict[str, Any]] = []
            eligible_rows: List[Dict[str, Any]] = []
            for evaluation_index, evaluation in enumerate(evaluated):
                proposal = evaluation["proposal"]
                extra = evaluation["extra"]
                assessment = evaluation["constraint_assessment"]
                preferred = bool(evaluation["preferred"])
                recovery_triggered = bool(evaluation_index in recovery_indices)
                eligible = bool(preferred or recovery_triggered)
                recovery_charge = (
                    float(assessment.get("recovery_charge", 0.0))
                    if recovery_triggered
                    else 0.0
                )
                final_selection_score = float(
                    evaluation["base_selection_score"]
                    + assessment["probabilistic_auxiliary_cost"]
                    + (
                        constraint_config.recovery_penalty * recovery_charge
                        if recovery_triggered
                        else 0.0
                    )
                )
                audit_assessment = dict(assessment)
                audit_assessment.pop("usage_after_tuple", None)
                audit_assessment.pop("duals_after_tuple", None)
                comparisons = audit_assessment.pop("hierarchy_comparisons", [])
                audit_assessment["hierarchy_comparison_count"] = len(comparisons)
                trial = {
                    "event_id": int(proposal.event_id),
                    "rank": int(proposal.rank),
                    "candidate_origin": str(
                        extra.get("candidate_origin", "retrieval")
                    ),
                    "safe": bool(proposal.safe),
                    "future_reachable": bool(evaluation["future_reachable"]),
                    "terminal_reachable": bool(evaluation["terminal_reachable"]),
                    "future_viability_depth": int(evaluation["future_viability_depth"]),
                    "reachable_until_slot": evaluation["future_reachability"].get(
                        "reachable_until_slot"
                    ),
                    "preferred": preferred,
                    "recovery_candidate": bool(
                        evaluation["hard_safe"]
                        and not preferred
                        and (
                            evaluation["terminal_reachable"]
                            or (
                                evaluation["future_viability_depth"]
                                >= constraint_config.recovery_minimum_viability_depth
                                and int(
                                    evaluation["future_reachability"].get(
                                        "future_safe_successor_count", 0
                                    )
                                )
                                > 0
                            )
                        )
                    ),
                    "recovery_triggered": recovery_triggered,
                    "eligible": eligible,
                    "combined_score": float(proposal.risk_score),
                    "base_selection_score": float(
                        evaluation["base_selection_score"]
                    ),
                    "selection_score": final_selection_score,
                    "physical_risk_score": float(
                        extra.get("physical_risk_score", proposal.risk_score)
                    ),
                    "heading_penalty": float(extra.get("heading_penalty", 0.0)),
                    "hard_reject": bool(
                        extra.get("heading_detail", {}).get("hard_reject", False)
                    ),
                    "observability": float(evaluation["observability"]),
                    "route_prior": dict(extra.get("route_prior", {})),
                    "source_calibration": dict(
                        extra.get("source_calibration", {})
                    ),
                    "future_reachability": dict(
                        evaluation["future_reachability"]
                    ),
                    "constraint_assessment": audit_assessment,
                    "diversity": dict(assessment["diversity"]),
                }
                trials.append(trial)
                if eligible:
                    eligible_row = dict(evaluation)
                    eligible_row["recovery_triggered"] = recovery_triggered
                    eligible_row["selection_score"] = final_selection_score
                    eligible_row["recovery_charge"] = recovery_charge
                    eligible_rows.append(eligible_row)

            diagnostic_summary = summarize_constraint_trials(
                trials,
                source_expansion=expansion_report,
                scarcity_context=scarcity_context.to_dict(),
            )
            diagnostic_summary["predicted_bottleneck_expansion"] = dict(
                bottleneck_report
            )
            if diagnostic_summary["constraint_collapse_detected"]:
                collapse_events.append(
                    {
                        "slot": int(slot_idx),
                        "state_index": int(state_index),
                        "prefix_event_ids": list(map(int, state.selected_event_ids)),
                        **dict(diagnostic_summary),
                    }
                )

            for evaluation in eligible_rows:
                proposal = evaluation["proposal"]
                extra = evaluation["extra"]
                assessment = evaluation["constraint_assessment"]
                observability = float(evaluation["observability"])
                recovery_triggered = bool(evaluation["recovery_triggered"])
                recovery_charge = float(evaluation["recovery_charge"])
                selection_score = float(evaluation["selection_score"])
                future_reachability = dict(evaluation["future_reachability"])
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
                candidate_origin = str(
                    extra.get("candidate_origin", "retrieval")
                )
                if recovery_triggered:
                    decision = "selected_continuous_safe_set_recovery"
                elif candidate_origin == "safe_source_expansion":
                    decision = "selected_safe_source_expansion"
                elif int(proposal.event_id) == primary_event_id:
                    decision = "selected_primary_soft_prior"
                else:
                    decision = "selected_state_aware_dynamic_beam"
                proposal.decision = decision
                source_frames = int(extra.get("source_frames", 0) or 0)
                if source_frames <= 0:
                    source_frames = base.load_event_motion(
                        motion_runtime,
                        proposal.event_path,
                        cfg,
                        "state_aware_routing_budget_warp_probe",
                    ).shape[0]
                recovery_budget_after = float(
                    state.recovery_budget_used + recovery_charge
                )
                recovery_count_after = int(
                    state.recovery_count + int(recovery_triggered)
                )
                recovery_metadata = controlled_recovery_metadata(
                    assessment,
                    triggered=recovery_triggered,
                    recovery_count_after=recovery_count_after,
                    recovery_budget_used_before=state.recovery_budget_used,
                    recovery_budget_used_after=recovery_budget_after,
                    recovery_budget_total=constraint_config.recovery_budget_total,
                )
                row_assessment = dict(assessment)
                row_assessment.pop("usage_after_tuple", None)
                row_assessment.pop("duals_after_tuple", None)
                comparisons = row_assessment.pop("hierarchy_comparisons", [])
                row_assessment["hierarchy_comparison_count"] = len(comparisons)
                row = {
                    "slot": int(slot_idx),
                    "event_id": int(proposal.event_id),
                    "candidate_rank": int(proposal.rank),
                    "candidate_origin": candidate_origin,
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
                    "dynamic_selection_score": selection_score,
                    "safe_predicted": bool(proposal.safe),
                    "decision": decision,
                    "primary_event_id": primary_event_id,
                    "planned_event_diverged": bool(
                        int(proposal.event_id) != primary_event_id
                    ),
                    "observability": observability,
                    "route_prior": dict(extra.get("route_prior", {})),
                    "source_calibration": dict(
                        extra.get("source_calibration", {})
                    ),
                    "motion_activity": dict(extra.get("motion_activity", {})),
                    "future_reachability": future_reachability,
                    "probabilistic_constraint_routing": row_assessment,
                    "controlled_recovery": recovery_metadata,
                    "safe_source_expansion": dict(expansion_report),
                    "predicted_bottleneck_expansion": dict(bottleneck_report),
                    "source_scarcity": scarcity_context.to_dict(),
                    "feasible_set_scarcity": scarcity_context.to_dict(),
                    "dynamic_hard_mask": {
                        "physical_anatomy_heading_safe": bool(proposal.safe),
                        "preference_budget_valid": bool(assessment["within_budget"]),
                        "future_state_reachable": bool(
                            future_reachability.get("terminal_reachable", False)
                        ),
                        "future_viability_depth": int(
                            future_reachability.get("future_viability_depth", 0) or 0
                        ),
                        "immutable_safety_relaxed": False,
                        "soft_reasons": list(
                            assessment["diversity"].get("soft_reasons", [])
                        ),
                    },
                    "diversity": dict(assessment["diversity"]),
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
                    "method": "viability-aware Routing Budget",
                }
                source_exemption_used = bool(
                    scarcity_context.source_scarcity_exemption
                    and (
                        float(assessment["raw_violations"].get("source_run", 0.0)) > 0.0
                        or float(assessment["raw_violations"].get("source_share", 0.0)) > 0.0
                    )
                )
                family_exemption_used = bool(
                    scarcity_context.family_scarcity_exemption
                    and (
                        float(assessment["raw_violations"].get("family_share", 0.0)) > 0.0
                        or float(assessment["raw_violations"].get("hierarchy_repetition", 0.0)) > 0.0
                    )
                )
                state_row = {
                    "slot": int(slot_idx),
                    "event_id": int(proposal.event_id),
                    "intent": intent,
                    "candidate_origin": candidate_origin,
                    "stage_heading_before_rad": stage_before,
                    "event_delta_rad": event_delta,
                    "stage_heading_after_rad": stage_after,
                    "cumulative_abs_yaw_rad": float(
                        state.cumulative_abs_yaw + abs(event_delta)
                    ),
                    "observability": observability,
                    "future_reachability_probability": float(
                        assessment["future_reachability_probability"]
                    ),
                    "terminal_reachable": bool(
                        future_reachability.get("terminal_reachable", False)
                    ),
                    "future_viability_depth": int(
                        future_reachability.get("future_viability_depth", 0) or 0
                    ),
                    "reachable_until_slot": future_reachability.get(
                        "reachable_until_slot"
                    ),
                    "future_first_dead_end_slot": future_reachability.get(
                        "future_first_dead_end_slot"
                    ),
                    "constraint_usage": dict(
                        assessment["constraint_usage_after"]
                    ),
                    "constraint_dual_variables": dict(
                        assessment["dual_variables_after"]
                    ),
                    "source_scarcity": scarcity_context.to_dict(),
                    "safe_source_expansion": dict(expansion_report),
                    "predicted_bottleneck_expansion": dict(bottleneck_report),
                    "controlled_recovery": recovery_metadata,
                    "prefix_score": float(state.score + selection_score),
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
                        score=state.score + selection_score,
                        observability=observability,
                        constraint_usage=tuple(assessment["usage_after_tuple"]),
                        constraint_duals=tuple(assessment["duals_after_tuple"]),
                        recovery_count=recovery_count_after,
                        recovery_budget_used=recovery_budget_after,
                        minimum_future_reachability=min(
                            float(state.minimum_future_reachability),
                            float(assessment["future_reachability_probability"]),
                        ),
                        minimum_future_viability_depth=min(
                            int(state.minimum_future_viability_depth),
                            int(future_reachability.get("future_viability_depth", 0) or 0),
                        ),
                        latest_future_viability_depth=int(
                            future_reachability.get("future_viability_depth", 0) or 0
                        ),
                        source_scarcity_exemption_count=(
                            state.source_scarcity_exemption_count
                            + int(source_exemption_used)
                        ),
                        family_scarcity_exemption_count=(
                            state.family_scarcity_exemption_count
                            + int(family_exemption_used)
                        ),
                        source_expansion_count=(
                            state.source_expansion_count
                            + int(candidate_origin == "safe_source_expansion")
                        ),
                        bottleneck_expansion_count=(
                            state.bottleneck_expansion_count
                            + int(bool(bottleneck_report.get("triggered", False)))
                        ),
                    )
                )

            state_diagnostics.append(
                {
                    "state_index": int(state_index),
                    "prefix_event_ids": list(map(int, state.selected_event_ids)),
                    "candidate_subset": list(map(int, subset)),
                    "source_expansion_candidates": list(map(int, expansion_ids)),
                    "predicted_bottleneck_expansion": dict(bottleneck_report),
                    **dict(diagnostic_summary),
                    "hard_contracts_relaxed": False,
                    "current_recovery_count": int(state.recovery_count),
                    "current_recovery_budget_used": float(
                        state.recovery_budget_used
                    ),
                    "recovery_budget_total": float(
                        constraint_config.recovery_budget_total
                    ),
                }
            )

        if not expanded:
            raise DynamicRouteDeadEnd(
                slot_idx,
                {
                    "retained_prefixes": int(len(beam)),
                    "candidate_count": int(len(candidates)),
                    "reservoir_candidate_count": int(
                        len(reservoir_layers[slot_idx])
                    ),
                    "state_diagnostics": state_diagnostics,
                    "hard_contracts_relaxed": False,
                    "preference_constraints_are_probabilistic": True,
                    "source_scarcity_budgeting_enabled": bool(
                        constraint_config.source_scarcity_enabled
                    ),
                    "family_scarcity_budgeting_enabled": bool(
                        constraint_config.family_scarcity_enabled
                    ),
                    "continuous_recovery_enabled": bool(
                        constraint_config.controlled_recovery_enabled
                    ),
                    "backward_reachability": reachability_model.runtime_summary(),
                },
            )

        width = adaptive_beam_width(
            search,
            [state.observability for state in expanded],
        )
        beam = prune_states(expanded, db, width=width)
        layer_summary = {
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
            "retained_recovery_counts": [
                int(state.recovery_count) for state in beam
            ],
            "retained_recovery_budget_used": [
                float(state.recovery_budget_used) for state in beam
            ],
            "retained_source_scarcity_exemptions": [
                int(state.source_scarcity_exemption_count) for state in beam
            ],
            "retained_family_scarcity_exemptions": [
                int(state.family_scarcity_exemption_count) for state in beam
            ],
            "retained_source_expansion_counts": [
                int(state.source_expansion_count) for state in beam
            ],
            "retained_minimum_future_reachability": [
                float(state.minimum_future_reachability) for state in beam
            ],
            "retained_latest_future_viability_depth": [
                int(state.latest_future_viability_depth) for state in beam
            ],
            "retained_minimum_future_viability_depth": [
                int(state.minimum_future_viability_depth) for state in beam
            ],
            "retained_bottleneck_expansion_counts": [
                int(state.bottleneck_expansion_count) for state in beam
            ],
            "constraint_collapse_states": int(
                sum(
                    bool(row.get("constraint_collapse_detected", False))
                    for row in state_diagnostics
                )
            ),
            "state_diagnostics": state_diagnostics,
        }
        layer_trace.append(layer_summary)
        progress_call("slot_finish", slot_idx, len(expanded), len(beam))

    best = min(beam, key=lambda state: float(state.score))
    final = base.enforce_contract(
        motion_runtime,
        np.asarray(best.motion, dtype=np.float32),
        cfg,
        source_hint="viability_aware_routing_budget_dynamic_heading_reference_final",
    )
    _LAST_HEADING_PLAN = {
        "schema": "viability_aware_bidirectional_hierarchical_probabilistic_route",
        "method": "viability-aware Routing Budget",
        "search": {
            "beam_width": int(search.beam_width),
            "maximum_beam_width": int(search.maximum_beam_width),
            "branch_topk": int(search.branch_topk),
            "candidates_per_source": int(search.candidates_per_source),
            "primary_is_soft_prior": True,
            "exact_simulation_authoritative": True,
            "physical_anatomy_heading_gates_immutable": True,
        },
        "constraint_configuration": constraint_config.to_dict(),
        "source_coverage_configuration": source_coverage_config.to_dict(),
        "source_candidate_reservoir": reservoir_report,
        "backward_reachability": reachability_model.runtime_summary(),
        "constraint_collapse_events": collapse_events,
        "predicted_bottleneck_expansion_events": bottleneck_expansion_events,
        "graph_route_prior": route_prior_summary(),
        "initial_stage_heading_rad": float(initial_heading),
        "final_stage_heading_rad": float(best.stage_heading),
        "cumulative_abs_event_yaw_rad": float(best.cumulative_abs_yaw),
        "cumulative_abs_event_yaw_deg": float(
            np.degrees(best.cumulative_abs_yaw)
        ),
        "final_route_score": float(best.score),
        "final_constraint_usage": dict(
            zip(CONSTRAINT_NAMES, map(float, best.constraint_usage))
        ),
        "final_constraint_dual_variables": dict(
            zip(CONSTRAINT_NAMES, map(float, best.constraint_duals))
        ),
        "controlled_recovery_count": int(best.recovery_count),
        "continuous_recovery_budget_used": float(best.recovery_budget_used),
        "continuous_recovery_budget_total": float(
            constraint_config.recovery_budget_total
        ),
        "source_scarcity_exemption_count": int(
            best.source_scarcity_exemption_count
        ),
        "family_scarcity_exemption_count": int(
            best.family_scarcity_exemption_count
        ),
        "safe_source_expansion_selection_count": int(
            best.source_expansion_count
        ),
        "minimum_future_reachability": float(
            best.minimum_future_reachability
        ),
        "minimum_future_viability_depth": int(
            best.minimum_future_viability_depth
        ),
        "bottleneck_expansion_count": int(best.bottleneck_expansion_count),
        "state_trace": list(best.state_trace),
        "layer_trace": layer_trace,
    }
    selected = [
        [int(event_id), int(rank)]
        for event_id, rank in zip(best.selected_event_ids, best.selected_ranks)
    ]
    progress_call("finish")
    return final, list(best.report), selected

def apply_generators_with_heading_guard(
    motion_runtime: Any,
    motion_ref: np.ndarray,
    cond: np.ndarray,
    seam_mask: np.ndarray,
    args: Any,
    cfg: Any,
    *,
    sliding_support_eligible: Optional[np.ndarray] = None,
    protected_geometry_mask: Optional[np.ndarray] = None,
    ik_protected_frame_mask: Optional[np.ndarray] = None,
    ik_candidate_guard: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Refiner/diffusion edit seams; planner root heading remains authoritative."""
    stage: Dict[str, Any] = {}
    motion = np.asarray(motion_ref, dtype=np.float32).copy()
    protected = None
    if protected_geometry_mask is not None:
        protected = np.asarray(protected_geometry_mask, dtype=bool).reshape(-1)
        if len(protected) != len(motion):
            raise ValueError("protected_geometry_mask length mismatch")

    def restore_protected_geometry(value):
        if protected is None or not np.any(protected):
            return value
        restored = np.asarray(value, dtype=np.float32).copy()
        restored[protected, 4:] = motion_ref[protected, 4:]
        return restored

    if protected is not None:
        stage["protected_geometry_contract"] = {
            "frames": int(protected.sum()),
            "channels": [4, int(motion.shape[1])],
            "contact_channels_recomputed": True,
        }

    def physical_audit(value: np.ndarray) -> Dict[str, Any]:
        if not hasattr(motion_runtime, "audit_motion_np"):
            return {}
        return dict(
            motion_runtime.audit_motion_np(
                value,
                cfg,
                sliding_support_eligible=sliding_support_eligible,
            )
        )

    stage["pre_refine_audit"] = physical_audit(motion)
    stage["motion_activity_retrieval"] = save_stage_snapshot(
        getattr(args, "out", None),
        "retrieval",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )

    if bool(getattr(cfg, "refiner_enable", False)) and base.env_bool(
        "BOUNDARY_USE_REFINER", True
    ):
        motion = motion_runtime.apply_refiner_model(
            motion,
            cond,
            seam_mask,
            getattr(args, "refiner", None),
            cfg,
            sliding_support_eligible=sliding_support_eligible,
        )
        motion = restore_protected_geometry(motion)
        stage["boundary_refiner_audit"] = physical_audit(motion)
    stage["motion_activity_refiner"] = save_stage_snapshot(
        getattr(args, "out", None),
        "refiner",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )

    if bool(getattr(cfg, "diffusion_enable", False)) and base.env_bool(
        "BOUNDARY_USE_DIFFUSION", True
    ):
        motion = motion_runtime.apply_diffusion_model(
            motion,
            cond,
            seam_mask,
            getattr(args, "diffusion", None),
            cfg,
            sliding_support_eligible=sliding_support_eligible,
        )
        motion = restore_protected_geometry(motion)
        stage["motion_diffusion_audit"] = physical_audit(motion)
    stage["motion_activity_diffusion"] = save_stage_snapshot(
        getattr(args, "out", None),
        "diffusion",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )

    if env_bool("EVENT_HEADING_PROTECT_PLANNED_ROOT_HEADING", True):
        motion, heading_guard_pre_ik = restore_planned_root_heading_np(
            motion,
            motion_ref,
        )
        motion = base.enforce_contract(
            motion_runtime,
            motion,
            cfg,
            source_hint="event_heading_heading_guard_pre_ik",
        )
    else:
        heading_guard_pre_ik = {"enabled": False}
    stage["event_heading_heading_guard_pre_ik"] = heading_guard_pre_ik

    ik_report = {"enabled": False}
    if bool(getattr(cfg, "ik_enable", False)) and base.env_bool(
        "BOUNDARY_USE_IK", True
    ):
        motion, ik_report = motion_runtime.true_lower_body_ik(
            motion,
            cfg,
            sliding_support_eligible=sliding_support_eligible,
            protected_frame_mask=ik_protected_frame_mask,
            candidate_guard=ik_candidate_guard,
        )
    stage["lower_body_ik_true_ik"] = ik_report

    if env_bool("EVENT_HEADING_PROTECT_PLANNED_ROOT_HEADING", True):
        motion, heading_guard_post_ik = restore_planned_root_heading_np(
            motion,
            motion_ref,
        )
        motion = base.enforce_contract(
            motion_runtime,
            motion,
            cfg,
            source_hint="event_heading_heading_guard_post_ik",
        )
    else:
        heading_guard_post_ik = {"enabled": False}
    stage["event_heading_heading_guard_post_ik"] = heading_guard_post_ik
    stage["event_heading_final_heading_metrics"] = heading_metrics_np(
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )
    stage["final_audit"] = physical_audit(motion)
    stage["final_physical_gate"] = base.physical_quality_gate(stage["final_audit"])
    stage["motion_activity_full_ik"] = save_stage_snapshot(
        getattr(args, "out", None),
        "full_ik",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )
    return motion.astype(np.float32), stage

def _patch_final_report(args: Any) -> None:
    path = Path(
        args.json
        or str(
            Path(args.out).with_name(
                Path(args.out).stem + ".boundary_closed_loop_closed_loop_report.json"
            )
        )
    )
    if not path.is_file():
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    report["version"] = "event_heading_closed_loop_scheduler"
    report["event_heading_planner"] = _LAST_HEADING_PLAN
    report["event_heading_env"] = {
        k: v for k, v in os.environ.items() if k.startswith("EVENT_HEADING_")
    }
    motion_path = Path(args.out)
    if motion_path.is_file():
        x = np.load(motion_path, allow_pickle=True).astype(np.float32)
        if x.ndim == 3:
            x = x[0]
        report["event_heading_final_heading_metrics"] = heading_metrics_np(
            x,
            fps=float(getattr(args, "fps", os.environ.get("GENERATION_FPS", 30.0))),
        )
    path.write_text(
        json.dumps(base.jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = base.parse_args(argv)
    if args.cmd != "generate":
        raise RuntimeError(args.cmd)

    # Monkey-patch only the policies owned by Event-Heading. All other current code,
    # including Semantic Routing routing and Boundary Closed-Loop boundary reselection, remains latest.
    base.assemble_closed_loop_reference = assemble_event_heading_reference
    base.apply_generators = apply_generators_with_heading_guard

    rc = base.generate_closed_loop(args)
    _patch_final_report(args)
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
