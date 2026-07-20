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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.v46_46_boundary_closed_loop as base  # noqa: E402
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
        piece = base.resample_motion(v46, piece, int(target_len))
        piece = base.enforce_contract(
            v46,
            piece,
            cfg,
            source_hint=f"v46_50_slot_exact_len:{event_id}",
        )
        length_info["slot_exact_repair_applied"] = True
        length_info["slot_exact_frames_after"] = int(piece.shape[0])

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
    """Planner-owned stage heading + boundary-safe candidate assembly."""
    global _LAST_HEADING_PLAN
    _heading_schema_guard(db)

    paths = np.asarray(db["paths"], dtype=object)
    banned = banned or {}
    pieces: List[np.ndarray] = []
    report: List[Dict[str, Any]] = []
    selected: List[List[int]] = []
    cursor = 0

    stage_heading = float(
        np.radians(env_float("V46_50_STAGE_INITIAL_HEADING_DEG", 0.0))
    )
    initial_heading = stage_heading
    recent_turn_count = 0
    cumulative_abs_yaw = 0.0
    state_trace: List[Dict[str, Any]] = []

    for slot_idx, slot in enumerate(slots):
        target_len = base.slot_target_frames(slot, cfg)
        prev = (
            np.concatenate(pieces, axis=0).astype(np.float32)
            if pieces
            else None
        )
        candidates = [
            int(x)
            for x in candidate_lists[slot_idx]
            if int(x) not in banned.get(slot_idx, set())
            and 0 <= int(x) < len(paths)
        ]
        if not candidates:
            raise RuntimeError(f"No candidates remain for slot {slot_idx}")

        # Heading-aware pre-ordering before expensive simulated stitching.
        preordered: List[Tuple[float, int, Dict[str, Any]]] = []
        for original_rank, event_id in enumerate(candidates):
            meta = event_meta_from_db(db, event_id)
            pen, detail = candidate_heading_penalty(
                meta,
                slot,
                stage_heading,
                recent_turn_count=recent_turn_count,
            )
            if not _heading_valid(db, event_id):
                detail["hard_reject"] = True
                pen += 1e6
            preordered.append((float(pen), int(event_id), detail))
        preordered.sort(key=lambda row: row[0])

        max_trials = max(
            1,
            min(
                len(preordered),
                base.env_int("V46_50_HEADING_TRIAL_TOPK", 32),
            ),
        )
        proposals: List[Tuple[base.CandidateProposal, Dict[str, Any]]] = []
        for rank, (_, event_id, _) in enumerate(preordered[:max_trials]):
            proposal, extra = _build_heading_proposal(
                v46=v46,
                prev_motion=prev,
                event_id=event_id,
                event_path=str(paths[event_id]),
                slot=dict(slot),
                slot_idx=slot_idx,
                candidate_rank=rank,
                target_len=target_len,
                cfg=cfg,
                db=db,
                stage_heading_rad=stage_heading,
                recent_turn_count=recent_turn_count,
            )
            proposals.append((proposal, extra))

        safe = [row for row in proposals if row[0].safe]
        if safe:
            selected_prop, selected_extra = min(
                safe, key=lambda row: float(row[0].risk_score)
            )
            selected_prop.decision = (
                "accepted_heading_physics_safe"
                if selected_prop.rank == 0
                else "reselected_heading_physics_safe"
            )
        else:
            non_hard = [
                row
                for row in proposals
                if not bool(row[1]["heading_detail"].get("hard_reject", False))
            ]
            if not non_hard:
                raise RuntimeError(
                    f"V46.50 heading contract exhausted candidates for slot {slot_idx}"
                )
            selected_prop, selected_extra = min(
                non_hard, key=lambda row: float(row[0].risk_score)
            )
            selected_prop.decision = "minimum_violation_rescue"

        piece = selected_prop.motion_piece.astype(np.float32)
        transition_span = None
        if selected_prop.transition_span_local is not None:
            transition_span = [
                int(cursor + selected_prop.transition_span_local[0]),
                int(cursor + selected_prop.transition_span_local[1]),
            ]
        core_span = [
            int(cursor + selected_prop.core_span_local[0]),
            int(cursor + selected_prop.core_span_local[1]),
        ]

        event_meta = selected_extra["event_meta"]
        event_delta = float(
            event_meta.get(
                "event_stage_delta_yaw_rad",
                _event_delta(db, selected_prop.event_id),
            )
        )
        stage_before = float(stage_heading)
        stage_after = float(wrap_angle(stage_before + event_delta))
        cumulative_abs_yaw += abs(event_delta)
        intent = str(event_meta.get("event_turn_intent", "none"))
        if intent in {"turn", "explicit_spin", "uncertain_turn"}:
            recent_turn_count += 1
        else:
            recent_turn_count = 0

        pieces.append(piece)
        selected.append([int(selected_prop.event_id), int(selected_prop.rank)])
        row = {
            "slot": int(slot_idx),
            "event_id": int(selected_prop.event_id),
            "candidate_rank": int(selected_prop.rank),
            "event_path": selected_prop.event_path,
            "target_frames": int(target_len),
            "piece_frames": int(piece.shape[0]),
            "transition_span": transition_span,
            "transition_spans": [transition_span] if transition_span else [],
            "core_span": core_span,
            "transition_in_frames": int(len(selected_prop.bridge)),
            "core_frames": int(len(selected_prop.core)),
            "core_warp": float(
                len(selected_prop.core)
                / max(
                    1,
                    base.load_event_motion(
                        v46,
                        selected_prop.event_path,
                        cfg,
                        "v46_50_warp_probe",
                    ).shape[0],
                )
            ),
            "risk_predicted": selected_prop.risk,
            "risk_score_predicted": float(selected_extra["physical_risk_score"]),
            "heading_penalty": float(selected_extra["heading_penalty"]),
            "combined_candidate_score": float(selected_prop.risk_score),
            "safe_predicted": bool(selected_prop.safe),
            "decision": selected_prop.decision,
            "length_policy": selected_prop.length_info,
            "contract_after_align": selected_prop.align_report,
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
            "candidate_trials": [
                {
                    "event_id": int(pp.event_id),
                    "rank": int(pp.rank),
                    "safe": bool(pp.safe),
                    "combined_score": float(pp.risk_score),
                    "physical_risk_score": float(ex["physical_risk_score"]),
                    "heading_penalty": float(ex["heading_penalty"]),
                    "hard_reject": bool(
                        ex["heading_detail"].get("hard_reject", False)
                    ),
                    "event_intent": str(
                        ex["event_meta"].get("event_turn_intent", "none")
                    ),
                    "decision": pp.decision,
                }
                for pp, ex in proposals
            ],
            "version": "v46_50_event_heading_closed_loop_reference",
        }
        report.append(row)
        state_trace.append({
            "slot": int(slot_idx),
            "event_id": int(selected_prop.event_id),
            "intent": intent,
            "stage_heading_before_rad": stage_before,
            "event_delta_rad": event_delta,
            "stage_heading_after_rad": stage_after,
            "cumulative_abs_yaw_rad": float(cumulative_abs_yaw),
        })
        stage_heading = stage_after
        cursor += int(piece.shape[0])

    final = (
        np.concatenate(pieces, axis=0).astype(np.float32)
        if pieces
        else np.zeros((0, EDGE_DIM), dtype=np.float32)
    )
    final = base.enforce_contract(
        v46,
        final,
        cfg,
        source_hint="v46_50_event_heading_reference_final",
    )
    _LAST_HEADING_PLAN = {
        "schema": "v46_50_stage_heading_state",
        "initial_stage_heading_rad": float(initial_heading),
        "final_stage_heading_rad": float(stage_heading),
        "cumulative_abs_event_yaw_rad": float(cumulative_abs_yaw),
        "cumulative_abs_event_yaw_deg": float(np.degrees(cumulative_abs_yaw)),
        "state_trace": state_trace,
    }
    return final, report, selected


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
    return motion.astype(np.float32), stage


def _patch_final_report(args: Any) -> None:
    path = Path(
        args.json
        or str(args.out).replace(
            ".npy",
            ".v46_46_closed_loop_report.json",
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
            fps=30.0,
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
