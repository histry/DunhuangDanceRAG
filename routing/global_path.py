#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.53 whole-song closed loop.

This module keeps the public V46.52 Fresh-WAV/Heading/Anatomy transaction and
adds the strongest low-resource-safe parts of the research reconstruction:

- dual-branch semantic + intrinsic-geometry candidate grounding;
- entropy-regularised global path pre-ordering (Schroedinger-inspired discrete
  path prior, not claimed as a continuous Schrödinger Bridge solver);
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

SCHEMA = "v46_53_geometry_probabilistic_eventrag_closed_loop"
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
                ckpt = str(Path(out_root) / "v46_53_dual_branch_grounder.pt")
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
    omega = _vector_gap(db, "v46_53_exit_omega", "v46_53_entry_omega", a, b)
    alpha = _vector_gap(db, "v46_53_exit_alpha", "v46_53_entry_alpha", a, b)
    posture = _posture_gap(db, a, b)
    pelvis = abs(
        float(_db_value(db, "pelvis_height_exit_norm", a, 0.8))
        - float(_db_value(db, "pelvis_height_entry_norm", b, 0.8))
    )
    contact = 0.0
    try:
        contact = float(np.mean(np.abs(
            np.asarray(db["contact_exit"], np.float32)[a]
            - np.asarray(db["contact_entry"], np.float32)[b]
        )))
    except Exception:
        pass
    return float(
        _env_float("V46_53_GLOBAL_OMEGA_W", 0.10) * omega
        + _env_float("V46_53_GLOBAL_ALPHA_W", 0.002) * alpha
        + _env_float("V46_53_GLOBAL_POSTURE_W", 0.35) * posture
        + _env_float("V46_53_GLOBAL_PELVIS_W", 1.8) * pelvis
        + _env_float("V46_53_GLOBAL_CONTACT_W", 0.45) * contact
    )


def _global_route_preorder(
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
        merged = tangent_masked_merge(motion_ref, proposal_motion, masks)
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
    report["v46_53_env"] = {k: v for k, v in os.environ.items() if k.startswith("V46_53_")}
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
