#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Formal CTSR boundary-safe whole-song generator for EDGE151 motion.

Research objective
------------------
Upgrade the existing patch-stack style two-layer safety idea into a closed-loop
boundary safety scheduler:

    search-time candidate ranking
        -> lightweight simulated stitching risk
        -> risk-adaptive transition budget
        -> real stitching
        -> Contact and Boundary Transition-style cross-boundary risk audit
        -> candidate reselection / repair / rollback
        -> unified boundary-level audit table

Key properties
--------------
1. Search risk is no longer only metadata-level.  For top-k candidates, the
   scheduler quickly simulates yaw/XZ alignment + root-Hermite / rotation-SLERP
   transition and evaluates a lightweight Contact Transition-style risk.
2. Transition length is adapted by pose/yaw/contact/FK risk, not only target
   duration.
3. Unsafe boundaries can trigger local candidate reselection before relying on
   refiner/diffusion/IK.
4. All predicted and actual boundary signals are exported as JSON and CSV for
   paper tables.

Formal CTSR example:
    python routing/boundary_closed_loop.py generate \
        --config configs/motion_model.json \
        --audio dunhuangwu2.wav \
        --db output/.../db \
        --refiner output/.../boundary_refiner.pt \
        --diffusion output/.../motion_runtime.pt \
        --out output/.../dunhuangwu2_boundary_closed_loop_closed_loop.npy \
        --json output/.../dunhuangwu2_boundary_closed_loop_closed_loop.report.json
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
# MOTION_ACTIVITY_INTEGRATION_BEGIN
from evaluation.motion_activity_analysis import (
    evaluate_final_motion_activity,
    save_stage_snapshot,
    write_activity_report,
)
# MOTION_ACTIVITY_INTEGRATION_END
from contracts.boundary_continuity import (
    BoundaryContinuityLimits,
    boundary_risk_reasons,
    evaluate_boundary_continuity,
)
from motion_geometry.resampling import resample_edge151_np
from contracts.physical_quality import (
    PhysicalQualityLimits,
    StageAcceptancePolicy,
    build_repair_mask,
    evaluate_physical_audit,
    evaluate_stage_candidate,
    run_stage_transaction,
)
from support.common import make_geodesic_transition
from support.transition_quality import transition_risk as canonical_transition_risk
from support.event_identity import (
    assert_same_event_db_contract,
    event_uids_from_generation_db,
    make_event_db_contract,
    normalize_event_db_contract,
)
from scheduling.schedule_hard_constraints import (
    DEFAULT_MAX_POSE_HOLD_RATIO,
    DEFAULT_MAX_SINGLE_SOURCE_RATIO,
    DEFAULT_MIN_CORE_FRAME_RATIO,
    DEFAULT_MIN_UNIQUE_EVENTS,
    assert_schedule_hard_constraints,
    final_selection_constraint_rows,
)
from evaluation.gar_evaluation_readiness import (
    GAR_READINESS_INTERFACE_SCHEMA,
    behavior_config_fingerprint,
    build_closed_loop_trace,
    canonical_fingerprint,
    checkpoint_bundle_fingerprint,
    current_git_commit,
    write_trace as write_gar_trace,
    make_boundary_id,
    make_evaluation_case_id,
    make_sequence_id,
)


EDGE_DIM = 151
CONTACT = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT6D_START = 7
ROT6D_END = 151

GAR_SELECTION_POLICY_ID = (
    "boundary_closed_loop_first_safe_minimum_risk_post_audit_reselection_v1"
)
GAR_GENERATOR_ID = "edge151_motion_generation_pipeline"
GAR_GENERATOR_VERSION = "edge151_refiner_diffusion_ik_pipeline_v1"
GAR_REPAIR_OPERATOR_ID = "so3_endpoint_velocity_bridge"
GAR_REPAIR_OPERATOR_VERSION = "so3_endpoint_velocity_bridge_v1"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def jsonable(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_everything(motion_runtime: Any, seed: int) -> None:
    """Reset all generator RNGs so candidate comparisons use common randomness."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    if hasattr(motion_runtime, "torch") and motion_runtime.torch is not None:
        motion_runtime.torch.manual_seed(int(seed))
        if motion_runtime.torch.cuda.is_available():
            motion_runtime.torch.cuda.manual_seed_all(int(seed))


def _comma_separated_ints(value: str) -> List[int]:
    result: List[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed not in result:
            result.append(parsed)
    return result


def repairability_runtime_settings(args: argparse.Namespace) -> Dict[str, Any]:
    mode = str(
        getattr(args, "repairability_mode", None)
        or os.environ.get("REPAIRABILITY_MODE", "off")
    ).strip().lower()
    if mode not in {"off", "shadow", "rank"}:
        raise ValueError("REPAIRABILITY_MODE must be off, shadow, or rank")
    checkpoint = str(
        getattr(args, "repairability_checkpoint", None)
        or os.environ.get("REPAIRABILITY_CHECKPOINT", "")
    ).strip()
    if mode != "off" and not checkpoint:
        raise ValueError("repairability shadow/rank mode requires a checkpoint")
    return {
        "mode": mode,
        "checkpoint": checkpoint or None,
        "device": str(os.environ.get("REPAIRABILITY_DEVICE", "cpu")).strip(),
    }


def load_repairability_ranker(settings: Mapping[str, Any]) -> Optional[Any]:
    if settings.get("mode") == "off":
        return None
    from model.repairability_predictor import RepairabilityRanker

    ranker = RepairabilityRanker.load(
        str(settings["checkpoint"]), device=str(settings.get("device", "cpu"))
    )
    if settings.get("mode") == "rank" and not bool(
        ranker.checkpoint_metadata.get("rank_authorized", False)
    ):
        raise RuntimeError(
            "repairability checkpoint is shadow-only because its promotion gate did not pass"
        )
    return ranker


def candidate_router_probability_maps(
    retrieval_report: Sequence[Mapping[str, Any]],
) -> List[Dict[int, float]]:
    result: List[Dict[int, float]] = []
    for row in retrieval_report:
        indices = list(row.get("candidate_event_indices", ()))
        probabilities = list(row.get("candidate_router_probabilities", ()))
        if len(indices) != len(probabilities):
            raise RuntimeError("candidate Router probability alignment mismatch")
        result.append(
            {int(index): float(probability) for index, probability in zip(indices, probabilities)}
        )
    return result


def import_motion_runtime():
    return importlib.import_module("training.motion_models")


def _as_motion_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] < EDGE_DIM:
        raise ValueError(f"Expected EDGE motion [T,151], got {arr.shape}")
    return arr[:, :EDGE_DIM].astype(np.float32)


def enforce_contract(motion_runtime, motion: np.ndarray, cfg: Any, source_hint: str) -> np.ndarray:
    x = _as_motion_array(motion)
    if hasattr(motion_runtime, "enforce_edge151_contract_np"):
        y, _ = motion_runtime.enforce_edge151_contract_np(
            x, cfg, source_hint=source_hint, derive_contact=True, project_rot=True
        )
        return _as_motion_array(y)
    return x.astype(np.float32)


def resample_motion(motion_runtime, motion: np.ndarray, target_len: int) -> np.ndarray:
    target_len = max(1, int(target_len))
    x = _as_motion_array(motion)
    if x.shape[0] == target_len:
        return x.copy().astype(np.float32)
    return _as_motion_array(resample_edge151_np(x, target_frames=target_len))


def load_event_motion(motion_runtime, path: str | Path, cfg: Any, source_hint: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    obj = np.load(str(p), allow_pickle=True)
    motion = _as_motion_array(obj)
    return enforce_contract(motion_runtime, motion, cfg, source_hint=source_hint)


def angle_diff(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def root_yaw(motion_runtime, motion: np.ndarray) -> np.ndarray:
    x = _as_motion_array(motion)
    if hasattr(motion_runtime, "root_yaw_np"):
        try:
            return np.asarray(motion_runtime.root_yaw_np(x), dtype=np.float32).reshape(-1)
        except Exception:
            pass
    # Fallback: derive a rough facing/yaw from root XZ velocity.
    root = x[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    v = np.zeros_like(root)
    if len(root) > 1:
        v[1:] = root[1:] - root[:-1]
        v[0] = v[1]
    yaw = np.arctan2(v[:, 0], v[:, 1] + 1e-8).astype(np.float32)
    return yaw


def fk_positions(motion_runtime, motion: np.ndarray) -> Optional[np.ndarray]:
    x = _as_motion_array(motion)
    for name in ("fk_24_np", "motion_to_joint_positions_np"):
        fn = getattr(motion_runtime, name, None)
        if fn is not None:
            try:
                return np.asarray(fn(x), dtype=np.float32)
            except Exception:
                pass
    return None


def transition_risk(motion_runtime, previous: np.ndarray, transition: np.ndarray, following: np.ndarray, fps: float) -> Dict[str, Any]:
    """Evaluate one seam through the canonical transition-quality contract."""
    del motion_runtime
    previous = np.asarray(previous, dtype=np.float32)
    transition = np.asarray(transition, dtype=np.float32)
    following = np.asarray(following, dtype=np.float32)
    risk = dict(
        canonical_transition_risk(
            previous,
            transition,
            following,
            fps=float(fps),
        )
    )
    violations = boundary_risk_reasons(risk)
    contract_failures = [
        reason
        for reason in violations
        if reason.startswith("missing_or_nonfinite")
        or reason.startswith("missing_or_invalid")
    ]
    if contract_failures:
        # Search-time ranking must not treat missing/non-finite safety metrics
        # as zeros.  The detailed reasons remain available to the final gate.
        risk["total"] = max(float(risk.get("total", 0.0)), 1.0e9)
        risk["contract_violations"] = contract_failures
    return risk


def risk_score(risk: Dict[str, Any]) -> float:
    # Normalized scalar used for search and fallback selection.  Every
    # kinematic threshold is named with its SI unit so 30/60 FPS runs share
    # exactly the same physical contract.
    bj = float(risk.get("boundary_joint_jerk_max", risk.get("joint_jerk", 0.0))) / max(env_float("BOUNDARY_NORM_BOUNDARY_JERK_MPS3", 5000.0), 1e-6)
    fk = max(
        float(risk.get("entry_fk_jump", 0.0)),
        float(risk.get("exit_fk_jump", 0.0)),
        float(risk.get("entry_fk_jump_max_m", 0.0)),
        float(risk.get("exit_fk_jump_max_m", 0.0)),
    ) / max(env_float("BOUNDARY_NORM_EXIT_FK_JUMP_M", 0.040), 1e-6)
    rot = max(
        float(risk.get("entry_rotation_step_rad", 0.0)),
        float(risk.get("exit_rotation_step_rad", 0.0)),
    ) / max(env_float("BOUNDARY_NORM_EXIT_ROT_RAD", 0.12), 1e-6)
    slip = max(
        float(risk.get("foot_slip", 0.0)),
        float(risk.get("foot_slip_p95", 0.0)),
        float(risk.get("foot_slip_max", 0.0)),
    ) / max(env_float("BOUNDARY_NORM_FOOT_SLIP_MPS", 0.22), 1e-6)
    cs = float(risk.get("contact_switch", 0.0)) / max(env_float("BOUNDARY_NORM_CONTACT_SWITCH", 0.45), 1e-6)
    total = float(risk.get("total", 0.0)) / max(env_float("BOUNDARY_NORM_TOTAL", 1.0), 1e-6)
    return float(0.30 * total + 0.24 * bj + 0.22 * fk + 0.14 * rot + 0.07 * slip + 0.03 * cs)


def risk_safe(risk: Dict[str, Any]) -> bool:
    return not boundary_risk_reasons(risk)


def estimate_boundary_features(motion_runtime, prev: np.ndarray, curr: np.ndarray, cfg: Any) -> Dict[str, float]:
    p = _as_motion_array(prev)
    c = _as_motion_array(curr)
    pose_gap = float(np.linalg.norm(p[-1, ROT6D_START:ROT6D_END] - c[0, ROT6D_START:ROT6D_END]) / math.sqrt(max(1, ROT6D_END - ROT6D_START)))
    if len(p) > 1 and len(c) > 1:
        pv = p[-1, ROT6D_START:ROT6D_END] - p[-2, ROT6D_START:ROT6D_END]
        cv = c[1, ROT6D_START:ROT6D_END] - c[0, ROT6D_START:ROT6D_END]
        velocity_gap = float(np.linalg.norm(pv - cv) / math.sqrt(max(1, ROT6D_END - ROT6D_START)))
    else:
        velocity_gap = 0.0
    yaw_prev = float(root_yaw(motion_runtime, p[-1:])[0])
    yaw_curr = float(root_yaw(motion_runtime, c[:1])[0])
    yaw_gap = abs(angle_diff(yaw_prev, yaw_curr))
    contact_gap = float(np.abs(p[-1, CONTACT] - c[0, CONTACT]).mean())
    fk_gap = 0.0
    fp = fk_positions(motion_runtime, p[-1:])
    fc = fk_positions(motion_runtime, c[:1])
    if fp is not None and fc is not None:
        try:
            fk_gap = float(np.sqrt(np.mean((fp[0] - fc[0]) ** 2)))
        except Exception:
            fk_gap = 0.0
    root_prev = p[-min(len(p), 4):, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_curr = c[:min(len(c), 4), [ROOT_X_IDX, ROOT_Z_IDX]]
    root_direction_gap = 0.0
    if len(root_prev) > 1 and len(root_curr) > 1:
        vp = root_prev[-1] - root_prev[0]
        vc = root_curr[-1] - root_curr[0]
        denom = float(np.linalg.norm(vp) * np.linalg.norm(vc))
        if denom > 1e-8:
            root_direction_gap = float(1.0 - np.dot(vp, vc) / denom)
    return {
        "pose_gap": pose_gap,
        "velocity_gap": velocity_gap,
        "yaw_gap_rad": float(yaw_gap),
        "contact_gap": contact_gap,
        "predicted_fk_gap": fk_gap,
        "root_direction_gap": root_direction_gap,
    }


def choose_transition_lengths(motion_runtime, prev: Optional[np.ndarray], source_len: int, target_len: int, raw_curr: np.ndarray, slot: Dict[str, Any], cfg: Any) -> Tuple[int, int, Dict[str, Any]]:
    target_len = max(1, int(target_len))
    source_len = max(1, int(source_len))
    has_prev = prev is not None and len(prev) > 0
    supported_max = int(round(float(getattr(cfg,"transition_train_max_seconds",28/30))*float(getattr(cfg,"fps",30))))
    configured_max = env_int("MOTION_TRANSITION_MAX_FRAMES",supported_max)
    if configured_max > supported_max:
        raise ValueError("generation bridge exceeds the trained seam length; change and revalidate the protocol first")
    if hasattr(motion_runtime, "_choose_core_and_transition_lengths"):
        core_len, trans_len, info = motion_runtime._choose_core_and_transition_lengths(source_len, target_len, has_prev, cfg)
        info = dict(info)
    else:
        if not has_prev:
            return target_len, 0, {"reason": "first_slot_no_transition"}
        min_trans = env_int("MOTION_TRANSITION_MIN_FRAMES", 10)
        max_trans = configured_max
        trans_len = int(round(target_len * env_float("MOTION_TRANSITION_RATIO", 0.18)))
        trans_len = max(min_trans, min(max_trans, trans_len))
        core_len = target_len - trans_len
        info = {"reason": "local_default", "transition_frames": trans_len, "core_frames": core_len}

    if not has_prev or not env_bool("BOUNDARY_RISK_ADAPT_TRANSITION_ENABLE", True):
        core_len = max(1, min(int(core_len), target_len))
        trans_len = max(0, target_len - core_len)
        info.update({"risk_adaptive": False})
        return int(core_len), int(trans_len), info

    # Estimate boundary features after a rough core resample but before final transition.
    rough_core = resample_motion(motion_runtime, raw_curr, max(1, int(core_len)))
    rough_core = enforce_contract(motion_runtime, rough_core, cfg, source_hint="boundary_closed_loop_transition_len_rough_core")
    # Align for a more realistic yaw/root measurement.
    aligned, align_info = align_core_to_prev(motion_runtime, prev, rough_core, cfg, transition_frames=trans_len)
    feats = estimate_boundary_features(motion_runtime, prev, aligned, cfg)

    extra = 0.0
    extra += env_float("BOUNDARY_TLEN_POSE_W", 10.0) * feats["pose_gap"]
    extra += env_float("BOUNDARY_TLEN_VEL_W", 4.0) * feats["velocity_gap"]
    extra += env_float("BOUNDARY_TLEN_YAW_W", 3.0) * min(feats["yaw_gap_rad"], math.pi)
    extra += env_float("BOUNDARY_TLEN_CONTACT_W", 8.0) * feats["contact_gap"]
    extra += env_float("BOUNDARY_TLEN_FK_W", 80.0) * feats["predicted_fk_gap"]

    if str(slot.get("router_architecture", "")) == "ctsr_weak_temporal_v1":
        # Use the observable structure event directly.  MSSD's historical
        # compatibility alias (for example build_up -> turning_climax) must
        # not inject a body-action interpretation into transition timing.
        label = str(slot.get("music_event", "neutral_flow")).lower()
    else:
        label = str(slot.get("music_alignment_label", slot.get("music_semantic_top_label", slot.get("role", "")))).lower()
    if any(k in label for k in ["calm", "lyrical", "pose", "release", "resolution"]):
        extra += env_float("BOUNDARY_TLEN_SMOOTH_MUSIC_BONUS", 3.0)
    if any(k in label for k in ["accent", "percussive", "climax"]):
        extra -= env_float("BOUNDARY_TLEN_ACCENT_REDUCE", 2.0)

    min_trans = env_int("MOTION_TRANSITION_MIN_FRAMES", 10)
    max_trans = configured_max
    min_core = env_int("MOTION_TRANSITION_MIN_CORE_FRAMES", 30)
    trans_len2 = int(round(float(trans_len) + np.clip(extra, -4.0, env_float("BOUNDARY_TLEN_EXTRA_MAX", 14.0))))
    trans_len2 = max(min_trans, min(max_trans, trans_len2))
    trans_len2 = min(trans_len2, max(0, target_len - min_core))
    core_len2 = max(1, target_len - trans_len2)
    info.update({
        "risk_adaptive": True,
        "base_transition_frames": int(trans_len),
        "risk_transition_extra": float(extra),
        "risk_transition_frames": int(trans_len2),
        "risk_core_frames": int(core_len2),
        "boundary_features": feats,
        "rough_align": align_info,
    })
    return int(core_len2), int(trans_len2), info


def align_core_to_prev(motion_runtime, prev: np.ndarray, core: np.ndarray, cfg: Any,
                       *, transition_frames: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = _as_motion_array(prev)
    c = _as_motion_array(core)
    formal = getattr(motion_runtime, "_align_core_to_previous", None)
    if formal is not None:
        out, rep = formal(p, c, cfg, transition_frames=transition_frames)
        return enforce_contract(motion_runtime, out, cfg, source_hint="boundary_closed_loop_align:duration"), dict(rep)
    for name in ("align_event_core_to_prev_np",):
        fn = getattr(motion_runtime, name, None)
        if fn is not None:
            try:
                out, rep = fn(p, c, cfg)
                return enforce_contract(motion_runtime, out, cfg, source_hint=f"boundary_closed_loop_align:{name}"), dict(rep or {})
            except Exception:
                pass
    out = c.copy().astype(np.float32)
    delta = p[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += float(delta[0])
    out[:, ROOT_Z_IDX] += float(delta[1])
    out = enforce_contract(motion_runtime, out, cfg, source_hint="boundary_closed_loop_align:fallback_xz")
    return out, {"mode": "fallback_xz_only", "delta_xz_applied": [float(delta[0]), float(delta[1])]}


def build_bridge(motion_runtime, prev: np.ndarray, core: np.ndarray, trans_len: int, cfg: Any,
                 *, report: Optional[Dict[str, Any]] = None) -> np.ndarray:
    trans_len = int(trans_len)
    if trans_len <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    prev_tail_n = min(max(2, trans_len // 2), len(prev))
    curr_head_n = min(max(2, trans_len // 2), len(core))
    formal = getattr(motion_runtime, "reference_motion_inbetween_np", None)
    if formal is not None:
        # An error in the formal algorithm must never silently select legacy SLERP.
        bridge = formal(prev[-prev_tail_n:], core[:curr_head_n], trans_len, cfg, report=report)
        return enforce_contract(motion_runtime, bridge, cfg, source_hint="boundary_closed_loop_bridge:duration_c2")
    for name in ("motion_inbetween_np",):
        fn = getattr(motion_runtime, name, None)
        if fn is not None:
            try:
                bridge = fn(prev[-prev_tail_n:], core[:curr_head_n], trans_len, cfg)
                return enforce_contract(motion_runtime, bridge, cfg, source_hint=f"boundary_closed_loop_bridge:{name}")
            except Exception:
                pass
    # The final fallback still has to honor the same geometry contract as the
    # primary bridge: Euclidean root translation, discrete contact, and SO(3)
    # interpolation for every joint.  Projecting a linearly blended Rot6D
    # vector is not equivalent, especially close to pi.
    bridge = make_geodesic_transition(prev, core, trans_len)
    return enforce_contract(motion_runtime, bridge, cfg, source_hint="boundary_closed_loop_bridge:fallback_geodesic")


@dataclass
class CandidateProposal:
    slot: int
    event_id: int
    rank: int
    event_path: str
    motion_piece: np.ndarray
    bridge: np.ndarray
    core: np.ndarray
    transition_span_local: Optional[List[int]]
    core_span_local: List[int]
    risk: Dict[str, Any]
    risk_score: float
    safe: bool
    length_info: Dict[str, Any]
    align_report: Dict[str, Any]
    decision: str
    diagnostic_arrays: Optional[Dict[str, np.ndarray]] = None


def build_candidate_proposal(
    motion_runtime,
    prev_motion: Optional[np.ndarray],
    event_id: int,
    event_path: str,
    slot: Dict[str, Any],
    slot_idx: int,
    candidate_rank: int,
    target_len: int,
    cfg: Any,
) -> CandidateProposal:
    raw = load_event_motion(motion_runtime, event_path, cfg, source_hint=f"boundary_closed_loop_load_event:{event_id}")
    trace = None
    if env_bool("BOUNDARY_STAGE_DIAGNOSTICS", False):
        from training.generation_stage_diagnostics import selected_diagnostic_slots
        if slot_idx in selected_diagnostic_slots():
            trace = {
                "source_file_motion": _as_motion_array(np.load(event_path, allow_pickle=False)),
                "loaded_event": raw.copy(),
            }
    has_prev = prev_motion is not None and len(prev_motion) > 0
    core_len, trans_len, length_info = choose_transition_lengths(motion_runtime, prev_motion, raw.shape[0], target_len, raw, slot, cfg)
    length_info = dict(length_info)
    length_info.setdefault("source_frames", int(raw.shape[0]))
    core = resample_motion(motion_runtime, raw, core_len)
    core = enforce_contract(motion_runtime, core, cfg, source_hint=f"boundary_closed_loop_core_resample:{event_id}")
    if trace is not None:
        trace["resampled_core"] = core.copy()
    align_report: Dict[str, Any] = {"mode": "none"}
    bridge = np.zeros((0, EDGE_DIM), dtype=np.float32)
    if has_prev:
        core, align_report = align_core_to_prev(motion_runtime, prev_motion, core, cfg, transition_frames=trans_len)
        if trace is not None:
            trace["aligned_core"] = core.copy()
        bridge_report: Dict[str, Any] = {}
        bridge = build_bridge(motion_runtime, prev_motion, core, trans_len, cfg, report=bridge_report)
        align_report["bridge"] = bridge_report
        risk = transition_risk(motion_runtime, prev_motion[-4:], bridge, core[:4], fps=float(getattr(cfg, "fps", 30.0)))
    else:
        risk = {"total": 0.0, "boundary_joint_jerk_max": 0.0, "exit_fk_jump": 0.0, "exit_rotation_step_rad": 0.0, "foot_slip": 0.0, "foot_penetration": 0.0, "contact_switch": 0.0}
    piece = np.concatenate([bridge, core], axis=0).astype(np.float32)
    # Guarantee exact slot length; this should almost always be a no-op.
    if piece.shape[0] != int(target_len):
        piece = resample_motion(motion_runtime, piece, int(target_len))
        piece = enforce_contract(motion_runtime, piece, cfg, source_hint=f"boundary_closed_loop_slot_exact_len:{event_id}")
        # If exact-length repair changed the bridge/core split, keep the recorded split but mark it.
        length_info["slot_exact_repair_applied"] = True
        length_info["slot_exact_frames_after"] = int(piece.shape[0])
    score = risk_score(risk)
    if trace is not None:
        trace["bridge"] = bridge.copy()
        trace["assembled_piece"] = piece.copy()
    safe = risk_safe(risk) if has_prev else True
    return CandidateProposal(
        slot=slot_idx,
        event_id=int(event_id),
        rank=int(candidate_rank),
        event_path=str(event_path),
        motion_piece=piece,
        bridge=bridge.astype(np.float32),
        core=core.astype(np.float32),
        transition_span_local=[0, int(bridge.shape[0])] if has_prev and bridge.shape[0] > 0 else None,
        core_span_local=[int(bridge.shape[0]), int(bridge.shape[0] + core.shape[0])],
        risk=risk,
        risk_score=float(score),
        safe=bool(safe),
        length_info=length_info,
        align_report=align_report,
        decision="candidate",
        diagnostic_arrays=trace,
    )


def slot_target_frames(slot: Dict[str, Any], cfg: Any) -> int:
    if slot.get("target_frames") is not None:
        try:
            return max(1, int(slot["target_frames"]))
        except Exception:
            pass
    dur = float(slot.get("duration", slot.get("duration_sec", 1.0)))
    return max(int(getattr(cfg, "min_event_frames", 1)), int(round(dur * float(getattr(cfg, "fps", 30.0)))))


def extract_candidate_lists(path_idx: Sequence[int], retrieval_report: Sequence[Dict[str, Any]], db: Dict[str, Any], cfg: Any) -> List[List[int]]:
    n = len(np.asarray(db["paths"], dtype=object))
    topk = max(1, env_int("BOUNDARY_RESELECT_TOPK", env_int("BOUNDARY_CANDIDATE_TOPK", 32)))
    out: List[List[int]] = []
    for i, sel in enumerate(path_idx):
        ids: List[int] = []
        if 0 <= int(sel) < n:
            ids.append(int(sel))
        if i < len(retrieval_report):
            for row in retrieval_report[i].get("candidate_preview", []) or []:
                try:
                    eid = int(row.get("event_id"))
                except Exception:
                    continue
                if 0 <= eid < n and eid not in ids:
                    ids.append(eid)
        out.append(ids[:topk] if ids else [int(sel)])
    return out


def assemble_closed_loop_reference(
    motion_runtime,
    slots: Sequence[Dict[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Dict[str, Any],
    cfg: Any,
    banned: Optional[Dict[int, set]] = None,
    diagnostic_dir: Optional[Path] = None,
    candidate_router_probabilities: Optional[Sequence[Mapping[int, float]]] = None,
    repairability_ranker: Optional[Any] = None,
    repairability_mode: str = "off",
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[List[int]]]:
    paths = np.asarray(db["paths"], dtype=object)
    banned = banned or {}
    pieces: List[np.ndarray] = []
    report: List[Dict[str, Any]] = []
    selected: List[List[int]] = []
    cursor = 0
    if repairability_mode not in {"off", "shadow", "rank"}:
        raise ValueError("repairability_mode must be off, shadow, or rank")
    if repairability_mode != "off" and repairability_ranker is None:
        raise ValueError("repairability mode requires a loaded ranker")
    for slot_idx, slot in enumerate(slots):
        target_len = slot_target_frames(slot, cfg)
        prev = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else None
        original_rank_by_event = {
            int(event_id): int(rank)
            for rank, event_id in enumerate(candidate_lists[slot_idx])
        }
        candidates = [int(x) for x in candidate_lists[slot_idx] if int(x) not in banned.get(slot_idx, set())]
        if not candidates:
            candidates = [int(candidate_lists[slot_idx][0])]
        proposals: List[CandidateProposal] = []
        best: Optional[CandidateProposal] = None
        selected_prop: Optional[CandidateProposal] = None
        evaluate_complete_pool = (
            slot_idx >= 1 and repairability_mode in {"shadow", "rank"}
        )
        for evaluation_rank, event_id in enumerate(candidates):
            original_rank = (
                original_rank_by_event.get(event_id, evaluation_rank)
                if repairability_mode != "off"
                else evaluation_rank
            )
            p = build_candidate_proposal(
                motion_runtime=motion_runtime,
                prev_motion=prev,
                event_id=event_id,
                event_path=str(paths[event_id]),
                slot=slot,
                slot_idx=slot_idx,
                candidate_rank=original_rank,
                target_len=target_len,
                cfg=cfg,
            )
            proposals.append(p)
            if best is None or p.risk_score < best.risk_score:
                best = p
            if p.safe:
                if selected_prop is None:
                    selected_prop = p
                if not evaluate_complete_pool:
                    break
        repairability_by_event: Dict[int, Dict[str, Any]] = {}
        repairability_abstained = False
        if repairability_ranker is not None and slot_idx >= 1:
            from model.repairability_predictor import proposal_feature_mapping

            probability_map = (
                dict(candidate_router_probabilities[slot_idx])
                if candidate_router_probabilities is not None
                and slot_idx < len(candidate_router_probabilities)
                else {}
            )
            eligible_predictions = []
            eligible_proposals = []
            for proposal in proposals:
                features = proposal_feature_mapping(
                    router_probability=float(probability_map.get(proposal.event_id, 0.0)),
                    original_rank=int(proposal.rank),
                    pool_size=len(candidate_lists[slot_idx]),
                    target_frames=target_len,
                    transition_frames=len(proposal.bridge),
                    core_frames=len(proposal.core),
                    core_warp=float(
                        len(proposal.core)
                        / max(
                            1,
                            int(
                                proposal.length_info.get(
                                    "raw_core_frames",
                                    proposal.length_info.get(
                                        "source_frames", len(proposal.core)
                                    ),
                                )
                            ),
                        )
                    ),
                    pre_safe=proposal.safe,
                    pre_risk_score=proposal.risk_score,
                    risk=proposal.risk,
                )
                prediction = repairability_ranker.predict(features)
                repairability_by_event[proposal.event_id] = {
                    "features": features,
                    "prediction": prediction.as_dict(),
                }
                if proposal.safe:
                    eligible_proposals.append(proposal)
                    eligible_predictions.append(prediction)
            repairability_abstained = repairability_ranker.should_abstain(
                eligible_predictions
            )
            if (
                repairability_mode == "rank"
                and eligible_proposals
                and not repairability_abstained
            ):
                selected_prop = max(
                    eligible_proposals,
                    key=lambda proposal: repairability_by_event[proposal.event_id][
                        "prediction"
                    ]["utility"],
                )
        if selected_prop is None:
            selected_prop = best
            if selected_prop is None:
                raise RuntimeError(f"No proposal for slot {slot_idx}")
            selected_prop.decision = "accepted_best_unsafe_fallback"
        elif (
            repairability_mode == "rank"
            and selected_prop.event_id in repairability_by_event
            and not repairability_abstained
        ):
            selected_prop.decision = "accepted_repairability_ranked_pre_safe"
        else:
            selected_prop.decision = (
                "accepted_first_safe"
                if selected_prop.rank == 0
                else "reselected_safe"
            )
        piece = selected_prop.motion_piece.astype(np.float32)
        transition_span = None
        if selected_prop.transition_span_local is not None:
            transition_span = [int(cursor + selected_prop.transition_span_local[0]), int(cursor + selected_prop.transition_span_local[1])]
        core_span = [int(cursor + selected_prop.core_span_local[0]), int(cursor + selected_prop.core_span_local[1])]
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
            "transition_in_frames": int(selected_prop.bridge.shape[0]),
            "core_frames": int(selected_prop.core.shape[0]),
            "core_warp": float(
                selected_prop.core.shape[0]
                / max(1, int(selected_prop.length_info.get("source_frames", selected_prop.core.shape[0])))
            ),
            "risk_predicted": selected_prop.risk,
            "risk_score_predicted": float(selected_prop.risk_score),
            "safe_predicted": bool(selected_prop.safe),
            "decision": selected_prop.decision,
            "repairability": {
                "mode": repairability_mode,
                "abstained": bool(repairability_abstained),
                "selected_prediction": repairability_by_event.get(
                    selected_prop.event_id
                ),
                "guard_is_authoritative": True,
                "pre_safe_gate_is_authoritative": True,
            },
            "conditioning_contract": str(
                slot.get(
                    "closed_loop_conditioning_contract",
                    "legacy_music_slot_feature",
                )
            ),
            "length_policy": selected_prop.length_info,
            "contract_after_align": selected_prop.align_report,
            "candidate_trials": [
                {
                    "event_id": int(pp.event_id),
                    "rank": int(pp.rank),
                    "safe": bool(pp.safe),
                    "risk_score": float(pp.risk_score),
                    "risk": pp.risk,
                    "transition_frames": int(pp.bridge.shape[0]),
                    "decision": pp.decision,
                    "repairability": repairability_by_event.get(pp.event_id),
                }
                for pp in proposals
            ],
            "version": "boundary_closed_loop_boundary_simulated_closed_loop_reference",
        }
        if diagnostic_dir is not None and selected_prop.diagnostic_arrays is not None:
            from training.generation_stage_diagnostics import save_upstream_stages
            row["stage_diagnostics"] = save_upstream_stages(
                diagnostic_dir / f"slot_{slot_idx:03d}",
                selected_prop.diagnostic_arrays, selected_prop.event_path,
                motion_runtime, cfg,
            )
        report.append(row)
        cursor += int(piece.shape[0])
    final = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else np.zeros((0, EDGE_DIM), dtype=np.float32)
    final = enforce_contract(motion_runtime, final, cfg, source_hint="boundary_closed_loop_closed_loop_reference_final")
    return final, report, selected


def transition_spans_from_report(assembly_report: Sequence[Dict[str, Any]]) -> List[List[int]]:
    out: List[List[int]] = []
    for r in assembly_report:
        sp = r.get("transition_span")
        if sp is not None and len(sp) >= 2 and int(sp[1]) > int(sp[0]):
            out.append([int(sp[0]), int(sp[1])])
    return out


def make_seam_mask(motion_runtime, T: int, transition_spans: Sequence[Sequence[int]], cfg: Any) -> Tuple[np.ndarray, List[int], str]:
    def finish(raw_mask: np.ndarray, centers: List[int], policy: str):
        mask = np.asarray(raw_mask, dtype=np.float32).reshape(int(T), -1)
        max_ratio = float(np.clip(env_float("ROUTING_SAFETY_MAX_TRANSITION_MASK_RATIO", 0.25), 0.0, 1.0))
        active = np.flatnonzero(mask[:, 0] > 1e-6)
        budget = int(math.floor(int(T) * max_ratio))
        if len(active) > budget and budget >= 0:
            keep = np.zeros((int(T),), dtype=bool)
            if centers and budget > 0:
                radius = max(0, budget // max(1, 2 * len(centers)))
                for center in centers:
                    keep[max(0, center - radius) : min(int(T), center + radius + 1)] = True
                if int(keep.sum()) > budget:
                    kept = np.flatnonzero(keep)[:budget]
                    keep[:] = False
                    keep[kept] = True
            mask[~keep, :] = 0.0
            policy += "+coverage_cap"
        return mask, centers, policy

    if transition_spans and hasattr(motion_runtime, "make_transition_budget_mask"):
        try:
            mask = motion_runtime.make_transition_budget_mask(T, transition_spans, cfg)
            centers = [int((int(a) + int(b)) // 2) for a, b in transition_spans]
            return finish(mask, centers, "boundary_closed_loop_transition_spans")
        except Exception:
            pass
    centers = [int((int(a) + int(b)) // 2) for a, b in transition_spans]
    if hasattr(motion_runtime, "make_boundary_mask"):
        try:
            mask = motion_runtime.make_boundary_mask(T, centers, width=env_int("BOUNDARY_FALLBACK_MASK_WIDTH", 24))
            return finish(mask, centers, "boundary_closed_loop_fallback_boundary_mask")
        except Exception:
            pass
    mask = np.zeros((int(T), 1), dtype=np.float32)
    width = env_int("BOUNDARY_FALLBACK_MASK_WIDTH", 24)
    for c in centers:
        mask[max(0, c - width):min(T, c + width), 0] = 1.0
    return finish(mask, centers, "boundary_closed_loop_local_fallback_boundary_mask")


def compute_condition(
    motion_runtime: Any,
    slot_feat: np.ndarray,
    assembly_report: Sequence[Dict[str, Any]],
    total_frames: int,
    db: Dict[str, Any],
) -> np.ndarray:
    """Build frame-local music conditioning for one assembled motion stream.

    The closed-loop scheduler may reselect events and therefore change transition
    spans between rounds. Conditioning is rebuilt after each assembly using the
    same authoritative helper as the direct Motion Generation entrypoint.
    """
    if not hasattr(motion_runtime, "build_frame_local_conditioning"):
        raise RuntimeError(
            "Closed-loop Generation requires "
            "training.motion_models.build_frame_local_conditioning"
        )

    features = np.asarray(slot_feat, dtype=np.float32)
    conditioning_contracts = {
        str(row.get("conditioning_contract", "missing_conditioning_contract"))
        for row in assembly_report
    }
    if conditioning_contracts == {"selected_event_motion_descriptor_v1"}:
        descriptors = np.asarray(db.get("desc", []), dtype=np.float32)
        selected = np.asarray(
            [int(row.get("event_id", -1)) for row in assembly_report],
            dtype=np.int64,
        )
        if (
            descriptors.ndim != 2
            or np.any(selected < 0)
            or np.any(selected >= len(descriptors))
        ):
            raise RuntimeError(
                "Formal closed-loop conditioning cannot resolve selected Event-DB descriptors"
            )
        # Music controls phrase timing and CTSR retrieval.  Refiner/diffusion
        # conditioning describes the motion actually selected (including a
        # boundary-safe reselection); it does not fabricate body semantics
        # from an old categorical music label.
        features = descriptors[selected].astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"slot_feat must be [S,C], got {features.shape}")
    if features.shape[0] != len(assembly_report):
        raise RuntimeError(
            "slot feature/assembly report count mismatch: "
            f"{features.shape[0]} features vs {len(assembly_report)} reports"
        )

    try:
        descriptor_mean = np.asarray(db["desc_mean"], dtype=np.float32).reshape(-1)
        descriptor_std = np.asarray(db["desc_std"], dtype=np.float32).reshape(-1)
    except KeyError as exc:
        raise RuntimeError(
            "Closed-loop frame-local conditioning requires desc_mean and desc_std "
            "from the Generation Event-DB"
        ) from exc

    condition = motion_runtime.build_frame_local_conditioning(
        features,
        assembly_report,
        total_frames=int(total_frames),
        descriptor_mean=descriptor_mean,
        descriptor_std=descriptor_std,
    )
    condition = np.asarray(condition, dtype=np.float32)
    expected_shape = (int(total_frames), int(features.shape[1]))
    if condition.shape != expected_shape:
        raise RuntimeError(
            "frame-local conditioning shape mismatch: "
            f"expected {expected_shape}, got {condition.shape}"
        )
    if not np.isfinite(condition).all():
        raise RuntimeError("frame-local conditioning contains non-finite values")
    return condition.astype(np.float32)

def sliding_support_eligibility(
    db: Dict[str, Any],
    assembly_report: Sequence[Dict[str, Any]],
    frames: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Map explicit Chang-E cloud/pivot semantics onto assembled frame spans."""
    mask = np.zeros(int(frames), dtype=bool)
    tokens = {
        token.strip().lower()
        for token in os.environ.get(
            "CHANG_E_SLIDING_SUPPORT_TOKENS",
            "sogdian_whirl,lotus_steps,turning_travel,traveling_steps,"
            "alternating_or_pivot_support,alternating_foot_support",
        ).split(",")
        if token.strip()
    }
    dance = np.asarray(db.get("dance_keys", []), dtype=object)
    locomotion = np.asarray(db.get("locomotion_labels", []), dtype=object)
    support = np.asarray(db.get("support_labels", []), dtype=object)
    eligible_events: list[int] = []
    for row in assembly_report:
        event_id = int(row.get("event_id", -1))
        if event_id < 0:
            continue
        values = []
        for array in (dance, locomotion, support):
            if event_id < len(array):
                values.append(str(array[event_id]).strip().lower())
        eligible = any(
            token in " ".join(values)
            for token in tokens
        )
        if not eligible:
            continue
        eligible_events.append(event_id)
        for key in ("transition_span", "core_span"):
            span = row.get(key)
            if isinstance(span, Sequence) and len(span) == 2:
                start = max(0, int(span[0]))
                end = min(len(mask), int(span[1]))
                mask[start:end] = True
    return mask, {
        "schema": "chang_e_sliding_support_eligibility_v1",
        "eligible_frames": int(mask.sum()),
        "eligible_frame_ratio": float(mask.mean()) if len(mask) else 0.0,
        "eligible_event_ids": sorted(set(eligible_events)),
        "tokens": sorted(tokens),
        "kinematic_confirmation_required": True,
    }


def apply_generators(
    motion_runtime,
    motion_ref: np.ndarray,
    cond: np.ndarray,
    seam_mask: np.ndarray,
    args: argparse.Namespace,
    cfg: Any,
    *,
    sliding_support_eligible: Optional[np.ndarray] = None,
    protected_geometry_mask: Optional[np.ndarray] = None,
    ik_protected_frame_mask: Optional[np.ndarray] = None,
    ik_candidate_guard: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply repair stages with Peak-Jerk support and transactional rollback."""
    if not hasattr(motion_runtime, "audit_motion_np"):
        raise RuntimeError(
            "The generation runtime must provide audit_motion_np for audited stages"
        )

    limits = PhysicalQualityLimits.from_environment()
    policy = StageAcceptancePolicy.from_environment()
    motion = np.asarray(motion_ref, dtype=np.float32).copy()
    protected = None
    if protected_geometry_mask is not None:
        protected = np.asarray(protected_geometry_mask, dtype=bool).reshape(-1)
        if len(protected) != len(motion):
            raise ValueError("protected_geometry_mask length mismatch")
    stage: Dict[str, Any] = {}
    if protected is not None:
        stage["protected_geometry_contract"] = {
            "frames": int(protected.sum()),
            "channels": [4, int(motion.shape[1])],
            "contact_channels_recomputed": True,
        }

    def restore_protected_geometry(value):
        if protected is None or not np.any(protected):
            return value
        restored = np.asarray(value, dtype=np.float32).copy()
        restored[protected, 4:] = motion_ref[protected, 4:]
        return restored

    def capture_candidate(name, value):
        if env_bool("BOUNDARY_STAGE_DIAGNOSTICS", False):
            from training.generation_stage_diagnostics import save_guarded_candidate
            stage[name + "_guarded_candidate"] = save_guarded_candidate(
                Path(args.out).parent, name, value,
            )
        return value

    def audit_fn(value: np.ndarray) -> Dict[str, Any]:
        return dict(
            motion_runtime.audit_motion_np(
                value,
                cfg,
                sliding_support_eligible=sliding_support_eligible,
            )
        )

    stage["pre_refine_audit"] = audit_fn(motion)
    stage["motion_activity_retrieval"] = save_stage_snapshot(
        getattr(args, "out", None),
        "retrieval",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )

    if bool(getattr(cfg, "refiner_enable", False)) and env_bool(
        "BOUNDARY_USE_REFINER", True
    ):
        refiner_mask, mask_report = build_repair_mask(
            motion,
            seam_mask,
            fps=float(getattr(cfg, "fps", 30.0)),
        )
        stage["boundary_refiner_repair_mask"] = mask_report
        composite_model = getattr(args, "refiner_composite_model", None)
        composite_contract = getattr(args, "refiner_composite_contract", None)
        if bool(composite_model) != bool(composite_contract):
            raise RuntimeError(
                "--refiner_composite_model and --refiner_composite_contract "
                "must be provided together"
            )
        pre_refiner = motion.copy()
        if composite_model:
            from routing.refiner_v15_15h_composite import (
                apply_composite_refiner,
            )

            motion, transaction = apply_composite_refiner(
                motion,
                cond,
                refiner_mask,
                composite_model,
                composite_contract,
                cfg,
                audit_fn=audit_fn,
                limits=limits,
                policy=policy,
            )
            motion = capture_candidate("refiner_composite", motion)
        else:
            motion, transaction = run_stage_transaction(
                stage_name="refiner",
                motion=motion,
                apply_fn=lambda value: capture_candidate("refiner", motion_runtime.apply_refiner_model(
                    value,
                    cond,
                    refiner_mask,
                    getattr(args, "refiner", None),
                    cfg,
                    sliding_support_eligible=sliding_support_eligible,
                )),
                audit_fn=audit_fn,
                limits=limits,
                policy=policy,
                require_repair_gain=True,
            )
        motion = restore_protected_geometry(motion)
        if (
            transaction.get("accepted")
            and protected is not None
            and np.any(protected)
        ):
            protected_reaudit = audit_fn(motion)
            protected_decision = evaluate_stage_candidate(
                audit_fn(pre_refiner),
                protected_reaudit,
                limits=limits,
                policy=policy,
                require_repair_gain=True,
            )
            transaction["post_protected_restore_full_transaction_reaudit"] = (
                protected_reaudit
            )
            transaction["post_protected_restore_guard"] = protected_decision
            if not protected_decision["accepted"]:
                motion = pre_refiner
                transaction.update({
                    "accepted": False,
                    "rolled_back": True,
                    "selection": "identity",
                    "reason": "post_protected_restore_full_guard_failed",
                    "selected_audit": audit_fn(pre_refiner),
                })
            else:
                transaction["selected_audit"] = protected_reaudit
        stage["boundary_refiner_transaction"] = transaction
        stage["boundary_refiner_audit"] = audit_fn(motion)
    stage["motion_activity_refiner"] = save_stage_snapshot(
        getattr(args, "out", None),
        "refiner",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )

    if bool(getattr(cfg, "diffusion_enable", False)) and env_bool(
        "BOUNDARY_USE_DIFFUSION", True
    ):
        pre_diffusion = motion.copy()
        diffusion_mask, mask_report = build_repair_mask(
            motion,
            seam_mask,
            fps=float(getattr(cfg, "fps", 30.0)),
        )
        stage["motion_diffusion_repair_mask"] = mask_report
        motion, transaction = run_stage_transaction(
            stage_name="diffusion",
            motion=motion,
            apply_fn=lambda value: capture_candidate("diffusion", motion_runtime.apply_diffusion_model(
                value,
                cond,
                diffusion_mask,
                getattr(args, "diffusion", None),
                cfg,
                sliding_support_eligible=sliding_support_eligible,
            )),
            audit_fn=audit_fn,
            limits=limits,
            policy=policy,
            require_repair_gain=True,
        )
        motion = restore_protected_geometry(motion)
        if (
            transaction.get("accepted")
            and protected is not None
            and np.any(protected)
        ):
            protected_reaudit = audit_fn(motion)
            protected_decision = evaluate_stage_candidate(
                audit_fn(pre_diffusion),
                protected_reaudit,
                limits=limits,
                policy=policy,
                require_repair_gain=True,
            )
            transaction["post_protected_restore_full_transaction_reaudit"] = (
                protected_reaudit
            )
            transaction["post_protected_restore_guard"] = protected_decision
            if not protected_decision["accepted"]:
                motion = pre_diffusion
                transaction.update({
                    "accepted": False,
                    "rolled_back": True,
                    "selection": "identity",
                    "reason": "post_protected_restore_full_guard_failed",
                    "selected_audit": audit_fn(pre_diffusion),
                })
            else:
                transaction["selected_audit"] = protected_reaudit
        stage["motion_diffusion_transaction"] = transaction
        stage["motion_diffusion_audit"] = audit_fn(motion)
    stage["motion_activity_diffusion"] = save_stage_snapshot(
        getattr(args, "out", None),
        "diffusion",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )

    # Long-horizon root drift is a full-sequence metric and cannot be repaired
    # reliably by a short ownership window whose local median recentres the
    # trajectory.  Generate one deterministic, low-frequency candidate by
    # scaling only root XZ displacement about the first-frame anchor.  This
    # preserves the path shape and introduces no new temporal frequencies.
    # The normal full physical transaction audit remains authoritative; any
    # foot/contact/jerk/rotation regression atomically restores the snapshot.
    pre_root_drift = motion.copy()
    root_drift_before = audit_fn(pre_root_drift)
    root_limits = {
        "root_horizontal_radius_p95_m": float(
            limits.root_horizontal_radius_p95_m
        ),
        "root_horizontal_radius_max_m": float(
            limits.root_horizontal_radius_max_m
        ),
        "root_horizontal_net_displacement_m": float(
            limits.root_horizontal_net_displacement_m
        ),
        "root_horizontal_drift_speed_mps": float(
            limits.root_horizontal_drift_speed_mps
        ),
        "root_horizontal_window_displacement_max_m": float(
            limits.root_horizontal_window_displacement_max_m
        ),
    }
    required_scales = []
    for key, maximum in root_limits.items():
        value = float(root_drift_before.get(key, float("nan")))
        if np.isfinite(value) and value > maximum and value > 0.0:
            required_scales.append(float(maximum / value))
    root_scale = (
        max(0.0, min(1.0, 0.98 * min(required_scales)))
        if required_scales
        else 1.0
    )
    root_stage_enabled = env_bool(
        "BOUNDARY_ROOT_DRIFT_RESTORATION", True
    )
    if root_stage_enabled and root_scale < 1.0 - 1.0e-8:
        def root_drift_candidate(value: np.ndarray) -> np.ndarray:
            candidate = np.asarray(value, dtype=np.float32).copy()
            anchor = candidate[0, [ROOT_X_IDX, ROOT_Z_IDX]].copy()
            candidate[:, [ROOT_X_IDX, ROOT_Z_IDX]] = (
                anchor[None]
                + float(root_scale)
                * (
                    candidate[:, [ROOT_X_IDX, ROOT_Z_IDX]]
                    - anchor[None]
                )
            )
            if protected is not None and np.any(protected):
                candidate[protected, 4:] = motion_ref[protected, 4:]
            return candidate

        motion, root_transaction = run_stage_transaction(
            stage_name="root_drift_restoration",
            motion=motion,
            apply_fn=root_drift_candidate,
            audit_fn=audit_fn,
            limits=limits,
            policy=policy,
            require_repair_gain=False,
        )
        triggered_root_metrics = [
            key
            for key, maximum in root_limits.items()
            if float(root_drift_before.get(key, float("-inf"))) > maximum
        ]
        selected_root_audit = dict(
            root_transaction.get("selected_audit", root_drift_before)
        )
        meaningful_root_gain = any(
            float(selected_root_audit.get(key, float("inf")))
            <= float(root_drift_before[key])
            - max(1.0e-7, 0.01 * (
                float(root_drift_before[key]) - root_limits[key]
            ))
            for key in triggered_root_metrics
        )
        if root_transaction.get("accepted") and not meaningful_root_gain:
            motion = pre_root_drift
            root_transaction.update({
                "accepted": False,
                "rolled_back": True,
                "selection": "identity",
                "reason": "no_meaningful_root_drift_gain",
                "selected_audit": root_drift_before,
            })
        root_transaction["candidate_contract"] = {
            "mode": "first_anchor_uniform_root_xz_path_scale_v1",
            "scale": float(root_scale),
            "triggered_metrics": triggered_root_metrics,
            "meaningful_root_gain": bool(meaningful_root_gain),
            "protected_geometry_restored_before_audit": bool(
                protected is not None and np.any(protected)
            ),
            "final_guard_unchanged": True,
        }
    else:
        root_transaction = {
            "stage": "root_drift_restoration",
            "accepted": False,
            "rolled_back": False,
            "reason": (
                "disabled" if not root_stage_enabled else "not_required"
            ),
            "candidate_contract": {
                "mode": "first_anchor_uniform_root_xz_path_scale_v1",
                "scale": float(root_scale),
                "triggered_metrics": [],
                "final_guard_unchanged": True,
            },
            "before_audit": root_drift_before,
            "selected_audit": root_drift_before,
        }
    stage["root_drift_restoration_transaction"] = root_transaction
    stage["root_drift_restoration_audit"] = audit_fn(motion)
    stage["motion_activity_root_drift_restoration"] = save_stage_snapshot(
        getattr(args, "out", None),
        "root_drift_restoration",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )

    ik_report = {"enabled": False}
    if bool(getattr(cfg, "ik_enable", False)) and env_bool(
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
    stage["final_audit"] = audit_fn(motion)
    stage["final_physical_gate"] = physical_quality_gate(
        stage["final_audit"]
    )
    stage["motion_activity_full_ik"] = save_stage_snapshot(
        getattr(args, "out", None),
        "full_ik",
        motion,
        fps=float(getattr(cfg, "fps", 30.0)),
    )
    return motion.astype(np.float32), stage


def physical_quality_gate(audit: Dict[str, Any]) -> Dict[str, Any]:
    """Delegate to the single authoritative SI physical-quality contract."""
    return evaluate_physical_audit(
        audit,
        limits=PhysicalQualityLimits.from_environment(),
    )


def audit_boundaries(
    motion_runtime,
    motion: np.ndarray,
    assembly_report: Sequence[Dict[str, Any]],
    cfg: Any,
    *,
    active_span: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selected_span = None
    if active_span is not None:
        if len(active_span) != 2:
            raise ValueError("active_span must contain [start, end]")
        selected_span = tuple(map(int, active_span))
    for i in range(1, len(assembly_report)):
        prev_core = assembly_report[i - 1].get("core_span")
        transition = assembly_report[i].get("transition_span")
        curr_core = assembly_report[i].get("core_span")
        if not prev_core or not curr_core:
            continue
        prev_end = int(prev_core[1])
        if transition is None:
            # No explicit transition: use an empty/short bridge, still audit direct join.
            t0 = prev_end
            t1 = prev_end
        else:
            t0, t1 = int(transition[0]), int(transition[1])
        c0, c1 = int(curr_core[0]), int(curr_core[1])
        dependency_start = min(max(0, prev_end - 4), t0, c0)
        dependency_end = max(prev_end, t1, min(c1, c0 + 4))
        if selected_span is not None and (
            dependency_end <= selected_span[0]
            or selected_span[1] <= dependency_start
        ):
            continue
        previous = motion[max(0, prev_end - 4):prev_end]
        bridge = motion[t0:t1]
        following = motion[c0:min(c1, c0 + 4)]
        risk = transition_risk(motion_runtime, previous, bridge, following, fps=float(getattr(cfg, "fps", 30.0)))
        failure_reasons = boundary_risk_reasons(risk)
        safe = not failure_reasons
        pred = assembly_report[i].get("risk_predicted", {})
        row = {
            "slot": int(i),
            "prev_event_id": int(assembly_report[i - 1].get("event_id", -1)),
            "curr_event_id": int(assembly_report[i].get("event_id", -1)),
            "candidate_rank": int(assembly_report[i].get("candidate_rank", -1)),
            "transition_start": int(t0),
            "transition_end": int(t1),
            "content_start": int(c0),
            "content_end": int(c1),
            "boundary_span": [
                int(dependency_start),
                int(dependency_end),
            ],
            "predicted_risk_score": float(assembly_report[i].get("risk_score_predicted", 0.0)),
            "predicted_boundary_jerk": float(pred.get("boundary_joint_jerk_max", 0.0)) if isinstance(pred, dict) else 0.0,
            "predicted_entry_fk_jump": float(pred.get("entry_fk_jump", 0.0)) if isinstance(pred, dict) else 0.0,
            "predicted_exit_fk_jump": float(pred.get("exit_fk_jump", 0.0)) if isinstance(pred, dict) else 0.0,
            "actual_risk_score": float(risk_score(risk)),
            "actual_boundary_jerk": float(risk.get("boundary_joint_jerk_max", 0.0)),
            "actual_entry_fk_jump": float(risk.get("entry_fk_jump", 0.0)),
            "actual_exit_fk_jump": float(risk.get("exit_fk_jump", 0.0)),
            "actual_entry_fk_jump_max_m": float(
                risk.get("entry_fk_jump_max_m", 0.0)
            ),
            "actual_exit_fk_jump_max_m": float(
                risk.get("exit_fk_jump_max_m", 0.0)
            ),
            "actual_entry_rotation_step_rad": float(risk.get("entry_rotation_step_rad", 0.0)),
            "actual_exit_rotation_step_rad": float(risk.get("exit_rotation_step_rad", 0.0)),
            "actual_foot_slip": float(risk.get("foot_slip", 0.0)),
            "actual_foot_slip_p95_mps": float(
                risk.get("foot_slip_p95", 0.0)
            ),
            "actual_foot_slip_peak_mps": float(
                risk.get("foot_slip_max", 0.0)
            ),
            "actual_foot_penetration": float(risk.get("foot_penetration", 0.0)),
            "actual_foot_penetration_depth_max_m": float(
                risk.get("foot_penetration_max_m", 0.0)
            ),
            "actual_contact_switch": float(risk.get("contact_switch", 0.0)),
            "safe": bool(safe),
            "failure_reasons": list(failure_reasons),
            "risk": risk,
            "decision": str(assembly_report[i].get("decision", "")),
            "transition_len": int(max(0, t1 - t0)),
            "core_warp": float(assembly_report[i].get("core_warp", 0.0)),
        }
        # Explicit-unit fields are the canonical report API.  The historical
        # names above remain for old analysis notebooks during migration.
        row.update(
            {
                "predicted_boundary_jerk_mps3": row["predicted_boundary_jerk"],
                "predicted_entry_fk_jump_m": row["predicted_entry_fk_jump"],
                "predicted_exit_fk_jump_m": row["predicted_exit_fk_jump"],
                "actual_boundary_jerk_mps3": row["actual_boundary_jerk"],
                "actual_entry_fk_jump_m": row["actual_entry_fk_jump"],
                "actual_exit_fk_jump_m": row["actual_exit_fk_jump"],
                "actual_foot_slip_mps": row["actual_foot_slip"],
                "actual_foot_penetration_m2": row["actual_foot_penetration"],
            }
        )
        rows.append(row)
    return rows


def write_audit_csv(rows: Sequence[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "slot", "prev_event_id", "curr_event_id", "candidate_rank",
        "transition_start", "transition_end", "transition_len", "content_start",
        "predicted_risk_score", "predicted_boundary_jerk", "predicted_entry_fk_jump", "predicted_exit_fk_jump",
        "actual_risk_score", "actual_boundary_jerk", "actual_entry_fk_jump", "actual_exit_fk_jump",
        "actual_entry_fk_jump_max_m", "actual_exit_fk_jump_max_m",
        "actual_entry_rotation_step_rad", "actual_exit_rotation_step_rad", "actual_foot_slip", "actual_foot_penetration",
        "actual_foot_slip_p95_mps", "actual_foot_slip_peak_mps", "actual_foot_penetration_depth_max_m",
        "actual_contact_switch", "core_warp", "safe", "decision",
        "predicted_boundary_jerk_mps3", "predicted_entry_fk_jump_m", "predicted_exit_fk_jump_m",
        "actual_boundary_jerk_mps3", "actual_entry_fk_jump_m", "actual_exit_fk_jump_m",
        "actual_foot_slip_mps", "actual_foot_penetration_m2",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def render_if_possible(
    motion_runtime,
    motion_path: str,
    audio_path: Optional[str],
    output_mp4: Optional[str],
    render_script: str = "rendering/render_motion.py",
    fps: float = 30.0,
) -> None:
    if not output_mp4:
        return
    if hasattr(motion_runtime, "render_if_possible"):
        try:
            motion_runtime.render_if_possible(
                motion_path,
                audio_path,
                output_mp4,
                render_script,
                fps=float(fps),
            )
            return
        except Exception as exc:
            print(f"[Boundary Closed-Loop WARN] motion_runtime.render_if_possible failed: {exc}", file=sys.stderr)
    if not audio_path or not Path(render_script).exists():
        print("[Boundary Closed-Loop WARN] render skipped", file=sys.stderr)
        return
    cmd = [
        sys.executable,
        render_script,
        "--motion", motion_path,
        "--audio", audio_path,
        "--output", output_mp4,
        "--fps", str(float(fps)),
    ]
    subprocess.run(cmd, check=False)


def set_cfg_runtime_knobs(cfg: Any) -> None:
    # Force routing reports to expose enough candidate_preview rows for closed-loop reselection.
    candidate_topk = env_int("BOUNDARY_CANDIDATE_TOPK", 48)
    try:
        setattr(cfg, "classification_report_topk", max(int(getattr(cfg, "classification_report_topk", 8)), candidate_topk))
    except Exception:
        pass


def merge_short_terminal_slot(
    slots: Sequence[Dict[str, Any]],
    slot_feat: np.ndarray,
    cfg: Any,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Merge a tiny terminal residual slot into the preceding slot.

    A final 1–3 frame residual cannot represent a meaningful motion event and
    causes extreme time compression, zero-length transitions, and false
    1e9 boundary-risk sentinels.
    """
    out_slots = [dict(s) for s in slots]
    feat = np.asarray(slot_feat, dtype=np.float32)

    if len(out_slots) < 2 or feat.shape[0] != len(out_slots):
        return out_slots, feat

    fps = float(getattr(cfg, "fps", 30.0))
    default_min = max(
        int(getattr(cfg, "min_event_frames", 1)),
        int(round(1.0 * fps)),
    )
    min_tail_frames = env_int(
        "BOUNDARY_MIN_TERMINAL_SLOT_FRAMES",
        default_min,
    )

    prev_frames = slot_target_frames(out_slots[-2], cfg)
    tail_frames = slot_target_frames(out_slots[-1], cfg)

    if tail_frames >= min_tail_frames:
        return out_slots, feat

    previous = dict(out_slots[-2])
    tail = dict(out_slots[-1])
    total_frames = int(prev_frames + tail_frames)

    merged = previous
    merged["target_frames"] = total_frames

    if "duration" in previous or "duration" in tail:
        merged["duration"] = float(total_frames / fps)
    if "duration_sec" in previous or "duration_sec" in tail:
        merged["duration_sec"] = float(total_frames / fps)

    for key in (
        "end",
        "end_sec",
        "end_time",
        "end_frame",
        "audio_end",
    ):
        if key in tail:
            merged[key] = tail[key]

    merged["terminal_tail_merge"] = {
        "enabled": True,
        "previous_frames": int(prev_frames),
        "tail_frames": int(tail_frames),
        "merged_frames": int(total_frames),
        "minimum_terminal_frames": int(min_tail_frames),
    }

    denom = float(max(1, total_frames))
    merged_feat = (
        feat[-2] * float(prev_frames)
        + feat[-1] * float(tail_frames)
    ) / denom

    out_slots[-2] = merged
    out_slots.pop()

    feat2 = feat[:-1].copy()
    feat2[-1] = merged_feat.astype(np.float32)

    print(
        f"[Terminal Merge TAIL MERGE] merged terminal slot: "
        f"{tail_frames} frames -> previous slot, "
        f"new_frames={total_frames}, slots={len(out_slots)}",
        file=sys.stderr,
    )
    return out_slots, feat2.astype(np.float32)


def formal_ctsr_schedule_contract(slots: Sequence[Dict[str, Any]]) -> bool:
    """Validate and identify the fail-closed formal Scheduler hand-off."""

    architectures = [str(slot.get("router_architecture", "")) for slot in slots]
    has_ctsr = any(value == "ctsr_weak_temporal_v1" for value in architectures)
    if not has_ctsr:
        return False
    if not slots or any(value != "ctsr_weak_temporal_v1" for value in architectures):
        raise RuntimeError("Formal CTSR schedule mixes Router architectures")
    for index, slot in enumerate(slots):
        if str(slot.get("router_supervision_source", "")) != "semantic_ot_teacher":
            raise RuntimeError(f"Formal slot {index} has invalid Router supervision")
        if bool(slot.get("router_compatibility_is_ground_truth", True)):
            raise RuntimeError(f"Formal slot {index} misdeclares Router evidence as ground truth")
        if bool(slot.get("action_compatibility_is_ground_truth", True)):
            raise RuntimeError(f"Formal slot {index} misdeclares action evidence as ground truth")
        if (
            str(slot.get("hierarchy_semantic_contract", ""))
            != "semantic_ot_teacher_x_weak_motion_local_action"
        ):
            raise RuntimeError(f"Formal slot {index} has a legacy hierarchy contract")
        if (
            str(slot.get("formal_candidate_contract", ""))
            != "ctsr_weak_scheduler_siblings_v1"
        ):
            raise RuntimeError(f"Formal slot {index} has no audited CTSR candidate set")
        candidates = slot.get("formal_candidate_event_uids")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError(f"Formal slot {index} has an empty CTSR candidate set")
        probabilities = slot.get("formal_candidate_router_probabilities")
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != len(candidates)
            or not np.isfinite(np.asarray(probabilities, dtype=np.float64)).all()
            or np.any(np.asarray(probabilities, dtype=np.float64) < 0.0)
        ):
            raise RuntimeError(
                f"Formal slot {index} has invalid candidate Router probabilities"
            )
    return True


def formal_candidate_state_from_slots(
    slots: Sequence[Dict[str, Any]],
    event_uids: Sequence[Any],
    *,
    boundary_top_k: int,
) -> Tuple[List[int], List[List[int]], List[Dict[str, Any]]]:
    """Resolve only Scheduler-issued CTSR candidates to Generation-DB rows.

    This function is intentionally shared with the feasibility slot splitter.
    Splitting a music slot must preserve its CTSR sibling set; it must never
    trigger a second retriever or construct new audio--motion pseudo-pairs.
    """

    formal_ctsr_schedule_contract(slots)
    uid_to_index = {str(uid): index for index, uid in enumerate(event_uids)}
    path_idx: List[int] = []
    candidate_lists: List[List[int]] = []
    retrieval_report: List[Dict[str, Any]] = []
    top_k = max(1, int(boundary_top_k))
    for slot_index, slot in enumerate(slots):
        scheduled_uid = str(
            slot.get("whole_song_event_uid", slot.get("event_uid", ""))
        )
        if scheduled_uid not in uid_to_index:
            raise RuntimeError(
                f"Formal slot {slot_index} references unknown event_uid={scheduled_uid!r}"
            )
        declared_uids = [str(value) for value in slot["formal_candidate_event_uids"]]
        declared_probabilities = [
            float(value) for value in slot["formal_candidate_router_probabilities"]
        ]
        probability_by_uid = dict(zip(declared_uids, declared_probabilities))
        candidate_indices: List[int] = []
        for uid in declared_uids:
            if uid not in uid_to_index:
                raise RuntimeError(
                    f"Formal slot {slot_index} candidate event_uid={uid!r} "
                    "is outside Generation DB"
                )
            event_index = int(uid_to_index[uid])
            if event_index not in candidate_indices:
                candidate_indices.append(event_index)
        exact_index = int(uid_to_index[scheduled_uid])
        candidate_indices = [exact_index] + [
            value for value in candidate_indices if value != exact_index
        ]
        selected = candidate_indices[:top_k]
        path_idx.append(exact_index)
        candidate_lists.append(selected)
        slot["closed_loop_conditioning_contract"] = (
            "selected_event_motion_descriptor_v1"
        )
        retrieval_report.append(
            {
                "slot": int(slot_index),
                "routing_policy": "formal_ctsr_scheduler_locked_candidates",
                "candidate_contract": "ctsr_weak_scheduler_siblings_v1",
                "candidate_event_uids": [str(event_uids[value]) for value in selected],
                "candidate_event_indices": list(map(int, selected)),
                "candidate_router_probabilities": [
                    float(probability_by_uid[str(event_uids[value])])
                    for value in selected
                ],
            }
        )
    return path_idx, candidate_lists, retrieval_report


def load_slots_and_candidates(motion_runtime, args: argparse.Namespace, cfg: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, List[int], List[Dict[str, Any]], List[List[int]]]:
    db = motion_runtime.load_db(args.db)
    if hasattr(motion_runtime, "_training_db_contract"):
        motion_runtime._training_db_contract(db, cfg, "Closed-loop Generation")
    event_uids = event_uids_from_generation_db(db)
    db["event_uids"] = event_uids
    db_contract = make_event_db_contract(event_uids)
    cfg._event_db_contract = db_contract
    strict_identity = env_bool("ROUTING_SAFETY_REQUIRE_ALIGNED_EVENT_DB", True)
    descriptor_contract = None
    slots_json = getattr(args, "slots_json", None)
    if slots_json and Path(slots_json).is_file():
        descriptor_obj = json.loads(Path(slots_json).read_text(encoding="utf-8"))
        descriptor_contract = normalize_event_db_contract(
            descriptor_obj.get("event_db_contract")
        )
    if strict_identity:
        assert_same_event_db_contract(
            db_contract,
            descriptor_contract,
            context="Scheduler/Generation Event-DB alignment",
        )
    slots, slot_feat = motion_runtime.audio_slots(args.audio, cfg, args.slot_seconds, getattr(args, "slots_json", None))
    slots, slot_feat = merge_short_terminal_slot(slots, slot_feat, cfg)
    uid_to_index = {str(uid): index for index, uid in enumerate(event_uids)}
    if not formal_ctsr_schedule_contract(slots):
        raise RuntimeError(
            "Closed-loop generation accepts only the formal CTSR-Weak schedule; "
            "historical retrievers are available only from the archived Git tag"
        )
    path_idx, candidate_lists, retrieval_report = formal_candidate_state_from_slots(
        slots,
        event_uids,
        boundary_top_k=max(1, env_int("BOUNDARY_RESELECT_TOPK", 32)),
    )
    descriptors = np.asarray(db.get("desc", []), dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] != 32:
        raise RuntimeError(
            f"Formal closed loop requires Event-DB desc=[N,32], got {descriptors.shape}"
        )
    slot_feat = descriptors[np.asarray(path_idx, dtype=np.int64)]
    for slot_index, slot in enumerate(slots):
        scheduled_uid = slot.get("whole_song_event_uid", slot.get("event_uid"))
        if not scheduled_uid:
            if strict_identity:
                raise RuntimeError(f"Slot {slot_index} has no stable whole_song_event_uid")
            continue
        scheduled_uid = str(scheduled_uid)
        if scheduled_uid not in uid_to_index:
            raise RuntimeError(
                f"Slot {slot_index} references event_uid={scheduled_uid!r} outside Generation DB"
            )
        exact_index = int(uid_to_index[scheduled_uid])
        path_idx[slot_index] = exact_index
        candidate_lists[slot_index] = [exact_index] + [
            int(value) for value in candidate_lists[slot_index] if int(value) != exact_index
        ]
        retrieval_report[slot_index]["scheduled_event_uid"] = scheduled_uid
        retrieval_report[slot_index]["scheduled_generation_event_index"] = exact_index
        retrieval_report[slot_index]["event_db_contract"] = db_contract
        retrieval_report[slot_index]["formal_ctsr_schedule"] = True
    return db, list(slots), np.asarray(slot_feat, dtype=np.float32), list(map(int, path_idx)), list(retrieval_report), candidate_lists


def gar_evaluation_trace_context(
    motion_runtime: Any,
    args: argparse.Namespace,
    cfg: Any,
    db: Mapping[str, Any],
    assembly_report: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve provenance only after the production decision path has finished."""

    if dataclasses.is_dataclass(cfg):
        config = dataclasses.asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        config = dict(jsonable(vars(cfg)))
    elif isinstance(cfg, Mapping):
        config = dict(jsonable(cfg))
    else:
        raise TypeError("GAR trace requires a mapping-like resolved configuration")
    runtime_environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(
            (
                "BOUNDARY_",
                "ROUTING_SAFETY_",
                "GENERATION_",
                "MOTION_ACTIVITY_",
                "GROUNDING_",
                "GRAPH_ROUTE_",
                "EVENT_HEADING_",
                "ROUTING_BUDGET_",
                "REPAIRABILITY_",
            )
        )
    }
    refiner_active = bool(getattr(cfg, "refiner_enable", False)) and env_bool(
        "BOUNDARY_USE_REFINER", True
    )
    diffusion_active = bool(getattr(cfg, "diffusion_enable", False)) and env_bool(
        "BOUNDARY_USE_DIFFUSION", True
    )
    ik_active = bool(getattr(cfg, "ik_enable", False)) and env_bool(
        "BOUNDARY_USE_IK", True
    )
    generator_config = {
        "pipeline": GAR_GENERATOR_VERSION,
        "refiner_active": refiner_active,
        "diffusion_active": diffusion_active,
        "ik_active": ik_active,
        "refiner_model_version": getattr(
            motion_runtime, "REFINER_MODEL_VERSION", None
        ),
        "diffusion_model_version": getattr(
            motion_runtime, "DIFFUSION_MODEL_VERSION", None
        ),
        "behavior_config_fingerprint": behavior_config_fingerprint(
            config,
            runtime_environment={
                key: value
                for key, value in runtime_environment.items()
                if not key.startswith("REPAIRABILITY_")
            },
        ),
    }
    checkpoint_fingerprint = checkpoint_bundle_fingerprint(
        {
            "refiner": getattr(args, "refiner", None) if refiner_active else None,
            "diffusion": (
                getattr(args, "diffusion", None) if diffusion_active else None
            ),
        }
    )
    repair_config = {
        "operator": GAR_REPAIR_OPERATOR_ID,
        "transition_train_min_seconds": getattr(
            cfg, "transition_train_min_seconds", None
        ),
        "transition_train_max_seconds": getattr(
            cfg, "transition_train_max_seconds", None
        ),
        "transition_root_tangent_max_mps": getattr(
            cfg, "transition_root_tangent_max_mps", None
        ),
        "transition_root_vertical_tangent_max_mps": getattr(
            cfg, "transition_root_vertical_tangent_max_mps", None
        ),
        "transition_angular_speed_max_rps": getattr(
            cfg, "transition_angular_speed_max_rps", None
        ),
        "transition_root_tangent_margin_m": getattr(
            cfg, "transition_root_tangent_margin_m", None
        ),
        "transition_tangent_smoothing_passes": getattr(
            cfg, "transition_tangent_smoothing_passes", None
        ),
        "risk_adaptive_transition_enabled": env_bool(
            "BOUNDARY_RISK_ADAPT_TRANSITION_ENABLE", True
        ),
        "boundary_environment": {
            key: value
            for key, value in runtime_environment.items()
            if key.startswith("BOUNDARY_")
        },
    }
    event_db_contract = make_event_db_contract(db["event_uids"])
    row_methods = {
        str(row.get("method", "")).strip()
        for row in assembly_report
        if str(row.get("method", "")).strip()
    }
    geometry_grounding_present = any(
        isinstance(row.get("risk_predicted"), Mapping)
        and "event_geometry_grounding" in row.get("risk_predicted", {})
        for row in assembly_report
    )
    repairability_modes = {
        str(
            row.get("repairability", {}).get("mode", "off")
            if isinstance(row.get("repairability", {}), Mapping)
            else "off"
        )
        for row in assembly_report
    }
    if "rank" in repairability_modes:
        selection_policy_id = (
            "fisher_rao_graph_sb_preorder_repairability_ranked_"
            "pre_safe_boundary_reselection_v1"
        )
        inferred_method_variant_id = "repairability_ranked_boundary_routing"
    elif "shadow" in repairability_modes:
        selection_policy_id = (
            "fisher_rao_graph_sb_preorder_repairability_shadow_"
            "boundary_reselection_v1"
        )
        inferred_method_variant_id = "repairability_shadow_boundary_routing"
    elif geometry_grounding_present and env_bool(
        "GROUNDING_GLOBAL_ROUTE_ENABLE", True
    ):
        selection_policy_id = (
            "fisher_rao_graph_sb_preorder_viability_aware_"
            "boundary_reselection_v1"
        )
        inferred_method_variant_id = "current_geometry_aware_routing"
    elif row_methods:
        selection_policy_id = "viability_aware_dynamic_beam_boundary_reselection_v1"
        inferred_method_variant_id = "current_viability_aware_routing"
    else:
        selection_policy_id = GAR_SELECTION_POLICY_ID
        inferred_method_variant_id = "current_boundary_closed_loop"
    configured_method_variant_id = str(
        getattr(
            cfg,
            "gar_evaluation_method_variant_id",
            "current_boundary_closed_loop",
        )
    )
    method_variant_id = (
        inferred_method_variant_id
        if configured_method_variant_id == "current_boundary_closed_loop"
        else configured_method_variant_id
    )
    return {
        "runtime_commit": current_git_commit(Path(__file__).resolve().parents[1]),
        "config_fingerprint": behavior_config_fingerprint(
            config, runtime_environment=runtime_environment
        ),
        "retrieval_index_fingerprint": str(
            event_db_contract["ordered_event_uid_sha256"]
        ),
        "generator_id": GAR_GENERATOR_ID,
        "generator_version": GAR_GENERATOR_VERSION,
        "generator_checkpoint_fingerprint": checkpoint_fingerprint,
        "generator_config_fingerprint": canonical_fingerprint(generator_config),
        "repair_operator_id": GAR_REPAIR_OPERATOR_ID,
        "repair_operator_version": GAR_REPAIR_OPERATOR_VERSION,
        "repair_config_fingerprint": canonical_fingerprint(repair_config),
        "selection_policy_id": selection_policy_id,
        "method_variant_id": method_variant_id,
        "random_seed": int(getattr(cfg, "seed", 0)),
        "risk_threshold_value": None,
        "risk_threshold_source": (
            "contracts.boundary_continuity."
            "BoundaryContinuityLimits.from_environment"
        ),
        "risk_thresholds": dataclasses.asdict(
            BoundaryContinuityLimits.from_environment()
        ),
        "capabilities": {
            "candidate_simulation_enabled": True,
            "adaptive_transition_enabled": env_bool(
                "BOUNDARY_RISK_ADAPT_TRANSITION_ENABLE", True
            ),
            "post_audit_enabled": True,
            "reselection_enabled": env_bool("BOUNDARY_RESELECT_ENABLE", True),
            "repairability_shadow_enabled": "shadow" in repairability_modes,
            "repairability_rank_enabled": "rank" in repairability_modes,
            "guard_remains_authoritative": True,
        },
    }


def _gar_add_runtime(
    runtime: Dict[str, Optional[float]],
    key: str,
    started_at: Optional[float],
) -> None:
    if started_at is None:
        return
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    previous = runtime.get(key)
    runtime[key] = float(elapsed_ms + (0.0 if previous is None else previous))


def collect_repairability_outcome_bank(
    *,
    motion_runtime: Any,
    args: argparse.Namespace,
    cfg: Any,
    db: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    slot_feat: np.ndarray,
    candidate_lists: Sequence[Sequence[int]],
    router_probabilities: Sequence[Mapping[int, float]],
    baseline_assembly: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Generate exhaustive local counterfactuals without changing production selection.

    Every trial freezes the final baseline event path except for one candidate at
    one boundary, resets a common random seed, runs the complete downstream
    generator and records the authoritative post-audit result.  The append-only
    writer makes a preempted 4090 run resumable.
    """

    enabled = bool(getattr(args, "outcome_bank", None)) or env_bool(
        "GAR_OUTCOME_BANK_ENABLE", False
    )
    if not enabled:
        return None
    runtime_mode = str(
        getattr(args, "repairability_mode", None)
        or os.environ.get("REPAIRABILITY_MODE", "off")
    ).strip().lower()
    if runtime_mode != "off":
        raise RuntimeError(
            "Outcome Bank capture requires REPAIRABILITY_MODE=off to avoid circular labels"
        )
    required_final_gates = {
        "ROUTING_SAFETY_REQUIRE_FINAL_PHYSICAL_GATE": env_bool(
            "ROUTING_SAFETY_REQUIRE_FINAL_PHYSICAL_GATE", True
        ),
        "BOUNDARY_REQUIRE_FINAL_BOUNDARY_GATE": env_bool(
            "BOUNDARY_REQUIRE_FINAL_BOUNDARY_GATE", True
        ),
        "MOTION_ACTIVITY_FINAL_GATE": env_bool(
            "MOTION_ACTIVITY_FINAL_GATE", True
        ),
    }
    disabled_final_gates = sorted(
        name for name, enabled in required_final_gates.items() if not enabled
    )
    if disabled_final_gates:
        raise RuntimeError(
            "Outcome Bank capture requires every production final quality gate; "
            "disabled=" + ",".join(disabled_final_gates)
        )
    if env_bool("BOUNDARY_STAGE_DIAGNOSTICS", False):
        raise RuntimeError(
            "Outcome Bank capture cannot be combined with stage snapshot diagnostics"
        )
    bank_path = str(
        getattr(args, "outcome_bank", None)
        or os.environ.get("GAR_OUTCOME_BANK_PATH", "")
    ).strip()
    if not bank_path:
        raise ValueError("Outcome Bank capture requires an output JSONL path")
    seed_text = str(
        getattr(args, "outcome_seeds", None)
        or os.environ.get("GAR_OUTCOME_BANK_SEEDS", "")
    )
    seeds = _comma_separated_ints(seed_text)
    if len(seeds) < 2:
        raise ValueError(
            "Outcome Bank probability supervision requires at least two common seeds"
        )
    full_pool_size = max((len(pool) for pool in candidate_lists[1:]), default=1)
    configured_top_k = max(
        1,
        int(
            getattr(args, "outcome_topk", None)
            or env_int("GAR_OUTCOME_BANK_TOPK", full_pool_size)
        ),
    )
    if configured_top_k < full_pool_size:
        raise ValueError(
            "GAR_OUTCOME_BANK_TOPK cannot truncate the production candidate pool; "
            "reduce BOUNDARY_RESELECT_TOPK for the entire frozen protocol instead"
        )
    top_k = full_pool_size
    requested_trials = sum(
        min(top_k, len(candidate_lists[slot])) * len(seeds)
        for slot in range(1, len(candidate_lists))
    )
    max_trials = max(1, env_int("GAR_OUTCOME_BANK_MAX_TRIALS", 4096))
    if requested_trials > max_trials:
        raise RuntimeError(
            f"Outcome Bank requests {requested_trials} trials, above "
            f"GAR_OUTCOME_BANK_MAX_TRIALS={max_trials}"
        )

    from evaluation.repairability_outcome_bank import (
        OUTCOME_BANK_RECORD_SCHEMA,
        OutcomeBankWriter,
        OutcomeRecord,
        canonical_fingerprint as bank_fingerprint,
        reason_families,
        read_outcome_bank,
        summarize_outcome_bank,
    )
    from model.repairability_predictor import proposal_feature_mapping

    event_uids = [str(value) for value in db["event_uids"]]
    sequence_id = make_sequence_id(args.audio, slots)
    baseline_events = [int(row["event_id"]) for row in baseline_assembly]
    if len(baseline_events) != len(slots):
        raise RuntimeError("Outcome Bank baseline path/slot count mismatch")
    context = gar_evaluation_trace_context(
        motion_runtime, args, cfg, db, baseline_assembly
    )
    generator_fingerprint = bank_fingerprint(
        {
            "id": context["generator_id"],
            "version": context["generator_version"],
            "checkpoint": context["generator_checkpoint_fingerprint"],
            "config": context["generator_config_fingerprint"],
        }
    )
    repair_fingerprint = bank_fingerprint(
        {
            "id": context["repair_operator_id"],
            "version": context["repair_operator_version"],
            "config": context["repair_config_fingerprint"],
        }
    )
    writer = OutcomeBankWriter(bank_path)
    bank_args = argparse.Namespace(**vars(args))
    bank_args.out = None
    bank_args.render_output = None
    completed_now = 0
    skipped_existing = 0
    started = time.perf_counter()
    for slot_index in range(1, len(slots)):
        pool = [int(value) for value in candidate_lists[slot_index][:top_k]]
        probability_map = dict(router_probabilities[slot_index])
        pool_fingerprint = bank_fingerprint(
            [
                {
                    "candidate_id": event_uids[event_id],
                    "rank": rank,
                    "router_probability": float(probability_map.get(event_id, 0.0)),
                }
                for rank, event_id in enumerate(pool)
            ]
        )
        boundary_id = make_boundary_id(sequence_id, slot_index)
        case_id = make_evaluation_case_id(sequence_id, slot_index)
        for original_rank, event_id in enumerate(pool):
            candidate_id = event_uids[event_id]
            for trial_seed in seeds:
                key = (case_id, candidate_id, int(trial_seed))
                existing_record = writer.get(key)
                if existing_record is not None:
                    if (
                        existing_record.candidate_pool_fingerprint
                        != pool_fingerprint
                        or existing_record.original_rank != original_rank
                        or existing_record.candidate_pool_size != len(pool)
                    ):
                        raise RuntimeError(
                            "existing Outcome Bank trial uses a different candidate pool"
                        )
                    skipped_existing += 1
                    continue
                _seed_everything(motion_runtime, trial_seed)
                forced_lists = [[event_id_] for event_id_ in baseline_events]
                forced_lists[slot_index] = [event_id]
                trial_started = time.perf_counter()
                motion_ref, assembly, _ = assemble_closed_loop_reference(
                    motion_runtime,
                    slots,
                    forced_lists,
                    dict(db),
                    cfg,
                    candidate_router_probabilities=router_probabilities,
                    repairability_mode="off",
                )
                selected_row = assembly[slot_index]
                risk = dict(selected_row.get("risk_predicted", {}))
                source_frames = max(
                    1,
                    int(
                        selected_row.get("length_policy", {}).get(
                            "source_frames", selected_row.get("core_frames", 1)
                        )
                    ),
                )
                features = proposal_feature_mapping(
                    router_probability=float(probability_map.get(event_id, 0.0)),
                    original_rank=original_rank,
                    pool_size=len(pool),
                    target_frames=int(selected_row["target_frames"]),
                    transition_frames=int(selected_row["transition_in_frames"]),
                    core_frames=int(selected_row["core_frames"]),
                    core_warp=float(selected_row["core_frames"]) / source_frames,
                    pre_safe=bool(selected_row["safe_predicted"]),
                    pre_risk_score=float(selected_row["risk_score_predicted"]),
                    risk=risk,
                )
                condition = compute_condition(
                    motion_runtime,
                    slot_feat,
                    assembly,
                    motion_ref.shape[0],
                    dict(db),
                )
                transition_spans = transition_spans_from_report(assembly)
                seam_mask, _, _ = make_seam_mask(
                    motion_runtime, motion_ref.shape[0], transition_spans, cfg
                )
                slide_eligible, _ = sliding_support_eligibility(
                    dict(db), assembly, motion_ref.shape[0]
                )
                motion, trial_stage_reports = apply_generators(
                    motion_runtime,
                    motion_ref,
                    condition,
                    seam_mask,
                    bank_args,
                    cfg,
                    sliding_support_eligible=slide_eligible,
                )
                audits = audit_boundaries(motion_runtime, motion, assembly, cfg)
                matching = [
                    row for row in audits if int(row.get("slot", -1)) == slot_index
                ]
                if len(matching) != 1:
                    raise RuntimeError(
                        "Outcome Bank trial did not produce exactly one target audit"
                    )
                audit = matching[0]
                boundary_gate = evaluate_boundary_continuity(
                    audits,
                    expected_boundaries=max(0, len(assembly) - 1),
                )
                boundary_reasons = tuple(
                    f"boundary:{value}"
                    for value in boundary_gate.get("reasons", ())
                )
                physical_gate = dict(
                    trial_stage_reports.get("final_physical_gate", {})
                )
                if physical_gate.get("schema") != "final_generation_physical_gate_v1":
                    raise RuntimeError(
                        "Outcome Bank trial is missing the authoritative physical gate"
                    )
                physical_reasons = tuple(
                    f"physical:{value}"
                    for value in physical_gate.get("reasons", ())
                )
                activity_gate = evaluate_final_motion_activity(
                    motion,
                    slots=slots,
                    assembly_report=assembly,
                    fps=float(getattr(cfg, "fps", 30.0)),
                )
                activity_reasons = tuple(
                    f"activity:{value}"
                    for value in activity_gate.get("reasons", ())
                )
                schedule_safe = True
                schedule_reasons: tuple[str, ...] = ()
                try:
                    constraint_rows = final_selection_constraint_rows(
                        dict(db), assembly
                    )
                    assert_schedule_hard_constraints(
                        constraint_rows,
                        max_pose_hold_ratio=env_float(
                            "GENERATION_MAX_POSE_HOLD_RATIO",
                            DEFAULT_MAX_POSE_HOLD_RATIO,
                        ),
                        max_single_source_ratio=env_float(
                            "ROUTING_SAFETY_MAX_SOURCE_SHARE",
                            DEFAULT_MAX_SINGLE_SOURCE_RATIO,
                        ),
                        max_single_recording_ratio=env_float(
                            "ROUTING_SAFETY_MAX_RECORDING_SHARE",
                            DEFAULT_MAX_SINGLE_SOURCE_RATIO,
                        ),
                        min_unique_events=env_int(
                            "GENERATION_MIN_UNIQUE_EVENTS",
                            DEFAULT_MIN_UNIQUE_EVENTS,
                        ),
                        min_core_frame_ratio=env_float(
                            "GENERATION_MIN_CORE_FRAME_RATIO",
                            DEFAULT_MIN_CORE_FRAME_RATIO,
                        ),
                    )
                except (RuntimeError, ValueError, AssertionError) as exc:
                    schedule_safe = False
                    schedule_reasons = (
                        f"schedule:{type(exc).__name__}:{exc}",
                    )
                boundary_safe = bool(boundary_gate.get("ok", False))
                physical_safe = bool(physical_gate.get("ok", False))
                activity_safe = bool(activity_gate.get("ok", False))
                if boundary_safe:
                    boundary_reasons = ()
                if physical_safe:
                    physical_reasons = ()
                if activity_safe:
                    activity_reasons = ()
                if not boundary_safe and not boundary_reasons:
                    boundary_reasons = ("boundary:unspecified_failure",)
                if not physical_safe and not physical_reasons:
                    physical_reasons = ("physical:unspecified_failure",)
                if not activity_safe and not activity_reasons:
                    activity_reasons = ("activity:unspecified_failure",)
                reasons = tuple(
                    dict.fromkeys(
                        boundary_reasons
                        + physical_reasons
                        + activity_reasons
                        + schedule_reasons
                    )
                )
                recording_value = (
                    None
                    if "recording_uids" not in db
                    else np.asarray(db["recording_uids"], dtype=object)[event_id]
                )
                performer_value = (
                    None
                    if "dancer_ids" not in db
                    else np.asarray(db["dancer_ids"], dtype=object)[event_id]
                )
                record = OutcomeRecord(
                    schema=OUTCOME_BANK_RECORD_SCHEMA,
                    sequence_id=sequence_id,
                    boundary_id=boundary_id,
                    evaluation_case_id=case_id,
                    group_id=sequence_id,
                    slot_index=slot_index,
                    candidate_id=candidate_id,
                    candidate_event_index=event_id,
                    source_recording_id=(
                        None if recording_value in (None, "") else str(recording_value)
                    ),
                    source_performer_id=(
                        None if performer_value in (None, "") else str(performer_value)
                    ),
                    original_rank=original_rank,
                    candidate_pool_size=len(pool),
                    random_seed=int(trial_seed),
                    features=features,
                    pre_safe=bool(selected_row["safe_predicted"]),
                    pre_risk=float(selected_row["risk_score_predicted"]),
                    boundary_safe=boundary_safe,
                    physical_safe=physical_safe,
                    activity_safe=activity_safe,
                    schedule_safe=schedule_safe,
                    post_safe=bool(
                        boundary_safe
                        and physical_safe
                        and activity_safe
                        and schedule_safe
                    ),
                    post_risk=float(audit.get("actual_risk_score", 0.0)),
                    failure_reasons=reasons,
                    violation_families=reason_families(reasons),
                    runtime_ms=(time.perf_counter() - trial_started) * 1000.0,
                    runtime_commit=str(context["runtime_commit"]),
                    config_fingerprint=str(context["config_fingerprint"]),
                    candidate_pool_fingerprint=pool_fingerprint,
                    generator_fingerprint=generator_fingerprint,
                    repair_fingerprint=repair_fingerprint,
                )
                writer.append(record)
                completed_now += 1
    _seed_everything(motion_runtime, int(getattr(cfg, "seed", 42)))
    all_records = read_outcome_bank(bank_path)
    summary = summarize_outcome_bank(all_records, required_seeds=seeds)
    summary.update(
        {
            "path": bank_path,
            "requested_trials_this_sequence": requested_trials,
            "completed_now": completed_now,
            "skipped_existing": skipped_existing,
            "capture_runtime_ms": (time.perf_counter() - started) * 1000.0,
            "common_random_seeds": seeds,
            "top_k": top_k,
            "production_selection_changed": False,
        }
    )
    return summary


def generate_closed_loop(args: argparse.Namespace) -> int:
    motion_runtime = import_motion_runtime()
    cfg = motion_runtime.MotionGenerationConfig.from_json(args.config).apply_env()
    set_cfg_runtime_knobs(cfg)
    diagnostic_root = None
    if env_bool("BOUNDARY_STAGE_DIAGNOSTICS", False):
        from uuid import uuid4
        diagnostic_root = Path(args.out).resolve().parent / (
            Path(args.out).stem + ".diagnostics." + uuid4().hex[:12]
        )
        diagnostic_root.mkdir(parents=True, exist_ok=False)
    gar_trace_enabled = bool(getattr(cfg, "gar_evaluation_trace_enable", False))
    gar_sequence_started = time.perf_counter() if gar_trace_enabled else None
    gar_runtime: Dict[str, Optional[float]] = {
        "retrieval_runtime_ms": None,
        "candidate_simulation_runtime_ms": None,
        "generation_runtime_ms": None,
        "post_audit_runtime_ms": None,
        "reselection_runtime_ms": None,
        "sequence_total_runtime_ms": None,
    }
    gar_round_records: List[Dict[str, Any]] = []

    seed = int(getattr(cfg, "seed", 1234))
    _seed_everything(motion_runtime, seed)

    gar_stage_started = time.perf_counter() if gar_trace_enabled else None
    db, slots, slot_feat, path_idx, retrieval_report, candidate_lists = load_slots_and_candidates(motion_runtime, args, cfg)
    router_probability_maps = candidate_router_probability_maps(retrieval_report)
    repairability_settings = repairability_runtime_settings(args)
    repairability_ranker = load_repairability_ranker(repairability_settings)
    _gar_add_runtime(gar_runtime, "retrieval_runtime_ms", gar_stage_started)

    banned: Dict[int, set] = {}
    rounds: List[Dict[str, Any]] = []
    best_payload: Optional[Dict[str, Any]] = None
    max_rounds = max(0, env_int("BOUNDARY_MAX_RESELECT_ROUNDS", 2))
    enable_reselect = env_bool("BOUNDARY_RESELECT_ENABLE", True)

    for round_id in range(max_rounds + 1):
        gar_stage_started = time.perf_counter() if gar_trace_enabled else None
        round_diagnostic_dir = None
        if diagnostic_root is not None:
            round_diagnostic_dir = diagnostic_root / f"round_{round_id:03d}"
            round_diagnostic_dir.mkdir(exist_ok=False)
        motion_ref, assembly_report, selected_pairs = assemble_closed_loop_reference(
            motion_runtime, slots, candidate_lists, db, cfg, banned=banned,
            diagnostic_dir=round_diagnostic_dir,
            candidate_router_probabilities=router_probability_maps,
            repairability_ranker=repairability_ranker,
            repairability_mode=str(repairability_settings["mode"]),
        )
        _gar_add_runtime(
            gar_runtime, "candidate_simulation_runtime_ms", gar_stage_started
        )
        cond = compute_condition(
            motion_runtime,
            slot_feat,
            assembly_report,
            motion_ref.shape[0],
            db,
        )
        transition_spans = transition_spans_from_report(assembly_report)
        seam_mask, seam_positions, mask_policy = make_seam_mask(motion_runtime, motion_ref.shape[0], transition_spans, cfg)
        slide_eligible, slide_report = sliding_support_eligibility(
            db,
            assembly_report,
            motion_ref.shape[0],
        )
        gar_stage_started = time.perf_counter() if gar_trace_enabled else None
        generator_args = args
        if round_diagnostic_dir is not None:
            generator_args = argparse.Namespace(**vars(args))
            generator_args.out = str(round_diagnostic_dir / "motion.npy")
        motion, stage_reports = apply_generators(
            motion_runtime,
            motion_ref,
            cond,
            seam_mask,
            generator_args,
            cfg,
            sliding_support_eligible=slide_eligible,
        )
        _gar_add_runtime(gar_runtime, "generation_runtime_ms", gar_stage_started)
        stage_reports["sliding_support_eligibility"] = slide_report
        if round_diagnostic_dir is not None:
            from training.generation_stage_diagnostics import save_round_bundle
            stage_reports["generation_stage_diagnostics"] = save_round_bundle(
                round_diagnostic_dir, round_id, motion_ref, motion, cond,
                seam_mask, slide_eligible, assembly_report, stage_reports,
                cfg, db, args,
            )
        conditioning_contract = (
            str(assembly_report[0].get("conditioning_contract", "unknown"))
            if assembly_report
            else "unknown"
        )
        stage_reports["neural_music_conditioning"] = {
            "mode": (
                "selected_event_motion_descriptor_repair_conditioning"
                if conditioning_contract == "selected_event_motion_descriptor_v1"
                else "invalid_non_event_descriptor_conditioning"
            ),
            "conditioning_contract": conditioning_contract,
            "shape": list(cond.shape),
            "slot_count": int(len(slots)),
            "recomputed_after_each_reselection": True,
            "transition_policy": "linear_previous_to_current_slot_descriptor",
            "normalization": "generation_event_db_descriptor_coordinates",
            "whole_song_mean_conditioning": False,
            "music_controls_routing_and_timing": bool(
                conditioning_contract == "selected_event_motion_descriptor_v1"
            ),
            "categorical_music_label_used_as_body_semantics": False
            if conditioning_contract == "selected_event_motion_descriptor_v1"
            else None,
        }
        gar_stage_started = time.perf_counter() if gar_trace_enabled else None
        boundary_rows = audit_boundaries(motion_runtime, motion, assembly_report, cfg)
        _gar_add_runtime(gar_runtime, "post_audit_runtime_ms", gar_stage_started)
        unsafe_rows = [r for r in boundary_rows if not bool(r.get("safe"))]
        round_summary = {
            "round": int(round_id),
            "unsafe_boundaries": int(len(unsafe_rows)),
            "num_boundaries": int(len(boundary_rows)),
            "selected_pairs": selected_pairs,
            "banned": {str(k): sorted(map(int, v)) for k, v in banned.items()},
            "worst_actual_risk_score": float(max([r.get("actual_risk_score", 0.0) for r in boundary_rows], default=0.0)),
            "motion_ref_frames": int(motion_ref.shape[0]),
            "final_frames": int(motion.shape[0]),
        }
        rounds.append(round_summary)
        if gar_trace_enabled:
            gar_round_records.append(
                {
                    "round": int(round_id),
                    "assembly_report": jsonable(assembly_report),
                    "boundary_rows": jsonable(boundary_rows),
                    "selected_pairs": jsonable(selected_pairs),
                }
            )
        payload = {
            "round": round_id,
            "motion_ref": motion_ref,
            "motion": motion,
            "assembly_report": assembly_report,
            "transition_spans": transition_spans,
            "seam_mask": seam_mask,
            "seam_positions": seam_positions,
            "mask_policy": mask_policy,
            "stage_reports": stage_reports,
            "boundary_rows": boundary_rows,
            "unsafe_rows": unsafe_rows,
            "selected_pairs": selected_pairs,
        }
        if best_payload is None or len(unsafe_rows) < len(best_payload["unsafe_rows"]):
            best_payload = payload
        if not unsafe_rows or not enable_reselect:
            best_payload = payload
            break
        # Ban the current event for the worst unsafe current slot and rerun whole assembly.
        gar_stage_started = time.perf_counter() if gar_trace_enabled else None
        worst = max(unsafe_rows, key=lambda r: float(r.get("actual_risk_score", 0.0)))
        slot = int(worst.get("slot", -1))
        curr = int(worst.get("curr_event_id", -1))
        if slot < 0 or curr < 0:
            break
        banned.setdefault(slot, set()).add(curr)
        _gar_add_runtime(gar_runtime, "reselection_runtime_ms", gar_stage_started)
        # Stop if we have exhausted candidates for this slot.
        remaining = [x for x in candidate_lists[slot] if x not in banned.get(slot, set())]
        if not remaining:
            break

    if best_payload is None:
        raise RuntimeError("Closed-loop generation produced no payload")

    outcome_bank_summary = None

    final_constraint_rows = final_selection_constraint_rows(
        db,
        best_payload["assembly_report"],
    )
    final_schedule_hard_constraints = assert_schedule_hard_constraints(
        final_constraint_rows,
        max_pose_hold_ratio=env_float(
            "GENERATION_MAX_POSE_HOLD_RATIO", DEFAULT_MAX_POSE_HOLD_RATIO
        ),
        max_single_source_ratio=env_float(
            "ROUTING_SAFETY_MAX_SOURCE_SHARE", DEFAULT_MAX_SINGLE_SOURCE_RATIO
        ),
        max_single_recording_ratio=env_float(
            "ROUTING_SAFETY_MAX_RECORDING_SHARE", DEFAULT_MAX_SINGLE_SOURCE_RATIO
        ),
        min_unique_events=env_int(
            "GENERATION_MIN_UNIQUE_EVENTS", DEFAULT_MIN_UNIQUE_EVENTS
        ),
        min_core_frame_ratio=env_float(
            "GENERATION_MIN_CORE_FRAME_RATIO", DEFAULT_MIN_CORE_FRAME_RATIO
        ),
    )
    best_payload["stage_reports"][
        "final_music_independent_hard_constraints"
    ] = final_schedule_hard_constraints

    final_gate = physical_quality_gate(
        best_payload["stage_reports"].get("final_audit", {})
    )
    best_payload["stage_reports"]["final_physical_gate"] = final_gate
    final_boundary_continuity = evaluate_boundary_continuity(
        best_payload["boundary_rows"],
        expected_boundaries=max(0, len(best_payload["assembly_report"]) - 1),
    )
    best_payload["stage_reports"][
        "final_boundary_continuity_gate"
    ] = final_boundary_continuity

    # The activity gate is additive and is evaluated only after immutable
    # physical/anatomy/heading processing has completed.
    final_motion_activity = evaluate_final_motion_activity(
        best_payload["motion"],
        slots=slots,
        assembly_report=best_payload["assembly_report"],
        fps=float(getattr(cfg, "fps", 30.0)),
    )
    best_payload["stage_reports"]["final_motion_activity"] = final_motion_activity

    quality_layers = {
        "anti_freeze_anti_collapse": {
            "ok": bool(final_motion_activity["ok"]),
            "reasons": list(final_motion_activity["reasons"]),
        },
        **dict(final_gate.get("layers", {})),
        "boundary_continuity": {
            "ok": bool(final_boundary_continuity["ok"]),
            "reasons": list(final_boundary_continuity["reasons"]),
        },
    }
    required_failures: list[str] = []
    if env_bool("ROUTING_SAFETY_REQUIRE_FINAL_PHYSICAL_GATE", True) and not bool(
        final_gate["ok"]
    ):
        required_failures.append(
            "physical:" + ",".join(str(value) for value in final_gate["reasons"])
        )
    if env_bool("BOUNDARY_REQUIRE_FINAL_BOUNDARY_GATE", True) and not bool(
        final_boundary_continuity["ok"]
    ):
        required_failures.append(
            "boundary:"
            + ",".join(str(value) for value in final_boundary_continuity["reasons"])
        )
    if env_bool("MOTION_ACTIVITY_FINAL_GATE", True) and not bool(
        final_motion_activity["ok"]
    ):
        required_failures.append(
            "activity:"
            + ",".join(str(value) for value in final_motion_activity["reasons"])
        )
    final_quality_gate = {
        "schema": "final_motion_quality_layers_v1",
        "ok": not required_failures,
        "reasons": list(required_failures),
        "layers": quality_layers,
        "rejected_output_is_renderable": False,
    }
    best_payload["stage_reports"]["final_quality_gate"] = final_quality_gate

    out = Path(args.out)
    if out.suffix.lower() != ".npy":
        raise ValueError(f"--out must end in .npy, got {out}")
    artifact_out = (
        out
        if not required_failures
        else out.with_name(out.stem + ".rejected.npy")
    )
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifact_out, best_payload["motion"].astype(np.float32))
    motion_ref_path = str(
        artifact_out.with_name(artifact_out.stem + ".motion_ref.npy")
    )
    mask_path = str(
        artifact_out.with_name(artifact_out.stem + ".transition_mask.npy")
    )
    audit_csv_path = str(out.with_name(out.stem + ".boundary_audit.csv"))
    audit_json_path = str(out.with_name(out.stem + ".boundary_audit.json"))
    motion_activity_path = str(
        out.with_name(out.stem + ".motion_activity.json")
    )
    write_activity_report(final_motion_activity, motion_activity_path)
    np.save(motion_ref_path, best_payload["motion_ref"].astype(np.float32))
    np.save(mask_path, best_payload["seam_mask"].astype(np.float32))
    write_audit_csv(best_payload["boundary_rows"], audit_csv_path)
    save_json(best_payload["boundary_rows"], audit_json_path)

    paths = np.asarray(db["paths"], dtype=object)
    selected_event_indices = [int(x[0]) for x in best_payload["selected_pairs"]]
    selected_paths = [str(paths[i]) for i in selected_event_indices]

    gar_trace_path: Optional[str] = None
    gar_method_variant_id = str(
        getattr(
            cfg,
            "gar_evaluation_method_variant_id",
            "current_boundary_closed_loop",
        )
    )
    if gar_trace_enabled:
        _gar_add_runtime(
            gar_runtime, "sequence_total_runtime_ms", gar_sequence_started
        )
        gar_context = gar_evaluation_trace_context(
            motion_runtime,
            args,
            cfg,
            db,
            best_payload["assembly_report"],
        )
        gar_trace = build_closed_loop_trace(
            audio=args.audio,
            slots=slots,
            db=db,
            event_uids=db["event_uids"],
            candidate_lists=candidate_lists,
            retrieval_report=retrieval_report,
            round_records=gar_round_records,
            final_round=int(best_payload["round"]),
            runtime=gar_runtime,
            **gar_context,
        )
        gar_trace_path = str(
            out.with_name(out.stem + ".gar_selection_trace.json")
        )
        write_gar_trace(gar_trace, gar_trace_path)
        gar_method_variant_id = gar_trace.method_variant_id

    outcome_bank_summary = collect_repairability_outcome_bank(
        motion_runtime=motion_runtime,
        args=args,
        cfg=cfg,
        db=db,
        slots=slots,
        slot_feat=slot_feat,
        candidate_lists=candidate_lists,
        router_probabilities=router_probability_maps,
        baseline_assembly=best_payload["assembly_report"],
    )

    gar_readiness = {
        "schema": GAR_READINESS_INTERFACE_SCHEMA,
        "trace_enabled": gar_trace_enabled,
        "trace_path": gar_trace_path,
        "method_variant_id": gar_method_variant_id,
        "paper1_experiments_implemented": False,
        "oracle_implemented": False,
        "statistical_tests_implemented": False,
        "long_horizon_benchmark_implemented": False,
        "production_selection_behavior_changed": bool(
            repairability_settings["mode"] == "rank"
        ),
    }

    report = {
        "version": "boundary_closed_loop_boundary_simulated_closed_loop_scheduler",
        "audio": args.audio,
        "db": args.db,
        "motion_path": str(artifact_out),
        "requested_motion_path": str(out),
        "fps": float(getattr(cfg, "fps", 30.0)),
        "event_db_contract": make_event_db_contract(db["event_uids"]),
        "config": dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else jsonable(cfg),
        "selected_event_indices_initial": path_idx,
        "selected_event_indices_final": selected_event_indices,
        "selected_event_paths_final": selected_paths,
        "slots": slots,
        "motion_ref_path": motion_ref_path,
        "transition_mask_path": mask_path,
        "boundary_audit_csv": audit_csv_path,
        "boundary_audit_json": audit_json_path,
        "motion_activity_json": motion_activity_path,
        "motion_activity": final_motion_activity,
        "gar_evaluation_readiness": gar_readiness,
        "heading_contract": {
            "schema": "formal_boundary_aligned_heading_v1",
            "authoritative_reference": "motion_ref_path",
            "reference_construction": "selected_event_core_boundary_alignment",
            "post_repair_heading_must_match_reference": True,
            "event_turn_budget_source": "generation_event_db",
            "legacy_event_heading_planner_required": False,
        },
        "final_quality_gate": final_quality_gate,
        "closed_loop": {
            "enabled": True,
            "rounds": rounds,
            "final_round": int(best_payload["round"]),
            "candidate_topk": int(env_int("BOUNDARY_CANDIDATE_TOPK", 48)),
            "reselect_topk": int(env_int("BOUNDARY_RESELECT_TOPK", env_int("BOUNDARY_CANDIDATE_TOPK", 32))),
            "reselect_enabled": bool(enable_reselect),
            "risk_adaptive_transition_enabled": env_bool("BOUNDARY_RISK_ADAPT_TRANSITION_ENABLE", True),
            "simulated_edge_risk_enabled": True,
            "repairability": {
                "mode": str(repairability_settings["mode"]),
                "checkpoint": repairability_settings.get("checkpoint"),
                "ranker_metadata": (
                    None
                    if repairability_ranker is None
                    else dict(repairability_ranker.checkpoint_metadata)
                ),
                "graph_sb_unary_changed": False,
                "guard_is_authoritative": True,
                "pre_safe_gate_is_authoritative": True,
            },
            "outcome_bank": outcome_bank_summary,
            "env": {k: v for k, v in os.environ.items() if k.startswith("BOUNDARY_")},
            "diversity_env": {k: v for k, v in os.environ.items() if k.startswith("ROUTING_SAFETY_")},
        },
        "stage_reports": {
            "retrieval": retrieval_report,
            "closed_loop_concat": best_payload["assembly_report"],
            "seams": best_payload["seam_positions"],
            "transition_spans": best_payload["transition_spans"],
            "seam_mask_policy": best_payload["mask_policy"],
            **best_payload["stage_reports"],
        },
        "boundary_audit_summary": {
            "num_boundaries": int(len(best_payload["boundary_rows"])),
            "safe_boundaries": int(sum(bool(r.get("safe")) for r in best_payload["boundary_rows"])),
            "unsafe_boundaries": int(sum(not bool(r.get("safe")) for r in best_payload["boundary_rows"])),
            "actual_boundary_jerk_p95": float(np.percentile([r.get("actual_boundary_jerk", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_entry_fk_jump_p95": float(np.percentile([r.get("actual_entry_fk_jump", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_exit_fk_jump_p95": float(np.percentile([r.get("actual_exit_fk_jump", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_foot_slip_p95": float(np.percentile([r.get("actual_foot_slip", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_boundary_jerk_p95_mps3": float(np.percentile([r.get("actual_boundary_jerk_mps3", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_entry_fk_jump_p95_m": float(np.percentile([r.get("actual_entry_fk_jump_m", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_exit_fk_jump_p95_m": float(np.percentile([r.get("actual_exit_fk_jump_m", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_foot_slip_p95_mps": float(np.percentile([r.get("actual_foot_slip_mps", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_entry_fk_joint_jump_max_m": float(max([r.get("actual_entry_fk_jump_max_m", 0.0) for r in best_payload["boundary_rows"]], default=0.0)),
            "actual_exit_fk_joint_jump_max_m": float(max([r.get("actual_exit_fk_jump_max_m", 0.0) for r in best_payload["boundary_rows"]], default=0.0)),
            "actual_supported_foot_slip_p95_max_mps": float(max([r.get("actual_foot_slip_p95_mps", 0.0) for r in best_payload["boundary_rows"]], default=0.0)),
            "actual_supported_foot_slip_peak_max_mps": float(max([r.get("actual_foot_slip_peak_mps", 0.0) for r in best_payload["boundary_rows"]], default=0.0)),
            "actual_foot_penetration_depth_max_m": float(max([r.get("actual_foot_penetration_depth_max_m", 0.0) for r in best_payload["boundary_rows"]], default=0.0)),
            "physical_units": {
                "boundary_jerk": "m/s^3",
                "entry_fk_jump": "m",
                "exit_fk_jump": "m",
                "entry_fk_joint_jump_max": "m",
                "exit_fk_joint_jump_max": "m",
                "exit_rotation_step": "rad/frame",
                "foot_slip": "m/s",
                "foot_penetration": "m^2_mean_squared_depth",
                "foot_penetration_depth_max": "m",
            },
        },
        "final_audit": best_payload["stage_reports"].get("final_audit", {}),
    }
    json_path = args.json or str(
        out.with_name(out.stem + ".boundary_closed_loop_closed_loop_report.json")
    )
    save_json(report, json_path)

    if args.render_output and not required_failures:
        render_if_possible(
            motion_runtime,
            str(artifact_out),
            args.audio,
            args.render_output,
            args.render_script,
            fps=float(getattr(cfg, "fps", 30.0)),
        )

    print(json.dumps(jsonable({
        "motion": str(artifact_out),
        "requested_motion": str(out),
        "motion_ref": motion_ref_path,
        "transition_mask": mask_path,
        "json": json_path,
        "boundary_audit_csv": audit_csv_path,
        "motion_activity_json": motion_activity_path,
        "gar_selection_trace_json": gar_trace_path,
        "frames": int(best_payload["motion"].shape[0]),
        "boundary_audit_summary": report["boundary_audit_summary"],
        "final_audit": report["final_audit"],
        "final_motion_activity": final_motion_activity,
        "final_quality_gate": final_quality_gate,
    }), ensure_ascii=False, indent=2))

    # Preserve NPY/JSON diagnostics but never render or silently accept an
    # output that fails any required final quality layer.
    if required_failures:
        raise RuntimeError(
            "Final motion quality gate rejected generated motion; diagnostics "
            f"were preserved at {json_path}; reasons="
            + ";".join(required_failures)
        )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Boundary Closed-Loop closed-loop boundary-safe generator for EDGE 151D")
    p.add_argument("cmd", choices=["generate"], help="subcommand")
    p.add_argument("--config", default="configs/motion_model.json")
    p.add_argument("--audio", required=True)
    p.add_argument("--slots_json", default=None)
    p.add_argument("--slot_seconds", type=float, default=4.0)
    p.add_argument("--db", required=True)
    p.add_argument("--refiner", default=None)
    p.add_argument("--refiner_composite_model", default=None)
    p.add_argument("--refiner_composite_contract", default=None)
    p.add_argument("--diffusion", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--render_output", default=None)
    p.add_argument("--render_script", default="rendering/render_motion.py")
    p.add_argument(
        "--repairability-mode",
        choices=["off", "shadow", "rank"],
        default=None,
        help="off preserves production; shadow records scores; rank reorders only pre-safe candidates",
    )
    p.add_argument("--repairability-checkpoint", default=None)
    p.add_argument(
        "--outcome-bank",
        default=None,
        help="append-only JSONL path; enabling this runs full multi-seed counterfactual generation",
    )
    p.add_argument(
        "--outcome-seeds",
        default=None,
        help="comma-separated common random seeds (at least two)",
    )
    p.add_argument("--outcome-topk", type=int, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "generate":
        return generate_closed_loop(args)
    raise RuntimeError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
