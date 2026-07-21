#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.46 Boundary-Simulated Closed-Loop Scheduler for EDGE 151D
=============================================================

This file is an additive research patch for the histry/EDGE V46.x codebase.
It does not replace training/motion_models.py.  Instead, it imports the
latest V46.38/V46.41/V46.45 functions and wraps them into a closed-loop
boundary-safe generation pipeline.

Research objective
------------------
Upgrade the existing patch-stack style two-layer safety idea into a closed-loop
boundary safety scheduler:

    search-time candidate ranking
        -> lightweight simulated stitching risk
        -> risk-adaptive transition budget
        -> real stitching
        -> V32/V34-style cross-boundary risk audit
        -> candidate reselection / repair / rollback
        -> unified boundary-level audit table

Key properties
--------------
1. Search risk is no longer only metadata-level.  For top-k candidates, the
   scheduler quickly simulates yaw/XZ alignment + root-Hermite / rotation-SLERP
   transition and evaluates a lightweight V32-style risk.
2. Transition length is adapted by pose/yaw/contact/FK risk, not only target
   duration.
3. Unsafe boundaries can trigger local candidate reselection before relying on
   refiner/diffusion/IK.
4. All predicted and actual boundary signals are exported as JSON and CSV for
   paper tables.

Expected location
-----------------
Copy this file to:
    <EDGE_ROOT>/routing/boundary_closed_loop.py

Run from EDGE root:
    python routing/boundary_closed_loop.py generate \
        --config configs/motion_model.json \
        --audio dunhuangwu2.wav \
        --db output/.../db \
        --contrastive output/.../v44.pt \
        --refiner output/.../v45.pt \
        --diffusion output/.../v46.pt \
        --out output/.../dunhuangwu2_v46_46_closed_loop.npy \
        --json output/.../dunhuangwu2_v46_46_closed_loop.report.json
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from motion_geometry.resampling import resample_edge151_np
from support.common import make_geodesic_transition
from support.event_identity import (
    assert_same_event_db_contract,
    event_uids_from_generation_db,
    make_event_db_contract,
    normalize_event_db_contract,
)


EDGE_DIM = 151
CONTACT = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT6D_START = 7
ROT6D_END = 151
NUM_JOINTS = 24
DEFAULT_FOOT_JOINTS = (7, 8, 10, 11)


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


def import_v46():
    return importlib.import_module("training.motion_models")


def import_v32_transition_risk():
    try:
        mod = importlib.import_module("support.transition_quality")
        return mod.transition_risk
    except Exception:
        return None


def _as_motion_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] < EDGE_DIM:
        raise ValueError(f"Expected EDGE motion [T,151], got {arr.shape}")
    return arr[:, :EDGE_DIM].astype(np.float32)


def enforce_contract(v46, motion: np.ndarray, cfg: Any, source_hint: str) -> np.ndarray:
    x = _as_motion_array(motion)
    if hasattr(v46, "enforce_edge151_contract_np"):
        y, _ = v46.enforce_edge151_contract_np(
            x, cfg, source_hint=source_hint, derive_contact=True, project_rot=True
        )
        return _as_motion_array(y)
    return x.astype(np.float32)


def resample_motion(v46, motion: np.ndarray, target_len: int) -> np.ndarray:
    target_len = max(1, int(target_len))
    x = _as_motion_array(motion)
    if x.shape[0] == target_len:
        return x.copy().astype(np.float32)
    return _as_motion_array(resample_edge151_np(x, target_frames=target_len))


def load_event_motion(v46, path: str | Path, cfg: Any, source_hint: str) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    obj = np.load(str(p), allow_pickle=True)
    motion = _as_motion_array(obj)
    return enforce_contract(v46, motion, cfg, source_hint=source_hint)


def angle_diff(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def root_yaw(v46, motion: np.ndarray) -> np.ndarray:
    x = _as_motion_array(motion)
    if hasattr(v46, "root_yaw_np"):
        try:
            return np.asarray(v46.root_yaw_np(x), dtype=np.float32).reshape(-1)
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


def fk_positions(v46, motion: np.ndarray) -> Optional[np.ndarray]:
    x = _as_motion_array(motion)
    for name in ("fk_24_np", "motion_to_joint_positions_np"):
        fn = getattr(v46, name, None)
        if fn is not None:
            try:
                return np.asarray(fn(x), dtype=np.float32)
            except Exception:
                pass
    return None


def simple_boundary_risk(previous: np.ndarray, transition: np.ndarray, following: np.ndarray, fps: float) -> Dict[str, float]:
    ctx = np.concatenate([previous[-4:], transition, following[:4]], axis=0).astype(np.float32)
    if len(ctx) < 4:
        return {"total": 1e9, "boundary_joint_jerk_max": 1e9, "exit_fk_jump": 1e9, "exit_rotation_step_rad": 1e9, "foot_slip": 1e9, "contact_switch": 1e9}
    root = ctx[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    vel = np.diff(root, axis=0) * fps
    acc = np.diff(vel, axis=0) * fps
    jerk = np.diff(acc, axis=0) * fps if len(acc) > 1 else np.zeros((0, 3), dtype=np.float32)
    boundary_jerk = float(np.max(np.linalg.norm(jerk, axis=-1))) if jerk.size else 0.0
    left = min(4, len(previous[-4:]))
    right = left + len(transition)
    exit_jump = 0.0
    if 0 < right < len(ctx):
        exit_jump = float(np.linalg.norm(ctx[right, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - ctx[right - 1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]))
    contact = transition[:, CONTACT] if len(transition) else np.zeros((0, 4), dtype=np.float32)
    contact_switch = float(np.abs(np.diff(contact, axis=0)).mean()) if len(contact) > 1 else 0.0
    total = 0.002 * boundary_jerk + 3.0 * exit_jump + 0.25 * contact_switch
    return {
        "total": float(total),
        "boundary_joint_jerk_max": float(boundary_jerk),
        "exit_fk_jump": float(exit_jump),
        "entry_fk_jump": 0.0,
        "exit_rotation_step_rad": 0.0,
        "entry_rotation_step_rad": 0.0,
        "foot_slip": 0.0,
        "foot_penetration": 0.0,
        "contact_switch": float(contact_switch),
    }


def transition_risk(v46, previous: np.ndarray, transition: np.ndarray, following: np.ndarray, fps: float) -> Dict[str, float]:
    previous = np.asarray(previous, dtype=np.float32)
    transition = np.asarray(transition, dtype=np.float32)
    following = np.asarray(following, dtype=np.float32)

    # V46.48: v32 transition_risk may return 1e9 sentinel values when the
    # explicit bridge is empty, especially for a tiny terminal residual slot.
    # An empty bridge is a direct join and should be evaluated as such.
    if transition.shape[0] == 0:
        return simple_boundary_risk(previous, transition, following, fps)

    fn = import_v32_transition_risk()
    if fn is not None:
        try:
            risk = dict(fn(previous, transition, following, fps=fps))
            probe_keys = (
                "total",
                "boundary_joint_jerk_max",
                "exit_fk_jump",
                "exit_rotation_step_rad",
                "foot_slip",
                "foot_penetration",
            )
            values = [
                float(risk.get(k, 0.0))
                for k in probe_keys
            ]
            sentinel = any(
                (not np.isfinite(v)) or abs(v) >= 1.0e8
                for v in values
            )
            if not sentinel:
                return risk
        except Exception:
            pass

    return simple_boundary_risk(previous, transition, following, fps)


def risk_score(risk: Dict[str, Any]) -> float:
    # Normalized scalar used for search and fallback selection.  Every
    # kinematic threshold is named with its SI unit so 30/60 FPS runs share
    # exactly the same physical contract.
    bj = float(risk.get("boundary_joint_jerk_max", risk.get("joint_jerk", 0.0))) / max(env_float("V46_46_NORM_BOUNDARY_JERK_MPS3", 5000.0), 1e-6)
    fk = float(risk.get("exit_fk_jump", 0.0)) / max(env_float("V46_46_NORM_EXIT_FK_JUMP_M", 0.040), 1e-6)
    rot = float(risk.get("exit_rotation_step_rad", 0.0)) / max(env_float("V46_46_NORM_EXIT_ROT_RAD", 0.12), 1e-6)
    slip = float(risk.get("foot_slip", 0.0)) / max(env_float("V46_46_NORM_FOOT_SLIP_MPS", 0.22), 1e-6)
    cs = float(risk.get("contact_switch", 0.0)) / max(env_float("V46_46_NORM_CONTACT_SWITCH", 0.45), 1e-6)
    total = float(risk.get("total", 0.0)) / max(env_float("V46_46_NORM_TOTAL", 1.0), 1e-6)
    return float(0.30 * total + 0.24 * bj + 0.22 * fk + 0.14 * rot + 0.07 * slip + 0.03 * cs)


def risk_safe(risk: Dict[str, Any]) -> bool:
    return bool(
        float(risk.get("boundary_joint_jerk_max", 0.0)) <= env_float("V46_46_MAX_BOUNDARY_JERK_MPS3", 650.0)
        and float(risk.get("exit_fk_jump", 0.0)) <= env_float("V46_46_MAX_EXIT_FK_JUMP_M", 0.015)
        and float(risk.get("exit_rotation_step_rad", 0.0)) <= env_float("V46_46_MAX_EXIT_ROT_RAD", 0.08)
        and float(risk.get("foot_slip", 0.0)) <= env_float("V46_46_MAX_FOOT_SLIP_MPS", 0.06)
        and float(risk.get("foot_penetration", 0.0)) <= env_float("V46_46_MAX_FOOT_PENETRATION_M2", 0.001)
    )


def estimate_boundary_features(v46, prev: np.ndarray, curr: np.ndarray, cfg: Any) -> Dict[str, float]:
    p = _as_motion_array(prev)
    c = _as_motion_array(curr)
    pose_gap = float(np.linalg.norm(p[-1, ROT6D_START:ROT6D_END] - c[0, ROT6D_START:ROT6D_END]) / math.sqrt(max(1, ROT6D_END - ROT6D_START)))
    if len(p) > 1 and len(c) > 1:
        pv = p[-1, ROT6D_START:ROT6D_END] - p[-2, ROT6D_START:ROT6D_END]
        cv = c[1, ROT6D_START:ROT6D_END] - c[0, ROT6D_START:ROT6D_END]
        velocity_gap = float(np.linalg.norm(pv - cv) / math.sqrt(max(1, ROT6D_END - ROT6D_START)))
    else:
        velocity_gap = 0.0
    yaw_prev = float(root_yaw(v46, p[-1:])[0])
    yaw_curr = float(root_yaw(v46, c[:1])[0])
    yaw_gap = abs(angle_diff(yaw_prev, yaw_curr))
    contact_gap = float(np.abs(p[-1, CONTACT] - c[0, CONTACT]).mean())
    fk_gap = 0.0
    fp = fk_positions(v46, p[-1:])
    fc = fk_positions(v46, c[:1])
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


def choose_transition_lengths(v46, prev: Optional[np.ndarray], source_len: int, target_len: int, raw_curr: np.ndarray, slot: Dict[str, Any], cfg: Any) -> Tuple[int, int, Dict[str, Any]]:
    target_len = max(1, int(target_len))
    source_len = max(1, int(source_len))
    has_prev = prev is not None and len(prev) > 0
    if hasattr(v46, "_v46_33_choose_core_and_transition_lengths"):
        try:
            core_len, trans_len, info = v46._v46_33_choose_core_and_transition_lengths(source_len, target_len, has_prev, cfg)
            info = dict(info)
        except Exception:
            core_len, trans_len, info = target_len, 0, {"reason": "fallback_exception"}
    else:
        if not has_prev:
            return target_len, 0, {"reason": "first_slot_no_transition"}
        min_trans = env_int("V46_TRANSITION_MIN_FRAMES", 10)
        max_trans = env_int("V46_TRANSITION_MAX_FRAMES", 28)
        trans_len = int(round(target_len * env_float("V46_TRANSITION_RATIO", 0.18)))
        trans_len = max(min_trans, min(max_trans, trans_len))
        core_len = target_len - trans_len
        info = {"reason": "local_default", "transition_frames": trans_len, "core_frames": core_len}

    if not has_prev or not env_bool("V46_46_RISK_ADAPT_TRANSITION_ENABLE", True):
        core_len = max(1, min(int(core_len), target_len))
        trans_len = max(0, target_len - core_len)
        info.update({"risk_adaptive": False})
        return int(core_len), int(trans_len), info

    # Estimate boundary features after a rough core resample but before final transition.
    rough_core = resample_motion(v46, raw_curr, max(1, int(core_len)))
    rough_core = enforce_contract(v46, rough_core, cfg, source_hint="v46_46_transition_len_rough_core")
    # Align for a more realistic yaw/root measurement.
    aligned, align_info = align_core_to_prev(v46, prev, rough_core, cfg)
    feats = estimate_boundary_features(v46, prev, aligned, cfg)

    extra = 0.0
    extra += env_float("V46_46_TLEN_POSE_W", 10.0) * feats["pose_gap"]
    extra += env_float("V46_46_TLEN_VEL_W", 4.0) * feats["velocity_gap"]
    extra += env_float("V46_46_TLEN_YAW_W", 3.0) * min(feats["yaw_gap_rad"], math.pi)
    extra += env_float("V46_46_TLEN_CONTACT_W", 8.0) * feats["contact_gap"]
    extra += env_float("V46_46_TLEN_FK_W", 80.0) * feats["predicted_fk_gap"]

    label = str(slot.get("music_alignment_label", slot.get("music_semantic_top_label", slot.get("role", "")))).lower()
    if any(k in label for k in ["calm", "lyrical", "pose", "release", "resolution"]):
        extra += env_float("V46_46_TLEN_SMOOTH_MUSIC_BONUS", 3.0)
    if any(k in label for k in ["accent", "percussive", "climax"]):
        extra -= env_float("V46_46_TLEN_ACCENT_REDUCE", 2.0)

    min_trans = env_int("V46_TRANSITION_MIN_FRAMES", 10)
    max_trans = env_int("V46_TRANSITION_MAX_FRAMES", 36)
    min_core = env_int("V46_TRANSITION_MIN_CORE_FRAMES", 30)
    trans_len2 = int(round(float(trans_len) + np.clip(extra, -4.0, env_float("V46_46_TLEN_EXTRA_MAX", 14.0))))
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


def align_core_to_prev(v46, prev: np.ndarray, core: np.ndarray, cfg: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = _as_motion_array(prev)
    c = _as_motion_array(core)
    for name in ("_v46_33_align_core_to_prev", "align_event_core_to_prev_np"):
        fn = getattr(v46, name, None)
        if fn is not None:
            try:
                out, rep = fn(p, c, cfg)
                return enforce_contract(v46, out, cfg, source_hint=f"v46_46_align:{name}"), dict(rep or {})
            except Exception:
                pass
    out = c.copy().astype(np.float32)
    delta = p[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += float(delta[0])
    out[:, ROOT_Z_IDX] += float(delta[1])
    out = enforce_contract(v46, out, cfg, source_hint="v46_46_align:fallback_xz")
    return out, {"mode": "fallback_xz_only", "delta_xz_applied": [float(delta[0]), float(delta[1])]}


def build_bridge(v46, prev: np.ndarray, core: np.ndarray, trans_len: int, cfg: Any) -> np.ndarray:
    trans_len = int(trans_len)
    if trans_len <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    prev_tail_n = min(max(2, trans_len // 2), len(prev))
    curr_head_n = min(max(2, trans_len // 2), len(core))
    for name in ("v46_33_motion_inbetween_np", "motion_inbetween_np"):
        fn = getattr(v46, name, None)
        if fn is not None:
            try:
                bridge = fn(prev[-prev_tail_n:], core[:curr_head_n], trans_len, cfg)
                return enforce_contract(v46, bridge, cfg, source_hint=f"v46_46_bridge:{name}")
            except Exception:
                pass
    # The final fallback still has to honor the same geometry contract as the
    # primary bridge: Euclidean root translation, discrete contact, and SO(3)
    # interpolation for every joint.  Projecting a linearly blended Rot6D
    # vector is not equivalent, especially close to pi.
    bridge = make_geodesic_transition(prev, core, trans_len)
    return enforce_contract(v46, bridge, cfg, source_hint="v46_46_bridge:fallback_geodesic")


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


def build_candidate_proposal(
    v46,
    prev_motion: Optional[np.ndarray],
    event_id: int,
    event_path: str,
    slot: Dict[str, Any],
    slot_idx: int,
    candidate_rank: int,
    target_len: int,
    cfg: Any,
) -> CandidateProposal:
    raw = load_event_motion(v46, event_path, cfg, source_hint=f"v46_46_load_event:{event_id}")
    has_prev = prev_motion is not None and len(prev_motion) > 0
    core_len, trans_len, length_info = choose_transition_lengths(v46, prev_motion, raw.shape[0], target_len, raw, slot, cfg)
    core = resample_motion(v46, raw, core_len)
    core = enforce_contract(v46, core, cfg, source_hint=f"v46_46_core_resample:{event_id}")
    align_report: Dict[str, Any] = {"mode": "none"}
    bridge = np.zeros((0, EDGE_DIM), dtype=np.float32)
    if has_prev:
        core, align_report = align_core_to_prev(v46, prev_motion, core, cfg)
        bridge = build_bridge(v46, prev_motion, core, trans_len, cfg)
        risk = transition_risk(v46, prev_motion[-4:], bridge, core[:4], fps=float(getattr(cfg, "fps", 30.0)))
    else:
        risk = {"total": 0.0, "boundary_joint_jerk_max": 0.0, "exit_fk_jump": 0.0, "exit_rotation_step_rad": 0.0, "foot_slip": 0.0, "foot_penetration": 0.0, "contact_switch": 0.0}
    piece = np.concatenate([bridge, core], axis=0).astype(np.float32)
    # Guarantee exact slot length; this should almost always be a no-op.
    if piece.shape[0] != int(target_len):
        piece = resample_motion(v46, piece, int(target_len))
        piece = enforce_contract(v46, piece, cfg, source_hint=f"v46_46_slot_exact_len:{event_id}")
        # If exact-length repair changed the bridge/core split, keep the recorded split but mark it.
        length_info["slot_exact_repair_applied"] = True
        length_info["slot_exact_frames_after"] = int(piece.shape[0])
    score = risk_score(risk)
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
    topk = max(1, env_int("V46_46_RESELECT_TOPK", env_int("V46_46_CANDIDATE_TOPK", 32)))
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
    v46,
    slots: Sequence[Dict[str, Any]],
    candidate_lists: Sequence[Sequence[int]],
    db: Dict[str, Any],
    cfg: Any,
    banned: Optional[Dict[int, set]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[List[int]]]:
    paths = np.asarray(db["paths"], dtype=object)
    banned = banned or {}
    pieces: List[np.ndarray] = []
    report: List[Dict[str, Any]] = []
    selected: List[List[int]] = []
    cursor = 0
    for slot_idx, slot in enumerate(slots):
        target_len = slot_target_frames(slot, cfg)
        prev = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else None
        candidates = [int(x) for x in candidate_lists[slot_idx] if int(x) not in banned.get(slot_idx, set())]
        if not candidates:
            candidates = [int(candidate_lists[slot_idx][0])]
        proposals: List[CandidateProposal] = []
        best: Optional[CandidateProposal] = None
        selected_prop: Optional[CandidateProposal] = None
        for rank, event_id in enumerate(candidates):
            p = build_candidate_proposal(
                v46=v46,
                prev_motion=prev,
                event_id=event_id,
                event_path=str(paths[event_id]),
                slot=slot,
                slot_idx=slot_idx,
                candidate_rank=rank,
                target_len=target_len,
                cfg=cfg,
            )
            proposals.append(p)
            if best is None or p.risk_score < best.risk_score:
                best = p
            if p.safe:
                selected_prop = p
                selected_prop.decision = "accepted_first_safe" if rank == 0 else "reselected_safe"
                break
        if selected_prop is None:
            selected_prop = best
            if selected_prop is None:
                raise RuntimeError(f"No proposal for slot {slot_idx}")
            selected_prop.decision = "accepted_best_unsafe_fallback"
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
            "core_warp": float(selected_prop.core.shape[0] / max(1, load_event_motion(v46, selected_prop.event_path, cfg, "v46_46_warp_probe").shape[0])),
            "risk_predicted": selected_prop.risk,
            "risk_score_predicted": float(selected_prop.risk_score),
            "safe_predicted": bool(selected_prop.safe),
            "decision": selected_prop.decision,
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
                }
                for pp in proposals
            ],
            "version": "v46_46_boundary_simulated_closed_loop_reference",
        }
        report.append(row)
        cursor += int(piece.shape[0])
    final = np.concatenate(pieces, axis=0).astype(np.float32) if pieces else np.zeros((0, EDGE_DIM), dtype=np.float32)
    final = enforce_contract(v46, final, cfg, source_hint="v46_46_closed_loop_reference_final")
    return final, report, selected


def transition_spans_from_report(assembly_report: Sequence[Dict[str, Any]]) -> List[List[int]]:
    out: List[List[int]] = []
    for r in assembly_report:
        sp = r.get("transition_span")
        if sp is not None and len(sp) >= 2 and int(sp[1]) > int(sp[0]):
            out.append([int(sp[0]), int(sp[1])])
    return out


def make_seam_mask(v46, T: int, transition_spans: Sequence[Sequence[int]], cfg: Any) -> Tuple[np.ndarray, List[int], str]:
    def finish(raw_mask: np.ndarray, centers: List[int], policy: str):
        mask = np.asarray(raw_mask, dtype=np.float32).reshape(int(T), -1)
        max_ratio = float(np.clip(env_float("V46_54_MAX_TRANSITION_MASK_RATIO", 0.25), 0.0, 1.0))
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

    if transition_spans and hasattr(v46, "make_transition_budget_mask"):
        try:
            mask = v46.make_transition_budget_mask(T, transition_spans, cfg)
            centers = [int((int(a) + int(b)) // 2) for a, b in transition_spans]
            return finish(mask, centers, "v46_46_transition_spans")
        except Exception:
            pass
    centers = [int((int(a) + int(b)) // 2) for a, b in transition_spans]
    if hasattr(v46, "make_boundary_mask"):
        try:
            mask = v46.make_boundary_mask(T, centers, width=env_int("V46_46_FALLBACK_MASK_WIDTH", 24))
            return finish(mask, centers, "v46_46_fallback_boundary_mask")
        except Exception:
            pass
    mask = np.zeros((int(T), 1), dtype=np.float32)
    width = env_int("V46_46_FALLBACK_MASK_WIDTH", 24)
    for c in centers:
        mask[max(0, c - width):min(T, c + width), 0] = 1.0
    return finish(mask, centers, "v46_46_local_fallback_boundary_mask")


def compute_condition(slot_feat: np.ndarray, db: Dict[str, Any]) -> np.ndarray:
    cond = np.mean(np.asarray(slot_feat, dtype=np.float32), axis=0).astype(np.float32)
    try:
        mean = np.asarray(db["desc_mean"], dtype=np.float32)[0]
        std = np.asarray(db["desc_std"], dtype=np.float32)[0]
        cond = (cond - mean) / np.maximum(std, 1e-6)
    except Exception:
        pass
    return cond.astype(np.float32)


def apply_generators(v46, motion_ref: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, args: argparse.Namespace, cfg: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    stage: Dict[str, Any] = {}
    motion = motion_ref.astype(np.float32)
    stage["pre_refine_audit"] = v46.audit_motion_np(motion, cfg) if hasattr(v46, "audit_motion_np") else {}
    if bool(getattr(cfg, "refiner_enable", False)) and env_bool("V46_46_USE_REFINER", True):
        motion = v46.apply_refiner_model(motion, cond, seam_mask, getattr(args, "refiner", None), cfg)
        stage["v45_refiner_audit"] = v46.audit_motion_np(motion, cfg) if hasattr(v46, "audit_motion_np") else {}
    if bool(getattr(cfg, "diffusion_enable", False)) and env_bool("V46_46_USE_DIFFUSION", True):
        motion = v46.apply_diffusion_model(motion, cond, seam_mask, getattr(args, "diffusion", None), cfg)
        stage["v46_diffusion_audit"] = v46.audit_motion_np(motion, cfg) if hasattr(v46, "audit_motion_np") else {}
    ik_report = {"enabled": False}
    if bool(getattr(cfg, "ik_enable", False)) and env_bool("V46_46_USE_IK", True):
        motion, ik_report = v46.true_lower_body_ik(motion, cfg)
    stage["v43_true_ik"] = ik_report
    stage["final_audit"] = v46.audit_motion_np(motion, cfg) if hasattr(v46, "audit_motion_np") else {}
    stage["final_physical_gate"] = physical_quality_gate(stage["final_audit"])
    return motion.astype(np.float32), stage


def physical_quality_gate(audit: Dict[str, Any]) -> Dict[str, Any]:
    limits = {
        "foot_skate_mps_p95": env_float("V46_54_MAX_FOOT_SKATE_P95_MPS", 0.18),
        "foot_skate_mps_max": env_float("V46_54_MAX_FOOT_SKATE_MAX_MPS", 0.60),
        "foot_penetration_min_m": env_float("V46_54_MIN_FOOT_PENETRATION_M", -0.050),
        "joint_jerk_mps3_p95": env_float("V46_54_MAX_JOINT_JERK_P95_MPS3", 810.0),
        "joint_jerk_mps3_max": env_float("V46_54_MAX_JOINT_JERK_MAX_MPS3", 1620.0),
        "root_y_range_m": env_float("V46_54_MAX_ROOT_Y_RANGE_M", 0.45),
    }
    reasons: List[str] = []
    for key in (
        "foot_skate_mps_p95",
        "foot_skate_mps_max",
        "joint_jerk_mps3_p95",
        "joint_jerk_mps3_max",
        "root_y_range_m",
    ):
        if float(audit.get(key, float("inf"))) > float(limits[key]):
            reasons.append(f"{key}_too_high")
    if float(audit.get("foot_penetration_min_m", float("-inf"))) < float(
        limits["foot_penetration_min_m"]
    ):
        reasons.append("foot_penetration_too_low")
    return {"ok": not reasons, "reasons": reasons, "limits": limits, "audit": dict(audit)}


def audit_boundaries(v46, motion: np.ndarray, assembly_report: Sequence[Dict[str, Any]], cfg: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
        previous = motion[max(0, prev_end - 4):prev_end]
        bridge = motion[t0:t1]
        following = motion[c0:min(c1, c0 + 4)]
        risk = transition_risk(v46, previous, bridge, following, fps=float(getattr(cfg, "fps", 30.0)))
        safe = risk_safe(risk)
        pred = assembly_report[i].get("risk_predicted", {})
        row = {
            "slot": int(i),
            "prev_event_id": int(assembly_report[i - 1].get("event_id", -1)),
            "curr_event_id": int(assembly_report[i].get("event_id", -1)),
            "candidate_rank": int(assembly_report[i].get("candidate_rank", -1)),
            "transition_start": int(t0),
            "transition_end": int(t1),
            "content_start": int(c0),
            "predicted_risk_score": float(assembly_report[i].get("risk_score_predicted", 0.0)),
            "predicted_boundary_jerk": float(pred.get("boundary_joint_jerk_max", 0.0)) if isinstance(pred, dict) else 0.0,
            "predicted_exit_fk_jump": float(pred.get("exit_fk_jump", 0.0)) if isinstance(pred, dict) else 0.0,
            "actual_risk_score": float(risk_score(risk)),
            "actual_boundary_jerk": float(risk.get("boundary_joint_jerk_max", 0.0)),
            "actual_exit_fk_jump": float(risk.get("exit_fk_jump", 0.0)),
            "actual_exit_rotation_step_rad": float(risk.get("exit_rotation_step_rad", 0.0)),
            "actual_foot_slip": float(risk.get("foot_slip", 0.0)),
            "actual_foot_penetration": float(risk.get("foot_penetration", 0.0)),
            "actual_contact_switch": float(risk.get("contact_switch", 0.0)),
            "safe": bool(safe),
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
                "predicted_exit_fk_jump_m": row["predicted_exit_fk_jump"],
                "actual_boundary_jerk_mps3": row["actual_boundary_jerk"],
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
        "predicted_risk_score", "predicted_boundary_jerk", "predicted_exit_fk_jump",
        "actual_risk_score", "actual_boundary_jerk", "actual_exit_fk_jump",
        "actual_exit_rotation_step_rad", "actual_foot_slip", "actual_foot_penetration",
        "actual_contact_switch", "core_warp", "safe", "decision",
        "predicted_boundary_jerk_mps3", "predicted_exit_fk_jump_m",
        "actual_boundary_jerk_mps3", "actual_exit_fk_jump_m",
        "actual_foot_slip_mps", "actual_foot_penetration_m2",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def render_if_possible(
    v46,
    motion_path: str,
    audio_path: Optional[str],
    output_mp4: Optional[str],
    render_script: str = "rendering/render_motion.py",
    fps: float = 30.0,
) -> None:
    if not output_mp4:
        return
    if hasattr(v46, "render_if_possible"):
        try:
            v46.render_if_possible(
                motion_path,
                audio_path,
                output_mp4,
                render_script,
                fps=float(fps),
            )
            return
        except Exception as exc:
            print(f"[V46.46 WARN] v46.render_if_possible failed: {exc}", file=sys.stderr)
    if not audio_path or not Path(render_script).exists():
        print("[V46.46 WARN] render skipped", file=sys.stderr)
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
    candidate_topk = env_int("V46_46_CANDIDATE_TOPK", 48)
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
        "V46_46_MIN_TERMINAL_SLOT_FRAMES",
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

    merged["v46_48_terminal_tail_merge"] = {
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
        f"[V46.48 TAIL MERGE] merged terminal slot: "
        f"{tail_frames} frames -> previous slot, "
        f"new_frames={total_frames}, slots={len(out_slots)}",
        file=sys.stderr,
    )
    return out_slots, feat2.astype(np.float32)


def load_slots_and_candidates(v46, args: argparse.Namespace, cfg: Any) -> Tuple[Dict[str, Any], Any, List[Dict[str, Any]], np.ndarray, List[int], List[Dict[str, Any]], List[List[int]]]:
    db = v46.load_db(args.db)
    event_uids = event_uids_from_generation_db(db)
    db["event_uids"] = event_uids
    db_contract = make_event_db_contract(event_uids)
    strict_identity = env_bool("V46_54_REQUIRE_ALIGNED_EVENT_DB", True)
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
    contrastive = v46.load_contrastive(getattr(args, "contrastive", None), cfg)
    slots, slot_feat = v46.audio_slots(args.audio, cfg, args.slot_seconds, getattr(args, "slots_json", None))
    slots, slot_feat = merge_short_terminal_slot(slots, slot_feat, cfg)
    path_idx, retrieval_report = v46.retrieve_schedule(slots, slot_feat, db, cfg, contrastive)
    candidate_lists = extract_candidate_lists(path_idx, retrieval_report, db, cfg)
    uid_to_index = {str(uid): index for index, uid in enumerate(event_uids)}
    for slot_index, slot in enumerate(slots):
        scheduled_uid = slot.get("v26_event_uid", slot.get("event_uid"))
        if not scheduled_uid:
            if strict_identity:
                raise RuntimeError(f"Slot {slot_index} has no stable v26_event_uid")
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
    return db, contrastive, list(slots), np.asarray(slot_feat, dtype=np.float32), list(map(int, path_idx)), list(retrieval_report), candidate_lists


def generate_closed_loop(args: argparse.Namespace) -> int:
    v46 = import_v46()
    cfg = v46.V46Config.from_json(args.config).apply_env()
    if getattr(args, "music_semantic_dirs", None):
        cfg.external_music_semantic_dirs = os.pathsep.join([str(x) for x in args.music_semantic_dirs])
    if getattr(args, "external_music_semantic_cmd", None):
        cfg.external_music_semantic_cmd = str(args.external_music_semantic_cmd)
    set_cfg_runtime_knobs(cfg)

    seed = int(getattr(cfg, "seed", 1234))
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(v46, "torch") and v46.torch is not None:
        try:
            v46.torch.manual_seed(seed)
        except Exception:
            pass

    db, _contrastive, slots, slot_feat, path_idx, retrieval_report, candidate_lists = load_slots_and_candidates(v46, args, cfg)
    cond = compute_condition(slot_feat, db)

    banned: Dict[int, set] = {}
    rounds: List[Dict[str, Any]] = []
    best_payload: Optional[Dict[str, Any]] = None
    max_rounds = max(0, env_int("V46_46_MAX_RESELECT_ROUNDS", 2))
    enable_reselect = env_bool("V46_46_RESELECT_ENABLE", True)

    for round_id in range(max_rounds + 1):
        motion_ref, assembly_report, selected_pairs = assemble_closed_loop_reference(v46, slots, candidate_lists, db, cfg, banned=banned)
        transition_spans = transition_spans_from_report(assembly_report)
        seam_mask, seam_positions, mask_policy = make_seam_mask(v46, motion_ref.shape[0], transition_spans, cfg)
        motion, stage_reports = apply_generators(v46, motion_ref, cond, seam_mask, args, cfg)
        boundary_rows = audit_boundaries(v46, motion, assembly_report, cfg)
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
        worst = max(unsafe_rows, key=lambda r: float(r.get("actual_risk_score", 0.0)))
        slot = int(worst.get("slot", -1))
        curr = int(worst.get("curr_event_id", -1))
        if slot < 0 or curr < 0:
            break
        banned.setdefault(slot, set()).add(curr)
        # Stop if we have exhausted candidates for this slot.
        remaining = [x for x in candidate_lists[slot] if x not in banned.get(slot, set())]
        if not remaining:
            break

    if best_payload is None:
        raise RuntimeError("Closed-loop generation produced no payload")

    final_gate = physical_quality_gate(
        best_payload["stage_reports"].get("final_audit", {})
    )
    best_payload["stage_reports"]["final_physical_gate"] = final_gate
    if env_bool("V46_54_REQUIRE_FINAL_PHYSICAL_GATE", True) and not final_gate["ok"]:
        raise RuntimeError(
            "Final physical gate rejected generated motion: "
            + ",".join(final_gate["reasons"])
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, best_payload["motion"].astype(np.float32))
    motion_ref_path = str(out).replace(".npy", ".motion_ref.npy")
    mask_path = str(out).replace(".npy", ".transition_mask.npy")
    audit_csv_path = str(out).replace(".npy", ".boundary_audit.csv")
    audit_json_path = str(out).replace(".npy", ".boundary_audit.json")
    np.save(motion_ref_path, best_payload["motion_ref"].astype(np.float32))
    np.save(mask_path, best_payload["seam_mask"].astype(np.float32))
    write_audit_csv(best_payload["boundary_rows"], audit_csv_path)
    save_json(best_payload["boundary_rows"], audit_json_path)

    paths = np.asarray(db["paths"], dtype=object)
    selected_event_indices = [int(x[0]) for x in best_payload["selected_pairs"]]
    selected_paths = [str(paths[i]) for i in selected_event_indices]

    report = {
        "version": "v46_46_boundary_simulated_closed_loop_scheduler",
        "audio": args.audio,
        "db": args.db,
        "config": dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else jsonable(cfg),
        "selected_event_indices_initial_v46": path_idx,
        "selected_event_indices_final": selected_event_indices,
        "selected_event_paths_final": selected_paths,
        "slots": slots,
        "motion_ref_path": motion_ref_path,
        "transition_mask_path": mask_path,
        "boundary_audit_csv": audit_csv_path,
        "boundary_audit_json": audit_json_path,
        "closed_loop": {
            "enabled": True,
            "rounds": rounds,
            "final_round": int(best_payload["round"]),
            "candidate_topk": int(env_int("V46_46_CANDIDATE_TOPK", 48)),
            "reselect_topk": int(env_int("V46_46_RESELECT_TOPK", env_int("V46_46_CANDIDATE_TOPK", 32))),
            "reselect_enabled": bool(enable_reselect),
            "risk_adaptive_transition_enabled": env_bool("V46_46_RISK_ADAPT_TRANSITION_ENABLE", True),
            "simulated_edge_risk_enabled": True,
            "env": {k: v for k, v in os.environ.items() if k.startswith("V46_46_")},
            "diversity_env": {k: v for k, v in os.environ.items() if k.startswith("V46_54_")},
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
            "actual_exit_fk_jump_p95": float(np.percentile([r.get("actual_exit_fk_jump", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_foot_slip_p95": float(np.percentile([r.get("actual_foot_slip", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_boundary_jerk_p95_mps3": float(np.percentile([r.get("actual_boundary_jerk_mps3", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_exit_fk_jump_p95_m": float(np.percentile([r.get("actual_exit_fk_jump_m", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "actual_foot_slip_p95_mps": float(np.percentile([r.get("actual_foot_slip_mps", 0.0) for r in best_payload["boundary_rows"]], 95)) if best_payload["boundary_rows"] else 0.0,
            "physical_units": {
                "boundary_jerk": "m/s^3",
                "exit_fk_jump": "m",
                "exit_rotation_step": "rad/frame",
                "foot_slip": "m/s",
                "foot_penetration": "m^2_mean_squared_depth",
            },
        },
        "final_audit": best_payload["stage_reports"].get("final_audit", {}),
    }
    json_path = args.json or str(out).replace(".npy", ".v46_46_closed_loop_report.json")
    save_json(report, json_path)

    if args.render_output:
        render_if_possible(
            v46,
            str(out),
            args.audio,
            args.render_output,
            args.render_script,
            fps=float(getattr(cfg, "fps", 30.0)),
        )

    print(json.dumps(jsonable({
        "motion": str(out),
        "motion_ref": motion_ref_path,
        "transition_mask": mask_path,
        "json": json_path,
        "boundary_audit_csv": audit_csv_path,
        "frames": int(best_payload["motion"].shape[0]),
        "boundary_audit_summary": report["boundary_audit_summary"],
        "final_audit": report["final_audit"],
    }), ensure_ascii=False, indent=2))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V46.46 closed-loop boundary-safe generator for EDGE 151D")
    p.add_argument("cmd", choices=["generate"], help="subcommand")
    p.add_argument("--config", default="configs/motion_model.json")
    p.add_argument("--audio", required=True)
    p.add_argument("--slots_json", default=None)
    p.add_argument("--music_semantic_dirs", nargs="*", default=None)
    p.add_argument("--external_music_semantic_cmd", default=None)
    p.add_argument("--slot_seconds", type=float, default=4.0)
    p.add_argument("--db", required=True)
    p.add_argument("--contrastive", default=None)
    p.add_argument("--refiner", default=None)
    p.add_argument("--diffusion", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--render_output", default=None)
    p.add_argument("--render_script", default="rendering/render_motion.py")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "generate":
        return generate_closed_loop(args)
    raise RuntimeError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
