#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.50 event-level heading contract for EDGE 151D.

This module separates three quantities that must not be conflated:

1. local pose semantics inside a motion event;
2. relative heading change carried by the event;
3. whole-song stage-facing state maintained by the planner.

Expected EDGE representation:
    contacts:          x[:, 0:4]
    root translation:  x[:, 4:7] = X, Y, Z
    local Rot6D:       x[:, 7:151].reshape(T, 24, 6)

Rot6D follows the repository's column-concatenation convention.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EDGE_DIM = 151
NUM_JOINTS = 24
ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX = 4, 5, 6
ROT6D_START, ROT6D_END = 7, 151


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def wrap_angle(x: Any) -> Any:
    return np.arctan2(np.sin(x), np.cos(x))


def angle_diff(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    """Column-concatenated Rot6D -> SO(3)."""
    r = np.asarray(x, dtype=np.float32)
    if r.shape[-1] != 6:
        raise ValueError(f"Rot6D expected last dimension 6, got {r.shape}")
    a1 = r[..., 0:3]
    a2 = r[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def matrix_to_rot6d_np(r: np.ndarray) -> np.ndarray:
    m = np.asarray(r, dtype=np.float32)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrix expected [...,3,3], got {m.shape}")
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1).astype(np.float32)


def yaw_matrix_np(yaw: np.ndarray | float) -> np.ndarray:
    y = np.asarray(yaw, dtype=np.float32)
    c = np.cos(y)
    s = np.sin(y)
    out = np.zeros(y.shape + (3, 3), dtype=np.float32)
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def root_yaw_np(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < EDGE_DIM:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    r6 = x[:, ROT6D_START:ROT6D_START + 6]
    r = rot6d_to_matrix_np(r6)
    forward = r[..., :, 2]
    return np.arctan2(forward[:, 0], forward[:, 2]).astype(np.float32)


def unwrap_root_yaw_np(motion: np.ndarray) -> np.ndarray:
    return np.unwrap(root_yaw_np(motion)).astype(np.float32)


def rotate_motion_constant_yaw_np(
    motion: np.ndarray,
    yaw_delta: float,
    pivot_xz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rigid world-space yaw around an XZ pivot."""
    x = np.asarray(motion, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] < EDGE_DIM or len(x) == 0:
        return x
    d = float(yaw_delta)
    if abs(d) < 1e-10:
        return x

    if pivot_xz is None:
        pivot = x[0, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    else:
        pivot = np.asarray(pivot_xz, dtype=np.float32).reshape(2)

    c, s = float(math.cos(d)), float(math.sin(d))
    rx = x[:, ROOT_X_IDX] - float(pivot[0])
    rz = x[:, ROOT_Z_IDX] - float(pivot[1])
    x[:, ROOT_X_IDX] = c * rx + s * rz + float(pivot[0])
    x[:, ROOT_Z_IDX] = -s * rx + c * rz + float(pivot[1])

    root = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_START + 6])
    ry = yaw_matrix_np(np.asarray(d, dtype=np.float32))
    root = np.matmul(ry[None], root)
    x[:, ROT6D_START:ROT6D_START + 6] = matrix_to_rot6d_np(root)
    return x.astype(np.float32)


def apply_framewise_yaw_delta_np(
    motion: np.ndarray,
    yaw_delta: np.ndarray,
    pivot_xz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a smooth frame-wise world yaw correction.

    The root trajectory is rotated about the event-entry pivot and the target
    root rotation is left-multiplied by the same frame-wise yaw.
    """
    x = np.asarray(motion, dtype=np.float32).copy()
    d = np.asarray(yaw_delta, dtype=np.float32).reshape(-1)
    if len(x) != len(d):
        raise ValueError(f"motion/yaw length mismatch: {len(x)} vs {len(d)}")
    if len(x) == 0:
        return x
    pivot = (
        x[0, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
        if pivot_xz is None
        else np.asarray(pivot_xz, dtype=np.float32).reshape(2)
    )
    c, s = np.cos(d), np.sin(d)
    rx = x[:, ROOT_X_IDX].copy() - float(pivot[0])
    rz = x[:, ROOT_Z_IDX].copy() - float(pivot[1])
    x[:, ROOT_X_IDX] = c * rx + s * rz + float(pivot[0])
    x[:, ROOT_Z_IDX] = -s * rx + c * rz + float(pivot[1])

    root = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_START + 6])
    root = np.matmul(yaw_matrix_np(d), root)
    x[:, ROT6D_START:ROT6D_START + 6] = matrix_to_rot6d_np(root)
    return x.astype(np.float32)


def moving_average(x: np.ndarray, size: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    size = max(1, int(size))
    if size <= 1 or len(a) <= 2:
        return a.copy()
    if size % 2 == 0:
        size += 1
    pad = size // 2
    ap = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    kernel = np.ones(size, dtype=np.float32) / float(size)
    if a.ndim == 1:
        return np.convolve(ap, kernel, mode="valid").astype(np.float32)
    out = np.empty_like(a, dtype=np.float32)
    for j in range(a.shape[1]):
        out[:, j] = np.convolve(ap[:, j], kernel, mode="valid")
    return out


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _robust_scale(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    p10, p90 = np.percentile(a, [10, 90]) if a.size else (0.0, 1.0)
    return np.clip((a - p10) / max(float(p90 - p10), 1e-6), 0.0, 1.0)


def local_rotation_energy_np(motion: np.ndarray, fps: float = 30.0) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    r = rot6d_to_matrix_np(
        x[:, ROT6D_START:ROT6D_END].reshape(len(x), NUM_JOINTS, 6)
    )
    if len(r) <= 1:
        return np.zeros(len(r), dtype=np.float32)
    rel = np.matmul(np.swapaxes(r[:-1], -1, -2), r[1:])
    trace = np.trace(rel, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    # Exclude root so heading does not masquerade as local artistic activity.
    e = np.mean(np.abs(angle[:, 1:]), axis=1) * float(fps)
    return np.concatenate([[e[0]], e]).astype(np.float32)


def heading_metrics_np(motion: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
    x = np.asarray(motion, dtype=np.float32)
    if len(x) == 0:
        return {
            "frames": 0,
            "duration_seconds": 0.0,
            "entry_heading_rad": 0.0,
            "exit_heading_rad": 0.0,
            "net_yaw_rad": 0.0,
            "net_yaw_deg": 0.0,
            "absolute_yaw_rad": 0.0,
            "absolute_yaw_deg": 0.0,
            "monotonicity": 0.0,
            "yaw_speed_deg_s_p50": 0.0,
            "yaw_speed_deg_s_p95": 0.0,
            "yaw_speed_deg_s_max": 0.0,
            "longest_same_sign_turn_seconds": 0.0,
            "mechanical_spin_ratio": 0.0,
        }

    yaw = unwrap_root_yaw_np(x)
    speed = np.gradient(yaw) * float(fps) if len(yaw) > 1 else np.zeros_like(yaw)
    speed_deg = np.degrees(speed)
    abs_total = float(np.sum(np.abs(np.diff(yaw)))) if len(yaw) > 1 else 0.0
    net = float(yaw[-1] - yaw[0])
    monotonicity = float(abs(net) / max(abs_total, 1e-8))

    sm = moving_average(speed_deg, max(3, int(round(0.6 * fps))))
    active = np.abs(sm) >= env_float("V46_50_MECHANICAL_MIN_SPEED_DEG_S", 7.0)
    same_sign = np.sign(sm)
    longest = 0
    mechanical = np.zeros(len(sm), dtype=bool)
    for a, b in contiguous_regions(active):
        if b <= a:
            continue
        seg = sm[a:b]
        pos = float(np.mean(seg > 0))
        neg = float(np.mean(seg < 0))
        consistency = max(pos, neg)
        variation = float(np.std(seg))
        mean_abs = float(np.mean(np.abs(seg)))
        if consistency >= 0.80 and variation <= max(6.0, 0.35 * mean_abs):
            mechanical[a:b] = True
            longest = max(longest, b - a)

    return {
        "frames": int(len(x)),
        "duration_seconds": float(len(x) / max(float(fps), 1e-8)),
        "entry_heading_rad": float(yaw[0]),
        "exit_heading_rad": float(yaw[-1]),
        "net_yaw_rad": net,
        "net_yaw_deg": float(np.degrees(net)),
        "absolute_yaw_rad": abs_total,
        "absolute_yaw_deg": float(np.degrees(abs_total)),
        "monotonicity": monotonicity,
        "yaw_speed_deg_s_p50": float(np.percentile(np.abs(speed_deg), 50)),
        "yaw_speed_deg_s_p95": float(np.percentile(np.abs(speed_deg), 95)),
        "yaw_speed_deg_s_max": float(np.max(np.abs(speed_deg))),
        "longest_same_sign_turn_seconds": float(longest / max(float(fps), 1e-8)),
        "mechanical_spin_ratio": float(np.mean(mechanical)),
    }


def canonicalize_event_entry_heading_np(
    motion: np.ndarray,
    fps: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x = np.asarray(motion, dtype=np.float32).copy()
    if len(x) == 0:
        return x, {"entry_heading_before_rad": 0.0, "entry_heading_after_rad": 0.0}
    n = max(1, min(len(x), int(round(env_float("V46_50_ENTRY_HEADING_MEDIAN_SECONDS", 0.15) * fps))))
    y = root_yaw_np(x[:n])
    # Circular mean is robust to +/-pi wrapping.
    entry = float(math.atan2(float(np.mean(np.sin(y))), float(np.mean(np.cos(y)))))
    pivot = x[0, [ROOT_X_IDX, ROOT_Z_IDX]].copy()
    x = rotate_motion_constant_yaw_np(x, -entry, pivot_xz=pivot)
    x[:, ROOT_X_IDX] -= float(x[0, ROOT_X_IDX])
    x[:, ROOT_Z_IDX] -= float(x[0, ROOT_Z_IDX])
    after = float(root_yaw_np(x[:1])[0])
    return x.astype(np.float32), {
        "entry_heading_before_rad": entry,
        "entry_heading_before_deg": float(np.degrees(entry)),
        "entry_heading_after_rad": after,
        "entry_heading_after_deg": float(np.degrees(after)),
    }


def semantic_turn_strength(meta: Mapping[str, Any]) -> float:
    text = " ".join(
        str(meta.get(k, "")).lower()
        for k in (
            "dance_key",
            "dance_category",
            "event_family",
            "spatial_label",
            "music_alignment_label",
            "semantic_role",
            "locomotion_label",
            "label",
        )
    )
    score = 0.0
    if any(k in text for k in ("sogdian", "whirl", "turning_flow", "explicit_spin")):
        score = max(score, 1.0)
    if any(k in text for k in ("turning", "turn", "spin", "turning_climax")):
        score = max(score, 0.85)
    if any(k in text for k in ("aerial_curve", "flying_apsaras", "traveling")):
        score = max(score, 0.35)
    if any(k in text for k in ("pose_hold", "thirty_six", "meditation", "instrument_motif")):
        score = min(score, 0.20)
    return float(score)


def default_yaw_budget_deg(meta: Mapping[str, Any], intent: str) -> float:
    if intent == "explicit_spin":
        return env_float("V46_50_BUDGET_EXPLICIT_SPIN_DEG", 540.0)
    if intent == "turn":
        return env_float("V46_50_BUDGET_TURN_DEG", 360.0)

    text = " ".join(str(meta.get(k, "")).lower() for k in meta.keys())
    if any(k in text for k in ("pose_hold", "pose_motif", "calm_flow", "meditation", "thirty_six")):
        return env_float("V46_50_BUDGET_POSE_CALM_DEG", 30.0)
    if any(k in text for k in ("instrument", "pipa", "upper_body_phrase")):
        return env_float("V46_50_BUDGET_INSTRUMENT_DEG", 45.0)
    if any(k in text for k in ("footwork", "lotus", "locomotion", "traveling_steps")):
        return env_float("V46_50_BUDGET_FOOTWORK_DEG", 90.0)
    if any(k in text for k in ("aerial", "flying", "transition")):
        return env_float("V46_50_BUDGET_AERIAL_TRANSITION_DEG", 120.0)
    return env_float("V46_50_BUDGET_DEFAULT_DEG", 60.0)


def infer_turn_intent(
    motion: np.ndarray,
    meta: Mapping[str, Any],
    fps: float = 30.0,
) -> Dict[str, Any]:
    metrics = heading_metrics_np(motion, fps=fps)
    strength = semantic_turn_strength(meta)
    net = abs(float(metrics["net_yaw_deg"]))
    absolute = float(metrics["absolute_yaw_deg"])
    monotonic = float(metrics["monotonicity"])
    p95 = float(metrics["yaw_speed_deg_s_p95"])
    duration = float(metrics["duration_seconds"])

    local_e = local_rotation_energy_np(motion, fps=fps)
    local_p95 = float(np.percentile(local_e, 95)) if local_e.size else 0.0
    root_dominance = float(
        np.radians(p95) / max(local_p95, 1e-5)
    )

    if strength >= 0.80 and (net >= 270.0 or absolute >= 360.0):
        intent = "explicit_spin"
        confidence = min(1.0, 0.55 + 0.35 * strength + 0.10 * monotonic)
    elif strength >= 0.55 and (net >= 45.0 or p95 >= 25.0):
        intent = "turn"
        confidence = min(1.0, 0.55 + 0.30 * strength + 0.15 * monotonic)
    else:
        provisional_budget = default_yaw_budget_deg(meta, "none")
        reset_like = bool(
            duration >= env_float("V46_50_RESET_MIN_SECONDS", 1.5)
            and net > provisional_budget * env_float("V46_50_RESET_BUDGET_MULTIPLIER", 1.20)
            and monotonic >= env_float("V46_50_RESET_MONOTONICITY_MIN", 0.62)
            and strength < 0.45
        )
        if reset_like:
            intent = "reset_or_drift"
            confidence = min(1.0, 0.55 + 0.25 * monotonic + 0.20 * min(1.0, net / 360.0))
        elif net >= 45.0 and p95 >= 20.0:
            intent = "uncertain_turn"
            confidence = 0.45 + 0.25 * monotonic
        else:
            intent = "none"
            confidence = min(1.0, 0.65 + 0.25 * max(0.0, 1.0 - net / max(provisional_budget, 1.0)))

    return {
        "intent": intent,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "semantic_turn_strength": strength,
        "local_rotation_energy_p95_rad_s": local_p95,
        "root_rotation_dominance": root_dominance,
        "metrics": metrics,
    }


def _scaled_heading_curve(yaw: np.ndarray, budget_rad: float) -> np.ndarray:
    rel = np.asarray(yaw, dtype=np.float32) - float(yaw[0])
    net = float(rel[-1]) if len(rel) else 0.0
    if abs(net) <= budget_rad + 1e-8:
        return rel
    scale = float(budget_rad / max(abs(net), 1e-8))
    # Smooth endpoint-preserving scaling. This keeps internal turn timing while
    # reducing the accumulated stage-facing change.
    return (rel * scale).astype(np.float32)


def enforce_event_heading_contract(
    motion: np.ndarray,
    meta: Mapping[str, Any],
    fps: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Canonicalize event entry and enforce semantic yaw budget.

    reset_or_drift events are excluded by default rather than cosmetically
    repaired, because they are likely capture reset or non-semantic rotation.
    """
    x0 = np.asarray(motion, dtype=np.float32)
    canonical, entry_report = canonicalize_event_entry_heading_np(x0, fps=fps)
    inferred = infer_turn_intent(canonical, meta, fps=fps)
    intent = str(inferred["intent"])
    budget_deg = float(default_yaw_budget_deg(meta, intent))
    budget_rad = float(np.radians(budget_deg))
    before = heading_metrics_np(canonical, fps=fps)

    valid = True
    reason = "within_budget"
    corrected = canonical.copy()
    excess_ratio = abs(float(before["net_yaw_rad"])) / max(budget_rad, 1e-8)

    if intent == "reset_or_drift" and env_bool("V46_50_DROP_RESET_OR_DRIFT", True):
        valid = False
        reason = "drop_reset_or_drift"
    elif excess_ratio > 1.0:
        hard_mult = env_float("V46_50_NON_TURN_HARD_EXCESS_MULTIPLIER", 1.50)
        if intent in {"none", "uncertain_turn"} and excess_ratio > hard_mult:
            valid = False
            reason = "non_turn_yaw_exceeds_hard_budget"
        else:
            yaw = unwrap_root_yaw_np(canonical)
            desired = _scaled_heading_curve(yaw, budget_rad)
            delta = desired - (yaw - float(yaw[0]))
            corrected = apply_framewise_yaw_delta_np(
                canonical,
                delta,
                pivot_xz=canonical[0, [ROOT_X_IDX, ROOT_Z_IDX]],
            )
            reason = "smooth_relative_yaw_projection"

    after = heading_metrics_np(corrected, fps=fps)
    budget_violation_deg = max(0.0, abs(float(after["net_yaw_deg"])) - budget_deg)
    quality = float(np.exp(-max(0.0, excess_ratio - 1.0)))
    if intent == "reset_or_drift":
        quality *= 0.25
    if budget_violation_deg > 2.0:
        quality *= 0.5

    report = {
        "schema": "v46_50_event_heading_contract",
        "valid": bool(valid),
        "reason": reason,
        "intent": intent,
        "turn_confidence": float(inferred["confidence"]),
        "semantic_turn_strength": float(inferred["semantic_turn_strength"]),
        "yaw_budget_deg": budget_deg,
        "yaw_budget_rad": budget_rad,
        "heading_quality": float(np.clip(quality, 0.0, 1.0)),
        "entry": entry_report,
        "before_budget": before,
        "after_budget": after,
        "budget_violation_deg": float(budget_violation_deg),
        "local_rotation_energy_p95_rad_s": float(
            inferred["local_rotation_energy_p95_rad_s"]
        ),
        "root_rotation_dominance": float(inferred["root_rotation_dominance"]),
    }
    return corrected.astype(np.float32), report


def _semantic_duration_range(meta: Mapping[str, Any]) -> Tuple[float, float]:
    raw = meta.get("natural_duration_range_sec")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return max(0.5, float(raw[0])), max(float(raw[0]), float(raw[-1]))

    text = " ".join(str(meta.get(k, "")).lower() for k in meta.keys())
    if "thirty_six" in text or "pose_motif" in text:
        return 1.2, 3.8
    if "meditation" in text or "calm_flow" in text:
        return 2.0, 6.0
    if "sogdian" in text or "turning_flow" in text:
        return 1.6, 4.5
    if "lotus" in text or "footwork_flow" in text:
        return 1.5, 4.0
    if "drum" in text or "percussive" in text:
        return 1.2, 3.5
    return 1.5, 4.5


def adaptive_event_segments(
    motion: np.ndarray,
    meta: Mapping[str, Any],
    fps: float = 30.0,
    min_event_frames: Optional[int] = None,
    max_event_frames: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Motion-adaptive event segmentation.

    Boundaries prefer local energy/yaw valleys and contact-stable frames. Turn
    interval edges are promoted so a spin is not mixed with a following pose.
    """
    x = np.asarray(motion, dtype=np.float32)
    T = len(x)
    min_sec, max_sec = _semantic_duration_range(meta)
    semantic_min_frames = max(
        env_int("V46_50_MIN_EVENT_FRAMES", int(round(min_sec * fps))),
        int(round(0.8 * fps)),
    )
    semantic_max_frames = min(
        env_int("V46_50_MAX_EVENT_FRAMES", int(round(max_sec * fps))),
        int(round(env_float("V46_50_GLOBAL_MAX_EVENT_SECONDS", 6.0) * fps)),
    )
    # The project-level motion contract is authoritative.  The semantic range
    # only chooses a preferred duration inside those reproducible bounds.
    hard_min = (
        max(1, int(min_event_frames))
        if min_event_frames is not None
        else max(1, semantic_min_frames)
    )
    hard_max = (
        max(2, int(max_event_frames))
        if max_event_frames is not None
        else max(hard_min + 1, semantic_max_frames)
    )
    if hard_max <= hard_min:
        raise ValueError(
            f"Invalid event duration contract: min={hard_min}, max={hard_max}"
        )
    min_frames = int(np.clip(semantic_min_frames, hard_min, hard_max - 1))
    max_frames = int(np.clip(semantic_max_frames, min_frames + 1, hard_max))
    preferred = int(round(0.5 * (min_frames + max_frames)))

    if T <= max_frames:
        return [(0, T)], {
            "schema": "v46_50_motion_adaptive_segmentation",
            "frames": T,
            "segments": 1,
            "min_frames": min_frames,
            "max_frames": max_frames,
            "reason": "short_sequence",
        }

    local_e = moving_average(
        local_rotation_energy_np(x, fps=fps),
        max(3, int(round(0.20 * fps))),
    )
    root = x[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    root_speed = np.zeros(T, dtype=np.float32)
    if T > 1:
        root_speed[1:] = np.linalg.norm(root[1:] - root[:-1], axis=-1) * float(fps)
        root_speed[0] = root_speed[1]
    yaw = unwrap_root_yaw_np(x)
    yaw_speed = np.abs(np.gradient(yaw) * fps)
    yaw_speed = moving_average(yaw_speed, max(3, int(round(0.25 * fps))))

    contact = np.asarray(x[:, 0:4], dtype=np.float32)
    contact_change = np.zeros(T, dtype=np.float32)
    if T > 1:
        contact_change[1:] = np.mean(np.abs(contact[1:] - contact[:-1]), axis=1)

    boundary_score = (
        0.48 * (1.0 - _robust_scale(local_e))
        + 0.22 * (1.0 - _robust_scale(root_speed))
        + 0.22 * (1.0 - _robust_scale(yaw_speed))
        + 0.08 * (1.0 - _robust_scale(contact_change))
    ).astype(np.float32)

    turn_thr = np.radians(env_float("V46_50_TURN_INTERVAL_SPEED_DEG_S", 18.0))
    turn_mask = yaw_speed >= turn_thr
    # Close small holes and remove tiny bursts without scipy.
    gap = max(1, int(round(0.20 * fps)))
    regions = contiguous_regions(turn_mask)
    for (a, b), (c, d) in zip(regions[:-1], regions[1:]):
        if c - b <= gap:
            turn_mask[b:c] = True
    min_turn = max(2, int(round(env_float("V46_50_MIN_TURN_INTERVAL_SECONDS", 0.40) * fps)))
    for a, b in contiguous_regions(turn_mask):
        if b - a < min_turn:
            turn_mask[a:b] = False
    turn_edges = set()
    for a, b in contiguous_regions(turn_mask):
        turn_edges.add(int(a))
        turn_edges.add(int(b))

    segments: List[Tuple[int, int]] = []
    cursor = 0
    while T - cursor > max_frames:
        lo = cursor + min_frames
        hi = min(T - min_frames, cursor + max_frames)
        if hi <= lo:
            break
        desired = min(hi, cursor + preferred)
        ids = np.arange(lo, hi + 1, dtype=np.int64)
        length_term = np.exp(
            -0.5 * ((ids - desired) / max(0.25 * preferred, 1.0)) ** 2
        )
        scores = boundary_score[ids] + 0.20 * length_term.astype(np.float32)
        for j, idx in enumerate(ids):
            if int(idx) in turn_edges:
                scores[j] += env_float("V46_50_TURN_EDGE_BOUNDARY_BONUS", 0.45)
            # Cutting through an active spin is undesirable.
            if 0 <= int(idx) < T and turn_mask[int(idx)]:
                scores[j] -= env_float("V46_50_CUT_THROUGH_TURN_PENALTY", 0.55)
        end = int(ids[int(np.argmax(scores))])
        if end <= cursor:
            end = min(T, cursor + max_frames)
        segments.append((cursor, end))
        cursor = end

    if cursor < T:
        if segments and T - cursor < min_frames:
            a, _ = segments[-1]
            total = T - a
            if total <= max_frames:
                segments[-1] = (a, T)
            else:
                # Rebalance the tail instead of merging a short remainder into
                # an over-long event.  Both final events remain within the
                # configured project-level duration contract.
                split = T - min_frames
                segments[-1] = (a, split)
                segments.append((split, T))
        else:
            segments.append((cursor, T))

    # Safety: no overlap, no gaps, no empty ranges.
    clean: List[Tuple[int, int]] = []
    last = 0
    for a, b in segments:
        a = max(last, int(a))
        b = min(T, int(b))
        if b > a:
            clean.append((a, b))
            last = b
    if not clean or clean[-1][1] != T:
        clean.append((last, T))

    return clean, {
        "schema": "v46_50_motion_adaptive_segmentation",
        "frames": int(T),
        "segments": int(len(clean)),
        "min_frames": int(min_frames),
        "max_frames": int(max_frames),
        "preferred_frames": int(preferred),
        "turn_intervals": [[int(a), int(b)] for a, b in contiguous_regions(turn_mask)],
        "boundary_score_p50": float(np.percentile(boundary_score, 50)),
        "boundary_score_p95": float(np.percentile(boundary_score, 95)),
    }


def slot_turn_policy(slot: Mapping[str, Any]) -> Dict[str, Any]:
    text = " ".join(
        str(slot.get(k, "")).lower()
        for k in (
            "role",
            "slot_role",
            "music_alignment_label",
            "music_semantic_top_label",
            "energy_label",
            "rhythm_label",
        )
    )
    if any(k in text for k in ("turning_climax", "turn", "spin")):
        intent = "turn"
        allow_spin = True
    elif any(k in text for k in ("climax", "accent", "percussive")):
        intent = "turn_allowed"
        allow_spin = True
    elif any(k in text for k in ("calm", "pose", "meditative", "resolution", "release", "intro")):
        intent = "non_turn_anchor"
        allow_spin = False
    else:
        intent = "neutral"
        allow_spin = False
    return {
        "slot_turn_intent": intent,
        "allow_explicit_spin": allow_spin,
        "front_anchor": intent == "non_turn_anchor",
    }


def candidate_heading_penalty(
    event_meta: Mapping[str, Any],
    slot: Mapping[str, Any],
    stage_heading_rad: float,
    recent_turn_count: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    policy = slot_turn_policy(slot)
    intent = str(event_meta.get("event_turn_intent", event_meta.get("turn_intent", "none")))
    delta = float(event_meta.get("event_stage_delta_yaw_rad", event_meta.get("net_yaw_rad", 0.0)))
    valid = bool(event_meta.get("event_heading_valid", True))
    quality = float(event_meta.get("event_heading_quality", 1.0))

    hard_reject = not valid
    mismatch = 0.0
    if policy["slot_turn_intent"] == "non_turn_anchor":
        if intent == "explicit_spin":
            hard_reject = True
            mismatch += 20.0
        elif intent in {"turn", "uncertain_turn"}:
            mismatch += 4.0
    elif policy["slot_turn_intent"] == "turn":
        if intent == "none":
            mismatch += 2.0
        elif intent == "explicit_spin":
            mismatch -= 0.5
    elif policy["slot_turn_intent"] == "turn_allowed":
        if intent == "explicit_spin":
            mismatch += 0.2
    else:
        if intent == "explicit_spin":
            mismatch += 2.5

    future = float(wrap_angle(stage_heading_rad + delta))
    anchor_penalty = 0.0
    if policy["front_anchor"]:
        anchor_penalty = abs(float(wrap_angle(future))) / math.pi

    repeat_penalty = 0.0
    if intent in {"turn", "explicit_spin", "uncertain_turn"}:
        repeat_penalty = env_float("V46_50_CONSECUTIVE_TURN_PENALTY", 0.75) * max(0, recent_turn_count)

    quality_penalty = max(0.0, 1.0 - quality)
    total = (
        env_float("V46_50_INTENT_MISMATCH_WEIGHT", 1.0) * mismatch
        + env_float("V46_50_STAGE_FRONT_ANCHOR_WEIGHT", 1.25) * anchor_penalty
        + repeat_penalty
        + env_float("V46_50_HEADING_QUALITY_WEIGHT", 0.75) * quality_penalty
    )
    return float(total), {
        "slot_policy": policy,
        "event_intent": intent,
        "event_delta_yaw_rad": delta,
        "event_delta_yaw_deg": float(np.degrees(delta)),
        "stage_heading_before_rad": float(stage_heading_rad),
        "stage_heading_after_rad": future,
        "stage_heading_after_deg": float(np.degrees(future)),
        "intent_mismatch_penalty": float(mismatch),
        "front_anchor_penalty": float(anchor_penalty),
        "consecutive_turn_penalty": float(repeat_penalty),
        "heading_quality_penalty": float(quality_penalty),
        "hard_reject": bool(hard_reject),
        "total_heading_penalty": float(total),
    }


def restore_planned_root_heading_np(
    generated: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Restore frame-wise planned root yaw while preserving generated tilt/roll."""
    out = np.asarray(generated, dtype=np.float32).copy()
    ref = np.asarray(reference, dtype=np.float32)
    if out.shape != ref.shape or out.ndim != 2 or out.shape[1] < EDGE_DIM:
        return out, {
            "enabled": False,
            "reason": f"shape_mismatch generated={out.shape} reference={ref.shape}",
        }
    gy = root_yaw_np(out)
    ry = root_yaw_np(ref)
    delta = wrap_angle(ry - gy).astype(np.float32)
    root = rot6d_to_matrix_np(out[:, ROT6D_START:ROT6D_START + 6])
    root = np.matmul(yaw_matrix_np(delta), root)
    out[:, ROT6D_START:ROT6D_START + 6] = matrix_to_rot6d_np(root)

    after = wrap_angle(root_yaw_np(out) - ry)
    return out.astype(np.float32), {
        "enabled": True,
        "schema": "v46_50_planned_root_heading_guard",
        "yaw_error_before_deg_p50": float(np.percentile(np.abs(np.degrees(wrap_angle(gy - ry))), 50)),
        "yaw_error_before_deg_p95": float(np.percentile(np.abs(np.degrees(wrap_angle(gy - ry))), 95)),
        "yaw_error_after_deg_p95": float(np.percentile(np.abs(np.degrees(after)), 95)),
        "correction_deg_p95": float(np.percentile(np.abs(np.degrees(delta)), 95)),
        "correction_deg_max": float(np.max(np.abs(np.degrees(delta)))),
    }


def event_meta_from_db(db: Mapping[str, Any], event_id: int) -> Dict[str, Any]:
    i = int(event_id)
    out: Dict[str, Any] = {}
    mapping = {
        "event_turn_intent": "event_turn_intents",
        "event_turn_confidence": "event_turn_confidence",
        "event_yaw_budget_rad": "event_yaw_budget_rad",
        "event_heading_quality": "event_heading_quality",
        "event_heading_valid": "event_heading_valid",
        "event_stage_delta_yaw_rad": "event_stage_delta_yaw_rad",
        "event_net_yaw_rad": "event_net_yaw_rad",
        "event_abs_yaw_rad": "event_abs_yaw_rad",
        "dance_key": "dance_keys",
        "event_family": "event_families",
        "spatial_label": "spatial_labels",
        "music_alignment_label": "music_alignment_labels",
        "motion_stage_role": "motion_stage_roles",
        "source_uid": "source_uids",
    }
    for dst, src in mapping.items():
        try:
            value = np.asarray(db[src], dtype=object)[i]
            if isinstance(value, np.generic):
                value = value.item()
            out[dst] = value
        except Exception:
            pass
    return out
