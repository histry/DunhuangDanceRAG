#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for scalable multi-music Dunhuang ChoreoRAG.

The module depends on the repository's canonical motion-geometry contract and
keeps the EDGE 151D representation:
  [0:4]   foot contacts
  [4:7]   root xyz
  [7:151] 24 joints x 6D rotations
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from motion_geometry.rotations import (
    angular_acceleration_np,
    angular_velocity_np,
    matrix_to_rot6d_np,
    relative_rotvec_np,
    rot6d_to_matrix_np,
    so3_geodesic_np,
    tangent_blend_np,
)
from motion_geometry.resampling import resample_edge151_np
from motion_geometry.heading import root_yaw_np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)
EVENT_TYPES = (
    "pose_hold",
    "calm_flow",
    "neutral_flow",
    "build_up",
    "release",
    "support_shift",
    "high_tension",
    "arm_flourish",
)
EVENT_TO_ID = {name: idx for idx, name in enumerate(EVENT_TYPES)}


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_json_items(path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data, data["items"]
    if isinstance(data, list):
        return {"items": data}, data
    raise ValueError(f"Invalid event database: {p}")


def _first_motion_in_object(obj: Any) -> np.ndarray | None:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.size == 1:
            return _first_motion_in_object(obj.reshape(-1)[0])
        arr = np.asarray(obj)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2 and arr.shape[-1] == 151:
            return arr.astype(np.float32)
        return None
    if isinstance(obj, dict):
        for key in ("motion", "motion_151", "poses", "canonical_motion", "unit_motions_physical"):
            if key in obj:
                found = _first_motion_in_object(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = _first_motion_in_object(value)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _first_motion_in_object(value)
            if found is not None:
                return found
    return None


def load_motion(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if p.suffix == ".npy":
        obj = np.load(p, allow_pickle=True)
    elif p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        obj = {k: z[k] for k in z.files}
    else:
        with p.open("rb") as f:
            obj = pickle.load(f)
    motion = _first_motion_in_object(obj)
    if motion is None:
        raise ValueError(f"No [T,151] motion found in {p}")
    if not np.isfinite(motion).all():
        raise ValueError(f"NaN/Inf motion in {p}")
    return motion.astype(np.float32)


def localize_root(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32).copy()
    if len(x):
        x[:, ROOT_X] -= x[0, ROOT_X]
        x[:, ROOT_Z] -= x[0, ROOT_Z]
    return x


def resample_motion(motion: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if x.ndim == 2 and x.shape[1] == 151:
        return resample_edge151_np(x, target_frames=int(target_len))
    if len(x) == target_len:
        return x.copy()
    if len(x) == 1:
        return np.repeat(x, target_len, axis=0)
    old_t = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.stack(
        [np.interp(new_t, old_t, x[:, d]) for d in range(x.shape[1])],
        axis=-1,
    ).astype(np.float32)


def robust_scale(values: np.ndarray, lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float32)
    if lo is None:
        lo = np.percentile(x, 10, axis=0).astype(np.float32)
    if hi is None:
        hi = np.percentile(x, 90, axis=0).astype(np.float32)
    scaled = np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)
    return scaled, np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)


def _joint_group_activity(rot_vel: np.ndarray, start: int, end: int) -> float:
    if rot_vel.size == 0:
        return 0.0
    group = rot_vel[:, start:end]
    if group.size == 0:
        return 0.0
    return float(np.linalg.norm(group.reshape(len(group), -1), axis=-1).mean())


def motion_descriptor_raw(motion: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Return 12D motion descriptor aligned with the music query descriptor.

    Dimensions:
      0 energy/full activity
      1 upper-body activity
      2 torso activity
      3 lower-body activity
      4 style tension / pose range
      5 calmness / smoothness
      6 support-change amount
      7 build-up trend
      8 release trend
      9 accent / peakiness
      10 phrase-change / entry-exit contrast
      11 duration (normalized later)
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    rot = x[:, ROT].reshape(len(x), 24, 6)
    matrices = rot6d_to_matrix_np(rot)
    if len(x) > 1:
        d = angular_velocity_np(matrices, fps=float(fps))
        frame_energy = np.linalg.norm(d.reshape(len(d), -1), axis=-1)
    else:
        d = np.zeros((0, 24, 3), dtype=np.float32)
        frame_energy = np.zeros((0,), dtype=np.float32)
    full = float(frame_energy.mean()) if len(frame_energy) else 0.0
    upper = _joint_group_activity(d, 14, 24)
    torso = _joint_group_activity(d, 8, 14)
    lower = _joint_group_activity(d, 0, 8)
    pose_range = float(np.std(rot.reshape(len(rot), -1), axis=0).mean()) if len(rot) else 0.0
    smoothness = 1.0 / (1.0 + float(np.var(frame_energy)) if len(frame_energy) else 1.0)
    contacts = x[:, CONTACT]
    duration_seconds = max((len(x) - 1) / max(float(fps), 1.0e-8), 1.0 / max(float(fps), 1.0e-8))
    support = float(np.abs(np.diff(contacts, axis=0)).sum() / duration_seconds) if len(x) > 1 else 0.0
    q = max(2, min(len(frame_energy), int(round(0.25 * float(fps)))))
    if len(frame_energy):
        entry = float(frame_energy[:q].mean())
        exit_ = float(frame_energy[-q:].mean())
        peak = float(np.percentile(frame_energy, 90))
        mean_e = float(frame_energy.mean())
    else:
        entry = exit_ = peak = mean_e = 0.0
    build = max(0.0, exit_ - entry)
    release = max(0.0, entry - exit_)
    accent = peak / (mean_e + 1e-6)
    change = float(np.mean(so3_geodesic_np(matrices[0], matrices[-1]))) if len(rot) else 0.0
    # This descriptor measures the sampled physical trajectory span.  Keep it
    # separate from Scheduler occupancy, which intentionally uses N / FPS for
    # an exclusive [start, end) frame interval.
    duration = float(duration_seconds)
    return np.asarray(
        [full, upper, torso, lower, pose_range, smoothness, support, build, release, accent, change, duration],
        dtype=np.float32,
    )


def _fixed_projection(in_dim: int, out_dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(20260605 + in_dim * 31 + out_dim)
    matrix = rng.standard_normal((in_dim, out_dim), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-8
    return matrix.astype(np.float32)


def motion_mmr_embedding(motion: np.ndarray, out_dim: int = 64, fps: float = 30.0) -> np.ndarray:
    """Compact deterministic style/dynamics embedding for MMR diversity."""
    x = resample_motion(localize_root(motion), 48)
    rot = x[:, ROT].reshape(48, 24, 6)
    duration_seconds = max((len(motion) - 1) / max(float(fps), 1.0e-8), 1.0 / max(float(fps), 1.0e-8))
    effective_fps = float(47.0 / duration_seconds)
    matrices = rot6d_to_matrix_np(rot)
    velocity_core = angular_velocity_np(matrices, fps=effective_fps)
    acceleration_core = angular_acceleration_np(matrices, fps=effective_fps)
    vel = np.concatenate([np.zeros((1, 24, 3), dtype=np.float32), velocity_core], axis=0)
    acc = np.concatenate([np.zeros((2, 24, 3), dtype=np.float32), acceleration_core], axis=0)[:48]
    groups = ((0, 8), (8, 14), (14, 24))
    feats: List[np.ndarray] = []
    for start, end in groups:
        g = rot[:, start:end].reshape(48, -1)
        gv = vel[:, start:end].reshape(48, -1)
        ga = acc[:, start:end].reshape(48, -1)
        feats.extend([
            g.mean(axis=0),
            g.std(axis=0),
            np.abs(gv).mean(axis=0),
            np.abs(ga).mean(axis=0),
        ])
        spec = np.abs(np.fft.rfft(g - g.mean(axis=0, keepdims=True), axis=0))[1:4]
        feats.append(spec.mean(axis=0))
    root = x[:, 4:7]
    root_velocity = np.diff(root, axis=0) * effective_fps
    contacts = x[:, CONTACT]
    feats.extend([
        root.mean(axis=0), root.std(axis=0), np.abs(root_velocity).mean(axis=0),
        contacts.mean(axis=0), np.abs(np.diff(contacts, axis=0)).mean(axis=0) * effective_fps,
    ])
    raw = np.concatenate([np.asarray(f, dtype=np.float32).reshape(-1) for f in feats])
    proj = raw @ _fixed_projection(raw.size, out_dim)
    norm = float(np.linalg.norm(proj))
    return (proj / (norm + 1e-8)).astype(np.float32)


def event_compatibility(music_event: str, motion_event: str) -> float:
    m = str(music_event or "neutral_flow")
    e = str(motion_event or "neutral_flow")
    if m == e:
        return 1.0
    table: Dict[str, Dict[str, float]] = {
        "accent": {"arm_flourish": 1.0, "high_tension": 0.95, "support_shift": 0.65, "build_up": 0.60, "neutral_flow": 0.20, "calm_flow": -0.15, "pose_hold": -0.25},
        "climax": {"arm_flourish": 1.0, "high_tension": 1.0, "build_up": 0.75, "support_shift": 0.55, "neutral_flow": 0.10},
        "section_change": {"support_shift": 1.0, "build_up": 0.85, "release": 0.75, "arm_flourish": 0.55, "high_tension": 0.45, "neutral_flow": 0.20},
        "build_up": {"build_up": 1.0, "high_tension": 0.85, "arm_flourish": 0.70, "support_shift": 0.45, "neutral_flow": 0.20},
        "release": {"release": 1.0, "calm_flow": 0.80, "pose_hold": 0.55, "neutral_flow": 0.30, "high_tension": -0.20},
        "calm_flow": {"calm_flow": 1.0, "pose_hold": 0.65, "neutral_flow": 0.50, "release": 0.40, "high_tension": -0.30, "arm_flourish": -0.25},
        "neutral_flow": {"neutral_flow": 0.80, "calm_flow": 0.55, "pose_hold": 0.30, "build_up": 0.25, "support_shift": 0.20},
    }
    return float(table.get(m, {}).get(e, 0.0))


def family_id(item: Dict[str, Any], family_span: int = 600) -> str:
    sid = int(item.get("source_id", -1))
    start = int(item.get("source_start", item.get("start", 0)))
    return f"{sid}:{start // max(1, int(family_span))}"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(aa, bb) / ((np.linalg.norm(aa) + 1e-8) * (np.linalg.norm(bb) + 1e-8)))


def transition_cost_from_arrays(
    prev_exit: np.ndarray,
    prev_vel: np.ndarray,
    next_entry: np.ndarray,
    next_vel: np.ndarray,
) -> float:
    prev_rotation = rot6d_to_matrix_np(
        np.asarray(prev_exit[ROT], dtype=np.float32).reshape(24, 6)
    )
    next_rotation = rot6d_to_matrix_np(
        np.asarray(next_entry[ROT], dtype=np.float32).reshape(24, 6)
    )
    relative = np.swapaxes(prev_rotation, -1, -2) @ next_rotation
    trace = np.trace(relative, axis1=-2, axis2=-1)
    pose_angles = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    pose_jump = float(np.sqrt(np.mean(np.square(pose_angles))))

    # The stored velocity arrays belong to historical Router/Planner assets and
    # remain Euclidean 151D values.  Decode the physical endpoint rotations for
    # pose distance, while keeping this legacy velocity term checkpoint-safe.
    vel_jump = float(
        np.linalg.norm(prev_vel[ROT] - next_vel[ROT]) / math.sqrt(144.0)
    )
    root_y_jump = abs(float(prev_exit[ROOT_Y] - next_entry[ROOT_Y]))
    contact_jump = float(np.abs(prev_exit[CONTACT] - next_entry[CONTACT]).mean())
    return pose_jump + 0.35 * vel_jump + 0.50 * root_y_jump + 0.15 * contact_jump


def motion_boundary_metrics(
    prev: np.ndarray,
    nxt: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, float]:
    """Intrinsic pose/dynamics metrics for two EDGE151 motion endpoints."""
    previous = np.asarray(prev, dtype=np.float32)
    following = np.asarray(nxt, dtype=np.float32)
    previous_rotation = rot6d_to_matrix_np(
        previous[:, ROT].reshape(len(previous), 24, 6)
    )
    following_rotation = rot6d_to_matrix_np(
        following[:, ROT].reshape(len(following), 24, 6)
    )
    previous_velocity = (
        relative_rotvec_np(previous_rotation[-2], previous_rotation[-1])
        if len(previous) > 1
        else np.zeros((24, 3), dtype=np.float32)
    )
    following_velocity = (
        relative_rotvec_np(following_rotation[0], following_rotation[1])
        if len(following) > 1
        else np.zeros((24, 3), dtype=np.float32)
    )
    previous_velocity_before = (
        relative_rotvec_np(previous_rotation[-3], previous_rotation[-2])
        if len(previous) > 2
        else previous_velocity
    )
    following_velocity_after = (
        relative_rotvec_np(following_rotation[1], following_rotation[2])
        if len(following) > 2
        else following_velocity
    )
    previous_velocity *= float(fps)
    following_velocity *= float(fps)
    previous_velocity_before *= float(fps)
    following_velocity_after *= float(fps)
    previous_acceleration = (previous_velocity - previous_velocity_before) * float(fps)
    following_acceleration = (following_velocity_after - following_velocity) * float(fps)
    yaw = root_yaw_np(
        np.stack([previous[-1], following[0]], axis=0).astype(np.float32)
    )
    yaw_delta = (
        float(np.arctan2(np.sin(yaw[1] - yaw[0]), np.cos(yaw[1] - yaw[0])))
        if len(yaw) == 2
        else 0.0
    )
    return {
        "pose_jump": float(
            np.sqrt(
                np.mean(
                    np.square(
                        so3_geodesic_np(previous_rotation[-1], following_rotation[0])
                    )
                )
            )
        ),
        "angular_velocity_jump_radps": float(
            np.sqrt(np.mean(np.square(previous_velocity - following_velocity)))
        ),
        "angular_acceleration_jump_radps2": float(
            np.sqrt(
                np.mean(
                    np.square(previous_acceleration - following_acceleration)
                )
            )
        ),
        "contact_jump": float(
            np.abs(previous[-1, CONTACT] - following[0, CONTACT]).mean()
        ),
        "yaw_gap_deg": float(abs(yaw_delta) * 180.0 / np.pi),
    }


def smoothstep(x: float) -> float:
    v = float(np.clip(x, 0.0, 1.0))
    return v * v * (3.0 - 2.0 * v)


def make_geodesic_transition(prev: np.ndarray, nxt: np.ndarray, length: int) -> np.ndarray:
    """Build a geometry-aware EDGE151 transition.

    Root translation is eased in Euclidean space, rotations follow the SO(3)
    shortest path, and contact channels switch discretely.
    """
    k = max(0, int(length))
    if k == 0:
        return np.zeros((0, 151), dtype=np.float32)
    out = np.zeros((k, 151), dtype=np.float32)
    a0 = np.asarray(prev[-1], dtype=np.float32)
    a1 = np.asarray(nxt[0], dtype=np.float32)
    r0 = rot6d_to_matrix_np(a0[ROT].reshape(24, 6))
    r1 = rot6d_to_matrix_np(a1[ROT].reshape(24, 6))
    for i in range(k):
        alpha = smoothstep((i + 1) / (k + 1))
        out[i] = (1.0 - alpha) * a0 + alpha * a1
        # Contacts are categorical support observations.  Switch once at the
        # temporal midpoint instead of activating the target support pattern
        # on the first bridge frame.
        out[i, CONTACT] = a0[CONTACT] if alpha < 0.5 else a1[CONTACT]
        blended = tangent_blend_np(
            r0,
            r1,
            np.full((24,), alpha, dtype=np.float32),
        )
        out[i, ROT] = matrix_to_rot6d_np(blended).reshape(-1)
    return out


def make_linear_transition(prev: np.ndarray, nxt: np.ndarray, length: int) -> np.ndarray:
    """Compatibility alias for historical Scheduler/checkpoint tooling."""
    return make_geodesic_transition(prev, nxt, length)


def apply_start_anchor(motion: np.ndarray, start_pose: np.ndarray, blend_frames: int = 8) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32).copy()
    s = np.asarray(start_pose, dtype=np.float32).reshape(-1)
    if s.shape[0] != 151 or len(x) == 0:
        return x
    bf = max(1, min(int(blend_frames), len(x)))
    x[0, CONTACT] = s[CONTACT]
    x[0, ROOT_Y] = s[ROOT_Y]
    x[0, ROT] = s[ROT]
    x[:, ROOT_X] = 0.0
    x[:, ROOT_Z] = 0.0
    anchor_rotation = rot6d_to_matrix_np(s[ROT].reshape(24, 6))
    target_rotations = rot6d_to_matrix_np(x[:bf, ROT].reshape(bf, 24, 6))
    for t in range(1, bf):
        alpha = smoothstep(t / max(bf - 1, 1))
        x[t, ROOT_Y] = (1.0 - alpha) * s[ROOT_Y] + alpha * x[t, ROOT_Y]
        blended = tangent_blend_np(
            anchor_rotation,
            target_rotations[t],
            np.full((24,), alpha, dtype=np.float32),
        )
        x[t, ROT] = matrix_to_rot6d_np(blended).reshape(-1)
    return x


def pad_or_trim_motion(motion: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if len(x) >= target_len:
        return x[:target_len].copy()
    if len(x) == 0:
        return np.zeros((target_len, 151), dtype=np.float32)
    pad = np.repeat(x[-1:], target_len - len(x), axis=0)
    return np.concatenate([x, pad], axis=0).astype(np.float32)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
