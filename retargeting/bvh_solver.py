#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimization-based Chang-E BVH -> EDGE 151D retargeting.

Why this exists
---------------
Chang-E BVH and EDGE's 24-joint SMPL-like skeleton do not share joint-local
frames.  Copying local Euler rotations by joint name is therefore not a valid
retargeting operation.  This implementation follows the scientific logic of
the Chang-E paper: source BVH *global keypoints* supervise optimisation of
target root orientation, root translation, body pose and a global scale.

The output is directly compatible with the V46 MotionRAG-Diff contract:
  [contacts(4), root_xyz(3), local_rot6d(24*6)] = 151 dimensions.

The fitter intentionally uses the repository's fixed 24-joint kinematic tree,
so database construction, training, IK, audit and rendering share one skeleton.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for V46.49 retargeting") from exc

try:
    from scipy.ndimage import median_filter
except Exception:  # pragma: no cover
    median_filter = None

from contracts.gravity import (
    EDGE_DIM,
    FOOT_JOINTS,
    NUM_JOINTS,
    OFFSETS,
    PARENTS,
    GravityThresholds,
    evaluate_gravity_contract,
    gravity_metrics_np,
    identity6d_np,
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
)

CONTACT = slice(0, 4)
ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX = 4, 5, 6
ROT6D_START, ROT6D_END = 7, 151


@dataclass
class BVHJoint:
    name: str
    parent: int
    offset: np.ndarray
    channels: List[str]
    channel_start: int
    is_end_site: bool = False


@dataclass
class BVHMotion:
    path: str
    joints: List[BVHJoint]
    values: np.ndarray
    frame_time: float

    @property
    def fps(self) -> float:
        return 1.0 / max(float(self.frame_time), 1e-8)


@dataclass
class RetargetConfig:
    target_fps: float = 30.0
    device: str = "cuda"
    chunk_frames: int = 256
    chunk_overlap: int = 32
    iterations: int = 90
    learning_rate: float = 0.035
    keypoint_weight: float = 30.0
    root_weight: float = 8.0
    temporal_velocity_weight: float = 0.10
    temporal_acceleration_weight: float = 0.025
    pose_prior_weight: float = 0.006
    upright_weight: float = 5.0
    head_order_weight: float = 2.0
    feet_order_weight: float = 1.5
    floor_weight: float = 4.0
    root_velocity_weight: float = 1.5
    # ===== V46.49.4 ABSOLUTE ROOT ORIENTATION CONTRACT =====
    root_orientation_lock: bool = True
    # ===== V46.49.4 ABSOLUTE ROOT ORIENTATION CONTRACT END =====
    gradient_clip: float = 2.0
    contact_height_m: float = 0.055
    contact_speed_mpf: float = 0.025
    contact_median_size: int = 5
    localize_root_xz: bool = True
    floor_to_zero: bool = True
    hard_gravity_gate: bool = True
    gravity_torso_p05_min: float = 0.45
    fit_rmse_p95_max_m: float = 0.18
    seed: int = 1234

    @classmethod
    def from_env(cls) -> "RetargetConfig":
        def f(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return float(default)

        def i(name: str, default: int) -> int:
            try:
                return int(float(os.environ.get(name, default)))
            except Exception:
                return int(default)

        def b(name: str, default: bool) -> bool:
            raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
            return raw in {"1", "true", "yes", "y", "on"}

        return cls(
            target_fps=f("V46_49_RETARGET_FPS", 30.0),
            device=os.environ.get("V46_49_RETARGET_DEVICE", os.environ.get("V46_DEVICE", "cuda")),
            chunk_frames=i("V46_49_RETARGET_CHUNK", 256),
            chunk_overlap=i("V46_49_RETARGET_OVERLAP", 32),
            iterations=i("V46_49_RETARGET_ITERS", 90),
            learning_rate=f("V46_49_RETARGET_LR", 0.035),
            keypoint_weight=f("V46_49_RETARGET_KEYPOINT_W", 30.0),
            root_weight=f("V46_49_RETARGET_ROOT_W", 8.0),
            temporal_velocity_weight=f("V46_49_RETARGET_VEL_W", 0.10),
            temporal_acceleration_weight=f("V46_49_RETARGET_ACC_W", 0.025),
            pose_prior_weight=f("V46_49_RETARGET_POSE_W", 0.006),
            upright_weight=f("V46_49_RETARGET_UPRIGHT_W", 5.0),
            head_order_weight=f("V46_49_RETARGET_HEAD_W", 2.0),
            feet_order_weight=f("V46_49_RETARGET_FEET_W", 1.5),
            floor_weight=f("V46_49_RETARGET_FLOOR_W", 4.0),
            root_velocity_weight=f("V46_49_RETARGET_ROOT_VEL_W", 1.5),
            root_orientation_lock=b("V46_49_ROOT_ORIENTATION_LOCK", True),
            gradient_clip=f("V46_49_RETARGET_GRAD_CLIP", 2.0),
            contact_height_m=f("V46_49_CONTACT_HEIGHT_M", 0.055),
            contact_speed_mpf=f("V46_49_CONTACT_SPEED_MPF", 0.025),
            contact_median_size=i("V46_49_CONTACT_MEDIAN", 5),
            localize_root_xz=b("V46_49_LOCALIZE_ROOT_XZ", True),
            floor_to_zero=b("V46_49_FLOOR_TO_ZERO", True),
            hard_gravity_gate=b("V46_49_GRAVITY_HARD_FAIL", True),
            gravity_torso_p05_min=f(
                "V46_49_GRAVITY_TORSO_P05_MIN", 0.45
            ),
            fit_rmse_p95_max_m=f(
                "V46_49_FIT_RMSE_P95_MAX_M", 0.18
            ),
            seed=i("V46_49_SEED", 1234),
        )


TARGET_ALIASES: List[List[str]] = [
    ["hips", "hip", "pelvis", "root", "mixamorighips"],
    ["leftupleg", "lefthip", "leftthigh", "lhip", "lthigh"],
    ["rightupleg", "righthip", "rightthigh", "rhip", "rthigh"],
    ["spine", "spine0", "lowerspine", "abdomen", "lowerchest"],
    ["leftleg", "leftknee", "leftshin", "lknee", "lshin"],
    ["rightleg", "rightknee", "rightshin", "rknee", "rshin"],
    ["spine1", "midspine", "chest"],
    ["leftfoot", "leftankle", "lfoot", "lankle"],
    ["rightfoot", "rightankle", "rfoot", "rankle"],
    ["spine2", "spine3", "upperchest", "chest2", "thorax"],
    ["lefttoebase", "lefttoe", "leftball", "ltoe", "leftankleendsite"],
    ["righttoebase", "righttoe", "rightball", "rtoe", "rightankleendsite"],
    ["neck", "neck1"],
    ["leftcollar", "leftclavicle", "lcollar"],
    ["rightcollar", "rightclavicle", "rcollar"],
    ["head", "headtop"],
    ["leftshoulder", "leftarm", "leftupperarm", "lupperarm"],
    ["rightshoulder", "rightarm", "rightupperarm", "rupperarm"],
    ["leftforearm", "leftlowerarm", "leftelbow", "lelbow"],
    ["rightforearm", "rightlowerarm", "rightelbow", "relbow"],
    ["lefthand", "leftwrist", "lwrist"],
    ["righthand", "rightwrist", "rwrist"],
    ["leftwristendsite", "lefthandendsite", "lefthandend", "leftfinger", "leftthumb"],
    ["rightwristendsite", "righthandendsite", "righthandend", "rightfinger", "rightthumb"],
]

# Exact profile used by the new Chang-E files currently placed under EDGE/change.
# The source hierarchy is:
#   Hips -> Chest -> Chest2 -> Neck -> Head,
# with Collar/Shoulder/Elbow/Wrist arms and Hip/Knee/Ankle legs.
# There is one fewer explicit spine joint than the EDGE/SMPL-like target, so
# target joint 3 (belly) is generated as a virtual interpolation between Hips
# and Chest; target joints 6 and 9 are supervised by Chest and Chest2.
CHANGE_SIMPLIFIED_PROFILE: Dict[int, str] = {
    0: "hips",
    1: "lefthip",
    2: "righthip",
    4: "leftknee",
    5: "rightknee",
    6: "chest",
    7: "leftankle",
    8: "rightankle",
    9: "chest2",
    10: "leftankleendsite",
    11: "rightankleendsite",
    12: "neck",
    13: "leftcollar",
    14: "rightcollar",
    15: "head",
    16: "leftshoulder",
    17: "rightshoulder",
    18: "leftelbow",
    19: "rightelbow",
    20: "leftwrist",
    21: "rightwrist",
    22: "leftwristendsite",
    23: "rightwristendsite",
}

# Strong torso/hips supervision, moderate limbs, lighter end-effectors.
TARGET_JOINT_WEIGHTS = np.asarray(
    [4.0, 2.5, 2.5, 3.0, 1.5, 1.5, 3.0, 1.5, 1.5, 3.5,
     1.0, 1.0, 3.0, 1.8, 1.8, 2.5, 1.5, 1.5, 1.2, 1.2,
     1.0, 1.0, 0.5, 0.5],
    dtype=np.float32,
)


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def parse_bvh(path: str | Path) -> BVHMotion:
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    motion_line = next((i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION"), None)
    if motion_line is None:
        raise ValueError(f"No MOTION section: {p}")

    joints: List[BVHJoint] = []
    stack: List[int] = []
    pending: Optional[int] = None
    channel_cursor = 0
    end_counts: Dict[int, int] = {}

    for raw in lines[:motion_line]:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()
        if key in {"ROOT", "JOINT"} and len(parts) >= 2:
            parent = stack[-1] if stack else -1
            joints.append(BVHJoint(parts[1], parent, np.zeros(3, np.float32), [], channel_cursor, False))
            pending = len(joints) - 1
        elif key == "END":
            if not stack:
                continue
            parent = stack[-1]
            count = end_counts.get(parent, 0)
            end_counts[parent] = count + 1
            suffix = "EndSite" if count == 0 else f"EndSite{count + 1}"
            joints.append(
                BVHJoint(
                    f"{joints[parent].name}_{suffix}",
                    parent,
                    np.zeros(3, np.float32),
                    [],
                    channel_cursor,
                    True,
                )
            )
            pending = len(joints) - 1
        elif key == "{":
            if pending is not None:
                stack.append(pending)
                pending = None
        elif key == "}":
            if stack:
                stack.pop()
        elif key == "OFFSET" and len(parts) >= 4 and stack:
            joints[stack[-1]].offset = np.asarray(
                [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32
            )
        elif key == "CHANNELS" and len(parts) >= 2 and stack:
            n = int(parts[1])
            ch = list(parts[2:2 + n])
            j = joints[stack[-1]]
            j.channels = ch
            j.channel_start = channel_cursor
            channel_cursor += n

    frames = None
    frame_time = None
    data_start = None
    for i in range(motion_line + 1, len(lines)):
        s = lines[i].strip()
        low = s.lower()
        if low.startswith("frames"):
            frames = int(s.replace(":", " ").split()[-1])
        elif low.startswith("frame time"):
            frame_time = float(s.replace(":", " ").split()[-1])
            data_start = i + 1
            break
    if frame_time is None or data_start is None:
        raise ValueError(f"Missing Frame Time: {p}")

    rows = []
    for raw in lines[data_start:]:
        s = raw.strip()
        if not s:
            continue
        vals = [float(v) for v in s.split()]
        if len(vals) >= channel_cursor:
            rows.append(vals[:channel_cursor])
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Empty BVH frames: {p}")
    if frames is not None and frames != len(values):
        frames = len(values)

    return BVHMotion(str(p), joints, values, float(frame_time))


def _axis_matrix_batch(axis: str, angle_deg: np.ndarray) -> np.ndarray:
    a = np.deg2rad(np.asarray(angle_deg, dtype=np.float32))
    c, s = np.cos(a), np.sin(a)
    out = np.zeros((len(a), 3, 3), dtype=np.float32)
    axis = axis.upper()[0]
    if axis == "X":
        out[:, 0, 0] = 1
        out[:, 1, 1] = c
        out[:, 1, 2] = -s
        out[:, 2, 1] = s
        out[:, 2, 2] = c
    elif axis == "Y":
        out[:, 1, 1] = 1
        out[:, 0, 0] = c
        out[:, 0, 2] = s
        out[:, 2, 0] = -s
        out[:, 2, 2] = c
    else:
        out[:, 2, 2] = 1
        out[:, 0, 0] = c
        out[:, 0, 1] = -s
        out[:, 1, 0] = s
        out[:, 1, 1] = c
    return out


def source_fk(bvh: BVHMotion, use_motion: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return source global positions and global rotations in native BVH units."""
    T, J = len(bvh.values), len(bvh.joints)
    gp = np.zeros((T, J, 3), dtype=np.float32)
    gr = np.zeros((T, J, 3, 3), dtype=np.float32)
    local_r = np.tile(np.eye(3, dtype=np.float32), (T, J, 1, 1))
    local_t = np.zeros((T, J, 3), dtype=np.float32)

    # ===== V46.49.2 NONROOT POSITION CONTRACT =====
    # Chang-E files expose XYZ position channels on every articulated joint.
    # Dataset inspection shows that non-root position channels are approximately
    # static calibration values. They must not be added on top of hierarchy
    # OFFSET in the formal retargeting path, otherwise each bone chain is
    # translated twice and global keypoint fitting becomes physically impossible.
    #
    # Modes:
    #   ignore (formal/default): keep root XYZ only; non-root uses hierarchy OFFSET
    #   delta: preserve only non-root displacement relative to frame 0
    #   raw: old diagnostic behaviour; add full non-root position channels
    _position_mode = str(
        os.environ.get("V46_49_NONROOT_POSITION_MODE", "ignore")
    ).strip().lower()
    if _position_mode not in {"ignore", "delta", "raw"}:
        raise ValueError(
            "V46_49_NONROOT_POSITION_MODE must be ignore/delta/raw, "
            f"got {_position_mode!r}"
        )

    for j_idx, joint in enumerate(bvh.joints):
        local_t[:, j_idx] = joint.offset[None]
        if use_motion and joint.channels:
            _pos_values = {}
            for k, ch in enumerate(joint.channels):
                col = joint.channel_start + k
                low = ch.lower()
                if low.endswith("position"):
                    axis = "xyz".index(low[0])
                    _pos_values[axis] = bvh.values[:, col].astype(np.float32)

            # Root translation is the only absolute translation retained.
            if j_idx == 0:
                for axis, values in _pos_values.items():
                    local_t[:, j_idx, axis] += values
            elif _position_mode == "raw":
                for axis, values in _pos_values.items():
                    local_t[:, j_idx, axis] += values
            elif _position_mode == "delta":
                for axis, values in _pos_values.items():
                    local_t[:, j_idx, axis] += values - values[:1]
            # ignore: hierarchy OFFSET alone defines non-root translation.

            for k, ch in enumerate(joint.channels):
                if ch.lower().endswith("rotation"):
                    local_r[:, j_idx] = local_r[:, j_idx] @ _axis_matrix_batch(
                        ch[0], bvh.values[:, joint.channel_start + k]
                    )
    # ===== V46.49.2 NONROOT POSITION CONTRACT END =====

    for j in range(J):
        p = bvh.joints[j].parent
        if p < 0:
            gp[:, j] = local_t[:, j]
            gr[:, j] = local_r[:, j]
        else:
            gp[:, j] = gp[:, p] + (gr[:, p] @ local_t[:, j, :, None])[..., 0]
            gr[:, j] = gr[:, p] @ local_r[:, j]
    return gp, gr


def target_rest_positions() -> np.ndarray:
    p = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for j in range(NUM_JOINTS):
        parent = int(PARENTS[j])
        p[j] = np.zeros(3, np.float32) if parent < 0 else p[parent] + OFFSETS[j]
    return p


def _exact_index_by_name(joints: Sequence[BVHJoint]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, joint in enumerate(joints):
        out.setdefault(_norm_name(joint.name), i)
    return out


def _looks_like_change_simplified_profile(names: set[str]) -> bool:
    required = {
        "hips", "chest", "chest2", "neck", "head",
        "leftcollar", "leftshoulder", "leftelbow", "leftwrist",
        "rightcollar", "rightshoulder", "rightelbow", "rightwrist",
        "lefthip", "leftknee", "leftankle",
        "righthip", "rightknee", "rightankle",
    }
    return required.issubset(names)


def build_joint_mapping(joints: Sequence[BVHJoint]) -> Dict[int, int]:
    """Build a deterministic semantic mapping from source BVH to EDGE joints.

    Important: Collar and Shoulder are different anatomical levels.  The old
    fuzzy matcher allowed target inner-shoulder to consume source Shoulder,
    leaving no upper-arm joint.  Exact profile mapping is therefore preferred
    whenever the current Chang-E simplified hierarchy is detected.
    """
    names = [_norm_name(j.name) for j in joints]
    name_to_idx = _exact_index_by_name(joints)
    name_set = set(names)

    if _looks_like_change_simplified_profile(name_set):
        mapping: Dict[int, int] = {}
        for tgt, source_name in CHANGE_SIMPLIFIED_PROFILE.items():
            src = name_to_idx.get(source_name)
            if src is not None:
                mapping[tgt] = int(src)
        # End sites are optional.  If absent, use their parent endpoint only as
        # a low-weight proxy; core body mapping remains exact.
        if 10 not in mapping and 7 in mapping:
            mapping[10] = mapping[7]
        if 11 not in mapping and 8 in mapping:
            mapping[11] = mapping[8]
        if 22 not in mapping and 20 in mapping:
            mapping[22] = mapping[20]
        if 23 not in mapping and 21 in mapping:
            mapping[23] = mapping[21]
        return mapping

    used: set[int] = set()
    mapping = {}
    for tgt, aliases in enumerate(TARGET_ALIASES):
        alias_n = [_norm_name(a) for a in aliases]

        # 1) Exact matching in alias priority order.
        src = -1
        for alias in alias_n:
            candidates = [
                i for i, name in enumerate(names)
                if name == alias and (i not in used or tgt in {10, 11, 22, 23})
            ]
            if candidates:
                # Prefer real joints except for explicit end-effector targets.
                candidates.sort(key=lambda i: (joints[i].is_end_site, i))
                src = candidates[0]
                break

        # 2) Conservative contains matching only when exact names are absent.
        if src < 0:
            scored = []
            for i, name in enumerate(names):
                if i in used and tgt not in {10, 11, 22, 23}:
                    continue
                best = None
                for rank, alias in enumerate(alias_n):
                    if alias and (alias in name or name in alias):
                        score = 100 - 5 * rank - abs(len(name) - len(alias))
                        if joints[i].is_end_site and tgt not in {10, 11, 22, 23}:
                            score -= 50
                        best = score if best is None else max(best, score)
                if best is not None:
                    scored.append((best, -i, i))
            if scored:
                scored.sort(reverse=True)
                src = scored[0][2]

        if src >= 0:
            mapping[tgt] = int(src)
            if tgt not in {10, 11, 22, 23}:
                used.add(int(src))

    if 10 not in mapping and 7 in mapping:
        mapping[10] = mapping[7]
    if 11 not in mapping and 8 in mapping:
        mapping[11] = mapping[8]
    if 22 not in mapping and 20 in mapping:
        mapping[22] = mapping[20]
    if 23 not in mapping and 21 in mapping:
        mapping[23] = mapping[21]

    # Belly/spine can be synthesised from neighbouring torso observations.
    required = {0, 1, 2, 4, 5, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21}
    missing = sorted(required - set(mapping))
    if missing:
        raise RuntimeError(
            f"Missing required BVH->target joints: {missing}; "
            f"names={[j.name for j in joints]}; mapping={mapping}"
        )
    return mapping


def similarity_umeyama(X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None):
    """Fit Y ~= scale * (R @ X) + t."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if weights is None:
        w = np.ones(len(X), np.float64)
    else:
        w = np.asarray(weights, np.float64)
    w = w / max(w.sum(), 1e-12)
    mx = (X * w[:, None]).sum(axis=0)
    my = (Y * w[:, None]).sum(axis=0)
    Xc, Yc = X - mx, Y - my
    C = (Yc * w[:, None]).T @ Xc
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    var = float((w[:, None] * (Xc ** 2)).sum())
    scale = float(np.sum(S * np.diag(D)) / max(var, 1e-12))
    t = my - scale * (R @ mx)
    return scale, R.astype(np.float32), t.astype(np.float32)


def apply_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (float(scale) * (np.asarray(points) @ np.asarray(R).T) + np.asarray(t)).astype(np.float32)


def resample_global_positions(pos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(source_fps - target_fps) < 1e-5:
        return pos.astype(np.float32)
    duration = (len(pos) - 1) / max(source_fps, 1e-8)
    n = max(2, int(round(duration * target_fps)) + 1)
    old_t = np.arange(len(pos), dtype=np.float64) / source_fps
    new_t = np.arange(n, dtype=np.float64) / target_fps
    new_t = np.minimum(new_t, old_t[-1])
    flat = pos.reshape(len(pos), -1)
    out = np.empty((n, flat.shape[1]), dtype=np.float32)
    for d in range(flat.shape[1]):
        out[:, d] = np.interp(new_t, old_t, flat[:, d]).astype(np.float32)
    return out.reshape(n, *pos.shape[1:])


def _body_frame(points: np.ndarray) -> np.ndarray:
    """Build a right/up/forward frame from pelvis, hips and neck."""
    if points.ndim == 2:
        points = points[None]
    pelvis = points[:, 0]
    lhip = points[:, 1]
    rhip = points[:, 2]
    up_ref = points[:, 12] - pelvis
    right = rhip - lhip
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8)
    up = up_ref - np.sum(up_ref * right, axis=-1, keepdims=True) * right
    up /= np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-8)
    forward = np.cross(right, up)
    forward /= np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)
    up = np.cross(forward, right)
    up /= np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-8)
    R = np.stack([right, up, forward], axis=-1).astype(np.float32)
    bad = ~np.isfinite(R).all(axis=(1, 2))
    R[bad] = np.eye(3, dtype=np.float32)
    return R


def body_frame_from_keypoints(target_pos: np.ndarray) -> np.ndarray:
    """Estimate root rotation relative to the target skeleton's rest frame."""
    current = _body_frame(target_pos)
    rest = _body_frame(target_rest_positions())[0]
    return (current @ rest.T[None]).astype(np.float32)


def _rot6d_to_matrix_torch(x):
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2o = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2o, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _matrix_to_rot6d_torch(m):
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def _project6d_torch(x):
    return _matrix_to_rot6d_torch(_rot6d_to_matrix_torch(x))


def _fk_target_torch(root, rot6d):
    """root [T,3], rot6d [T,24,6] -> joints [T,24,3]."""
    local = _rot6d_to_matrix_torch(rot6d)
    offsets = torch.as_tensor(OFFSETS, dtype=root.dtype, device=root.device)
    gp, gr = [], []
    for j in range(NUM_JOINTS):
        p = int(PARENTS[j])
        if p < 0:
            rj, pj = local[:, j], root
        else:
            rj = gr[p] @ local[:, j]
            pj = gp[p] + (gr[p] @ offsets[j].view(1, 3, 1)).squeeze(-1)
        gp.append(pj)
        gr.append(rj)
    return torch.stack(gp, dim=1)


def _fit_chunk(
    source_target_pos: np.ndarray,
    source_mask: np.ndarray,
    init_root: np.ndarray,
    init_rot6d: np.ndarray,
    floor_y: float,
    cfg: RetargetConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    target = torch.as_tensor(source_target_pos, dtype=torch.float32, device=device)
    mask = torch.as_tensor(source_mask, dtype=torch.float32, device=device)
    weights = torch.as_tensor(TARGET_JOINT_WEIGHTS, dtype=torch.float32, device=device)
    w = mask * weights.view(1, -1)

    root = torch.tensor(init_root, dtype=torch.float32, device=device, requires_grad=True)
    rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device, requires_grad=True)
    init_rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device)
    reference_root_rot6d = _project6d_torch(init_rot[:, 0]).detach()
    source_root = target[:, 0].detach()
    floor = torch.tensor(float(floor_y), dtype=torch.float32, device=device)

    opt = torch.optim.Adam([root, rot], lr=float(cfg.learning_rate))
    last = {}
    for _ in range(int(cfg.iterations)):
        rp = _project6d_torch(rot)
        if cfg.root_orientation_lock:
            rp = torch.cat(
                [reference_root_rot6d[:, None, :], rp[:, 1:]],
                dim=1,
            )
        joints = _fk_target_torch(root, rp)
        diff = F.smooth_l1_loss(joints, target, reduction="none", beta=0.03).sum(dim=-1)
        key = (diff * w).sum() / w.sum().clamp_min(1.0)

        root_loss = F.smooth_l1_loss(root, source_root, beta=0.03)
        if len(root) > 1:
            root_vel = F.smooth_l1_loss(root[1:] - root[:-1], source_root[1:] - source_root[:-1], beta=0.02)
            rot_vel = (rp[1:] - rp[:-1]).pow(2).mean()
        else:
            root_vel = root.new_zeros(())
            rot_vel = root.new_zeros(())
        if len(root) > 2:
            rot_acc = (rp[2:] - 2 * rp[1:-1] + rp[:-2]).pow(2).mean()
        else:
            rot_acc = root.new_zeros(())

        pose_prior = (rp[:, 1:] - init_rot[:, 1:]).pow(2).mean()

        pelvis = joints[:, 0]
        head = joints[:, 15]
        feet = joints[:, list(FOOT_JOINTS)]
        torso_cos = F.normalize(head - pelvis, dim=-1, eps=1e-8)[:, 1]
        upright = F.relu(0.45 - torso_cos).pow(2).mean()
        head_order = F.relu(0.18 - (head[:, 1] - pelvis[:, 1])).pow(2).mean()
        feet_order = F.relu(0.30 - (pelvis[:, 1] - feet[..., 1].mean(dim=1))).pow(2).mean()
        penetration = F.relu(floor + 0.004 - feet[..., 1]).pow(2).mean()

        loss = (
            cfg.keypoint_weight * key
            + cfg.root_weight * root_loss
            + cfg.root_velocity_weight * root_vel
            + cfg.temporal_velocity_weight * rot_vel
            + cfg.temporal_acceleration_weight * rot_acc
            + cfg.pose_prior_weight * pose_prior
            + cfg.upright_weight * upright
            + cfg.head_order_weight * head_order
            + cfg.feet_order_weight * feet_order
            + cfg.floor_weight * penetration
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([root, rot], float(cfg.gradient_clip))
        opt.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "key": float(key.detach().cpu()),
            "upright": float(upright.detach().cpu()),
            "penetration": float(penetration.detach().cpu()),
        }

    with torch.no_grad():
        final_rot_t = _project6d_torch(rot)
        if cfg.root_orientation_lock:
            final_rot_t = torch.cat(
                [reference_root_rot6d[:, None, :], final_rot_t[:, 1:]],
                dim=1,
            )
        final_rot = final_rot_t.cpu().numpy().astype(np.float32)
        final_root = root.cpu().numpy().astype(np.float32)
    return final_root, final_rot, last



# ===== V46.49.3 ABSOLUTE HEADING CONTRACT =====
def _v46_49_3_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _v46_49_3_moving_average(x: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if size <= 1:
        return x.copy()
    pad = size // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(size, dtype=np.float32) / float(size)
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def _v46_49_3_moving_median(x: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    if median_filter is not None:
        return median_filter(x, size=size, mode="nearest").astype(np.float32)
    return _v46_49_3_moving_average(x, size)


def _v46_49_3_runs(mask: np.ndarray):
    m = np.asarray(mask, dtype=bool)
    if not m.size:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _v46_49_3_body_yaw(
    positions: np.ndarray,
    mapping: Dict[int, int],
) -> np.ndarray:
    p = np.asarray(positions, dtype=np.float32)
    pelvis = p[:, mapping[0]]
    lhip = p[:, mapping[1]]
    rhip = p[:, mapping[2]]
    neck = p[:, mapping[12]]

    vectors = []
    hip_right = rhip - lhip
    vectors.append(
        hip_right / np.maximum(
            np.linalg.norm(hip_right, axis=-1, keepdims=True), 1e-8
        )
    )
    if 13 in mapping and 14 in mapping:
        shoulder_right = p[:, mapping[14]] - p[:, mapping[13]]
        vectors.append(
            shoulder_right / np.maximum(
                np.linalg.norm(shoulder_right, axis=-1, keepdims=True), 1e-8
            )
        )

    right = np.mean(np.stack(vectors, axis=0), axis=0)
    right /= np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8)

    up = neck - pelvis
    up -= np.sum(up * right, axis=-1, keepdims=True) * right
    up /= np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-8)

    forward = np.cross(right, up)
    forward /= np.maximum(
        np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8
    )
    return np.unwrap(
        np.arctan2(forward[:, 0], forward[:, 2])
    ).astype(np.float32)


def _v46_49_3_heading_metrics(
    yaw: np.ndarray,
    fps: float,
    min_rate_deg_s: float,
) -> dict:
    yaw = np.asarray(yaw, dtype=np.float32)
    rate = (
        np.gradient(yaw) * float(fps)
        if len(yaw) > 1
        else np.zeros_like(yaw)
    )
    rate_deg = np.degrees(rate)
    active = np.abs(rate_deg) >= float(min_rate_deg_s)
    longest = max(
        (b - a for a, b in _v46_49_3_runs(active)),
        default=0,
    )
    return {
        "net_turns": float((yaw[-1] - yaw[0]) / (2 * np.pi))
        if len(yaw) else 0.0,
        "absolute_turns": float(
            np.sum(np.abs(np.diff(yaw))) / (2 * np.pi)
        ) if len(yaw) > 1 else 0.0,
        "yaw_speed_deg_s_p50": float(
            np.percentile(np.abs(rate_deg), 50)
        ) if len(rate_deg) else 0.0,
        "yaw_speed_deg_s_p95": float(
            np.percentile(np.abs(rate_deg), 95)
        ) if len(rate_deg) else 0.0,
        "yaw_speed_deg_s_max": float(
            np.max(np.abs(rate_deg))
        ) if len(rate_deg) else 0.0,
        "active_turn_ratio": float(active.mean())
        if len(active) else 0.0,
        "longest_active_turn_seconds": float(
            longest / max(float(fps), 1e-8)
        ),
    }


def stabilize_source_heading_positions(
    positions: np.ndarray,
    mapping: Dict[int, int],
    fps: float,
) -> Tuple[np.ndarray, dict]:
    x = np.asarray(positions, dtype=np.float32).copy()
    mode = str(
        os.environ.get("V46_49_HEADING_MODE", "stabilize")
    ).strip().lower()
    if mode not in {"stabilize", "raw", "lock"}:
        raise ValueError(
            "V46_49_HEADING_MODE must be stabilize/raw/lock, "
            f"got {mode!r}"
        )

    raw_yaw = _v46_49_3_body_yaw(x, mapping)

    smooth_seconds = _v46_49_3_env_float(
        "V46_49_HEADING_SMOOTH_SECONDS", 0.45
    )
    baseline_seconds = _v46_49_3_env_float(
        "V46_49_HEADING_BASELINE_SECONDS", 4.0
    )
    min_rate_deg_s = _v46_49_3_env_float(
        "V46_49_HEADING_MIN_DRIFT_DEG_S", 7.0
    )
    min_persist_seconds = _v46_49_3_env_float(
        "V46_49_HEADING_MIN_PERSIST_SECONDS", 3.0
    )
    consistency_min = _v46_49_3_env_float(
        "V46_49_HEADING_SIGN_CONSISTENCY", 0.82
    )
    max_variation_deg_s = _v46_49_3_env_float(
        "V46_49_HEADING_MAX_BASELINE_VARIATION_DEG_S", 14.0
    )
    max_correction_deg_s = _v46_49_3_env_float(
        "V46_49_HEADING_MAX_CORRECTION_DEG_S", 60.0
    )

    smooth_n = max(3, int(round(smooth_seconds * fps)))
    baseline_n = max(5, int(round(baseline_seconds * fps)))
    min_persist_n = max(2, int(round(min_persist_seconds * fps)))

    yaw_smooth = _v46_49_3_moving_average(raw_yaw, smooth_n)
    yaw_rate = np.gradient(yaw_smooth) * float(fps)
    baseline = _v46_49_3_moving_median(yaw_rate, baseline_n)

    same_sign = (
        np.sign(yaw_rate) == np.sign(baseline)
    ).astype(np.float32)
    consistency = _v46_49_3_moving_average(
        same_sign, baseline_n
    )
    variation = _v46_49_3_moving_average(
        np.abs(yaw_rate - baseline), baseline_n
    )

    candidate = (
        (np.abs(np.degrees(baseline)) >= min_rate_deg_s)
        & (consistency >= consistency_min)
        & (np.degrees(variation) <= max_variation_deg_s)
    )

    persistent = np.zeros(len(x), dtype=bool)
    longest_candidate = 0
    for a, b in _v46_49_3_runs(candidate):
        longest_candidate = max(longest_candidate, b - a)
        if b - a >= min_persist_n:
            persistent[a:b] = True

    if mode == "raw":
        correction = np.zeros_like(raw_yaw)
        drift_rate = np.zeros_like(raw_yaw)
    elif mode == "lock":
        correction = raw_yaw - raw_yaw[:1]
        drift_rate = np.gradient(correction) * float(fps)
        persistent[:] = True
    else:
        max_rate = np.deg2rad(max_correction_deg_s)
        drift_rate = np.where(
            persistent,
            np.clip(baseline, -max_rate, max_rate),
            0.0,
        ).astype(np.float32)
        drift_rate = _v46_49_3_moving_average(
            drift_rate,
            max(3, int(round(0.75 * fps))),
        )
        correction = np.cumsum(
            drift_rate / float(fps)
        ).astype(np.float32)
        correction -= correction[:1]

    pelvis = x[:, mapping[0]].copy()
    rel = x - pelvis[:, None, :]
    theta = -correction
    c, s = np.cos(theta), np.sin(theta)
    old_x = rel[..., 0].copy()
    old_z = rel[..., 2].copy()
    rel[..., 0] = c[:, None] * old_x + s[:, None] * old_z
    rel[..., 2] = -s[:, None] * old_x + c[:, None] * old_z
    corrected = rel + pelvis[:, None, :]

    corrected_yaw = _v46_49_3_body_yaw(corrected, mapping)
    report = {
        "version": "v46_49_3_absolute_heading_contract",
        "mode": mode,
        "persistent_drift_ratio": float(persistent.mean()),
        "longest_candidate_drift_seconds": float(
            longest_candidate / max(float(fps), 1e-8)
        ),
        "removed_turns": float(
            (correction[-1] - correction[0]) / (2 * np.pi)
        ) if len(correction) else 0.0,
        "correction_speed_deg_s_p95": float(
            np.percentile(np.abs(np.degrees(drift_rate)), 95)
        ) if len(drift_rate) else 0.0,
        "raw": _v46_49_3_heading_metrics(
            raw_yaw, fps, min_rate_deg_s
        ),
        "corrected": _v46_49_3_heading_metrics(
            corrected_yaw, fps, min_rate_deg_s
        ),
    }
    return corrected.astype(np.float32), report
# ===== V46.49.3 ABSOLUTE HEADING CONTRACT END =====


def fit_target_motion(
    aligned_source_positions: np.ndarray,
    mapping: Dict[int, int],
    cfg: RetargetConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    T = len(aligned_source_positions)
    target_pos = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    mask = np.zeros((T, NUM_JOINTS), dtype=np.float32)
    for tgt, src in mapping.items():
        target_pos[:, tgt] = aligned_source_positions[:, src]
        mask[:, tgt] = 1.0

    # The current Chang-E simplified hierarchy has Hips->Chest->Chest2 but the
    # EDGE target has pelvis->belly->spine->chest.  Do not duplicate a source
    # rotation.  Instead create geometrically meaningful virtual observations.
    # These are global keypoint targets only; the optimiser still solves all
    # target local rotations on the EDGE skeleton.
    if 3 not in mapping and 0 in mapping and 6 in mapping:
        target_pos[:, 3] = 0.45 * target_pos[:, 0] + 0.55 * target_pos[:, 6]
        mask[:, 3] = 0.75
    if 6 not in mapping and 3 in mapping and 9 in mapping:
        target_pos[:, 6] = 0.50 * target_pos[:, 3] + 0.50 * target_pos[:, 9]
        mask[:, 6] = 0.75
    if 9 not in mapping and 6 in mapping and 12 in mapping:
        target_pos[:, 9] = 0.55 * target_pos[:, 6] + 0.45 * target_pos[:, 12]
        mask[:, 9] = 0.75

    init_root = target_pos[:, 0].copy()
    init_rot = identity6d_np((T, NUM_JOINTS))
    root_R = body_frame_from_keypoints(target_pos)
    init_rot[:, 0] = matrix_to_rot6d_np(root_R)

    source_foot_ids = [mapping[t] for t in FOOT_JOINTS if t in mapping]
    if source_foot_ids:
        floor_y = float(np.percentile(aligned_source_positions[:, source_foot_ids, 1], 5))
    else:
        floor_y = float(np.percentile(target_pos[:, [7, 8], 1], 5))

    chunk = max(32, int(cfg.chunk_frames))
    overlap = max(0, min(int(cfg.chunk_overlap), chunk // 2))
    stride = max(1, chunk - overlap)
    accum_root = np.zeros((T, 3), dtype=np.float32)
    accum_rot = np.zeros((T, NUM_JOINTS, 6), dtype=np.float32)
    weight_sum = np.zeros((T, 1), dtype=np.float32)
    chunk_reports = []

    for st in range(0, T, stride):
        ed = min(T, st + chunk)
        if ed - st < 4:
            continue
        r, q, rep = _fit_chunk(
            target_pos[st:ed],
            mask[st:ed],
            init_root[st:ed],
            init_rot[st:ed],
            floor_y,
            cfg,
        )
        L = ed - st
        weight = np.ones((L, 1), dtype=np.float32)
        ov = min(overlap, L // 2)
        if ov > 1 and st > 0:
            weight[:ov, 0] = np.linspace(1e-3, 1.0, ov, dtype=np.float32)
        if ov > 1 and ed < T:
            weight[-ov:, 0] = np.linspace(1.0, 1e-3, ov, dtype=np.float32)
        accum_root[st:ed] += r * weight
        accum_rot[st:ed] += q * weight[:, None]
        weight_sum[st:ed] += weight
        chunk_reports.append({"start": st, "end": ed, **rep})

    valid = weight_sum[:, 0] > 0
    if not np.all(valid):
        accum_root[~valid] = init_root[~valid]
        accum_rot[~valid] = init_rot[~valid]
        weight_sum[~valid] = 1.0
    root = accum_root / weight_sum
    rot6d = accum_rot / weight_sum[:, None]
    rot6d = matrix_to_rot6d_np(rot6d_to_matrix_np(rot6d))

    motion = np.zeros((T, EDGE_DIM), dtype=np.float32)
    motion[:, 4:7] = root
    motion[:, 7:151] = rot6d.reshape(T, -1)

    # Optional stage localisation; Y is retained until floor normalisation.
    if cfg.localize_root_xz:
        motion[:, ROOT_X_IDX] -= motion[0, ROOT_X_IDX]
        motion[:, ROOT_Z_IDX] -= motion[0, ROOT_Z_IDX]

    from contracts.gravity import fk24_np

    joints = fk24_np(motion)
    fitted_floor = float(np.percentile(joints[:, list(FOOT_JOINTS), 1], 5))
    if cfg.floor_to_zero:
        motion[:, ROOT_Y_IDX] -= fitted_floor
        joints[..., 1] -= fitted_floor
        fitted_floor = 0.0

    # Contacts use both height and horizontal speed; median filtering removes flicker.
    feet = joints[:, list(FOOT_JOINTS)]
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    if T > 1:
        speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
    height = feet[..., 1] - fitted_floor
    contacts = (height <= float(cfg.contact_height_m)) & (speed <= float(cfg.contact_speed_mpf))
    if median_filter is not None and cfg.contact_median_size > 1:
        contacts = median_filter(
            contacts.astype(np.uint8),
            size=(int(cfg.contact_median_size), 1),
            mode="nearest",
        ).astype(bool)
    motion[:, 0:4] = contacts.astype(np.float32)

    # Fit error after final root localisation/floor shift must use the same transform.
    target_eval = target_pos.copy()
    if cfg.localize_root_xz:
        target_eval[..., 0] -= target_pos[0, 0, 0]
        target_eval[..., 2] -= target_pos[0, 0, 2]
    if cfg.floor_to_zero:
        target_eval[..., 1] -= float(
            np.percentile(target_eval[:, [t for t in FOOT_JOINTS if t in mapping], 1], 5)
        )
    pred = fk24_np(motion)
    per_frame = []
    for t in range(T):
        ids = np.where(mask[t] > 0.5)[0]
        if len(ids):
            per_frame.append(float(np.sqrt(np.mean((pred[t, ids] - target_eval[t, ids]) ** 2))))
    fit_arr = np.asarray(per_frame, dtype=np.float32)

    return motion, {
        "floor_y_after": float(fitted_floor),
        "contact_ratio": float(contacts.mean()),
        "fit_rmse_mean_m": float(fit_arr.mean()) if fit_arr.size else 0.0,
        "fit_rmse_p95_m": float(np.percentile(fit_arr, 95)) if fit_arr.size else 0.0,
        "root_orientation_contract": {
            "version": "v46_49_4_absolute_root_orientation_contract",
            "mode": "absolute_reference_lock"
            if cfg.root_orientation_lock
            else "unconstrained_ablation",
            "root_translation": "optimized",
            "root_orientation": "fixed_to_corrected_source_body_frame"
            if cfg.root_orientation_lock
            else "optimized",
            "local_joints_1_to_23": "optimized",
        },
        "chunk_reports": chunk_reports,
    }


def retarget_bvh(path: str | Path, cfg: Optional[RetargetConfig] = None):
    cfg = cfg or RetargetConfig.from_env()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    bvh = parse_bvh(path)
    native_pos, _ = source_fk(bvh, use_motion=True)
    rest_pos, _ = source_fk(
        BVHMotion(bvh.path, bvh.joints, np.zeros_like(bvh.values), bvh.frame_time),
        use_motion=False,
    )
    mapping = build_joint_mapping(bvh.joints)
    target_rest = target_rest_positions()

    # Use unique source joints for calibration to avoid duplicated hand fallbacks.
    pairs = []
    used_src = set()
    for tgt, src in mapping.items():
        if src in used_src or tgt in {22, 23}:
            continue
        used_src.add(src)
        pairs.append((tgt, src))
    X = np.asarray([rest_pos[0, src] for tgt, src in pairs], dtype=np.float32)
    Y = np.asarray([target_rest[tgt] for tgt, src in pairs], dtype=np.float32)
    W = np.asarray([TARGET_JOINT_WEIGHTS[tgt] for tgt, src in pairs], dtype=np.float32)
    scale, basis_R, trans = similarity_umeyama(X, Y, W)

    aligned = apply_similarity(native_pos, scale, basis_R, trans)
    aligned = resample_global_positions(aligned, bvh.fps, cfg.target_fps)
    aligned, heading_report = stabilize_source_heading_positions(
        aligned,
        mapping,
        float(cfg.target_fps),
    )
    motion, fit_report = fit_target_motion(aligned, mapping, cfg)
    fit_report["heading_contract"] = heading_report

    gravity = gravity_metrics_np(motion, cfg.target_fps)
    gravity_thresholds = GravityThresholds(
        torso_up_cos_p05_min=float(cfg.gravity_torso_p05_min),
    )
    gravity_ok, gravity_reasons = evaluate_gravity_contract(
        gravity,
        gravity_thresholds,
    )
    fit_ok = float(fit_report["fit_rmse_p95_m"]) <= float(cfg.fit_rmse_p95_max_m)
    ok = gravity_ok and fit_ok

    report = {
        "version": "v46_49_optimization_keypoint_retarget",
        "source": str(path),
        "source_fps": float(bvh.fps),
        "target_fps": float(cfg.target_fps),
        "source_frames": int(len(bvh.values)),
        "target_frames": int(len(motion)),
        "source_joint_count": int(len(bvh.joints)),
        "source_position_contract": {
            "version": "v46_49_2_nonroot_position_contract",
            "root_position": "retained",
            "nonroot_position_mode": str(
                os.environ.get("V46_49_NONROOT_POSITION_MODE", "ignore")
            ).strip().lower(),
            "hierarchy_offsets": "retained",
        },
        "mapping": {
            str(tgt): {
                "source_index": int(src),
                "source_name": bvh.joints[src].name,
            }
            for tgt, src in mapping.items()
        },
        "similarity": {
            "scale": float(scale),
            "basis_rotation": basis_R.tolist(),
            "translation": trans.tolist(),
            "det_basis_rotation": float(np.linalg.det(basis_R)),
        },
        "config": dataclasses.asdict(cfg),
        "fit": fit_report,
        "gravity": gravity,
        "gravity_thresholds": gravity_thresholds.to_dict(),
        "gravity_ok": bool(gravity_ok),
        "gravity_reasons": gravity_reasons,
        "fit_ok": bool(fit_ok),
        "ok": bool(ok),
    }
    if cfg.hard_gravity_gate and not ok:
        raise RuntimeError(
            f"Retarget contract failed for {path}: gravity={gravity_reasons}; "
            f"fit_rmse_p95={fit_report['fit_rmse_p95_m']:.4f} "
            f"(limit={cfg.fit_rmse_p95_max_m:.4f})"
        )
    return motion.astype(np.float32), report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow_failed_contract", action="store_true")
    args = ap.parse_args(argv)

    cfg = RetargetConfig.from_env()
    if args.device:
        cfg.device = args.device
    if args.allow_failed_contract:
        cfg.hard_gravity_gate = False

    motion, report = retarget_bvh(args.input, cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, motion)
    rp = Path(args.report) if args.report else out.with_suffix(".retarget.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "motion": str(out),
        "report": str(rp),
        "frames": int(len(motion)),
        "ok": report["ok"],
        "fit_rmse_p95_m": report["fit"]["fit_rmse_p95_m"],
        "gravity": report["gravity"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
