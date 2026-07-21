#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.53.1 posture-aware anatomy and source/event safety contracts.

This module is API-compatible with ``tools.v46_52_anatomy_contract`` while
separating two decisions that must not be conflated in a 12-source low-resource
project:

1. source safety: reject only catastrophic numerical/kinematic failures;
2. event quality: retain the source, then remove or down-weight poor events.

EDGE 151D:
    [0:4] contacts, [4:7] root XYZ (Y-up), [7:151] 24 x column Rot6D.
"""
from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover
    Rotation = None

from contracts.gravity import (
    CONTACT,
    EDGE_DIM,
    FOOT_JOINTS,
    NUM_JOINTS,
    OFFSETS,
    PARENTS,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
    ROT6D_END,
    ROT6D_START,
    fk24_np,
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
)

PELVIS = 0
LH, RH = 1, 2
BELLY = 3
LK, RK = 4, 5
SPINE = 6
LA, RA = 7, 8
CHEST = 9
LTOE, RTOE = 10, 11
NECK = 12
LCOLLAR, RCOLLAR = 13, 14
HEAD = 15
LSHOULDER, RSHOULDER = 16, 17
LELBOW, RELBOW = 18, 19
LWRIST, RWRIST = 20, 21
LHAND, RHAND = 22, 23

SPINE_JOINTS = (BELLY, SPINE, CHEST, NECK)
UPPER_BODY_JOINTS = (
    BELLY, SPINE, CHEST, NECK, LCOLLAR, RCOLLAR,
    LSHOULDER, RSHOULDER, LELBOW, RELBOW, LWRIST, RWRIST,
)
LOWER_BODY_JOINTS = (LH, RH, LK, RK, LA, RA, LTOE, RTOE)

# Robust magnitude caps, not clinical Euler-angle limits.
DEFAULT_LOCAL_MAX_DEG: Dict[int, float] = {
    LH: 145.0, RH: 145.0, BELLY: 62.0, LK: 165.0, RK: 165.0,
    SPINE: 62.0, LA: 95.0, RA: 95.0, CHEST: 75.0,
    LTOE: 70.0, RTOE: 70.0, NECK: 75.0,
    LCOLLAR: 85.0, RCOLLAR: 85.0, HEAD: 80.0,
    LSHOULDER: 170.0, RSHOULDER: 170.0,
    LELBOW: 165.0, RELBOW: 165.0,
    LWRIST: 125.0, RWRIST: 125.0, LHAND: 105.0, RHAND: 105.0,
}

POSTURE_ORDER = {
    "floor_pose": 0,
    "kneeling": 1,
    "deep_squat": 2,
    "half_squat": 3,
    "standing": 4,
    "aerial": 5,
}


def env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _as_motion(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    if y.ndim == 3 and y.shape[0] == 1:
        y = y[0]
    if y.ndim != 2 or y.shape[-1] < EDGE_DIM:
        raise ValueError(f"Expected [T,{EDGE_DIM}], got {y.shape}")
    return y[:, :EDGE_DIM]


def _rest_positions() -> np.ndarray:
    p = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for j in range(NUM_JOINTS):
        parent = int(PARENTS[j])
        p[j] = np.zeros(3, np.float32) if parent < 0 else p[parent] + OFFSETS[j]
    return p


REST_POS = _rest_positions()
REST_HEAD_PELVIS = float(np.linalg.norm(REST_POS[HEAD] - REST_POS[PELVIS]))
REST_NECK_PELVIS = float(np.linalg.norm(REST_POS[NECK] - REST_POS[PELVIS]))
REST_BODY_HEIGHT = float(REST_POS[:, 1].max() - REST_POS[:, 1].min())
REST_LEG_LENGTH = float(
    0.5 * (
        np.linalg.norm(OFFSETS[LK]) + np.linalg.norm(OFFSETS[LA])
        + np.linalg.norm(OFFSETS[RK]) + np.linalg.norm(OFFSETS[RA])
    )
)


@dataclass
class AnatomyThresholds:
    """Event-level hard safety thresholds.

    Mild style-dependent limit excess is handled continuously through
    ``anatomy_quality`` and is not a source-level one-vote veto.
    """
    nonfinite_count_max: int = 0
    rot_orthogonality_p95_max: float = 2.0e-4
    rot_det_abs_error_p95_max: float = 2.0e-4
    local_angle_violation_ratio_max: float = 0.060
    local_angle_severe_ratio_max: float = 0.006
    spine_cumulative_angle_p95_max_rad: float = math.radians(170.0)
    torso_compression_ratio_p05_min: float = 0.46
    neck_compression_ratio_p05_min: float = 0.48
    self_collision_severe_ratio_max: float = 0.050
    knee_collapse_ratio_max: float = 0.045
    elbow_collapse_ratio_max: float = 0.060
    foot_penetration_p01_min_m: float = -0.080
    bone_length_drift_max: float = 2.0e-4
    anatomy_quality_min: float = 0.30

    @classmethod
    def from_env(cls) -> "AnatomyThresholds":
        return cls(
            nonfinite_count_max=env_int("V46_52_NONFINITE_MAX", 0),
            rot_orthogonality_p95_max=env_float("V46_52_ROT_ORTHO_P95_MAX", 2e-4),
            rot_det_abs_error_p95_max=env_float("V46_52_ROT_DET_P95_MAX", 2e-4),
            local_angle_violation_ratio_max=env_float("V46_52_EVENT_LOCAL_LIMIT_RATIO_MAX", 0.060),
            local_angle_severe_ratio_max=env_float("V46_52_EVENT_LOCAL_SEVERE_RATIO_MAX", 0.006),
            spine_cumulative_angle_p95_max_rad=math.radians(
                env_float("V46_52_EVENT_SPINE_CUM_P95_MAX_DEG", 170.0)
            ),
            torso_compression_ratio_p05_min=env_float("V46_52_EVENT_TORSO_RATIO_P05_MIN", 0.46),
            neck_compression_ratio_p05_min=env_float("V46_52_EVENT_NECK_RATIO_P05_MIN", 0.48),
            self_collision_severe_ratio_max=env_float("V46_52_EVENT_COLLISION_RATIO_MAX", 0.050),
            knee_collapse_ratio_max=env_float("V46_52_EVENT_KNEE_COLLAPSE_RATIO_MAX", 0.045),
            elbow_collapse_ratio_max=env_float("V46_52_EVENT_ELBOW_COLLAPSE_RATIO_MAX", 0.060),
            foot_penetration_p01_min_m=env_float("V46_52_FOOT_PENETRATION_P01_MIN_M", -0.080),
            bone_length_drift_max=env_float("V46_52_BONE_DRIFT_MAX", 2e-4),
            anatomy_quality_min=env_float("V46_52_EVENT_ANATOMY_HARD_QUALITY_MIN", 0.30),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SourceAnatomyThresholds:
    """Catastrophic source-level gate used before event slicing."""
    nonfinite_count_max: int = 0
    rot_orthogonality_p95_max: float = 3.0e-4
    rot_det_abs_error_p95_max: float = 3.0e-4
    local_angle_severe_ratio_max: float = 0.012
    spine_cumulative_angle_p95_max_rad: float = math.radians(185.0)
    torso_compression_ratio_p01_min: float = 0.34
    neck_compression_ratio_p01_min: float = 0.36
    self_collision_severe_ratio_max: float = 0.080
    foot_penetration_p01_min_m: float = -0.080
    bone_length_drift_max: float = 3.0e-4
    anatomy_quality_min: float = 0.18

    @classmethod
    def from_env(cls) -> "SourceAnatomyThresholds":
        return cls(
            nonfinite_count_max=env_int("V46_52_SOURCE_NONFINITE_MAX", 0),
            rot_orthogonality_p95_max=env_float("V46_52_SOURCE_ROT_ORTHO_P95_MAX", 3e-4),
            rot_det_abs_error_p95_max=env_float("V46_52_SOURCE_ROT_DET_P95_MAX", 3e-4),
            local_angle_severe_ratio_max=env_float("V46_52_SOURCE_LOCAL_SEVERE_RATIO_MAX", 0.012),
            spine_cumulative_angle_p95_max_rad=math.radians(
                env_float("V46_52_SOURCE_SPINE_CUM_P95_MAX_DEG", 185.0)
            ),
            torso_compression_ratio_p01_min=env_float("V46_52_SOURCE_TORSO_RATIO_P01_MIN", 0.34),
            neck_compression_ratio_p01_min=env_float("V46_52_SOURCE_NECK_RATIO_P01_MIN", 0.36),
            self_collision_severe_ratio_max=env_float("V46_52_SOURCE_COLLISION_RATIO_MAX", 0.080),
            foot_penetration_p01_min_m=env_float("V46_52_FOOT_PENETRATION_P01_MIN_M", -0.080),
            bone_length_drift_max=env_float("V46_52_SOURCE_BONE_DRIFT_MAX", 3e-4),
            anatomy_quality_min=env_float("V46_52_SOURCE_ANATOMY_QUALITY_MIN", 0.18),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class AnatomyLossWeights:
    local_limit: float = 3.0
    local_severe: float = 5.0
    spine: float = 4.0
    torso: float = 4.5
    bend: float = 1.2
    collision: float = 2.5
    symmetry: float = 0.0

    @classmethod
    def from_env(cls) -> "AnatomyLossWeights":
        return cls(
            local_limit=env_float("V46_52_LOSS_LOCAL_LIMIT_W", 3.0),
            local_severe=env_float("V46_52_LOSS_LOCAL_SEVERE_W", 5.0),
            spine=env_float("V46_52_LOSS_SPINE_W", 4.0),
            torso=env_float("V46_52_LOSS_TORSO_W", 4.5),
            bend=env_float("V46_52_LOSS_BEND_W", 1.2),
            collision=env_float("V46_52_LOSS_COLLISION_W", 2.5),
            symmetry=env_float("V46_52_LOSS_SYMMETRY_W", 0.0),
        )


def _matrix_angle_np(m: np.ndarray) -> np.ndarray:
    tr = np.trace(np.asarray(m), axis1=-2, axis2=-1)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(c).astype(np.float32)


def _joint_angle_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u = a - b
    v = c - b
    un = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-8)
    vn = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    return np.arccos(np.clip(np.sum(un * vn, axis=-1), -1.0, 1.0)).astype(np.float32)


def _point_segment_distance_np(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.sum((p - a) * ab, axis=-1) / np.maximum(np.sum(ab * ab, axis=-1), 1e-8)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[..., None] * ab
    return np.linalg.norm(p - q, axis=-1)


def _bone_length_drift(joints: np.ndarray) -> float:
    worst = 0.0
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        lengths = np.linalg.norm(joints[:, j] - joints[:, p], axis=-1)
        med = float(np.median(lengths))
        if med > 1e-8:
            worst = max(worst, float((np.max(lengths) - np.min(lengths)) / med))
    return worst


def posture_labels_np(motion: np.ndarray, joints: Optional[np.ndarray] = None) -> np.ndarray:
    x = _as_motion(motion)
    j = fk24_np(x) if joints is None else np.asarray(joints, dtype=np.float32)
    feet_y = j[:, list(FOOT_JOINTS), 1]
    floor = float(np.percentile(feet_y.reshape(-1), 5))
    pelvis_h = (j[:, PELVIS, 1] - floor) / max(REST_LEG_LENGTH, 1e-6)
    body_h = (j[:, :, 1].max(axis=1) - j[:, :, 1].min(axis=1)) / max(REST_BODY_HEIGHT, 1e-6)
    torso = j[:, HEAD] - j[:, PELVIS]
    torso_cos = torso[:, 1] / np.maximum(np.linalg.norm(torso, axis=-1), 1e-8)
    foot_clearance = np.min(feet_y - floor, axis=1)

    labels = np.full(len(x), "standing", dtype=object)
    labels[pelvis_h < 0.78] = "half_squat"
    labels[pelvis_h < 0.58] = "deep_squat"
    labels[pelvis_h < 0.38] = "kneeling"
    labels[(body_h < 0.62) | ((torso_cos < 0.26) & (pelvis_h < 0.62))] = "floor_pose"
    labels[foot_clearance > 0.10] = "aerial"
    return labels


def _mode_object(values: Sequence[Any]) -> Any:
    if not len(values):
        return "unknown"
    vals, counts = np.unique(np.asarray(values, dtype=object), return_counts=True)
    return vals[int(np.argmax(counts))]


def _base_limit_vector() -> np.ndarray:
    limits = np.full(NUM_JOINTS, np.inf, dtype=np.float32)
    for idx, deg in DEFAULT_LOCAL_MAX_DEG.items():
        limits[int(idx)] = math.radians(float(deg))
    return limits


def _posture_limit_slack(labels: np.ndarray) -> np.ndarray:
    """Per-frame cultural-pose slack in radians.

    Severe violations are still checked against the adaptive limit + 25 degrees.
    """
    T = len(labels)
    slack = np.zeros((T, NUM_JOINTS), dtype=np.float32)
    groups = {
        "half_squat": {
            (LH, RH): 8, (BELLY, SPINE, CHEST): 6, (LA, RA): 4,
        },
        "deep_squat": {
            (LH, RH): 18, (LK, RK): 8, (BELLY, SPINE, CHEST): 10,
            (LA, RA): 8, (LWRIST, RWRIST, LHAND, RHAND): 6,
        },
        "kneeling": {
            (LH, RH): 24, (LK, RK): 12, (BELLY, SPINE, CHEST): 14,
            (LA, RA): 12, (LWRIST, RWRIST, LHAND, RHAND): 8,
        },
        "floor_pose": {
            (LH, RH): 28, (LK, RK): 15, (BELLY, SPINE, CHEST, NECK): 18,
            (LA, RA): 12, (LCOLLAR, RCOLLAR, LWRIST, RWRIST, LHAND, RHAND): 10,
        },
        "aerial": {
            (LH, RH): 12, (LK, RK): 8, (LSHOULDER, RSHOULDER): 6,
        },
    }
    for posture, entries in groups.items():
        rows = np.where(labels == posture)[0]
        if not len(rows):
            continue
        for joints, degrees in entries.items():
            slack[np.ix_(rows, np.asarray(joints, dtype=np.int64))] = math.radians(float(degrees))
    return slack


def anatomy_metrics_np(motion: np.ndarray, fps: float = 30.0) -> Dict[str, Any]:
    x = _as_motion(motion)
    T = len(x)
    local = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6))
    joints = fk24_np(x)
    labels = posture_labels_np(x, joints)

    ident = np.eye(3, dtype=np.float32)
    ortho = local.transpose(0, 1, 3, 2) @ local
    ortho_err = np.linalg.norm(ortho - ident, axis=(-2, -1))
    det = np.linalg.det(local)
    det_err = np.abs(det - 1.0)
    local_angle = _matrix_angle_np(local)

    base_limits = _base_limit_vector()
    adaptive_limits = base_limits[None] + _posture_limit_slack(labels)
    finite = np.isfinite(adaptive_limits)
    raw_violations = (local_angle > base_limits[None]) & np.isfinite(base_limits)[None]
    violations = (local_angle > adaptive_limits) & finite
    severe = (local_angle > adaptive_limits + math.radians(25.0)) & finite

    spine_angles = local_angle[:, list(SPINE_JOINTS)].sum(axis=1)
    head_pelvis = np.linalg.norm(joints[:, HEAD] - joints[:, PELVIS], axis=-1)
    neck_pelvis = np.linalg.norm(joints[:, NECK] - joints[:, PELVIS], axis=-1)
    torso_ratio = head_pelvis / max(REST_HEAD_PELVIS, 1e-6)
    neck_ratio = neck_pelvis / max(REST_NECK_PELVIS, 1e-6)
    body_height = joints[:, :, 1].max(axis=1) - joints[:, :, 1].min(axis=1)
    body_ratio = body_height / max(REST_BODY_HEIGHT, 1e-6)
    torso_vec = joints[:, HEAD] - joints[:, PELVIS]
    torso_cos = torso_vec[:, 1] / np.maximum(np.linalg.norm(torso_vec, axis=-1), 1e-8)

    knee_l = _joint_angle_np(joints[:, LH], joints[:, LK], joints[:, LA])
    knee_r = _joint_angle_np(joints[:, RH], joints[:, RK], joints[:, RA])
    elbow_l = _joint_angle_np(joints[:, LSHOULDER], joints[:, LELBOW], joints[:, LWRIST])
    elbow_r = _joint_angle_np(joints[:, RSHOULDER], joints[:, RELBOW], joints[:, RWRIST])
    knee_collapse = np.minimum(knee_l, knee_r) < math.radians(15.0)
    elbow_collapse = np.minimum(elbow_l, elbow_r) < math.radians(9.0)

    torso_a, torso_b = joints[:, PELVIS], joints[:, NECK]
    wrist_torso = np.minimum(
        _point_segment_distance_np(joints[:, LWRIST], torso_a, torso_b),
        _point_segment_distance_np(joints[:, RWRIST], torso_a, torso_b),
    )
    elbow_torso = np.minimum(
        _point_segment_distance_np(joints[:, LELBOW], torso_a, torso_b),
        _point_segment_distance_np(joints[:, RELBOW], torso_a, torso_b),
    )
    knee_sep = np.linalg.norm(joints[:, LK] - joints[:, RK], axis=-1)
    wrist_head = np.minimum(
        np.linalg.norm(joints[:, LWRIST] - joints[:, HEAD], axis=-1),
        np.linalg.norm(joints[:, RWRIST] - joints[:, HEAD], axis=-1),
    )
    severe_collision = (
        ((wrist_torso < 0.030) & (elbow_torso < 0.050))
        | (knee_sep < 0.040)
        | (wrist_head < 0.020)
    )

    feet = joints[:, list(FOOT_JOINTS)]
    support_floor = float(np.percentile(feet[..., 1], 5))
    floor_mode = str(os.environ.get("V46_52_FLOOR_REFERENCE_MODE", "stage_zero")).strip().lower()
    stage_floor = support_floor if floor_mode in {"auto", "auto_quantile", "sequence_quantile"} else env_float(
        "V46_52_STAGE_FLOOR_Y_M", 0.0
    )
    penetration = feet[..., 1] - stage_floor
    pelvis_h = (joints[:, PELVIS, 1] - support_floor) / max(REST_LEG_LENGTH, 1e-6)

    violation_ratio = float(violations.mean())
    raw_violation_ratio = float(raw_violations.mean())
    severe_ratio = float(severe.mean())
    quality_penalty = (
        4.5 * violation_ratio
        + 8.0 * severe_ratio
        + 1.5 * max(0.0, 0.52 - float(np.percentile(torso_ratio, 5)))
        + 2.0 * float(severe_collision.mean())
        + 0.35 * max(0.0, float(np.percentile(spine_angles, 95)) - math.radians(135.0))
    )
    quality = float(np.clip(math.exp(-quality_penalty), 0.0, 1.0))

    return {
        "schema": "v46_53_1_posture_aware_anatomy_contract",
        "frames": int(T),
        "fps": float(fps),
        "nonfinite_count": int((~np.isfinite(x)).sum()),
        "rot_orthogonality_p95": float(np.percentile(ortho_err, 95)),
        "rot_orthogonality_max": float(np.max(ortho_err)),
        "rot_det_abs_error_p95": float(np.percentile(det_err, 95)),
        "rot_det_min": float(np.min(det)),
        "raw_local_angle_violation_ratio": raw_violation_ratio,
        "local_angle_violation_ratio": violation_ratio,
        "local_angle_severe_ratio": severe_ratio,
        "local_angle_p95_deg": float(np.degrees(np.percentile(local_angle, 95))),
        "local_angle_max_deg": float(np.degrees(np.max(local_angle))),
        "spine_cumulative_angle_p95_rad": float(np.percentile(spine_angles, 95)),
        "spine_cumulative_angle_p95_deg": float(np.degrees(np.percentile(spine_angles, 95))),
        "spine_cumulative_angle_max_deg": float(np.degrees(np.max(spine_angles))),
        "torso_compression_ratio_p01": float(np.percentile(torso_ratio, 1)),
        "torso_compression_ratio_p05": float(np.percentile(torso_ratio, 5)),
        "torso_compression_ratio_median": float(np.median(torso_ratio)),
        "neck_compression_ratio_p01": float(np.percentile(neck_ratio, 1)),
        "neck_compression_ratio_p05": float(np.percentile(neck_ratio, 5)),
        "body_height_ratio_p05": float(np.percentile(body_ratio, 5)),
        "body_height_ratio_median": float(np.median(body_ratio)),
        "torso_up_cos_p05": float(np.percentile(torso_cos, 5)),
        "torso_up_cos_median": float(np.median(torso_cos)),
        "pelvis_height_norm_p05": float(np.percentile(pelvis_h, 5)),
        "pelvis_height_norm_median": float(np.median(pelvis_h)),
        "pelvis_height_norm_p95": float(np.percentile(pelvis_h, 95)),
        "knee_angle_p05_deg": float(np.degrees(np.percentile(np.minimum(knee_l, knee_r), 5))),
        "elbow_angle_p05_deg": float(np.degrees(np.percentile(np.minimum(elbow_l, elbow_r), 5))),
        "knee_collapse_ratio": float(knee_collapse.mean()),
        "elbow_collapse_ratio": float(elbow_collapse.mean()),
        "self_collision_severe_ratio": float(severe_collision.mean()),
        "wrist_torso_distance_p01_m": float(np.percentile(wrist_torso, 1)),
        "knee_separation_p01_m": float(np.percentile(knee_sep, 1)),
        "floor_y": support_floor,
        "support_floor_y": support_floor,
        "stage_floor_y": float(stage_floor),
        "floor_reference_mode": floor_mode,
        "foot_penetration_p01_m": float(np.percentile(penetration, 1)),
        "foot_penetration_min_m": float(np.min(penetration)),
        "bone_length_drift_max": float(_bone_length_drift(joints)),
        "posture_entry": str(labels[0]) if len(labels) else "unknown",
        "posture_exit": str(labels[-1]) if len(labels) else "unknown",
        "posture_mode": str(_mode_object(labels)),
        "posture_distribution": {
            str(k): float(v / max(1, len(labels)))
            for k, v in zip(*np.unique(labels, return_counts=True))
        },
        "anatomy_quality": quality,
    }


def _check_high(metrics: Mapping[str, Any], name: str, limit: float, reasons: List[str]) -> None:
    value = float(metrics.get(name, float("inf")))
    if not np.isfinite(value) or value > float(limit):
        reasons.append(f"{name}={value:.6g} > {float(limit):.6g}")


def _check_low(metrics: Mapping[str, Any], name: str, limit: float, reasons: List[str]) -> None:
    value = float(metrics.get(name, -float("inf")))
    if not np.isfinite(value) or value < float(limit):
        reasons.append(f"{name}={value:.6g} < {float(limit):.6g}")


def evaluate_anatomy_contract_detailed(
    metrics: Mapping[str, Any],
    thresholds: Optional[AnatomyThresholds] = None,
) -> Dict[str, Any]:
    th = thresholds or AnatomyThresholds.from_env()
    hard: List[str] = []
    soft: List[str] = []
    _check_high(metrics, "nonfinite_count", th.nonfinite_count_max, hard)
    _check_high(metrics, "rot_orthogonality_p95", th.rot_orthogonality_p95_max, hard)
    _check_high(metrics, "rot_det_abs_error_p95", th.rot_det_abs_error_p95_max, hard)
    _check_high(metrics, "local_angle_severe_ratio", th.local_angle_severe_ratio_max, hard)
    _check_high(metrics, "spine_cumulative_angle_p95_rad", th.spine_cumulative_angle_p95_max_rad, hard)
    _check_low(metrics, "torso_compression_ratio_p05", th.torso_compression_ratio_p05_min, hard)
    _check_high(metrics, "self_collision_severe_ratio", th.self_collision_severe_ratio_max, hard)
    _check_high(metrics, "knee_collapse_ratio", th.knee_collapse_ratio_max, hard)
    _check_high(metrics, "elbow_collapse_ratio", th.elbow_collapse_ratio_max, hard)
    _check_low(metrics, "foot_penetration_p01_m", th.foot_penetration_p01_min_m, hard)
    _check_high(metrics, "bone_length_drift_max", th.bone_length_drift_max, hard)
    _check_low(metrics, "anatomy_quality", th.anatomy_quality_min, hard)

    _check_high(metrics, "local_angle_violation_ratio", th.local_angle_violation_ratio_max, soft)
    _check_low(metrics, "neck_compression_ratio_p05", th.neck_compression_ratio_p05_min, soft)
    return {
        "hard_ok": not hard,
        "hard_reasons": hard,
        "soft_ok": not soft,
        "soft_reasons": soft,
        "thresholds": th.to_dict(),
    }


def evaluate_anatomy_contract(
    metrics: Mapping[str, Any],
    thresholds: Optional[AnatomyThresholds] = None,
) -> Tuple[bool, List[str]]:
    result = evaluate_anatomy_contract_detailed(metrics, thresholds)
    return bool(result["hard_ok"]), list(result["hard_reasons"])


def evaluate_source_anatomy_contract(
    metrics: Mapping[str, Any],
    thresholds: Optional[SourceAnatomyThresholds] = None,
) -> Tuple[bool, List[str]]:
    th = thresholds or SourceAnatomyThresholds.from_env()
    reasons: List[str] = []
    _check_high(metrics, "nonfinite_count", th.nonfinite_count_max, reasons)
    _check_high(metrics, "rot_orthogonality_p95", th.rot_orthogonality_p95_max, reasons)
    _check_high(metrics, "rot_det_abs_error_p95", th.rot_det_abs_error_p95_max, reasons)
    _check_high(metrics, "local_angle_severe_ratio", th.local_angle_severe_ratio_max, reasons)
    _check_high(metrics, "spine_cumulative_angle_p95_rad", th.spine_cumulative_angle_p95_max_rad, reasons)
    _check_low(metrics, "torso_compression_ratio_p01", th.torso_compression_ratio_p01_min, reasons)
    _check_low(metrics, "neck_compression_ratio_p01", th.neck_compression_ratio_p01_min, reasons)
    _check_high(metrics, "self_collision_severe_ratio", th.self_collision_severe_ratio_max, reasons)
    _check_low(metrics, "foot_penetration_p01_m", th.foot_penetration_p01_min_m, reasons)
    _check_high(metrics, "bone_length_drift_max", th.bone_length_drift_max, reasons)
    _check_low(metrics, "anatomy_quality", th.anatomy_quality_min, reasons)
    return not reasons, reasons


def event_anatomy_features(motion: np.ndarray, fps: float = 30.0) -> Dict[str, Any]:
    x = _as_motion(motion)
    joints = fk24_np(x)
    labels = posture_labels_np(x, joints)
    feet = joints[:, list(FOOT_JOINTS), 1]
    floor = float(np.percentile(feet, 5))
    pelvis_norm = (joints[:, PELVIS, 1] - floor) / max(REST_LEG_LENGTH, 1e-6)
    body_norm = (
        joints[:, :, 1].max(axis=1) - joints[:, :, 1].min(axis=1)
    ) / max(REST_BODY_HEIGHT, 1e-6)
    metrics = anatomy_metrics_np(x, fps=fps)
    detail = evaluate_anatomy_contract_detailed(metrics)
    edge_n = max(1, min(len(x), env_int("V46_52_EVENT_EDGE_FRAMES", 6)))
    return {
        "anatomy_valid": bool(detail["hard_ok"]),
        "anatomy_hard_valid": bool(detail["hard_ok"]),
        "anatomy_soft_valid": bool(detail["soft_ok"]),
        "anatomy_reasons": list(detail["hard_reasons"]),
        "anatomy_soft_reasons": list(detail["soft_reasons"]),
        "anatomy_quality": float(metrics["anatomy_quality"]),
        "posture_entry": str(_mode_object(labels[:edge_n])),
        "posture_exit": str(_mode_object(labels[-edge_n:])),
        "posture_mode": str(_mode_object(labels)),
        "posture_distribution": dict(metrics["posture_distribution"]),
        "pelvis_height_entry_norm": float(np.median(pelvis_norm[:edge_n])),
        "pelvis_height_exit_norm": float(np.median(pelvis_norm[-edge_n:])),
        "pelvis_height_median_norm": float(np.median(pelvis_norm)),
        "body_height_entry_norm": float(np.median(body_norm[:edge_n])),
        "body_height_exit_norm": float(np.median(body_norm[-edge_n:])),
        "body_height_median_norm": float(np.median(body_norm)),
        "entry_floor_offset_m": float(np.median(feet[:edge_n]) - floor),
        "exit_floor_offset_m": float(np.median(feet[-edge_n:]) - floor),
        "torso_compression_ratio_p05": float(metrics["torso_compression_ratio_p05"]),
        "local_angle_violation_ratio": float(metrics["local_angle_violation_ratio"]),
        "raw_local_angle_violation_ratio": float(metrics["raw_local_angle_violation_ratio"]),
        "local_angle_severe_ratio": float(metrics["local_angle_severe_ratio"]),
        "self_collision_severe_ratio": float(metrics["self_collision_severe_ratio"]),
        "spine_cumulative_angle_p95_deg": float(metrics["spine_cumulative_angle_p95_deg"]),
    }


def posture_distance(a: str, b: str) -> int:
    return abs(int(POSTURE_ORDER.get(str(a), 3)) - int(POSTURE_ORDER.get(str(b), 3)))


def transition_anatomy_risk(
    previous: np.ndarray,
    transition: np.ndarray,
    following: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, Any]:
    p = _as_motion(previous)
    b = _as_motion(transition) if len(transition) else np.zeros((0, EDGE_DIM), np.float32)
    f = _as_motion(following)
    context = np.concatenate([p[-8:], b, f[:8]], axis=0)
    m = anatomy_metrics_np(context, fps=fps)
    pfeat = event_anatomy_features(p[-min(len(p), 12):], fps=fps)
    ffeat = event_anatomy_features(f[:min(len(f), 12)], fps=fps)
    posture_gap = posture_distance(pfeat["posture_exit"], ffeat["posture_entry"])
    pelvis_gap = abs(float(pfeat["pelvis_height_exit_norm"]) - float(ffeat["pelvis_height_entry_norm"]))
    body_gap = abs(float(pfeat["body_height_exit_norm"]) - float(ffeat["body_height_entry_norm"]))
    floor_gap = abs(float(pfeat["exit_floor_offset_m"]) - float(ffeat["entry_floor_offset_m"]))
    transition_seconds = len(b) / max(float(fps), 1e-8)
    required = 0.20 + 0.30 * posture_gap + 0.90 * max(0.0, pelvis_gap - 0.08)
    hard = (
        not bool(ffeat["anatomy_hard_valid"])
        or pelvis_gap > env_float("V46_52_PELVIS_GAP_HARD", 0.34)
        or (posture_gap >= 3 and transition_seconds + 1e-6 < required)
    )
    score = (
        env_float("V46_52_RISK_PELVIS_W", 2.5) * pelvis_gap
        + env_float("V46_52_RISK_BODY_W", 1.0) * body_gap
        + env_float("V46_52_RISK_POSTURE_W", 0.45) * posture_gap
        + env_float("V46_52_RISK_FLOOR_W", 4.0) * floor_gap
        + env_float("V46_52_RISK_ANATOMY_W", 2.0) * (1.0 - float(m["anatomy_quality"]))
    )
    return {
        "anatomy_quality": float(m["anatomy_quality"]),
        "anatomy_valid": bool(evaluate_anatomy_contract(m)[0]),
        "posture_exit": pfeat["posture_exit"],
        "posture_entry": ffeat["posture_entry"],
        "posture_gap": int(posture_gap),
        "pelvis_height_gap_norm": float(pelvis_gap),
        "body_height_gap_norm": float(body_gap),
        "floor_offset_gap_m": float(floor_gap),
        "required_transition_seconds": float(required),
        "available_transition_seconds": float(transition_seconds),
        "anatomy_risk_score": float(score),
        "anatomy_hard_reject": bool(hard),
        "context_metrics": m,
    }


def align_core_floor_np(
    previous: Optional[np.ndarray],
    core: np.ndarray,
    fps: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    c = _as_motion(core).copy()
    if previous is None or not len(previous) or not len(c):
        return c, {"enabled": False, "reason": "no_previous"}
    p = _as_motion(previous)
    n = max(2, min(12, len(p), len(c)))
    jp = fk24_np(p[-n:])
    jc = fk24_np(c[:n])
    prev_floor = float(np.percentile(jp[:, list(FOOT_JOINTS), 1], 10))
    curr_floor = float(np.percentile(jc[:, list(FOOT_JOINTS), 1], 10))
    delta = float(np.clip(prev_floor - curr_floor, -0.08, 0.08))
    c[:, ROOT_Y_IDX] += delta
    return c, {
        "enabled": True,
        "mode": "support_floor_only",
        "delta_root_y": delta,
        "prev_floor": prev_floor,
        "curr_floor_before": curr_floor,
        "curr_floor_after": curr_floor + delta,
        "pelvis_height_forced": False,
    }


def _rotvec_from_matrix_np(m: np.ndarray) -> np.ndarray:
    shape = m.shape[:-2]
    flat = np.asarray(m, dtype=np.float64).reshape(-1, 3, 3)
    if Rotation is not None:
        return Rotation.from_matrix(flat).as_rotvec().reshape(*shape, 3).astype(np.float32)
    angle = _matrix_angle_np(flat)
    skew = np.stack(
        [flat[:, 2, 1] - flat[:, 1, 2], flat[:, 0, 2] - flat[:, 2, 0], flat[:, 1, 0] - flat[:, 0, 1]],
        axis=-1,
    )
    denom = np.maximum(2.0 * np.sin(angle)[:, None], 1e-6)
    return (skew / denom * angle[:, None]).reshape(*shape, 3).astype(np.float32)


def _matrix_from_rotvec_np(v: np.ndarray) -> np.ndarray:
    shape = v.shape[:-1]
    flat = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    if Rotation is not None:
        return Rotation.from_rotvec(flat).as_matrix().reshape(*shape, 3, 3).astype(np.float32)
    theta = np.linalg.norm(flat, axis=-1)
    axis = flat / np.maximum(theta[:, None], 1e-8)
    K = np.zeros((len(flat), 3, 3), dtype=np.float64)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    I = np.eye(3, dtype=np.float64)[None]
    out = I + np.sin(theta)[:, None, None] * K + (1.0 - np.cos(theta))[:, None, None] * (K @ K)
    out[theta < 1e-8] = I
    return out.reshape(*shape, 3, 3).astype(np.float32)


def geodesic_c2_bridge_np(
    previous: np.ndarray,
    following: np.ndarray,
    frames: int,
    fps: float = 30.0,
) -> np.ndarray:
    frames = int(frames)
    if frames <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    p = _as_motion(previous)
    f = _as_motion(following)
    a, z = p[-1], f[0]
    u = np.arange(1, frames + 1, dtype=np.float32) / float(frames + 1)
    s = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3

    out = np.zeros((frames, EDGE_DIM), dtype=np.float32)
    split = frames // 2
    out[:split, CONTACT] = (a[CONTACT] >= 0.5).astype(np.float32)
    out[split:, CONTACT] = (z[CONTACT] >= 0.5).astype(np.float32)

    p0 = a[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    p1 = z[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    v0 = (p[-1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - p[-2, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]) * float(fps) if len(p) > 1 else np.zeros(3, np.float32)
    v1 = (f[1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - f[0, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]) * float(fps) if len(f) > 1 else np.zeros(3, np.float32)
    max_vel = env_float("V46_52_BRIDGE_ROOT_VEL_CLIP_MPS", 2.4)
    duration_seconds = float(frames + 1) / max(float(fps), 1.0e-8)
    v0 = np.clip(v0, -max_vel, max_vel) * duration_seconds
    v1 = np.clip(v1, -max_vel, max_vel) * duration_seconds
    h00, h10 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u
    h01, h11 = -2*u**3 + 3*u**2, u**3 - u**2
    root = h00[:, None]*p0 + h10[:, None]*v0 + h01[:, None]*p1 + h11[:, None]*v1
    root[:, 1] = np.clip(root[:, 1], min(float(p0[1]), float(p1[1])) - 0.03, max(float(p0[1]), float(p1[1])) + 0.03)
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root

    r0 = rot6d_to_matrix_np(a[ROT6D_START:ROT6D_END].reshape(NUM_JOINTS, 6))
    r1 = rot6d_to_matrix_np(z[ROT6D_START:ROT6D_END].reshape(NUM_JOINTS, 6))
    delta = _rotvec_from_matrix_np(np.swapaxes(r0, -1, -2) @ r1)
    mats = r0[None] @ _matrix_from_rotvec_np(s[:, None, None] * delta[None])
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(mats).reshape(frames, -1)
    return out.astype(np.float32)


def frame_anomaly_score_np(motion: np.ndarray) -> np.ndarray:
    x = _as_motion(motion)
    T = len(x)
    local = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6))
    joints = fk24_np(x)
    labels = posture_labels_np(x, joints)
    angles = _matrix_angle_np(local)
    limits = _base_limit_vector()[None] + _posture_limit_slack(labels)
    excess = np.maximum(angles - limits, 0.0)
    torso_ratio = np.linalg.norm(joints[:, HEAD] - joints[:, PELVIS], axis=-1) / max(REST_HEAD_PELVIS, 1e-6)
    spine = angles[:, list(SPINE_JOINTS)].sum(axis=1)
    return (
        excess.mean(axis=1) * 4.0
        + np.maximum(0.0, 0.52 - torso_ratio) * 4.0
        + np.maximum(0.0, spine - math.radians(145.0))
    ).astype(np.float32)


def _matrix_angle_torch(m: "torch.Tensor") -> "torch.Tensor":
    tr = torch.diagonal(m, dim1=-2, dim2=-1).sum(dim=-1)
    c = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(c)


def _joint_angle_torch(a: "torch.Tensor", b: "torch.Tensor", c: "torch.Tensor") -> "torch.Tensor":
    u = F.normalize(a - b, dim=-1, eps=1e-8)
    v = F.normalize(c - b, dim=-1, eps=1e-8)
    return torch.acos((u * v).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def _point_segment_distance_torch(
    p: "torch.Tensor", a: "torch.Tensor", b: "torch.Tensor"
) -> "torch.Tensor":
    ab = b - a
    t = ((p - a) * ab).sum(dim=-1) / ab.square().sum(dim=-1).clamp_min(1e-8)
    t = t.clamp(0.0, 1.0)
    return torch.linalg.norm(p - (a + t.unsqueeze(-1) * ab), dim=-1)


def anatomy_losses_torch(
    joints: "torch.Tensor",
    local_matrices: "torch.Tensor",
    weights: Optional[AnatomyLossWeights] = None,
) -> Dict[str, "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("PyTorch is required for anatomy losses")
    w = weights or AnatomyLossWeights.from_env()
    angles = _matrix_angle_torch(local_matrices)

    base = torch.full((NUM_JOINTS,), float("inf"), device=angles.device, dtype=angles.dtype)
    for idx, deg in DEFAULT_LOCAL_MAX_DEG.items():
        base[int(idx)] = math.radians(float(deg))

    pelvis, head = joints[:, PELVIS], joints[:, HEAD]
    torso_cos = F.normalize(head - pelvis, dim=-1, eps=1e-8)[:, 1]
    knee_l = _joint_angle_torch(joints[:, LH], joints[:, LK], joints[:, LA])
    knee_r = _joint_angle_torch(joints[:, RH], joints[:, RK], joints[:, RA])
    min_knee = torch.minimum(knee_l, knee_r)
    low_pose_gate = torch.sigmoid((0.62 - torso_cos) * 8.0)
    bend_gate = torch.sigmoid((math.radians(80.0) - min_knee) * 5.0)
    style_gate = torch.clamp(torch.maximum(low_pose_gate, bend_gate), 0.0, 1.0)

    slack_deg = torch.zeros(NUM_JOINTS, device=angles.device, dtype=angles.dtype)
    for ids, deg in (
        ((LH, RH), 22.0), ((LK, RK), 10.0), ((BELLY, SPINE, CHEST, NECK), 14.0),
        ((LA, RA), 10.0), ((LWRIST, RWRIST, LHAND, RHAND), 8.0),
    ):
        slack_deg[list(ids)] = float(deg)
    limits = base.view(1, -1) + style_gate[:, None] * torch.deg2rad(slack_deg).view(1, -1)
    finite = torch.isfinite(limits)
    excess = torch.where(finite, F.relu(angles - limits), torch.zeros_like(angles))
    severe = torch.where(finite, F.relu(angles - limits - math.radians(25.0)), torch.zeros_like(angles))
    local_limit = excess.square().mean()
    local_severe = severe.square().mean()

    spine_sum = angles[:, list(SPINE_JOINTS)].sum(dim=-1)
    spine_cap = math.radians(125.0) + style_gate * math.radians(25.0)
    spine = F.relu(spine_sum - spine_cap).square().mean()

    hp = torch.linalg.norm(joints[:, HEAD] - joints[:, PELVIS], dim=-1)
    npel = torch.linalg.norm(joints[:, NECK] - joints[:, PELVIS], dim=-1)
    torso_floor = (0.58 - 0.10 * style_gate) * REST_HEAD_PELVIS
    neck_floor = (0.58 - 0.08 * style_gate) * REST_NECK_PELVIS
    torso = (
        F.relu(torso_floor - hp).square().mean()
        + F.relu(neck_floor - npel).square().mean()
    )

    elbow_l = _joint_angle_torch(joints[:, LSHOULDER], joints[:, LELBOW], joints[:, LWRIST])
    elbow_r = _joint_angle_torch(joints[:, RSHOULDER], joints[:, RELBOW], joints[:, RWRIST])
    bend = (
        F.relu(math.radians(15.0) - knee_l).square().mean()
        + F.relu(math.radians(15.0) - knee_r).square().mean()
        + F.relu(math.radians(9.0) - elbow_l).square().mean()
        + F.relu(math.radians(9.0) - elbow_r).square().mean()
    )

    torso_a, torso_b = joints[:, PELVIS], joints[:, NECK]
    wrist_dist = torch.minimum(
        _point_segment_distance_torch(joints[:, LWRIST], torso_a, torso_b),
        _point_segment_distance_torch(joints[:, RWRIST], torso_a, torso_b),
    )
    elbow_dist = torch.minimum(
        _point_segment_distance_torch(joints[:, LELBOW], torso_a, torso_b),
        _point_segment_distance_torch(joints[:, RELBOW], torso_a, torso_b),
    )
    knee_sep = torch.linalg.norm(joints[:, LK] - joints[:, RK], dim=-1)
    collision = (
        (F.relu(0.030 - wrist_dist) * F.relu(0.050 - elbow_dist)).mean()
        + F.relu(0.040 - knee_sep).square().mean()
    )

    symmetry_raw = (
        (angles[:, LH] - angles[:, RH]).square()
        + (angles[:, LK] - angles[:, RK]).square()
        + (angles[:, LSHOULDER] - angles[:, RSHOULDER]).square()
    )
    # Only suppress simultaneous bilateral collapse; preserve expressive asymmetry.
    collapse_gate = (1.0 - style_gate).detach()
    symmetry = (symmetry_raw * collapse_gate).mean()

    total = (
        w.local_limit * local_limit
        + w.local_severe * local_severe
        + w.spine * spine
        + w.torso * torso
        + w.bend * bend
        + w.collision * collision
        + w.symmetry * symmetry
    )
    return {
        "total": total,
        "local_limit": local_limit,
        "local_severe": local_severe,
        "spine": spine,
        "torso": torso,
        "bend": bend,
        "collision": collision,
        "symmetry": symmetry,
        "style_gate_mean": style_gate.mean(),
    }
