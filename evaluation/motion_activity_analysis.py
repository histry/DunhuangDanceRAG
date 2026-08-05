#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motion-activity analysis and anti-collapse acceptance for EDGE151 motion.

The EDGE151 contract is ``contact(4) + root_xyz(3) + 24 x Rot6D``.  Rotation
channels follow the repository's canonical column-concatenated convention,
``[R[:, 0], R[:, 1]]``.  The diagnostics are deliberately independent from
boundary, anatomy and severe-heading checks: those contracts remain immutable,
while this module prevents a physically safe but nearly static sequence from
being accepted as a valid whole-song result.

The module supports three research uses:

1. candidate-level activity assessment during route search;
2. stage-wise measurements for retrieval, refiner, diffusion and IK outputs;
3. final whole-song acceptance with per-slot music-density alignment.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from motion_geometry.rotations import (
        angular_velocity_np as _repository_angular_velocity,
        rot6d_to_matrix_np as _repository_rot6d_to_matrix,
    )
except Exception:  # Standalone diagnostic use outside a full checkout.
    _repository_angular_velocity = None
    _repository_rot6d_to_matrix = None

EDGE_DIM = 151
CONTACT_SLICE = slice(0, 4)
ROOT_SLICE = slice(4, 7)
ROT6D_SLICE = slice(7, 151)
NUM_JOINTS = 24
EPS = 1.0e-8


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _as_motion(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] < EDGE_DIM:
        raise ValueError(f"Expected EDGE151 motion [T,151], got {x.shape}")
    if not np.isfinite(x[:, :EDGE_DIM]).all():
        raise ValueError("Motion contains NaN or Inf values")
    return x[:, :EDGE_DIM].astype(np.float32, copy=False)


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    shape = x.shape
    flat = x.reshape(-1, 3, 3)
    u, _, vt = np.linalg.svd(flat)
    rotation = u @ vt
    negative = np.linalg.det(rotation) < 0.0
    if np.any(negative):
        u = u.copy()
        u[negative, :, -1] *= -1.0
        rotation = u @ vt
    return rotation.reshape(shape).astype(np.float32)


def _canonical_rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    """Decode column-concatenated Rot6D values with stable Gram--Schmidt."""

    x = np.asarray(values, dtype=np.float32)
    if x.shape[-1] != 6:
        raise ValueError(f"Rot6D last dimension must be 6, got {x.shape}")
    if _repository_rot6d_to_matrix is not None:
        return np.asarray(_repository_rot6d_to_matrix(x), dtype=np.float32)

    first = x[..., :3]
    second = x[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    basis1 = first / np.maximum(first_norm, EPS)
    second_orthogonal = second - np.sum(basis1 * second, axis=-1, keepdims=True) * basis1
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    basis2 = second_orthogonal / np.maximum(second_norm, EPS)
    basis3 = np.cross(basis1, basis2)
    matrix = np.stack([basis1, basis2, basis3], axis=-1)
    invalid = (
        ~np.isfinite(matrix).all(axis=(-2, -1))
        | (first_norm[..., 0] < 1.0e-7)
        | (second_norm[..., 0] < 1.0e-7)
    )
    if np.any(invalid):
        matrix = matrix.copy()
        matrix[invalid] = np.eye(3, dtype=np.float32)
    return _project_to_so3(matrix)


def _angular_velocity(rotations: np.ndarray, fps: float) -> np.ndarray:
    r = _project_to_so3(rotations)
    if r.shape[0] < 2:
        return np.zeros((0,) + r.shape[1:-2] + (3,), dtype=np.float32)
    if _repository_angular_velocity is not None:
        return np.asarray(
            _repository_angular_velocity(r, fps=float(fps)), dtype=np.float32
        )

    relative = np.swapaxes(r[:-1], -1, -2) @ r[1:]
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    vee = np.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        axis=-1,
    )
    sine = 0.5 * np.linalg.norm(vee, axis=-1)
    angle = np.arctan2(sine, cosine)
    axis = vee / np.maximum(2.0 * sine[..., None], 1.0e-7)
    rotvec = axis * angle[..., None]
    small = sine < 1.0e-6
    if np.any(small):
        rotvec = rotvec.copy()
        rotvec[small] = 0.5 * vee[small]
    return (rotvec * float(fps)).astype(np.float32)


def _longest_true_streak(mask: np.ndarray) -> Tuple[int, int, int]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    best_start = best_end = 0
    current_start: Optional[int] = None
    for index, value in enumerate(values):
        if value and current_start is None:
            current_start = index
        is_terminal = index == len(values) - 1
        if current_start is not None and ((not value) or is_terminal):
            current_end = index + 1 if value and is_terminal else index
            if current_end - current_start > best_end - best_start:
                best_start, best_end = current_start, current_end
            current_start = None
    return best_end - best_start, best_start, best_end


@dataclass(frozen=True)
class ActivityThresholds:
    """Numerical thresholds for candidate and whole-song activity contracts."""

    static_joint_speed_rad_s: float = 0.08
    static_root_horizontal_speed_m_s: float = 0.015
    joint_speed_reference_rad_s: float = 0.55
    root_speed_reference_m_s: float = 0.35

    candidate_required_target: float = 0.55
    candidate_max_static_ratio: float = 0.84
    candidate_max_static_seconds: float = 1.75
    candidate_min_joint_speed_rad_s: float = 0.065
    candidate_max_density_gap: float = 0.48
    candidate_penalty_weight: float = 1.25

    final_max_static_ratio: float = 0.72
    final_max_static_seconds: float = 4.0
    final_min_joint_speed_rad_s: float = 0.045
    final_min_root_travel_per_second_m_s: float = 0.006
    high_target_slot_threshold: float = 0.55
    high_target_slot_max_static_ratio: float = 0.75
    high_target_slot_max_density_gap: float = 0.50
    high_target_failed_slot_fraction: float = 0.34

    @classmethod
    def from_environment(cls) -> "ActivityThresholds":
        return cls(
            static_joint_speed_rad_s=max(
                0.0, _env_float("MOTION_ACTIVITY_STATIC_JOINT_RADPS", 0.08)
            ),
            static_root_horizontal_speed_m_s=max(
                0.0, _env_float("MOTION_ACTIVITY_STATIC_ROOT_MPS", 0.015)
            ),
            joint_speed_reference_rad_s=max(
                1.0e-6, _env_float("MOTION_ACTIVITY_JOINT_REFERENCE_RADPS", 0.55)
            ),
            root_speed_reference_m_s=max(
                1.0e-6, _env_float("MOTION_ACTIVITY_ROOT_REFERENCE_MPS", 0.35)
            ),
            candidate_required_target=float(
                np.clip(_env_float("MOTION_ACTIVITY_CANDIDATE_REQUIRED_TARGET", 0.55), 0.0, 1.0)
            ),
            candidate_max_static_ratio=float(
                np.clip(_env_float("MOTION_ACTIVITY_CANDIDATE_MAX_STATIC_RATIO", 0.84), 0.0, 1.0)
            ),
            candidate_max_static_seconds=max(
                0.0, _env_float("MOTION_ACTIVITY_CANDIDATE_MAX_STATIC_SECONDS", 1.75)
            ),
            candidate_min_joint_speed_rad_s=max(
                0.0, _env_float("MOTION_ACTIVITY_CANDIDATE_MIN_JOINT_RADPS", 0.065)
            ),
            candidate_max_density_gap=float(
                np.clip(_env_float("MOTION_ACTIVITY_CANDIDATE_MAX_DENSITY_GAP", 0.48), 0.0, 1.0)
            ),
            candidate_penalty_weight=max(
                0.0, _env_float("MOTION_ACTIVITY_CANDIDATE_PENALTY_WEIGHT", 1.25)
            ),
            final_max_static_ratio=float(
                np.clip(_env_float("MOTION_ACTIVITY_FINAL_MAX_STATIC_RATIO", 0.72), 0.0, 1.0)
            ),
            final_max_static_seconds=max(
                0.0, _env_float("MOTION_ACTIVITY_FINAL_MAX_STATIC_SECONDS", 4.0)
            ),
            final_min_joint_speed_rad_s=max(
                0.0, _env_float("MOTION_ACTIVITY_FINAL_MIN_JOINT_RADPS", 0.045)
            ),
            final_min_root_travel_per_second_m_s=max(
                0.0, _env_float("MOTION_ACTIVITY_FINAL_MIN_ROOT_TRAVEL_MPS", 0.006)
            ),
            high_target_slot_threshold=float(
                np.clip(_env_float("MOTION_ACTIVITY_HIGH_TARGET_THRESHOLD", 0.55), 0.0, 1.0)
            ),
            high_target_slot_max_static_ratio=float(
                np.clip(_env_float("MOTION_ACTIVITY_HIGH_TARGET_MAX_STATIC_RATIO", 0.75), 0.0, 1.0)
            ),
            high_target_slot_max_density_gap=float(
                np.clip(_env_float("MOTION_ACTIVITY_HIGH_TARGET_MAX_DENSITY_GAP", 0.50), 0.0, 1.0)
            ),
            high_target_failed_slot_fraction=float(
                np.clip(_env_float("MOTION_ACTIVITY_HIGH_TARGET_FAILED_FRACTION", 0.34), 0.0, 1.0)
            ),
        )


def motion_activity_metrics(
    motion: np.ndarray,
    fps: float = 30.0,
    thresholds: Optional[ActivityThresholds] = None,
    *,
    include_signals: bool = False,
) -> Dict[str, Any]:
    """Measure rotational, root and static-streak activity for one sequence."""

    x = _as_motion(motion)
    cfg = thresholds or ActivityThresholds.from_environment()
    fps = max(float(fps), 1.0e-6)
    frames = len(x)
    duration = frames / fps

    rotations = _canonical_rot6d_to_matrix(
        x[:, ROT6D_SLICE].reshape(frames, NUM_JOINTS, 6)
    )
    angular_velocity = _angular_velocity(rotations, fps=fps)
    if len(angular_velocity):
        per_joint_speed = np.linalg.norm(angular_velocity, axis=-1)
        joint_speed_step = per_joint_speed.mean(axis=1)
        joint_speed_frame = np.concatenate([joint_speed_step[:1], joint_speed_step])[:frames]
    else:
        per_joint_speed = np.zeros((0, NUM_JOINTS), dtype=np.float32)
        joint_speed_step = np.zeros((0,), dtype=np.float32)
        joint_speed_frame = np.zeros((frames,), dtype=np.float32)

    root = x[:, ROOT_SLICE]
    root_delta = np.diff(root, axis=0)
    root_step_3d = np.linalg.norm(root_delta, axis=-1)
    root_step_horizontal = np.linalg.norm(root_delta[:, [0, 2]], axis=-1)
    root_speed_horizontal_step = root_step_horizontal * fps
    if len(root_speed_horizontal_step):
        root_speed_frame = np.concatenate(
            [root_speed_horizontal_step[:1], root_speed_horizontal_step]
        )[:frames]
    else:
        root_speed_frame = np.zeros((frames,), dtype=np.float32)

    static_mask = (
        joint_speed_frame < cfg.static_joint_speed_rad_s
    ) & (
        root_speed_frame < cfg.static_root_horizontal_speed_m_s
    )
    streak_frames, streak_start, streak_end = _longest_true_streak(static_mask)

    normalized_joint = np.clip(
        joint_speed_frame / cfg.joint_speed_reference_rad_s, 0.0, 2.0
    )
    normalized_root = np.clip(
        root_speed_frame / cfg.root_speed_reference_m_s, 0.0, 2.0
    )
    density_signal = np.clip(
        0.78 * normalized_joint + 0.22 * normalized_root,
        0.0,
        1.0,
    ).astype(np.float32)

    report: Dict[str, Any] = {
        "shape": [int(frames), int(x.shape[1])],
        "fps": float(fps),
        "duration_seconds": float(duration),
        "root_total_travel_m": float(root_step_3d.sum()) if root_step_3d.size else 0.0,
        "root_horizontal_travel_m": float(root_step_horizontal.sum())
        if root_step_horizontal.size
        else 0.0,
        "root_travel_per_second_m_s": float(root_step_3d.sum() / max(duration, EPS))
        if root_step_3d.size
        else 0.0,
        "root_horizontal_speed_mean_m_s": float(root_speed_horizontal_step.mean())
        if root_speed_horizontal_step.size
        else 0.0,
        "root_horizontal_speed_p95_m_s": float(np.percentile(root_speed_horizontal_step, 95))
        if root_speed_horizontal_step.size
        else 0.0,
        "joint_speed_mean_rad_s": float(joint_speed_step.mean())
        if joint_speed_step.size
        else 0.0,
        "joint_speed_p95_rad_s": float(np.percentile(joint_speed_step, 95))
        if joint_speed_step.size
        else 0.0,
        "joint_speed_max_rad_s": float(joint_speed_step.max())
        if joint_speed_step.size
        else 0.0,
        "per_joint_speed_mean_rad_s": (
            per_joint_speed.mean(axis=0).astype(float).tolist()
            if per_joint_speed.size
            else [0.0] * NUM_JOINTS
        ),
        "static_frame_ratio": float(static_mask.mean()) if frames else 1.0,
        "static_frames": int(static_mask.sum()),
        "longest_static_streak_frames": int(streak_frames),
        "longest_static_streak_seconds": float(streak_frames / fps),
        "longest_static_streak_span": [int(streak_start), int(streak_end)],
        "motion_density_mean": float(density_signal.mean()) if frames else 0.0,
        "motion_density_p95": float(np.percentile(density_signal, 95)) if frames else 0.0,
        "contact_mean": x[:, CONTACT_SLICE].mean(axis=0).astype(float).tolist()
        if frames
        else [0.0] * 4,
        "thresholds": {
            "static_joint_speed_rad_s": float(cfg.static_joint_speed_rad_s),
            "static_root_horizontal_speed_m_s": float(
                cfg.static_root_horizontal_speed_m_s
            ),
        },
    }
    if include_signals:
        report["motion_density_signal"] = density_signal
        report["static_mask"] = static_mask
        report["joint_speed_signal_rad_s"] = joint_speed_frame.astype(np.float32)
        report["root_speed_signal_m_s"] = root_speed_frame.astype(np.float32)
    return report


def _numeric_slot_value(slot: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = slot.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            return float(np.clip(number, 0.0, 1.0))
    return None


def slot_activity_target(slot: Mapping[str, Any]) -> Optional[float]:
    """Infer the desired motion density from scheduler semantics.

    Priority:
    1. Explicit continuous motion-density or energy fields;
    2. Probability-weighted expectation over the scheduler's semantic classes;
    3. Coarse slot-role prior.

    ``1 - pose_hold`` is intentionally not used as an activity target. A low
    probability for one class only indicates that the slot is not a pose hold;
    it does not imply maximum motion intensity.
    """

    direct = _numeric_slot_value(
        slot,
        (
            "target_motion_density",
            "motion_density",
            "desired_motion_density",
            "music_motion_density",
            "motion_energy",
            "music_energy",
            "energy",
            "arousal",
        ),
    )
    if direct is not None:
        return direct

    probabilities: Optional[Mapping[str, Any]] = None
    for key in (
        "music_semantic_probs",
        "semantic_probs",
        "event_semantic_probs",
    ):
        value = slot.get(key)
        if isinstance(value, Mapping):
            probabilities = value
            break

    # Values represent relative motion-density priors, not class scores.
    semantic_weights = {
        # Current music-semantic taxonomy.
        "pose_hold": 0.05,
        "calm_meditative": 0.20,
        "lyrical_flow": 0.42,
        "instrument_phrase": 0.50,
        "percussive_accent": 0.72,
        "turning_climax": 0.90,
        "footwork_flow": 0.82,
        "aerial_curve": 0.68,

        # Backward-compatible aliases used by older schedules.
        "hold": 0.06,
        "calm": 0.20,
        "calm_flow": 0.30,
        "resolution": 0.32,
        "transition": 0.55,
        "motif_recall": 0.58,
        "turning_flow": 0.76,
        "turn": 0.78,
        "build_up": 0.80,
        "dynamic": 0.86,
        "climax": 0.96,
    }

    if probabilities is not None:
        weighted_sum = 0.0
        recognized_mass = 0.0

        for raw_key, raw_value in probabilities.items():
            try:
                probability = float(raw_value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(probability) or probability < 0.0:
                continue

            key = str(raw_key).strip().lower()
            if key not in semantic_weights:
                continue

            weighted_sum += probability * semantic_weights[key]
            recognized_mass += probability

        # Avoid normalizing numerical probability dust such as 4.9e-13.
        if recognized_mass >= 1.0e-6:
            return float(
                np.clip(weighted_sum / recognized_mass, 0.0, 1.0)
            )

    role_weights = {
        "pose_hold": 0.05,
        "hold": 0.06,
        "calm": 0.20,
        "intro": 0.25,
        "calm_flow": 0.30,
        "resolution": 0.32,
        "lyrical_flow": 0.42,
        "instrument_phrase": 0.50,
        "transition": 0.55,
        "motif_recall": 0.58,
        "aerial_curve": 0.68,
        "percussive_accent": 0.72,
        "build_up": 0.80,
        "footwork_flow": 0.82,
        "turning_flow": 0.86,
        "turning_climax": 0.90,
        "climax": 0.96,
    }

    for field in ("role", "phrase_role", "semantic_role", "event_role"):
        raw_role = slot.get(field)
        if raw_role is None:
            continue
        role = str(raw_role).strip().lower()
        if role in role_weights:
            return float(role_weights[role])

    return None

def candidate_activity_assessment(
    metrics: Mapping[str, Any],
    target_activity: Optional[float],
    thresholds: Optional[ActivityThresholds] = None,
) -> Dict[str, Any]:
    """Return additive route penalty and a conservative static-core hard gate."""

    cfg = thresholds or ActivityThresholds.from_environment()
    target = (
        None
        if target_activity is None
        else float(np.clip(float(target_activity), 0.0, 1.0))
    )
    measured = float(metrics.get("motion_density_mean", 0.0))
    density_gap = abs(measured - target) if target is not None else 0.0
    active_required = bool(
        target is not None and target >= cfg.candidate_required_target
    )
    reasons: List[str] = []
    if active_required:
        if float(metrics.get("static_frame_ratio", 1.0)) > cfg.candidate_max_static_ratio:
            reasons.append("candidate_static_frame_ratio")
        if float(metrics.get("longest_static_streak_seconds", 0.0)) > cfg.candidate_max_static_seconds:
            reasons.append("candidate_static_streak")
        if float(metrics.get("joint_speed_mean_rad_s", 0.0)) < cfg.candidate_min_joint_speed_rad_s:
            reasons.append("candidate_joint_speed")
        if density_gap > cfg.candidate_max_density_gap:
            reasons.append("candidate_music_density_gap")

    hard_reject = bool(
        _env_bool("MOTION_ACTIVITY_CANDIDATE_HARD_GATE", True)
        and active_required
        and len(reasons) >= 2
    )
    under_activity = max(0.0, (target or measured) - measured)
    penalty = cfg.candidate_penalty_weight * (
        under_activity
        + 0.20 * density_gap
        + 0.15 * float(metrics.get("static_frame_ratio", 0.0))
    )
    return {
        "target_activity": target,
        "measured_activity": measured,
        "activity_gap": float(density_gap),
        "active_motion_required": active_required,
        "penalty": float(penalty),
        "hard_reject": hard_reject,
        "reasons": reasons,
        "immutable_physical_anatomy_heading_gates_relaxed": False,
    }


def _slot_frame_count(slot: Mapping[str, Any], fps: float) -> int:
    for key in ("target_frames", "music_length"):
        value = slot.get(key)
        if value is not None:
            try:
                return max(1, int(round(float(value))))
            except Exception:
                pass
    start = slot.get("start_frame", slot.get("music_start"))
    end = slot.get("end_frame", slot.get("music_end"))
    if start is not None and end is not None:
        try:
            return max(1, int(round(float(end) - float(start))))
        except Exception:
            pass
    duration = slot.get("duration", slot.get("duration_sec", 1.0))
    try:
        return max(1, int(round(float(duration) * float(fps))))
    except Exception:
        return max(1, int(round(float(fps))))


def _slot_spans(
    slots: Sequence[Mapping[str, Any]],
    assembly_report: Optional[Sequence[Mapping[str, Any]]],
    total_frames: int,
    fps: float,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for index, slot in enumerate(slots):
        start: Optional[int] = None
        end: Optional[int] = None
        if assembly_report is not None and index < len(assembly_report):
            row = assembly_report[index]
            core_span = row.get("core_span")
            transition_span = row.get("transition_span")
            if isinstance(core_span, Sequence) and len(core_span) == 2:
                start = int(core_span[0])
                end = int(core_span[1])
                if isinstance(transition_span, Sequence) and len(transition_span) == 2:
                    start = min(start, int(transition_span[0]))
        if start is None or end is None:
            count = _slot_frame_count(slot, fps)
            start, end = cursor, cursor + count
        start = int(np.clip(start, 0, total_frames))
        end = int(np.clip(max(start + 1, end), 0, total_frames))
        spans.append((start, end))
        cursor = end
    if spans and spans[-1][1] < total_frames:
        spans[-1] = (spans[-1][0], total_frames)
    return spans


def per_slot_activity(
    motion: np.ndarray,
    slots: Sequence[Mapping[str, Any]],
    assembly_report: Optional[Sequence[Mapping[str, Any]]] = None,
    fps: float = 30.0,
    thresholds: Optional[ActivityThresholds] = None,
) -> List[Dict[str, Any]]:
    x = _as_motion(motion)
    cfg = thresholds or ActivityThresholds.from_environment()
    spans = _slot_spans(slots, assembly_report, len(x), fps)
    rows: List[Dict[str, Any]] = []
    for index, (start, end) in enumerate(spans):
        if end <= start:
            continue
        metrics = motion_activity_metrics(
            x[start:end], fps=fps, thresholds=cfg, include_signals=False
        )
        target = slot_activity_target(slots[index])
        measured = float(metrics["motion_density_mean"])
        gap = None if target is None else abs(float(target) - measured)
        high_target = bool(
            target is not None and target >= cfg.high_target_slot_threshold
        )
        failed = bool(
            high_target
            and (
                float(metrics["static_frame_ratio"])
                > cfg.high_target_slot_max_static_ratio
                or (gap is not None and gap > cfg.high_target_slot_max_density_gap)
            )
        )
        rows.append(
            {
                "slot": int(index),
                "frame_span": [int(start), int(end)],
                "target_activity": target,
                "measured_activity": measured,
                "activity_gap": gap,
                "high_activity_target": high_target,
                "high_activity_target_failed": failed,
                "static_frame_ratio": float(metrics["static_frame_ratio"]),
                "longest_static_streak_seconds": float(
                    metrics["longest_static_streak_seconds"]
                ),
                "joint_speed_mean_rad_s": float(metrics["joint_speed_mean_rad_s"]),
                "root_travel_per_second_m_s": float(
                    metrics["root_travel_per_second_m_s"]
                ),
            }
        )
    return rows


def motion_density_alignment(slot_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = [
        (float(row["target_activity"]), float(row["measured_activity"]))
        for row in slot_rows
        if row.get("target_activity") is not None
    ]
    if not pairs:
        return {
            "available": False,
            "slot_count": 0,
            "mae": None,
            "correlation": None,
        }
    target = np.asarray([row[0] for row in pairs], dtype=np.float64)
    measured = np.asarray([row[1] for row in pairs], dtype=np.float64)
    correlation: Optional[float]
    if len(pairs) >= 2 and target.std() > EPS and measured.std() > EPS:
        correlation = float(np.corrcoef(target, measured)[0, 1])
    else:
        correlation = None
    return {
        "available": True,
        "slot_count": int(len(pairs)),
        "mae": float(np.mean(np.abs(target - measured))),
        "correlation": correlation,
        "target_mean": float(target.mean()),
        "measured_mean": float(measured.mean()),
    }


def evaluate_final_motion_activity(
    motion: np.ndarray,
    slots: Optional[Sequence[Mapping[str, Any]]] = None,
    assembly_report: Optional[Sequence[Mapping[str, Any]]] = None,
    fps: float = 30.0,
    thresholds: Optional[ActivityThresholds] = None,
) -> Dict[str, Any]:
    """Reject whole-song outputs only when multiple collapse signals agree."""

    cfg = thresholds or ActivityThresholds.from_environment()
    metrics = motion_activity_metrics(motion, fps=fps, thresholds=cfg)
    slot_rows = per_slot_activity(
        motion,
        slots or [],
        assembly_report=assembly_report,
        fps=fps,
        thresholds=cfg,
    )
    alignment = motion_density_alignment(slot_rows)

    global_reasons: List[str] = []
    if float(metrics["static_frame_ratio"]) > cfg.final_max_static_ratio:
        global_reasons.append("final_static_frame_ratio")
    if float(metrics["longest_static_streak_seconds"]) > cfg.final_max_static_seconds:
        global_reasons.append("final_static_streak")
    if float(metrics["joint_speed_mean_rad_s"]) < cfg.final_min_joint_speed_rad_s:
        global_reasons.append("final_joint_speed")
    if (
        float(metrics["root_travel_per_second_m_s"])
        < cfg.final_min_root_travel_per_second_m_s
    ):
        global_reasons.append("final_root_travel")

    high_target = [row for row in slot_rows if row["high_activity_target"]]
    high_target_failed = [
        row for row in high_target if row["high_activity_target_failed"]
    ]
    failed_fraction = len(high_target_failed) / max(1, len(high_target))
    slot_failure = bool(
        high_target
        and failed_fraction > cfg.high_target_failed_slot_fraction
    )
    reasons = list(global_reasons)
    if slot_failure:
        reasons.append("high_activity_slot_failure_fraction")

    # A music-conditioned slot failure is an independent collapse signal.
    # When the failed fraction exceeds the configured threshold, the output
    # violates the requested activity profile even if loose global statistics
    # do not independently trigger two collapse indicators.
    collapse = bool(
        len(global_reasons) >= 2
        or slot_failure
    )
    gate_enabled = _env_bool("MOTION_ACTIVITY_FINAL_GATE", True)
    return {
        "schema": "edge151_motion_activity_acceptance",
        "ok": bool((not gate_enabled) or (not collapse)),
        "gate_enabled": bool(gate_enabled),
        "collapse_detected": bool(collapse),
        "reasons": reasons,
        "global_collapse_indicators": global_reasons,
        "whole_sequence": metrics,
        "per_slot": slot_rows,
        "motion_density_alignment": alignment,
        "high_activity_target_slots": int(len(high_target)),
        "failed_high_activity_target_slots": int(len(high_target_failed)),
        "failed_high_activity_target_fraction": float(failed_fraction),
        "thresholds": asdict(cfg),
        "physical_anatomy_severe_heading_gates_relaxed": False,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_activity_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(dict(report)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def save_stage_snapshot(
    output_motion_path: str | Path | None,
    stage_name: str,
    motion: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    """Persist a stage NPY and its compact activity metrics when enabled."""

    metrics = motion_activity_metrics(motion, fps=fps)
    result: Dict[str, Any] = {
        "stage": str(stage_name),
        "metrics": metrics,
        "snapshot_saved": False,
        "snapshot_path": None,
    }
    if not output_motion_path or not _env_bool("MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS", True):
        return result
    output = Path(output_motion_path)
    snapshot = output.with_name(f"{output.stem}.stage_{stage_name}.npy")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    np.save(snapshot, _as_motion(motion).astype(np.float32))
    result["snapshot_saved"] = True
    result["snapshot_path"] = str(snapshot)
    return result


def _load_slots_from_report(report_path: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    if not report_path:
        return [], None
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    slots = list(report.get("slots", []))
    stage_reports = report.get("stage_reports", {})
    assembly = stage_reports.get("closed_loop_concat")
    if assembly is None:
        assembly = report.get("event_heading_planner", {}).get("state_trace")
    return slots, assembly


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze EDGE151 motion activity and reject static collapse"
    )
    parser.add_argument("--input", required=True, help="EDGE151 .npy motion")
    parser.add_argument("--report", default=None, help="generation report JSON")
    parser.add_argument("--output", default=None, help="activity report JSON")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fail-on-collapse", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    motion_path = Path(args.input)
    motion = np.load(motion_path, allow_pickle=True)
    slots, assembly = _load_slots_from_report(args.report)
    result = evaluate_final_motion_activity(
        motion,
        slots=slots,
        assembly_report=assembly,
        fps=float(args.fps),
    )
    output = Path(args.output) if args.output else motion_path.with_name(
        motion_path.stem + ".motion_activity.json"
    )
    write_activity_report(result, output)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    if args.fail_on_collapse and not result["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
