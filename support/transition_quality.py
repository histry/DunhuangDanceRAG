#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V34 cross-boundary transition risk and absolute safety gate.

The historical filename is retained for import compatibility.  V34 evaluates
previous[-4:] + transition + following[:4], so entry/exit discontinuities can no
longer hide outside the transition-only window.  Candidate acceptance requires
both relative improvement and absolute physical safety.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from support.motion_geometry import (
    CONTACT,
    NUM_JOINTS,
    angular_velocity_np,
    motion_rotation_matrices_np,
    motion_to_joint_positions_np,
)

FOOT_JOINTS = (7, 8, 10, 11)


def _rms(values: np.ndarray) -> float:
    x = np.asarray(values, np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _ratio(value: float, reference: float, floor: float) -> float:
    return float(value / max(reference, floor))


def _rotation_step(motion: np.ndarray) -> np.ndarray:
    matrix = motion_rotation_matrices_np(motion)
    if len(matrix) < 2:
        return np.zeros((0, NUM_JOINTS), np.float32)
    with torch.no_grad():
        first = torch.from_numpy(matrix[:-1])
        second = torch.from_numpy(matrix[1:])
        relative = torch.matmul(first.transpose(-1, -2), second)
        angle = torch.linalg.norm(
            matrix_to_axis_angle(relative), dim=-1
        )
    return angle.cpu().numpy().astype(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _kinematic_contact_proxy(
    feet: np.ndarray,
    fps: float,
) -> np.ndarray:
    if len(feet) == 0:
        return np.zeros((0, 4), np.float32)
    height = feet[..., 1]
    velocity = np.diff(feet, axis=0, prepend=feet[:1]) * fps
    horizontal = np.linalg.norm(velocity[..., (0, 2)], axis=-1)
    ground = np.percentile(height, 5, axis=0)
    height_probability = _sigmoid(
        (ground[None] + 0.045 - height) / 0.010
    )
    speed_probability = _sigmoid((0.16 - horizontal) / 0.035)
    return (
        height_probability ** 0.70 * speed_probability ** 0.30
    ).astype(np.float32)


def _boundary_jerk_regions(
    values: np.ndarray,
    left_count: int,
    transition_count: int,
) -> Tuple[float, float, float]:
    """Return entry, exit and global maxima for third differences.

    A third-difference row i spans source frames i..i+3.  Rows whose span
    intersects either concatenation junction are included in the matching
    boundary region.
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    score = np.linalg.norm(values, axis=-1)
    if score.ndim > 1:
        score = score.mean(axis=-1)
    entry = left_count
    exit_ = left_count + transition_count
    indices = np.arange(len(score))
    entry_mask = (indices <= entry) & (indices + 3 >= entry - 1)
    exit_mask = (indices <= exit_) & (indices + 3 >= exit_ - 1)
    return (
        float(score[entry_mask].max()) if np.any(entry_mask) else 0.0,
        float(score[exit_mask].max()) if np.any(exit_mask) else 0.0,
        float(score.max()),
    )


def transition_risk(
    previous: np.ndarray,
    transition: np.ndarray,
    following: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, float]:
    prev = np.asarray(previous, np.float32)
    trans = np.asarray(transition, np.float32)
    nxt = np.asarray(following, np.float32)
    if len(trans) == 0:
        return {
            key: 1e9 for key in (
                "total", "entry_velocity", "exit_velocity",
                "entry_acceleration", "exit_acceleration",
                "joint_jerk", "angular_jerk", "boundary_joint_jerk_max",
                "entry_boundary_jerk", "exit_boundary_jerk",
                "boundary_angular_jerk_max", "foot_slip",
                "foot_penetration", "contact_switch",
                "max_rotation_step_rad", "entry_rotation_step_rad",
                "exit_rotation_step_rad", "entry_fk_jump", "exit_fk_jump",
                "high_frequency",
            )
        }

    prev_context = prev[-4:] if len(prev) >= 4 else prev
    next_context = nxt[:4] if len(nxt) >= 4 else nxt
    context = np.concatenate([prev_context, trans, next_context], axis=0)
    positions = motion_to_joint_positions_np(context)
    velocity = np.diff(positions, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps

    left = len(prev_context)
    right = left + len(trans)
    entry_step_index = max(left - 1, 0)
    exit_step_index = max(right - 1, 0)

    entry_velocity = (
        _rms(velocity[entry_step_index] - velocity[max(entry_step_index - 1, 0)])
        if len(velocity) else 0.0
    )
    exit_velocity = (
        _rms(
            velocity[min(exit_step_index, len(velocity) - 1)]
            - velocity[min(exit_step_index + 1, len(velocity) - 1)]
        ) if len(velocity) else 0.0
    )

    entry_acceleration = 0.0
    exit_acceleration = 0.0
    if len(acceleration):
        entry_acceleration = _rms(
            acceleration[min(max(entry_step_index - 1, 0), len(acceleration) - 1)]
            - acceleration[min(max(entry_step_index - 2, 0), len(acceleration) - 1)]
        )
        exit_acceleration = _rms(
            acceleration[min(max(exit_step_index - 1, 0), len(acceleration) - 1)]
            - acceleration[min(max(exit_step_index, 0), len(acceleration) - 1)]
        )

    entry_boundary_jerk, exit_boundary_jerk, boundary_jerk_max = (
        _boundary_jerk_regions(jerk, left, len(trans))
    )

    transition_positions = motion_to_joint_positions_np(trans)
    tv = np.diff(transition_positions, axis=0) * fps
    ta = np.diff(tv, axis=0) * fps
    tj = np.diff(ta, axis=0) * fps
    joint_jerk = _rms(tj)
    high_frequency = _rms(np.diff(tj, axis=0)) if len(tj) > 1 else joint_jerk

    context_angular = angular_velocity_np(context) * fps
    context_angular_acc = np.diff(context_angular, axis=0) * fps
    context_angular_jerk = np.diff(context_angular_acc, axis=0) * fps
    _, _, boundary_angular_jerk_max = _boundary_jerk_regions(
        context_angular_jerk, left, len(trans)
    )
    angular = angular_velocity_np(trans) * fps
    angular_acc = np.diff(angular, axis=0) * fps
    angular_jerk_values = np.diff(angular_acc, axis=0) * fps
    angular_jerk = _rms(angular_jerk_values)

    feet = transition_positions[:, FOOT_JOINTS]
    foot_velocity = np.diff(feet, axis=0, prepend=feet[:1]) * fps
    horizontal_speed = np.linalg.norm(
        foot_velocity[..., (0, 2)], axis=-1
    )
    predicted_contact = np.asarray(
        trans[:, CONTACT], np.float32
    ).clip(0.0, 1.0)
    kinematic_contact = _kinematic_contact_proxy(feet, fps)
    gate_contact = np.maximum(predicted_contact, kinematic_contact)
    foot_slip = float(
        np.sum(horizontal_speed * gate_contact)
        / max(float(gate_contact.sum()), 1e-6)
    )

    context_feet_y = positions[..., FOOT_JOINTS, 1]
    ground = float(np.percentile(context_feet_y, 5))
    penetration = np.maximum(ground - feet[..., 1] - 0.008, 0.0)
    foot_penetration = float(np.mean(penetration**2))
    contact_switch = (
        float(np.abs(np.diff(predicted_contact, axis=0)).mean())
        if len(predicted_contact) > 1 else 0.0
    )

    rotation_step = _rotation_step(context)
    max_rotation_step = float(np.max(rotation_step)) if rotation_step.size else 0.0
    entry_rotation_step = (
        float(np.max(rotation_step[entry_step_index]))
        if len(rotation_step) > entry_step_index else 0.0
    )
    exit_rotation_step = (
        float(np.max(rotation_step[exit_step_index]))
        if len(rotation_step) > exit_step_index else 0.0
    )
    entry_fk_jump = (
        _rms(positions[left] - positions[left - 1])
        if left > 0 and left < len(positions) else 0.0
    )
    exit_fk_jump = (
        _rms(positions[right] - positions[right - 1])
        if right > 0 and right < len(positions) else 0.0
    )

    total = (
        1.20 * entry_velocity
        + 1.60 * exit_velocity
        + 0.18 * entry_acceleration
        + 0.28 * exit_acceleration
        + 0.06 * joint_jerk
        + 0.03 * angular_jerk
        + 0.002 * boundary_jerk_max
        + 0.001 * boundary_angular_jerk_max
        + 2.00 * foot_slip
        + 6.00 * foot_penetration
        + 0.25 * contact_switch
        + 2.00 * max_rotation_step
        + 4.00 * exit_rotation_step
        + 3.00 * exit_fk_jump
        + 0.04 * high_frequency
    )
    return {
        "total": float(total),
        "entry_velocity": float(entry_velocity),
        "exit_velocity": float(exit_velocity),
        "entry_acceleration": float(entry_acceleration),
        "exit_acceleration": float(exit_acceleration),
        "joint_jerk": float(joint_jerk),
        "angular_jerk": float(angular_jerk),
        "entry_boundary_jerk": float(entry_boundary_jerk),
        "exit_boundary_jerk": float(exit_boundary_jerk),
        "boundary_joint_jerk_max": float(boundary_jerk_max),
        "boundary_angular_jerk_max": float(boundary_angular_jerk_max),
        "foot_slip": float(foot_slip),
        "foot_penetration": float(foot_penetration),
        "contact_switch": float(contact_switch),
        "max_rotation_step_rad": float(max_rotation_step),
        "entry_rotation_step_rad": float(entry_rotation_step),
        "exit_rotation_step_rad": float(exit_rotation_step),
        "entry_fk_jump": float(entry_fk_jump),
        "exit_fk_jump": float(exit_fk_jump),
        "high_frequency": float(high_frequency),
        "predicted_contact_rate": float(predicted_contact.mean()),
        "kinematic_contact_rate": float(kinematic_contact.mean()),
        "gate_contact_rate": float(gate_contact.mean()),
    }


def accept_candidate(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    max_total_ratio: float = 1.02,
    max_entry_ratio: float = 1.05,
    max_exit_ratio: float = 1.03,
    max_jerk_ratio: float = 1.03,
    max_foot_ratio: float = 1.02,
    max_penetration_ratio: float = 1.02,
    max_rotation_step_rad: float = 0.20,
    max_boundary_jerk_abs: float = 5000.0,
    max_boundary_angular_jerk_abs: float = 5000.0,
    max_entry_rotation_step_rad: float = 0.16,
    max_exit_rotation_step_rad: float = 0.12,
    max_entry_fk_jump: float = 0.060,
    max_exit_fk_jump: float = 0.040,
    max_exit_acceleration: float = 12.0,
) -> Tuple[bool, Dict[str, object]]:
    finite = all(np.isfinite(float(v)) for v in candidate.values())
    ratios = {
        "total": _ratio(candidate["total"], baseline["total"], 1e-5),
        "entry_velocity": _ratio(
            candidate["entry_velocity"], baseline["entry_velocity"], 1e-4
        ),
        "exit_velocity": _ratio(
            candidate["exit_velocity"], baseline["exit_velocity"], 1e-4
        ),
        "joint_jerk": _ratio(
            candidate["joint_jerk"], baseline["joint_jerk"], 1e-3
        ),
        "angular_jerk": _ratio(
            candidate["angular_jerk"], baseline["angular_jerk"], 1e-3
        ),
        "foot_slip": _ratio(
            candidate["foot_slip"], baseline["foot_slip"], 1e-4
        ),
        "foot_penetration": _ratio(
            candidate["foot_penetration"],
            baseline["foot_penetration"], 1e-7,
        ),
    }
    checks = {
        "finite": finite,
        "total_relative": ratios["total"] <= max_total_ratio,
        "entry_relative": ratios["entry_velocity"] <= max_entry_ratio,
        "exit_relative": ratios["exit_velocity"] <= max_exit_ratio,
        "joint_jerk_relative": ratios["joint_jerk"] <= max_jerk_ratio,
        "angular_jerk_relative": ratios["angular_jerk"] <= max_jerk_ratio,
        "foot_slip_relative": ratios["foot_slip"] <= max_foot_ratio,
        "penetration_relative": (
            ratios["foot_penetration"] <= max_penetration_ratio
        ),
        "rotation_step_absolute": (
            candidate["max_rotation_step_rad"] <= max_rotation_step_rad
        ),
        "entry_rotation_absolute": (
            candidate["entry_rotation_step_rad"] <= max_entry_rotation_step_rad
        ),
        "exit_rotation_absolute": (
            candidate["exit_rotation_step_rad"] <= max_exit_rotation_step_rad
        ),
        "entry_fk_absolute": candidate["entry_fk_jump"] <= max_entry_fk_jump,
        "exit_fk_absolute": candidate["exit_fk_jump"] <= max_exit_fk_jump,
        "exit_acceleration_absolute": (
            candidate["exit_acceleration"] <= max_exit_acceleration
        ),
        "boundary_jerk_absolute": (
            candidate["boundary_joint_jerk_max"] <= max_boundary_jerk_abs
        ),
        "boundary_angular_jerk_absolute": (
            candidate["boundary_angular_jerk_max"]
            <= max_boundary_angular_jerk_abs
        ),
    }
    accepted = bool(all(checks.values()))
    return accepted, {
        "accepted": accepted,
        "checks": checks,
        "ratios": ratios,
        "absolute_thresholds": {
            "max_boundary_jerk_abs": max_boundary_jerk_abs,
            "max_boundary_angular_jerk_abs": max_boundary_angular_jerk_abs,
            "max_entry_rotation_step_rad": max_entry_rotation_step_rad,
            "max_exit_rotation_step_rad": max_exit_rotation_step_rad,
            "max_entry_fk_jump": max_entry_fk_jump,
            "max_exit_fk_jump": max_exit_fk_jump,
            "max_exit_acceleration": max_exit_acceleration,
        },
    }
