#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 motion geometry utilities for EDGE 151D motions.

Representation:
    [0:4]   foot contacts
    [4:7]   root xyz
    [7:151] 24 local joint rotations in continuous 6D representation

This module centralizes all geometry-sensitive operations used by the V29
research pipeline.  In particular, rotations are never interpolated directly
in raw 6D coordinates.  They are converted to SO(3), interpolated or filtered
there, and projected back to valid 6D rotations.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)
from motion_geometry.rotations import (
    matrix_to_rot6d_torch as matrix_to_rotation_6d,
    rot6d_to_matrix_torch as rotation_6d_to_matrix,
)
from motion_geometry.smpl24 import (
    CONTACT,
    FOOT_JOINTS,
    JOINT_NAMES as SMPL_JOINT_NAMES,
    MOTION_DIM,
    NUM_JOINTS,
    OFFSETS as SMPL_OFFSETS,
    PARENTS as SMPL_PARENTS,
    ROOT,
    ROOT_X_IDX as ROOT_X,
    ROOT_Y_IDX as ROOT_Y,
    ROOT_Z_IDX as ROOT_Z,
    ROT6D_END,
    ROT6D_START,
)

ROT = slice(ROT6D_START, ROT6D_END)
ROOT_ROT6D = slice(ROT6D_START, ROT6D_START + 6)


def smootherstep01(x):
    if torch.is_tensor(x):
        y = x.clamp(0.0, 1.0)
        return y * y * y * (y * (y * 6.0 - 15.0) + 10.0)
    y = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return y * y * y * (y * (y * 6.0 - 15.0) + 10.0)


def _validate_motion_np(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != MOTION_DIM:
        raise ValueError(f"Expected [T,{MOTION_DIM}], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Motion contains NaN or Inf")
    return x


def project_motion_rotations_torch(motion: torch.Tensor) -> torch.Tensor:
    """Project raw 6D rotation channels onto the valid SO(3) manifold."""
    if motion.shape[-1] != MOTION_DIM:
        raise ValueError(f"Expected last dim {MOTION_DIM}, got {motion.shape}")
    rot = motion[..., ROT].reshape(*motion.shape[:-1], NUM_JOINTS, 6)
    matrix = rotation_6d_to_matrix(rot)
    rot6d = matrix_to_rotation_6d(matrix).reshape(*motion.shape[:-1], NUM_JOINTS * 6)
    out = motion.clone()
    out[..., ROT] = rot6d
    return out


def project_motion_rotations_np(motion: np.ndarray) -> np.ndarray:
    x = _validate_motion_np(motion)
    with torch.no_grad():
        out = project_motion_rotations_torch(torch.from_numpy(x))
    return out.cpu().numpy().astype(np.float32)


def motion_rotation_matrices_torch(motion: torch.Tensor) -> torch.Tensor:
    if motion.shape[-1] != MOTION_DIM:
        raise ValueError(f"Expected last dim {MOTION_DIM}, got {motion.shape}")
    return rotation_6d_to_matrix(
        motion[..., ROT].reshape(*motion.shape[:-1], NUM_JOINTS, 6)
    )


def motion_rotation_matrices_np(motion: np.ndarray) -> np.ndarray:
    x = _validate_motion_np(motion)
    with torch.no_grad():
        result = motion_rotation_matrices_torch(torch.from_numpy(x))
    return result.cpu().numpy().astype(np.float32)


def motion_to_joint_positions_torch(motion: torch.Tensor) -> torch.Tensor:
    """Differentiable 24-joint forward kinematics for EDGE motions."""
    if motion.shape[-1] != MOTION_DIM:
        raise ValueError(f"Expected last dim {MOTION_DIM}, got {motion.shape}")
    rotations_local = motion_rotation_matrices_torch(motion)
    root_positions = motion[..., ROOT]
    offsets = torch.as_tensor(SMPL_OFFSETS, device=motion.device, dtype=motion.dtype)

    positions_world = []
    rotations_world = []
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations_local[..., joint, :, :])
        else:
            parent_rotation = rotations_world[parent]
            offset = offsets[joint]
            rotated_offset = torch.matmul(
                parent_rotation, offset.reshape(3, 1)
            ).squeeze(-1)
            positions_world.append(positions_world[parent] + rotated_offset)
            rotations_world.append(
                torch.matmul(parent_rotation, rotations_local[..., joint, :, :])
            )
    return torch.stack(positions_world, dim=-2)


def motion_to_joint_positions_np(motion: np.ndarray) -> np.ndarray:
    x = _validate_motion_np(motion)
    with torch.no_grad():
        pos = motion_to_joint_positions_torch(torch.from_numpy(x))
    return pos.cpu().numpy().astype(np.float32)


def geodesic_rotation_error_torch(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return per-frame, per-joint SO(3) angular errors in radians."""
    pred_m = motion_rotation_matrices_torch(predicted)
    target_m = motion_rotation_matrices_torch(target)
    relative = torch.matmul(pred_m.transpose(-1, -2), target_m)
    return torch.linalg.vector_norm(matrix_to_axis_angle(relative), dim=-1)


def angular_velocity_torch(motion: torch.Tensor) -> torch.Tensor:
    """Local angular velocity vectors in radians/frame, shape [..., T-1, J, 3]."""
    matrices = motion_rotation_matrices_torch(motion)
    if matrices.shape[-4] < 2:
        return matrices.new_zeros((*matrices.shape[:-4], 0, NUM_JOINTS, 3))
    relative = torch.matmul(
        matrices[..., :-1, :, :, :].transpose(-1, -2),
        matrices[..., 1:, :, :, :],
    )
    return matrix_to_axis_angle(relative)


def angular_velocity_np(motion: np.ndarray) -> np.ndarray:
    x = _validate_motion_np(motion)
    with torch.no_grad():
        vel = angular_velocity_torch(torch.from_numpy(x))
    return vel.cpu().numpy().astype(np.float32)


def _so3_hermite_rotations(
    r0: torch.Tensor,
    r1: torch.Tensor,
    start_velocity: torch.Tensor,
    end_velocity: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """Cubic Hermite interpolation in the tangent space at the first endpoint."""
    k = max(0, int(length))
    if k == 0:
        return r0.new_zeros((0, *r0.shape))
    relative = torch.matmul(r0.transpose(-1, -2), r1)
    delta = matrix_to_axis_angle(relative)
    scale = float(k + 1)

    # Limit endpoint tangents to avoid large Hermite overshoot.
    max_tangent = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) + 0.35
    m0 = start_velocity * scale

    # ``end_velocity`` is a body tangent at R1, whereas the Hermite polynomial
    # lives in log coordinates at R0.  Map the requested endpoint derivative
    # through the inverse SO(3) right Jacobian so the generated curve has the
    # intended physical angular velocity at u=1.
    theta = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    x, y, z = delta.unbind(dim=-1)
    zero = torch.zeros_like(x)
    hat = torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*delta.shape[:-1], 3, 3)
    theta2 = theta * theta
    small = theta < 1e-4
    coefficient = torch.where(
        small,
        1.0 / 12.0 + theta2 / 720.0,
        1.0 / theta2.clamp_min(1e-8)
        - 1.0
        / (
            2.0
            * theta.clamp_min(1e-8)
            * torch.tan(0.5 * theta).clamp_min(1e-8)
        ),
    )
    identity = torch.eye(3, dtype=delta.dtype, device=delta.device)
    identity = identity.expand(*delta.shape[:-1], 3, 3)
    right_jacobian_inverse = (
        identity + 0.5 * hat + coefficient[..., None] * torch.matmul(hat, hat)
    )
    m1 = torch.matmul(
        right_jacobian_inverse,
        (end_velocity * scale).unsqueeze(-1),
    )[..., 0]
    m0_norm = torch.linalg.vector_norm(m0, dim=-1, keepdim=True).clamp_min(1e-8)
    m1_norm = torch.linalg.vector_norm(m1, dim=-1, keepdim=True).clamp_min(1e-8)
    m0 = m0 * torch.minimum(torch.ones_like(m0_norm), max_tangent / m0_norm)
    m1 = m1 * torch.minimum(torch.ones_like(m1_norm), max_tangent / m1_norm)

    u = torch.linspace(
        1.0 / (k + 1), k / (k + 1), k,
        device=r0.device, dtype=r0.dtype,
    ).reshape(k, *([1] * (delta.ndim)))
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    tangent = h10 * m0.unsqueeze(0) + h01 * delta.unsqueeze(0) + h11 * m1.unsqueeze(0)

    # A final norm cap prevents pathological random pseudo-pairs from wrapping.
    delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-8)
    cap = delta_norm.unsqueeze(0) + 0.5
    tangent = tangent * torch.minimum(torch.ones_like(tangent_norm), cap / tangent_norm)
    return torch.matmul(r0.unsqueeze(0), axis_angle_to_matrix(tangent))


def _clip_vector_norm_torch(value: torch.Tensor, maximum: float) -> torch.Tensor:
    limit = max(float(maximum), 0.0)
    if limit <= 0.0:
        return torch.zeros_like(value)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-8)
    return value * torch.minimum(torch.ones_like(norm), value.new_tensor(limit) / norm)


def _clip_root_velocity_np(
    velocity: np.ndarray,
    *,
    fps: float,
    horizontal_speed_cap_mps: float,
    vertical_speed_cap_mps: float,
) -> np.ndarray:
    value = np.asarray(velocity, dtype=np.float32).copy()
    rate = max(float(fps), 1e-6)
    horizontal = value[[0, 2]]
    horizontal_norm = float(np.linalg.norm(horizontal))
    horizontal_limit = max(float(horizontal_speed_cap_mps), 0.0) / rate
    if horizontal_norm > horizontal_limit > 0.0:
        value[[0, 2]] *= horizontal_limit / max(horizontal_norm, 1e-8)
    elif horizontal_limit <= 0.0:
        value[[0, 2]] = 0.0
    value[1] = float(
        np.clip(
            value[1],
            -max(float(vertical_speed_cap_mps), 0.0) / rate,
            max(float(vertical_speed_cap_mps), 0.0) / rate,
        )
    )
    return value


def make_so3_transition(
    prev: np.ndarray,
    nxt: np.ndarray,
    length: int,
    *,
    fps: float = 30.0,
    angular_speed_cap_radps: float = 8.0,
    root_horizontal_speed_cap_mps: float = 1.5,
    root_vertical_speed_cap_mps: float = 0.9,
    root_tangent_margin_m: float = 0.12,
) -> np.ndarray:
    """Generate an endpoint-velocity-aware SO(3) transition.

    The first argument may contain one or more preceding frames; the second may
    contain one or more following frames.  Endpoint angular and root velocities
    are estimated when context frames are available.
    """
    a = _validate_motion_np(prev)
    b = _validate_motion_np(nxt)
    k = max(0, int(length))
    if k == 0:
        return np.zeros((0, MOTION_DIM), dtype=np.float32)

    with torch.no_grad():
        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)
        a_rot = motion_rotation_matrices_torch(a_t)
        b_rot = motion_rotation_matrices_torch(b_t)
        r0 = a_rot[-1]
        r1 = b_rot[0]

        if len(a) >= 2:
            start_v = matrix_to_axis_angle(
                torch.matmul(a_rot[-2].transpose(-1, -2), a_rot[-1])
            )
        else:
            start_v = torch.zeros((NUM_JOINTS, 3), dtype=a_t.dtype)
        if len(b) >= 2:
            end_v = matrix_to_axis_angle(
                torch.matmul(b_rot[0].transpose(-1, -2), b_rot[1])
            )
        else:
            end_v = torch.zeros((NUM_JOINTS, 3), dtype=b_t.dtype)
        per_frame_angular_cap = max(float(angular_speed_cap_radps), 0.0) / max(
            float(fps), 1e-6
        )
        start_v = _clip_vector_norm_torch(start_v, per_frame_angular_cap)
        end_v = _clip_vector_norm_torch(end_v, per_frame_angular_cap)

        rotations = _so3_hermite_rotations(r0, r1, start_v, end_v, k)
        rot6d = matrix_to_rotation_6d(rotations).reshape(k, NUM_JOINTS * 6)

    u = np.linspace(1.0 / (k + 1), k / (k + 1), k, dtype=np.float32)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    root0 = a[-1, ROOT]
    root1 = b[0, ROOT]
    root_v0 = a[-1, ROOT] - a[-2, ROOT] if len(a) >= 2 else np.zeros((3,), np.float32)
    root_v1 = b[1, ROOT] - b[0, ROOT] if len(b) >= 2 else np.zeros((3,), np.float32)
    root_v0 = _clip_root_velocity_np(
        root_v0,
        fps=fps,
        horizontal_speed_cap_mps=root_horizontal_speed_cap_mps,
        vertical_speed_cap_mps=root_vertical_speed_cap_mps,
    )
    root_v1 = _clip_root_velocity_np(
        root_v1,
        fps=fps,
        horizontal_speed_cap_mps=root_horizontal_speed_cap_mps,
        vertical_speed_cap_mps=root_vertical_speed_cap_mps,
    )
    root_delta = root1 - root0
    tangent_limit = float(np.linalg.norm(root_delta)) + max(
        float(root_tangent_margin_m), 0.0
    )
    for velocity in (root_v0, root_v1):
        tangent = velocity * float(k + 1)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > tangent_limit > 0.0:
            velocity *= tangent_limit / max(tangent_norm, 1e-8)
        elif tangent_limit <= 0.0:
            velocity[:] = 0.0
    roots = (
        h00[:, None] * root0[None]
        + h10[:, None] * float(k + 1) * root_v0[None]
        + h01[:, None] * root1[None]
        + h11[:, None] * float(k + 1) * root_v1[None]
    )

    contact_alpha = smootherstep01(u)[:, None]
    contacts = np.where(
        contact_alpha < 0.5,
        a[-1, CONTACT][None],
        b[0, CONTACT][None],
    )

    out = np.zeros((k, MOTION_DIM), dtype=np.float32)
    out[:, CONTACT] = contacts
    out[:, ROOT] = roots
    out[:, ROT] = rot6d.cpu().numpy().astype(np.float32)
    return project_motion_rotations_np(out)


def canonicalize_event_root_np(
    motion: np.ndarray,
    *,
    target_floor_y: float = 0.0,
    floor_quantile: float = 5.0,
    max_floor_penetration_m: float = 0.005,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Localize one event in XZ and align its robust foot floor in Y.

    Only a constant root translation is applied.  Relative stage travel,
    crouches, kneels, jumps, and every joint rotation remain unchanged.
    """

    from contracts.gravity import fk24_np

    x = _validate_motion_np(motion).copy()
    if len(x) == 0:
        return x, {
            "source_floor_y_m": float(target_floor_y),
            "target_floor_y_m": float(target_floor_y),
            "root_y_offset_m": 0.0,
            "root_xz_origin_m": [0.0, 0.0],
        }
    origin = x[0, [ROOT_X, ROOT_Z]].astype(np.float32).copy()
    x[:, ROOT_X] -= float(origin[0])
    x[:, ROOT_Z] -= float(origin[1])
    joints = fk24_np(x)
    foot_y = joints[:, list(FOOT_JOINTS), 1]
    robust_floor = float(
        np.percentile(foot_y, np.clip(float(floor_quantile), 0.0, 25.0))
    )
    # A robust percentile is less sensitive to a single corrupt foot sample,
    # but using it without a lower-envelope guard can leave an event already
    # below the stage floor.  Limit that residual with one constant Root-Y
    # translation; the event's internal crouch/jump profile is untouched.
    minimum_floor = float(np.min(foot_y))
    source_floor = min(
        robust_floor,
        minimum_floor + max(float(max_floor_penetration_m), 0.0),
    )
    offset = float(target_floor_y) - source_floor
    x[:, ROOT_Y] += offset
    return x.astype(np.float32), {
        "source_floor_y_m": source_floor,
        "robust_floor_y_m": robust_floor,
        "minimum_foot_y_m": minimum_floor,
        "target_floor_y_m": float(target_floor_y),
        "max_floor_penetration_m": float(max_floor_penetration_m),
        "root_y_offset_m": offset,
        "root_xz_origin_m": [float(origin[0]), float(origin[1])],
        "relative_root_xz_travel_m": float(
            np.linalg.norm(x[-1, [ROOT_X, ROOT_Z]] - x[0, [ROOT_X, ROOT_Z]])
        ),
    }


def compose_event_root_xz_np(
    motion: np.ndarray,
    start_xz: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Translate a localized event so its first XZ equals ``start_xz``."""

    x = _validate_motion_np(motion).copy()
    start = np.asarray(start_xz, dtype=np.float32).reshape(2)
    if len(x):
        delta = start - x[0, [ROOT_X, ROOT_Z]]
        x[:, ROOT_X] += float(delta[0])
        x[:, ROOT_Z] += float(delta[1])
    return x.astype(np.float32), {
        "stage_start_xz_m": start.tolist(),
        "stage_end_xz_m": (
            x[-1, [ROOT_X, ROOT_Z]].astype(float).tolist()
            if len(x)
            else start.astype(float).tolist()
        ),
    }


def event_endpoint_geometry_np(
    motion: np.ndarray,
    *,
    floor_quantile: float = 5.0,
    window_frames: int = 5,
) -> Dict[str, float]:
    """Return floor-relative endpoint state for routing edge decisions."""

    from contracts.gravity import fk24_np

    x = _validate_motion_np(motion)
    if len(x) == 0:
        return {
            "floor_y_m": 0.0,
            "entry_floor_relative_m": 0.0,
            "exit_floor_relative_m": 0.0,
            "entry_root_height_m": 0.0,
            "exit_root_height_m": 0.0,
        }
    joints = fk24_np(x)
    foot_y = joints[:, list(FOOT_JOINTS), 1]
    quantile = np.clip(float(floor_quantile), 0.0, 25.0)
    floor = float(np.percentile(foot_y, quantile))
    count = max(1, min(int(window_frames), len(x)))
    entry_floor = float(np.percentile(foot_y[:count], quantile)) - floor
    exit_floor = float(np.percentile(foot_y[-count:], quantile)) - floor
    return {
        "floor_y_m": floor,
        "entry_floor_relative_m": entry_floor,
        "exit_floor_relative_m": exit_floor,
        "entry_root_height_m": float(np.median(x[:count, ROOT_Y]) - floor),
        "exit_root_height_m": float(np.median(x[-count:, ROOT_Y]) - floor),
    }


def project_transition_floor_np(
    transition: np.ndarray,
    *,
    target_floor_y: float = 0.0,
    clearance_m: float = 0.002,
    smoothing_frames: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Raise only penetrating bridge frames using a conservative smooth lift."""

    from contracts.gravity import fk24_np

    x = _validate_motion_np(transition).copy()
    if len(x) == 0:
        return x, {
            "applied": False,
            "max_root_y_correction_m": 0.0,
            "penetration_before_m": 0.0,
            "penetration_after_m": 0.0,
        }
    feet = fk24_np(x)[:, list(FOOT_JOINTS)]
    minimum = feet[..., 1].min(axis=1)
    floor = float(target_floor_y) + max(float(clearance_m), 0.0)
    required = np.maximum(floor - minimum, 0.0).astype(np.float32)
    correction = required.copy()
    window = max(1, int(smoothing_frames))
    if window % 2 == 0:
        window += 1
    if window > 1 and len(x) > 1:
        radius = window // 2
        grid = np.arange(-radius, radius + 1, dtype=np.float32)
        sigma = max(float(radius) / 1.5, 0.8)
        kernel = np.exp(-0.5 * np.square(grid / sigma))
        kernel /= np.sum(kernel)
        padded = np.pad(required, (radius, radius), mode="constant")
        smoothed = np.convolve(padded, kernel, mode="valid").astype(np.float32)
        # Never smooth below the exact correction needed to satisfy the floor.
        correction = np.maximum(required, smoothed)
    x[:, ROOT_Y] += correction
    feet_after = fk24_np(x)[:, list(FOOT_JOINTS)]
    residual = np.maximum(floor - feet_after[..., 1].min(axis=1), 0.0)
    if np.any(residual > 1e-6):
        x[:, ROOT_Y] += residual.astype(np.float32)
        correction += residual.astype(np.float32)
        feet_after = fk24_np(x)[:, list(FOOT_JOINTS)]
    return x.astype(np.float32), {
        "applied": bool(np.any(correction > 1e-7)),
        "target_floor_y_m": float(target_floor_y),
        "clearance_m": float(clearance_m),
        "smoothing_frames": int(window),
        "max_root_y_correction_m": float(np.max(correction)),
        "correction_p95_m": float(np.percentile(correction, 95)),
        "penetration_before_m": float(
            min(0.0, float(np.min(minimum - float(target_floor_y))))
        ),
        "penetration_after_m": float(
            min(
                0.0,
                float(
                    np.min(
                        feet_after[..., 1].min(axis=1) - float(target_floor_y)
                    )
                ),
            )
        ),
    }


def recompute_transition_contacts_np(
    transition: np.ndarray,
    *,
    fps: float,
    floor_y: float = 0.0,
    left_contact: np.ndarray | None = None,
    right_contact: np.ndarray | None = None,
    ramp_seconds: float = 4.0 / 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rebuild bridge contacts from FK and soften only support on/off edges."""

    from contracts.gravity import fk24_np
    from motion_geometry.physical import contact_from_joints_np

    x = _validate_motion_np(transition).copy()
    if len(x) == 0:
        return x, {"contact_ratio": 0.0, "ramp_frames": 0}
    binary = contact_from_joints_np(
        fk24_np(x),
        fps=float(fps),
        floor_y=float(floor_y),
    )
    left = (
        np.asarray(left_contact, dtype=np.float32).reshape(4) > 0.5
        if left_contact is not None
        else np.zeros((4,), dtype=bool)
    )
    right = (
        np.asarray(right_contact, dtype=np.float32).reshape(4) > 0.5
        if right_contact is not None
        else np.zeros((4,), dtype=bool)
    )
    ramp = max(1, int(round(max(float(ramp_seconds), 0.0) * float(fps))))
    weights = binary.astype(np.float32)
    for foot in range(binary.shape[1]):
        index = 0
        while index < len(binary):
            if not binary[index, foot]:
                index += 1
                continue
            end = index + 1
            while end < len(binary) and binary[end, foot]:
                end += 1
            for frame in range(index, end):
                in_phase = 1.0 if index == 0 and left[foot] else min(
                    1.0, (frame - index + 1) / float(ramp)
                )
                out_phase = 1.0 if end == len(binary) and right[foot] else min(
                    1.0, (end - frame) / float(ramp)
                )
                weights[frame, foot] = float(
                    min(smootherstep01(in_phase), smootherstep01(out_phase))
                )
            index = end
    x[:, CONTACT] = weights
    return x.astype(np.float32), {
        "contact_ratio": float(np.mean(weights > 0.5)),
        "contact_confidence_mean": float(np.mean(weights)),
        "ramp_frames": int(ramp),
        "left_contact": left.astype(int).tolist(),
        "right_contact": right.astype(int).tolist(),
    }


def resample_motion_so3_np(motion: np.ndarray, positions: np.ndarray) -> np.ndarray:
    from motion_geometry.resampling import resample_edge151_np

    return resample_edge151_np(_validate_motion_np(motion), positions=positions)


def dampen_event_edges_so3(
    motion: np.ndarray,
    edge_frames: int,
    strength: float,
) -> np.ndarray:
    """Reduce boundary velocity by local SO(3) time reparameterization.

    Unlike direct 6D coordinate blending, this function preserves valid
    rotations and caps the modified region to at most one sixth of the event on
    each side.  The event center is left untouched.
    """
    x = _validate_motion_np(motion).copy()
    if len(x) < 8:
        return x
    n = min(max(0, int(edge_frames)), max(2, len(x) // 6))
    s = float(np.clip(strength, 0.0, 1.0))
    if n <= 1 or s <= 0.0:
        return x

    positions = np.arange(len(x), dtype=np.float32)

    left_u = np.arange(n + 1, dtype=np.float32) / float(n)
    # f'(0)=0 and f'(1)=1: only ease the event start.
    left_eased = 2.0 * left_u**2 - left_u**3
    left_map = (1.0 - s) * np.arange(n + 1, dtype=np.float32) + s * n * left_eased
    positions[: n + 1] = left_map

    right_u = np.arange(n + 1, dtype=np.float32) / float(n)
    # f'(0)=1 and f'(1)=0: only ease into the event endpoint.
    right_eased = right_u + right_u**2 - right_u**3
    start = len(x) - n - 1
    right_map = start + (1.0 - s) * np.arange(n + 1, dtype=np.float32) + s * n * right_eased
    positions[start:] = right_map

    result = resample_motion_so3_np(x, positions)
    return result.astype(np.float32)


def apply_start_anchor_so3(
    motion: np.ndarray,
    start_pose: np.ndarray,
    blend_frames: int = 8,
) -> np.ndarray:
    """Apply one constant stage-XZ translation and nothing else.

    ``blend_frames`` is retained for API compatibility with old checkpoints
    and launchers.  A start anchor must not overwrite pose, Root-Y, contacts or
    any frame-relative trajectory; those are event content contracts.
    """
    x = _validate_motion_np(motion).copy()
    s = np.asarray(start_pose, dtype=np.float32).reshape(-1)
    if s.shape[0] != MOTION_DIM or len(x) == 0:
        return x
    del blend_frames
    stage_offset = s[[ROOT_X, ROOT_Z]] - x[0, [ROOT_X, ROOT_Z]]
    x[:, ROOT_X] += float(stage_offset[0])
    x[:, ROOT_Z] += float(stage_offset[1])
    return x.astype(np.float32)


def temporal_so3_filter_np(
    motion: np.ndarray,
    window: int = 5,
    strength: float = 0.25,
    preserve_contacts: bool = True,
) -> np.ndarray:
    """Apply a short SO(3)-aware low-pass filter to a local transition."""
    x = _validate_motion_np(motion)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or len(x) < 3 or strength <= 0.0:
        return x.copy()
    radius = w // 2
    sigma = max(float(radius) / 1.5, 0.8)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    with torch.no_grad():
        m = motion_rotation_matrices_torch(torch.from_numpy(x))
        filtered = []
        for t in range(len(x)):
            indices = np.clip(
                np.arange(t - radius, t + radius + 1), 0, len(x) - 1
            )
            local = m[torch.from_numpy(indices).long()]
            weighted = (
                local * torch.from_numpy(kernel).float()[:, None, None, None]
            ).sum(dim=0)
            u, _, vh = torch.linalg.svd(weighted)
            r = torch.matmul(u, vh)
            det = torch.det(r)
            if torch.any(det < 0):
                u = u.clone()
                u[det < 0, :, -1] *= -1.0
                r = torch.matmul(u, vh)
            filtered.append(r)
        filtered_m = torch.stack(filtered, dim=0)
        relative = torch.matmul(m.transpose(-1, -2), filtered_m)
        tangent = matrix_to_axis_angle(relative) * float(np.clip(strength, 0.0, 1.0))
        blended = torch.matmul(m, axis_angle_to_matrix(tangent))

    out = x.copy()
    out[:, ROT] = matrix_to_rotation_6d(blended).reshape(len(x), -1).cpu().numpy()
    root_y = np.pad(x[:, ROOT_Y], (radius, radius), mode="edge")
    smooth_y = np.convolve(root_y, kernel, mode="valid")
    out[:, ROOT_Y] = (
        (1.0 - strength) * x[:, ROOT_Y] + strength * smooth_y
    ).astype(np.float32)
    if preserve_contacts:
        out[:, CONTACT] = x[:, CONTACT]
    return project_motion_rotations_np(out)


def transition_blend_envelope(length: int, power: float = 2.0) -> np.ndarray:
    k = max(0, int(length))
    if k == 0:
        return np.zeros((0,), dtype=np.float32)
    u = np.linspace(1.0 / (k + 1), k / (k + 1), k, dtype=np.float32)
    return np.sin(np.pi * u).astype(np.float32) ** float(max(power, 0.25))


def root_yaw_np(motion: np.ndarray) -> np.ndarray:
    x = _validate_motion_np(motion)
    matrices = motion_rotation_matrices_np(x)[:, 0]
    return np.unwrap(np.arctan2(matrices[:, 0, 2], matrices[:, 2, 2])).astype(np.float32)


def endpoint_metrics_np(prev: np.ndarray, nxt: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
    a = _validate_motion_np(prev)
    b = _validate_motion_np(nxt)
    with torch.no_grad():
        ma = motion_rotation_matrices_torch(torch.from_numpy(a))
        mb = motion_rotation_matrices_torch(torch.from_numpy(b))
        pose_rel = torch.matmul(ma[-1].transpose(-1, -2), mb[0])
        pose_angle = torch.linalg.vector_norm(matrix_to_axis_angle(pose_rel), dim=-1)

        va = (
            matrix_to_axis_angle(torch.matmul(ma[-2].transpose(-1, -2), ma[-1]))
            if len(a) >= 2 else torch.zeros((NUM_JOINTS, 3))
        )
        vb = (
            matrix_to_axis_angle(torch.matmul(mb[0].transpose(-1, -2), mb[1]))
            if len(b) >= 2 else torch.zeros((NUM_JOINTS, 3))
        )
        velocity_jump = torch.linalg.vector_norm(va - vb, dim=-1)

        aa = (
            matrix_to_axis_angle(torch.matmul(ma[-3].transpose(-1, -2), ma[-2]))
            if len(a) >= 3 else va
        )
        ab = (
            matrix_to_axis_angle(torch.matmul(mb[1].transpose(-1, -2), mb[2]))
            if len(b) >= 3 else vb
        )
        acceleration_jump = torch.linalg.vector_norm((va - aa) - (ab - vb), dim=-1)

    pair_yaw = root_yaw_np(np.stack([a[-1], b[0]], axis=0))
    yaw_gap = abs(float(pair_yaw[1] - pair_yaw[0]))
    return {
        "pose_jump": float(torch.sqrt(torch.mean(pose_angle**2)).item() / np.pi),
        "angular_velocity_jump_radps_rms": float(
            torch.sqrt(torch.mean(velocity_jump**2)).item() * fps
        ),
        "angular_acceleration_jump_radps2_rms": float(
            torch.sqrt(torch.mean(acceleration_jump**2)).item() * fps * fps
        ),
        "contact_jump": float(np.abs(a[-1, CONTACT] - b[0, CONTACT]).mean()),
        "yaw_gap_deg": float(yaw_gap * 180.0 / np.pi),
        # Interpretable scientific fields.
        "pose_jump_deg_rms": float(torch.sqrt(torch.mean(pose_angle**2)).item() * 180.0 / np.pi),
        "velocity_jump_deg_s_rms": float(torch.sqrt(torch.mean(velocity_jump**2)).item() * fps * 180.0 / np.pi),
        "acceleration_jump_deg_s2_rms": float(torch.sqrt(torch.mean(acceleration_jump**2)).item() * fps * fps * 180.0 / np.pi),
        "root_y_jump": abs(float(a[-1, ROOT_Y] - b[0, ROOT_Y])),
    }


def jitter_statistics_np(motion: np.ndarray, fps: float = 30.0) -> Dict[str, object]:
    x = _validate_motion_np(motion)
    positions = motion_to_joint_positions_np(x)
    velocity = np.diff(positions, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    jerk_norm = np.linalg.norm(jerk, axis=-1) if len(jerk) else np.zeros((0, NUM_JOINTS))
    acc_norm = np.linalg.norm(acceleration, axis=-1) if len(acceleration) else np.zeros((0, NUM_JOINTS))
    angular = angular_velocity_np(x) * fps
    angular_acc = np.diff(angular, axis=0) * fps
    angular_jerk = np.diff(angular_acc, axis=0) * fps

    per_joint = []
    for j, name in enumerate(SMPL_JOINT_NAMES):
        values = jerk_norm[:, j] if len(jerk_norm) else np.zeros((0,))
        per_joint.append(
            {
                "joint": name,
                "jerk_p95_mps3": float(np.percentile(values, 95)) if len(values) else 0.0,
                "jerk_max_mps3": float(values.max()) if len(values) else 0.0,
            }
        )
    per_joint.sort(key=lambda row: row["jerk_p95_mps3"], reverse=True)

    frame_score = jerk_norm.mean(axis=1) if len(jerk_norm) else np.zeros((0,))
    top_indices = np.argsort(frame_score)[::-1][: min(20, len(frame_score))]
    return {
        "joint_acceleration_p95_mps2": float(np.percentile(acc_norm, 95)) if acc_norm.size else 0.0,
        "joint_acceleration_max_mps2": float(acc_norm.max()) if acc_norm.size else 0.0,
        "joint_jerk_p95_mps3": float(np.percentile(jerk_norm, 95)) if jerk_norm.size else 0.0,
        "joint_jerk_max_mps3": float(jerk_norm.max()) if jerk_norm.size else 0.0,
        "angular_jerk_p95_radps3": float(np.percentile(np.linalg.norm(angular_jerk, axis=-1), 95)) if angular_jerk.size else 0.0,
        "root_y_acceleration_p95_mps2": float(
            np.percentile(np.abs(np.diff(x[:, ROOT_Y], n=2)) * fps * fps, 95)
        ) if len(x) >= 3 else 0.0,
        "per_joint_jerk": per_joint,
        "worst_frames": [
            {
                "frame": int(i + 2),
                "time_seconds": float((i + 2) / fps),
                "mean_joint_jerk_mps3": float(frame_score[i]),
            }
            for i in top_indices
        ],
    }
