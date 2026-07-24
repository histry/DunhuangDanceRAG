#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V34 boundary dynamics for continuous Dunhuang transitions.

This module provides four pieces used by both training and inference:

1. robust endpoint angular/root velocity, acceleration and jerk estimates;
2. a regularised septic Hermite path on SO(3) tangent spaces;
3. an arbitrary-length NumPy transition constructor;
4. a length-preserving local exit handshake for the following event.

The septic path has eight degrees of freedom and can match pose, velocity,
acceleration and (regularised) jerk at both ends.  Learned INR residuals are
handled separately and use a C3-zero endpoint envelope in v32_contact_inr.py.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from support.motion_geometry import (
    CONTACT,
    MOTION_DIM,
    NUM_JOINTS,
    ROOT,
    ROOT_X,
    ROOT_Y,
    ROOT_Z,
    ROT,
)


def _limit_norm(vector: torch.Tensor, maximum: torch.Tensor | float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)
    limit = torch.as_tensor(maximum, dtype=vector.dtype, device=vector.device)
    return vector * torch.minimum(torch.ones_like(norm), limit / norm)


def _native_rotation_matrices_torch(motion: torch.Tensor) -> torch.Tensor:
    """Decode the historical transition model's PyTorch3D-row layout."""

    return rotation_6d_to_matrix(
        motion[..., ROT].reshape(*motion.shape[:-1], NUM_JOINTS, 6)
    )


def _project_native_motion_torch(motion: torch.Tensor) -> torch.Tensor:
    matrices = _native_rotation_matrices_torch(motion)
    out = motion.clone()
    out[..., ROT] = matrix_to_rotation_6d(matrices).reshape(
        *motion.shape[:-1],
        NUM_JOINTS * 6,
    )
    return out


def _project_native_motion_np(motion: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        projected = _project_native_motion_torch(
            torch.from_numpy(np.asarray(motion, dtype=np.float32))
        )
    return projected.cpu().numpy().astype(np.float32)


def _difference(values: torch.Tensor, order: int) -> torch.Tensor:
    result = values
    for _ in range(int(order)):
        if result.shape[0] < 2:
            return result.new_zeros((0, *result.shape[1:]))
        result = result[1:] - result[:-1]
    return result


def _angular_increments(motion: torch.Tensor) -> torch.Tensor:
    matrices = _native_rotation_matrices_torch(motion)
    if len(matrices) < 2:
        return matrices.new_zeros((0, NUM_JOINTS, 3))
    relative = torch.matmul(
        matrices[:-1].transpose(-1, -2), matrices[1:]
    )
    return matrix_to_axis_angle(relative)


def _endpoint_state_one(sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Derivatives for a complete [start, target..., end] sequence."""
    increments = _angular_increments(sequence)
    root = sequence[:, ROOT]
    root_velocity = _difference(root, 1)

    zero_w = sequence.new_zeros((NUM_JOINTS, 3))
    zero_r = sequence.new_zeros((3,))

    start_omega = increments[0] if len(increments) else zero_w
    end_omega = increments[-1] if len(increments) else zero_w
    start_alpha = (
        increments[1] - increments[0] if len(increments) >= 2 else zero_w
    )
    end_alpha = (
        increments[-1] - increments[-2] if len(increments) >= 2 else zero_w
    )
    start_jerk = (
        increments[2] - 2.0 * increments[1] + increments[0]
        if len(increments) >= 3 else zero_w
    )
    end_jerk = (
        increments[-1] - 2.0 * increments[-2] + increments[-3]
        if len(increments) >= 3 else zero_w
    )

    start_root_v = root_velocity[0] if len(root_velocity) else zero_r
    end_root_v = root_velocity[-1] if len(root_velocity) else zero_r
    start_root_a = (
        root_velocity[1] - root_velocity[0]
        if len(root_velocity) >= 2 else zero_r
    )
    end_root_a = (
        root_velocity[-1] - root_velocity[-2]
        if len(root_velocity) >= 2 else zero_r
    )
    start_root_j = (
        root_velocity[2] - 2.0 * root_velocity[1] + root_velocity[0]
        if len(root_velocity) >= 3 else zero_r
    )
    end_root_j = (
        root_velocity[-1] - 2.0 * root_velocity[-2] + root_velocity[-3]
        if len(root_velocity) >= 3 else zero_r
    )
    return {
        "start_omega": start_omega,
        "end_omega": end_omega,
        "start_alpha": start_alpha,
        "end_alpha": end_alpha,
        "start_angular_jerk": start_jerk,
        "end_angular_jerk": end_jerk,
        "start_root_velocity": start_root_v,
        "end_root_velocity": end_root_v,
        "start_root_acceleration": start_root_a,
        "end_root_acceleration": end_root_a,
        "start_root_jerk": start_root_j,
        "end_root_jerk": end_root_j,
    }


def boundary_state_from_training_batch(
    start: torch.Tensor,
    target: torch.Tensor,
    end: torch.Tensor,
    lengths: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build endpoint dynamics from real masked training intervals.

    The state is deterministic supervision derived from the same complete event
    frames that produced each window.  It is not a learnable input parameter.
    """
    rows: Dict[str, list[torch.Tensor]] = {}
    for batch_index in range(len(start)):
        length = max(1, min(int(lengths[batch_index].item()), target.shape[1]))
        sequence = torch.cat([
            start[batch_index : batch_index + 1],
            target[batch_index, :length],
            end[batch_index : batch_index + 1],
        ], dim=0)
        state = _endpoint_state_one(sequence)
        for key, value in state.items():
            rows.setdefault(key, []).append(value)
    return {key: torch.stack(value, dim=0) for key, value in rows.items()}


def boundary_state_from_context_np(
    previous: np.ndarray,
    following: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Estimate derivatives at the last previous and first following frames."""
    prev = np.asarray(previous, np.float32)
    nxt = np.asarray(following, np.float32)
    if prev.ndim != 2 or nxt.ndim != 2:
        raise ValueError("previous/following must be [T,151]")
    if prev.shape[-1] != MOTION_DIM or nxt.shape[-1] != MOTION_DIM:
        raise ValueError("Invalid EDGE motion dimension")

    with torch.no_grad():
        prev_t = torch.from_numpy(prev[-4:])
        next_t = torch.from_numpy(nxt[:4])
        prev_inc = _angular_increments(prev_t)
        next_inc = _angular_increments(next_t)

    zero_w = np.zeros((NUM_JOINTS, 3), np.float32)
    start_omega = prev_inc[-1].cpu().numpy() if len(prev_inc) else zero_w
    end_omega = next_inc[0].cpu().numpy() if len(next_inc) else zero_w
    start_alpha = (
        (prev_inc[-1] - prev_inc[-2]).cpu().numpy()
        if len(prev_inc) >= 2 else zero_w
    )
    end_alpha = (
        (next_inc[1] - next_inc[0]).cpu().numpy()
        if len(next_inc) >= 2 else zero_w
    )
    start_jerk = (
        (prev_inc[-1] - 2.0 * prev_inc[-2] + prev_inc[-3]).cpu().numpy()
        if len(prev_inc) >= 3 else zero_w
    )
    end_jerk = (
        (next_inc[2] - 2.0 * next_inc[1] + next_inc[0]).cpu().numpy()
        if len(next_inc) >= 3 else zero_w
    )

    prev_root = prev[-4:, ROOT]
    next_root = nxt[:4, ROOT]
    prev_v = np.diff(prev_root, axis=0)
    next_v = np.diff(next_root, axis=0)
    zero_r = np.zeros((3,), np.float32)
    return {
        "start_omega": start_omega.astype(np.float32),
        "end_omega": end_omega.astype(np.float32),
        "start_alpha": start_alpha.astype(np.float32),
        "end_alpha": end_alpha.astype(np.float32),
        "start_angular_jerk": start_jerk.astype(np.float32),
        "end_angular_jerk": end_jerk.astype(np.float32),
        "start_root_velocity": (prev_v[-1] if len(prev_v) else zero_r).astype(np.float32),
        "end_root_velocity": (next_v[0] if len(next_v) else zero_r).astype(np.float32),
        "start_root_acceleration": (
            prev_v[-1] - prev_v[-2] if len(prev_v) >= 2 else zero_r
        ).astype(np.float32),
        "end_root_acceleration": (
            next_v[1] - next_v[0] if len(next_v) >= 2 else zero_r
        ).astype(np.float32),
        "start_root_jerk": (
            prev_v[-1] - 2.0 * prev_v[-2] + prev_v[-3]
            if len(prev_v) >= 3 else zero_r
        ).astype(np.float32),
        "end_root_jerk": (
            next_v[2] - 2.0 * next_v[1] + next_v[0]
            if len(next_v) >= 3 else zero_r
        ).astype(np.float32),
    }


def boundary_state_to_torch(
    state: Dict[str, np.ndarray | torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.ndim in (1, 2):
            tensor = tensor.unsqueeze(0)
        result[key] = tensor
    return result


def _septic_coefficients(
    endpoint: torch.Tensor,
    velocity0: torch.Tensor,
    velocity1: torch.Tensor,
    acceleration0: torch.Tensor,
    acceleration1: torch.Tensor,
    jerk0: torch.Tensor,
    jerk1: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Coefficients c0..c7 for p(0)=0 and p(1)=endpoint."""
    c0 = torch.zeros_like(endpoint)
    c1 = velocity0
    c2 = 0.5 * acceleration0
    c3 = jerk0 / 6.0
    rhs = torch.stack([
        endpoint - c1 - c2 - c3,
        velocity1 - c1 - 2.0 * c2 - 3.0 * c3,
        acceleration1 - 2.0 * c2 - 6.0 * c3,
        jerk1 - 6.0 * c3,
    ], dim=-1)
    matrix = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [4.0, 5.0, 6.0, 7.0],
        [12.0, 20.0, 30.0, 42.0],
        [24.0, 60.0, 120.0, 210.0],
    ], dtype=endpoint.dtype, device=endpoint.device)
    solution = torch.linalg.solve(matrix, rhs.reshape(-1, 4).T).T
    solution = solution.reshape(*rhs.shape[:-1], 4)
    c4, c5, c6, c7 = solution.unbind(dim=-1)
    return c0, c1, c2, c3, c4, c5, c6, c7


def _evaluate_polynomial(
    coefficients: Tuple[torch.Tensor, ...],
    t: torch.Tensor,
) -> torch.Tensor:
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = result * t + coefficient
    return result


def septic_so3_root_base(
    start: torch.Tensor,
    end: torch.Tensor,
    t: torch.Tensor,
    length_frames: torch.Tensor,
    boundary_state: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contact logits, root and local rotations for the V34 C3 base."""
    batch, count, _ = t.shape
    r0 = rotation_6d_to_matrix(
        start[..., ROT].reshape(batch, NUM_JOINTS, 6)
    )
    r1 = rotation_6d_to_matrix(
        end[..., ROT].reshape(batch, NUM_JOINTS, 6)
    )
    relative = torch.matmul(r0.transpose(-1, -2), r1)
    delta = matrix_to_axis_angle(relative)

    length = (length_frames.reshape(batch, 1, 1) + 1.0).to(start.dtype)
    omega0 = boundary_state["start_omega"]
    omega1 = torch.matmul(
        relative, boundary_state["end_omega"].unsqueeze(-1)
    ).squeeze(-1)
    alpha0 = boundary_state["start_alpha"]
    alpha1 = torch.matmul(
        relative, boundary_state["end_alpha"].unsqueeze(-1)
    ).squeeze(-1)
    jerk0 = boundary_state["start_angular_jerk"]
    jerk1 = torch.matmul(
        relative, boundary_state["end_angular_jerk"].unsqueeze(-1)
    ).squeeze(-1)

    jerk_shrink = float(os.getenv("V34_JERK_MATCH_SHRINK", "0.35"))
    velocity0 = omega0 * length
    velocity1 = omega1 * length
    acceleration0 = alpha0 * length.square()
    acceleration1 = alpha1 * length.square()
    jerk0 = jerk0 * length.pow(3) * jerk_shrink
    jerk1 = jerk1 * length.pow(3) * jerk_shrink

    path_norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
    velocity_cap = torch.minimum(
        path_norm + float(os.getenv("V34_VELOCITY_TANGENT_MARGIN", "0.30")),
        torch.full_like(path_norm, float(os.getenv("V34_VELOCITY_TANGENT_CAP", "0.90"))),
    )
    velocity0 = _limit_norm(velocity0, velocity_cap)
    velocity1 = _limit_norm(velocity1, velocity_cap)
    acceleration0 = _limit_norm(
        acceleration0,
        float(os.getenv("V34_ACCELERATION_TANGENT_CAP", "1.40")),
    )
    acceleration1 = _limit_norm(
        acceleration1,
        float(os.getenv("V34_ACCELERATION_TANGENT_CAP", "1.40")),
    )
    jerk0 = _limit_norm(jerk0, float(os.getenv("V34_JERK_TANGENT_CAP", "2.20")))
    jerk1 = _limit_norm(jerk1, float(os.getenv("V34_JERK_TANGENT_CAP", "2.20")))

    coefficients = _septic_coefficients(
        delta, velocity0, velocity1,
        acceleration0, acceleration1, jerk0, jerk1,
    )
    u = t.reshape(batch, count, 1, 1)
    tangent = _evaluate_polynomial(
        tuple(coefficient[:, None] for coefficient in coefficients), u
    )
    tangent = _limit_norm(
        tangent,
        torch.minimum(
            path_norm[:, None] + 0.35,
            torch.full_like(path_norm[:, None], 1.50),
        ),
    )
    rotations = torch.matmul(r0[:, None], axis_angle_to_matrix(tangent))

    root0 = start[..., ROOT]
    root_delta = end[..., ROOT] - root0
    root_length = (length_frames.reshape(batch, 1) + 1.0).to(start.dtype)
    rv0 = boundary_state["start_root_velocity"] * root_length
    rv1 = boundary_state["end_root_velocity"] * root_length
    ra0 = boundary_state["start_root_acceleration"] * root_length.square()
    ra1 = boundary_state["end_root_acceleration"] * root_length.square()
    rj0 = boundary_state["start_root_jerk"] * root_length.pow(3) * jerk_shrink
    rj1 = boundary_state["end_root_jerk"] * root_length.pow(3) * jerk_shrink
    rv0 = rv0.clamp(-0.25, 0.25)
    rv1 = rv1.clamp(-0.25, 0.25)
    ra0 = ra0.clamp(-0.35, 0.35)
    ra1 = ra1.clamp(-0.35, 0.35)
    rj0 = rj0.clamp(-0.50, 0.50)
    rj1 = rj1.clamp(-0.50, 0.50)
    root_coefficients = _septic_coefficients(
        root_delta, rv0, rv1, ra0, ra1, rj0, rj1
    )
    root = root0[:, None] + _evaluate_polynomial(
        tuple(coefficient[:, None] for coefficient in root_coefficients), t
    )
    # Preserve the endpoint-velocity-aware Root trajectory.  Older snapshots
    # zeroed XZ here because boundary clips were treated as root-local poses;
    # doing so breaks cumulative event composition and introduces a pair of
    # artificial velocity impulses at every transition.

    smooth = t.clamp(0.0, 1.0) ** 3 * (
        10.0 - 15.0 * t.clamp(0.0, 1.0) + 6.0 * t.clamp(0.0, 1.0) ** 2
    )
    start_logits = torch.logit(start[..., CONTACT].clamp(1e-4, 1 - 1e-4))[:, None]
    end_logits = torch.logit(end[..., CONTACT].clamp(1e-4, 1 - 1e-4))[:, None]
    contact_logits = (1.0 - smooth) * start_logits + smooth * end_logits
    return contact_logits, root, rotations


def make_v34_transition_np(
    previous: np.ndarray,
    following: np.ndarray,
    length: int,
) -> np.ndarray:
    prev = np.asarray(previous, np.float32)
    nxt = np.asarray(following, np.float32)
    k = max(0, int(length))
    if k == 0:
        return np.zeros((0, MOTION_DIM), np.float32)
    state_np = boundary_state_from_context_np(prev, nxt)
    with torch.no_grad():
        start = torch.from_numpy(prev[-1]).reshape(1, -1)
        end = torch.from_numpy(nxt[0]).reshape(1, -1)
        coordinate = torch.linspace(
            1.0 / (k + 1), k / (k + 1), k
        ).reshape(1, k, 1)
        length_frames = torch.tensor([[float(k)]])
        state = boundary_state_to_torch(state_np, torch.device("cpu"))
        logits, root, rotation = septic_so3_root_base(
            start, end, coordinate, length_frames, state
        )
        rot6d = matrix_to_rotation_6d(rotation).reshape(k, -1)
        motion = torch.cat([
            torch.sigmoid(logits)[0], root[0], rot6d
        ], dim=-1)
        motion = _project_native_motion_torch(motion)
    return motion.cpu().numpy().astype(np.float32)


def _smootherstep(values: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(values, np.float32), 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def apply_exit_handshake_np(
    transition: np.ndarray,
    content: np.ndarray,
    frames: int = 10,
    strength: float = 1.0,
    max_rotation_deg: float = 18.0,
    max_root_y: float = 0.08,
) -> Tuple[np.ndarray, Dict[str, float | int | bool]]:
    """Replace only the following event's first frames with a C3 handshake.

    Total sequence length is unchanged.  The bridge starts from the transition's
    true terminal dynamics and reaches the untouched event at frame ``h``.
    """
    trans = np.asarray(transition, np.float32)
    original = np.asarray(content, np.float32)
    h = min(max(0, int(frames)), max(0, len(original) - 4))
    if h < 3 or len(trans) < 3 or strength <= 0.0:
        return original.copy(), {
            "enabled": False,
            "frames": int(h),
            "reason": "insufficient_context",
        }

    bridge = make_v34_transition_np(trans[-4:], original[h:], h)
    mode = os.getenv("V34_HANDSHAKE_MODE", "replace").strip().lower()
    if mode == "replace":
        # A full bridge preserves the septic endpoint derivatives.  A tapered
        # blend is retained only as an explicit ablation because it can shift
        # the discontinuity from the transition exit to the handshake tail.
        weights = np.full(
            (h,), float(np.clip(strength, 0.0, 1.0)), dtype=np.float32
        )
    elif mode == "taper":
        weights = float(np.clip(strength, 0.0, 1.0)) * (
            1.0 - _smootherstep(
                np.arange(1, h + 1, dtype=np.float32) / float(h + 1)
            )
        )
        weights[: min(3, h)] = float(np.clip(strength, 0.0, 1.0))
    else:
        raise ValueError(f"Unknown V34_HANDSHAKE_MODE={mode!r}")

    with torch.no_grad():
        orig_m = _native_rotation_matrices_torch(
            torch.from_numpy(original[:h])
        )
        bridge_m = _native_rotation_matrices_torch(
            torch.from_numpy(bridge)
        )
        relative = torch.matmul(orig_m.transpose(-1, -2), bridge_m)
        tangent_raw = matrix_to_axis_angle(relative)
        raw_correction_deg = (
            torch.linalg.norm(tangent_raw, dim=-1).max().item()
            * 180.0 / np.pi
        )
        cap = float(max_rotation_deg) * np.pi / 180.0
        tangent = _limit_norm(tangent_raw, cap)
        weighted = tangent * torch.from_numpy(weights)[:, None, None]
        rotation = torch.matmul(orig_m, axis_angle_to_matrix(weighted))
        rot6d = matrix_to_rotation_6d(rotation).reshape(h, -1).cpu().numpy()
        correction_deg = (
            torch.linalg.norm(tangent, dim=-1).max().item() * 180.0 / np.pi
        )

    result = original.copy()
    result[:h, ROT] = rot6d.astype(np.float32)
    root_weight = weights[:, None]
    root_delta = np.clip(
        bridge[:, ROOT] - original[:h, ROOT],
        -float(max_root_y), float(max_root_y),
    )
    result[:h, ROOT] = (
        original[:h, ROOT] + root_weight * root_delta
    ).astype(np.float32)
    result[:h, CONTACT] = (
        (1.0 - root_weight) * original[:h, CONTACT]
        + root_weight * bridge[:, CONTACT]
    ).astype(np.float32)
    if os.getenv("V34_HARD_CONTACT_OUTPUT", "0").lower() in {
        "1", "true", "yes", "on"
    }:
        result[:h, CONTACT] = (result[:h, CONTACT] >= 0.5).astype(np.float32)
    # The handshake owns only its local prefix.  Preserve the original event's
    # global Root trajectory outside that window and the geodesic bridge Root
    # inside it; never reset a whole event to the origin.
    result = _project_native_motion_np(result)
    return result, {
        "enabled": True,
        "frames": int(h),
        "strength": float(strength),
        "mode": mode,
        "max_rotation_correction_deg": float(correction_deg),
        "raw_rotation_correction_deg": float(raw_correction_deg),
        "rotation_correction_clipped": bool(raw_correction_deg > max_rotation_deg),
        "max_root_correction": float(np.max(np.abs(root_delta))),
    }
