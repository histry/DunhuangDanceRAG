#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.53 single-source-of-truth rotation and tangent-space contract.

EDGE uses a *column-concatenated* 6D rotation representation:
``[R[:, 0], R[:, 1]]``.  This module is intentionally independent of the
large MotionRAG core and is safe to import from data building, routing,
refinement, auditing and rendering code.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover
    Rotation = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

EPS = 1.0e-8

ROT6D_LAYOUT_COLUMN = "column"
ROT6D_LAYOUT_PYTORCH3D_ROW = "pytorch3d_row"
CANONICAL_ROT6D_LAYOUT = ROT6D_LAYOUT_COLUMN


def normalize_rot6d_layout(layout: str | None) -> str:
    """Return a stable serialized name for a supported Rot6D layout."""

    value = str(layout or CANONICAL_ROT6D_LAYOUT).strip().lower().replace("-", "_")
    aliases = {
        "column": ROT6D_LAYOUT_COLUMN,
        "columns": ROT6D_LAYOUT_COLUMN,
        "column_concat": ROT6D_LAYOUT_COLUMN,
        "edge_column": ROT6D_LAYOUT_COLUMN,
        "row": ROT6D_LAYOUT_PYTORCH3D_ROW,
        "rows": ROT6D_LAYOUT_PYTORCH3D_ROW,
        "row_concat": ROT6D_LAYOUT_PYTORCH3D_ROW,
        "pytorch3d": ROT6D_LAYOUT_PYTORCH3D_ROW,
        "pytorch3d_row": ROT6D_LAYOUT_PYTORCH3D_ROW,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Rot6D layout {layout!r}; expected one of "
            f"{sorted(set(aliases.values()))}"
        ) from exc


def _as_float_np(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def project_to_so3_np(m: np.ndarray) -> np.ndarray:
    """Project arbitrary 3x3 matrices to the nearest proper rotation."""
    x = _as_float_np(m)
    shape = x.shape
    flat = x.reshape(-1, 3, 3).astype(np.float64)
    u, _, vt = np.linalg.svd(flat)
    r = u @ vt
    det = np.linalg.det(r)
    bad = det < 0.0
    if np.any(bad):
        u = u.copy()
        u[bad, :, -1] *= -1.0
        r = u @ vt
    return r.reshape(shape).astype(np.float32)


def rot6d_to_matrix_np(x: np.ndarray, project: bool = True) -> np.ndarray:
    """Decode column-concatenated Rot6D with stable Gram--Schmidt."""
    y = _as_float_np(x)
    if y.shape[-1] != 6:
        raise ValueError(f"Rot6D requires last dimension 6, got {y.shape}")
    a1, a2 = y[..., :3], y[..., 3:6]
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    b1 = a1 / np.maximum(n1, EPS)
    a2o = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(a2o, axis=-1, keepdims=True)
    b2 = a2o / np.maximum(n2, EPS)
    b3 = np.cross(b1, b2)
    out = np.stack([b1, b2, b3], axis=-1)
    bad = (
        ~np.isfinite(out).all(axis=(-2, -1))
        | (n1[..., 0] < 1.0e-7)
        | (n2[..., 0] < 1.0e-7)
    )
    if np.any(bad):
        out = out.copy()
        out[bad] = np.eye(3, dtype=np.float32)
    return project_to_so3_np(out) if project else out.astype(np.float32)


def matrix_to_rot6d_np(m: np.ndarray, project: bool = True) -> np.ndarray:
    """Encode rotations by concatenating their first two *columns*."""
    r = project_to_so3_np(m) if project else _as_float_np(m)
    if r.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices require (...,3,3), got {r.shape}")
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1).astype(np.float32)


def rot6d_to_matrix_layout_np(
    x: np.ndarray,
    layout: str = CANONICAL_ROT6D_LAYOUT,
    project: bool = True,
) -> np.ndarray:
    """Decode either the canonical column or historical PyTorch3D row layout."""

    normalized = normalize_rot6d_layout(layout)
    matrix = rot6d_to_matrix_np(x, project=False)
    if normalized == ROT6D_LAYOUT_PYTORCH3D_ROW:
        matrix = np.swapaxes(matrix, -1, -2)
    return project_to_so3_np(matrix) if project else matrix.astype(np.float32)


def matrix_to_rot6d_layout_np(
    m: np.ndarray,
    layout: str = CANONICAL_ROT6D_LAYOUT,
    project: bool = True,
) -> np.ndarray:
    """Encode a rotation using an explicitly declared Rot6D layout."""

    normalized = normalize_rot6d_layout(layout)
    matrix = project_to_so3_np(m) if project else _as_float_np(m)
    if normalized == ROT6D_LAYOUT_PYTORCH3D_ROW:
        matrix = np.swapaxes(matrix, -1, -2)
    return matrix_to_rot6d_np(matrix, project=False)


def convert_rot6d_layout_np(
    x: np.ndarray,
    source_layout: str,
    target_layout: str,
) -> np.ndarray:
    """Convert Rot6D values while preserving their physical rotations."""

    source = normalize_rot6d_layout(source_layout)
    target = normalize_rot6d_layout(target_layout)
    values = _as_float_np(x)
    if source == target:
        return values.copy()
    matrix = rot6d_to_matrix_layout_np(values, source)
    return matrix_to_rot6d_layout_np(matrix, target)


def convert_motion_rot6d_layout_np(
    motion: np.ndarray,
    source_layout: str,
    target_layout: str,
    rotation_start: int = 7,
    rotation_end: int = 151,
    num_joints: int = 24,
) -> np.ndarray:
    """Convert the rotation channels of one frame or a motion batch."""

    values = _as_float_np(motion).copy()
    if values.shape[-1] < rotation_end:
        raise ValueError(
            f"Motion requires at least {rotation_end} channels, got {values.shape}"
        )
    rotations = values[..., rotation_start:rotation_end].reshape(
        values.shape[:-1] + (num_joints, 6)
    )
    values[..., rotation_start:rotation_end] = convert_rot6d_layout_np(
        rotations,
        source_layout,
        target_layout,
    ).reshape(values.shape[:-1] + (num_joints * 6,))
    return values.astype(np.float32)


def so3_geodesic_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle on SO(3), in radians.

    ``atan2(sin(theta), cos(theta))`` is used instead of a raw ``acos`` so
    round-trip tests remain accurate near the identity in float32.
    """
    ra = project_to_so3_np(a).astype(np.float64)
    rb = project_to_so3_np(b).astype(np.float64)
    rel = np.swapaxes(ra, -1, -2) @ rb
    tr = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    vee = np.stack([
        rel[..., 2, 1] - rel[..., 1, 2],
        rel[..., 0, 2] - rel[..., 2, 0],
        rel[..., 1, 0] - rel[..., 0, 1],
    ], axis=-1)
    sin = 0.5 * np.linalg.norm(vee, axis=-1)
    return np.arctan2(sin, cos).astype(np.float32)


def so3_log_np(m: np.ndarray) -> np.ndarray:
    """Log map from SO(3) to rotation vectors in so(3)."""
    r = project_to_so3_np(m)
    shape = r.shape[:-2]
    flat = r.reshape(-1, 3, 3).astype(np.float64)
    if Rotation is not None:
        return Rotation.from_matrix(flat).as_rotvec().reshape(*shape, 3).astype(np.float32)

    tr = np.trace(flat, axis1=-2, axis2=-1)
    cosine = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    vee = np.stack(
        [
            flat[:, 2, 1] - flat[:, 1, 2],
            flat[:, 0, 2] - flat[:, 2, 0],
            flat[:, 1, 0] - flat[:, 0, 1],
        ],
        axis=-1,
    )
    sine = 0.5 * np.linalg.norm(vee, axis=-1)
    angle = np.arctan2(sine, cosine)
    out = np.zeros_like(vee)
    small = angle < 1.0e-5
    near_pi = np.abs(np.pi - angle) < 1.0e-4
    regular = ~(small | near_pi)
    out[small] = 0.5 * vee[small]
    if np.any(regular):
        scale = angle[regular] / np.maximum(2.0 * np.sin(angle[regular]), 1.0e-7)
        out[regular] = vee[regular] * scale[:, None]
    if np.any(near_pi):
        near = flat[near_pi]
        symmetric = 0.5 * (near + np.eye(3, dtype=np.float64)[None])
        diagonal = np.clip(np.diagonal(symmetric, axis1=-2, axis2=-1), 0.0, None)
        largest = np.argmax(diagonal, axis=-1)
        axes = np.zeros((len(near), 3), dtype=np.float64)
        for axis_index in range(3):
            mask = largest == axis_index
            if not np.any(mask):
                continue
            component = np.sqrt(np.maximum(diagonal[mask, axis_index], EPS))
            selected = symmetric[mask]
            candidate = selected[:, :, axis_index] / component[:, None]
            axes[mask] = candidate
        axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), EPS)
        near_vee = vee[near_pi]
        flip = np.sum(axes * near_vee, axis=-1) < 0.0
        axes[flip] *= -1.0
        out[near_pi] = axes * angle[near_pi, None]
    return out.reshape(*shape, 3).astype(np.float32)


def so3_exp_np(v: np.ndarray) -> np.ndarray:
    """Exponential map from rotation vectors to SO(3)."""
    x = _as_float_np(v)
    shape = x.shape[:-1]
    flat = x.reshape(-1, 3).astype(np.float64)
    if Rotation is not None:
        return Rotation.from_rotvec(flat).as_matrix().reshape(*shape, 3, 3).astype(np.float32)

    theta = np.linalg.norm(flat, axis=-1)
    axis = flat / np.maximum(theta[:, None], EPS)
    k = np.zeros((len(flat), 3, 3), dtype=np.float64)
    k[:, 0, 1] = -axis[:, 2]
    k[:, 0, 2] = axis[:, 1]
    k[:, 1, 0] = axis[:, 2]
    k[:, 1, 2] = -axis[:, 0]
    k[:, 2, 0] = -axis[:, 1]
    k[:, 2, 1] = axis[:, 0]
    ident = np.eye(3, dtype=np.float64)[None]
    out = ident + np.sin(theta)[:, None, None] * k + (1.0 - np.cos(theta))[:, None, None] * (k @ k)
    out[theta < 1.0e-8] = ident
    return out.reshape(*shape, 3, 3).astype(np.float32)


def relative_rotvec_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return Log(a^T b)."""
    ra = project_to_so3_np(a)
    rb = project_to_so3_np(b)
    return so3_log_np(np.swapaxes(ra, -1, -2) @ rb)


def angular_velocity_np(rotations: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Body-frame angular velocity, shape ``[T-1,...,3]`` in rad/s."""
    r = project_to_so3_np(rotations)
    if r.shape[0] < 2:
        return np.zeros((0,) + r.shape[1:-2] + (3,), dtype=np.float32)
    return relative_rotvec_np(r[:-1], r[1:]) * float(fps)


def angular_acceleration_np(rotations: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """First difference of body-frame angular velocity in rad/s^2."""
    w = angular_velocity_np(rotations, fps=fps)
    if w.shape[0] < 2:
        return np.zeros((0,) + w.shape[1:], dtype=np.float32)
    return np.diff(w, axis=0).astype(np.float32) * float(fps)


def tangent_blend_np(reference: np.ndarray, proposal: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Geodesically blend rotations using tangent residuals at ``reference``.

    ``weight`` is broadcast to the rotation batch prefix and clipped to [0, 1].
    """
    r0 = project_to_so3_np(reference)
    r1 = project_to_so3_np(proposal)
    delta = relative_rotvec_np(r0, r1)
    w = np.asarray(weight, dtype=np.float32)
    while w.ndim < delta.ndim:
        w = w[..., None]
    out = r0 @ so3_exp_np(np.clip(w, 0.0, 1.0) * delta)
    return project_to_so3_np(out)


def rot6d_to_matrix_torch(x: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required")
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1.0e-8)
    a2o = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2o, dim=-1, eps=1.0e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d_torch(m: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required")
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def rot6d_to_matrix_layout_torch(
    x: "torch.Tensor",
    layout: str = CANONICAL_ROT6D_LAYOUT,
) -> "torch.Tensor":
    normalized = normalize_rot6d_layout(layout)
    matrix = rot6d_to_matrix_torch(x)
    if normalized == ROT6D_LAYOUT_PYTORCH3D_ROW:
        matrix = matrix.transpose(-1, -2)
    return matrix


def matrix_to_rot6d_layout_torch(
    m: "torch.Tensor",
    layout: str = CANONICAL_ROT6D_LAYOUT,
) -> "torch.Tensor":
    normalized = normalize_rot6d_layout(layout)
    matrix = m.transpose(-1, -2) if normalized == ROT6D_LAYOUT_PYTORCH3D_ROW else m
    return matrix_to_rot6d_torch(matrix)


def convert_rot6d_layout_torch(
    x: "torch.Tensor",
    source_layout: str,
    target_layout: str,
) -> "torch.Tensor":
    source = normalize_rot6d_layout(source_layout)
    target = normalize_rot6d_layout(target_layout)
    if source == target:
        return x.clone()
    return matrix_to_rot6d_layout_torch(
        rot6d_to_matrix_layout_torch(x, source),
        target,
    )


def so3_geodesic_torch(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required")
    rel = a.transpose(-1, -2) @ b
    tr = torch.diagonal(rel, dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = torch.stack([
        rel[..., 2, 1] - rel[..., 1, 2],
        rel[..., 0, 2] - rel[..., 2, 0],
        rel[..., 1, 0] - rel[..., 0, 1],
    ], dim=-1)
    sin = 0.5 * torch.linalg.norm(vee, dim=-1)
    return torch.atan2(sin, cos)


def so3_log_torch(m: "torch.Tensor") -> "torch.Tensor":
    """Differentiable Log map for the small relative rotations in motion data."""
    if torch is None:
        raise RuntimeError("PyTorch is required")
    trace = torch.diagonal(m, dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    vee = torch.stack(
        [
            m[..., 2, 1] - m[..., 1, 2],
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 1, 0] - m[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(vee, dim=-1)
    angle = torch.atan2(sine, cosine)
    small = angle.abs() < 1.0e-4
    near_pi = (torch.pi - angle).abs() < 1.0e-4
    regular_scale = angle / (2.0 * torch.sin(angle).clamp_min(1.0e-7))
    small_scale = 0.5 + angle.square() / 12.0
    scale = torch.where(small, small_scale, regular_scale)
    result = vee * scale[..., None]

    if torch.any(near_pi):
        identity = torch.eye(3, dtype=m.dtype, device=m.device)
        symmetric = 0.5 * (m[near_pi] + identity)
        diagonal = torch.diagonal(symmetric, dim1=-2, dim2=-1).clamp_min(0.0)
        largest = diagonal.argmax(dim=-1)
        columns = symmetric.transpose(-1, -2)
        gather_index = largest[:, None, None].expand(-1, 1, 3)
        selected = torch.gather(columns, 1, gather_index).squeeze(1)
        component = torch.sqrt(
            torch.gather(diagonal, 1, largest[:, None]).squeeze(1).clamp_min(EPS)
        )
        axis = selected / component[:, None]
        axis = F.normalize(axis, dim=-1, eps=EPS)
        near_vee = vee[near_pi]
        flip = (axis * near_vee).sum(dim=-1) < 0.0
        axis = torch.where(flip[:, None], -axis, axis)
        result = result.clone()
        result[near_pi] = axis * angle[near_pi, None]
    return result


def validate_rot6d_roundtrip_np(x: np.ndarray) -> dict:
    m = rot6d_to_matrix_np(x)
    x2 = matrix_to_rot6d_np(m)
    m2 = rot6d_to_matrix_np(x2)
    err = so3_geodesic_np(m, m2)
    return {
        "max_geodesic_rad": float(np.max(err)) if err.size else 0.0,
        "mean_geodesic_rad": float(np.mean(err)) if err.size else 0.0,
        "finite": bool(np.isfinite(m2).all()),
    }
