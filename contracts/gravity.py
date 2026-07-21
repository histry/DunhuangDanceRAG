#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.49 gravity/body-frame contract for EDGE 151D.

Representation:
  [0:4]   contacts
  [4:7]   root XYZ, Y-up
  [7:151] 24 local joint rotations, column-concatenated 6D

This module is deliberately independent from tools.v46_motionrag_diff to avoid
circular imports when it is used as a training loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

from motion_geometry.rotations import matrix_to_rot6d_np, rot6d_to_matrix_np
from motion_geometry.smpl24 import (
    CONTACT,
    FOOT_JOINTS,
    MOTION_DIM as EDGE_DIM,
    NUM_JOINTS,
    OFFSETS,
    PARENTS,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
    ROT6D_END,
    ROT6D_START,
)

PELVIS = 0
NECK = 12
HEAD = 15


@dataclass
class GravityThresholds:
    torso_up_cos_p05_min: float = 0.45
    torso_up_cos_median_min: float = 0.70
    head_above_pelvis_ratio_min: float = 0.92
    feet_below_pelvis_ratio_min: float = 0.90
    horizontal_body_ratio_max: float = 0.10
    nonfinite_count_max: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def identity6d_np(shape_prefix: Tuple[int, ...] = ()) -> np.ndarray:
    base = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    return np.broadcast_to(base, tuple(shape_prefix) + (6,)).copy()


def fk24_np(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] < EDGE_DIM:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    T = x.shape[0]
    local = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6))
    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    gp = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    gr = np.zeros((T, NUM_JOINTS, 3, 3), dtype=np.float32)
    for j in range(NUM_JOINTS):
        p = int(PARENTS[j])
        if p < 0:
            gr[:, j] = local[:, j]
            gp[:, j] = root
        else:
            gr[:, j] = gr[:, p] @ local[:, j]
            gp[:, j] = gp[:, p] + (gr[:, p] @ OFFSETS[j].reshape(1, 3, 1))[..., 0]
    return gp


def gravity_metrics_np(motion: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    joints = fk24_np(x)
    pelvis = joints[:, PELVIS]
    neck = joints[:, NECK]
    head = joints[:, HEAD]
    feet = joints[:, FOOT_JOINTS]

    torso = head - pelvis
    torso_n = np.linalg.norm(torso, axis=-1)
    torso_u = torso / np.maximum(torso_n[:, None], 1e-8)
    torso_cos = torso_u[:, 1]
    horizontal = np.abs(torso_cos) < 0.35

    root_r = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_START + 6].reshape(-1, 1, 6))[:, 0]
    root_up_cos = root_r[:, 1, 1]

    foot_center_y = feet[..., 1].mean(axis=1)
    floor_y = float(np.percentile(feet[..., 1].reshape(-1), 5))
    foot_vel_xz = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet) > 1:
        foot_vel_xz[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)

    return {
        "frames": int(len(x)),
        "fps": float(fps),
        "nonfinite_count": int((~np.isfinite(x)).sum()),
        "torso_up_cos_p05": float(np.percentile(torso_cos, 5)),
        "torso_up_cos_median": float(np.median(torso_cos)),
        "torso_up_cos_p95": float(np.percentile(torso_cos, 95)),
        "torso_abs_up_cos_p05": float(np.percentile(np.abs(torso_cos), 5)),
        "horizontal_body_ratio": float(horizontal.mean()),
        "head_above_pelvis_ratio": float((head[:, 1] > pelvis[:, 1] + 0.05).mean()),
        "neck_above_pelvis_ratio": float((neck[:, 1] > pelvis[:, 1] + 0.03).mean()),
        "feet_below_pelvis_ratio": float((foot_center_y < pelvis[:, 1] - 0.15).mean()),
        "root_up_cos_p05": float(np.percentile(root_up_cos, 5)),
        "root_up_cos_median": float(np.median(root_up_cos)),
        "pelvis_y_p05": float(np.percentile(pelvis[:, 1], 5)),
        "pelvis_y_p95": float(np.percentile(pelvis[:, 1], 95)),
        "floor_y": floor_y,
        "foot_height_above_floor_p95": float(np.percentile(feet[..., 1] - floor_y, 95)),
        "foot_speed_xz_p95_mps": float(np.percentile(foot_vel_xz * float(fps), 95)),
    }


def evaluate_gravity_contract(
    metrics: Dict[str, float],
    thresholds: Optional[GravityThresholds] = None,
) -> Tuple[bool, list[str]]:
    th = thresholds or GravityThresholds()
    reasons: list[str] = []
    checks = [
        ("torso_up_cos_p05", metrics.get("torso_up_cos_p05", -1e9), th.torso_up_cos_p05_min, ">="),
        ("torso_up_cos_median", metrics.get("torso_up_cos_median", -1e9), th.torso_up_cos_median_min, ">="),
        ("head_above_pelvis_ratio", metrics.get("head_above_pelvis_ratio", -1e9), th.head_above_pelvis_ratio_min, ">="),
        ("feet_below_pelvis_ratio", metrics.get("feet_below_pelvis_ratio", -1e9), th.feet_below_pelvis_ratio_min, ">="),
    ]
    for name, value, limit, _ in checks:
        if float(value) < float(limit):
            reasons.append(f"{name}={value:.6g} < {limit:.6g}")
    if float(metrics.get("horizontal_body_ratio", 1e9)) > th.horizontal_body_ratio_max:
        reasons.append(
            f"horizontal_body_ratio={metrics.get('horizontal_body_ratio'):.6g} "
            f"> {th.horizontal_body_ratio_max:.6g}"
        )
    if int(metrics.get("nonfinite_count", 10**9)) > th.nonfinite_count_max:
        reasons.append(
            f"nonfinite_count={metrics.get('nonfinite_count')} > {th.nonfinite_count_max}"
        )
    return not reasons, reasons


def _rot6d_to_matrix_torch(x):
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2o = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2o, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _fk24_torch(motion):
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    B, T, _ = motion.shape
    local = _rot6d_to_matrix_torch(
        motion[..., ROT6D_START:ROT6D_END].reshape(B, T, NUM_JOINTS, 6)
    )
    root = motion[..., [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    parents = torch.as_tensor(PARENTS, device=motion.device, dtype=torch.long)
    offsets = torch.as_tensor(OFFSETS, device=motion.device, dtype=motion.dtype)
    gp, gr = [], []
    for j in range(NUM_JOINTS):
        p = int(parents[j].item())
        if p < 0:
            rj = local[:, :, j]
            pj = root
        else:
            rj = gr[p] @ local[:, :, j]
            off = offsets[j].view(1, 1, 3, 1)
            pj = gp[p] + (gr[p] @ off).squeeze(-1)
        gr.append(rj)
        gp.append(pj)
    return torch.stack(gp, dim=2)


def gravity_loss_torch(
    motion,
    reference=None,
    min_torso_cos: float = 0.45,
    min_head_margin: float = 0.18,
    min_feet_margin: float = 0.30,
    reference_margin: float = 0.08,
) -> Dict[str, "torch.Tensor"]:
    """Differentiable body-up contract.

    The absolute terms stop whole-body collapse.  The reference-relative term
    preserves legitimate Dunhuang bends by only penalising predicted torso-up
    values that are substantially worse than the clean/reference sequence.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required")
    j = _fk24_torch(motion)
    pelvis, head = j[:, :, PELVIS], j[:, :, HEAD]
    feet = j[:, :, FOOT_JOINTS]
    torso = head - pelvis
    torso_cos = F.normalize(torso, dim=-1, eps=1e-8)[..., 1]
    head_margin = head[..., 1] - pelvis[..., 1]
    feet_margin = pelvis[..., 1] - feet[..., 1].mean(dim=2)

    upright = F.relu(float(min_torso_cos) - torso_cos).pow(2).mean()
    head_order = F.relu(float(min_head_margin) - head_margin).pow(2).mean()
    feet_order = F.relu(float(min_feet_margin) - feet_margin).pow(2).mean()

    reference_term = motion.new_zeros(())
    if reference is not None:
        with torch.no_grad():
            jr = _fk24_torch(reference)
            tr = jr[:, :, HEAD] - jr[:, :, PELVIS]
            ref_cos = F.normalize(tr, dim=-1, eps=1e-8)[..., 1]
        reference_term = F.relu(ref_cos - float(reference_margin) - torso_cos).pow(2).mean()

    total = upright + 0.5 * head_order + 0.5 * feet_order + reference_term
    return {
        "total": total,
        "upright": upright,
        "head_order": head_order,
        "feet_order": feet_order,
        "reference": reference_term,
        "torso_cos_mean": torso_cos.mean(),
    }


__all__ = [
    "EDGE_DIM", "PARENTS", "OFFSETS", "FOOT_JOINTS",
    "GravityThresholds", "identity6d_np", "rot6d_to_matrix_np",
    "matrix_to_rot6d_np", "fk24_np", "gravity_metrics_np",
    "evaluate_gravity_contract", "gravity_loss_torch",
]
