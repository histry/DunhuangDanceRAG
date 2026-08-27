#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared SMPL14 motion geometry, training, refinement, diffusion, and IK.

Event-DB construction lives in :mod:`events.build_database`; formal retrieval
and generation orchestration live in the CTSR Scheduler and Graph-SB closed
loop.  This module intentionally contains no BVH, filename-semantic, external
pretrained music, or historical contrastive-retrieval business path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from motion_geometry.smpl24 import (
    FOOT_JOINTS as DEFAULT_FOOT_JOINTS,
    MOTION_DIM as EDGE_DIM,
    NUM_JOINTS,
    OFFSETS,
    PARENTS,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
    ROT6D_END,
    ROT6D_START,
    SMPL24_SKELETON_SCHEMA,
    skeleton_contract,
    skeleton_fingerprint,
)
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    matrix_to_rot6d_np as _contract_matrix_to_rot6d_np,
    matrix_to_rot6d_torch as _contract_matrix_to_rot6d_torch,
    rot6d_to_matrix_np as _contract_rot6d_to_matrix_np,
    rot6d_to_matrix_torch as _contract_rot6d_to_matrix_torch,
    so3_log_torch,
)
from motion_geometry.product_manifold import (
    PRODUCT_STATE_DIM,
    masked_retract_np,
    masked_retract_torch,
    product_log_np,
    product_log_torch,
)
from contracts.boundary import BODY_PARTS, build_frame_joint_risk_mask
from contracts.physical_quality import (
    PeakJerkMaskConfig,
    PhysicalQualityLimits,
    StageAcceptancePolicy,
    compute_joint_kinematic_metrics,
    evaluate_physical_audit,
    evaluate_stage_candidate,
    evaluate_stage_reference_fidelity,
)
from support.event_identity import (
    assert_same_event_db_contract,
    event_uids_from_generation_db,
    make_event_db_contract,
    normalize_event_db_contract,
)
from motion_geometry.resampling import blend_edge151_geodesic_np

LOWER_BODY_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)
FK_TREE_SOURCE = SMPL24_SKELETON_SCHEMA
REFINER_MODEL_VERSION = "product_manifold_boundary_refiner_v5"


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Optional[str | Path], default: Optional[dict] = None) -> dict:
    if not path:
        return dict(default or {})
    p = Path(path)
    if not p.exists():
        return dict(default or {})
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    base = dict(default or {})
    base.update(data)
    return base


def _json_safe(x):
    """Make report/meta objects JSON serializable.

    Event Semantics hotfix:
    Chang-E semantic ontology may contain Python set values, e.g. aliases.
    events_meta.json must remain writable, so convert sets/numpy/Path safely.
    """
    import dataclasses as _dataclasses
    import numpy as _np
    from pathlib import Path as _Path

    if _dataclasses.is_dataclass(x):
        return _json_safe(_dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, set):
        return sorted([_json_safe(v) for v in x], key=lambda z: str(z))
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, _Path):
        return str(x)
    if isinstance(x, _np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, _np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
def smooth_np(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or ndi is None:
        return x
    return ndi.gaussian_filter1d(x, sigma=float(sigma), axis=0, mode="nearest")


def median_bool_filter(x: np.ndarray, size: int) -> np.ndarray:
    if size <= 1 or ndi is None:
        return x.astype(bool)
    return ndi.median_filter(x.astype(np.uint8), size=size).astype(bool)


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def resample_motion_np(motion: np.ndarray, new_len: int) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if new_len <= 1 or motion.shape[0] <= 1:
        return np.repeat(motion[:1], max(1, new_len), axis=0)
    if motion.ndim == 2 and motion.shape[1] >= EDGE_DIM:
        from motion_geometry.resampling import resample_edge151_np

        canonical = resample_edge151_np(motion[:, :EDGE_DIM], target_frames=int(new_len))
        if motion.shape[1] == EDGE_DIM:
            return canonical
        # Non-EDGE extension channels remain ordinary Euclidean signals.
        old_x = np.linspace(0.0, 1.0, motion.shape[0])
        new_x = np.linspace(0.0, 1.0, int(new_len))
        extra = np.stack(
            [np.interp(new_x, old_x, motion[:, d]) for d in range(EDGE_DIM, motion.shape[1])],
            axis=-1,
        ).astype(np.float32)
        return np.concatenate([canonical, extra], axis=-1)
    old_x = np.linspace(0.0, 1.0, motion.shape[0])
    new_x = np.linspace(0.0, 1.0, new_len)
    out = np.empty((new_len, motion.shape[1]), dtype=np.float32)
    for d in range(motion.shape[1]):
        out[:, d] = np.interp(new_x, old_x, motion[:, d])
    return out


def rot6d_to_matrix_np(
    x: np.ndarray,
    *,
    project: bool = True,
) -> np.ndarray:
    return _contract_rot6d_to_matrix_np(x, project=project)


def matrix_to_rot6d_np(
    mat: np.ndarray,
    *,
    project: bool = True,
) -> np.ndarray:
    """Convert rotation matrices to EDGE/Zhou 6D in column-concatenated form.

    Motion Loading/Event Semantics critical fix:
    The inverse of rot6d_to_matrix_np() must concatenate the first two matrix
    columns as [R[:,0], R[:,1]].  The previous row-major expression
    ``mat[..., :, 0:2].reshape(..., 6)`` interleaves rows as
    [R00, R01, R10, R11, R20, R21], which turns the identity matrix into
    [1, 0, 0, 1, 0, 0] instead of [1, 0, 0, 0, 1, 0].  That silently corrupts
    saved Event-RAG clips and makes strict raw-rot6d audit fail even after
    projection.
    """
    return _contract_matrix_to_rot6d_np(mat, project=project)


def fk_24_np(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < ROT6D_END:
        raise ValueError(f"Expected EDGE 151D motion [T,151], got {motion.shape}")
    T = motion.shape[0]
    root = motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    rot6d = motion[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)
    # Rot6D decoding already performs a stable Gram-Schmidt projection. The
    # additional nearest-SO(3) SVD is redundant inside the repeated FK path.
    local_r = rot6d_to_matrix_np(rot6d, project=False)
    global_r = np.zeros((T, NUM_JOINTS, 3, 3), dtype=np.float32)
    joints = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    global_r[:, 0] = local_r[:, 0]
    joints[:, 0] = root
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        if p < 0:
            global_r[:, j] = local_r[:, j]
            joints[:, j] = root
        else:
            global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
            offset = OFFSETS[j].astype(np.float32)[None, :, None]
            joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], offset)[..., 0]
    return joints


def rot6d_to_matrix_torch(x):
    return _contract_rot6d_to_matrix_torch(x)


def matrix_to_rot6d_torch(mat):
    """Convert rotation matrices to Motion Generation/EDGE column-concatenated 6D.

    Must match matrix_to_rot6d_np(): [R[:,0], R[:,1]].  The old
    mat[..., :, 0:2].reshape(...) interleaves rows and corrupts identity
    rotations as [1,0,0,1,0,0] instead of [1,0,0,0,1,0].
    """
    return _contract_matrix_to_rot6d_torch(mat)


def project_rot6d_torch(x):
    return matrix_to_rot6d_torch(rot6d_to_matrix_torch(x))


_FK_OFFSETS_TORCH_CACHE: Dict[Tuple[str, Any], Any] = {}


def fk_24_torch(motion, parents=None, offsets=None):
    if parents is None:
        parent_ids = tuple(int(value) for value in PARENTS)
    elif torch.is_tensor(parents):
        parent_ids = tuple(
            int(value) for value in parents.detach().cpu().tolist()
        )
    else:
        parent_ids = tuple(int(value) for value in parents)
    if offsets is None:
        offset_key = (str(motion.device), motion.dtype)
        offsets_t = _FK_OFFSETS_TORCH_CACHE.get(offset_key)
        if offsets_t is None:
            offsets_t = torch.as_tensor(
                OFFSETS,
                device=motion.device,
                dtype=motion.dtype,
            )
            _FK_OFFSETS_TORCH_CACHE[offset_key] = offsets_t
    else:
        offsets_t = torch.as_tensor(
            offsets,
            device=motion.device,
            dtype=motion.dtype,
        )
    if motion.ndim < 2 or motion.shape[-1] != EDGE_DIM:
        raise ValueError(f"Expected [...,{EDGE_DIM}], got {tuple(motion.shape)}")
    leading = motion.shape[:-1]
    flat = motion.reshape(-1, EDGE_DIM)
    root = flat[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    rot6d = flat[:, ROT6D_START:ROT6D_END].reshape(-1, NUM_JOINTS, 6)
    local_r = rot6d_to_matrix_torch(rot6d)
    global_r = []
    joints = []
    for j in range(NUM_JOINTS):
        p = parent_ids[j]
        if j == 0 or p < 0:
            gr = local_r[:, j]
            pos = root
        else:
            gr = torch.matmul(global_r[p], local_r[:, j])
            off = offsets_t[j].view(1, 3, 1)
            pos = joints[p] + torch.matmul(global_r[p], off).squeeze(-1)
        global_r.append(gr)
        joints.append(pos)
    return torch.stack(joints, dim=1).reshape(*leading, NUM_JOINTS, 3)


def root_yaw_np(motion: np.ndarray) -> np.ndarray:
    root_r = rot6d_to_matrix_np(motion[:, ROT6D_START:ROT6D_START + 6].reshape(-1, 1, 6))[:, 0]
    forward = root_r[:, :, 2]
    return np.arctan2(forward[:, 0], forward[:, 2]).astype(np.float32)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b)).astype(np.float32)


@dataclasses.dataclass
class MotionGenerationConfig:
    fps: float = 30.0
    window_len: int = 120
    hop_len: int = 60
    min_event_frames: int = 36
    max_event_frames: int = 120
    overlap: int = 12
    transition_train_min_seconds: float = 10.0 / 30.0
    transition_train_max_seconds: float = 28.0 / 30.0
    transition_mask_halo_seconds: float = 6.0 / 30.0
    transition_tangent_smoothing_passes: int = 4
    ik_enable: bool = True
    ik_iters: int = 120
    ik_lr: float = 0.020
    ik_chunk: int = 240
    ik_pose_w: float = 0.035
    ik_temporal_w: float = 0.055
    ik_root_w: float = 0.010
    ik_contact_w: float = 8.0
    ik_penetration_w: float = 12.0
    ik_contact_high: float = 0.70
    ik_contact_low: float = 0.45
    ik_height_margin: float = 0.035
    ik_speed_gate_mps: float = 0.36
    ik_contact_break_speed_mps: float = 0.54
    ik_hard_contact_lock: bool = True
    ik_hard_contact_min_confidence: float = 0.85
    ik_contact_ramp_seconds: float = 4.0 / 30.0
    ik_max_delta_rot: float = 0.30
    # Sliding-Support IK: cloud-step is not a release / continue any more.  Large XZ travel
    # in contact is classified by speed and then mapped to a sliding anchor.
    # This avoids the Footskate Forgiveness Paradox: severe slow AI drifting is
    # still locked, while true Dunhuang cloud-step gets a smooth moving target.
    ik_slide_release_m: float = 0.05
    ik_slide_release_min_seconds: float = 4.0 / 30.0
    ik_cloud_step_speed_mps: float = 0.15
    ik_sliding_anchor_seconds: float = 10.0 / 30.0
    ik_cloud_speed_cv_max: float = 1.75
    # Root-Vertical Dynamics: root-Y ballistic/damping pass. It is deliberately C1-safe and
    # never breaks a damping cycle mid-contact.
    # Disabled by default: contact labels must first pass the final contact
    # reconstruction gate.  Enabling ballistics on sparse/corrupt contacts can
    # manufacture long root-Y excursions that are not present in the event.
    root_y_physics_enable: bool = False
    root_y_flight_strength: float = 0.18
    root_y_min_flight_seconds: float = 3.0 / 30.0
    # Flight Safety Fuse: biological fuse. If the no-contact interval is longer than this,
    # treat it as corrupted contact labels / bad upstream generation, not a
    # real human jump. Do not inject a huge ballistic parabola or landing dip.
    root_y_max_flight_seconds: float = 1.20
    root_y_damping_max_dip: float = 0.018
    # Root-Aware Motion Safety: cap landing damping to an early post-touchdown window.  The window
    # still starts and ends at zero dip, but it no longer stretches across a
    # multi-second support island and therefore cannot create delayed squats.
    root_y_damping_max_seconds: float = 0.28

    # Root-Aware Motion Safety: root-aware cloud-step guard. A true cloud-step requires foot travel
    # to be consistent with root/CoM translation; smooth AI dark-drift with no
    # body support is kept on the static-anchor repair path.
    ik_cloud_root_min_travel_m: float = 0.045
    ik_cloud_direction_cos_min: float = 0.35
    ik_cloud_root_foot_rel_max_m: float = 0.18

    # Root-Aware Motion Safety: long-sequence IK stitching and rollback safety.
    ik_chunk_overlap: int = 24
    rollback_root_delta_max_m: float = 0.12
    ik_post_stabilize_enable: bool = True
    ik_post_stabilize_passes: int = 2

    lower_body_only: bool = True
    refiner_enable: bool = True
    diffusion_enable: bool = True
    # Geometry-safe Motion Refiner and Motion Diffusion use a 79D
    # contact/root/joint-tangent state.
    product_refiner_rotation_cap_rad: float = 0.35
    product_refiner_root_cap_m: float = 0.08
    product_refiner_outside_weight: float = 0.25
    # The synthetic transition occupies only a minority of a training window.
    # Keep a small whole-window fidelity term, but normalize the main geometry
    # objective on the actually corrupted seam so authentic high-motion frames
    # selected by the Peak-Jerk mask cannot dilute the repair signal.
    product_refiner_global_reconstruction_weight: float = 0.05
    product_refiner_active_reconstruction_weight: float = 0.0
    product_refiner_tangent_supervision_weight: float = 0.0
    product_refiner_repair_margin_weight: float = 1.0
    product_refiner_clean_preservation_weight: float = 1.0
    # Endpoint inbetweening deliberately removes unobservable interior detail.
    # Train a bounded relative improvement rather than asking a deterministic
    # refiner to reconstruct that missing trajectory exactly.
    product_refiner_training_target_repair_gain: float = 0.10
    product_refiner_temporal_supervision_weight: float = 0.0
    # A degraded-only training distribution never teaches the network that an
    # already-clean transition must remain unchanged. The paired identity
    # branch uses the same seam/risk contract with authentic clean input.
    product_refiner_clean_identity_weight: float = 0.50
    product_refiner_clean_identity_temporal_weight: float = 1.0
    product_refiner_relative_temporal_weight: float = 0.25
    product_refiner_residual_smoothing_passes: int = 2
    product_refiner_residual_taper_frames: int = 3
    product_refiner_endpoint_continuity_weight: float = 0.50
    product_refiner_seam_velocity_weight: float = 0.20
    product_refiner_seam_acceleration_weight: float = 0.25
    product_refiner_seam_jerk_weight: float = 0.30
    # A full endpoint-only inbetween discards the unknown clean interior and
    # makes source-disjoint reconstruction ill posed.  Retain half of the
    # authentic local trajectory while injecting the same transition artifact
    # seen by the boundary stage.
    transition_bridge_mix: float = 0.50
    # Differentiable world-space supervision shared by Motion Refiner and Motion Generation.  The
    # acceleration/jerk residuals are normalized inside the loss, so these
    # weights are comparable to the product-manifold reconstruction term.
    physics_fk_loss_weight: float = 0.08
    physics_foot_loss_weight: float = 0.12
    physics_support_loss_weight: float = 0.10
    physics_penetration_loss_weight: float = 0.08
    physics_acceleration_loss_weight: float = 0.02
    physics_jerk_loss_weight: float = 0.01
    physics_static_support_speed_mps: float = 0.18
    # Checkpoint validation is a stage-relative contract.  The final absolute
    # generation gate remains diagnostic here and becomes authoritative only
    # after Diffusion, boundary repair and IK.
    checkpoint_validation_fail_closed: bool = True
    checkpoint_validation_min_stage_repair_rate: float = 0.75
    checkpoint_validation_min_temporal_repair_rate: float = 0.75
    # Kept for report compatibility; full-window fidelity is diagnostic only.
    checkpoint_validation_min_clean_fidelity_rate: float = 0.75
    checkpoint_validation_min_clean_identity_rate: float = 0.75
    checkpoint_validation_max_fk_p95_m: float = 0.15
    checkpoint_validation_max_fk_max_m: float = 1.0
    checkpoint_validation_max_refiner_product_log_l1: float = 0.03
    checkpoint_validation_min_geometry_repair_gain: float = 0.03
    checkpoint_validation_min_temporal_repair_gain: float = 0.03
    checkpoint_validation_min_endpoint_repair_gain: float = 0.03
    checkpoint_validation_max_clean_identity_product_log_l1: float = 0.005
    checkpoint_validation_max_clean_identity_contact_l1: float = 0.05
    # Whole-sequence acceptance is authoritative. A candidate that fails the
    # SI physical gate is written only as an explicitly rejected diagnostic,
    # never to the requested accepted-output path.
    final_physical_gate_enable: bool = True
    final_physical_gate_fail_closed: bool = True
    final_physical_gate_save_rejected: bool = True
    tangent_diffusion_rotation_cap_rad: float = 0.45
    tangent_diffusion_root_cap_m: float = 0.10
    riemannian_trust_region_enable: bool = True
    riemannian_trust_region_steps: int = 5
    riemannian_trust_region_initial_radius: float = 1.0
    riemannian_trust_region_min_radius: float = 0.0625
    diffusion_steps: int = 50
    diffusion_train_steps: int = 15000
    refiner_train_steps: int = 8000
    training_snapshot_interval_steps: int = 200
    training_validation_interval_steps: int = 1000
    batch_size: int = 64
    inference_window_batch_size: int = 16
    gpu_preprocessing: bool = True
    # Bound diffusion inference noise independently of whole-song duration.
    # One window stores [diffusion_steps, window_len, PRODUCT_STATE_DIM]
    # float values; the runtime further caps the configured inference batch so
    # this temporary tensor cannot grow without limit.
    diffusion_noise_batch_max_mib: float = 256.0
    lr: float = 2e-4
    seed: int = 42
    device: str = "cuda"

    @staticmethod
    def from_json(path: Optional[str | Path]) -> "MotionGenerationConfig":
        cfg = MotionGenerationConfig()
        if path and Path(path).exists():
            data = load_json(path)
            if "fps" in data:
                cfg.fps = float(data["fps"])
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def apply_env(self) -> "MotionGenerationConfig":
        env_map = {
            "MOTION_FPS": ("fps", float),
            "MOTION_ENABLE_TRUE_IK": ("ik_enable", lambda x: bool(int(x))),
            "MOTION_ENABLE_REFINER": ("refiner_enable", lambda x: bool(int(x))),
            "MOTION_ENABLE_DIFFUSION": ("diffusion_enable", lambda x: bool(int(x))),
            "MOTION_PRODUCT_REFINER_ROTATION_CAP_RAD": ("product_refiner_rotation_cap_rad", float),
            "MOTION_PRODUCT_REFINER_ROOT_CAP_M": ("product_refiner_root_cap_m", float),
            "MOTION_PRODUCT_REFINER_OUTSIDE_WEIGHT": ("product_refiner_outside_weight", float),
            "MOTION_PRODUCT_REFINER_GLOBAL_RECONSTRUCTION_WEIGHT": (
                "product_refiner_global_reconstruction_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_ACTIVE_RECONSTRUCTION_WEIGHT": (
                "product_refiner_active_reconstruction_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_TANGENT_SUPERVISION_WEIGHT": (
                "product_refiner_tangent_supervision_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_REPAIR_MARGIN_WEIGHT": (
                "product_refiner_repair_margin_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_CLEAN_PRESERVATION_WEIGHT": (
                "product_refiner_clean_preservation_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_TRAINING_TARGET_REPAIR_GAIN": (
                "product_refiner_training_target_repair_gain",
                float,
            ),
            "MOTION_PRODUCT_REFINER_TEMPORAL_SUPERVISION_WEIGHT": (
                "product_refiner_temporal_supervision_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_CLEAN_IDENTITY_WEIGHT": (
                "product_refiner_clean_identity_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_CLEAN_IDENTITY_TEMPORAL_WEIGHT": (
                "product_refiner_clean_identity_temporal_weight", float,
            ),
            "MOTION_PRODUCT_REFINER_RELATIVE_TEMPORAL_WEIGHT": (
                "product_refiner_relative_temporal_weight", float,
            ),
            "MOTION_PRODUCT_REFINER_RESIDUAL_SMOOTHING_PASSES": (
                "product_refiner_residual_smoothing_passes", int,
            ),
            "MOTION_PRODUCT_REFINER_RESIDUAL_TAPER_FRAMES": (
                "product_refiner_residual_taper_frames", int,
            ),
            "MOTION_PRODUCT_REFINER_ENDPOINT_CONTINUITY_WEIGHT": (
                "product_refiner_endpoint_continuity_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_SEAM_VELOCITY_WEIGHT": (
                "product_refiner_seam_velocity_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_SEAM_ACCELERATION_WEIGHT": (
                "product_refiner_seam_acceleration_weight",
                float,
            ),
            "MOTION_PRODUCT_REFINER_SEAM_JERK_WEIGHT": (
                "product_refiner_seam_jerk_weight",
                float,
            ),
            "MOTION_TRANSITION_BRIDGE_MIX": ("transition_bridge_mix", float),
            "MOTION_PHYSICS_FK_LOSS_WEIGHT": ("physics_fk_loss_weight", float),
            "MOTION_PHYSICS_FOOT_LOSS_WEIGHT": ("physics_foot_loss_weight", float),
            "MOTION_PHYSICS_SUPPORT_LOSS_WEIGHT": ("physics_support_loss_weight", float),
            "MOTION_PHYSICS_PENETRATION_LOSS_WEIGHT": ("physics_penetration_loss_weight", float),
            "MOTION_PHYSICS_ACCELERATION_LOSS_WEIGHT": ("physics_acceleration_loss_weight", float),
            "MOTION_PHYSICS_JERK_LOSS_WEIGHT": ("physics_jerk_loss_weight", float),
            "MOTION_PHYSICS_STATIC_SUPPORT_SPEED_MPS": ("physics_static_support_speed_mps", float),
            "CONTACT_STATIC_SUPPORT_SPEED_MPS": ("physics_static_support_speed_mps", float),
            "MOTION_FINAL_PHYSICAL_GATE_ENABLE": ("final_physical_gate_enable", lambda x: bool(int(x))),
            "MOTION_FINAL_PHYSICAL_GATE_FAIL_CLOSED": ("final_physical_gate_fail_closed", lambda x: bool(int(x))),
            "MOTION_FINAL_PHYSICAL_GATE_SAVE_REJECTED": ("final_physical_gate_save_rejected", lambda x: bool(int(x))),
            "MOTION_TANGENT_DIFFUSION_ROTATION_CAP_RAD": ("tangent_diffusion_rotation_cap_rad", float),
            "MOTION_TANGENT_DIFFUSION_ROOT_CAP_M": ("tangent_diffusion_root_cap_m", float),
            "MOTION_RIEMANNIAN_TRUST_REGION_ENABLE": ("riemannian_trust_region_enable", lambda x: bool(int(x))),
            "MOTION_RIEMANNIAN_TRUST_REGION_STEPS": ("riemannian_trust_region_steps", int),
            "MOTION_RIEMANNIAN_TRUST_REGION_INITIAL_RADIUS": ("riemannian_trust_region_initial_radius", float),
            "MOTION_RIEMANNIAN_TRUST_REGION_MIN_RADIUS": ("riemannian_trust_region_min_radius", float),
            "MOTION_OVERLAP": ("overlap", int),
            "MOTION_TRANSITION_TRAIN_MIN_SECONDS": ("transition_train_min_seconds", float),
            "MOTION_TRANSITION_TRAIN_MAX_SECONDS": ("transition_train_max_seconds", float),
            "MOTION_TRANSITION_MASK_HALO_SECONDS": ("transition_mask_halo_seconds", float),
            "MOTION_TRANSITION_TANGENT_SMOOTHING_PASSES": (
                "transition_tangent_smoothing_passes",
                int,
            ),
            "MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED": (
                "checkpoint_validation_fail_closed",
                lambda x: bool(int(x)),
            ),
            "MOTION_CHECKPOINT_VALIDATION_MIN_STAGE_REPAIR_RATE": (
                "checkpoint_validation_min_stage_repair_rate",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MIN_TEMPORAL_REPAIR_RATE": (
                "checkpoint_validation_min_temporal_repair_rate",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MIN_CLEAN_FIDELITY_RATE": (
                "checkpoint_validation_min_clean_fidelity_rate",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MIN_CLEAN_IDENTITY_RATE": (
                "checkpoint_validation_min_clean_identity_rate",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MAX_FK_P95_M": (
                "checkpoint_validation_max_fk_p95_m",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MAX_FK_MAX_M": (
                "checkpoint_validation_max_fk_max_m",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MAX_REFINER_PRODUCT_LOG_L1": (
                "checkpoint_validation_max_refiner_product_log_l1",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MIN_GEOMETRY_REPAIR_GAIN": (
                "checkpoint_validation_min_geometry_repair_gain",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MIN_TEMPORAL_REPAIR_GAIN": (
                "checkpoint_validation_min_temporal_repair_gain",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MIN_ENDPOINT_REPAIR_GAIN": (
                "checkpoint_validation_min_endpoint_repair_gain",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MAX_CLEAN_IDENTITY_PRODUCT_LOG_L1": (
                "checkpoint_validation_max_clean_identity_product_log_l1",
                float,
            ),
            "MOTION_CHECKPOINT_VALIDATION_MAX_CLEAN_IDENTITY_CONTACT_L1": (
                "checkpoint_validation_max_clean_identity_contact_l1",
                float,
            ),
            "MOTION_WINDOW_LEN": ("window_len", int),
            "MOTION_HOP_LEN": ("hop_len", int),
            "MOTION_INFERENCE_WINDOW_BATCH_SIZE": (
                "inference_window_batch_size",
                int,
            ),
            "MOTION_GPU_PREPROCESSING": (
                "gpu_preprocessing",
                lambda x: bool(int(x)),
            ),
            "MOTION_DIFFUSION_NOISE_BATCH_MAX_MIB": (
                "diffusion_noise_batch_max_mib",
                float,
            ),
            "MOTION_TRAINING_SNAPSHOT_INTERVAL_STEPS": (
                "training_snapshot_interval_steps",
                int,
            ),
            "MOTION_TRAINING_VALIDATION_INTERVAL_STEPS": (
                "training_validation_interval_steps",
                int,
            ),
            "MOTION_MIN_EVENT_FRAMES": ("min_event_frames", int),
            "MOTION_MAX_EVENT_FRAMES": ("max_event_frames", int),
            "MOTION_IK_ITERS": ("ik_iters", int),
            "MOTION_IK_CONTACT_W": ("ik_contact_w", float),
            "MOTION_IK_PENETRATION_W": ("ik_penetration_w", float),
            "MOTION_IK_CONTACT_HIGH": ("ik_contact_high", float),
            "MOTION_IK_CONTACT_LOW": ("ik_contact_low", float),
            "MOTION_IK_SPEED_GATE_MPS": ("ik_speed_gate_mps", float),
            "CONTACT_IK_LOCK_SPEED_MPS": ("ik_speed_gate_mps", float),
            "MOTION_IK_CONTACT_BREAK_SPEED_MPS": ("ik_contact_break_speed_mps", float),
            "MOTION_IK_HARD_CONTACT_LOCK": ("ik_hard_contact_lock", lambda x: bool(int(x))),
            "MOTION_IK_HARD_CONTACT_MIN_CONFIDENCE": ("ik_hard_contact_min_confidence", float),
            "MOTION_IK_SLIDE_RELEASE_M": ("ik_slide_release_m", float),
            "MOTION_IK_CLOUD_STEP_SPEED_MPS": ("ik_cloud_step_speed_mps", float),
            "MOTION_IK_SLIDING_ANCHOR_SECONDS": ("ik_sliding_anchor_seconds", float),
            "MOTION_IK_CLOUD_SPEED_CV_MAX": ("ik_cloud_speed_cv_max", float),
            "MOTION_IK_CLOUD_ROOT_MIN_TRAVEL_M": ("ik_cloud_root_min_travel_m", float),
            "MOTION_IK_CLOUD_DIRECTION_COS_MIN": ("ik_cloud_direction_cos_min", float),
            "MOTION_IK_CLOUD_ROOT_FOOT_REL_MAX_M": ("ik_cloud_root_foot_rel_max_m", float),
            "MOTION_IK_CHUNK_OVERLAP": ("ik_chunk_overlap", int),
            "MOTION_IK_CONTACT_RAMP_SECONDS": ("ik_contact_ramp_seconds", float),
            "MOTION_IK_POST_STABILIZE_ENABLE": ("ik_post_stabilize_enable", lambda x: bool(int(x))),
            "MOTION_IK_POST_STABILIZE_PASSES": ("ik_post_stabilize_passes", int),
            "MOTION_ROLLBACK_ROOT_DELTA_MAX_M": ("rollback_root_delta_max_m", float),
            "MOTION_ROOT_Y_DAMPING_MAX_SECONDS": ("root_y_damping_max_seconds", float),
            "MOTION_ENABLE_ROOT_Y_PHYSICS": ("root_y_physics_enable", lambda x: bool(int(x))),
            "MOTION_ROOT_Y_MIN_FLIGHT_SECONDS": ("root_y_min_flight_seconds", float),
            "MOTION_ROOT_Y_MAX_FLIGHT_SECONDS": ("root_y_max_flight_seconds", float),
            "MOTION_DIFFUSION_STEPS": ("diffusion_steps", int),
            "MOTION_DEVICE": ("device", str),
        }
        for e, (attr, caster) in env_map.items():
            if e in os.environ:
                setattr(self, attr, caster(os.environ[e]))
        if torch is None:
            if str(self.device).startswith("cuda"):
                self.device = "cpu"
            return self
        try:
            resolved_device = torch.device(self.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid MOTION_DEVICE {self.device!r}") from exc
        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                print(
                    json.dumps(
                        {
                            "stage": "motion_device_resolution",
                            "requested": str(self.device),
                            "resolved": "cpu",
                            "reason": "cuda_unavailable",
                        }
                    ),
                    flush=True,
                )
                self.device = "cpu"
            elif (
                resolved_device.index is not None
                and resolved_device.index >= torch.cuda.device_count()
            ):
                raise ValueError(
                    f"MOTION_DEVICE {self.device!r} selects CUDA index "
                    f"{resolved_device.index}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are visible"
                )
        return self


MOTION_CHECKPOINT_CONTRACT_SCHEMA = "dunhuang_motion_checkpoint_contract_v2"


def motion_checkpoint_contract(cfg: MotionGenerationConfig, role: str) -> Dict[str, Any]:
    """Return the immutable representation/time contract embedded in a checkpoint."""
    contract = {
        "schema": MOTION_CHECKPOINT_CONTRACT_SCHEMA,
        "role": str(role),
        "fps": float(cfg.fps),
        "motion_dim": int(EDGE_DIM),
        "window_len": int(cfg.window_len),
        "window_seconds": float(cfg.window_len) / max(float(cfg.fps), 1.0e-8),
        "rot6d_layout": CANONICAL_ROT6D_LAYOUT,
        "skeleton_schema": SMPL24_SKELETON_SCHEMA,
        "skeleton_sha256": skeleton_fingerprint(),
        "derivative_units": {
            "linear_velocity": "m/s",
            "linear_acceleration": "m/s^2",
            "linear_jerk": "m/s^3",
            "angular_velocity": "rad/s",
            "angular_acceleration": "rad/s^2",
        },
    }
    if role == "boundary_refiner":
        contract["refiner_tangent_mask_semantics"] = (
            "soft_scale_then_cap_applied_tangent_v1"
        )
    return contract


def assert_motion_checkpoint_contract(
    checkpoint: Dict[str, Any],
    cfg: MotionGenerationConfig,
    path: str | Path,
    role: str,
) -> None:
    """Reject mixed-FPS, mixed-Rot6D and mixed-skeleton model assets."""
    actual = checkpoint.get("motion_contract")
    if not isinstance(actual, dict):
        raise RuntimeError(
            f"Checkpoint {path} has no {MOTION_CHECKPOINT_CONTRACT_SCHEMA}. "
            "It must be rebuilt for the current SMPL14 repository contract."
        )

    expected = motion_checkpoint_contract(cfg, role)
    mismatches: List[str] = []
    for key in ("schema", "role", "motion_dim", "rot6d_layout", "skeleton_schema", "skeleton_sha256"):
        actual_value = actual.get(key)
        expected_value = expected[key]
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: checkpoint={actual.get(key)!r}, runtime={expected[key]!r}"
            )
    try:
        if abs(float(actual.get("fps")) - expected["fps"]) > 1.0e-6:
            mismatches.append(f"fps: checkpoint={actual.get('fps')!r}, runtime={expected['fps']!r}")
    except (TypeError, ValueError):
        mismatches.append(f"fps: checkpoint={actual.get('fps')!r}, runtime={expected['fps']!r}")
    if role in {"boundary_refiner", "motion_diffusion"}:
        if int(actual.get("window_len", -1)) != expected["window_len"]:
            mismatches.append(
                f"window_len: checkpoint={actual.get('window_len')!r}, runtime={expected['window_len']!r}"
            )
    if role == "boundary_refiner":
        key = "refiner_tangent_mask_semantics"
        if actual.get(key) != expected[key]:
            mismatches.append(
                f"{key}: checkpoint={actual.get(key)!r}, "
                f"runtime={expected[key]!r}"
            )
    if mismatches:
        raise RuntimeError(
            f"Checkpoint contract mismatch for {path}: " + "; ".join(mismatches)
        )
    expected_db = normalize_event_db_contract(
        getattr(cfg, "_event_db_contract", None)
    )
    if expected_db is not None:
        checkpoint_db = normalize_event_db_contract(
            checkpoint.get("training_event_db_contract")
        )
        assert_same_event_db_contract(
            expected_db,
            checkpoint_db,
            context=f"{role} checkpoint/Generation Event-DB alignment ({path})",
        )


def event_descriptor(motion: np.ndarray, fps: float = 30.0) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    T = motion.shape[0]
    joints = fk_24_np(motion)
    root = motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    root_v = np.zeros_like(root)
    root_v[1:] = (root[1:] - root[:-1]) * float(fps)
    joint_v = np.zeros_like(joints)
    joint_v[1:] = (joints[1:] - joints[:-1]) * float(fps)
    foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
    foot_vxz = np.zeros(foot.shape[:2], dtype=np.float32)
    foot_vxz[1:] = (
        np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
        * float(fps)
    )
    foot_y = foot[..., 1]
    floor = np.percentile(foot_y.reshape(-1), 5)
    contact = (foot_y < floor + 0.05) & (foot_vxz < 0.75)
    yaw = root_yaw_np(motion)
    yaw_v = np.zeros_like(yaw)
    yaw_v[1:] = angle_diff(yaw[1:], yaw[:-1]) * float(fps)
    lower_ids = [1, 2, 4, 5, 7, 8, 10, 11]
    upper_ids = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_energy = float(np.mean(np.linalg.norm(joint_v[:, lower_ids], axis=-1)))
    upper_energy = float(np.mean(np.linalg.norm(joint_v[:, upper_ids], axis=-1)))
    desc = np.array(
        [
            T / fps,
            np.linalg.norm(root[-1, [0, 2]] - root[0, [0, 2]]),
            np.mean(np.linalg.norm(root_v[:, [0, 2]], axis=-1)),
            np.percentile(np.linalg.norm(root_v[:, [0, 2]], axis=-1), 95),
            np.mean(np.abs(root_v[:, 1])),
            np.mean(np.linalg.norm(joint_v.reshape(T, -1, 3), axis=-1)),
            np.percentile(np.linalg.norm(joint_v.reshape(T, -1, 3), axis=-1), 95),
            lower_energy,
            upper_energy,
            lower_energy / max(upper_energy, 1e-6),
            np.mean(contact),
            np.mean(contact[:, :2]),
            np.mean(contact[:, 2:]),
            np.mean(foot_vxz),
            np.percentile(foot_vxz, 95),
            float(angle_diff(yaw[-1:], yaw[:1])[0]) if len(yaw) else 0.0,
            float(np.mean(np.abs(yaw_v))),
            float(np.percentile(np.abs(yaw_v), 95)),
            float(np.max(root[:, 1]) - np.min(root[:, 1])),
            float(np.mean(np.abs(motion[:, ROT6D_START:ROT6D_END]))),
        ],
        dtype=np.float32,
    )
    stats = []
    for q in [5, 25, 50, 75, 95]:
        stats.append(np.percentile(np.linalg.norm(joint_v.reshape(T, -1, 3), axis=-1), q))
    desc = np.concatenate([desc, np.asarray(stats, dtype=np.float32)], axis=0)
    if desc.shape[0] < 32:
        desc = np.pad(desc, (0, 32 - desc.shape[0]))
    # v2 local-action features occupy previously unused descriptor channels.
    # They are computed from the window itself, never from its dance theme.
    pelvis_height = joints[:, 0, 1] - float(floor)
    floorwork_ratio = float(np.mean(pelvis_height < 0.55))
    airborne_ratio = float(np.mean(np.sum(contact, axis=1) == 0))
    joint_speed = np.linalg.norm(joint_v.reshape(T, -1, 3), axis=-1)
    mean_speed = float(np.mean(joint_speed))
    burstiness = float(
        np.clip(
            (float(np.percentile(joint_speed, 95)) / max(mean_speed, 1.0e-6) - 1.0)
            / 4.0,
            0.0,
            1.0,
        )
    )
    midpoint = max(1, T // 2)
    first_energy = float(np.mean(joint_speed[:midpoint]))
    second_energy = float(np.mean(joint_speed[midpoint:])) if midpoint < T else first_energy
    transition_contrast = float(
        np.clip(
            abs(second_energy - first_energy)
            / max(first_energy + second_energy, 1.0e-6),
            0.0,
            1.0,
        )
    )
    desc[25] = floorwork_ratio
    desc[26] = airborne_ratio
    desc[27] = burstiness
    desc[28] = float(np.percentile(pelvis_height, 10))
    desc[29] = float(upper_energy / max(lower_energy + upper_energy, 1.0e-6))
    desc[30] = transition_contrast
    desc[31] = 2.0  # descriptor schema marker: local_action_v2
    return desc[:32].astype(np.float32)


def motion_boundary_state(
    motion: np.ndarray,
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joints = fk_24_np(motion)
    v = np.zeros_like(joints)
    v[1:] = (joints[1:] - joints[:-1]) * float(fps)
    entry = np.concatenate([joints[0].reshape(-1), v[min(1, len(v) - 1)].reshape(-1)], axis=0)
    exit_ = np.concatenate([joints[-1].reshape(-1), v[-1].reshape(-1)], axis=0)
    foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
    foot_vxz = np.zeros(foot.shape[:2], dtype=np.float32)
    foot_vxz[1:] = (
        np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
        * float(fps)
    )
    floor = np.percentile(foot[..., 1].reshape(-1), 5)
    contact = ((foot[..., 1] < floor + 0.05) & (foot_vxz < 0.75)).astype(np.float32)
    return entry.astype(np.float32), exit_.astype(np.float32), contact[0], contact[-1]


































# -----------------------------------------------------------------------------
# Event Semantics research contract guards for Chang-E/change RAG DB
# -----------------------------------------------------------------------------
def identity6d_np(shape_prefix: Tuple[int, ...] = ()) -> np.ndarray:
    """Return identity rotation in the repository's 6D convention."""
    base = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if not shape_prefix:
        return base.copy()
    return np.broadcast_to(base, tuple(shape_prefix) + (6,)).copy().astype(np.float32)


def sanitize_rot6d_np(rot6d: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Replace invalid / degenerate 6D rotations with identity before projection."""
    r = np.asarray(rot6d, dtype=np.float32).copy()
    if r.size == 0:
        return r.astype(np.float32), {"bad_joint_count": 0, "bad_joint_ratio": 0.0}
    r = r.reshape(-1, NUM_JOINTS, 6)
    finite = np.isfinite(r).all(axis=-1)
    a1 = r[..., 0:3]
    a2 = r[..., 3:6]
    a1_clean = np.nan_to_num(a1, nan=0.0, posinf=0.0, neginf=0.0)
    a2_clean = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0)
    n1 = np.linalg.norm(a1_clean, axis=-1)
    n2 = np.linalg.norm(a2_clean, axis=-1)
    # Event Semantics: also reject near-collinear 6D vectors.  Gram-Schmidt
    # can collapse when a1 and a2 are parallel/anti-parallel even if both
    # vector norms are valid, which can happen during early diffusion denoising.
    denom = np.maximum(n1 * n2, 1e-8)
    cross_norm = np.linalg.norm(np.cross(a1_clean, a2_clean), axis=-1) / denom
    bad = (~finite) | (n1 < 1e-5) | (n2 < 1e-5) | (cross_norm < 1e-5)
    bad_count = int(np.sum(bad))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if bad_count:
        r[bad] = identity6d_np((bad_count,))
    report = {
        "bad_joint_count": bad_count,
        "bad_joint_ratio": float(bad_count / max(1, bad.size)),
        "min_a1_norm_before_identity": float(np.nanmin(n1)) if n1.size else 0.0,
        "min_a2_norm_before_identity": float(np.nanmin(n2)) if n2.size else 0.0,
        "min_cross_norm_before_identity": float(np.nanmin(cross_norm)) if cross_norm.size else 0.0,
        "near_collinear_joint_count": int(np.sum(cross_norm < 1e-5)) if cross_norm.size else 0,
    }
    return r.reshape(np.asarray(rot6d).shape).astype(np.float32), report


def project_edge151_rot6d_np(motion: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Project every EDGE 6D rotation channel back to SO(3)-derived 6D safely."""
    x = np.asarray(motion, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] < EDGE_DIM or x.shape[0] <= 0:
        return x.astype(np.float32), {"projected": False, "reason": "invalid_shape"}
    rot = x[:, ROT6D_START:ROT6D_END].reshape(x.shape[0], NUM_JOINTS, 6)
    rot, sanitize_report = sanitize_rot6d_np(rot)
    x[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(
        rot6d_to_matrix_np(
            rot.reshape(x.shape[0], NUM_JOINTS, 6),
            project=False,
        ),
        project=False,
    ).reshape(x.shape[0], -1)
    sanitize_report["projected"] = True
    return x.astype(np.float32), sanitize_report


def rotate_motion_around_y_np(motion: np.ndarray, yaw_delta: float, pivot_xz: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate a whole EDGE-151D motion around the vertical Y axis.

    This is a world-space rigid yaw transform for event stitching. It rotates
    the root XZ trajectory around ``pivot_xz`` and left-multiplies the root
    joint rotation by R_y(yaw_delta). Child joint local rotations remain valid
    because the root orientation carries the global heading change.
    """
    out = np.asarray(motion, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < ROT6D_END or out.shape[0] <= 0:
        return out.astype(np.float32)
    yaw = float(yaw_delta)
    if not np.isfinite(yaw) or abs(yaw) < 1e-8:
        return out.astype(np.float32)
    c = float(np.cos(yaw))
    ss = float(np.sin(yaw))
    if pivot_xz is None:
        pivot = out[0, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    else:
        pivot = np.asarray(pivot_xz, dtype=np.float32).reshape(2)
    rel_x = out[:, ROOT_X_IDX].copy() - float(pivot[0])
    rel_z = out[:, ROOT_Z_IDX].copy() - float(pivot[1])
    out[:, ROOT_X_IDX] = c * rel_x + ss * rel_z + float(pivot[0])
    out[:, ROOT_Z_IDX] = -ss * rel_x + c * rel_z + float(pivot[1])

    ry = np.asarray([[c, 0.0, ss], [0.0, 1.0, 0.0], [-ss, 0.0, c]], dtype=np.float32)
    root6 = out[:, ROT6D_START:ROT6D_START + 6].reshape(out.shape[0], 1, 6)
    root_r = rot6d_to_matrix_np(root6)
    root_r = np.matmul(ry[None, None, :, :], root_r).astype(np.float32)
    out[:, ROT6D_START:ROT6D_START + 6] = matrix_to_rot6d_np(root_r).reshape(out.shape[0], 6)
    return out.astype(np.float32)


def _safe_percentile(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    try:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            return float(default)
        return float(np.nanpercentile(a, q))
    except Exception:
        return float(default)


def heuristic_contacts_fallback_np(motion: np.ndarray, cfg: MotionGenerationConfig, source_hint: str = "") -> Tuple[np.ndarray, dict]:
    """Kinematic fallback for contact channels when the main FK contact builder fails.

    Contact Reconstruction fix: never replace a time-varying foot contact signal with a static
    scalar such as 0.50 or 0.60.  First try a simple foot-height + foot-velocity
    heuristic from FK joints.  If even FK is unavailable, fall back to a root
    height/speed heuristic so the signal remains temporally varying rather than
    permanently locking or releasing both feet.
    """
    x = np.asarray(motion, dtype=np.float32)[:, :EDGE_DIM]
    T = int(x.shape[0])
    margin = float(getattr(cfg, "ik_height_margin", 0.05))
    speed_gate = float(getattr(cfg, "ik_speed_gate_mps", 0.36))
    report = {"source_hint": str(source_hint), "mode": "uninitialized"}
    contacts = np.zeros((T, 4), dtype=np.float32)
    if T <= 0:
        report["mode"] = "empty"
        return contacts, report

    try:
        joints = fk_24_np(x)
        foot_ids = list(DEFAULT_FOOT_JOINTS)
        foot = joints[:, foot_ids]
        foot_vxz = np.zeros(foot.shape[:2], dtype=np.float32)
        if T > 1:
            foot_vxz[1:] = (
                np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
                * float(cfg.fps)
            )
        floor_y = float(np.nanpercentile(foot[..., 1].reshape(-1), 5))
        near = foot[..., 1] <= floor_y + max(0.015, margin)
        slow = foot_vxz <= max(0.01, speed_gate)
        contacts = (near & slow).astype(np.float32)
        # Avoid all-zero output caused by overly strict speed thresholds on noisy
        # data.  Use near-floor alone as a second-stage fallback, still per-frame.
        if float(contacts.mean()) < 0.02:
            contacts = near.astype(np.float32)
            report["secondary_mode"] = "near_floor_without_speed_gate"
        report.update({
            "mode": "fk_height_velocity_heuristic",
            "floor_y": floor_y,
            "contact_ratio": float(contacts.mean()),
            "height_margin": float(margin),
            "speed_gate_mps": float(speed_gate),
        })
        return contacts.astype(np.float32), report
    except Exception as exc:
        report["fk_heuristic_error"] = str(exc)

    # Last-resort fallback when FK is unavailable.  Without foot joints, there is
    # no physically reliable way to decide left/right support.  Therefore Contact Reconstruction
    # deliberately avoids copying one root-level state to all four foot contacts:
    # that would weld both feet on near-root frames and release both feet otherwise.
    # Instead, produce a weak, non-anchoring, time-varying uncertainty signal that
    # stays below ik_contact_high, so IK will not impose a false strong foot lock.
    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    root_speed = np.zeros((T,), dtype=np.float32)
    if T > 1:
        root_speed[1:] = (
            np.linalg.norm(root[1:, [0, 2]] - root[:-1, [0, 2]], axis=-1)
            * float(cfg.fps)
        )
    root_floor = float(np.nanpercentile(root[:, 1], 20)) if T else 0.0
    near_root = root[:, 1] <= root_floor + max(0.02, margin)
    slow_root = root_speed <= max(0.02, speed_gate * 2.0)
    support_like = (near_root & slow_root).astype(np.float32)
    uncertain = min(0.50, max(0.42, float(getattr(cfg, "ik_contact_low", 0.38)) + 0.06))
    release = max(0.05, min(0.25, float(getattr(cfg, "ik_contact_low", 0.38)) - 0.10))
    base = release + (uncertain - release) * support_like
    contacts = np.repeat(base[:, None], 4, axis=1).astype(np.float32)
    report.update({
        "mode": "root_uncertain_nonlocking_no_fk",
        "root_floor_y": root_floor,
        "contact_ratio": float(contacts.mean()),
        "uncertain_contact_value": float(uncertain),
        "release_contact_value": float(release),
        "height_margin": float(margin),
        "speed_gate_mps": float(speed_gate),
        "warning": "FK unavailable; foot-specific contacts cannot be recovered, so fallback intentionally avoids strong IK anchoring.",
    })
    return contacts.astype(np.float32), report


def enforce_edge151_contract_np(
    motion: np.ndarray,
    cfg: Optional[MotionGenerationConfig] = None,
    source_hint: str = "",
    derive_contact: bool = True,
    project_rot: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Return a valid EDGE-151D motion tensor and an audit report.

    Official SMPL preprocessing and every downstream model must agree that
    EDGE channels ``[0:4]`` are contacts. The guard also projects rotations and
    reconstructs contacts after learned or geometric transformations.
    """
    cfg = cfg or MotionGenerationConfig()
    x0 = np.asarray(motion, dtype=np.float32)
    report = {
        "version": "edge151_contract_edge151_contract_guard",
        "source_hint": str(source_hint),
        "input_shape": list(x0.shape),
    }
    if x0.ndim != 2 or x0.shape[1] < EDGE_DIM:
        raise ValueError(
            f"EDGE151 contract violation: expected [T,151+], got {tuple(x0.shape)} from {source_hint}"
        )

    x = x0[:, :EDGE_DIM].astype(np.float32).copy()
    finite_before = bool(np.isfinite(x).all())
    report["finite_before"] = finite_before

    # Handle root/contact/other scalar channels conservatively, but never let
    # invalid rot6d become all-zero rotations.  Rot6D is sanitized separately.
    scalar_idx = list(range(0, ROT6D_START))
    x[:, scalar_idx] = np.nan_to_num(x[:, scalar_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rot_flat, rot_sanitize_report = sanitize_rot6d_np(x[:, ROT6D_START:ROT6D_END])
    x[:, ROT6D_START:ROT6D_END] = rot_flat.reshape(x.shape[0], -1)
    report["rot6d_sanitize"] = rot_sanitize_report

    contact_before = x[:, 0:4].copy()
    report["contact_before_min"] = float(np.min(contact_before)) if contact_before.size else 0.0
    report["contact_before_max"] = float(np.max(contact_before)) if contact_before.size else 0.0
    report["contact_before_abs_p95"] = _safe_percentile(np.abs(contact_before), 95)
    contact_polluted = bool(report["contact_before_abs_p95"] > 1.5 or report["contact_before_min"] < -0.05)
    report["contact_metadata_pollution_detected"] = contact_polluted

    if project_rot:
        report["rot6d_abs_p95_before_project"] = _safe_percentile(np.abs(x[:, ROT6D_START:ROT6D_END]), 95)
        x, project_report = project_edge151_rot6d_np(x)
        report["rot6d_project"] = project_report
        report["rot6d_projected"] = True
    else:
        report["rot6d_projected"] = False

    if derive_contact:
        try:
            contacts, conf, floor_y, _ = derive_contacts_np(x, cfg)
            x[:, 0:4] = contacts.astype(np.float32)
            report["contact_rebuilt_from_fk"] = True
            report["contact_ratio"] = float(contacts.mean())
            report["contact_conf_mean"] = float(np.mean(conf))
            report["floor_y"] = float(floor_y)
        except Exception as exc:
            # Contact Reconstruction: do not replace a dynamic contact signal with a global
            # constant such as 0.50 or 0.60.  Generate a per-frame kinematic
            # fallback so IK neither releases nor welds both feet for the whole clip.
            if contact_polluted:
                contacts_fb, fb_report = heuristic_contacts_fallback_np(
                    x, cfg, source_hint=f"contact_rebuild_failed:{source_hint}"
                )
                x[:, 0:4] = contacts_fb.astype(np.float32)
                report["contact_fallback_mode"] = "time_varying_kinematic_heuristic_due_to_metadata_pollution"
                report["contact_fallback_report"] = fb_report
            else:
                x[:, 0:4] = np.clip(np.nan_to_num(x[:, 0:4], nan=0.0), 0.0, 1.0)
                report["contact_fallback_mode"] = "clipped_existing_contact"
            report["contact_rebuilt_from_fk"] = False
            report["contact_rebuild_error"] = str(exc)
    else:
        if contact_polluted:
            contacts_fb, fb_report = heuristic_contacts_fallback_np(
                x, cfg, source_hint=f"derive_contact_false:{source_hint}"
            )
            x[:, 0:4] = contacts_fb.astype(np.float32)
            report["contact_fallback_mode"] = "derive_contact_false_time_varying_kinematic_heuristic_due_to_metadata_pollution"
            report["contact_fallback_report"] = fb_report
        else:
            x[:, 0:4] = np.clip(np.nan_to_num(x[:, 0:4], nan=0.0), 0.0, 1.0)
            report["contact_fallback_mode"] = "derive_contact_false_clipped_existing_contact"
        report["contact_rebuilt_from_fk"] = False

    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    report["root_min"] = [float(v) for v in np.min(root, axis=0)]
    report["root_max"] = [float(v) for v in np.max(root, axis=0)]
    report["root_y_range_m"] = float(np.max(x[:, ROOT_Y_IDX]) - np.min(x[:, ROOT_Y_IDX]))
    report["root_xz_travel_m"] = float(
        np.linalg.norm(x[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - x[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    )
    report["contact_after_abs_p95"] = _safe_percentile(np.abs(x[:, 0:4]), 95)
    report["rot6d_abs_p95_after"] = _safe_percentile(np.abs(x[:, ROT6D_START:ROT6D_END]), 95)
    return x.astype(np.float32), report


def sliding_window_ranges(T: int, window: int, hop: int) -> List[Tuple[int, int]]:
    """Return coverage-complete sliding windows for long-sequence inference."""
    T = int(T)
    window = max(1, int(window))
    hop = max(1, int(hop))
    if T <= window:
        return [(0, T)]
    starts = list(range(0, max(1, T - window + 1), hop))
    last = T - window
    if starts[-1] != last:
        starts.append(last)
    return [(int(s), int(min(T, s + window))) for s in starts]


def overlap_add_weight_np(length: int, start: int, total: int, hop: int, window: int) -> np.ndarray:
    """Raised-cosine weight with global-boundary one-sided protection.

    Contact Reconstruction fix:
    a full symmetric Hann window attenuates the very first and very last global
    frames even though no outside window can compensate them.  We therefore keep
    the non-overlapped side of the first/last chunk at weight 1.0 and only use
    cosine weights inside actual cross-window transition regions.
    """
    length = int(length)
    start = int(start)
    total = int(total)
    if length <= 0:
        return np.zeros((0, 1), dtype=np.float32)
    if length == 1 or total <= length:
        return np.ones((length, 1), dtype=np.float32)

    n = np.arange(length, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / float(max(length - 1, 1)))
    w = np.maximum(w, 1e-4).astype(np.float32)

    # The first global chunk has no previous chunk on its left side; do not
    # attenuate the leading half.  The last global chunk has no following chunk
    # on its right side; do not attenuate the trailing half.
    half = max(1, length // 2)
    if start <= 0:
        w[:half] = 1.0
    if start + length >= total:
        w[half:] = 1.0
    return w[:, None].astype(np.float32)

def normalize_quat_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    out = q / np.maximum(norm, 1e-8)
    bad = (~np.isfinite(out).all(axis=-1)) | (norm[..., 0] < 1e-8)
    if np.any(bad):
        out = out.copy()
        out[bad] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return out.astype(np.float32)


def matrix_to_quat_np(R: np.ndarray) -> np.ndarray:
    """Vectorized rotation-matrix to unit quaternion conversion [w,x,y,z].

    Contact Reconstruction fix: avoid per-matrix Python loops.  Long whole-song inference calls
    this function many times for [T,24,3,3] arrays, so the branch logic is
    implemented with NumPy masks to keep official evaluation practical.
    """
    arr = np.asarray(R, dtype=np.float32)
    prefix = arr.shape[:-2]
    m = arr.reshape(-1, 3, 3)
    q = np.zeros((m.shape[0], 4), dtype=np.float32)
    if m.shape[0] == 0:
        return q.reshape(prefix + (4,)).astype(np.float32)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    tr = m00 + m11 + m22

    mask = tr > 0.0
    if np.any(mask):
        s = np.sqrt(np.maximum(tr[mask] + 1.0, 1e-8)) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (m21[mask] - m12[mask]) / s
        q[mask, 2] = (m02[mask] - m20[mask]) / s
        q[mask, 3] = (m10[mask] - m01[mask]) / s

    rem = ~mask
    mask_x = rem & (m00 > m11) & (m00 > m22)
    if np.any(mask_x):
        s = np.sqrt(np.maximum(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x], 1e-8)) * 2.0
        q[mask_x, 0] = (m21[mask_x] - m12[mask_x]) / s
        q[mask_x, 1] = 0.25 * s
        q[mask_x, 2] = (m01[mask_x] + m10[mask_x]) / s
        q[mask_x, 3] = (m02[mask_x] + m20[mask_x]) / s

    mask_y = rem & (~mask_x) & (m11 > m22)
    if np.any(mask_y):
        s = np.sqrt(np.maximum(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y], 1e-8)) * 2.0
        q[mask_y, 0] = (m02[mask_y] - m20[mask_y]) / s
        q[mask_y, 1] = (m01[mask_y] + m10[mask_y]) / s
        q[mask_y, 2] = 0.25 * s
        q[mask_y, 3] = (m12[mask_y] + m21[mask_y]) / s

    mask_z = rem & (~mask_x) & (~mask_y)
    if np.any(mask_z):
        s = np.sqrt(np.maximum(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z], 1e-8)) * 2.0
        q[mask_z, 0] = (m10[mask_z] - m01[mask_z]) / s
        q[mask_z, 1] = (m02[mask_z] + m20[mask_z]) / s
        q[mask_z, 2] = (m12[mask_z] + m21[mask_z]) / s
        q[mask_z, 3] = 0.25 * s

    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return normalize_quat_np(q.reshape(prefix + (4,)))

def quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions [w,x,y,z] to rotation matrices."""
    q = normalize_quat_np(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R.astype(np.float32)


def init_motion_window_accumulators(T: int, D: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accum = np.zeros((int(T), int(D)), dtype=np.float32)
    weight_sum = np.zeros((int(T), 1), dtype=np.float32)
    rot_quat_accum = np.zeros((int(T), NUM_JOINTS, 4), dtype=np.float32)
    rot_quat_weight = np.zeros((int(T), 1, 1), dtype=np.float32)
    return accum, weight_sum, rot_quat_accum, rot_quat_weight


def accumulate_motion_window_np(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    rot_quat_accum: np.ndarray,
    rot_quat_weight: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    start: int,
    end: int,
) -> None:
    """Accumulate a generated chunk without linearly averaging Rot6D.

    Root/contact/scalar channels use Euclidean overlap-add.  Rotation channels
    are converted to quaternions and accumulated on S^3 with sign alignment
    before a final normalized quaternion-to-rot6d projection.  This avoids the
    near-zero Rot6D cancellation and snap risk caused by direct Rot6D averaging.
    """
    y = np.asarray(y, dtype=np.float32)[: int(end - start), :EDGE_DIM]
    w = np.asarray(w, dtype=np.float32).reshape(-1, 1)[: y.shape[0]]
    if y.shape[0] == 0:
        return
    y_linear = y.copy()
    y_linear[:, ROT6D_START:ROT6D_END] = 0.0
    accum[start:end] += y_linear * w
    weight_sum[start:end] += w

    R = rot6d_to_matrix_np(y[:, ROT6D_START:ROT6D_END].reshape(y.shape[0], NUM_JOINTS, 6))
    q = matrix_to_quat_np(R)
    for li, gi in enumerate(range(int(start), int(end))):
        wi = float(w[li, 0])
        if wi <= 0.0:
            continue
        qi = q[li]
        if float(rot_quat_weight[gi, 0, 0]) > 1e-8:
            ref = normalize_quat_np(rot_quat_accum[gi])
            dots = np.sum(qi * ref, axis=-1, keepdims=True)
            qi = np.where(dots < 0.0, -qi, qi)
        rot_quat_accum[gi] += qi * wi
        rot_quat_weight[gi, 0, 0] += wi


def finalize_motion_window_accum_np(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    rot_quat_accum: np.ndarray,
    rot_quat_weight: np.ndarray,
    cfg: MotionGenerationConfig,
    source_hint: str,
    derive_contact: bool = True,
) -> Tuple[np.ndarray, dict]:
    out = accum / np.maximum(weight_sum, 1e-8)
    valid = rot_quat_weight[:, 0, 0] > 1e-8
    q = np.zeros((accum.shape[0], NUM_JOINTS, 4), dtype=np.float32)
    q[:] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if np.any(valid):
        q[valid] = normalize_quat_np(rot_quat_accum[valid])
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(accum.shape[0], -1)
    out, report = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint=source_hint,
        derive_contact=derive_contact,
        project_rot=True,
    )
    report["rotation_overlap_mode"] = "quaternion_sign_aligned_weighted_average"
    report["scalar_overlap_mode"] = "hann_weighted_overlap_add"
    report["weight_sum_min"] = float(np.min(weight_sum)) if weight_sum.size else 0.0
    report["weight_sum_p05"] = float(np.percentile(weight_sum, 5)) if weight_sum.size else 0.0
    return out.astype(np.float32), report


def blend_motion_overlap_np(
    a: np.ndarray,
    b: np.ndarray,
    w_b: np.ndarray,
    cfg: MotionGenerationConfig,
    source_hint: str = "blend_motion_overlap",
) -> Tuple[np.ndarray, dict]:
    """Blend two overlap clips with quaternion rotation fusion, not Rot6D LERP.

    a and b must have the same temporal length. Scalar/root/contact channels
    use Euclidean weights; Rot6D channels are converted to quaternions with
    sign alignment and then mapped back to Rot6D. This is used for RAG event
    boundary blending, where adjacent retrieved clips can have large pose gaps.
    """
    a = np.asarray(a, dtype=np.float32)[:, :EDGE_DIM]
    b = np.asarray(b, dtype=np.float32)[:, :EDGE_DIM]
    L = int(min(len(a), len(b)))
    if L <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32), {"blend_mode": "empty"}
    a = a[:L]
    b = b[:L]
    wb = np.asarray(w_b, dtype=np.float32).reshape(-1, 1)[:L]
    wb = np.clip(wb, 0.0, 1.0)
    wa = 1.0 - wb
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(L, EDGE_DIM)
    accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, a, wa, 0, L)
    accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, b, wb, 0, L)
    out, report = finalize_motion_window_accum_np(
        accum, weight_sum, rot_quat_accum, rot_quat_weight, cfg, source_hint=source_hint
    )
    report["blend_mode"] = "scalar_linear_quaternion_rotation"
    report["w_b_min"] = float(np.min(wb)) if wb.size else 0.0
    report["w_b_max"] = float(np.max(wb)) if wb.size else 0.0
    return out.astype(np.float32), report










# Formal theme profiles contain context only.  In particular, names such as
# Pipa, Drum, Ribbon, or Sogdian Whirl do not inject motion energy, rotation,
# footwork, prop visibility, or percussive action into local SMPL windows.
CHANG_E_CATEGORY_PROFILES = {
    "flying_apsaras": {
        "aliases": {"flying", "apsaras", "flying_apsara", "flying_apsaras", "feitian", "fei_tian"},
        "display": "Flying Apsaras",
        "cultural_context": ["dunhuang_flying_apsaras_theme"],
    },
    "lotus_steps": {
        "aliases": {"lotus", "lotussteps", "lotus_step", "lotus_steps"},
        "display": "Lotus Steps",
        "cultural_context": ["lotus_steps_theme"],
    },
    "thirty_six_postures": {
        "aliases": {"36pose", "36posture", "36postures", "thirtysix", "thirty_six", "thirty_six_postures", "jiyuetian"},
        "display": "Ji Yue Tian Thirty-Six Postures",
        "cultural_context": ["thirty_six_postures_theme"],
    },
    "revelation_meditation": {
        "aliases": {"meditation", "mediation", "revelation", "revelation_meditation", "revelation_mediation"},
        "display": "Revelation Meditation",
        "cultural_context": ["revelation_meditation_theme"],
    },
    "sogdian_whirl": {
        "aliases": {"sogdian", "sogdian_whirl", "whirl"},
        "display": "Sogdian Whirl",
        "cultural_context": ["sogdian_whirl_theme"],
    },
    "pipa_behind_back": {
        "aliases": {"pipa", "pipa1", "pipa2", "playing_pipa", "playing_the_pipa", "pipa_behind_back"},
        "display": "Playing the Pipa Behind the Back",
        "cultural_context": ["pipa_source_context"],
    },
    "lei_gong_drum": {
        "aliases": {"drum", "lei_gong", "leigong", "lei_gong_drum"},
        "display": "Lei Gong Drum",
        "cultural_context": ["drum_source_context"],
    },
    "unknown": {
        "aliases": set(),
        "display": "Unknown Chang-E Theme",
        "cultural_context": [],
    },
}

ENERGY_LABELS = ["calm", "moderate", "high", "percussive"]
RHYTHM_LABELS = ["sustained", "lyrical", "accented", "percussive"]
BODY_FOCUS_LABELS = ["pose", "lower_body", "upper_body", "full_body", "turning_flow"]
SPATIAL_LABELS = ["in_place", "traveling", "turning", "aerial_leaning"]
MUSIC_ALIGNMENT_LABELS = ["unknown", "calm_meditative", "lyrical_flow", "pose_hold", "instrument_phrase", "percussive_accent", "turning_climax", "footwork_flow", "aerial_curve"]
EVENT_FAMILY_LABELS = ["pose_hold", "locomotion", "turn_spin", "jump_aerial", "floorwork", "upper_body_gesture", "rhythmic_accent", "transition", "unknown"]
STAGE_ROLE_LABELS = ["intro", "development", "build_up", "motif_recall", "anchor_or_resolution", "intro_or_resolution", "opening_or_climax", "accent_or_climax", "climax", "resolution"]
CATEGORY_CLASS_OVERRIDES = {}


def canonicalize_chang_e_key(key: object) -> str:
    key_s = str(key or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        key_s = re.sub(r"_take\d+$", "", key_s)
    except Exception:
        pass
    aliases = {"mediation": "revelation_meditation", "female_mediation": "revelation_meditation", "male_mediation": "revelation_meditation", "meditation": "revelation_meditation", "36pose": "thirty_six_postures", "36postures": "thirty_six_postures", "thirtysix": "thirty_six_postures", "lotus": "lotus_steps", "pipa": "pipa_behind_back", "drum": "lei_gong_drum", "leigong": "lei_gong_drum", "sogdian": "sogdian_whirl", "whirl": "sogdian_whirl", "flying": "flying_apsaras", "apsaras": "flying_apsaras", "feitian": "flying_apsaras"}
    if key_s in aliases:
        return aliases[key_s]
    for k, prof in CHANG_E_CATEGORY_PROFILES.items():
        if key_s == k or key_s in set(prof.get("aliases", set())):
            return k
    return key_s if key_s in CHANG_E_CATEGORY_PROFILES else "unknown"


def _safe_profile_key(meta: dict) -> str:
    return canonicalize_chang_e_key(
        meta.get("dance_category")
        or meta.get("dance_theme")
        or meta.get("dance_key")
        or "unknown"
    )


def _parse_numeric_semantic(meta: dict) -> Dict[str, float]:
    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]
    vals = [x for x in re.split(r"[;, ]+", str(meta.get("semantic_numeric", "") or "")) if x]
    out = {}
    for k, v in zip(keys, vals):
        try: out[k] = float(v)
        except Exception: pass
    return out


def strong_action_semantics_from_meta(meta: dict, desc: Optional[np.ndarray] = None) -> Dict[str, object]:
    if str(meta.get("source_format", "")) != "chang_e_official_smpl":
        raise RuntimeError(
            "Current local-action semantics accept only Chang-E official SMPL metadata"
        )
    key = _safe_profile_key(meta)
    theme = dict(
        CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"])
    )
    context = meta.get("source_context", [])
    if isinstance(context, str):
        context = [value for value in re.split(r"[;,|]", context) if value]
    out: Dict[str, object] = {
        "semantics_schema": "chang_e_five_layer_event_semantics_v2",
        "dance_theme": key,
        "theme_display": theme.get("display", key),
        "theme_label_status": meta.get("theme_label_status", "unknown"),
        "candidate_dance_category": meta.get("candidate_dance_category"),
        "cultural_context": list(theme.get("cultural_context", [])),
        "source_context": [str(value) for value in context],
        "source_context_is_local_action_truth": False,
        "prop_observation_available": False,
        "hand_capture_available": False,
        "preferred_dance_keys": [],
        "preferred_music_roles": [],
        "natural_duration_range_sec": [1.5, 4.0],
    }
    return refine_chang_e_event_semantics(meta, desc, out)


def _float_meta(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        v = meta.get(key, default)
        if v is None or str(v).lower() in {"nan", "none", "null", ""}:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _bounded01(x: float) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return 0.0


def chang_e_event_quality_from_numbers(nums: Dict[str, float], family: str, duration: float, natural_range: Sequence[float]) -> float:
    """Quality gate for converting long Chang-E recordings into RAG events."""
    energy = _bounded01(nums.get("energy", 0.0)); travel = _bounded01(nums.get("travel", 0.0))
    turn = _bounded01(nums.get("turn", 0.0)); lower = _bounded01(nums.get("lower", 0.0)); upper = _bounded01(nums.get("upper", 0.0))
    pose_hold = _bounded01(nums.get("pose_hold", 0.0)); jump = _bounded01(nums.get("jump", 0.0)); onset = _bounded01(nums.get("onset", 0.0))
    contact_ratio = _bounded01(nums.get("contact_ratio", 0.5))
    root_y = max(0.0, float(nums.get("root_y_range", 0.0)))
    lo, hi = 1.5, 4.0
    try:
        if natural_range and len(natural_range) >= 2:
            lo, hi = float(natural_range[0]), float(natural_range[-1])
    except Exception:
        pass
    dur = max(1e-3, float(duration or 0.0))
    center = max(1e-3, 0.5 * (lo + hi))
    dur_score = 1.0 if (lo <= dur <= hi) else float(np.exp(-abs(np.log(dur / center))))
    content = max(energy, travel, turn, lower, upper, onset, jump)
    if family in {"pose_motif", "calm_flow", "pose_hold", "floorwork"}:
        content = max(content * 0.65, pose_hold)
    # Event Semantics: stationary Dunhuang postures / meditation motifs are supposed to
    # have long stable support. Do not score contact_ratio=1.0 as bad gait.
    if family in {"pose_motif", "calm_flow", "pose_hold", "floorwork"} or pose_hold > 0.70:
        contact_score = 1.0 if contact_ratio >= 0.70 else float(contact_ratio / 0.70)
    else:
        contact_score = 1.0 - min(1.0, abs(contact_ratio - 0.46) / 0.54)
    root_y_penalty = max(0.0, min(0.25, (root_y - 0.35) * 0.35))
    dead_penalty = 0.0
    if family not in {"pose_motif", "calm_flow", "pose_hold", "floorwork"} and content < 0.20 and pose_hold < 0.45:
        dead_penalty = 0.25
    q = 0.42 * content + 0.22 * pose_hold + 0.20 * dur_score + 0.16 * contact_score - root_y_penalty - dead_penalty
    return float(np.clip(q, 0.02, 1.0))


def refine_chang_e_event_semantics(
    meta: dict,
    desc: Optional[np.ndarray],
    prof: Dict[str, object],
) -> Dict[str, object]:
    """Build five-layer semantics from provenance, theme context, and motion.

    The dance theme never contributes a numeric local-action score.  The local
    layer is multi-label and is computed only from the current motion window.
    Music compatibility is explicitly weak/probabilistic and is not copied from
    the selected local-action family.
    """

    out = dict(prof)
    scores = {label: 0.0 for label in EVENT_FAMILY_LABELS if label != "unknown"}
    nums = {
        "energy": 0.0,
        "onset": 0.0,
        "travel": 0.0,
        "turn": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "floorwork": 0.0,
        "jump": 0.0,
        "spin": 0.0,
        "pose_hold": 0.0,
        "instrument": 0.0,
        "prop": 0.0,
    }

    descriptor_available = desc is not None and len(desc) >= 31
    if descriptor_available:
        x = np.asarray(desc, dtype=np.float32)
        energy = _bounded01(float(x[5]) / 0.35)
        lower = _bounded01(float(x[7]) / 0.30)
        upper = _bounded01(float(x[8]) / 0.30)
        travel = max(
            _bounded01(float(x[1]) / 1.20),
            _bounded01(float(x[2]) / 0.45),
        )
        turn = max(
            _bounded01(abs(float(x[15])) / 1.40),
            _bounded01(abs(float(x[17])) / 1.20),
        )
        floorwork = _bounded01(float(x[25]))
        airborne = _bounded01(float(x[26]))
        burst = _bounded01(float(x[27]))
        upper_fraction = _bounded01(float(x[29]))
        transition = _bounded01(float(x[30]))
        contact_ratio = _bounded01(float(x[10]))
        root_y_range = max(0.0, float(x[18]))
        jump = max(
            _bounded01((root_y_range - 0.05) / 0.25),
            airborne,
        )
        pose_hold = _bounded01(1.0 - energy) * _bounded01(
            0.35 + 0.65 * contact_ratio
        )
        upper_gesture = _bounded01(upper * (0.55 + 0.90 * upper_fraction))
        rhythmic = _bounded01(0.65 * burst + 0.35 * _bounded01(float(x[6]) / 1.2))

        scores.update(
            {
                "pose_hold": pose_hold,
                "locomotion": travel,
                "turn_spin": turn,
                "jump_aerial": jump,
                "floorwork": floorwork,
                "upper_body_gesture": upper_gesture,
                "rhythmic_accent": rhythmic,
                "transition": transition,
            }
        )
        nums.update(
            {
                "energy": energy,
                "onset": rhythmic,
                "travel": travel,
                "turn": turn,
                "lower": lower,
                "upper": upper,
                "floorwork": floorwork,
                "jump": jump,
                "spin": turn,
                "pose_hold": pose_hold,
                "contact_ratio": contact_ratio,
                "root_y_range": root_y_range,
            }
        )

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    primary, primary_score = ordered[0] if ordered else ("unknown", 0.0)
    if not descriptor_available or primary_score < 0.40:
        primary = "unknown"
    local_labels = [name for name, score in ordered if score >= 0.50]
    if not local_labels:
        local_labels = [primary]

    frac_mid = _float_meta(
        meta,
        "event_position_mid",
        _float_meta(meta, "event_position_fraction", 0.5),
    )
    if frac_mid < 0.12:
        stage = "intro"
    elif frac_mid > 0.88:
        stage = "resolution"
    elif primary in {"turn_spin", "jump_aerial", "rhythmic_accent"}:
        stage = "climax"
    elif primary == "transition":
        stage = "development"
    else:
        stage = "development"

    compatibility_by_action = {
        "pose_hold": {"pose_hold": 0.45, "calm_meditative": 0.40, "lyrical_flow": 0.15},
        "locomotion": {"footwork_flow": 0.55, "lyrical_flow": 0.30, "percussive_accent": 0.15},
        "turn_spin": {"turning_climax": 0.60, "lyrical_flow": 0.25, "footwork_flow": 0.15},
        "jump_aerial": {"aerial_curve": 0.55, "lyrical_flow": 0.30, "turning_climax": 0.15},
        "floorwork": {"calm_meditative": 0.35, "lyrical_flow": 0.35, "pose_hold": 0.30},
        "upper_body_gesture": {"lyrical_flow": 0.50, "instrument_phrase": 0.20, "unknown": 0.30},
        "rhythmic_accent": {"percussive_accent": 0.60, "footwork_flow": 0.25, "lyrical_flow": 0.15},
        "transition": {"lyrical_flow": 0.50, "unknown": 0.50},
        "unknown": {"unknown": 1.0},
    }
    compatibility = dict(compatibility_by_action[primary])
    compatibility_top = max(
        compatibility.items(), key=lambda item: (item[1], item[0])
    )[0]

    duration = float(
        desc[0] if descriptor_available else _float_meta(meta, "duration", 0.0)
    )
    q = chang_e_event_quality_from_numbers(
        nums, primary, duration, [1.5, 4.0]
    )
    out.update(
        {
            "local_action_labels": local_labels,
            "local_action_scores": scores,
            "local_action_scores_json": json.dumps(scores, sort_keys=True),
            "local_action_descriptor_available": bool(descriptor_available),
            "event_family": primary,
            "motion_stage_role": stage,
            "music_alignment_label": "unknown",
            "music_alignment_tags": list(compatibility),
            "music_compatibility_top_label": compatibility_top,
            "music_compatibility_scores": compatibility,
            "music_compatibility_scores_json": json.dumps(
                compatibility, sort_keys=True
            ),
            "music_compatibility_supervision": "weak_kinematic_heuristic",
            "music_compatibility_is_ground_truth": False,
            "cultural_context_is_source_only": True,
            "prop_proxy_label": "not_observed_in_smpl",
            "event_position_mid": float(frac_mid),
            "event_quality_score": float(q),
            "semantic_confidence": float(
                np.clip(0.20 + 0.55 * q + 0.25 * primary_score, 0.10, 1.0)
            ),
        }
    )
    if primary == "turn_spin":
        out.update({"locomotion_label": "turning", "spatial_label": "turning"})
    elif primary == "locomotion":
        out.update({"locomotion_label": "traveling", "spatial_label": "traveling"})
    elif primary == "jump_aerial":
        out.update({"locomotion_label": "aerial", "spatial_label": "aerial"})
    elif primary == "floorwork":
        out.update({"locomotion_label": "floor_level", "spatial_label": "in_place"})
    elif primary == "pose_hold":
        out.update({"locomotion_label": "in_place_pose", "spatial_label": "in_place"})
    else:
        out.update({"locomotion_label": "unknown", "spatial_label": "unknown"})

    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]
    out["semantic_numeric"] = ";".join(
        str(float(nums.get(name, 0.0))) for name in keys
    )
    out["classification_text"] = (
        f"theme={out.get('dance_theme', 'unknown')}"
        f"[{out.get('theme_label_status', 'unknown')}]; "
        f"local_actions={','.join(local_labels)}; primary={primary}; "
        f"music_compatibility={compatibility_top}[weak]; "
        f"source_context={','.join(out.get('source_context', [])) or 'none'}"
    )
    return out


def official_smpl_semantics_from_metadata(meta: Mapping[str, Any]) -> Dict[str, object]:
    """Resolve formal source/theme context without filenames or BVH metadata."""

    if str(meta.get("source_format", "")).strip() != "chang_e_official_smpl":
        raise ValueError(
            "Official SMPL semantics require "
            "source_format=chang_e_official_smpl"
        )
    source_id = str(meta.get("source_id") or meta.get("source_uid") or "").strip()
    recording_uid = str(meta.get("recording_uid") or "").strip()
    sequence_id = str(meta.get("sequence_id") or "").strip()
    if not source_id or not recording_uid or not sequence_id:
        raise ValueError(
            "Official SMPL semantics require source_id, recording_uid, and sequence_id"
        )

    theme_status = str(meta.get("theme_label_status") or "").strip()
    dance_category = str(meta.get("dance_category") or "unknown").strip()
    if theme_status not in {"confirmed", "pending_official_confirmation"}:
        raise ValueError(f"Invalid official theme_label_status={theme_status!r}")
    if theme_status != "confirmed" and dance_category != "unknown":
        raise ValueError(
            "Unconfirmed official themes must remain dance_category=unknown"
        )

    performer_group = str(meta.get("performer_group") or "unknown").strip().lower()
    context = meta.get("source_context", [])
    if isinstance(context, str):
        context = [value for value in re.split(r"[;,|]", context) if value]
    context = [str(value) for value in context]
    display = str(
        CHANG_E_CATEGORY_PROFILES.get(
            dance_category, CHANG_E_CATEGORY_PROFILES["unknown"]
        ).get("display", dance_category)
    )
    theme_label = dance_category if theme_status == "confirmed" else "unknown_theme"
    return {
        "source_format": "chang_e_official_smpl",
        "source_uid": source_id,
        "source_group": source_id,
        "source_id": source_id,
        "recording_uid": recording_uid,
        "sequence_id": sequence_id,
        "dancer_id": meta.get("dancer_id"),
        "dancer_id_status": meta.get("dancer_id_status", "unverified"),
        "performer_track_id": meta.get("performer_track_id", -1),
        "recording_performer_count": int(meta.get("recording_performer_count", 1)),
        "solo_compatibility": meta.get("solo_compatibility", "unknown"),
        "solo_compatible": bool(meta.get("solo_compatible", False)),
        "solo_review_status": meta.get("solo_review_status", "unknown"),
        "sequence_index": meta.get("sequence_index", -1),
        "performer_group": performer_group,
        "gender": performer_group,
        "dance_key": dance_category,
        "dance_category": dance_category,
        "dance_theme": dance_category,
        "candidate_dance_category": meta.get("candidate_dance_category"),
        "theme_label_status": theme_status,
        "source_context": context,
        "take_id": meta.get("take_id"),
        "source_take": meta.get("take_id"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "coordinate_system": meta.get("coordinate_system"),
        "translation_units": meta.get("translation_units"),
        "pose_layout": meta.get("pose_layout"),
        "label": theme_label,
        "parent_label": theme_label,
        "semantic_role": "dance_theme_context",
        "semantic_text": (
            f"theme={display}; theme_status={theme_status}; "
            f"source_context={','.join(context) if context else 'none'}"
        ),
    }


def add_event_to_db_lists(
    clip: np.ndarray,
    event_idx: int,
    out_path: Path,
    cfg: MotionGenerationConfig,
    source: str,
    matched_audio: Optional[str],
    st: int,
    base_meta: dict,
    descs: List[np.ndarray], entries: List[np.ndarray], exits: List[np.ndarray], c0s: List[np.ndarray], c1s: List[np.ndarray],
    music_feats: List[np.ndarray], music_masks: List[float], meta: List[dict],
) -> None:
    """Write one canonical official-SMPL event into the shared Event-DB arrays.

    The former writer also accepted BVH/name-derived records and optional paired
    audio.  Those paths are deliberately rejected here: the formal SMPL14
    Event-DB is motion-only and receives all source/theme context from the
    authoritative manifest.
    """
    if str(base_meta.get("source_format", "")) != "chang_e_official_smpl":
        raise ValueError("Event-DB writer accepts only chang_e_official_smpl")
    if matched_audio is not None:
        raise ValueError("Formal SMPL14 Event-DB must not contain paired audio")

    clip, contract_report = enforce_edge151_contract_np(
        clip,
        cfg,
        source_hint=str(base_meta.get("source_file", out_path)),
        derive_contact=True,
        project_rot=True,
    )
    np.save(out_path, clip.astype(np.float32))
    desc = event_descriptor(clip, cfg.fps)
    entry, exit_, c0, c1 = motion_boundary_state(clip, fps=float(cfg.fps))
    music_feat = np.zeros(32, dtype=np.float32)
    music_mask = 0.0
    descs.append(desc)
    entries.append(entry)
    exits.append(exit_)
    c0s.append(c0)
    c1s.append(c1)
    music_feats.append(music_feat.astype(np.float32))
    music_masks.append(float(music_mask))
    item = {
        "event_id": event_idx,
        "path": str(out_path),
        "source_file": str(base_meta.get("source_file", base_meta.get("load_path", ""))),
        "source_format": "chang_e_official_smpl",
        "source_asset": str(base_meta.get("source_asset", base_meta.get("source_file", ""))),
        "source_group": source,
        "has_real_audio_feature": False,
        "seq_id": int(base_meta.get("seq_id", 0)),
        "start": int(st),
        "end": int(st + clip.shape[0]),
        "frames": int(clip.shape[0]),
        "duration": float(clip.shape[0] / max(float(cfg.fps), 1e-6)),
        "source_start_seconds": float(
            base_meta.get("source_start_seconds", st / max(float(cfg.fps), 1e-6))
        ),
        "source_end_seconds": float(
            base_meta.get(
                "source_end_seconds",
                (st + clip.shape[0]) / max(float(cfg.fps), 1e-6),
            )
        ),
        "canonical_fps": float(base_meta.get("canonical_fps", cfg.fps)),
        "label": str(base_meta.get("label") or "unknown_theme"),
        "parent_label": str(base_meta.get("parent_label", base_meta.get("label", "unknown"))),
        "fragment_index": int(base_meta.get("fragment_index", 0) or 0),
        "manifest_id": base_meta.get("manifest_id"),
        "manifest_path": base_meta.get("manifest_path"),
        "input_mode": base_meta.get("input_mode", "direct_files"),
        "edge151_contract_report": contract_report,
    }
    sem = official_smpl_semantics_from_metadata(base_meta)
    item.update(strong_action_semantics_from_meta({**sem, **item}, desc))
    for k in ["source_uid", "source_id", "recording_uid", "sequence_id", "dancer_id", "dancer_id_status", "performer_track_id", "recording_performer_count", "solo_compatibility", "solo_compatible", "solo_review_status", "sequence_index", "performer_group", "gender", "dance_key", "dance_category", "dance_theme", "candidate_dance_category", "theme_label_status", "source_context", "manifest_sha256", "coordinate_system", "translation_units", "pose_layout", "semantic_role", "semantic_text", "take_id", "source_take", "raw_stem"]:
        item[k] = base_meta.get(k, sem.get(k))
    strong_sem = strong_action_semantics_from_meta(item, desc)
    item.update(strong_sem)
    if item.get("semantic_text"):
        item["semantic_text"] = str(item["semantic_text"]) + "; " + str(strong_sem.get("classification_text", ""))
    item["label"] = str(sem["label"])
    item["parent_label"] = str(sem["parent_label"])
    if "resample_report" in base_meta:
        item["resample_report"] = base_meta["resample_report"]
    meta.append(item)

def audio_slots(
    path: str | Path,
    cfg: MotionGenerationConfig,
    slot_seconds: float = 4.0,
    slots_json: Optional[str] = None,
) -> Tuple[List[dict], np.ndarray]:
    """Load the final CTSR Scheduler hand-off without semantic fallback.

    ``path`` and ``slot_seconds`` remain in the shared generation interface, but
    slot construction belongs exclusively to the trained Librosa-12D Router and
    continuous Planner. Generation therefore requires their final descriptor.
    """
    del path, slot_seconds
    if not slots_json:
        raise RuntimeError("Formal generation requires --slots_json from CTSR Scheduler")
    descriptor_path = Path(slots_json)
    if not descriptor_path.is_file():
        raise FileNotFoundError(str(descriptor_path))
    from events.semantic_descriptor import parse_descriptor_file

    slots, features, _meta = parse_descriptor_file(
        descriptor_path,
        require_final_schedule=True,
        fps=float(cfg.fps),
        usage="generate_schedule",
    )
    return slots, features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _object_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _resolve_training_motion_path(raw_value: Any, db_path: Path) -> Path:
    """Resolve an Event-DB motion without depending on the process CWD."""
    raw = Path(str(raw_value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        PROJECT_ROOT / raw,
        db_path.parent / raw,
    ]
    checked: List[str] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve training motion {raw_value!r} from Event-DB {db_path}; "
        f"checked={checked}"
    )


def load_db(db_path: str | Path) -> dict:
    path = Path(db_path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as data:
        db = {k: data[k] for k in data.files}
    if "paths" in db:
        db["paths"] = np.asarray(
            [_resolve_training_motion_path(value, path) for value in db["paths"]],
            dtype=object,
        )
    db["_database_path"] = str(path)
    return db


def _training_db_contract(db: Dict[str, Any], cfg: MotionGenerationConfig, label: str) -> Dict[str, Any]:
    """Validate the immutable geometry/time/identity contract of a training DB."""
    paths = np.asarray(db.get("paths", []), dtype=object)
    desc = np.asarray(db.get("desc", []), dtype=np.float32)
    desc_z = np.asarray(db.get("desc_z", []), dtype=np.float32)
    count = int(len(paths))
    if count < 1:
        raise RuntimeError(f"{label} Event-DB is empty")
    if desc.shape != (count, 32) or desc_z.shape != (count, 32):
        raise RuntimeError(
            f"{label} descriptor contract mismatch: desc={desc.shape}, "
            f"desc_z={desc_z.shape}, expected=({count}, 32)"
        )

    fps_values = np.asarray(db.get("canonical_fps", []), dtype=np.float64).reshape(-1)
    if fps_values.size == 1:
        fps_values = np.full(count, float(fps_values[0]), dtype=np.float64)
    if fps_values.size != count or not np.all(np.isfinite(fps_values)) or np.any(fps_values <= 0.0):
        raise RuntimeError(f"{label} Event-DB has no valid per-event canonical_fps contract")
    unique_fps = np.unique(np.round(fps_values, decimals=6))
    if len(unique_fps) != 1:
        raise RuntimeError(f"{label} Event-DB mixes canonical FPS values: {unique_fps.tolist()}")
    database_fps = float(unique_fps[0])
    if abs(database_fps - float(cfg.fps)) > 1.0e-6:
        raise RuntimeError(
            f"{label} Event-DB FPS={database_fps:g} does not match runtime/config FPS={float(cfg.fps):g}"
        )

    raw_skeleton = _object_scalar(db.get("skeleton_contract_json"))
    if isinstance(raw_skeleton, bytes):
        raw_skeleton = raw_skeleton.decode("utf-8")
    if not isinstance(raw_skeleton, str):
        raise RuntimeError(f"{label} Event-DB has no SMPL24 skeleton contract")
    try:
        declared_skeleton = json.loads(raw_skeleton)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} Event-DB skeleton contract is invalid JSON") from exc
    expected_skeleton = skeleton_contract()
    for key in ("schema", "motion_dim", "rot6d_layout", "sha256"):
        if declared_skeleton.get(key) != expected_skeleton[key]:
            raise RuntimeError(
                f"{label} Event-DB skeleton {key} mismatch: "
                f"database={declared_skeleton.get(key)!r}, runtime={expected_skeleton[key]!r}"
            )

    event_uids = event_uids_from_generation_db(db)
    identity = make_event_db_contract(event_uids)
    declared_identity = normalize_event_db_contract(db.get("event_db_contract_json"))
    if declared_identity is None:
        raise RuntimeError(f"{label} Event-DB has no declared event identity contract")
    assert_same_event_db_contract(
        identity,
        declared_identity,
        context=f"{label} Event-DB identity",
    )
    return {
        "database_path": str(db.get("_database_path", "")),
        "canonical_fps": database_fps,
        "num_events": count,
        "event_db_contract": identity,
        "skeleton_schema": expected_skeleton["schema"],
        "skeleton_sha256": expected_skeleton["sha256"],
        "descriptor_dim": 32,
    }


def _validate_source_disjoint(
    train_db: Dict[str, Any],
    validation_db: Dict[str, Any],
) -> Dict[str, Any]:
    train_sources = {str(value) for value in np.asarray(train_db.get("source_uids", []), dtype=object)}
    validation_sources = {str(value) for value in np.asarray(validation_db.get("source_uids", []), dtype=object)}
    if not train_sources or not validation_sources:
        raise RuntimeError("Source-disjoint validation requires source_uids in both Event-DBs")
    overlap = sorted(train_sources & validation_sources)
    if overlap:
        raise RuntimeError(f"Train/validation source leakage detected: {overlap[:20]}")
    train_recordings = {
        str(value)
        for value in np.asarray(
            train_db.get("recording_uids", train_db.get("source_uids", [])),
            dtype=object,
        )
    }
    validation_recordings = {
        str(value)
        for value in np.asarray(
            validation_db.get(
                "recording_uids", validation_db.get("source_uids", [])
            ),
            dtype=object,
        )
    }
    recording_overlap = sorted(train_recordings & validation_recordings)
    if recording_overlap:
        raise RuntimeError(
            "Train/validation recording leakage detected: "
            f"{recording_overlap[:20]}"
        )
    return {
        "train_sources": len(train_sources),
        "validation_sources": len(validation_sources),
        "overlap": overlap,
        "train_recordings": len(train_recordings),
        "validation_recordings": len(validation_recordings),
        "recording_overlap": recording_overlap,
    }


def _descriptor_values_in_training_coordinates(
    db: Dict[str, Any],
    train_db: Dict[str, Any],
) -> np.ndarray:
    raw = np.asarray(db["desc"], dtype=np.float32)
    mean = np.asarray(train_db["desc_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(train_db["desc_std"], dtype=np.float32).reshape(1, -1)
    if raw.shape[1:] != mean.shape[1:] or mean.shape != std.shape:
        raise RuntimeError(
            f"Descriptor normalization mismatch: raw={raw.shape}, mean={mean.shape}, std={std.shape}"
        )
    return np.clip((raw - mean) / np.maximum(std, 1.0e-6), -8.0, 8.0).astype(np.float32)


def _expand_temporal_condition_torch(cond, frames: int):
    """Return condition as ``[batch, frames, features]``.

    Training may pass one descriptor per sequence, while whole-song generation
    passes a local descriptor for every frame.
    """

    if cond.ndim == 2:
        return cond[:, None, :].expand(cond.shape[0], int(frames), cond.shape[-1])
    if cond.ndim == 3 and cond.shape[1] == int(frames):
        return cond
    raise ValueError(
        f"condition must be [B,C] or [B,T,C] with T={frames}, got {tuple(cond.shape)}"
    )


def _condition_with_time_torch(cond, time_embedding, frames: int, projector):
    frame_cond = _expand_temporal_condition_torch(cond, frames)
    frame_time = time_embedding[:, None, :].expand(
        frame_cond.shape[0], int(frames), time_embedding.shape[-1]
    )
    return projector(torch.cat([frame_cond, frame_time], dim=-1)).transpose(1, 2)


def _condition_chunk_np(
    condition: np.ndarray,
    start: int,
    end: int,
    target_frames: Optional[int] = None,
) -> np.ndarray:
    """Slice frame-local conditioning while retaining training-time vectors."""

    value = np.asarray(condition, dtype=np.float32)
    if value.ndim == 1:
        return value
    if value.ndim != 2:
        raise ValueError(f"condition must be [C] or [T,C], got {value.shape}")
    chunk = value[int(start):int(end)]
    if chunk.shape[0] < 1:
        raise ValueError(f"empty condition slice [{start}:{end}] for {value.shape}")
    desired = int(target_frames if target_frames is not None else end - start)
    if chunk.shape[0] != desired:
        chunk = resample_motion_np(chunk, desired)
    return np.asarray(chunk, dtype=np.float32)


def build_frame_local_conditioning(
    slot_features: np.ndarray,
    concat_report: Sequence[Mapping[str, Any]],
    total_frames: int,
    descriptor_mean: np.ndarray,
    descriptor_std: np.ndarray,
) -> np.ndarray:
    """Expand slot descriptors to frames and interpolate every transition.

    Each slot owns exactly ``target_frames`` in the reference concatenation.
    A transition at the start of slot *i* interpolates from slot *i-1* to slot
    *i*, so neural repair observes the local musical change instead of one
    whole-song average descriptor.
    """

    features = np.asarray(slot_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(f"slot_features must be non-empty [S,C], got {features.shape}")
    if len(concat_report) != features.shape[0]:
        raise ValueError(
            "slot feature/report count mismatch: "
            f"{features.shape[0]} features vs {len(concat_report)} reports"
        )

    frame_parts: List[np.ndarray] = []
    transition_spans: List[Tuple[int, int, int]] = []
    for index, report in enumerate(concat_report):
        count = int(report.get("target_frames", report.get("slot_total_frames", 0)))
        if count <= 0:
            raise ValueError(f"slot {index} has invalid target_frames={count}")
        frame_parts.append(np.repeat(features[index:index + 1], count, axis=0))
        span = report.get("transition_span")
        if index > 0 and span is not None and len(span) >= 2:
            transition_spans.append((int(span[0]), int(span[1]), index))
    frame_condition = np.concatenate(frame_parts, axis=0).astype(np.float32)
    if frame_condition.shape[0] != int(total_frames):
        frame_condition = resample_motion_np(frame_condition, int(total_frames))

    for start, end, index in transition_spans:
        start = max(0, min(int(total_frames), start))
        end = max(start, min(int(total_frames), end))
        if end <= start:
            continue
        alpha = np.linspace(0.0, 1.0, end - start, dtype=np.float32)[:, None]
        frame_condition[start:end] = (
            (1.0 - alpha) * features[index - 1][None]
            + alpha * features[index][None]
        )

    mean = np.asarray(descriptor_mean, dtype=np.float32).reshape(-1)
    std = np.asarray(descriptor_std, dtype=np.float32).reshape(-1)
    if mean.shape[0] != features.shape[1] or std.shape[0] != features.shape[1]:
        raise ValueError(
            f"descriptor normalization mismatch: features={features.shape[1]}, "
            f"mean={mean.shape[0]}, std={std.shape[0]}"
        )
    return ((frame_condition - mean[None]) / np.maximum(std[None], 1.0e-8)).astype(np.float32)


class ProductManifoldTemporalRefiner(nn.Module):
    """Boundary refiner with a joint-risk-conditioned 79D geometric output.

    Output layout:
      - 4 contact logits;
      - 3 root-translation tangent residuals;
      - 24 x 3 local SO(3) tangent residuals.
    """

    def __init__(
        self,
        motion_dim: int = EDGE_DIM,
        cond_dim: int = 32,
        hidden: int = 256,
    ):
        super().__init__()
        # Kernel-5 dilations [1,2,5] give a 33-frame receptive field. A
        # 28-frame corrupted seam needs both external clean endpoints, not
        # merely 28 in-seam frames; the former 29-frame field missed one
        # endpoint at the centre of a maximum-length transition.
        self.temporal_dilations = (1, 2, 5)
        self.temporal_receptive_field_frames = 1 + 4 * sum(
            self.temporal_dilations
        )
        self.in_proj = nn.Conv1d(
            motion_dim + cond_dim + 1 + NUM_JOINTS, hidden, 1
        )
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2, dilation=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=10, dilation=5),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
        )
        self.out = nn.Conv1d(hidden, PRODUCT_STATE_DIM, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, cond, seam_mask, joint_mask):
        # x: B,T,151; cond: B,32 or B,T,32; seam: B,T,1.
        batch, frames, _ = x.shape
        c = _expand_temporal_condition_torch(cond, frames)
        y = torch.cat([x, c, seam_mask, joint_mask], dim=-1).transpose(1, 2)
        h = self.in_proj(y)
        h = h + self.net(h)
        return self.out(h).transpose(1, 2)


def _risk_masks_for_batch_np(
    motion_batch: np.ndarray,
    seam_batch: np.ndarray,
    cfg: MotionGenerationConfig,
    *,
    executor: Optional[ThreadPoolExecutor] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ordered frame x joint risk masks for a complete training batch."""
    motions = np.asarray(motion_batch, dtype=np.float32)
    seams = np.asarray(seam_batch, dtype=np.float32)
    if motions.ndim != 3 or motions.shape[-1] != EDGE_DIM:
        raise ValueError(
            f"motion_batch must have shape [B,T,{EDGE_DIM}], got "
            f"{motions.shape}"
        )
    if seams.ndim not in {2, 3} or seams.shape[:2] != motions.shape[:2]:
        raise ValueError(
            f"seam_batch must be event-aligned with {motions.shape[:2]}, "
            f"got {seams.shape}"
        )

    def build_one(pair: Tuple[np.ndarray, np.ndarray]):
        motion, seam = pair
        return build_frame_joint_risk_mask(
            motion,
            seam,
            fps=float(cfg.fps),
        )

    pairs = list(zip(motions, seams))
    masks_batch = (
        list(executor.map(build_one, pairs))
        if executor is not None and len(pairs) > 1
        else [build_one(pair) for pair in pairs]
    )
    joint_masks: List[np.ndarray] = []
    root_masks: List[np.ndarray] = []
    contact_masks: List[np.ndarray] = []
    for masks in masks_batch:
        joint_masks.append(np.asarray(masks["joint"], dtype=np.float32))
        root_masks.append(np.asarray(masks["root"], dtype=np.float32)[:, None])
        contact_masks.append(
            np.asarray(masks["contact"], dtype=np.float32)[:, None]
        )
    return (
        np.stack(joint_masks),
        np.stack(root_masks),
        np.stack(contact_masks),
    )


_RISK_ANCESTORS_TORCH_CACHE: Dict[Tuple[str, int], Any] = {}


def _risk_mask_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _risk_mask_env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _risk_ancestor_matrix_torch(device: Any, parent_depth: int):
    key = (str(device), int(parent_depth))
    cached = _RISK_ANCESTORS_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    ancestors = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.float32)
    for source_joint in range(NUM_JOINTS):
        chain_joint = source_joint
        for _ in range(max(0, int(parent_depth)) + 1):
            if chain_joint < 0 or chain_joint >= NUM_JOINTS:
                break
            ancestors[source_joint, chain_joint] = 1.0
            chain_joint = int(PARENTS[chain_joint])
    cached = torch.as_tensor(ancestors, device=device)
    _RISK_ANCESTORS_TORCH_CACHE[key] = cached
    return cached


def _risk_masks_for_batch_torch(
    motion_batch: Any,
    seam_batch: Any,
    cfg: MotionGenerationConfig,
) -> Tuple[Any, Any, Any]:
    """Build the formal repair masks as one device-resident batch.

    This mirrors ``build_frame_joint_risk_mask``: seam dilation, robust
    angular risk, world-space Peak-Jerk, parent-chain expansion, body-part
    propagation, root acceleration and contact changes.  Keeping the result on
    the model device removes one CPU/SciPy pass and three host-to-device copies
    per sample.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for batched risk masks")
    motion = motion_batch
    seam = seam_batch
    if motion.ndim != 3 or motion.shape[-1] != EDGE_DIM:
        raise ValueError(
            f"motion_batch must have shape [B,T,{EDGE_DIM}], got "
            f"{tuple(motion.shape)}"
        )
    if seam.ndim == 2:
        seam = seam.unsqueeze(-1)
    if seam.ndim != 3 or seam.shape[:2] != motion.shape[:2]:
        raise ValueError(
            f"seam_batch must be event-aligned with {tuple(motion.shape[:2])}, "
            f"got {tuple(seam.shape)}"
        )
    if motion.device != seam.device:
        raise ValueError("motion_batch and seam_batch must share one device")
    if not motion.is_floating_point() or not seam.is_floating_point():
        raise ValueError("motion_batch and seam_batch must be floating tensors")

    batch, frames = int(motion.shape[0]), int(motion.shape[1])
    dtype, device = motion.dtype, motion.device
    seam_active = seam.amax(dim=-1) > 0.01
    dilation = max(
        0,
        int(
            round(
                _risk_mask_env_int("GROUNDING_MASK_DILATE", 3)
                * float(cfg.fps)
                / 30.0
            )
        ),
    )
    if dilation > 0:
        seam_active = F.max_pool1d(
            seam_active.to(dtype).unsqueeze(1),
            kernel_size=2 * dilation + 1,
            stride=1,
            padding=dilation,
        ).squeeze(1) > 0.0

    rotations = rot6d_to_matrix_torch(
        motion[..., ROT6D_START:ROT6D_END].reshape(
            batch, frames, NUM_JOINTS, 6
        )
    )
    angular_speed = torch.zeros(
        (batch, frames, NUM_JOINTS),
        dtype=dtype,
        device=device,
    )
    angular_acceleration = torch.zeros_like(angular_speed)
    if frames > 1:
        relative = rotations[:, :-1].transpose(-1, -2) @ rotations[:, 1:]
        omega = so3_log_torch(relative) * float(cfg.fps)
        angular_speed[:, 1:] = torch.linalg.vector_norm(omega, dim=-1)
        if frames > 2:
            alpha = torch.diff(omega, dim=1) * float(cfg.fps)
            angular_acceleration[:, 2:] = torch.linalg.vector_norm(
                alpha,
                dim=-1,
            )

    def robust_z(values: Any):
        median = torch.quantile(
            values,
            0.5,
            dim=1,
            keepdim=True,
            interpolation="linear",
        )
        mad = torch.quantile(
            (values - median).abs(),
            0.5,
            dim=1,
            keepdim=True,
            interpolation="linear",
        ) + 1.0e-5
        return torch.clamp_min(
            (values - median) / (1.4826 * mad),
            0.0,
        )

    risk = (
        0.55 * torch.clamp(robust_z(angular_speed) / 5.0, 0.0, 1.0)
        + 0.45
        * torch.clamp(
            robust_z(angular_acceleration) / 5.0,
            0.0,
            1.0,
        )
    )
    risk = risk * seam_active[..., None].to(dtype)
    minimum = _risk_mask_env_float("GROUNDING_MASK_MIN_ON_SEAM", 0.18)
    risk = torch.where(
        seam_active[..., None],
        torch.clamp_min(risk, minimum),
        risk,
    )

    peak_cfg = PeakJerkMaskConfig.from_environment()
    peak_joint = torch.zeros_like(risk)
    if peak_cfg.enabled and frames >= 4:
        # The authoritative NumPy Peak-Jerk contract promotes FK positions to
        # float64 before taking a third difference.  Match that precision here
        # so samples close to the formal threshold cannot change support merely
        # because GPU preprocessing is enabled.
        joints = fk_24_torch(motion).to(dtype=torch.float64)
        jerk = torch.diff(joints, n=3, dim=1) * float(cfg.fps) ** 3
        jerk_norm = torch.linalg.vector_norm(jerk, dim=-1)
        percentile = float(np.clip(peak_cfg.percentile, 0.0, 100.0))
        adaptive = torch.quantile(
            jerk_norm.reshape(batch, -1),
            percentile / 100.0,
            dim=1,
            interpolation="linear",
        )
        threshold = torch.maximum(
            adaptive,
            torch.full_like(adaptive, peak_cfg.absolute_threshold_mps3),
        )
        risky = jerk_norm >= threshold[:, None, None]
        radius = max(
            0,
            int(
                round(
                    int(peak_cfg.radius_frames_at_30fps)
                    * float(cfg.fps)
                    / 30.0
                )
            ),
        )
        # Derivative sample d owns motion frames [d-r, d+3+r].
        expanded = F.max_pool1d(
            F.pad(
                risky.permute(0, 2, 1).to(dtype),
                (3 + radius, 3 + radius),
            ),
            kernel_size=4 + 2 * radius,
            stride=1,
        ).permute(0, 2, 1) > 0.0
        ancestors = _risk_ancestor_matrix_torch(
            device,
            int(peak_cfg.parent_depth),
        ).to(dtype=dtype)
        peak_joint = (
            expanded.to(dtype) @ ancestors > 0.0
        ).to(dtype)

    risk = torch.maximum(risk, peak_joint)
    for ids in BODY_PARTS.values():
        indices = list(map(int, ids))
        part = risk[..., indices].amax(dim=-1, keepdim=True)
        risk[..., indices] = torch.maximum(
            risk[..., indices],
            part
            * _risk_mask_env_float("GROUNDING_PART_PROPAGATION", 0.65),
        )

    root_velocity = torch.zeros(
        (batch, frames, 3), dtype=dtype, device=device
    )
    root_acceleration = torch.zeros_like(root_velocity)
    root = motion[..., [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    if frames > 1:
        root_velocity[:, 1:] = torch.diff(root, dim=1) * float(cfg.fps)
    if frames > 2:
        root_acceleration[:, 2:] = (
            torch.diff(root_velocity[:, 1:], dim=1) * float(cfg.fps)
        )
    root_score = torch.clamp(
        torch.linalg.vector_norm(root_acceleration, dim=-1)
        / _risk_mask_env_float("GROUNDING_ROOT_ACC_MASK_SCALE", 8.0),
        0.0,
        1.0,
    ) * seam_active.to(dtype)
    root_score = torch.maximum(
        root_score,
        seam_active.to(dtype)
        * _risk_mask_env_float("GROUNDING_ROOT_MASK_MIN_ON_SEAM", 0.18),
    )
    root_score = torch.maximum(root_score, peak_joint[..., 0])

    contact_score = torch.zeros(
        (batch, frames), dtype=dtype, device=device
    )
    if frames > 1:
        contact_score[:, 1:] = (
            torch.diff(motion[..., :4].clamp(0.0, 1.0), dim=1)
            .abs()
            .amax(dim=-1)
            * float(cfg.fps)
            / 30.0
        )
    contact_score = torch.maximum(
        contact_score * seam_active.to(dtype),
        seam_active.to(dtype) * 0.25,
    )
    contact_score = torch.maximum(
        contact_score,
        peak_joint[..., list(DEFAULT_FOOT_JOINTS)].amax(dim=-1),
    )

    return (
        risk.clamp(0.0, 1.0),
        root_score.clamp(0.0, 1.0).unsqueeze(-1),
        contact_score.clamp(0.0, 1.0).unsqueeze(-1),
    )


def _gpu_preprocessing_enabled(
    cfg: MotionGenerationConfig,
    device: Any,
) -> bool:
    resolved = torch.device(device) if torch is not None else None
    return bool(
        torch is not None
        and bool(cfg.gpu_preprocessing)
        and resolved is not None
        and resolved.type == "cuda"
        and torch.cuda.is_available()
    )


def _diffusion_noise_window_cap(
    cfg: MotionGenerationConfig,
    *,
    steps: int,
    window: int,
    dtype: Any,
) -> int:
    """Maximum noise windows allowed in one inference allocation."""
    if int(steps) < 1 or int(window) < 1:
        raise ValueError(
            "diffusion noise requires positive steps and window length, got "
            f"steps={steps!r}, window={window!r}"
        )
    limit_mib = float(cfg.diffusion_noise_batch_max_mib)
    if not np.isfinite(limit_mib) or limit_mib <= 0.0:
        raise ValueError(
            "diffusion_noise_batch_max_mib must be a positive finite value, "
            f"got {limit_mib!r}"
        )
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_window = (
        int(steps)
        * int(window)
        * int(PRODUCT_STATE_DIM)
        * int(element_size)
    )
    limit_bytes = int(limit_mib * (1024 ** 2))
    if limit_bytes < bytes_per_window:
        minimum_mib = bytes_per_window / float(1024 ** 2)
        raise ValueError(
            "diffusion_noise_batch_max_mib is smaller than one diffusion "
            f"window ({limit_mib:.3f} MiB configured, "
            f"{minimum_mib:.3f} MiB required)"
        )
    return max(1, limit_bytes // bytes_per_window)


def _diffusion_noise_batch_torch(
    window_indices: Sequence[int],
    *,
    base_seed: int,
    steps: int,
    window: int,
    dtype: Any,
    device: Any,
):
    """Create bounded per-window noise independent of batch partitioning.

    A separate deterministic generator per global window avoids both the old
    full-song allocation and batch-size-dependent random streams.  Each window
    still uses one device allocation instead of one allocation per diffusion
    step.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for diffusion noise")
    indices = [int(value) for value in window_indices]
    if not indices:
        raise ValueError("window_indices must not be empty")
    resolved = torch.device(device)
    modulus = (1 << 63) - 1
    noise = torch.empty(
        (len(indices), int(steps), int(window), int(PRODUCT_STATE_DIM)),
        dtype=dtype,
        device=resolved,
    )
    for row_index, global_index in enumerate(indices):
        if global_index < 0:
            raise ValueError("window indices must be non-negative")
        generator = torch.Generator(device=resolved)
        window_seed = (
            int(base_seed) + 0x9E3779B1 * int(global_index)
        ) % modulus
        generator.manual_seed(window_seed)
        noise[row_index].normal_(generator=generator)
    return noise


_SYMMETRIC_PAD_INDEX_TORCH_CACHE: Dict[Tuple[str, int, int], Any] = {}


def _symmetric_pad_indices_torch(
    frames: int,
    radius: int,
    device: Any,
):
    key = (str(device), int(frames), int(radius))
    cached = _SYMMETRIC_PAD_INDEX_TORCH_CACHE.get(key)
    if cached is not None:
        return cached
    positions = torch.arange(
        -int(radius),
        int(frames) + int(radius),
        device=device,
        dtype=torch.long,
    )
    period = max(2, 2 * int(frames))
    folded = torch.remainder(positions, period)
    indices = torch.where(
        folded < int(frames),
        folded,
        2 * int(frames) - 1 - folded,
    )
    _SYMMETRIC_PAD_INDEX_TORCH_CACHE[key] = indices
    return indices


def _median_bool_filter_torch(values: Any, size: int):
    window = max(1, int(size))
    if window <= 1 or values.shape[1] <= 1:
        return values.to(dtype=torch.bool)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    indices = _symmetric_pad_indices_torch(
        int(values.shape[1]),
        radius,
        values.device,
    )
    padded = values.index_select(1, indices).to(dtype=torch.int16)
    votes = padded.unfold(1, window, 1).sum(dim=-1)
    return votes >= (radius + 1)


def _enforce_internal_batch_contract_torch(
    motion_batch: Any,
    cfg: MotionGenerationConfig,
    *,
    derive_contact: bool = True,
):
    """GPU contract finalization for validated internal motion batches.

    The general NumPy guard retains fallback reports for arbitrary external
    input.  Training corruption is narrower: its source is already validated,
    only finite root/Rot6D noise is added, and the batch needs rotation
    projection plus FK contact reconstruction.  This specialized path mirrors
    those operations without moving every sample back through CPU/SciPy.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for GPU training finalization")
    if motion_batch.ndim != 3 or motion_batch.shape[-1] != EDGE_DIM:
        raise ValueError(
            f"training motion must be [B,T,{EDGE_DIM}], got "
            f"{tuple(motion_batch.shape)}"
        )
    motion = motion_batch.clone()
    motion[..., :ROT6D_START] = torch.nan_to_num(
        motion[..., :ROT6D_START],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    batch, frames = int(motion.shape[0]), int(motion.shape[1])
    rotations6d = motion[..., ROT6D_START:ROT6D_END].reshape(
        batch,
        frames,
        NUM_JOINTS,
        6,
    )
    finite_rotation = torch.isfinite(rotations6d).all(dim=-1)
    rotations6d = torch.nan_to_num(
        rotations6d,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    a1, a2 = rotations6d[..., :3], rotations6d[..., 3:]
    n1 = torch.linalg.vector_norm(a1, dim=-1)
    n2 = torch.linalg.vector_norm(a2, dim=-1)
    cross_norm = torch.linalg.vector_norm(torch.cross(a1, a2, dim=-1), dim=-1)
    invalid = (
        ~finite_rotation
        | (n1 < 1.0e-5)
        | (n2 < 1.0e-5)
        | (cross_norm < 1.0e-5)
    )
    identity = torch.as_tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=motion.dtype,
        device=motion.device,
    )
    rotations6d = torch.where(
        invalid[..., None],
        identity,
        rotations6d,
    )
    projected = matrix_to_rot6d_torch(
        rot6d_to_matrix_torch(rotations6d)
    )
    motion[..., ROT6D_START:ROT6D_END] = projected.reshape(
        batch,
        frames,
        -1,
    )
    if not derive_contact:
        motion[..., :4] = motion[..., :4].clamp(0.0, 1.0)
        return motion

    joints = fk_24_torch(motion)
    feet = joints[..., list(DEFAULT_FOOT_JOINTS), :]
    foot_y = feet[..., 1]
    floor_y = torch.quantile(
        foot_y.reshape(batch, -1),
        0.05,
        dim=1,
        interpolation="linear",
    )
    velocity = torch.zeros(
        feet.shape[:-1],
        dtype=motion.dtype,
        device=motion.device,
    )
    if frames > 1:
        velocity[:, 1:] = (
            torch.linalg.vector_norm(
                feet[:, 1:, :, (0, 2)] - feet[:, :-1, :, (0, 2)],
                dim=-1,
            )
            * float(cfg.fps)
        )
    height_score = torch.clamp(
        1.0
        - (foot_y - floor_y[:, None, None])
        / max(float(cfg.ik_height_margin), 1.0e-6),
        0.0,
        1.0,
    )
    speed_score = torch.clamp(
        1.0 - velocity / max(float(cfg.ik_speed_gate_mps), 1.0e-6),
        0.0,
        1.0,
    )
    confidence = 0.62 * height_score + 0.38 * speed_score
    event = torch.where(
        confidence >= float(cfg.ik_contact_high),
        torch.ones_like(confidence, dtype=torch.int8),
        torch.where(
            confidence <= float(cfg.ik_contact_low),
            -torch.ones_like(confidence, dtype=torch.int8),
            torch.zeros_like(confidence, dtype=torch.int8),
        ),
    )
    frame_index = torch.arange(
        frames,
        device=motion.device,
        dtype=torch.long,
    )[None, :, None]
    last_event_index = torch.where(
        event != 0,
        frame_index,
        torch.full_like(frame_index, -1),
    ).cummax(dim=1).values
    gathered_event = torch.gather(
        event,
        1,
        last_event_index.clamp_min(0).expand_as(event),
    )
    contact = (last_event_index >= 0) & (gathered_event > 0)
    median_frames = max(1, int(round(float(cfg.fps) / 6.0)))
    if median_frames % 2 == 0:
        median_frames += 1
    contact = _median_bool_filter_torch(contact, median_frames)
    contact &= velocity <= float(
        getattr(cfg, "ik_contact_break_speed_mps", 0.54)
    )
    motion[..., :4] = contact.to(dtype=motion.dtype)
    return motion


def _smooth_supported_residual_torch(value, support, cfg):
    """Smooth corrections (never source motion), with no support expansion.

    A hard 0 -> risk-mask jump used to turn even a constant network output
    into an impulsive FK jerk. Normalized convolution and an inward quintic
    taper remove that discontinuity without touching any protected frame.
    """
    passes = int(cfg.product_refiner_residual_smoothing_passes)
    radius = int(cfg.product_refiner_residual_taper_frames)
    if not 0 <= passes <= 8 or not 0 <= radius <= 16:
        raise ValueError("refiner residual smoothing/taper is out of bounds")
    active = torch.broadcast_to((support > 0).to(value.dtype), value.shape)
    weights = active.transpose(1, 2)
    filtered = (value * active).transpose(1, 2)

    def smooth(x):
        pad = F.pad(x, (1, 1), mode="replicate")
        return 0.25 * pad[..., :-2] + 0.50 * pad[..., 1:-1] + 0.25 * pad[..., 2:]

    for _ in range(passes):
        filtered, weights = smooth(filtered), smooth(weights)
    filtered = filtered / weights.clamp_min(1e-6)
    eroded = active.transpose(1, 2)
    distance = eroded.clone()
    for _ in range(radius):
        eroded = -F.max_pool1d(
            -F.pad(eroded, (1, 1), mode="replicate"), 3, stride=1
        )
        distance = distance + eroded
    phase = distance / float(radius + 1)
    taper = phase.pow(3) * (10.0 - 15.0 * phase + 6.0 * phase.square())
    return filtered.transpose(1, 2) * taper.transpose(1, 2) * active


def _decode_product_refiner_output(
    reference,
    output,
    joint_mask,
    root_mask,
    contact_mask,
    cfg: MotionGenerationConfig,
):
    if output.shape[-1] != PRODUCT_STATE_DIM:
        raise ValueError(
            f"product refiner output must be {PRODUCT_STATE_DIM}D, "
            f"got {output.shape[-1]}"
        )
    # Risk masks are soft confidences. Scale the proposed tangent first; the
    # safety cap must constrain the applied tangent after that scaling. The
    # former cap-then-scale order saturated raw outputs at low mask values
    # (0.18 on a seam), obstructing supervised repair and radial gradients.
    root_weight = root_mask.clamp(0.0, 1.0)
    if root_weight.ndim == output.ndim - 1:
        root_weight = root_weight.unsqueeze(-1)
    joint_weight = joint_mask.clamp(0.0, 1.0)
    raw_tangent = output[..., 4:]
    scaled_tangent = torch.cat(
        [
            raw_tangent[..., :3] * root_weight,
            (
                raw_tangent[..., 3:].reshape(
                    raw_tangent.shape[:-1] + (NUM_JOINTS, 3)
                )
                * joint_weight[..., None]
            ).reshape(
                raw_tangent.shape[:-1] + (NUM_JOINTS * 3,)
            ),
        ],
        dim=-1,
    )
    tangent_support = torch.cat(
        [root_weight.expand_as(scaled_tangent[..., :3]),
         joint_weight[..., None].expand(joint_weight.shape + (3,)).reshape(
             scaled_tangent.shape[:-1] + (NUM_JOINTS * 3,))],
        dim=-1,
    )
    scaled_tangent = _smooth_supported_residual_torch(
        scaled_tangent, tangent_support, cfg
    )
    geometry = masked_retract_torch(
        reference,
        scaled_tangent,
        joint_mask=(joint_weight > 0.0).to(reference.dtype),
        root_mask=(root_weight > 0.0).to(reference.dtype),
        max_rotation_rad=float(cfg.product_refiner_rotation_cap_rad),
        max_root_m=float(cfg.product_refiner_root_cap_m),
    )
    contact_weight = contact_mask.clamp(0.0, 1.0)
    observed_contact = reference[..., :4].clamp(0.0, 1.0)
    contact_logits = torch.logit(observed_contact.clamp(0.01, 0.99))
    # Zero network output must be a true identity, including observed contact.
    # Absolute sigmoid logits used to replace clean 0/1 observations with 0.5.
    contact_delta = torch.sigmoid(contact_logits + output[..., :4]) - torch.sigmoid(contact_logits)
    contacts = (observed_contact + contact_weight * contact_delta).clamp(0.0, 1.0)
    return torch.cat([contacts, geometry[..., 4:]], dim=-1)


def _derivative_support_torch(
    seam_mask,
    order: int,
    *,
    boundary_only: bool = False,
):
    """Return frame support for a temporal derivative touching the seam.

    ``torch.diff(..., n=order)`` consumes ``order + 1`` source frames.  The
    support therefore has the derivative's shorter time axis.  Boundary-only
    support selects stencils that cross between clean context and the corrupt
    core; non-boundary support selects every stencil that touches the core.
    """

    if order < 1:
        raise ValueError("temporal derivative order must be positive")
    if seam_mask.ndim == 3:
        core = seam_mask[..., 0]
    elif seam_mask.ndim == 2:
        core = seam_mask
    else:
        raise ValueError(
            "seam_mask must be [B,T] or [B,T,1], got "
            f"{tuple(seam_mask.shape)}"
        )
    core = (core >= 0.5).to(dtype=seam_mask.dtype)
    width = int(core.shape[1]) - int(order)
    if width <= 0:
        return core[:, :0]
    stencil = torch.stack(
        [core[:, offset:offset + width] for offset in range(order + 1)],
        dim=-1,
    )
    maximum = stencil.amax(dim=-1)
    if boundary_only:
        return maximum - stencil.amin(dim=-1)
    return maximum


def _masked_temporal_fk_loss_torch(
    predicted_joints,
    clean_joints,
    seam_mask,
    *,
    order: int,
    scale: float,
    boundary_only: bool = False,
):
    if predicted_joints.shape[1] <= order:
        return predicted_joints.new_zeros(())
    support = _derivative_support_torch(
        seam_mask,
        order,
        boundary_only=boundary_only,
    )
    predicted_delta = torch.diff(
        predicted_joints, n=order, dim=1
    ) / float(scale)
    clean_delta = torch.diff(clean_joints, n=order, dim=1) / float(scale)
    error = F.smooth_l1_loss(
        predicted_delta,
        clean_delta,
        reduction="none",
    ).mean(dim=(-1, -2))
    return (error * support).sum() / support.sum().clamp_min(1.0)


def _relative_repair_margin_torch(error, baseline, minimum_gain):
    """Dimensionless per-window objective matching the relative audit gate."""
    baseline = baseline.detach()
    denominator = baseline.clamp_min(1e-6)
    target = (1.0 - float(minimum_gain)) * baseline
    return (torch.relu(error - target) / denominator).mean()


def _seam_fk_errors_torch(candidate_joints, clean_joints, seam_mask, cfg):
    """Differentiable counterpart of _seam_fk_temporal_errors_np, per window."""
    result = {}
    for name, order, scale, boundary in (
        ("endpoint_velocity_error", 1, 1.0 / cfg.fps, True),
        ("seam_velocity_error", 1, 1.0 / cfg.fps, False),
        ("seam_acceleration_error", 2, 10.0 / cfg.fps**2, False),
        ("seam_jerk_error", 3, 1000.0 / cfg.fps**3, False),
    ):
        if candidate_joints.shape[1] <= order:
            result[name] = candidate_joints.new_zeros(candidate_joints.shape[0])
            continue
        support = _derivative_support_torch(seam_mask, order, boundary_only=boundary)
        error = (
            torch.diff(candidate_joints, n=order, dim=1)
            - torch.diff(clean_joints, n=order, dim=1)
        ).abs().mean(dim=(-1, -2)) / float(scale)
        result[name] = (error * support).sum(dim=1) / support.sum(dim=1).clamp_min(1.0)
    return result


def _world_space_physics_losses(
    prediction,
    clean,
    cfg: MotionGenerationConfig,
    *,
    seam_mask=None,
    reference=None,
):
    """Differentiable FK/foot/dynamics losses for EDGE-151D batches.

    Static support is defined from the clean target's contact and world-space
    speed. This avoids forcing a legitimate moving/cloud-step support to zero
    velocity while still supervising planted feet and penetration explicitly.
    """
    if prediction.ndim != 3 or clean.shape != prediction.shape:
        raise ValueError(
            "physics loss expects matching [B,T,151] tensors, got "
            f"{tuple(prediction.shape)} and {tuple(clean.shape)}"
        )
    # One batched FK halves the Python joint-loop and GPU kernel-launch count.
    # Batch items remain independent, so this is numerically equivalent to two
    # separate FK calls while keeping gradients for ``prediction`` intact.
    batch_size = prediction.shape[0]
    motions = [prediction, clean]
    if reference is not None:
        motions.append(reference.detach())
    combined_joints = fk_24_torch(torch.cat(motions, dim=0))
    predicted_joints = combined_joints[:batch_size]
    clean_joints = combined_joints[batch_size:2 * batch_size]
    foot_ids = list(DEFAULT_FOOT_JOINTS)
    predicted_feet = predicted_joints[:, :, foot_ids]
    clean_feet = clean_joints[:, :, foot_ids]

    fk_loss = F.smooth_l1_loss(predicted_joints, clean_joints)
    foot_loss = F.smooth_l1_loss(predicted_feet, clean_feet)

    if prediction.shape[1] > 1:
        fps = float(cfg.fps)
        predicted_velocity = (
            predicted_feet[:, 1:] - predicted_feet[:, :-1]
        ) * fps
        clean_velocity = (clean_feet[:, 1:] - clean_feet[:, :-1]) * fps
        clean_horizontal_speed = torch.linalg.vector_norm(
            clean_velocity[..., (0, 2)], dim=-1
        )
        clean_contact = clean[:, 1:, :4].clamp(0.0, 1.0)
        static_support = (
            clean_horizontal_speed
            <= float(cfg.physics_static_support_speed_mps)
        ).to(clean.dtype) * clean_contact
        support_error = F.smooth_l1_loss(
            predicted_velocity,
            clean_velocity,
            reduction="none",
        ).mean(dim=-1)
        support_loss = (
            (support_error * static_support).sum()
            / static_support.sum().clamp_min(1.0)
        )
    else:
        support_loss = fk_loss.new_zeros(())

    clean_floor = torch.quantile(
        clean_feet[..., 1].detach().reshape(clean.shape[0], -1),
        0.05,
        dim=1,
    )
    predicted_penetration = torch.relu(
        clean_floor[:, None, None] - predicted_feet[..., 1] - 0.008
    )
    clean_penetration = torch.relu(
        clean_floor[:, None, None] - clean_feet[..., 1].detach() - 0.008
    )
    penetration_loss = F.smooth_l1_loss(
        predicted_penetration,
        clean_penetration,
    )

    def derivative_loss(order: int, scale: float):
        if prediction.shape[1] <= order:
            return fk_loss.new_zeros(())
        predicted_delta = torch.diff(
            predicted_joints, n=order, dim=1
        ) * (float(cfg.fps) ** order / scale)
        clean_delta = torch.diff(
            clean_joints, n=order, dim=1
        ) * (float(cfg.fps) ** order / scale)
        return F.smooth_l1_loss(predicted_delta, clean_delta)

    acceleration_loss = derivative_loss(2, 10.0)
    jerk_loss = derivative_loss(3, 1000.0)
    if seam_mask is None:
        endpoint_continuity_loss = fk_loss.new_zeros(())
        seam_velocity_loss = fk_loss.new_zeros(())
        seam_acceleration_loss = fk_loss.new_zeros(())
        seam_jerk_loss = fk_loss.new_zeros(())
    else:
        endpoint_continuity_loss = _masked_temporal_fk_loss_torch(
            predicted_joints,
            clean_joints,
            seam_mask,
            order=1,
            scale=1.0 / float(cfg.fps),
            boundary_only=True,
        )
        seam_velocity_loss = _masked_temporal_fk_loss_torch(
            predicted_joints,
            clean_joints,
            seam_mask,
            order=1,
            scale=1.0 / float(cfg.fps),
        )
        seam_acceleration_loss = _masked_temporal_fk_loss_torch(
            predicted_joints,
            clean_joints,
            seam_mask,
            order=2,
            scale=10.0 / (float(cfg.fps) ** 2),
        )
        seam_jerk_loss = _masked_temporal_fk_loss_torch(
            predicted_joints,
            clean_joints,
            seam_mask,
            order=3,
            scale=1000.0 / (float(cfg.fps) ** 3),
        )
    relative_temporal_loss = fk_loss.new_zeros(())
    if reference is not None and seam_mask is not None:
        proposed = _seam_fk_errors_torch(predicted_joints, clean_joints, seam_mask, cfg)
        baseline = _seam_fk_errors_torch(
            combined_joints[2 * batch_size:], clean_joints.detach(), seam_mask, cfg
        )
        temporal_keys = ("seam_velocity_error", "seam_acceleration_error", "seam_jerk_error")
        proposed_mean = torch.stack([proposed[key] for key in temporal_keys]).mean(dim=0)
        baseline_mean = torch.stack([baseline[key] for key in temporal_keys]).mean(dim=0)
        gain = float(cfg.product_refiner_training_target_repair_gain)
        temporal_margin = _relative_repair_margin_torch(proposed_mean, baseline_mean, gain)
        endpoint_margin = _relative_repair_margin_torch(
            proposed["endpoint_velocity_error"], baseline["endpoint_velocity_error"], gain
        )
        jerk_margin = (
            torch.relu(proposed["seam_jerk_error"] - 1.02 * baseline["seam_jerk_error"].detach() - 1e-6)
            / baseline["seam_jerk_error"].detach().clamp_min(1e-6)
        ).mean()
        relative_temporal_loss = (temporal_margin + endpoint_margin + jerk_margin) / 3.0
    terms = {
        "fk": fk_loss,
        "foot": foot_loss,
        "support": support_loss,
        "penetration": penetration_loss,
        "acceleration": acceleration_loss,
        "jerk": jerk_loss,
        "endpoint_continuity": endpoint_continuity_loss,
        "seam_velocity": seam_velocity_loss,
        "seam_acceleration": seam_acceleration_loss,
        "seam_jerk": seam_jerk_loss,
        "relative_temporal": relative_temporal_loss,
    }
    total = (
        float(cfg.physics_fk_loss_weight) * fk_loss
        + float(cfg.physics_foot_loss_weight) * foot_loss
        + float(cfg.physics_support_loss_weight) * support_loss
        + float(cfg.physics_penetration_loss_weight) * penetration_loss
        + float(cfg.physics_acceleration_loss_weight) * acceleration_loss
        + float(cfg.physics_jerk_loss_weight) * jerk_loss
        + float(cfg.product_refiner_endpoint_continuity_weight)
        * endpoint_continuity_loss
        + float(cfg.product_refiner_seam_velocity_weight)
        * seam_velocity_loss
        + float(cfg.product_refiner_seam_acceleration_weight)
        * seam_acceleration_loss
        + float(cfg.product_refiner_seam_jerk_weight) * seam_jerk_loss
        + float(cfg.product_refiner_relative_temporal_weight) * relative_temporal_loss
    )
    return total, terms


def _training_risk_mask_executor(
    batch_size: int,
) -> Tuple[Optional[ThreadPoolExecutor], int]:
    configured = os.environ.get("MOTION_RISK_MASK_WORKERS")
    if configured is None:
        requested = min(8, max(1, int(os.cpu_count() or 1)))
    else:
        try:
            requested = int(configured)
        except ValueError as exc:
            raise ValueError(
                "MOTION_RISK_MASK_WORKERS must be an integer"
            ) from exc
    workers = min(max(1, requested), max(1, int(batch_size)))
    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    print(
        json.dumps(
            {
                "stage": "risk_mask_batch",
                "batch_size": int(batch_size),
                "workers": workers,
                "parallel": executor is not None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return executor, workers


def _product_motion_losses(
    prediction,
    clean,
    reference,
    joint_mask,
    root_mask,
    contact_mask,
    cfg: MotionGenerationConfig,
    seam_mask=None,
):
    contact_target = clean[..., :4].clamp(0.0, 1.0)
    contact_weight = 0.25 + 0.75 * contact_mask
    # Affine interior mapping keeps the derivative at authentic 0/1 contacts.
    # Hard clamping here would freeze an identity-initialized residual head
    # when a corrupted binary contact is wrong.
    contact_probability = 1.0e-4 + (1.0 - 2.0e-4) * prediction[..., :4]
    contact_loss = (
        F.binary_cross_entropy(
            contact_probability,
            contact_target,
            reduction="none",
        )
        * contact_weight
    ).mean()

    reconstruction_tangent = product_log_torch(prediction, clean)
    reconstruction_loss = F.smooth_l1_loss(
        reconstruction_tangent, torch.zeros_like(reconstruction_tangent)
    )
    if prediction.shape[1] > 1:
        predicted_velocity = product_log_torch(
            prediction[:, :-1], prediction[:, 1:]
        )
        clean_velocity = product_log_torch(clean[:, :-1], clean[:, 1:])
        velocity_loss = F.smooth_l1_loss(predicted_velocity, clean_velocity)
    else:
        velocity_loss = reconstruction_loss.new_zeros(())

    reference_delta = product_log_torch(reference, prediction)
    product_mask = torch.cat(
        [
            root_mask.expand(reference_delta.shape[:-1] + (3,)),
            joint_mask[..., None]
            .expand(joint_mask.shape + (3,))
            .reshape(reference_delta.shape[:-1] + (NUM_JOINTS * 3,)),
        ],
        dim=-1,
    ).clamp(0.0, 1.0)

    # Risk confidence controls edits, not the measurement of successful repair.
    # Match validation exactly: unweighted 75D error over the seam core. The
    # old confidence/halo-weighted loss measured a different objective.
    if seam_mask is None:
        corruption_support = torch.ones_like(product_mask[..., :1])
    else:
        corruption_support = seam_mask
        if corruption_support.ndim == product_mask.ndim - 1:
            corruption_support = corruption_support.unsqueeze(-1)
        if corruption_support.shape[:-1] != product_mask.shape[:-1]:
            raise ValueError(
                "seam_mask must be frame-aligned with the product motion loss"
            )
        corruption_support = corruption_support.amax(
            dim=-1,
            keepdim=True,
        ).clamp(0.0, 1.0)
    repair_mask = (corruption_support >= 0.5).to(product_mask.dtype)

    def weighted_per_sample(value, weight):
        expanded = torch.broadcast_to(weight, value.shape)
        numerator = (value * expanded).sum(dim=(1, 2))
        denominator = expanded.sum(dim=(1, 2)).clamp_min(1.0)
        return numerator / denominator

    active_reconstruction_per_sample = weighted_per_sample(
        reconstruction_tangent.abs(),
        repair_mask,
    )
    active_reconstruction_loss = active_reconstruction_per_sample.mean()

    target_delta = product_log_torch(reference, clean)
    applied_delta = product_log_torch(reference, prediction)
    tangent_supervision_loss = weighted_per_sample(
        (applied_delta - target_delta).abs(),
        repair_mask,
    ).mean()
    degraded_error = weighted_per_sample(
        target_delta.detach().abs(),
        repair_mask,
    )
    minimum_gain = float(cfg.product_refiner_training_target_repair_gain)
    if not np.isfinite(minimum_gain) or not 0.0 < minimum_gain <= 1.0:
        raise ValueError(
            "product_refiner_training_target_repair_gain must be finite in "
            f"(0, 1], got {minimum_gain!r}"
        )
    repair_margin_loss = _relative_repair_margin_torch(
        active_reconstruction_per_sample, degraded_error, minimum_gain
    )
    clean_preservation_loss = weighted_per_sample(
        applied_delta.abs(),
        product_mask * (1.0 - corruption_support),
    ).mean()

    # Per-frame reconstruction alone permits a correction tangent with small
    # framewise error but damaging high-frequency variation. Match temporal
    # derivatives of the supervised correction on the corrupted seam itself;
    # whole-window velocity/physics losses otherwise dilute this signal.
    tangent_error = applied_delta - target_delta
    temporal_parts = []
    for order, order_weight in ((1, 1.0), (2, 0.5), (3, 0.25)):
        if tangent_error.shape[1] <= order:
            continue
        derivative = torch.diff(tangent_error, n=order, dim=1).abs()
        support = repair_mask[:, order:].clone()
        width = derivative.shape[1]
        for offset in range(order):
            support = torch.maximum(
                support,
                repair_mask[:, offset:offset + width],
            )
        temporal_parts.append(
            float(order_weight)
            * weighted_per_sample(derivative, support).mean()
        )
    temporal_supervision_loss = (
        torch.stack(temporal_parts).sum()
        if temporal_parts
        else reconstruction_loss.new_zeros(())
    )

    outside_loss = (
        F.smooth_l1_loss(
            reference_delta,
            torch.zeros_like(reference_delta),
            reduction="none",
        )
        * (1.0 - product_mask)
    ).mean()
    physics_loss, physics_terms = _world_space_physics_losses(
        prediction,
        clean,
        cfg,
        seam_mask=seam_mask,
        reference=reference,
    )
    total = (
        float(cfg.product_refiner_global_reconstruction_weight)
        * reconstruction_loss
        + float(cfg.product_refiner_active_reconstruction_weight)
        * active_reconstruction_loss
        + float(cfg.product_refiner_tangent_supervision_weight)
        * tangent_supervision_loss
        + float(cfg.product_refiner_repair_margin_weight)
        * repair_margin_loss
        + float(cfg.product_refiner_clean_preservation_weight)
        * clean_preservation_loss
        + float(cfg.product_refiner_temporal_supervision_weight)
        * temporal_supervision_loss
        + 0.25 * velocity_loss
        + 0.20 * contact_loss
        + float(cfg.product_refiner_outside_weight) * outside_loss
        + physics_loss
    )
    terms = {
        "reconstruction": reconstruction_loss,
        "active_reconstruction": active_reconstruction_loss,
        "tangent_supervision": tangent_supervision_loss,
        "repair_margin": repair_margin_loss,
        "clean_preservation": clean_preservation_loss,
        "temporal_supervision": temporal_supervision_loss,
        "degraded_active_product_l1": degraded_error.mean(),
        "velocity": velocity_loss,
        "contact": contact_loss,
        "outside": outside_loss,
        "physics": physics_loss,
        "endpoint_continuity": physics_terms["endpoint_continuity"],
        "seam_velocity": physics_terms["seam_velocity"],
        "seam_acceleration": physics_terms["seam_acceleration"],
        "seam_jerk": physics_terms["seam_jerk"],
        "relative_temporal": physics_terms["relative_temporal"],
    }
    terms.update({f"physics_{key}": value for key, value in physics_terms.items()})
    return total, terms


def _product_refiner_clean_identity_loss(
    prediction,
    clean,
    joint_mask,
    root_mask,
    contact_mask,
    cfg: MotionGenerationConfig,
):
    """Penalize any Refiner geometry edit on an authentic clean transition."""

    identity_delta = product_log_torch(clean, prediction)
    product_mask = torch.cat(
        [
            root_mask.expand(identity_delta.shape[:-1] + (3,)),
            joint_mask[..., None]
            .expand(joint_mask.shape + (3,))
            .reshape(identity_delta.shape[:-1] + (NUM_JOINTS * 3,)),
        ],
        dim=-1,
    ).clamp(0.0, 1.0)

    def weighted(value, weight):
        expanded = torch.broadcast_to(weight, value.shape)
        return (value * expanded).sum() / expanded.sum().clamp_min(1.0)

    reconstruction = weighted(identity_delta.abs(), product_mask)
    contact_weight = 0.25 + 0.75 * contact_mask.clamp(0.0, 1.0)
    contact = weighted(
        (prediction[..., :4] - clean[..., :4]).abs(),
        contact_weight,
    )
    # Independent of the retired exact-repair tangent objective: switching that
    # objective off must not silently disable clean-input temporal protection.
    joints = fk_24_torch(torch.cat([prediction, clean], dim=0))
    predicted_joints, clean_joints = joints.split(prediction.shape[0], dim=0)
    fk_temporal = reconstruction.new_zeros(())
    for order, scale, weight in ((1, 1.0, 1.0), (2, 10.0, 0.5), (3, 1000.0, 0.25)):
        if prediction.shape[1] <= order:
            continue
        delta = (
            torch.diff(predicted_joints, n=order, dim=1)
            - torch.diff(clean_joints, n=order, dim=1)
        ) * (float(cfg.fps)**order / scale)
        norms = torch.linalg.vector_norm(delta, dim=-1).flatten(1)
        # Include the worst edited joint/frame; a full-window mean alone can
        # hide a short, high-jerk artifact that fails the identity audit.
        fk_temporal = fk_temporal + weight * (
            0.5 * norms.mean(dim=1) + 0.5 * norms.amax(dim=1)
        ).mean()
    total = (
        reconstruction / max(float(cfg.checkpoint_validation_max_clean_identity_product_log_l1), 1e-6)
        + float(cfg.product_refiner_clean_identity_temporal_weight) * fk_temporal
        + 0.25 * contact
    )
    return total, {
        "reconstruction": reconstruction,
        "temporal": fk_temporal,
        "fk_temporal": fk_temporal,
        "contact": contact,
    }


def sample_motion_window(paths: np.ndarray, target_len: int, cfg: Optional[MotionGenerationConfig] = None) -> np.ndarray:
    """Sample a training window and keep the EDGE-151D contract after resampling."""
    p = str(random.choice(paths.tolist()))
    return load_motion_window(p, target_len, cfg)


def _load_motion_array(path: str | Path) -> np.ndarray:
    """Load one event array once and validate the training representation."""
    motion = np.asarray(np.load(str(path), allow_pickle=False), dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != EDGE_DIM:
        raise RuntimeError(
            f"motion event must have shape [frames,{EDGE_DIM}], got "
            f"{motion.shape} for {path}"
        )
    if motion.shape[0] < 1:
        raise RuntimeError(f"motion event has no frames: {path}")
    if not np.isfinite(motion).all():
        raise RuntimeError(f"motion event contains non-finite values: {path}")
    return np.ascontiguousarray(motion)


def _motion_window_from_array(
    motion: np.ndarray,
    target_len: int,
    cfg: Optional[MotionGenerationConfig],
    *,
    source_hint: str,
    random_crop: bool = True,
) -> np.ndarray:
    """Create a contract-valid window without performing file I/O."""
    m = np.asarray(motion, dtype=np.float32)
    if m.shape[0] == target_len:
        out = m
    elif m.shape[0] > target_len:
        start = (
            random.randint(0, m.shape[0] - target_len)
            if random_crop
            else (m.shape[0] - target_len) // 2
        )
        out = m[start:start + target_len]
    else:
        out = resample_motion_np(m, target_len)
    out, _ = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint=source_hint,
        derive_contact=True,
        project_rot=True,
    )
    return np.ascontiguousarray(out, dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class PreloadedMotionWindowPool:
    """Event-aligned in-memory windows for formal motion-model training.

    Refiner events longer than the requested window remain raw so every sample
    keeps the original random-crop distribution. Shorter/equal events and all
    diffusion inputs are deterministic, so their contract-valid windows are
    prepared once instead of being rebuilt at every optimization step.
    """

    paths: Tuple[str, ...]
    raw_motions: Tuple[np.ndarray, ...]
    fixed_windows: Tuple[Optional[np.ndarray], ...]
    target_len: int
    mode: str
    cfg: MotionGenerationConfig
    unique_paths: int
    resident_bytes: int

    @classmethod
    def preload(
        cls,
        paths: Sequence[str | Path],
        target_len: int,
        cfg: MotionGenerationConfig,
        *,
        mode: str,
    ) -> "PreloadedMotionWindowPool":
        if mode not in {"refiner", "diffusion"}:
            raise ValueError(f"unsupported motion preload mode: {mode!r}")
        normalized_paths = tuple(str(path) for path in paths)
        if not normalized_paths:
            raise RuntimeError(f"cannot preload an empty {mode} motion dataset")

        array_cache: Dict[str, np.ndarray] = {}
        fixed_cache: Dict[str, np.ndarray] = {}
        raw_motions: List[np.ndarray] = []
        fixed_windows: List[Optional[np.ndarray]] = []
        for path in normalized_paths:
            if path not in array_cache:
                raw = _load_motion_array(path)
                raw.setflags(write=False)
                array_cache[path] = raw
            raw = array_cache[path]
            raw_motions.append(raw)

            deterministic = mode == "diffusion" or raw.shape[0] <= target_len
            fixed: Optional[np.ndarray] = None
            if deterministic:
                if path not in fixed_cache:
                    if mode == "diffusion":
                        prepared = resample_motion_np(raw, target_len)
                        prepared, _ = enforce_edge151_contract_np(
                            prepared,
                            cfg,
                            source_hint=f"train_diffusion_clean:{path}",
                            derive_contact=True,
                            project_rot=True,
                        )
                        prepared = np.ascontiguousarray(
                            prepared, dtype=np.float32
                        )
                    else:
                        prepared = _motion_window_from_array(
                            raw,
                            target_len,
                            cfg,
                            source_hint=f"sample_motion_window:{path}",
                            random_crop=False,
                        )
                    prepared.setflags(write=False)
                    fixed_cache[path] = prepared
                fixed = fixed_cache[path]
            fixed_windows.append(fixed)

        resident_bytes = sum(value.nbytes for value in array_cache.values())
        resident_bytes += sum(value.nbytes for value in fixed_cache.values())
        pool = cls(
            paths=normalized_paths,
            raw_motions=tuple(raw_motions),
            fixed_windows=tuple(fixed_windows),
            target_len=int(target_len),
            mode=mode,
            cfg=cfg,
            unique_paths=len(array_cache),
            resident_bytes=int(resident_bytes),
        )
        print(
            json.dumps(
                {
                    "stage": f"{mode}_motion_preload",
                    "events": len(normalized_paths),
                    "unique_paths": pool.unique_paths,
                    "fixed_windows": sum(
                        value is not None for value in fixed_windows
                    ),
                    "random_crop_windows": sum(
                        value is None for value in fixed_windows
                    ),
                    "resident_mib": round(pool.resident_bytes / (1024 ** 2), 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return pool

    def sample(self, index: int) -> np.ndarray:
        fixed = self.fixed_windows[index]
        if fixed is not None:
            # Training corruption copies its input before mutation and the clean
            # batch is stacked later, so returning the read-only cached view
            # avoids an otherwise redundant full-window memcpy per sample.
            return fixed
        path = self.paths[index]
        return _motion_window_from_array(
            self.raw_motions[index],
            self.target_len,
            self.cfg,
            source_hint=f"sample_motion_window:{path}",
            random_crop=True,
        )


def load_motion_window(
    path: str | Path,
    target_len: int,
    cfg: Optional[MotionGenerationConfig] = None,
    *,
    random_crop: bool = True,
) -> np.ndarray:
    """Load one event and return a contract-valid fixed-length training window."""
    p = str(path)
    return _motion_window_from_array(
        _load_motion_array(p),
        target_len,
        cfg,
        source_hint=f"sample_motion_window:{p}",
        random_crop=random_crop,
    )





def _smooth_standard_normal_time_np(
    shape: Tuple[int, ...],
    *,
    passes: int,
) -> np.ndarray:
    """Return temporally correlated, per-channel unit-RMS Gaussian noise."""

    if not shape or int(shape[0]) < 1:
        return np.zeros(shape, dtype=np.float32)
    value = np.random.normal(0.0, 1.0, size=shape).astype(np.float32)
    for _ in range(max(1, int(passes))):
        pad_width = [(1, 1)] + [(0, 0)] * (value.ndim - 1)
        padded = np.pad(value, pad_width, mode="edge")
        value = (
            0.25 * padded[:-2]
            + 0.50 * padded[1:-1]
            + 0.25 * padded[2:]
        ).astype(np.float32)
    value -= np.mean(value, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(value * value, axis=0, keepdims=True))
    return (value / np.maximum(rms, 1.0e-6)).astype(np.float32)


def degrade_for_refiner(
    clean: np.ndarray,
    severity: float = 0.06,
    cfg: Optional[MotionGenerationConfig] = None,
    *,
    finalize_contract: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference Inbetweening transition-masked corruption for Motion Refiner and Motion Diffusion training.

    Instead of arbitrary global drift only, corrupt a local transition region by
    replacing it with a weak root-Hermite / rotation-SLERP inbetweening path plus
    smooth product-manifold tangent perturbations. This matches the
    inference-time transition-budget mask: the model learns to repair motion_ref
    only near boundaries while preserving core clips exactly.
    """
    cfg = cfg or MotionGenerationConfig()
    x = np.asarray(clean, dtype=np.float32).copy()
    T = x.shape[0]
    seam = np.zeros((T, 1), dtype=np.float32)
    if T <= 12:
        if finalize_contract:
            x, _ = enforce_edge151_contract_np(x, cfg, source_hint="inbetween_degrade_too_short", derive_contact=True, project_rot=True)
        return x.astype(np.float32), seam

    min_w = max(1, int(round(float(cfg.transition_train_min_seconds) * float(cfg.fps))))
    max_w = max(min_w, int(round(float(cfg.transition_train_max_seconds) * float(cfg.fps))))
    halo = max(0, int(round(float(cfg.transition_mask_halo_seconds) * float(cfg.fps))))
    max_w = max(min_w, min(max_w, max(4, T // 3)))
    w = random.randint(max(4, min_w), max_w)
    c = random.randint(max(2, T // 5), max(3, 4 * T // 5))
    a = max(1, c - w // 2)
    b = min(T - 1, a + w)
    a = max(1, b - w)
    if b - a >= 3:
        prev_tail = x[max(0, a - 4):a]
        curr_head = x[b:min(T, b + 4)]
        if prev_tail.shape[0] >= 1 and curr_head.shape[0] >= 1:
            bridge = reference_motion_inbetween_np(
                prev_tail,
                curr_head,
                b - a,
                cfg,
                finalize_contract=False,
            )
            # Never add Euclidean noise directly to Rot6D.  Build one C1-tapered,
            # temporally correlated tangent perturbation and retract it on
            # R^3 x SO(3)^24.  This creates realistic boundary uncertainty
            # without manufacturing frame-independent jerk across the song.
            span = int(bridge.shape[0])
            phase = (
                (np.arange(span, dtype=np.float32) + 1.0)
                / float(span + 1)
            )
            taper = (np.sin(np.pi * phase) ** 2).astype(np.float32)
            passes = max(
                1,
                int(getattr(cfg, "transition_tangent_smoothing_passes", 4)),
            )
            root_noise = _smooth_standard_normal_time_np(
                (span, 3), passes=passes
            )
            joint_noise = _smooth_standard_normal_time_np(
                (span, NUM_JOINTS, 3), passes=passes
            )
            tangent = np.zeros((span, 3 + NUM_JOINTS * 3), dtype=np.float32)
            tangent[:, :3] = (
                root_noise * taper[:, None] * float(severity) * 0.18
            )
            tangent[:, 3:] = (
                joint_noise
                * taper[:, None, None]
                * float(severity)
                * 0.08
            ).reshape(span, -1)
            bridge_mix = float(getattr(cfg, "transition_bridge_mix", 0.50))
            if not np.isfinite(bridge_mix) or not 0.0 < bridge_mix <= 1.0:
                raise ValueError(
                    "transition_bridge_mix must be finite in (0, 1], got "
                    f"{bridge_mix!r}"
                )
            clean_segment = x[a:b].copy()
            bridge_delta = product_log_np(clean_segment, bridge)
            mixed_bridge = masked_retract_np(
                clean_segment,
                bridge_delta * bridge_mix,
                joint_mask=np.ones((span, NUM_JOINTS), dtype=np.float32),
                root_mask=np.ones((span, 1), dtype=np.float32),
            )
            x[a:b] = masked_retract_np(
                mixed_bridge,
                tangent,
                joint_mask=np.ones((span, NUM_JOINTS), dtype=np.float32),
                root_mask=np.ones((span, 1), dtype=np.float32),
                max_rotation_rad=max(1.0e-4, float(severity) * 0.20),
                max_root_m=max(1.0e-4, float(severity) * 0.30),
            )
            seam[max(0, a - halo):min(T, b + halo), 0] = 0.35
            seam[a:b, 0] = 1.0
    if finalize_contract:
        x, _ = enforce_edge151_contract_np(x, cfg, source_hint="inbetween_degrade_for_transition_refiner", derive_contact=True, project_rot=True)
    return x.astype(np.float32), np.clip(seam, 0.0, 1.0).astype(np.float32)


def _validation_indices(count: int, maximum: int = 16) -> List[int]:
    if count < 1:
        return []
    return sorted(set(np.linspace(0, count - 1, min(count, maximum), dtype=np.int64).tolist()))


_VALIDATION_PHYSICAL_KEYS = (
    "foot_skate_mps_p95",
    "foot_skate_mps_max",
    "foot_support_drift_m_p95",
    "foot_support_drift_m_max",
    "foot_contact_height_m_max",
    "foot_contact_mismatch_ratio",
    "foot_penetration_min_m",
    "joint_jerk_mps3_p95",
    "joint_jerk_mps3_max",
    "joint_jerk_window_p95_max_mps3",
    "extremity_jerk_mps3_p95",
    "extremity_jerk_window_p95_max_mps3",
    "joint_rotation_step_rad_p95",
    "joint_rotation_step_rad_max",
    "joint_rotation_step_window_p95_max_rad",
    "root_y_robust_range_m",
    "root_vertical_speed_mps_p95",
    "root_horizontal_radius_p95_m",
    "root_horizontal_net_displacement_m",
    "root_horizontal_window_displacement_max_m",
)


def _new_validation_physical_accumulator() -> Dict[str, Any]:
    return {
        "prediction_audits": [],
        "clean_audits": [],
        "degraded_audits": [],
        "stage_repair_gates": [],
        "temporal_repair_gates": [],
        "clean_non_regression_gates": [],
        "clean_fidelity_gates": [],
        "clean_identity_prediction_audits": [],
        "clean_identity_gates": [],
        "clean_identity_product_errors": [],
        "clean_identity_contact_errors": [],
        "prediction_final_diagnostic_gates": [],
        "clean_final_diagnostic_gates": [],
        "fk_errors": [],
    }


def _derivative_support_np(
    seam_mask: np.ndarray,
    order: int,
    *,
    boundary_only: bool = False,
) -> np.ndarray:
    seam = np.asarray(seam_mask, dtype=np.float32)
    if seam.ndim == 2 and seam.shape[-1] == 1:
        seam = seam[:, 0]
    if seam.ndim != 1:
        raise ValueError(f"seam_mask must be [T] or [T,1], got {seam.shape}")
    width = len(seam) - int(order)
    if width <= 0:
        return np.zeros((0,), dtype=np.float32)
    core = (seam >= 0.5).astype(np.float32)
    stencil = np.stack(
        [core[offset:offset + width] for offset in range(order + 1)],
        axis=-1,
    )
    maximum = np.max(stencil, axis=-1)
    if boundary_only:
        return (maximum - np.min(stencil, axis=-1)).astype(np.float32)
    return maximum.astype(np.float32)


def _seam_fk_temporal_errors_np(
    candidate: np.ndarray,
    clean: np.ndarray,
    seam_mask: np.ndarray,
    cfg: MotionGenerationConfig,
) -> Dict[str, float]:
    candidate_joints = fk_24_np(np.asarray(candidate, dtype=np.float32))
    clean_joints = fk_24_np(np.asarray(clean, dtype=np.float32))

    def error(order: int, scale: float, *, boundary_only: bool = False) -> float:
        if len(candidate_joints) <= order:
            return 0.0
        support = _derivative_support_np(
            seam_mask,
            order,
            boundary_only=boundary_only,
        )
        if not np.any(support > 0.0):
            return 0.0
        candidate_delta = np.diff(candidate_joints, n=order, axis=0) / float(scale)
        clean_delta = np.diff(clean_joints, n=order, axis=0) / float(scale)
        per_frame = np.mean(np.abs(candidate_delta - clean_delta), axis=(1, 2))
        return float(np.sum(per_frame * support) / max(float(np.sum(support)), 1.0))

    fps = float(cfg.fps)
    return {
        "endpoint_velocity_error": error(
            1,
            1.0 / fps,
            boundary_only=True,
        ),
        "seam_velocity_error": error(1, 1.0 / fps),
        "seam_acceleration_error": error(2, 10.0 / (fps ** 2)),
        "seam_jerk_error": error(3, 1000.0 / (fps ** 3)),
    }


def _temporal_repair_gate_np(
    prediction: np.ndarray,
    degraded: np.ndarray,
    clean: np.ndarray,
    seam_mask: np.ndarray,
    cfg: MotionGenerationConfig,
) -> Dict[str, Any]:
    prediction_error = _seam_fk_temporal_errors_np(
        prediction,
        clean,
        seam_mask,
        cfg,
    )
    degraded_error = _seam_fk_temporal_errors_np(
        degraded,
        clean,
        seam_mask,
        cfg,
    )
    floor = 1.0e-6

    def relative_gain(reference: float, candidate: float) -> float:
        if reference <= floor:
            return 1.0 if candidate <= floor else float("-inf")
        return float((reference - candidate) / reference)

    temporal_keys = (
        "seam_velocity_error",
        "seam_acceleration_error",
        "seam_jerk_error",
    )
    degraded_temporal = float(np.mean([degraded_error[key] for key in temporal_keys]))
    prediction_temporal = float(np.mean([prediction_error[key] for key in temporal_keys]))
    temporal_gain = relative_gain(degraded_temporal, prediction_temporal)
    endpoint_gain = relative_gain(
        degraded_error["endpoint_velocity_error"],
        prediction_error["endpoint_velocity_error"],
    )
    jerk_non_regression = bool(
        prediction_error["seam_jerk_error"]
        <= degraded_error["seam_jerk_error"] * 1.02 + floor
    )
    reasons: List[str] = []
    if (
        not np.isfinite(temporal_gain)
        or temporal_gain
        < float(cfg.checkpoint_validation_min_temporal_repair_gain)
    ):
        reasons.append("no_meaningful_temporal_repair_gain")
    if (
        not np.isfinite(endpoint_gain)
        or endpoint_gain
        < float(cfg.checkpoint_validation_min_endpoint_repair_gain)
    ):
        reasons.append("no_meaningful_endpoint_continuity_gain")
    if not jerk_non_regression:
        reasons.append("seam_fk_jerk_regressed")
    return {
        "schema": "checkpoint_seam_fk_temporal_repair_gate_v1",
        "accepted": not reasons,
        "reasons": reasons,
        "detail": {
            "prediction": prediction_error,
            "degraded": degraded_error,
            "temporal_repair_gain": temporal_gain,
            "endpoint_repair_gain": endpoint_gain,
            "minimum_temporal_repair_gain": float(
                cfg.checkpoint_validation_min_temporal_repair_gain
            ),
            "minimum_endpoint_repair_gain": float(
                cfg.checkpoint_validation_min_endpoint_repair_gain
            ),
            "jerk_non_regression": jerk_non_regression,
        },
    }


def _safe_validation_audit(
    motion: np.ndarray,
    cfg: MotionGenerationConfig,
    *,
    role: str,
    support_policy: str,
) -> Dict[str, Any]:
    try:
        return audit_motion_np(
            np.asarray(motion, dtype=np.float32),
            cfg,
            support_policy=support_policy,
        )
    except Exception as exc:
        # Never project or sanitize a broken model prediction into an
        # apparently healthy checkpoint validation sample.
        return {
            "schema": f"invalid_checkpoint_validation_{role}",
            "validation_audit_error": f"{type(exc).__name__}: {exc}",
        }


def _record_validation_physical_prediction(
    accumulator: Dict[str, Any],
    prediction: np.ndarray,
    clean: np.ndarray,
    cfg: MotionGenerationConfig,
    *,
    degraded: Optional[np.ndarray] = None,
    seam_mask: Optional[np.ndarray] = None,
) -> None:
    predicted = np.asarray(prediction, dtype=np.float32)
    target, _ = enforce_edge151_contract_np(
        np.asarray(clean, dtype=np.float32),
        cfg,
        source_hint="checkpoint_validation_target",
        derive_contact=True,
        project_rot=True,
    )
    damaged = (
        target
        if degraded is None
        else np.asarray(degraded, dtype=np.float32)
    )

    # Checkpoint validation is source/event-stage aware.  Fast low-foot source
    # observations remain swing, and authentic short-event travel is compared
    # to its own clean reference instead of being called whole-song drift.
    prediction_audit = _safe_validation_audit(
        predicted,
        cfg,
        role="prediction",
        support_policy="source_observation",
    )
    clean_audit = _safe_validation_audit(
        target,
        cfg,
        role="clean_target",
        support_policy="source_observation",
    )
    degraded_audit = _safe_validation_audit(
        damaged,
        cfg,
        role="degraded_input",
        support_policy="source_observation",
    )
    degraded_physical_diagnostic = evaluate_stage_candidate(
        degraded_audit,
        prediction_audit,
        require_repair_gain=False,
        ignored_layers=("long_horizon_root_drift",),
    )
    fidelity_gate = evaluate_stage_reference_fidelity(
        clean_audit,
        prediction_audit,
    )
    clean_non_regression_gate = evaluate_stage_candidate(
        clean_audit,
        prediction_audit,
        require_repair_gain=False,
        ignored_layers=("long_horizon_root_drift",),
    )
    try:
        degraded_product_delta = np.abs(product_log_np(target, damaged))
        prediction_product_delta = np.abs(product_log_np(target, predicted))
        if seam_mask is not None:
            seam = np.asarray(seam_mask, dtype=np.float32).reshape(-1)
            local = seam >= 0.5
            if np.any(local):
                degraded_product_delta = degraded_product_delta[local]
                prediction_product_delta = prediction_product_delta[local]
        degraded_product_error = float(np.mean(degraded_product_delta))
        prediction_product_error = float(np.mean(prediction_product_delta))
        if not np.isfinite(degraded_product_error):
            degraded_product_error = float("inf")
        if not np.isfinite(prediction_product_error):
            prediction_product_error = float("inf")
    except Exception:
        degraded_product_error = float("inf")
        prediction_product_error = float("inf")
    repair_floor = 1.0e-6
    stage_reasons: List[str] = []
    stage_detail: Dict[str, Any] = {
        "degraded_product_log_l1_to_clean": degraded_product_error,
        "prediction_product_log_l1_to_clean": prediction_product_error,
        "minimum_geometry_repair_gain": float(
            cfg.checkpoint_validation_min_geometry_repair_gain
        ),
    }
    if not np.isfinite(degraded_product_error) or not np.isfinite(
        prediction_product_error
    ):
        geometry_repair_gain = float("-inf")
        stage_reasons.append("nonfinite_product_geometry_error")
    elif degraded is not None and degraded_product_error > repair_floor:
        geometry_repair_gain = (
            degraded_product_error - prediction_product_error
        ) / degraded_product_error
        minimum_gain = float(
            cfg.checkpoint_validation_min_geometry_repair_gain
        )
        if not np.isfinite(geometry_repair_gain) or geometry_repair_gain < minimum_gain:
            stage_reasons.append("no_meaningful_geometry_repair_gain")
    else:
        # Compatibility for direct validation helpers without synthetic
        # corruption: exact/near-exact clean reconstruction is healthy.
        geometry_repair_gain = (
            1.0 if prediction_product_error <= repair_floor else float("-inf")
        )
        if prediction_product_error > repair_floor:
            stage_reasons.append("uncorrupted_reference_not_reconstructed")
    stage_detail["geometry_repair_gain"] = float(geometry_repair_gain)
    stage_gate = {
        "schema": "checkpoint_product_geometry_repair_gate_v1",
        "accepted": not stage_reasons,
        "reasons": list(dict.fromkeys(stage_reasons)),
        "detail": stage_detail,
        "acceptance_basis": "product_geometry_repair_toward_clean",
        # Smoothing corruption can score physically better than authentic
        # motion.  Keep this comparison for diagnosis, never as acceptance.
        "degraded_physical_comparison_diagnostic": (
            degraded_physical_diagnostic
        ),
    }
    if seam_mask is None:
        temporal_gate = {
            "schema": "checkpoint_seam_fk_temporal_repair_gate_v1",
            "accepted": True,
            "reasons": [],
            "detail": {
                "compatibility_without_seam": True,
                "temporal_repair_gain": 1.0,
                "endpoint_repair_gain": 1.0,
                "jerk_non_regression": True,
            },
        }
    else:
        temporal_gate = _temporal_repair_gate_np(
            predicted,
            damaged,
            target,
            seam_mask,
            cfg,
        )

    # Preserve the strict final gate as a named diagnostic only.  It is not a
    # checkpoint criterion before Diffusion, boundary repair and IK.
    prediction_final_audit = _safe_validation_audit(
        predicted,
        cfg,
        role="prediction_final_diagnostic",
        support_policy="final_fail_closed",
    )
    clean_final_audit = _safe_validation_audit(
        target,
        cfg,
        role="clean_final_diagnostic",
        support_policy="final_fail_closed",
    )
    prediction_final_gate = evaluate_physical_audit(prediction_final_audit)
    clean_final_gate = evaluate_physical_audit(clean_final_audit)
    try:
        fk_error = np.linalg.norm(
            fk_24_np(predicted) - fk_24_np(target), axis=-1
        )
        if not np.isfinite(fk_error).all():
            fk_error = np.full((1,), np.inf, dtype=np.float32)
    except Exception:
        fk_error = np.full((1,), np.inf, dtype=np.float32)
    accumulator["prediction_audits"].append(prediction_audit)
    accumulator["clean_audits"].append(clean_audit)
    accumulator["degraded_audits"].append(degraded_audit)
    accumulator["stage_repair_gates"].append(stage_gate)
    accumulator["temporal_repair_gates"].append(temporal_gate)
    accumulator["clean_non_regression_gates"].append(
        clean_non_regression_gate
    )
    accumulator["clean_fidelity_gates"].append(fidelity_gate)
    accumulator["prediction_final_diagnostic_gates"].append(
        prediction_final_gate
    )
    accumulator["clean_final_diagnostic_gates"].append(clean_final_gate)
    accumulator["fk_errors"].append(fk_error.reshape(-1))


def _record_validation_clean_identity_prediction(
    accumulator: Dict[str, Any],
    prediction: np.ndarray,
    clean: np.ndarray,
    cfg: MotionGenerationConfig,
) -> None:
    """Audit the Refiner's response when its input is already authentic."""

    target, _ = enforce_edge151_contract_np(
        np.asarray(clean, dtype=np.float32),
        cfg,
        source_hint="checkpoint_validation_clean_identity_target",
        derive_contact=True,
        project_rot=True,
    )
    predicted = np.asarray(prediction, dtype=np.float32)
    target_audit = _safe_validation_audit(
        target,
        cfg,
        role="clean_identity_target",
        support_policy="source_observation",
    )
    prediction_audit = _safe_validation_audit(
        predicted,
        cfg,
        role="clean_identity_prediction",
        support_policy="source_observation",
    )
    gate = dict(
        evaluate_stage_reference_fidelity(target_audit, prediction_audit)
    )
    try:
        error = float(np.mean(np.abs(product_log_np(target, predicted))))
        if not np.isfinite(error):
            error = float("inf")
    except Exception:
        error = float("inf")
    try:
        contact_error = float(
            np.mean(np.abs(predicted[:, :4] - target[:, :4]))
        )
        if not np.isfinite(contact_error):
            contact_error = float("inf")
    except Exception:
        contact_error = float("inf")
    identity_reasons = list(gate.get("reasons", []))
    if error > float(
        cfg.checkpoint_validation_max_clean_identity_product_log_l1
    ):
        identity_reasons.append("clean_identity_product_log_l1_too_high")
    if contact_error > float(
        cfg.checkpoint_validation_max_clean_identity_contact_l1
    ):
        identity_reasons.append("clean_identity_contact_l1_too_high")
    gate["reasons"] = list(dict.fromkeys(identity_reasons))
    gate["accepted"] = not gate["reasons"]
    gate["identity_detail"] = {
        "product_log_l1": error,
        "maximum_product_log_l1": float(
            cfg.checkpoint_validation_max_clean_identity_product_log_l1
        ),
        "contact_l1": contact_error,
        "maximum_contact_l1": float(
            cfg.checkpoint_validation_max_clean_identity_contact_l1
        ),
    }
    accumulator["clean_identity_prediction_audits"].append(prediction_audit)
    accumulator["clean_identity_gates"].append(gate)
    accumulator["clean_identity_product_errors"].append(error)
    accumulator["clean_identity_contact_errors"].append(contact_error)


def _summarize_validation_audits(
    audits: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    worst_window: Dict[str, Optional[float]] = {}
    mean_across_windows: Dict[str, Optional[float]] = {}
    for key in _VALIDATION_PHYSICAL_KEYS:
        values = [float(audit[key]) for audit in audits if key in audit]
        if not values:
            worst_window[key] = None
            mean_across_windows[key] = None
            continue
        worst_window[key] = (
            float(min(values))
            if key == "foot_penetration_min_m"
            else float(max(values))
        )
        mean_across_windows[key] = float(np.mean(values))
    return {
        "worst_window": worst_window,
        "mean_across_windows": mean_across_windows,
    }


def _summarize_validation_gates(
    gates: Sequence[Mapping[str, Any]],
    *,
    accepted_key: str,
) -> Dict[str, Any]:
    failure_counts: Dict[str, int] = {}
    passed = 0
    for gate in gates:
        passed += int(bool(gate.get(accepted_key, False)))
        for reason in gate.get("reasons", []):
            failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + 1
    count = len(gates)
    return {
        "pass_rate": float(passed / count) if count else None,
        "passed_windows": int(passed),
        "failed_windows": int(count - passed),
        "failure_reasons": failure_counts,
    }


def _summarize_validation_physical_metrics(
    accumulator: Mapping[str, Any],
) -> Dict[str, Any]:
    prediction_audits = list(accumulator.get("prediction_audits", []))
    clean_audits = list(accumulator.get("clean_audits", []))
    degraded_audits = list(accumulator.get("degraded_audits", []))
    stage_repair_gates = list(accumulator.get("stage_repair_gates", []))
    temporal_repair_gates = list(
        accumulator.get("temporal_repair_gates", [])
    )
    error_parts = list(accumulator.get("fk_errors", []))
    errors = np.concatenate(error_parts) if error_parts else np.zeros(0, dtype=np.float32)
    prediction_summary = _summarize_validation_audits(prediction_audits)
    clean_identity_errors = np.asarray(
        accumulator.get("clean_identity_product_errors", []),
        dtype=np.float64,
    )
    clean_identity_contact_errors = np.asarray(
        accumulator.get("clean_identity_contact_errors", []),
        dtype=np.float64,
    )
    geometry_gain_minimum = next(
        (
            gate.get("detail", {}).get("minimum_geometry_repair_gain")
            for gate in stage_repair_gates
            if gate.get("detail", {}).get("minimum_geometry_repair_gain")
            is not None
        ),
        None,
    )

    def summarize_gate_detail(
        gates: Sequence[Mapping[str, Any]],
        key: str,
    ) -> Dict[str, Optional[float]]:
        values = [
            float(gate.get("detail", {}).get(key))
            for gate in gates
            if gate.get("detail", {}).get(key) is not None
        ]
        finite = np.asarray(
            [value for value in values if np.isfinite(value)],
            dtype=np.float64,
        )
        return {
            "mean": float(np.mean(finite)) if finite.size else None,
            "p50": float(np.percentile(finite, 50)) if finite.size else None,
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
        }

    def summarize_stage_detail(key: str) -> Dict[str, Optional[float]]:
        return summarize_gate_detail(stage_repair_gates, key)

    return {
        "schema": "motion_checkpoint_stage_validation_v3",
        "num_windows": len(prediction_audits),
        "fk_position_error_m_mean": float(np.mean(errors)) if errors.size else None,
        "fk_position_error_m_p95": float(np.percentile(errors, 95)) if errors.size else None,
        "fk_position_error_m_max": float(np.max(errors)) if errors.size else None,
        # Compatibility locations now unambiguously describe the prediction,
        # while acceptance is reported in named stage contracts below.
        **prediction_summary,
        "prediction": prediction_summary,
        "clean_target": _summarize_validation_audits(clean_audits),
        "degraded_input": _summarize_validation_audits(degraded_audits),
        "stage_repair": {
            **_summarize_validation_gates(
                stage_repair_gates,
                accepted_key="accepted",
            ),
            "gate": "checkpoint_product_geometry_repair_gate_v1",
            "ignored_layers": ["long_horizon_root_drift"],
            "degraded_physical_comparison_is_diagnostic_only": True,
            "geometry_repair_gain": {
                "metric": "mean_absolute_product_log_to_clean",
                "minimum": geometry_gain_minimum,
                "configured_per_window": True,
                "observed": summarize_stage_detail(
                    "geometry_repair_gain"
                ),
            },
            "degraded_product_log_l1_to_clean": summarize_stage_detail(
                "degraded_product_log_l1_to_clean"
            ),
            "prediction_product_log_l1_to_clean": summarize_stage_detail(
                "prediction_product_log_l1_to_clean"
            ),
        },
        "temporal_repair": {
            **_summarize_validation_gates(
                temporal_repair_gates,
                accepted_key="accepted",
            ),
            "gate": "checkpoint_seam_fk_temporal_repair_gate_v1",
            "criteria": [
                "endpoint_velocity_continuity_gain",
                "seam_fk_velocity_acceleration_jerk_gain",
                "seam_fk_jerk_non_regression",
            ],
            "temporal_repair_gain": {
                "minimum": next(
                    (
                        gate.get("detail", {}).get(
                            "minimum_temporal_repair_gain"
                        )
                        for gate in temporal_repair_gates
                        if gate.get("detail", {}).get(
                            "minimum_temporal_repair_gain"
                        ) is not None
                    ),
                    None,
                ),
                "observed": summarize_gate_detail(
                    temporal_repair_gates,
                    "temporal_repair_gain",
                ),
            },
            "endpoint_repair_gain": {
                "minimum": next(
                    (
                        gate.get("detail", {}).get(
                            "minimum_endpoint_repair_gain"
                        )
                        for gate in temporal_repair_gates
                        if gate.get("detail", {}).get(
                            "minimum_endpoint_repair_gain"
                        ) is not None
                    ),
                    None,
                ),
                "observed": summarize_gate_detail(
                    temporal_repair_gates,
                    "endpoint_repair_gain",
                ),
            },
        },
        "clean_reference_fidelity": {
            **_summarize_validation_gates(
                list(accumulator.get("clean_fidelity_gates", [])),
                accepted_key="accepted",
            ),
            "gate": (
                "contracts.physical_quality."
                "evaluate_stage_reference_fidelity"
            ),
            "checkpoint_criterion": False,
            "diagnostic_note": (
                "full-window all-metric conjunction is diagnostic; checkpoint "
                "publication uses separate local geometry and seam temporal gates"
            ),
        },
        "clean_input_identity": {
            **_summarize_validation_gates(
                list(accumulator.get("clean_identity_gates", [])),
                accepted_key="accepted",
            ),
            "gate": (
                "contracts.physical_quality."
                "evaluate_stage_reference_fidelity"
            ),
            "input_role": "authentic_clean_motion",
            "prediction": _summarize_validation_audits(
                list(
                    accumulator.get(
                        "clean_identity_prediction_audits", []
                    )
                )
            ),
            "product_log_l1": {
                "mean": (
                    float(np.mean(clean_identity_errors))
                    if clean_identity_errors.size
                    else None
                ),
                "maximum": (
                    float(np.max(clean_identity_errors))
                    if clean_identity_errors.size
                    else None
                ),
            },
            "contact_l1": {
                "mean": (
                    float(np.mean(clean_identity_contact_errors))
                    if clean_identity_contact_errors.size
                    else None
                ),
                "maximum": (
                    float(np.max(clean_identity_contact_errors))
                    if clean_identity_contact_errors.size
                    else None
                ),
            },
        },
        "clean_physical_non_regression": {
            **_summarize_validation_gates(
                list(accumulator.get("clean_non_regression_gates", [])),
                accepted_key="accepted",
            ),
            "gate": "contracts.physical_quality.evaluate_stage_candidate",
            "reference": "authentic_clean_motion",
            "ignored_layers": ["long_horizon_root_drift"],
            "absolute_final_gate_used": False,
            "checkpoint_criterion": False,
            "diagnostic_note": (
                "strict multi-metric stage comparison of repaired versus clean; "
                "local geometry and seam FK temporal repair are the checkpoint "
                "criteria"
            ),
        },
        "final_generation_gate_diagnostic": {
            "prediction": _summarize_validation_gates(
                list(
                    accumulator.get(
                        "prediction_final_diagnostic_gates", []
                    )
                ),
                accepted_key="ok",
            ),
            "clean_target": _summarize_validation_gates(
                list(accumulator.get("clean_final_diagnostic_gates", [])),
                accepted_key="ok",
            ),
            "gate": "contracts.physical_quality.evaluate_physical_audit",
            "checkpoint_criterion": False,
            "authoritative_after": [
                "diffusion",
                "boundary_closed_loop",
                "lower_body_ik",
            ],
        },
        "support_policy": "source_observation",
        "aggregation_note": (
            "geometry and seam temporal repair are separate deterministic "
            "validation gates; full-window fidelity and final-generation "
            "absolute pass rate are diagnostic only"
        ),
    }


def _evaluate_refiner_validation(
    model: Any,
    validation_db: Dict[str, Any],
    train_db: Dict[str, Any],
    cfg: MotionGenerationConfig,
    device: Any,
    *,
    maximum_windows: int = 16,
) -> Dict[str, Any]:
    indices = _validation_indices(len(validation_db["paths"]), maximum=maximum_windows)
    cond_z = _descriptor_values_in_training_coordinates(validation_db, train_db)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(int(cfg.seed) + 45001)
    np.random.seed(int(cfg.seed) + 45001)
    rec_values: List[float] = []
    velocity_values: List[float] = []
    physical = _new_validation_physical_accumulator()
    window_details = []
    gpu_preprocessing = _gpu_preprocessing_enabled(cfg, device)
    model.eval()
    try:
        with torch.no_grad():
            for idx in indices:
                clean = load_motion_window(
                    validation_db["paths"][idx],
                    cfg.window_len,
                    cfg,
                    random_crop=False,
                )
                bad, seam = degrade_for_refiner(
                    clean,
                    cfg=cfg,
                    finalize_contract=not gpu_preprocessing,
                )
                clean_t = torch.from_numpy(clean[None]).float().to(device)
                bad_t = torch.from_numpy(bad[None]).float().to(device)
                seam_t = torch.from_numpy(seam[None]).float().to(device)
                cond_t = torch.from_numpy(cond_z[idx][None]).float().to(device)
                if gpu_preprocessing:
                    bad_t = _enforce_internal_batch_contract_torch(
                        bad_t,
                        cfg,
                    )
                    pair_motion = torch.cat([bad_t, clean_t], dim=0)
                    pair_seam = torch.cat([seam_t, seam_t], dim=0)
                    pair_joint, pair_root, pair_contact = (
                        _risk_masks_for_batch_torch(
                            pair_motion,
                            pair_seam,
                            cfg,
                        )
                    )
                    joint_t, clean_joint_t = pair_joint.split(1, dim=0)
                    root_t, clean_root_t = pair_root.split(1, dim=0)
                    contact_t, clean_contact_t = pair_contact.split(1, dim=0)
                else:
                    pair_np = np.concatenate([bad[None], clean[None]], axis=0)
                    pair_seam_np = np.concatenate(
                        [seam[None], seam[None]], axis=0
                    )
                    pair_joint_np, pair_root_np, pair_contact_np = (
                        _risk_masks_for_batch_np(
                            pair_np, pair_seam_np, cfg
                        )
                    )
                    joint_t = torch.from_numpy(pair_joint_np[:1]).float().to(device)
                    clean_joint_t = torch.from_numpy(pair_joint_np[1:]).float().to(device)
                    root_t = torch.from_numpy(pair_root_np[:1]).float().to(device)
                    clean_root_t = torch.from_numpy(pair_root_np[1:]).float().to(device)
                    contact_t = torch.from_numpy(pair_contact_np[:1]).float().to(device)
                    clean_contact_t = torch.from_numpy(pair_contact_np[1:]).float().to(device)
                    pair_motion = torch.cat([bad_t, clean_t], dim=0)
                    pair_seam = torch.cat([seam_t, seam_t], dim=0)
                pair_output = model(
                    pair_motion,
                    torch.cat([cond_t, cond_t], dim=0),
                    pair_seam,
                    torch.cat([joint_t, clean_joint_t], dim=0),
                )
                output, clean_output = pair_output.split(1, dim=0)
                pred = _decode_product_refiner_output(
                    bad_t, output, joint_t, root_t, contact_t, cfg
                )
                clean_identity_pred = _decode_product_refiner_output(
                    clean_t,
                    clean_output,
                    clean_joint_t,
                    clean_root_t,
                    clean_contact_t,
                    cfg,
                )
                rec_values.append(
                    float(product_log_torch(pred, clean_t).abs().mean().cpu())
                )
                velocity_values.append(
                    float(
                        F.smooth_l1_loss(
                            product_log_torch(pred[:, :-1], pred[:, 1:]),
                            product_log_torch(clean_t[:, :-1], clean_t[:, 1:]),
                        ).cpu()
                    )
                )
                _record_validation_physical_prediction(
                    physical,
                    pred[0].detach().cpu().numpy(),
                    clean,
                    cfg,
                    degraded=bad_t[0].detach().cpu().numpy(),
                    seam_mask=seam,
                )
                _record_validation_clean_identity_prediction(
                    physical,
                    clean_identity_pred[0].detach().cpu().numpy(),
                    clean,
                    cfg,
                )
                window_details.append({
                    "event_index": int(idx),
                    "path": str(validation_db["paths"][idx]),
                    "source_uid": str(validation_db.get("source_uids", validation_db["paths"])[idx]),
                    "seam_core_frames": np.flatnonzero(seam[:, 0] >= 0.5).tolist(),
                    "geometry": physical["stage_repair_gates"][-1],
                    "temporal": physical["temporal_repair_gates"][-1],
                    "clean_identity": physical["clean_identity_gates"][-1],
                })
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.train()
    return {
        "num_windows": len(indices),
        "reconstruction_product_log_l1": (
            float(np.mean(rec_values)) if rec_values else None
        ),
        "velocity_smooth_l1_per_frame": (
            float(np.mean(velocity_values)) if velocity_values else None
        ),
        "representation": "product_manifold_79d",
        "descriptor_coordinates": "training_event_db",
        "physical_quality": _summarize_validation_physical_metrics(physical),
        "windows": window_details,
    }


def _evaluate_diffusion_validation(
    model: Any,
    validation_db: Dict[str, Any],
    train_db: Dict[str, Any],
    cfg: MotionGenerationConfig,
    device: Any,
    abar: Any,
    diffusion_steps: int,
) -> Dict[str, Any]:
    indices = _validation_indices(len(validation_db["paths"]))
    cond_z = _descriptor_values_in_training_coordinates(validation_db, train_db)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(int(cfg.seed) + 46001)
    np.random.seed(int(cfg.seed) + 46001)
    torch.manual_seed(int(cfg.seed) + 46001)
    noise_values: List[float] = []
    velocity_values: List[float] = []
    physical = _new_validation_physical_accumulator()
    gpu_preprocessing = _gpu_preprocessing_enabled(cfg, device)
    model.eval()
    try:
        with torch.no_grad():
            for sample_index, idx in enumerate(indices):
                clean = load_motion_window(
                    validation_db["paths"][idx],
                    cfg.window_len,
                    cfg,
                    random_crop=False,
                )
                retrieval, seam = degrade_for_refiner(
                    clean,
                    severity=0.045,
                    cfg=cfg,
                    finalize_contract=not gpu_preprocessing,
                )
                x0 = torch.from_numpy(clean[None]).float().to(device)
                retr = torch.from_numpy(retrieval[None]).float().to(device)
                seam_t = torch.from_numpy(seam[None]).float().to(device)
                cond_t = torch.from_numpy(cond_z[idx][None]).float().to(device)
                timestep = int(
                    round(
                        sample_index
                        * max(diffusion_steps - 1, 0)
                        / max(len(indices) - 1, 1)
                    )
                )
                t = torch.full((1,), timestep, dtype=torch.long, device=device)
                a = abar[t].view(1, 1, 1)
                if gpu_preprocessing:
                    retr = _enforce_internal_batch_contract_torch(
                        retr,
                        cfg,
                    )
                    joint_t, root_t, contact_t = _risk_masks_for_batch_torch(
                        retr,
                        seam_t,
                        cfg,
                    )
                else:
                    joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                        retrieval[None], seam[None], cfg
                    )
                    joint_t = torch.from_numpy(joint_np).float().to(device)
                    root_t = torch.from_numpy(root_np).float().to(device)
                    contact_t = torch.from_numpy(contact_np).float().to(device)
                state0 = _encode_reference_tangent_state(retr, x0)
                state_mask = _tangent_state_mask(
                    joint_t, root_t, contact_t
                )
                active_mask = (state_mask > 0.0).to(state0.dtype)
                noise = torch.randn_like(state0) * active_mask
                x_t = (
                    torch.sqrt(a) * state0
                    + torch.sqrt(1.0 - a) * noise
                ) * active_mask
                pred_noise = model(
                    x_t, retr, cond_t, seam_t, joint_t, t
                ) * active_mask
                denominator = state_mask.sum().clamp_min(1.0)
                noise_error = ((pred_noise - noise) ** 2 * state_mask).sum()
                noise_values.append(float((noise_error / denominator).cpu()))
                state0_hat = (
                    x_t - torch.sqrt(1.0 - a) * pred_noise
                ) / torch.sqrt(a).clamp_min(1.0e-6)
                decoded = _decode_reference_tangent_state(
                    retr,
                    state0_hat,
                    joint_t,
                    root_t,
                    contact_t,
                    cfg,
                )
                velocity_values.append(
                    float(
                        F.smooth_l1_loss(
                            product_log_torch(decoded[:, :-1], decoded[:, 1:]),
                            product_log_torch(x0[:, :-1], x0[:, 1:]),
                        ).cpu()
                    )
                )
                _record_validation_physical_prediction(
                    physical,
                    decoded[0].detach().cpu().numpy(),
                    clean,
                    cfg,
                    degraded=retr[0].detach().cpu().numpy(),
                    seam_mask=seam,
                )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        model.train()
    return {
        "num_windows": len(indices),
        "noise_mse": float(np.mean(noise_values)) if noise_values else None,
        "velocity_smooth_l1_per_frame": (
            float(np.mean(velocity_values)) if velocity_values else None
        ),
        "representation": "reference_tangent_product_manifold_79d",
        "descriptor_coordinates": "training_event_db",
        "physical_quality": _summarize_validation_physical_metrics(physical),
    }


def _emit_training_progress(
    label: str,
    step: int,
    total_steps: int,
    started_at: float,
    **metrics: float,
) -> None:
    """Emit an unbuffered progress line with measured throughput and ETA."""
    completed = step + 1
    elapsed_seconds = max(time.perf_counter() - started_at, 1.0e-6)
    steps_per_second = completed / elapsed_seconds
    remaining_seconds = max(total_steps - completed, 0) / max(
        steps_per_second, 1.0e-9
    )
    metric_text = " ".join(
        f"{name}={float(value):.6f}" for name, value in metrics.items()
    )
    print(
        f"{label} step={completed}/{total_steps} {metric_text} "
        f"steps_per_second={steps_per_second:.4f} "
        f"elapsed_minutes={elapsed_seconds / 60.0:.1f} "
        f"eta_minutes={remaining_seconds / 60.0:.1f}",
        flush=True,
    )


def _checkpoint_validation_decision(
    metrics: Mapping[str, Any],
    cfg: MotionGenerationConfig,
    *,
    stage: str,
) -> Dict[str, Any]:
    """Decide whether a neural-stage checkpoint may become a formal asset.

    This deliberately does not consume the strict final-generation diagnostic.
    Refiner and diffusion checkpoints are accepted on two independent local
    contracts: product-geometry repair and seam FK temporal repair.  The
    full-window all-metric clean comparison is retained as a diagnostic because
    it is an overly strict conjunction for a deliberately edited transition.
    The absolute whole-song physical gate remains downstream of repair, IK and
    closed-loop composition.
    """

    physical = metrics.get("physical_quality", {})
    stage_repair = physical.get("stage_repair", {})
    temporal_repair = physical.get("temporal_repair", {})
    clean_fidelity = physical.get("clean_reference_fidelity", {})
    clean_non_regression = physical.get(
        "clean_physical_non_regression", {}
    )
    clean_identity = physical.get("clean_input_identity", {})
    observed: Dict[str, Optional[float]] = {
        "num_windows": physical.get("num_windows"),
        "stage_repair_rate": stage_repair.get("pass_rate"),
        "temporal_repair_rate": temporal_repair.get("pass_rate"),
        "clean_fidelity_rate": clean_fidelity.get("pass_rate"),
        "clean_non_regression_rate": clean_non_regression.get("pass_rate"),
        "clean_identity_rate": clean_identity.get("pass_rate"),
        "fk_position_error_m_p95": physical.get("fk_position_error_m_p95"),
        "fk_position_error_m_max": physical.get("fk_position_error_m_max"),
    }
    thresholds: Dict[str, float] = {
        "min_stage_repair_rate": float(
            cfg.checkpoint_validation_min_stage_repair_rate
        ),
        "min_temporal_repair_rate": float(
            cfg.checkpoint_validation_min_temporal_repair_rate
        ),
        "min_clean_identity_rate": float(
            cfg.checkpoint_validation_min_clean_identity_rate
        ),
        "max_fk_position_error_m_p95": float(
            cfg.checkpoint_validation_max_fk_p95_m
        ),
        "max_fk_position_error_m_max": float(
            cfg.checkpoint_validation_max_fk_max_m
        ),
    }
    reasons: List[str] = []

    def _require_min(name: str, minimum: float) -> None:
        raw = observed.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            reasons.append(f"missing_or_nonfinite:{name}")
            return
        if not np.isfinite(value):
            reasons.append(f"missing_or_nonfinite:{name}")
        elif value < float(minimum):
            reasons.append(f"{name}_too_low")

    def _require_max(name: str, maximum: float) -> None:
        raw = observed.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            reasons.append(f"missing_or_nonfinite:{name}")
            return
        if not np.isfinite(value):
            reasons.append(f"missing_or_nonfinite:{name}")
        elif value > float(maximum):
            reasons.append(f"{name}_too_high")

    try:
        num_windows = int(observed["num_windows"])
    except (TypeError, ValueError):
        num_windows = 0
    if num_windows <= 0:
        reasons.append("validation_windows_missing")
    _require_min(
        "stage_repair_rate",
        thresholds["min_stage_repair_rate"],
    )
    _require_min(
        "temporal_repair_rate",
        thresholds["min_temporal_repair_rate"],
    )
    _require_max(
        "fk_position_error_m_p95",
        thresholds["max_fk_position_error_m_p95"],
    )
    _require_max(
        "fk_position_error_m_max",
        thresholds["max_fk_position_error_m_max"],
    )

    if stage == "refiner":
        _require_min(
            "clean_identity_rate",
            thresholds["min_clean_identity_rate"],
        )
        metric_name = "reconstruction_product_log_l1"
        observed[metric_name] = metrics.get(metric_name)
        thresholds["max_reconstruction_product_log_l1"] = float(
            cfg.checkpoint_validation_max_refiner_product_log_l1
        )
        _require_max(
            metric_name,
            thresholds["max_reconstruction_product_log_l1"],
        )
    elif stage != "diffusion":
        reasons.append(f"unsupported_checkpoint_stage:{stage}")

    scientific_acceptance = not reasons
    fail_closed = bool(cfg.checkpoint_validation_fail_closed)
    return {
        "schema": "motion_checkpoint_publication_decision_v2",
        "stage": str(stage),
        "scientific_acceptance": bool(scientific_acceptance),
        "publish_allowed": bool(scientific_acceptance or not fail_closed),
        "fail_closed_enforced": fail_closed,
        "reasons": reasons,
        "thresholds": thresholds,
        "observed": observed,
        "final_generation_gate_used": False,
        "clean_non_regression_diagnostic_only": True,
        "clean_reference_fidelity_diagnostic_only": True,
        "final_generation_gate_role": (
            "diagnostic_only_until_diffusion_boundary_ik_closed_loop"
        ),
    }


def _save_checkpoint_after_validation(
    payload: Mapping[str, Any],
    requested_path: Path,
    decision: Mapping[str, Any],
) -> Tuple[Path, bool]:
    """Publish an accepted checkpoint, or quarantine a rejected candidate."""

    publish = bool(decision.get("publish_allowed", False))
    actual_path = requested_path
    if not publish:
        suffix = requested_path.suffix or ".pt"
        actual_path = requested_path.with_name(
            f"{requested_path.stem}.rejected_validation{suffix}"
        )
    actual_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), actual_path)
    return actual_path, publish


TRAINING_RESUME_SNAPSHOT_SCHEMA = "dunhuang_motion_training_resume_v1"
REFINER_BEST_VALIDATION_SCHEMA = "dunhuang_refiner_best_validation_v1"


def _training_stage_contract(stage: str) -> Tuple[str, str]:
    if stage == "refiner":
        return "boundary_refiner", REFINER_MODEL_VERSION
    if stage == "diffusion":
        return "motion_diffusion", "reference_tangent_motion_diffusion_v2"
    raise ValueError(f"unsupported training snapshot stage: {stage!r}")


def _training_code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _training_config_sha256(
    cfg: MotionGenerationConfig,
    *,
    stage: str,
    diffusion_steps: Optional[int] = None,
) -> str:
    _, model_version = _training_stage_contract(stage)
    training_config = dataclasses.asdict(cfg)
    # Recovery frequency changes only durability, never the optimization or
    # scientific model contract, so it must not invalidate an otherwise exact
    # resume snapshot.
    training_config.pop("training_snapshot_interval_steps", None)
    payload = {
        "stage": str(stage),
        "model_version": model_version,
        "config": training_config,
        "diffusion_steps": (
            None if diffusion_steps is None else int(diffusion_steps)
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_training_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_training_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(
            f"training resume snapshot is missing RNG fields: {missing}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "training resume snapshot contains CUDA RNG state but CUDA "
                "is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "training resume snapshot CUDA device-count mismatch: "
                f"snapshot={len(cuda_states)}, runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}"
    )
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite_score_value(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _refiner_validation_score(
    metrics: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Tuple[float, ...]:
    """Rank deterministic validation candidates without hiding a failed gate."""

    physical = metrics.get("physical_quality", {})
    geometry_rate = _finite_score_value(
        physical.get("stage_repair", {}).get("pass_rate"),
        default=-1.0,
    )
    temporal_rate = _finite_score_value(
        physical.get("temporal_repair", {}).get("pass_rate"),
        default=-1.0,
    )
    identity_rate = _finite_score_value(
        physical.get("clean_input_identity", {}).get("pass_rate"),
        default=-1.0,
    )
    reconstruction = _finite_score_value(
        metrics.get("reconstruction_product_log_l1"),
        default=float("inf"),
    )
    fk_p95 = _finite_score_value(
        physical.get("fk_position_error_m_p95"),
        default=float("inf"),
    )
    return (
        float(bool(decision.get("scientific_acceptance", False))),
        min(geometry_rate, temporal_rate, identity_rate),
        geometry_rate + temporal_rate + identity_rate,
        -reconstruction,
        -fk_p95,
    )


def _save_refiner_best_validation_candidate(
    path: Path,
    *,
    model: Any,
    completed_steps: int,
    metrics: Mapping[str, Any],
    decision: Mapping[str, Any],
    cfg: MotionGenerationConfig,
    training_contract: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    score = _refiner_validation_score(metrics, decision)
    payload = {
        "schema": REFINER_BEST_VALIDATION_SCHEMA,
        "formal_checkpoint": False,
        "publication_state": "validated_training_candidate_not_formal",
        "model_version": REFINER_MODEL_VERSION,
        "completed_steps": int(completed_steps),
        "score": list(score),
        "metrics": dict(metrics),
        "checkpoint_decision": dict(decision),
        "motion_contract": motion_checkpoint_contract(cfg, "boundary_refiner"),
        "training_event_db_contract": dict(
            training_contract["event_db_contract"]
        ),
        "validation_event_db_contract": dict(
            validation_contract["event_db_contract"]
        ),
        "training_config_sha256": _training_config_sha256(
            cfg,
            stage="refiner",
        ),
        "code_revision": _training_code_revision(),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
    }
    _atomic_torch_save(payload, path)
    print(
        json.dumps(
            {
                "stage": "refiner_best_validation_candidate",
                "path": str(path),
                "completed_steps": int(completed_steps),
                "score": list(score),
                "scientific_acceptance": bool(
                    decision.get("scientific_acceptance", False)
                ),
                "formal_checkpoint": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


def _load_refiner_best_validation_candidate(
    path: Path,
    *,
    cfg: MotionGenerationConfig,
    training_contract: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = _trusted_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"invalid Refiner best-validation payload: {path}")
    if payload.get("schema") != REFINER_BEST_VALIDATION_SCHEMA:
        raise RuntimeError(f"Refiner best-validation schema mismatch: {path}")
    if payload.get("model_version") != REFINER_MODEL_VERSION:
        raise RuntimeError(f"Refiner best-validation model mismatch: {path}")
    if bool(payload.get("formal_checkpoint", True)):
        raise RuntimeError(f"Refiner best-validation candidate is marked formal: {path}")
    assert_motion_checkpoint_contract(
        dict(payload),
        cfg,
        path,
        "boundary_refiner",
    )
    assert_same_event_db_contract(
        training_contract["event_db_contract"],
        payload.get("training_event_db_contract"),
        context="Refiner best candidate training Event-DB",
    )
    assert_same_event_db_contract(
        validation_contract["event_db_contract"],
        payload.get("validation_event_db_contract"),
        context="Refiner best candidate validation Event-DB",
    )
    expected_config = _training_config_sha256(cfg, stage="refiner")
    if payload.get("training_config_sha256") != expected_config:
        raise RuntimeError(
            f"Refiner best-validation training configuration mismatch: {path}"
        )
    snapshot_revision = str(payload.get("code_revision", "unknown"))
    runtime_revision = _training_code_revision()
    if (
        snapshot_revision != "unknown"
        and runtime_revision != "unknown"
        and snapshot_revision != runtime_revision
    ):
        raise RuntimeError(
            "Refiner best-validation code revision mismatch for "
            f"{path}: candidate={snapshot_revision}, runtime={runtime_revision}"
        )
    return dict(payload)


def _save_training_resume_snapshot(
    path: Path,
    *,
    stage: str,
    model: Any,
    optimizer: Any,
    completed_steps: int,
    target_steps: int,
    elapsed_seconds: float,
    cfg: MotionGenerationConfig,
    training_contract: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
    diffusion_steps: Optional[int] = None,
) -> Dict[str, Any]:
    role, model_version = _training_stage_contract(stage)
    if completed_steps < 0 or completed_steps > target_steps:
        raise ValueError(
            "completed training steps must be within the target: "
            f"{completed_steps}/{target_steps}"
        )
    payload: Dict[str, Any] = {
        "schema": TRAINING_RESUME_SNAPSHOT_SCHEMA,
        "stage": str(stage),
        "formal_checkpoint": False,
        "publication_state": "resume_only_not_validated",
        "model_version": model_version,
        "completed_steps": int(completed_steps),
        "target_steps": int(target_steps),
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "diffusion_steps": (
            None if diffusion_steps is None else int(diffusion_steps)
        ),
        "motion_contract": motion_checkpoint_contract(cfg, role),
        "training_event_db_contract": dict(
            training_contract["event_db_contract"]
        ),
        "validation_event_db_contract": dict(
            validation_contract["event_db_contract"]
        ),
        "training_config_sha256": _training_config_sha256(
            cfg,
            stage=stage,
            diffusion_steps=diffusion_steps,
        ),
        "code_revision": _training_code_revision(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _capture_training_rng_state(),
        "checkpoint_decision": {
            "publish_allowed": False,
            "reason": "training_resume_snapshot_has_not_passed_validation",
        },
    }
    _atomic_torch_save(payload, path)
    print(
        json.dumps(
            {
                "stage": f"{stage}_training_snapshot",
                "path": str(path),
                "completed_steps": int(completed_steps),
                "target_steps": int(target_steps),
                "formal_checkpoint": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return payload


def _load_training_resume_snapshot(
    path: Path,
    *,
    stage: str,
    model: Any,
    optimizer: Any,
    target_steps: int,
    cfg: MotionGenerationConfig,
    training_contract: Mapping[str, Any],
    validation_contract: Mapping[str, Any],
    device: Any,
    diffusion_steps: Optional[int] = None,
) -> Tuple[int, float, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"training resume snapshot missing: {path}")
    snapshot = _trusted_torch_load(path, map_location=device)
    if not isinstance(snapshot, Mapping):
        raise RuntimeError(f"invalid training resume snapshot payload: {path}")
    if snapshot.get("schema") != TRAINING_RESUME_SNAPSHOT_SCHEMA:
        raise RuntimeError(
            f"training resume snapshot schema mismatch for {path}: "
            f"{snapshot.get('schema')!r}"
        )
    if bool(snapshot.get("formal_checkpoint", True)):
        raise RuntimeError(
            f"training resume snapshot {path} is not marked resume-only"
        )
    role, model_version = _training_stage_contract(stage)
    if snapshot.get("stage") != stage:
        raise RuntimeError(
            f"training resume snapshot stage mismatch for {path}: "
            f"snapshot={snapshot.get('stage')!r}, runtime={stage!r}"
        )
    if snapshot.get("model_version") != model_version:
        raise RuntimeError(
            f"training resume snapshot model mismatch for {path}: "
            f"snapshot={snapshot.get('model_version')!r}, "
            f"runtime={model_version!r}"
        )
    assert_motion_checkpoint_contract(snapshot, cfg, path, role)
    assert_same_event_db_contract(
        training_contract["event_db_contract"],
        snapshot.get("training_event_db_contract"),
        context=f"{stage} resume training Event-DB",
    )
    assert_same_event_db_contract(
        validation_contract["event_db_contract"],
        snapshot.get("validation_event_db_contract"),
        context=f"{stage} resume validation Event-DB",
    )
    expected_config = _training_config_sha256(
        cfg,
        stage=stage,
        diffusion_steps=diffusion_steps,
    )
    if snapshot.get("training_config_sha256") != expected_config:
        raise RuntimeError(
            f"{stage} resume training configuration mismatch for {path}"
        )
    snapshot_diffusion_steps = snapshot.get("diffusion_steps")
    expected_diffusion_steps = (
        None if diffusion_steps is None else int(diffusion_steps)
    )
    if snapshot_diffusion_steps != expected_diffusion_steps:
        raise RuntimeError(
            f"{stage} resume diffusion schedule mismatch for {path}: "
            f"snapshot={snapshot_diffusion_steps!r}, "
            f"runtime={expected_diffusion_steps!r}"
        )
    completed_steps = int(snapshot.get("completed_steps", -1))
    if completed_steps < 0 or completed_steps > int(target_steps):
        raise RuntimeError(
            f"{stage} resume step mismatch for {path}: "
            f"completed={completed_steps}, target={target_steps}"
        )
    if int(snapshot.get("target_steps", -1)) != int(target_steps):
        raise RuntimeError(
            f"{stage} resume target-step mismatch for {path}: "
            f"snapshot={snapshot.get('target_steps')!r}, "
            f"runtime={target_steps!r}"
        )
    snapshot_revision = str(snapshot.get("code_revision", "unknown"))
    runtime_revision = _training_code_revision()
    if (
        snapshot_revision != "unknown"
        and runtime_revision != "unknown"
        and snapshot_revision != runtime_revision
    ):
        raise RuntimeError(
            f"{stage} resume code revision mismatch for {path}: "
            f"snapshot={snapshot_revision}, runtime={runtime_revision}"
        )
    try:
        model.load_state_dict(snapshot["model_state_dict"], strict=True)
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        _restore_training_rng_state(snapshot["rng_state"])
    except KeyError as exc:
        raise RuntimeError(
            f"{stage} resume snapshot {path} is missing {exc.args[0]!r}"
        ) from exc
    elapsed_seconds = max(0.0, float(snapshot.get("elapsed_seconds", 0.0)))
    resume_info = {
        "resumed": True,
        "snapshot_path": str(path),
        "completed_steps": completed_steps,
        "snapshot_code_revision": snapshot_revision,
        "runtime_code_revision": runtime_revision,
        "cross_revision": snapshot_revision != runtime_revision,
    }
    print(
        json.dumps(
            {
                "stage": f"{stage}_training_resume",
                **resume_info,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return completed_steps, elapsed_seconds, resume_info


def _resolve_training_snapshot_options(
    args: argparse.Namespace,
    cfg: MotionGenerationConfig,
    requested_checkpoint: Path,
) -> Tuple[Path, Optional[Path], int]:
    resume_value = getattr(args, "resume_snapshot", None)
    output_value = getattr(args, "snapshot_path", None)
    resume_path = Path(resume_value).expanduser() if resume_value else None
    if output_value:
        snapshot_path = Path(output_value).expanduser()
    elif resume_path is not None:
        snapshot_path = resume_path
    else:
        suffix = requested_checkpoint.suffix or ".pt"
        snapshot_path = requested_checkpoint.with_name(
            f"{requested_checkpoint.stem}.training_snapshot{suffix}"
        )
    if (
        resume_path is not None
        and resume_path.resolve() != snapshot_path.resolve()
    ):
        raise ValueError(
            "--resume_snapshot and --snapshot_path must identify the same "
            "recovery file"
        )
    raw_interval = getattr(args, "snapshot_every", None)
    interval = int(
        cfg.training_snapshot_interval_steps
        if raw_interval is None
        else raw_interval
    )
    if interval < 1:
        raise ValueError("training snapshot interval must be at least one step")
    if resume_path is None and snapshot_path.exists():
        raise RuntimeError(
            f"training snapshot already exists: {snapshot_path}. Pass "
            "--resume_snapshot to continue it, or archive it before a fresh run."
        )
    return snapshot_path, resume_path, interval


def train_refiner(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for Motion Refiner training.")
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    db = load_db(args.db)
    database_contract = _training_db_contract(db, cfg, "Motion Refiner training")
    paths = db["paths"]
    desc_z = _descriptor_values_in_training_coordinates(db, db)
    validation_db = load_db(args.val_db)
    validation_contract = _training_db_contract(
        validation_db, cfg, "Motion Refiner validation"
    )
    validation_report: Dict[str, Any] = {
        "enabled": True,
        "database": validation_contract,
        "source_disjoint": _validate_source_disjoint(db, validation_db),
    }
    device = torch.device(cfg.device)
    model = ProductManifoldTemporalRefiner(EDGE_DIM, 32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = int(args.steps or cfg.refiner_train_steps)
    if steps < 1:
        raise ValueError("Motion Refiner training requires at least one step")
    stop_after = getattr(args, "stop_after_steps", None)
    stop_after = None if stop_after is None else int(stop_after)
    if stop_after is not None and not 0 < stop_after < steps:
        raise ValueError("--stop_after_steps must be between 1 and target steps - 1")
    probe_windows = int(getattr(args, "train_probe_windows", 8))
    if not 0 <= probe_windows <= 16:
        raise ValueError("--train_probe_windows must be within [0,16]")
    out = Path(args.out)
    snapshot_path, resume_path, snapshot_interval = (
        _resolve_training_snapshot_options(args, cfg, out)
    )
    validation_interval = int(
        cfg.training_validation_interval_steps
        if getattr(args, "validation_every", None) is None
        else args.validation_every
    )
    if validation_interval < 1:
        raise ValueError("training validation interval must be at least one step")
    suffix = out.suffix or ".pt"
    best_validation_path = out.with_name(
        f"{out.stem}.best_validation{suffix}"
    )
    if resume_path is None and best_validation_path.exists():
        raise RuntimeError(
            f"Refiner best-validation candidate already exists: "
            f"{best_validation_path}. Archive it before a fresh run."
        )
    start_step = 0
    resumed_elapsed_seconds = 0.0
    resume_info: Dict[str, Any] = {
        "resumed": False,
        "snapshot_path": None,
        "completed_steps": 0,
        "snapshot_code_revision": None,
        "runtime_code_revision": _training_code_revision(),
        "cross_revision": False,
    }
    if resume_path is not None:
        start_step, resumed_elapsed_seconds, resume_info = (
            _load_training_resume_snapshot(
                resume_path,
                stage="refiner",
                model=model,
                optimizer=opt,
                target_steps=steps,
                cfg=cfg,
                training_contract=database_contract,
                validation_contract=validation_contract,
                device=device,
            )
        )
    if stop_after is not None and stop_after <= start_step:
        raise ValueError("--stop_after_steps must be after the resumed step")
    best_validation: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[float, ...]] = None
    if resume_path is not None and best_validation_path.exists():
        best_validation = _load_refiner_best_validation_candidate(
            best_validation_path,
            cfg=cfg,
            training_contract=database_contract,
            validation_contract=validation_contract,
        )
        best_score = tuple(float(value) for value in best_validation["score"])
    validation_history: List[Dict[str, Any]] = []
    bs = min(cfg.batch_size, max(2, len(paths)))
    motion_pool = PreloadedMotionWindowPool.preload(
        paths,
        cfg.window_len,
        cfg,
        mode="refiner",
    )
    gpu_preprocessing = _gpu_preprocessing_enabled(cfg, device)
    risk_executor = None
    if gpu_preprocessing:
        print(
            json.dumps(
                {
                    "stage": "risk_mask_batch",
                    "backend": "torch_cuda",
                    "batch_size": int(bs),
                    "device": str(device),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        risk_executor, _ = _training_risk_mask_executor(bs)
    training_started_at = time.perf_counter() - resumed_elapsed_seconds
    for step in range(start_step, steps):
        clean_batch = []
        bad_batch = []
        seam_batch = []
        cond_batch = []
        for _ in range(bs):
            idx = random.randrange(len(paths))
            clean = motion_pool.sample(idx)
            bad, seam = degrade_for_refiner(
                clean,
                cfg=cfg,
                finalize_contract=not gpu_preprocessing,
            )
            clean_batch.append(clean)
            bad_batch.append(bad)
            seam_batch.append(seam)
            cond_batch.append(desc_z[idx])
        clean_np = np.stack(clean_batch)
        bad_np = np.stack(bad_batch)
        seam_np = np.stack(seam_batch)
        clean_t = torch.from_numpy(clean_np).float().to(device)
        bad_t = torch.from_numpy(bad_np).float().to(device)
        seam_t = torch.from_numpy(seam_np).float().to(device)
        cond_t = torch.from_numpy(np.stack(cond_batch)).float().to(device)
        if gpu_preprocessing:
            bad_t = _enforce_internal_batch_contract_torch(
                bad_t,
                cfg,
            )
            pair_motion = torch.cat([bad_t, clean_t], dim=0)
            pair_seam = torch.cat([seam_t, seam_t], dim=0)
            pair_joint, pair_root, pair_contact = _risk_masks_for_batch_torch(
                pair_motion,
                pair_seam,
                cfg,
            )
            joint_t, clean_joint_t = pair_joint.split(bs, dim=0)
            root_t, clean_root_t = pair_root.split(bs, dim=0)
            contact_t, clean_contact_t = pair_contact.split(bs, dim=0)
        else:
            pair_np = np.concatenate([bad_np, clean_np], axis=0)
            pair_seam_np = np.concatenate([seam_np, seam_np], axis=0)
            pair_joint_np, pair_root_np, pair_contact_np = _risk_masks_for_batch_np(
                pair_np,
                pair_seam_np,
                cfg,
                executor=risk_executor,
            )
            joint_t = torch.from_numpy(pair_joint_np[:bs]).float().to(device)
            clean_joint_t = torch.from_numpy(pair_joint_np[bs:]).float().to(device)
            root_t = torch.from_numpy(pair_root_np[:bs]).float().to(device)
            clean_root_t = torch.from_numpy(pair_root_np[bs:]).float().to(device)
            contact_t = torch.from_numpy(pair_contact_np[:bs]).float().to(device)
            clean_contact_t = torch.from_numpy(pair_contact_np[bs:]).float().to(device)
            pair_motion = torch.cat([bad_t, clean_t], dim=0)
            pair_seam = torch.cat([seam_t, seam_t], dim=0)
        pair_output = model(
            pair_motion,
            torch.cat([cond_t, cond_t], dim=0),
            pair_seam,
            torch.cat([joint_t, clean_joint_t], dim=0),
        )
        output, clean_output = pair_output.split(bs, dim=0)
        pred = _decode_product_refiner_output(
            bad_t, output, joint_t, root_t, contact_t, cfg
        )
        clean_identity_pred = _decode_product_refiner_output(
            clean_t,
            clean_output,
            clean_joint_t,
            clean_root_t,
            clean_contact_t,
            cfg,
        )
        repair_loss, loss_terms = _product_motion_losses(
            pred,
            clean_t,
            bad_t,
            joint_t,
            root_t,
            contact_t,
            cfg,
            seam_mask=seam_t,
        )
        identity_loss, identity_terms = _product_refiner_clean_identity_loss(
            clean_identity_pred,
            clean_t,
            clean_joint_t,
            clean_root_t,
            clean_contact_t,
            cfg,
        )
        loss = repair_loss + float(
            cfg.product_refiner_clean_identity_weight
        ) * identity_loss
        rec = loss_terms["reconstruction"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        completed_steps = step + 1
        pausing = stop_after is not None and completed_steps >= stop_after
        if (
            completed_steps % snapshot_interval == 0
            or completed_steps == steps
            or pausing
        ):
            _save_training_resume_snapshot(
                snapshot_path,
                stage="refiner",
                model=model,
                optimizer=opt,
                completed_steps=completed_steps,
                target_steps=steps,
                elapsed_seconds=(
                    time.perf_counter() - training_started_at
                ),
                cfg=cfg,
                training_contract=database_contract,
                validation_contract=validation_contract,
            )
        if (
            completed_steps % validation_interval == 0
            or completed_steps == steps
            or pausing
        ):
            periodic_metrics = _evaluate_refiner_validation(
                model,
                validation_db,
                db,
                cfg,
                device,
            )
            periodic_decision = _checkpoint_validation_decision(
                periodic_metrics,
                cfg,
                stage="refiner",
            )
            periodic_score = _refiner_validation_score(
                periodic_metrics,
                periodic_decision,
            )
            improved = best_score is None or periodic_score > best_score
            validation_history.append(
                {
                    "completed_steps": int(completed_steps),
                    "score": list(periodic_score),
                    "scientific_acceptance": bool(
                        periodic_decision.get("scientific_acceptance", False)
                    ),
                    "reasons": list(periodic_decision.get("reasons", [])),
                    "selected_as_best": bool(improved),
                    "observed": periodic_decision.get("observed", {}),
                }
            )
            # Train-fit is a diagnostic, never an input to model selection or
            # source-disjoint publication. Same fixed corruption/metric path.
            train_probe = None
            if probe_windows:
                probe_metrics = _evaluate_refiner_validation(
                    model, db, db, cfg, device, maximum_windows=probe_windows
                )
                train_probe = {
                    "role": "training_fit_diagnostic_not_validation",
                    "used_for_checkpoint_selection": False,
                    "metrics": probe_metrics,
                    "decision": _checkpoint_validation_decision(probe_metrics, cfg, stage="refiner"),
                }
            save_json({
                "completed_steps": completed_steps,
                "code_revision": _training_code_revision(),
                "validation": periodic_metrics,
                "checkpoint_decision": periodic_decision,
                "training_probe": train_probe,
            }, out.with_name(f"{out.stem}.validation_step_{completed_steps:06d}.json"))
            print(
                json.dumps(
                    {
                        "stage": "refiner_periodic_validation",
                        **validation_history[-1],
                        "training_probe_observed": (
                            train_probe["decision"].get("observed", {}) if train_probe else None
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if improved:
                best_validation = _save_refiner_best_validation_candidate(
                    best_validation_path,
                    model=model,
                    completed_steps=completed_steps,
                    metrics=periodic_metrics,
                    decision=periodic_decision,
                    cfg=cfg,
                    training_contract=database_contract,
                    validation_contract=validation_contract,
                )
                best_score = periodic_score
        if step == 0 or (step + 1) % 200 == 0 or step == steps - 1:
            _emit_training_progress(
                "[Boundary Refiner]",
                step,
                steps,
                training_started_at,
                loss=loss.item(),
                rec=rec.item(),
                active=loss_terms["active_reconstruction"].item(),
                repair=loss_terms["repair_margin"].item(),
                preserve=loss_terms["clean_preservation"].item(),
                temporal=loss_terms["temporal_supervision"].item(),
                endpoint=loss_terms["endpoint_continuity"].item(),
                seam_velocity=loss_terms["seam_velocity"].item(),
                seam_acceleration=loss_terms["seam_acceleration"].item(),
                seam_jerk=loss_terms["seam_jerk"].item(),
                identity=identity_terms["reconstruction"].item(),
                identity_contact=identity_terms["contact"].item(),
                identity_fk_temporal=identity_terms["fk_temporal"].item(),
                relative_temporal=loss_terms["relative_temporal"].item(),
            )
        if pausing:
            if risk_executor is not None:
                risk_executor.shutdown(wait=True)
            print(json.dumps({
                "stage": "refiner_pilot_paused",
                "completed_steps": completed_steps,
                "target_steps": steps,
                "snapshot": str(snapshot_path),
                "best_candidate": str(best_validation_path),
                "published": False,
                "next_action": "inspect validation before resuming the same target/config/revision",
            }), flush=True)
            return 0
    if risk_executor is not None:
        risk_executor.shutdown(wait=True)
    if best_validation is None:
        raise RuntimeError("Refiner training produced no validation candidate")
    model.load_state_dict(best_validation["model_state_dict"], strict=True)
    validation_report["metrics"] = best_validation["metrics"]
    decision = dict(best_validation["checkpoint_decision"])
    validation_report["checkpoint_decision"] = decision
    validation_report["selection"] = {
        "schema": REFINER_BEST_VALIDATION_SCHEMA,
        "best_candidate": str(best_validation_path),
        "selected_completed_steps": int(
            best_validation["completed_steps"]
        ),
        "score": list(best_validation["score"]),
        "validation_interval_steps": int(validation_interval),
        "history": validation_history,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": REFINER_MODEL_VERSION,
        "output_mode": "contact_residual_logits_smooth_supported_root_joint_tangent",
        "output_dim": PRODUCT_STATE_DIM,
        "joint_mask_conditioned": True,
        "temporal_architecture": {
            "dilations": list(model.temporal_dilations),
            "receptive_field_frames": int(
                model.temporal_receptive_field_frames
            ),
        },
        "training_objective": {
            "schema": "dimensionless_relative_repair_clean_fk_identity_v4",
            "repair_support": "unweighted_seam_core_same_as_validation",
            "geometry_repair_normalization": "per_window_degraded_error",
            "clean_identity_temporal_weight": float(cfg.product_refiner_clean_identity_temporal_weight),
            "relative_temporal_weight": float(cfg.product_refiner_relative_temporal_weight),
            "residual_smoothing_passes": int(cfg.product_refiner_residual_smoothing_passes),
            "residual_taper_frames": int(cfg.product_refiner_residual_taper_frames),
            "corruption_support": "synthetic_transition_seam",
            "minimum_relative_geometry_gain": float(
                cfg.checkpoint_validation_min_geometry_repair_gain
            ),
            "training_target_relative_geometry_gain": float(
                cfg.product_refiner_training_target_repair_gain
            ),
            "clean_identity_weight": float(
                cfg.product_refiner_clean_identity_weight
            ),
            "temporal_supervision_weight": float(
                cfg.product_refiner_temporal_supervision_weight
            ),
            "endpoint_continuity_weight": float(
                cfg.product_refiner_endpoint_continuity_weight
            ),
            "seam_fk_temporal_weights": {
                "velocity": float(cfg.product_refiner_seam_velocity_weight),
                "acceleration": float(
                    cfg.product_refiner_seam_acceleration_weight
                ),
                "jerk": float(cfg.product_refiner_seam_jerk_weight),
            },
            "unobservable_clean_interior_exact_reconstruction": False,
            "tangent_mask_semantics": (
                "soft_scale_then_cap_applied_tangent_v1"
            ),
            "transition_bridge_mix": float(cfg.transition_bridge_mix),
        },
        "state_dict": model.state_dict(),
        "config": dataclasses.asdict(cfg),
        "motion_contract": motion_checkpoint_contract(cfg, "boundary_refiner"),
        "training_event_db_contract": database_contract["event_db_contract"],
        "training_database": database_contract,
        "descriptor_normalization": {
            "source": "training_event_db",
            "mean": np.asarray(db["desc_mean"], dtype=np.float32),
            "std": np.asarray(db["desc_std"], dtype=np.float32),
        },
        "training_resume": resume_info,
        "training_snapshot": {
            "schema": TRAINING_RESUME_SNAPSHOT_SCHEMA,
            "interval_steps": int(snapshot_interval),
            "resume_only": True,
        },
        "best_validation_candidate": {
            "path": str(best_validation_path),
            "selected_completed_steps": int(
                best_validation["completed_steps"]
            ),
            "formal_checkpoint": False,
        },
        "validation": validation_report,
    }
    saved_path, published = _save_checkpoint_after_validation(
        payload,
        out,
        decision,
    )
    print(
        json.dumps(
            {
                "refiner_ckpt": str(saved_path),
                "requested_ckpt": str(out),
                "published": bool(published),
                "steps": steps,
                "validation": validation_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if published else 2


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device).float() * (-math.log(10000.0) / max(half - 1, 1)))
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class TangentDiffusionDenoiser(nn.Module):
    """Reference-point diffusion denoiser in the local 79D product tangent."""

    def __init__(
        self,
        tangent_dim: int = PRODUCT_STATE_DIM,
        cond_dim: int = 32,
        hidden: int = 256,
        time_dim: int = 128,
    ):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.in_proj = nn.Conv1d(
            tangent_dim + EDGE_DIM + 1 + NUM_JOINTS, hidden, 1
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2),
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4),
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, 5, padding=2),
                    nn.GroupNorm(8, hidden),
                    nn.SiLU(),
                ),
            ]
        )
        self.out = nn.Conv1d(hidden, tangent_dim, 1)

    def forward(self, x_tangent, retrieval, cond, seam_mask, joint_mask, t):
        frames = x_tangent.shape[1]
        inp = torch.cat(
            [x_tangent, retrieval, seam_mask, joint_mask], dim=-1
        ).transpose(1, 2)
        h = self.in_proj(inp)
        te = self.time(t)
        ce = _condition_with_time_torch(cond, te, frames, self.cond_proj)
        h = h + ce
        for block in self.blocks:
            h = h + block(h)
        return self.out(h).transpose(1, 2)


def _contact_logit_torch(contact):
    value = contact.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(value) - torch.log1p(-value)


def _encode_reference_tangent_state(reference, target):
    contact_delta = (
        _contact_logit_torch(target[..., :4])
        - _contact_logit_torch(reference[..., :4])
    ).clamp(-8.0, 8.0)
    return torch.cat(
        [contact_delta, product_log_torch(reference, target)], dim=-1
    )


def _tangent_state_mask(joint_mask, root_mask, contact_mask):
    return torch.cat(
        [
            contact_mask.expand(contact_mask.shape[:-1] + (4,)),
            root_mask.expand(root_mask.shape[:-1] + (3,)),
            joint_mask[..., None]
            .expand(joint_mask.shape + (3,))
            .reshape(joint_mask.shape[:-1] + (NUM_JOINTS * 3,)),
        ],
        dim=-1,
    ).clamp(0.0, 1.0)


def _decode_reference_tangent_state(
    reference,
    state,
    joint_mask,
    root_mask,
    contact_mask,
    cfg: MotionGenerationConfig,
):
    if state.shape[-1] != PRODUCT_STATE_DIM:
        raise ValueError(
            f"tangent diffusion state must be {PRODUCT_STATE_DIM}D"
        )
    geometry = masked_retract_torch(
        reference,
        state[..., 4:],
        joint_mask=joint_mask,
        root_mask=root_mask,
        max_rotation_rad=float(cfg.tangent_diffusion_rotation_cap_rad),
        max_root_m=float(cfg.tangent_diffusion_root_cap_m),
    )
    contact_delta = state[..., :4] * contact_mask
    contacts = torch.sigmoid(
        _contact_logit_torch(reference[..., :4]) + contact_delta
    )
    return torch.cat([contacts, geometry[..., 4:]], dim=-1)


def make_beta_schedule(n: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, n, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def train_diffusion(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for Motion Generation diffusion training.")
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    db = load_db(args.db)
    database_contract = _training_db_contract(db, cfg, "Motion Generation training")
    paths = db["paths"]
    desc_z = _descriptor_values_in_training_coordinates(db, db)
    validation_db = load_db(args.val_db)
    validation_contract = _training_db_contract(
        validation_db, cfg, "Motion Generation validation"
    )
    validation_report: Dict[str, Any] = {
        "enabled": True,
        "database": validation_contract,
        "source_disjoint": _validate_source_disjoint(db, validation_db),
    }
    device = torch.device(cfg.device)
    model = TangentDiffusionDenoiser(PRODUCT_STATE_DIM, 32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = int(args.steps or cfg.diffusion_train_steps)
    Tdiff = int(args.diffusion_steps or cfg.diffusion_steps)
    if steps < 1:
        raise ValueError("Motion Diffusion training requires at least one step")
    if Tdiff < 2:
        raise ValueError("Motion Diffusion requires at least two diffusion steps")
    out = Path(args.out)
    snapshot_path, resume_path, snapshot_interval = (
        _resolve_training_snapshot_options(args, cfg, out)
    )
    start_step = 0
    resumed_elapsed_seconds = 0.0
    resume_info: Dict[str, Any] = {
        "resumed": False,
        "snapshot_path": None,
        "completed_steps": 0,
        "snapshot_code_revision": None,
        "runtime_code_revision": _training_code_revision(),
        "cross_revision": False,
    }
    if resume_path is not None:
        start_step, resumed_elapsed_seconds, resume_info = (
            _load_training_resume_snapshot(
                resume_path,
                stage="diffusion",
                model=model,
                optimizer=opt,
                target_steps=steps,
                cfg=cfg,
                training_contract=database_contract,
                validation_contract=validation_contract,
                device=device,
                diffusion_steps=Tdiff,
            )
        )
    _, _, abar = make_beta_schedule(Tdiff, device)
    bs = min(cfg.batch_size, max(2, len(paths)))
    motion_pool = PreloadedMotionWindowPool.preload(
        paths,
        cfg.window_len,
        cfg,
        mode="diffusion",
    )
    gpu_preprocessing = _gpu_preprocessing_enabled(cfg, device)
    risk_executor = None
    if gpu_preprocessing:
        print(
            json.dumps(
                {
                    "stage": "risk_mask_batch",
                    "backend": "torch_cuda",
                    "batch_size": int(bs),
                    "device": str(device),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        risk_executor, _ = _training_risk_mask_executor(bs)
    training_started_at = time.perf_counter() - resumed_elapsed_seconds
    for step in range(start_step, steps):
        clean_batch = []
        retr_batch = []
        seam_batch = []
        cond_batch = []
        for _ in range(bs):
            idx = random.randrange(len(paths))
            clean = motion_pool.sample(idx)
            retr, seam = degrade_for_refiner(
                clean,
                severity=0.045,
                cfg=cfg,
                finalize_contract=not gpu_preprocessing,
            )
            clean_batch.append(clean)
            retr_batch.append(retr)
            seam_batch.append(seam)
            cond_batch.append(desc_z[idx])
        clean_np = np.stack(clean_batch)
        retr_np = np.stack(retr_batch)
        seam_np = np.stack(seam_batch)
        x0 = torch.from_numpy(clean_np).float().to(device)
        retr = torch.from_numpy(retr_np).float().to(device)
        seam = torch.from_numpy(seam_np).float().to(device)
        cond = torch.from_numpy(np.stack(cond_batch)).float().to(device)
        t = torch.randint(0, Tdiff, (bs,), device=device)
        a = abar[t].view(bs, 1, 1)
        if gpu_preprocessing:
            retr = _enforce_internal_batch_contract_torch(
                retr,
                cfg,
            )
            joint_mask, root_mask, contact_mask = _risk_masks_for_batch_torch(
                retr,
                seam,
                cfg,
            )
        else:
            joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                retr_np,
                seam_np,
                cfg,
                executor=risk_executor,
            )
            joint_mask = torch.from_numpy(joint_np).float().to(device)
            root_mask = torch.from_numpy(root_np).float().to(device)
            contact_mask = torch.from_numpy(contact_np).float().to(device)
        state0 = _encode_reference_tangent_state(retr, x0)
        state_mask = _tangent_state_mask(
            joint_mask, root_mask, contact_mask
        )
        active_mask = (state_mask > 0.0).to(state0.dtype)
        noise = torch.randn_like(state0) * active_mask
        x_t = (
            torch.sqrt(a) * state0
            + torch.sqrt(1.0 - a) * noise
        ) * active_mask
        pred_noise = model(
            x_t, retr, cond, seam, joint_mask, t
        ) * active_mask
        loss_noise = (
            ((pred_noise - noise) ** 2 * state_mask).sum()
            / state_mask.sum().clamp_min(1.0)
        )
        state0_hat = (
            x_t - torch.sqrt(1.0 - a) * pred_noise
        ) / torch.sqrt(a).clamp_min(1.0e-6)
        decoded = _decode_reference_tangent_state(
            retr,
            state0_hat,
            joint_mask,
            root_mask,
            contact_mask,
            cfg,
        )
        loss_vel = F.smooth_l1_loss(
            product_log_torch(decoded[:, :-1], decoded[:, 1:]),
            product_log_torch(x0[:, :-1], x0[:, 1:]),
        )
        loss_physics, physics_terms = _world_space_physics_losses(
            decoded,
            x0,
            cfg,
        )
        loss = loss_noise + 0.10 * loss_vel + loss_physics
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        completed_steps = step + 1
        if (
            completed_steps % snapshot_interval == 0
            or completed_steps == steps
        ):
            _save_training_resume_snapshot(
                snapshot_path,
                stage="diffusion",
                model=model,
                optimizer=opt,
                completed_steps=completed_steps,
                target_steps=steps,
                elapsed_seconds=(
                    time.perf_counter() - training_started_at
                ),
                cfg=cfg,
                training_contract=database_contract,
                validation_contract=validation_contract,
                diffusion_steps=Tdiff,
            )
        if step == 0 or (step + 1) % 250 == 0 or step == steps - 1:
            _emit_training_progress(
                "[Motion Generation diffusion]",
                step,
                steps,
                training_started_at,
                loss=loss.item(),
                noise=loss_noise.item(),
                physics=loss_physics.item(),
                jerk=physics_terms["jerk"].item(),
            )
    if risk_executor is not None:
        risk_executor.shutdown(wait=True)
    validation_report["metrics"] = _evaluate_diffusion_validation(
        model, validation_db, db, cfg, device, abar, Tdiff
    )
    decision = _checkpoint_validation_decision(
        validation_report["metrics"],
        cfg,
        stage="diffusion",
    )
    validation_report["checkpoint_decision"] = decision
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "reference_tangent_motion_diffusion_v2",
        "diffusion_space": "reference_point_product_tangent",
        "state_dim": PRODUCT_STATE_DIM,
        "joint_mask_conditioned": True,
        "training_corruption": {
            "schema": "retained_clean_transition_bridge_v1",
            "transition_bridge_mix": float(cfg.transition_bridge_mix),
            "rotation_noise": "smooth_so3_tangent",
        },
        "state_dict": model.state_dict(),
        "config": dataclasses.asdict(cfg),
        "diffusion_steps": Tdiff,
        "motion_contract": motion_checkpoint_contract(cfg, "motion_diffusion"),
        "training_event_db_contract": database_contract["event_db_contract"],
        "training_database": database_contract,
        "descriptor_normalization": {
            "source": "training_event_db",
            "mean": np.asarray(db["desc_mean"], dtype=np.float32),
            "std": np.asarray(db["desc_std"], dtype=np.float32),
        },
        "training_resume": resume_info,
        "training_snapshot": {
            "schema": TRAINING_RESUME_SNAPSHOT_SCHEMA,
            "interval_steps": int(snapshot_interval),
            "resume_only": True,
        },
        "validation": validation_report,
    }
    saved_path, published = _save_checkpoint_after_validation(
        payload,
        out,
        decision,
    )
    print(
        json.dumps(
            {
                "diffusion_ckpt": str(saved_path),
                "requested_ckpt": str(out),
                "published": bool(published),
                "steps": steps,
                "validation": validation_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if published else 2





# ===== TRUSTED LOCAL CKPT LOAD FIX START =====
def _trusted_torch_load(path, map_location=None, **_unused_kwargs):
    """Load trusted local checkpoints saved by this project.

    Project checkpoints include contract metadata and NumPy arrays, so trusted
    local assets are loaded with ``weights_only=False`` explicitly.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to load checkpoints.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch has no weights_only argument.
        return torch.load(path, map_location=map_location)
# ===== TRUSTED LOCAL CKPT LOAD FIX END =====

def transition_cost(exit_state: np.ndarray, entry_state: np.ndarray, cexit: np.ndarray, centry: np.ndarray) -> float:
    pose_exit = exit_state[: NUM_JOINTS * 3]
    vel_exit = exit_state[NUM_JOINTS * 3 :]
    pose_entry = entry_state[: NUM_JOINTS * 3]
    vel_entry = entry_state[NUM_JOINTS * 3 :]
    pose = float(np.mean((pose_exit - pose_entry) ** 2))
    vel = float(np.mean((vel_exit - vel_entry) ** 2))
    contact = float(np.mean(np.abs(cexit - centry)))
    return pose * 0.8 + vel * 1.6 + contact * 0.12




def align_next_to_prev(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    out = nxt.copy()
    delta = prev[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += delta[0]
    out[:, ROOT_Z_IDX] += delta[1]
    # Soft root-y adjustment only; do not force same height completely.
    dy = prev[-1, ROOT_Y_IDX] - out[0, ROOT_Y_IDX]
    ramp = np.linspace(1.0, 0.0, min(18, len(out)), dtype=np.float32)
    out[: len(ramp), ROOT_Y_IDX] += dy * ramp
    return out


def concatenate_events_with_overlap(event_paths: Sequence[str], target_durations: Sequence[float], cfg: MotionGenerationConfig) -> Tuple[np.ndarray, List[dict]]:
    """Concatenate retrieved RAG events under the EDGE-151D contract.

    Event Semantics fix:
    The overlap cross-fade is now compensated locally per segment, not by a
    whole-song global resample.  Every music slot keeps its assigned net frame
    budget after overlap trimming, so local beat/phrase boundaries do not drift.
    We also keep the Overlap Alignment/Overlap Alignment yaw-aligned overlap start, no root-Y ramp,
    ov==1 midpoint weighting, and safe one-frame overlap slicing.
    """
    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    target_lens = [max(cfg.min_event_frames, int(round(float(d) * cfg.fps))) for d in target_durations]
    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(
            m_raw, cfg, source_hint=f"concat_load:{p}", derive_contact=True, project_rot=True
        )
        target_len = int(target_lens[i])
        # Event Semantics: compensate overlap locally.  Incoming clips lose ov frames
        # when m = m[ov:] removes the overlapped prefix.  Rather than globally
        # resampling the entire final song, pre-extend non-first clips by the
        # maximum plausible overlap and then locally normalize their post-overlap
        # remainder back to target_len.  This preserves per-slot music timing.
        overlap_budget = int(max(0, getattr(cfg, "overlap", 0))) if pieces else 0
        local_resample_len = int(max(cfg.min_event_frames, target_len + overlap_budget))
        warp = local_resample_len / max(1, m.shape[0])
        m = resample_motion_np(m, local_resample_len).astype(np.float32)
        m, post_resample_report = enforce_edge151_contract_np(
            m, cfg, source_hint=f"concat_resample_local_timing:{p}", derive_contact=True, project_rot=True
        )
        used_overlap = 0
        align_report = None
        blend_report = None
        local_timing_report = {
            "expected_net_frames": int(target_len),
            "local_resample_frames_before_overlap": int(local_resample_len),
            "overlap_budget_frames": int(overlap_budget),
            "overlap_trim_frames": 0,
            "post_overlap_frames_before_local_fix": int(local_resample_len),
            "local_timing_fix_applied": False,
            "local_timing_fix_mode": "none",
        }
        if pieces:
            ov = min(int(cfg.overlap), len(pieces[-1]) // 3, len(m) // 3)
            used_overlap = int(max(0, ov))
            if ov > 0:
                # Align incoming m[0] to the previous overlap start in both yaw
                # and XZ position.  This avoids both speed surge and cross-heading
                # tearing inside the quaternion overlap window.
                ref = pieces[-1][-ov].copy()
                try:
                    yaw_ref = float(root_yaw_np(pieces[-1][-ov:][:1])[0])
                    yaw_m = float(root_yaw_np(m[:1])[0])
                    dyaw = float(np.arctan2(np.sin(yaw_ref - yaw_m), np.cos(yaw_ref - yaw_m)))
                except Exception:
                    yaw_ref, yaw_m, dyaw = 0.0, 0.0, 0.0
                m = rotate_motion_around_y_np(m, dyaw, pivot_xz=m[0, [ROOT_X_IDX, ROOT_Z_IDX]])
                delta_xz = ref[[ROOT_X_IDX, ROOT_Z_IDX]] - m[0, [ROOT_X_IDX, ROOT_Z_IDX]]
                m[:, ROOT_X_IDX] += float(delta_xz[0])
                m[:, ROOT_Z_IDX] += float(delta_xz[1])
                # Deliberately do not apply any root-Y ramp.  Height/contact
                # continuity is handled only inside the real overlap blend.
                m, align_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_overlap_start_yaw_align:{p}", derive_contact=True, project_rot=True
                )
                if align_report is None:
                    align_report = {}
                align_report.update({
                    "overlap_alignment_mode": "yaw_and_xz_to_overlap_start_no_root_y_ramp",
                    "overlap_ref_frame": "previous_event[-overlap]",
                    "yaw_ref": float(yaw_ref),
                    "yaw_incoming_before": float(yaw_m),
                    "dyaw_applied": float(dyaw),
                    "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
                    "root_y_ramp_applied": False,
                })
                a = pieces[-1][-ov:].copy()
                b = m[:ov].copy()
                if ov == 1:
                    w_b = np.asarray([[0.5]], dtype=np.float32)
                else:
                    w_b = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
                blend, blend_report = blend_motion_overlap_np(
                    a, b, w_b, cfg, source_hint=f"concat_overlap_quat:{Path(str(p)).name}"
                )
                pieces[-1] = np.concatenate([pieces[-1][:-ov], blend], axis=0)
                pieces[-1], _ = enforce_edge151_contract_np(
                    pieces[-1], cfg, source_hint="concat_piece_after_quat_overlap", derive_contact=True, project_rot=True
                )
                m = m[ov:]
                local_timing_report["overlap_trim_frames"] = int(ov)
                local_timing_report["post_overlap_frames_before_local_fix"] = int(m.shape[0])
            else:
                m = align_next_to_prev(pieces[-1], m)
                m, align_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_align_no_overlap:{p}", derive_contact=True, project_rot=True
                )
                local_timing_report["post_overlap_frames_before_local_fix"] = int(m.shape[0])

            # Event Semantics: after overlap handling, repair only this incoming segment's
            # net length.  This prevents whole-song interpolation from smearing
            # contact steps and preserves local music slot boundaries.
            if int(m.shape[0]) != int(target_len):
                m = resample_motion_np(m, int(target_len)).astype(np.float32)
                m, local_fix_report = enforce_edge151_contract_np(
                    m, cfg, source_hint=f"concat_local_timing_fix:{p}", derive_contact=True, project_rot=True
                )
                local_timing_report.update({
                    "local_timing_fix_applied": True,
                    "local_timing_fix_mode": "segment_local_resample_after_overlap_trim",
                    "frames_after_local_timing_fix": int(m.shape[0]),
                    "contract_after_local_timing_fix": local_fix_report,
                })
            else:
                local_timing_report["frames_after_local_timing_fix"] = int(m.shape[0])

        pieces.append(m.astype(np.float32))
        rep.append({
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "local_resample_frames": int(local_resample_len),
            "warp": float(warp),
            "overlap": int(used_overlap),
            "boundary_blend_mode": "quaternion_rotation" if used_overlap > 0 else "none",
            "contract_pre": pre_report,
            "contract_after_resample": post_resample_report,
            "contract_after_align": align_report,
            "contract_overlap_blend": blend_report,
            "segment_local_timing": local_timing_report,
        })

    final = np.concatenate(pieces, axis=0).astype(np.float32)
    total_target_frames = int(sum(target_lens))
    timing_report = {
        "target_total_frames": int(total_target_frames),
        "frames_before_terminal_guard": int(final.shape[0]),
        "timing_frame_delta_before_terminal_guard": int(total_target_frames - final.shape[0]),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "segment_local_overlap_compensation_no_global_resample",
        "global_resample_applied": False,
    }
    # Terminal guard only.  It should normally be a no-op because each segment is
    # locally length-corrected.  If a pathological one-frame edge case remains,
    # trim or hold the last frame instead of globally resampling thousands of
    # frames, so local beat/contact timing is not redistributed across the song.
    if total_target_frames > 0 and int(final.shape[0]) != int(total_target_frames):
        delta = int(total_target_frames - final.shape[0])
        if delta > 0:
            pad = np.repeat(final[-1:, :], delta, axis=0).astype(np.float32)
            final = np.concatenate([final, pad], axis=0).astype(np.float32)
            mode = "terminal_hold_last_frame_pad_no_global_resample"
        else:
            final = final[:total_target_frames].astype(np.float32)
            mode = "terminal_trim_no_global_resample"
        timing_report.update({
            "timing_compensation_applied": True,
            "timing_compensation_mode": mode,
            "terminal_delta_frames": int(delta),
        })
    timing_report["frames_after_terminal_guard"] = int(final.shape[0])
    final, final_report = enforce_edge151_contract_np(
        final, cfg, source_hint="concat_final", derive_contact=True, project_rot=True
    )
    if rep:
        rep[-1]["concat_timing_compensation"] = timing_report
        rep[-1]["concat_final_contract"] = final_report
    return final.astype(np.float32), rep





# ===== Transition Budget TRANSITION-BUDGET PATCH START =====
def _transition_env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _transition_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _transition_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def quat_slerp_np(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Vectorized shortest-path quaternion SLERP / nlerp fallback.

    q0, q1: [...,4], w: broadcastable [...,1]. Returns normalized [...,4].
    This function is intentionally NumPy-only so it is usable during concat.
    """
    q0 = normalize_quat_np(np.asarray(q0, dtype=np.float32))
    q1 = normalize_quat_np(np.asarray(q1, dtype=np.float32))
    w = np.asarray(w, dtype=np.float32)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    # Use normalized linear interpolation near zero angle; true SLERP elsewhere.
    near = dot > 0.9995
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - w) * theta0) / np.maximum(sin0, 1e-8)
    s1 = np.sin(w * theta0) / np.maximum(sin0, 1e-8)
    qs = s0 * q0 + s1 * q1
    ql = (1.0 - w) * q0 + w * q1
    out = np.where(near, ql, qs)
    return normalize_quat_np(out).astype(np.float32)


def _root_velocity(m: np.ndarray, at_end: bool) -> np.ndarray:
    if m.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)
    if at_end:
        return (m[-1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - m[-2, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)
    return (m[1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - m[0, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)


def transition_len_for_boundary(prev: np.ndarray, curr: np.ndarray, target_len: int, cfg: MotionGenerationConfig) -> int:
    """Choose a boundary transition budget for the *incoming* slot.

    The length is risk-aware but capped by the current music slot so that the
    retrieved core motion remains dominant.  This implements the paper position:
    real events preserve cultural vocabulary, local generation repairs only seams.
    """
    min_t = _transition_env_int("MOTION_TRANSITION_MIN_FRAMES", 8)
    max_t = _transition_env_int("MOTION_TRANSITION_MAX_FRAMES", 24)
    ratio = _transition_env_float("MOTION_TRANSITION_RATIO", 0.18)
    min_core = _transition_env_int("MOTION_TRANSITION_MIN_CORE_FRAMES", max(18, int(getattr(cfg, "min_event_frames", 36) * 0.55)))
    base = int(round(float(target_len) * float(ratio)))

    try:
        exit_j = fk_24_np(prev[-min(len(prev), 3):])[-1]
        entry_j = fk_24_np(curr[:min(len(curr), 3)])[0]
        pose_gap = float(np.linalg.norm(exit_j - entry_j, axis=-1).mean())
    except Exception:
        pose_gap = 0.0
    try:
        yaw_gap = abs(float(np.arctan2(np.sin(root_yaw_np(prev[-1:])[0] - root_yaw_np(curr[:1])[0]),
                                      np.cos(root_yaw_np(prev[-1:])[0] - root_yaw_np(curr[:1])[0]))))
    except Exception:
        yaw_gap = 0.0
    # Small risk schedule: larger pose/yaw gap gets a longer bridge.
    risk_extra = int(round(np.clip(pose_gap * 16.0 + yaw_gap * 4.0, 0.0, 10.0)))
    L = int(np.clip(base + risk_extra, min_t, max_t))
    L = min(L, max(0, int(target_len) - int(min_core)))
    return int(max(0, L))


def motion_inbetween_np(left_ctx: np.ndarray, right_ctx: np.ndarray, length: int, cfg: MotionGenerationConfig,
                        source_hint: str = "reference_inbetween") -> np.ndarray:
    """Generate a kinematic transition in EDGE-151D space.

    The bridge interpolates root trajectory with cubic Hermite and rotations with
    quaternion shortest-path interpolation.  Contact channels are rebuilt by FK in
    enforce_edge151_contract_np, so no invalid gray contacts are preserved.
    """
    L = int(length)
    if L <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    a = np.asarray(left_ctx[-1], dtype=np.float32).copy()
    b = np.asarray(right_ctx[0], dtype=np.float32).copy()
    out = np.repeat(a[None, :], L, axis=0).astype(np.float32)

    # Phase excludes exact endpoints to avoid duplicating previous last or next first.
    u = (np.arange(1, L + 1, dtype=np.float32) / float(L + 1))[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2

    p0 = a[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    p1 = b[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    v0 = _root_velocity(left_ctx, at_end=True)
    v1 = _root_velocity(right_ctx, at_end=False)
    # Limit velocity to prevent a transition budget from launching the body.
    vmax = _transition_env_float("MOTION_TRANSITION_ROOT_VEL_CLAMP_MPF", 0.055)
    v0 = np.clip(v0, -vmax, vmax)
    v1 = np.clip(v1, -vmax, vmax)
    root = h00 * p0[None] + h10 * (L * v0[None]) + h01 * p1[None] + h11 * (L * v1[None])
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root.astype(np.float32)

    # Rotation SLERP for all joints.
    Ra = rot6d_to_matrix_np(a[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    Rb = rot6d_to_matrix_np(b[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    qa = matrix_to_quat_np(Ra)[None, :, :]
    qb = matrix_to_quat_np(Rb)[None, :, :]
    q = quat_slerp_np(np.repeat(qa, L, axis=0), np.repeat(qb, L, axis=0), u[:, None, :])
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(L, -1)

    # Contacts are not linearly interpolated. Rebuild from FK/contact thresholds.
    out[:, 0:4] = 0.0
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=source_hint, derive_contact=True, project_rot=True)
    return out.astype(np.float32)


def align_event_core_to_prev_np(prev: np.ndarray, curr: np.ndarray, cfg: MotionGenerationConfig) -> Tuple[np.ndarray, dict]:
    """Yaw + XZ align the incoming event core to the previous exit."""
    out = np.asarray(curr, dtype=np.float32).copy()
    rep: Dict[str, object] = {"mode": "none"}
    if prev.shape[0] == 0 or out.shape[0] == 0:
        return out, rep
    try:
        yaw_ref = float(root_yaw_np(prev[-1:])[0])
        yaw_m = float(root_yaw_np(out[:1])[0])
        dyaw = float(np.arctan2(np.sin(yaw_ref - yaw_m), np.cos(yaw_ref - yaw_m)))
    except Exception:
        yaw_ref, yaw_m, dyaw = 0.0, 0.0, 0.0
    out = rotate_motion_around_y_np(out, dyaw, pivot_xz=out[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    delta_xz = prev[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += float(delta_xz[0])
    out[:, ROOT_Z_IDX] += float(delta_xz[1])
    out, contract = enforce_edge151_contract_np(out, cfg, source_hint="align_event_core_to_previous", derive_contact=True, project_rot=True)
    rep = {"mode": "yaw_xz_entry_to_prev_exit", "yaw_ref": yaw_ref, "yaw_incoming_before": yaw_m,
           "dyaw_applied": dyaw, "delta_xz_applied": [float(delta_xz[0]), float(delta_xz[1])],
           "root_y_ramp_applied": False, "contract": contract}
    return out.astype(np.float32), rep



# === Reference Inbetweening reference-conditioned transition budget begin ===





















# === Reference Inbetweening reference-conditioned transition budget begin ===





















# === Reference Inbetweening reference-conditioned transition budget begin ===
def _inbetween_env_bool(name: str, default: bool) -> bool:
    if name in os.environ:
        try:
            return bool(int(os.environ[name]))
        except Exception:
            return str(os.environ[name]).strip().lower() in {"true", "yes", "on"}
    return bool(default)


def _inbetween_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _inbetween_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _inbetween_config_bool(cfg: MotionGenerationConfig, attr: str, env: str, default: bool) -> bool:
    return _inbetween_env_bool(env, bool(getattr(cfg, attr, default)))


def _inbetween_config_int(cfg: MotionGenerationConfig, attr: str, env: str, default: int) -> int:
    return _inbetween_env_int(env, int(getattr(cfg, attr, default)))


def _inbetween_config_float(cfg: MotionGenerationConfig, attr: str, env: str, default: float) -> float:
    return _inbetween_env_float(env, float(getattr(cfg, attr, default)))


def _slerp_quaternion_np(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorized quaternion SLERP. q0/q1: [J,4], t: [T,1,1]."""
    q0 = normalize_quat_np(np.asarray(q0, dtype=np.float32))
    q1 = normalize_quat_np(np.asarray(q1, dtype=np.float32))
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    tt = np.asarray(t, dtype=np.float32)
    q0b = q0[None]
    q1b = q1[None]
    dotb = dot[None]
    thetab = theta[None]
    sinb = sin_theta[None]
    lerp = normalize_quat_np((1.0 - tt) * q0b + tt * q1b)
    s0 = np.sin((1.0 - tt) * thetab) / np.maximum(sinb, 1e-6)
    s1 = np.sin(tt * thetab) / np.maximum(sinb, 1e-6)
    slerp = normalize_quat_np(s0 * q0b + s1 * q1b)
    use_lerp = (dotb > 0.9995) | (np.abs(sinb) < 1e-6)
    return np.where(use_lerp, lerp, slerp).astype(np.float32)


def reference_motion_inbetween_np(
    prev_tail: np.ndarray,
    curr_head: np.ndarray,
    n_frames: int,
    cfg: MotionGenerationConfig,
    *,
    finalize_contract: bool = True,
) -> np.ndarray:
    """Kinematic inbetweening in EDGE-151D: root Hermite + per-joint rotation SLERP.

    prev_tail and curr_head are short clips. The generated bridge excludes both
    endpoints, so it can be inserted between previous core and current core
    without duplicating boundary frames.
    """
    n = int(n_frames)
    if n <= 0:
        return np.zeros((0, EDGE_DIM), dtype=np.float32)
    a_clip = np.asarray(prev_tail, dtype=np.float32)
    b_clip = np.asarray(curr_head, dtype=np.float32)
    a = a_clip[-1].copy()
    b = b_clip[0].copy()
    out = np.zeros((n, EDGE_DIM), dtype=np.float32)
    phase = (np.arange(n, dtype=np.float32) + 1.0) / float(n + 1)
    s = phase[:, None]
    smooth = (s * s * (3.0 - 2.0 * s)).astype(np.float32)

    # Contact channels are re-derived after FK; keep them as conservative blends here.
    out[:, 0:4] = ((1.0 - smooth) * a[None, 0:4] + smooth * b[None, 0:4]).astype(np.float32)

    # Root position: C1 Hermite using local endpoint velocities.
    p0 = a[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    p1 = b[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    v0 = np.zeros(3, dtype=np.float32)
    v1 = np.zeros(3, dtype=np.float32)
    if a_clip.shape[0] >= 2:
        v0 = (a_clip[-1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - a_clip[-2, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)
    if b_clip.shape[0] >= 2:
        v1 = (b_clip[1, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - b_clip[0, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]).astype(np.float32)
    # Bound bridge tangents to avoid long-range root launches at mismatched clips.
    fps = max(float(cfg.fps), 1.0e-8)
    v0 *= fps
    v1 *= fps
    max_step = _inbetween_config_float(cfg, "transition_root_tangent_max_mps", "MOTION_TRANSITION_ROOT_TANGENT_MAX_MPS", 1.35)
    for vv in (v0, v1):
        norm = float(np.linalg.norm(vv[[0, 2]]))
        if norm > max_step:
            vv[[0, 2]] *= max_step / max(norm, 1e-8)
    tt = phase[:, None]
    h00 = 2 * tt ** 3 - 3 * tt ** 2 + 1
    h10 = tt ** 3 - 2 * tt ** 2 + tt
    h01 = -2 * tt ** 3 + 3 * tt ** 2
    h11 = tt ** 3 - tt ** 2
    scale = float(n + 1) / fps
    root = h00 * p0[None] + h10 * (v0[None] * scale) + h01 * p1[None] + h11 * (v1[None] * scale)
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root.astype(np.float32)

    # Rotation: joint-wise quaternion SLERP, then convert back to legal Rot6D.
    Ra = rot6d_to_matrix_np(
        a[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6),
        project=False,
    )[0]
    Rb = rot6d_to_matrix_np(
        b[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6),
        project=False,
    )[0]
    qa = matrix_to_quat_np(Ra)
    qb = matrix_to_quat_np(Rb)
    q = _slerp_quaternion_np(qa, qb, phase.reshape(n, 1, 1))
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(
        R,
        project=False,
    ).reshape(n, -1)

    if not finalize_contract:
        return out.astype(np.float32)
    out, _ = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint="inbetween_motion_inbetween",
        derive_contact=True,
        project_rot=True,
    )
    return out.astype(np.float32)


def _align_core_to_previous(prev_piece: np.ndarray, core: np.ndarray, cfg: MotionGenerationConfig) -> Tuple[np.ndarray, dict]:
    """Align current core to previous endpoint in yaw and XZ only."""
    out = core.copy().astype(np.float32)
    report: Dict[str, object] = {"mode": "yaw_xz_to_previous_endpoint_no_root_y_ramp"}
    if prev_piece.size == 0 or out.size == 0:
        return out, report
    try:
        yaw_prev = float(root_yaw_np(prev_piece[-1:])[0])
        yaw_core = float(root_yaw_np(out[:1])[0])
        dyaw = float(np.arctan2(np.sin(yaw_prev - yaw_core), np.cos(yaw_prev - yaw_core)))
    except Exception:
        yaw_prev, yaw_core, dyaw = 0.0, 0.0, 0.0
    out = rotate_motion_around_y_np(out, dyaw, pivot_xz=out[0, [ROOT_X_IDX, ROOT_Z_IDX]])
    delta = prev_piece[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - out[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, ROOT_X_IDX] += float(delta[0])
    out[:, ROOT_Z_IDX] += float(delta[1])
    out, contract = enforce_edge151_contract_np(out, cfg, source_hint="inbetween_align_core_to_prev", derive_contact=True, project_rot=True)
    report.update({
        "yaw_prev": float(yaw_prev),
        "yaw_core_before": float(yaw_core),
        "dyaw_applied": float(dyaw),
        "delta_xz_applied": [float(delta[0]), float(delta[1])],
        "root_y_ramp_applied": False,
        "contract": contract,
    })
    return out.astype(np.float32), report


def _choose_core_and_transition_lengths(source_len: int, target_len: int, has_prev: bool, cfg: MotionGenerationConfig) -> Tuple[int, int, dict]:
    """Return (core_len, transition_in_len) while preserving target_len exactly."""
    target_len = max(1, int(target_len))
    source_len = max(1, int(source_len))
    if not has_prev:
        return target_len, 0, {"reason": "first_slot_no_transition", "core_warp": float(target_len / source_len)}

    min_trans = _inbetween_config_int(cfg, "transition_min_frames", "MOTION_TRANSITION_MIN_FRAMES", 10)
    max_trans = _inbetween_config_int(cfg, "transition_max_frames", "MOTION_TRANSITION_MAX_FRAMES", 28)
    ratio = _inbetween_config_float(cfg, "transition_ratio", "MOTION_TRANSITION_RATIO", 0.18)
    min_core = _inbetween_config_int(cfg, "transition_min_core_frames", "MOTION_TRANSITION_MIN_CORE_FRAMES", 30)
    warp_min = _inbetween_config_float(cfg, "core_warp_min", "MOTION_CORE_WARP_MIN", 0.72)
    warp_max = _inbetween_config_float(cfg, "core_warp_max", "MOTION_CORE_WARP_MAX", 1.38)

    if target_len <= min_core + 2:
        return target_len, 0, {"reason": "slot_too_short_for_transition", "core_warp": float(target_len / source_len)}

    trans = int(round(target_len * ratio))
    trans = max(min_trans, min(max_trans, trans))
    trans = min(trans, max(0, target_len - min_core))
    core = max(min_core, target_len - trans)

    # Prefer natural core duration, but never violate total slot length.
    lower = max(min_core, int(round(source_len * warp_min)))
    upper = max(lower, int(round(source_len * warp_max)))
    desired = int(np.clip(core, lower, upper))
    desired = min(max(min_core, desired), target_len - max(1, min_trans))
    if desired > 0:
        core = desired
        trans = target_len - core

    if trans < 0:
        trans = 0
        core = target_len
    info = {
        "target_len": int(target_len),
        "source_len": int(source_len),
        "transition_frames": int(trans),
        "core_frames": int(core),
        "core_warp": float(core / max(1, source_len)),
        "warp_min": float(warp_min),
        "warp_max": float(warp_max),
        "ratio": float(ratio),
    }
    return int(core), int(trans), info


def concat_events(event_paths: Sequence[str], target_durations: Sequence[float], cfg: MotionGenerationConfig) -> Tuple[np.ndarray, List[dict]]:
    """Reference Inbetweening reference-conditioned transition-budget concatenation.

    This constructs a strong reference motion stream (motion_ref): each music
    slot contributes exactly target_frames.  For non-first slots, part of the
    slot is reserved as transition budget; the core event is lightly resampled,
    aligned in yaw/XZ, and connected through root-Hermite + rotation-SLERP
    inbetweening.  The generated transition spans are reported so generate()
    can build a precise transition mask for Motion Refiner and Motion Diffusion.
    """
    if not _inbetween_config_bool(cfg, "transition_budget_enable", "MOTION_TRANSITION_BUDGET_ENABLE", True):
        if "concatenate_events_with_overlap" in globals():
            return concatenate_events_with_overlap(event_paths, target_durations, cfg)

    pieces: List[np.ndarray] = []
    rep: List[dict] = []
    target_lens = [max(cfg.min_event_frames, int(round(float(d) * cfg.fps))) for d in target_durations]
    cursor = 0
    transition_spans_global: List[Tuple[int, int]] = []

    for i, (p, dur) in enumerate(zip(event_paths, target_durations)):
        m_raw = np.load(str(p)).astype(np.float32)
        m, pre_report = enforce_edge151_contract_np(
            m_raw, cfg, source_hint=f"inbetween_concat_load:{p}", derive_contact=True, project_rot=True
        )
        target_len = int(target_lens[i])
        has_prev = bool(pieces)
        core_len, trans_len, length_info = _choose_core_and_transition_lengths(m.shape[0], target_len, has_prev, cfg)
        core = resample_motion_np(m, int(core_len)).astype(np.float32)
        core, core_report = enforce_edge151_contract_np(
            core, cfg, source_hint=f"inbetween_core_resample:{p}", derive_contact=True, project_rot=True
        )
        align_report = None
        bridge_report: Dict[str, object] = {"enabled": False, "frames": 0}
        transition_span = None

        if has_prev and trans_len > 0 and _inbetween_config_bool(cfg, "transition_inbetween_enable", "MOTION_TRANSITION_INBETWEEN_ENABLE", True):
            core, align_report = _align_core_to_previous(pieces[-1], core, cfg)
            prev_tail_n = min(max(2, trans_len // 2), len(pieces[-1]))
            curr_head_n = min(max(2, trans_len // 2), len(core))
            bridge = reference_motion_inbetween_np(pieces[-1][-prev_tail_n:], core[:curr_head_n], trans_len, cfg)
            start = cursor
            end = cursor + int(bridge.shape[0])
            transition_span = [int(start), int(end)]
            transition_spans_global.append((int(start), int(end)))
            pieces.append(bridge.astype(np.float32))
            cursor += int(bridge.shape[0])
            bridge_report = {
                "enabled": True,
                "mode": "root_hermite_rotation_slerp_motion_space_inbetweening",
                "frames": int(bridge.shape[0]),
                "span": transition_span,
                "prev_tail_frames": int(prev_tail_n),
                "curr_head_frames": int(curr_head_n),
            }
        elif has_prev:
            core, align_report = _align_core_to_previous(pieces[-1], core, cfg)

        pieces.append(core.astype(np.float32))
        core_span = [int(cursor), int(cursor + core.shape[0])]
        cursor += int(core.shape[0])
        rep.append({
            "version": "inbetween_reference_conditioned_transition_budget",
            "path": str(p),
            "target_frames": int(target_len),
            "source_frames": int(m_raw.shape[0]),
            "core_frames": int(core.shape[0]),
            "transition_in_frames": int(trans_len if has_prev else 0),
            "slot_total_frames": int((trans_len if has_prev else 0) + core.shape[0]),
            "core_span": core_span,
            "transition_span": transition_span,
            "transition_spans": [transition_span] if transition_span else [],
            "core_warp": float(core.shape[0] / max(1, m_raw.shape[0])),
            "length_policy": length_info,
            "contract_pre": pre_report,
            "contract_core": core_report,
            "contract_after_align": align_report,
            "boundary_inbetween": bridge_report,
            "reference_conditioning": {
                "motion_ref_role": "strong_reference_trajectory",
                "diffusion_should_edit": "transition_mask_regions_only_by_default",
                "core_motion_preservation": True,
            },
        })

    if pieces:
        final = np.concatenate(pieces, axis=0).astype(np.float32)
    else:
        final = np.zeros((0, EDGE_DIM), dtype=np.float32)
    total_target_frames = int(sum(target_lens))
    timing_report = {
        "target_total_frames": int(total_target_frames),
        "frames_before_terminal_guard": int(final.shape[0]),
        "timing_frame_delta_before_terminal_guard": int(total_target_frames - final.shape[0]),
        "timing_compensation_applied": False,
        "timing_compensation_mode": "inbetween_slot_exact_transition_budget_no_global_resample",
        "global_resample_applied": False,
        "transition_spans_global": [[int(a), int(b)] for a, b in transition_spans_global],
    }
    if total_target_frames > 0 and int(final.shape[0]) != int(total_target_frames):
        delta = int(total_target_frames - final.shape[0])
        if delta > 0:
            pad = np.repeat(final[-1:, :], delta, axis=0).astype(np.float32)
            final = np.concatenate([final, pad], axis=0).astype(np.float32)
            mode = "terminal_hold_last_frame_pad_no_global_resample"
        else:
            final = final[:total_target_frames].astype(np.float32)
            mode = "terminal_trim_no_global_resample"
        timing_report.update({
            "timing_compensation_applied": True,
            "timing_compensation_mode": mode,
            "terminal_delta_frames": int(delta),
        })
    timing_report["frames_after_terminal_guard"] = int(final.shape[0])
    final, final_report = enforce_edge151_contract_np(
        final, cfg, source_hint="inbetween_concat_final_motion_ref", derive_contact=True, project_rot=True
    )
    if rep:
        rep[-1]["concat_timing_compensation"] = timing_report
        rep[-1]["concat_final_contract"] = final_report
    return final.astype(np.float32), rep


# === Reference Inbetweening reference-conditioned transition budget end ===


# === Reference Inbetweening reference-conditioned transition budget end ===


def make_transition_budget_mask(T: int, transition_spans: Sequence[Sequence[int]], cfg: MotionGenerationConfig) -> np.ndarray:
    """Build precise transition mask with optional halo and low core mask."""
    core_val = _inbetween_config_float(cfg, "transition_core_mask_value", "MOTION_TRANSITION_CORE_MASK_VALUE", 0.0)
    halo = max(0, int(round(float(cfg.transition_mask_halo_seconds) * float(cfg.fps))))
    mask = np.full((int(T), 1), float(core_val), dtype=np.float32)
    for sp in transition_spans:
        if sp is None or len(sp) < 2:
            continue
        a, b = int(sp[0]), int(sp[1])
        a0 = max(0, a - halo)
        b0 = min(int(T), b + halo)
        if b0 <= a0:
            continue
        # Raised plateau: transition core = 1, halo ramps down to core_val.
        mask[a:b, 0] = 1.0
        if halo > 0:
            la = max(0, a - halo)
            if a > la:
                ramp = np.linspace(float(core_val), 1.0, a - la, endpoint=False, dtype=np.float32)
                mask[la:a, 0] = np.maximum(mask[la:a, 0], ramp)
            rb = min(int(T), b + halo)
            if rb > b:
                ramp = np.linspace(1.0, float(core_val), rb - b, endpoint=False, dtype=np.float32)
                mask[b:rb, 0] = np.maximum(mask[b:rb, 0], ramp)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)
# === Reference Inbetweening reference-conditioned transition budget end ===


def make_boundary_mask(T: int, seams: Sequence[int], width: int = 18) -> np.ndarray:
    mask = np.zeros((T, 1), dtype=np.float32)
    for s in seams:
        a = max(0, int(s) - width)
        b = min(T, int(s) + width)
        mask[a:b, 0] = 1.0
    return mask


def analytic_residual_refine(motion: np.ndarray, seam_positions: Sequence[int], width: int = 24) -> np.ndarray:
    out = motion.copy().astype(np.float32)
    for s in seam_positions:
        a = max(0, s - width)
        b = min(len(out), s + width)
        if b - a < 4:
            continue
        left = out[a].copy()
        right = out[b - 1].copy()
        x = np.linspace(0, 1, b - a, dtype=np.float32)[:, None]
        cubic = x * x * (3 - 2 * x)
        bridge = resample_motion_np(
            np.stack([left, right], axis=0).astype(np.float32),
            b - a,
        )
        bridge[:, ROOT_X_IDX:ROOT_Z_IDX + 1] = (
            (1 - cubic) * left[None, ROOT_X_IDX:ROOT_Z_IDX + 1]
            + cubic * right[None, ROOT_X_IDX:ROOT_Z_IDX + 1]
        )
        # Only blend root and rotations near boundary; keep original high-frequency content.
        w = np.sin(np.linspace(0, math.pi, b - a, dtype=np.float32))[:, None] ** 2
        out[a:b] = blend_edge151_geodesic_np(out[a:b], bridge, 0.35 * w)
    out[:, ROOT_Y_IDX] = smooth_np(out[:, ROOT_Y_IDX:ROOT_Y_IDX + 1], 1.0)[:, 0]
    return out.astype(np.float32)


_INFERENCE_MODEL_CACHE: Dict[Tuple[str, str, int, int, str], Dict[str, Any]] = {}


def _cached_inference_model(
    role: str,
    ckpt_path: str | Path,
    cfg: MotionGenerationConfig,
) -> Dict[str, Any]:
    """Load one immutable inference checkpoint/model per file revision."""
    path = Path(ckpt_path).resolve()
    stat = path.stat()
    key = (
        str(role),
        str(path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(cfg.device),
    )
    cached = _INFERENCE_MODEL_CACHE.get(key)
    if cached is None:
        checkpoint = _trusted_torch_load(path, map_location=cfg.device)
        if role == "boundary_refiner":
            expected_version = REFINER_MODEL_VERSION
            if str(checkpoint.get("version", "")) != expected_version:
                raise RuntimeError(
                    "Formal generation rejects a non-product refiner checkpoint"
                )
            model = ProductManifoldTemporalRefiner(EDGE_DIM, 32).to(cfg.device)
            schedule = None
        elif role == "motion_diffusion":
            expected_version = "reference_tangent_motion_diffusion_v2"
            if str(checkpoint.get("version", "")) != expected_version:
                raise RuntimeError(
                    "Formal generation rejects a non-tangent diffusion checkpoint"
                )
            model = TangentDiffusionDenoiser(PRODUCT_STATE_DIM, 32).to(
                cfg.device
            )
            diffusion_steps = int(
                checkpoint.get("diffusion_steps", cfg.diffusion_steps)
            )
            schedule = make_beta_schedule(
                diffusion_steps,
                torch.device(cfg.device),
            )
        else:
            raise ValueError(f"unsupported inference model role: {role!r}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        cached = {
            "checkpoint": checkpoint,
            "model": model,
            "schedule": schedule,
        }
        # Discard stale revisions of the same role/path/device while retaining
        # the independent Refiner and Diffusion entries.
        stale_keys = [
            candidate
            for candidate in _INFERENCE_MODEL_CACHE
            if candidate[0] == str(role)
            and candidate[1] == str(path)
            and candidate[4] == str(cfg.device)
        ]
        for stale_key in stale_keys:
            _INFERENCE_MODEL_CACHE.pop(stale_key, None)
        _INFERENCE_MODEL_CACHE[key] = cached
        print(
            json.dumps(
                {
                    "stage": "motion_inference_model_cache",
                    "role": role,
                    "checkpoint": str(path),
                    "cache_hit": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if role == "boundary_refiner":
        stored_config = cached["checkpoint"].get("config", {})
        for field, default in (
            ("product_refiner_residual_smoothing_passes", 2),
            ("product_refiner_residual_taper_frames", 3),
        ):
            if int(getattr(cfg, field)) != int(stored_config.get(field, default)):
                raise RuntimeError(f"Refiner decoder configuration mismatch: {field}")
    assert_motion_checkpoint_contract(
        cached["checkpoint"],
        cfg,
        path,
        role,
    )
    return cached





def apply_refiner_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: MotionGenerationConfig) -> np.ndarray:
    """Apply Motion Refiner as reference-conditioned transition residual refiner.

    Core regions are strongly locked.  By default only a tiny residual is allowed
    outside transition masks; transition regions receive the full correction.
    """
    core_strength = _inbetween_config_float(cfg, "refiner_core_strength", "MOTION_REFINER_CORE_STRENGTH", 0.02)
    trans_strength = _inbetween_config_float(cfg, "refiner_transition_strength", "MOTION_REFINER_TRANSITION_STRENGTH", 1.00)
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        seam_centers = []
        for a, b in contiguous_regions(seam_mask[:, 0] > 0.5):
            seam_centers.append((a + b) // 2)
        refined = analytic_residual_refine(motion, seam_centers)
        # Blend analytic fallback back to the reference outside transition mask.
        w = np.clip(core_strength + (trans_strength - core_strength) * seam_mask.astype(np.float32), 0.0, 1.0)
        refined = motion.astype(np.float32) * (1.0 - w) + refined.astype(np.float32) * w
        refined, _ = enforce_edge151_contract_np(
            refined, cfg, source_hint="apply_refiner_model:inbetween_reference_analytic", derive_contact=True, project_rot=True
        )
        return refined.astype(np.float32)

    cached_model = _cached_inference_model(
        "boundary_refiner",
        ckpt_path,
        cfg,
    )
    model = cached_model["model"]
    gpu_preprocessing = _gpu_preprocessing_enabled(cfg, cfg.device)

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)
    records: List[Dict[str, Any]] = []
    for st, ed in sliding_window_ranges(T, win, hop):
        chunk = motion[st:ed]
        mask = seam_mask[st:ed]
        orig_len = len(chunk)
        cond_in = _condition_chunk_np(
            cond,
            st,
            ed,
            win if orig_len < win else orig_len,
        )
        if orig_len < win:
            chunk_in = resample_motion_np(chunk, win)
            mask_in = resample_motion_np(mask, win)
        else:
            chunk_in = chunk
            mask_in = mask
        if not gpu_preprocessing:
            chunk_in, _ = enforce_edge151_contract_np(
                chunk_in,
                cfg,
                source_hint="apply_refiner_model:inbetween_input_chunk",
                derive_contact=True,
                project_rot=True,
            )
        records.append(
            {
                "start": st,
                "end": ed,
                "original_length": orig_len,
                "chunk": chunk_in,
                "condition": cond_in,
                "mask": mask_in,
            }
        )

    inference_batch_size = min(
        max(1, int(cfg.inference_window_batch_size)),
        max(1, len(records)),
    )
    risk_workers = (
        0
        if gpu_preprocessing
        else min(
            8,
            max(1, int(os.cpu_count() or 1)),
            inference_batch_size,
        )
    )
    risk_executor = (
        ThreadPoolExecutor(max_workers=risk_workers)
        if risk_workers > 1
        else None
    )
    print(
        json.dumps(
            {
                "stage": "boundary_refiner_inference_windows",
                "windows": len(records),
                "batch_size": inference_batch_size,
                "risk_workers": risk_workers,
                "risk_backend": (
                    "torch_cuda" if gpu_preprocessing else "numpy_cpu"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with torch.no_grad():
        for batch_start in range(0, len(records), inference_batch_size):
            batch_records = records[
                batch_start:batch_start + inference_batch_size
            ]
            chunk_batch = np.stack([item["chunk"] for item in batch_records])
            condition_batch = np.stack(
                [item["condition"] for item in batch_records]
            ).astype(np.float32)
            mask_batch = np.stack(
                [item["mask"] for item in batch_records]
            ).astype(np.float32)
            x = torch.from_numpy(chunk_batch).float().to(cfg.device)
            c = torch.from_numpy(condition_batch).float().to(cfg.device)
            sm = torch.from_numpy(mask_batch).float().to(cfg.device)
            if gpu_preprocessing:
                x = _enforce_internal_batch_contract_torch(x, cfg)
            strength = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * sm, 0.0, 1.0)
            if gpu_preprocessing:
                joint_t, root_t, contact_t = _risk_masks_for_batch_torch(
                    x,
                    sm,
                    cfg,
                )
            else:
                joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                    chunk_batch,
                    mask_batch,
                    cfg,
                    executor=risk_executor,
                )
                joint_t = torch.from_numpy(joint_np).float().to(cfg.device)
                root_t = torch.from_numpy(root_np).float().to(cfg.device)
                contact_t = torch.from_numpy(contact_np).float().to(cfg.device)
            output = model(x, c, sm, joint_t)
            y = _decode_product_refiner_output(
                x,
                output,
                joint_t * strength,
                root_t * strength,
                contact_t * strength,
                cfg,
            )
            if gpu_preprocessing:
                y = _enforce_internal_batch_contract_torch(
                    y,
                    cfg,
                    derive_contact=False,
                )
            proposals = y.detach().cpu().numpy()
            for item, y_np in zip(batch_records, proposals):
                orig_len = int(item["original_length"])
                st = int(item["start"])
                ed = int(item["end"])
                if orig_len < win:
                    y_np = resample_motion_np(y_np, orig_len)
                    y_np, _ = enforce_edge151_contract_np(
                        y_np,
                        cfg,
                        source_hint="apply_refiner_model:resampled_output_chunk",
                        derive_contact=False,
                        project_rot=True,
                    )
                elif not gpu_preprocessing:
                    y_np, _ = enforce_edge151_contract_np(
                        y_np,
                        cfg,
                        source_hint="apply_refiner_model:inbetween_output_chunk",
                        derive_contact=False,
                        project_rot=True,
                    )
                w = overlap_add_weight_np(orig_len, st, T, hop, win)
                accumulate_motion_window_np(
                    accum,
                    weight_sum,
                    rot_quat_accum,
                    rot_quat_weight,
                    y_np,
                    w,
                    st,
                    ed,
                )
            print(
                f"[Boundary Refiner inference] "
                f"windows={min(batch_start + len(batch_records), len(records))}"
                f"/{len(records)}",
                flush=True,
            )
    if risk_executor is not None:
        risk_executor.shutdown(wait=True)

    out, _ = finalize_motion_window_accum_np(
        accum,
        weight_sum,
        rot_quat_accum,
        rot_quat_weight,
        cfg,
        source_hint="apply_refiner_model:inbetween_final",
        derive_contact=False,
    )
    # Hard blend with original reference according to the exact transition mask.
    w = np.clip(core_strength + (trans_strength - core_strength) * seam_mask.astype(np.float32), 0.0, 1.0)
    if gpu_preprocessing:
        with torch.no_grad():
            full_masks = _risk_masks_for_batch_torch(
                torch.from_numpy(motion[None]).float().to(cfg.device),
                torch.from_numpy(seam_mask[None]).float().to(cfg.device),
                cfg,
            )
        full_joint, full_root, full_contact = (
            value.cpu().numpy() for value in full_masks
        )
    else:
        full_joint, full_root, full_contact = _risk_masks_for_batch_np(
            motion[None], seam_mask[None], cfg
        )
    tangent = product_log_np(motion, out)
    joint_support = (full_joint[0] * w > 0.0).astype(np.float32)
    root_support = (full_root[0] * w > 0.0).astype(np.float32)
    out_geometry = masked_retract_np(
        motion,
        tangent,
        joint_mask=joint_support,
        root_mask=root_support,
        max_rotation_rad=float(cfg.product_refiner_rotation_cap_rad),
        max_root_m=float(cfg.product_refiner_root_cap_m),
    )
    contact_weight = (full_contact[0] * w > 0.0).astype(np.float32)
    out_geometry[:, :4] = (
        motion[:, :4] * (1.0 - contact_weight)
        + out[:, :4] * contact_weight
    )
    out = out_geometry
    out, _ = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint="apply_refiner_model:inbetween_reference_blend",
        derive_contact=False,
        project_rot=True,
    )
    return out.astype(np.float32)



def apply_diffusion_model(
    motion: np.ndarray,
    cond: np.ndarray,
    seam_mask: np.ndarray,
    ckpt_path: Optional[str],
    cfg: MotionGenerationConfig,
) -> np.ndarray:
    """Apply only the current reference-tangent motion diffusion checkpoint."""

    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        motion, _ = enforce_edge151_contract_np(
            motion,
            cfg,
            source_hint="apply_diffusion_model:disabled",
            derive_contact=True,
            project_rot=True,
        )
        return motion.astype(np.float32)

    core_strength = _inbetween_config_float(
        cfg, "diffusion_core_strength", "MOTION_DIFFUSION_CORE_STRENGTH", 0.0
    )
    trans_strength = _inbetween_config_float(
        cfg,
        "diffusion_transition_strength",
        "MOTION_DIFFUSION_TRANSITION_STRENGTH",
        0.72,
    )
    noise_scale = _inbetween_config_float(
        cfg,
        "diffusion_reference_noise_scale",
        "MOTION_DIFFUSION_REFERENCE_NOISE_SCALE",
        0.03,
    )
    cached_model = _cached_inference_model(
        "motion_diffusion",
        ckpt_path,
        cfg,
    )
    checkpoint = cached_model["checkpoint"]
    model = cached_model["model"]
    steps = int(checkpoint.get("diffusion_steps", cfg.diffusion_steps))
    betas, alphas, abar = cached_model["schedule"]
    gpu_preprocessing = _gpu_preprocessing_enabled(cfg, cfg.device)

    total = int(motion.shape[0])
    window = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", window)), window))
    accumulators = init_motion_window_accumulators(total, EDGE_DIM)
    records: List[Dict[str, Any]] = []
    for start, end in sliding_window_ranges(total, window, hop):
        retrieval_np = motion[start:end]
        mask_np = seam_mask[start:end]
        original_length = len(retrieval_np)
        condition_np = _condition_chunk_np(
            cond,
            start,
            end,
            window if original_length < window else original_length,
        )
        if original_length < window:
            retrieval_input = resample_motion_np(retrieval_np, window)
            mask_input = resample_motion_np(mask_np, window)
        else:
            retrieval_input = retrieval_np
            mask_input = mask_np
        if not gpu_preprocessing:
            retrieval_input, _ = enforce_edge151_contract_np(
                retrieval_input,
                cfg,
                source_hint="apply_diffusion_model:retrieval_chunk",
                derive_contact=True,
                project_rot=True,
            )
        records.append(
            {
                "start": start,
                "end": end,
                "original_length": original_length,
                "retrieval": retrieval_input,
                "condition": condition_np,
                "mask": mask_input,
            }
        )

    noise_window_cap = _diffusion_noise_window_cap(
        cfg,
        steps=steps,
        window=window,
        dtype=torch.float32,
    )
    inference_batch_size = min(
        max(1, int(cfg.inference_window_batch_size)),
        max(1, len(records)),
        noise_window_cap,
    )
    # Consume the caller-controlled global stream once.  All subsequent noise
    # is derived from the global window index, so changing batch size cannot
    # change the generated song while memory remains bounded to one batch.
    noise_seed = int(
        torch.randint(
            0,
            (1 << 31) - 1,
            (1,),
            device="cpu",
            dtype=torch.int64,
        ).item()
    )
    risk_workers = (
        0
        if gpu_preprocessing
        else min(
            8,
            max(1, int(os.cpu_count() or 1)),
            inference_batch_size,
        )
    )
    risk_executor = (
        ThreadPoolExecutor(max_workers=risk_workers)
        if risk_workers > 1
        else None
    )
    print(
        json.dumps(
            {
                "stage": "motion_diffusion_inference_windows",
                "windows": len(records),
                "batch_size": inference_batch_size,
                "diffusion_steps": steps,
                "risk_workers": risk_workers,
                "risk_backend": (
                    "torch_cuda" if gpu_preprocessing else "numpy_cpu"
                ),
                "noise_backend": "seeded_per_window_device_batch",
                "noise_window_cap": noise_window_cap,
                "noise_batch_max_mib": float(
                    cfg.diffusion_noise_batch_max_mib
                ),
                "noise_seed": noise_seed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with torch.no_grad():
        for batch_start in range(0, len(records), inference_batch_size):
            batch_records = records[
                batch_start:batch_start + inference_batch_size
            ]
            retrieval_batch = np.stack(
                [item["retrieval"] for item in batch_records]
            )
            mask_batch = np.stack(
                [item["mask"] for item in batch_records]
            ).astype(np.float32)
            condition_batch = np.stack(
                [item["condition"] for item in batch_records]
            ).astype(np.float32)
            retrieval = torch.from_numpy(retrieval_batch).float().to(cfg.device)
            raw_mask = torch.from_numpy(mask_batch).float().to(cfg.device)
            if gpu_preprocessing:
                retrieval = _enforce_internal_batch_contract_torch(
                    retrieval,
                    cfg,
                )
            strength = torch.clamp(
                float(core_strength)
                + (float(trans_strength) - float(core_strength)) * raw_mask,
                0.0,
                1.0,
            )
            condition = torch.from_numpy(
                condition_batch
            ).float().to(cfg.device)
            if gpu_preprocessing:
                joint, root, contact = _risk_masks_for_batch_torch(
                    retrieval,
                    raw_mask,
                    cfg,
                )
            else:
                joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                    retrieval_batch,
                    mask_batch,
                    cfg,
                    executor=risk_executor,
                )
                joint = torch.from_numpy(joint_np).float().to(cfg.device)
                root = torch.from_numpy(root_np).float().to(cfg.device)
                contact = torch.from_numpy(contact_np).float().to(cfg.device)
            effective_joint = joint * strength
            effective_root = root * strength
            effective_contact = contact * strength
            state_mask = _tangent_state_mask(
                effective_joint, effective_root, effective_contact
            )
            active = (state_mask > 0.0).to(retrieval.dtype)

            batch_noise = _diffusion_noise_batch_torch(
                range(batch_start, batch_start + len(batch_records)),
                base_seed=noise_seed,
                steps=steps,
                window=window,
                dtype=retrieval.dtype,
                device=cfg.device,
            )
            state = (
                float(noise_scale)
                * batch_noise[:, 0]
                * active
            )
            for step in reversed(range(steps)):
                timestep = torch.full(
                    (len(batch_records),),
                    step,
                    device=cfg.device,
                    dtype=torch.long,
                )
                prediction = model(
                    state, retrieval, condition, raw_mask, joint, timestep
                ) * active
                beta = betas[step]
                alpha = alphas[step]
                cumulative = abar[step]
                mean = (1 / torch.sqrt(alpha)) * (
                    state
                    - beta
                    / torch.sqrt(1 - cumulative).clamp_min(1.0e-6)
                    * prediction
                )
                if step > 0:
                    state = (
                        mean
                        + torch.sqrt(beta)
                        * batch_noise[:, steps - step]
                        * 0.35
                        * active
                    )
                else:
                    state = mean
                state = state * active
            proposals = _decode_reference_tangent_state(
                retrieval,
                state,
                effective_joint,
                effective_root,
                effective_contact,
                cfg,
            )
            if gpu_preprocessing:
                proposals = _enforce_internal_batch_contract_torch(
                    proposals,
                    cfg,
                    derive_contact=False,
                )
            proposals = proposals.detach().cpu().numpy()
            for item, proposal in zip(batch_records, proposals):
                original_length = int(item["original_length"])
                start = int(item["start"])
                end = int(item["end"])
                if original_length < window:
                    proposal = resample_motion_np(proposal, original_length)
                    proposal, _ = enforce_edge151_contract_np(
                        proposal,
                        cfg,
                        source_hint="apply_diffusion_model:resampled_output_chunk",
                        derive_contact=False,
                        project_rot=True,
                    )
                elif not gpu_preprocessing:
                    proposal, _ = enforce_edge151_contract_np(
                        proposal,
                        cfg,
                        source_hint="apply_diffusion_model:output_chunk",
                        derive_contact=False,
                        project_rot=True,
                    )
                weight = overlap_add_weight_np(
                    original_length, start, total, hop, window
                )
                accumulate_motion_window_np(
                    *accumulators,
                    proposal,
                    weight,
                    start,
                    end,
                )
            print(
                f"[Motion diffusion inference] "
                f"windows={min(batch_start + len(batch_records), len(records))}"
                f"/{len(records)}",
                flush=True,
            )
    if risk_executor is not None:
        risk_executor.shutdown(wait=True)

    out, _ = finalize_motion_window_accum_np(
        *accumulators,
        cfg,
        source_hint="apply_diffusion_model:final",
        derive_contact=False,
    )
    weight = np.clip(
        core_strength
        + (trans_strength - core_strength) * seam_mask.astype(np.float32),
        0.0,
        1.0,
    )
    if gpu_preprocessing:
        with torch.no_grad():
            full_masks = _risk_masks_for_batch_torch(
                torch.from_numpy(motion[None]).float().to(cfg.device),
                torch.from_numpy(seam_mask[None]).float().to(cfg.device),
                cfg,
            )
        full_joint, full_root, full_contact = (
            value.cpu().numpy() for value in full_masks
        )
    else:
        full_joint, full_root, full_contact = _risk_masks_for_batch_np(
            motion[None], seam_mask[None], cfg
        )
    tangent = product_log_np(motion, out)
    geometry = masked_retract_np(
        motion,
        tangent,
        joint_mask=(full_joint[0] * weight > 0.0).astype(np.float32),
        root_mask=(full_root[0] * weight > 0.0).astype(np.float32),
        max_rotation_rad=float(cfg.tangent_diffusion_rotation_cap_rad),
        max_root_m=float(cfg.tangent_diffusion_root_cap_m),
    )
    contact_weight = (full_contact[0] * weight > 0.0).astype(np.float32)
    geometry[:, :4] = (
        motion[:, :4] * (1.0 - contact_weight)
        + out[:, :4] * contact_weight
    )
    geometry, _ = enforce_edge151_contract_np(
        geometry,
        cfg,
        source_hint="apply_diffusion_model:reference_blend",
        derive_contact=False,
        project_rot=True,
    )
    return geometry.astype(np.float32)


def derive_contacts_np(
    motion: np.ndarray,
    cfg: MotionGenerationConfig,
    *,
    precomputed_joints: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    joints = (
        fk_24_np(motion)
        if precomputed_joints is None
        else np.asarray(precomputed_joints, dtype=np.float32)
    )
    expected_joints = (len(np.asarray(motion)), NUM_JOINTS, 3)
    if joints.shape != expected_joints:
        raise ValueError(
            f"Expected precomputed joints {expected_joints}, got {joints.shape}"
        )
    foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
    foot_y = foot[..., 1]
    floor_y = float(np.percentile(foot_y.reshape(-1), 5))
    vel = np.zeros(foot.shape[:2], dtype=np.float32)
    vel[1:] = (
        np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
        * float(cfg.fps)
    )
    height_score = np.clip(1.0 - (foot_y - floor_y) / max(cfg.ik_height_margin, 1e-6), 0.0, 1.0)
    speed_score = np.clip(1.0 - vel / max(cfg.ik_speed_gate_mps, 1e-6), 0.0, 1.0)
    conf = 0.62 * height_score + 0.38 * speed_score
    clean = np.zeros_like(conf, dtype=bool)
    for f in range(conf.shape[1]):
        state = False
        for t, p in enumerate(conf[:, f]):
            if p >= cfg.ik_contact_high:
                state = True
            elif p <= cfg.ik_contact_low:
                state = False
            clean[t, f] = state
        median_frames = max(1, int(round(float(cfg.fps) / 6.0)))
        if median_frames % 2 == 0:
            median_frames += 1
        clean[:, f] = median_bool_filter(clean[:, f], median_frames)
        # A support contact cannot coexist with a large horizontal foot step.
        # This hard veto prevents hysteresis from carrying a stale contact
        # label across a transition spike and makes foot-skate metrics honest.
        clean[vel[:, f] > float(getattr(cfg, "ik_contact_break_speed_mps", 0.54)), f] = False
    return clean, conf.astype(np.float32), floor_y, foot.astype(np.float32)


def c1_hanning_window01(phase: np.ndarray | float) -> np.ndarray | float:
    """C1-safe 0->1->0 window. Value and first derivative are zero at both ends."""
    ph = np.clip(np.asarray(phase, dtype=np.float32), 0.0, 1.0)
    out = np.sin(np.pi * ph) ** 2
    if np.isscalar(phase):
        return float(out)
    return out.astype(np.float32)


def smoothstep01(x: np.ndarray | float) -> np.ndarray | float:
    y = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    out = y * y * (3.0 - 2.0 * y)
    if np.isscalar(x):
        return float(out)
    return out.astype(np.float32)


def apply_root_y_c1_physics_np(motion: np.ndarray, contacts: np.ndarray, cfg: MotionGenerationConfig) -> Tuple[np.ndarray, dict]:
    """
    Root-Vertical Safety root-Y safety pass.

    Fixes three common post-process artifacts:
    1) Damping Snap Bug: landing damping length is the real contact duration,
       so the damping dip is exactly zero before the next flight frame.
    2) C1 Discontinuity: flight/parabola blending uses a Hanning/sin^2 gate,
       whose first derivative is zero at takeoff and landing boundaries.
    3) Micro-Flight Shock Bug: landing damping is applied only when the
       immediately preceding flight island is a real effective flight. The same
       physical-time flight threshold gates both parabola and damping,
       so a 1-2 frame denoising gap cannot create a ghost landing hit.
    4) Space-Jump Bug: extremely long no-contact islands are treated as broken
       contact labels / bad upstream generation and are fused off before the
       ballistic parabola can launch the root several meters into the air.

    Only the legal EDGE root-Y channel is edited here; lower-body IK later writes
    legal lower-body rot6d channels.
    """
    out = motion.copy().astype(np.float32)
    if not bool(cfg.root_y_physics_enable) or out.shape[0] < 4:
        return out, {"enabled": False, "reason": "disabled_or_too_short"}

    root_y0 = out[:, ROOT_Y_IDX].copy()
    any_contact = contacts.any(axis=1)
    is_flight = ~any_contact
    fps = max(float(cfg.fps), 1e-6)
    min_effective_flight = max(1, int(round(float(cfg.root_y_min_flight_seconds) * fps)))
    max_biological_flight_s = float(max(cfg.root_y_max_flight_seconds, 1.0 / fps))
    max_biological_flight_frames = max(min_effective_flight, int(round(max_biological_flight_s * fps)))

    flight_applied = 0
    flight_skipped_micro = 0
    flight_skipped_space_jump = 0
    for start, end in contiguous_regions(is_flight):
        n = end - start
        if n < min_effective_flight:
            flight_skipped_micro += 1
            continue
        air_duration = n / fps
        # Flight Safety Fuse biological fuse: a 2+ second no-contact interval in generated
        # long dance is almost always broken contact labels / bad retrieval, not
        # a valid human jump. Injecting a physical parabola would create a
        # multi-meter "space launch". Preserve the native root trajectory instead.
        if air_duration > max_biological_flight_s:
            flight_skipped_space_jump += 1
            continue
        left = max(0, start - 1)
        right = min(len(root_y0) - 1, end)
        if right <= left:
            continue
        y0 = float(root_y0[left])
        y1 = float(root_y0[right])
        duration = max((right - left) / fps, 1.0 / fps)
        v0 = (y1 - y0 + 0.5 * 9.81 * duration * duration) / duration
        for k, ti in enumerate(range(start, end)):
            # Use exact endpoint phases for zero-value/zero-slope blend.
            phase = 0.0 if n <= 1 else k / float(n - 1)
            gate = float(cfg.root_y_flight_strength) * float(c1_hanning_window01(phase))
            tau = (ti - left) / fps
            parabola = y0 + v0 * tau - 0.5 * 9.81 * tau * tau
            out[ti, ROOT_Y_IDX] = (1.0 - gate) * out[ti, ROOT_Y_IDX] + gate * parabola
        flight_applied += 1

    damping_applied = 0
    damping_skipped_micro_flight = 0
    damping_skipped_space_jump_flight = 0
    damping_preview: List[Dict[str, object]] = []
    for start, end in contiguous_regions(any_contact):
        # Only damp actual landings: a contact island that follows flight.
        if start <= 0 or any_contact[start - 1]:
            continue

        # Root-Vertical Safety terminal guard: trace the immediately preceding no-contact
        # island. Damping is allowed only if this flight island was long enough
        # to receive the ballistic treatment. This preserves logical
        # conservation between parabola and landing damping, and prevents a
        # 1-2 frame per-foot-filter gap from creating a ghost impact dip.
        prev_flight_end = start
        prev_flight_start = prev_flight_end
        while prev_flight_start > 0 and not bool(any_contact[prev_flight_start - 1]):
            prev_flight_start -= 1
        prev_flight_len = prev_flight_end - prev_flight_start
        if prev_flight_len < min_effective_flight:
            damping_skipped_micro_flight += 1
            if len(damping_preview) < 24:
                damping_preview.append({
                    "start": int(start),
                    "end": int(end),
                    "frames": int(end - start),
                    "skipped": True,
                    "reason": "preceding_micro_flight",
                    "preceding_flight_start": int(prev_flight_start),
                    "preceding_flight_end": int(prev_flight_end),
                    "preceding_flight_frames": int(prev_flight_len),
                    "min_effective_flight_frames": int(min_effective_flight),
                })
            continue
        prev_flight_duration = prev_flight_len / fps
        if prev_flight_duration > max_biological_flight_s or prev_flight_len > max_biological_flight_frames:
            damping_skipped_space_jump_flight += 1
            if len(damping_preview) < 24:
                damping_preview.append({
                    "start": int(start),
                    "end": int(end),
                    "frames": int(end - start),
                    "skipped": True,
                    "reason": "preceding_space_jump_flight",
                    "preceding_flight_start": int(prev_flight_start),
                    "preceding_flight_end": int(prev_flight_end),
                    "preceding_flight_frames": int(prev_flight_len),
                    "preceding_flight_seconds": float(prev_flight_duration),
                    "max_biological_flight_seconds": float(max_biological_flight_s),
                })
            continue

        contact_len = end - start
        max_damp_frames = max(3, int(round(float(cfg.root_y_damping_max_seconds) * fps)))
        n = min(contact_len, max_damp_frames)
        if n <= 2:
            continue
        # Root-Aware Motion Safety critical fix: damping is capped to an early post-touchdown window
        # instead of stretching across the whole contact island.  The chosen
        # window still has zero dip at both ends, so it is C0-safe at the next
        # untouched frame but cannot create a delayed squat in long support.
        max_abs_dip = 0.0
        for k, ti in enumerate(range(start, start + n)):
            phase = k / float(max(n - 1, 1))
            gate = float(c1_hanning_window01(phase))
            # Decay biases the cushion toward the landing instant while the
            # Hanning gate keeps both value and first derivative zero at ends.
            dip = float(cfg.root_y_damping_max_dip) * math.exp(-4.0 * phase) * gate
            out[ti, ROOT_Y_IDX] -= dip
            max_abs_dip = max(max_abs_dip, abs(dip))
        damping_applied += 1
        if len(damping_preview) < 24:
            damping_preview.append({
                "start": int(start),
                "end": int(end),
                "frames": int(contact_len),
                "damping_frames": int(n),
                "max_damping_frames": int(max_damp_frames),
                "max_dip_m": float(max_abs_dip),
                "capped_early_duration": True,
                "capped_seconds": float(cfg.root_y_damping_max_seconds),
                "preceding_flight_frames": int(prev_flight_len),
                "effective_flight_gated": True,
            })

    delta = out[:, ROOT_Y_IDX] - root_y0
    return out.astype(np.float32), {
        "enabled": True,
        "version": "biological_flight_root_y_physics",
        "fixes_damping_snap": True,
        "fixes_c1_discontinuity": True,
        "fixes_micro_flight_shock": True,
        "fixes_space_jump_bug": True,
        "flight_gate": "hanning_sin_squared",
        "damping_duration": "capped_early_contact_window",
        "damping_max_seconds": float(cfg.root_y_damping_max_seconds),
        "damping_requires_effective_preceding_flight": True,
        "ballistic_requires_biological_flight_duration": True,
        "max_biological_flight_seconds": float(max_biological_flight_s),
        "max_biological_flight_frames": int(max_biological_flight_frames),
        "min_effective_flight_frames": int(min_effective_flight),
        "flight_segments_applied": int(flight_applied),
        "flight_segments_skipped_micro": int(flight_skipped_micro),
        "flight_segments_skipped_space_jump": int(flight_skipped_space_jump),
        "landing_damping_applied": int(damping_applied),
        "landing_damping_skipped_micro_flight": int(damping_skipped_micro_flight),
        "landing_damping_skipped_space_jump_flight": int(damping_skipped_space_jump_flight),
        "damping_preview": damping_preview,
        "delta_mean": float(delta.mean()),
        "delta_p95_abs": float(np.percentile(np.abs(delta), 95)),
        "delta_max_abs": float(np.max(np.abs(delta))),
    }


def contact_ramp_weights_np(
    contacts: np.ndarray,
    *,
    fps: float,
    ramp_seconds: float,
) -> np.ndarray:
    """Return smooth support weights without changing the binary contact state."""

    state = np.asarray(contacts, dtype=bool)
    if state.ndim != 2:
        raise ValueError(f"Expected [T,F] contacts, got {state.shape}")
    ramp = max(1, int(round(max(float(ramp_seconds), 0.0) * float(fps))))
    weights = state.astype(np.float32)
    for foot in range(state.shape[1]):
        index = 0
        while index < len(state):
            if not state[index, foot]:
                index += 1
                continue
            end = index + 1
            while end < len(state) and state[end, foot]:
                end += 1
            for frame in range(index, end):
                phase_in = min(1.0, (frame - index + 1) / float(ramp))
                phase_out = min(1.0, (end - frame) / float(ramp))
                weights[frame, foot] = float(
                    min(smoothstep01(phase_in), smoothstep01(phase_out))
                )
            index = end
    return weights.astype(np.float32)


def generate_ik_targets_np(native_foot: np.ndarray, contacts: np.ndarray, cfg: MotionGenerationConfig, root_xz: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Generate Lower-Body IK lower-body IK targets with a Sliding-Support IK sliding-anchor cloud-step guard.

    The old span-only guard caused a Footskate Forgiveness Paradox: the worse a
    slow AI foot-drift became, the more likely it was to exceed the XZ threshold
    and be released from repair. This version never uses ``continue`` as an
    amnesty for large contact travel.

    Decision rule:
    - Large span + sufficiently high mean velocity (+ not absurdly bursty) is
      considered cloud-step only when it is also root/CoM-consistent: the root
      moves, the foot direction agrees with root direction, and foot-relative-to-root
      drift stays bounded. It then receives a moving local-window anchor.
    - Large span + low mean velocity is treated as AI dark drift / footskate and
      is still locked to a static contact-internal anchor for IK repair.
    - Short/static contacts use the same static anchor as before.

    Targets are initialized from native FK positions, and non-contact frames are
    never edited.
    """
    targets = native_foot.copy().astype(np.float32)
    locked_segments = 0
    sliding_anchor_segments = 0
    dark_drift_locked_segments = 0
    root_inconsistent_locked_segments = 0
    skipped_short = 0
    preview: List[Dict[str, object]] = []

    win = max(3, int(round(float(cfg.ik_sliding_anchor_seconds) * float(cfg.fps))))
    if win % 2 == 0:
        win += 1
    half = win // 2
    speed_thr = float(cfg.ik_cloud_step_speed_mps)
    span_thr = float(cfg.ik_slide_release_m)
    min_frames = max(1, int(round(float(cfg.ik_slide_release_min_seconds) * float(cfg.fps))))
    speed_cv_max = float(cfg.ik_cloud_speed_cv_max)
    root_min_travel = float(cfg.ik_cloud_root_min_travel_m)
    direction_cos_min = float(cfg.ik_cloud_direction_cos_min)
    rel_span_max = float(cfg.ik_cloud_root_foot_rel_max_m)
    if root_xz is not None:
        root_xz = np.asarray(root_xz, dtype=np.float32)
        if root_xz.ndim != 2 or root_xz.shape[0] != native_foot.shape[0] or root_xz.shape[1] != 2:
            root_xz = None

    for f in range(native_foot.shape[1]):
        for start, end in contiguous_regions(contacts[:, f]):
            length = end - start
            if length < 3:
                skipped_short += 1
                continue

            seg = native_foot[start:end, f, :].astype(np.float32)
            seg_xz = seg[:, [0, 2]]
            span = float(np.linalg.norm(seg_xz.max(axis=0) - seg_xz.min(axis=0))) if length > 1 else 0.0
            step = np.linalg.norm(seg_xz[1:] - seg_xz[:-1], axis=-1) if length > 1 else np.zeros((0,), dtype=np.float32)
            arc = float(step.sum())
            duration_s = max((length - 1) / max(float(cfg.fps), 1e-6), 1.0 / max(float(cfg.fps), 1e-6))
            mean_speed_mps = float(arc / duration_s)
            inst_speed = step * float(cfg.fps) if step.size else np.zeros((0,), dtype=np.float32)
            speed_std = float(inst_speed.std()) if inst_speed.size else 0.0
            speed_mean = float(inst_speed.mean()) if inst_speed.size else 0.0
            speed_cv = float(speed_std / max(speed_mean, 1e-6)) if inst_speed.size else 0.0
            path_efficiency = float(span / max(arc, 1e-6)) if arc > 1e-6 else 1.0

            is_large_contact_travel = length >= min_frames and span > span_thr

            # Root-Aware Motion Safety root-foot relative test.  Smooth high-speed foot travel is
            # not sufficient evidence for an intentional Dunhuang cloud-step:
            # severe AI footskate can also be smooth.  A true cloud-step should
            # be supported by root/CoM translation in a compatible direction and
            # should not show unbounded foot motion relative to the root.
            root_span = 0.0
            root_foot_rel_span = 0.0
            foot_root_cos = 0.0
            root_consistent = False
            if root_xz is not None and length > 1:
                root_seg = root_xz[start:end].astype(np.float32)
                root_span = float(np.linalg.norm(root_seg.max(axis=0) - root_seg.min(axis=0)))
                foot_delta = seg_xz[-1] - seg_xz[0]
                root_delta = root_seg[-1] - root_seg[0]
                denom = float(np.linalg.norm(foot_delta) * np.linalg.norm(root_delta))
                foot_root_cos = float(np.dot(foot_delta, root_delta) / max(denom, 1e-8)) if denom > 1e-8 else 0.0
                rel = seg_xz - root_seg
                root_foot_rel_span = float(np.linalg.norm(rel.max(axis=0) - rel.min(axis=0)))
                root_consistent = bool(
                    root_span >= root_min_travel
                    and foot_root_cos >= direction_cos_min
                    and root_foot_rel_span <= rel_span_max
                )
            else:
                # Backward-compatible fallback when root is unavailable.  It is
                # intentionally conservative: only very efficient, stable motion
                # can use sliding anchor without root evidence.
                root_consistent = bool(path_efficiency > 0.72 and speed_cv <= min(speed_cv_max, 1.0))

            velocity_consistent = bool(mean_speed_mps >= speed_thr and speed_cv <= speed_cv_max)
            is_cloud_step = bool(is_large_contact_travel and velocity_consistent and root_consistent)

            if is_cloud_step:
                # Sliding-Support IK critical fix: use a sliding local-window anchor rather
                # than releasing the target. This preserves intentional support
                # travel while smoothing high-frequency foot jitter.
                sliding_anchor_segments += 1
                for k, t in enumerate(range(start, end)):
                    lo = max(0, k - half)
                    hi = min(length, k + half + 1)
                    local_anchor = seg[lo:hi].mean(axis=0)
                    # Full XYZ mean is used intentionally: XZ follows the slide,
                    # Y is smoothed to avoid contact-height flicker.
                    targets[t, f] = local_anchor
                if len(preview) < 32:
                    preview.append({
                        "foot": int(f), "start": int(start), "end": int(end),
                        "frames": int(length), "mode": "sliding_anchor_cloud_step",
                        "xz_span_m": span, "arc_m": arc,
                        "mean_speed_mps": mean_speed_mps,
                        "speed_cv": speed_cv,
                        "path_efficiency": path_efficiency,
                        "root_span_m": root_span,
                        "root_foot_rel_span_m": root_foot_rel_span,
                        "foot_root_direction_cos": foot_root_cos,
                        "root_consistent": bool(root_consistent),
                        "span_threshold_m": span_thr,
                        "speed_threshold_mps": speed_thr,
                        "root_min_travel_m": root_min_travel,
                        "direction_cos_min": direction_cos_min,
                        "root_foot_rel_max_m": rel_span_max,
                        "window_frames": int(win),
                    })
                continue

            # Static anchor path. This deliberately catches large but slow AI
            # drift instead of forgiving it.
            if is_large_contact_travel:
                dark_drift_locked_segments += 1
                if velocity_consistent and not root_consistent:
                    root_inconsistent_locked_segments += 1

            anchor_end = min(start + 3, end)
            anchor = native_foot[start:anchor_end, f].mean(axis=0)
            locked_segments += 1
            for k, t in enumerate(range(start, end)):
                if bool(getattr(cfg, "ik_hard_contact_lock", True)):
                    targets[t, f] = anchor
                else:
                    phase_in = min(1.0, k / 6.0)
                    phase_out = min(1.0, (end - 1 - t) / 6.0)
                    w = min(float(smoothstep01(phase_in)), float(smoothstep01(phase_out)))
                    targets[t, f] = (1 - w) * native_foot[t, f] + w * anchor
            if len(preview) < 32:
                preview.append({
                    "foot": int(f), "start": int(start), "end": int(end),
                    "frames": int(length), "mode": "locked_footplant",
                    "xz_span_m": span, "arc_m": arc,
                    "mean_speed_mps": mean_speed_mps,
                    "speed_cv": speed_cv,
                    "path_efficiency": path_efficiency,
                    "root_span_m": root_span,
                    "root_foot_rel_span_m": root_foot_rel_span,
                    "foot_root_direction_cos": foot_root_cos,
                    "root_consistent": bool(root_consistent),
                    "velocity_consistent": bool(velocity_consistent),
                    "large_slow_drift_locked": bool(is_large_contact_travel),
                    "root_inconsistent_locked": bool(is_large_contact_travel and velocity_consistent and not root_consistent),
                    "anchor_source": "contact_internal_first_frames",
                })

    diff = np.linalg.norm(targets - native_foot, axis=-1)
    non_contact = ~contacts
    meta = {
        "version": "sliding_anchor_step_target_generator",
        "fixes_footskate_forgiveness_paradox": True,
        "intentional_slide_guard": "root_aware_velocity_classified_sliding_anchor",
        "no_span_only_release": True,
        "fixes_smooth_dark_drift_cloudstep_false_positive": True,
        "cloud_step_speed_threshold_mps": float(speed_thr),
        "slide_span_threshold_m": float(span_thr),
        "sliding_anchor_window_frames": int(win),
        "cloud_speed_cv_max": float(speed_cv_max),
        "cloud_root_min_travel_m": float(root_min_travel),
        "cloud_direction_cos_min": float(direction_cos_min),
        "cloud_root_foot_rel_max_m": float(rel_span_max),
        "root_aware_guard_enabled": bool(root_xz is not None),
        "locked_segments": int(locked_segments),
        "sliding_anchor_segments": int(sliding_anchor_segments),
        "dark_drift_locked_segments": int(dark_drift_locked_segments),
        "root_inconsistent_locked_segments": int(root_inconsistent_locked_segments),
        "released_slide_segments": 0,
        "skipped_short_segments": int(skipped_short),
        "non_contact_diff_max": float(diff[non_contact].max()) if non_contact.any() else 0.0,
        "contact_diff_p95": float(np.percentile(diff[contacts], 95)) if contacts.any() else 0.0,
        "preview": preview,
    }
    return targets.astype(np.float32), meta


def true_lower_body_ik(motion: np.ndarray, cfg: MotionGenerationConfig) -> Tuple[np.ndarray, dict]:
    if torch is None:
        return motion, {"enabled": False, "reason": "torch_unavailable"}
    contacts0, _, _, _ = derive_contacts_np(motion, cfg)
    motion_base, root_y_report = apply_root_y_c1_physics_np(motion, contacts0, cfg)
    contacts, conf, floor_y, native_foot = derive_contacts_np(motion_base, cfg)
    contact_ramps = contact_ramp_weights_np(
        contacts,
        fps=float(cfg.fps),
        ramp_seconds=float(cfg.ik_contact_ramp_seconds),
    )
    targets, target_meta = generate_ik_targets_np(native_foot, contacts, cfg, root_xz=motion_base[:, [ROOT_X_IDX, ROOT_Z_IDX]])
    device = torch.device(cfg.device)
    out_all = motion_base.copy().astype(np.float32)
    reports = []
    T = motion.shape[0]
    chunk = int(cfg.ik_chunk)
    overlap = max(0, int(cfg.ik_chunk_overlap))
    stride = max(1, chunk - overlap)
    # Root-Aware Motion Safety: independent chunk solves are merged by weighted accumulation,
    # rather than half-overlap overwrite.  This avoids long-sequence IK seams at
    # chunk boundaries and preserves every overlapping frame as a blend.
    starts = list(range(0, T, stride))
    accum = np.zeros_like(out_all, dtype=np.float32)
    weight_sum = np.zeros((T, 1), dtype=np.float32)
    lower_idx = torch.as_tensor(
        LOWER_BODY_JOINTS,
        device=device,
        dtype=torch.long,
    )
    floor = torch.tensor(floor_y, device=device, dtype=torch.float32)
    for st in starts:
        ed = min(T, st + chunk)
        if ed - st < 4:
            continue
        base_np = motion_base[st:ed].copy()
        L = base_np.shape[0]
        base = torch.from_numpy(base_np).float().to(device)
        rot_full = base[:, ROT6D_START:ROT6D_END].reshape(L, NUM_JOINTS, 6).detach().clone()
        root = base[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].detach().clone().requires_grad_(True)
        lower_rot = rot_full[:, lower_idx].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([lower_rot, root], lr=cfg.ik_lr)
        target = torch.from_numpy(targets[st:ed]).float().to(device)
        contact = torch.from_numpy(contacts[st:ed].astype(np.float32)).float().to(device)
        contact_ramp = torch.from_numpy(contact_ramps[st:ed]).float().to(device)
        confidence = torch.from_numpy(conf[st:ed]).float().to(device)
        base_rot = rot_full[:, lower_idx].detach().clone()
        base_root = root.detach().clone()
        best_loss_device = torch.full(
            (),
            float("inf"),
            dtype=base.dtype,
            device=device,
        )
        best_motion_device = base.detach().clone()
        for it in range(int(cfg.ik_iters)):
            rr = project_rot6d_torch(lower_rot)
            rr = base_rot + torch.clamp(rr - base_rot, -cfg.ik_max_delta_rot, cfg.ik_max_delta_rot)
            rot = rot_full.clone()
            rot[:, lower_idx] = rr
            mm = base.clone()
            mm[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root
            mm[:, ROT6D_START:ROT6D_END] = rot.reshape(L, -1)
            joints = fk_24_torch(mm)
            foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
            minimum_confidence = float(
                getattr(cfg, "ik_hard_contact_min_confidence", 0.85)
            )
            effective_confidence = torch.where(
                contact > 0,
                torch.clamp(confidence, min=minimum_confidence),
                confidence,
            )
            w = (contact * contact_ramp * effective_confidence).unsqueeze(-1)
            foot_loss = ((foot - target) ** 2 * w).sum() / w.sum().clamp_min(1.0)
            pose_loss = F.smooth_l1_loss(rr, base_rot)
            if L > 1:
                vel_loss = F.smooth_l1_loss(rr[1:] - rr[:-1], base_rot[1:] - base_rot[:-1])
                root_vel = F.smooth_l1_loss(root[1:] - root[:-1], base_root[1:] - base_root[:-1])
            else:
                vel_loss = torch.tensor(0.0, device=device)
                root_vel = torch.tensor(0.0, device=device)
            pen = F.relu(floor + 0.003 - foot[..., 1]).pow(2).mean()
            root_loss = F.smooth_l1_loss(root, base_root) + root_vel
            loss = cfg.ik_contact_w * foot_loss + cfg.ik_pose_w * pose_loss + cfg.ik_temporal_w * vel_loss + cfg.ik_root_w * root_loss + cfg.ik_penetration_w * pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([lower_rot, root], 1.0)
            opt.step()
            with torch.no_grad():
                root.copy_(
                    base_root
                    + torch.clamp(
                        root - base_root,
                        -float(cfg.rollback_root_delta_max_m),
                        float(cfg.rollback_root_delta_max_m),
                    )
                )
            # Keep best-state selection entirely on the accelerator.  The old
            # implementation synchronized the GPU and copied [L,151] back to
            # the host on nearly every improving iteration.
            detached_loss = loss.detach()
            improved = torch.isfinite(detached_loss) & (
                detached_loss < best_loss_device
            )
            best_motion_device = torch.where(
                improved,
                mm.detach(),
                best_motion_device,
            )
            best_loss_device = torch.where(
                improved,
                detached_loss,
                best_loss_device,
            )
        best_loss = float(best_loss_device.cpu())
        best_motion = (
            best_motion_device.cpu().numpy()
            if np.isfinite(best_loss)
            else None
        )
        if best_motion is not None:
            weight = np.ones((L, 1), dtype=np.float32)
            ov = min(overlap, L // 2 if L > 1 else 0)
            if ov > 1 and st > 0:
                weight[:ov, 0] *= np.linspace(0.0, 1.0, ov, dtype=np.float32)
            if ov > 1 and ed < T:
                weight[-ov:, 0] *= np.linspace(1.0, 0.0, ov, dtype=np.float32)
            # Avoid exact zero-only coverage on pathological tiny chunks.
            weight = np.maximum(weight, 1e-4)
            accum[st:ed] += best_motion.astype(np.float32) * weight
            weight_sum[st:ed] += weight
        reports.append({"start": int(st), "end": int(ed), "best_loss": float(best_loss), "contact_ratio": float(contacts[st:ed].mean())})
    valid = weight_sum[:, 0] > 1e-8
    out_all = motion_base.copy().astype(np.float32)
    out_all[valid] = accum[valid] / weight_sum[valid]
    # Re-orthogonalize all rotation channels after optimization.
    if torch is not None:
        with torch.no_grad():
            x = torch.from_numpy(
                out_all[:, ROT6D_START:ROT6D_END].reshape(
                    T,
                    NUM_JOINTS,
                    6,
                )
            ).float().to(device)
            out_all[:, ROT6D_START:ROT6D_END] = (
                project_rot6d_torch(x).cpu().numpy().reshape(T, -1)
            )
    post_stabilize_report: Dict[str, Any] = {
        "enabled": bool(cfg.ik_post_stabilize_enable),
        "applied": False,
        "passes": 0,
        "kernel": [0.0625, 0.25, 0.375, 0.25, 0.0625],
        "scope": "root_xyz_and_lower_body_rot6d",
    }
    if bool(cfg.ik_post_stabilize_enable) and T >= 5:
        candidate_before_stabilize = out_all.copy()
        audit_before_stabilize = audit_motion_np(candidate_before_stabilize, cfg)
        kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0

        def binomial_filter_time(values: np.ndarray) -> np.ndarray:
            pad = len(kernel) // 2
            padded = np.pad(
                values,
                [(pad, pad)] + [(0, 0)] * (values.ndim - 1),
                mode="edge",
            )
            filtered = np.zeros_like(values, dtype=np.float32)
            for offset, weight in enumerate(kernel):
                filtered += float(weight) * padded[offset:offset + len(values)]
            return filtered

        stabilized = candidate_before_stabilize.copy()
        rotations = stabilized[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)
        lower = np.asarray(LOWER_BODY_JOINTS, dtype=np.int64)
        passes = max(0, int(cfg.ik_post_stabilize_passes))
        for _ in range(passes):
            stabilized[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = binomial_filter_time(
                stabilized[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
            )
            rotations[:, lower] = binomial_filter_time(rotations[:, lower])
            rotations[:, lower] = matrix_to_rot6d_np(
                rot6d_to_matrix_np(rotations[:, lower])
            )
        stabilized[:, ROT6D_START:ROT6D_END] = rotations.reshape(T, -1)
        audit_stabilized = audit_motion_np(stabilized, cfg)
        stabilization_decision = evaluate_stage_candidate(
            audit_before_stabilize,
            audit_stabilized,
            limits=PhysicalQualityLimits.from_environment(),
            policy=StageAcceptancePolicy.from_environment(),
            require_repair_gain=False,
        )
        stabilization_safe = bool(stabilization_decision["accepted"])
        if passes > 0:
            # Local transaction gates below decide which ownership windows may
            # commit.  Keep the lower-jerk stabilized proposal available even
            # when a different global window prevents a whole-song commit.
            out_all = stabilized
        post_stabilize_report.update({
            "applied": bool(passes > 0),
            "globally_safe": bool(stabilization_safe),
            "commit_scope": "local_transaction_candidate",
            "passes": int(passes),
            "safe": bool(stabilization_safe),
            "stage_decision": stabilization_decision,
            "audit_before": audit_before_stabilize,
            "audit_candidate": audit_stabilized,
        })
    audit_before = audit_motion_np(motion, cfg)
    audit_after = audit_motion_np(out_all, cfg)
    root_delta = np.linalg.norm(out_all[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]], axis=1)
    ik_limits = PhysicalQualityLimits.from_environment()
    ik_policy = StageAcceptancePolicy.from_environment()

    def transaction_reasons(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        maximum_root_delta: float,
    ) -> Tuple[List[str], List[str]]:
        decision = evaluate_stage_candidate(
            before,
            after,
            limits=ik_limits,
            policy=ik_policy,
            require_repair_gain=False,
        )
        absolute_prefixes = (
            "absolute_",
            "candidate_missing_or_",
            "candidate_missing_or_invalid_schema",
        )
        relative = [
            str(reason)
            for reason in decision["reasons"]
            if not str(reason).startswith(absolute_prefixes)
        ]
        absolute = [
            str(reason)
            for reason in decision["reasons"]
            if str(reason).startswith(absolute_prefixes)
        ]
        if maximum_root_delta > float(cfg.rollback_root_delta_max_m):
            absolute.append("absolute_root_delta")
        return relative, absolute

    rollback_reasons, absolute_commit_reasons = transaction_reasons(
        audit_before,
        audit_after,
        float(root_delta.max()) if root_delta.size else 0.0,
    )

    # Commit IK by non-overlapping ownership windows.  A bad solve in one
    # support interval must not discard improvements in unrelated intervals.
    final = np.asarray(motion, dtype=np.float32).copy()
    transaction_reports: List[Dict[str, Any]] = []
    accepted_transactions = 0
    solved_ranges = [
        (int(report["start"]), int(report["end"]))
        for report in reports
        if int(report["end"]) - int(report["start"]) >= 4
    ]
    for transaction_index, (start, end) in enumerate(solved_ranges):
        own_start = start
        own_end = end
        if transaction_index > 0:
            own_start = min(own_end, own_start + overlap // 2)
        if transaction_index + 1 < len(solved_ranges) and own_end < T:
            own_end = max(
                own_start,
                own_end - (overlap - overlap // 2),
            )
        if own_end - own_start < 4:
            continue
        has_contact = bool(np.any(contacts[own_start:own_end]))
        if not has_contact:
            transaction_reports.append(
                {
                    "start": int(own_start),
                    "end": int(own_end),
                    "committed": False,
                    "reason": "no_contact_in_ownership_window",
                }
            )
            continue

        fade = min(
            max(2, int(round(3.0 * float(cfg.fps) / 30.0))),
            max(2, (own_end - own_start) // 4),
        )
        weight = np.ones((own_end - own_start, 1), dtype=np.float32)
        if own_start > 0:
            weight[:fade, 0] = smoothstep01(
                np.linspace(0.0, 1.0, fade, dtype=np.float32)
            )
        if own_end < T:
            weight[-fade:, 0] = np.minimum(
                weight[-fade:, 0],
                smoothstep01(
                    np.linspace(1.0, 0.0, fade, dtype=np.float32)
                ),
            )

        trial = final.copy()
        trial[own_start:own_end] = blend_edge151_geodesic_np(
            final[own_start:own_end],
            out_all[own_start:own_end],
            weight,
        )
        halo = max(4, int(round(6.0 * float(cfg.fps) / 30.0)))
        audit_start = max(0, own_start - halo)
        audit_end = min(T, own_end + halo)
        before_local = audit_motion_np(final[audit_start:audit_end], cfg)
        after_local = audit_motion_np(trial[audit_start:audit_end], cfg)
        local_root_delta = np.linalg.norm(
            trial[own_start:own_end, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
            - final[own_start:own_end, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]],
            axis=1,
        )
        relative_reasons, absolute_reasons = transaction_reasons(
            before_local,
            after_local,
            float(local_root_delta.max()) if local_root_delta.size else 0.0,
        )
        kbo_reasons: List[str] = []
        kbo_detail: Dict[str, Any] = {}
        if (
            not relative_reasons
            and not absolute_reasons
            and callable(globals().get("_kinematic_barrier_oracle"))
        ):
            kbo_ok, kbo_reasons, kbo_detail = globals()["_kinematic_barrier_oracle"](
                trial[audit_start:audit_end],
                final[audit_start:audit_end],
                cfg,
                stage="ik_local_transaction",
                global_start=int(audit_start),
            )
            if not kbo_ok and not kbo_reasons:
                kbo_reasons = ["local_kbo_rejected"]
        committed = (
            not relative_reasons
            and not absolute_reasons
            and not kbo_reasons
        )
        if committed:
            final = trial
            accepted_transactions += 1
        transaction_reports.append(
            {
                "start": int(own_start),
                "end": int(own_end),
                "audit_start": int(audit_start),
                "audit_end": int(audit_end),
                "committed": bool(committed),
                "relative_reasons": relative_reasons,
                "absolute_reasons": absolute_reasons,
                "kbo_reasons": kbo_reasons,
                "kbo_detail": kbo_detail,
                "root_delta_max_m": float(
                    local_root_delta.max()
                    if local_root_delta.size
                    else 0.0
                ),
                "audit_before": before_local,
                "audit_after": after_local,
            }
        )
    rollback = bool(solved_ranges and accepted_transactions == 0)
    # Contacts are an observation of the final FK state.  Never retain stale
    # logits from the pre-IK/refiner motion after geometry has changed.
    final_contacts, final_confidence, final_floor_y, _ = derive_contacts_np(final, cfg)
    final = final.copy().astype(np.float32)
    final[:, :4] = final_contacts.astype(np.float32)
    report = {
        "version": "lower_body_ik_contact_transactions",
        "enabled": True,
        "writes_lower_body_rot6d": True,
        "root_y_physics": root_y_report,
        "ik_target_generator": target_meta,
        "post_ik_stabilization": post_stabilize_report,
        "lower_body_joints": list(map(int, LOWER_BODY_JOINTS)),
        "foot_joint_ids": list(map(int, DEFAULT_FOOT_JOINTS)),
        "floor_y": float(floor_y),
        "contact_ratio": float(contacts.mean()),
        "contact_ramp": {
            "seconds": float(cfg.ik_contact_ramp_seconds),
            "frames": int(
                max(
                    1,
                    round(
                        float(cfg.ik_contact_ramp_seconds) * float(cfg.fps)
                    ),
                )
            ),
            "mean_weight": float(contact_ramps.mean()),
        },
        "chunks": reports,
        "chunk_stitching": {
            "mode": "weighted_accumulation",
            "chunk": int(chunk),
            "overlap": int(overlap),
            "stride": int(stride),
            "coverage_min": float(weight_sum[:, 0].min()) if weight_sum.size else 0.0,
            "coverage_p95": float(np.percentile(weight_sum[:, 0], 95)) if weight_sum.size else 0.0,
        },
        "rollback_policy": {
            "mode": "local_ownership_window_transactions",
            "physical_metric_registry": "contracts.physical_quality.physical_metric_specs",
            "root_delta_max_m": float(cfg.rollback_root_delta_max_m),
        },
        "local_transactions": {
            "attempted": int(len(transaction_reports)),
            "accepted": int(accepted_transactions),
            "rejected": int(
                sum(
                    1
                    for transaction in transaction_reports
                    if not transaction.get("committed", False)
                )
            ),
            "ownership_overlap_frames": int(overlap),
            "transactions": transaction_reports,
        },
        "root_delta_max_m": float(root_delta.max()) if root_delta.size else 0.0,
        "root_delta_p95_m": float(np.percentile(root_delta, 95)) if root_delta.size else 0.0,
        "rollback_reasons": rollback_reasons,
        "absolute_commit_reasons": absolute_commit_reasons,
        "absolute_commit_gate_passed": not absolute_commit_reasons,
        "audit_before": audit_before,
        "audit_after_candidate": audit_after,
        "rollback_triggered": rollback,
        "final_contact_recomputed": True,
        "final_contact_ratio": float(final_contacts.mean()),
        "final_contact_confidence_mean": float(final_confidence.mean()),
        "final_contact_floor_y": float(final_floor_y),
        "audit_final": audit_motion_np(final, cfg),
    }
    return final.astype(np.float32), report


def audit_motion_np(
    motion: np.ndarray,
    cfg: Optional[MotionGenerationConfig] = None,
    *,
    support_policy: str = "final_fail_closed",
    sliding_support_eligible: Optional[np.ndarray] = None,
    precomputed_joints: Optional[np.ndarray] = None,
) -> dict:
    cfg = cfg or MotionGenerationConfig()
    reusable_joints = precomputed_joints
    if reusable_joints is None and np.isfinite(motion).all():
        reusable_joints = fk_24_np(motion)
    contacts, _, _, _ = derive_contacts_np(
        motion,
        cfg,
        precomputed_joints=reusable_joints,
    )
    audited_motion = np.asarray(motion, dtype=np.float32).copy()
    audited_motion[:, :4] = contacts.astype(np.float32)
    from motion_geometry.physical import motion_physical_metrics_np

    report = motion_physical_metrics_np(
        audited_motion,
        fps=float(cfg.fps),
        support_policy=support_policy,
        sliding_support_eligible=sliding_support_eligible,
        precomputed_joints=reusable_joints,
    )
    report["root_y_range_m"] = float(
        np.max(audited_motion[:, ROOT_Y_IDX]) - np.min(audited_motion[:, ROOT_Y_IDX])
    )
    return report


def render_if_possible(
    motion_path: str,
    audio_path: Optional[str],
    output_mp4: Optional[str],
    render_script: str = "rendering/render_motion.py",
    fps: float = 30.0,
) -> None:
    if not output_mp4 or not audio_path:
        return
    if not Path(render_script).exists() or not Path(audio_path).exists():
        print("[Motion Generation WARN] render skipped: render script or audio missing", file=sys.stderr)
        return
    cmd = [
        sys.executable,
        render_script,
        "--motion", motion_path,
        "--audio", audio_path,
        "--output", output_mp4,
        "--fps", str(float(fps)),
        "--camera_mode", "follow",
        "--render_smooth_window", "5",
    ]
    print("[Motion Generation RENDER]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_ik(args: argparse.Namespace) -> int:
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    motion = np.load(args.input).astype(np.float32)
    out_motion, report = true_lower_body_ik(motion, cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out_motion)
    save_json(report, args.json or str(args.output).replace(".npy", ".lower_body_ik_true_ik.json"))
    print(json.dumps({"output": args.output, "audit_final": report.get("audit_final")}, ensure_ascii=False, indent=2))
    return 0


def run_audit(args: argparse.Namespace) -> int:
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    motion = np.load(args.input).astype(np.float32)
    report = audit_motion_np(motion, cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        save_json(report, args.json)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MotionRAG-Diff generation for EDGE 151D")
    p.add_argument("--config", default="configs/motion_model.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("train-refiner", help="Motion Refiner residual Motion Refiner training")
    r.add_argument("--db", required=True)
    r.add_argument("--val_db", required=True, help="Required source-disjoint validation Event-DB used for physical, contract and leakage gates")
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=None)
    r.add_argument(
        "--snapshot_path",
        default=None,
        help="Resume-only atomic training snapshot output (never a formal checkpoint)",
    )
    r.add_argument(
        "--resume_snapshot",
        default=None,
        help="Resume the exact Refiner optimizer/RNG state from this snapshot",
    )
    r.add_argument("--snapshot_every", type=int, default=None)
    r.add_argument(
        "--validation_every",
        type=int,
        default=None,
        help="Run deterministic source-disjoint validation every N steps",
    )
    r.add_argument(
        "--stop_after_steps", type=int, default=None,
        help="Pause a pilot at this absolute step; preserve the full target and resume snapshot, publish nothing",
    )
    r.add_argument(
        "--train_probe_windows", type=int, default=8,
        help="Fixed train-fit diagnostic windows (0 disables); never used to select or accept checkpoints",
    )
    r.set_defaults(func=train_refiner)

    d = sub.add_parser("train-diffusion", help="Motion Generation conditional residual diffusion training")
    d.add_argument("--db", required=True)
    d.add_argument("--val_db", required=True, help="Required source-disjoint validation Event-DB used for physical, contract and leakage gates")
    d.add_argument("--out", required=True)
    d.add_argument("--steps", type=int, default=None)
    d.add_argument("--diffusion_steps", type=int, default=None)
    d.add_argument(
        "--snapshot_path",
        default=None,
        help="Resume-only atomic training snapshot output (never a formal checkpoint)",
    )
    d.add_argument(
        "--resume_snapshot",
        default=None,
        help="Resume the exact Diffusion optimizer/RNG state from this snapshot",
    )
    d.add_argument("--snapshot_every", type=int, default=None)
    d.set_defaults(func=train_diffusion)

    ik = sub.add_parser("ik", help="Run Lower-Body IK true lower-body IK on an existing EDGE 151D npy")
    ik.add_argument("--input", required=True)
    ik.add_argument("--output", required=True)
    ik.add_argument("--json", default=None)
    ik.set_defaults(func=run_ik)

    a = sub.add_parser("audit", help="Audit EDGE 151D foot skate, floor penetration and jerk")
    a.add_argument("--input", required=True)
    a.add_argument("--json", default=None)
    a.set_defaults(func=run_audit)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))





















# ===== Stage Guard STAGE-ANCHORED GUIDED TGT PATCH START =====
# Stage Guard: Macroscopic Stage Anchoring + KBO-guided Temporal Generative Transactions.
# This layer is intentionally generation-time only. It preserves the Semantic Routing
# MSSD/AESD routing objective and protects Motion Refiner and Motion Diffusion/IK from long-horizon drift.

_stage_guard_orig_concat_events = concat_events
_stage_guard_orig_apply_refiner_model = apply_refiner_model
_stage_guard_orig_apply_diffusion_model = apply_diffusion_model
_stage_guard_orig_true_lower_body_ik = true_lower_body_ik

_STAGE_TRANSACTION_AUDIT = []
_STAGE_PRIOR_XZ = None
_STAGE_PRIOR_METADATA = {}


def _stage_guard_env_bool(name, default=True):
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _stage_guard_env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _stage_guard_env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _stage_guard_jsonable(x):
    try:
        return _json_safe(x)
    except Exception:
        if isinstance(x, dict):
            return {str(k): _stage_guard_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_stage_guard_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _reset_stage_transaction_audit():
    global _STAGE_TRANSACTION_AUDIT
    _STAGE_TRANSACTION_AUDIT = []


def _append_stage_transaction_audit(item):
    global _STAGE_TRANSACTION_AUDIT
    if len(_STAGE_TRANSACTION_AUDIT) < _stage_guard_env_int("STAGE_GUARD_AUDIT_MAX_RECORDS", 4000):
        _STAGE_TRANSACTION_AUDIT.append(_stage_guard_jsonable(dict(item)))


def _stage_guard_torch_load(path, map_location=None):
    if torch is None:
        raise RuntimeError("PyTorch is required")
    if "_trusted_torch_load" in globals():
        return _trusted_torch_load(path, map_location=map_location)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _build_stage_prior_xz(motion, target_durations=None, cfg=None):
    """Build a low-frequency root-XZ anchor prior from the retrieved motion.

    The prior is conservative: it keeps the macro route but pulls it back into a
    bounded stage radius.  It is intentionally not a learned generator here;
    the MSSD slot durations provide the temporal scaffold, while the retrieved
    motion supplies the cultural trajectory skeleton.
    """
    m = np.asarray(motion, dtype=np.float32)
    T = int(m.shape[0])
    if T <= 1:
        return np.zeros((T, 2), dtype=np.float32), {"enabled": False, "reason": "too_short"}
    xz = m[:, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    center = np.median(xz, axis=0, keepdims=True).astype(np.float32)
    rel = xz - center
    if ndi is not None:
        sigma = max(8.0, T / max(8.0, _stage_guard_env_float("STAGE_GUARD_MSA_SMOOTH_DIV", 72.0)))
        rel_s = ndi.gaussian_filter1d(rel, sigma=float(sigma), axis=0, mode="nearest")
    else:
        rel_s = rel
    radius = _stage_guard_env_float("STAGE_GUARD_RADIUS_M", 1.80)
    norm = np.linalg.norm(rel_s, axis=1, keepdims=True)
    rel_clamped = rel_s * np.minimum(1.0, radius / np.maximum(norm, 1e-6))
    prior = (center + rel_clamped).astype(np.float32)
    meta = {
        "enabled": True,
        "version": "stage_guard_macroscopic_stage_anchoring",
        "stage_radius_m": float(radius),
        "prior_root_xz_range_before": (xz.max(axis=0) - xz.min(axis=0)).tolist(),
        "prior_root_xz_range_after": (prior.max(axis=0) - prior.min(axis=0)).tolist(),
        "num_target_durations": int(len(target_durations) if target_durations is not None else 0),
    }
    return prior, meta




def concat_events(event_paths, target_durations, cfg):
    global _STAGE_PRIOR_XZ, _STAGE_PRIOR_METADATA
    motion, rep = _stage_guard_orig_concat_events(event_paths, target_durations, cfg)
    if _stage_guard_env_bool("STAGE_GUARD_MSA_ENABLE", True):
        _STAGE_PRIOR_XZ, _STAGE_PRIOR_METADATA = _build_stage_prior_xz(motion, target_durations, cfg)
        motion2, meta = _apply_guarded_stage_prior(motion, cfg, strength=_stage_guard_env_float("STAGE_GUARD_MSA_REFERENCE_STRENGTH", 0.10))
        _STAGE_PRIOR_METADATA.update(meta)
        if isinstance(rep, list) and rep:
            rep[-1].setdefault("stage_guard_macroscopic_stage_anchor", _STAGE_PRIOR_METADATA)
        _append_stage_transaction_audit({"mechanism": "MSA", "stage": "concat", "commit_state": "anchor_applied", "meta": _STAGE_PRIOR_METADATA})
        return motion2.astype(np.float32), rep
    return motion, rep


def _kinematic_stability_metrics(motion, cfg):
    """Return KBO statistics using the same true frame-joint SI metrics as final audit."""
    m = np.asarray(motion, dtype=np.float32)
    stats = {"finite": bool(np.isfinite(m).all()), "shape": list(m.shape)}
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < EDGE_DIM:
        stats["valid"] = False
        return stats
    joints = None
    try:
        joints = fk_24_np(m)
        stats["fk_finite"] = bool(np.isfinite(joints).all())
        foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
        foot_y = foot[..., 1]
        stats["floor_y"] = float(np.percentile(foot_y.reshape(-1), 5))
        stats["foot_penetration_min_m"] = float(
            np.min(foot_y - stats["floor_y"])
        )
        stats.update(
            compute_joint_kinematic_metrics(joints, fps=float(cfg.fps))
        )
        bone_vars = []
        for joint_id in range(1, min(NUM_JOINTS, len(PARENTS))):
            parent_id = int(PARENTS[joint_id])
            if parent_id < 0 or parent_id >= NUM_JOINTS:
                continue
            lengths = np.linalg.norm(
                joints[:, joint_id] - joints[:, parent_id], axis=-1
            )
            bone_vars.append(
                float(np.max(np.abs(lengths - np.median(lengths))))
            )
        stats["bone_length_violation_max_m"] = float(
            max(bone_vars) if bone_vars else 0.0
        )
    except Exception as exc:
        stats["fk_finite"] = False
        stats["fk_error"] = str(exc)
    try:
        stats.update(audit_motion_np(m, cfg, precomputed_joints=joints))
    except Exception as exc:
        stats["audit_error"] = str(exc)
    stats["root_y_range_m"] = float(
        np.max(m[:, ROOT_Y_IDX]) - np.min(m[:, ROOT_Y_IDX])
    )
    xz = m[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    stats["root_xz_radius_p95_m"] = float(
        np.percentile(
            np.linalg.norm(
                xz - np.median(xz, axis=0, keepdims=True), axis=-1
            ),
            95,
        )
    )
    stats["valid"] = True
    return stats




def _kinematic_barrier_oracle(candidate, reference, cfg, stage="stage", global_start=0):
    """Kinematic barrier oracle aligned with the final SI physical gate."""
    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if cand.shape != ref.shape:
        return False, ["shape_changed"], {
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }

    candidate_stats = _kinematic_stability_metrics(cand, cfg)
    reference_stats = _kinematic_stability_metrics(ref, cfg)
    reasons = []

    if not candidate_stats.get("finite", False) or not candidate_stats.get(
        "fk_finite", False
    ):
        reasons.append("nan_or_inf_or_fk_invalid")

    if float(candidate_stats.get("bone_length_violation_max_m", 0.0)) > _stage_guard_env_float(
        "PHYSICAL_STAGE_BONE_LENGTH_EPS_M", 0.02
    ):
        reasons.append("bone_length_violation")

    if abs(
        float(candidate_stats.get("floor_y", 0.0))
        - float(reference_stats.get("floor_y", 0.0))
    ) > _stage_guard_env_float("PHYSICAL_STAGE_FLOOR_SHIFT_MAX_M", 1.50):
        reasons.append("floor_shift_exceeded")

    if float(candidate_stats.get("joint_acceleration_mps2_max", 0.0)) > _stage_guard_env_float(
        "PHYSICAL_STAGE_ACCELERATION_MAX_MPS2", 2700.0
    ):
        reasons.append("acceleration_spike")

    limits = PhysicalQualityLimits.from_environment()
    stage_key = str(stage).strip().lower()

    if stage_key.startswith("ik_"):
        # IK has its own local ownership-window transaction checks, including
        # relative skate/jerk constraints, penetration control and root-delta
        # limits.  KBO must therefore apply only the authoritative absolute SI
        # gate to IK candidates; reapplying the neural-stage relative policy
        # would reject physically valid IK trade-offs twice.
        decision = evaluate_physical_audit(
            candidate_stats,
            limits=limits,
        )
    else:
        # Neural repair stages must not regress relative to their input.
        decision = evaluate_stage_candidate(
            reference_stats,
            candidate_stats,
            limits=limits,
            policy=StageAcceptancePolicy.from_environment(),
            require_repair_gain=False,
        )

    reasons.extend(decision["reasons"])

    if _stage_guard_env_bool("STAGE_GUARD_KBO_STAGE_ANCHOR_ENABLE", True):
        anchor_error = _stage_anchor_error(cand, global_start)
        if anchor_error > _stage_guard_env_float(
            "STAGE_GUARD_KBO_ANCHOR_P95_MAX_M", 0.85
        ):
            reasons.append("stage_anchor_deviation")
        candidate_stats["stage_anchor_error_p95_m"] = anchor_error

    reasons = list(dict.fromkeys(reasons))
    detail = {
        "candidate": candidate_stats,
        "reference": reference_stats,
        "stage": stage,
        "global_start": int(global_start),
        "stage_decision": decision,
    }
    return len(reasons) == 0, reasons, detail


def _save_hard_negative_pair(stage, tx_id, snapshot, rejected, accepted, reasons, global_span):
    if not _stage_guard_env_bool("STAGE_GUARD_HN_DPO_SAVE_PAIRS", True):
        return {}
    root = Path(os.environ.get("STAGE_GUARD_HN_DPO_DIR", "output/stage_guard_hn_dpo_pairs"))
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{stage}_tx{int(tx_id):04d}_{int(time.time()*1000)}"
    snap_p = root / f"{tag}_snapshot.npy"
    rej_p = root / f"{tag}_rejected.npy"
    acc_p = root / f"{tag}_accepted.npy"
    np.save(snap_p, np.asarray(snapshot, dtype=np.float32))
    np.save(rej_p, np.asarray(rejected, dtype=np.float32))
    np.save(acc_p, np.asarray(accepted, dtype=np.float32))
    meta = {"stage": stage, "transaction_id": int(tx_id), "span": list(map(int, global_span)), "snapshot": str(snap_p), "rejected": str(rej_p), "accepted": str(acc_p), "reasons": list(map(str, reasons))}
    with open(root / "pairs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(_stage_guard_jsonable(meta), ensure_ascii=False) + "\n")
    return meta






def _repair_regions(seam_mask, T):
    sm = np.asarray(seam_mask, dtype=np.float32)
    if sm.ndim == 1:
        sm = sm[:, None]
    active = sm[:, 0] > _stage_guard_env_float("STAGE_GUARD_TGT_ACTIVE_THRESHOLD", 0.05)
    raw = contiguous_regions(active)
    if not raw:
        return []
    halo = _stage_guard_env_int("STAGE_GUARD_TGT_HALO", 12)
    min_len = _stage_guard_env_int("STAGE_GUARD_TGT_MIN_FRAMES", 16)
    max_len = _stage_guard_env_int("STAGE_GUARD_TGT_MAX_FRAMES", 96)
    out = []
    for a, b in raw:
        a = max(0, int(a) - halo); b = min(int(T), int(b) + halo)
        if b - a < min_len:
            mid = (a + b) // 2
            a = max(0, mid - min_len // 2)
            b = min(int(T), a + min_len)
            a = max(0, b - min_len)
        while b - a > max_len:
            out.append((a, a + max_len))
            a = a + max_len - halo
        out.append((a, b))
    out.sort()
    merged = []
    for a, b in out:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(int(a), int(b)) for a, b in merged if int(b) > int(a)]




def _apply_guarded_stage(stage, orig_func, motion, cond, seam_mask, ckpt_path, cfg):
    preserve_neural_contacts = False
    if torch is not None and ckpt_path and Path(ckpt_path).exists():
        try:
            checkpoint_meta = _stage_guard_torch_load(
                ckpt_path, map_location="cpu"
            )
            checkpoint_version = str(checkpoint_meta.get("version", ""))
            preserve_neural_contacts = bool(
                (
                    stage == "refiner"
                    and checkpoint_version.startswith(
                        "product_manifold_boundary_refiner"
                    )
                )
                or (
                    stage == "diffusion"
                    and checkpoint_version.startswith(
                        "reference_tangent_motion_diffusion"
                    )
                )
            )
            del checkpoint_meta
        except Exception:
            # The actual model loader below remains the authority and will
            # surface invalid checkpoints; this metadata probe is optional.
            preserve_neural_contacts = False
    if not _stage_guard_env_bool("STAGE_GUARD_TGT_ENABLE", True):
        cand = orig_func(motion, cond, seam_mask, ckpt_path, cfg)
        return _bounded_residual_update(
            cand,
            motion,
            seam_mask,
            cfg,
            stage=stage,
            global_start=0,
            preserve_contacts=preserve_neural_contacts,
        )
    ref_all = np.asarray(motion, dtype=np.float32)
    out = ref_all.copy().astype(np.float32)
    regions = _repair_regions(seam_mask, ref_all.shape[0])
    if not regions:
        _append_stage_transaction_audit({"mechanism": "TGT", "stage": stage, "event": "no_transaction_regions", "commit_state": "return_reference"})
        return out.astype(np.float32)
    for tx_id, (a, b) in enumerate(regions):
        snapshot = out[a:b].copy().astype(np.float32)
        sm_win = np.asarray(seam_mask[a:b], dtype=np.float32).copy()
        cond_win = _condition_chunk_np(cond, a, b)
        token = {"mechanism": "TGT+KBO", "stage": stage, "temporal_transaction_id": int(tx_id), "atomic_window": [int(a), int(b)], "frames": int(b-a), "commit_state": "pending"}
        rejected_candidate = None
        try:
            if stage == "diffusion" and _stage_guard_env_bool("STAGE_GUARD_DIFFUSION_EARLY_ABORT_ENABLE", True):
                cand = _diffusion_window_proposal(snapshot.copy(), cond_win, sm_win, ckpt_path, cfg, global_start=a)
            else:
                cand = orig_func(snapshot.copy(), cond_win, sm_win, ckpt_path, cfg)
            rejected_candidate = np.asarray(cand, dtype=np.float32)
            cand = _bounded_residual_update(
                cand,
                snapshot,
                sm_win,
                cfg,
                stage=stage,
                global_start=a,
                preserve_contacts=preserve_neural_contacts,
                validate_barrier=False,
            )
            ok, reasons, detail = _kinematic_barrier_oracle(cand, snapshot, cfg, stage=f"{stage}_neural_commit", global_start=a)
            if ok:
                out[a:b] = cand.astype(np.float32)
                token.update({"commit_state": "committed", "fallback_level": "neural_bounded_commit", "kbo_status": "pass", "hard_negative": False})
            else:
                token.update({"commit_state": "neural_rejected", "kbo_status": "fail", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
                raise RuntimeError("kbo_reject:" + ",".join(reasons))
        except Exception as exc:
            token["neural_exception"] = str(exc)[:500]
            fb, fb_report = _deterministic_repair_bridge(snapshot, sm_win, cfg, stage=stage, global_start=a)
            if fb_report.get("committed"):
                out[a:b] = fb.astype(np.float32)
                token.update({"commit_state": "committed", "fallback_level": "deterministic_root_rotation_prior", "kbo_status": "fallback_pass", "fallback_report": fb_report, "hard_negative": True})
                if rejected_candidate is not None:
                    token["hn_dpo_pair"] = _save_hard_negative_pair(stage, tx_id, snapshot, rejected_candidate, fb, token.get("barrier_violations", [str(exc)]), [a, b])
            else:
                out[a:b] = snapshot.astype(np.float32)
                token.update({"commit_state": "rolled_back", "fallback_level": "snapshot_rollback", "kbo_status": "fallback_fail", "fallback_report": fb_report, "hard_negative": True})
                if rejected_candidate is not None:
                    token["hn_dpo_pair"] = _save_hard_negative_pair(stage, tx_id, snapshot, rejected_candidate, snapshot, token.get("barrier_violations", [str(exc)]), [a, b])
        _append_stage_transaction_audit(token)
    out, _ = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint=f"stage_guard_tgt_final:{stage}",
        derive_contact=not preserve_neural_contacts,
        project_rot=True,
    )
    out, _ = _apply_guarded_stage_prior(
        out,
        cfg,
        strength=_stage_guard_env_float(
            "STAGE_GUARD_MSA_STAGE_FINAL_STRENGTH", 0.08
        ),
        preserve_contacts=preserve_neural_contacts,
    )
    ok, reasons, detail = _kinematic_barrier_oracle(out, ref_all, cfg, stage=f"{stage}_whole_stage_guard", global_start=0)
    if not ok:
        _append_stage_transaction_audit({"mechanism": "KBO", "stage": stage, "event": "whole_stage_rollback", "commit_state": "rolled_back", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
        return ref_all.astype(np.float32)
    return out.astype(np.float32)


def apply_refiner_model(motion, cond, seam_mask, ckpt_path, cfg):
    return _apply_guarded_stage("refiner", _stage_guard_orig_apply_refiner_model, motion, cond, seam_mask, ckpt_path, cfg)


def apply_diffusion_model(motion, cond, seam_mask, ckpt_path, cfg):
    return _apply_guarded_stage("diffusion", _stage_guard_orig_apply_diffusion_model, motion, cond, seam_mask, ckpt_path, cfg)


def true_lower_body_ik(motion, cfg):
    if not _stage_guard_env_bool("STAGE_GUARD_IK_TGT_ENABLE", True):
        return _stage_guard_orig_true_lower_body_ik(motion, cfg)
    snapshot = np.asarray(motion, dtype=np.float32).copy()
    try:
        out, report = _stage_guard_orig_true_lower_body_ik(snapshot.copy(), cfg)
        local_transactions = dict(report.get("local_transactions", {}))
        local_mode = str(
            report.get("rollback_policy", {}).get("mode", "")
        ) == "local_ownership_window_transactions"
        # Local transactions have already passed physical and KBO checks with
        # derivative halos.  A second whole-song stage prior would modify
        # frames outside those audited ownership windows.
        if not local_mode:
            out, _ = _apply_guarded_stage_prior(
                out,
                cfg,
                strength=_stage_guard_env_float(
                    "STAGE_GUARD_MSA_IK_STRENGTH", 0.04
                ),
            )
        ok, reasons, detail = _kinematic_barrier_oracle(out, snapshot, cfg, stage="ik_final", global_start=0)
        if ok:
            _append_stage_transaction_audit({"mechanism": "IK_TGT", "stage": "ik", "commit_state": "committed", "fallback_level": "ik_commit", "kbo_status": "pass", "frames": int(snapshot.shape[0])})
            return out.astype(np.float32), report
        if local_mode and int(local_transactions.get("accepted", 0)) > 0:
            _append_stage_transaction_audit(
                {
                    "mechanism": "IK_TGT",
                    "stage": "ik",
                    "commit_state": "partially_committed",
                    "fallback_level": "local_transaction_commit",
                    "kbo_status": "global_baseline_still_invalid",
                    "barrier_violations": reasons,
                    "detail": detail,
                    "frames": int(snapshot.shape[0]),
                }
            )
            report = dict(report)
            report["stage_guard_global_kbo_after_local_transactions"] = {
                "ok": False,
                "reasons": reasons,
                "detail": detail,
                "policy": (
                    "retain_individually_audited_local_commits; "
                    "do_not_rollback_unrelated_windows"
                ),
            }
            return out.astype(np.float32), report
        _append_stage_transaction_audit({"mechanism": "IK_TGT", "stage": "ik", "commit_state": "rolled_back", "fallback_level": "fk_snapshot_rollback", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
        try:
            report = dict(report)
            report["stage_guard_ik_rollback_to_fk"] = True
            report["stage_guard_rollback_reasons"] = reasons
        except Exception:
            pass
        return snapshot.astype(np.float32), report
    except Exception as exc:
        _append_stage_transaction_audit({"mechanism": "IK_TGT", "stage": "ik", "commit_state": "rolled_back", "fallback_level": "ik_exception_to_fk", "exception": str(exc)[:500], "hard_negative": True})
        return snapshot.astype(np.float32), {"enabled": True, "stage_guard_ik_exception_to_fk": True, "exception": str(exc)[:500]}


def _summarize_stage_transactions(records):
    out = {"version": "stage_guard_stage_anchored_guided_tgt_kbo", "num_records": int(len(records)), "by_stage": {}, "fallback_counts": {}, "hard_negatives": 0}
    for r in records:
        st = str(r.get("stage", "unknown"))
        out["by_stage"].setdefault(st, {"records": 0, "committed": 0, "rolled_back": 0})
        out["by_stage"][st]["records"] += 1
        cs = str(r.get("commit_state", ""))
        if cs == "committed":
            out["by_stage"][st]["committed"] += 1
        elif cs in ("rolled_back", "neural_rejected"):
            out["by_stage"][st]["rolled_back"] += 1
        fb = str(r.get("fallback_level", r.get("commit_state", "unknown")))
        out["fallback_counts"][fb] = out["fallback_counts"].get(fb, 0) + 1
        if bool(r.get("hard_negative", False)):
            out["hard_negatives"] += 1
    out["stage_anchor"] = _stage_guard_jsonable(_STAGE_PRIOR_METADATA)
    return out


# ===== Stage Guard STAGE-ANCHORED GUIDED TGT PATCH END =====



# ===== Energy Stability STABILITY ALIGNMENT PATCH START =====
# Fixes for Stage Guard scientific loopholes:
# 1) Tweedie jitter false positives: early-abort probe is low-pass filtered and
#    checked with relaxed early thresholds.
# 2) Rubber-band MSA: stage-anchor strength is modulated by MSSD energy/role and
#    local root velocity; high-energy leaps are not over-constrained.
# 3) Audit exposes Energy Stability policy; kinetic HN-DPO is implemented in the separate
#    energy_stability_train_hn_dpo_diffusion.py tool.


_FRAME_STAGE_ANCHOR_WEIGHT = None
_STAGE_WEIGHT_METADATA = {}


def _energy_gate_env_bool(name, default=True):
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _energy_gate_env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _energy_gate_env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _slot_energy_weight(slot):
    """Return anchor weight in [min,max]; lower weight means freer movement."""
    label = " ".join([
        str(slot.get("music_event", "")),
        str(slot.get("role", "")),
        str(slot.get("slot_role", "")),
        str(slot.get("energy_label", "")),
    ]).lower()
    high_words = ("climax", "turn", "percussive", "accent", "footwork", "leap", "jump", "build", "high")
    calm_words = ("calm", "meditative", "pose", "hold", "sustain", "resolution", "intro")
    energy = float(slot.get("energy", slot.get("boundary_accent_strength", 0.0)) or 0.0)
    tension = float(slot.get("tension", 0.0) or 0.0)
    speed = float(slot.get("music_speed_factor", 1.0) or 1.0)
    w = _energy_gate_env_float("ENERGY_STABILITY_MSA_CALM_WEIGHT", 1.0)
    if any(x in label for x in high_words):
        w *= _energy_gate_env_float("ENERGY_STABILITY_MSA_HIGH_ENERGY_SCALE", 0.22)
    elif any(x in label for x in calm_words):
        w *= _energy_gate_env_float("ENERGY_STABILITY_MSA_CALM_SCALE", 1.00)
    # Continuous attenuation from music dynamics.
    dyn = max(0.0, min(1.0, 0.60 * energy + 0.30 * tension + 0.10 * max(0.0, speed - 1.0)))
    w *= (1.0 - dyn * _energy_gate_env_float("ENERGY_STABILITY_MSA_DYNAMIC_ATTENUATION", 0.75))
    return float(np.clip(w, _energy_gate_env_float("ENERGY_STABILITY_MSA_MIN_WEIGHT", 0.05), _energy_gate_env_float("ENERGY_STABILITY_MSA_MAX_WEIGHT", 1.0)))


def _load_stage_anchor_weights(slots_json, total_frames_hint=0):
    global _FRAME_STAGE_ANCHOR_WEIGHT, _STAGE_WEIGHT_METADATA
    _FRAME_STAGE_ANCHOR_WEIGHT = None
    _STAGE_WEIGHT_METADATA = {"enabled": False, "reason": "no_slots_json"}
    if not slots_json:
        return
    try:
        p = Path(slots_json)
        if not p.exists():
            _STAGE_WEIGHT_METADATA = {"enabled": False, "reason": f"missing:{slots_json}"}
            return
        obj = json.load(open(p, "r", encoding="utf-8"))
        slots = obj.get("slots", []) if isinstance(obj, dict) else []
        if not isinstance(slots, list) or not slots:
            _STAGE_WEIGHT_METADATA = {"enabled": False, "reason": "no_slots"}
            return
        total = int(obj.get("total_target_frames", 0) or 0)
        if total <= 0:
            total = max(int(s.get("end_frame", 0) or 0) for s in slots)
        if total <= 0:
            total = int(total_frames_hint or 0)
        if total <= 0:
            _STAGE_WEIGHT_METADATA = {"enabled": False, "reason": "invalid_total_frames"}
            return
        w = np.ones((total,), dtype=np.float32)
        hist = {}
        for i, s in enumerate(slots):
            a = int(s.get("start_frame", 0) or 0)
            b = int(s.get("end_frame", a + int(s.get("target_frames", 0) or 0)) or a)
            if b <= a:
                b = a + int(s.get("target_frames", 1) or 1)
            a = max(0, min(total, a)); b = max(a, min(total, b))
            sw = _slot_energy_weight(s)
            if b > a:
                w[a:b] = sw
            key = str(s.get("music_event", "unknown"))
            hist[key] = hist.get(key, 0) + 1
        if ndi is not None and len(w) > 7:
            w = ndi.gaussian_filter1d(w, sigma=float(_energy_gate_env_float("ENERGY_STABILITY_MSA_WEIGHT_SMOOTH_SIGMA", 3.0)), mode="nearest").astype(np.float32)
        _FRAME_STAGE_ANCHOR_WEIGHT = np.clip(w, _energy_gate_env_float("ENERGY_STABILITY_MSA_MIN_WEIGHT", 0.05), 1.0).astype(np.float32)
        _STAGE_WEIGHT_METADATA = {
            "enabled": True,
            "source": str(p),
            "total_frames": int(total),
            "min": float(np.min(_FRAME_STAGE_ANCHOR_WEIGHT)),
            "mean": float(np.mean(_FRAME_STAGE_ANCHOR_WEIGHT)),
            "p95": float(np.percentile(_FRAME_STAGE_ANCHOR_WEIGHT, 95)),
            "semantic_histogram": hist,
            "interpretation": "lower weights indicate high-energy/climax windows where MSA is relaxed",
        }
    except Exception as exc:
        _FRAME_STAGE_ANCHOR_WEIGHT = None
        _STAGE_WEIGHT_METADATA = {"enabled": False, "reason": str(exc)}


def _frame_stage_weights(T, global_start=0):
    if _FRAME_STAGE_ANCHOR_WEIGHT is None or T <= 0:
        return np.ones((int(T), 1), dtype=np.float32)
    a = int(global_start)
    b = a + int(T)
    if a < 0 or b > len(_FRAME_STAGE_ANCHOR_WEIGHT):
        # Defensive resize for rare report/motion length drifts.
        idx = np.linspace(0, len(_FRAME_STAGE_ANCHOR_WEIGHT) - 1, int(T)).clip(0, len(_FRAME_STAGE_ANCHOR_WEIGHT) - 1).astype(int)
        return _FRAME_STAGE_ANCHOR_WEIGHT[idx, None].astype(np.float32)
    return _FRAME_STAGE_ANCHOR_WEIGHT[a:b, None].astype(np.float32)


def _root_velocity_gate(motion):
    m = np.asarray(motion, dtype=np.float32)
    if m.shape[0] < 3:
        return np.ones((m.shape[0], 1), dtype=np.float32)
    v = np.linalg.norm(np.diff(m[:, [ROOT_X_IDX, ROOT_Z_IDX]], axis=0), axis=-1)
    v = np.concatenate([[v[0]], v]).astype(np.float32)
    thr = _energy_gate_env_float("ENERGY_STABILITY_MSA_ROOT_SPEED_RELAX_THRESH", 0.045)
    if thr <= 0:
        return np.ones((m.shape[0], 1), dtype=np.float32)
    # High root speed means possible leap/large travel; attenuate anchor.
    g = 1.0 / (1.0 + (v / max(thr, 1e-6)) ** 2)
    g = np.clip(g, _energy_gate_env_float("ENERGY_STABILITY_MSA_VELOCITY_MIN_GATE", 0.12), 1.0)
    if ndi is not None and len(g) > 7:
        g = ndi.gaussian_filter1d(g, sigma=2.0, mode="nearest")
    return g[:, None].astype(np.float32)






def _energy_gate_jsonable(x):
    try:
        return _stage_guard_jsonable(x)
    except Exception:
        if isinstance(x, dict):
            return {str(k): _energy_gate_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_energy_gate_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _lowpass_motion_for_barrier_oracle(motion, cfg, sigma=None):
    """Low-pass a Tweedie/intermediate probe before high-order KBO.

    This prevents high-frequency residual noise from creating false positive jerk
    spikes during early-abort checks. The committed sample is not replaced by
    this smoothed probe; smoothing is only for the oracle decision.
    """
    m = np.asarray(motion, dtype=np.float32).copy()
    if sigma is None:
        sigma = _energy_gate_env_float("ENERGY_STABILITY_EARLY_ABORT_KBO_SMOOTH_SIGMA", 1.35)
    if ndi is not None and m.shape[0] > 5 and float(sigma) > 0:
        idx = [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX] + list(range(ROT6D_START, ROT6D_END))
        m[:, idx] = ndi.gaussian_filter1d(m[:, idx], sigma=float(sigma), axis=0, mode="nearest")
    m, _ = enforce_edge151_contract_np(m, cfg, source_hint="energy_stability_lowpass_tweedie_probe", derive_contact=True, project_rot=True)
    return m.astype(np.float32)




def _bounded_residual_update(
    candidate,
    reference,
    seam_mask,
    cfg,
    stage="stage",
    global_start=0,
    preserve_contacts=False,
    validate_barrier=True,
):
    cand = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    if cand.shape != ref.shape:
        return ref.astype(np.float32)
    sm = np.asarray(seam_mask, dtype=np.float32)
    if sm.ndim == 1:
        sm = sm[:, None]
    if sm.shape[0] != ref.shape[0]:
        sm = resample_motion_np(sm, ref.shape[0])
    core = _stage_guard_env_float(f"STAGE_GUARD_{stage.upper()}_CORE_COMMIT", 0.0)
    trans_default = 0.18 if stage == "refiner" else 0.12
    trans = _stage_guard_env_float(f"STAGE_GUARD_{stage.upper()}_TRANSITION_COMMIT", trans_default)
    w = np.clip(core + (trans - core) * sm.astype(np.float32), 0.0, 1.0)
    delta = cand - ref
    bounded = cand.copy().astype(np.float32)
    root_xz_max = _stage_guard_env_float("STAGE_GUARD_ROOT_XZ_DELTA_MAX_M", 0.05)
    root_y_max = _stage_guard_env_float("STAGE_GUARD_ROOT_Y_DELTA_MAX_M", 0.02)
    for idx, mx in [(ROOT_X_IDX, root_xz_max), (ROOT_Y_IDX, root_y_max), (ROOT_Z_IDX, root_xz_max)]:
        bounded[:, idx] = ref[:, idx] + np.clip(delta[:, idx], -mx, mx)
    max_rotation_rad = _stage_guard_env_float(
        "STAGE_GUARD_ROTATION_DELTA_MAX_RAD",
        _stage_guard_env_float("STAGE_GUARD_ROT6D_DELTA_MAX", 0.12),
    )
    out = blend_edge151_geodesic_np(
        ref,
        bounded,
        w,
        max_rotation_rad=max_rotation_rad,
    )
    if preserve_contacts:
        out[:, :4] = (
            ref[:, :4] * (1.0 - w)
            + np.clip(bounded[:, :4], 0.0, 1.0) * w
        )
    out, _ = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint=f"energy_stability_safe_residual:{stage}",
        derive_contact=not preserve_contacts,
        project_rot=True,
    )
    out, _ = _apply_guarded_stage_prior(
        out,
        cfg,
        strength=_stage_guard_env_float(
            "STAGE_GUARD_MSA_TRANSACTION_STRENGTH", 0.08
        ),
        global_start=global_start,
        preserve_contacts=preserve_contacts,
    )
    if validate_barrier:
        ok, reasons, detail = _kinematic_barrier_oracle(out, ref, cfg, stage=f"{stage}_bounded_residual", global_start=global_start)
        if not ok:
            _append_stage_transaction_audit({"mechanism": "KBO", "version": "motion_42", "stage": stage, "event": "bounded_residual_rejected", "barrier_violations": reasons, "detail": detail, "hard_negative": True})
            return ref.astype(np.float32)
    return out.astype(np.float32)


def _deterministic_repair_bridge(reference, seam_mask, cfg, stage="fallback", global_start=0):
    ref = np.asarray(reference, dtype=np.float32).copy()
    if ref.shape[0] < 4:
        return ref.astype(np.float32), {"mode": "snapshot_too_short", "committed": False}
    sm = np.asarray(seam_mask, dtype=np.float32)
    if sm.ndim == 1:
        sm = sm[:, None]
    active = sm[:, 0] > _stage_guard_env_float("STAGE_GUARD_TGT_ACTIVE_THRESHOLD", 0.05)
    regs = contiguous_regions(active)
    if not regs:
        return ref.astype(np.float32), {"mode": "no_active_mask", "committed": False}
    out = ref.copy().astype(np.float32)
    fallback_strength = _stage_guard_env_float("STAGE_GUARD_DETERMINISTIC_FALLBACK_STRENGTH", 0.35)
    reports = []
    for a, b in regs:
        a = max(1, int(a)); b = min(int(b), ref.shape[0] - 1)
        if b - a < 2:
            continue
        n = b - a
        try:
            if "reference_motion_inbetween_np" in globals():
                bridge = reference_motion_inbetween_np(ref[max(0, a-2):a], ref[b:min(ref.shape[0], b+2)], n, cfg)
            else:
                raise RuntimeError("reference_motion_inbetween_np unavailable")
        except Exception:
            left = ref[a - 1].copy(); right = ref[b].copy()
            x = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            cubic = x * x * (3.0 - 2.0 * x)
            bridge = resample_motion_np(np.stack([left, right], axis=0), n)
            bridge[:, ROOT_X_IDX:ROOT_Z_IDX + 1] = (
                (1.0 - cubic) * left[None, ROOT_X_IDX:ROOT_Z_IDX + 1]
                + cubic * right[None, ROOT_X_IDX:ROOT_Z_IDX + 1]
            )
        w = np.clip(sm[a:b], 0.0, 1.0) * float(fallback_strength)
        out[a:b] = blend_edge151_geodesic_np(out[a:b], bridge, w)
        reports.append({"span": [int(a), int(b)], "frames": int(n)})
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"energy_stability_deterministic_bridge:{stage}", derive_contact=True, project_rot=True)
    out, _ = _apply_guarded_stage_prior(out, cfg, strength=_stage_guard_env_float("STAGE_GUARD_MSA_FALLBACK_STRENGTH", 0.10), global_start=global_start)
    ok, reasons, detail = _kinematic_barrier_oracle(out, ref, cfg, stage=f"{stage}_deterministic_bridge", global_start=global_start)
    if not ok:
        return ref.astype(np.float32), {"mode": "deterministic_bridge_rejected", "committed": False, "reasons": reasons, "detail": detail}
    return out.astype(np.float32), {"mode": "deterministic_root_rotation_bridge", "committed": True, "regions": reports, "energy_stability_dynamic_msa": True}




# ===== Energy Stability STABILITY ALIGNMENT PATCH END =====



# ===== Physics Stability PHYSICS-CONSISTENT STABILITY PATCH START =====
# Physics-consistent fixes after Energy Stability.
# Runtime functions below replace the earlier stability implementations by
# global name, but generation itself has one explicit orchestration entrypoint.

_EARLY_ABORT_TRACE = []


def _physics_stability_env_bool(name, default=True):
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _physics_stability_env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _physics_stability_env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _physics_stability_jsonable(x):
    try:
        return _energy_gate_jsonable(x)
    except Exception:
        if isinstance(x, dict):
            return {str(k): _physics_stability_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_physics_stability_jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)


def _lowpass_motion_channels(motion, cfg, sigma=2.25):
    """Only used for oracle decisions, never as committed sample."""
    m = np.asarray(motion, dtype=np.float32).copy()
    if ndi is not None and m.ndim == 2 and m.shape[0] > 7 and float(sigma) > 0:
        idx = [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX] + list(range(ROT6D_START, ROT6D_END))
        m[:, idx] = ndi.gaussian_filter1d(m[:, idx], sigma=float(sigma), axis=0, mode="nearest")
    m, _ = enforce_edge151_contract_np(m, cfg, source_hint="physics_stability_derivative_safe_lowpass_probe", derive_contact=True, project_rot=True)
    return m.astype(np.float32)


def _robust_derivative_metrics(motion, cfg):
    """Robust derivative statistics after low-pass filtering.

    Uses p95/p99 rather than raw max to avoid one-frame Tweedie jitter causing
    false positive early-abort. Raw max remains available for diagnostics only.
    """
    m = np.asarray(motion, dtype=np.float32)
    st = {"finite": bool(np.isfinite(m).all()), "shape": list(m.shape)}
    if m.ndim != 2 or m.shape[0] < 4 or m.shape[1] < EDGE_DIM:
        st["valid"] = False
        return st
    joints = None
    try:
        joints = fk_24_np(m)
        st["fk_finite"] = bool(np.isfinite(joints).all())
        fps = float(cfg.fps)
        acc = np.diff(joints, n=2, axis=0) * fps ** 2
        jerk = np.diff(joints, n=3, axis=0) * fps ** 3
        acc_n = np.linalg.norm(acc, axis=-1).mean(axis=-1) if acc.size else np.zeros((1,), dtype=np.float32)
        jerk_n = np.linalg.norm(jerk, axis=-1).mean(axis=-1) if jerk.size else np.zeros((1,), dtype=np.float32)
        st["joint_acceleration_p95_mps2"] = float(np.percentile(acc_n, 95))
        st["joint_acceleration_p99_mps2"] = float(np.percentile(acc_n, 99))
        st["joint_acceleration_max_mps2_diag"] = float(np.max(acc_n))
        st["joint_jerk_p95_mps3"] = float(np.percentile(jerk_n, 95))
        st["joint_jerk_p99_mps3"] = float(np.percentile(jerk_n, 99))
        st["joint_jerk_max_mps3_diag"] = float(np.max(jerk_n))
        # Bone-length variance should be extremely small for a valid FK skeleton.
        bone_vars = []
        for j in range(1, min(NUM_JOINTS, len(PARENTS))):
            pa = int(PARENTS[j])
            if pa < 0 or pa >= NUM_JOINTS:
                continue
            L = np.linalg.norm(joints[:, j] - joints[:, pa], axis=-1)
            if L.size:
                bone_vars.append(float(np.max(np.abs(L - np.median(L)))))
        st["bone_length_violation_max_m"] = float(max(bone_vars) if bone_vars else 0.0)
    except Exception as exc:
        st["fk_finite"] = False
        st["fk_error"] = str(exc)
    try:
        st.update(audit_motion_np(m, cfg, precomputed_joints=joints))
    except Exception as exc:
        st["audit_error"] = str(exc)
    st["root_y_range_m"] = float(np.max(m[:, ROOT_Y_IDX]) - np.min(m[:, ROOT_Y_IDX])) if m.size else 0.0
    st["valid"] = True
    return st


def _physics_early_abort_oracle(candidate, reference, cfg, stage="diffusion_early_abort_probe", global_start=0):
    """Derivative-safe early-abort oracle.

    It deliberately separates fatal low-frequency barriers from derivative-only
    barriers. A derivative spike on a Tweedie/intermediate probe cannot abort by
    itself, because differentiation amplifies high-frequency noise.
    """
    raw = np.asarray(candidate, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)
    sigma = _physics_stability_env_float("PHYSICS_STABILITY_EARLY_ABORT_LOWPASS_SIGMA", 2.25)
    relax = _physics_stability_env_float("PHYSICS_STABILITY_EARLY_ABORT_RELAX", 4.0)
    smooth = _lowpass_motion_channels(raw, cfg, sigma=sigma)
    c = _robust_derivative_metrics(smooth, cfg)
    r = _robust_derivative_metrics(ref, cfg)
    fatal = []
    soft = []

    if not c.get("finite", False) or not c.get("fk_finite", False):
        fatal.append("non_finite_or_fk_invalid")
    if float(c.get("root_y_range_m", 0.0)) > _stage_guard_env_float("STAGE_GUARD_KBO_ROOT_RANGE_ABS_MAX_M", 2.50) * max(1.0, relax * 0.75):
        fatal.append("root_y_range_abs_exceeded")
    if abs(float(c.get("floor_y", 0.0)) - float(r.get("floor_y", 0.0))) > _stage_guard_env_float("STAGE_GUARD_KBO_FLOOR_SHIFT_MAX_M", 1.50) * max(1.0, relax):
        fatal.append("floor_shift_exceeded")
    if float(c.get("bone_length_violation_max_m", 0.0)) > _stage_guard_env_float("STAGE_GUARD_KBO_BONE_LENGTH_EPS_M", 0.02) * max(1.0, relax):
        fatal.append("bone_length_violation")

    # Derivative barriers are soft in early-abort mode. They need co-occurring
    # fatal evidence, or a very large robust p99 excursion when the user enables it.
    acc_thr = _stage_guard_env_float("STAGE_GUARD_KBO_ACC_MAX_MPS2", 2700.0) * max(1.0, relax)
    jerk_thr = _stage_guard_env_float("STAGE_GUARD_KBO_JERK_MAX_MPS3", 81000.0) * max(1.0, relax)
    if float(c.get("joint_acceleration_p99_mps2", 0.0)) > acc_thr:
        soft.append("robust_acc_p99_spike")
    if float(c.get("joint_jerk_p99_mps3", 0.0)) > jerk_thr:
        soft.append("robust_jerk_p99_spike")

    if _stage_guard_env_bool("STAGE_GUARD_KBO_STAGE_ANCHOR_ENABLE", True):
        ae = _stage_anchor_error(smooth, global_start)
        # Anchor is a soft early signal; high-energy windows already get low
        # weights through Physics Stability anchor_error.
        c["stage_anchor_error_p95_m"] = float(ae)
        if ae > _stage_guard_env_float("STAGE_GUARD_KBO_ANCHOR_P95_MAX_M", 0.85) * max(1.0, relax):
            soft.append("weighted_stage_anchor_deviation")

    derivative_only_abort = _physics_stability_env_bool("PHYSICS_STABILITY_EARLY_ABORT_ALLOW_DERIVATIVE_ONLY_FATAL", False)
    if fatal:
        ok = False
        reasons = fatal + soft
    elif derivative_only_abort and len(soft) >= _physics_stability_env_int("PHYSICS_STABILITY_EARLY_ABORT_MIN_SOFT_BARRIERS", 2):
        ok = False
        reasons = soft
    else:
        ok = True
        reasons = soft  # diagnostic only

    detail = {
        "kbo_mode": "physics_stability_derivative_safe_early_abort",
        "lowpass_sigma": float(sigma),
        "relax": float(relax),
        "fatal_barriers": fatal,
        "soft_barriers": soft,
        "candidate_lowpass": c,
        "reference": r,
        "raw_probe_shape": list(raw.shape),
        "global_start": int(global_start),
        "interpretation": "soft derivative barriers alone do not abort Tweedie probes",
    }
    return ok, reasons, detail


# Preserve the Energy Stability function name used by diffusion proposal, but replace its logic.
def _barrier_oracle_early_abort(candidate, reference, cfg, stage="diffusion_early_abort_probe", global_start=0):
    return _physics_early_abort_oracle(candidate, reference, cfg, stage=stage, global_start=global_start)


def _stage_anchor_weight_for_motion(motion, global_start=0):
    m = np.asarray(motion, dtype=np.float32)
    T = len(m)
    if T <= 0:
        return np.ones((0, 1), dtype=np.float32)
    try:
        frame_w = _frame_stage_weights(T, global_start=global_start)
    except Exception:
        frame_w = np.ones((T, 1), dtype=np.float32)
    try:
        vel_gate = _root_velocity_gate(m)
    except Exception:
        vel_gate = np.ones((T, 1), dtype=np.float32)
    # Harder leap gate: if root speed is high, do not pull the body back to a
    # low-pass prior. Dilate high-speed regions to include takeoff/landing.
    if T >= 3:
        v = np.linalg.norm(np.diff(m[:, [ROOT_X_IDX, ROOT_Z_IDX]], axis=0), axis=-1)
        v = np.concatenate([[v[0]], v]).astype(np.float32)
        leap_thr = _physics_stability_env_float("PHYSICS_STABILITY_MSA_LEAP_SPEED_THRESH", 0.070)
        leap = v > leap_thr
        if ndi is not None and np.any(leap):
            leap = ndi.binary_dilation(leap.astype(bool), iterations=_physics_stability_env_int("PHYSICS_STABILITY_MSA_LEAP_DILATE", 4))
        leap_gate = np.where(leap, _physics_stability_env_float("PHYSICS_STABILITY_MSA_LEAP_MIN_GATE", 0.0), 1.0).astype(np.float32)[:, None]
    else:
        leap_gate = np.ones((T, 1), dtype=np.float32)
    w = frame_w * vel_gate * leap_gate
    return np.clip(w, _energy_gate_env_float("ENERGY_STABILITY_MSA_MIN_WEIGHT", 0.05), 1.0).astype(np.float32)


def _apply_guarded_stage_prior(
    motion,
    cfg,
    strength=None,
    global_start=0,
    preserve_contacts=False,
):
    """Velocity-preserving MSA.

    Instead of dragging root to the low-frequency prior frame-by-frame, correct
    only low-frequency drift with capped, smoothed offsets. Leap/high-speed
    frames are gated out to avoid moonwalk/airborne rubber-band artifacts.
    """
    global _STAGE_PRIOR_XZ
    if not _stage_guard_env_bool("STAGE_GUARD_MSA_ENABLE", True):
        return np.asarray(motion, dtype=np.float32), {"enabled": False}
    m = np.asarray(motion, dtype=np.float32).copy()
    prior = _STAGE_PRIOR_XZ
    if prior is None or len(prior) < int(global_start) + len(m):
        prior_local, meta = _build_stage_prior_xz(m, None, cfg)
    else:
        prior_local = prior[int(global_start):int(global_start)+len(m)]
        meta = dict(_STAGE_PRIOR_METADATA)
    base_alpha = _stage_guard_env_float("STAGE_GUARD_MSA_COMMIT_STRENGTH", 0.16) if strength is None else float(strength)
    w = _stage_anchor_weight_for_motion(m, global_start=global_start)
    raw_corr = prior_local - m[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    sigma = _physics_stability_env_float("PHYSICS_STABILITY_MSA_CORRECTION_LOWPASS_SIGMA", 10.0)
    if ndi is not None and len(raw_corr) > 7 and sigma > 0:
        corr = ndi.gaussian_filter1d(raw_corr, sigma=float(sigma), axis=0, mode="nearest")
    else:
        corr = raw_corr
    # Capping the correction magnitude and its frame-to-frame velocity preserves
    # local foot/root dynamics and prevents rubber-band deceleration.
    max_delta = _physics_stability_env_float("PHYSICS_STABILITY_MSA_MAX_OFFSET_DELTA_M", _stage_guard_env_float("STAGE_GUARD_MSA_MAX_DELTA_M", 0.06))
    corr = np.clip(corr, -max_delta, max_delta)
    max_corr_vel = _physics_stability_env_float("PHYSICS_STABILITY_MSA_MAX_CORRECTION_VEL_MPS", 0.18)
    max_corr_step = max_corr_vel / max(float(cfg.fps), 1.0e-8)
    if len(corr) > 1 and max_corr_vel > 0:
        smooth_corr = corr.copy()
        for t in range(1, len(smooth_corr)):
            step = np.clip(smooth_corr[t] - smooth_corr[t-1], -max_corr_step, max_corr_step)
            smooth_corr[t] = smooth_corr[t-1] + step
        corr = smooth_corr
    alpha = float(base_alpha) * w[:, 0]
    m[:, ROOT_X_IDX] = m[:, ROOT_X_IDX] + alpha * corr[:, 0]
    m[:, ROOT_Z_IDX] = m[:, ROOT_Z_IDX] + alpha * corr[:, 1]
    m, _ = enforce_edge151_contract_np(
        m,
        cfg,
        source_hint="physics_stability_velocity_preserving_msa",
        derive_contact=not preserve_contacts,
        project_rot=True,
    )
    meta.update({
        "applied": True,
        "version": "physics_stability_velocity_preserving_dynamic_msa",
        "base_strength": float(base_alpha),
        "effective_strength_mean": float(base_alpha * float(np.mean(w))) if len(w) else 0.0,
        "effective_strength_min": float(base_alpha * float(np.min(w))) if len(w) else 0.0,
        "correction_lowpass_sigma": float(sigma),
        "max_offset_delta_m": float(max_delta),
        "max_correction_velocity_mps": float(max_corr_vel),
        "interpretation": "low-frequency drift correction only; leap/high-root-speed frames are released",
    })
    return m.astype(np.float32), meta


def _stage_anchor_error(candidate, a0=0):
    """Anchor error for KBO, with leap/high-energy weighting.

    High-energy or high-root-speed windows are not rejected only because they
    deviate from the low-frequency stage prior.
    """
    global _STAGE_PRIOR_XZ
    cand = np.asarray(candidate, dtype=np.float32)
    if _STAGE_PRIOR_XZ is None:
        return 0.0
    a = int(a0); b = a + len(cand)
    if a < 0 or b > len(_STAGE_PRIOR_XZ):
        return 0.0
    prior = _STAGE_PRIOR_XZ[a:b]
    err = np.linalg.norm(cand[:, [ROOT_X_IDX, ROOT_Z_IDX]] - prior, axis=-1)
    w = _stage_anchor_weight_for_motion(cand, global_start=a)[:, 0]
    weighted = err * np.clip(w, _energy_gate_env_float("ENERGY_STABILITY_MSA_MIN_WEIGHT", 0.05), 1.0)
    return float(np.percentile(weighted, 95))


def _diffusion_window_proposal(snapshot, cond, sm_win, ckpt_path, cfg, global_start=0):
    """Delegate guarded windows to the formal reference-tangent diffusion."""

    return _stage_guard_orig_apply_diffusion_model(
        snapshot,
        cond,
        sm_win,
        ckpt_path,
        cfg,
    )


# ===== Physics Stability PHYSICS-CONSISTENT STABILITY PATCH END =====


if __name__ == "__main__":
    raise SystemExit(main())
