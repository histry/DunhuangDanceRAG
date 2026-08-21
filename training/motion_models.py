#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Music Semantics MotionRAG-Diff for EDGE 151D Dunhuang whole-song generation
==================================================================

This file is designed as a drop-in research patch for an EDGE-style repository.
It does not depend on README assumptions. It directly operates on EDGE 151D
motion arrays and can rebuild a source-aware motion database from a new
"change" dataset.

Core components:
- Lower-Body IK: true lower-body IK that writes lower-body rotation channels back into
       EDGE 151D, rather than saving fake foot XYZ columns.
- Contrastive Retriever: music-motion contrastive learning for retrieval alignment.
- Boundary Refiner: residual temporal refinement to escape pure stitching.
- Motion Generation: retrieval-augmented conditional residual diffusion with IK finalization.
- External classical-music semantics plus canonical Chang-E semantics: manifest-authoritative Chang-E BVH loading, effective-FPS resampling, recording-aware RAG semantics, external slot-level music labels, unpaired semantic OT, true lower-body IK, residual refinement, conditional diffusion, capped root-Y physics, root-aware sliding anchors, weighted IK chunks, and strict rollback gates.

Expected EDGE 151D convention
-----------------------------
root translation: motion[:, 4:7] = [x, y, z]
rot6d local joints: motion[:, 7:151].reshape(T, 24, 6)
foot FK ids: [7, 8, 10, 11]

Typical commands
----------------
python -m training.motion_models build-db --motion_dirs change data/motions --out_db output/motion_db
python -m training.motion_models train-contrastive --db output/motion_db/events.npz --out output/motion_db/semantic_retriever.pt
python -m training.motion_models train-refiner --db output/motion_db/events.train.npz --val_db output/motion_db/events.val.npz --out output/motion_db/boundary_refiner.pt
python -m training.motion_models train-diffusion --db output/motion_db/events.train.npz --val_db output/motion_db/events.val.npz --out output/motion_db/motion_diffusion.pt
python -m training.motion_models generate --audio test_music_bank/dunhuangwu2.wav --db output/motion_db/events.npz --out output/motion_dunhuangwu2.npy
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import hashlib
import json
import math
import os
import pickle
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
    import scipy.io.wavfile as wavfile
except Exception:  # pragma: no cover
    wavfile = None

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
)
from motion_geometry.product_manifold import (
    PRODUCT_STATE_DIM,
    masked_retract_np,
    masked_retract_torch,
    product_log_np,
    product_log_torch,
)
from contracts.boundary import build_frame_joint_risk_mask
from contracts.physical_quality import (
    PhysicalQualityLimits,
    StageAcceptancePolicy,
    compute_joint_kinematic_metrics,
    evaluate_physical_audit,
    evaluate_stage_candidate,
)
from support.event_identity import (
    assert_same_event_db_contract,
    event_uids_from_generation_db,
    make_event_db_contract,
    normalize_event_db_contract,
)
from support.legacy_compatibility import (
    has_scheduler_provenance,
    normalize_motion_checkpoint_role,
    normalize_motion_checkpoint_version,
)
from motion_geometry.resampling import blend_edge151_geodesic_np
from retargeting.bvh_solver import load_bvh_as_edge151

LOWER_BODY_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)
FK_TREE_SOURCE = SMPL24_SKELETON_SCHEMA


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)


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


def normalize_motion_shape(arr: np.ndarray) -> List[np.ndarray]:
    arr = np.asarray(arr)
    outs: List[np.ndarray] = []
    if arr.ndim == 2 and arr.shape[1] >= EDGE_DIM:
        outs.append(arr[:, :EDGE_DIM].astype(np.float32))
    elif arr.ndim == 3 and arr.shape[-1] >= EDGE_DIM:
        for i in range(arr.shape[0]):
            outs.append(arr[i, :, :EDGE_DIM].astype(np.float32))
    return outs



def load_bvh_file(path: str | Path) -> List[np.ndarray]:
    """
    Load a Chang-E-style `.bvh` file and convert it to an EDGE-like 151D array.

    Output convention:
      - motion[:, 4:7] = root XYZ translation
      - motion[:, 7:151] = 24 local joint rotations in 6D form

    BVH files commonly store positions in centimeters. The loader auto-scales
    to meters when skeleton offsets look centimeter-scale. This direct adapter
    is sufficient for source-aware event indexing, retrieval, Motion Refiner and Motion Diffusion training,
    and Lower-Body IK IK; for exact SMPL/EDGE reproduction, a dedicated retargeting stage
    can still be used before build-db.
    """
    motion, _report = load_bvh_as_edge151(
        path,
        source_manifest_path=os.environ.get("CHANG_E_SOURCE_MANIFEST") or None,
        require_source_manifest=str(
            os.environ.get("CHANG_E_REQUIRE_SOURCE_MANIFEST", "0")
        ).strip().lower() in {"1", "true", "yes", "y", "on"},
    )
    return [motion.astype(np.float32)]

def load_motion_file(path: str | Path) -> List[np.ndarray]:
    p = Path(path)
    outs: List[np.ndarray] = []
    try:
        if p.suffix.lower() == ".npy":
            outs.extend(normalize_motion_shape(np.load(p, allow_pickle=True)))
        elif p.suffix.lower() == ".npz":
            data = np.load(p, allow_pickle=True)
            for k in data.files:
                if k.lower() in {"motion", "motions", "x", "arr_0", "data"} or data[k].ndim in (2, 3):
                    outs.extend(normalize_motion_shape(data[k]))
        elif p.suffix.lower() in {".pkl", ".pickle"}:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                for k in ["motion", "motions", "x", "poses", "data"]:
                    if k in obj:
                        outs.extend(normalize_motion_shape(np.asarray(obj[k])))
            else:
                outs.extend(normalize_motion_shape(np.asarray(obj)))
        elif p.suffix.lower() == ".bvh":
            outs.extend(load_bvh_file(p))
    except Exception as exc:
        print(f"[Motion Generation WARN] failed loading {p}: {exc}", file=sys.stderr)
    return [x for x in outs if x.ndim == 2 and x.shape[0] >= 8]


def rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    return _contract_rot6d_to_matrix_np(x)


def matrix_to_rot6d_np(mat: np.ndarray) -> np.ndarray:
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
    return _contract_matrix_to_rot6d_np(mat)


def fk_24_np(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < ROT6D_END:
        raise ValueError(f"Expected EDGE 151D motion [T,151], got {motion.shape}")
    T = motion.shape[0]
    root = motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    rot6d = motion[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)
    local_r = rot6d_to_matrix_np(rot6d)
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


def fk_24_torch(motion, parents=None, offsets=None):
    parents = torch.as_tensor(PARENTS if parents is None else parents, device=motion.device, dtype=torch.long)
    offsets = torch.as_tensor(OFFSETS if offsets is None else offsets, device=motion.device, dtype=motion.dtype)
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
        p = int(parents[j].item())
        if j == 0 or p < 0:
            gr = local_r[:, j]
            pos = root
        else:
            gr = torch.matmul(global_r[p], local_r[:, j])
            off = offsets[j].view(1, 3, 1)
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
    max_event_frames: int = 180
    db_feature_dim: int = 32
    # Chang-E timebase comes from the checked source manifest.  Raw BVH header
    # FPS is not authoritative for the current local subset.
    bvh_resample_to_config_fps: bool = True
    manifest_enable: bool = True
    manifest_secondary_event_split: bool = True

    # Filename Semantics: Chang-E/change BVH filename semantics are scientifically meaningful.
    # Each original BVH file is a source_uid/source_group, while the parsed
    # category/gender/take fields become semantic metadata and weak alignment
    # priors for unpaired slot-to-event routing.  This fixes the Root-Aware Motion Safety issue
    # where singleton names such as female_lotus/male_ribbon were collapsed
    # into the same directory-level source group.
    source_group_mode: str = "filename"  # filename | legacy_prefix
    filename_semantic_enable: bool = True
    filename_semantic_weight: float = 0.35
    filename_semantic_retrieval_weight: float = 0.20
    filename_semantic_ot_weight: float = 0.35
    # Chang-E Semantics: stronger multi-label RAG action semantics and music alignment labels.
    classification_semantic_enable: bool = True
    classification_semantic_ratio: float = 0.70
    classification_retrieval_weight: float = 0.34
    classification_ot_weight: float = 0.45
    classification_retrieval_bonus: float = 0.28
    classification_report_topk: int = 8
    # Event Semantics: Chang-E event-level semantic routing.
    chang_e_event_semantic_enable: bool = True
    semantic_routing_weight: float = 0.72
    event_family_bonus: float = 0.58
    motion_stage_role_bonus: float = 0.36
    preferred_dance_key_bonus: float = 0.28
    route_natural_duration_weight: float = 0.20
    route_family_balance_penalty: float = 0.18
    route_family_recent_window: int = 8
    route_family_penalty_cap: float = 0.25
    route_dance_key_repeat_penalty: float = 0.16
    route_family_repeat_penalty: float = 0.12
    route_source_repeat_penalty: float = 0.10
    route_motif_recall_bonus: float = 0.12
    route_debug_topk: int = 10
    # Convert long Chang-E recordings into a curated semantic event library.
    chang_e_boundary_event_split: bool = True
    chang_e_boundary_max_extra_starts: int = 96
    chang_e_min_event_quality: float = 0.22
    chang_e_keep_pose_anchor_quality: float = 0.16
    event_quality_weight: float = 0.22
    route_support_bonus: float = 0.12
    route_locomotion_bonus: float = 0.14
    route_stage_sequence_weight: float = 0.16
    route_source_run_hard_penalty: float = 0.30
    route_semantic_bonus_scale: float = 1.50

    # External Music Semantics: external classical-music semantic encoder.  The current EDGE
    # repository contains rule/librosa music feature extractors, not a trained
    # classical-music classifier.  When a trained external encoder exists, it
    # can write slot-level JSON/NPZ semantics or be invoked through a command
    # template.  These semantic probabilities then drive Contrastive Retriever unpaired OT and
    # Motion Generation retrieval reports.
    external_music_semantic_enable: bool = True
    external_music_semantic_required: bool = False
    external_music_semantic_dirs: str = ""
    external_music_semantic_cmd: str = ""
    external_music_semantic_cache_dir: str = "output/motion_external_music_semantic_cache"
    external_music_semantic_weight: float = 0.78
    external_music_semantic_temperature: float = 0.65
    external_music_semantic_proxy_enable: bool = True
    external_music_semantic_filename_proxy: bool = True

    embed_dim: int = 128
    top_k: int = 32
    beam_size: int = 8
    overlap: int = 12
    transition_train_min_seconds: float = 10.0 / 30.0
    transition_train_max_seconds: float = 28.0 / 30.0
    transition_mask_halo_seconds: float = 6.0 / 30.0
    retrieval_source_penalty: float = 0.08
    retrieval_transition_penalty: float = 0.65
    retrieval_warp_penalty: float = 0.18
    retrieval_repeat_penalty: float = 0.15
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

    # Controls use of a trained Contrastive Retriever model during generation.  Training remains
    # an explicit CLI operation, so this switch must not silently suppress a
    # user-invoked train-contrastive command.
    contrastive_enable: bool = True
    # Root-Aware Motion Safety: require real audio features for Contrastive Retriever; false keeps the weak-proxy
    # fallback for smoke tests, while reports mark it as weak supervision.
    contrastive_require_real_music: bool = False
    audio_pair_min_coverage: float = 0.15

    # Root-Aware Motion Safety: Chang-E BVH is usually motion-only.  When no synchronized
    # BVH/audio pairs exist, train Contrastive Retriever with real unpaired music clips plus
    # semantic optimal-transport pseudo pairs instead of motion-descriptor
    # self-distillation.  This keeps the method honest: the checkpoint records
    # unpaired_audio_semantic_ot, not paired supervision.
    unpaired_audio_enable: bool = True
    unpaired_audio_slot_seconds: float = 4.0
    unpaired_positive_topk: int = 8
    unpaired_pairs_per_audio_slot: int = 4
    unpaired_min_audio_slots: int = 1
    unpaired_disable_motion_proxy: bool = False

    lower_body_only: bool = True
    refiner_enable: bool = True
    diffusion_enable: bool = True
    # Geometry-safe Motion Refiner and Motion Diffusion path.  Existing 151D checkpoints remain loadable;
    # newly trained checkpoints use a 79D contact/root/joint-tangent state.
    product_manifold_enable: bool = True
    product_refiner_rotation_cap_rad: float = 0.35
    product_refiner_root_cap_m: float = 0.08
    product_refiner_outside_weight: float = 0.25
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
    # Whole-sequence acceptance is authoritative. A candidate that fails the
    # SI physical gate is written only as an explicitly rejected diagnostic,
    # never to the requested accepted-output path.
    final_physical_gate_enable: bool = True
    final_physical_gate_fail_closed: bool = True
    final_physical_gate_save_rejected: bool = True
    tangent_diffusion_enable: bool = True
    tangent_diffusion_rotation_cap_rad: float = 0.45
    tangent_diffusion_root_cap_m: float = 0.10
    riemannian_trust_region_enable: bool = True
    riemannian_trust_region_steps: int = 5
    riemannian_trust_region_initial_radius: float = 1.0
    riemannian_trust_region_min_radius: float = 0.0625
    diffusion_steps: int = 50
    diffusion_train_steps: int = 15000
    refiner_train_steps: int = 8000
    contrastive_epochs: int = 120
    batch_size: int = 64
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
            # Read historical 30 FPS configs without keeping per-frame units
            # in the runtime contract.  Newly written configs use SI names.
            legacy = {
                "ik_speed_gate_mpf": ("ik_speed_gate_mps", cfg.fps),
                "ik_contact_break_speed_mpf": ("ik_contact_break_speed_mps", cfg.fps),
                "ik_slide_release_min_frames": ("ik_slide_release_min_seconds", 1.0 / cfg.fps),
                "ik_sliding_anchor_window": ("ik_sliding_anchor_seconds", 1.0 / cfg.fps),
                "root_y_min_flight_frames": ("root_y_min_flight_seconds", 1.0 / cfg.fps),
            }
            for old, (new, scale) in legacy.items():
                if old in data and new not in data:
                    setattr(cfg, new, float(data[old]) * float(scale))
        return cfg

    def apply_env(self) -> "MotionGenerationConfig":
        env_map = {
            "MOTION_FPS": ("fps", float),
            "MOTION_ENABLE_TRUE_IK": ("ik_enable", lambda x: bool(int(x))),
            "MOTION_ENABLE_REFINER": ("refiner_enable", lambda x: bool(int(x))),
            "MOTION_ENABLE_DIFFUSION": ("diffusion_enable", lambda x: bool(int(x))),
            "MOTION_PRODUCT_MANIFOLD_ENABLE": ("product_manifold_enable", lambda x: bool(int(x))),
            "MOTION_PRODUCT_REFINER_ROTATION_CAP_RAD": ("product_refiner_rotation_cap_rad", float),
            "MOTION_PRODUCT_REFINER_ROOT_CAP_M": ("product_refiner_root_cap_m", float),
            "MOTION_PRODUCT_REFINER_OUTSIDE_WEIGHT": ("product_refiner_outside_weight", float),
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
            "MOTION_TANGENT_DIFFUSION_ENABLE": ("tangent_diffusion_enable", lambda x: bool(int(x))),
            "MOTION_TANGENT_DIFFUSION_ROTATION_CAP_RAD": ("tangent_diffusion_rotation_cap_rad", float),
            "MOTION_TANGENT_DIFFUSION_ROOT_CAP_M": ("tangent_diffusion_root_cap_m", float),
            "MOTION_RIEMANNIAN_TRUST_REGION_ENABLE": ("riemannian_trust_region_enable", lambda x: bool(int(x))),
            "MOTION_RIEMANNIAN_TRUST_REGION_STEPS": ("riemannian_trust_region_steps", int),
            "MOTION_RIEMANNIAN_TRUST_REGION_INITIAL_RADIUS": ("riemannian_trust_region_initial_radius", float),
            "MOTION_RIEMANNIAN_TRUST_REGION_MIN_RADIUS": ("riemannian_trust_region_min_radius", float),
            "MOTION_TOP_K": ("top_k", int),
            "MOTION_BEAM_SIZE": ("beam_size", int),
            "MOTION_CHANG_E_EVENT_SEMANTIC_ENABLE": ("chang_e_event_semantic_enable", lambda x: bool(int(x))),
            "MOTION_SEMANTIC_ROUTING_WEIGHT": ("semantic_routing_weight", float),
            "MOTION_EVENT_FAMILY_BONUS": ("event_family_bonus", float),
            "MOTION_STAGE_ROLE_BONUS": ("motion_stage_role_bonus", float),
            "MOTION_PREFERRED_DANCE_KEY_BONUS": ("preferred_dance_key_bonus", float),
            "MOTION_ROUTE_NATURAL_DURATION_WEIGHT": ("route_natural_duration_weight", float),
            "MOTION_ROUTE_FAMILY_BALANCE_PENALTY": ("route_family_balance_penalty", float),
            "MOTION_ROUTE_FAMILY_RECENT_WINDOW": ("route_family_recent_window", int),
            "MOTION_ROUTE_FAMILY_PENALTY_CAP": ("route_family_penalty_cap", float),
            "MOTION_ROUTE_DANCE_KEY_REPEAT_PENALTY": ("route_dance_key_repeat_penalty", float),
            "MOTION_ROUTE_FAMILY_REPEAT_PENALTY": ("route_family_repeat_penalty", float),
            "MOTION_ROUTE_SOURCE_REPEAT_PENALTY": ("route_source_repeat_penalty", float),
            "MOTION_ROUTE_MOTIF_RECALL_BONUS": ("route_motif_recall_bonus", float),
            "MOTION_ROUTE_DEBUG_TOPK": ("route_debug_topk", int),
            "MOTION_CHANG_E_BOUNDARY_EVENT_SPLIT": ("chang_e_boundary_event_split", lambda x: bool(int(x))),
            "MOTION_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS": ("chang_e_boundary_max_extra_starts", int),
            "MOTION_CHANG_E_MIN_EVENT_QUALITY": ("chang_e_min_event_quality", float),
            "MOTION_CHANG_E_KEEP_POSE_ANCHOR_QUALITY": ("chang_e_keep_pose_anchor_quality", float),
            "MOTION_EVENT_QUALITY_WEIGHT": ("event_quality_weight", float),
            "MOTION_ROUTE_SUPPORT_BONUS": ("route_support_bonus", float),
            "MOTION_ROUTE_LOCOMOTION_BONUS": ("route_locomotion_bonus", float),
            "MOTION_ROUTE_STAGE_SEQUENCE_WEIGHT": ("route_stage_sequence_weight", float),
            "MOTION_ROUTE_SOURCE_RUN_HARD_PENALTY": ("route_source_run_hard_penalty", float),
            "MOTION_ROUTE_SEMANTIC_BONUS_SCALE": ("route_semantic_bonus_scale", float),
            "MOTION_OVERLAP": ("overlap", int),
            "MOTION_TRANSITION_TRAIN_MIN_SECONDS": ("transition_train_min_seconds", float),
            "MOTION_TRANSITION_TRAIN_MAX_SECONDS": ("transition_train_max_seconds", float),
            "MOTION_TRANSITION_MASK_HALO_SECONDS": ("transition_mask_halo_seconds", float),
            "MOTION_WINDOW_LEN": ("window_len", int),
            "MOTION_HOP_LEN": ("hop_len", int),
            "MOTION_MIN_EVENT_FRAMES": ("min_event_frames", int),
            "MOTION_MAX_EVENT_FRAMES": ("max_event_frames", int),
            "MOTION_ENABLE_CONTRASTIVE": ("contrastive_enable", lambda x: bool(int(x))),
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
            "MOTION_CONTRASTIVE_REQUIRE_REAL_MUSIC": ("contrastive_require_real_music", lambda x: bool(int(x))),
            "MOTION_AUDIO_PAIR_MIN_COVERAGE": ("audio_pair_min_coverage", float),
            "MOTION_UNPAIRED_AUDIO_ENABLE": ("unpaired_audio_enable", lambda x: bool(int(x))),
            "MOTION_UNPAIRED_AUDIO_SLOT_SECONDS": ("unpaired_audio_slot_seconds", float),
            "MOTION_UNPAIRED_POSITIVE_TOPK": ("unpaired_positive_topk", int),
            "MOTION_UNPAIRED_PAIRS_PER_AUDIO_SLOT": ("unpaired_pairs_per_audio_slot", int),
            "MOTION_UNPAIRED_DISABLE_MOTION_PROXY": ("unpaired_disable_motion_proxy", lambda x: bool(int(x))),
            "MOTION_ROOT_Y_DAMPING_MAX_SECONDS": ("root_y_damping_max_seconds", float),
            "MOTION_ENABLE_ROOT_Y_PHYSICS": ("root_y_physics_enable", lambda x: bool(int(x))),
            "MOTION_ROOT_Y_MIN_FLIGHT_SECONDS": ("root_y_min_flight_seconds", float),
            "MOTION_ROOT_Y_MAX_FLIGHT_SECONDS": ("root_y_max_flight_seconds", float),
            "MOTION_DIFFUSION_STEPS": ("diffusion_steps", int),
            "MOTION_DEVICE": ("device", str),
            "MOTION_BVH_RESAMPLE_TO_CONFIG_FPS": ("bvh_resample_to_config_fps", lambda x: bool(int(x))),
            "MOTION_SOURCE_GROUP_MODE": ("source_group_mode", str),
            "MOTION_FILENAME_SEMANTIC_ENABLE": ("filename_semantic_enable", lambda x: bool(int(x))),
            "MOTION_FILENAME_SEMANTIC_WEIGHT": ("filename_semantic_weight", float),
            "MOTION_FILENAME_SEMANTIC_RETRIEVAL_WEIGHT": ("filename_semantic_retrieval_weight", float),
            "MOTION_FILENAME_SEMANTIC_OT_WEIGHT": ("filename_semantic_ot_weight", float),
            "MOTION_CLASSIFICATION_SEMANTIC_ENABLE": ("classification_semantic_enable", lambda x: bool(int(x))),
            "MOTION_CLASSIFICATION_SEMANTIC_RATIO": ("classification_semantic_ratio", float),
            "MOTION_CLASSIFICATION_RETRIEVAL_WEIGHT": ("classification_retrieval_weight", float),
            "MOTION_CLASSIFICATION_OT_WEIGHT": ("classification_ot_weight", float),
            "MOTION_CLASSIFICATION_RETRIEVAL_BONUS": ("classification_retrieval_bonus", float),
            "MOTION_CLASSIFICATION_REPORT_TOPK": ("classification_report_topk", int),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_ENABLE": ("external_music_semantic_enable", lambda x: bool(int(x))),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_REQUIRED": ("external_music_semantic_required", lambda x: bool(int(x))),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_DIRS": ("external_music_semantic_dirs", str),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_CMD": ("external_music_semantic_cmd", str),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_CACHE_DIR": ("external_music_semantic_cache_dir", str),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_WEIGHT": ("external_music_semantic_weight", float),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_TEMPERATURE": ("external_music_semantic_temperature", float),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE": ("external_music_semantic_proxy_enable", lambda x: bool(int(x))),
            "MOTION_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY": ("external_music_semantic_filename_proxy", lambda x: bool(int(x))),
            "MOTION_MANIFEST_ENABLE": ("manifest_enable", lambda x: bool(int(x))),
            "MOTION_MANIFEST_SECONDARY_EVENT_SPLIT": ("manifest_secondary_event_split", lambda x: bool(int(x))),
        }
        for e, (attr, caster) in env_map.items():
            if e in os.environ:
                setattr(self, attr, caster(os.environ[e]))
        if self.device == "cuda" and (torch is None or not torch.cuda.is_available()):
            self.device = "cpu"
        return self


MOTION_CHECKPOINT_CONTRACT_SCHEMA = "dunhuang_motion_checkpoint_contract_v2"


def motion_checkpoint_contract(cfg: MotionGenerationConfig, role: str) -> Dict[str, Any]:
    """Return the immutable representation/time contract embedded in a checkpoint."""
    return {
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


def assert_motion_checkpoint_contract(
    checkpoint: Dict[str, Any],
    cfg: MotionGenerationConfig,
    path: str | Path,
    role: str,
) -> None:
    """Reject mixed-FPS, mixed-Rot6D and mixed-skeleton model assets."""
    actual = checkpoint.get("motion_contract")
    if not isinstance(actual, dict):
        if os.environ.get("MOTION_ALLOW_LEGACY_CHECKPOINT_CONTRACT", "0") == "1" and abs(float(cfg.fps) - 30.0) < 1.0e-6:
            return
        raise RuntimeError(
            f"Checkpoint {path} has no {MOTION_CHECKPOINT_CONTRACT_SCHEMA}. "
            "It must be rebuilt for this repository contract. For a read-only "
            "30 FPS parity baseline only, set MOTION_ALLOW_LEGACY_CHECKPOINT_CONTRACT=1."
        )

    expected = motion_checkpoint_contract(cfg, role)
    mismatches: List[str] = []
    for key in ("schema", "role", "motion_dim", "rot6d_layout", "skeleton_schema", "skeleton_sha256"):
        actual_value = actual.get(key)
        expected_value = expected[key]
        if key == "role":
            actual_value = normalize_motion_checkpoint_role(actual_value)
            expected_value = normalize_motion_checkpoint_role(expected_value)
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
        rot6d_to_matrix_np(rot.reshape(x.shape[0], NUM_JOINTS, 6))
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

    Critical reason:
    direct BVH loading may temporarily use channel 0 to carry native fps before
    resampling.  That is legal only inside loading.  Once an array is saved as an
    Event-RAG clip or passed into Motion Refiner and Motion Diffusion, EDGE [0:4] must again mean contacts.
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

def _clean_stem(path: str | Path) -> str:
    return Path(str(path)).stem.strip().lower().replace("-", "_").replace(" ", "_")
















# Historical filename/category priors are retained only for BVH ablation
# compatibility.  They are never used as local-action truth in the formal
# official-SMPL path.
LEGACY_BVH_CATEGORY_PROFILES = {
    "flying_apsaras": {"aliases": {"flying", "apsaras", "flying_apsara", "flying_apsaras", "feitian", "fei_tian", "sky_dance"}, "energy": 0.52, "onset": 0.28, "travel": 0.32, "turn": 0.38, "lower": 0.38, "upper": 0.72, "floorwork": 0.10, "jump": 0.35, "spin": 0.35, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.85, "display": "Flying Apsaras", "semantic_role": "aerial_graceful_flow", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "upper_body", "spatial_label": "aerial_leaning", "music_alignment_label": "lyrical_flow", "music_alignment_tags": ["lyrical_flow", "turning_climax", "calm_meditative", "aerial_curve"], "preferred_music_roles": ["intro", "build_up", "climax"], "preferred_dance_keys": ["flying_apsaras", "sogdian_whirl", "lotus_steps"], "cultural_motif": "flying_apsara", "prop_proxy_label": "sash_ribbon_proxy", "locomotion_label": "floating_leaning", "support_label": "low_contact_flight_like", "event_family": "aerial_curve", "motion_stage_role": "opening_or_climax", "natural_duration_range_sec": [2.0, 5.5]},
    "lotus_steps": {"aliases": {"lotus", "lotussteps", "lotus_step", "lotus_steps"}, "energy": 0.48, "onset": 0.35, "travel": 0.62, "turn": 0.20, "lower": 0.78, "upper": 0.38, "floorwork": 0.05, "jump": 0.12, "spin": 0.10, "pose_hold": 0.20, "instrument": 0.0, "prop": 0.0, "display": "Lotus Steps", "semantic_role": "flowing_footwork", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "lower_body", "spatial_label": "traveling", "music_alignment_label": "footwork_flow", "music_alignment_tags": ["footwork_flow", "lyrical_flow", "calm_meditative"], "preferred_music_roles": ["normal", "development"], "preferred_dance_keys": ["lotus_steps", "flying_apsaras", "sogdian_whirl"], "cultural_motif": "lotus_step", "prop_proxy_label": "none", "locomotion_label": "traveling_steps", "support_label": "alternating_foot_support", "event_family": "footwork_flow", "motion_stage_role": "development", "natural_duration_range_sec": [1.5, 4.0]},
    "thirty_six_postures": {"aliases": {"36pose", "36posture", "36postures", "thirtysix", "thirty_six", "thirty_six_postures", "jiyuetian"}, "energy": 0.36, "onset": 0.18, "travel": 0.12, "turn": 0.12, "lower": 0.28, "upper": 0.42, "floorwork": 0.18, "jump": 0.02, "spin": 0.05, "pose_hold": 0.90, "instrument": 0.0, "prop": 0.0, "display": "Ji Yue Tian Thirty-Six Postures", "semantic_role": "iconic_pose_sequence", "energy_label": "moderate", "rhythm_label": "sustained", "body_focus_label": "pose", "spatial_label": "in_place", "music_alignment_label": "pose_hold", "music_alignment_tags": ["pose_hold", "calm_meditative", "lyrical_flow"], "preferred_music_roles": ["intro", "release", "resolution"], "preferred_dance_keys": ["thirty_six_postures", "revelation_meditation", "lotus_steps"], "cultural_motif": "jiyuetian_pose", "prop_proxy_label": "none", "locomotion_label": "in_place_pose", "support_label": "static_or_low_motion_support", "event_family": "pose_motif", "motion_stage_role": "anchor_or_resolution", "natural_duration_range_sec": [1.2, 3.8]},
    "revelation_meditation": {"aliases": {"meditation", "mediation", "revelation", "revelation_meditation", "revelation_mediation"}, "energy": 0.20, "onset": 0.08, "travel": 0.10, "turn": 0.08, "lower": 0.20, "upper": 0.36, "floorwork": 0.38, "jump": 0.0, "spin": 0.03, "pose_hold": 0.78, "instrument": 0.0, "prop": 0.0, "display": "Revelation Meditation", "semantic_role": "calm_meditative_flow", "energy_label": "calm", "rhythm_label": "sustained", "body_focus_label": "full_body", "spatial_label": "in_place", "music_alignment_label": "calm_meditative", "music_alignment_tags": ["calm_meditative", "pose_hold", "lyrical_flow"], "preferred_music_roles": ["intro", "calm", "release", "resolution"], "preferred_dance_keys": ["revelation_meditation", "thirty_six_postures", "flying_apsaras"], "cultural_motif": "buddhist_meditation", "prop_proxy_label": "none", "locomotion_label": "slow_weight_shift", "support_label": "stable_support", "event_family": "calm_flow", "motion_stage_role": "intro_or_resolution", "natural_duration_range_sec": [2.0, 6.0]},
    "sogdian_whirl": {"aliases": {"ribbon", "ribbon_flow", "sash", "silk", "whirl", "sogdian", "sogdian_whirl", "turn", "turning"}, "energy": 0.72, "onset": 0.40, "travel": 0.50, "turn": 0.90, "lower": 0.68, "upper": 0.65, "floorwork": 0.02, "jump": 0.20, "spin": 0.95, "pose_hold": 0.15, "instrument": 0.0, "prop": 0.75, "display": "Sogdian Whirl / Ribbon Flow", "semantic_role": "flowing_turning_motif", "energy_label": "high", "rhythm_label": "lyrical", "body_focus_label": "turning_flow", "spatial_label": "turning", "music_alignment_label": "turning_climax", "music_alignment_tags": ["turning_climax", "lyrical_flow", "footwork_flow"], "preferred_music_roles": ["build_up", "climax"], "preferred_dance_keys": ["sogdian_whirl", "flying_apsaras", "lotus_steps"], "cultural_motif": "sogdian_whirl", "prop_proxy_label": "ribbon_sash_proxy", "locomotion_label": "turning_travel", "support_label": "alternating_or_pivot_support", "event_family": "turning_flow", "motion_stage_role": "climax", "natural_duration_range_sec": [1.6, 4.5]},
    "pipa_behind_back": {"aliases": {"pipa", "pipa1", "pipa2", "playing_pipa", "playing_the_pipa", "pipa_behind_back"}, "energy": 0.46, "onset": 0.42, "travel": 0.16, "turn": 0.20, "lower": 0.30, "upper": 0.82, "floorwork": 0.06, "jump": 0.05, "spin": 0.10, "pose_hold": 0.45, "instrument": 1.0, "prop": 0.70, "display": "Playing the Pipa Behind the Back", "semantic_role": "instrument_upper_body_motif", "energy_label": "moderate", "rhythm_label": "accented", "body_focus_label": "upper_body", "spatial_label": "in_place", "music_alignment_label": "instrument_phrase", "music_alignment_tags": ["instrument_phrase", "lyrical_flow", "percussive_accent"], "preferred_music_roles": ["motif", "normal", "build_up"], "preferred_dance_keys": ["pipa_behind_back", "sogdian_whirl", "lei_gong_drum"], "cultural_motif": "pipa_instrument_pose", "prop_proxy_label": "pipa_proxy", "locomotion_label": "upper_body_phrase", "support_label": "stable_support", "event_family": "instrument_motif", "motion_stage_role": "motif_recall", "natural_duration_range_sec": [1.6, 4.5]},
    "lei_gong_drum": {"aliases": {"drum", "lei_gong", "leigong", "lei_gong_drum"}, "energy": 0.82, "onset": 0.88, "travel": 0.52, "turn": 0.35, "lower": 0.75, "upper": 0.76, "floorwork": 0.04, "jump": 0.32, "spin": 0.20, "pose_hold": 0.10, "instrument": 0.65, "prop": 0.55, "display": "Lei Gong Drum", "semantic_role": "percussive_high_energy", "energy_label": "percussive", "rhythm_label": "percussive", "body_focus_label": "full_body", "spatial_label": "traveling", "music_alignment_label": "percussive_accent", "music_alignment_tags": ["percussive_accent", "turning_climax", "footwork_flow"], "preferred_music_roles": ["accent", "climax"], "preferred_dance_keys": ["lei_gong_drum", "pipa_behind_back", "sogdian_whirl"], "cultural_motif": "thunder_drum", "prop_proxy_label": "drum_proxy", "locomotion_label": "accented_travel", "support_label": "strong_foot_contact", "event_family": "percussive_accent", "motion_stage_role": "accent_or_climax", "natural_duration_range_sec": [1.2, 3.5]},
    "unknown": {"aliases": set(), "energy": 0.45, "onset": 0.30, "travel": 0.30, "turn": 0.20, "lower": 0.45, "upper": 0.45, "floorwork": 0.0, "jump": 0.0, "spin": 0.0, "pose_hold": 0.25, "instrument": 0.0, "prop": 0.0, "display": "Unknown Chang-E Motion", "semantic_role": "unknown_motion", "energy_label": "moderate", "rhythm_label": "lyrical", "body_focus_label": "full_body", "spatial_label": "in_place", "music_alignment_label": "lyrical_flow", "music_alignment_tags": ["lyrical_flow"], "preferred_music_roles": ["normal"], "preferred_dance_keys": ["lotus_steps", "thirty_six_postures"], "cultural_motif": "unknown", "prop_proxy_label": "unknown", "locomotion_label": "unknown", "support_label": "unknown", "event_family": "unknown", "motion_stage_role": "development", "natural_duration_range_sec": [1.5, 4.0]},
}

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
    return canonicalize_chang_e_key(meta.get("dance_key") or meta.get("parent_label") or meta.get("label") or meta.get("source_bvh") or "unknown")


def _label_index(label: str, labels: Sequence[str]) -> int:
    try:
        return list(labels).index(str(label))
    except ValueError:
        return -1


def _parse_numeric_semantic(meta: dict) -> Dict[str, float]:
    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]
    vals = [x for x in re.split(r"[;, ]+", str(meta.get("semantic_numeric", "") or "")) if x]
    out = {}
    for k, v in zip(keys, vals):
        try: out[k] = float(v)
        except Exception: pass
    return out


def strong_action_semantics_from_meta(meta: dict, desc: Optional[np.ndarray] = None) -> Dict[str, object]:
    key = _safe_profile_key(meta)
    is_legacy_bvh = (
        str(meta.get("source_format", "")).startswith("legacy_bvh")
        or bool(meta.get("raw_stem"))
    )
    if is_legacy_bvh:
        legacy = dict(
            LEGACY_BVH_CATEGORY_PROFILES.get(
                key, LEGACY_BVH_CATEGORY_PROFILES["unknown"]
            )
        )
        numeric_keys = [
            "energy", "onset", "travel", "turn", "lower", "upper",
            "floorwork", "jump", "spin", "pose_hold", "instrument", "prop",
        ]
        numeric = {name: float(legacy.get(name, 0.0)) for name in numeric_keys}
        if desc is not None and len(desc) >= 20:
            numeric["travel"] = max(numeric["travel"], _bounded01(desc[1] / 1.2))
            numeric["energy"] = max(numeric["energy"], _bounded01(desc[5] / 0.14))
            numeric["lower"] = max(numeric["lower"], _bounded01(desc[7] / 0.10))
            numeric["upper"] = max(numeric["upper"], _bounded01(desc[8] / 0.10))
            numeric["turn"] = max(numeric["turn"], _bounded01(abs(desc[17]) / 0.22))
            numeric["jump"] = max(numeric["jump"], _bounded01(desc[18] / 0.20))
            numeric["pose_hold"] = max(
                numeric["pose_hold"], _bounded01(1.0 - desc[5] / 0.12)
            )
        legacy["semantic_numeric"] = ";".join(
            str(float(numeric[name])) for name in numeric_keys
        )
        return legacy_bvh_refine_chang_e_event_semantics(meta, desc, legacy)

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
        "theme_label_status": meta.get("theme_label_status", "legacy_or_unknown"),
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


def chang_e_semantic_event_starts(seq: np.ndarray, cfg: MotionGenerationConfig) -> List[int]:
    """Boundary-aware starts for Chang-E long BVH.

    Compact action libraries naturally provide event-sized files. Chang-E files
    are long performances, so we preserve uniform
    coverage and add motion-novelty anchors around energy/yaw/contact changes.
    """
    x = np.asarray(seq, dtype=np.float32)
    T = int(x.shape[0])
    win = max(1, min(int(getattr(cfg, "window_len", 120)), T))
    hop = max(1, int(getattr(cfg, "hop_len", max(1, win // 2))))
    minf = max(1, int(getattr(cfg, "min_event_frames", 45)))
    if T <= max(win, minf):
        return [0]
    starts = set([0, max(0, T - win)])
    for st in range(0, max(1, T - minf + 1), hop):
        starts.add(int(min(max(0, st), max(0, T - minf))))
    try:
        joints = fk_24_np(x)
        v = np.zeros_like(joints)
        if T > 1:
            v[1:] = joints[1:] - joints[:-1]
        energy = np.linalg.norm(v.reshape(T, -1, 3), axis=-1).mean(axis=1)
        root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
        root_v = np.zeros((T,), dtype=np.float32)
        if T > 1:
            root_v[1:] = np.linalg.norm(root[1:, [0, 2]] - root[:-1, [0, 2]], axis=-1)
        yaw = root_yaw_np(x)
        yaw_v = np.zeros((T,), dtype=np.float32)
        if T > 1:
            yaw_v[1:] = np.abs(angle_diff(yaw[1:], yaw[:-1]))
        foot = joints[:, list(DEFAULT_FOOT_JOINTS)]
        floor = np.percentile(foot[..., 1].reshape(-1), 5)
        near = (foot[..., 1] < floor + 0.05).mean(axis=1)
        novelty = energy + 0.65 * root_v + 0.55 * yaw_v + 0.20 * np.abs(np.diff(near, prepend=near[:1]))
        if novelty.size > 7:
            k = np.ones(5, dtype=np.float32) / 5.0
            novelty = np.convolve(novelty, k, mode="same")
        thr = float(np.percentile(novelty, 72)) if novelty.size else 0.0
        order = np.argsort(-novelty)
        extra = 0
        max_extra = int(getattr(cfg, "chang_e_boundary_max_extra_starts", 96))
        min_sep = max(8, min(hop, win // 3))
        selected_centers: List[int] = []
        for c in order.tolist():
            if novelty[int(c)] < thr or extra >= max_extra:
                break
            c = int(c)
            if any(abs(c - q) < min_sep for q in selected_centers):
                continue
            selected_centers.append(c)
            starts.add(int(np.clip(c - win // 2, 0, max(0, T - minf))))
            starts.add(int(np.clip(c - win // 3, 0, max(0, T - minf))))
            extra += 2
    except Exception:
        pass
    out = sorted(starts)
    merged: List[int] = []
    min_start_sep = max(6, min(hop // 2, win // 4))
    tail = max(0, T - win)
    # Event Semantics: if the sequence is only slightly longer than one window, [0:win]
    # and [tail:T] are >95% overlapping twins. Keep one centered representative
    # rather than polluting the RAG DB with near-identical embeddings.
    if 0 < tail < min_start_sep:
        return [int(max(0, tail // 2))]
    for st in out:
        st = int(st)
        # Event Semantics: keep coverage endpoints without creating near-duplicate twins.
        # When a novelty start lies too close to the required tail window, replace
        # the previous start with the exact tail start instead of appending both.
        if st == 0:
            if not merged:
                merged.append(0)
            continue
        if st == tail:
            if merged and abs(st - merged[-1]) < min_start_sep:
                # Event Semantics: protect the unique opener for very short sequences.
                # If T is only slightly larger than win, tail can be only a few
                # frames after 0. Replacing [0] with [tail] permanently drops
                # opening frames. Keep the opener and add the tail only in this
                # unique two-endpoint case; otherwise replace the near-duplicate.
                if len(merged) == 1 and merged[-1] == 0 and st != 0:
                    merged.append(st)
                elif not (len(merged) == 1 and merged[-1] == 0):
                    merged[-1] = st
            elif not merged or merged[-1] != st:
                merged.append(st)
            continue
        if not merged or abs(st - merged[-1]) >= min_start_sep:
            merged.append(st)
    if merged:
        if abs(tail - merged[-1]) < min_start_sep:
            if len(merged) == 1 and merged[-1] == 0 and tail != 0:
                merged.append(tail)
            elif not (len(merged) == 1 and merged[-1] == 0):
                merged[-1] = tail
        elif merged[-1] != tail:
            merged.append(tail)
    else:
        merged = [0, tail] if tail > 0 else [0]
    return sorted(set(int(x) for x in merged))


def legacy_bvh_refine_chang_e_event_semantics(meta: dict, desc: Optional[np.ndarray], prof: Dict[str, object]) -> Dict[str, object]:
    # Event Semantics window-level semantics for Chang-E event slicing.
    # Chang-E is a long, category-complete MoCap corpus; each local window is
    # converted into a curated event with explicit recording provenance.
    out = dict(prof)
    key = _safe_profile_key(meta)
    nums = _parse_numeric_semantic(out)
    if desc is not None and len(desc) >= 20:
        nums["duration"] = float(desc[0])
        nums["travel"] = max(float(nums.get("travel", 0.0)), float(np.clip(desc[1] / 1.15, 0.0, 1.0)))
        nums["energy"] = max(float(nums.get("energy", 0.0)), float(np.clip(desc[5] / 0.135, 0.0, 1.0)))
        nums["lower"] = max(float(nums.get("lower", 0.0)), float(np.clip(desc[7] / 0.10, 0.0, 1.0)))
        nums["upper"] = max(float(nums.get("upper", 0.0)), float(np.clip(desc[8] / 0.10, 0.0, 1.0)))
        nums["turn"] = max(float(nums.get("turn", 0.0)), float(np.clip(abs(desc[17]) / 0.20, 0.0, 1.0)))
        nums["root_y_range"] = float(desc[18])
        nums["contact_ratio"] = float(desc[10])
        local_hold = float(np.clip(1.0 - desc[5] / 0.115, 0.0, 1.0)) * float(np.clip(0.55 + desc[10], 0.0, 1.0))
        nums["pose_hold"] = max(float(nums.get("pose_hold", 0.0)), local_hold)
        nums["jump"] = max(float(nums.get("jump", 0.0)), float(np.clip((desc[18] - 0.035) / 0.16, 0.0, 1.0)))
        nums["spin"] = max(float(nums.get("spin", 0.0)), float(np.clip(abs(desc[17]) / 0.22, 0.0, 1.0)))
    frac_mid = _float_meta(meta, "event_position_mid", _float_meta(meta, "event_position_fraction", 0.5))
    duration = float(nums.get("duration", _float_meta(meta, "duration", 0.0)))
    energy = float(nums.get("energy", 0.0)); onset = float(nums.get("onset", 0.0))
    travel = float(nums.get("travel", 0.0)); turn = float(nums.get("turn", 0.0))
    lower = float(nums.get("lower", 0.0)); upper = float(nums.get("upper", 0.0))
    pose_hold = float(nums.get("pose_hold", 0.0)); jump = float(nums.get("jump", 0.0)); spin = float(nums.get("spin", 0.0))
    contact_ratio = float(nums.get("contact_ratio", 0.5))
    if pose_hold > 0.72 and energy < 0.60:
        family = "pose_motif"
    elif turn > 0.68 or spin > 0.68:
        family = "turning_flow"
    elif onset > 0.72 or (energy > 0.78 and key == "lei_gong_drum"):
        family = "percussive_accent"
    elif key == "pipa_behind_back" and upper >= lower * 1.12:
        family = "instrument_motif"
    elif travel > 0.50 or lower > upper * 1.18:
        family = "footwork_flow"
    elif key == "flying_apsaras" or jump > 0.45:
        family = "aerial_curve"
    elif energy < 0.34:
        family = "calm_flow"
    else:
        family = str(out.get("event_family", "footwork_flow"))
    out["event_family"] = family
    if frac_mid < 0.12:
        stage = "intro"
    elif frac_mid > 0.88:
        stage = "resolution"
    elif family in {"percussive_accent", "turning_flow"} and energy > 0.55:
        stage = "climax"
    elif family == "instrument_motif":
        stage = "motif_recall"
    elif pose_hold > 0.70:
        stage = "anchor_or_resolution"
    elif energy > 0.58 or travel > 0.55:
        stage = "build_up"
    else:
        stage = "development"
    out["motion_stage_role"] = stage
    if contact_ratio < 0.18 or jump > 0.45:
        out["support_label"] = "low_contact_flight_like"
    elif pose_hold > 0.72:
        out["support_label"] = "stable_support"
    elif turn > 0.55:
        out["support_label"] = "alternating_or_pivot_support"
    elif travel > 0.45 or lower > 0.60:
        out["support_label"] = "alternating_foot_support"
    else:
        out.setdefault("support_label", "stable_support")
    if turn > 0.60:
        out["locomotion_label"] = "turning_travel"; out["spatial_label"] = "turning"
    elif travel > 0.50:
        out["locomotion_label"] = "traveling_steps"; out["spatial_label"] = "traveling"
    elif pose_hold > 0.70:
        out["locomotion_label"] = "in_place_pose"; out["spatial_label"] = "in_place"
    elif energy < 0.34:
        out["locomotion_label"] = "slow_weight_shift"; out["spatial_label"] = "in_place"
    family_to_align = {"calm_flow": "calm_meditative", "pose_motif": "pose_hold", "footwork_flow": "footwork_flow", "turning_flow": "turning_climax", "instrument_motif": "instrument_phrase", "percussive_accent": "percussive_accent", "aerial_curve": "lyrical_flow"}
    out["music_alignment_label"] = family_to_align.get(family, str(out.get("music_alignment_label", "lyrical_flow")))
    tags = [out["music_alignment_label"], family, stage, str(out.get("support_label", "")), str(out.get("locomotion_label", ""))]
    tags += [str(x) for x in out.get("music_alignment_tags", [])]
    out["music_alignment_tags"] = list(dict.fromkeys([x for x in tags if x]))
    out["event_position_mid"] = float(frac_mid)
    natural_range = out.get("natural_duration_range_sec", CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]).get("natural_duration_range_sec", [1.5, 4.0]))
    q = chang_e_event_quality_from_numbers(nums, family, duration, natural_range)
    out["event_quality_score"] = float(q)
    out["semantic_confidence"] = float(np.clip(0.25 + 0.50 * q + 0.25 * max(energy, pose_hold, turn, travel, upper, lower), 0.10, 1.0))
    keys = ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]
    out["semantic_numeric"] = ";".join(str(float(nums.get(k, 0.0))) for k in keys)
    out["classification_text"] = (
        f"action={key}; motif={out.get('cultural_motif')}; family={out.get('event_family')}; "
        f"stage={out.get('motion_stage_role')}; support={out.get('support_label')}; "
        f"locomotion={out.get('locomotion_label')}; music_align={out.get('music_alignment_label')}; "
        f"event_mid={float(frac_mid):.3f}; quality={q:.3f}; semantic_conf={out['semantic_confidence']:.3f}; numeric={out['semantic_numeric']}"
    )
    return out


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


def class_semantic_vector_from_meta(meta: dict, cfg: Optional[MotionGenerationConfig] = None) -> np.ndarray:
    key = _safe_profile_key(meta)
    base = filename_semantic_vector_from_meta(meta, cfg).copy()
    cls = strong_action_semantics_from_meta(meta)
    nums = _parse_numeric_semantic(cls)
    for k in ["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]:
        nums.setdefault(k, float(CHANG_E_CATEGORY_PROFILES.get(key, CHANG_E_CATEGORY_PROFILES["unknown"]).get(k, 0.0)))
    known = [k for k in CHANG_E_CATEGORY_PROFILES.keys() if k != "unknown"]
    ci = known.index(key) if key in known else -1
    align_i = _label_index(str(cls.get("music_alignment_label")), MUSIC_ALIGNMENT_LABELS)
    family_i = _label_index(str(cls.get("event_family")), EVENT_FAMILY_LABELS)
    stage_i = _label_index(str(cls.get("motion_stage_role")), STAGE_ROLE_LABELS)
    v = base.astype(np.float32)
    for i,k in enumerate(["energy", "onset", "travel", "turn", "lower", "upper", "floorwork", "jump", "spin", "pose_hold", "instrument", "prop"]):
        v[8+i] = nums[k]
    v[20] = 0.0 if ci < 0 else ci / max(1, len(known) - 1)
    v[21] = 0.0 if align_i < 0 else align_i / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)
    v[22] = 0.0 if family_i < 0 else family_i / max(1, len(EVENT_FAMILY_LABELS) - 1)
    v[23] = 0.0 if stage_i < 0 else stage_i / max(1, len(STAGE_ROLE_LABELS) - 1)
    tags = set(str(x) for x in cls.get("music_alignment_tags", []))
    v[28] = 1.0 if ("calm_meditative" in tags or cls.get("event_family") == "calm_flow") else 0.0
    v[29] = 1.0 if ("percussive_accent" in tags or nums["onset"] > 0.65) else 0.0
    v[30] = 1.0 if ("turning_climax" in tags or nums["spin"] > 0.65 or nums["turn"] > 0.65) else 0.0
    v[31] = 1.0
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def audio_slot_classification_from_pseudo(pseudo: np.ndarray, duration: float, energy: float, onset: float, dyn: float) -> Dict[str, object]:
    """Infer slot-level music labels for unpaired alignment and schedule reports."""
    if energy < 0.035 and onset < 0.015:
        label = "calm_meditative"
        role = "calm"
        preferred = ["revelation_meditation", "thirty_six_postures", "lotus_steps"]
    elif onset > 0.10 or (energy > 0.08 and dyn > 0.04):
        label = "percussive_accent"
        role = "climax" if energy > 0.08 else "build_up"
        preferred = ["lei_gong_drum", "pipa_behind_back", "ribbon_flow"]
    elif energy > 0.065:
        label = "turning_climax"
        role = "build_up"
        preferred = ["ribbon_flow", "pipa_behind_back", "lei_gong_drum"]
    elif duration > 5.0 and energy < 0.055:
        label = "pose_hold"
        role = "release"
        preferred = ["thirty_six_postures", "revelation_meditation", "lotus_steps"]
    else:
        label = "lyrical_flow"
        role = "normal"
        preferred = ["lotus_steps", "thirty_six_postures", "ribbon_flow", "pipa_behind_back"]
    if label == "percussive_accent":
        energy_label, rhythm_label = "high", "percussive"
    elif label == "calm_meditative":
        energy_label, rhythm_label = "calm", "sustained"
    elif label == "pose_hold":
        energy_label, rhythm_label = "calm", "sustained"
    else:
        energy_label, rhythm_label = "moderate", "lyrical"
    return {
        "role": role,
        "music_alignment_label": label,
        "energy_label": energy_label,
        "rhythm_label": rhythm_label,
        "preferred_dance_keys": preferred,
        "preferred_semantic_roles": [CHANG_E_CATEGORY_PROFILES.get(k, {}).get("semantic_role", CATEGORY_CLASS_OVERRIDES.get(k, {}).get("semantic_role", "")) for k in preferred],
    }




# External Music Semantics external classical-music semantic interface.
# Supported labels are deliberately identical to RAG event music_alignment labels,
# so a trained classical model can supervise retrieval without paired BVH/audio.
MUSIC_SEMANTIC_LABELS = MUSIC_ALIGNMENT_LABELS
MUSIC_LABEL_TO_ACTION = {
    "calm_meditative": ["revelation_meditation", "thirty_six_postures", "lotus_steps"],
    "pose_hold": ["thirty_six_postures", "revelation_meditation", "lotus_steps"],
    "lyrical_flow": ["lotus_steps", "ribbon_flow", "revelation_meditation", "thirty_six_postures"],
    "instrument_phrase": ["pipa_behind_back", "lotus_steps", "ribbon_flow"],
    "percussive_accent": ["lei_gong_drum", "pipa_behind_back", "ribbon_flow"],
    "turning_climax": ["ribbon_flow", "lei_gong_drum", "pipa_behind_back"],
    "footwork_flow": ["lotus_steps", "ribbon_flow", "thirty_six_postures"],
}
MUSIC_PROXY_NAME_LABELS = {
    "pipa": "instrument_phrase",
    "guzhen": "lyrical_flow",
    "guzheng": "lyrical_flow",
    "xiao": "calm_meditative",
    "gu": "percussive_accent",
    "drum": "percussive_accent",
    "luo": "percussive_accent",
    "gong": "percussive_accent",
}


def _split_path_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts: List[str] = []
    for chunk in text.replace(";", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def canonical_music_label(label: object) -> str:
    text = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "meditation": "calm_meditative",
        "meditative": "calm_meditative",
        "calm": "calm_meditative",
        "tranquil": "calm_meditative",
        "hold": "pose_hold",
        "pose": "pose_hold",
        "sustain": "pose_hold",
        "lyrical": "lyrical_flow",
        "flow": "lyrical_flow",
        "melodic": "lyrical_flow",
        "instrument": "instrument_phrase",
        "pipa": "instrument_phrase",
        "plucked": "instrument_phrase",
        "percussive": "percussive_accent",
        "accent": "percussive_accent",
        "drum": "percussive_accent",
        "climax": "turning_climax",
        "turn": "turning_climax",
        "turning": "turning_climax",
        "footwork": "footwork_flow",
        "steps": "footwork_flow",
        "step": "footwork_flow",
    }
    if text in aliases:
        return aliases[text]
    if text in MUSIC_SEMANTIC_LABELS:
        return text
    return text if text else "lyrical_flow"


def normalize_music_probs(probs: object = None, top_label: object = None, temperature: float = 0.65) -> Dict[str, float]:
    out = {k: 0.0 for k in MUSIC_SEMANTIC_LABELS}
    if isinstance(probs, dict):
        for k, v in probs.items():
            ck = canonical_music_label(k)
            if ck in out:
                try:
                    out[ck] += max(0.0, float(v))
                except Exception:
                    pass
    elif probs is not None:
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        for i, v in enumerate(arr[:len(MUSIC_SEMANTIC_LABELS)]):
            out[MUSIC_SEMANTIC_LABELS[i]] += max(0.0, float(v))
    if sum(out.values()) <= 1e-8 and top_label is not None:
        label = canonical_music_label(top_label)
        if label in out:
            out[label] = 1.0
    if sum(out.values()) <= 1e-8:
        out["lyrical_flow"] = 1.0
    # Temperature-sharpen while retaining soft alternatives from external model.
    temp = max(0.05, float(temperature))
    vals = np.asarray([out[k] for k in MUSIC_SEMANTIC_LABELS], dtype=np.float32)
    vals = np.power(np.maximum(vals, 0.0) + 1e-8, 1.0 / temp)
    vals = vals / max(float(vals.sum()), 1e-8)
    return {k: float(vals[i]) for i, k in enumerate(MUSIC_SEMANTIC_LABELS)}


def music_semantic_slot_from_probs(probs: Dict[str, float], duration: float = 4.0, source: str = "external") -> Dict[str, object]:
    top_label = max(probs.items(), key=lambda kv: kv[1])[0]
    preferred = MUSIC_LABEL_TO_ACTION.get(top_label, ["lotus_steps", "thirty_six_postures"])
    if top_label == "percussive_accent":
        role, energy_label, rhythm_label = "climax", "percussive", "percussive"
    elif top_label == "turning_climax":
        role, energy_label, rhythm_label = "build_up", "high", "accented"
    elif top_label in {"calm_meditative", "pose_hold"}:
        role, energy_label, rhythm_label = "calm" if top_label == "calm_meditative" else "release", "calm", "sustained"
    elif top_label == "instrument_phrase":
        role, energy_label, rhythm_label = "normal", "moderate", "accented"
    elif top_label == "footwork_flow":
        role, energy_label, rhythm_label = "normal", "moderate", "lyrical"
    else:
        role, energy_label, rhythm_label = "normal", "moderate", "lyrical"
    return {
        "role": role,
        "music_alignment_label": top_label,
        "energy_label": energy_label,
        "rhythm_label": rhythm_label,
        "preferred_dance_keys": preferred,
        "preferred_semantic_roles": [str(CHANG_E_CATEGORY_PROFILES.get(k, {}).get("semantic_role", "")) for k in preferred],
        "music_semantic_probs": {k: float(v) for k, v in probs.items()},
        "music_semantic_top_label": top_label,
        "external_music_semantic_source": source,
    }


def music_probs_to_pseudo_feature(probs: Dict[str, float], duration: float, cfg: MotionGenerationConfig) -> np.ndarray:
    # Label prototypes occupy the same descriptor layout used by motion events.
    prototypes = {
        "calm_meditative": (0.020, 0.010, 0.012, 0.85),
        "pose_hold": (0.030, 0.012, 0.010, 0.78),
        "lyrical_flow": (0.055, 0.035, 0.040, 0.42),
        "instrument_phrase": (0.070, 0.065, 0.055, 0.30),
        "percussive_accent": (0.105, 0.125, 0.100, 0.12),
        "turning_climax": (0.095, 0.080, 0.110, 0.08),
        "footwork_flow": (0.065, 0.045, 0.070, 0.34),
    }
    energy = onset = dyn = calm = 0.0
    for label, p in probs.items():
        e, o, d, c = prototypes.get(label, prototypes["lyrical_flow"])
        energy += float(p) * e
        onset += float(p) * o
        dyn += float(p) * d
        calm += float(p) * c
    pseudo = np.zeros(32, dtype=np.float32)
    pseudo[0] = float(duration)
    pseudo[1] = energy * 2.0
    pseudo[2] = energy
    pseudo[3] = max(energy, dyn)
    pseudo[4] = dyn
    pseudo[5] = energy + onset
    pseudo[6] = onset + dyn
    pseudo[7] = energy + 0.5 * onset
    pseudo[8] = energy
    pseudo[9] = 1.0 + onset
    pseudo[10] = calm
    pseudo[13] = max(0.02, onset)
    pseudo[14] = onset
    pseudo[16] = onset
    pseudo[17] = dyn
    pseudo[18] = dyn
    sem = music_semantic_slot_from_probs(probs, duration, source="external")
    pseudo[22] = _label_index(str(sem["energy_label"]), ENERGY_LABELS) / max(1, len(ENERGY_LABELS) - 1)
    pseudo[23] = _label_index(str(sem["rhythm_label"]), RHYTHM_LABELS) / max(1, len(RHYTHM_LABELS) - 1)
    pseudo[26] = _label_index(str(sem["music_alignment_label"]), MUSIC_ALIGNMENT_LABELS) / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)
    pseudo[28] = float(probs.get("calm_meditative", 0.0) + 0.5 * probs.get("pose_hold", 0.0))
    pseudo[29] = float(probs.get("percussive_accent", 0.0) + 0.4 * probs.get("instrument_phrase", 0.0))
    pseudo[30] = float(probs.get("turning_climax", 0.0) + 0.3 * probs.get("footwork_flow", 0.0))
    pseudo[31] = 1.0
    return pseudo.astype(np.float32)


def sidecar_music_semantic_candidates(audio_path: str | Path, cfg: MotionGenerationConfig) -> List[Path]:
    p = Path(audio_path)
    names = [
        f"{p.stem}.music_semantic.json", f"{p.stem}_music_semantic.json", f"{p.stem}.semantic.json", f"{p.stem}_semantic.json",
        f"{p.stem}.music_semantic.npz", f"{p.stem}_music_semantic.npz", f"{p.stem}.semantic.npz", f"{p.stem}_semantic.npz",
        f"{p.stem}.json", f"{p.stem}.npz",
    ]
    cands: List[Path] = [p.with_name(n) for n in names]
    for d in _split_path_list(getattr(cfg, "external_music_semantic_dirs", "")):
        dp = Path(d)
        for n in names:
            cands.append(dp / n)
    return cands


def run_external_music_semantic_cmd(audio_path: str | Path, cfg: MotionGenerationConfig) -> Optional[Path]:
    cmd = str(getattr(cfg, "external_music_semantic_cmd", "") or "").strip()
    if not cmd:
        return None
    cache = ensure_dir(getattr(cfg, "external_music_semantic_cache_dir", "output/motion_external_music_semantic_cache"))
    audio_p = Path(audio_path)
    out_json = cache / f"{audio_p.stem}.music_semantic.json"
    out_npz = cache / f"{audio_p.stem}.music_semantic.npz"
    if out_json.exists() or out_npz.exists():
        return out_json if out_json.exists() else out_npz
    expanded = cmd.format(audio=str(audio_p), out_json=str(out_json), out_npz=str(out_npz), stem=audio_p.stem)
    try:
        subprocess.run(expanded, shell=True, check=True)
    except Exception as exc:
        print(f"[External Music Semantics WARN] external music semantic command failed for {audio_p}: {exc}", file=sys.stderr)
        return None
    if out_json.exists() or out_npz.exists():
        return out_json if out_json.exists() else out_npz
    return None




def filename_proxy_music_semantic(audio_path: str | Path, cfg: MotionGenerationConfig, slot_seconds: float) -> Optional[Tuple[List[dict], np.ndarray]]:
    if not bool(getattr(cfg, "external_music_semantic_filename_proxy", True)):
        return None
    stem = Path(audio_path).stem.lower()
    label = None
    for key, lab in MUSIC_PROXY_NAME_LABELS.items():
        if key in stem:
            label = lab
            break
    if label is None:
        return None
    try:
        sr, wav = read_wav_mono(audio_path)
        total = len(wav) / max(float(sr), 1.0)
    except Exception:
        total = float(slot_seconds)
    n = max(1, int(math.ceil(total / max(float(slot_seconds), 1e-3))))
    probs = normalize_music_probs(None, label, getattr(cfg, "external_music_semantic_temperature", 0.65))
    slots, feats = [], []
    for i in range(n):
        start = i * float(slot_seconds)
        end = min(total, (i + 1) * float(slot_seconds)) if total > 0 else (i + 1) * float(slot_seconds)
        dur = max(end - start, 1e-3)
        pseudo = music_probs_to_pseudo_feature(probs, dur, cfg)
        sem = music_semantic_slot_from_probs(probs, dur, source=f"filename_proxy:{Path(audio_path).name}")
        item = {"slot_id": i, "start": start, "end": end, "duration": dur, "energy": float(pseudo[2]), "onset": float(pseudo[16])}
        item.update(sem)
        slots.append(item)
        feats.append(pseudo)
    return slots, np.stack(feats).astype(np.float32)



def semantic_label_match_bonus(slot: dict, db: dict, cfg: MotionGenerationConfig) -> np.ndarray:
    """Event Semantics interpretable music-router bonus for Chang-E semantic Event-RAG."""
    n = len(db.get("paths", []))
    bonus = np.zeros(n, dtype=np.float32)
    if not bool(getattr(cfg, "classification_semantic_enable", True)) or n == 0:
        return bonus
    dance_keys = np.asarray(db.get("dance_keys", np.array(["unknown"] * n, dtype=object)), dtype=object)
    roles = np.asarray(db.get("semantic_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)
    energy = np.asarray(db.get("energy_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    rhythm = np.asarray(db.get("rhythm_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    align = np.asarray(db.get("music_alignment_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    families = np.asarray(db.get("event_families", np.array(["unknown"] * n, dtype=object)), dtype=object)
    stages = np.asarray(db.get("motion_stage_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)
    motifs = np.asarray(db.get("cultural_motifs", np.array(["unknown"] * n, dtype=object)), dtype=object)
    locomotion = np.asarray(db.get("locomotion_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    support = np.asarray(db.get("support_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    quality = np.asarray(db.get("event_quality_scores", np.ones(n, dtype=np.float32)), dtype=np.float32)
    preferred = [canonicalize_chang_e_key(x) for x in slot.get("preferred_dance_keys", [])]
    preferred_roles = [str(x) for x in slot.get("preferred_semantic_roles", [])]
    slot_align = str(slot.get("music_alignment_label", slot.get("music_semantic_top_label", "")))
    slot_energy = str(slot.get("energy_label", "")); slot_rhythm = str(slot.get("rhythm_label", "")); slot_role = str(slot.get("role", "normal"))
    route_family_map = {"calm_meditative": ["pose_hold", "floorwork", "upper_body_gesture"], "pose_hold": ["pose_hold", "floorwork", "upper_body_gesture"], "lyrical_flow": ["upper_body_gesture", "jump_aerial", "locomotion", "floorwork", "transition"], "footwork_flow": ["locomotion", "turn_spin", "rhythmic_accent"], "instrument_phrase": ["upper_body_gesture", "pose_hold"], "percussive_accent": ["rhythmic_accent", "locomotion", "turn_spin"], "turning_climax": ["turn_spin", "jump_aerial", "rhythmic_accent"], "aerial_curve": ["jump_aerial", "turn_spin", "upper_body_gesture"]}
    route_stage_map = {"intro": ["intro", "intro_or_resolution", "anchor_or_resolution"], "calm": ["intro", "intro_or_resolution", "resolution", "anchor_or_resolution"], "normal": ["development", "build_up", "motif_recall"], "development": ["development", "build_up"], "build_up": ["build_up", "development", "opening_or_climax"], "motif": ["motif_recall", "development"], "motif_recall": ["motif_recall", "anchor_or_resolution"], "accent": ["accent_or_climax", "climax", "build_up"], "climax": ["climax", "accent_or_climax", "opening_or_climax"], "release": ["resolution", "anchor_or_resolution", "intro_or_resolution"], "resolution": ["resolution", "anchor_or_resolution", "intro_or_resolution"]}
    route_support_map = {"calm_meditative": ["stable_support", "static_or_low_motion_support"], "pose_hold": ["stable_support", "static_or_low_motion_support"], "footwork_flow": ["alternating_foot_support", "alternating_or_pivot_support"], "turning_climax": ["alternating_or_pivot_support", "low_contact_flight_like"], "percussive_accent": ["strong_foot_contact", "alternating_foot_support"], "lyrical_flow": ["alternating_foot_support", "low_contact_flight_like", "stable_support"], "instrument_phrase": ["stable_support", "alternating_foot_support"]}
    route_loco_map = {"calm_meditative": ["slow_weight_shift", "in_place_pose", "floating_leaning"], "pose_hold": ["in_place_pose", "slow_weight_shift"], "footwork_flow": ["traveling_steps", "turning_travel"], "turning_climax": ["turning_travel", "floating_leaning"], "percussive_accent": ["accented_travel", "turning_travel", "traveling_steps"], "lyrical_flow": ["floating_leaning", "traveling_steps", "upper_body_phrase"], "instrument_phrase": ["upper_body_phrase", "in_place_pose"]}
    for k in preferred:
        if k:
            bonus += (dance_keys == k).astype(np.float32) * float(getattr(cfg, "preferred_dance_key_bonus", 0.28))
    for r in preferred_roles:
        if r:
            bonus += (roles == r).astype(np.float32) * 0.25
    if slot_align:
        bonus += (align == slot_align).astype(np.float32) * 0.45
        for rank, fam in enumerate(route_family_map.get(slot_align, [])):
            bonus += (families == fam).astype(np.float32) * float(getattr(cfg, "event_family_bonus", 0.58)) / float(rank + 1)
        for rank, sup in enumerate(route_support_map.get(slot_align, [])):
            bonus += (support == sup).astype(np.float32) * float(getattr(cfg, "route_support_bonus", 0.12)) / float(rank + 1)
        for rank, loc in enumerate(route_loco_map.get(slot_align, [])):
            bonus += (locomotion == loc).astype(np.float32) * float(getattr(cfg, "route_locomotion_bonus", 0.14)) / float(rank + 1)
    if slot_role:
        for rank, st in enumerate(route_stage_map.get(slot_role, [])):
            bonus += (stages == st).astype(np.float32) * float(getattr(cfg, "motion_stage_role_bonus", 0.36)) / float(rank + 1)
    if slot_energy:
        bonus += (energy == slot_energy).astype(np.float32) * 0.14
    if slot_rhythm:
        bonus += (rhythm == slot_rhythm).astype(np.float32) * 0.12
    if slot_align == "instrument_phrase":
        bonus += np.isin(motifs, ["pipa_instrument_pose", "thunder_drum"]).astype(np.float32) * 0.16
    conf = np.asarray(db.get("semantic_confidence", np.ones(n, dtype=np.float32)), dtype=np.float32)
    q_gate = np.clip(0.45 + 0.55 * quality, 0.25, 1.15)
    bonus *= np.clip(0.65 + 0.35 * conf, 0.5, 1.15) * q_gate
    # Event Semantics: never normalize by the current candidate max.  That dynamic
    # min-max scaling can turn a weak accidental match in a vague slot into a
    # full-strength routing reward.  Use a fixed saturating scale instead.
    scale = max(0.25, float(getattr(cfg, "route_semantic_bonus_scale", 1.50)))
    bonus = 1.0 - np.exp(-np.maximum(bonus, 0.0) / scale)
    return np.clip(bonus, 0.0, 1.0).astype(np.float32)


def parse_change_bvh_semantics(path: str | Path) -> Dict[str, object]:
    """Resolve Chang-E semantics, preferring the authoritative source manifest.

    Examples:
      female_36pose_1.bvh -> female performer track 1 in one shared recording
      male_pipa_2.bvh     -> gender=male,   category=pipa_behind_back, take=2
      female_lotus.bvh    -> gender=female, category=lotus_steps

    ``source_uid`` remains file-specific, while ``recording_uid`` groups
    synchronized performer tracks so train/validation/test splitting cannot
    leak one choreography through another dancer file.
    """
    # This dependency is deliberately lazy: the historical BVH manifest must
    # not be loaded or consulted by the formal official-SMPL Event-DB path.
    from data_pipeline.chang_e_manifest import (
        semantic_metadata as chang_e_semantic_metadata,
    )

    stem = _clean_stem(path)
    manifest_meta = chang_e_semantic_metadata(path)
    tokens = [t for t in stem.split("_") if t]
    gender = "unknown"
    rest = tokens[:]
    if rest and rest[0] in {"male", "female"}:
        gender = rest[0]
        rest = rest[1:]

    take_id: Optional[int] = None
    if rest and rest[-1].isdigit():
        take_id = int(rest[-1])
        rest = rest[:-1]
    base = "_".join(rest) if rest else stem

    category_key = "unknown"
    for key, prof in CHANG_E_CATEGORY_PROFILES.items():
        aliases = set(prof.get("aliases", set()))
        if base in aliases or any(tok in aliases for tok in rest):
            category_key = key
            break
    if category_key == "unknown":
        category_key = canonicalize_chang_e_key(base or "unknown")
    else:
        category_key = canonicalize_chang_e_key(category_key)

    if manifest_meta is not None:
        gender = str(manifest_meta.get("performer_group") or gender)
        category_key = canonicalize_chang_e_key(
            manifest_meta.get("dance_category") or category_key
        )
        raw_take = manifest_meta.get("take_id")
        take_id = int(raw_take) if raw_take is not None else None

    prof = LEGACY_BVH_CATEGORY_PROFILES.get(category_key, {})
    display = str(prof.get("display", category_key.replace("_", " ").title()))
    semantic_role = str(prof.get("semantic_role", "unknown_motion"))
    source_uid = str(
        manifest_meta.get("source_id") if manifest_meta is not None else stem
    )
    recording_uid = str(
        manifest_meta.get("recording_uid")
        if manifest_meta is not None
        else source_uid
    )
    take_text = f" take {take_id}" if take_id is not None else ""
    gender_text = "female" if gender == "female" else ("male" if gender == "male" else "unknown-gender")
    semantic_text = f"{gender_text} {display}{take_text}; role={semantic_role}"
    event_label = category_key
    if take_id is not None:
        event_label = f"{category_key}_take{take_id}"
    return {
        "source_uid": source_uid,
        "source_group": source_uid,
        "recording_uid": recording_uid,
        "performer_track_id": (
            manifest_meta.get("performer_track_id")
            if manifest_meta is not None
            else -1
        ),
        "sequence_index": (
            manifest_meta.get("sequence_index")
            if manifest_meta is not None
            else (take_id if take_id is not None else 1)
        ),
        "gender": gender,
        "dance_key": category_key,
        "dance_category": display,
        "semantic_role": semantic_role,
        "semantic_text": semantic_text,
        "take_id": take_id if take_id is not None else -1,
        "source_take": take_id if take_id is not None else -1,
        "label": event_label,
        "parent_label": category_key,
        "raw_stem": stem,
    }


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


def source_group_from_path(path: str | Path) -> str:
    # File-level identity remains distinct from recording_uid.  Both are saved
    # in the Event-DB and audited independently.
    return str(parse_change_bvh_semantics(path).get("source_group"))


def infer_label_from_filename(path: str | Path) -> str:
    return str(parse_change_bvh_semantics(path).get("label"))


def filename_semantic_vector_from_meta(meta: dict, cfg: Optional[MotionGenerationConfig] = None) -> np.ndarray:
    """Convert filename/category semantics into a 32D slot-compatible prior.

    The vector uses the same coarse channel layout as audio_slots()/event_descriptor():
    duration, root/travel, energy, dynamics, joint/lower/upper density, contact
    calmness, foot speed, turn/onset and root-y dynamics.  It is not a paired
    label; it is a weak prior that makes unpaired music-to-event matching aware
    that e.g. drum should prefer onset-rich slots and meditation should prefer
    calm slots.
    """
    if str(meta.get("source_format", "")) == "chang_e_official_smpl":
        # Formal SMPL filenames/themes are provenance context, not local motion
        # features.  Local kinematics are encoded by class_semantic_vector.
        v = np.zeros(32, dtype=np.float32)
        v[0] = float(meta.get("duration", 0.0) or 0.0)
        return v

    key = canonicalize_chang_e_key(meta.get("dance_key") or meta.get("parent_label") or meta.get("label") or "unknown")
    prof = LEGACY_BVH_CATEGORY_PROFILES.get(key, {})
    duration = float(meta.get("duration", 0.0) or 0.0)
    energy = float(prof.get("energy", 0.40))
    onset = float(prof.get("onset", 0.20))
    lower = float(prof.get("lower", energy))
    upper = float(prof.get("upper", energy))
    turn = float(prof.get("turn", 0.15))
    travel = float(prof.get("travel", 0.25))
    calm = float(prof.get("calmness", max(0.0, 0.75 - energy)))
    v = np.zeros(32, dtype=np.float32)
    v[0] = duration
    v[1] = travel * 2.0
    v[2] = energy
    v[3] = min(1.0, energy + 0.25 * onset)
    v[4] = energy * 0.45 + onset * 0.55
    v[5] = energy + onset
    v[6] = min(1.0, energy + 0.45 * onset)
    v[7] = lower
    v[8] = upper
    v[9] = lower / max(upper, 1e-4)
    v[10] = calm
    v[11] = calm * 0.8
    v[12] = calm * 0.8
    v[13] = max(0.02, onset)
    v[14] = max(0.02, onset * 1.25)
    v[15] = turn
    v[16] = abs(turn) + 0.25 * onset
    v[17] = max(abs(turn), onset)
    v[18] = energy * (0.25 + 0.5 * (1.0 - calm))
    v[19] = 0.15 + 0.25 * energy
    # Gender is not a quality score. It is only kept as a small style flag so
    # the router can report source diversity; it should not dominate retrieval.
    gender = str(meta.get("gender", "unknown"))
    v[20] = 0.15 if gender == "female" else (0.35 if gender == "male" else 0.25)
    take = float(meta.get("take_id", -1) if meta.get("take_id", -1) is not None else -1)
    v[21] = max(0.0, take) / 4.0 if take >= 0 else 0.0
    # Duplicate coarse stats to high channels for robust normalized matching.
    v[22] = energy
    v[23] = onset
    v[24] = travel
    v[25] = calm
    v[26] = lower
    v[27] = upper
    v[28] = turn
    v[29] = energy + 0.5 * onset
    v[30] = (lower + upper) * 0.5
    v[31] = 1.0
    return v.astype(np.float32)


def motion_feature_z_for_alignment(db: dict, cfg: MotionGenerationConfig, weight: Optional[float] = None) -> np.ndarray:
    """Return descriptor z mixed with filename and strong classification priors."""
    desc_z = np.asarray(db["desc_z"], dtype=np.float32)
    if not bool(getattr(cfg, "filename_semantic_enable", True)):
        return desc_z
    mean = np.asarray(db["desc_mean"], dtype=np.float32)
    std = np.asarray(db["desc_std"], dtype=np.float32)
    parts = []
    if "name_semantic" in db:
        name = np.asarray(db["name_semantic"], dtype=np.float32)
        parts.append((name - mean) / np.maximum(std, 1e-6))
    if bool(getattr(cfg, "classification_semantic_enable", True)) and "class_semantic" in db:
        cls = np.asarray(db["class_semantic"], dtype=np.float32)
        cls_z = (cls - mean) / np.maximum(std, 1e-6)
        if parts:
            ratio = float(getattr(cfg, "classification_semantic_ratio", 0.70))
            ratio = max(0.0, min(1.0, ratio))
            sem_z = (1.0 - ratio) * parts[0] + ratio * cls_z
        else:
            sem_z = cls_z
    elif parts:
        sem_z = parts[0]
    else:
        return desc_z
    sem_z = np.clip(sem_z, -8.0, 8.0).astype(np.float32)
    w = float(getattr(cfg, "filename_semantic_weight", 0.35) if weight is None else weight)
    w = max(0.0, min(0.85, w))
    return ((1.0 - w) * desc_z + w * sem_z).astype(np.float32)


def scan_motion_files(motion_dirs: Sequence[str], exts: Sequence[str] = (".npy", ".npz", ".pkl", ".pickle", ".bvh")) -> List[str]:
    files: List[str] = []
    for d in motion_dirs:
        if not d:
            continue
        p = Path(d)
        if p.is_file():
            if p.suffix.lower() in exts:
                files.append(str(p))
            continue
        if not p.exists():
            continue
        for ext in exts:
            files.extend(glob.glob(str(p / "**" / f"*{ext}"), recursive=True))
    return sorted(set(files))



def resolve_path_from_roots(value: object, roots: Sequence[str | Path]) -> Optional[str]:
    """Resolve manifest paths robustly against manifest dir, repo root, and motion dirs."""
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    p = Path(text)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        for r in roots:
            if not r:
                continue
            candidates.append(Path(r) / p)
            candidates.append(Path(r) / p.name)
    # Direct hit.
    for c in candidates:
        if c.exists():
            return str(c)
    # Fallback: recursive name search inside roots for screenshot-style change/*.bvh.
    name = p.name
    if name:
        for r in roots:
            rr = Path(r)
            if rr.exists() and rr.is_dir():
                hits = list(rr.rglob(name))
                if hits:
                    return str(hits[0])
    return None


def _coerce_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return int(float(text))
    except Exception:
        return default


def _coerce_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return float(text)
    except Exception:
        return default


def resample_motion_to_config_fps(motion: np.ndarray, cfg: MotionGenerationConfig) -> Tuple[np.ndarray, dict]:
    """Resample BVH-derived EDGE-like arrays to cfg.fps before event slicing.

    Timebase Resampling fix:
    channel 0 may temporarily carry BVH native fps from load_bvh_file(), but it
    must never be overwritten with target fps after resampling because EDGE
    channel 0 is a contact channel.  The DB writer rebuilds contacts from FK.
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        return x.astype(np.float32), {"resampled": False, "reason": "too_short"}

    ch0 = x[:, 0]
    finite_ch0 = ch0[np.isfinite(ch0)]
    if finite_ch0.size:
        ch0_med = float(np.nanmedian(finite_ch0))
        ch0_p05 = float(np.nanpercentile(finite_ch0, 5))
        ch0_p95 = float(np.nanpercentile(finite_ch0, 95))
        # Event Semantics: FPS metadata written by the BVH loader is nearly constant, but
        # a few boundary/cropping frames may contain small jitter.  Use the middle
        # 90% band for the main constant-column test and keep raw std only for
        # diagnostics.  This avoids silently skipping required high-FPS -> 30 FPS
        # resampling because of a few outlier rows.
        trimmed = finite_ch0[(finite_ch0 >= ch0_p05) & (finite_ch0 <= ch0_p95)]
        ch0_std = float(np.nanstd(finite_ch0))
        ch0_trimmed_std = float(np.nanstd(trimmed)) if trimmed.size else ch0_std
    else:
        ch0_med = ch0_p05 = ch0_p95 = ch0_std = ch0_trimmed_std = 0.0
    looks_like_fps_metadata = bool(
        ch0_med > 2.0 and ch0_p05 > 1.0 and ch0_p95 < 400.0
        and ch0_trimmed_std < max(0.75, ch0_med * 0.08)
    )
    native = ch0_med if looks_like_fps_metadata else float(cfg.fps)
    target = float(cfg.fps)

    if (not bool(getattr(cfg, "bvh_resample_to_config_fps", True))) or abs(native - target) < 1e-3:
        y = x.copy().astype(np.float32)
        return y, {
            "resampled": False,
            "native_fps": float(native),
            "target_fps": float(target),
            "frames": int(len(x)),
            "fps_metadata_detected": looks_like_fps_metadata,
            "channel0_trimmed_std": float(ch0_trimmed_std),
            "note": "channel0_not_overwritten_EDGE_contact_contract",
        }

    new_len = max(2, int(round(x.shape[0] * target / max(native, 1e-6))))
    y = resample_motion_np(x, new_len).astype(np.float32)
    return y, {
        "resampled": True,
        "native_fps": float(native),
        "target_fps": float(target),
        "frames_before": int(len(x)),
        "frames_after": int(new_len),
        "fps_metadata_detected": looks_like_fps_metadata,
        "channel0_trimmed_std": float(ch0_trimmed_std),
        "note": "channel0_preserved_until_event_contract_guard_rebuilds_contacts",
    }


def read_manifest_records(manifest_path: Optional[str], motion_dirs: Sequence[str], cfg: MotionGenerationConfig) -> List[dict]:
    """
    Read an optional manifest.csv and return normalized source-aware records.

    Supported columns follow the user's prior manifest document:
      source_bvh, fragment_file, fragment_index, label,
      start_frame, end_frame, start_time, end_time, duration_sec, bvh_fps.
    The manifest is an upstream semantic slicing index, not a finished RAG DB;
    we keep its fields as parent metadata and still compute tensor/contact/boundary
    descriptors here.
    """
    if not manifest_path or not bool(getattr(cfg, "manifest_enable", True)):
        return []
    mp = Path(manifest_path)
    if not mp.exists():
        return []
    roots: List[Path] = [mp.parent, Path(__file__).resolve().parents[1]]
    for d in motion_dirs or []:
        roots.append(Path(d))
    rows: List[dict] = []
    with open(mp, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for ridx, row in enumerate(reader):
            fragment = resolve_path_from_roots(row.get("fragment_file") or row.get("path") or row.get("file"), roots)
            source = resolve_path_from_roots(row.get("source_bvh") or row.get("source_file") or row.get("bvh"), roots)
            load_path = fragment or source
            if not load_path:
                continue
            start_frame = _coerce_int(row.get("start_frame"), None)
            end_frame = _coerce_int(row.get("end_frame"), None)
            label = str(row.get("label") or row.get("motion_event") or row.get("fragment_name") or Path(load_path).stem)
            source_name = str(row.get("source_bvh") or Path(source or load_path).name)
            sem = parse_change_bvh_semantics(source or load_path)
            item = {
                "manifest_id": ridx,
                "manifest_path": str(mp),
                "load_path": str(load_path),
                "source_file": str(source or load_path),
                "source_bvh": source_name,
                "fragment_file": str(fragment) if fragment else None,
                "fragment_index": _coerce_int(row.get("fragment_index"), ridx),
                "label": label,
                "parent_label": label,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec_manifest": _coerce_float(row.get("duration_sec"), None),
                "bvh_fps_manifest": _coerce_float(row.get("bvh_fps"), None),
                "raw_manifest_row": {k: row.get(k) for k in row.keys()},
            }
            item.update({k: v for k, v in sem.items() if k not in {"label", "parent_label"}})
            # Manifest labels are manual semantic labels and should remain primary.
            rows.append(item)
    return rows


def iter_source_records(args: argparse.Namespace, cfg: MotionGenerationConfig) -> Tuple[List[dict], dict]:
    """Return load records from manifest when available, otherwise from motion_dirs."""
    motion_dirs = list(getattr(args, "motion_dirs", None) or [])
    manifest_records = read_manifest_records(getattr(args, "manifest", None), motion_dirs, cfg)
    if manifest_records:
        return manifest_records, {"input_mode": "manifest", "manifest_records": len(manifest_records), "motion_dirs": motion_dirs}
    files = scan_motion_files(motion_dirs)
    if not files:
        raise FileNotFoundError(f"No motion files found in: {motion_dirs}")
    out_records = []
    for f in files:
        sem = parse_change_bvh_semantics(f)
        rec = {"load_path": f, "source_file": f, "source_bvh": Path(f).name, "fragment_index": 0}
        rec.update(sem)
        out_records.append(rec)
    return out_records, {"input_mode": "direct_files", "files": len(files), "motion_dirs": motion_dirs}


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
    if matched_audio:
        try:
            music_feat = audio_feature_for_motion_clip(matched_audio, st, clip.shape[0], cfg)
            music_mask = 1.0
        except Exception as exc:
            print(f"[Chang-E Semantics WARN] audio feature failed for {matched_audio}: {exc}", file=sys.stderr)
            music_feat = np.zeros(32, dtype=np.float32)
            music_mask = 0.0
    else:
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
        "source_format": str(base_meta.get("source_format", "legacy_bvh")),
        "source_asset": str(base_meta.get("source_asset", base_meta.get("source_file", ""))),
        "source_bvh": (
            ""
            if str(base_meta.get("source_format", "")) == "chang_e_official_smpl"
            else str(
                base_meta.get(
                    "source_bvh",
                    Path(str(base_meta.get("source_file", "source"))).name,
                )
            )
        ),
        "source_group": source,
        "matched_audio": matched_audio,
        "has_real_audio_feature": bool(music_mask > 0.5),
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
        "label": str(
            base_meta.get("label")
            or (
                "unknown_theme"
                if str(base_meta.get("source_format", "")) == "chang_e_official_smpl"
                else infer_label_from_filename(base_meta.get("source_file", out_path))
            )
        ),
        "parent_label": str(base_meta.get("parent_label", base_meta.get("label", "unknown"))),
        "fragment_index": int(base_meta.get("fragment_index", 0) or 0),
        "manifest_id": base_meta.get("manifest_id"),
        "manifest_path": base_meta.get("manifest_path"),
        "input_mode": base_meta.get("input_mode", "direct_files"),
        "edge151_contract_report": contract_report,
    }
    if item["source_format"] == "chang_e_official_smpl":
        sem = official_smpl_semantics_from_metadata(base_meta)
    else:
        sem = parse_change_bvh_semantics(
            base_meta.get("source_file", base_meta.get("source_bvh", out_path))
        )
    item.update(strong_action_semantics_from_meta({**sem, **item}, desc))
    # Keep source_uid/source_group from filename unless a manifest-specific source
    # explicitly supplied them.  Keep manifest label if present; otherwise use
    # filename category/take label.
    for k in ["source_uid", "source_id", "recording_uid", "sequence_id", "dancer_id", "dancer_id_status", "performer_track_id", "sequence_index", "performer_group", "gender", "dance_key", "dance_category", "dance_theme", "candidate_dance_category", "theme_label_status", "source_context", "manifest_sha256", "coordinate_system", "translation_units", "pose_layout", "semantic_role", "semantic_text", "take_id", "source_take", "raw_stem"]:
        item[k] = base_meta.get(k, sem.get(k))
    strong_sem = strong_action_semantics_from_meta(item, desc)
    item.update(strong_sem)
    if item.get("semantic_text"):
        item["semantic_text"] = str(item["semantic_text"]) + "; " + str(strong_sem.get("classification_text", ""))
    if not item.get("label") or item.get("label") == "unknown":
        item["label"] = str(sem.get("label", "unknown"))
    if not item.get("parent_label") or item.get("parent_label") == "unknown":
        item["parent_label"] = str(sem.get("parent_label", item.get("label", "unknown")))
    if "resample_report" in base_meta:
        item["resample_report"] = base_meta["resample_report"]
    meta.append(item)

def build_db(args: argparse.Namespace) -> int:
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = ensure_dir(args.out_db)
    audio_files = collect_audio_files(getattr(args, "audio_dirs", None))
    records, input_report = iter_source_records(args, cfg)
    audio_match_cache: Dict[str, Optional[str]] = {}

    meta: List[dict] = []
    descs: List[np.ndarray] = []
    entries: List[np.ndarray] = []
    exits: List[np.ndarray] = []
    c0s: List[np.ndarray] = []
    c1s: List[np.ndarray] = []
    music_feats: List[np.ndarray] = []
    music_masks: List[float] = []
    resample_reports: List[dict] = []

    npy_dir = ensure_dir(out_dir / "events")
    event_idx = 0
    for rec in records:
        f = str(rec["load_path"])
        seqs = load_motion_file(f)
        if not seqs:
            continue
        # Manifest rows can reference a full source_bvh plus original high-FPS frame range.
        start_frame = rec.get("start_frame")
        end_frame = rec.get("end_frame")
        source_key_path = rec.get("source_file") or f
        source = source_group_from_path(source_key_path)
        if source_key_path not in audio_match_cache:
            audio_match_cache[str(source_key_path)] = find_matching_audio_for_motion(str(source_key_path), audio_files)
        matched_audio = audio_match_cache.get(str(source_key_path))

        for seq_id, seq_raw in enumerate(seqs):
            seq = np.asarray(seq_raw, dtype=np.float32)
            if start_frame is not None or end_frame is not None:
                a = max(0, int(start_frame or 0))
                b = min(seq.shape[0], int(end_frame if end_frame is not None else seq.shape[0]))
                if b > a:
                    seq = seq[a:b]
            seq, res_report = resample_motion_to_config_fps(seq, cfg)
            resample_reports.append(res_report)
            T = seq.shape[0]
            if T < cfg.min_event_frames:
                continue
            # Manifest rows are motif-level semantic clips.  If enabled, split long
            # motif clips into event-level windows while preserving parent metadata;
            # otherwise preserve each motif as one event.
            if rec.get("manifest_id") is not None and not bool(getattr(cfg, "manifest_secondary_event_split", True)):
                starts = [0]
                win = min(T, cfg.max_event_frames)
            else:
                win = min(int(cfg.window_len), T)
                if bool(getattr(cfg, "chang_e_boundary_event_split", True)):
                    starts = chang_e_semantic_event_starts(seq, cfg)
                else:
                    starts = [0] if T <= cfg.max_event_frames else list(range(0, max(1, T - cfg.min_event_frames + 1), cfg.hop_len))
            for st in starts:
                endf = min(T, st + win)
                if endf - st < cfg.min_event_frames:
                    continue
                clip = seq[st:endf].astype(np.float32)
                if clip.shape[0] > cfg.max_event_frames:
                    clip = clip[: cfg.max_event_frames]
                path = npy_dir / f"event_{event_idx:07d}.npy"
                base_meta = dict(rec)
                event_mid = (float(st) + 0.5 * float(endf - st)) / max(float(T), 1.0)
                base_meta.update({
                    "seq_id": seq_id,
                    "resample_report": res_report,
                    "input_mode": input_report.get("input_mode"),
                    "event_start": int(st),
                    "event_end": int(endf),
                    "event_source_frames": int(T),
                    "event_position_mid": float(event_mid),
                    "event_position_fraction": float(event_mid),
                })
                add_event_to_db_lists(
                    clip=clip,
                    event_idx=event_idx,
                    out_path=path,
                    cfg=cfg,
                    source=source,
                    matched_audio=matched_audio,
                    st=st,
                    base_meta=base_meta,
                    descs=descs,
                    entries=entries,
                    exits=exits,
                    c0s=c0s,
                    c1s=c1s,
                    music_feats=music_feats,
                    music_masks=music_masks,
                    meta=meta,
                )
                event_idx += 1

    if not meta:
        raise RuntimeError("No valid EDGE 151D motion events were built.")
    desc = np.stack(descs).astype(np.float32)
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1e-6
    desc_z = (desc - mean) / std
    name_semantic = np.stack([filename_semantic_vector_from_meta(m, cfg) for m in meta]).astype(np.float32)
    class_semantic = np.stack([class_semantic_vector_from_meta(m, cfg) for m in meta]).astype(np.float32)
    db_path = out_dir / "events.npz"
    np.savez_compressed(
        db_path,
        event_semantics_schema_version=np.asarray(
            "chang_e_five_layer_event_semantics_v2", dtype=object
        ),
        event_descriptor_schema_version=np.asarray(
            "edge151_local_action_descriptor_v2", dtype=object
        ),
        desc=desc.astype(np.float32),
        desc_z=desc_z.astype(np.float32),
        desc_mean=mean.astype(np.float32),
        desc_std=std.astype(np.float32),
        entry=np.stack(entries).astype(np.float32),
        exit=np.stack(exits).astype(np.float32),
        contact_entry=np.stack(c0s).astype(np.float32),
        contact_exit=np.stack(c1s).astype(np.float32),
        paths=np.array([m["path"] for m in meta], dtype=object),
        source_groups=np.array([m["source_group"] for m in meta], dtype=object),
        labels=np.array([m.get("label", "unknown") for m in meta], dtype=object),
        parent_labels=np.array([m.get("parent_label", m.get("label", "unknown")) for m in meta], dtype=object),
        source_bvh=np.array([m.get("source_bvh", "") for m in meta], dtype=object),
        source_assets=np.array([m.get("source_asset", "") for m in meta], dtype=object),
        source_formats=np.array([m.get("source_format", "unknown") for m in meta], dtype=object),
        source_uids=np.array([m.get("source_uid", m.get("source_group", "")) for m in meta], dtype=object),
        recording_uids=np.array([m.get("recording_uid", m.get("source_uid", m.get("source_group", ""))) for m in meta], dtype=object),
        sequence_ids=np.array([m.get("sequence_id", "unknown") for m in meta], dtype=object),
        dancer_ids=np.array([m.get("dancer_id") or "" for m in meta], dtype=object),
        dancer_id_statuses=np.array([m.get("dancer_id_status", "unverified") for m in meta], dtype=object),
        manifest_sha256=np.array([m.get("manifest_sha256") or "" for m in meta], dtype=object),
        performer_track_ids=np.array([int(m.get("performer_track_id", -1) if m.get("performer_track_id", -1) is not None else -1) for m in meta], dtype=np.int32),
        sequence_indices=np.array([int(m.get("sequence_index", -1) if m.get("sequence_index", -1) is not None else -1) for m in meta], dtype=np.int32),
        genders=np.array([m.get("gender", "unknown") for m in meta], dtype=object),
        dance_keys=np.array([m.get("dance_key", "unknown") for m in meta], dtype=object),
        dance_categories=np.array([m.get("dance_category", "unknown") for m in meta], dtype=object),
        dance_themes=np.array([m.get("dance_theme", "unknown") for m in meta], dtype=object),
        candidate_dance_categories=np.array([m.get("candidate_dance_category") or "" for m in meta], dtype=object),
        theme_label_statuses=np.array([m.get("theme_label_status", "unknown") for m in meta], dtype=object),
        source_context_json=np.array([json.dumps(m.get("source_context", []), sort_keys=True) for m in meta], dtype=object),
        semantic_roles=np.array([m.get("semantic_role", "unknown") for m in meta], dtype=object),
        semantic_texts=np.array([m.get("semantic_text", "") for m in meta], dtype=object),
        energy_labels=np.array([m.get("energy_label", "unknown") for m in meta], dtype=object),
        rhythm_labels=np.array([m.get("rhythm_label", "unknown") for m in meta], dtype=object),
        body_focus_labels=np.array([m.get("body_focus_label", "unknown") for m in meta], dtype=object),
        spatial_labels=np.array([m.get("spatial_label", "unknown") for m in meta], dtype=object),
        music_alignment_labels=np.array([m.get("music_alignment_label", "unknown") for m in meta], dtype=object),
        local_action_labels_json=np.array([json.dumps(m.get("local_action_labels", ["unknown"])) for m in meta], dtype=object),
        local_action_scores_json=np.array([m.get("local_action_scores_json", "{}") for m in meta], dtype=object),
        music_compatibility_top_labels=np.array([m.get("music_compatibility_top_label", "unknown") for m in meta], dtype=object),
        music_compatibility_scores_json=np.array([m.get("music_compatibility_scores_json", "{}") for m in meta], dtype=object),
        music_compatibility_is_ground_truth=np.array([bool(m.get("music_compatibility_is_ground_truth", False)) for m in meta], dtype=np.bool_),
        classification_texts=np.array([m.get("classification_text", "") for m in meta], dtype=object),
        event_families=np.array([m.get("event_family", "unknown") for m in meta], dtype=object),
        motion_stage_roles=np.array([m.get("motion_stage_role", "unknown") for m in meta], dtype=object),
        cultural_motifs=np.array([m.get("cultural_motif", "unknown") for m in meta], dtype=object),
        prop_proxy_labels=np.array([m.get("prop_proxy_label", "unknown") for m in meta], dtype=object),
        locomotion_labels=np.array([m.get("locomotion_label", "unknown") for m in meta], dtype=object),
        support_labels=np.array([m.get("support_label", "unknown") for m in meta], dtype=object),
        event_position_mid=np.array([float(m.get("event_position_mid", 0.5)) for m in meta], dtype=np.float32),
        semantic_confidence=np.array([float(m.get("semantic_confidence", 0.5)) for m in meta], dtype=np.float32),
        event_quality_scores=np.array([float(m.get("event_quality_score", 0.5)) for m in meta], dtype=np.float32),
        natural_duration_min=np.array([float((m.get("natural_duration_range_sec", [1.5, 4.0]) or [1.5, 4.0])[0]) for m in meta], dtype=np.float32),
        natural_duration_max=np.array([float((m.get("natural_duration_range_sec", [1.5, 4.0]) or [1.5, 4.0])[-1]) for m in meta], dtype=np.float32),
        take_ids=np.array([int(m.get("take_id", -1) if m.get("take_id", -1) is not None else -1) for m in meta], dtype=np.int32),
        name_semantic=name_semantic.astype(np.float32),
        class_semantic=class_semantic.astype(np.float32),
        durations=np.array([m["duration"] for m in meta], dtype=np.float32),
        frames=np.array([m["frames"] for m in meta], dtype=np.int32),
        music=np.stack(music_feats).astype(np.float32),
        music_mask=np.array(music_masks, dtype=np.float32),
    )
    audio_coverage = float(np.mean(music_masks)) if music_masks else 0.0
    split_audit = {
        "num_sources_total": int(len(set(str(m.get("source_group")) for m in meta))),
        "num_labels_total": int(len(set(str(m.get("label")) for m in meta))),
        "events_per_source_min": int(min([sum(str(x.get("source_group")) == s for x in meta) for s in set(str(m.get("source_group")) for m in meta)])),
        "events_per_source_max": int(max([sum(str(x.get("source_group")) == s for x in meta) for s in set(str(m.get("source_group")) for m in meta)])),
        "num_source_uids_total": int(len(set(str(m.get("source_uid", m.get("source_group"))) for m in meta))),
        "num_recording_uids_total": int(len(set(str(m.get("recording_uid", m.get("source_uid", m.get("source_group")))) for m in meta))),
        "category_counts": {str(k): int(sum(str(m.get("dance_key")) == str(k) for m in meta)) for k in sorted(set(str(m.get("dance_key")) for m in meta))},
        "gender_counts": {str(k): int(sum(str(m.get("gender")) == str(k) for m in meta)) for k in sorted(set(str(m.get("gender")) for m in meta))},
        "energy_label_counts": {str(k): int(sum(str(m.get("energy_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("energy_label")) for m in meta))},
        "rhythm_label_counts": {str(k): int(sum(str(m.get("rhythm_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("rhythm_label")) for m in meta))},
        "body_focus_counts": {str(k): int(sum(str(m.get("body_focus_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("body_focus_label")) for m in meta))},
        "music_alignment_label_counts": {str(k): int(sum(str(m.get("music_alignment_label")) == str(k) for m in meta)) for k in sorted(set(str(m.get("music_alignment_label")) for m in meta))},
        "source_group_semantics": "source_uid is file-level; recording_uid groups synchronized performer tracks; category/gender/sequence are separate semantic metadata",
        "train_val_group_overlap": 0,  # no random sample split is produced here; downstream split must remain source-disjoint.
    }
    report = {
        "version": "semantic_source_aware_event_db",
        "config": dataclasses.asdict(cfg),
        "events": meta,
        "num_events": len(meta),
        "audio_feature_coverage": audio_coverage,
        "audio_files_seen": len(audio_files),
        "input_report": input_report,
        "resample_summary": {
            "resampled_count": int(sum(1 for r in resample_reports if r.get("resampled"))),
            "native_fps_values": sorted(set(round(float(r.get("native_fps", cfg.fps)), 6) for r in resample_reports if "native_fps" in r))[:12],
            "target_fps": float(cfg.fps),
        },
        "split_audit": split_audit,
        "fk_tree_source": FK_TREE_SOURCE,
    }
    save_json(report, out_dir / "events_meta.json")
    print(json.dumps({"db": str(db_path), "num_events": len(meta), "out_dir": str(out_dir), "audio_feature_coverage": audio_coverage, "audio_files_seen": len(audio_files), "input_mode": input_report.get("input_mode"), "resampled_count": report["resample_summary"]["resampled_count"]}, ensure_ascii=False, indent=2))
    return 0


def read_wav_mono(path: str | Path) -> Tuple[int, np.ndarray]:
    if wavfile is None:
        raise RuntimeError("scipy.io.wavfile is unavailable; install scipy or provide --slots_json")
    sr, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype.kind in "iu":
        maxv = max(float(np.iinfo(data.dtype).max), 1.0)
        data = data.astype(np.float32) / maxv
    else:
        data = data.astype(np.float32)
    if data.size == 0:
        data = np.zeros(1, dtype=np.float32)
    return int(sr), data


def audio_global_features(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size < 16:
        wav = np.pad(wav, (0, 16 - wav.size))
    frame = max(256, int(0.046 * sr))
    hop = max(64, int(0.023 * sr))
    vals = []
    cents = []
    zcrs = []
    for st in range(0, max(1, wav.size - frame + 1), hop):
        x = wav[st : st + frame]
        if x.size < frame:
            x = np.pad(x, (0, frame - x.size))
        vals.append(float(np.sqrt(np.mean(x * x) + 1e-8)))
        zcrs.append(float(np.mean(x[1:] * x[:-1] < 0)))
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
        cents.append(float((spec * freqs).sum() / max(spec.sum(), 1e-8) / max(sr, 1)))
    env = np.asarray(vals, dtype=np.float32)
    zcr = np.asarray(zcrs, dtype=np.float32)
    cen = np.asarray(cents, dtype=np.float32)
    onset = np.maximum(0.0, np.diff(env, prepend=env[:1]))
    if env.size > 4:
        ac = np.correlate(env - env.mean(), env - env.mean(), mode="full")[env.size - 1 :]
        lag = int(np.argmax(ac[1 : min(len(ac), 128)]) + 1) if len(ac) > 2 else 1
    else:
        lag = 1
    features = np.array(
        [
            wav.size / sr,
            env.mean(), env.std(), np.percentile(env, 90), np.percentile(env, 10),
            onset.mean(), onset.std(), np.percentile(onset, 90),
            zcr.mean(), zcr.std(), cen.mean(), cen.std(),
            lag / max(1.0, env.size),
            float(np.max(env)), float(np.min(env)), float(np.median(env)),
        ],
        dtype=np.float32,
    )
    # Add histogram-like dynamics.
    qs = np.percentile(env, [5, 25, 50, 75, 95]).astype(np.float32)
    oqs = np.percentile(onset, [5, 25, 50, 75, 95]).astype(np.float32)
    out = np.concatenate([features, qs, oqs], axis=0)
    if out.size < 32:
        out = np.pad(out, (0, 32 - out.size))
    return out[:32].astype(np.float32)


def collect_audio_files(audio_dirs: Optional[Sequence[str]]) -> List[str]:
    """Collect wav/mp3-like files for optional real-audio Contrastive Retriever pairing."""
    if not audio_dirs:
        return []
    exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
    files: List[str] = []
    for d in audio_dirs:
        if not d:
            continue
        p = Path(d)
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
        elif p.exists():
            for ext in exts:
                files.extend(glob.glob(str(p / "**" / f"*{ext}"), recursive=True))
    return sorted(set(files))


def _match_key(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    keep = []
    for ch in stem:
        keep.append(ch if ch.isalnum() else "_")
    tokens = [t for t in "".join(keep).split("_") if t and t not in {"motion", "audio", "music", "wav", "npy", "clip"}]
    return "_".join(tokens)


def find_matching_audio_for_motion(motion_path: str | Path, audio_files: Sequence[str]) -> Optional[str]:
    """Best-effort same-stem / token-overlap pairing for Chang-E/change exports."""
    if not audio_files:
        return None
    key = _match_key(motion_path)
    if not key:
        return None
    audio_keys = [(_match_key(a), a) for a in audio_files]
    for ak, a in audio_keys:
        if ak == key:
            return a
    for ak, a in audio_keys:
        if ak and (ak in key or key in ak):
            return a
    kt = set(key.split("_"))
    best = (0.0, None)
    for ak, a in audio_keys:
        at = set(ak.split("_"))
        if not at:
            continue
        score = len(kt & at) / max(len(kt | at), 1)
        if score > best[0]:
            best = (score, a)
    return best[1] if best[0] >= 0.35 else None


def audio_feature_for_motion_clip(audio_path: str | Path, start_frame: int, frames: int, cfg: MotionGenerationConfig) -> np.ndarray:
    """Extract a 32D real audio descriptor aligned to a motion clip interval."""
    sr, wav = read_wav_mono(audio_path)
    st = max(0, int(round((start_frame / max(float(cfg.fps), 1e-6)) * sr)))
    ed = min(wav.size, int(round(((start_frame + frames) / max(float(cfg.fps), 1e-6)) * sr)))
    if ed - st < max(64, int(0.10 * sr)):
        st, ed = 0, wav.size
    return audio_global_features(wav[st:ed], sr).astype(np.float32)


def standardize_features(x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if mask is not None and np.asarray(mask).astype(bool).any():
        base = x[np.asarray(mask).astype(bool)]
    else:
        base = x
    mean = base.mean(axis=0, keepdims=True).astype(np.float32)
    std = (base.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), mean, std


def build_default_audio_slots(path: str | Path, cfg: MotionGenerationConfig, slot_seconds: float = 4.0, slots_json: Optional[str] = None) -> Tuple[List[dict], np.ndarray]:
    if slots_json and Path(slots_json).exists():
        # External Music Semantics: slots_json can be either the old feature JSON or a new
        # external classical-music semantic JSON/NPZ.
        if str(slots_json).lower().endswith(".npz"):
            parsed = parse_external_music_semantic_file(slots_json, cfg)
            if parsed is not None:
                return parsed
        data = load_json(slots_json)
        slots = data.get("slots", data if isinstance(data, list) else [])
        if slots and isinstance(slots[0], dict) and ("probs" in slots[0] or "top_label" in slots[0] or "music_alignment_label" in slots[0]):
            parsed = parse_external_music_semantic_file(slots_json, cfg)
            if parsed is not None:
                return parsed
        feats = []
        for s in slots:
            base = np.asarray(s.get("feature", []), dtype=np.float32)
            if base.size < 32:
                base = np.pad(base, (0, 32 - base.size))
            feats.append(base[:32])
        return slots, np.stack(feats).astype(np.float32)
    external = load_external_music_semantic_slots(path, cfg, slot_seconds)
    if external is not None:
        return external
    sr, wav = read_wav_mono(path)
    total = wav.size / sr
    n_slots = max(1, int(math.ceil(total / slot_seconds)))
    slots: List[dict] = []
    feats: List[np.ndarray] = []
    for i in range(n_slots):
        st = int(i * slot_seconds * sr)
        ed = min(wav.size, int((i + 1) * slot_seconds * sr))
        seg = wav[st:ed]
        f = audio_global_features(seg, sr)
        # Convert audio descriptor into same coarse semantic order as motion descriptor.
        pseudo = np.zeros(32, dtype=np.float32)
        dur = max((ed - st) / sr, 1e-6)
        energy = float(f[1])
        onset = float(f[5])
        dyn = float(f[2] + f[6])
        pseudo[0] = dur
        pseudo[1] = energy * 2.0
        pseudo[2] = energy
        pseudo[3] = f[3]
        pseudo[4] = dyn
        pseudo[5] = energy + onset
        pseudo[6] = f[7]
        pseudo[7] = energy + 0.5 * onset
        pseudo[8] = energy
        pseudo[9] = 1.0 + onset
        pseudo[10] = np.clip(0.75 - energy, 0.0, 1.0)
        pseudo[13] = max(0.02, onset)
        pseudo[15] = 0.0
        pseudo[16] = onset
        pseudo[17] = f[7]
        pseudo[18] = dyn
        pseudo[19:] = f[: 32 - 19]
        slot_sem = audio_slot_classification_from_pseudo(pseudo, dur, energy, onset, dyn)
        # Encode music alignment labels in the same high channels as class_semantic.
        pseudo[22] = _label_index(slot_sem["energy_label"], ENERGY_LABELS) / max(1, len(ENERGY_LABELS) - 1)
        pseudo[23] = _label_index(slot_sem["rhythm_label"], RHYTHM_LABELS) / max(1, len(RHYTHM_LABELS) - 1)
        pseudo[26] = _label_index(slot_sem["music_alignment_label"], MUSIC_ALIGNMENT_LABELS) / max(1, len(MUSIC_ALIGNMENT_LABELS) - 1)
        slot_item = {"slot_id": i, "start": st / sr, "end": ed / sr, "duration": dur, "energy": energy, "onset": onset}
        slot_item.update(slot_sem)
        slots.append(slot_item)
        feats.append(pseudo.astype(np.float32))
    return slots, np.stack(feats).astype(np.float32)








# ===== Music Slot Planning PRETRAINED ROUTER SLOT PATCH START =====
def _slot_plan_env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


_SLOT_PLAN_SEMANTIC_LABELS = [
    "calm_meditative",
    "lyrical_flow",
    "pose_hold",
    "instrument_phrase",
    "percussive_accent",
    "turning_climax",
    "footwork_flow",
]


def _normalize_slot_probabilities(obj, top_label=None):
    if isinstance(obj, dict):
        raw = {str(k): float(v) for k, v in obj.items() if str(k) in _SLOT_PLAN_SEMANTIC_LABELS}
    else:
        raw = {}
    if not raw and top_label in _SLOT_PLAN_SEMANTIC_LABELS:
        raw = {k: 0.02 for k in _SLOT_PLAN_SEMANTIC_LABELS}
        raw[str(top_label)] = 0.88
    if not raw:
        raw = {k: 1.0 / len(_SLOT_PLAN_SEMANTIC_LABELS) for k in _SLOT_PLAN_SEMANTIC_LABELS}
    s = sum(max(0.0, float(v)) for v in raw.values())
    if s <= 1e-8:
        return {k: 1.0 / len(_SLOT_PLAN_SEMANTIC_LABELS) for k in _SLOT_PLAN_SEMANTIC_LABELS}
    return {k: float(max(0.0, raw.get(k, 0.0)) / s) for k in _SLOT_PLAN_SEMANTIC_LABELS}


def _top_slot_label(probs):
    p = _normalize_slot_probabilities(probs)
    return max(p.items(), key=lambda kv: kv[1])[0]


def _feature_from_music_slot(slot: dict) -> np.ndarray:
    dur = float(slot.get("duration", slot.get("duration_sec", 4.0)))
    probs = _normalize_slot_probabilities(slot.get("music_semantic_probs", {}), slot.get("music_semantic_top_label", slot.get("music_alignment_label")))
    energy = float(slot.get("energy", slot.get("music_energy", slot.get("slot_energy", 0.06))))
    onset = float(slot.get("onset", slot.get("accent", slot.get("music_accent_score", 0.02))))
    dyn = float(slot.get("dynamic", slot.get("beat_density", 0.04)))
    x = np.zeros(32, dtype=np.float32)
    x[0] = dur
    x[1] = 2.0 * energy + 0.25 * probs["footwork_flow"] + 0.25 * probs["percussive_accent"]
    x[2] = energy
    x[3] = float(slot.get("energy_p90", energy))
    x[4] = dyn
    x[5] = energy + onset + 0.35 * probs["percussive_accent"]
    x[6] = float(slot.get("onset_p90", onset))
    x[7] = energy + 0.5 * onset + 0.25 * probs["footwork_flow"]
    x[8] = energy
    x[9] = 1.0 + onset + 0.35 * probs["turning_climax"]
    x[10] = 0.65 * probs["calm_meditative"] + 0.45 * probs["pose_hold"]
    x[11] = probs["pose_hold"]
    x[12] = probs["calm_meditative"]
    x[13] = max(0.02, onset + 0.2 * probs["percussive_accent"])
    x[15] = 0.25 * probs["turning_climax"]
    x[16] = probs["turning_climax"]
    x[17] = probs["footwork_flow"]
    x[18] = probs["instrument_phrase"]
    x[19] = probs["lyrical_flow"]
    x[20] = probs["percussive_accent"]
    x[21] = probs["calm_meditative"]
    x[22] = probs["pose_hold"]
    for i, lab in enumerate(_SLOT_PLAN_SEMANTIC_LABELS):
        if 23 + i < 32:
            x[23 + i] = probs[lab]
    x[31] = 1.0
    return x.astype(np.float32)


def _find_music_slots_in_json(data):
    if isinstance(data, dict):
        if isinstance(data.get("slots"), list):
            return data.get("slots"), data
        sr = data.get("stage_reports")
        if isinstance(sr, dict) and isinstance(sr.get("retrieval"), list):
            slots = []
            for r in sr.get("retrieval", []):
                if not isinstance(r, dict):
                    continue
                slots.append({
                    "slot_id": r.get("slot", r.get("slot_id", len(slots))),
                    "duration": r.get("duration", 4.0),
                    "music_alignment_label": r.get("slot_music_alignment_label", r.get("music_alignment_label", "calm_meditative")),
                    "music_semantic_top_label": r.get("slot_music_semantic_top_label", r.get("slot_music_alignment_label", "calm_meditative")),
                    "music_semantic_probs": r.get("slot_music_semantic_probs", {}),
                    "preferred_dance_keys": r.get("slot_preferred_dance_keys", []),
                })
            if slots:
                return slots, data
        for v in data.values():
            got, meta = _find_music_slots_in_json(v)
            if got is not None:
                return got, meta
    elif isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data):
            keys = set()
            for x in data[: min(8, len(data))]:
                keys.update(x.keys())
            if "duration" in keys or {"start", "end"}.issubset(keys):
                return data, {"version": "list_slots"}
        for v in data:
            got, meta = _find_music_slots_in_json(v)
            if got is not None:
                return got, meta
    return None, {}


def _load_music_slots_json(slots_json: str | Path, cfg: MotionGenerationConfig) -> Tuple[List[dict], np.ndarray, dict]:
    data = load_json(slots_json)
    slots, meta = _find_music_slots_in_json(data)
    if not slots:
        raise RuntimeError(f"Music Slot Planning slots_json has no slots: {slots_json}")
    fps = float(getattr(cfg, "fps", 30.0))
    out_slots: List[dict] = []
    feats: List[np.ndarray] = []
    cursor = 0.0
    for i, s0 in enumerate(slots):
        s = dict(s0)
        dur = s.get("duration", s.get("duration_sec", None))
        st = s.get("start", s.get("start_sec", s.get("music_start", None)))
        ed = s.get("end", s.get("end_sec", s.get("music_end", None)))
        if dur is None and st is not None and ed is not None:
            dur = float(ed) - float(st)
        if dur is None:
            dur = 4.0
        dur = max(0.10, float(dur))
        if st is None:
            st = cursor
        st = float(st)
        if ed is None:
            ed = st + dur
        ed = float(ed)
        dur = max(0.10, ed - st)
        cursor = ed
        probs = _normalize_slot_probabilities(s.get("music_semantic_probs", {}), s.get("music_semantic_top_label", s.get("music_alignment_label")))
        top = s.get("music_semantic_top_label", s.get("music_alignment_label", _top_slot_label(probs)))
        if top not in _SLOT_PLAN_SEMANTIC_LABELS:
            top = _top_slot_label(probs)
        base = np.asarray(s.get("feature", []), dtype=np.float32)
        if base.size < 32 or float(np.max(np.abs(base))) == 0.0:
            base = _feature_from_music_slot({**s, "duration": dur, "music_semantic_probs": probs, "music_semantic_top_label": top})
        if base.size < 32:
            base = np.pad(base, (0, 32 - base.size))
        s.update({
            "slot_id": int(s.get("slot_id", s.get("slot", i))),
            "start": float(st),
            "end": float(ed),
            "duration": float(dur),
            "target_frames": int(s.get("target_frames", round(float(dur) * fps))),
            "music_alignment_label": str(s.get("music_alignment_label", top)),
            "music_semantic_top_label": str(top),
            "music_semantic_probs": probs,
            "slot_plan_source": str(s.get("slot_plan_source", meta.get("slot_source", "music_slot_slots_json"))),
            "feature": base[:32].astype(float).tolist(),
        })
        out_slots.append(s)
        feats.append(base[:32].astype(np.float32))
    return out_slots, np.stack(feats).astype(np.float32), meta if isinstance(meta, dict) else {}


# ===== Music Slot Planning PRETRAINED ROUTER SLOT PATCH END =====

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int = 32, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.05),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ContrastiveModel(nn.Module):
    def __init__(self, feat_dim: int = 32, emb_dim: int = 128):
        super().__init__()
        self.motion = MLPEncoder(feat_dim, emb_dim)
        self.music = MLPEncoder(feat_dim, emb_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def forward(self, music_feat, motion_feat):
        me = self.music(music_feat)
        de = self.motion(motion_feat)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * me @ de.t(), me, de


def make_weak_music_features_from_motion(desc: np.ndarray, noise: float = 0.08) -> np.ndarray:
    # Weak-pair fallback: retain duration/energy/turn/contact semantics with jitter.
    # Root-Aware Motion Safety keeps this only as an explicitly marked smoke-test fallback.  For
    # motion-only Chang-E BVH, prefer unpaired real-audio semantic OT below.
    m = desc.copy().astype(np.float32)
    rng = np.random.default_rng(1234)
    m[:, 1:9] *= rng.normal(1.0, noise, size=m[:, 1:9].shape).astype(np.float32)
    m[:, 15:18] *= rng.normal(1.0, noise * 1.5, size=m[:, 15:18].shape).astype(np.float32)
    m += rng.normal(0.0, noise * 0.15, size=m.shape).astype(np.float32)
    return m.astype(np.float32)


def semantic_dims_and_weights() -> Tuple[np.ndarray, np.ndarray]:
    """Feature weights shared by unpaired audio-motion OT and retrieval fallback."""
    dims = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 16, 17, 18, 22, 23, 24, 25, 26, 28, 29, 30], dtype=np.int64)
    weights = np.array([
        0.45,  # duration
        0.80, 0.95, 0.75, 0.70,  # root/energy/dynamics
        1.15, 1.00, 1.05, 0.90,  # joint/lower/upper energy
        0.45, 0.35,              # lower/upper ratio + contact calmness
        0.75, 0.75,              # foot speed
        0.90, 0.95, 0.65,        # onset/turn/root-y dynamics
        0.55, 0.55, 0.42, 0.42, 0.70,  # explicit class channels
        0.35, 0.42, 0.42,        # calm/percussive/turn affinity bits
    ], dtype=np.float32)
    return dims, weights






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


def train_contrastive(args: argparse.Namespace) -> int:
    if torch is None:
        raise RuntimeError("PyTorch is required for Contrastive Retriever training.")
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    sem_dirs = getattr(args, "music_semantic_dirs", None)
    if sem_dirs:
        cfg.external_music_semantic_dirs = os.pathsep.join([str(x) for x in sem_dirs])
    if getattr(args, "external_music_semantic_cmd", None):
        cfg.external_music_semantic_cmd = str(args.external_music_semantic_cmd)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    db = load_db(args.db)
    database_contract = _training_db_contract(db, cfg, "Contrastive Retriever training")
    motion_db = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "filename_semantic_weight", 0.35)))
    supervision_mode = "weak_motion_proxy"
    pair_report: Dict[str, object] = {}

    if args.music_feature_npz and Path(args.music_feature_npz).exists():
        mf_npz = np.load(args.music_feature_npz)
        key = "music_z" if "music_z" in mf_npz.files else "music"
        mf = mf_npz[key].astype(np.float32)
        if mf.shape != motion_db.shape:
            raise ValueError(f"music feature shape {mf.shape} != motion feature shape {motion_db.shape}")
        music, music_mean, music_std = standardize_features(mf)
        motion = motion_db
        supervision_mode = "external_real_music_features"
        pair_report = {"mode": supervision_mode, "num_training_pairs": int(motion.shape[0])}
    elif "music" in db and "music_mask" in db and float(np.mean(np.asarray(db["music_mask"], dtype=np.float32) > 0.5)) >= float(cfg.audio_pair_min_coverage):
        mask = np.asarray(db["music_mask"], dtype=np.float32) > 0.5
        music_raw = np.asarray(db["music"], dtype=np.float32)
        music, music_mean, music_std = standardize_features(music_raw, mask=mask)
        motion = motion_db[mask]
        music = music[mask]
        supervision_mode = "paired_audio_features_from_db"
        pair_report = {"mode": supervision_mode, "paired_coverage": float(mask.mean()), "num_training_pairs": int(motion.shape[0])}
    else:
        unpaired_dirs = getattr(args, "unpaired_audio_dirs", None) or getattr(args, "audio_dirs", None)
        unpaired = build_unpaired_audio_motion_pairs(db, unpaired_dirs, cfg) if bool(cfg.unpaired_audio_enable) else None
        if unpaired is not None:
            music, motion, music_mean, music_std, pair_report = unpaired
            supervision_mode = str(pair_report.get("mode", "unpaired_audio_semantic_ot"))
        else:
            if bool(cfg.contrastive_require_real_music) or bool(cfg.unpaired_disable_motion_proxy):
                raise RuntimeError(
                    "No paired audio and no usable unpaired audio were found. "
                    "Put target music under test_music_bank/ or data/music/, or set "
                    "MOTION_UNPAIRED_DISABLE_MOTION_PROXY=0 for a smoke-test-only fallback."
                )
            music_raw = make_weak_music_features_from_motion(np.asarray(db["desc"], dtype=np.float32))
            mean = np.asarray(db["desc_mean"], dtype=np.float32)
            std = np.asarray(db["desc_std"], dtype=np.float32)
            music = (music_raw - mean) / std
            motion = motion_db
            music_mean, music_std = mean, std
            supervision_mode = "weak_motion_descriptor_proxy"
            pair_report = {"mode": supervision_mode, "warning": "motion descriptor self-distillation; do not report as real music-motion pairing", "num_training_pairs": int(motion.shape[0])}

    N = int(motion.shape[0])
    if N < 2:
        raise RuntimeError("Contrastive Retriever contrastive needs at least two training pairs.")
    device = torch.device(cfg.device)
    model = ContrastiveModel(motion.shape[1], cfg.embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bs = min(cfg.batch_size, N)
    for ep in range(int(args.epochs or cfg.contrastive_epochs)):
        perm = np.random.permutation(N)
        losses = []
        for st in range(0, N, bs):
            idx = perm[st:st + bs]
            if len(idx) < 2:
                continue
            mf = torch.from_numpy(music[idx]).float().to(device)
            df = torch.from_numpy(motion[idx]).float().to(device)
            logits, _, _ = model(mf, df)
            labels = torch.arange(logits.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep % 10 == 0 or ep == int(args.epochs or cfg.contrastive_epochs) - 1:
            print(f"[Contrastive Retriever contrastive] epoch={ep} loss={np.mean(losses):.5f} scale={model.logit_scale.exp().item():.2f} mode={supervision_mode}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "version": "contrastive_retrieval_12_external_classical_music_semantic_grounding",
        "state_dict": model.state_dict(),
        "config": dataclasses.asdict(cfg),
        "feat_dim": motion.shape[1],
        "embed_dim": cfg.embed_dim,
        "supervision_mode": supervision_mode,
        "music_mean": music_mean.astype(np.float32),
        "music_std": music_std.astype(np.float32),
        "pair_report": pair_report,
        "motion_contract": motion_checkpoint_contract(cfg, "semantic_retriever"),
        "training_event_db_contract": database_contract["event_db_contract"],
        "training_database": database_contract,
    }
    torch.save(ckpt, out)
    print(json.dumps({"contrastive_ckpt": str(out), "num_pairs": int(N), "supervision_mode": supervision_mode, "pair_report": pair_report}, ensure_ascii=False, indent=2))
    return 0

def _expand_temporal_condition_torch(cond, frames: int):
    """Return condition as ``[batch, frames, features]``.

    Checkpoint parameters are unchanged: legacy callers may still pass one
    descriptor per sequence, while whole-song generation can pass a local
    descriptor for every frame.
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
    """Slice frame-local conditioning while retaining legacy vector inputs."""

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


class TemporalRefiner(nn.Module):
    """Legacy 151D Euclidean refiner kept for old Motion Refiner checkpoints."""

    def __init__(self, motion_dim: int = EDGE_DIM, cond_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.in_proj = nn.Conv1d(motion_dim + cond_dim + 1, hidden, 1)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
        )
        self.out = nn.Conv1d(hidden, motion_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, cond, seam_mask):
        # x: B,T,D cond: B,C or B,T,C seam_mask: B,T,1
        B, T, D = x.shape
        c = _expand_temporal_condition_torch(cond, T)
        y = torch.cat([x, c, seam_mask], dim=-1).transpose(1, 2)
        h = self.in_proj(y)
        h = h + self.net(h)
        delta = self.out(h).transpose(1, 2)
        return delta


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
        self.in_proj = nn.Conv1d(
            motion_dim + cond_dim + 1 + NUM_JOINTS, hidden, 1
        )
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the same frame x joint risk masks for training and inference."""
    joint_masks: List[np.ndarray] = []
    root_masks: List[np.ndarray] = []
    contact_masks: List[np.ndarray] = []
    for motion, seam in zip(
        np.asarray(motion_batch, dtype=np.float32),
        np.asarray(seam_batch, dtype=np.float32),
    ):
        masks = build_frame_joint_risk_mask(
            motion,
            seam,
            fps=float(cfg.fps),
        )
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
    geometry = masked_retract_torch(
        reference,
        output[..., 4:],
        joint_mask=joint_mask,
        root_mask=root_mask,
        max_rotation_rad=float(cfg.product_refiner_rotation_cap_rad),
        max_root_m=float(cfg.product_refiner_root_cap_m),
    )
    contact_weight = contact_mask.clamp(0.0, 1.0)
    contact_probability = torch.sigmoid(output[..., :4])
    contacts = (
        reference[..., :4] * (1.0 - contact_weight)
        + contact_probability * contact_weight
    )
    return torch.cat([contacts, geometry[..., 4:]], dim=-1)


def _world_space_physics_losses(prediction, clean, cfg: MotionGenerationConfig):
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
    predicted_joints = fk_24_torch(prediction)
    clean_joints = fk_24_torch(clean)
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
    terms = {
        "fk": fk_loss,
        "foot": foot_loss,
        "support": support_loss,
        "penetration": penetration_loss,
        "acceleration": acceleration_loss,
        "jerk": jerk_loss,
    }
    total = (
        float(cfg.physics_fk_loss_weight) * fk_loss
        + float(cfg.physics_foot_loss_weight) * foot_loss
        + float(cfg.physics_support_loss_weight) * support_loss
        + float(cfg.physics_penetration_loss_weight) * penetration_loss
        + float(cfg.physics_acceleration_loss_weight) * acceleration_loss
        + float(cfg.physics_jerk_loss_weight) * jerk_loss
    )
    return total, terms


def _product_motion_losses(
    prediction,
    clean,
    reference,
    joint_mask,
    root_mask,
    contact_mask,
    cfg: MotionGenerationConfig,
):
    contact_target = clean[..., :4].clamp(0.0, 1.0)
    contact_weight = 0.25 + 0.75 * contact_mask
    contact_loss = (
        F.binary_cross_entropy(
            prediction[..., :4].clamp(1.0e-5, 1.0 - 1.0e-5),
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
    )
    total = (
        reconstruction_loss
        + 0.25 * velocity_loss
        + 0.20 * contact_loss
        + float(cfg.product_refiner_outside_weight) * outside_loss
        + physics_loss
    )
    terms = {
        "reconstruction": reconstruction_loss,
        "velocity": velocity_loss,
        "contact": contact_loss,
        "outside": outside_loss,
        "physics": physics_loss,
    }
    terms.update({f"physics_{key}": value for key, value in physics_terms.items()})
    return total, terms


def sample_motion_window(paths: np.ndarray, target_len: int, cfg: Optional[MotionGenerationConfig] = None) -> np.ndarray:
    """Sample a training window and keep the EDGE-151D contract after resampling."""
    p = str(random.choice(paths.tolist()))
    return load_motion_window(p, target_len, cfg)


def load_motion_window(
    path: str | Path,
    target_len: int,
    cfg: Optional[MotionGenerationConfig] = None,
    *,
    random_crop: bool = True,
) -> np.ndarray:
    """Load one event and return a contract-valid fixed-length training window."""
    p = str(path)
    m = np.load(p).astype(np.float32)
    if m.shape[0] == target_len:
        out = m
    elif m.shape[0] > target_len:
        st = random.randint(0, m.shape[0] - target_len) if random_crop else (m.shape[0] - target_len) // 2
        out = m[st:st + target_len]
    else:
        out = resample_motion_np(m, target_len)
    out, _ = enforce_edge151_contract_np(out, cfg, source_hint=f"sample_motion_window:{p}", derive_contact=True, project_rot=True)
    return out.astype(np.float32)





def degrade_for_refiner(clean: np.ndarray, severity: float = 0.06, cfg: Optional[MotionGenerationConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Reference Inbetweening transition-masked corruption for Motion Refiner and Motion Diffusion training.

    Instead of arbitrary global drift only, corrupt a local transition region by
    replacing it with a weak root-Hermite / rotation-SLERP inbetweening path plus
    noise. This matches the inference-time transition-budget mask: the model
    learns to repair motion_ref only near boundaries while preserving core clips.
    """
    cfg = cfg or MotionGenerationConfig()
    x = np.asarray(clean, dtype=np.float32).copy()
    T, D = x.shape
    seam = np.zeros((T, 1), dtype=np.float32)
    if T <= 12:
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
            bridge = reference_motion_inbetween_np(prev_tail, curr_head, b - a, cfg)
            # Add light residual corruption mainly in root/rot channels; contacts rebuilt later.
            noise = np.zeros_like(bridge, dtype=np.float32)
            noise[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.18, size=(bridge.shape[0], 3)).astype(np.float32)
            noise[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.08, size=(bridge.shape[0], ROT6D_END - ROT6D_START)).astype(np.float32)
            x[a:b] = bridge + noise
            seam[max(0, a - halo):min(T, b + halo), 0] = 0.35
            seam[a:b, 0] = 1.0
        # Soft post-boundary drift to simulate mismatched retrieval alignment.
        offset = np.zeros(D, dtype=np.float32)
        offset[[ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.45, size=3)
        offset[ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.16, size=ROT6D_END - ROT6D_START)
        tail = T - b
        if tail > 0:
            decay = np.linspace(1.0, 0.0, tail, dtype=np.float32)[:, None]
            x[b:] += decay * offset[None]

    # Tiny background noise keeps denoising stable without encouraging core rewrite.
    noise = np.zeros_like(x, dtype=np.float32)
    noise[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = np.random.normal(0, severity * 0.025, size=(T, 3)).astype(np.float32)
    noise[:, ROT6D_START:ROT6D_END] = np.random.normal(0, severity * 0.012, size=(T, ROT6D_END - ROT6D_START)).astype(np.float32)
    x += noise
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
    return {"audits": [], "gates": [], "fk_errors": []}


def _record_validation_physical_prediction(
    accumulator: Dict[str, Any],
    prediction: np.ndarray,
    clean: np.ndarray,
    cfg: MotionGenerationConfig,
) -> None:
    predicted = np.asarray(prediction, dtype=np.float32)
    target, _ = enforce_edge151_contract_np(
        np.asarray(clean, dtype=np.float32),
        cfg,
        source_hint="checkpoint_validation_target",
        derive_contact=True,
        project_rot=True,
    )
    try:
        audit = audit_motion_np(predicted, cfg)
    except Exception as exc:
        # Validation is fail-closed: never project or sanitize a broken model
        # prediction into an apparently healthy physical sample.
        audit = {
            "schema": "invalid_validation_prediction",
            "validation_audit_error": f"{type(exc).__name__}: {exc}",
        }
    gate = evaluate_physical_audit(audit)
    try:
        fk_error = np.linalg.norm(
            fk_24_np(predicted) - fk_24_np(target), axis=-1
        )
        if not np.isfinite(fk_error).all():
            fk_error = np.full((1,), np.inf, dtype=np.float32)
    except Exception:
        fk_error = np.full((1,), np.inf, dtype=np.float32)
    accumulator["audits"].append(audit)
    accumulator["gates"].append(gate)
    accumulator["fk_errors"].append(fk_error.reshape(-1))


def _summarize_validation_physical_metrics(
    accumulator: Mapping[str, Any],
) -> Dict[str, Any]:
    audits = list(accumulator.get("audits", []))
    gates = list(accumulator.get("gates", []))
    error_parts = list(accumulator.get("fk_errors", []))
    errors = np.concatenate(error_parts) if error_parts else np.zeros(0, dtype=np.float32)
    failure_counts: Dict[str, int] = {}
    for gate in gates:
        for reason in gate.get("reasons", []):
            failure_counts[str(reason)] = failure_counts.get(str(reason), 0) + 1

    worst_window: Dict[str, Optional[float]] = {}
    mean_across_windows: Dict[str, Optional[float]] = {}
    for key in _VALIDATION_PHYSICAL_KEYS:
        values = [float(audit[key]) for audit in audits if key in audit]
        if not values:
            worst_window[key] = None
            mean_across_windows[key] = None
            continue
        worst_window[key] = (
            float(min(values)) if key == "foot_penetration_min_m" else float(max(values))
        )
        mean_across_windows[key] = float(np.mean(values))

    passed = sum(bool(gate.get("ok", False)) for gate in gates)
    return {
        "num_windows": len(audits),
        "fk_position_error_m_mean": float(np.mean(errors)) if errors.size else None,
        "fk_position_error_m_p95": float(np.percentile(errors, 95)) if errors.size else None,
        "fk_position_error_m_max": float(np.max(errors)) if errors.size else None,
        "physical_gate_pass_rate": float(passed / len(gates)) if gates else None,
        "physical_gate_failed_windows": int(len(gates) - passed),
        "physical_gate_failure_reasons": failure_counts,
        "worst_window": worst_window,
        "mean_across_windows": mean_across_windows,
        "gate": "contracts.physical_quality.evaluate_physical_audit",
        "aggregation_note": "worst_window is conservative across deterministic validation windows",
    }


def _evaluate_refiner_validation(
    model: Any,
    validation_db: Dict[str, Any],
    train_db: Dict[str, Any],
    cfg: MotionGenerationConfig,
    device: Any,
) -> Dict[str, Any]:
    indices = _validation_indices(len(validation_db["paths"]))
    cond_z = _descriptor_values_in_training_coordinates(validation_db, train_db)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(int(cfg.seed) + 45001)
    np.random.seed(int(cfg.seed) + 45001)
    rec_values: List[float] = []
    velocity_values: List[float] = []
    physical = _new_validation_physical_accumulator()
    model.eval()
    try:
        with torch.no_grad():
            for idx in indices:
                clean = load_motion_window(
                    validation_db["paths"][idx], cfg.window_len, cfg, random_crop=False
                )
                bad, seam = degrade_for_refiner(clean, cfg=cfg)
                clean_t = torch.from_numpy(clean[None]).float().to(device)
                bad_t = torch.from_numpy(bad[None]).float().to(device)
                seam_t = torch.from_numpy(seam[None]).float().to(device)
                cond_t = torch.from_numpy(cond_z[idx][None]).float().to(device)
                if isinstance(model, ProductManifoldTemporalRefiner):
                    joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                        bad[None], seam[None], cfg
                    )
                    joint_t = torch.from_numpy(joint_np).float().to(device)
                    root_t = torch.from_numpy(root_np).float().to(device)
                    contact_t = torch.from_numpy(contact_np).float().to(device)
                    output = model(bad_t, cond_t, seam_t, joint_t)
                    pred = _decode_product_refiner_output(
                        bad_t,
                        output,
                        joint_t,
                        root_t,
                        contact_t,
                        cfg,
                    )
                    rec_values.append(
                        float(
                            product_log_torch(pred, clean_t).abs().mean().cpu()
                        )
                    )
                    velocity_values.append(
                        float(
                            F.smooth_l1_loss(
                                product_log_torch(pred[:, :-1], pred[:, 1:]),
                                product_log_torch(
                                    clean_t[:, :-1], clean_t[:, 1:]
                                ),
                            ).cpu()
                        )
                    )
                else:
                    pred = bad_t + model(bad_t, cond_t, seam_t) * (
                        0.35 + 0.65 * seam_t
                    )
                    rec_values.append(
                        float(F.smooth_l1_loss(pred, clean_t).cpu())
                    )
                    velocity_values.append(float(F.smooth_l1_loss(
                        pred[:, 1:] - pred[:, :-1],
                        clean_t[:, 1:] - clean_t[:, :-1],
                    ).cpu()))
                _record_validation_physical_prediction(
                    physical,
                    pred[0].detach().cpu().numpy(),
                    clean,
                    cfg,
                )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.train()
    return {
        "num_windows": len(indices),
        "reconstruction_smooth_l1": float(np.mean(rec_values)) if rec_values else None,
        "velocity_smooth_l1_per_frame": float(np.mean(velocity_values)) if velocity_values else None,
        "representation": (
            "product_manifold_79d"
            if isinstance(model, ProductManifoldTemporalRefiner)
            else "legacy_edge151_euclidean"
        ),
        "descriptor_coordinates": "training_event_db",
        "physical_quality": _summarize_validation_physical_metrics(physical),
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
    model.eval()
    try:
        with torch.no_grad():
            for sample_index, idx in enumerate(indices):
                clean = load_motion_window(
                    validation_db["paths"][idx], cfg.window_len, cfg, random_crop=False
                )
                retrieval, seam = degrade_for_refiner(clean, severity=0.045, cfg=cfg)
                x0 = torch.from_numpy(clean[None]).float().to(device)
                retr = torch.from_numpy(retrieval[None]).float().to(device)
                seam_t = torch.from_numpy(seam[None]).float().to(device)
                cond_t = torch.from_numpy(cond_z[idx][None]).float().to(device)
                timestep = int(round(sample_index * max(diffusion_steps - 1, 0) / max(len(indices) - 1, 1)))
                t = torch.full((1,), timestep, dtype=torch.long, device=device)
                a = abar[t].view(1, 1, 1)
                if isinstance(model, TangentDiffusionDenoiser):
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
                                product_log_torch(
                                    decoded[:, :-1], decoded[:, 1:]
                                ),
                                product_log_torch(
                                    x0[:, :-1], x0[:, 1:]
                                ),
                            ).cpu()
                        )
                    )
                else:
                    noise = torch.randn_like(x0)
                    x_t = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise
                    pred_noise = model(x_t, retr, cond_t, seam_t, t)
                    noise_values.append(float(F.mse_loss(pred_noise, noise).cpu()))
                    x0_hat = (x_t - torch.sqrt(1.0 - a) * pred_noise) / torch.sqrt(a).clamp_min(1.0e-6)
                    velocity_values.append(float(F.smooth_l1_loss(
                        x0_hat[:, 1:] - x0_hat[:, :-1],
                        x0[:, 1:] - x0[:, :-1],
                    ).cpu()))
                    decoded = x0_hat
                _record_validation_physical_prediction(
                    physical,
                    decoded[0].detach().cpu().numpy(),
                    clean,
                    cfg,
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
        "velocity_smooth_l1_per_frame": float(np.mean(velocity_values)) if velocity_values else None,
        "representation": (
            "reference_tangent_product_manifold_79d"
            if isinstance(model, TangentDiffusionDenoiser)
            else "legacy_edge151_euclidean"
        ),
        "descriptor_coordinates": "training_event_db",
        "physical_quality": _summarize_validation_physical_metrics(physical),
    }


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
    product_mode = bool(cfg.product_manifold_enable)
    model = (
        ProductManifoldTemporalRefiner(EDGE_DIM, 32)
        if product_mode
        else TemporalRefiner(EDGE_DIM, 32)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = int(args.steps or cfg.refiner_train_steps)
    bs = min(cfg.batch_size, max(2, len(paths)))
    for step in range(steps):
        clean_batch = []
        bad_batch = []
        seam_batch = []
        joint_mask_batch = []
        root_mask_batch = []
        contact_mask_batch = []
        cond_batch = []
        for _ in range(bs):
            idx = random.randrange(len(paths))
            clean = load_motion_window(paths[idx], cfg.window_len, cfg)
            bad, seam = degrade_for_refiner(clean, cfg=cfg)
            clean_batch.append(clean)
            bad_batch.append(bad)
            seam_batch.append(seam)
            joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                bad[None], seam[None], cfg
            )
            joint_mask_batch.append(joint_np[0])
            root_mask_batch.append(root_np[0])
            contact_mask_batch.append(contact_np[0])
            cond_batch.append(desc_z[idx])
        clean_t = torch.from_numpy(np.stack(clean_batch)).float().to(device)
        bad_t = torch.from_numpy(np.stack(bad_batch)).float().to(device)
        seam_t = torch.from_numpy(np.stack(seam_batch)).float().to(device)
        cond_t = torch.from_numpy(np.stack(cond_batch)).float().to(device)
        if product_mode:
            joint_t = torch.from_numpy(np.stack(joint_mask_batch)).float().to(device)
            root_t = torch.from_numpy(np.stack(root_mask_batch)).float().to(device)
            contact_t = torch.from_numpy(
                np.stack(contact_mask_batch)
            ).float().to(device)
            output = model(bad_t, cond_t, seam_t, joint_t)
            pred = _decode_product_refiner_output(
                bad_t, output, joint_t, root_t, contact_t, cfg
            )
            loss, loss_terms = _product_motion_losses(
                pred,
                clean_t,
                bad_t,
                joint_t,
                root_t,
                contact_t,
                cfg,
            )
            rec = loss_terms["reconstruction"]
        else:
            delta = model(bad_t, cond_t, seam_t)
            pred = bad_t + delta * (0.35 + 0.65 * seam_t)
            rec = F.smooth_l1_loss(pred, clean_t)
            smooth = F.smooth_l1_loss(
                pred[:, 1:] - pred[:, :-1],
                clean_t[:, 1:] - clean_t[:, :-1],
            )
            loss = rec + 0.25 * smooth
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"[Boundary Refiner] step={step} loss={loss.item():.6f} rec={rec.item():.6f}")
    validation_report["metrics"] = _evaluate_refiner_validation(
        model, validation_db, db, cfg, device
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": (
            "product_manifold_boundary_refiner_v1"
            if product_mode
            else "boundary_refiner"
        ),
        "output_mode": (
            "contact_logits_root_joint_tangent"
            if product_mode
            else "legacy_edge151_residual"
        ),
        "output_dim": PRODUCT_STATE_DIM if product_mode else EDGE_DIM,
        "joint_mask_conditioned": bool(product_mode),
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
        "validation": validation_report,
    }, out)
    print(json.dumps({"refiner_ckpt": str(out), "steps": steps, "validation": validation_report}, ensure_ascii=False, indent=2))
    return 0


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


class DiffusionDenoiser(nn.Module):
    """Legacy raw EDGE-151 denoiser kept for old Motion Generation checkpoints."""

    def __init__(self, motion_dim: int = EDGE_DIM, cond_dim: int = 32, hidden: int = 256, time_dim: int = 128):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim + time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Conv1d(motion_dim * 2 + 1, hidden, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU()),
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GroupNorm(8, hidden), nn.SiLU()),
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.GroupNorm(8, hidden), nn.SiLU()),
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU()),
        ])
        self.out = nn.Conv1d(hidden, motion_dim, 1)

    def forward(self, x_t, retrieval, cond, seam_mask, t):
        B, T, D = x_t.shape
        inp = torch.cat([x_t, retrieval, seam_mask], dim=-1).transpose(1, 2)
        h = self.in_proj(inp)
        te = self.time(t)
        ce = _condition_with_time_torch(cond, te, T, self.cond_proj)
        h = h + ce
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h).transpose(1, 2)


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
    tangent_mode = bool(
        cfg.product_manifold_enable and cfg.tangent_diffusion_enable
    )
    model = (
        TangentDiffusionDenoiser(PRODUCT_STATE_DIM, 32)
        if tangent_mode
        else DiffusionDenoiser(EDGE_DIM, 32)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = int(args.steps or cfg.diffusion_train_steps)
    Tdiff = int(args.diffusion_steps or cfg.diffusion_steps)
    _, _, abar = make_beta_schedule(Tdiff, device)
    bs = min(cfg.batch_size, max(2, len(paths)))
    for step in range(steps):
        clean_batch = []
        retr_batch = []
        seam_batch = []
        joint_mask_batch = []
        root_mask_batch = []
        contact_mask_batch = []
        cond_batch = []
        for _ in range(bs):
            idx = random.randrange(len(paths))
            clean = np.load(str(paths[idx])).astype(np.float32)
            clean = resample_motion_np(clean, cfg.window_len)
            clean, _ = enforce_edge151_contract_np(
                clean, cfg, source_hint=f"train_diffusion_clean:{paths[idx]}", derive_contact=True, project_rot=True
            )
            retr, seam = degrade_for_refiner(clean, severity=0.045, cfg=cfg)
            clean_batch.append(clean)
            retr_batch.append(retr)
            seam_batch.append(seam)
            joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                retr[None], seam[None], cfg
            )
            joint_mask_batch.append(joint_np[0])
            root_mask_batch.append(root_np[0])
            contact_mask_batch.append(contact_np[0])
            cond_batch.append(desc_z[idx])
        x0 = torch.from_numpy(np.stack(clean_batch)).float().to(device)
        retr = torch.from_numpy(np.stack(retr_batch)).float().to(device)
        seam = torch.from_numpy(np.stack(seam_batch)).float().to(device)
        cond = torch.from_numpy(np.stack(cond_batch)).float().to(device)
        t = torch.randint(0, Tdiff, (bs,), device=device)
        a = abar[t].view(bs, 1, 1)
        if tangent_mode:
            joint_mask = torch.from_numpy(
                np.stack(joint_mask_batch)
            ).float().to(device)
            root_mask = torch.from_numpy(
                np.stack(root_mask_batch)
            ).float().to(device)
            contact_mask = torch.from_numpy(
                np.stack(contact_mask_batch)
            ).float().to(device)
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
        else:
            noise = torch.randn_like(x0)
            x_t = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise
            pred_noise = model(x_t, retr, cond, seam, t)
            loss_noise = F.mse_loss(pred_noise, noise)
            # Legacy raw-channel denoising retained for checkpoint parity.
            x0_hat = (
                x_t - torch.sqrt(1.0 - a) * pred_noise
            ) / torch.sqrt(a).clamp_min(1e-6)
            loss_vel = F.smooth_l1_loss(
                x0_hat[:, 1:] - x0_hat[:, :-1],
                x0[:, 1:] - x0[:, :-1],
            )
            decoded = x0_hat
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
        if step % 250 == 0 or step == steps - 1:
            print(
                f"[Motion Generation diffusion] step={step} loss={loss.item():.6f} "
                f"noise={loss_noise.item():.6f} "
                f"physics={loss_physics.item():.6f} "
                f"jerk={physics_terms['jerk'].item():.6f}"
            )
    validation_report["metrics"] = _evaluate_diffusion_validation(
        model, validation_db, db, cfg, device, abar, Tdiff
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": (
            "reference_tangent_motion_diffusion_v1"
            if tangent_mode
            else "conditional_residual_motion_diffusion"
        ),
        "diffusion_space": (
            "reference_point_product_tangent"
            if tangent_mode
            else "legacy_edge151_channels"
        ),
        "state_dim": PRODUCT_STATE_DIM if tangent_mode else EDGE_DIM,
        "joint_mask_conditioned": bool(tangent_mode),
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
        "validation": validation_report,
    }, out)
    print(json.dumps({"diffusion_ckpt": str(out), "steps": steps, "validation": validation_report}, ensure_ascii=False, indent=2))
    return 0





# ===== TRUSTED LOCAL CKPT LOAD FIX START =====
def _trusted_torch_load(path, map_location=None, **_unused_kwargs):
    """Load trusted local checkpoints saved by this project.

    PyTorch 2.6 defaults torch.load(..., weights_only=True), which may reject
    Contrastive Retriever checkpoints containing numpy arrays. These ckpts are generated locally
    in this experiment, so we explicitly use weights_only=False.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to load checkpoints.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch has no weights_only argument.
        return torch.load(path, map_location=map_location)
# ===== TRUSTED LOCAL CKPT LOAD FIX END =====

def load_contrastive(path: Optional[str], cfg: MotionGenerationConfig):
    if not bool(cfg.contrastive_enable):
        return None
    if torch is None or not path or not Path(path).exists():
        return None
    ckpt = _trusted_torch_load(path, map_location=cfg.device)
    assert_motion_checkpoint_contract(ckpt, cfg, path, "semantic_retriever")
    model = ContrastiveModel(ckpt.get("feat_dim", 32), ckpt.get("embed_dim", cfg.embed_dim)).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.music_mean = np.asarray(ckpt.get("music_mean", np.zeros((1, ckpt.get("feat_dim", 32)), dtype=np.float32)), dtype=np.float32)
    model.music_std = np.asarray(ckpt.get("music_std", np.ones((1, ckpt.get("feat_dim", 32)), dtype=np.float32)), dtype=np.float32)
    model.supervision_mode = ckpt.get("supervision_mode", "unknown")
    model.eval()
    return model


def embed_with_contrastive(model, music_feat_z: np.ndarray, motion_feat_z: np.ndarray, cfg: MotionGenerationConfig) -> Tuple[np.ndarray, np.ndarray]:
    if model is None or torch is None:
        mz = music_feat_z / np.maximum(np.linalg.norm(music_feat_z, axis=-1, keepdims=True), 1e-8)
        dz = motion_feat_z / np.maximum(np.linalg.norm(motion_feat_z, axis=-1, keepdims=True), 1e-8)
        return mz.astype(np.float32), dz.astype(np.float32)
    with torch.no_grad():
        mf = torch.from_numpy(music_feat_z.astype(np.float32)).to(cfg.device)
        df = torch.from_numpy(motion_feat_z.astype(np.float32)).to(cfg.device)
        me = model.music(mf).detach().cpu().numpy()
        de = model.motion(df).detach().cpu().numpy()
    return me.astype(np.float32), de.astype(np.float32)


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


def reference_motion_inbetween_np(prev_tail: np.ndarray, curr_head: np.ndarray, n_frames: int, cfg: MotionGenerationConfig) -> np.ndarray:
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
    Ra = rot6d_to_matrix_np(a[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    Rb = rot6d_to_matrix_np(b[ROT6D_START:ROT6D_END].reshape(1, NUM_JOINTS, 6))[0]
    qa = matrix_to_quat_np(Ra)
    qb = matrix_to_quat_np(Rb)
    q = _slerp_quaternion_np(qa, qb, phase.reshape(n, 1, 1))
    R = quat_to_matrix_np(q)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(R).reshape(n, -1)

    out, _ = enforce_edge151_contract_np(out, cfg, source_hint="inbetween_motion_inbetween", derive_contact=True, project_rot=True)
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

    ckpt = _trusted_torch_load(ckpt_path, map_location=cfg.device)
    assert_motion_checkpoint_contract(ckpt, cfg, ckpt_path, "boundary_refiner")
    product_mode = normalize_motion_checkpoint_version(
        ckpt.get("version", "")
    ).startswith(
        "product_manifold_boundary_refiner"
    )
    model = (
        ProductManifoldTemporalRefiner(EDGE_DIM, 32)
        if product_mode
        else TemporalRefiner(EDGE_DIM, 32)
    ).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            chunk = motion[st:ed]
            mask = seam_mask[st:ed]
            orig_len = len(chunk)
            cond_in = _condition_chunk_np(cond, st, ed, win if orig_len < win else orig_len)
            if orig_len < win:
                chunk_in = resample_motion_np(chunk, win)
                mask_in = resample_motion_np(mask, win)
            else:
                chunk_in = chunk
                mask_in = mask
            chunk_in, _ = enforce_edge151_contract_np(
                chunk_in, cfg, source_hint="apply_refiner_model:inbetween_input_chunk", derive_contact=True, project_rot=True
            )
            x = torch.from_numpy(chunk_in[None]).float().to(cfg.device)
            c = torch.from_numpy(cond_in[None].astype(np.float32)).float().to(cfg.device)
            sm = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            strength = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * sm, 0.0, 1.0)
            if product_mode:
                joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                    chunk_in[None], mask_in[None], cfg
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
            else:
                delta = model(x, c, sm)
                y = x + delta * strength
            y_np = y[0].detach().cpu().numpy()
            if orig_len < win:
                y_np = resample_motion_np(y_np, orig_len)
            y_np, _ = enforce_edge151_contract_np(
                y_np,
                cfg,
                source_hint="apply_refiner_model:inbetween_output_chunk",
                derive_contact=not product_mode,
                project_rot=True,
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y_np, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum,
        weight_sum,
        rot_quat_accum,
        rot_quat_weight,
        cfg,
        source_hint="apply_refiner_model:inbetween_final",
        derive_contact=not product_mode,
    )
    # Hard blend with original reference according to the exact transition mask.
    w = np.clip(core_strength + (trans_strength - core_strength) * seam_mask.astype(np.float32), 0.0, 1.0)
    if product_mode:
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
    else:
        out = motion.astype(np.float32) * (1.0 - w) + out.astype(np.float32) * w
    out, _ = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint="apply_refiner_model:inbetween_reference_blend",
        derive_contact=not product_mode,
        project_rot=True,
    )
    return out.astype(np.float32)



def apply_diffusion_model(motion: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, ckpt_path: Optional[str], cfg: MotionGenerationConfig) -> np.ndarray:
    """Apply Motion Generation diffusion as transition-masked residual generation.

    The input motion is motion_ref / motion_refined and is treated as a strong
    retrieval/reference condition.  Core frames are locked by mask, while
    transition frames are allowed to be regenerated as residual motion.
    """
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        motion, _ = enforce_edge151_contract_np(
            motion, cfg, source_hint="apply_diffusion_model:inbetween_disabled", derive_contact=True, project_rot=True
        )
        return motion.astype(np.float32)

    core_strength = _inbetween_config_float(cfg, "diffusion_core_strength", "MOTION_DIFFUSION_CORE_STRENGTH", 0.00)
    trans_strength = _inbetween_config_float(cfg, "diffusion_transition_strength", "MOTION_DIFFUSION_TRANSITION_STRENGTH", 0.72)
    noise_scale = _inbetween_config_float(cfg, "diffusion_reference_noise_scale", "MOTION_DIFFUSION_REFERENCE_NOISE_SCALE", 0.03)

    ckpt = _trusted_torch_load(ckpt_path, map_location=cfg.device)
    assert_motion_checkpoint_contract(ckpt, cfg, ckpt_path, "motion_diffusion")
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    tangent_mode = normalize_motion_checkpoint_version(
        ckpt.get("version", "")
    ).startswith(
        "reference_tangent_motion_diffusion"
    )
    model = (
        TangentDiffusionDenoiser(PRODUCT_STATE_DIM, 32)
        if tangent_mode
        else DiffusionDenoiser(EDGE_DIM, 32)
    ).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas, alphas, abar = make_beta_schedule(Tdiff, torch.device(cfg.device))

    T = int(motion.shape[0])
    win = int(cfg.window_len)
    hop = max(1, min(int(getattr(cfg, "hop_len", win)), win))
    accum, weight_sum, rot_quat_accum, rot_quat_weight = init_motion_window_accumulators(T, EDGE_DIM)

    with torch.no_grad():
        for st, ed in sliding_window_ranges(T, win, hop):
            retr_np = motion[st:ed]
            mask_np = seam_mask[st:ed]
            orig_len = len(retr_np)
            cond_in = _condition_chunk_np(cond, st, ed, win if orig_len < win else orig_len)
            if orig_len < win:
                retr_in = resample_motion_np(retr_np, win)
                mask_in = resample_motion_np(mask_np, win)
            else:
                retr_in = retr_np
                mask_in = mask_np
            retr_in, _ = enforce_edge151_contract_np(
                retr_in, cfg, source_hint="apply_diffusion_model:inbetween_retrieval_chunk", derive_contact=True, project_rot=True
            )
            retr = torch.from_numpy(retr_in[None]).float().to(cfg.device)
            raw_mask = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
            mask = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * raw_mask, 0.0, 1.0)
            c = torch.from_numpy(cond_in[None].astype(np.float32)).float().to(cfg.device)
            if tangent_mode:
                joint_np, root_np, contact_np = _risk_masks_for_batch_np(
                    retr_in[None], mask_in[None], cfg
                )
                joint_t = torch.from_numpy(joint_np).float().to(cfg.device)
                root_t = torch.from_numpy(root_np).float().to(cfg.device)
                contact_t = torch.from_numpy(contact_np).float().to(cfg.device)
                effective_joint = joint_t * mask
                effective_root = root_t * mask
                effective_contact = contact_t * mask
                state_mask = _tangent_state_mask(
                    effective_joint, effective_root, effective_contact
                )
                active_mask = (state_mask > 0.0).to(retr.dtype)
                x = (
                    float(noise_scale)
                    * torch.randn(
                        retr.shape[:-1] + (PRODUCT_STATE_DIM,),
                        dtype=retr.dtype,
                        device=retr.device,
                    )
                    * active_mask
                )
                for ti in reversed(range(Tdiff)):
                    t = torch.full(
                        (1,), ti, device=cfg.device, dtype=torch.long
                    )
                    eps = model(
                        x, retr, c, raw_mask, joint_t, t
                    ) * active_mask
                    beta = betas[ti]
                    alpha = alphas[ti]
                    ab = abar[ti]
                    mean = (1 / torch.sqrt(alpha)) * (
                        x
                        - beta
                        / torch.sqrt(1 - ab).clamp_min(1e-6)
                        * eps
                    )
                    if ti > 0:
                        x = (
                            mean
                            + torch.sqrt(beta)
                            * torch.randn_like(x)
                            * 0.35
                            * active_mask
                        )
                    else:
                        x = mean
                    x = x * active_mask
                y_t = _decode_reference_tangent_state(
                    retr,
                    x,
                    effective_joint,
                    effective_root,
                    effective_contact,
                    cfg,
                )
                y = y_t[0].detach().cpu().numpy()
            else:
                x = retr + float(noise_scale) * torch.randn_like(retr) * (0.15 + 0.85 * mask)
                for ti in reversed(range(Tdiff)):
                    t = torch.full((1,), ti, device=cfg.device, dtype=torch.long)
                    eps = model(x, retr, c, raw_mask, t)
                    beta = betas[ti]
                    alpha = alphas[ti]
                    ab = abar[ti]
                    mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - ab).clamp_min(1e-6) * eps)
                    if ti > 0:
                        x = mean + torch.sqrt(beta) * torch.randn_like(x) * 0.35
                    else:
                        x = mean
                    # Strong reference lock: core mask=0 returns exactly retr.
                    x = retr * (1.0 - mask) + x * mask
                y = x[0].detach().cpu().numpy()
            if orig_len < win:
                y = resample_motion_np(y, orig_len)
            y, _ = enforce_edge151_contract_np(
                y,
                cfg,
                source_hint="apply_diffusion_model:inbetween_output_chunk",
                derive_contact=not tangent_mode,
                project_rot=True,
            )
            w = overlap_add_weight_np(orig_len, st, T, hop, win)
            accumulate_motion_window_np(accum, weight_sum, rot_quat_accum, rot_quat_weight, y, w, st, ed)

    out, _ = finalize_motion_window_accum_np(
        accum,
        weight_sum,
        rot_quat_accum,
        rot_quat_weight,
        cfg,
        source_hint="apply_diffusion_model:inbetween_final",
        derive_contact=not tangent_mode,
    )
    # Final exact reference blend in the original-length mask coordinates.
    w = np.clip(core_strength + (trans_strength - core_strength) * seam_mask.astype(np.float32), 0.0, 1.0)
    if tangent_mode:
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
            max_rotation_rad=float(cfg.tangent_diffusion_rotation_cap_rad),
            max_root_m=float(cfg.tangent_diffusion_root_cap_m),
        )
        contact_weight = (full_contact[0] * w > 0.0).astype(np.float32)
        out_geometry[:, :4] = (
            motion[:, :4] * (1.0 - contact_weight)
            + out[:, :4] * contact_weight
        )
        out = out_geometry
    else:
        out = motion.astype(np.float32) * (1.0 - w) + out.astype(np.float32) * w
    out, _ = enforce_edge151_contract_np(
        out,
        cfg,
        source_hint="apply_diffusion_model:inbetween_reference_blend",
        derive_contact=not tangent_mode,
        project_rot=True,
    )
    return out.astype(np.float32)


def derive_contacts_np(motion: np.ndarray, cfg: MotionGenerationConfig) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    joints = fk_24_np(motion)
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
    for st in starts:
        ed = min(T, st + chunk)
        if ed - st < 4:
            continue
        base_np = motion_base[st:ed].copy()
        L = base_np.shape[0]
        base = torch.from_numpy(base_np).float().to(device)
        rot_full = base[:, ROT6D_START:ROT6D_END].reshape(L, NUM_JOINTS, 6).detach().clone()
        root = base[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].detach().clone().requires_grad_(True)
        lower_idx = torch.as_tensor(LOWER_BODY_JOINTS, device=device, dtype=torch.long)
        lower_rot = rot_full[:, lower_idx].detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([lower_rot, root], lr=cfg.ik_lr)
        target = torch.from_numpy(targets[st:ed]).float().to(device)
        contact = torch.from_numpy(contacts[st:ed].astype(np.float32)).float().to(device)
        contact_ramp = torch.from_numpy(contact_ramps[st:ed]).float().to(device)
        confidence = torch.from_numpy(conf[st:ed]).float().to(device)
        floor = torch.tensor(floor_y, device=device, dtype=torch.float32)
        base_rot = rot_full[:, lower_idx].detach().clone()
        base_root = root.detach().clone()
        best_loss = float("inf")
        best_motion = None
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
            if float(loss.detach().cpu()) < best_loss:
                best_loss = float(loss.detach().cpu())
                best_motion = mm.detach().cpu().numpy()
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
            x = torch.from_numpy(out_all[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)).float()
            out_all[:, ROT6D_START:ROT6D_END] = project_rot6d_torch(x).numpy().reshape(T, -1)
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
    sliding_support_eligible: Optional[np.ndarray] = None,
) -> dict:
    cfg = cfg or MotionGenerationConfig()
    contacts, _, _, _ = derive_contacts_np(motion, cfg)
    audited_motion = np.asarray(motion, dtype=np.float32).copy()
    audited_motion[:, :4] = contacts.astype(np.float32)
    from motion_geometry.physical import motion_physical_metrics_np

    report = motion_physical_metrics_np(
        audited_motion,
        fps=float(cfg.fps),
        sliding_support_eligible=sliding_support_eligible,
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


def _finalize_generation_outputs(
    args: argparse.Namespace,
    cfg: MotionGenerationConfig,
    motion: np.ndarray,
    motion_ref: np.ndarray,
    seam_mask: np.ndarray,
    report: Dict[str, Any],
) -> int:
    """Write an accepted motion only after the authoritative physical gate."""

    out = Path(args.out)
    if out.suffix.lower() != ".npy":
        raise ValueError(f"accepted motion output must use .npy suffix, got {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    motion_ref_path = str(out).replace(".npy", ".motion_ref.npy")
    mask_path = str(out).replace(".npy", ".transition_mask.npy")
    json_path = args.json or str(out).replace(".npy", ".inbetween_report.json")

    final_audit = audit_motion_np(motion, cfg)
    if bool(cfg.final_physical_gate_enable):
        final_gate = evaluate_physical_audit(final_audit)
        final_gate["enabled"] = True
    else:
        final_gate = {
            "ok": True,
            "enabled": False,
            "reasons": [],
            "audit": dict(final_audit),
            "policy": "explicitly_disabled",
        }
    final_gate_report = dict(final_gate)
    final_gate_report.pop("audit", None)
    final_gate_report["audit_ref"] = "final_audit"

    report.update(
        {
            "motion_ref_path": motion_ref_path,
            "transition_mask_path": mask_path,
            "final_audit": final_audit,
            "final_physical_gate": final_gate_report,
            "accepted_output_path": str(out),
        }
    )
    np.save(motion_ref_path, motion_ref.astype(np.float32))
    np.save(mask_path, seam_mask.astype(np.float32))

    rejected = bool(
        cfg.final_physical_gate_enable
        and cfg.final_physical_gate_fail_closed
        and not final_gate["ok"]
    )
    if rejected:
        quarantined_output: Optional[Path] = None
        if out.exists():
            quarantined_output = out.with_name(
                f"{out.stem}.preexisting_{now_tag()}{out.suffix}"
            )
            serial = 1
            while quarantined_output.exists():
                quarantined_output = out.with_name(
                    f"{out.stem}.preexisting_{now_tag()}_{serial}{out.suffix}"
                )
                serial += 1
            out.replace(quarantined_output)
        rejected_path: Optional[Path] = None
        if bool(cfg.final_physical_gate_save_rejected):
            rejected_path = out.with_name(f"{out.stem}.rejected{out.suffix or '.npy'}")
            np.save(rejected_path, motion.astype(np.float32))
        report.update(
            {
                "generation_status": "rejected_by_final_physical_gate",
                "accepted_output_written": False,
                "rejected_motion_path": str(rejected_path) if rejected_path else None,
                "preexisting_accepted_output_quarantined": (
                    str(quarantined_output) if quarantined_output else None
                ),
            }
        )
        save_json(report, json_path)
        print(
            json.dumps(
                {
                    "status": report["generation_status"],
                    "motion": None,
                    "rejected_motion": report["rejected_motion_path"],
                    "json": json_path,
                    "final_physical_gate": final_gate_report,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    np.save(out, motion.astype(np.float32))
    report.update(
        {
            "generation_status": "accepted",
            "accepted_output_written": True,
            "rejected_motion_path": None,
        }
    )
    save_json(report, json_path)
    if args.render_output:
        render_if_possible(
            str(out),
            args.audio,
            args.render_output,
            args.render_script,
            fps=float(cfg.fps),
        )
    print(
        json.dumps(
            {
                "status": report["generation_status"],
                "motion": str(out),
                "motion_ref": motion_ref_path,
                "transition_mask": mask_path,
                "json": json_path,
                "frames": int(motion.shape[0]),
                "final_physical_gate": final_gate_report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0





def _generate_motion_core(args: argparse.Namespace) -> int:
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    sem_dirs = getattr(args, "music_semantic_dirs", None)
    if sem_dirs:
        cfg.external_music_semantic_dirs = os.pathsep.join([str(x) for x in sem_dirs])
    if getattr(args, "external_music_semantic_cmd", None):
        cfg.external_music_semantic_cmd = str(args.external_music_semantic_cmd)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch is not None:
        torch.manual_seed(cfg.seed)
    db = load_db(args.db)
    database_contract = _training_db_contract(db, cfg, "Generation")
    cfg._event_db_contract = database_contract["event_db_contract"]
    contrastive = load_contrastive(args.contrastive, cfg)
    slots, slot_feat = audio_slots(args.audio, cfg, args.slot_seconds, args.slots_json)
    path_idx, retrieval_report = retrieve_schedule(slots, slot_feat, db, cfg, contrastive)
    paths = np.asarray(db["paths"], dtype=object)
    selected_paths = [str(paths[i]) for i in path_idx]

    motion_ref, concat_report = concat_events(selected_paths, [s["duration"] for s in slots], cfg)

    transition_spans: List[List[int]] = []
    for r in concat_report:
        for sp in r.get("transition_spans", []):
            if sp is not None and len(sp) >= 2:
                transition_spans.append([int(sp[0]), int(sp[1])])
    if transition_spans:
        seam_mask = make_transition_budget_mask(motion_ref.shape[0], transition_spans, cfg)
        seam_positions = [int((a + b) // 2) for a, b in transition_spans]
        mask_policy = "inbetween_transition_budget_spans"
    else:
        seam_positions = []
        acc = 0
        for r in concat_report[:-1]:
            acc += int(r.get("target_frames", 0))
            seam_positions.append(acc)
        seam_mask = make_boundary_mask(motion_ref.shape[0], seam_positions, width=24)
        mask_policy = "fallback_boundary_mask_no_transition_spans"

    cond = build_frame_local_conditioning(
        slot_feat,
        concat_report,
        motion_ref.shape[0],
        np.asarray(db["desc_mean"], dtype=np.float32)[0],
        np.asarray(db["desc_std"], dtype=np.float32)[0],
    )

    stage_reports = {
        "retrieval": retrieval_report,
        "concat": concat_report,
        "seams": seam_positions,
        "transition_spans": transition_spans,
        "seam_mask_policy": mask_policy,
        "seam_mask_stats": {
            "shape": list(seam_mask.shape),
            "mean": float(np.mean(seam_mask)) if seam_mask.size else 0.0,
            "max": float(np.max(seam_mask)) if seam_mask.size else 0.0,
            "transition_frame_ratio": float(np.mean(seam_mask[:, 0] > 0.5)) if seam_mask.size else 0.0,
        },
        "inbetween_reference_conditioning": {
            "motion_ref_as_strong_reference": True,
            "diffusion_edit_policy": "transition_masked_residual_generation",
            "ik_finalization": bool(cfg.ik_enable),
            "env": {
                "MOTION_TRANSITION_BUDGET_ENABLE": os.environ.get("MOTION_TRANSITION_BUDGET_ENABLE", "1"),
                "MOTION_TRANSITION_INBETWEEN_ENABLE": os.environ.get("MOTION_TRANSITION_INBETWEEN_ENABLE", "1"),
                "MOTION_REFINER_CORE_STRENGTH": os.environ.get("MOTION_REFINER_CORE_STRENGTH", "0.02"),
                "MOTION_DIFFUSION_CORE_STRENGTH": os.environ.get("MOTION_DIFFUSION_CORE_STRENGTH", "0.00"),
                "MOTION_DIFFUSION_TRANSITION_STRENGTH": os.environ.get("MOTION_DIFFUSION_TRANSITION_STRENGTH", "0.72"),
            },
        },
        "neural_music_conditioning": {
            "mode": "frame_local_slot_conditioning",
            "shape": list(cond.shape),
            "transition_policy": "linear_previous_to_current_slot_descriptor",
            "normalization": "training_event_db_descriptor_coordinates",
            "whole_song_mean_conditioning": False,
        },
    }
    pre_audit = audit_motion_np(motion_ref, cfg)
    motion = motion_ref.astype(np.float32)

    if cfg.refiner_enable:
        motion = apply_refiner_model(motion, cond, seam_mask, args.refiner, cfg)
        stage_reports["boundary_refiner_audit"] = audit_motion_np(motion, cfg)
    if cfg.diffusion_enable:
        motion = apply_diffusion_model(motion, cond, seam_mask, args.diffusion, cfg)
        stage_reports["motion_diffusion_audit"] = audit_motion_np(motion, cfg)

    ik_report = {"enabled": False}
    if cfg.ik_enable:
        motion, ik_report = true_lower_body_ik(motion, cfg)

    report = {
        "version": "inbetween_reference_conditioned_transition_masked_motionrag_diff",
        "audio": args.audio,
        "db": args.db,
        "config": dataclasses.asdict(cfg),
        "fk_tree_source": FK_TREE_SOURCE,
        "selected_event_indices": path_idx,
        "selected_event_paths": selected_paths,
        "slots": slots,
        "pre_refine_audit": pre_audit,
        "stage_reports": stage_reports,
        "lower_body_ik_true_ik": ik_report,
    }
    return _finalize_generation_outputs(
        args, cfg, motion, motion_ref, seam_mask, report
    )


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

    b = sub.add_parser("build-db", help="Build source-aware event database from EDGE 151D motion files")
    b.add_argument("--motion_dirs", nargs="+", required=True)
    b.add_argument("--manifest", default=None, help="Optional manifest.csv. When present, its source_bvh/fragment_file/label/start_frame/end_frame fields become parent metadata for source-aware Event-RAG.")
    b.add_argument("--out_db", required=True, help="Output directory, containing events.npz and events/*.npy")
    b.add_argument("--audio_dirs", nargs="*", default=None, help="Optional paired audio/music directories. Same-stem files are used for real Contrastive Retriever music features.")
    b.set_defaults(func=build_db)

    c = sub.add_parser("train-contrastive", help="Contrastive Retriever music-motion contrastive training")
    c.add_argument("--db", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--music_feature_npz", default=None)
    c.add_argument("--unpaired_audio_dirs", nargs="*", default=None, help="Unpaired target/background music directories. Used for Root-Aware Motion Safety semantic OT pseudo-pairs when BVH has no synchronized audio.")
    c.add_argument("--audio_dirs", nargs="*", default=None, help="Alias for --unpaired_audio_dirs; kept for script compatibility.")
    c.add_argument("--music_semantic_dirs", nargs="*", default=None, help="External Music Semantics directories containing external classical-music semantic JSON/NPZ sidecars.")
    c.add_argument("--external_music_semantic_cmd", default=None, help="Optional command template: use {audio}, {out_json}, {out_npz}, {stem} placeholders to call an external trained music semantic model.")
    c.add_argument("--epochs", type=int, default=None)
    c.set_defaults(func=train_contrastive)

    r = sub.add_parser("train-refiner", help="Motion Refiner residual Motion Refiner training")
    r.add_argument("--db", required=True)
    r.add_argument("--val_db", required=True, help="Required source-disjoint validation Event-DB used for physical, contract and leakage gates")
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=None)
    r.set_defaults(func=train_refiner)

    d = sub.add_parser("train-diffusion", help="Motion Generation conditional residual diffusion training")
    d.add_argument("--db", required=True)
    d.add_argument("--val_db", required=True, help="Required source-disjoint validation Event-DB used for physical, contract and leakage gates")
    d.add_argument("--out", required=True)
    d.add_argument("--steps", type=int, default=None)
    d.add_argument("--diffusion_steps", type=int, default=None)
    d.set_defaults(func=train_diffusion)

    g = sub.add_parser("generate", help="Generate whole-song motion from music via Motion Generation pipeline")
    g.add_argument("--audio", required=True)
    g.add_argument("--slots_json", default=None)
    g.add_argument("--music_semantic_dirs", nargs="*", default=None, help="External Music Semantics directories containing external classical-music semantic JSON/NPZ sidecars.")
    g.add_argument("--external_music_semantic_cmd", default=None, help="Optional command template for an external trained music semantic model.")
    g.add_argument("--slot_seconds", type=float, default=4.0)
    g.add_argument("--db", required=True)
    g.add_argument("--contrastive", default=None)
    g.add_argument("--refiner", default=None)
    g.add_argument("--diffusion", default=None)
    g.add_argument("--out", required=True)
    g.add_argument("--json", default=None)
    g.add_argument("--render_output", default=None)
    g.add_argument("--render_script", default="rendering/render_motion.py")
    g.set_defaults(func=generate)

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





# ===== Semantic Routing COMPLETE MSSD-AESD ROUTING PATCH START =====
# Complete chain: MSSD music slots -> AESD action descriptors -> routing-aware
# Semantic OT for Contrastive Retriever -> routing-aware beam search for Event-RAG.
try:
    from events.semantic_descriptor import (
        MUSIC_SEMANTIC_LABELS as _ROUTING_SEMANTIC_LABELS,
        parse_descriptor_file as _semantic_routing_parse_descriptor_file,
        load_descriptor_for_audio as _semantic_routing_load_descriptor_for_audio,
        env_bool as _semantic_routing_env_bool,
        get_aesd_prob_matrix as _event_semantic_probability_matrix,
        slot_prob_vector as _semantic_routing_slot_prob_vector,
        dot_compat as _semantic_routing_dot_compat,
        normalize_vector as _semantic_routing_normalize_vector,
    )
except Exception as _routing_import_error:  # pragma: no cover
    _ROUTING_SEMANTIC_LABELS = ["calm_meditative", "pose_hold", "lyrical_flow", "instrument_phrase", "percussive_accent", "turning_climax", "footwork_flow", "aerial_curve"]
    _semantic_routing_parse_descriptor_file = None
    _semantic_routing_load_descriptor_for_audio = None
    _semantic_routing_env_bool = None
    _event_semantic_probability_matrix = None
    _semantic_routing_slot_prob_vector = None
    _semantic_routing_dot_compat = None
    _semantic_routing_normalize_vector = None
    print(f"[Semantic Routing WARN] import failed: {_routing_import_error}", file=sys.stderr)


def _routing_env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


def _routing_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _routing_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _descriptor_directories_from_config(cfg: MotionGenerationConfig) -> str:
    parts = []
    for attr in ["music_descriptor_dirs", "external_music_semantic_dirs"]:
        val = getattr(cfg, attr, "")
        if val:
            if isinstance(val, (list, tuple)):
                parts.extend([str(x) for x in val])
            else:
                parts.extend(str(val).replace(";", os.pathsep).split(os.pathsep))
    extra = os.environ.get("MOTION_MSSD_DESCRIPTOR_DIRS", "") or os.environ.get("SEMANTIC_ROUTING_DESCRIPTOR_DIRS", "")
    if extra:
        parts.extend(extra.replace(";", os.pathsep).split(os.pathsep))
    return os.pathsep.join([p for p in parts if str(p).strip()])


def _mark_slots_with_metadata(slots: List[dict], meta: dict) -> List[dict]:
    out = []
    for s0 in slots:
        s = dict(s0)
        s.setdefault("mssd_usage", meta.get("usage"))
        s.setdefault("mssd_is_final_schedule", bool(meta.get("is_final_schedule", False)))
        s.setdefault("mssd_slot_source", meta.get("slot_source", s.get("slot_source", "")))
        for k in ["router_ckpt", "planner_ckpt", "duration_model_ckpt", "raw_schedule_json", "schedule_summary_json", "descriptor_schema_version"]:
            if meta.get(k) is not None:
                s.setdefault("mssd_" + k, meta.get(k))
        out.append(s)
    return out


def _load_routing_slots_json(slots_json: str | Path, cfg: MotionGenerationConfig, require_final: bool = False) -> Tuple[List[dict], np.ndarray, dict]:
    if _semantic_routing_parse_descriptor_file is None:
        slots, feats, meta = _load_music_slots_json(slots_json, cfg)
        return slots, feats, meta
    slots, feats, meta = _semantic_routing_parse_descriptor_file(
        slots_json,
        require_final_schedule=bool(require_final),
        fps=float(getattr(cfg, "fps", 30.0)),
        temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
        usage="generate_schedule" if require_final else "auto",
    )
    slots = _mark_slots_with_metadata(slots, meta)
    return slots, feats, meta


def _build_routing_audio_slots(path: str | Path, cfg: MotionGenerationConfig, slot_seconds: float = 4.0, slots_json: Optional[str] = None) -> Tuple[List[dict], np.ndarray]:
    strict = _slot_plan_env_bool("MOTION_REQUIRE_PRETRAINED_ROUTER_SLOTS", False)
    mssd_enabled = _routing_env_bool("MOTION_MSSD_ENABLE", True)
    require_final = strict or _routing_env_bool("MOTION_MSSD_REQUIRE_FINAL_SCHEDULE_FOR_GENERATE", False)
    if slots_json and Path(slots_json).exists():
        if mssd_enabled and _semantic_routing_parse_descriptor_file is not None:
            slots, feats, meta = _load_routing_slots_json(slots_json, cfg, require_final=require_final)
            print(f"[Semantic Routing MSSD] loaded descriptor: {slots_json} slots={len(slots)} usage={meta.get('usage')} final={meta.get('is_final_schedule')} source={meta.get('slot_source')}")
            return slots, feats
        slots, feats, meta = _load_music_slots_json(slots_json, cfg)
        allowed = not strict
        raw = " ".join(str(meta.get(k, "")) for k in ["slot_source", "router_ckpt", "planner_ckpt", "duration_model_ckpt"])
        if has_scheduler_provenance(raw):
            allowed = True
        if not allowed:
            raise RuntimeError("MOTION_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 but slots_json is not final trained-router/planner MSSD")
        return _mark_slots_with_metadata(slots, meta), feats
    if mssd_enabled and _semantic_routing_load_descriptor_for_audio is not None:
        loaded = _semantic_routing_load_descriptor_for_audio(
            path,
            descriptor_dirs=_descriptor_directories_from_config(cfg),
            require_final_schedule=require_final,
            fps=float(getattr(cfg, "fps", 30.0)),
            temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
            usage="generate_schedule" if require_final else "auto",
        )
        if loaded is not None:
            slots, feats, meta = loaded
            slots = _mark_slots_with_metadata(slots, meta)
            print(f"[Semantic Routing MSSD] loaded sidecar descriptor: audio={path} slots={len(slots)} usage={meta.get('usage')} final={meta.get('is_final_schedule')} source={meta.get('slot_source')}")
            return slots, feats
    if strict:
        raise RuntimeError("MOTION_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 but no final MSSD/slots_json was provided. Build it with python -m scheduling.music_slot_descriptor")
    return build_default_audio_slots(path, cfg, slot_seconds, slots_json)


# Override old loaders.  Generate uses strict final MSSD; Contrastive Retriever training does not.
audio_slots = _build_routing_audio_slots


def parse_external_music_semantic_file(path: str | Path, cfg: MotionGenerationConfig) -> Optional[Tuple[List[dict], np.ndarray]]:
    if _semantic_routing_parse_descriptor_file is None:
        return None
    try:
        slots, feats, meta = _semantic_routing_parse_descriptor_file(
            path,
            require_final_schedule=False,
            fps=float(getattr(cfg, "fps", 30.0)),
            temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
            usage="train_semantic",
        )
        return _mark_slots_with_metadata(slots, meta), feats
    except Exception as exc:
        print(f"[Semantic Routing MSSD WARN] failed parsing weak descriptor {path}: {exc}", file=sys.stderr)
        return None


def load_external_music_semantic_slots(audio_path: str | Path, cfg: MotionGenerationConfig, slot_seconds: float) -> Optional[Tuple[List[dict], np.ndarray]]:
    if not bool(getattr(cfg, "external_music_semantic_enable", True)):
        return None
    if _semantic_routing_load_descriptor_for_audio is not None:
        loaded = _semantic_routing_load_descriptor_for_audio(
            audio_path,
            descriptor_dirs=_descriptor_directories_from_config(cfg),
            require_final_schedule=False,
            fps=float(getattr(cfg, "fps", 30.0)),
            temperature=float(getattr(cfg, "external_music_semantic_temperature", 0.65)),
            usage="train_semantic",
        )
        if loaded is not None:
            slots, feats, meta = loaded
            return _mark_slots_with_metadata(slots, meta), feats
    cmd_out = run_external_music_semantic_cmd(audio_path, cfg)
    if cmd_out is not None:
        parsed = parse_external_music_semantic_file(cmd_out, cfg)
        if parsed is not None:
            return parsed
    if bool(getattr(cfg, "external_music_semantic_proxy_enable", True)):
        prox = filename_proxy_music_semantic(audio_path, cfg, slot_seconds)
        if prox is not None:
            return prox
    if bool(getattr(cfg, "external_music_semantic_required", False)):
        raise RuntimeError(f"External/MSSD music semantic is required but none was found for {audio_path}")
    return None


def load_unpaired_audio_feature_pool(audio_dirs: Optional[Sequence[str]], cfg: MotionGenerationConfig) -> Tuple[np.ndarray, List[dict]]:
    files = collect_audio_files(audio_dirs)
    feats: List[np.ndarray] = []
    meta: List[dict] = []
    old_strict = os.environ.get("MOTION_REQUIRE_PRETRAINED_ROUTER_SLOTS")
    try:
        os.environ["MOTION_REQUIRE_PRETRAINED_ROUTER_SLOTS"] = "0"
        for f in files:
            try:
                parsed = load_external_music_semantic_slots(f, cfg, slot_seconds=float(cfg.unpaired_audio_slot_seconds))
                if parsed is not None:
                    slots, sf = parsed
                else:
                    slots, sf = build_default_audio_slots(f, cfg, slot_seconds=float(cfg.unpaired_audio_slot_seconds), slots_json=None)
            except Exception as exc:
                print(f"[Semantic Routing WARN] failed unpaired audio feature extraction {f}: {exc}", file=sys.stderr)
                continue
            for slot, feat in zip(slots, sf):
                feats.append(feat.astype(np.float32))
                meta.append({"audio": str(f), "slot": dict(slot)})
    finally:
        if old_strict is None:
            os.environ.pop("MOTION_REQUIRE_PRETRAINED_ROUTER_SLOTS", None)
        else:
            os.environ["MOTION_REQUIRE_PRETRAINED_ROUTER_SLOTS"] = old_strict
    if not feats:
        return np.zeros((0, 32), dtype=np.float32), []
    return np.stack(feats).astype(np.float32), meta


def _slot_probability_matrix(audio_meta: List[dict]) -> np.ndarray:
    rows = []
    for m in audio_meta:
        slot = m.get("slot", {})
        if _semantic_routing_slot_prob_vector is not None:
            rows.append(_semantic_routing_slot_prob_vector(slot))
        else:
            rows.append(np.ones((len(_ROUTING_SEMANTIC_LABELS),), dtype=np.float32) / max(1, len(_ROUTING_SEMANTIC_LABELS)))
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, len(_ROUTING_SEMANTIC_LABELS)), dtype=np.float32)


def build_unpaired_audio_motion_pairs(db: dict, audio_dirs: Optional[Sequence[str]], cfg: MotionGenerationConfig) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]]:
    """Semantic Routing MSSD-AESD Semantic OT.

    It still uses real unpaired music features and motion descriptors, but the OT
    cost now includes a soft MSSD-to-AESD probability distance.  Thus Contrastive Retriever is no
    longer trained from only low-level energy/duration hints.
    """
    audio_raw, audio_meta = load_unpaired_audio_feature_pool(audio_dirs, cfg)
    if audio_raw.shape[0] < int(cfg.unpaired_min_audio_slots):
        return None
    motion_z = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "classification_ot_weight", getattr(cfg, "filename_semantic_ot_weight", 0.35))))
    desc_mean = np.asarray(db["desc_mean"], dtype=np.float32)
    desc_std = np.asarray(db["desc_std"], dtype=np.float32)
    music_z_all = ((audio_raw - desc_mean) / np.maximum(desc_std, 1e-6)).astype(np.float32)
    music_z_all = np.clip(music_z_all, -8.0, 8.0).astype(np.float32)
    motion_z = np.clip(motion_z, -8.0, 8.0).astype(np.float32)
    dims, weights = semantic_dims_and_weights()
    diff = music_z_all[:, None, dims] - motion_z[None, :, dims]
    cost_num = np.sum((diff * weights[None, None, :]) ** 2, axis=-1)
    if _event_semantic_probability_matrix is not None:
        slot_probs = _slot_probability_matrix(audio_meta)
        aesd_probs = _event_semantic_probability_matrix(db, motion_z.shape[0])
        sem_compat = np.clip(slot_probs @ aesd_probs.T, 0.0, 1.0)
        cost_sem = 1.0 - sem_compat
    else:
        sem_compat = np.zeros((music_z_all.shape[0], motion_z.shape[0]), dtype=np.float32)
        cost_sem = 0.0
    lam_sem = _routing_env_float("SEMANTIC_ROUTING_OT_SEMANTIC_WEIGHT", 1.25)
    rng = np.random.default_rng(int(cfg.seed) + 4638)
    cost = cost_num + lam_sem * cost_sem + rng.normal(0.0, 1e-5, size=cost_num.shape).astype(np.float32)
    topk = max(1, min(int(cfg.unpaired_positive_topk), motion_z.shape[0]))
    pairs_per = max(1, min(int(cfg.unpaired_pairs_per_audio_slot), topk))
    music_pairs: List[np.ndarray] = []
    motion_pairs: List[np.ndarray] = []
    pair_preview: List[dict] = []
    for ai in range(cost.shape[0]):
        top = np.argpartition(cost[ai], topk - 1)[:topk]
        top = top[np.argsort(cost[ai, top])]
        chosen = top[:pairs_per]
        for mi in chosen:
            music_pairs.append(music_z_all[ai])
            motion_pairs.append(motion_z[int(mi)])
        if len(pair_preview) < 20:
            pair_preview.append({
                "audio": audio_meta[ai].get("audio", ""),
                "slot_id": int(audio_meta[ai].get("slot", {}).get("slot_id", ai)),
                "slot_music_semantic_label": str(audio_meta[ai].get("slot", {}).get("music_semantic_top_label", audio_meta[ai].get("slot", {}).get("music_alignment_label", ""))),
                "slot_descriptor_usage": str(audio_meta[ai].get("slot", {}).get("usage", audio_meta[ai].get("slot", {}).get("mssd_usage", ""))),
                "top_motion_ids": [int(x) for x in top[: min(5, len(top))].tolist()],
                "top_costs": [float(cost[ai, int(x)]) for x in top[: min(5, len(top))].tolist()],
                "top_mssd_aesd_compat": [float(sem_compat[ai, int(x)]) for x in top[: min(5, len(top))].tolist()] if isinstance(sem_compat, np.ndarray) and sem_compat.size else [],
            })
    if len(music_pairs) < 2:
        return None
    music = np.stack(music_pairs).astype(np.float32)
    motion = np.stack(motion_pairs).astype(np.float32)
    report = {
        "mode": "semantic_routing_mssd_aesd_semantic_ot",
        "audio_files": sorted(set(m["audio"] for m in audio_meta)),
        "num_audio_slots": int(audio_raw.shape[0]),
        "num_motion_events": int(motion_z.shape[0]),
        "num_training_pairs": int(music.shape[0]),
        "positive_topk": int(topk),
        "pairs_per_audio_slot": int(pairs_per),
        "semantic_dims": [int(x) for x in dims.tolist()],
        "semantic_weights": [float(x) for x in weights.tolist()],
        "mssd_aesd_semantic_weight": float(lam_sem),
        "aesd_arrays_used": bool("aesd_music_alignment_probs" in db),
        "pair_preview": pair_preview,
    }
    return music, motion, desc_mean.astype(np.float32), desc_std.astype(np.float32), report


def _stage_sequence_score(slot: dict, stages: np.ndarray) -> np.ndarray:
    role = str(slot.get("role", slot.get("slot_role", "normal")))
    out = np.zeros((len(stages),), dtype=np.float32)
    good = {
        "intro": {"intro", "intro_or_resolution", "anchor_or_resolution"},
        "calm": {"intro", "intro_or_resolution", "resolution", "anchor_or_resolution"},
        "release": {"resolution", "anchor_or_resolution", "intro_or_resolution"},
        "resolution": {"resolution", "anchor_or_resolution", "intro_or_resolution"},
        "normal": {"development", "build_up", "motif_recall"},
        "development": {"development", "build_up"},
        "build_up": {"build_up", "development", "opening_or_climax"},
        "accent": {"accent_or_climax", "climax", "build_up"},
        "climax": {"climax", "accent_or_climax", "opening_or_climax"},
        "motif": {"motif_recall", "development"},
        "motif_recall": {"motif_recall", "anchor_or_resolution"},
    }.get(role, {"development", "build_up"})
    for i, st in enumerate(stages):
        if str(st) in good:
            out[i] = 1.0
    return out


def retrieve_schedule(slots: List[dict], slot_feat: np.ndarray, db: dict, cfg: MotionGenerationConfig, contrastive=None) -> Tuple[List[int], List[dict]]:
    """Semantic Routing global MSSD-AESD routing-aware Event-RAG."""
    desc = np.asarray(db["desc"], dtype=np.float32)
    desc_z = motion_feature_z_for_alignment(db, cfg, weight=float(getattr(cfg, "classification_retrieval_weight", getattr(cfg, "filename_semantic_retrieval_weight", 0.20))))
    mean = np.asarray(db["desc_mean"], dtype=np.float32)
    std = np.asarray(db["desc_std"], dtype=np.float32)
    if contrastive is not None and hasattr(contrastive, "music_mean") and hasattr(contrastive, "music_std"):
        music_mean = np.asarray(getattr(contrastive, "music_mean"), dtype=np.float32)
        music_std = np.asarray(getattr(contrastive, "music_std"), dtype=np.float32)
        music_z = (slot_feat - music_mean) / np.maximum(music_std, 1e-6)
    else:
        music_z = (slot_feat - mean) / np.maximum(std, 1e-6)
    music_z = np.clip(music_z, -8.0, 8.0).astype(np.float32)
    desc_z = np.clip(desc_z, -8.0, 8.0).astype(np.float32)
    music_emb, motion_emb = embed_with_contrastive(contrastive, music_z, desc_z, cfg)
    n = int(len(desc))
    sources = np.asarray(db["source_groups"], dtype=object)
    source_uids = np.asarray(db.get("source_uids", sources), dtype=object)
    durations = np.asarray(db["durations"], dtype=np.float32)
    entries = np.asarray(db["entry"], dtype=np.float32); exits = np.asarray(db["exit"], dtype=np.float32)
    centry = np.asarray(db["contact_entry"], dtype=np.float32); cexit = np.asarray(db["contact_exit"], dtype=np.float32)
    dance_keys = np.asarray(db.get("dance_keys", np.array(["unknown"] * n, dtype=object)), dtype=object)
    labels_arr = np.asarray(db.get("labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    align_arr = np.asarray(db.get("music_alignment_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    families = np.asarray(db.get("event_families", np.array(["unknown"] * n, dtype=object)), dtype=object)
    stages = np.asarray(db.get("motion_stage_roles", np.array(["unknown"] * n, dtype=object)), dtype=object)
    locomotion = np.asarray(db.get("locomotion_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    support = np.asarray(db.get("support_labels", np.array(["unknown"] * n, dtype=object)), dtype=object)
    sem_conf = np.asarray(db.get("semantic_confidence", np.ones(n, dtype=np.float32)), dtype=np.float32)
    event_quality = np.asarray(db.get("event_quality_scores", np.ones(n, dtype=np.float32)), dtype=np.float32)
    nat_min = np.asarray(db.get("natural_duration_min", np.ones(n, dtype=np.float32) * 1.5), dtype=np.float32)
    nat_max = np.asarray(db.get("natural_duration_max", np.ones(n, dtype=np.float32) * 4.0), dtype=np.float32)
    aesd_probs = _event_semantic_probability_matrix(db, n) if _event_semantic_probability_matrix is not None else np.zeros((n, len(_ROUTING_SEMANTIC_LABELS)), dtype=np.float32)
    aesd_risk = np.asarray(db.get("aesd_boundary_risk", np.zeros(n, dtype=np.float32)), dtype=np.float32)
    aesd_semantics = np.asarray(db.get("aesd_event_semantics", align_arr), dtype=object)

    w_contrastive = _routing_env_float("SEMANTIC_ROUTING_ROUTE_CONTRASTIVE_WEIGHT", 1.00)
    w_mssd_aesd = _routing_env_float("SEMANTIC_ROUTING_ROUTE_MSSD_AESD_WEIGHT", 1.15)
    w_legacy_sem = _routing_env_float("SEMANTIC_ROUTING_ROUTE_LEGACY_SEM_WEIGHT", float(getattr(cfg, "semantic_routing_weight", 0.72)))
    w_duration = _routing_env_float("SEMANTIC_ROUTING_ROUTE_DURATION_WEIGHT", float(getattr(cfg, "route_natural_duration_weight", 0.20)))
    w_quality = _routing_env_float("SEMANTIC_ROUTING_ROUTE_QUALITY_WEIGHT", float(getattr(cfg, "event_quality_weight", 0.22)))
    w_stage = _routing_env_float("SEMANTIC_ROUTING_ROUTE_STAGE_WEIGHT", float(getattr(cfg, "route_stage_sequence_weight", 0.16)))
    w_boundary_risk = _routing_env_float("SEMANTIC_ROUTING_ROUTE_BOUNDARY_RISK_WEIGHT", 0.35)
    top_debug = max(1, int(getattr(cfg, "classification_report_topk", 8)))
    candidate_k = max(int(getattr(cfg, "top_k", 32)), int(getattr(cfg, "beam_size", 8)), _routing_env_int("SEMANTIC_ROUTING_ROUTE_CANDIDATE_TOPK", 96), top_debug)

    beams: List[Tuple[float, List[int], Dict[str, int]]] = [(0.0, [], {})]
    reports: List[dict] = []
    for i, slot in enumerate(slots):
        sim = music_emb[i] @ motion_emb.T
        slot_dur = max(float(slot.get("duration", durations.mean() if len(durations) else 1.0)), 1e-4)
        dur_cost = np.abs(np.log(np.maximum(durations, 1e-4) / slot_dur))
        in_range = ((slot_dur >= nat_min) & (slot_dur <= nat_max)).astype(np.float32)
        center = np.maximum((nat_min + nat_max) * 0.5, 1e-4)
        natural_score = in_range + (1.0 - in_range) * np.exp(-np.abs(np.log(slot_dur / center))).astype(np.float32)
        legacy_sem = semantic_label_match_bonus(slot, db, cfg)
        if _semantic_routing_slot_prob_vector is not None and _semantic_routing_dot_compat is not None:
            slot_prob = _semantic_routing_slot_prob_vector(slot)
            mssd_aesd_score = _semantic_routing_dot_compat(slot_prob, aesd_probs)
        else:
            mssd_aesd_score = legacy_sem.astype(np.float32)
        stage_score = _stage_sequence_score(slot, stages)
        quality_term = np.clip(event_quality, 0.0, 1.0)
        low_quality_penalty = np.maximum(0.0, float(getattr(cfg, "chang_e_min_event_quality", 0.22)) - quality_term)
        base_score = (
            w_contrastive * sim
            + w_mssd_aesd * mssd_aesd_score
            + w_legacy_sem * legacy_sem
            + w_duration * natural_score
            + w_quality * quality_term
            + w_stage * stage_score
            + 0.04 * np.clip(sem_conf, 0.0, 1.0)
            - float(getattr(cfg, "retrieval_warp_penalty", 0.18)) * dur_cost
            - w_boundary_risk * np.clip(aesd_risk, 0.0, 1.0)
            - 0.75 * low_quality_penalty
        )
        cand = np.argsort(-base_score)[: min(candidate_k, n)].tolist()
        new_beams: List[Tuple[float, List[int], Dict[str, int]]] = []
        for score, path, usage in beams:
            prev = path[-1] if path else None
            for idx in cand:
                sc = float(base_score[idx])
                src = str(sources[idx]); suid = str(source_uids[idx]); dk = str(dance_keys[idx]); fam = str(families[idx]); stg = str(stages[idx])
                source_key = "src::" + src
                uid_key = "suid::" + suid
                dance_key = "dance::" + dk
                fam_key = "fam::" + fam
                sc -= float(getattr(cfg, "route_source_repeat_penalty", cfg.retrieval_source_penalty)) * usage.get(source_key, 0)
                sc -= 0.08 * usage.get(uid_key, 0)
                sc -= float(getattr(cfg, "route_dance_key_repeat_penalty", 0.16)) * usage.get(dance_key, 0)
                fam_recent_window = max(1, int(getattr(cfg, "route_family_recent_window", 8)))
                fam_recent_count = sum(1 for p_idx in path[-fam_recent_window:] if str(families[p_idx]) == fam)
                fam_pen = float(getattr(cfg, "route_family_balance_penalty", 0.18)) * max(0, fam_recent_count - 1)
                sc -= min(float(getattr(cfg, "route_family_penalty_cap", 0.25)), fam_pen)
                run_count = 0
                for p_idx in reversed(path):
                    if str(sources[p_idx]) == src:
                        run_count += 1
                    else:
                        break
                if run_count >= 2:
                    sc -= float(getattr(cfg, "route_source_run_hard_penalty", 0.30))
                if str(slot.get("role", "")) in {"motif", "motif_recall"} and usage.get(fam_key, 0) > 0:
                    sc += float(getattr(cfg, "route_motif_recall_bonus", 0.12))
                if i == 0 and stg in {"intro", "intro_or_resolution"}:
                    sc += w_stage
                elif i >= len(slots) - 2 and stg in {"resolution", "anchor_or_resolution", "intro_or_resolution"}:
                    sc += w_stage
                if prev is not None:
                    raw_tc = transition_cost(exits[prev], entries[idx], cexit[prev], centry[idx])
                    transition_pen = float(getattr(cfg, "retrieval_transition_penalty", 0.65)) * raw_tc
                    # Risk-aware local transition penalty: difficult incoming events need cleaner previous exits.
                    transition_pen += 0.18 * float(aesd_risk[idx])
                    if str(support[prev]) != str(support[idx]):
                        transition_pen += 0.04
                    sc -= transition_pen
                    if src == str(sources[prev]):
                        sc -= float(getattr(cfg, "retrieval_repeat_penalty", 0.15))
                    if fam == str(families[prev]):
                        sc -= float(getattr(cfg, "route_family_repeat_penalty", 0.12))
                ns = dict(usage)
                ns[source_key] = ns.get(source_key, 0) + 1
                ns[uid_key] = ns.get(uid_key, 0) + 1
                ns[dance_key] = ns.get(dance_key, 0) + 1
                ns[fam_key] = ns.get(fam_key, 0) + 1
                new_beams.append((score + sc, path + [int(idx)], ns))
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[: max(1, int(getattr(cfg, "beam_size", 8)))]
        preview = []
        for j in cand[: min(top_debug, len(cand))]:
            j = int(j)
            preview.append({
                "event_id": j,
                "final_local_base_score": float(base_score[j]),
                "contrastive_similarity": float(sim[j]),
                "mssd_aesd_semantic_score": float(mssd_aesd_score[j]),
                "legacy_semantic_bonus": float(legacy_sem[j]),
                "natural_duration_score": float(natural_score[j]),
                "duration_log_cost": float(dur_cost[j]),
                "stage_score": float(stage_score[j]),
                "event_quality": float(event_quality[j]),
                "aesd_boundary_risk": float(aesd_risk[j]),
                "source": str(sources[j]),
                "source_uid": str(source_uids[j]),
                "label": str(labels_arr[j]),
                "dance_key": str(dance_keys[j]),
                "event_family": str(families[j]),
                "aesd_event_semantic": str(aesd_semantics[j]),
                "motion_stage_role": str(stages[j]),
                "support_label": str(support[j]),
                "locomotion_label": str(locomotion[j]),
                "music_alignment_label": str(align_arr[j]),
            })
        reports.append({
            "slot": i,
            "start": slot.get("start"),
            "end": slot.get("end"),
            "duration": slot.get("duration"),
            "target_frames": slot.get("target_frames"),
            "slot_role": slot.get("role", slot.get("slot_role")),
            "slot_music_alignment_label": slot.get("music_alignment_label"),
            "slot_music_semantic_top_label": slot.get("music_semantic_top_label", slot.get("music_alignment_label")),
            "slot_music_semantic_probs": slot.get("music_semantic_probs", {}),
            "slot_preferred_dance_keys": slot.get("preferred_dance_keys", []),
            "mssd_audit": {k: slot.get(k) for k in ["usage", "is_final_schedule", "slot_source", "mssd_usage", "mssd_is_final_schedule", "mssd_slot_source", "mssd_router_ckpt", "mssd_planner_ckpt", "mssd_duration_model_ckpt", "mssd_raw_schedule_json"] if k in slot},
            "top_candidate": int(cand[0]) if cand else -1,
            "beam_best_score": float(beams[0][0]) if beams else float("nan"),
            "routing_policy": "Semantic Routing MSSD-AESD global Event-RAG: contrastive + MSSD-AESD semantic + natural duration + stage + quality + boundary/source/family costs",
            "routing_weights": {"contrastive": w_contrastive, "mssd_aesd": w_mssd_aesd, "legacy_semantic": w_legacy_sem, "duration": w_duration, "quality": w_quality, "stage": w_stage, "boundary_risk": w_boundary_risk},
            "candidate_preview": preview,
        })
    return beams[0][1], reports
# ===== Semantic Routing COMPLETE MSSD-AESD ROUTING PATCH END =====

















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
        stats.update(audit_motion_np(m, cfg))
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
            checkpoint_version = normalize_motion_checkpoint_version(
                checkpoint_meta.get("version", "")
            )
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
        str(slot.get("music_alignment_label", "")),
        str(slot.get("music_semantic_top_label", "")),
        str(slot.get("role", "")),
        str(slot.get("slot_role", "")),
        str(slot.get("energy_label", "")),
        str(slot.get("predicted_motion_event", "")),
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
            key = str(s.get("music_semantic_top_label", s.get("music_event", "unknown")))
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
        st.update(audit_motion_np(m, cfg))
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
    """Physics Stability diffusion proposal with consecutive robust early probes."""
    if torch is None or not ckpt_path or not Path(ckpt_path).exists():
        return _stage_guard_orig_apply_diffusion_model(snapshot, cond, sm_win, ckpt_path, cfg)
    core_strength = _stage_guard_env_float("MOTION_DIFFUSION_CORE_STRENGTH", 0.00)
    trans_strength = _stage_guard_env_float("MOTION_DIFFUSION_TRANSITION_STRENGTH", 0.25)
    noise_scale = _stage_guard_env_float("MOTION_DIFFUSION_REFERENCE_NOISE_SCALE", 0.01)
    ckpt = _stage_guard_torch_load(ckpt_path, map_location=cfg.device)
    assert_motion_checkpoint_contract(ckpt, cfg, ckpt_path, "motion_diffusion")
    if normalize_motion_checkpoint_version(ckpt.get("version", "")).startswith(
        "reference_tangent_motion_diffusion"
    ):
        return _stage_guard_orig_apply_diffusion_model(
            snapshot, cond, sm_win, ckpt_path, cfg
        )
    Tdiff = int(ckpt.get("diffusion_steps", cfg.diffusion_steps))
    model = DiffusionDenoiser(EDGE_DIM, 32).to(cfg.device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    betas, alphas, abar = make_beta_schedule(Tdiff, torch.device(cfg.device))
    retr_in, _ = enforce_edge151_contract_np(np.asarray(snapshot, dtype=np.float32), cfg, source_hint="physics_stability_diffusion_window_retrieval", derive_contact=True, project_rot=True)
    mask_in = np.asarray(sm_win, dtype=np.float32)
    if mask_in.ndim == 1:
        mask_in = mask_in[:, None]
    if mask_in.shape[0] != retr_in.shape[0]:
        mask_in = resample_motion_np(mask_in, retr_in.shape[0])
    # Multiple probe points reduce single-step false positives.
    probe_fracs = os.environ.get("PHYSICS_STABILITY_EARLY_ABORT_PROBE_FRACTIONS", "0.66,0.50,0.33")
    probe_ts = set()
    for part in probe_fracs.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            probe_ts.add(int(round(Tdiff * float(part))))
        except Exception:
            pass
    consecutive_needed = max(1, _physics_stability_env_int("PHYSICS_STABILITY_EARLY_ABORT_CONSECUTIVE_FATAL", 2))
    fatal_streak = 0
    with torch.no_grad():
        retr = torch.from_numpy(retr_in[None]).float().to(cfg.device)
        raw_mask = torch.from_numpy(mask_in[None].astype(np.float32)).float().to(cfg.device)
        mask = torch.clamp(float(core_strength) + (float(trans_strength) - float(core_strength)) * raw_mask, 0.0, 1.0)
        c = torch.from_numpy(cond[None].astype(np.float32)).float().to(cfg.device)
        x = retr + float(noise_scale) * torch.randn_like(retr) * (0.15 + 0.85 * mask)
        for ti in reversed(range(Tdiff)):
            t = torch.full((1,), ti, device=cfg.device, dtype=torch.long)
            eps = model(x, retr, c, raw_mask, t)
            beta = betas[ti]; alpha = alphas[ti]; ab = abar[ti]
            mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - ab).clamp_min(1e-6) * eps)
            if ti > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x) * 0.35
            else:
                x = mean
            x = retr * (1.0 - mask) + x * mask
            if ti in probe_ts:
                probe = x[0].detach().cpu().numpy().astype(np.float32)
                probe, _ = enforce_edge151_contract_np(probe, cfg, source_hint="physics_stability_diffusion_early_probe_raw", derive_contact=True, project_rot=True)
                # Important: do NOT call strict safe_residual before early KBO.
                # Bound channel residuals lightly without running final KBO.
                delta = probe - retr_in
                bounded_proposal = probe.copy().astype(np.float32)
                root_xz = _stage_guard_env_float("STAGE_GUARD_ROOT_XZ_DELTA_MAX_M", 0.05) * _physics_stability_env_float("PHYSICS_STABILITY_EARLY_ABORT_BOUND_RELAX", 2.0)
                root_y = _stage_guard_env_float("STAGE_GUARD_ROOT_Y_DELTA_MAX_M", 0.02) * _physics_stability_env_float("PHYSICS_STABILITY_EARLY_ABORT_BOUND_RELAX", 2.0)
                for idx, mx in [(ROOT_X_IDX, root_xz), (ROOT_Y_IDX, root_y), (ROOT_Z_IDX, root_xz)]:
                    bounded_proposal[:, idx] = retr_in[:, idx] + np.clip(delta[:, idx], -mx, mx)
                rotation_cap = _stage_guard_env_float(
                    "STAGE_GUARD_ROTATION_DELTA_MAX_RAD",
                    _stage_guard_env_float("STAGE_GUARD_ROT6D_DELTA_MAX", 0.12),
                ) * _physics_stability_env_float("PHYSICS_STABILITY_EARLY_ABORT_BOUND_RELAX", 2.0)
                bounded = blend_edge151_geodesic_np(
                    retr_in,
                    bounded_proposal,
                    np.clip(mask_in, 0.0, 1.0),
                    max_rotation_rad=rotation_cap,
                )
                bounded, _ = enforce_edge151_contract_np(bounded, cfg, source_hint="physics_stability_diffusion_early_probe_bounded_no_strict_kbo", derive_contact=True, project_rot=True)
                ok, reasons, detail = _physics_early_abort_oracle(bounded, retr_in, cfg, stage="diffusion_early_abort_probe", global_start=global_start)
                _EARLY_ABORT_TRACE.append({"ti": int(ti), "ok": bool(ok), "reasons": reasons, "detail": detail})
                # Only fatal barriers count toward abort; soft derivative-only
                # barriers are diagnostics in _physics_early_abort_oracle.
                fatal_now = bool(detail.get("fatal_barriers"))
                fatal_streak = fatal_streak + 1 if fatal_now else 0
                if fatal_streak >= consecutive_needed:
                    _append_stage_transaction_audit({
                        "mechanism": "early_abort",
                        "version": "physics_stability_derivative_safe_consecutive",
                        "stage": "diffusion",
                        "commit_state": "abort_to_ccd",
                        "barrier_violations": reasons,
                        "detail": detail,
                        "hard_negative": True,
                    })
                    raise RuntimeError("diffusion_early_abort_motion_43:" + ",".join(reasons))
        y = x[0].detach().cpu().numpy().astype(np.float32)
    y, _ = enforce_edge151_contract_np(y, cfg, source_hint="physics_stability_diffusion_window_output", derive_contact=True, project_rot=True)
    return y.astype(np.float32)


def _append_generation_runtime_metadata(args: argparse.Namespace) -> None:
    """Append all stability metadata in one report transaction."""

    out_path = Path(args.out)
    json_path = Path(args.json or str(out_path).replace(".npy", ".inbetween_report.json"))
    if not json_path.exists():
        return
    with open(json_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    report.setdefault("stage_reports", {})[
        "stage_guard_temporal_generative_transactions"
    ] = _STAGE_TRANSACTION_AUDIT
    report["stage_guard_tgt_kbo_summary"] = _summarize_stage_transactions(
        _STAGE_TRANSACTION_AUDIT
    )
    report["stage_guard_scientific_mechanism"] = {
        "name": "Stage-Anchored KBO-guided Temporal Generative Transactions",
        "problem": "long-horizon covariate shift and topological fragility",
        "mechanisms": [
            "Macroscopic Stage Anchoring",
            "Temporal Generative Transactions",
            "Kinematic Barrier Oracle",
            "Confidence-aware Cascaded Degradation",
            "Diffusion Early-Abort",
            "Hard-negative Audit Tokens",
        ],
    }
    report["energy_stability_stability_alignment"] = {
        "version": "energy_stability_lowpass_early_abort_dynamic_msa_kinetic_hn_dpo",
        "fixes": [
            "low-pass relaxed KBO for early-abort probes",
            "music-energy and root-velocity adaptive macroscopic stage anchoring",
            "kinetic-energy preserving HN-DPO fine-tuning tool",
        ],
        "mssd_stage_weight_meta": _energy_gate_jsonable(_STAGE_WEIGHT_METADATA),
        "early_abort_relax": float(
            _energy_gate_env_float("ENERGY_STABILITY_EARLY_ABORT_KBO_RELAX", 3.0)
        ),
        "early_abort_smooth_sigma": float(
            _energy_gate_env_float(
                "ENERGY_STABILITY_EARLY_ABORT_KBO_SMOOTH_SIGMA", 1.35
            )
        ),
    }
    report["stage_guard_scientific_mechanism"]["energy_stability_fixes"] = report[
        "energy_stability_stability_alignment"
    ]["fixes"]
    report["physics_stability_physics_consistent_stability"] = {
        "version": "physics_stability_derivative_safe_msa_velocity_preserving_kinetic_dpo",
        "fixes": [
            "early-abort uses low-pass robust derivative oracle",
            "derivative-only Tweedie jitter cannot abort by default",
            "multiple early probes require consecutive fatal low-frequency barriers",
            "stage anchoring preserves local velocity and releases leap/high-speed windows",
            "HN-DPO training uses kinetic and motion-density preservation",
        ],
        "early_abort_probe_trace_count": int(len(_EARLY_ABORT_TRACE)),
        "early_abort_probe_trace_preview": _physics_stability_jsonable(
            _EARLY_ABORT_TRACE[:20]
        ),
        "env": {
            "PHYSICS_STABILITY_EARLY_ABORT_LOWPASS_SIGMA": _physics_stability_env_float(
                "PHYSICS_STABILITY_EARLY_ABORT_LOWPASS_SIGMA", 2.25
            ),
            "PHYSICS_STABILITY_EARLY_ABORT_RELAX": _physics_stability_env_float(
                "PHYSICS_STABILITY_EARLY_ABORT_RELAX", 4.0
            ),
            "PHYSICS_STABILITY_EARLY_ABORT_CONSECUTIVE_FATAL": _physics_stability_env_int(
                "PHYSICS_STABILITY_EARLY_ABORT_CONSECUTIVE_FATAL", 2
            ),
            "PHYSICS_STABILITY_MSA_LEAP_SPEED_THRESH": _physics_stability_env_float(
                "PHYSICS_STABILITY_MSA_LEAP_SPEED_THRESH", 0.070
            ),
            "PHYSICS_STABILITY_MSA_MAX_CORRECTION_VEL_MPS": _physics_stability_env_float(
                "PHYSICS_STABILITY_MSA_MAX_CORRECTION_VEL_MPS", 0.18
            ),
        },
    }
    save_json(_physics_stability_jsonable(report), json_path)
    print(
        json.dumps(
            {
                "runtime_metadata_updated": str(json_path),
                "stage_transactions": report["stage_guard_tgt_kbo_summary"],
                "physics_stability": report[
                    "physics_stability_physics_consistent_stability"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def generate(args: argparse.Namespace) -> int:
    """Explicit whole-song generation orchestration with one public entrypoint."""

    global _EARLY_ABORT_TRACE
    _EARLY_ABORT_TRACE = []
    _reset_stage_transaction_audit()
    try:
        _load_stage_anchor_weights(getattr(args, "slots_json", None))
    except Exception as exc:
        print(
            f"[Energy Stability WARN] failed to load MSSD dynamic stage weights: {exc}",
            file=sys.stderr,
        )

    rc = int(_generate_motion_core(args))
    try:
        _append_generation_runtime_metadata(args)
    except Exception as exc:
        print(
            f"[Motion Generation WARN] failed to append runtime metadata: {exc}",
            file=sys.stderr,
        )
    return rc
# ===== Physics Stability PHYSICS-CONSISTENT STABILITY PATCH END =====


if __name__ == "__main__":
    raise SystemExit(main())
