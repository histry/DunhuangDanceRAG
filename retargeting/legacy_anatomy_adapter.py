#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anatomy-constrained Chang-E/SMPL to EDGE151 retargeting for V46.52.

The module reuses the repository's validated BVH parser, source FK, semantic
mapping and heading stabilization, but replaces the under-constrained chunk
optimizer with anatomy-aware losses.  Official SMPL pose files are preferred
when available because the Chang-E paper provides fitted SMPL parameters.
"""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import median_filter
except Exception:  # pragma: no cover
    median_filter = None

try:
    from scipy.spatial.transform import Rotation, Slerp
except Exception:  # pragma: no cover
    Rotation = None
    Slerp = None

import retargeting.bvh_solver as legacy
from contracts.gravity import (
    EDGE_DIM,
    FOOT_JOINTS,
    GravityThresholds,
    evaluate_gravity_contract,
    fk24_np,
    gravity_metrics_np,
    matrix_to_rot6d_np,
)
from contracts.anatomy import (
    AnatomyLossWeights,
    AnatomyThresholds,
    anatomy_losses_torch,
    anatomy_metrics_np,
    env_bool,
    env_float,
    env_int,
    evaluate_anatomy_contract,
)

CONTACT = slice(0, 4)
ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX = 4, 5, 6
NUM_JOINTS = 24


def _fit_chunk_anatomy(
    source_target_pos: np.ndarray,
    source_mask: np.ndarray,
    init_root: np.ndarray,
    init_rot6d: np.ndarray,
    floor_y: float,
    cfg: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Drop-in replacement for legacy._fit_chunk with anatomy constraints."""
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    target = torch.as_tensor(source_target_pos, dtype=torch.float32, device=device)
    mask = torch.as_tensor(source_mask, dtype=torch.float32, device=device)
    weights = torch.as_tensor(legacy.TARGET_JOINT_WEIGHTS, dtype=torch.float32, device=device)
    weighted_mask = mask * weights.view(1, -1)

    root = torch.tensor(init_root, dtype=torch.float32, device=device, requires_grad=True)
    rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device, requires_grad=True)
    init_rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device)
    reference_root_rot6d = legacy._project6d_torch(init_rot[:, 0]).detach()
    source_root = target[:, 0].detach()
    floor = torch.tensor(float(floor_y), dtype=torch.float32, device=device)
    anatomy_w = AnatomyLossWeights.from_env()

    iters = env_int("V46_52_RETARGET_ITERS", max(160, int(getattr(cfg, "iterations", 90))))
    lr = env_float("V46_52_RETARGET_LR", min(float(getattr(cfg, "learning_rate", 0.035)), 0.025))
    optimizer = torch.optim.Adam([root, rot], lr=lr)
    last: Dict[str, float] = {}

    for step in range(iters):
        projected = legacy._project6d_torch(rot)
        if bool(getattr(cfg, "root_orientation_lock", True)):
            projected = torch.cat([reference_root_rot6d[:, None], projected[:, 1:]], dim=1)
        local_mats = legacy._rot6d_to_matrix_torch(projected)
        joints = legacy._fk_target_torch(root, projected)

        diff = F.smooth_l1_loss(joints, target, reduction="none", beta=0.025).sum(dim=-1)
        key = (diff * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)
        root_loss = F.smooth_l1_loss(root, source_root, beta=0.025)

        if len(root) > 1:
            root_vel = F.smooth_l1_loss(
                root[1:] - root[:-1],
                source_root[1:] - source_root[:-1],
                beta=0.015,
            )
            rot_vel = (projected[1:] - projected[:-1]).square().mean()
        else:
            root_vel = root.new_zeros(())
            rot_vel = root.new_zeros(())
        if len(root) > 2:
            rot_acc = (projected[2:] - 2 * projected[1:-1] + projected[:-2]).square().mean()
        else:
            rot_acc = root.new_zeros(())

        # Stronger reference prior than V46.49: enough to stop optimization from
        # solving sparse keypoints with implausible local rotations.
        pose_prior = (projected[:, 1:] - init_rot[:, 1:]).square().mean()

        pelvis = joints[:, 0]
        head = joints[:, 15]
        feet = joints[:, list(FOOT_JOINTS)]
        torso_cos = F.normalize(head - pelvis, dim=-1, eps=1e-8)[:, 1]
        upright = F.relu(0.52 - torso_cos).square().mean()
        head_order = F.relu(0.22 - (head[:, 1] - pelvis[:, 1])).square().mean()
        feet_order = F.relu(0.28 - (pelvis[:, 1] - feet[..., 1].mean(dim=1))).square().mean()
        penetration = F.relu(floor + 0.003 - feet[..., 1]).square().mean()

        anatomy = anatomy_losses_torch(joints, local_mats, anatomy_w)

        # A short warm-up prioritizes coarse keypoint placement, then anatomy
        # losses reach full strength.  This avoids poor local minima at frame 0.
        warm = min(1.0, (step + 1) / max(1.0, 0.20 * iters))
        key_w = env_float("V46_52_KEYPOINT_W", 24.0)
        pose_w = env_float("V46_52_POSE_PRIOR_W", 0.075)
        loss = (
            key_w * key
            + env_float("V46_52_ROOT_W", 8.0) * root_loss
            + env_float("V46_52_ROOT_VEL_W", 1.8) * root_vel
            + env_float("V46_52_ROT_VEL_W", 0.18) * rot_vel
            + env_float("V46_52_ROT_ACC_W", 0.055) * rot_acc
            + pose_w * pose_prior
            + env_float("V46_52_UPRIGHT_W", 7.0) * upright
            + env_float("V46_52_HEAD_ORDER_W", 3.0) * head_order
            + env_float("V46_52_FEET_ORDER_W", 2.0) * feet_order
            + env_float("V46_52_FLOOR_W", 8.0) * penetration
            + warm * anatomy["total"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([root, rot], env_float("V46_52_RETARGET_GRAD_CLIP", 1.5))
        optimizer.step()

        progress_every = max(1, env_int("V46_52_RETARGET_PROGRESS_EVERY", 25))
        should_report = step == 0 or step == iters - 1 or (step + 1) % progress_every == 0
        if should_report:
            last = {
                "loss": float(loss.detach().cpu()),
                "key": float(key.detach().cpu()),
                "upright": float(upright.detach().cpu()),
                "penetration": float(penetration.detach().cpu()),
                "anatomy_total": float(anatomy["total"].detach().cpu()),
                "anatomy_local_limit": float(anatomy["local_limit"].detach().cpu()),
                "anatomy_spine": float(anatomy["spine"].detach().cpu()),
                "anatomy_torso": float(anatomy["torso"].detach().cpu()),
                "anatomy_collision": float(anatomy["collision"].detach().cpu()),
                "iterations": int(iters),
                "learning_rate": float(lr),
            }
            if env_bool("V46_52_RETARGET_PROGRESS", True):
                print(
                    "[V46.52 FIT] "
                    f"device={device} frames={len(root)} "
                    f"step={step + 1}/{iters} "
                    f"loss={last['loss']:.6f} key={last['key']:.6f} "
                    f"anatomy={last['anatomy_total']:.6f} "
                    f"floor_loss={last['penetration']:.6f}",
                    flush=True,
                )

    with torch.no_grad():
        final_rot_t = legacy._project6d_torch(rot)
        if bool(getattr(cfg, "root_orientation_lock", True)):
            final_rot_t = torch.cat([reference_root_rot6d[:, None], final_rot_t[:, 1:]], dim=1)
        final_rot = final_rot_t.cpu().numpy().astype(np.float32)
        final_root = root.cpu().numpy().astype(np.float32)
    return final_root, final_rot, last


def _strict_contract(motion: np.ndarray, legacy_report: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    gravity = gravity_metrics_np(motion, 30.0)
    gravity_th = GravityThresholds(
        torso_up_cos_p05_min=env_float("V46_52_GRAVITY_TORSO_P05_MIN", 0.55),
        torso_up_cos_median_min=env_float("V46_52_GRAVITY_TORSO_MEDIAN_MIN", 0.76),
        head_above_pelvis_ratio_min=env_float("V46_52_HEAD_ABOVE_RATIO_MIN", 0.97),
        feet_below_pelvis_ratio_min=env_float("V46_52_FEET_BELOW_RATIO_MIN", 0.94),
        horizontal_body_ratio_max=env_float("V46_52_HORIZONTAL_BODY_RATIO_MAX", 0.04),
    )
    gravity_ok, gravity_reasons = evaluate_gravity_contract(gravity, gravity_th)
    anatomy = anatomy_metrics_np(motion, fps=30.0)
    anatomy_th = AnatomyThresholds.from_env()
    anatomy_ok, anatomy_reasons = evaluate_anatomy_contract(anatomy, anatomy_th)
    fit_p95 = float(legacy_report.get("fit", {}).get("fit_rmse_p95_m", 1e9))
    fit_limit = env_float("V46_52_FIT_RMSE_P95_MAX_M", 0.12)
    fit_ok = np.isfinite(fit_p95) and fit_p95 <= fit_limit
    reasons = []
    if not gravity_ok:
        reasons.extend(["gravity:" + r for r in gravity_reasons])
    if not anatomy_ok:
        reasons.extend(["anatomy:" + r for r in anatomy_reasons])
    if not fit_ok:
        reasons.append(f"fit_rmse_p95_m={fit_p95:.6g} > {fit_limit:.6g}")
    report = {
        "schema": "v46_52_strict_motion_contract",
        "ok": bool(gravity_ok and anatomy_ok and fit_ok),
        "reasons": reasons,
        "gravity": gravity,
        "gravity_thresholds": gravity_th.to_dict(),
        "gravity_ok": bool(gravity_ok),
        "anatomy": anatomy,
        "anatomy_thresholds": anatomy_th.to_dict(),
        "anatomy_ok": bool(anatomy_ok),
        "fit_rmse_p95_m": fit_p95,
        "fit_rmse_p95_max_m": fit_limit,
        "fit_ok": bool(fit_ok),
    }
    return bool(report["ok"]), report


def retarget_bvh_anatomy(path: str | Path, cfg: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg = copy.deepcopy(cfg or legacy.RetargetConfig.from_env())
    cfg.iterations = env_int("V46_52_RETARGET_ITERS", max(160, int(cfg.iterations)))
    cfg.learning_rate = env_float("V46_52_RETARGET_LR", min(float(cfg.learning_rate), 0.025))
    cfg.fit_rmse_p95_max_m = env_float("V46_52_FIT_RMSE_P95_MAX_M", 0.12)
    cfg.hard_gravity_gate = False  # V46.52 evaluates a stronger combined contract.

    original = legacy._fit_chunk
    legacy._fit_chunk = _fit_chunk_anatomy
    try:
        motion, report = legacy.retarget_bvh(path, cfg)
    finally:
        legacy._fit_chunk = original

    ok, contract = _strict_contract(motion, report)
    report = dict(report)
    report["version"] = "v46_52_anatomy_constrained_bvh_retarget"
    report["v46_52_contract"] = contract
    report["anatomy"] = contract["anatomy"]
    report["anatomy_ok"] = contract["anatomy_ok"]
    report["fit_ok"] = contract["fit_ok"]
    report["gravity_ok"] = contract["gravity_ok"]
    report["ok"] = bool(ok)
    if env_bool("V46_52_HARD_RETARGET_GATE", True) and not ok:
        raise RuntimeError(
            f"V46.52 retarget contract failed for {path}: " + " | ".join(contract["reasons"])
        )
    return motion.astype(np.float32), report


def _load_pickle_or_npz(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".npz":
        obj = np.load(path, allow_pickle=True)
        return {k: obj[k] for k in obj.files}
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Expected dict-like SMPL file: {path}")


def _first_key(data: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in data:
            return data[k]
    return None


def _resample_smpl(rotvec: np.ndarray, trans: np.ndarray, src_fps: float, dst_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    if abs(float(src_fps) - float(dst_fps)) < 1e-5:
        return rotvec.astype(np.float32), trans.astype(np.float32)
    duration = (len(rotvec) - 1) / max(float(src_fps), 1e-8)
    n = max(2, int(round(duration * float(dst_fps))) + 1)
    old_t = np.arange(len(rotvec), dtype=np.float64) / float(src_fps)
    new_t = np.minimum(np.arange(n, dtype=np.float64) / float(dst_fps), old_t[-1])
    trans_out = np.stack([np.interp(new_t, old_t, trans[:, d]) for d in range(3)], axis=-1).astype(np.float32)
    if Rotation is None or Slerp is None:
        flat = rotvec.reshape(len(rotvec), -1)
        rv = np.stack([np.interp(new_t, old_t, flat[:, d]) for d in range(flat.shape[1])], axis=-1)
        return rv.reshape(n, NUM_JOINTS, 3).astype(np.float32), trans_out
    out = np.empty((n, NUM_JOINTS, 3), dtype=np.float32)
    for j in range(NUM_JOINTS):
        r = Rotation.from_rotvec(rotvec[:, j])
        out[:, j] = Slerp(old_t, r)(new_t).as_rotvec().astype(np.float32)
    return out, trans_out


def load_official_smpl_motion(path: str | Path, target_fps: float = 30.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load official fitted SMPL pose/trans parameters directly into EDGE151D."""
    p = Path(path)
    data = _load_pickle_or_npz(p)
    poses = _first_key(data, ("poses", "pose", "smpl_pose", "body_pose", "full_pose"))
    trans = _first_key(data, ("trans", "transl", "translations", "root_translation", "root_trans"))
    if poses is None:
        raise ValueError(f"No SMPL pose key found in {p}; keys={sorted(data.keys())}")
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 2 and poses.shape[1] >= 72:
        rotvec = poses[:, :72].reshape(len(poses), NUM_JOINTS, 3)
    elif poses.ndim == 3 and poses.shape[1:] == (NUM_JOINTS, 3):
        rotvec = poses
    else:
        raise ValueError(f"Unsupported SMPL pose shape in {p}: {poses.shape}")
    if trans is None:
        trans_arr = np.zeros((len(rotvec), 3), dtype=np.float32)
    else:
        trans_arr = np.asarray(trans, dtype=np.float32).reshape(len(rotvec), 3)
    fps_value = _first_key(data, ("mocap_framerate", "fps", "frame_rate", "framerate"))
    src_fps = float(np.asarray(fps_value).reshape(-1)[0]) if fps_value is not None else 30.0
    rotvec, trans_arr = _resample_smpl(rotvec, trans_arr, src_fps, target_fps)

    if Rotation is not None:
        mats = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(len(rotvec), NUM_JOINTS, 3, 3).astype(np.float32)
    else:
        # Torch conversion is available in the project's training environment.
        from pytorch3d.transforms import axis_angle_to_matrix
        mats = axis_angle_to_matrix(torch.as_tensor(rotvec)).cpu().numpy().astype(np.float32)

    motion = np.zeros((len(rotvec), EDGE_DIM), dtype=np.float32)
    motion[:, 4:7] = trans_arr
    motion[:, 7:151] = matrix_to_rot6d_np(mats).reshape(len(rotvec), -1)
    if env_bool("V46_52_LOCALIZE_ROOT_XZ", True):
        motion[:, ROOT_X_IDX] -= motion[0, ROOT_X_IDX]
        motion[:, ROOT_Z_IDX] -= motion[0, ROOT_Z_IDX]
    joints = fk24_np(motion)
    floor = float(np.percentile(joints[:, list(FOOT_JOINTS), 1], 5))
    motion[:, ROOT_Y_IDX] -= floor
    joints[..., 1] -= floor

    feet = joints[:, list(FOOT_JOINTS)]
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet) > 1:
        speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
    contact = (feet[..., 1] <= env_float("V46_52_CONTACT_HEIGHT_M", 0.055)) & (
        speed <= env_float("V46_52_CONTACT_SPEED_MPF", 0.025)
    )
    if median_filter is not None:
        contact = median_filter(contact.astype(np.uint8), size=(5, 1), mode="nearest").astype(bool)
    motion[:, CONTACT] = contact.astype(np.float32)

    base_report = {
        "source": str(p),
        "source_format": "official_smpl_parameters",
        "source_fps": src_fps,
        "target_fps": float(target_fps),
        "source_frames": int(len(poses)),
        "target_frames": int(len(motion)),
        "fit": {"fit_rmse_p95_m": 0.0, "direct_smpl": True},
    }
    ok, contract = _strict_contract(motion, base_report)
    report = {
        "version": "v46_52_direct_official_smpl",
        **base_report,
        "v46_52_contract": contract,
        "anatomy": contract["anatomy"],
        "anatomy_ok": contract["anatomy_ok"],
        "gravity": contract["gravity"],
        "gravity_ok": contract["gravity_ok"],
        "fit_ok": True,
        "ok": bool(ok),
    }
    if env_bool("V46_52_HARD_RETARGET_GATE", True) and not ok:
        raise RuntimeError(f"V46.52 direct SMPL contract failed for {p}: " + " | ".join(contract["reasons"]))
    return motion.astype(np.float32), report
