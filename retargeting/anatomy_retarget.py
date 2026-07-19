#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.53.1 research retargeter for Chang-E BVH -> EDGE151.

Key corrections:
- soft root-orientation anchoring instead of a complete root lock;
- SO(3) velocity/acceleration regularisation instead of Euclidean Rot6D differences;
- source-structure-guided upright/order targets;
- nonlinear loss scheduling;
- matrix-space overlap fusion followed by SVD projection;
- catastrophic source gate separated from event-level quality filtering.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import retargeting.bvh_solver as legacy
from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    project_to_so3_np,
    rot6d_to_matrix_np,
    so3_geodesic_torch,
)
from contracts.gravity import (
    EDGE_DIM,
    FOOT_JOINTS,
    GravityThresholds,
    evaluate_gravity_contract,
    fk24_np,
)
from contracts.anatomy import (
    AnatomyLossWeights,
    SourceAnatomyThresholds,
    anatomy_losses_torch,
    anatomy_metrics_np,
    env_bool,
    env_float,
    env_int,
    evaluate_source_anatomy_contract,
)

CONTACT = slice(0, 4)
ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX = 4, 5, 6
NUM_JOINTS = 24


def _so3_log_torch(r: torch.Tensor) -> torch.Tensor:
    """Stable differentiable SO(3) log vector for regularisation."""
    tr = torch.diagonal(r, dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos)
    vee = torch.stack(
        [
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ],
        dim=-1,
    )
    scale = theta / (2.0 * torch.sin(theta).abs().clamp_min(1e-5))
    regular = vee * scale.unsqueeze(-1)
    small = 0.5 * vee
    return torch.where((theta < 1e-4).unsqueeze(-1), small, regular)


def _relative_rotvec_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _so3_log_torch(a.transpose(-1, -2) @ b)


def _smoothstep01(x: float) -> float:
    y = max(0.0, min(1.0, float(x)))
    return y * y * (3.0 - 2.0 * y)


def _fit_chunk_research(
    source_target_pos: np.ndarray,
    source_mask: np.ndarray,
    init_root: np.ndarray,
    init_rot6d: np.ndarray,
    floor_y: float,
    cfg: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    target = torch.as_tensor(source_target_pos, dtype=torch.float32, device=device)
    mask = torch.as_tensor(source_mask, dtype=torch.float32, device=device)
    weights = torch.as_tensor(legacy.TARGET_JOINT_WEIGHTS, dtype=torch.float32, device=device)
    weighted_mask = mask * weights.view(1, -1)

    root = torch.tensor(init_root, dtype=torch.float32, device=device, requires_grad=True)
    rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device, requires_grad=True)
    init_rot = torch.tensor(init_rot6d, dtype=torch.float32, device=device)
    init_projected = legacy._project6d_torch(init_rot).detach()
    init_mats = legacy._rot6d_to_matrix_torch(init_projected).detach()
    source_root = target[:, 0].detach()
    floor = torch.tensor(float(floor_y), dtype=torch.float32, device=device)
    anatomy_w = AnatomyLossWeights.from_env()

    iters = env_int("V46_53_1_RETARGET_ITERS", env_int("V46_52_RETARGET_ITERS", 280))
    lr = env_float("V46_53_1_RETARGET_LR", env_float("V46_52_RETARGET_LR", 0.018))
    optimizer = torch.optim.AdamW([root, rot], lr=lr, weight_decay=0.0)

    # Preserve the source body structure rather than imposing a universal upright pose.
    target_pelvis = target[:, 0]
    target_head = target[:, 15]
    target_feet = target[:, list(FOOT_JOINTS)]
    target_torso_cos = F.normalize(target_head - target_pelvis, dim=-1, eps=1e-8)[:, 1].detach()
    target_head_margin = (target_head[:, 1] - target_pelvis[:, 1]).detach()
    target_feet_margin = (target_pelvis[:, 1] - target_feet[..., 1].mean(dim=1)).detach()

    last: Dict[str, float] = {}
    fps = float(getattr(cfg, "target_fps", 30.0))
    for step in range(iters):
        progress = (step + 1) / max(1.0, float(iters))
        projected = legacy._project6d_torch(rot)
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
            omega = _relative_rotvec_torch(local_mats[:-1], local_mats[1:]) * fps
            init_omega = _relative_rotvec_torch(init_mats[:-1], init_mats[1:]) * fps
            # Match the reliable root body-frame dynamics, regularise other joints.
            rot_vel = (
                F.smooth_l1_loss(omega[:, 0], init_omega[:, 0], beta=0.08)
                + 0.30 * omega[:, 1:].square().mean()
            )
        else:
            root_vel = root.new_zeros(())
            rot_vel = root.new_zeros(())
            omega = root.new_zeros((0, NUM_JOINTS, 3))

        if len(root) > 2:
            alpha = (omega[1:] - omega[:-1]) * fps
            rot_acc = alpha.square().mean()
        else:
            rot_acc = root.new_zeros(())

        root_delta = _relative_rotvec_torch(init_mats[:, 0], local_mats[:, 0])
        root_axis_w = torch.as_tensor(
            [1.0, env_float("V46_53_1_ROOT_YAW_ANCHOR_MULT", 4.0), 1.0],
            dtype=root_delta.dtype,
            device=root_delta.device,
        )
        root_anchor = (root_delta.square() * root_axis_w).mean()
        pose_prior = so3_geodesic_torch(local_mats[:, 1:], init_mats[:, 1:]).square().mean()

        pelvis = joints[:, 0]
        head = joints[:, 15]
        feet = joints[:, list(FOOT_JOINTS)]
        torso_cos = F.normalize(head - pelvis, dim=-1, eps=1e-8)[:, 1]
        source_margin = env_float("V46_53_1_UPRIGHT_SOURCE_MARGIN", 0.10)
        target_floor = (target_torso_cos - source_margin).clamp_min(0.10)
        upright = F.relu(target_floor - torso_cos).square().mean()

        head_floor = (target_head_margin - env_float("V46_53_1_HEAD_SOURCE_MARGIN_M", 0.06)).clamp_min(0.06)
        feet_floor = (target_feet_margin - env_float("V46_53_1_FEET_SOURCE_MARGIN_M", 0.08)).clamp_min(0.12)
        head_order = F.relu(head_floor - (head[:, 1] - pelvis[:, 1])).square().mean()
        feet_order = F.relu(feet_floor - (pelvis[:, 1] - feet[..., 1].mean(dim=1))).square().mean()
        penetration = F.relu(floor + 0.003 - feet[..., 1]).square().mean()

        anatomy = anatomy_losses_torch(joints, local_mats, anatomy_w)

        # Paper-inspired nonlinear schedule: structure priors dominate early,
        # anatomy constraints rise smoothly after coarse alignment.
        anatomy_scale = _smoothstep01((progress - 0.08) / 0.55)
        prior_scale = 0.20 + 0.80 * math.exp(-2.8 * progress)
        key_scale = 1.0 + 0.20 * math.exp(-3.0 * progress)

        loss = (
            key_scale * env_float("V46_52_KEYPOINT_W", 24.0) * key
            + env_float("V46_52_ROOT_W", 8.0) * root_loss
            + env_float("V46_52_ROOT_VEL_W", 1.8) * root_vel
            + env_float("V46_53_1_SO3_VEL_W", 0.025) * rot_vel
            + env_float("V46_53_1_SO3_ACC_W", 0.00008) * rot_acc
            + prior_scale * env_float("V46_52_POSE_PRIOR_W", 0.08) * pose_prior
            + env_float("V46_53_1_ROOT_ANCHOR_W", 0.35) * root_anchor
            + env_float("V46_52_UPRIGHT_W", 4.0) * upright
            + env_float("V46_52_HEAD_ORDER_W", 2.0) * head_order
            + env_float("V46_52_FEET_ORDER_W", 1.5) * feet_order
            + env_float("V46_52_FLOOR_W", 8.0) * penetration
            + anatomy_scale * anatomy["total"]
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([root, rot], env_float("V46_52_RETARGET_GRAD_CLIP", 1.5))
        optimizer.step()

        every = max(1, env_int("V46_52_RETARGET_PROGRESS_EVERY", 25))
        if step == 0 or step == iters - 1 or (step + 1) % every == 0:
            last = {
                "loss": float(loss.detach().cpu()),
                "key": float(key.detach().cpu()),
                "root_anchor": float(root_anchor.detach().cpu()),
                "so3_velocity": float(rot_vel.detach().cpu()),
                "so3_acceleration": float(rot_acc.detach().cpu()),
                "upright": float(upright.detach().cpu()),
                "penetration": float(penetration.detach().cpu()),
                "anatomy_total": float(anatomy["total"].detach().cpu()),
                "anatomy_local_limit": float(anatomy["local_limit"].detach().cpu()),
                "anatomy_spine": float(anatomy["spine"].detach().cpu()),
                "anatomy_torso": float(anatomy["torso"].detach().cpu()),
                "anatomy_collision": float(anatomy["collision"].detach().cpu()),
                "style_gate_mean": float(anatomy["style_gate_mean"].detach().cpu()),
                "iterations": int(iters),
                "learning_rate": float(lr),
            }
            if env_bool("V46_52_RETARGET_PROGRESS", True):
                print(
                    "[V46.53.1 FIT] "
                    f"device={device} frames={len(root)} step={step + 1}/{iters} "
                    f"loss={last['loss']:.6f} key={last['key']:.6f} "
                    f"so3v={last['so3_velocity']:.6f} anatomy={last['anatomy_total']:.6f} "
                    f"floor_loss={last['penetration']:.6f}",
                    flush=True,
                )

    with torch.no_grad():
        final_rot = legacy._project6d_torch(rot).cpu().numpy().astype(np.float32)
        final_root = root.cpu().numpy().astype(np.float32)
    return final_root, final_rot, last


def _fit_target_motion_research(
    aligned_source_positions: np.ndarray,
    mapping: Dict[int, int],
    cfg: Any,
) -> Tuple[np.ndarray, Dict[str, object]]:
    T = len(aligned_source_positions)
    target_pos = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    mask = np.zeros((T, NUM_JOINTS), dtype=np.float32)
    for tgt, src in mapping.items():
        target_pos[:, tgt] = aligned_source_positions[:, src]
        mask[:, tgt] = 1.0

    if 3 not in mapping and 0 in mapping and 6 in mapping:
        target_pos[:, 3] = 0.45 * target_pos[:, 0] + 0.55 * target_pos[:, 6]
        mask[:, 3] = 0.75
    if 6 not in mapping and 3 in mapping and 9 in mapping:
        target_pos[:, 6] = 0.50 * target_pos[:, 3] + 0.50 * target_pos[:, 9]
        mask[:, 6] = 0.75
    if 9 not in mapping and 6 in mapping and 12 in mapping:
        target_pos[:, 9] = 0.55 * target_pos[:, 6] + 0.45 * target_pos[:, 12]
        mask[:, 9] = 0.75

    init_root = target_pos[:, 0].copy()
    init_rot = legacy.identity6d_np((T, NUM_JOINTS))
    root_R = legacy.body_frame_from_keypoints(target_pos)
    init_rot[:, 0] = legacy.matrix_to_rot6d_np(root_R)

    source_foot_ids = [mapping[t] for t in FOOT_JOINTS if t in mapping]
    floor_y = float(np.percentile(aligned_source_positions[:, source_foot_ids, 1], 5)) if source_foot_ids else float(
        np.percentile(target_pos[:, [7, 8], 1], 5)
    )

    chunk = max(32, int(cfg.chunk_frames))
    overlap = max(0, min(int(cfg.chunk_overlap), chunk // 2))
    stride = max(1, chunk - overlap)
    accum_root = np.zeros((T, 3), dtype=np.float64)
    accum_mat = np.zeros((T, NUM_JOINTS, 3, 3), dtype=np.float64)
    weight_sum = np.zeros((T, 1), dtype=np.float64)
    chunk_reports = []

    for st in range(0, T, stride):
        ed = min(T, st + chunk)
        if ed - st < 4:
            continue
        r, q, rep = _fit_chunk_research(
            target_pos[st:ed], mask[st:ed], init_root[st:ed], init_rot[st:ed], floor_y, cfg
        )
        mats = rot6d_to_matrix_np(q)
        L = ed - st
        weight = np.ones((L, 1), dtype=np.float64)
        ov = min(overlap, L // 2)
        if ov > 1 and st > 0:
            weight[:ov, 0] = np.linspace(1e-3, 1.0, ov, dtype=np.float64)
        if ov > 1 and ed < T:
            weight[-ov:, 0] = np.linspace(1.0, 1e-3, ov, dtype=np.float64)
        accum_root[st:ed] += r.astype(np.float64) * weight
        accum_mat[st:ed] += mats.astype(np.float64) * weight[:, None, None]
        weight_sum[st:ed] += weight
        chunk_reports.append({"start": st, "end": ed, **rep})

    valid = weight_sum[:, 0] > 0
    if not np.all(valid):
        accum_root[~valid] = init_root[~valid]
        accum_mat[~valid] = rot6d_to_matrix_np(init_rot[~valid])
        weight_sum[~valid] = 1.0
    root = (accum_root / weight_sum).astype(np.float32)
    mats = project_to_so3_np(accum_mat / weight_sum[:, None, None])
    rot6d = matrix_to_rot6d_np(mats)

    motion = np.zeros((T, EDGE_DIM), dtype=np.float32)
    motion[:, 4:7] = root
    motion[:, 7:151] = rot6d.reshape(T, -1)

    if cfg.localize_root_xz:
        motion[:, ROOT_X_IDX] -= motion[0, ROOT_X_IDX]
        motion[:, ROOT_Z_IDX] -= motion[0, ROOT_Z_IDX]

    joints = fk24_np(motion)
    fitted_floor = float(np.percentile(joints[:, list(FOOT_JOINTS), 1], 5))
    if cfg.floor_to_zero:
        motion[:, ROOT_Y_IDX] -= fitted_floor
        joints[..., 1] -= fitted_floor
        fitted_floor = 0.0

    feet = joints[:, list(FOOT_JOINTS)]
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    if T > 1:
        speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
    height = feet[..., 1] - fitted_floor
    contacts = (height <= float(cfg.contact_height_m)) & (speed <= float(cfg.contact_speed_mpf))
    if legacy.median_filter is not None and cfg.contact_median_size > 1:
        contacts = legacy.median_filter(
            contacts.astype(np.uint8),
            size=(int(cfg.contact_median_size), 1),
            mode="nearest",
        ).astype(bool)
    motion[:, CONTACT] = contacts.astype(np.float32)

    target_eval = target_pos.copy()
    if cfg.localize_root_xz:
        target_eval[..., 0] -= target_pos[0, 0, 0]
        target_eval[..., 2] -= target_pos[0, 0, 2]
    if cfg.floor_to_zero:
        floor_ids = [t for t in FOOT_JOINTS if t in mapping]
        if floor_ids:
            target_eval[..., 1] -= float(np.percentile(target_eval[:, floor_ids, 1], 5))
    pred = fk24_np(motion)
    per_frame = []
    for t in range(T):
        ids = np.where(mask[t] > 0.5)[0]
        if len(ids):
            per_frame.append(float(np.sqrt(np.mean((pred[t, ids] - target_eval[t, ids]) ** 2))))
    fit_arr = np.asarray(per_frame, dtype=np.float32)

    return motion, {
        "floor_y_after": float(fitted_floor),
        "contact_ratio": float(contacts.mean()),
        "fit_rmse_mean_m": float(fit_arr.mean()) if fit_arr.size else 0.0,
        "fit_rmse_p95_m": float(np.percentile(fit_arr, 95)) if fit_arr.size else 0.0,
        "root_orientation_contract": {
            "version": "v46_53_1_soft_source_body_frame_contract",
            "mode": "soft_geodesic_anchor",
            "root_translation": "optimized",
            "root_orientation": "optimized_with_source_body_frame_prior",
            "local_joints_1_to_23": "optimized",
        },
        "overlap_fusion": "weighted_rotation_matrix_then_svd_so3_projection",
        "chunk_reports": chunk_reports,
    }


def retarget_bvh_research(path: str | Path, cfg: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    cfg = copy.deepcopy(cfg or legacy.RetargetConfig.from_env())
    cfg.iterations = env_int("V46_53_1_RETARGET_ITERS", env_int("V46_52_RETARGET_ITERS", 280))
    cfg.learning_rate = env_float("V46_53_1_RETARGET_LR", env_float("V46_52_RETARGET_LR", 0.018))
    cfg.fit_rmse_p95_max_m = env_float("V46_52_FIT_RMSE_P95_MAX_M", 0.14)
    cfg.root_orientation_lock = False
    cfg.hard_gravity_gate = False

    old_fit = legacy._fit_chunk
    old_target = legacy.fit_target_motion
    legacy._fit_chunk = _fit_chunk_research
    legacy.fit_target_motion = _fit_target_motion_research
    try:
        motion, legacy_report = legacy.retarget_bvh(path, cfg)
    finally:
        legacy._fit_chunk = old_fit
        legacy.fit_target_motion = old_target

    anatomy = anatomy_metrics_np(motion, fps=float(cfg.target_fps))
    source_th = SourceAnatomyThresholds.from_env()
    anatomy_ok, anatomy_reasons = evaluate_source_anatomy_contract(anatomy, source_th)

    from contracts.gravity import gravity_metrics_np
    gravity = gravity_metrics_np(motion, float(cfg.target_fps))
    gravity_th = GravityThresholds(
        torso_up_cos_p05_min=env_float("V46_52_SOURCE_GRAVITY_TORSO_P05_MIN", 0.30),
        torso_up_cos_median_min=env_float("V46_52_SOURCE_GRAVITY_TORSO_MEDIAN_MIN", 0.55),
        head_above_pelvis_ratio_min=env_float("V46_52_SOURCE_HEAD_ABOVE_RATIO_MIN", 0.85),
        feet_below_pelvis_ratio_min=env_float("V46_52_SOURCE_FEET_BELOW_RATIO_MIN", 0.85),
        horizontal_body_ratio_max=env_float("V46_52_SOURCE_HORIZONTAL_BODY_RATIO_MAX", 0.20),
    )
    gravity_ok, gravity_reasons = evaluate_gravity_contract(gravity, gravity_th)

    fit_p95 = float(legacy_report.get("fit", {}).get("fit_rmse_p95_m", 1e9))
    fit_limit = env_float("V46_52_FIT_RMSE_P95_MAX_M", 0.14)
    fit_ok = np.isfinite(fit_p95) and fit_p95 <= fit_limit

    reasons = []
    reasons.extend(["anatomy:" + r for r in anatomy_reasons])
    reasons.extend(["gravity:" + r for r in gravity_reasons])
    if not fit_ok:
        reasons.append(f"fit_rmse_p95_m={fit_p95:.6g} > {fit_limit:.6g}")

    report = dict(legacy_report)
    report.update(
        {
            "version": "v46_52_anatomy_constrained_bvh_retarget_v46_53_1",
            "schema": "v46_53_1_source_safety_retarget",
            "source": str(path),
            "anatomy": anatomy,
            "anatomy_thresholds": source_th.to_dict(),
            "anatomy_ok": bool(anatomy_ok),
            "gravity": gravity,
            "gravity_thresholds": gravity_th.to_dict(),
            "gravity_ok": bool(gravity_ok),
            "fit_ok": bool(fit_ok),
            "fit_rmse_p95_m": fit_p95,
            "fit_rmse_p95_max_m": fit_limit,
            "source_gate_ok": bool(anatomy_ok and gravity_ok and fit_ok),
            "source_gate_reasons": reasons,
            "event_gate_required": True,
            "ok": bool(anatomy_ok and gravity_ok and fit_ok),
        }
    )

    if env_bool("V46_52_HARD_RETARGET_GATE", True) and not report["ok"]:
        raise RuntimeError(
            f"V46.53.1 source safety contract failed for {path}: " + " | ".join(reasons)
        )
    return motion.astype(np.float32), report


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow_failed_contract", action="store_true")
    args = ap.parse_args()

    cfg = legacy.RetargetConfig.from_env()
    if args.device:
        cfg.device = args.device
    if args.allow_failed_contract:
        os.environ["V46_52_HARD_RETARGET_GATE"] = "0"

    motion, report = retarget_bvh_research(args.input, cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, motion)
    rp = Path(args.report) if args.report else out.with_suffix(".retarget.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"motion": str(out), "report": str(rp), "frames": len(motion), "ok": report["ok"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
