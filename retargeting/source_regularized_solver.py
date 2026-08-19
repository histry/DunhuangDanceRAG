#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-relative temporal regularization wrapper for Chang-E retargeting.

This module deliberately leaves :mod:`retargeting.bvh_solver` unchanged as the
authoritative legacy/baseline implementation.  Production cache construction can
import this wrapper instead.  When the two source-relative bone-direction
regularization weights are zero (the default), the wrapped optimizer is
numerically equivalent to the baseline objective.  Targeted experiments can
enable the extra losses with environment variables:

    SOURCE_RETARGET_BONE_DIR_VEL_W
    SOURCE_RETARGET_BONE_DIR_ACC_W

The added losses compare *candidate* and *source* parent-relative unit-bone
direction derivatives on the strict common direct mapped-bone set.  They do not
smooth toward zero, so authentic high-frequency dance dynamics present in the
recorded source remain a valid target.

Final-generation SI jerk/support/penetration/SO(3) contracts are not imported or
modified here.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

import retargeting.bvh_solver as _legacy

# Re-export the API consumed by retargeting.build_cache and research utilities.
BVHJoint = _legacy.BVHJoint
BVHMotion = _legacy.BVHMotion
RetargetConfig = _legacy.RetargetConfig
TARGET_JOINT_WEIGHTS = _legacy.TARGET_JOINT_WEIGHTS
TARGET_ALIASES = _legacy.TARGET_ALIASES
CHANGE_SIMPLIFIED_PROFILE = _legacy.CHANGE_SIMPLIFIED_PROFILE
parse_bvh = _legacy.parse_bvh
source_fk = _legacy.source_fk
target_rest_positions = _legacy.target_rest_positions
build_joint_mapping = _legacy.build_joint_mapping
similarity_umeyama = _legacy.similarity_umeyama
apply_similarity = _legacy.apply_similarity
resample_global_positions = _legacy.resample_global_positions
stabilize_source_heading_positions = _legacy.stabilize_source_heading_positions

torch = _legacy.torch
F = _legacy.F
NUM_JOINTS = _legacy.NUM_JOINTS
PARENTS = _legacy.PARENTS
FOOT_JOINTS = _legacy.FOOT_JOINTS


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


@dataclass(frozen=True)
class SourceTemporalRegularization:
    """Optional source-relative temporal matching used only during retargeting.

    Defaults are intentionally zero until the two-source A/B/C ablation has
    selected a validated operating point.  This makes merging the mechanism into
    ``main`` safe: cache behavior remains baseline-equivalent unless an
    experiment explicitly opts in.
    """

    bone_direction_velocity_weight: float = 0.0
    bone_direction_acceleration_weight: float = 0.0
    velocity_beta: float = 0.02
    acceleration_beta: float = 0.01
    normalized_reference_fps: float = 30.0

    @classmethod
    def from_environment(cls) -> "SourceTemporalRegularization":
        return cls(
            bone_direction_velocity_weight=_env_float(
                "SOURCE_RETARGET_BONE_DIR_VEL_W", 0.0
            ),
            bone_direction_acceleration_weight=_env_float(
                "SOURCE_RETARGET_BONE_DIR_ACC_W", 0.0
            ),
            velocity_beta=max(
                1.0e-8,
                _env_float("SOURCE_RETARGET_BONE_DIR_VEL_BETA", 0.02),
            ),
            acceleration_beta=max(
                1.0e-8,
                _env_float("SOURCE_RETARGET_BONE_DIR_ACC_BETA", 0.01),
            ),
            normalized_reference_fps=max(
                1.0e-6,
                _env_float("SOURCE_RETARGET_BONE_DIR_REFERENCE_FPS", 30.0),
            ),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.bone_direction_velocity_weight > 0.0
            or self.bone_direction_acceleration_weight > 0.0
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["enabled"] = bool(self.enabled)
        return payload


def common_direct_mapped_bone_children(
    bvh: BVHMotion,
    mapping: Dict[int, int],
) -> Tuple[int, ...]:
    """Return strict target child indices comparable to the source hierarchy.

    A target bone participates only when child and parent are both directly
    mapped, do not share one source proxy, and the source child's direct parent
    is exactly the mapped source parent.  This is the same topology principle as
    the source physical gate and excludes virtual/interpolated target joints.
    """

    children: list[int] = []
    for child in range(1, int(NUM_JOINTS)):
        parent = int(PARENTS[child])
        if child not in mapping or parent not in mapping:
            continue
        src_child = int(mapping[child])
        src_parent = int(mapping[parent])
        if src_child == src_parent:
            continue
        if src_child < 0 or src_child >= len(bvh.joints):
            continue
        if src_parent < 0 or src_parent >= len(bvh.joints):
            continue
        if int(bvh.joints[src_child].parent) != src_parent:
            continue
        children.append(int(child))
    return tuple(children)


@dataclass(frozen=True)
class _FitContext:
    settings: SourceTemporalRegularization
    common_bone_children: Tuple[int, ...]


_ACTIVE_CONTEXT: ContextVar[Optional[_FitContext]] = ContextVar(
    "source_temporal_regularization_context",
    default=None,
)
_PATCH_LOCK = threading.RLock()


def _bone_direction_derivative_losses(
    candidate_joints,
    source_target_positions,
    *,
    children: Iterable[int],
    target_fps: float,
    settings: SourceTemporalRegularization,
):
    """Return source-relative direction velocity/acceleration matching losses."""

    child_ids = tuple(int(v) for v in children)
    zero = candidate_joints.new_zeros(())
    if not child_ids or len(candidate_joints) < 2:
        return zero, zero

    child = torch.as_tensor(
        child_ids,
        dtype=torch.long,
        device=candidate_joints.device,
    )
    parent = torch.as_tensor(
        [int(PARENTS[idx]) for idx in child_ids],
        dtype=torch.long,
        device=candidate_joints.device,
    )

    source_vec = (
        source_target_positions[:, child]
        - source_target_positions[:, parent]
    )
    candidate_vec = candidate_joints[:, child] - candidate_joints[:, parent]

    source_dir = F.normalize(source_vec, dim=-1, eps=1.0e-6)
    candidate_dir = F.normalize(candidate_vec, dim=-1, eps=1.0e-6)

    # Normalize discrete derivatives to a 30-fps-equivalent timebase.  For the
    # same physical trajectory, first differences scale approximately as 1/fps
    # and second differences as 1/fps^2.
    rate = float(target_fps) / float(settings.normalized_reference_fps)

    source_d1 = (source_dir[1:] - source_dir[:-1]) * rate
    candidate_d1 = (candidate_dir[1:] - candidate_dir[:-1]) * rate
    vel = F.smooth_l1_loss(
        candidate_d1,
        source_d1,
        beta=float(settings.velocity_beta),
    )

    if len(candidate_dir) < 3:
        return vel, zero

    source_d2 = (
        source_dir[2:] - 2.0 * source_dir[1:-1] + source_dir[:-2]
    ) * (rate * rate)
    candidate_d2 = (
        candidate_dir[2:]
        - 2.0 * candidate_dir[1:-1]
        + candidate_dir[:-2]
    ) * (rate * rate)
    acc = F.smooth_l1_loss(
        candidate_d2,
        source_d2,
        beta=float(settings.acceleration_beta),
    )
    return vel, acc


def _fit_chunk_source_regularized(
    source_target_pos: np.ndarray,
    source_mask: np.ndarray,
    init_root: np.ndarray,
    init_rot6d: np.ndarray,
    floor_y: float,
    cfg: RetargetConfig,
):
    """Baseline chunk optimizer plus optional source-relative bone dynamics."""

    context = _ACTIVE_CONTEXT.get()
    settings = (
        context.settings if context is not None
        else SourceTemporalRegularization()
    )
    common_bones = (
        context.common_bone_children if context is not None else ()
    )

    device = torch.device(
        cfg.device
        if (cfg.device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )
    target = torch.as_tensor(
        source_target_pos,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.as_tensor(
        source_mask,
        dtype=torch.float32,
        device=device,
    )
    weights = torch.as_tensor(
        TARGET_JOINT_WEIGHTS,
        dtype=torch.float32,
        device=device,
    )
    w = mask * weights.view(1, -1)

    root = torch.tensor(
        init_root,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    rot = torch.tensor(
        init_rot6d,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    init_rot = torch.tensor(
        init_rot6d,
        dtype=torch.float32,
        device=device,
    )
    reference_root_rot6d = _legacy._project6d_torch(init_rot[:, 0]).detach()
    source_root = target[:, 0].detach()
    floor = torch.tensor(
        float(floor_y),
        dtype=torch.float32,
        device=device,
    )

    opt = torch.optim.Adam(
        [root, rot],
        lr=float(cfg.learning_rate),
    )
    last: dict[str, float] = {}

    for _ in range(int(cfg.iterations)):
        rp = _legacy._project6d_torch(rot)
        if cfg.root_orientation_lock:
            rp = torch.cat(
                [
                    reference_root_rot6d[:, None, :],
                    rp[:, 1:],
                ],
                dim=1,
            )

        joints = _legacy._fk_target_torch(root, rp)

        diff = F.smooth_l1_loss(
            joints,
            target,
            reduction="none",
            beta=0.03,
        ).sum(dim=-1)
        key = (diff * w).sum() / w.sum().clamp_min(1.0)

        root_loss = F.smooth_l1_loss(
            root,
            source_root,
            beta=0.03,
        )

        if len(root) > 1:
            root_vel = F.smooth_l1_loss(
                root[1:] - root[:-1],
                source_root[1:] - source_root[:-1],
                beta=0.02,
            )
            # Keep the historical Rot6D regularizers unchanged.  The new
            # source-relative loss is additive so an ablation remains possible.
            rot_vel = (rp[1:] - rp[:-1]).pow(2).mean()
        else:
            root_vel = root.new_zeros(())
            rot_vel = root.new_zeros(())

        if len(root) > 2:
            rot_acc = (
                rp[2:] - 2.0 * rp[1:-1] + rp[:-2]
            ).pow(2).mean()
        else:
            rot_acc = root.new_zeros(())

        pose_prior = (
            rp[:, 1:] - init_rot[:, 1:]
        ).pow(2).mean()

        pelvis = joints[:, 0]
        head = joints[:, 15]
        feet = joints[:, list(FOOT_JOINTS)]
        torso_cos = F.normalize(
            head - pelvis,
            dim=-1,
            eps=1.0e-8,
        )[:, 1]
        upright = F.relu(
            0.45 - torso_cos
        ).pow(2).mean()
        head_order = F.relu(
            0.18 - (head[:, 1] - pelvis[:, 1])
        ).pow(2).mean()
        feet_order = F.relu(
            0.30 - (
                pelvis[:, 1] - feet[..., 1].mean(dim=1)
            )
        ).pow(2).mean()
        penetration = F.relu(
            floor + 0.004 - feet[..., 1]
        ).pow(2).mean()

        if settings.enabled and common_bones:
            bone_dir_vel, bone_dir_acc = (
                _bone_direction_derivative_losses(
                    joints,
                    target,
                    children=common_bones,
                    target_fps=float(cfg.target_fps),
                    settings=settings,
                )
            )
        else:
            bone_dir_vel = root.new_zeros(())
            bone_dir_acc = root.new_zeros(())

        loss = (
            cfg.keypoint_weight * key
            + cfg.root_weight * root_loss
            + cfg.root_velocity_weight * root_vel
            + cfg.temporal_velocity_weight * rot_vel
            + cfg.temporal_acceleration_weight * rot_acc
            + cfg.pose_prior_weight * pose_prior
            + cfg.upright_weight * upright
            + cfg.head_order_weight * head_order
            + cfg.feet_order_weight * feet_order
            + cfg.floor_weight * penetration
            + float(settings.bone_direction_velocity_weight) * bone_dir_vel
            + float(settings.bone_direction_acceleration_weight) * bone_dir_acc
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [root, rot],
            float(cfg.gradient_clip),
        )
        opt.step()

        last = {
            "loss": float(loss.detach().cpu()),
            "key": float(key.detach().cpu()),
            "upright": float(upright.detach().cpu()),
            "penetration": float(penetration.detach().cpu()),
            "bone_direction_velocity_match": float(
                bone_dir_vel.detach().cpu()
            ),
            "bone_direction_acceleration_match": float(
                bone_dir_acc.detach().cpu()
            ),
            "bone_direction_common_bone_count": int(len(common_bones)),
            "bone_direction_velocity_weight": float(
                settings.bone_direction_velocity_weight
            ),
            "bone_direction_acceleration_weight": float(
                settings.bone_direction_acceleration_weight
            ),
        }

    with torch.no_grad():
        final_rot_t = _legacy._project6d_torch(rot)
        if cfg.root_orientation_lock:
            final_rot_t = torch.cat(
                [
                    reference_root_rot6d[:, None, :],
                    final_rot_t[:, 1:],
                ],
                dim=1,
            )
        final_rot = final_rot_t.cpu().numpy().astype(np.float32)
        final_root = root.cpu().numpy().astype(np.float32)

    return final_root, final_rot, last


def retarget_bvh(
    path: str | Path,
    cfg: Optional[RetargetConfig] = None,
):
    """Retarget with optional source-relative temporal direction matching."""

    cfg = cfg or RetargetConfig.from_env()
    settings = SourceTemporalRegularization.from_environment()

    # Parse once here solely to establish the strict source/target comparison
    # bone set.  The baseline retargeter performs its own authoritative parse.
    bvh = parse_bvh(path)
    mapping = build_joint_mapping(bvh.joints)
    common_bones = common_direct_mapped_bone_children(bvh, mapping)
    context = _FitContext(
        settings=settings,
        common_bone_children=common_bones,
    )

    # ``fit_target_motion`` resolves ``_fit_chunk`` from the baseline module's
    # globals.  Patch it only inside this locked call and restore it reliably.
    with _PATCH_LOCK:
        token = _ACTIVE_CONTEXT.set(context)
        original = _legacy._fit_chunk
        _legacy._fit_chunk = _fit_chunk_source_regularized
        try:
            motion, report = _legacy.retarget_bvh(path, cfg)
        finally:
            _legacy._fit_chunk = original
            _ACTIVE_CONTEXT.reset(token)

    report = dict(report)
    report["source_temporal_regularization"] = {
        **settings.to_dict(),
        "common_direct_mapped_bone_children": list(common_bones),
        "common_direct_mapped_bone_count": int(len(common_bones)),
        "activation_contract": (
            "explicit_environment_opt_in"
            if not settings.enabled
            else "enabled_by_environment"
        ),
    }
    return motion, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow_failed_contract", action="store_true")
    args = ap.parse_args(argv)

    cfg = RetargetConfig.from_env()
    if args.device:
        cfg.device = args.device
    if args.allow_failed_contract:
        cfg.hard_gravity_gate = False

    motion, report = retarget_bvh(args.input, cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, motion)
    rp = (
        Path(args.report)
        if args.report
        else out.with_suffix(".retarget.json")
    )
    rp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "motion": str(out),
                "report": str(rp),
                "frames": int(len(motion)),
                "ok": report["ok"],
                "fit_rmse_p95_m": report["fit"]["fit_rmse_p95_m"],
                "gravity": report["gravity"],
                "source_temporal_regularization": report[
                    "source_temporal_regularization"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
