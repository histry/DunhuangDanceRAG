#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified SI physical-quality contracts for motion generation and repair.

The module centralizes three responsibilities that previously used different
statistics and thresholds in separate pipeline stages:

1. final whole-motion physical acceptance;
2. transactional acceptance of Refiner/Diffusion candidates;
3. frame-joint Peak-Jerk localization for local repair masks.

All jerk values use world-space FK positions and SI units (m/s^3). The hard
maximum is the true maximum over every frame-joint pair, not a frame-wise mean
across joints.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from contracts.gravity import fk24_np
from motion_geometry.smpl24 import NUM_JOINTS, PARENTS


def _env_float(primary: str, fallback: Optional[str], default: float) -> float:
    raw = os.environ.get(primary)
    if raw is None and fallback:
        raw = os.environ.get(fallback)
    try:
        return float(default if raw is None else raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(primary: str, fallback: Optional[str], default: int) -> int:
    raw = os.environ.get(primary)
    if raw is None and fallback:
        raw = os.environ.get(fallback)
    try:
        return int(float(default if raw is None else raw))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(primary: str, fallback: Optional[str], default: bool) -> bool:
    raw = os.environ.get(primary)
    if raw is None and fallback:
        raw = os.environ.get(fallback)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _metric(audit: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(audit.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


@dataclass(frozen=True)
class PhysicalQualityLimits:
    """Absolute whole-motion limits in SI units."""

    foot_skate_mps_p95: float = 0.18
    foot_skate_mps_max: float = 0.60
    foot_support_drift_m_p95: float = 0.06
    foot_support_drift_m_max: float = 0.12
    foot_contact_height_m_max: float = 0.10
    foot_penetration_min_m: float = -0.05
    joint_jerk_mps3_p95: float = 810.0
    joint_jerk_mps3_max: float = 1620.0
    joint_jerk_window_p95_max_mps3: float = 1080.0
    extremity_jerk_mps3_p95: float = 810.0
    extremity_jerk_window_p95_max_mps3: float = 1080.0
    root_y_robust_range_m: float = 0.90
    root_vertical_speed_mps_p95: float = 1.25
    root_vertical_speed_mps_max: float = 4.0
    root_horizontal_radius_p95_m: float = 1.80
    root_horizontal_radius_max_m: float = 2.20
    root_horizontal_net_displacement_m: float = 3.00
    root_horizontal_drift_speed_mps: float = 0.12
    root_horizontal_window_displacement_max_m: float = 1.50

    @classmethod
    def from_environment(cls) -> "PhysicalQualityLimits":
        return cls(
            foot_skate_mps_p95=_env_float(
                "PHYSICAL_MAX_FOOT_SKATE_P95_MPS",
                "V46_54_MAX_FOOT_SKATE_P95_MPS",
                0.18,
            ),
            foot_skate_mps_max=_env_float(
                "PHYSICAL_MAX_FOOT_SKATE_MAX_MPS",
                "V46_54_MAX_FOOT_SKATE_MAX_MPS",
                0.60,
            ),
            foot_support_drift_m_p95=_env_float(
                "PHYSICAL_MAX_FOOT_SUPPORT_DRIFT_P95_M",
                "V46_54_MAX_FOOT_SUPPORT_DRIFT_P95_M",
                0.06,
            ),
            foot_support_drift_m_max=_env_float(
                "PHYSICAL_MAX_FOOT_SUPPORT_DRIFT_MAX_M",
                "V46_54_MAX_FOOT_SUPPORT_DRIFT_MAX_M",
                0.12,
            ),
            foot_contact_height_m_max=_env_float(
                "PHYSICAL_MAX_FOOT_CONTACT_HEIGHT_M",
                "V46_54_MAX_FOOT_CONTACT_HEIGHT_M",
                0.10,
            ),
            foot_penetration_min_m=_env_float(
                "PHYSICAL_MIN_FOOT_PENETRATION_M",
                "V46_54_MIN_FOOT_PENETRATION_M",
                -0.05,
            ),
            joint_jerk_mps3_p95=_env_float(
                "PHYSICAL_MAX_JOINT_JERK_P95_MPS3",
                "V46_54_MAX_JOINT_JERK_P95_MPS3",
                810.0,
            ),
            joint_jerk_mps3_max=_env_float(
                "PHYSICAL_MAX_JOINT_JERK_MAX_MPS3",
                "V46_54_MAX_JOINT_JERK_MAX_MPS3",
                1620.0,
            ),
            joint_jerk_window_p95_max_mps3=_env_float(
                "PHYSICAL_MAX_JOINT_JERK_WINDOW_P95_MPS3",
                "V46_54_MAX_JOINT_JERK_WINDOW_P95_MPS3",
                1080.0,
            ),
            extremity_jerk_mps3_p95=_env_float(
                "PHYSICAL_MAX_EXTREMITY_JERK_P95_MPS3",
                "V46_54_MAX_EXTREMITY_JERK_P95_MPS3",
                810.0,
            ),
            extremity_jerk_window_p95_max_mps3=_env_float(
                "PHYSICAL_MAX_EXTREMITY_JERK_WINDOW_P95_MPS3",
                "V46_54_MAX_EXTREMITY_JERK_WINDOW_P95_MPS3",
                1080.0,
            ),
            root_y_robust_range_m=_env_float(
                "PHYSICAL_MAX_ROOT_Y_ROBUST_RANGE_M",
                "V46_54_MAX_ROOT_Y_ROBUST_RANGE_M",
                0.90,
            ),
            root_vertical_speed_mps_p95=_env_float(
                "PHYSICAL_MAX_ROOT_VERTICAL_SPEED_P95_MPS",
                "V46_54_MAX_ROOT_VERTICAL_SPEED_P95_MPS",
                1.25,
            ),
            root_vertical_speed_mps_max=_env_float(
                "PHYSICAL_MAX_ROOT_VERTICAL_SPEED_MAX_MPS",
                "V46_54_MAX_ROOT_VERTICAL_SPEED_MAX_MPS",
                4.0,
            ),
            root_horizontal_radius_p95_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_RADIUS_P95_M",
                "V46_54_MAX_ROOT_XZ_RADIUS_P95_M",
                1.80,
            ),
            root_horizontal_radius_max_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_RADIUS_MAX_M",
                "V46_54_MAX_ROOT_XZ_RADIUS_MAX_M",
                2.20,
            ),
            root_horizontal_net_displacement_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_NET_DISPLACEMENT_M",
                "V46_54_MAX_ROOT_XZ_NET_DISPLACEMENT_M",
                3.00,
            ),
            root_horizontal_drift_speed_mps=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_DRIFT_SPEED_MPS",
                "V46_54_MAX_ROOT_XZ_DRIFT_SPEED_MPS",
                0.12,
            ),
            root_horizontal_window_displacement_max_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_WINDOW_DISPLACEMENT_M",
                "V46_54_MAX_ROOT_XZ_WINDOW_DISPLACEMENT_M",
                1.50,
            ),
        )

    def as_audit_limits(self) -> Dict[str, float]:
        return {
            "foot_skate_mps_p95": float(self.foot_skate_mps_p95),
            "foot_skate_mps_max": float(self.foot_skate_mps_max),
            "foot_support_drift_m_p95": float(self.foot_support_drift_m_p95),
            "foot_support_drift_m_max": float(self.foot_support_drift_m_max),
            "foot_contact_height_m_max": float(self.foot_contact_height_m_max),
            "foot_penetration_min_m": float(self.foot_penetration_min_m),
            "joint_jerk_mps3_p95": float(self.joint_jerk_mps3_p95),
            "joint_jerk_mps3_max": float(self.joint_jerk_mps3_max),
            "joint_jerk_window_p95_max_mps3": float(
                self.joint_jerk_window_p95_max_mps3
            ),
            "extremity_jerk_mps3_p95": float(self.extremity_jerk_mps3_p95),
            "extremity_jerk_window_p95_max_mps3": float(
                self.extremity_jerk_window_p95_max_mps3
            ),
            "root_y_robust_range_m": float(self.root_y_robust_range_m),
            "root_vertical_speed_mps_p95": float(self.root_vertical_speed_mps_p95),
            "root_vertical_speed_mps_max": float(self.root_vertical_speed_mps_max),
            "root_horizontal_radius_p95_m": float(self.root_horizontal_radius_p95_m),
            "root_horizontal_radius_max_m": float(self.root_horizontal_radius_max_m),
            "root_horizontal_net_displacement_m": float(
                self.root_horizontal_net_displacement_m
            ),
            "root_horizontal_drift_speed_mps": float(
                self.root_horizontal_drift_speed_mps
            ),
            "root_horizontal_window_displacement_max_m": float(
                self.root_horizontal_window_displacement_max_m
            ),
        }

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class StageAcceptancePolicy:
    """Relative regression and minimum-repair policy for neural stages."""

    jerk_max_ratio: float = 1.02
    jerk_max_margin_mps3: float = 40.0
    jerk_p95_ratio: float = 1.10
    jerk_p95_margin_mps3: float = 25.0
    skate_p95_ratio: float = 1.10
    skate_p95_margin_mps: float = 0.01
    skate_max_ratio: float = 1.10
    skate_max_margin_mps: float = 0.03
    penetration_margin_m: float = 0.012
    root_range_ratio: float = 1.10
    root_range_margin_m: float = 0.02
    root_vertical_speed_ratio: float = 1.10
    root_vertical_speed_p95_margin_mps: float = 0.05
    root_vertical_speed_max_margin_mps: float = 0.10
    minimum_repair_gain: float = 0.03

    @classmethod
    def from_environment(cls) -> "StageAcceptancePolicy":
        return cls(
            jerk_max_ratio=_env_float(
                "PHYSICAL_STAGE_JERK_MAX_RATIO", None, 1.02
            ),
            jerk_max_margin_mps3=_env_float(
                "PHYSICAL_STAGE_JERK_MAX_MARGIN_MPS3", None, 40.0
            ),
            jerk_p95_ratio=_env_float(
                "PHYSICAL_STAGE_JERK_P95_RATIO", None, 1.10
            ),
            jerk_p95_margin_mps3=_env_float(
                "PHYSICAL_STAGE_JERK_P95_MARGIN_MPS3", None, 25.0
            ),
            skate_p95_ratio=_env_float(
                "PHYSICAL_STAGE_SKATE_P95_RATIO", None, 1.10
            ),
            skate_p95_margin_mps=_env_float(
                "PHYSICAL_STAGE_SKATE_P95_MARGIN_MPS", None, 0.01
            ),
            skate_max_ratio=_env_float(
                "PHYSICAL_STAGE_SKATE_MAX_RATIO", None, 1.10
            ),
            skate_max_margin_mps=_env_float(
                "PHYSICAL_STAGE_SKATE_MAX_MARGIN_MPS", None, 0.03
            ),
            penetration_margin_m=_env_float(
                "PHYSICAL_STAGE_PENETRATION_MARGIN_M", None, 0.012
            ),
            root_range_ratio=_env_float(
                "PHYSICAL_STAGE_ROOT_RANGE_RATIO", None, 1.10
            ),
            root_range_margin_m=_env_float(
                "PHYSICAL_STAGE_ROOT_RANGE_MARGIN_M", None, 0.02
            ),
            root_vertical_speed_ratio=_env_float(
                "PHYSICAL_STAGE_ROOT_VERTICAL_SPEED_RATIO", None, 1.10
            ),
            root_vertical_speed_p95_margin_mps=_env_float(
                "PHYSICAL_STAGE_ROOT_VERTICAL_SPEED_P95_MARGIN_MPS", None, 0.05
            ),
            root_vertical_speed_max_margin_mps=_env_float(
                "PHYSICAL_STAGE_ROOT_VERTICAL_SPEED_MAX_MARGIN_MPS", None, 0.10
            ),
            minimum_repair_gain=_env_float(
                "PHYSICAL_STAGE_MINIMUM_REPAIR_GAIN", None, 0.03
            ),
        )


@dataclass(frozen=True)
class PeakJerkMaskConfig:
    """Configuration for localized frame-joint Peak-Jerk repair masks."""

    enabled: bool = True
    absolute_threshold_mps3: float = 1400.0
    percentile: float = 99.5
    radius_frames_at_30fps: int = 4
    parent_depth: int = 2

    @classmethod
    def from_environment(cls) -> "PeakJerkMaskConfig":
        return cls(
            enabled=_env_bool(
                "PHYSICAL_PEAK_JERK_MASK_ENABLE", None, True
            ),
            absolute_threshold_mps3=_env_float(
                "PHYSICAL_PEAK_JERK_THRESHOLD_MPS3", None, 1400.0
            ),
            percentile=_env_float(
                "PHYSICAL_PEAK_JERK_PERCENTILE", None, 99.5
            ),
            radius_frames_at_30fps=_env_int(
                "PHYSICAL_PEAK_JERK_RADIUS_FRAMES", None, 4
            ),
            parent_depth=_env_int(
                "PHYSICAL_PEAK_JERK_PARENT_DEPTH", None, 2
            ),
        )


def compute_joint_kinematic_metrics(
    joints: np.ndarray,
    fps: float,
) -> Dict[str, float]:
    """Compute true frame-joint velocity/acceleration/jerk statistics."""

    positions = np.asarray(joints, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected joint positions [T,J,3], got {positions.shape}")
    if not np.isfinite(positions).all():
        return {
            "joint_velocity_mps_mean": float("inf"),
            "joint_velocity_mps_p95": float("inf"),
            "joint_velocity_mps_max": float("inf"),
            "joint_acceleration_mps2_mean": float("inf"),
            "joint_acceleration_mps2_p95": float("inf"),
            "joint_acceleration_mps2_max": float("inf"),
            "joint_jerk_mps3_mean": float("inf"),
            "joint_jerk_mps3_p95": float("inf"),
            "joint_jerk_mps3_max": float("inf"),
            "frame_mean_jerk_mps3_max": float("inf"),
        }

    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")

    def summarize(values: np.ndarray, prefix: str) -> Dict[str, float]:
        if values.size == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_p95": 0.0,
                f"{prefix}_max": 0.0,
            }
        norms = np.linalg.norm(values, axis=-1)
        return {
            f"{prefix}_mean": float(np.mean(norms)),
            f"{prefix}_p95": float(np.percentile(norms, 95)),
            f"{prefix}_max": float(np.max(norms)),
        }

    velocity = np.diff(positions, n=1, axis=0) * rate
    acceleration = np.diff(positions, n=2, axis=0) * rate ** 2
    jerk = np.diff(positions, n=3, axis=0) * rate ** 3

    metrics: Dict[str, float] = {}
    metrics.update(summarize(velocity, "joint_velocity_mps"))
    metrics.update(summarize(acceleration, "joint_acceleration_mps2"))
    metrics.update(summarize(jerk, "joint_jerk_mps3"))
    if jerk.size:
        jerk_norm = np.linalg.norm(jerk, axis=-1)
        metrics["frame_mean_jerk_mps3_max"] = float(
            np.max(np.mean(jerk_norm, axis=-1))
        )
    else:
        metrics["frame_mean_jerk_mps3_max"] = 0.0
    return metrics


def evaluate_physical_audit(
    audit: Mapping[str, Any],
    limits: Optional[PhysicalQualityLimits] = None,
) -> Dict[str, Any]:
    """Apply the single authoritative whole-motion physical gate."""

    lim = limits or PhysicalQualityLimits.from_environment()
    limit_map = lim.as_audit_limits()
    layer_reasons: Dict[str, list[str]] = {
        "anti_jitter": [],
        "foot_contact": [],
        "root_vertical": [],
        "long_horizon_root_drift": [],
    }

    def reject_high(layer: str, keys: Sequence[str]) -> None:
        for key in keys:
            if _metric(audit, key, float("inf")) > limit_map[key]:
                layer_reasons[layer].append(f"{key}_too_high")

    reject_high(
        "anti_jitter",
        (
            "joint_jerk_mps3_p95",
            "joint_jerk_mps3_max",
            "joint_jerk_window_p95_max_mps3",
            "extremity_jerk_mps3_p95",
            "extremity_jerk_window_p95_max_mps3",
        ),
    )
    reject_high(
        "foot_contact",
        (
            "foot_skate_mps_p95",
            "foot_skate_mps_max",
            "foot_support_drift_m_p95",
            "foot_support_drift_m_max",
            "foot_contact_height_m_max",
        ),
    )
    reject_high(
        "long_horizon_root_drift",
        (
            "root_horizontal_radius_p95_m",
            "root_horizontal_radius_max_m",
            "root_horizontal_net_displacement_m",
            "root_horizontal_drift_speed_mps",
            "root_horizontal_window_displacement_max_m",
        ),
    )

    robust_root_range = _metric(
        audit,
        "root_y_robust_range_m",
        _metric(audit, "root_y_range_m", float("inf")),
    )
    if robust_root_range > limit_map["root_y_robust_range_m"]:
        layer_reasons["root_vertical"].append("root_y_robust_range_m_too_high")

    for key in (
        "root_vertical_speed_mps_p95",
        "root_vertical_speed_mps_max",
    ):
        if key in audit and _metric(audit, key, float("inf")) > limit_map[key]:
            layer_reasons["root_vertical"].append(f"{key}_too_high")

    if _metric(audit, "foot_penetration_min_m", float("-inf")) < limit_map[
        "foot_penetration_min_m"
    ]:
        layer_reasons["foot_contact"].append("foot_penetration_too_low")

    reasons = [
        reason
        for layer in (
            "anti_jitter",
            "foot_contact",
            "root_vertical",
            "long_horizon_root_drift",
        )
        for reason in layer_reasons[layer]
    ]

    return {
        "ok": not reasons,
        "reasons": reasons,
        "limits": limit_map,
        "audit": dict(audit),
        "layers": {
            name: {"ok": not values, "reasons": list(values)}
            for name, values in layer_reasons.items()
        },
    }


def _allowed_after_stage(
    before: float,
    absolute_limit: float,
    ratio: float,
    margin: float,
) -> float:
    if before <= absolute_limit:
        return min(absolute_limit, before * ratio + margin)
    return before * ratio


def evaluate_stage_candidate(
    before_audit: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    *,
    limits: Optional[PhysicalQualityLimits] = None,
    policy: Optional[StageAcceptancePolicy] = None,
    require_repair_gain: bool = False,
) -> Dict[str, Any]:
    """Evaluate a neural-stage candidate against absolute and relative safety."""

    lim = limits or PhysicalQualityLimits.from_environment()
    pol = policy or StageAcceptancePolicy.from_environment()
    reasons = []
    detail: Dict[str, float] = {}

    metric_specs = (
        (
            "joint_jerk_mps3_max",
            lim.joint_jerk_mps3_max,
            pol.jerk_max_ratio,
            pol.jerk_max_margin_mps3,
            "joint_jerk_max_regressed",
        ),
        (
            "joint_jerk_mps3_p95",
            lim.joint_jerk_mps3_p95,
            pol.jerk_p95_ratio,
            pol.jerk_p95_margin_mps3,
            "joint_jerk_p95_regressed",
        ),
        (
            "foot_skate_mps_p95",
            lim.foot_skate_mps_p95,
            pol.skate_p95_ratio,
            pol.skate_p95_margin_mps,
            "foot_skate_p95_regressed",
        ),
        (
            "foot_skate_mps_max",
            lim.foot_skate_mps_max,
            pol.skate_max_ratio,
            pol.skate_max_margin_mps,
            "foot_skate_max_regressed",
        ),
        (
            "root_y_robust_range_m",
            lim.root_y_robust_range_m,
            pol.root_range_ratio,
            pol.root_range_margin_m,
            "root_y_robust_range_regressed",
        ),
        (
            "root_vertical_speed_mps_p95",
            lim.root_vertical_speed_mps_p95,
            pol.root_vertical_speed_ratio,
            pol.root_vertical_speed_p95_margin_mps,
            "root_vertical_speed_p95_regressed",
        ),
        (
            "root_vertical_speed_mps_max",
            lim.root_vertical_speed_mps_max,
            pol.root_vertical_speed_ratio,
            pol.root_vertical_speed_max_margin_mps,
            "root_vertical_speed_max_regressed",
        ),
    )

    for key, absolute_limit, ratio, margin, regression_reason in metric_specs:
        before = _metric(before_audit, key, float("inf"))
        after = _metric(candidate_audit, key, float("inf"))
        allowed = _allowed_after_stage(before, absolute_limit, ratio, margin)
        detail[f"before_{key}"] = before
        detail[f"candidate_{key}"] = after
        detail[f"allowed_{key}"] = allowed

        if after > allowed:
            reasons.append(regression_reason)
        if before <= absolute_limit and after > absolute_limit:
            reasons.append(f"absolute_{key}")

    before_penetration = _metric(
        before_audit, "foot_penetration_min_m", float("-inf")
    )
    after_penetration = _metric(
        candidate_audit, "foot_penetration_min_m", float("-inf")
    )
    detail["before_foot_penetration_min_m"] = before_penetration
    detail["candidate_foot_penetration_min_m"] = after_penetration
    detail["allowed_foot_penetration_min_m"] = max(
        lim.foot_penetration_min_m,
        before_penetration - pol.penetration_margin_m,
    )
    if after_penetration < detail["allowed_foot_penetration_min_m"]:
        reasons.append("foot_penetration_regressed")
    if before_penetration >= lim.foot_penetration_min_m and (
        after_penetration < lim.foot_penetration_min_m
    ):
        reasons.append("absolute_foot_penetration")

    before_jerk_max = detail["before_joint_jerk_mps3_max"]
    after_jerk_max = detail["candidate_joint_jerk_mps3_max"]
    if require_repair_gain and before_jerk_max > lim.joint_jerk_mps3_max:
        repair_gain = (before_jerk_max - after_jerk_max) / max(
            abs(before_jerk_max), 1.0e-8
        )
        detail["joint_jerk_max_repair_gain"] = float(repair_gain)
        detail["minimum_repair_gain"] = float(pol.minimum_repair_gain)
        if repair_gain < pol.minimum_repair_gain:
            reasons.append("no_meaningful_repair_gain")

    # Preserve reason order while removing duplicates.
    reasons = list(dict.fromkeys(reasons))
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "detail": detail,
        "limits": lim.as_audit_limits(),
        "policy": asdict(pol),
        "require_repair_gain": bool(require_repair_gain),
    }


def run_stage_transaction(
    *,
    stage_name: str,
    motion: np.ndarray,
    apply_fn: Any,
    audit_fn: Any,
    limits: Optional[PhysicalQualityLimits] = None,
    policy: Optional[StageAcceptancePolicy] = None,
    require_repair_gain: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run a repair stage as an auditable commit-or-rollback transaction."""

    snapshot = np.asarray(motion, dtype=np.float32).copy()
    before_audit = dict(audit_fn(snapshot))

    try:
        candidate = np.asarray(apply_fn(snapshot.copy()), dtype=np.float32)
        if candidate.shape != snapshot.shape:
            raise ValueError(
                f"stage changed motion shape from {snapshot.shape} to {candidate.shape}"
            )
        if not np.isfinite(candidate).all():
            raise ValueError("stage candidate contains NaN or Inf")
        candidate_audit = dict(audit_fn(candidate))
    except Exception as exc:
        return snapshot, {
            "stage": str(stage_name),
            "accepted": False,
            "rolled_back": True,
            "reasons": ["stage_exception"],
            "exception": repr(exc),
            "before_audit": before_audit,
            "selected_audit": before_audit,
        }

    decision = evaluate_stage_candidate(
        before_audit,
        candidate_audit,
        limits=limits,
        policy=policy,
        require_repair_gain=require_repair_gain,
    )
    accepted = bool(decision["accepted"])
    selected = candidate if accepted else snapshot
    selected_audit = candidate_audit if accepted else before_audit

    return selected.astype(np.float32), {
        "stage": str(stage_name),
        "accepted": accepted,
        "rolled_back": not accepted,
        "reasons": list(decision["reasons"]),
        "decision": decision,
        "before_audit": before_audit,
        "candidate_audit": candidate_audit,
        "selected_audit": selected_audit,
    }


def _frames_at_rate(frames_at_30fps: int, fps: float) -> int:
    return max(0, int(round(int(frames_at_30fps) * float(fps) / 30.0)))


def build_peak_jerk_risk_mask(
    motion: np.ndarray,
    *,
    fps: float,
    config: Optional[PeakJerkMaskConfig] = None,
    parents: Sequence[int] = PARENTS,
) -> Dict[str, Any]:
    """Build localized frame-joint masks from true world-space Peak Jerk."""

    cfg = config or PeakJerkMaskConfig.from_environment()
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected motion [T,D], got {x.shape}")

    frames = int(x.shape[0])
    joint_mask = np.zeros((frames, NUM_JOINTS), dtype=np.float32)
    frame_mask = np.zeros((frames,), dtype=np.float32)
    root_mask = np.zeros((frames,), dtype=np.float32)
    contact_mask = np.zeros((frames,), dtype=np.float32)

    if not cfg.enabled or frames < 4:
        return {
            "joint": joint_mask,
            "frame": frame_mask,
            "root": root_mask,
            "contact": contact_mask,
            "report": {
                "enabled": bool(cfg.enabled),
                "threshold_mps3": float(cfg.absolute_threshold_mps3),
                "peak_count": 0,
                "masked_frames": 0,
                "masked_joint_frame_pairs": 0,
                "radius_frames": 0,
                "parent_depth": int(cfg.parent_depth),
            },
        }

    joints = fk24_np(x).astype(np.float64)
    jerk = np.diff(joints, n=3, axis=0) * float(fps) ** 3
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    percentile = float(np.clip(cfg.percentile, 0.0, 100.0))
    adaptive = float(np.percentile(jerk_norm, percentile)) if jerk_norm.size else 0.0
    threshold = max(float(cfg.absolute_threshold_mps3), adaptive)
    risky = jerk_norm >= threshold
    radius = _frames_at_rate(cfg.radius_frames_at_30fps, fps)

    for derivative_frame, joint_id in np.argwhere(risky):
        derivative_frame = int(derivative_frame)
        joint_id = int(joint_id)
        # A third-order difference at k depends on original frames k..k+3.
        start = max(0, derivative_frame - radius)
        end = min(frames, derivative_frame + 4 + radius)
        chain_joint = joint_id
        for _ in range(max(0, int(cfg.parent_depth)) + 1):
            if chain_joint < 0 or chain_joint >= NUM_JOINTS:
                break
            joint_mask[start:end, chain_joint] = 1.0
            chain_joint = int(parents[chain_joint])
        frame_mask[start:end] = 1.0

    # Root translation is editable only when the pelvis/root chain is implicated.
    root_mask = joint_mask[:, 0].copy()

    peak_values = jerk_norm[risky]
    return {
        "joint": joint_mask,
        "frame": frame_mask,
        "root": root_mask,
        "contact": contact_mask,
        "report": {
            "enabled": True,
            "threshold_mps3": float(threshold),
            "absolute_threshold_mps3": float(cfg.absolute_threshold_mps3),
            "adaptive_percentile_mps3": float(adaptive),
            "percentile": percentile,
            "peak_count": int(risky.sum()),
            "peak_value_max_mps3": float(np.max(peak_values)) if peak_values.size else 0.0,
            "masked_frames": int(np.count_nonzero(frame_mask)),
            "masked_joint_frame_pairs": int(np.count_nonzero(joint_mask)),
            "radius_frames": int(radius),
            "parent_depth": int(cfg.parent_depth),
        },
    }


def build_repair_mask(
    motion: np.ndarray,
    seam_mask: np.ndarray,
    *,
    fps: float,
    config: Optional[PeakJerkMaskConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Union the original seam mask with the localized core Peak-Jerk mask."""

    x = np.asarray(motion, dtype=np.float32)
    seam = np.asarray(seam_mask, dtype=np.float32)
    if seam.ndim == 1:
        seam = seam[:, None]
    if seam.shape[0] != x.shape[0]:
        raise ValueError(
            f"seam mask length {seam.shape[0]} does not match motion {x.shape[0]}"
        )

    peak = build_peak_jerk_risk_mask(x, fps=fps, config=config)
    peak_frame = np.asarray(peak["frame"], dtype=np.float32)[:, None]
    repair = np.maximum(np.clip(seam, 0.0, 1.0), peak_frame)
    report = {
        "seam_active_frames": int(np.count_nonzero(seam[:, 0] > 1.0e-6)),
        "peak_active_frames": int(np.count_nonzero(peak_frame[:, 0] > 1.0e-6)),
        "repair_active_frames": int(np.count_nonzero(repair[:, 0] > 1.0e-6)),
        "repair_active_ratio": float(np.mean(repair[:, 0] > 1.0e-6)),
        "peak_jerk": peak["report"],
    }
    return repair.astype(np.float32), report
