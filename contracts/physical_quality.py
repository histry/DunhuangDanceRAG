#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical-quality contracts for source qualification and final generation.

The two contracts intentionally have different semantics:

* Final-generation contract: strict absolute SI limits and fail-closed support
  semantics.  This protects rendered/generated motion from jitter, foot skate,
  penetration, drift, malformed Rot6D, and rotation discontinuities.
* Source-retarget contract: qualification of recorded motion before Event-DB
  construction.  Authentic high-frequency dance dynamics are judged relative
  to the aligned pre-retarget recording; fast low-foot observations are not
  automatically treated as planted support; robust penetration statistics are
  used together with a catastrophic-minimum guard.

Neural-stage transactional acceptance uses reference-relative regression
checks.  The strict absolute final-generation contract remains authoritative
only after the complete repair/IK/closed-loop stack.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from contracts.gravity import fk24_np
from motion_geometry.physical import (
    PHYSICAL_METRICS_SCHEMA,
    SOURCE_REFERENCE_KINEMATICS_SCHEMA,
    SUPPORT_POLICY_SOURCE,
)
from motion_geometry.smpl24 import NUM_JOINTS, PARENTS


def _env_float(primary: str, fallback: Optional[str], default: float) -> float:
    raw = os.environ.get(primary)
    if raw is None and fallback:
        raw = os.environ.get(fallback)
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


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


@dataclass(frozen=True)
class PhysicalQualityLimits:
    """Strict absolute limits for final generated motion, in SI units."""

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
    rot6d_nonfinite_ratio: float = 0.0
    rot6d_degenerate_ratio: float = 0.0
    rot6d_collinearity_abs_p99: float = 0.995
    rotation_near_pi_step_ratio: float = 0.0
    joint_rotation_step_rad_p95: float = 0.35
    joint_rotation_step_rad_max: float = 1.20
    joint_rotation_step_window_p95_max_rad: float = 0.50
    extremity_rotation_step_rad_p95: float = 0.45
    extremity_rotation_step_rad_max: float = 1.40
    joint_angular_acceleration_rps2_p95: float = 900.0
    joint_angular_acceleration_rps2_max: float = 3000.0
    joint_angular_acceleration_window_p95_max_rps2: float = 1600.0

    @classmethod
    def from_environment(cls) -> "PhysicalQualityLimits":
        return cls(
            foot_skate_mps_p95=_env_float(
                "PHYSICAL_MAX_FOOT_SKATE_P95_MPS",
                "ROUTING_SAFETY_MAX_FOOT_SKATE_P95_MPS",
                0.18,
            ),
            foot_skate_mps_max=_env_float(
                "PHYSICAL_MAX_FOOT_SKATE_MAX_MPS",
                "ROUTING_SAFETY_MAX_FOOT_SKATE_MAX_MPS",
                0.60,
            ),
            foot_support_drift_m_p95=_env_float(
                "PHYSICAL_MAX_FOOT_SUPPORT_DRIFT_P95_M",
                "ROUTING_SAFETY_MAX_FOOT_SUPPORT_DRIFT_P95_M",
                0.06,
            ),
            foot_support_drift_m_max=_env_float(
                "PHYSICAL_MAX_FOOT_SUPPORT_DRIFT_MAX_M",
                "ROUTING_SAFETY_MAX_FOOT_SUPPORT_DRIFT_MAX_M",
                0.12,
            ),
            foot_contact_height_m_max=_env_float(
                "PHYSICAL_MAX_FOOT_CONTACT_HEIGHT_M",
                "ROUTING_SAFETY_MAX_FOOT_CONTACT_HEIGHT_M",
                0.10,
            ),
            foot_penetration_min_m=_env_float(
                "PHYSICAL_MIN_FOOT_PENETRATION_M",
                "ROUTING_SAFETY_MIN_FOOT_PENETRATION_M",
                -0.05,
            ),
            joint_jerk_mps3_p95=_env_float(
                "PHYSICAL_MAX_JOINT_JERK_P95_MPS3",
                "ROUTING_SAFETY_MAX_JOINT_JERK_P95_MPS3",
                810.0,
            ),
            joint_jerk_mps3_max=_env_float(
                "PHYSICAL_MAX_JOINT_JERK_MAX_MPS3",
                "ROUTING_SAFETY_MAX_JOINT_JERK_MAX_MPS3",
                1620.0,
            ),
            joint_jerk_window_p95_max_mps3=_env_float(
                "PHYSICAL_MAX_JOINT_JERK_WINDOW_P95_MPS3",
                "ROUTING_SAFETY_MAX_JOINT_JERK_WINDOW_P95_MPS3",
                1080.0,
            ),
            extremity_jerk_mps3_p95=_env_float(
                "PHYSICAL_MAX_EXTREMITY_JERK_P95_MPS3",
                "ROUTING_SAFETY_MAX_EXTREMITY_JERK_P95_MPS3",
                810.0,
            ),
            extremity_jerk_window_p95_max_mps3=_env_float(
                "PHYSICAL_MAX_EXTREMITY_JERK_WINDOW_P95_MPS3",
                "ROUTING_SAFETY_MAX_EXTREMITY_JERK_WINDOW_P95_MPS3",
                1080.0,
            ),
            root_y_robust_range_m=_env_float(
                "PHYSICAL_MAX_ROOT_Y_ROBUST_RANGE_M",
                "ROUTING_SAFETY_MAX_ROOT_Y_ROBUST_RANGE_M",
                0.90,
            ),
            root_vertical_speed_mps_p95=_env_float(
                "PHYSICAL_MAX_ROOT_VERTICAL_SPEED_P95_MPS",
                "ROUTING_SAFETY_MAX_ROOT_VERTICAL_SPEED_P95_MPS",
                1.25,
            ),
            root_vertical_speed_mps_max=_env_float(
                "PHYSICAL_MAX_ROOT_VERTICAL_SPEED_MAX_MPS",
                "ROUTING_SAFETY_MAX_ROOT_VERTICAL_SPEED_MAX_MPS",
                4.0,
            ),
            root_horizontal_radius_p95_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_RADIUS_P95_M",
                "ROUTING_SAFETY_MAX_ROOT_XZ_RADIUS_P95_M",
                1.80,
            ),
            root_horizontal_radius_max_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_RADIUS_MAX_M",
                "ROUTING_SAFETY_MAX_ROOT_XZ_RADIUS_MAX_M",
                2.20,
            ),
            root_horizontal_net_displacement_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_NET_DISPLACEMENT_M",
                "ROUTING_SAFETY_MAX_ROOT_XZ_NET_DISPLACEMENT_M",
                3.00,
            ),
            root_horizontal_drift_speed_mps=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_DRIFT_SPEED_MPS",
                "ROUTING_SAFETY_MAX_ROOT_XZ_DRIFT_SPEED_MPS",
                0.12,
            ),
            root_horizontal_window_displacement_max_m=_env_float(
                "PHYSICAL_MAX_ROOT_XZ_WINDOW_DISPLACEMENT_M",
                "ROUTING_SAFETY_MAX_ROOT_XZ_WINDOW_DISPLACEMENT_M",
                1.50,
            ),
            rot6d_nonfinite_ratio=_env_float(
                "PHYSICAL_MAX_ROT6D_NONFINITE_RATIO", None, 0.0
            ),
            rot6d_degenerate_ratio=_env_float(
                "PHYSICAL_MAX_ROT6D_DEGENERATE_RATIO", None, 0.0
            ),
            rot6d_collinearity_abs_p99=_env_float(
                "PHYSICAL_MAX_ROT6D_COLLINEARITY_P99", None, 0.995
            ),
            rotation_near_pi_step_ratio=_env_float(
                "PHYSICAL_MAX_ROTATION_NEAR_PI_STEP_RATIO", None, 0.0
            ),
            joint_rotation_step_rad_p95=_env_float(
                "PHYSICAL_MAX_JOINT_ROTATION_STEP_P95_RAD", None, 0.35
            ),
            joint_rotation_step_rad_max=_env_float(
                "PHYSICAL_MAX_JOINT_ROTATION_STEP_MAX_RAD", None, 1.20
            ),
            joint_rotation_step_window_p95_max_rad=_env_float(
                "PHYSICAL_MAX_JOINT_ROTATION_STEP_WINDOW_P95_RAD", None, 0.50
            ),
            extremity_rotation_step_rad_p95=_env_float(
                "PHYSICAL_MAX_EXTREMITY_ROTATION_STEP_P95_RAD", None, 0.45
            ),
            extremity_rotation_step_rad_max=_env_float(
                "PHYSICAL_MAX_EXTREMITY_ROTATION_STEP_MAX_RAD", None, 1.40
            ),
            joint_angular_acceleration_rps2_p95=_env_float(
                "PHYSICAL_MAX_JOINT_ANGULAR_ACCEL_P95_RPS2", None, 900.0
            ),
            joint_angular_acceleration_rps2_max=_env_float(
                "PHYSICAL_MAX_JOINT_ANGULAR_ACCEL_MAX_RPS2", None, 3000.0
            ),
            joint_angular_acceleration_window_p95_max_rps2=_env_float(
                "PHYSICAL_MAX_JOINT_ANGULAR_ACCEL_WINDOW_P95_RPS2",
                None,
                1600.0,
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
            "rot6d_nonfinite_ratio": float(self.rot6d_nonfinite_ratio),
            "rot6d_degenerate_ratio": float(self.rot6d_degenerate_ratio),
            "rot6d_collinearity_abs_p99": float(self.rot6d_collinearity_abs_p99),
            "rotation_near_pi_step_ratio": float(self.rotation_near_pi_step_ratio),
            "joint_rotation_step_rad_p95": float(self.joint_rotation_step_rad_p95),
            "joint_rotation_step_rad_max": float(self.joint_rotation_step_rad_max),
            "joint_rotation_step_window_p95_max_rad": float(
                self.joint_rotation_step_window_p95_max_rad
            ),
            "extremity_rotation_step_rad_p95": float(
                self.extremity_rotation_step_rad_p95
            ),
            "extremity_rotation_step_rad_max": float(
                self.extremity_rotation_step_rad_max
            ),
            "joint_angular_acceleration_rps2_p95": float(
                self.joint_angular_acceleration_rps2_p95
            ),
            "joint_angular_acceleration_rps2_max": float(
                self.joint_angular_acceleration_rps2_max
            ),
            "joint_angular_acceleration_window_p95_max_rps2": float(
                self.joint_angular_acceleration_window_p95_max_rps2
            ),
        }

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PretrainingRoutePhysicalPolicy:
    """Catastrophic-only policy for the pretraining Scheduler smoke test.

    The same-WAV regression runs before the Motion Refiner, diffusion model,
    and boundary closed loop. It must reject malformed motion and unsafe
    vertical/rotation states without pretending that the unrefined
    concatenation already satisfies the final render contract.
    """

    foot_penetration_catastrophic_threshold_m: float = -0.18
    foot_penetration_catastrophic_max_seconds: float = 0.08

    @classmethod
    def from_environment(cls) -> "PretrainingRoutePhysicalPolicy":
        return cls(
            foot_penetration_catastrophic_threshold_m=_env_float(
                "PRETRAIN_ROUTE_CATASTROPHIC_FOOT_PENETRATION_MIN_M",
                None,
                -0.18,
            ),
            foot_penetration_catastrophic_max_seconds=_env_float(
                "PRETRAIN_ROUTE_CATASTROPHIC_FOOT_PENETRATION_MAX_SECONDS",
                None,
                0.08,
            ),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SourcePhysicalQualityPolicy:
    """Pre-training Retarget Clean policy, never used by final generation.

    Source anti-jitter is measured on parent-relative unit-bone direction
    trajectories over the direct common source/target mapped-bone set.  A
    source-only absolute noise floor prevents very-low-dynamic recordings from
    turning harmless optimizer residuals into arbitrarily large ratios.  The
    absolute SI final-generation contract is intentionally not represented in
    these fields.
    """

    unit_bone_jerk_p95_ratio: float = 1.15
    unit_bone_jerk_p95_margin_s3: float = 75.0
    unit_bone_jerk_p99_ratio: float = 1.20
    unit_bone_jerk_p99_margin_s3: float = 150.0
    unit_bone_jerk_window_ratio: float = 1.20
    unit_bone_jerk_window_margin_s3: float = 100.0
    unit_bone_extremity_jerk_p95_ratio: float = 1.15
    unit_bone_extremity_jerk_p95_margin_s3: float = 75.0
    unit_bone_extremity_jerk_p99_ratio: float = 1.20
    unit_bone_extremity_jerk_p99_margin_s3: float = 150.0
    unit_bone_extremity_jerk_window_ratio: float = 1.20
    unit_bone_extremity_jerk_window_margin_s3: float = 150.0
    # Source-only calibration floors.  These never participate in final
    # generation auditing; they only stop low-dynamic reference clips from
    # producing unstable candidate/reference ratios.
    unit_bone_jerk_p95_floor_s3: float = 1900.0
    unit_bone_jerk_p99_floor_s3: float = 4500.0
    unit_bone_jerk_window_floor_s3: float = 11000.0
    unit_bone_extremity_jerk_p95_floor_s3: float = 2800.0
    unit_bone_extremity_jerk_p99_floor_s3: float = 7500.0
    unit_bone_extremity_jerk_window_floor_s3: float = 22000.0
    foot_drift_p95_ratio: float = 1.25
    foot_drift_p95_margin_m: float = 0.03
    foot_drift_max_ratio: float = 1.35
    foot_drift_max_margin_m: float = 0.08
    foot_contact_height_m_max: float = 0.10
    foot_penetration_p01_margin_m: float = 0.04
    foot_penetration_p01_floor_m: float = -0.10
    foot_penetration_p001_margin_m: float = 0.06
    foot_penetration_p001_floor_m: float = -0.14
    # Quantile comparison epsilon only.  Catastrophic penetration remains
    # governed independently by threshold/run-duration hard gates.
    foot_penetration_p001_comparison_epsilon_m: float = 0.002
    foot_penetration_catastrophic_threshold_m: float = -0.18
    foot_penetration_catastrophic_max_seconds: float = 0.08
    root_range_ratio: float = 1.20
    root_range_margin_m: float = 0.08
    root_vertical_speed_p95_ratio: float = 1.20
    root_vertical_speed_p95_margin_mps: float = 0.15
    root_vertical_speed_max_ratio: float = 1.25
    root_vertical_speed_max_margin_mps: float = 0.40
    frame_count_tolerance: int = 1

    @classmethod
    def from_environment(cls) -> "SourcePhysicalQualityPolicy":
        return cls(
            unit_bone_jerk_p95_ratio=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_P95_RATIO", None, 1.15
            ),
            unit_bone_jerk_p95_margin_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_P95_MARGIN_S3", None, 75.0
            ),
            unit_bone_jerk_p99_ratio=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_P99_RATIO", None, 1.20
            ),
            unit_bone_jerk_p99_margin_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_P99_MARGIN_S3", None, 150.0
            ),
            unit_bone_jerk_window_ratio=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_WINDOW_RATIO", None, 1.20
            ),
            unit_bone_jerk_window_margin_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_WINDOW_MARGIN_S3", None, 100.0
            ),
            unit_bone_extremity_jerk_p95_ratio=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P95_RATIO", None, 1.15
            ),
            unit_bone_extremity_jerk_p95_margin_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P95_MARGIN_S3", None, 75.0
            ),
            unit_bone_extremity_jerk_p99_ratio=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P99_RATIO", None, 1.20
            ),
            unit_bone_extremity_jerk_p99_margin_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P99_MARGIN_S3", None, 150.0
            ),
            unit_bone_extremity_jerk_window_ratio=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_WINDOW_RATIO", None, 1.20
            ),
            unit_bone_extremity_jerk_window_margin_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_WINDOW_MARGIN_S3", None, 150.0
            ),
            unit_bone_jerk_p95_floor_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_P95_FLOOR_S3", None, 1900.0
            ),
            unit_bone_jerk_p99_floor_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_P99_FLOOR_S3", None, 4500.0
            ),
            unit_bone_jerk_window_floor_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_JERK_WINDOW_FLOOR_S3", None, 11000.0
            ),
            unit_bone_extremity_jerk_p95_floor_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P95_FLOOR_S3", None, 2800.0
            ),
            unit_bone_extremity_jerk_p99_floor_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P99_FLOOR_S3", None, 7500.0
            ),
            unit_bone_extremity_jerk_window_floor_s3=_env_float(
                "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_WINDOW_FLOOR_S3", None, 22000.0
            ),
            foot_drift_p95_ratio=_env_float(
                "SOURCE_PHYSICAL_FOOT_DRIFT_P95_RATIO", None, 1.25
            ),
            foot_drift_p95_margin_m=_env_float(
                "SOURCE_PHYSICAL_FOOT_DRIFT_P95_MARGIN_M", None, 0.03
            ),
            foot_drift_max_ratio=_env_float(
                "SOURCE_PHYSICAL_FOOT_DRIFT_MAX_RATIO", None, 1.35
            ),
            foot_drift_max_margin_m=_env_float(
                "SOURCE_PHYSICAL_FOOT_DRIFT_MAX_MARGIN_M", None, 0.08
            ),
            foot_contact_height_m_max=_env_float(
                "SOURCE_PHYSICAL_MAX_FOOT_CONTACT_HEIGHT_M", None, 0.10
            ),
            foot_penetration_p01_margin_m=_env_float(
                "SOURCE_PHYSICAL_FOOT_PENETRATION_P01_MARGIN_M", None, 0.04
            ),
            foot_penetration_p01_floor_m=_env_float(
                "SOURCE_PHYSICAL_FOOT_PENETRATION_P01_FLOOR_M", None, -0.10
            ),
            foot_penetration_p001_margin_m=_env_float(
                "SOURCE_PHYSICAL_FOOT_PENETRATION_P001_MARGIN_M", None, 0.06
            ),
            foot_penetration_p001_floor_m=_env_float(
                "SOURCE_PHYSICAL_FOOT_PENETRATION_P001_FLOOR_M", None, -0.14
            ),
            foot_penetration_p001_comparison_epsilon_m=_env_float(
                "SOURCE_PHYSICAL_FOOT_PENETRATION_P001_EPS_M", None, 0.002
            ),
            foot_penetration_catastrophic_threshold_m=_env_float(
                "SOURCE_PHYSICAL_CATASTROPHIC_FOOT_PENETRATION_MIN_M", None, -0.18
            ),
            foot_penetration_catastrophic_max_seconds=_env_float(
                "SOURCE_PHYSICAL_CATASTROPHIC_FOOT_PENETRATION_MAX_SECONDS", None, 0.08
            ),
            root_range_ratio=_env_float(
                "SOURCE_PHYSICAL_ROOT_RANGE_RATIO", None, 1.20
            ),
            root_range_margin_m=_env_float(
                "SOURCE_PHYSICAL_ROOT_RANGE_MARGIN_M", None, 0.08
            ),
            root_vertical_speed_p95_ratio=_env_float(
                "SOURCE_PHYSICAL_ROOT_VERTICAL_SPEED_P95_RATIO", None, 1.20
            ),
            root_vertical_speed_p95_margin_mps=_env_float(
                "SOURCE_PHYSICAL_ROOT_VERTICAL_SPEED_P95_MARGIN_MPS", None, 0.15
            ),
            root_vertical_speed_max_ratio=_env_float(
                "SOURCE_PHYSICAL_ROOT_VERTICAL_SPEED_MAX_RATIO", None, 1.25
            ),
            root_vertical_speed_max_margin_mps=_env_float(
                "SOURCE_PHYSICAL_ROOT_VERTICAL_SPEED_MAX_MARGIN_MPS", None, 0.40
            ),
            frame_count_tolerance=_env_int(
                "SOURCE_PHYSICAL_FRAME_COUNT_TOLERANCE", None, 1
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
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
    secondary_metric_ratio: float = 1.10
    root_drift_ratio: float = 1.10
    root_drift_margin_m: float = 0.02
    rotation_step_ratio: float = 1.10
    rotation_step_margin_rad: float = 0.02
    angular_acceleration_ratio: float = 1.10
    angular_acceleration_margin_rps2: float = 25.0
    minimum_repair_gain: float = 0.03

    @classmethod
    def from_environment(cls) -> "StageAcceptancePolicy":
        return cls(
            jerk_max_ratio=_env_float("PHYSICAL_STAGE_JERK_MAX_RATIO", None, 1.02),
            jerk_max_margin_mps3=_env_float(
                "PHYSICAL_STAGE_JERK_MAX_MARGIN_MPS3", None, 40.0
            ),
            jerk_p95_ratio=_env_float("PHYSICAL_STAGE_JERK_P95_RATIO", None, 1.10),
            jerk_p95_margin_mps3=_env_float(
                "PHYSICAL_STAGE_JERK_P95_MARGIN_MPS3", None, 25.0
            ),
            skate_p95_ratio=_env_float("PHYSICAL_STAGE_SKATE_P95_RATIO", None, 1.10),
            skate_p95_margin_mps=_env_float(
                "PHYSICAL_STAGE_SKATE_P95_MARGIN_MPS", None, 0.01
            ),
            skate_max_ratio=_env_float("PHYSICAL_STAGE_SKATE_MAX_RATIO", None, 1.10),
            skate_max_margin_mps=_env_float(
                "PHYSICAL_STAGE_SKATE_MAX_MARGIN_MPS", None, 0.03
            ),
            penetration_margin_m=_env_float(
                "PHYSICAL_STAGE_PENETRATION_MARGIN_M", None, 0.012
            ),
            root_range_ratio=_env_float("PHYSICAL_STAGE_ROOT_RANGE_RATIO", None, 1.10),
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
            secondary_metric_ratio=_env_float("PHYSICAL_STAGE_SECONDARY_RATIO", None, 1.10),
            root_drift_ratio=_env_float("PHYSICAL_STAGE_ROOT_DRIFT_RATIO", None, 1.10),
            root_drift_margin_m=_env_float(
                "PHYSICAL_STAGE_ROOT_DRIFT_MARGIN_M", None, 0.02
            ),
            rotation_step_ratio=_env_float(
                "PHYSICAL_STAGE_ROTATION_STEP_RATIO", None, 1.10
            ),
            rotation_step_margin_rad=_env_float(
                "PHYSICAL_STAGE_ROTATION_STEP_MARGIN_RAD", None, 0.02
            ),
            angular_acceleration_ratio=_env_float(
                "PHYSICAL_STAGE_ANGULAR_ACCEL_RATIO", None, 1.10
            ),
            angular_acceleration_margin_rps2=_env_float(
                "PHYSICAL_STAGE_ANGULAR_ACCEL_MARGIN_RPS2", None, 25.0
            ),
            minimum_repair_gain=_env_float(
                "PHYSICAL_STAGE_MINIMUM_REPAIR_GAIN", None, 0.03
            ),
        )


@dataclass(frozen=True)
class PhysicalMetricSpec:
    """One required metric shared by final and neural-stage acceptance."""

    key: str
    layer: str
    direction: str
    absolute_limit: float
    stage_ratio: float
    stage_margin: float
    regression_reason: str


def physical_metric_specs(
    limits: PhysicalQualityLimits,
    policy: Optional[StageAcceptancePolicy] = None,
) -> Tuple[PhysicalMetricSpec, ...]:
    """Return the strict final/stage metric registry.

    Source qualification deliberately does not consume this registry wholesale;
    it has a separate reference-relative contract below.
    """

    pol = policy or StageAcceptancePolicy.from_environment()

    def high(
        key: str,
        layer: str,
        limit: float,
        ratio: float,
        margin: float,
        reason: Optional[str] = None,
    ) -> PhysicalMetricSpec:
        return PhysicalMetricSpec(
            key,
            layer,
            "high",
            float(limit),
            float(ratio),
            float(margin),
            reason or f"{key}_regressed",
        )

    return (
        high("joint_jerk_mps3_p95", "anti_jitter", limits.joint_jerk_mps3_p95, pol.jerk_p95_ratio, pol.jerk_p95_margin_mps3, "joint_jerk_p95_regressed"),
        high("joint_jerk_mps3_max", "anti_jitter", limits.joint_jerk_mps3_max, pol.jerk_max_ratio, pol.jerk_max_margin_mps3, "joint_jerk_max_regressed"),
        high("joint_jerk_window_p95_max_mps3", "anti_jitter", limits.joint_jerk_window_p95_max_mps3, pol.jerk_p95_ratio, pol.jerk_p95_margin_mps3),
        high("extremity_jerk_mps3_p95", "anti_jitter", limits.extremity_jerk_mps3_p95, pol.jerk_p95_ratio, pol.jerk_p95_margin_mps3),
        high("extremity_jerk_window_p95_max_mps3", "anti_jitter", limits.extremity_jerk_window_p95_max_mps3, pol.jerk_p95_ratio, pol.jerk_p95_margin_mps3),
        high("foot_skate_mps_p95", "foot_contact", limits.foot_skate_mps_p95, pol.skate_p95_ratio, pol.skate_p95_margin_mps, "foot_skate_p95_regressed"),
        high("foot_skate_mps_max", "foot_contact", limits.foot_skate_mps_max, pol.skate_max_ratio, pol.skate_max_margin_mps, "foot_skate_max_regressed"),
        high("foot_support_drift_m_p95", "foot_contact", limits.foot_support_drift_m_p95, pol.skate_p95_ratio, pol.skate_p95_margin_mps),
        high("foot_support_drift_m_max", "foot_contact", limits.foot_support_drift_m_max, pol.skate_max_ratio, pol.skate_max_margin_mps),
        high("foot_contact_height_m_max", "foot_contact", limits.foot_contact_height_m_max, pol.secondary_metric_ratio, 0.01),
        PhysicalMetricSpec("foot_penetration_min_m", "foot_contact", "low", limits.foot_penetration_min_m, 1.0, pol.penetration_margin_m, "foot_penetration_regressed"),
        high("root_y_robust_range_m", "root_vertical", limits.root_y_robust_range_m, pol.root_range_ratio, pol.root_range_margin_m),
        high("root_vertical_speed_mps_p95", "root_vertical", limits.root_vertical_speed_mps_p95, pol.root_vertical_speed_ratio, pol.root_vertical_speed_p95_margin_mps),
        high("root_vertical_speed_mps_max", "root_vertical", limits.root_vertical_speed_mps_max, pol.root_vertical_speed_ratio, pol.root_vertical_speed_max_margin_mps),
        high("root_horizontal_radius_p95_m", "long_horizon_root_drift", limits.root_horizontal_radius_p95_m, pol.root_drift_ratio, pol.root_drift_margin_m),
        high("root_horizontal_radius_max_m", "long_horizon_root_drift", limits.root_horizontal_radius_max_m, pol.root_drift_ratio, pol.root_drift_margin_m),
        high("root_horizontal_net_displacement_m", "long_horizon_root_drift", limits.root_horizontal_net_displacement_m, pol.root_drift_ratio, pol.root_drift_margin_m),
        high("root_horizontal_drift_speed_mps", "long_horizon_root_drift", limits.root_horizontal_drift_speed_mps, pol.root_drift_ratio, 0.01),
        high("root_horizontal_window_displacement_max_m", "long_horizon_root_drift", limits.root_horizontal_window_displacement_max_m, pol.root_drift_ratio, pol.root_drift_margin_m),
        high("rot6d_nonfinite_ratio", "rotation_quality", limits.rot6d_nonfinite_ratio, 1.0, 0.0),
        high("rot6d_degenerate_ratio", "rotation_quality", limits.rot6d_degenerate_ratio, 1.0, 0.0),
        high("rot6d_collinearity_abs_p99", "rotation_quality", limits.rot6d_collinearity_abs_p99, pol.secondary_metric_ratio, 0.01),
        high("rotation_near_pi_step_ratio", "rotation_quality", limits.rotation_near_pi_step_ratio, 1.0, 0.0),
        high("joint_rotation_step_rad_p95", "rotation_quality", limits.joint_rotation_step_rad_p95, pol.rotation_step_ratio, pol.rotation_step_margin_rad),
        high("joint_rotation_step_rad_max", "rotation_quality", limits.joint_rotation_step_rad_max, pol.rotation_step_ratio, pol.rotation_step_margin_rad),
        high("joint_rotation_step_window_p95_max_rad", "rotation_quality", limits.joint_rotation_step_window_p95_max_rad, pol.rotation_step_ratio, pol.rotation_step_margin_rad),
        high("extremity_rotation_step_rad_p95", "rotation_quality", limits.extremity_rotation_step_rad_p95, pol.rotation_step_ratio, pol.rotation_step_margin_rad),
        high("extremity_rotation_step_rad_max", "rotation_quality", limits.extremity_rotation_step_rad_max, pol.rotation_step_ratio, pol.rotation_step_margin_rad),
        high("joint_angular_acceleration_rps2_p95", "rotation_quality", limits.joint_angular_acceleration_rps2_p95, pol.angular_acceleration_ratio, pol.angular_acceleration_margin_rps2),
        high("joint_angular_acceleration_rps2_max", "rotation_quality", limits.joint_angular_acceleration_rps2_max, pol.angular_acceleration_ratio, pol.angular_acceleration_margin_rps2),
        high("joint_angular_acceleration_window_p95_max_rps2", "rotation_quality", limits.joint_angular_acceleration_window_p95_max_rps2, pol.angular_acceleration_ratio, pol.angular_acceleration_margin_rps2),
    )


def _required_metric(
    audit: Mapping[str, Any], key: str
) -> Tuple[bool, float]:
    if key not in audit:
        return False, float("nan")
    try:
        value = float(audit[key])
    except (TypeError, ValueError):
        return False, float("nan")
    return bool(np.isfinite(value)), value


def _append_required_high(
    reasons: list[str],
    detail: Dict[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    key: str,
    ratio: float,
    margin: float,
    reason: str,
) -> None:
    ref_ok, ref = _required_metric(reference, key)
    cand_ok, cand = _required_metric(candidate, key)
    if not ref_ok:
        reasons.append(f"reference_missing_or_nonfinite:{key}")
        return
    if not cand_ok:
        reasons.append(f"candidate_missing_or_nonfinite:{key}")
        return
    allowed = ref * float(ratio) + float(margin)
    detail[key] = {
        "reference": float(ref),
        "candidate": float(cand),
        "ratio": float(ratio),
        "margin": float(margin),
        "allowed": float(allowed),
    }
    if cand > allowed:
        reasons.append(reason)



def _append_required_high_with_source_floor(
    reasons: list[str],
    detail: Dict[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    key: str,
    ratio: float,
    margin: float,
    source_only_noise_floor: float,
    reason: str,
) -> None:
    """Relative high-is-bad source check with a source-only absolute noise floor.

    The floor is deliberately *not* a final-generation safety limit.  It only
    stabilizes the pre-training source comparison when the recorded reference
    is exceptionally low dynamic.  Above the floor the original relative
    regression rule remains authoritative.
    """

    ref_ok, ref = _required_metric(reference, key)
    cand_ok, cand = _required_metric(candidate, key)
    if not ref_ok:
        reasons.append(f"reference_missing_or_nonfinite:{key}")
        return
    if not cand_ok:
        reasons.append(f"candidate_missing_or_nonfinite:{key}")
        return

    relative_allowed = float(ref) * float(ratio) + float(margin)
    floor = float(source_only_noise_floor)
    allowed = max(relative_allowed, floor)
    detail[key] = {
        "reference": float(ref),
        "candidate": float(cand),
        "ratio": float(ratio),
        "margin": float(margin),
        "relative_allowed": float(relative_allowed),
        "source_only_noise_floor_s3": floor,
        "allowed": float(allowed),
        "semantics": "source_relative_plus_source_only_noise_floor",
    }
    if cand > allowed:
        reasons.append(reason)


def _append_required_low_relative(
    reasons: list[str],
    detail: Dict[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    key: str,
    margin: float,
    absolute_floor: float,
    reason: str,
) -> None:
    ref_ok, ref = _required_metric(reference, key)
    cand_ok, cand = _required_metric(candidate, key)
    if not ref_ok:
        reasons.append(f"reference_missing_or_nonfinite:{key}")
        return
    if not cand_ok:
        reasons.append(f"candidate_missing_or_nonfinite:{key}")
        return
    allowed = max(float(absolute_floor), float(ref) - float(margin))
    detail[key] = {
        "reference": float(ref),
        "candidate": float(cand),
        "margin": float(margin),
        "absolute_floor": float(absolute_floor),
        "allowed_minimum": float(allowed),
    }
    if cand < allowed:
        reasons.append(reason)


def _append_required_low_relative_reference_aware_floor(
    reasons: list[str],
    detail: Dict[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    key: str,
    margin: float,
    absolute_floor: float,
    comparison_epsilon: float = 0.0,
    reason: str,
) -> None:
    """Source-relative low metric with a floor only when the source is healthy.

    For penetration-like metrics, larger/less-negative is better.  If the
    recorded source already satisfies the normal floor, the candidate must both
    stay close to the source and remain above that floor.  If the source itself
    is below the floor, this relative gate no longer demands an artificial full
    repair; it only rejects additional regression beyond ``margin``.  Sustained
    catastrophic penetration is checked independently by its own hard gate.
    """

    ref_ok, ref = _required_metric(reference, key)
    cand_ok, cand = _required_metric(candidate, key)
    if not ref_ok:
        reasons.append(f"reference_missing_or_nonfinite:{key}")
        return
    if not cand_ok:
        reasons.append(f"candidate_missing_or_nonfinite:{key}")
        return

    floor = float(absolute_floor)
    reference_healthy = float(ref) >= floor
    if reference_healthy:
        allowed = max(floor, float(ref) - float(margin))
        semantics = "reference_healthy_enforce_floor_and_relative_margin"
    else:
        allowed = float(ref) - float(margin)
        semantics = "reference_below_floor_relative_non_regression_only"

    detail[key] = {
        "reference": float(ref),
        "candidate": float(cand),
        "margin": float(margin),
        "absolute_floor": floor,
        "reference_satisfies_floor": bool(reference_healthy),
        "semantics": semantics,
        "allowed_minimum": float(allowed),
        "comparison_epsilon_m": float(comparison_epsilon),
        "effective_allowed_minimum": float(allowed) - float(comparison_epsilon),
    }
    if cand < float(allowed) - float(comparison_epsilon):
        reasons.append(reason)


def _append_required_low_absolute(
    reasons: list[str],
    audit: Mapping[str, Any],
    *,
    key: str,
    minimum: float,
    reason: str,
) -> None:
    finite, value = _required_metric(audit, key)
    if not finite:
        reasons.append(f"missing_or_nonfinite:{key}")
    elif value < float(minimum):
        reasons.append(reason)


def _append_required_high_absolute(
    reasons: list[str],
    audit: Mapping[str, Any],
    *,
    key: str,
    maximum: float,
    reason: str,
) -> None:
    finite, value = _required_metric(audit, key)
    if not finite:
        reasons.append(f"missing_or_nonfinite:{key}")
    elif value > float(maximum):
        reasons.append(reason)


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
            enabled=_env_bool("PHYSICAL_PEAK_JERK_MASK_ENABLE", None, True),
            absolute_threshold_mps3=_env_float(
                "PHYSICAL_PEAK_JERK_THRESHOLD_MPS3", None, 1400.0
            ),
            percentile=_env_float("PHYSICAL_PEAK_JERK_PERCENTILE", None, 99.5),
            radius_frames_at_30fps=_env_int(
                "PHYSICAL_PEAK_JERK_RADIUS_FRAMES", None, 4
            ),
            parent_depth=_env_int("PHYSICAL_PEAK_JERK_PARENT_DEPTH", None, 2),
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
    """Apply the strict final-generation whole-motion physical gate."""

    lim = limits or PhysicalQualityLimits.from_environment()
    limit_map = lim.as_audit_limits()
    layer_reasons: Dict[str, list[str]] = {
        "contract": [],
        "anti_jitter": [],
        "foot_contact": [],
        "root_vertical": [],
        "long_horizon_root_drift": [],
        "rotation_quality": [],
    }

    schema = str(audit.get("schema", ""))
    if schema != PHYSICAL_METRICS_SCHEMA:
        layer_reasons["contract"].append(
            f"missing_or_invalid_schema:{schema or 'missing'}"
        )

    for spec in physical_metric_specs(lim):
        finite, value = _required_metric(audit, spec.key)
        if not finite:
            layer_reasons[spec.layer].append(
                f"missing_or_nonfinite:{spec.key}"
            )
            continue
        if spec.direction == "high" and value > spec.absolute_limit:
            layer_reasons[spec.layer].append(f"{spec.key}_too_high")
        elif spec.direction == "low" and value < spec.absolute_limit:
            layer_reasons[spec.layer].append(f"{spec.key}_too_low")

    layer_order = (
        "contract",
        "anti_jitter",
        "foot_contact",
        "root_vertical",
        "long_horizon_root_drift",
        "rotation_quality",
    )
    reasons = [
        reason for layer in layer_order for reason in layer_reasons[layer]
    ]

    return {
        "schema": "final_generation_physical_gate_v1",
        "contract_role": "final_generation",
        "ok": not reasons,
        "reasons": reasons,
        "limits": limit_map,
        "audit": dict(audit),
        "layers": {
            name: {"ok": not values, "reasons": list(values)}
            for name, values in layer_reasons.items()
        },
    }


def evaluate_pretraining_route_audit(
    audit: Mapping[str, Any],
    limits: Optional[PhysicalQualityLimits] = None,
    *,
    policy: Optional[PretrainingRoutePhysicalPolicy] = None,
) -> Dict[str, Any]:
    """Gate the same-WAV Scheduler smoke test without using final semantics.

    This stage executes before learned motion repair. Final anti-jitter,
    planted-foot, and long-horizon horizontal-drift failures are retained as
    diagnostics, but they cannot prevent the Refiner/diffusion/boundary stages
    from running. Contract failures, unsafe root-vertical motion, malformed
    rotations, and sustained catastrophic floor penetration remain hard.
    """

    final_diagnostic = evaluate_physical_audit(audit, limits=limits)
    pol = policy or PretrainingRoutePhysicalPolicy.from_environment()

    blocking_layers = ("contract", "root_vertical", "rotation_quality")
    diagnostic_only_layers = (
        "anti_jitter",
        "foot_contact",
        "long_horizon_root_drift",
    )
    hard_reasons = [
        str(reason)
        for layer in blocking_layers
        for reason in final_diagnostic["layers"][layer]["reasons"]
    ]

    catastrophic_reasons: list[str] = []
    threshold_ok, observed_threshold = _required_metric(
        audit, "foot_penetration_catastrophic_threshold_m"
    )
    if not threshold_ok:
        catastrophic_reasons.append(
            "missing_or_nonfinite:foot_penetration_catastrophic_threshold_m"
        )
    elif abs(
        observed_threshold - float(pol.foot_penetration_catastrophic_threshold_m)
    ) > 1.0e-9:
        catastrophic_reasons.append(
            "pretraining_penetration_catastrophic_threshold_mismatch"
        )

    run_ok, run_seconds = _required_metric(
        audit, "foot_penetration_catastrophic_run_max_seconds"
    )
    if not run_ok:
        catastrophic_reasons.append(
            "missing_or_nonfinite:foot_penetration_catastrophic_run_max_seconds"
        )
    elif run_seconds > float(pol.foot_penetration_catastrophic_max_seconds):
        catastrophic_reasons.append("foot_penetration_sustained_catastrophic")

    reasons = list(dict.fromkeys(hard_reasons + catastrophic_reasons))
    diagnostic_only_reasons = [
        str(reason)
        for layer in diagnostic_only_layers
        for reason in final_diagnostic["layers"][layer]["reasons"]
    ]
    return {
        "schema": "pretraining_scheduler_route_physical_gate_v1",
        "contract_role": "pretraining_scheduler_route_smoke_test",
        "ok": not reasons,
        "reasons": reasons,
        "blocking_layers": list(blocking_layers),
        "diagnostic_only_layers": list(diagnostic_only_layers),
        "diagnostic_only_reasons": diagnostic_only_reasons,
        "policy": pol.to_dict(),
        "catastrophic_penetration": {
            "observed_threshold_m": observed_threshold if threshold_ok else None,
            "observed_run_max_seconds": run_seconds if run_ok else None,
        },
        # The caller stores the authoritative full final diagnostic separately;
        # keep only a compact summary here to avoid duplicating its large audit.
        "final_generation_diagnostic": {
            "schema": final_diagnostic["schema"],
            "contract_role": final_diagnostic["contract_role"],
            "ok": final_diagnostic["ok"],
            "reasons": list(final_diagnostic["reasons"]),
            "layers": dict(final_diagnostic["layers"]),
        },
        "final_generation_gate_required_after_motion_repair": True,
        "required_downstream_stages": [
            "motion_refiner",
            "motion_diffusion",
            "boundary_closed_loop",
            "final_generation_physical_gate",
        ],
    }


def _evaluate_source_reference_relative(
    audit: Mapping[str, Any],
    source_reference_audit: Mapping[str, Any],
    *,
    final_limits: PhysicalQualityLimits,
    policy: SourcePhysicalQualityPolicy,
) -> Dict[str, Any]:
    layer_reasons: Dict[str, list[str]] = {
        "contract": [],
        "anti_jitter": [],
        "foot_contact": [],
        "root_vertical": [],
        "rotation_quality": [],
    }
    relative_checks: Dict[str, Any] = {}

    if str(audit.get("schema", "")) != PHYSICAL_METRICS_SCHEMA:
        layer_reasons["contract"].append(
            f"missing_or_invalid_schema:{audit.get('schema', 'missing')}"
        )
    support_contract = audit.get("support_state_contract")
    support_policy = (
        str(support_contract.get("policy", ""))
        if isinstance(support_contract, Mapping)
        else ""
    )
    if support_policy != SUPPORT_POLICY_SOURCE:
        layer_reasons["contract"].append(
            f"source_support_policy_required:{support_policy or 'missing'}"
        )
    if str(source_reference_audit.get("schema", "")) != SOURCE_REFERENCE_KINEMATICS_SCHEMA:
        layer_reasons["contract"].append(
            "missing_or_invalid_source_reference_schema"
        )

    cand_fps_ok, cand_fps = _required_metric(audit, "fps")
    ref_fps_ok, ref_fps = _required_metric(source_reference_audit, "fps")
    if (
        not cand_fps_ok
        or not ref_fps_ok
        or abs(cand_fps - ref_fps) > 1.0e-6
    ):
        layer_reasons["contract"].append("source_reference_fps_mismatch")
    try:
        if (
            abs(int(audit.get("frames")) - int(source_reference_audit.get("frames")))
            > int(policy.frame_count_tolerance)
        ):
            layer_reasons["contract"].append(
                "source_reference_frame_count_mismatch"
            )
    except (TypeError, ValueError):
        layer_reasons["contract"].append(
            "missing_source_reference_frame_count"
        )

    # V2.2 comparison set contract.  Candidate and reference must use exactly
    # the same direct common mapped target bones; otherwise distribution-level
    # source-relative jerk is not comparable and fails closed.
    try:
        cand_bones = tuple(int(v) for v in audit.get("unit_bone_comparison_bones", []))
        ref_bones = tuple(
            int(v) for v in source_reference_audit.get("unit_bone_comparison_bones", [])
        )
        cand_parents = tuple(
            int(v) for v in audit.get("unit_bone_comparison_parents", [])
        )
        ref_parents = tuple(
            int(v)
            for v in source_reference_audit.get("unit_bone_comparison_parents", [])
        )
    except (TypeError, ValueError):
        cand_bones = ref_bones = cand_parents = ref_parents = ()

    if not ref_bones:
        layer_reasons["contract"].append(
            "missing_source_reference_common_mapped_bones"
        )
    if not cand_bones:
        layer_reasons["contract"].append(
            "missing_candidate_common_mapped_bones"
        )
    if cand_bones and ref_bones and cand_bones != ref_bones:
        layer_reasons["contract"].append(
            "source_candidate_comparison_bone_set_mismatch"
        )
    if not ref_parents:
        layer_reasons["contract"].append(
            "missing_source_reference_common_mapped_bone_parents"
        )
    if not cand_parents:
        layer_reasons["contract"].append(
            "missing_candidate_common_mapped_bone_parents"
        )
    if cand_parents and ref_parents and cand_parents != ref_parents:
        layer_reasons["contract"].append(
            "source_candidate_comparison_parent_set_mismatch"
        )
    try:
        cand_extremity_bones = tuple(
            int(v) for v in audit.get("unit_bone_extremity_comparison_bones", [])
        )
        ref_extremity_bones = tuple(
            int(v)
            for v in source_reference_audit.get(
                "unit_bone_extremity_comparison_bones", []
            )
        )
    except (TypeError, ValueError):
        cand_extremity_bones = ref_extremity_bones = ()
    if not ref_extremity_bones:
        layer_reasons["contract"].append(
            "missing_source_reference_common_mapped_extremity_bones"
        )
    if cand_extremity_bones != ref_extremity_bones:
        layer_reasons["contract"].append(
            "source_candidate_extremity_comparison_bone_set_mismatch"
        )

    relative_checks["unit_bone_comparison_contract"] = {
        "reference_bones": list(ref_bones),
        "candidate_bones": list(cand_bones),
        "reference_parents": list(ref_parents),
        "candidate_parents": list(cand_parents),
        "reference_extremity_bones": list(ref_extremity_bones),
        "candidate_extremity_bones": list(cand_extremity_bones),
        "exact_match": bool(
            cand_bones
            and ref_bones
            and cand_bones == ref_bones
            and cand_parents == ref_parents
            and cand_extremity_bones == ref_extremity_bones
            and bool(ref_extremity_bones)
        ),
    }

    # Source anti-jitter: parent-relative unit-bone direction dynamics over the
    # common direct mapped-bone set only.  V2.1 pelvis-relative/radius-normalized
    # fields and all raw SI jerk values stay diagnostic here.  The final gate
    # continues to use the unchanged world-space m/s^3 contract.
    anti = layer_reasons["anti_jitter"]
    unit_bone_specs = (
        (
            "unit_bone_joint_jerk_s3_p95",
            policy.unit_bone_jerk_p95_ratio,
            policy.unit_bone_jerk_p95_margin_s3,
            policy.unit_bone_jerk_p95_floor_s3,
            "unit_bone_joint_jerk_p95_regressed_vs_source",
        ),
        (
            "unit_bone_joint_jerk_s3_p99",
            policy.unit_bone_jerk_p99_ratio,
            policy.unit_bone_jerk_p99_margin_s3,
            policy.unit_bone_jerk_p99_floor_s3,
            "unit_bone_joint_jerk_p99_regressed_vs_source",
        ),
        (
            "unit_bone_joint_jerk_window_p95_max_s3",
            policy.unit_bone_jerk_window_ratio,
            policy.unit_bone_jerk_window_margin_s3,
            policy.unit_bone_jerk_window_floor_s3,
            "unit_bone_joint_jerk_window_regressed_vs_source",
        ),
        (
            "unit_bone_extremity_jerk_s3_p95",
            policy.unit_bone_extremity_jerk_p95_ratio,
            policy.unit_bone_extremity_jerk_p95_margin_s3,
            policy.unit_bone_extremity_jerk_p95_floor_s3,
            "unit_bone_extremity_jerk_p95_regressed_vs_source",
        ),
        (
            "unit_bone_extremity_jerk_s3_p99",
            policy.unit_bone_extremity_jerk_p99_ratio,
            policy.unit_bone_extremity_jerk_p99_margin_s3,
            policy.unit_bone_extremity_jerk_p99_floor_s3,
            "unit_bone_extremity_jerk_p99_regressed_vs_source",
        ),
        (
            "unit_bone_extremity_jerk_window_p95_max_s3",
            policy.unit_bone_extremity_jerk_window_ratio,
            policy.unit_bone_extremity_jerk_window_margin_s3,
            policy.unit_bone_extremity_jerk_window_floor_s3,
            "unit_bone_extremity_jerk_window_regressed_vs_source",
        ),
    )
    for key, ratio, margin, source_floor, reason in unit_bone_specs:
        _append_required_high_with_source_floor(
            anti,
            relative_checks,
            audit,
            source_reference_audit,
            key=key,
            ratio=ratio,
            margin=margin,
            source_only_noise_floor=source_floor,
            reason=reason,
        )

    foot = layer_reasons["foot_contact"]
    _append_required_high(
        foot, relative_checks, audit, source_reference_audit,
        key="foot_support_drift_m_p95",
        ratio=policy.foot_drift_p95_ratio,
        margin=policy.foot_drift_p95_margin_m,
        reason="foot_support_drift_p95_regressed_vs_source",
    )
    _append_required_high(
        foot, relative_checks, audit, source_reference_audit,
        key="foot_support_drift_m_max",
        ratio=policy.foot_drift_max_ratio,
        margin=policy.foot_drift_max_margin_m,
        reason="foot_support_drift_max_regressed_vs_source",
    )
    _append_required_high_absolute(
        foot, audit, key="foot_contact_height_m_max",
        maximum=policy.foot_contact_height_m_max,
        reason="foot_contact_height_m_max_too_high",
    )

    # Keep the V2.1 p01 rule.  Only p0.1% changes semantics in V2.2.
    _append_required_low_relative(
        foot, relative_checks, audit, source_reference_audit,
        key="foot_penetration_p01_m",
        margin=policy.foot_penetration_p01_margin_m,
        absolute_floor=policy.foot_penetration_p01_floor_m,
        reason="foot_penetration_p01_regressed_vs_source",
    )
    _append_required_low_relative_reference_aware_floor(
        foot, relative_checks, audit, source_reference_audit,
        key="foot_penetration_p001_m",
        margin=policy.foot_penetration_p001_margin_m,
        absolute_floor=policy.foot_penetration_p001_floor_m,
        comparison_epsilon=policy.foot_penetration_p001_comparison_epsilon_m,
        reason="foot_penetration_p001_regressed_vs_source",
    )

    # Sustained catastrophic penetration is an independent hard source gate.
    # It is evaluated regardless of whether p0.1% improved versus a degraded
    # reference; a single raw minimum is diagnostic only.
    threshold_ok, observed_threshold = _required_metric(
        audit, "foot_penetration_catastrophic_threshold_m"
    )
    if (
        not threshold_ok
        or abs(
            observed_threshold
            - float(policy.foot_penetration_catastrophic_threshold_m)
        )
        > 1.0e-9
    ):
        layer_reasons["contract"].append(
            "source_penetration_catastrophic_threshold_mismatch"
        )
    finite, run_seconds = _required_metric(
        audit, "foot_penetration_catastrophic_run_max_seconds"
    )
    if not finite:
        foot.append(
            "missing_or_nonfinite:foot_penetration_catastrophic_run_max_seconds"
        )
    elif run_seconds > float(policy.foot_penetration_catastrophic_max_seconds):
        foot.append("foot_penetration_sustained_catastrophic")
    relative_checks["foot_penetration_sustained_catastrophic"] = {
        "independent_hard_gate": True,
        "threshold_m": float(policy.foot_penetration_catastrophic_threshold_m),
        "candidate_run_max_seconds": run_seconds,
        "allowed_max_seconds": float(
            policy.foot_penetration_catastrophic_max_seconds
        ),
        "raw_min_diagnostic_only_m": audit.get("foot_penetration_min_m"),
    }

    root = layer_reasons["root_vertical"]
    _append_required_high(
        root, relative_checks, audit, source_reference_audit,
        key="root_y_robust_range_m",
        ratio=policy.root_range_ratio,
        margin=policy.root_range_margin_m,
        reason="root_y_robust_range_regressed_vs_source",
    )
    _append_required_high(
        root, relative_checks, audit, source_reference_audit,
        key="root_vertical_speed_mps_p95",
        ratio=policy.root_vertical_speed_p95_ratio,
        margin=policy.root_vertical_speed_p95_margin_mps,
        reason="root_vertical_speed_p95_regressed_vs_source",
    )
    _append_required_high(
        root, relative_checks, audit, source_reference_audit,
        key="root_vertical_speed_mps_max",
        ratio=policy.root_vertical_speed_max_ratio,
        margin=policy.root_vertical_speed_max_margin_mps,
        reason="root_vertical_speed_max_regressed_vs_source",
    )

    # Keep final SO(3)/Rot6D integrity absolutely strict for source qualification.
    # In particular, male_pipa_2 joint_rotation_step_rad_max remains governed
    # by the unchanged final limit (1.20 rad by default).
    rotation = layer_reasons["rotation_quality"]
    for spec in physical_metric_specs(final_limits):
        if spec.layer != "rotation_quality":
            continue
        finite, value = _required_metric(audit, spec.key)
        if not finite:
            rotation.append(f"missing_or_nonfinite:{spec.key}")
        elif spec.direction == "high" and value > spec.absolute_limit:
            rotation.append(f"{spec.key}_too_high")
        elif spec.direction == "low" and value < spec.absolute_limit:
            rotation.append(f"{spec.key}_too_low")

    order = (
        "contract",
        "anti_jitter",
        "foot_contact",
        "root_vertical",
        "rotation_quality",
    )
    reasons = list(
        dict.fromkeys(
            reason for layer in order for reason in layer_reasons[layer]
        )
    )
    return {
        "schema": "source_physical_clean_gate_v5_calibrated_unit_bone",
        "contract_role": "pretraining_source_retarget",
        "mode": "calibrated_unit_bone_common_mapped_reference_relative_source_contract",
        "ok": not reasons,
        "reasons": reasons,
        "excluded_final_only_layers": [
            "long_horizon_root_drift",
            "absolute_final_si_anti_jitter",
            "final_fail_closed_foot_skate",
        ],
        "source_policy": policy.to_dict(),
        "final_rotation_integrity_limits": final_limits.as_audit_limits(),
        "source_si_jerk_diagnostic_only": True,
        "body_normalized_jerk_diagnostic_only": True,
        "relative_checks": relative_checks,
        "audit": dict(audit),
        "source_reference_audit": dict(source_reference_audit),
        "layers": {
            name: {"ok": not values, "reasons": list(values)}
            for name, values in layer_reasons.items()
        },
    }


def evaluate_source_physical_clean_audit(
    audit: Mapping[str, Any],
    limits: Optional[PhysicalQualityLimits] = None,
    *,
    source_reference_audit: Optional[Mapping[str, Any]] = None,
    policy: Optional[SourcePhysicalQualityPolicy] = None,
) -> Dict[str, Any]:
    """Fail-closed pre-training gate, distinct from final-generation quality.

    Formal Retarget Clean callers must provide ``source_reference_audit``
    computed from aligned/resampled pre-retarget recorded keypoints.  In that
    mode authentic dynamics are preserved and only *regression versus source*
    is rejected. Missing source evidence is a contract error.
    """

    if source_reference_audit is None:
        raise RuntimeError(
            "source_reference_audit is required for the formal pre-training source gate"
        )
    return _evaluate_source_reference_relative(
        audit,
        source_reference_audit,
        final_limits=limits or PhysicalQualityLimits.from_environment(),
        policy=policy or SourcePhysicalQualityPolicy.from_environment(),
    )


def _allowed_after_stage(
    before: float,
    absolute_limit: float,
    ratio: float,
    margin: float,
) -> float:
    if before <= absolute_limit:
        return min(absolute_limit, before * ratio + margin)
    return before


def _minimum_after_stage(
    before: float,
    absolute_limit: float,
    margin: float,
) -> float:
    if before >= absolute_limit:
        return max(absolute_limit, before - margin)
    return before


def evaluate_stage_candidate(
    before_audit: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    *,
    limits: Optional[PhysicalQualityLimits] = None,
    policy: Optional[StageAcceptancePolicy] = None,
    require_repair_gain: bool = False,
    ignored_layers: Sequence[str] = (),
) -> Dict[str, Any]:
    """Evaluate a neural-stage candidate relative to its damaged input.

    ``ignored_layers`` is intentionally explicit.  Checkpoint validation uses
    it only for long-horizon root travel: a four-second authentic locomotion
    event must not be treated as whole-song root drift.  Inference-time stage
    transactions retain the default empty set and therefore preserve their
    existing strict behaviour.
    """

    lim = limits or PhysicalQualityLimits.from_environment()
    pol = policy or StageAcceptancePolicy.from_environment()
    ignored = {str(layer) for layer in ignored_layers}
    reasons: list[str] = []
    detail: Dict[str, float] = {}

    for label, audit in (("before", before_audit), ("candidate", candidate_audit)):
        schema = str(audit.get("schema", ""))
        if schema != PHYSICAL_METRICS_SCHEMA:
            reasons.append(f"{label}_missing_or_invalid_schema")

    for spec in physical_metric_specs(lim, pol):
        if spec.layer in ignored:
            continue
        before_finite, before = _required_metric(before_audit, spec.key)
        after_finite, after = _required_metric(candidate_audit, spec.key)
        detail[f"before_{spec.key}"] = before
        detail[f"candidate_{spec.key}"] = after
        if not before_finite:
            reasons.append(f"before_missing_or_nonfinite:{spec.key}")
        if not after_finite:
            reasons.append(f"candidate_missing_or_nonfinite:{spec.key}")
        if not before_finite or not after_finite:
            continue

        if spec.direction == "high":
            allowed = _allowed_after_stage(
                before,
                spec.absolute_limit,
                spec.stage_ratio,
                spec.stage_margin,
            )
            detail[f"allowed_{spec.key}"] = allowed
            if after > allowed:
                reasons.append(spec.regression_reason)
            if before <= spec.absolute_limit and after > spec.absolute_limit:
                reasons.append(f"absolute_{spec.key}")
        else:
            allowed = _minimum_after_stage(
                before,
                spec.absolute_limit,
                spec.stage_margin,
            )
            detail[f"allowed_{spec.key}"] = allowed
            if after < allowed:
                reasons.append(spec.regression_reason)
            if before >= spec.absolute_limit and after < spec.absolute_limit:
                reasons.append(f"absolute_{spec.key}")

    before_jerk_max = detail.get("before_joint_jerk_mps3_max", float("nan"))
    after_jerk_max = detail.get("candidate_joint_jerk_mps3_max", float("nan"))
    if (
        require_repair_gain
        and np.isfinite(before_jerk_max)
        and np.isfinite(after_jerk_max)
        and before_jerk_max > lim.joint_jerk_mps3_max
    ):
        repair_gain = (before_jerk_max - after_jerk_max) / max(
            abs(before_jerk_max), 1.0e-8
        )
        detail["joint_jerk_max_repair_gain"] = float(repair_gain)
        detail["minimum_repair_gain"] = float(pol.minimum_repair_gain)
        if repair_gain < pol.minimum_repair_gain:
            reasons.append("no_meaningful_repair_gain")

    reasons = list(dict.fromkeys(reasons))
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "detail": detail,
        "limits": lim.as_audit_limits(),
        "policy": asdict(pol),
        "require_repair_gain": bool(require_repair_gain),
        "ignored_layers": sorted(ignored),
    }


def evaluate_stage_reference_fidelity(
    reference_audit: Mapping[str, Any],
    candidate_audit: Mapping[str, Any],
    *,
    limits: Optional[PhysicalQualityLimits] = None,
    policy: Optional[StageAcceptancePolicy] = None,
) -> Dict[str, Any]:
    """Require a neural prediction to remain close to authentic clean motion.

    This contract is deliberately reference-relative and does not reuse the
    final absolute gate.  Recorded Dunhuang locomotion may legitimately exceed
    a whole-song root-drift threshold, and fast low-foot source observations
    are not automatically planted support.  The candidate may vary within the
    same per-metric ratio/margin budget used by stage transactions, but it may
    not introduce a material regression relative to the clean target.
    """

    lim = limits or PhysicalQualityLimits.from_environment()
    pol = policy or StageAcceptancePolicy.from_environment()
    reasons: list[str] = []
    detail: Dict[str, float] = {}

    for label, audit in (
        ("reference", reference_audit),
        ("candidate", candidate_audit),
    ):
        schema = str(audit.get("schema", ""))
        if schema != PHYSICAL_METRICS_SCHEMA:
            reasons.append(f"{label}_missing_or_invalid_schema")

    for spec in physical_metric_specs(lim, pol):
        reference_finite, reference = _required_metric(
            reference_audit, spec.key
        )
        candidate_finite, candidate = _required_metric(
            candidate_audit, spec.key
        )
        detail[f"reference_{spec.key}"] = reference
        detail[f"candidate_{spec.key}"] = candidate
        if not reference_finite:
            reasons.append(f"reference_missing_or_nonfinite:{spec.key}")
        if not candidate_finite:
            reasons.append(f"candidate_missing_or_nonfinite:{spec.key}")
        if not reference_finite or not candidate_finite:
            continue

        if spec.direction == "high":
            allowed = max(
                reference * float(spec.stage_ratio),
                reference + float(spec.stage_margin),
            )
            detail[f"allowed_{spec.key}"] = float(allowed)
            if candidate > allowed:
                reasons.append(f"reference_fidelity_{spec.key}_regressed")
        else:
            allowed = reference - float(spec.stage_margin)
            detail[f"allowed_{spec.key}"] = float(allowed)
            if candidate < allowed:
                reasons.append(f"reference_fidelity_{spec.key}_regressed")

    reasons = list(dict.fromkeys(reasons))
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "detail": detail,
        "policy": asdict(pol),
        "absolute_final_gate_used": False,
        "reference_role": "authentic_clean_motion",
    }


def select_physical_candidate(
    *,
    stage_name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    audit_fn: Any,
    limits: Optional[PhysicalQualityLimits] = None,
    rollback_enabled: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Audit an already-built candidate and apply one shared fallback policy."""

    baseline = np.asarray(reference, dtype=np.float32)
    proposal = np.asarray(candidate, dtype=np.float32)
    if proposal.shape != baseline.shape:
        raise ValueError(
            f"{stage_name} candidate shape {proposal.shape} != {baseline.shape}"
        )
    if not np.isfinite(proposal).all():
        raise ValueError(f"{stage_name} candidate contains NaN or Inf")

    candidate_audit = dict(audit_fn(proposal))
    candidate_gate = evaluate_physical_audit(candidate_audit, limits=limits)
    selected = proposal
    selected_audit = candidate_audit
    selected_gate = candidate_gate
    reference_audit: Optional[Dict[str, Any]] = None
    reference_gate: Optional[Dict[str, Any]] = None
    rolled_back = False

    if rollback_enabled and not bool(candidate_gate["ok"]):
        reference_audit = dict(audit_fn(baseline))
        reference_gate = evaluate_physical_audit(reference_audit, limits=limits)
        if bool(reference_gate["ok"]):
            selected = baseline.copy()
            selected_audit = reference_audit
            selected_gate = reference_gate
            rolled_back = True

    return selected.astype(np.float32), {
        "stage": str(stage_name),
        "enabled": bool(rollback_enabled),
        "accepted": bool(candidate_gate["ok"]),
        "rolled_back": bool(rolled_back),
        "candidate_audit": candidate_audit,
        "candidate_gate": candidate_gate,
        "reference_audit": reference_audit,
        "reference_gate": reference_gate,
        "selected_audit": selected_audit,
        "selected_gate": selected_gate,
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


def _expand_peak_jerk_pairs(
    risky: np.ndarray,
    *,
    frames: int,
    radius: int,
    parent_depth: int,
    parents: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorize the exact temporal and parent-chain Peak-Jerk expansion."""
    risky_pairs = np.asarray(risky, dtype=bool)
    expected = (max(0, int(frames) - 3), NUM_JOINTS)
    if risky_pairs.shape != expected:
        raise ValueError(
            f"risky Peak-Jerk pairs must have shape {expected}, got "
            f"{risky_pairs.shape}"
        )
    expanded = np.zeros((int(frames), NUM_JOINTS), dtype=bool)
    derivative_frames = risky_pairs.shape[0]
    for offset in range(-max(0, int(radius)), 4 + max(0, int(radius))):
        source_start = max(0, -offset)
        source_end = min(derivative_frames, int(frames) - offset)
        if source_end <= source_start:
            continue
        expanded[source_start + offset:source_end + offset] |= risky_pairs[
            source_start:source_end
        ]

    ancestors = np.zeros((NUM_JOINTS, NUM_JOINTS), dtype=np.uint8)
    for source_joint in range(NUM_JOINTS):
        chain_joint = source_joint
        for _ in range(max(0, int(parent_depth)) + 1):
            if chain_joint < 0 or chain_joint >= NUM_JOINTS:
                break
            ancestors[source_joint, chain_joint] = 1
            chain_joint = int(parents[chain_joint])
    joint_mask = (
        expanded.astype(np.uint8) @ ancestors > 0
    ).astype(np.float32)
    frame_mask = np.any(expanded, axis=1).astype(np.float32)
    return joint_mask, frame_mask


def build_peak_jerk_risk_mask(
    motion: np.ndarray,
    *,
    fps: float,
    config: Optional[PeakJerkMaskConfig] = None,
    parents: Sequence[int] = PARENTS,
) -> Dict[str, Any]:
    """Build localized frame-joint masks from strict world-space Peak Jerk."""

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

    joint_mask, frame_mask = _expand_peak_jerk_pairs(
        risky,
        frames=frames,
        radius=radius,
        parent_depth=int(cfg.parent_depth),
        parents=parents,
    )

    root_mask = joint_mask[:, 0].copy()
    contact_mask = np.max(
        joint_mask[:, [7, 8, 10, 11]], axis=1
    ).astype(np.float32)

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
    """Union the original seam mask with the localized Peak-Jerk mask."""

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
