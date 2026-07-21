"""FPS-invariant contact and kinematic metrics in SI units."""
from __future__ import annotations

from typing import Any

import numpy as np

from contracts.gravity import fk24_np
from motion_geometry.smpl24 import CONTACT, FOOT_JOINTS, MOTION_DIM

PHYSICAL_METRICS_SCHEMA = "dunhuang_physical_metrics_si_v1"


def _odd_window(seconds: float, fps: float) -> int:
    size = max(1, int(round(max(0.0, float(seconds)) * float(fps))))
    return size if size % 2 == 1 else size + 1


def median_filter_bool_np(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=bool)
    if window <= 1 or len(x) <= 1:
        return x.copy()
    radius = window // 2
    padded = np.pad(x.astype(np.uint8), ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for index in range(len(x)):
        out[index] = np.median(padded[index:index + window], axis=0) >= 0.5
    return out


def contact_from_joints_np(
    joints: np.ndarray,
    *,
    fps: float,
    floor_y: float | None = None,
    height_margin_m: float = 0.055,
    speed_gate_mps: float = 0.75,
    median_seconds: float = 1.0 / 6.0,
) -> np.ndarray:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    j = np.asarray(joints, dtype=np.float32)
    feet = j[:, list(FOOT_JOINTS)]
    if floor_y is None:
        floor_y = float(np.percentile(feet[..., 1], 5))
    speed_mps = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet) > 1:
        speed_mps[1:] = (
            np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
            * float(fps)
        )
    contact = (feet[..., 1] <= float(floor_y) + float(height_margin_m)) & (
        speed_mps <= float(speed_gate_mps)
    )
    return median_filter_bool_np(contact, _odd_window(median_seconds, fps))


def recompute_contacts_np(
    motion: np.ndarray,
    *,
    fps: float,
    height_margin_m: float = 0.055,
    speed_gate_mps: float = 0.75,
    median_seconds: float = 1.0 / 6.0,
) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32).copy()
    if x.ndim != 2 or x.shape[1] != MOTION_DIM:
        raise ValueError(f"Expected [T,{MOTION_DIM}], got {x.shape}")
    joints = fk24_np(x)
    feet = joints[:, list(FOOT_JOINTS)]
    floor_y = float(np.percentile(feet[..., 1], 5))
    x[:, CONTACT] = contact_from_joints_np(
        joints,
        fps=fps,
        floor_y=floor_y,
        height_margin_m=height_margin_m,
        speed_gate_mps=speed_gate_mps,
        median_seconds=median_seconds,
    ).astype(np.float32)
    return x


def motion_physical_metrics_np(motion: np.ndarray, *, fps: float) -> dict[str, Any]:
    """Report physical translation derivatives and contact skate in SI units."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != MOTION_DIM:
        raise ValueError(f"Expected [T,{MOTION_DIM}], got {x.shape}")
    joints = fk24_np(x)
    velocity = np.diff(joints, axis=0) * float(fps)
    acceleration = np.diff(joints, n=2, axis=0) * float(fps) ** 2
    jerk = np.diff(joints, n=3, axis=0) * float(fps) ** 3

    feet = joints[:, list(FOOT_JOINTS)]
    foot_speed_mps = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet) > 1:
        foot_speed_mps[1:] = (
            np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
            * float(fps)
        )
    contacts = x[:, CONTACT] > 0.5
    skate = foot_speed_mps[contacts]
    floor_y = float(np.percentile(feet[..., 1], 5))

    def distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return {f"{prefix}_mean": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_max": 0.0}
        return {
            f"{prefix}_mean": float(np.mean(v)),
            f"{prefix}_p95": float(np.percentile(v, 95)),
            f"{prefix}_max": float(np.max(v)),
        }

    result: dict[str, Any] = {
        "schema": PHYSICAL_METRICS_SCHEMA,
        "frames": int(len(x)),
        "fps": float(fps),
        "duration_seconds": float((len(x) - 1) / fps) if len(x) > 1 else 0.0,
        "floor_y_m": floor_y,
        "foot_penetration_min_m": float(np.min(feet[..., 1] - floor_y)),
        "contact_ratio": float(np.mean(contacts)),
    }
    result.update(distribution(skate, "foot_skate_mps"))
    result.update(distribution(np.linalg.norm(velocity, axis=-1), "joint_velocity_mps"))
    result.update(distribution(np.linalg.norm(acceleration, axis=-1), "joint_acceleration_mps2"))
    result.update(distribution(np.linalg.norm(jerk, axis=-1), "joint_jerk_mps3"))
    return result
