"""FPS-invariant contact and kinematic metrics in SI units."""
from __future__ import annotations

from typing import Any

import numpy as np

from contracts.gravity import fk24_np
from motion_geometry.smpl24 import (
    CONTACT,
    FOOT_JOINTS,
    MOTION_DIM,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
)

PHYSICAL_METRICS_SCHEMA = "dunhuang_physical_metrics_si_v3_final_quality_layers"
EXTREMITY_JOINTS = (7, 8, 10, 11, 20, 21, 22, 23)


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


def _window_percentile_max(
    values: np.ndarray,
    *,
    fps: float,
    seconds: float = 1.0,
    percentile: float = 95.0,
) -> float:
    """Maximum local percentile over overlapping physical-time windows."""

    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or len(x) == 0:
        return 0.0
    window = min(len(x), max(1, int(round(float(seconds) * float(fps)))))
    hop = max(1, window // 2)
    starts = list(range(0, max(1, len(x) - window + 1), hop))
    final_start = max(0, len(x) - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return float(
        max(
            np.percentile(x[start : start + window], percentile)
            for start in starts
        )
    )


def _support_segment_drift_values(
    feet_xz: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Distances from each planted segment's first sample, for every foot."""

    positions = np.asarray(feet_xz, dtype=np.float64)
    mask = np.asarray(support, dtype=bool)
    values: list[np.ndarray] = []
    segment_count = 0
    for foot_index in range(mask.shape[1]):
        start: int | None = None
        for frame in range(len(mask) + 1):
            active = frame < len(mask) and bool(mask[frame, foot_index])
            if active and start is None:
                start = frame
            if start is not None and not active:
                end = frame
                segment = positions[start:end, foot_index]
                if len(segment):
                    values.append(
                        np.linalg.norm(segment - segment[:1], axis=-1)
                    )
                    segment_count += 1
                start = None
    if not values:
        return np.zeros((0,), dtype=np.float64), 0
    return np.concatenate(values, axis=0), int(segment_count)


def _root_window_displacement_max(
    root_xz: np.ndarray,
    *,
    fps: float,
    seconds: float = 10.0,
) -> float:
    positions = np.asarray(root_xz, dtype=np.float64)
    if len(positions) < 2:
        return 0.0
    span = min(len(positions) - 1, max(1, int(round(float(seconds) * fps))))
    displacement = np.linalg.norm(positions[span:] - positions[:-span], axis=-1)
    return float(np.max(displacement)) if displacement.size else 0.0


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
    """Report final anti-jitter, contact and long-horizon metrics in SI units."""
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
    declared_contacts = x[:, CONTACT] > 0.5
    floor_y = float(np.percentile(feet[..., 1], 5))
    # Contact auditing must not use a speed veto: doing so removes the fastest
    # sliding samples from the very metric intended to reject them.  A low-foot
    # support proxy is unioned with declared contact and temporally denoised.
    height_support = feet[..., 1] <= floor_y + 0.055
    height_support = median_filter_bool_np(
        height_support,
        _odd_window(1.0 / 12.0, fps),
    )
    support = declared_contacts | height_support
    skate = foot_speed_mps[support]
    support_drift, support_segment_count = _support_segment_drift_values(
        feet[..., (0, 2)],
        support,
    )
    contact_height = np.maximum(feet[..., 1] - floor_y, 0.0)[declared_contacts]
    contact_mismatch = np.logical_xor(declared_contacts, height_support)

    root_y = np.asarray(x[:, ROOT_Y_IDX], dtype=np.float32)
    root_xz = np.asarray(x[:, [ROOT_X_IDX, ROOT_Z_IDX]], dtype=np.float64)
    root_vertical_speed = (
        np.abs(np.diff(root_y)) * float(fps)
        if len(root_y) > 1
        else np.zeros(0, dtype=np.float32)
    )
    duration = float((len(x) - 1) / fps) if len(x) > 1 else 0.0
    root_center = (
        np.median(root_xz, axis=0)
        if len(root_xz)
        else np.zeros((2,), dtype=np.float64)
    )
    root_radius = (
        np.linalg.norm(root_xz - root_center[None], axis=-1)
        if len(root_xz)
        else np.zeros((0,), dtype=np.float64)
    )
    root_steps = (
        np.linalg.norm(np.diff(root_xz, axis=0), axis=-1)
        if len(root_xz) > 1
        else np.zeros((0,), dtype=np.float64)
    )
    root_net_displacement = (
        float(np.linalg.norm(root_xz[-1] - root_xz[0]))
        if len(root_xz) > 1
        else 0.0
    )
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    extremity_jerk = (
        jerk_norm[:, list(EXTREMITY_JOINTS)]
        if jerk_norm.size
        else np.zeros((0, len(EXTREMITY_JOINTS)), dtype=np.float64)
    )

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
        "duration_seconds": duration,
        "floor_y_m": floor_y,
        "foot_penetration_min_m": float(np.min(feet[..., 1] - floor_y)),
        "contact_ratio": float(np.mean(declared_contacts)),
        "foot_support_ratio": float(np.mean(support)),
        "foot_contact_mismatch_ratio": float(np.mean(contact_mismatch)),
        "foot_support_segment_count": int(support_segment_count),
        "root_y_range_m": float(np.ptp(root_y)) if root_y.size else 0.0,
        "root_y_robust_range_m": (
            float(np.percentile(root_y, 99) - np.percentile(root_y, 1))
            if root_y.size
            else 0.0
        ),
        "joint_jerk_window_p95_max_mps3": _window_percentile_max(
            jerk_norm,
            fps=fps,
            seconds=1.0,
        ),
        "extremity_jerk_window_p95_max_mps3": _window_percentile_max(
            extremity_jerk,
            fps=fps,
            seconds=1.0,
        ),
        "root_horizontal_center_m": root_center.astype(float).tolist(),
        "root_horizontal_radius_p95_m": (
            float(np.percentile(root_radius, 95)) if root_radius.size else 0.0
        ),
        "root_horizontal_radius_max_m": (
            float(np.max(root_radius)) if root_radius.size else 0.0
        ),
        "root_horizontal_net_displacement_m": root_net_displacement,
        "root_horizontal_path_length_m": float(np.sum(root_steps)),
        "root_horizontal_drift_speed_mps": (
            root_net_displacement / max(duration, 1.0e-8) if duration > 0.0 else 0.0
        ),
        "root_horizontal_window_seconds": 10.0,
        "root_horizontal_window_displacement_max_m": _root_window_displacement_max(
            root_xz,
            fps=fps,
            seconds=10.0,
        ),
    }
    result.update(distribution(skate, "foot_skate_mps"))
    result.update(distribution(support_drift, "foot_support_drift_m"))
    result.update(distribution(contact_height, "foot_contact_height_m"))
    result.update(distribution(np.linalg.norm(velocity, axis=-1), "joint_velocity_mps"))
    result.update(distribution(np.linalg.norm(acceleration, axis=-1), "joint_acceleration_mps2"))
    result.update(distribution(jerk_norm, "joint_jerk_mps3"))
    result.update(distribution(extremity_jerk, "extremity_jerk_mps3"))
    result.update(distribution(root_vertical_speed, "root_vertical_speed_mps"))
    return result
