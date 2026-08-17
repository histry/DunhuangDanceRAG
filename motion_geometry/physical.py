"""FPS-invariant contact and kinematic metrics in SI units."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
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
    ROT6D_END,
    ROT6D_START,
)
from motion_geometry.rotations import (
    angular_acceleration_np,
    rot6d_to_matrix_np,
    so3_geodesic_np,
)

PHYSICAL_METRICS_SCHEMA = "dunhuang_physical_metrics_si_v5_contact_states"
EXTREMITY_JOINTS = (7, 8, 10, 11, 20, 21, 22, 23)
SWING = 0
STATIC_SUPPORT = 1
SLIDING_SUPPORT = 2


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


@dataclass(frozen=True)
class ContactStateThresholds:
    """One role-labelled contact contract; values are intentionally distinct."""

    observation_speed_mps: float = 0.75
    ik_lock_speed_mps: float = 0.36
    static_support_speed_mps: float = 0.18
    slide_min_speed_mps: float = 0.15
    slide_min_foot_travel_m: float = 0.05
    slide_min_root_travel_m: float = 0.045
    slide_direction_cos_min: float = 0.35
    slide_root_foot_relative_max_m: float = 0.18

    @classmethod
    def from_environment(cls) -> "ContactStateThresholds":
        return cls(
            observation_speed_mps=_env_float(
                "CONTACT_OBSERVATION_SPEED_MPS", 0.75
            ),
            ik_lock_speed_mps=_env_float(
                "CONTACT_IK_LOCK_SPEED_MPS", 0.36
            ),
            static_support_speed_mps=_env_float(
                "CONTACT_STATIC_SUPPORT_SPEED_MPS", 0.18
            ),
            slide_min_speed_mps=_env_float(
                "CONTACT_SLIDING_MIN_SPEED_MPS", 0.15
            ),
            slide_min_foot_travel_m=_env_float(
                "CONTACT_SLIDING_MIN_FOOT_TRAVEL_M", 0.05
            ),
            slide_min_root_travel_m=_env_float(
                "CONTACT_SLIDING_MIN_ROOT_TRAVEL_M", 0.045
            ),
            slide_direction_cos_min=_env_float(
                "CONTACT_SLIDING_DIRECTION_COS_MIN", 0.35
            ),
            slide_root_foot_relative_max_m=_env_float(
                "CONTACT_SLIDING_ROOT_FOOT_RELATIVE_MAX_M", 0.18
            ),
        )


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
    speed_gate_mps: float | None = None,
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
    observation_speed = (
        ContactStateThresholds.from_environment().observation_speed_mps
        if speed_gate_mps is None
        else float(speed_gate_mps)
    )
    contact = (feet[..., 1] <= float(floor_y) + float(height_margin_m)) & (
        speed_mps <= observation_speed
    )
    return median_filter_bool_np(contact, _odd_window(median_seconds, fps))


def recompute_contacts_np(
    motion: np.ndarray,
    *,
    fps: float,
    height_margin_m: float = 0.055,
    speed_gate_mps: float | None = None,
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


def classify_support_states_np(
    joints: np.ndarray,
    declared_contacts: np.ndarray,
    *,
    fps: float,
    sliding_support_eligible: np.ndarray | None = None,
    thresholds: ContactStateThresholds | None = None,
    height_support: np.ndarray | None = None,
) -> np.ndarray:
    """Return SWING/STATIC_SUPPORT/SLIDING_SUPPORT per frame and foot.

    Sliding support is fail-closed: kinematics alone never grant an exemption.
    The caller must provide semantic eligibility, and the eligible segment must
    also move consistently with the root. This preserves the anti-label-bypass
    foot-skate check for ordinary generated motion.
    """
    limits = thresholds or ContactStateThresholds.from_environment()
    positions = np.asarray(joints, dtype=np.float32)
    contacts = np.asarray(declared_contacts, dtype=bool)
    if positions.ndim != 3 or positions.shape[1:] != (24, 3):
        raise ValueError(f"Expected joints [T,24,3], got {positions.shape}")
    if contacts.shape != (len(positions), 4):
        raise ValueError(f"Expected contacts [T,4], got {contacts.shape}")
    feet = positions[:, list(FOOT_JOINTS)]
    if height_support is None:
        floor_y = float(np.percentile(feet[..., 1], 5))
        height_support = median_filter_bool_np(
            feet[..., 1] <= floor_y + 0.055,
            _odd_window(1.0 / 12.0, fps),
        )
    else:
        height_support = np.asarray(height_support, dtype=bool)
        if height_support.shape != contacts.shape:
            raise ValueError(
                f"Expected height support [T,4], got {height_support.shape}"
            )
    support = contacts | height_support
    states = np.where(support, STATIC_SUPPORT, SWING).astype(np.uint8)
    if sliding_support_eligible is None:
        return states
    eligible = np.asarray(sliding_support_eligible, dtype=bool)
    if eligible.ndim == 1:
        eligible = np.repeat(eligible[:, None], 4, axis=1)
    if eligible.shape != support.shape:
        raise ValueError(
            f"Expected sliding eligibility [T] or [T,4], got {eligible.shape}"
        )

    root_xz = positions[:, 0][:, (0, 2)]
    feet_xz = feet[..., (0, 2)]
    for foot_index in range(4):
        active = support[:, foot_index] & eligible[:, foot_index]
        start: int | None = None
        for frame in range(len(active) + 1):
            is_active = frame < len(active) and bool(active[frame])
            if is_active and start is None:
                start = frame
            if start is not None and not is_active:
                end = frame
                if end - start >= 4:
                    foot_segment = feet_xz[start:end, foot_index]
                    root_segment = root_xz[start:end]
                    foot_delta = foot_segment[-1] - foot_segment[0]
                    root_delta = root_segment[-1] - root_segment[0]
                    foot_travel = float(np.linalg.norm(foot_delta))
                    root_travel = float(np.linalg.norm(root_delta))
                    step_speed = np.linalg.norm(
                        np.diff(foot_segment, axis=0), axis=-1
                    ) * float(fps)
                    median_speed = (
                        float(np.median(step_speed)) if step_speed.size else 0.0
                    )
                    direction_cos = float(
                        np.dot(foot_delta, root_delta)
                        / max(foot_travel * root_travel, 1.0e-8)
                    )
                    relative = foot_segment - root_segment
                    relative_span = float(
                        np.max(np.linalg.norm(relative - relative[:1], axis=-1))
                    )
                    is_sliding = (
                        foot_travel >= limits.slide_min_foot_travel_m
                        and root_travel >= limits.slide_min_root_travel_m
                        and median_speed >= limits.slide_min_speed_mps
                        and direction_cos >= limits.slide_direction_cos_min
                        and relative_span
                        <= limits.slide_root_foot_relative_max_m
                    )
                    if is_sliding:
                        states[start:end, foot_index] = SLIDING_SUPPORT
                start = None
    return states


def motion_physical_metrics_np(
    motion: np.ndarray,
    *,
    fps: float,
    sliding_support_eligible: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report FK physical metrics plus raw/temporal SO(3) rotation quality."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != MOTION_DIM:
        raise ValueError(f"Expected [T,{MOTION_DIM}], got {x.shape}")
    raw_rot6d = np.asarray(
        x[:, ROT6D_START:ROT6D_END].reshape(len(x), 24, 6),
        dtype=np.float64,
    )
    finite_rot6d = np.isfinite(raw_rot6d).all(axis=-1)
    safe_rot6d = np.nan_to_num(
        raw_rot6d,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    first = safe_rot6d[..., :3]
    second = safe_rot6d[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1)
    second_norm = np.linalg.norm(second, axis=-1)
    first_unit = first / np.maximum(first_norm[..., None], 1.0e-12)
    second_orthogonal = second - (
        np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    )
    second_orthogonal_norm = np.linalg.norm(second_orthogonal, axis=-1)
    collinearity = np.abs(np.sum(first * second, axis=-1)) / np.maximum(
        first_norm * second_norm,
        1.0e-12,
    )
    degenerate = (
        ~finite_rot6d
        | (first_norm < 1.0e-5)
        | (second_norm < 1.0e-5)
        | (second_orthogonal_norm < 1.0e-5)
    )
    rotation_matrices = rot6d_to_matrix_np(safe_rot6d.astype(np.float32))
    rotation_steps = (
        so3_geodesic_np(rotation_matrices[:-1], rotation_matrices[1:])
        if len(rotation_matrices) > 1
        else np.zeros((0, 24), dtype=np.float32)
    )
    extremity_rotation_steps = (
        rotation_steps[:, list(EXTREMITY_JOINTS)]
        if rotation_steps.size
        else np.zeros((0, len(EXTREMITY_JOINTS)), dtype=np.float32)
    )
    angular_acceleration = angular_acceleration_np(
        rotation_matrices,
        fps=float(fps),
    )
    angular_acceleration_norm = np.linalg.norm(angular_acceleration, axis=-1)

    safe_motion = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    joints = fk24_np(safe_motion)
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
    declared_contacts = safe_motion[:, CONTACT] > 0.5
    floor_y = float(np.percentile(feet[..., 1], 5))
    # Contact auditing must not use a speed veto: doing so removes the fastest
    # sliding samples from the very metric intended to reject them.  A low-foot
    # support proxy is unioned with declared contact and temporally denoised.
    height_support = feet[..., 1] <= floor_y + 0.055
    height_support = median_filter_bool_np(
        height_support,
        _odd_window(1.0 / 12.0, fps),
    )
    support_states = classify_support_states_np(
        joints,
        declared_contacts,
        fps=fps,
        sliding_support_eligible=sliding_support_eligible,
        height_support=height_support,
    )
    support = support_states != SWING
    static_support = support_states == STATIC_SUPPORT
    sliding_support = support_states == SLIDING_SUPPORT
    skate = foot_speed_mps[static_support]
    support_drift, support_segment_count = _support_segment_drift_values(
        feet[..., (0, 2)],
        static_support,
    )
    root_relative_feet_xz = (
        feet[..., (0, 2)]
        - joints[:, 0][:, (0, 2)][:, None, :]
    )
    sliding_relative_speed = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet) > 1:
        sliding_relative_speed[1:] = np.linalg.norm(
            np.diff(root_relative_feet_xz, axis=0), axis=-1
        ) * float(fps)
    contact_height = np.maximum(feet[..., 1] - floor_y, 0.0)[declared_contacts]
    contact_mismatch = np.logical_xor(declared_contacts, height_support)

    root_y = np.asarray(safe_motion[:, ROOT_Y_IDX], dtype=np.float32)
    root_xz = np.asarray(
        safe_motion[:, [ROOT_X_IDX, ROOT_Z_IDX]], dtype=np.float64
    )
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
        "static_support_ratio": float(np.mean(static_support)),
        "sliding_support_ratio": float(np.mean(sliding_support)),
        "support_state_contract": {
            "states": {
                "swing": SWING,
                "static_support": STATIC_SUPPORT,
                "sliding_support": SLIDING_SUPPORT,
            },
            "thresholds": asdict(ContactStateThresholds.from_environment()),
            "sliding_requires_explicit_semantic_eligibility": True,
        },
        "foot_contact_mismatch_ratio": float(np.mean(contact_mismatch)),
        "foot_support_segment_count": int(support_segment_count),
        "rot6d_nonfinite_ratio": float(np.mean(~finite_rot6d)),
        "rot6d_degenerate_ratio": float(np.mean(degenerate)),
        "rot6d_first_vector_norm_min": float(np.min(first_norm)),
        "rot6d_second_vector_norm_min": float(np.min(second_norm)),
        "rot6d_second_orthogonal_norm_min": float(
            np.min(second_orthogonal_norm)
        ),
        "rot6d_collinearity_abs_p99": float(
            np.percentile(collinearity, 99)
        ),
        "rotation_near_pi_step_ratio": float(
            np.mean(rotation_steps >= (np.pi - 0.05))
        ) if rotation_steps.size else 0.0,
        "joint_rotation_step_window_p95_max_rad": _window_percentile_max(
            rotation_steps,
            fps=fps,
            seconds=1.0,
        ),
        "joint_angular_acceleration_window_p95_max_rps2": _window_percentile_max(
            angular_acceleration_norm,
            fps=fps,
            seconds=1.0,
        ),
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
    result.update(
        distribution(
            sliding_relative_speed[sliding_support],
            "sliding_support_relative_speed_mps",
        )
    )
    result.update(distribution(support_drift, "foot_support_drift_m"))
    result.update(distribution(contact_height, "foot_contact_height_m"))
    result.update(distribution(np.linalg.norm(velocity, axis=-1), "joint_velocity_mps"))
    result.update(distribution(np.linalg.norm(acceleration, axis=-1), "joint_acceleration_mps2"))
    result.update(distribution(jerk_norm, "joint_jerk_mps3"))
    result.update(distribution(extremity_jerk, "extremity_jerk_mps3"))
    result.update(distribution(root_vertical_speed, "root_vertical_speed_mps"))
    result.update(distribution(rotation_steps, "joint_rotation_step_rad"))
    result.update(
        distribution(extremity_rotation_steps, "extremity_rotation_step_rad")
    )
    result.update(
        distribution(
            angular_acceleration_norm,
            "joint_angular_acceleration_rps2",
        )
    )
    return result
