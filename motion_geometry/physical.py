"""FPS-invariant contact and kinematic metrics in SI units.

Two support policies are intentionally exposed:

``final_fail_closed``
    Used by generated-motion auditing.  Declared contact is unioned with a
    low-foot proxy so a generator cannot evade foot-skate checks by clearing
    contact labels.

``source_observation``
    Used only while qualifying trusted/recorded source motion for the training
    database.  A low foot is *not* automatically a planted foot: only slow
    support is static, semantically eligible coherent travel may be sliding,
    and the remaining fast low-foot samples stay swing.  This prevents genuine
    low sweeping/turning steps in Change-E from being mislabeled as foot skate.

The default remains ``final_fail_closed`` so existing final-generation callers
retain the strict fail-closed behaviour.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from contracts.gravity import fk24_np
from motion_geometry.smpl24 import (
    CONTACT,
    FOOT_JOINTS,
    MOTION_DIM,
    NUM_JOINTS,
    PARENTS,
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
SOURCE_REFERENCE_KINEMATICS_SCHEMA = "source_reference_kinematics_unit_bone_v3"

EXTREMITY_JOINTS = (7, 8, 10, 11, 20, 21, 22, 23)
SWING = 0
STATIC_SUPPORT = 1
SLIDING_SUPPORT = 2

SUPPORT_POLICY_FINAL = "final_fail_closed"
SUPPORT_POLICY_SOURCE = "source_observation"
_SUPPORT_POLICIES = {SUPPORT_POLICY_FINAL, SUPPORT_POLICY_SOURCE}


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


def _distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_p95": float(np.percentile(v, 95)),
        f"{prefix}_max": float(np.max(v)),
    }


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


def _foot_speed_mps(feet: np.ndarray, fps: float) -> np.ndarray:
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet) > 1:
        speed[1:] = (
            np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
            * float(fps)
        )
    return speed


def classify_support_states_np(
    joints: np.ndarray,
    declared_contacts: np.ndarray,
    *,
    fps: float,
    sliding_support_eligible: np.ndarray | None = None,
    thresholds: ContactStateThresholds | None = None,
    height_support: np.ndarray | None = None,
    support_policy: str = SUPPORT_POLICY_FINAL,
) -> np.ndarray:
    """Return SWING/STATIC_SUPPORT/SLIDING_SUPPORT per frame and foot.

    ``final_fail_closed`` preserves the original anti-label-bypass behaviour:
    declared contact OR low height is provisionally support, then semantically
    eligible coherent travel may be upgraded to SLIDING_SUPPORT.

    ``source_observation`` is deliberately different.  Recorded/retargeted
    training sources may contain legitimate fast low-foot sweep/turn steps.
    Only low-speed support is considered STATIC_SUPPORT; semantically eligible
    coherent travel may be SLIDING_SUPPORT; other fast low-foot observations
    remain SWING.  This policy must never be used for final generated motion.
    """
    limits = thresholds or ContactStateThresholds.from_environment()
    policy = str(support_policy).strip().lower()
    if policy not in _SUPPORT_POLICIES:
        raise ValueError(
            f"support_policy must be one of {sorted(_SUPPORT_POLICIES)}, got {support_policy!r}"
        )

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

    support_observation = contacts | height_support
    foot_speed = _foot_speed_mps(feet, fps)

    if policy == SUPPORT_POLICY_FINAL:
        states = np.where(
            support_observation, STATIC_SUPPORT, SWING
        ).astype(np.uint8)
    else:
        static = support_observation & (
            foot_speed <= float(limits.static_support_speed_mps)
        )
        states = np.where(static, STATIC_SUPPORT, SWING).astype(np.uint8)

    if sliding_support_eligible is None:
        return states

    eligible = np.asarray(sliding_support_eligible, dtype=bool)
    if eligible.ndim == 1:
        eligible = np.repeat(eligible[:, None], 4, axis=1)
    if eligible.shape != support_observation.shape:
        raise ValueError(
            f"Expected sliding eligibility [T] or [T,4], got {eligible.shape}"
        )

    root_xz = positions[:, 0][:, (0, 2)]
    feet_xz = feet[..., (0, 2)]
    for foot_index in range(4):
        # Sliding classification uses the observation mask rather than only the
        # static subset, otherwise a fast legitimate slide can never be found.
        active = support_observation[:, foot_index] & eligible[:, foot_index]
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
                        and relative_span <= limits.slide_root_foot_relative_max_m
                    )
                    if is_sliding:
                        states[start:end, foot_index] = SLIDING_SUPPORT
                start = None
    return states


def _max_true_run_frames(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    best = run = 0
    for value in values:
        if value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _body_normalized_root_relative_jerk_metrics(
    joints: np.ndarray,
    *,
    fps: float,
) -> dict[str, Any]:
    """Legacy V2.1 pelvis-relative diagnostic, no longer a source gate.

    These values are retained in reports so V2.1/V2.2 runs remain directly
    comparable.  They are intentionally *not* used for V2.2 source acceptance:
    pelvis-relative Cartesian trajectories still mix articulated-chain bone
    proportions.  The authoritative V2.2 source anti-jitter representation is
    ``_unit_bone_direction_jerk_metrics`` below.  Final SI jerk remains the
    original world-space m/s^3 contract.
    """
    positions = np.asarray(joints, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (24, 3):
        raise ValueError(f"Expected joints [T,24,3], got {positions.shape}")
    relative = positions - positions[:, :1, :]
    radius = np.linalg.norm(relative, axis=-1)
    joint_radius = np.median(radius, axis=0) if len(radius) else np.zeros(24)
    nonroot = joint_radius[1:]
    reference_radius = float(np.percentile(nonroot, 75)) if nonroot.size else 1.0
    radius_floor = max(1.0e-4, 0.08 * max(reference_radius, 1.0e-4))
    denom = np.maximum(joint_radius, radius_floor)
    denom[0] = 1.0
    normalized = relative / denom[None, :, None]
    normalized[:, 0] = 0.0
    jerk = np.diff(normalized, n=3, axis=0) * float(fps) ** 3
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    joint_jerk = jerk_norm[:, 1:] if jerk_norm.size else np.zeros((0, 23))
    extremity_jerk = (
        jerk_norm[:, list(EXTREMITY_JOINTS)]
        if jerk_norm.size else np.zeros((0, len(EXTREMITY_JOINTS)))
    )
    out: dict[str, Any] = {
        "body_normalization_reference_radius_m": reference_radius,
        "body_normalization_radius_floor_m": radius_floor,
        "body_normalization_joint_radius_m": joint_radius.astype(float).tolist(),
        "body_normalized_joint_jerk_s3_p99": float(np.percentile(joint_jerk, 99)) if joint_jerk.size else 0.0,
        "body_normalized_joint_jerk_window_p95_max_s3": _window_percentile_max(joint_jerk, fps=fps, seconds=1.0),
        "body_normalized_extremity_jerk_s3_p99": float(np.percentile(extremity_jerk, 99)) if extremity_jerk.size else 0.0,
        "body_normalized_extremity_jerk_window_p95_max_s3": _window_percentile_max(extremity_jerk, fps=fps, seconds=1.0),
    }
    out.update(_distribution(joint_jerk, "body_normalized_joint_jerk_s3"))
    out.update(_distribution(extremity_jerk, "body_normalized_extremity_jerk_s3"))
    return out


def _normalize_source_comparison_bones(
    source_comparison_bones: Sequence[int] | np.ndarray | None,
) -> tuple[int, ...]:
    """Return a validated, ordered set of target child-joint bone indices.

    A target bone is identified by its child joint; its parent is read from the
    canonical SMPL24/EDGE tree.  Formal BVH callers pass the direct common
    source/target mapped-bone set produced by ``retargeting.build_cache``.
    ``None`` is retained only for canonical/self-reference diagnostics and
    means all non-root target bones.
    """

    if source_comparison_bones is None:
        values = list(range(1, NUM_JOINTS))
    else:
        values = [int(v) for v in source_comparison_bones]

    ordered: list[int] = []
    seen: set[int] = set()
    for child in values:
        if child in seen:
            continue
        if child <= 0 or child >= NUM_JOINTS:
            raise ValueError(
                "source comparison bones must be non-root SMPL24 child indices; "
                f"got {child}"
            )
        parent = int(PARENTS[child])
        if parent < 0 or parent >= child:
            raise ValueError(
                f"invalid canonical parent for comparison bone {child}: {parent}"
            )
        seen.add(child)
        ordered.append(child)

    if not ordered:
        raise ValueError("source comparison bone set must not be empty")
    return tuple(ordered)


def _unit_bone_direction_jerk_metrics(
    joints: np.ndarray,
    *,
    fps: float,
    source_comparison_bones: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Parent-relative unit-bone directional jerk for source qualification.

    For each selected target bone ``parent(j) -> j`` we form

        u_j(t) = (p_j(t) - p_parent(j)(t)) / ||p_j(t)-p_parent(j)(t)||

    and measure the third finite difference of ``u_j`` in physical time.  Bone
    length therefore cannot scale the statistic, and upstream chain lengths or
    root translation cancel before differentiation.  Only the common direct
    source/target mapped bones supplied by the cache builder are authoritative
    for BVH source qualification.

    Units are s^-3 because ``u_j`` is dimensionless.  This is a source-only
    relative diagnostic/gate; final generated motion continues to use the
    unchanged world-space SI jerk metrics in m/s^3.
    """

    positions = np.asarray(joints, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(
            f"Expected joints [T,{NUM_JOINTS},3], got {positions.shape}"
        )
    if not np.isfinite(positions).all():
        raise ValueError("unit-bone source trajectory contains NaN/Inf")
    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")

    bones = _normalize_source_comparison_bones(source_comparison_bones)
    children = np.asarray(bones, dtype=np.int64)
    parents = np.asarray([int(PARENTS[j]) for j in bones], dtype=np.int64)

    vectors = positions[:, children] - positions[:, parents]
    lengths = np.linalg.norm(vectors, axis=-1)
    median_lengths = (
        np.median(lengths, axis=0)
        if len(lengths)
        else np.zeros((len(bones),), dtype=np.float64)
    )
    length_epsilon_m = max(
        1.0e-8,
        _env_float("SOURCE_PHYSICAL_UNIT_BONE_LENGTH_EPS_M", 1.0e-5),
    )
    degenerate = median_lengths <= float(length_epsilon_m)
    if np.any(degenerate):
        bad = [int(bones[i]) for i in np.flatnonzero(degenerate)]
        raise ValueError(
            "degenerate source comparison bones after mapping: "
            f"children={bad}, epsilon_m={length_epsilon_m}"
        )

    unit = vectors / np.maximum(lengths[..., None], float(length_epsilon_m))
    jerk = np.diff(unit, n=3, axis=0) * rate**3
    jerk_norm = np.linalg.norm(jerk, axis=-1)

    extremity_columns = [
        i for i, child in enumerate(bones) if int(child) in EXTREMITY_JOINTS
    ]
    extremity_bones = [int(bones[i]) for i in extremity_columns]
    extremity_jerk = (
        jerk_norm[:, extremity_columns]
        if jerk_norm.size and extremity_columns
        else np.zeros((0, len(extremity_columns)), dtype=np.float64)
    )

    out: dict[str, Any] = {
        "unit_bone_comparison_bones": [int(v) for v in bones],
        "unit_bone_comparison_parents": [int(v) for v in parents],
        "unit_bone_comparison_count": int(len(bones)),
        "unit_bone_extremity_comparison_bones": extremity_bones,
        "unit_bone_extremity_comparison_count": int(len(extremity_bones)),
        "unit_bone_length_epsilon_m": float(length_epsilon_m),
        "unit_bone_median_lengths_m": median_lengths.astype(float).tolist(),
        "unit_bone_joint_jerk_s3_p99": (
            float(np.percentile(jerk_norm, 99)) if jerk_norm.size else 0.0
        ),
        "unit_bone_joint_jerk_window_p95_max_s3": _window_percentile_max(
            jerk_norm, fps=rate, seconds=1.0
        ),
        "unit_bone_extremity_jerk_s3_p99": (
            float(np.percentile(extremity_jerk, 99))
            if extremity_jerk.size
            else 0.0
        ),
        "unit_bone_extremity_jerk_window_p95_max_s3": _window_percentile_max(
            extremity_jerk, fps=rate, seconds=1.0
        ),
    }
    out.update(_distribution(jerk_norm, "unit_bone_joint_jerk_s3"))
    out.update(
        _distribution(extremity_jerk, "unit_bone_extremity_jerk_s3")
    )
    return out


def _penetration_robust_metrics(relative_height: np.ndarray, *, fps: float) -> dict[str, Any]:
    h = np.asarray(relative_height, dtype=np.float64)
    flat = h.reshape(-1)
    per_frame_min = np.min(h, axis=1) if h.size else np.zeros((0,), dtype=np.float64)
    threshold = _env_float("SOURCE_PHYSICAL_CATASTROPHIC_FOOT_PENETRATION_MIN_M", -0.18)
    catastrophic = per_frame_min < float(threshold)
    run_frames = _max_true_run_frames(catastrophic)
    return {
        "foot_penetration_p001_m": float(np.percentile(flat, 0.1)) if flat.size else 0.0,
        "foot_penetration_p005_m": float(np.percentile(flat, 0.5)) if flat.size else 0.0,
        "foot_penetration_catastrophic_threshold_m": float(threshold),
        "foot_penetration_catastrophic_frame_ratio": float(np.mean(catastrophic)) if catastrophic.size else 0.0,
        "foot_penetration_catastrophic_run_max_frames": int(run_frames),
        "foot_penetration_catastrophic_run_max_seconds": float(run_frames / float(fps)) if fps > 0 else float("inf"),
    }


def source_reference_kinematic_metrics_np(
    joints: np.ndarray,
    *,
    fps: float,
    source_comparison_bones: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Recorded pre-retarget reference metrics in target coordinates/FPS."""
    positions = np.asarray(joints, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (24, 3):
        raise ValueError(f"Expected source reference joints [T,24,3], got {positions.shape}")
    if not np.isfinite(positions).all() or not np.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("invalid source reference trajectory or fps")
    velocity = np.diff(positions, n=1, axis=0) * float(fps)
    acceleration = np.diff(positions, n=2, axis=0) * float(fps) ** 2
    jerk = np.diff(positions, n=3, axis=0) * float(fps) ** 3
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    extremity_jerk = jerk_norm[:, list(EXTREMITY_JOINTS)] if jerk_norm.size else np.zeros((0, len(EXTREMITY_JOINTS)))
    feet = positions[:, list(FOOT_JOINTS)]
    foot_speed = _foot_speed_mps(feet.astype(np.float32), float(fps)).astype(np.float64)
    floor_y = float(np.percentile(feet[..., 1], 5))
    height_support = median_filter_bool_np(feet[..., 1] <= floor_y + 0.055, _odd_window(1.0 / 12.0, fps))
    contact_limits = ContactStateThresholds.from_environment()
    static_support = height_support & (foot_speed <= float(contact_limits.static_support_speed_mps))
    support_drift, support_segment_count = _support_segment_drift_values(feet[..., (0, 2)], static_support)
    penetration = feet[..., 1] - floor_y
    root_y = positions[:, 0, 1]
    root_vertical_speed = np.abs(np.diff(root_y)) * float(fps) if len(root_y) > 1 else np.zeros((0,))
    result: dict[str, Any] = {
        "schema": SOURCE_REFERENCE_KINEMATICS_SCHEMA,
        "frames": int(len(positions)), "fps": float(fps), "floor_y_m": floor_y,
        "joint_jerk_window_p95_max_mps3": _window_percentile_max(jerk_norm, fps=fps, seconds=1.0),
        "extremity_jerk_window_p95_max_mps3": _window_percentile_max(extremity_jerk, fps=fps, seconds=1.0),
        "joint_jerk_mps3_p99": float(np.percentile(jerk_norm, 99)) if jerk_norm.size else 0.0,
        "extremity_jerk_mps3_p99": float(np.percentile(extremity_jerk, 99)) if extremity_jerk.size else 0.0,
        "source_static_support_ratio": float(np.mean(static_support)),
        "source_support_segment_count": int(support_segment_count),
        "foot_penetration_min_m": float(np.min(penetration)),
        "foot_penetration_p01_m": float(np.percentile(penetration, 1)),
        "root_y_robust_range_m": float(np.percentile(root_y, 99) - np.percentile(root_y, 1)) if root_y.size else 0.0,
    }
    result.update(_penetration_robust_metrics(penetration, fps=fps))
    # V2.1 metric is retained for historical diagnostics only.
    result.update(_body_normalized_root_relative_jerk_metrics(positions, fps=fps))
    # V2.2 authoritative source anti-jitter representation.
    result.update(
        _unit_bone_direction_jerk_metrics(
            positions,
            fps=fps,
            source_comparison_bones=source_comparison_bones,
        )
    )
    result.update(_distribution(np.linalg.norm(velocity, axis=-1), "joint_velocity_mps"))
    result.update(_distribution(np.linalg.norm(acceleration, axis=-1), "joint_acceleration_mps2"))
    result.update(_distribution(jerk_norm, "joint_jerk_mps3"))
    result.update(_distribution(extremity_jerk, "extremity_jerk_mps3"))
    result.update(_distribution(foot_speed[static_support], "foot_skate_mps"))
    result.update(_distribution(support_drift, "foot_support_drift_m"))
    result.update(_distribution(root_vertical_speed, "root_vertical_speed_mps"))
    return result


def motion_physical_metrics_np(
    motion: np.ndarray, *, fps: float,
    sliding_support_eligible: np.ndarray | None = None,
    support_policy: str = SUPPORT_POLICY_FINAL,
    source_comparison_bones: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Report final SI metrics plus source-only normalized diagnostics.

    Existing SI metric definitions are intentionally unchanged.  Extra
    body-normalized and robust-penetration fields are ignored by the final gate
    and are consumed only by the source-retarget contract.
    """
    if fps <= 0.0: raise ValueError("fps must be positive")
    policy = str(support_policy).strip().lower()
    if policy not in _SUPPORT_POLICIES: raise ValueError(f"support_policy must be one of {sorted(_SUPPORT_POLICIES)}, got {support_policy!r}")
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1: x = x[0]
    if x.ndim != 2 or x.shape[1] != MOTION_DIM: raise ValueError(f"Expected [T,{MOTION_DIM}], got {x.shape}")
    raw_rot6d = np.asarray(x[:, ROT6D_START:ROT6D_END].reshape(len(x), 24, 6), dtype=np.float64)
    finite_rot6d = np.isfinite(raw_rot6d).all(axis=-1)
    safe_rot6d = np.nan_to_num(raw_rot6d, nan=0.0, posinf=0.0, neginf=0.0)
    first, second = safe_rot6d[..., :3], safe_rot6d[..., 3:]
    first_norm, second_norm = np.linalg.norm(first, axis=-1), np.linalg.norm(second, axis=-1)
    first_unit = first / np.maximum(first_norm[..., None], 1e-12)
    second_orthogonal = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_orthogonal_norm = np.linalg.norm(second_orthogonal, axis=-1)
    collinearity = np.abs(np.sum(first * second, axis=-1)) / np.maximum(first_norm * second_norm, 1e-12)
    degenerate = ~finite_rot6d | (first_norm < 1e-5) | (second_norm < 1e-5) | (second_orthogonal_norm < 1e-5)
    rotation_matrices = rot6d_to_matrix_np(safe_rot6d.astype(np.float32))
    rotation_steps = so3_geodesic_np(rotation_matrices[:-1], rotation_matrices[1:]) if len(rotation_matrices) > 1 else np.zeros((0,24), dtype=np.float32)
    extremity_rotation_steps = rotation_steps[:, list(EXTREMITY_JOINTS)] if rotation_steps.size else np.zeros((0,len(EXTREMITY_JOINTS)), dtype=np.float32)
    angular_acceleration_norm = np.linalg.norm(angular_acceleration_np(rotation_matrices, fps=float(fps)), axis=-1)
    safe_motion = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    joints = fk24_np(safe_motion)
    velocity = np.diff(joints, axis=0) * float(fps)
    acceleration = np.diff(joints, n=2, axis=0) * float(fps)**2
    jerk = np.diff(joints, n=3, axis=0) * float(fps)**3
    feet = joints[:, list(FOOT_JOINTS)]
    foot_speed_mps = _foot_speed_mps(feet, fps)
    declared_contacts = safe_motion[:, CONTACT] > 0.5
    floor_y = float(np.percentile(feet[...,1],5))
    height_support = median_filter_bool_np(feet[...,1] <= floor_y + 0.055, _odd_window(1.0/12.0,fps))
    support_states = classify_support_states_np(joints, declared_contacts, fps=fps, sliding_support_eligible=sliding_support_eligible, height_support=height_support, support_policy=policy)
    support = support_states != SWING; static_support = support_states == STATIC_SUPPORT; sliding_support = support_states == SLIDING_SUPPORT
    skate = foot_speed_mps[static_support]
    support_drift, support_segment_count = _support_segment_drift_values(feet[..., (0,2)], static_support)
    root_relative_feet_xz = feet[..., (0,2)] - joints[:,0][:,(0,2)][:,None,:]
    sliding_relative_speed = np.zeros(feet.shape[:2], dtype=np.float32)
    if len(feet)>1: sliding_relative_speed[1:] = np.linalg.norm(np.diff(root_relative_feet_xz,axis=0),axis=-1)*float(fps)
    contact_height = np.maximum(feet[...,1]-floor_y,0.0)[declared_contacts]
    contact_mismatch = np.logical_xor(declared_contacts,height_support)
    foot_relative_height = feet[...,1]-floor_y
    root_y = np.asarray(safe_motion[:,ROOT_Y_IDX],dtype=np.float32)
    root_xz = np.asarray(safe_motion[:,[ROOT_X_IDX,ROOT_Z_IDX]],dtype=np.float64)
    root_vertical_speed = np.abs(np.diff(root_y))*float(fps) if len(root_y)>1 else np.zeros(0,dtype=np.float32)
    duration = float((len(x)-1)/fps) if len(x)>1 else 0.0
    root_center = np.median(root_xz,axis=0) if len(root_xz) else np.zeros((2,),dtype=np.float64)
    root_radius = np.linalg.norm(root_xz-root_center[None],axis=-1) if len(root_xz) else np.zeros((0,),dtype=np.float64)
    root_steps = np.linalg.norm(np.diff(root_xz,axis=0),axis=-1) if len(root_xz)>1 else np.zeros((0,),dtype=np.float64)
    root_net_displacement = float(np.linalg.norm(root_xz[-1]-root_xz[0])) if len(root_xz)>1 else 0.0
    jerk_norm = np.linalg.norm(jerk,axis=-1)
    extremity_jerk = jerk_norm[:,list(EXTREMITY_JOINTS)] if jerk_norm.size else np.zeros((0,len(EXTREMITY_JOINTS)),dtype=np.float64)
    result: dict[str,Any] = {
        "schema": PHYSICAL_METRICS_SCHEMA, "frames": int(len(x)), "fps": float(fps), "duration_seconds": duration, "floor_y_m": floor_y,
        "foot_penetration_min_m": float(np.min(foot_relative_height)),
        "foot_penetration_p01_m": float(np.percentile(foot_relative_height,1)),
        "contact_ratio": float(np.mean(declared_contacts)), "foot_support_ratio": float(np.mean(support)), "static_support_ratio": float(np.mean(static_support)), "sliding_support_ratio": float(np.mean(sliding_support)),
        "support_state_contract": {"policy":policy,"states":{"swing":SWING,"static_support":STATIC_SUPPORT,"sliding_support":SLIDING_SUPPORT},"thresholds":asdict(ContactStateThresholds.from_environment()),"sliding_requires_explicit_semantic_eligibility":True,"final_fail_closed_low_height_union":policy==SUPPORT_POLICY_FINAL,"source_fast_low_foot_remains_swing":policy==SUPPORT_POLICY_SOURCE},
        "foot_contact_mismatch_ratio": float(np.mean(contact_mismatch)), "foot_support_segment_count": int(support_segment_count),
        "rot6d_nonfinite_ratio": float(np.mean(~finite_rot6d)), "rot6d_degenerate_ratio": float(np.mean(degenerate)), "rot6d_first_vector_norm_min": float(np.min(first_norm)), "rot6d_second_vector_norm_min": float(np.min(second_norm)), "rot6d_second_orthogonal_norm_min": float(np.min(second_orthogonal_norm)), "rot6d_collinearity_abs_p99": float(np.percentile(collinearity,99)),
        "rotation_near_pi_step_ratio": float(np.mean(rotation_steps >= (np.pi-0.05))) if rotation_steps.size else 0.0,
        "joint_rotation_step_window_p95_max_rad": _window_percentile_max(rotation_steps,fps=fps,seconds=1.0),
        "joint_angular_acceleration_window_p95_max_rps2": _window_percentile_max(angular_acceleration_norm,fps=fps,seconds=1.0),
        "root_y_range_m": float(np.ptp(root_y)) if root_y.size else 0.0,
        "root_y_robust_range_m": float(np.percentile(root_y,99)-np.percentile(root_y,1)) if root_y.size else 0.0,
        "joint_jerk_window_p95_max_mps3": _window_percentile_max(jerk_norm,fps=fps,seconds=1.0),
        "extremity_jerk_window_p95_max_mps3": _window_percentile_max(extremity_jerk,fps=fps,seconds=1.0),
        "root_horizontal_center_m": root_center.astype(float).tolist(), "root_horizontal_radius_p95_m": float(np.percentile(root_radius,95)) if root_radius.size else 0.0, "root_horizontal_radius_max_m": float(np.max(root_radius)) if root_radius.size else 0.0,
        "root_horizontal_net_displacement_m": root_net_displacement, "root_horizontal_path_length_m": float(np.sum(root_steps)), "root_horizontal_drift_speed_mps": root_net_displacement/max(duration,1e-8) if duration>0 else 0.0, "root_horizontal_window_seconds":10.0,
        "root_horizontal_window_displacement_max_m": _root_window_displacement_max(root_xz,fps=fps,seconds=10.0),
    }
    result.update(_penetration_robust_metrics(foot_relative_height, fps=fps))
    # Preserve the exact V2.1 diagnostic on both source/final reports.  The final
    # gate ignores it; changing/removing it would create unnecessary report drift.
    result.update(_body_normalized_root_relative_jerk_metrics(joints, fps=fps))
    if policy == SUPPORT_POLICY_SOURCE:
        result.update(
            _unit_bone_direction_jerk_metrics(
                joints,
                fps=fps,
                source_comparison_bones=source_comparison_bones,
            )
        )
    result.update(_distribution(skate,"foot_skate_mps")); result.update(_distribution(sliding_relative_speed[sliding_support],"sliding_support_relative_speed_mps")); result.update(_distribution(support_drift,"foot_support_drift_m")); result.update(_distribution(contact_height,"foot_contact_height_m"))
    result.update(_distribution(np.linalg.norm(velocity,axis=-1),"joint_velocity_mps")); result.update(_distribution(np.linalg.norm(acceleration,axis=-1),"joint_acceleration_mps2")); result.update(_distribution(jerk_norm,"joint_jerk_mps3")); result.update(_distribution(extremity_jerk,"extremity_jerk_mps3")); result.update(_distribution(root_vertical_speed,"root_vertical_speed_mps")); result.update(_distribution(rotation_steps,"joint_rotation_step_rad")); result.update(_distribution(extremity_rotation_steps,"extremity_rotation_step_rad")); result.update(_distribution(angular_acceleration_norm,"joint_angular_acceleration_rps2"))
    return result
