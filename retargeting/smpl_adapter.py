"""Explicit SMPL/SMPL-X pose-layout adapter for canonical SMPL24."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from contracts.gravity import FOOT_JOINTS, fk24_np
from motion_geometry.physical import recompute_contacts_np
from motion_geometry.resampling import positions_for_fps, resample_rotations_so3_np
from motion_geometry.rotations import matrix_to_rot6d_np, so3_exp_np
from motion_geometry.smpl24 import (
    MOTION_DIM,
    NUM_JOINTS,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
    ROT6D_END,
    ROT6D_START,
    skeleton_contract,
)

AISTPLUSPLUS_SOURCE_FPS = 60.0
AISTPLUSPLUS_ADAPTER_SCHEMA = "dunhuang_smpl24_adapter_v2"
CHANG_E_POSE_LAYOUT = "smplx55_axis_angle_body22_to_smpl24_hands_zero_v1"
SMPL24_POSE_LAYOUT = "smpl24_axis_angle_v1"
CHANG_E_SOURCE_JOINTS = 55
CHANG_E_OBSERVED_BODY_JOINTS = 22


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        obj = np.load(path, allow_pickle=True)
        return {key: obj[key] for key in obj.files}
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected dict-like SMPL parameters: {path}")
    return dict(value)


def _first(data: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(np.asarray(value).reshape(-1)[0])


def _map_axis_angle_to_smpl24(
    poses: np.ndarray,
    *,
    pose_layout: str | None,
    path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map a declared source layout without silently truncating coordinates."""

    x = np.asarray(poses, dtype=np.float32)
    declared = str(pose_layout or "").strip()

    if x.ndim == 2 and x.shape[1] == CHANG_E_SOURCE_JOINTS * 3:
        inferred = CHANG_E_POSE_LAYOUT
        source = x.reshape(len(x), CHANG_E_SOURCE_JOINTS, 3)
        ignored = source[:, CHANG_E_OBSERVED_BODY_JOINTS:]
        ignored_max = float(np.max(np.abs(ignored))) if ignored.size else 0.0
        if declared and declared != inferred:
            raise ValueError(
                f"Pose-layout mismatch in {path}: declared={declared!r}, "
                f"shape={x.shape} implies {inferred!r}"
            )
        if ignored_max > 1.0e-6:
            raise ValueError(
                f"Chang-E 165D layout has non-zero unobserved hand/face joints "
                f"in {path}: max_abs={ignored_max:.6g}"
            )
        mapped = np.zeros((len(x), NUM_JOINTS, 3), dtype=np.float32)
        mapped[:, :CHANG_E_OBSERVED_BODY_JOINTS] = source[
            :, :CHANG_E_OBSERVED_BODY_JOINTS
        ]
        return mapped, {
            "pose_layout": inferred,
            "source_pose_shape": [int(v) for v in x.shape],
            "source_joint_count": CHANG_E_SOURCE_JOINTS,
            "observed_body_joint_count": CHANG_E_OBSERVED_BODY_JOINTS,
            "canonical_joint_count": NUM_JOINTS,
            "canonical_hand_joint_indices": [22, 23],
            "hand_rotation_policy": "zero_unobserved",
            "unobserved_joint_max_abs": ignored_max,
        }

    if (
        (x.ndim == 2 and x.shape[1] == NUM_JOINTS * 3)
        or (x.ndim == 3 and x.shape[1:] == (NUM_JOINTS, 3))
    ):
        inferred = SMPL24_POSE_LAYOUT
        if declared and declared != inferred:
            raise ValueError(
                f"Pose-layout mismatch in {path}: declared={declared!r}, "
                f"shape={x.shape} implies {inferred!r}"
            )
        mapped = x.reshape(len(x), NUM_JOINTS, 3)
        return mapped, {
            "pose_layout": inferred,
            "source_pose_shape": [int(v) for v in x.shape],
            "source_joint_count": NUM_JOINTS,
            "observed_body_joint_count": NUM_JOINTS,
            "canonical_joint_count": NUM_JOINTS,
            "canonical_hand_joint_indices": [22, 23],
            "hand_rotation_policy": "observed",
            "unobserved_joint_max_abs": 0.0,
        }

    raise ValueError(
        f"Unsupported SMPL pose shape/layout in {path}: "
        f"shape={x.shape}, declared_layout={declared or None!r}"
    )


def load_smpl24_parameters(
    path: str | Path,
    *,
    target_fps: float = 30.0,
    source_fps: float | None = None,
    pose_layout: str | None = None,
    coordinate_system: str = "y_up",
    translation_units: str = "m",
    scaling_mode: str = "canonical_body",
    localize_root_xz: bool = True,
    contact_height_m: float = 0.055,
    contact_speed_mps: float = 0.75,
    contact_median_seconds: float = 1.0 / 6.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert SMPL axis-angle parameters to canonical EDGE151.

    AIST++ fields ``smpl_poses``, ``smpl_trans`` and ``smpl_scaling`` are
    recognized directly.  ``canonical_body`` intentionally records but does
    not apply the fitted body scaling: all sources are mapped to the one fixed
    SMPL24 skeleton, while world-space root translation remains unchanged.
    ``scale_translation`` and ``inverse_scale_translation`` are explicit
    compatibility modes for non-canonical exports.
    """
    p = Path(path)
    data = _load_mapping(p)
    is_aistplusplus = "smpl_poses" in data
    poses_value = _first(data, ("smpl_poses", "poses", "pose", "smpl_pose", "body_pose", "full_pose"))
    trans_value = _first(data, ("smpl_trans", "trans", "transl", "translations", "root_translation", "root_trans"))
    if poses_value is None:
        raise ValueError(f"No SMPL pose field in {p}; keys={sorted(data.keys())}")
    poses = np.asarray(poses_value, dtype=np.float32)
    rotvec, pose_contract = _map_axis_angle_to_smpl24(
        poses,
        pose_layout=pose_layout,
        path=p,
    )
    normalized_coordinate_system = str(coordinate_system).strip().lower()
    if normalized_coordinate_system != "y_up":
        raise ValueError(
            f"Unsupported coordinate_system={coordinate_system!r} in {p}; "
            "convert explicitly to y_up before canonical EDGE151 adaptation"
        )
    normalized_translation_units = str(translation_units).strip().lower()
    if normalized_translation_units != "m":
        raise ValueError(
            f"Unsupported translation_units={translation_units!r} in {p}; "
            "convert explicitly to metres before canonical EDGE151 adaptation"
        )
    if trans_value is None:
        translation = np.zeros((len(rotvec), 3), dtype=np.float32)
    else:
        translation = np.asarray(trans_value, dtype=np.float32).reshape(len(rotvec), 3)

    scaling = _scalar(_first(data, ("smpl_scaling", "scaling", "scale")), 1.0)
    if not np.isfinite(scaling) or scaling <= 0.0:
        raise ValueError(f"Invalid smpl_scaling={scaling!r} in {p}")
    normalized_mode = str(scaling_mode).strip().lower()
    if normalized_mode == "scale_translation":
        translation = translation * scaling
    elif normalized_mode == "inverse_scale_translation":
        translation = translation / scaling
    elif normalized_mode != "canonical_body":
        raise ValueError(
            "scaling_mode must be canonical_body, scale_translation, or inverse_scale_translation"
        )

    fps_value = _first(data, ("mocap_framerate", "fps", "frame_rate", "framerate"))
    inferred_fps = AISTPLUSPLUS_SOURCE_FPS if is_aistplusplus else 30.0
    src_fps = float(source_fps) if source_fps is not None else _scalar(fps_value, inferred_fps)
    if src_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("source_fps and target_fps must be positive")

    matrices = so3_exp_np(rotvec)
    if abs(src_fps - float(target_fps)) > 1.0e-8:
        positions = positions_for_fps(len(rotvec), src_fps, float(target_fps))
        matrices = resample_rotations_so3_np(matrices, positions)
        source_axis = np.arange(len(translation), dtype=np.float32)
        translation = np.stack(
            [np.interp(positions, source_axis, translation[:, dim]) for dim in range(3)],
            axis=-1,
        ).astype(np.float32)

    motion = np.zeros((len(matrices), MOTION_DIM), dtype=np.float32)
    motion[:, 4:7] = translation
    motion[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(matrices).reshape(len(matrices), -1)
    if localize_root_xz and len(motion):
        motion[:, ROOT_X_IDX] -= motion[0, ROOT_X_IDX]
        motion[:, ROOT_Z_IDX] -= motion[0, ROOT_Z_IDX]
    joints = fk24_np(motion)
    floor_y = float(np.percentile(joints[:, list(FOOT_JOINTS), 1], 5))
    motion[:, ROOT_Y_IDX] -= floor_y
    motion = recompute_contacts_np(
        motion,
        fps=float(target_fps),
        height_margin_m=contact_height_m,
        speed_gate_mps=contact_speed_mps,
        median_seconds=contact_median_seconds,
    )

    report = {
        "schema": AISTPLUSPLUS_ADAPTER_SCHEMA,
        "source": str(p),
        "source_format": "aistplusplus_smpl" if is_aistplusplus else "smpl_parameters",
        "source_fps": float(src_fps),
        "target_fps": float(target_fps),
        "source_frames": int(len(poses)),
        "target_frames": int(len(motion)),
        "duration_seconds": float((len(poses) - 1) / src_fps) if len(poses) > 1 else 0.0,
        "smpl_scaling": float(scaling),
        "smpl_scaling_mode": normalized_mode,
        "gender": str(data.get("gender", "neutral")),
        "betas_present": bool(_first(data, ("smpl_betas", "betas", "shape")) is not None),
        **pose_contract,
        "coordinate_system": normalized_coordinate_system,
        "translation_units": normalized_translation_units,
        "skeleton_contract": skeleton_contract(),
        "contact_units": {"height": "m", "speed": "m/s", "median_window": "s"},
    }
    return motion.astype(np.float32), report
