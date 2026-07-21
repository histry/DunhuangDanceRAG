"""SMPL24 parameter adapter with first-class AIST++ support."""
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
AISTPLUSPLUS_ADAPTER_SCHEMA = "dunhuang_aistplusplus_smpl24_adapter_v1"


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


def load_smpl24_parameters(
    path: str | Path,
    *,
    target_fps: float = 30.0,
    source_fps: float | None = None,
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
    if poses.ndim == 2 and poses.shape[1] >= NUM_JOINTS * 3:
        rotvec = poses[:, : NUM_JOINTS * 3].reshape(len(poses), NUM_JOINTS, 3)
    elif poses.ndim == 3 and poses.shape[1:] == (NUM_JOINTS, 3):
        rotvec = poses
    else:
        raise ValueError(f"Unsupported SMPL pose shape in {p}: {poses.shape}")
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
        "skeleton_contract": skeleton_contract(),
        "contact_units": {"height": "m", "speed": "m/s", "median_window": "s"},
    }
    return motion.astype(np.float32), report
