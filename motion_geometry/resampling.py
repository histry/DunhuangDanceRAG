"""Frame-rate conversion for EDGE151 using physical channel semantics."""
from __future__ import annotations

from typing import Optional

import numpy as np

from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    relative_rotvec_np,
    rot6d_to_matrix_np,
    so3_exp_np,
)
from motion_geometry.smpl24 import (
    CONTACT,
    MOTION_DIM,
    NUM_JOINTS,
    ROOT,
    ROT6D_END,
    ROT6D_START,
)


def frame_positions(source_frames: int, target_frames: int) -> np.ndarray:
    if source_frames < 1 or target_frames < 1:
        raise ValueError("source_frames and target_frames must be positive")
    if target_frames == 1:
        return np.zeros((1,), dtype=np.float32)
    return np.linspace(0.0, float(source_frames - 1), target_frames, dtype=np.float32)


def target_frame_count(source_frames: int, source_fps: float, target_fps: float) -> int:
    if source_frames < 1:
        raise ValueError("source_frames must be positive")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("FPS values must be positive")
    if source_frames == 1:
        return 1
    duration_s = (source_frames - 1) / float(source_fps)
    return max(2, int(round(duration_s * float(target_fps))) + 1)


def positions_for_fps(source_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    count = target_frame_count(source_frames, source_fps, target_fps)
    source_times = np.arange(source_frames, dtype=np.float64) / float(source_fps)
    target_times = np.arange(count, dtype=np.float64) / float(target_fps)
    target_times = np.minimum(target_times, source_times[-1])
    return (target_times * float(source_fps)).astype(np.float32)


def resample_rotations_so3_np(rotations: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate rotations along the local SO(3) shortest path."""
    r = np.asarray(rotations, dtype=np.float32)
    p = np.asarray(positions, dtype=np.float32).reshape(-1)
    if r.ndim < 3 or r.shape[-2:] != (3, 3) or r.shape[0] < 1:
        raise ValueError(f"Expected [T,...,3,3], got {r.shape}")
    p = np.clip(p, 0.0, float(r.shape[0] - 1))
    if r.shape[0] == 1:
        return np.repeat(r, len(p), axis=0)
    lo = np.floor(p).astype(np.int64)
    hi = np.minimum(lo + 1, r.shape[0] - 1)
    alpha = p - lo.astype(np.float32)
    base = r[lo]
    tangent = relative_rotvec_np(base, r[hi])
    weight = alpha.reshape((len(alpha),) + (1,) * (tangent.ndim - 1))
    return (base @ so3_exp_np(tangent * weight)).astype(np.float32)


def resample_edge151_np(
    motion: np.ndarray,
    *,
    target_frames: Optional[int] = None,
    positions: Optional[np.ndarray] = None,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
) -> np.ndarray:
    """Resample EDGE151 by channel type.

    Contacts use nearest-neighbour sampling, root translation uses linear
    interpolation in R3, and every joint rotation is interpolated on SO(3).
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != MOTION_DIM or x.shape[0] < 1:
        raise ValueError(f"Expected [T,{MOTION_DIM}], got {x.shape}")
    if positions is None:
        if source_fps is not None or target_fps is not None:
            if source_fps is None or target_fps is None:
                raise ValueError("source_fps and target_fps must be supplied together")
            positions = positions_for_fps(len(x), source_fps, target_fps)
        elif target_frames is not None:
            positions = frame_positions(len(x), int(target_frames))
        else:
            raise ValueError("Provide positions, target_frames, or source_fps/target_fps")
    p = np.asarray(positions, dtype=np.float32).reshape(-1)
    p = np.clip(p, 0.0, float(len(x) - 1))
    out = np.zeros((len(p), MOTION_DIM), dtype=np.float32)
    nearest = np.rint(p).astype(np.int64)
    out[:, CONTACT] = (x[nearest, CONTACT] >= 0.5).astype(np.float32)
    source_positions = np.arange(len(x), dtype=np.float32)
    for dim in range(ROOT.start, ROOT.stop):
        out[:, dim] = np.interp(p, source_positions, x[:, dim]).astype(np.float32)
    rotations = rot6d_to_matrix_np(
        x[:, ROT6D_START:ROT6D_END].reshape(len(x), NUM_JOINTS, 6)
    )
    interpolated = resample_rotations_so3_np(rotations, p)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(interpolated).reshape(len(p), -1)
    return out
