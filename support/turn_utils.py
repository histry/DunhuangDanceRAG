#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn-aware utilities for V22 Dunhuang ChoreoRAG.

The module keeps the EDGE 151D representation contract:
    [0:4]   foot contacts
    [4:7]   root xyz
    [7:151] 24 joints x 6D rotations

All turn analysis is based on the root joint's 6D rotation.  The vertical axis
is Y, matching EDGE/SMPL's physical coordinate system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)
ROOT_ROT6D = slice(7, 13)
FPS_DEFAULT = 30.0


@dataclass(frozen=True)
class TurnEvent:
    peak_index: int
    start: int
    end: int
    peak_speed_dps: float
    mean_speed_dps: float
    net_angle_deg: float
    path_angle_deg: float

    @property
    def duration_frames(self) -> int:
        return int(self.end - self.start + 1)

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_index": int(self.peak_index),
            "start": int(self.start),
            "end": int(self.end),
            "duration_frames": int(self.duration_frames),
            "peak_speed_dps": float(self.peak_speed_dps),
            "mean_speed_dps": float(self.mean_speed_dps),
            "net_angle_deg": float(self.net_angle_deg),
            "path_angle_deg": float(self.path_angle_deg),
            "center": float(self.center),
        }


def _moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or window <= 1:
        return x.copy()
    window = min(int(window), len(x))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return x.copy()
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def root_rotation_matrices_np(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    root6d = torch.from_numpy(x[:, ROOT_ROT6D]).float()
    with torch.no_grad():
        matrices = rotation_6d_to_matrix(root6d)
    return matrices.cpu().numpy().astype(np.float32)


def root_yaw_np(motion: np.ndarray) -> np.ndarray:
    """Return unwrapped root yaw in radians.

    EDGE/SMPL is Y-up.  We use atan2(R[0,2], R[2,2]), which measures the
    forward-axis rotation around Y.
    """
    matrices = root_rotation_matrices_np(motion)
    yaw = np.arctan2(matrices[:, 0, 2], matrices[:, 2, 2])
    return np.unwrap(yaw).astype(np.float32)


def yaw_speed_dps_np(
    motion: np.ndarray,
    fps: float = FPS_DEFAULT,
    smooth_window: int = 5,
) -> np.ndarray:
    yaw = root_yaw_np(motion)
    if len(yaw) < 2:
        return np.zeros((0,), dtype=np.float32)
    speed = np.abs(np.diff(yaw)) * float(fps) * 180.0 / np.pi
    return _moving_average(speed.astype(np.float32), smooth_window)


def detect_turn_events(
    motion: np.ndarray,
    fps: float = FPS_DEFAULT,
    min_peak_dps: float = 40.0,
    threshold_ratio: float = 0.35,
    min_gap: int = 18,
    min_duration: int = 4,
    max_events: int | None = None,
    search_start: int = 0,
    search_end: int | None = None,
) -> List[TurnEvent]:
    """Detect non-overlapping root-yaw turn events.

    Detection is intentionally conservative.  It first identifies local peaks,
    then expands each peak to the region where yaw speed remains above a
    fraction of the peak.  Nearby candidates are suppressed by peak strength.
    """
    x = np.asarray(motion, dtype=np.float32)
    speed = yaw_speed_dps_np(x, fps=fps, smooth_window=5)
    if len(speed) == 0:
        return []

    lo = int(max(1, search_start))
    hi = int(min(len(speed) - 1, search_end if search_end is not None else len(speed) - 1))
    if hi <= lo:
        return []

    candidates: List[int] = []
    for i in range(lo, hi + 1):
        left = speed[i - 1] if i - 1 >= 0 else speed[i]
        right = speed[i + 1] if i + 1 < len(speed) else speed[i]
        if speed[i] >= float(min_peak_dps) and speed[i] >= left and speed[i] >= right:
            candidates.append(i)

    # Strongest-first NMS.
    candidates.sort(key=lambda i: float(speed[i]), reverse=True)
    kept: List[int] = []
    for i in candidates:
        if all(abs(i - j) >= int(min_gap) for j in kept):
            kept.append(i)
            if max_events is not None and len(kept) >= int(max_events):
                break
    kept.sort()

    yaw = root_yaw_np(x)
    events: List[TurnEvent] = []
    for peak in kept:
        threshold = max(float(min_peak_dps) * 0.45, float(speed[peak]) * float(threshold_ratio))
        left = peak
        while left > lo and speed[left - 1] >= threshold:
            left -= 1
        right = peak
        while right < hi and right + 1 < len(speed) and speed[right + 1] >= threshold:
            right += 1

        # speed[t] describes frame t -> t+1.
        start = max(0, left - 1)
        end = min(len(x) - 1, right + 2)
        if end - start + 1 < int(min_duration):
            continue

        local_speed = speed[max(0, left) : min(len(speed), right + 1)]
        path_angle = float(np.sum(np.abs(np.diff(yaw[start : end + 1]))) * 180.0 / np.pi)
        net_angle = float(abs(yaw[end] - yaw[start]) * 180.0 / np.pi)
        events.append(
            TurnEvent(
                peak_index=int(peak),
                start=int(start),
                end=int(end),
                peak_speed_dps=float(speed[peak]),
                mean_speed_dps=float(local_speed.mean()) if len(local_speed) else float(speed[peak]),
                net_angle_deg=net_angle,
                path_angle_deg=path_angle,
            )
        )
    return events


def summarize_turns(
    motion: np.ndarray,
    fps: float = FPS_DEFAULT,
    min_peak_dps: float = 35.0,
) -> Dict[str, Any]:
    events = detect_turn_events(
        motion,
        fps=fps,
        min_peak_dps=min_peak_dps,
        threshold_ratio=0.32,
        min_gap=12,
        min_duration=3,
        max_events=None,
    )
    length = max(len(motion), 1)
    if not events:
        return {
            "contains_turn": False,
            "turn_count": 0,
            "turn_angle_deg": 0.0,
            "turn_path_angle_deg": 0.0,
            "peak_yaw_speed_dps": 0.0,
            "mean_yaw_speed_dps": 0.0,
            "turn_duration_frames": 0,
            "turn_phase_center": 0.5,
            "turn_events": [],
        }

    strongest = max(events, key=lambda e: e.peak_speed_dps)
    return {
        "contains_turn": True,
        "turn_count": int(len(events)),
        "turn_angle_deg": float(strongest.net_angle_deg),
        "turn_path_angle_deg": float(strongest.path_angle_deg),
        "peak_yaw_speed_dps": float(strongest.peak_speed_dps),
        "mean_yaw_speed_dps": float(strongest.mean_speed_dps),
        "turn_duration_frames": int(strongest.duration_frames),
        "turn_phase_center": float(strongest.center / max(length - 1, 1)),
        "turn_events": [event.to_dict() for event in events],
    }


def allowed_yaw_speed_dps(music_event: str, query: Sequence[float] | np.ndarray | None = None) -> float:
    """Music-conditioned acceptable peak turn speed.

    Values are intentionally conservative for calm/release phrases and become
    looser for accent/climax phrases.  Arousal/tension adjust the base softly.
    """
    base = {
        "calm_flow": 82.0,
        "release": 92.0,
        "neutral_flow": 108.0,
        "build_up": 124.0,
        "section_change": 132.0,
        "accent": 145.0,
        "climax": 158.0,
    }.get(str(music_event or "neutral_flow"), 108.0)

    if query is not None:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        arousal = float(q[0]) if len(q) > 0 else 0.5
        tension = float(q[4]) if len(q) > 4 else 0.5
        calm = float(q[5]) if len(q) > 5 else 0.5
        adjustment = 14.0 * (0.55 * arousal + 0.45 * tension - 0.50) - 10.0 * max(0.0, calm - 0.55)
        base += float(np.clip(adjustment, -16.0, 18.0))
    return float(np.clip(base, 65.0, 175.0))


def effective_peak_after_resample(original_peak_dps: float, time_warp_ratio: float) -> float:
    """Peak speed after resampling an event to a new duration.

    ratio = output content length / original event length.  Compression
    (ratio < 1) increases speed approximately by 1/ratio.
    """
    ratio = max(float(time_warp_ratio), 1e-6)
    return float(original_peak_dps) / ratio


def turn_speed_penalty(
    original_peak_dps: float,
    original_turn_frames: int,
    turn_angle_deg: float,
    time_warp_ratio: float,
    music_event: str,
    query: Sequence[float] | np.ndarray | None,
) -> Dict[str, float]:
    allowed = allowed_yaw_speed_dps(music_event, query)
    effective_peak = effective_peak_after_resample(original_peak_dps, time_warp_ratio)
    effective_frames = float(original_turn_frames) * max(float(time_warp_ratio), 1e-6)
    required_frames = float(abs(turn_angle_deg)) / max(allowed, 1e-6) * FPS_DEFAULT

    speed_excess = max(0.0, effective_peak / max(allowed, 1e-6) - 1.0)
    duration_deficit = max(0.0, required_frames / max(effective_frames, 1.0) - 1.0)
    penalty = speed_excess**2 + 0.65 * duration_deficit**2
    return {
        "allowed_yaw_speed_dps": float(allowed),
        "effective_peak_yaw_speed_dps": float(effective_peak),
        "effective_turn_duration_frames": float(effective_frames),
        "required_turn_duration_frames": float(required_frames),
        "turn_speed_excess": float(speed_excess),
        "turn_duration_deficit": float(duration_deficit),
        "turn_speed_penalty": float(penalty),
    }


def resample_motion_so3(motion: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    """Resample EDGE motion at arbitrary floating source positions.

    Contacts use nearest-neighbour sampling, root xyz uses linear interpolation,
    and rotations use SO(3) geodesic interpolation.
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    positions = np.asarray(source_positions, dtype=np.float32).reshape(-1)
    positions = np.clip(positions, 0.0, max(len(x) - 1, 0))
    lo = np.floor(positions).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    alpha = positions - lo.astype(np.float32)

    out = np.zeros((len(positions), 151), dtype=np.float32)
    nearest = np.rint(positions).astype(np.int64)
    out[:, CONTACT] = x[nearest, CONTACT]

    base_t = np.arange(len(x), dtype=np.float32)
    for dim in (ROOT_X, ROOT_Y, ROOT_Z):
        out[:, dim] = np.interp(positions, base_t, x[:, dim]).astype(np.float32)

    rot6d = torch.from_numpy(x[:, ROT].reshape(len(x), 24, 6)).float()
    with torch.no_grad():
        matrices = rotation_6d_to_matrix(rot6d)
        lo_t = torch.from_numpy(lo).long()
        hi_t = torch.from_numpy(hi).long()
        alpha_t = torch.from_numpy(alpha).float()
        r0 = matrices[lo_t]
        r1 = matrices[hi_t]
        relative = torch.matmul(r0.transpose(-1, -2), r1)
        axis_angle = matrix_to_axis_angle(relative)
        delta = torch.from_numpy(alpha[:, None, None]).float() * axis_angle
        from pytorch3d.transforms import axis_angle_to_matrix
        interp = torch.matmul(r0, axis_angle_to_matrix(delta))
        out[:, ROT] = matrix_to_rotation_6d(interp).reshape(len(positions), 144).cpu().numpy().astype(np.float32)
    return out


def make_fast_turn_corruption(
    target: np.ndarray,
    turn_start: int,
    turn_end: int,
    speed_factor: float,
    min_context_frames: int = 4,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compress a natural turn within a fixed-length window.

    The turn consumes fewer output frames and surrounding context consumes the
    released frames.  The output length and endpoints are unchanged.  This
    mirrors the failure observed when a dynamic unit is compressed into a short
    music phrase.
    """
    x = np.asarray(target, dtype=np.float32)
    n = len(x)
    if n < 12:
        raise ValueError("Window too short for turn corruption")
    turn_start = int(np.clip(turn_start, 1, n - 4))
    turn_end = int(np.clip(turn_end, turn_start + 2, n - 2))
    original_span = int(turn_end - turn_start)
    desired_span = max(2, int(round(original_span / max(float(speed_factor), 1.0))))

    pre_source = turn_start
    post_source = (n - 1) - turn_end
    remaining = (n - 1) - desired_span
    min_pre = min(int(min_context_frames), pre_source)
    min_post = min(int(min_context_frames), post_source)
    if remaining < min_pre + min_post:
        desired_span = max(2, (n - 1) - min_pre - min_post)
        remaining = (n - 1) - desired_span

    total_context = max(pre_source + post_source, 1)
    output_pre = int(round(remaining * pre_source / total_context))
    output_pre = int(np.clip(output_pre, min_pre, remaining - min_post))
    output_post = remaining - output_pre

    output_control = np.asarray([0, output_pre, output_pre + desired_span, n - 1], dtype=np.float32)
    source_control = np.asarray([0, turn_start, turn_end, n - 1], dtype=np.float32)
    output_frames = np.arange(n, dtype=np.float32)
    source_positions = np.interp(output_frames, output_control, source_control).astype(np.float32)
    corrupted = resample_motion_so3(x, source_positions)

    edit_start = max(0, output_pre - 5)
    edit_end = min(n - 1, output_pre + desired_span + 5)
    mask = np.zeros((n,), dtype=np.float32)
    if edit_end > edit_start:
        core = np.linspace(0.0, 1.0, edit_end - edit_start + 1, dtype=np.float32)
        soft = np.sin(np.pi * core) ** 2
        mask[edit_start : edit_end + 1] = soft

    info = {
        "original_turn_start": int(turn_start),
        "original_turn_end": int(turn_end),
        "original_turn_span": int(original_span),
        "corrupted_turn_start": int(output_pre),
        "corrupted_turn_end": int(output_pre + desired_span),
        "corrupted_turn_span": int(desired_span),
        "requested_speed_factor": float(speed_factor),
        "effective_speed_factor": float(original_span / max(desired_span, 1)),
        "source_positions": source_positions,
    }
    return corrupted.astype(np.float32), mask.astype(np.float32), info


def motion_query_from_dynamics(motion: np.ndarray) -> np.ndarray:
    """Build a weak 12D music-like condition from motion dynamics."""
    x = np.asarray(motion, dtype=np.float32)
    rot = x[:, ROT].reshape(len(x), 24, 6)
    vel = np.diff(rot, axis=0, prepend=rot[:1])
    frame = np.linalg.norm(vel.reshape(len(x), -1), axis=-1)
    energy = float(np.mean(frame))
    p90 = float(np.percentile(frame, 90)) if len(frame) else 0.0
    half = max(1, len(frame) // 3)
    start = float(frame[:half].mean())
    end = float(frame[-half:].mean())
    build = max(0.0, end - start)
    release = max(0.0, start - end)
    upper = float(np.linalg.norm(vel[:, 14:24].reshape(len(x), -1), axis=-1).mean())
    torso = float(np.linalg.norm(vel[:, 8:14].reshape(len(x), -1), axis=-1).mean())
    lower = float(np.linalg.norm(vel[:, 0:8].reshape(len(x), -1), axis=-1).mean())
    support = float(np.abs(np.diff(x[:, CONTACT], axis=0)).sum() / max(len(x) - 1, 1))
    smooth = 1.0 / (1.0 + float(np.var(frame)))
    raw = np.asarray(
        [
            energy,
            upper,
            torso,
            lower,
            float(np.std(rot)),
            smooth,
            support,
            build,
            release,
            p90 / (energy + 1e-6),
            float(np.linalg.norm(rot[-1] - rot[0])) if len(rot) else 0.0,
            len(x) / 72.0,
        ],
        dtype=np.float32,
    )
    # Robust bounded scaling for weak conditioning only.
    scales = np.asarray([0.18, 0.12, 0.08, 0.08, 0.35, 1.0, 0.15, 0.08, 0.08, 5.0, 8.0, 1.0], dtype=np.float32)
    return np.clip(raw / (scales + 1e-8), 0.0, 1.0).astype(np.float32)


def project_rot6d_np(motion: np.ndarray) -> np.ndarray:
    """Project rotation channels back onto valid SO(3) 6D representation."""
    x = np.asarray(motion, dtype=np.float32).copy()
    rot = torch.from_numpy(x[:, ROT].reshape(len(x), 24, 6)).float()
    with torch.no_grad():
        x[:, ROT] = matrix_to_rotation_6d(rotation_6d_to_matrix(rot)).reshape(len(x), 144).cpu().numpy()
    return x.astype(np.float32)


def torch_root_yaw_velocity_dps(motion: torch.Tensor, fps: float = FPS_DEFAULT) -> torch.Tensor:
    """Differentiable approximate frame-wise root yaw velocity in deg/s.

    motion: [B,T,151].  The Y component of the relative rotation axis-angle is
    used as the signed yaw increment.  This is stable for the small frame-to-
    frame rotations present in dance motion.
    """
    root6d = motion[..., ROOT_ROT6D]
    matrices = rotation_6d_to_matrix(root6d)
    relative = torch.matmul(matrices[:, :-1].transpose(-1, -2), matrices[:, 1:])
    axis_angle = matrix_to_axis_angle(relative)
    yaw_delta = axis_angle[..., 1]
    return yaw_delta * float(fps) * (180.0 / np.pi)
