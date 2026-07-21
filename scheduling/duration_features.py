#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for V23-v2 natural-duration supervision and runtime.

The central design rule is that every feature used at training time must also be
available at inference time.  In particular, conditions are built from the
observed/corrupted motion only; no target-motion dynamics are leaked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    matrix_to_rot6d_np,
    relative_rotvec_np,
    rot6d_to_matrix_np,
    so3_exp_np,
)

from motion_geometry.heading import (
    CONTACT,
    ROT,
    ROOT_ROT6D,
    motion_query_from_dynamics,
    root_yaw_np,
    yaw_speed_dps_np,
)


ROOT = slice(4, 7)


@dataclass(frozen=True)
class NaturalTurnEvent:
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
        return 0.5 * float(self.start + self.end)

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


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
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
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid").astype(np.float32)


def _close_small_false_gaps(active: np.ndarray, max_gap: int = 2) -> np.ndarray:
    out = np.asarray(active, dtype=bool).copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= int(max_gap):
            out[i:j] = True
        i = j
    return out


def _expand_to_min_duration(start: int, end: int, peak: int, n_frames: int, minimum: int) -> Tuple[int, int]:
    start = int(start)
    end = int(end)
    minimum = int(max(3, minimum))
    while end - start + 1 < minimum:
        can_left = start > 0
        can_right = end < n_frames - 1
        if not can_left and not can_right:
            break
        if can_left and can_right:
            if (peak - start) <= (end - peak):
                start -= 1
            else:
                end += 1
        elif can_left:
            start -= 1
        else:
            end += 1
    return start, end


def _cap_interval_around_peak(start: int, end: int, peak: int, n_frames: int, maximum: int) -> Tuple[int, int]:
    maximum = int(max(3, maximum))
    if end - start + 1 <= maximum:
        return int(start), int(end)
    half_left = maximum // 2
    capped_start = int(peak - half_left)
    capped_end = int(capped_start + maximum - 1)
    if capped_start < 0:
        capped_start = 0
        capped_end = maximum - 1
    if capped_end >= n_frames:
        capped_end = n_frames - 1
        capped_start = max(0, capped_end - maximum + 1)
    return capped_start, capped_end


def _cumulative_angle_crop(
    yaw: np.ndarray,
    start: int,
    end: int,
    low_fraction: float,
    high_fraction: float,
) -> Tuple[int, int]:
    segment = np.asarray(yaw[start : end + 1], dtype=np.float32)
    if len(segment) < 4:
        return int(start), int(end)
    step = np.abs(np.diff(segment))
    total = float(step.sum())
    if total <= 1e-7:
        return int(start), int(end)
    cumulative = np.concatenate([[0.0], np.cumsum(step)]).astype(np.float32)
    lo_value = float(np.clip(low_fraction, 0.0, 0.45)) * total
    hi_value = float(np.clip(high_fraction, 0.55, 1.0)) * total
    local_start = int(np.searchsorted(cumulative, lo_value, side="left"))
    local_end = int(np.searchsorted(cumulative, hi_value, side="left"))
    local_end = max(local_start + 2, min(local_end, len(segment) - 1))
    return int(start + local_start), int(start + local_end)



def signed_yaw_speed_dps_np(
    motion: np.ndarray,
    fps: float = 30.0,
    smooth_window: int = 5,
    rot6d_layout: str = CANONICAL_ROT6D_LAYOUT,
) -> np.ndarray:
    """Signed root-yaw velocity in degrees per second."""
    yaw = root_yaw_np(
        np.asarray(motion, dtype=np.float32),
        rot6d_layout=rot6d_layout,
    )
    if len(yaw) < 2:
        return np.zeros((0,), dtype=np.float32)
    delta = np.diff(yaw).astype(np.float32)
    speed = delta * float(fps) * (180.0 / np.pi)
    return moving_average(speed, smooth_window)


def _robust_unit_scale(values: np.ndarray, low: float = 10.0, high: float = 90.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if len(x) == 0:
        return x.copy()
    lo = float(np.percentile(x, low))
    hi = float(np.percentile(x, high))
    if hi - lo < 1e-7:
        maximum = float(np.max(np.abs(x)))
        if maximum < 1e-7:
            return np.zeros_like(x)
        return np.clip(x / maximum, 0.0, 1.0).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def multi_scale_pose_progress(
    motion: np.ndarray,
    span: int = 10,
) -> np.ndarray:
    """Measure slow pose progression over a wider temporal baseline.

    Frame velocity alone treats Dunhuang slow motion as nearly static.  This
    descriptor compares poses several frames apart, so a slow but continuously
    unfolding arm/torso trajectory remains active while a genuine hold becomes
    quiet.
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    t = len(x)
    if t < 2:
        return np.zeros((t,), dtype=np.float32)
    span = int(max(2, min(span, max(2, (t - 1) // 2))))
    rot = x[:, ROT].reshape(t, 24, 6)
    result = np.zeros((t,), dtype=np.float32)
    for index in range(t):
        left = max(0, index - span)
        right = min(t - 1, index + span)
        distance = max(right - left, 1)
        pose_delta = np.linalg.norm(rot[right] - rot[left], axis=-1).mean()
        root_delta = np.linalg.norm(x[right, ROOT] - x[left, ROOT])
        result[index] = float(pose_delta / distance + 0.35 * root_delta / distance)
    return moving_average(result, max(3, span // 2 * 2 + 1))


def full_body_activity_envelope(
    motion: np.ndarray,
    fps: float = 30.0,
    smooth_window: int = 9,
    slow_pose_span: int = 10,
    rot6d_layout: str = CANONICAL_ROT6D_LAYOUT,
) -> np.ndarray:
    """Slow-motion-aware full-body phase activity.

    The envelope combines short-term joint velocity with multi-scale pose
    progression.  Therefore a slowly unfolding Dunhuang movement stays active,
    while a true pose hold, a support pause, or a direction-change valley can
    still become a valid split point.
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    t = len(x)
    if t < 2:
        return np.zeros((t,), dtype=np.float32)

    rot = x[:, ROT].reshape(t, 24, 6)
    rot_velocity = np.zeros((t, 24), dtype=np.float32)
    rot_velocity[1:] = np.linalg.norm(np.diff(rot, axis=0), axis=-1)
    rot_velocity[0] = rot_velocity[1]

    lower_ids = np.asarray([1, 2, 4, 5, 7, 8, 10, 11], dtype=np.int64)
    torso_ids = np.asarray([0, 3, 6, 9, 12, 13, 14], dtype=np.int64)
    upper_ids = np.asarray([15, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int64)
    lower = rot_velocity[:, lower_ids].mean(axis=1)
    torso = rot_velocity[:, torso_ids].mean(axis=1)
    upper = rot_velocity[:, upper_ids].mean(axis=1)
    short_body = 0.28 * lower + 0.30 * torso + 0.42 * upper

    slow_progress = multi_scale_pose_progress(x, span=slow_pose_span)

    root_velocity = np.zeros((t,), dtype=np.float32)
    root_velocity[1:] = np.linalg.norm(np.diff(x[:, ROOT], axis=0), axis=-1)
    root_velocity[0] = root_velocity[1]

    contact_change = np.zeros((t,), dtype=np.float32)
    contact_change[1:] = np.abs(np.diff(x[:, CONTACT], axis=0)).mean(axis=1)
    contact_change[0] = contact_change[1]

    signed_yaw = signed_yaw_speed_dps_np(
        x,
        fps=fps,
        smooth_window=9,
        rot6d_layout=rot6d_layout,
    )
    yaw_frame = np.zeros((t,), dtype=np.float32)
    if len(signed_yaw):
        yaw_frame[1:] = np.abs(signed_yaw)
        yaw_frame[0] = yaw_frame[1]

    combined = (
        0.30 * _robust_unit_scale(short_body)
        + 0.27 * _robust_unit_scale(slow_progress)
        + 0.24 * _robust_unit_scale(yaw_frame)
        + 0.11 * _robust_unit_scale(root_velocity)
        + 0.08 * _robust_unit_scale(contact_change)
    )
    return moving_average(combined.astype(np.float32), smooth_window)



def _search_phrase_boundary(
    active: np.ndarray,
    signed_yaw_frame: np.ndarray,
    peak_frame: int,
    dominant_sign: float,
    lo: int,
    hi: int,
    quiet_run: int,
    opposite_run: int,
    direction: int,
) -> int:
    last_active = int(peak_frame)
    quiet = 0
    opposite = 0
    index = int(peak_frame) + int(direction)
    while lo <= index <= hi:
        if active[index]:
            last_active = index
            quiet = 0
        else:
            quiet += 1

        signed_value = float(signed_yaw_frame[index])
        if dominant_sign != 0.0 and signed_value * dominant_sign < -1e-6:
            opposite += 1
        else:
            opposite = 0

        if quiet >= int(quiet_run) or opposite >= int(opposite_run):
            break
        index += int(direction)
    return int(last_active)


def _activity_cumulative_crop(
    activity: np.ndarray,
    start: int,
    end: int,
    low_fraction: float,
    high_fraction: float,
    margin: int,
) -> Tuple[int, int]:
    segment = np.asarray(activity[start : end + 1], dtype=np.float32)
    if len(segment) < 5:
        return int(start), int(end)
    baseline = float(np.percentile(segment, 10.0))
    weights = np.maximum(segment - baseline, 0.0) + 1e-4
    cumulative = np.cumsum(weights)
    total = float(cumulative[-1])
    if total <= 1e-7:
        return int(start), int(end)
    lo_value = float(np.clip(low_fraction, 0.0, 0.40)) * total
    hi_value = float(np.clip(high_fraction, 0.60, 1.0)) * total
    local_start = int(np.searchsorted(cumulative, lo_value, side="left"))
    local_end = int(np.searchsorted(cumulative, hi_value, side="left"))
    local_start = max(0, local_start - int(margin))
    local_end = min(len(segment) - 1, local_end + int(margin))
    local_end = max(local_start + 2, local_end)
    return int(start + local_start), int(start + local_end)



def _local_maxima(values: np.ndarray, minimum: float) -> List[int]:
    x = np.asarray(values, dtype=np.float32)
    result: List[int] = []
    for index in range(1, len(x) - 1):
        if x[index] >= float(minimum) and x[index] >= x[index - 1] and x[index] >= x[index + 1]:
            result.append(index)
    return result


def _slow_angle_candidates(
    yaw: np.ndarray,
    window: int,
    minimum_angle_deg: float,
) -> Tuple[List[int], np.ndarray]:
    """Find slow turns using accumulated yaw instead of instantaneous speed."""
    yaw = np.asarray(yaw, dtype=np.float32)
    n = len(yaw)
    window = int(max(6, min(window, max(6, n - 1))))
    half = window // 2
    score = np.zeros((n,), dtype=np.float32)
    for index in range(n):
        left = max(0, index - half)
        right = min(n - 1, index + half)
        if right <= left:
            continue
        segment = yaw[left : right + 1]
        path = float(np.sum(np.abs(np.diff(segment))) * 180.0 / np.pi)
        net = float(abs(segment[-1] - segment[0]) * 180.0 / np.pi)
        # Prefer coherent progression, but retain expressive turns with small
        # counter-motion in the arms/torso preparation phase.
        score[index] = 0.65 * net + 0.35 * path
    candidates = _local_maxima(score, max(4.0, 0.75 * float(minimum_angle_deg)))
    return candidates, score


def _nms_candidates(
    candidates: Sequence[int],
    scores: np.ndarray,
    min_gap: int,
    maximum: int | None,
) -> List[int]:
    order = sorted(set(int(i) for i in candidates), key=lambda i: float(scores[i]), reverse=True)
    kept: List[int] = []
    for index in order:
        if all(abs(index - other) >= int(min_gap) for other in kept):
            kept.append(index)
            if maximum is not None and len(kept) >= int(maximum):
                break
    return sorted(kept)


def _mean_near(values: np.ndarray, center: int, radius: int) -> float:
    lo = max(0, int(center) - int(radius))
    hi = min(len(values), int(center) + int(radius) + 1)
    if hi <= lo:
        return float(values[int(np.clip(center, 0, len(values) - 1))])
    return float(np.mean(values[lo:hi]))


def _integrated_signed_angle(
    signed_speed: np.ndarray,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> float:
    lo = max(0, int(start_frame))
    hi = min(len(signed_speed), max(lo, int(end_frame)))
    if hi <= lo:
        return 0.0
    return float(np.sum(signed_speed[lo:hi]) / max(float(fps), 1e-6))


def _best_slow_aware_split(
    start: int,
    end: int,
    activity: np.ndarray,
    slow_progress: np.ndarray,
    yaw_abs_frame: np.ndarray,
    signed_speed: np.ndarray,
    fps: float,
    min_duration: int,
    valley_radius: int,
    reversal_angle_deg: float,
    secondary_peak_ratio: float,
) -> Tuple[int | None, float, bool]:
    """Return the best semantic phase boundary inside an interval.

    A valid split is supported by one or more of:
    - a sustained full-body / slow-progress valley;
    - a root-yaw direction reversal with meaningful angle on both sides;
    - two strong yaw phases separated by a genuine valley.
    """
    first = int(start + min_duration)
    last = int(end - min_duration)
    if last <= first:
        return None, -1.0, False

    interval_activity = activity[start : end + 1]
    interval_slow = slow_progress[start : end + 1]
    interval_yaw = yaw_abs_frame[start : end + 1]
    act_scale = max(float(np.percentile(interval_activity, 75.0)), 1e-6)
    slow_scale = max(float(np.percentile(interval_slow, 75.0)), 1e-6)
    yaw_scale = max(float(np.percentile(interval_yaw, 85.0)), 1e-6)
    global_peak = max(float(np.max(interval_yaw)), 1e-6)

    best_index: int | None = None
    best_score = -1.0
    best_structural = False
    for cut in range(first, last + 1):
        local_activity = _mean_near(activity, cut, valley_radius)
        local_slow = _mean_near(slow_progress, cut, valley_radius)
        local_yaw = _mean_near(yaw_abs_frame, cut, valley_radius)

        activity_valley = 1.0 - float(np.clip(local_activity / act_scale, 0.0, 1.0))
        slow_valley = 1.0 - float(np.clip(local_slow / slow_scale, 0.0, 1.0))
        yaw_valley = 1.0 - float(np.clip(local_yaw / yaw_scale, 0.0, 1.0))

        left_peak = float(np.max(yaw_abs_frame[start : cut + 1]))
        right_peak = float(np.max(yaw_abs_frame[cut : end + 1]))
        dual_peak = (
            left_peak >= float(secondary_peak_ratio) * global_peak
            and right_peak >= float(secondary_peak_ratio) * global_peak
        )

        look = max(int(min_duration), 12)
        left_angle = _integrated_signed_angle(signed_speed, max(start, cut - look), cut, fps)
        right_angle = _integrated_signed_angle(signed_speed, cut, min(end, cut + look), fps)
        reversal = (
            left_angle * right_angle < 0.0
            and abs(left_angle) >= float(reversal_angle_deg)
            and abs(right_angle) >= float(reversal_angle_deg)
        )

        valley_support = 0.34 * activity_valley + 0.30 * slow_valley + 0.24 * yaw_valley
        structural = bool(
            reversal
            or (
                dual_peak
                and activity_valley >= 0.25
                and slow_valley >= 0.20
                and yaw_valley >= 0.20
            )
        )
        score = valley_support + (0.32 if reversal else 0.0) + (0.14 if dual_peak else 0.0)
        if score > best_score:
            best_score = float(score)
            best_index = int(cut)
            best_structural = structural
    return best_index, best_score, best_structural


def _recursive_phase_split(
    start: int,
    end: int,
    activity: np.ndarray,
    slow_progress: np.ndarray,
    yaw_abs_frame: np.ndarray,
    signed_speed: np.ndarray,
    fps: float,
    min_duration: int,
    max_duration: int,
    valley_radius: int,
    reversal_angle_deg: float,
    secondary_peak_ratio: float,
    split_score_threshold: float,
    long_split_score_threshold: float,
    depth: int = 0,
) -> List[Tuple[int, int]]:
    duration = int(end - start + 1)
    if duration < int(min_duration):
        return []
    if depth >= 8:
        return [(int(start), int(end))] if duration <= int(max_duration) else []

    cut, score, structural = _best_slow_aware_split(
        start,
        end,
        activity,
        slow_progress,
        yaw_abs_frame,
        signed_speed,
        fps,
        min_duration,
        valley_radius,
        reversal_angle_deg,
        secondary_peak_ratio,
    )
    must_split = duration > int(max_duration)
    should_split = bool(
        cut is not None
        and (
            (must_split and score >= float(long_split_score_threshold))
            or (structural and score >= float(split_score_threshold))
        )
    )
    if not should_split:
        # Never cap a long event to the maximum label.  A slow single-phase event
        # that exceeds the model capacity is excluded and can later be handled by
        # a longer-window model.
        return [(int(start), int(end))] if not must_split else []

    left = _recursive_phase_split(
        start, cut, activity, slow_progress, yaw_abs_frame, signed_speed, fps,
        min_duration, max_duration, valley_radius, reversal_angle_deg,
        secondary_peak_ratio, split_score_threshold, long_split_score_threshold,
        depth + 1,
    )
    right = _recursive_phase_split(
        cut + 1, end, activity, slow_progress, yaw_abs_frame, signed_speed, fps,
        min_duration, max_duration, valley_radius, reversal_angle_deg,
        secondary_peak_ratio, split_score_threshold, long_split_score_threshold,
        depth + 1,
    )
    return left + right


def detect_natural_turn_events(
    motion: np.ndarray,
    fps: float = 30.0,
    min_peak_dps: float = 14.0,
    min_turn_angle_deg: float = 10.0,
    min_gap: int = 16,
    min_duration: int = 12,
    max_duration: int = 88,
    threshold_ratio: float = 0.10,
    cumulative_low: float = 0.03,
    cumulative_high: float = 0.97,
    max_events: int | None = None,
    activity_threshold_ratio: float = 0.22,
    boundary_yaw_ratio: float = 0.04,
    quiet_run: int = 8,
    opposite_run: int = 4,
    phrase_margin: int = 3,
    slow_pose_span: int = 10,
    slow_angle_window: int = 24,
    search_duration_multiplier: float = 1.80,
    split_valley_radius: int = 3,
    reversal_angle_deg: float = 7.0,
    secondary_peak_ratio: float = 0.48,
    split_score_threshold: float = 0.68,
    long_split_score_threshold: float = 0.42,
    min_direction_consistency: float = 0.18,
    rot6d_layout: str = CANONICAL_ROT6D_LAYOUT,
    smooth_window: int = 9,
    minimum_input_frames: int = 24,
) -> List[NaturalTurnEvent]:
    """Detect slow, complete and phase-consistent Dunhuang turn events.

    Key differences from V23-v2.1:
    1. slow turns are proposed by accumulated yaw over a long window, not only
       by instantaneous high-speed peaks;
    2. multi-scale pose progression keeps slow preparation and recovery active;
    3. broad intervals are recursively split at sustained valleys, direction
       reversals, or dual-peak phase boundaries;
    4. unsplittable events longer than ``max_duration`` are rejected instead of
       being cropped to a constant maximum-duration label.
    """
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 151 or len(x) < max(int(minimum_input_frames), min_duration + 6):
        return []

    yaw = root_yaw_np(x, rot6d_layout=rot6d_layout)
    signed_speed = signed_yaw_speed_dps_np(
        x,
        fps=fps,
        smooth_window=max(1, int(smooth_window)),
        rot6d_layout=rot6d_layout,
    )
    absolute_speed = np.abs(signed_speed)
    if len(absolute_speed) < 3:
        return []

    n_frames = len(x)
    yaw_abs_frame = np.zeros((n_frames,), dtype=np.float32)
    signed_yaw_frame = np.zeros((n_frames,), dtype=np.float32)
    yaw_abs_frame[1:] = absolute_speed
    signed_yaw_frame[1:] = signed_speed
    yaw_abs_frame[0] = yaw_abs_frame[1]
    signed_yaw_frame[0] = signed_yaw_frame[1]

    activity = full_body_activity_envelope(
        x,
        fps=fps,
        smooth_window=max(1, int(smooth_window)),
        slow_pose_span=slow_pose_span,
        rot6d_layout=rot6d_layout,
    )
    slow_progress = multi_scale_pose_progress(x, span=slow_pose_span)

    speed_peaks = _local_maxima(absolute_speed, float(min_peak_dps))
    slow_candidates, slow_scores = _slow_angle_candidates(
        yaw,
        window=slow_angle_window,
        minimum_angle_deg=min_turn_angle_deg,
    )
    candidate_scores = np.zeros((n_frames,), dtype=np.float32)
    for transition in speed_peaks:
        frame = min(n_frames - 1, int(transition) + 1)
        candidate_scores[frame] = max(candidate_scores[frame], float(absolute_speed[transition]))
    for frame in slow_candidates:
        candidate_scores[frame] = max(
            candidate_scores[frame],
            float(min_peak_dps) + float(slow_scores[frame]),
        )
    candidate_frames = [min(n_frames - 1, int(index) + 1) for index in speed_peaks] + slow_candidates
    kept = _nms_candidates(candidate_frames, candidate_scores, min_gap=min_gap, maximum=max_events)

    broad_intervals: List[Tuple[int, int]] = []
    search_radius = int(max(max_duration, round(max_duration * float(search_duration_multiplier) / 2.0)))
    for peak_frame in kept:
        lo = max(0, int(peak_frame) - search_radius)
        hi = min(n_frames - 1, int(peak_frame) + search_radius)
        local_activity = activity[lo : hi + 1]
        activity_floor = float(np.percentile(local_activity, 15.0))
        activity_high = float(np.percentile(local_activity, 85.0))
        activity_threshold = activity_floor + float(activity_threshold_ratio) * max(
            activity_high - activity_floor, 1e-6
        )
        local_slow = slow_progress[lo : hi + 1]
        slow_floor = float(np.percentile(local_slow, 15.0))
        slow_high = float(np.percentile(local_slow, 85.0))
        slow_threshold = slow_floor + 0.20 * max(slow_high - slow_floor, 1e-6)

        peak_yaw = float(yaw_abs_frame[peak_frame])
        yaw_threshold = max(
            1.5,
            float(min_peak_dps) * 0.10,
            peak_yaw * max(float(boundary_yaw_ratio), float(threshold_ratio) * 0.30),
        )
        active = (
            (activity >= activity_threshold)
            | (slow_progress >= slow_threshold)
            | (yaw_abs_frame >= yaw_threshold)
        )
        active = _close_small_false_gaps(active, max_gap=max(2, int(quiet_run) // 2))

        speed_index = int(np.clip(peak_frame - 1, 0, len(signed_speed) - 1))
        dominant_sign = float(np.sign(signed_speed[speed_index]))
        left = _search_phrase_boundary(
            active, signed_yaw_frame, peak_frame, dominant_sign, lo, hi,
            quiet_run=quiet_run, opposite_run=opposite_run, direction=-1,
        )
        right = _search_phrase_boundary(
            active, signed_yaw_frame, peak_frame, dominant_sign, lo, hi,
            quiet_run=quiet_run, opposite_run=opposite_run, direction=1,
        )
        start = max(lo, left - int(phrase_margin))
        end = min(hi, right + int(phrase_margin))
        start, end = _activity_cumulative_crop(
            activity,
            start,
            end,
            low_fraction=cumulative_low,
            high_fraction=cumulative_high,
            margin=max(1, int(phrase_margin) // 2),
        )
        if end - start + 1 >= int(min_duration):
            broad_intervals.append((int(start), int(end)))

    # Merge only heavily overlapping proposals before semantic phase splitting.
    broad_intervals.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in broad_intervals:
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        intersection = max(0, min(end, previous_end) - max(start, previous_start) + 1)
        smaller = max(1, min(end - start + 1, previous_end - previous_start + 1))
        if intersection / smaller >= 0.65:
            merged[-1] = (min(start, previous_start), max(end, previous_end))
        else:
            merged.append((start, end))

    split_intervals: List[Tuple[int, int]] = []
    for start, end in merged:
        split_intervals.extend(
            _recursive_phase_split(
                start,
                end,
                activity,
                slow_progress,
                yaw_abs_frame,
                signed_speed,
                fps,
                min_duration,
                max_duration,
                split_valley_radius,
                reversal_angle_deg,
                secondary_peak_ratio,
                split_score_threshold,
                long_split_score_threshold,
            )
        )

    events: List[NaturalTurnEvent] = []
    for start, end in split_intervals:
        duration = int(end - start + 1)
        if duration < int(min_duration) or duration > int(max_duration):
            continue
        segment_yaw = yaw[start : end + 1]
        path_angle = float(np.sum(np.abs(np.diff(segment_yaw))) * 180.0 / np.pi)
        net_angle = float(abs(segment_yaw[-1] - segment_yaw[0]) * 180.0 / np.pi)
        if path_angle < float(min_turn_angle_deg):
            continue
        consistency = net_angle / max(path_angle, 1e-6)
        if consistency < float(min_direction_consistency):
            continue
        lo_speed = max(0, start - 1)
        hi_speed = min(len(absolute_speed), end)
        local_speed = absolute_speed[lo_speed:hi_speed]
        if len(local_speed) == 0:
            continue
        local_peak = int(np.argmax(local_speed)) + lo_speed
        peak_frame = min(n_frames - 1, local_peak + 1)
        events.append(
            NaturalTurnEvent(
                peak_index=int(peak_frame),
                start=int(start),
                end=int(end),
                peak_speed_dps=float(absolute_speed[local_peak]),
                mean_speed_dps=float(local_speed.mean()),
                net_angle_deg=net_angle,
                path_angle_deg=path_angle,
            )
        )

    # Keep the strongest representative for highly overlapping final events.
    events.sort(key=lambda event: (event.peak_speed_dps, event.path_angle_deg), reverse=True)
    selected: List[NaturalTurnEvent] = []
    for event in events:
        duplicate = False
        for other in selected:
            intersection = max(0, min(event.end, other.end) - max(event.start, other.start) + 1)
            union = max(event.end, other.end) - min(event.start, other.start) + 1
            if union > 0 and intersection / union > 0.55:
                duplicate = True
                break
        if not duplicate:
            selected.append(event)
    selected.sort(key=lambda event: event.start)
    return selected


def make_fast_turn_corruption_v2(
    target: np.ndarray,
    turn_start: int,
    turn_end: int,
    speed_factor: float,
    min_context_frames: int = 4,
    min_corrupted_duration: int = 4,
    max_effective_factor: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compress a complete turn phrase while bounding the effective factor."""
    x = np.asarray(target, dtype=np.float32)
    n = len(x)
    if n < 12:
        raise ValueError("Window too short for turn corruption")
    turn_start = int(np.clip(turn_start, 1, n - 4))
    turn_end = int(np.clip(turn_end, turn_start + 2, n - 2))
    original_duration = int(turn_end - turn_start + 1)
    requested = float(max(speed_factor, 1.0))
    maximum = float(max(max_effective_factor, 1.0))
    minimum_duration_from_factor = int(np.ceil(original_duration / maximum))
    desired_duration = int(round(original_duration / requested))
    desired_duration = max(int(min_corrupted_duration), minimum_duration_from_factor, desired_duration)
    desired_duration = min(original_duration - 1, desired_duration)
    desired_duration = max(3, desired_duration)
    desired_span = int(desired_duration - 1)

    pre_source = int(turn_start)
    post_source = int((n - 1) - turn_end)
    remaining = int((n - 1) - desired_span)
    min_pre = min(int(min_context_frames), pre_source)
    min_post = min(int(min_context_frames), post_source)
    if remaining < min_pre + min_post:
        desired_span = max(2, (n - 1) - min_pre - min_post)
        desired_duration = desired_span + 1
        remaining = (n - 1) - desired_span

    total_context = max(pre_source + post_source, 1)
    output_pre = int(round(remaining * pre_source / total_context))
    output_pre = int(np.clip(output_pre, min_pre, max(min_pre, remaining - min_post)))
    output_post = int(remaining - output_pre)
    del output_post

    output_control = np.asarray(
        [0, output_pre, output_pre + desired_span, n - 1], dtype=np.float32
    )
    source_control = np.asarray([0, turn_start, turn_end, n - 1], dtype=np.float32)
    output_frames = np.arange(n, dtype=np.float32)
    source_positions = np.interp(output_frames, output_control, source_control).astype(np.float32)

    from motion_geometry.heading import resample_motion_so3

    corrupted = resample_motion_so3(x, source_positions)
    corrupted_start = int(output_pre)
    corrupted_end = int(output_pre + desired_span)
    mask = make_soft_event_mask(n, corrupted_start, corrupted_end, context=max(4, min_context_frames))
    info = {
        "original_turn_start": int(turn_start),
        "original_turn_end": int(turn_end),
        "original_turn_span": int(original_duration - 1),
        "original_turn_duration_frames": int(original_duration),
        "corrupted_turn_start": int(corrupted_start),
        "corrupted_turn_end": int(corrupted_end),
        "corrupted_turn_span": int(desired_span),
        "corrupted_turn_duration_frames": int(desired_duration),
        "requested_speed_factor": float(requested),
        "effective_speed_factor": float(original_duration / max(desired_duration, 1)),
        "source_positions": source_positions,
    }
    return corrupted.astype(np.float32), mask.astype(np.float32), info

def make_soft_event_mask(length: int, start: int, end: int, context: int = 6) -> np.ndarray:
    length = int(length)
    start = int(np.clip(start, 0, length - 1))
    end = int(np.clip(end, start, length - 1))
    context = int(max(0, context))
    mask = np.zeros((length,), dtype=np.float32)
    mask[start : end + 1] = 1.0
    if context > 0:
        left_start = max(0, start - context)
        left_len = start - left_start
        if left_len > 0:
            phase = np.linspace(0.0, 1.0, left_len + 1, dtype=np.float32)[:-1]
            mask[left_start:start] = 0.5 - 0.5 * np.cos(np.pi * phase)
        right_end = min(length - 1, end + context)
        right_len = right_end - end
        if right_len > 0:
            phase = np.linspace(1.0, 0.0, right_len + 1, dtype=np.float32)[1:]
            mask[end + 1 : right_end + 1] = 0.5 - 0.5 * np.cos(np.pi * phase)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def extract_window_with_event(
    motion: np.ndarray,
    event: NaturalTurnEvent,
    window_len: int,
    center_jitter: int = 0,
) -> Tuple[np.ndarray, int, int, int]:
    x = np.asarray(motion, dtype=np.float32)
    center = int(round(event.center)) + int(center_jitter)
    start = center - int(window_len) // 2
    start = int(np.clip(start, 0, max(0, len(x) - int(window_len))))
    end = start + int(window_len)
    window = x[start:end]
    if len(window) < int(window_len):
        if len(window) == 0:
            raise ValueError("Empty motion window")
        window = np.concatenate([window, np.repeat(window[-1:], int(window_len) - len(window), axis=0)], axis=0)
    local_start = int(np.clip(event.start - start, 1, int(window_len) - 4))
    local_end = int(np.clip(event.end - start, local_start + 2, int(window_len) - 2))
    return window.astype(np.float32), start, local_start, local_end


def inverse_time_map(source_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(source_positions, dtype=np.float32)
    positions = np.maximum.accumulate(positions)
    n = len(positions)
    inverse = np.interp(
        np.arange(n, dtype=np.float32),
        positions,
        np.arange(n, dtype=np.float32),
    ).astype(np.float32)
    inverse[0] = 0.0
    inverse[-1] = float(n - 1)
    return np.clip(inverse / max(n - 1, 1), 0.0, 1.0).astype(np.float32)


def build_v23_condition(
    observed_motion: np.ndarray,
    event_start: int,
    event_end: int,
    fps: float = 30.0,
    rot6d_layout: str = CANONICAL_ROT6D_LAYOUT,
) -> np.ndarray:
    """Build the 17D inference-safe condition from observed motion only."""
    x = np.asarray(observed_motion, dtype=np.float32)
    query = motion_query_from_dynamics(x, fps=fps, rot6d_layout=rot6d_layout)
    event_start = int(np.clip(event_start, 0, len(x) - 2))
    event_end = int(np.clip(event_end, event_start + 1, len(x) - 1))
    speed = yaw_speed_dps_np(
        x,
        fps=fps,
        smooth_window=5,
        rot6d_layout=rot6d_layout,
    )
    local_speed = speed[event_start : min(event_end, len(speed))]
    yaw = root_yaw_np(x, rot6d_layout=rot6d_layout)
    path_angle = float(np.sum(np.abs(np.diff(yaw[event_start : event_end + 1]))) * 180.0 / np.pi)
    peak = float(local_speed.max()) if len(local_speed) else 0.0
    mean = float(local_speed.mean()) if len(local_speed) else 0.0
    span = int(event_end - event_start + 1)
    center_phase = 0.5 * float(event_start + event_end) / max(len(x) - 1, 1)
    extra = np.asarray(
        [
            np.log1p(max(peak, 0.0)) / np.log1p(1600.0),
            np.log1p(max(mean, 0.0)) / np.log1p(800.0),
            np.clip(path_angle / 180.0, 0.0, 2.0),
            np.clip(span / max(len(x), 1), 0.0, 1.0),
            np.clip(center_phase, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([query, extra], axis=0).astype(np.float32)


def parse_duration_bins(text: str | Sequence[int]) -> np.ndarray:
    if isinstance(text, str):
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    else:
        values = [int(value) for value in text]
    if len(values) < 2:
        raise ValueError("At least two duration-bin edges are required")
    edges = np.asarray(sorted(set(values)), dtype=np.int32)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Duration-bin edges must be strictly increasing")
    return edges


def duration_bin_ids(duration_frames: np.ndarray, edges: np.ndarray) -> np.ndarray:
    duration = np.asarray(duration_frames, dtype=np.float32)
    # edges are inclusive lower boundaries, e.g. 8,16,...,57 -> six bins.
    return np.clip(np.digitize(duration, edges[1:-1], right=False), 0, len(edges) - 2).astype(np.int64)


def rotation_activity_np(motion: np.ndarray) -> float:
    x = np.asarray(motion, dtype=np.float32)
    if len(x) < 2:
        return 0.0
    velocity = np.diff(x[:, ROT], axis=0)
    return float(np.linalg.norm(velocity, axis=-1).mean())


def rotation_range_np(motion: np.ndarray) -> float:
    x = np.asarray(motion, dtype=np.float32)
    if len(x) < 2:
        return 0.0
    center = x[:, ROT].mean(axis=0, keepdims=True)
    return float(np.linalg.norm(x[:, ROT] - center, axis=-1).mean())


def max_rotation_jump_np(motion: np.ndarray) -> float:
    x = np.asarray(motion, dtype=np.float32)
    if len(x) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(x[:, ROT], axis=0), axis=-1).max())


def blend_motion_so3_np(base: np.ndarray, candidate: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32).reshape(-1)
    if base.shape != candidate.shape or base.ndim != 2 or base.shape[-1] != 151:
        raise ValueError(f"Expected matching [T,151], got {base.shape} and {candidate.shape}")
    if len(alpha) != len(base):
        raise ValueError("alpha length mismatch")
    a = np.clip(alpha, 0.0, 1.0)
    out = np.empty_like(base)
    out[:, CONTACT] = np.where(a[:, None] >= 0.5, candidate[:, CONTACT], base[:, CONTACT])
    out[:, ROOT] = (1.0 - a[:, None]) * base[:, ROOT] + a[:, None] * candidate[:, ROOT]

    rb = rot6d_to_matrix_np(base[:, ROT].reshape(len(base), 24, 6))
    rc = rot6d_to_matrix_np(candidate[:, ROT].reshape(len(base), 24, 6))
    tangent = relative_rotvec_np(rb, rc)
    blended = rb @ so3_exp_np(tangent * a[:, None, None])
    out[:, ROT] = matrix_to_rot6d_np(blended).reshape(len(base), 144)
    return out.astype(np.float32)


def cosine_window(length: int, edge: int = 8) -> np.ndarray:
    length = int(length)
    edge = int(max(0, min(edge, length // 2)))
    alpha = np.ones((length,), dtype=np.float32)
    if edge > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, edge + 2, dtype=np.float32)[1:-1])
        alpha[:edge] = ramp
        alpha[-edge:] = ramp[::-1]
    return alpha
