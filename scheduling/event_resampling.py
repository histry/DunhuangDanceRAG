#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-event SO(3) resampling for whole-song scheduling.

V23 Tau is never allowed to cross an event boundary.  Turn-bearing events use a
V23-conditioned local time map; other events use uniform SO(3) resampling.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from motion_geometry.heading import resample_motion_so3
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    convert_motion_rot6d_layout_np,
    normalize_rot6d_layout,
)
from scheduling.duration_features import (
    build_v23_condition,
    detect_natural_turn_events,
    extract_window_with_event,
    make_soft_event_mask,
)


def _uniform_positions(source_len: int, target_len: int) -> np.ndarray:
    if source_len <= 1:
        return np.zeros((target_len,), dtype=np.float32)
    return np.linspace(0.0, source_len - 1, target_len, dtype=np.float32)


def resample_event(
    motion: np.ndarray,
    target_len: int,
    v23_bundle: Dict[str, Any] | None,
    device: torch.device,
    fps: float = 30.0,
    min_turn_angle: float = 10.0,
    min_peak_dps: float = 14.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    source = np.asarray(motion, dtype=np.float32)
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if float(fps) <= 0.0:
        raise ValueError("fps must be positive")
    if len(source) < 2:
        return np.repeat(source[:1], target_len, axis=0), {
            "method": "repeat",
            "fps": float(fps),
        }

    # The detector's historical defaults are frame counts calibrated at
    # 30 FPS.  Convert them to equal physical durations before running a
    # rate-specific 30/60 FPS Scheduler branch.
    frame_scale = float(fps) / 30.0

    def scaled(frames_at_30fps: int, minimum: int = 1) -> int:
        return max(int(minimum), int(round(float(frames_at_30fps) * frame_scale)))

    minimum_turn_frames = min(
        scaled(12),
        max(scaled(4), len(source) // 3),
    )

    events = detect_natural_turn_events(
        source,
        fps=float(fps),
        min_peak_dps=min_peak_dps,
        min_turn_angle_deg=min_turn_angle,
        min_gap=scaled(16),
        min_duration=minimum_turn_frames,
        max_duration=scaled(88),
        max_events=2,
        quiet_run=scaled(8),
        opposite_run=scaled(4),
        phrase_margin=scaled(3),
        slow_pose_span=scaled(10, minimum=2),
        slow_angle_window=scaled(24, minimum=2),
        split_valley_radius=scaled(3),
        smooth_window=scaled(9),
        minimum_input_frames=scaled(24),
    )
    if v23_bundle is None or not events:
        positions = _uniform_positions(len(source), target_len)
        return resample_motion_so3(source, positions).astype(np.float32), {
            "method": "uniform_so3",
            "source_len": len(source),
            "target_len": target_len,
            "fps": float(fps),
        }

    event = max(events, key=lambda x: (x.path_angle_deg, x.peak_speed_dps))
    checkpoint_layout = normalize_rot6d_layout(
        v23_bundle.get("rot6d_layout", ROT6D_LAYOUT_PYTORCH3D_ROW)
    )
    model_source = convert_motion_rot6d_layout_np(
        source,
        CANONICAL_ROT6D_LAYOUT,
        checkpoint_layout,
    )
    window_len = int(v23_bundle["config"].get("window_len", 120))
    window, _, local_start, local_end = extract_window_with_event(
        model_source,
        event,
        window_len,
    )
    mask = make_soft_event_mask(
        window_len,
        local_start,
        local_end,
        context=scaled(6),
    )
    condition = build_v23_condition(
        window,
        local_start,
        local_end,
        fps=float(fps),
        rot6d_layout=checkpoint_layout,
    )
    model = v23_bundle["model"]
    with torch.no_grad():
        tau_output = model.predict_tau(
            torch.from_numpy(window[None]).to(device),
            torch.from_numpy(mask[None]).to(device),
            torch.from_numpy(condition[None]).to(device),
            torch.tensor([float(target_len)], dtype=torch.float32, device=device),
        )
    tau = tau_output["tau"][0].detach().cpu().numpy().astype(np.float32)
    source_window_positions = tau * float(window_len - 1)

    # Sample the learned map only over the event's own output domain.  Convert
    # resulting window coordinates back into source-event coordinates.
    output_window_grid = np.linspace(local_start, local_end, target_len, dtype=np.float32)
    mapped_window = np.interp(
        output_window_grid,
        np.arange(window_len, dtype=np.float32),
        source_window_positions,
    )
    denominator = max(float(local_end - local_start), 1.0)
    local_normalized = np.clip((mapped_window - local_start) / denominator, 0.0, 1.0)
    positions = local_normalized * float(len(source) - 1)
    positions[0] = 0.0
    positions[-1] = float(len(source) - 1)
    positions = np.maximum.accumulate(positions)
    result = resample_motion_so3(source, positions).astype(np.float32)
    return result, {
        "method": "v23_local_tau",
        "source_len": len(source),
        "target_len": target_len,
        "fps": float(fps),
        "turn_start": int(event.start),
        "turn_end": int(event.end),
        "turn_peak_dps": float(event.peak_speed_dps),
        "turn_angle_deg": float(event.path_angle_deg),
        "input_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
        "checkpoint_rot6d_layout": checkpoint_layout,
    }
