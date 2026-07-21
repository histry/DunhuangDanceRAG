#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Augment a V21 shared Event-RAG index with natural-duration priors.

Dynamic-event source length is used for non-turn events.  For turn-bearing
events, V23 Stage 1 supplies an inference-safe calibrated natural duration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from model.duration_predictor import load_duration_checkpoint
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    convert_motion_rot6d_layout_np,
    normalize_rot6d_layout,
)
from scheduling.index_io import load_shared_index, resolve_event_motion_path
from support.common import load_motion
from scheduling.duration_features import (
    build_v23_condition,
    detect_natural_turn_events,
    extract_window_with_event,
    make_soft_event_mask,
)


def estimate_one(
    motion: np.ndarray,
    bundle: Dict[str, Any],
    device: torch.device,
    min_turn_angle: float,
    min_peak_dps: float,
    fps: float,
) -> Dict[str, Any]:
    raw_length = int(len(motion))
    frame_scale = float(fps) / 30.0

    def scaled(frames_at_30fps: int, minimum: int = 1) -> int:
        return max(int(minimum), int(round(float(frames_at_30fps) * frame_scale)))

    events = detect_natural_turn_events(
        motion,
        fps=float(fps),
        min_peak_dps=min_peak_dps,
        min_turn_angle_deg=min_turn_angle,
        min_gap=scaled(16),
        min_duration=scaled(12, minimum=2),
        max_duration=scaled(88, minimum=3),
        max_events=3,
        quiet_run=scaled(8),
        opposite_run=scaled(4),
        phrase_margin=scaled(3),
        slow_pose_span=scaled(10, minimum=2),
        slow_angle_window=scaled(24, minimum=2),
        split_valley_radius=scaled(3),
        smooth_window=scaled(9),
        minimum_input_frames=scaled(24),
    )
    if not events:
        return {
            "natural_duration": float(raw_length),
            "raw_length": raw_length,
            "v23_used": False,
            "turn_peak_dps": 0.0,
            "turn_angle_deg": 0.0,
            "duration_confidence": 1.0,
        }
    event = max(events, key=lambda x: (x.path_angle_deg, x.peak_speed_dps))
    checkpoint_layout = normalize_rot6d_layout(
        bundle.get("rot6d_layout", ROT6D_LAYOUT_PYTORCH3D_ROW)
    )
    model_motion = convert_motion_rot6d_layout_np(
        motion,
        CANONICAL_ROT6D_LAYOUT,
        checkpoint_layout,
    )
    window_len = int(bundle["config"].get("window_len", 120))
    window, _, local_start, local_end = extract_window_with_event(
        model_motion,
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
    model = bundle["model"]
    with torch.no_grad():
        output = model.predict_duration(
            torch.from_numpy(window[None]).to(device),
            torch.from_numpy(mask[None]).to(device),
            torch.from_numpy(condition[None]).to(device),
            use_hard_duration=False,
        )
    predicted = float(output["duration_frames"][0].item())
    confidence = float(output["duration_bin_confidence"][0].item())
    # The index event is already a real natural action.  V23 is a calibration
    # prior rather than permission to make an arbitrary large change.
    lower = max(0.4 * float(fps), 0.70 * raw_length)
    upper = min((88.0 / 30.0) * float(fps), 1.35 * raw_length)
    calibrated = float(np.clip(predicted, lower, upper))
    return {
        "natural_duration": calibrated,
        "raw_length": raw_length,
        "v23_used": True,
        "turn_peak_dps": float(event.peak_speed_dps),
        "turn_angle_deg": float(event.path_angle_deg),
        "duration_confidence": confidence,
        "checkpoint_rot6d_layout": checkpoint_layout,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--index_npz", required=True)
    parser.add_argument("--v23_checkpoint", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--min_turn_angle", type=float, default=10.0)
    parser.add_argument("--min_peak_dps", type=float, default=14.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    index_json = Path(args.index_json).resolve()
    meta, arrays, items = load_shared_index(index_json, Path(args.index_npz))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = load_duration_checkpoint(args.v23_checkpoint, device=device)
    declared_fps = bundle.get("config", {}).get("fps")
    if declared_fps is None:
        legacy_ok = (
            abs(float(args.fps) - 30.0) < 1.0e-6
            and os.environ.get("DUNHUANG_ALLOW_LEGACY_30FPS_CHECKPOINTS", "0") == "1"
        )
        if not legacy_ok:
            raise RuntimeError(
                "Duration checkpoint has no FPS contract. Rebuild a rate-specific checkpoint; "
                "legacy weights are allowed only for the explicit 30 FPS parity baseline."
            )
        declared_fps = 30.0
    checkpoint_fps = float(declared_fps)
    if abs(checkpoint_fps - float(args.fps)) > 1.0e-6:
        raise RuntimeError(
            "Duration checkpoint FPS contract mismatch: "
            f"checkpoint={checkpoint_fps}, requested={args.fps}. "
            "Train/rebuild a rate-specific duration checkpoint."
        )

    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        path = resolve_event_motion_path(item, index_json, metadata=meta)
        motion = load_motion(path)
        row = estimate_one(
            motion,
            bundle,
            device,
            min_turn_angle=args.min_turn_angle,
            min_peak_dps=args.min_peak_dps,
            fps=float(args.fps),
        )
        row["event_index"] = index
        row["event_id"] = str(item.get("event_id", index))
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(items):
            print(f"[V26 duration index] {index + 1}/{len(items)}", flush=True)

    out_arrays = {key: arrays[key] for key in arrays.files}
    out_arrays.update(
        {
            "natural_duration": np.asarray([r["natural_duration"] for r in rows], dtype=np.float32),
            "duration_confidence": np.asarray([r["duration_confidence"] for r in rows], dtype=np.float32),
            "v23_duration_used": np.asarray([r["v23_used"] for r in rows], dtype=np.bool_),
            "turn_peak_dps": np.asarray([r["turn_peak_dps"] for r in rows], dtype=np.float32),
            "turn_angle_deg": np.asarray([r["turn_angle_deg"] for r in rows], dtype=np.float32),
        }
    )
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out_arrays)

    payload = {
        "version": "multirate_duration_augmented_event_index_v2",
        "fps": float(args.fps),
        "natural_duration_units": "frames_at_fps",
        "turn_peak_units": "deg/s",
        "base_index_json": str(args.index_json),
        "base_index_npz": str(args.index_npz),
        "v23_checkpoint": str(args.v23_checkpoint),
        "num_events": len(rows),
        "v23_used_count": int(sum(int(r["v23_used"]) for r in rows)),
        "natural_duration_percentiles": np.percentile(
            np.asarray([r["natural_duration"] for r in rows]),
            [0, 10, 25, 50, 75, 90, 100],
        ).tolist(),
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_npz}")
    print(f"[SAVED] {out_json}")


if __name__ == "__main__":
    main()
