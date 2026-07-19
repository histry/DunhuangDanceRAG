#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audio-derived dynamic-duration contract for whole-song generation.

The generated motion length is determined by the current WAV and its fresh slot
schedule.  No fixed-duration assumption is permitted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def motion_frame_count(path: str | Path) -> int:
    path = Path(path)
    x = np.asarray(np.load(path, allow_pickle=True))
    if x.ndim == 2:
        return int(x.shape[0])
    if x.ndim == 3 and x.shape[0] == 1:
        return int(x.shape[1])
    raise ValueError(
        f"Expected generated motion [T,D] or [1,T,D], got {x.shape} from {path}"
    )


def audit_dynamic_duration(
    output_path: str | Path,
    contract: Mapping[str, Any],
    fps: float,
    output_frame_tolerance: int = 2,
    schedule_audio_tolerance: int = 2,
) -> Dict[str, Any]:
    """Compare generated frames with the fresh schedule and current WAV budget."""
    output_path = Path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(
            f"Generated motion missing for duration guard: {output_path}"
        )

    actual_frames = motion_frame_count(output_path)
    schedule_frames = int(
        contract.get(
            "total_target_frames",
            contract.get("expected_audio_target_frames", -1),
        )
    )
    audio_frames = int(contract.get("expected_audio_target_frames", schedule_frames))
    if schedule_frames < 0 or audio_frames < 0:
        raise RuntimeError(
            "Fresh-WAV contract did not expose an audio-derived frame budget"
        )

    schedule_error = int(actual_frames - schedule_frames)
    audio_error = int(actual_frames - audio_frames)
    output_frame_tolerance = max(0, int(output_frame_tolerance))
    audio_tolerance = output_frame_tolerance + max(
        0, int(schedule_audio_tolerance)
    )
    ok = bool(
        abs(schedule_error) <= output_frame_tolerance
        and abs(audio_error) <= audio_tolerance
    )

    return {
        "schema": "v46_53_audio_derived_dynamic_duration_contract",
        "policy": "fresh_wav_audio_duration_determines_output_frames",
        "fixed_duration_seconds": None,
        "audio": contract.get("audio"),
        "schedule": contract.get("schedule_path"),
        "output": str(output_path),
        "fps": float(fps),
        "expected_audio_frames": int(audio_frames),
        "schedule_target_frames": int(schedule_frames),
        "actual_output_frames": int(actual_frames),
        "expected_audio_seconds": float(audio_frames / max(float(fps), 1e-8)),
        "schedule_target_seconds": float(
            schedule_frames / max(float(fps), 1e-8)
        ),
        "actual_output_seconds": float(
            actual_frames / max(float(fps), 1e-8)
        ),
        "output_minus_schedule_frames": int(schedule_error),
        "output_minus_audio_frames": int(audio_error),
        "schedule_tolerance_frames": int(output_frame_tolerance),
        "audio_tolerance_frames": int(audio_tolerance),
        "ok": ok,
    }


def save_duration_report(report: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
