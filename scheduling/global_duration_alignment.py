#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact whole-song integer duration allocation for music-dominant V26.

Principle:
1. music phrase length and local speed factor define the target rhythm;
2. natural duration defines a feasible range and calibration prior;
3. planner duration is a weak auxiliary prior;
4. exact whole-song frame budget is always enforced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


def event_elasticity(event_type: str) -> float:
    table = {
        "pose_hold": 1.80,
        "calm_flow": 1.60,
        "neutral_flow": 1.25,
        "release": 1.25,
        "build_up": 1.05,
        "arm_flourish": 0.90,
        "support_shift": 0.80,
        "high_tension": 0.72,
    }
    return float(table.get(str(event_type), 1.0))


def event_importance(event_type: str, music_event: str) -> float:
    table = {
        "pose_hold": 0.70,
        "calm_flow": 0.85,
        "neutral_flow": 1.00,
        "release": 1.00,
        "build_up": 1.15,
        "arm_flourish": 1.30,
        "support_shift": 1.40,
        "high_tension": 1.50,
    }
    value = float(table.get(str(event_type), 1.0))
    if str(music_event) in {"climax", "accent", "section_change"}:
        value *= 1.12
    return value


def _as_array(values, n: int, default: float) -> np.ndarray:
    if values is None:
        return np.full((n,), float(default), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"Expected sequence length {n}, got shape {arr.shape}")
    return arr


def _continuous_projection(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    flexibility: np.ndarray,
    budget: float,
    iterations: int = 160,
) -> np.ndarray:
    x = np.clip(np.asarray(target, dtype=np.float64), lower, upper)
    flex = np.maximum(np.asarray(flexibility, dtype=np.float64), 1e-5)
    for _ in range(iterations):
        error = float(budget - x.sum())
        if abs(error) < 1e-7:
            break
        room = np.maximum(upper - x, 0.0) if error > 0 else np.maximum(x - lower, 0.0)
        active = room > 1e-8
        if not np.any(active):
            break
        weight = room * flex * active
        if weight.sum() <= 1e-12:
            break
        delta = error * weight / weight.sum()
        if error > 0:
            delta = np.minimum(delta, room)
        else:
            delta = -np.minimum(-delta, room)
        x += delta
        x = np.clip(x, lower, upper)
    return x


def _integer_exact(
    continuous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    priority: np.ndarray,
) -> np.ndarray:
    base = np.floor(continuous).astype(np.int64)
    base = np.maximum(base, lower.astype(np.int64))
    base = np.minimum(base, upper.astype(np.int64))
    remainder = int(budget - int(base.sum()))
    fractional = continuous - np.floor(continuous)
    if remainder > 0:
        order = np.argsort(-(fractional + 1e-4 * priority))
        guard = 0
        while remainder > 0:
            changed = False
            for index in order:
                if base[index] < int(upper[index]):
                    base[index] += 1
                    remainder -= 1
                    changed = True
                    if remainder == 0:
                        break
            guard += 1
            if not changed or guard > len(base) + 4:
                break
    elif remainder < 0:
        order = np.argsort(fractional + 1e-4 * priority)
        guard = 0
        while remainder < 0:
            changed = False
            for index in order:
                if base[index] > int(lower[index]):
                    base[index] -= 1
                    remainder += 1
                    changed = True
                    if remainder == 0:
                        break
            guard += 1
            if not changed or guard > len(base) + 4:
                break
    if int(base.sum()) != int(budget):
        raise RuntimeError(
            f"Could not allocate exact frame budget: allocated={base.sum()} budget={budget}. "
            "Relax natural bounds, phrase count, or transition lengths."
        )
    return base.astype(np.int32)


def allocate_whole_song_durations(
    phrase_lengths: Sequence[int],
    natural_durations: Sequence[float],
    planner_durations: Sequence[float],
    event_types: Sequence[str],
    music_events: Sequence[str],
    transition_lengths: Sequence[int],
    total_frames: int,
    music_weight: float = 1.60,
    natural_weight: float = 0.85,
    planner_weight: float = 0.75,
    min_content_frames: int = 12,
    min_warp: float = 0.70,
    max_warp: float = 1.50,
    music_speed_factors: Sequence[float] | None = None,
    music_content_targets: Sequence[float] | None = None,
    allow_music_bound_override: bool = True,
    lock_music_boundaries: bool = False,
) -> Dict[str, Any]:
    phrase = np.asarray(phrase_lengths, dtype=np.float64)
    natural = np.asarray(natural_durations, dtype=np.float64)
    planned = np.asarray(planner_durations, dtype=np.float64)
    transitions = np.asarray(transition_lengths, dtype=np.int32)
    n = len(phrase)
    if not (len(natural) == len(planned) == len(event_types) == len(music_events) == n):
        raise ValueError("All phrase/event sequences must have identical length")
    if len(transitions) != n:
        raise ValueError("transition_lengths must have one value per phrase; first must be zero")
    if n == 0:
        raise ValueError("At least one phrase is required")
    if int(transitions[0]) != 0:
        raise ValueError("The first transition length must be zero")

    speed = np.clip(_as_array(music_speed_factors, n, 1.0), 0.45, 2.20)
    phrase_content = np.maximum(phrase - transitions, float(min_content_frames))
    if music_content_targets is not None:
        phrase_content = np.maximum(_as_array(music_content_targets, n, min_content_frames), float(min_content_frames))
    reference = np.maximum(natural, 1.0)
    lower = np.maximum(float(min_content_frames), np.floor(reference * float(min_warp)))
    upper = np.maximum(lower, np.ceil(reference * float(max_warp)))
    importance = np.asarray(
        [event_importance(e, m) for e, m in zip(event_types, music_events)],
        dtype=np.float64,
    )
    elasticity = np.asarray([event_elasticity(e) for e in event_types], dtype=np.float64)

    if lock_music_boundaries:
        full_lengths = phrase.astype(np.int64)
        if int(full_lengths.sum()) != int(total_frames):
            raise RuntimeError(
                f"Locked music boundaries require phrase length sum {int(full_lengths.sum())} "
                f"to equal total_frames {int(total_frames)}"
            )
        allocation = full_lengths - transitions.astype(np.int64)
        if np.any(allocation < int(min_content_frames)):
            bad = np.where(allocation < int(min_content_frames))[0].tolist()
            raise RuntimeError(
                f"Locked music boundaries leave too little content for slots {bad}. "
                "Reduce transition_min_frames, split less aggressively, or lower min_content_frames."
            )
        over = allocation.astype(np.float64) > upper
        under = allocation.astype(np.float64) < lower
        boundaries = [0]
        for length in full_lengths:
            boundaries.append(boundaries[-1] + int(length))
        override_reason = "music_boundaries_locked"
        if bool(np.any(over) or np.any(under)):
            override_reason = "music_boundaries_locked_with_natural_warp_override"
        return {
            "version": "v26_music_dominant_duration_alignment",
            "total_frames": int(total_frames),
            "content_budget": int(allocation.sum()),
            "content_lengths": allocation.astype(int).tolist(),
            "transition_lengths": transitions.astype(int).tolist(),
            "phrase_total_lengths": full_lengths.astype(int).tolist(),
            "output_boundaries": boundaries,
            "target_continuous": phrase_content.tolist(),
            "music_duration_target": phrase_content.tolist(),
            "music_content_targets": phrase_content.tolist(),
            "music_speed_factors": speed.tolist(),
            "natural_durations": natural.tolist(),
            "planner_durations": planned.tolist(),
            "music_phrase_lengths": phrase.astype(int).tolist(),
            "warp_ratios": (allocation / np.maximum(natural, 1.0)).tolist(),
            "strict_natural_min": (reference * float(min_warp)).tolist(),
            "strict_natural_max": (reference * float(max_warp)).tolist(),
            "actual_lower_bounds": lower.tolist(),
            "actual_upper_bounds": upper.tolist(),
            "bound_override": override_reason,
            "lock_music_boundaries": True,
            "num_warp_over_max": int(np.sum(over)),
            "num_warp_under_min": int(np.sum(under)),
            "importance": importance.tolist(),
            "elasticity": elasticity.tolist(),
            "weights": {
                "music": float(music_weight),
                "natural": float(natural_weight),
                "planner": float(planner_weight),
            },
        }

    content_budget = int(total_frames - int(transitions.sum()))
    if content_budget < n * int(min_content_frames):
        raise RuntimeError(
            f"Transitions consume too much of the song: content_budget={content_budget}, phrases={n}"
        )

    music_duration_target = 0.72 * phrase_content + 0.28 * (np.maximum(natural, 1.0) / speed)

    denominator = music_weight + natural_weight + planner_weight
    target = (
        music_weight * music_duration_target
        + natural_weight * (np.maximum(natural, 1.0) / speed)
        + planner_weight * planned
    ) / max(denominator, 1e-8)

    strict_lower_sum = float(lower.sum())
    strict_upper_sum = float(upper.sum())
    override_reason = "none"
    if content_budget < strict_lower_sum:
        if not allow_music_bound_override:
            raise RuntimeError(
                f"Content budget {content_budget} below natural lower bound {strict_lower_sum:.1f}"
            )
        shrink = max(float(content_budget) / max(strict_lower_sum, 1.0), 0.35)
        lower = np.maximum(float(min_content_frames), np.floor(lower * shrink))
        upper = np.maximum(upper, lower)
        override_reason = "lower_relaxed_for_exact_music_alignment"
    elif content_budget > strict_upper_sum:
        if not allow_music_bound_override:
            raise RuntimeError(
                f"Content budget {content_budget} above natural upper bound {strict_upper_sum:.1f}"
            )
        # One event per music phrase cannot always satisfy a 0.7-1.5 natural range
        # for long songs.  Expand bounds toward phrase targets but keep the report
        # explicit so the paper can discuss where exact alignment used extra stretch.
        upper = np.maximum(upper, np.ceil(phrase_content))
        if content_budget > float(upper.sum()):
            extra = content_budget - float(upper.sum())
            upper = upper + np.ceil(extra / max(n, 1))
        override_reason = "upper_expanded_for_exact_music_alignment"

    flexibility = elasticity / np.maximum(importance, 1e-6)
    continuous = _continuous_projection(
        target,
        lower,
        upper,
        flexibility,
        float(content_budget),
    )
    allocation = _integer_exact(
        continuous,
        lower,
        upper,
        content_budget,
        priority=importance,
    )
    full_lengths = allocation + transitions
    if int(full_lengths.sum()) != int(total_frames):
        raise AssertionError("Whole-song duration allocation is not exact")

    boundaries = [0]
    for length in full_lengths:
        boundaries.append(boundaries[-1] + int(length))

    return {
        "version": "v26_music_dominant_duration_alignment",
        "total_frames": int(total_frames),
        "content_budget": int(content_budget),
        "content_lengths": allocation.tolist(),
        "transition_lengths": transitions.tolist(),
        "phrase_total_lengths": full_lengths.astype(int).tolist(),
        "output_boundaries": boundaries,
        "target_continuous": continuous.tolist(),
        "music_duration_target": music_duration_target.tolist(),
        "music_content_targets": phrase_content.tolist(),
        "music_speed_factors": speed.tolist(),
        "natural_durations": natural.tolist(),
        "planner_durations": planned.tolist(),
        "music_phrase_lengths": phrase.astype(int).tolist(),
        "warp_ratios": (allocation / np.maximum(natural, 1.0)).tolist(),
        "strict_natural_min": (reference * float(min_warp)).tolist(),
        "strict_natural_max": (reference * float(max_warp)).tolist(),
        "actual_lower_bounds": lower.tolist(),
        "actual_upper_bounds": upper.tolist(),
        "bound_override": override_reason,
        "lock_music_boundaries": False,
        "importance": importance.tolist(),
        "elasticity": elasticity.tolist(),
        "weights": {
            "music": float(music_weight),
            "natural": float(natural_weight),
            "planner": float(planner_weight),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = allocate_whole_song_durations(**payload)
    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {args.out_json}")


if __name__ == "__main__":
    main()
