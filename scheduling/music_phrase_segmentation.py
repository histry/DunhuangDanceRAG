#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whole-song music feature extraction and variable-length phrase segmentation.

V26 music-dominant revision:
- keeps the original V21 12D music frame representation;
- adds phrase-level rhythm control fields used by the scheduler:
  speed_factor, transition_base_frames, transition_profile, and boundary accent;
- keeps all frame indices in target motion FPS.
"""
from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from scheduling.extract_music_features import extract_audio_features
from scheduling.music_event_calibrated import build_phrase_query as calibrated_phrase_query


@dataclass(frozen=True)
class MusicPhrase:
    index: int
    start: int
    end: int
    music_event: str
    query: List[float]
    planner_feature: List[float]
    boundary_confidence: float
    speed_factor: float
    transition_base_frames: int
    transition_profile: str
    boundary_accent_strength: float
    energy: float
    onset: float
    beat_density: float
    tempo_norm: float
    arousal: float
    tension: float
    calmness: float
    novelty: float
    section_change: float
    accent: float

    @property
    def length(self) -> int:
        return int(self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["length"] = self.length
        return value


def audio_duration_seconds(path: str | Path) -> float:
    audio = Path(path)
    if not audio.is_file():
        raise FileNotFoundError(audio)
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(audio))
        if info.samplerate > 0:
            return float(info.frames / info.samplerate)
    except Exception:
        pass
    try:
        import librosa  # type: ignore

        return float(librosa.get_duration(path=str(audio)))
    except Exception:
        pass
    if audio.suffix.lower() == ".wav":
        with wave.open(str(audio), "rb") as wf:
            if wf.getframerate() <= 0:
                raise RuntimeError(f"Invalid WAV sample rate: {audio}")
            return float(wf.getnframes() / wf.getframerate())
    raise RuntimeError(
        f"Could not determine audio duration for {audio}. "
        "Install soundfile or librosa, or use WAV input."
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
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


def robust_01(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return x.copy()
    lo, hi = np.percentile(x, [10.0, 90.0])
    if hi - lo < 1e-7:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def whole_song_features(
    audio_path: str | Path,
    fps: float = 30.0,
    cache_dir: str | Path | None = None,
    max_seconds: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    duration = audio_duration_seconds(audio_path)
    if max_seconds > 0:
        duration = min(duration, float(max_seconds))
    num_frames = max(2, int(round(duration * float(fps))))
    cache_path: Path | None = None
    meta_path: Path | None = None
    if cache_dir:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        key = Path(audio_path).stem
        cache_path = cache / f"{key}_v26_fps{float(fps):g}_{num_frames}.npy"
        meta_path = cache_path.with_suffix(".json")
        if cache_path.is_file() and meta_path.is_file():
            features = np.load(cache_path).astype(np.float32)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if features.shape == (num_frames, 12):
                return features, meta

    features, extractor_meta = extract_audio_features(audio_path, num_frames=num_frames)
    meta = {
        "audio": str(audio_path),
        "duration_sec": float(duration),
        "fps": float(fps),
        "num_frames": int(num_frames),
        "feature_dim": int(features.shape[1]),
        "extractor": extractor_meta,
    }
    if cache_path is not None and meta_path is not None:
        np.save(cache_path, features.astype(np.float32))
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return features.astype(np.float32), meta


def structural_novelty(features: np.ndarray, fps: float = 30.0) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 12:
        raise ValueError(f"Expected [T,12+] features, got {x.shape}")
    base = (
        0.38 * robust_01(x[:, 10])
        + 0.25 * robust_01(x[:, 8])
        + 0.16 * robust_01(x[:, 1])
        + 0.11 * robust_01(np.abs(x[:, 5]))
        + 0.10 * robust_01(x[:, 11])
    )
    short = moving_average(base, max(3, int(round(0.25 * fps)) | 1))
    medium = moving_average(base, max(3, int(round(0.75 * fps)) | 1))
    long = moving_average(base, max(3, int(round(2.0 * fps)) | 1))
    contrast = np.maximum(0.0, short - long)
    return robust_01(0.45 * short + 0.35 * medium + 0.20 * contrast)


def _local_maxima(score: np.ndarray, minimum: float) -> List[int]:
    x = np.asarray(score, dtype=np.float32)
    return [
        i
        for i in range(1, len(x) - 1)
        if x[i] >= minimum and x[i] >= x[i - 1] and x[i] >= x[i + 1]
    ]


def _snap_to_rhythm(
    index: int,
    features: np.ndarray,
    radius: int,
    lower: int,
    upper: int,
) -> int:
    lo = max(int(lower), int(index) - int(radius))
    hi = min(int(upper), int(index) + int(radius))
    if hi <= lo:
        return int(np.clip(index, lower, upper))
    evidence = (
        0.50 * features[lo : hi + 1, 2]
        + 0.30 * features[lo : hi + 1, 1]
        + 0.20 * features[lo : hi + 1, 10]
    )
    return int(lo + np.argmax(evidence))


def boundary_evidence(features: np.ndarray) -> np.ndarray:
    """Return frame-level evidence for phrase or sub-phrase boundaries."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 12:
        raise ValueError(f"Expected [T,12+] features, got {x.shape}")
    novelty = structural_novelty(x)
    evidence = (
        0.34 * robust_01(x[:, 2])
        + 0.28 * robust_01(x[:, 1])
        + 0.20 * robust_01(x[:, 10])
        + 0.12 * robust_01(x[:, 11])
        + 0.06 * novelty
    )
    return robust_01(evidence).astype(np.float32)


def _repair_phrase_lengths(
    boundaries: List[int],
    score: np.ndarray,
    features: np.ndarray,
    min_frames: int,
    max_frames: int,
    snap_radius: int,
) -> List[int]:
    total = int(len(score))
    result = [0]
    source = sorted(set(int(x) for x in boundaries if 0 < int(x) < total))
    source.append(total)

    cursor = 0
    for proposed in source:
        proposed = int(proposed)
        while proposed - cursor > max_frames:
            ideal = cursor + max_frames
            lo = max(cursor + min_frames, ideal - max_frames // 4)
            hi = min(proposed - min_frames, ideal + max_frames // 4)
            if hi <= lo:
                split = ideal
            else:
                split = lo + int(np.argmax(score[lo : hi + 1]))
            split = _snap_to_rhythm(split, features, snap_radius, cursor + min_frames, proposed - min_frames)
            if split <= cursor:
                break
            result.append(split)
            cursor = split
        if proposed - cursor < min_frames and proposed != total:
            continue
        if proposed > cursor:
            result.append(proposed)
            cursor = proposed

    if result[-1] != total:
        result[-1] = total
    while len(result) >= 3 and result[-1] - result[-2] < min_frames:
        result.pop(-2)
    return result


def _subslot_boundaries(
    features: np.ndarray,
    phrase: MusicPhrase,
    slots: int,
    min_frames: int,
    snap_radius: int,
) -> List[int]:
    slots = max(1, int(slots))
    start = int(phrase.start)
    end = int(phrase.end)
    if slots <= 1 or end - start < slots * min_frames:
        return [start, end]
    evidence = boundary_evidence(features)
    boundaries = [start]
    for i in range(1, slots):
        ideal = int(round(start + i * (end - start) / float(slots)))
        lo = max(boundaries[-1] + min_frames, ideal - snap_radius)
        hi = min(end - (slots - i) * min_frames, ideal + snap_radius)
        if hi <= lo:
            split = ideal
        else:
            split = lo + int(np.argmax(evidence[lo : hi + 1]))
        split = _snap_to_rhythm(split, features, snap_radius, lo, hi)
        split = int(np.clip(split, boundaries[-1] + min_frames, end - (slots - i) * min_frames))
        boundaries.append(split)
    boundaries.append(end)
    return boundaries


def split_music_phrases_for_events(
    features: np.ndarray,
    phrases: List[MusicPhrase],
    fps: float = 30.0,
    enabled: bool = True,
    max_slot_seconds: float = 3.20,
    min_slot_seconds: float = 1.60,
    max_events_per_phrase: int = 4,
    beat_snap_seconds: float = 0.25,
    calm_max_slot_seconds: float = 2.80,
) -> Tuple[List[MusicPhrase], Dict[str, Any]]:
    """Split long music phrases into multiple event slots.

    The original phrase boundaries are kept exactly.  Only internal sub-slot
    boundaries are inserted, so downstream scheduling can lock output
    boundaries to music while avoiding excessive time-warp on a single action.
    """
    x = np.asarray(features, dtype=np.float32)
    if not enabled:
        return list(phrases), {
            "enabled": False,
            "num_source_phrases": len(phrases),
            "num_slots": len(phrases),
            "slot_meta": [
                {
                    "slot_index": i,
                    "source_phrase_index": int(p.index),
                    "source_phrase_start": int(p.start),
                    "source_phrase_end": int(p.end),
                    "subslot_index": 0,
                    "subslot_count": 1,
                    "split_reason": "disabled",
                }
                for i, p in enumerate(phrases)
            ],
        }

    max_frames = max(12, int(round(float(max_slot_seconds) * float(fps))))
    calm_max_frames = max(12, int(round(float(calm_max_slot_seconds) * float(fps))))
    min_frames = max(12, int(round(float(min_slot_seconds) * float(fps))))
    snap_radius = max(1, int(round(float(beat_snap_seconds) * float(fps))))
    max_events = max(1, int(max_events_per_phrase))

    slots_out: List[MusicPhrase] = []
    slot_meta: List[Dict[str, Any]] = []
    for phrase in phrases:
        local_max = calm_max_frames if phrase.music_event in {"calm_flow", "release"} else max_frames
        local_max = max(local_max, min_frames)
        count = int(math.ceil(phrase.length / float(local_max))) if phrase.length > local_max else 1
        count = min(max(count, 1), max_events)
        if phrase.length < count * min_frames:
            count = max(1, phrase.length // max(min_frames, 1))
        count = max(1, int(count))
        reason = "long_phrase" if count > 1 else "single"
        if count > 1 and phrase.music_event in {"calm_flow", "release"}:
            reason = "anti_static_calm_phrase"
        boundaries = _subslot_boundaries(x, phrase, count, min_frames=min_frames, snap_radius=snap_radius)
        actual_count = len(boundaries) - 1
        for sub_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            query, event = calibrated_phrase_query(x[start:end], int(start), int(end))
            planner = phrase_planner_feature(x, int(start), int(end), np.asarray(query, dtype=np.float32))
            rhythm = phrase_rhythm_profile(x, int(start), int(end), fps)
            slot_index = len(slots_out)
            slots_out.append(
                MusicPhrase(
                    index=slot_index,
                    start=int(start),
                    end=int(end),
                    music_event=str(event),
                    query=np.asarray(query, dtype=np.float32).tolist(),
                    planner_feature=planner.tolist(),
                    boundary_confidence=float(phrase.boundary_confidence),
                    **rhythm,
                )
            )
            slot_meta.append(
                {
                    "slot_index": slot_index,
                    "source_phrase_index": int(phrase.index),
                    "source_phrase_start": int(phrase.start),
                    "source_phrase_end": int(phrase.end),
                    "subslot_index": int(sub_index),
                    "subslot_count": int(actual_count),
                    "split_reason": reason,
                    "source_music_event": str(phrase.music_event),
                    "slot_start": int(start),
                    "slot_end": int(end),
                    "slot_length": int(end - start),
                }
            )

    return slots_out, {
        "enabled": True,
        "num_source_phrases": len(phrases),
        "num_slots": len(slots_out),
        "max_slot_seconds": float(max_slot_seconds),
        "min_slot_seconds": float(min_slot_seconds),
        "max_events_per_phrase": int(max_events),
        "beat_snap_seconds": float(beat_snap_seconds),
        "calm_max_slot_seconds": float(calm_max_slot_seconds),
        "slot_meta": slot_meta,
        "source_boundaries": [int(phrases[0].start)] + [int(p.end) for p in phrases] if phrases else [],
        "slot_boundaries": [int(slots_out[0].start)] + [int(p.end) for p in slots_out] if slots_out else [],
    }


def phrase_rhythm_profile(
    features: np.ndarray,
    start: int,
    end: int,
    fps: float,
) -> Dict[str, Any]:
    """Build a music-dominant local tempo/transition profile for one phrase.

    speed_factor > 1 means the phrase can move faster, so the natural-duration
    target will become shorter downstream.
    """
    x = np.asarray(features[start:end], dtype=np.float32)
    if len(x) == 0:
        x = np.zeros((1, features.shape[1]), dtype=np.float32)
    energy = float(np.mean(x[:, 0]))
    onset = float(np.mean(x[:, 1]))
    beat = float(np.mean(x[:, 2]))
    tempo = float(np.mean(x[:, 3]))
    arousal = float(np.mean(x[:, 4]))
    tension = float(np.mean(x[:, 6]))
    calm = float(np.mean(x[:, 7]))
    novelty = float(np.mean(x[:, 8]))
    section = float(np.max(x[:, 10]))
    accent = float(np.mean(x[:, 11]))
    boundary_accent = float(np.percentile(x[:, 11], 90))

    fast_drive = np.clip(
        0.25 * tempo
        + 0.22 * beat
        + 0.20 * onset
        + 0.16 * energy
        + 0.12 * tension
        + 0.05 * accent,
        0.0,
        1.0,
    )
    slow_drive = np.clip(0.42 * calm + 0.22 * (1.0 - onset) + 0.18 * (1.0 - beat) + 0.18 * (1.0 - energy), 0.0, 1.0)
    raw_speed = 1.0 + 0.55 * (fast_drive - 0.50) - 0.45 * (slow_drive - 0.50)
    speed_factor = float(np.clip(raw_speed, 0.72, 1.38))

    transition_slow = np.clip(
        0.45 * calm + 0.20 * (1.0 - onset) + 0.20 * (1.0 - beat) + 0.15 * (1.0 - accent),
        0.0,
        1.0,
    )
    transition_fast = np.clip(
        0.34 * onset + 0.25 * beat + 0.22 * accent + 0.19 * tension,
        0.0,
        1.0,
    )
    transition_frames = int(round(12.0 + 36.0 * transition_slow - 12.0 * transition_fast))
    transition_frames = int(np.clip(transition_frames, 12, 48))
    if section > 0.70 and calm > 0.55:
        profile = "section_sustain"
    elif transition_fast > 0.62 and accent > 0.50:
        profile = "accent_cut"
    elif calm > 0.62:
        profile = "calm_sustain"
    elif tension > 0.62:
        profile = "tense_drive"
    else:
        profile = "balanced"

    return {
        "speed_factor": speed_factor,
        "transition_base_frames": transition_frames,
        "transition_profile": profile,
        "boundary_accent_strength": boundary_accent,
        "energy": energy,
        "onset": onset,
        "beat_density": beat,
        "tempo_norm": tempo,
        "arousal": arousal,
        "tension": tension,
        "calmness": calm,
        "novelty": novelty,
        "section_change": section,
        "accent": accent,
    }


def phrase_planner_feature(
    features: np.ndarray,
    start: int,
    end: int,
    query: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features[start:end], dtype=np.float32)
    if len(x) == 0:
        x = np.zeros((1, features.shape[1]), dtype=np.float32)
    mean = x[:, :12].mean(axis=0)
    std = x[:, :12].std(axis=0)
    trend = x[-1, :4] - x[0, :4]
    total = max(len(features), 1)
    relative = np.asarray(
        [
            start / total,
            end / total,
            (end - start) / total,
            0.5 * (start + end) / total,
        ],
        dtype=np.float32,
    )
    return np.concatenate([query[:12], mean[[0, 1, 2, 4, 6, 7, 8, 10]], std[[0, 1, 6, 8]], trend, relative]).astype(np.float32)


def segment_music_phrases(
    features: np.ndarray,
    fps: float = 30.0,
    min_phrase_seconds: float = 2.5,
    max_phrase_seconds: float = 7.5,
    boundary_quantile: float = 0.68,
    beat_snap_seconds: float = 0.35,
) -> Tuple[List[MusicPhrase], Dict[str, Any]]:
    x = np.asarray(features, dtype=np.float32)
    score = structural_novelty(x, fps=fps)
    min_frames = max(12, int(round(float(min_phrase_seconds) * fps)))
    max_frames = max(min_frames + 1, int(round(float(max_phrase_seconds) * fps)))
    threshold = float(np.quantile(score, np.clip(boundary_quantile, 0.1, 0.95)))
    candidates = _local_maxima(score, threshold)
    candidates = sorted(candidates, key=lambda i: float(score[i]), reverse=True)
    kept: List[int] = []
    for index in candidates:
        if all(abs(index - other) >= min_frames for other in kept):
            kept.append(index)
    kept.sort()
    snap_radius = max(1, int(round(float(beat_snap_seconds) * fps)))
    snapped = [
        _snap_to_rhythm(i, x, snap_radius, max(1, i - snap_radius), min(len(x) - 1, i + snap_radius))
        for i in kept
    ]
    boundaries = _repair_phrase_lengths(
        [0, *snapped, len(x)],
        score,
        x,
        min_frames=min_frames,
        max_frames=max_frames,
        snap_radius=snap_radius,
    )

    phrases: List[MusicPhrase] = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        query, event = calibrated_phrase_query(x[start:end], start, end)
        confidence = float(score[start]) if 0 < start < len(score) else 0.0
        planner = phrase_planner_feature(x, start, end, query)
        rhythm = phrase_rhythm_profile(x, start, end, fps)
        phrases.append(
            MusicPhrase(
                index=index,
                start=int(start),
                end=int(end),
                music_event=str(event),
                query=np.asarray(query, dtype=np.float32).tolist(),
                planner_feature=planner.tolist(),
                boundary_confidence=confidence,
                **rhythm,
            )
        )
    meta = {
        "num_frames": int(len(x)),
        "fps": float(fps),
        "num_phrases": len(phrases),
        "boundaries": boundaries,
        "min_phrase_frames": min_frames,
        "max_phrase_frames": max_frames,
        "novelty_threshold": threshold,
        "novelty_score": score.tolist(),
        "music_speed_factors": [p.speed_factor for p in phrases],
        "transition_base_frames": [p.transition_base_frames for p in phrases],
        "transition_profiles": [p.transition_profile for p in phrases],
    }
    return phrases, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_features", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min_phrase_seconds", type=float, default=2.5)
    parser.add_argument("--max_phrase_seconds", type=float, default=7.5)
    parser.add_argument("--boundary_quantile", type=float, default=0.68)
    parser.add_argument("--beat_snap_seconds", type=float, default=0.35)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--max_seconds", type=float, default=0.0)
    args = parser.parse_args()

    features, audio_meta = whole_song_features(
        args.audio,
        fps=args.fps,
        cache_dir=args.cache_dir or None,
        max_seconds=args.max_seconds,
    )
    phrases, segmentation = segment_music_phrases(
        features,
        fps=args.fps,
        min_phrase_seconds=args.min_phrase_seconds,
        max_phrase_seconds=args.max_phrase_seconds,
        boundary_quantile=args.boundary_quantile,
        beat_snap_seconds=args.beat_snap_seconds,
    )
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v26_music_dominant_phrase_segmentation",
        "audio_meta": audio_meta,
        "segmentation": segmentation,
        "phrases": [phrase.to_dict() for phrase in phrases],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_features:
        feature_path = Path(args.out_features)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features.astype(np.float32))
    print(f"[SAVED] {out}")
    print(f"frames={len(features)} phrases={len(phrases)}")


if __name__ == "__main__":
    main()
