#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract a compact 12D frame-level music event stream for V21.

Output convention per frame:
  0 energy
  1 onset strength
  2 beat pulse
  3 tempo (track-level, normalized)
  4 arousal
  5 delta arousal
  6 tension
  7 calmness
  8 novelty / phrase-change score
  9 spectral brightness
 10 section-change score
 11 accent score
"""
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _robust_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, [10, 90])
    if hi - lo < 1e-8:
        return np.full_like(x, 0.5, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def _resize_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x
    if len(x) <= 1:
        value = float(x[0]) if len(x) else 0.0
        return np.full((target_len,), value, dtype=np.float32)
    old_t = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(new_t, old_t, x).astype(np.float32)


def _fallback_read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        width = int(wf.getsampwidth())
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        y = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width={width}: {path}")
    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)
    return y.astype(np.float32), sr


def _fallback_features(path: Path, num_frames: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    y, sr = _fallback_read_wav(path)
    if len(y) == 0:
        return np.zeros((num_frames, 12), dtype=np.float32), {"backend": "wave_fallback", "tempo_bpm": 0.0}
    frame_edges = np.linspace(0, len(y), num_frames + 1).astype(int)
    rms = np.zeros((num_frames,), dtype=np.float32)
    zcr = np.zeros((num_frames,), dtype=np.float32)
    centroid = np.zeros((num_frames,), dtype=np.float32)
    for i in range(num_frames):
        seg = y[frame_edges[i] : frame_edges[i + 1]]
        if len(seg) == 0:
            continue
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-8))
        zcr[i] = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0)) if len(seg) > 1 else 0.0
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        centroid[i] = float((spec * freqs).sum() / (spec.sum() + 1e-8))
    energy = _robust_01(rms)
    onset = _robust_01(np.maximum(0.0, np.diff(energy, prepend=energy[:1])))
    beat = (_robust_01(onset) > 0.75).astype(np.float32)
    brightness = _robust_01(centroid)
    novelty = _robust_01(np.abs(np.diff(np.stack([energy, brightness], axis=1), axis=0, prepend=np.zeros((1, 2)))).mean(axis=1))
    tempo = np.full_like(energy, 0.5)
    arousal = np.clip(0.50 * energy + 0.30 * onset + 0.20 * beat, 0.0, 1.0)
    delta = np.diff(arousal, prepend=arousal[:1])
    tension = np.clip(0.40 * onset + 0.35 * brightness + 0.25 * novelty, 0.0, 1.0)
    calm = np.clip(1.0 - 0.55 * arousal - 0.45 * tension, 0.0, 1.0)
    section = np.clip(0.65 * novelty + 0.35 * np.abs(delta), 0.0, 1.0)
    accent = np.clip(0.60 * onset + 0.40 * beat, 0.0, 1.0)
    feat = np.stack([energy, onset, beat, tempo, arousal, delta, tension, calm, novelty, brightness, section, accent], axis=1).astype(np.float32)
    return feat, {"backend": "wave_fallback", "tempo_bpm": 0.0, "sample_rate": sr}


def extract_audio_features(audio_path: str | Path, num_frames: int = 150) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = Path(audio_path)
    try:
        import librosa  # type: ignore

        y, sr = librosa.load(str(path), sr=None, mono=True)
        if y.size == 0:
            raise RuntimeError("empty audio")
        hop = max(64, int(math.ceil(len(y) / max(num_frames, 1))))
        rms = librosa.feature.rms(y=y, frame_length=max(512, hop * 2), hop_length=hop, center=True)[0]
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
        try:
            tempo_value, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
            tempo_bpm = float(np.asarray(tempo_value).reshape(-1)[0])
        except Exception:
            tempo_bpm = 0.0
            beat_frames = np.asarray([], dtype=int)
        beat_pulse = np.zeros((max(len(rms), len(onset), chroma.shape[1]),), dtype=np.float32)
        beat_frames = np.asarray(beat_frames, dtype=int)
        beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < len(beat_pulse))]
        beat_pulse[beat_frames] = 1.0

        energy = _robust_01(_resize_1d(rms, num_frames))
        onset_n = _robust_01(_resize_1d(onset, num_frames))
        beat = _resize_1d(beat_pulse, num_frames)
        beat = np.clip(beat, 0.0, 1.0)
        brightness = _robust_01(_resize_1d(centroid, num_frames))
        chroma_t = chroma.T.astype(np.float32)
        if len(chroma_t) > 1:
            chroma_novelty = np.linalg.norm(np.diff(chroma_t, axis=0, prepend=chroma_t[:1]), axis=1)
        else:
            chroma_novelty = np.zeros((len(chroma_t),), dtype=np.float32)
        novelty = _robust_01(_resize_1d(chroma_novelty, num_frames))
        tempo_norm = float(np.clip((tempo_bpm - 50.0) / 150.0, 0.0, 1.0))
        tempo = np.full((num_frames,), tempo_norm, dtype=np.float32)
        arousal = np.clip(0.45 * energy + 0.35 * onset_n + 0.20 * beat, 0.0, 1.0)
        delta = np.diff(arousal, prepend=arousal[:1]).astype(np.float32)
        tension = np.clip(0.35 * onset_n + 0.30 * brightness + 0.35 * novelty, 0.0, 1.0)
        calm = np.clip(1.0 - 0.55 * arousal - 0.45 * tension, 0.0, 1.0)
        section = np.clip(0.70 * novelty + 0.30 * _robust_01(np.abs(delta)), 0.0, 1.0)
        accent = np.clip(0.65 * onset_n + 0.35 * beat, 0.0, 1.0)
        feat = np.stack([energy, onset_n, beat, tempo, arousal, delta, tension, calm, novelty, brightness, section, accent], axis=1).astype(np.float32)
        meta = {"backend": "librosa", "tempo_bpm": tempo_bpm, "sample_rate": int(sr), "duration_sec": float(len(y) / sr)}
        return feat, meta
    except Exception as exc:
        feat, meta = _fallback_features(path, num_frames)
        meta["librosa_error"] = str(exc)
        return feat, meta


def classify_frame_events(features: np.ndarray) -> list[str]:
    events: list[str] = []
    for row in np.asarray(features, dtype=np.float32):
        energy, onset, beat, _, arousal, delta, tension, calm, novelty, _, section, accent = row.tolist()
        if section > 0.72:
            event = "section_change"
        elif calm > 0.70 and tension < 0.45:
            event = "calm_flow"
        elif arousal > 0.72 and tension > 0.68:
            event = "climax"
        elif delta > 0.04 or (arousal > 0.62 and tension > 0.55):
            event = "build_up"
        elif delta < -0.04 and arousal < 0.60:
            event = "release"
        elif accent > 0.70 or (beat > 0.5 and onset > 0.55):
            event = "accent"
        else:
            event = "neutral_flow"
        events.append(event)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--num_frames", type=int, default=150)
    args = ap.parse_args()

    feat, meta = extract_audio_features(args.audio, args.num_frames)
    events = classify_frame_events(feat)
    out_npy = Path(args.out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, feat.astype(np.float32))
    out_json = Path(args.out_json) if args.out_json else out_npy.with_suffix(".json")
    payload = {
        "version": "v21_music_event_stream",
        "audio": str(args.audio),
        "npy": str(out_npy),
        "num_frames": int(len(feat)),
        "feature_dim": int(feat.shape[1]),
        "meta": meta,
        "events": events,
        "event_counts": {name: events.count(name) for name in sorted(set(events))},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved_npy:", out_npy)
    print("saved_json:", out_json)
    print("event_counts:", payload["event_counts"])


if __name__ == "__main__":
    main()
