#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attach real CLAP and temporal features to scheduler slot JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from grounding.paired_data import (
    DEFAULT_TEMPORAL_FRAMES,
    _resample_sequence,
)
from scheduling.audio_features import extract_audio_features
from scheduling.deep_music_features import phrase_deep_embedding_matrix


SCHEMA = "v46_53_mixed_grounding_audio_query_v1"


def _slot_list(payload: Any) -> tuple[list[dict[str, Any]], Optional[str]]:
    if isinstance(payload, list):
        rows = payload
        key = None
    elif isinstance(payload, Mapping):
        key = next(
            (
                candidate
                for candidate in ("slots", "schedule", "phrases", "events")
                if isinstance(payload.get(candidate), list)
            ),
            None,
        )
        if key is None:
            raise RuntimeError(
                "Schedule JSON must be a list or contain slots/schedule/phrases/events"
            )
        rows = payload[key]
    else:
        raise RuntimeError("Schedule JSON root must be an object or list")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise RuntimeError("Schedule slot list is empty or contains non-object rows")
    return [dict(row) for row in rows], key


def _slot_interval(slot: Mapping[str, Any], index: int) -> tuple[float, float]:
    start = float(slot.get("start_sec", slot.get("start", 0.0)))
    end = float(
        slot.get(
            "end_sec",
            slot.get("end", start + float(slot.get("duration", 2.0))),
        )
    )
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise RuntimeError(
            f"Invalid audio interval in schedule slot {index}: [{start}, {end}]"
        )
    return start, end


def enrich_schedule_audio(
    audio_path: Path,
    schedule_path: Path,
    out_path: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    model_name: str = "clap",
    cache_dir: Optional[Path] = None,
    temporal_frames: int = DEFAULT_TEMPORAL_FRAMES,
    temporal_source_frames: int = 2048,
    phrase_fps: float = 30.0,
) -> dict[str, Any]:
    audio = audio_path.expanduser().resolve()
    if not audio.is_file():
        raise FileNotFoundError(str(audio))
    original = json.loads(schedule_path.read_text(encoding="utf-8-sig"))
    slots, container_key = _slot_list(original)
    intervals = [_slot_interval(slot, index) for index, slot in enumerate(slots)]
    phrases = [
        SimpleNamespace(
            start=int(round(start * phrase_fps)),
            end=int(round(end * phrase_fps)),
            length=int(round((end - start) * phrase_fps)),
            music_event=str(slot.get("music_event", "neutral_flow")),
            energy=float(slot.get("energy", 0.5)),
            onset=float(slot.get("onset", 0.0)),
            beat_density=float(slot.get("beat_density", 0.0)),
            tension=float(slot.get("tension", 0.0)),
            calmness=float(slot.get("calmness", 0.0)),
            boundary_accent_strength=float(
                slot.get("boundary_accent_strength", 0.0)
            ),
        )
        for slot, (start, end) in zip(slots, intervals)
    ]
    clap, clap_meta = phrase_deep_embedding_matrix(
        audio,
        phrases,
        model_name=model_name,
        cache_dir=cache_dir,
        require_deep=True,
        min_deep_success=1.0,
        fps=phrase_fps,
    )
    checkpoint_contract: Optional[dict[str, Any]] = None
    if checkpoint_path is not None:
        from grounding.mixed_curvature import (
            MixedGrounderConfig,
            _load_torch_checkpoint,
        )

        checkpoint = _load_torch_checkpoint(
            checkpoint_path.expanduser().resolve()
        )
        config = MixedGrounderConfig(**dict(checkpoint["config"]))
        if int(clap.shape[1]) != int(config.clap_dim):
            raise RuntimeError(
                "Runtime CLAP dimension does not match mixed-grounder "
                f"checkpoint: audio={clap.shape[1]}, "
                f"checkpoint={config.clap_dim}"
            )
        if int(config.temporal_dim) != 12:
            raise RuntimeError(
                "Mixed-grounder checkpoint temporal feature dimension is "
                f"{config.temporal_dim}, but the runtime extractor emits 12"
            )
        checkpoint_contract = {
            "path": str(checkpoint_path.expanduser().resolve()),
            "schema": str(checkpoint["schema"]),
            "clap_dim": int(config.clap_dim),
            "temporal_dim": int(config.temporal_dim),
        }
    temporal_full, temporal_meta = extract_audio_features(
        audio, num_frames=int(temporal_source_frames)
    )
    duration = float(
        temporal_meta.get(
            "duration_sec", max(end for _, end in intervals)
        )
    )
    enriched: list[dict[str, Any]] = []
    for index, (slot, (start, end)) in enumerate(zip(slots, intervals)):
        begin = int(np.floor(start / max(duration, 1.0e-6) * len(temporal_full)))
        finish = int(np.ceil(end / max(duration, 1.0e-6) * len(temporal_full)))
        begin = int(np.clip(begin, 0, max(len(temporal_full) - 1, 0)))
        finish = int(np.clip(finish, begin + 1, len(temporal_full)))
        temporal = _resample_sequence(
            temporal_full[begin:finish], int(temporal_frames)
        )
        enriched.append(
            {
                **slot,
                "clap_embedding": clap[index].astype(float).tolist(),
                "temporal_features": temporal.astype(float).tolist(),
                "mixed_audio_feature_schema": SCHEMA,
                "mixed_audio_feature_interval_sec": [float(start), float(end)],
            }
        )
    if container_key is None:
        output: Any = enriched
    else:
        output = dict(original)
        output[container_key] = enriched
        output["mixed_audio_features"] = {
            "schema": SCHEMA,
            "audio": str(audio),
            "clap": clap_meta,
            "temporal": temporal_meta,
            "temporal_frames": int(temporal_frames),
            "checkpoint_contract": checkpoint_contract,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "schema": SCHEMA,
        "audio": str(audio),
        "input_schedule": str(schedule_path.resolve()),
        "output_schedule": str(out_path.resolve()),
        "slots": len(enriched),
        "clap_dim": int(clap.shape[1]),
        "temporal_frames": int(temporal_frames),
        "checkpoint_contract": checkpoint_contract,
        "ok": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--model_name", default="clap")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--temporal_frames", type=int, default=DEFAULT_TEMPORAL_FRAMES)
    parser.add_argument("--temporal_source_frames", type=int, default=2048)
    parser.add_argument("--phrase_fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    report = enrich_schedule_audio(
        Path(args.audio),
        Path(args.schedule),
        Path(args.out),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        model_name=args.model_name,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        temporal_frames=args.temporal_frames,
        temporal_source_frames=args.temporal_source_frames,
        phrase_fps=args.phrase_fps,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
