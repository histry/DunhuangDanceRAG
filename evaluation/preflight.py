#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.53.1 real-data preflight before expensive retargeting/training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def _count_audio(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def _check_file(path: Path, label: str, errors: List[str]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        errors.append(f"missing {label}: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--audio", required=True)
    ap.add_argument("--music_dir", required=True)
    ap.add_argument("--change_dir", default="change")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    audio = Path(args.audio).resolve()
    music_dir = Path(args.music_dir).resolve()
    change_dir = Path(args.change_dir)
    if not change_dir.is_absolute(): change_dir = (root / change_dir).resolve()
    errors: List[str] = []
    warnings: List[str] = []

    _check_file(audio, "input audio", errors)
    if not music_dir.is_dir(): errors.append(f"missing training music directory: {music_dir}")
    if "test_music_bank" in str(music_dir): errors.append("test_music_bank must not enter V44 training")
    if not change_dir.is_dir(): errors.append(f"missing change source directory: {change_dir}")

    required = [
        "contracts/anatomy.py",
        "retargeting/anatomy_retarget.py",
        "retargeting/build_cache.py",
        "data_pipeline/split_sources.py",
        "events/filter_anatomy.py",
        "motion_geometry/rotations.py",
        "events/build_pipeline.py",
        "grounding/model.py",
        "scripts/pipeline.sh",
        "configs/scheduler.env",
        "configs/research.env",
    ]
    for rel in required: _check_file(root / rel, rel, errors)

    sources = sorted(change_dir.rglob("*.bvh")) if change_dir.is_dir() else []
    min_sources = int(float(os.environ.get("V46_52_MIN_OK_SOURCES", "8")))
    if len(sources) < min_sources:
        errors.append(f"change BVH count={len(sources)} < minimum source requirement={min_sources}")
    if len(sources) < 12:
        warnings.append(f"expected project inventory is 12 BVH sources; discovered {len(sources)}")

    music_count = _count_audio(music_dir) if music_dir.is_dir() else 0
    expected_music = int(float(os.environ.get("V46_53_1_EXPECTED_TRAIN_MUSIC", "788")))
    if expected_music > 0 and music_count != expected_music:
        errors.append(f"training music count={music_count}; expected={expected_music}")

    runtime: Dict[str, Any] = {}
    try:
        import torch
        runtime.update({
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
        if not torch.cuda.is_available(): errors.append("CUDA is unavailable")
    except Exception as exc:
        errors.append(f"PyTorch import failed: {exc}")

    try:
        from data_pipeline.split_sources import exact_split_counts
        runtime["split_counts_at_discovered_sources"] = exact_split_counts(
            max(3, len(sources)),
            float(os.environ.get("V46_51_TRAIN_RATIO", "0.67")),
            float(os.environ.get("V46_51_VAL_RATIO", "0.165")),
            float(os.environ.get("V46_51_TEST_RATIO", "0.165")),
        )
    except Exception as exc:
        errors.append(f"split contract import/self-test failed: {exc}")

    report = {
        "schema": "real_data_preflight",
        "root": str(root), "audio": str(audio), "music_dir": str(music_dir), "change_dir": str(change_dir),
        "bvh_sources": [str(p) for p in sources], "num_bvh_sources": len(sources),
        "training_music_count": music_count, "expected_training_music_count": expected_music,
        "runtime": runtime, "warnings": warnings, "errors": errors, "ok": not errors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
