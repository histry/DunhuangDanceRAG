#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight for the Chang-E official-SMPL source-aware main route."""
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

from data_pipeline.chang_e_manifest import load_manifest, manifest_sha256
from retargeting.official_smpl_source_preprocess import (
    discover_official_smpl_files,
    load_name_map,
    match_manifest_entry,
)

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def _count_audio(path: Path) -> int:
    return sum(
        1
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXT
    )


def _check_file(path: Path, label: str, errors: List[str]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        errors.append(f"missing {label}: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--audio", required=True)
    ap.add_argument("--music_dir", required=True)
    ap.add_argument("--smpl_dir", required=True)
    ap.add_argument("--source_manifest", required=True)
    ap.add_argument("--name_map", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    audio = Path(args.audio).resolve()
    music_dir = Path(args.music_dir).resolve()
    smpl_dir = Path(args.smpl_dir).expanduser().resolve()
    manifest_path = Path(args.source_manifest).expanduser().resolve()
    errors: List[str] = []
    warnings: List[str] = []

    _check_file(audio, "input audio", errors)
    if not music_dir.is_dir():
        errors.append(f"missing training music directory: {music_dir}")
    if "test_music_bank" in str(music_dir):
        errors.append("test_music_bank must not enter training")
    if not smpl_dir.is_dir():
        errors.append(f"missing official SMPL directory: {smpl_dir}")
    _check_file(manifest_path, "Chang-E source manifest", errors)

    required = [
        "retargeting/official_smpl_source_preprocess.py",
        "data_pipeline/split_sources.py",
        "events/build_database.py",
        "events/filter_anatomy.py",
        "training/motion_models.py",
        "training/music_router.py",
        "training/duration_model.py",
        "training/whole_song_planner.py",
        "routing/closed_loop.py",
        "rendering/render_motion.py",
        "scripts/pipeline.sh",
        "configs/experiment.env",
    ]
    for rel in required:
        _check_file(root / rel, rel, errors)

    source_rows: List[Dict[str, Any]] = []
    source_ids: set[str] = set()
    recording_uids: set[str] = set()
    discovered = discover_official_smpl_files(smpl_dir) if smpl_dir.is_dir() else []
    name_map = load_name_map(
        Path(args.name_map).expanduser().resolve() if args.name_map else None
    )
    try:
        manifest = load_manifest(manifest_path, required=True)
        assert manifest is not None
        for source in discovered:
            explicit = name_map.get(source.name) or name_map.get(
                str(source.relative_to(smpl_dir))
            )
            row = match_manifest_entry(
                source, manifest, explicit_source_id=explicit
            )
            source_id = str(row["source_id"])
            if source_id in source_ids:
                errors.append(
                    f"duplicate official SMPL mapping for source_id={source_id}"
                )
                continue
            source_ids.add(source_id)
            recording_uids.add(str(row["recording_uid"]))
            source_rows.append(
                {
                    "source": str(source),
                    "source_id": source_id,
                    "recording_uid": str(row["recording_uid"]),
                    "performer_group": row.get("performer_group", "unknown"),
                    "dance_category": row.get("dance_category", "unknown"),
                    "effective_fps": float(row.get("effective_fps", 60.0)),
                }
            )
    except Exception as exc:
        errors.append(f"official SMPL / Chang-E manifest resolution failed: {exc}")

    min_sources = int(float(os.environ.get("RETARGET_MIN_OK_SOURCES", "8")))
    if len(source_ids) < min_sources:
        errors.append(
            f"official SMPL source count={len(source_ids)} "
            f"< minimum source requirement={min_sources}"
        )
    if len(recording_uids) < 3:
        errors.append(
            f"recording groups={len(recording_uids)} < minimum split requirement=3"
        )

    music_count = _count_audio(music_dir) if music_dir.is_dir() else 0
    expected_music = int(
        float(os.environ.get("RETARGET_CLEAN_EXPECTED_TRAIN_MUSIC", "788"))
    )
    if expected_music > 0 and music_count != expected_music:
        errors.append(
            f"training music count={music_count}; expected={expected_music}"
        )

    runtime: Dict[str, Any] = {}
    try:
        import torch

        runtime.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
        if not torch.cuda.is_available():
            errors.append("CUDA is unavailable")
    except Exception as exc:
        errors.append(f"PyTorch import failed: {exc}")

    try:
        from data_pipeline.split_sources import exact_split_counts

        runtime["split_counts_at_recording_groups"] = exact_split_counts(
            max(3, len(recording_uids)),
            float(os.environ.get("GENERATION_TRAIN_RATIO", "0.67")),
            float(os.environ.get("GENERATION_VAL_RATIO", "0.165")),
            float(os.environ.get("GENERATION_TEST_RATIO", "0.165")),
        )
    except Exception as exc:
        errors.append(f"split contract import/self-test failed: {exc}")

    report = {
        "schema": "chang_e_official_smpl_source_aware_preflight_v1",
        "root": str(root),
        "audio": str(audio),
        "music_dir": str(music_dir),
        "smpl_dir": str(smpl_dir),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": (
            manifest_sha256(manifest_path) if manifest_path.is_file() else None
        ),
        "official_smpl_files": [str(p) for p in discovered],
        "num_official_smpl_files": len(discovered),
        "num_source_ids": len(source_ids),
        "source_rows": source_rows,
        "num_recording_groups": len(recording_uids),
        "recording_uids": sorted(recording_uids),
        "training_music_count": music_count,
        "expected_training_music_count": expected_music,
        "runtime": runtime,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
