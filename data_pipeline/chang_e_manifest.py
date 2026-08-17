#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoritative Chang-E source metadata and timebase validation.

The local BVH headers are not a reliable sampling-rate authority: eleven files
declare roughly 24 FPS even though their frame counts align with the published
sequence durations at 60 FPS.  Every formal Chang-E path therefore resolves
effective FPS, recording identity and source provenance from ``sources.json``.
Unknown datasets may still use their BVH header when strict manifest mode is
not requested.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "motion" / "bvh" / "sources.json"
MANIFEST_SCHEMA = "chang_e_source_manifest_v2"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=64)
def _sha256_file_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    source = Path(path).resolve()
    stat = source.stat()
    return _sha256_file_cached(str(source), int(stat.st_size), int(stat.st_mtime_ns))


def manifest_path(path: str | Path | None = None) -> Path:
    raw = path or os.environ.get("CHANG_E_SOURCE_MANIFEST") or DEFAULT_MANIFEST
    return Path(raw).expanduser().resolve()


def manifest_sha256(path: str | Path | None = None) -> str:
    target = manifest_path(path)
    return _sha256_bytes(target.read_bytes())


def load_manifest(
    path: str | Path | None = None,
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    target = manifest_path(path)
    if not target.is_file():
        if required:
            raise FileNotFoundError(f"Chang-E source manifest does not exist: {target}")
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Chang-E source manifest must be a JSON object: {target}")
    if str(payload.get("schema", "")) != MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported Chang-E source manifest schema {payload.get('schema')!r}; "
            f"expected {MANIFEST_SCHEMA!r}"
        )
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Chang-E source manifest requires a non-empty sources list")
    if int(payload.get("num_sources", -1)) != len(sources):
        raise ValueError(
            "Chang-E manifest num_sources does not match sources length: "
            f"{payload.get('num_sources')!r}!={len(sources)}"
        )

    files: set[str] = set()
    source_ids: set[str] = set()
    for index, raw in enumerate(sources):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Chang-E source row {index} is not an object")
        source_id = str(raw.get("source_id", "")).strip()
        filename = Path(str(raw.get("file", ""))).name
        recording_uid = str(raw.get("recording_uid", "")).strip()
        effective_fps = float(raw.get("effective_fps", 0.0))
        if not source_id or not filename or not recording_uid:
            raise ValueError(
                f"Chang-E source row {index} requires source_id, file and recording_uid"
            )
        if source_id in source_ids or filename.lower() in files:
            raise ValueError(f"Duplicate Chang-E source_id/file at row {index}")
        if not math.isfinite(effective_fps) or effective_fps <= 0.0:
            raise ValueError(f"Invalid effective_fps for {source_id}: {effective_fps!r}")
        expected_hash = str(raw.get("sha256", "")).strip().lower()
        if len(expected_hash) != 64:
            raise ValueError(f"Missing/invalid SHA256 for Chang-E source {source_id}")
        source_ids.add(source_id)
        files.add(filename.lower())
    return payload


def find_source_entry(
    source: str | Path,
    *,
    path: str | Path | None = None,
    required: bool = False,
) -> Optional[Dict[str, Any]]:
    payload = load_manifest(path, required=required)
    if payload is None:
        return None
    token = Path(str(source)).name.lower()
    stem = Path(str(source)).stem.lower()
    matches = [
        dict(row)
        for row in payload["sources"]
        if Path(str(row.get("file", ""))).name.lower() == token
        or str(row.get("source_id", "")).lower() == stem
    ]
    if not matches:
        if required:
            raise KeyError(f"Source is absent from Chang-E manifest: {source}")
        return None
    if len(matches) != 1:
        raise ValueError(f"Ambiguous Chang-E manifest source match: {source}")
    entry = matches[0]
    entry["manifest_path"] = str(manifest_path(path))
    entry["manifest_sha256"] = manifest_sha256(path)
    return entry


def read_bvh_header(path: str | Path) -> Tuple[int, float]:
    frames: Optional[int] = None
    frame_time: Optional[float] = None
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            lower = line.lower()
            if lower.startswith("frames"):
                frames = int(line.replace(":", " ").split()[-1])
            elif lower.startswith("frame time"):
                frame_time = float(line.replace(":", " ").split()[-1])
                break
    if frames is None or frame_time is None or frame_time <= 0.0:
        raise ValueError(f"BVH frame metadata is missing or invalid: {path}")
    return int(frames), float(frame_time)


def validate_source(
    source: str | Path,
    *,
    path: str | Path | None = None,
    required: bool = True,
    verify_hash: bool = True,
) -> Dict[str, Any]:
    source_path = Path(source).resolve()
    entry = find_source_entry(source_path, path=path, required=required)
    if (
        entry is not None
        and not required
        and path is None
        and not os.environ.get("CHANG_E_SOURCE_MANIFEST")
        and source_path.parent != Path(str(entry["manifest_path"])).parent
    ):
        # A generic dataset may reuse one of the Chang-E filenames.  Automatic
        # manifest resolution is only safe inside the canonical source folder;
        # formal callers pass an explicit manifest and strict mode.
        entry = None
    frames, frame_time = read_bvh_header(source_path)
    declared_fps = 1.0 / frame_time
    if entry is None:
        return {
            "manifest_resolved": False,
            "source": str(source_path),
            "declared_frame_time_seconds": float(frame_time),
            "declared_fps": float(declared_fps),
            "effective_fps": float(declared_fps),
            "frames": int(frames),
            "duration_seconds": float(max(0, frames - 1) / declared_fps),
        }

    expected_frames = int(entry.get("expected_frames", -1))
    expected_frame_time = float(entry.get("declared_frame_time_seconds", 0.0))
    effective_fps = float(entry["effective_fps"])
    errors: list[str] = []
    if expected_frames != frames:
        errors.append(f"frame_count:{frames}!={expected_frames}")
    if not math.isclose(frame_time, expected_frame_time, rel_tol=0.0, abs_tol=1.0e-7):
        errors.append(
            f"declared_frame_time:{frame_time:.9f}!={expected_frame_time:.9f}"
        )
    actual_hash = file_sha256(source_path) if verify_hash else None
    if actual_hash is not None and actual_hash != str(entry["sha256"]).lower():
        errors.append(f"sha256:{actual_hash}!={entry['sha256']}")

    duration = float(max(0, frames - 1) / effective_fps)
    published = entry.get("published_duration_seconds")
    if isinstance(published, list) and published:
        distance = min(abs(duration - float(value)) for value in published)
    elif published is not None:
        distance = abs(duration - float(published))
    else:
        distance = 0.0
    tolerance = float(entry.get("published_duration_tolerance_seconds", 3.0))
    if distance > tolerance:
        errors.append(
            f"published_duration:{duration:.3f}s outside tolerance {tolerance:.3f}s"
        )
    if errors:
        raise ValueError(
            f"Chang-E source provenance mismatch for {source_path.name}: "
            + "; ".join(errors)
        )

    return {
        "manifest_resolved": True,
        "source": str(source_path),
        "source_id": str(entry["source_id"]),
        "recording_uid": str(entry["recording_uid"]),
        "performer_track_id": entry.get("performer_track_id"),
        "sequence_index": entry.get("sequence_index"),
        "declared_frame_time_seconds": float(frame_time),
        "declared_fps": float(declared_fps),
        "effective_fps": float(effective_fps),
        "frames": int(frames),
        "duration_seconds": duration,
        "header_to_effective_fps_ratio": float(declared_fps / effective_fps),
        "source_sha256": actual_hash or str(entry["sha256"]).lower(),
        "manifest_path": str(entry["manifest_path"]),
        "manifest_sha256": str(entry["manifest_sha256"]),
        "entry": {
            key: value
            for key, value in entry.items()
            if key not in {"manifest_path", "manifest_sha256"}
        },
    }


def semantic_metadata(
    source: str | Path,
    *,
    path: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    entry = find_source_entry(source, path=path, required=False)
    if entry is None:
        return None
    source_path = Path(source).expanduser()
    if (
        path is None
        and not os.environ.get("CHANG_E_SOURCE_MANIFEST")
        and source_path.exists()
        and source_path.resolve().parent
        != Path(str(entry["manifest_path"])).parent
    ):
        return None
    keys = (
        "source_id",
        "recording_uid",
        "performer_track_id",
        "sequence_index",
        "performer_group",
        "dance_category",
        "take_id",
        "skeleton_id",
        "effective_fps",
    )
    return {key: entry.get(key) for key in keys}
