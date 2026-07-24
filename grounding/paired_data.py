#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an auditable real-audio/motion dataset for mixed-curvature grounding.

The legacy grounder intentionally supports unpaired semantic learning.  The
research architecture in this module has a stricter contract: every row names a
stable ``event_uid`` and either supplies precomputed audio features or a real
audio interval from which unprojected CLAP and temporal features are extracted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from events.intrinsic_geometry import (
    GAUSSIAN_BODY_PARTS,
    GAUSSIAN_FEATURE_DIM,
)
from scheduling.audio_features import extract_audio_features
from scheduling.deep_music_features import phrase_deep_embedding_matrix
from support.event_identity import (
    event_uids_from_generation_db,
    make_event_db_contract,
)


SCHEMA = "v46_53_real_audio_motion_paired_grounding_v1"
TEMPORAL_DIM = 12
DEFAULT_TEMPORAL_FRAMES = 64
CONTROL_NAMES = (
    "duration_normalized",
    "combined_quality",
    "semantic_confidence",
    "event_position_mid",
)


def _load_npz_mapping(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        payload = json.loads(text)
        rows = payload.get("pairs", []) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Pair manifest is empty or invalid: {path}")
    if not all(isinstance(row, Mapping) for row in rows):
        raise RuntimeError("Every pair-manifest row must be a JSON object")
    return [dict(row) for row in rows]


def _resample_sequence(sequence: np.ndarray, target_frames: int) -> np.ndarray:
    value = np.asarray(sequence, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != TEMPORAL_DIM:
        raise ValueError(
            f"temporal audio features must be [T,{TEMPORAL_DIM}], got {value.shape}"
        )
    target = int(target_frames)
    if target < 2:
        raise ValueError("temporal_frames must be at least two")
    if len(value) == target:
        return value
    if len(value) == 0:
        raise ValueError("temporal audio features cannot be empty")
    if len(value) == 1:
        return np.broadcast_to(value, (target, value.shape[1])).copy()
    old = np.linspace(0.0, 1.0, len(value), dtype=np.float64)
    new = np.linspace(0.0, 1.0, target, dtype=np.float64)
    return np.stack(
        [np.interp(new, old, value[:, index]) for index in range(value.shape[1])],
        axis=-1,
    ).astype(np.float32)


def _feature_payload_from_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            clap = np.asarray(
                loaded["clap"]
                if "clap" in loaded.files
                else loaded["clap_embedding"],
                dtype=np.float32,
            ).reshape(-1)
            temporal = np.asarray(
                loaded["temporal"]
                if "temporal" in loaded.files
                else loaded["temporal_features"],
                dtype=np.float32,
            )
        finally:
            loaded.close()
        return clap, temporal
    array = np.asarray(loaded, dtype=np.float32)
    if array.ndim != 1:
        raise RuntimeError(
            "A .npy audio feature path is interpreted as a CLAP vector; "
            f"use .npz with clap+temporal arrays for {path}"
        )
    raise RuntimeError(
        f"Audio feature file {path} provides CLAP only; temporal features are required"
    )


def _row_audio_features(
    row: Mapping[str, Any],
    *,
    manifest_dir: Path,
    model_name: str,
    cache_dir: Optional[Path],
    temporal_frames: int,
    temporal_source_frames: int,
    phrase_fps: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if "audio_feature_path" in row:
        feature_path = Path(str(row["audio_feature_path"])).expanduser()
        if not feature_path.is_absolute():
            feature_path = manifest_dir / feature_path
        clap, temporal = _feature_payload_from_path(feature_path.resolve())
        mode = {"mode": "precomputed", "path": str(feature_path.resolve())}
    elif "clap_features" in row and "temporal_features" in row:
        clap = np.asarray(row["clap_features"], dtype=np.float32).reshape(-1)
        temporal = np.asarray(row["temporal_features"], dtype=np.float32)
        mode = {"mode": "manifest_arrays"}
    else:
        if "audio_path" not in row:
            raise RuntimeError(
                "Each pair row requires audio_feature_path, explicit "
                "clap_features+temporal_features, or audio_path"
            )
        audio_path = Path(str(row["audio_path"])).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest_dir / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(str(audio_path))
        start_sec = float(row.get("start_sec", 0.0))
        end_sec = float(row.get("end_sec", start_sec + 2.0))
        if not np.isfinite(start_sec) or not np.isfinite(end_sec) or end_sec <= start_sec:
            raise RuntimeError(
                f"Invalid audio interval [{start_sec}, {end_sec}] for {audio_path}"
            )
        phrase = SimpleNamespace(
            start=int(round(start_sec * phrase_fps)),
            end=int(round(end_sec * phrase_fps)),
            length=int(round((end_sec - start_sec) * phrase_fps)),
            music_event=str(row.get("music_event", "neutral_flow")),
            energy=float(row.get("energy", 0.5)),
            onset=float(row.get("onset", 0.0)),
            beat_density=float(row.get("beat_density", 0.0)),
            tension=float(row.get("tension", 0.0)),
            calmness=float(row.get("calmness", 0.0)),
            boundary_accent_strength=float(
                row.get("boundary_accent_strength", 0.0)
            ),
        )
        clap_matrix, clap_meta = phrase_deep_embedding_matrix(
            audio_path,
            [phrase],
            model_name=model_name,
            cache_dir=cache_dir,
            require_deep=True,
            min_deep_success=1.0,
            fps=phrase_fps,
        )
        clap = clap_matrix[0]
        whole_temporal, temporal_meta = extract_audio_features(
            audio_path, num_frames=int(temporal_source_frames)
        )
        duration = float(
            temporal_meta.get(
                "duration_sec", max(end_sec, start_sec + 1.0e-6)
            )
        )
        begin = int(np.floor(start_sec / max(duration, 1.0e-6) * len(whole_temporal)))
        finish = int(np.ceil(end_sec / max(duration, 1.0e-6) * len(whole_temporal)))
        begin = int(np.clip(begin, 0, max(len(whole_temporal) - 1, 0)))
        finish = int(np.clip(finish, begin + 1, len(whole_temporal)))
        temporal = whole_temporal[begin:finish]
        mode = {
            "mode": "real_audio_extracted",
            "audio_path": str(audio_path),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "clap": clap_meta,
            "temporal": temporal_meta,
        }

    if clap.ndim != 1 or clap.size == 0 or not np.isfinite(clap).all():
        raise RuntimeError(f"Invalid CLAP feature vector: {clap.shape}")
    clap_norm = float(np.linalg.norm(clap))
    if clap_norm <= 1.0e-8:
        raise RuntimeError("CLAP feature vector has zero norm")
    clap = (clap / clap_norm).astype(np.float32)
    temporal = _resample_sequence(temporal, temporal_frames)
    if not np.isfinite(temporal).all():
        raise RuntimeError("Temporal audio features contain NaN or Inf")
    return clap, temporal, mode


def validate_paired_payload(payload: Mapping[str, Any]) -> Dict[str, int]:
    """Validate aligned shapes and SPD constraints without requiring PyTorch."""

    required = (
        "clap",
        "temporal",
        "motion_geometry",
        "bodypart_flow",
        "gaussian_mean",
        "gaussian_covariance",
        "controls",
        "quality",
        "pair_ids",
        "family_ids",
        "source_ids",
        "event_indices",
        "event_uids",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Paired grounding payload misses fields: {missing}")
    arrays = {key: np.asarray(payload[key]) for key in required}
    rows = int(len(arrays["clap"]))
    if rows < 2:
        raise RuntimeError("Mixed-curvature training requires at least two pair rows")
    misaligned = {
        key: int(len(value))
        for key, value in arrays.items()
        if value.ndim == 0 or len(value) != rows
    }
    if "audio_group_ids" in payload:
        audio_groups = np.asarray(payload["audio_group_ids"])
        if audio_groups.ndim == 0 or len(audio_groups) != rows:
            misaligned["audio_group_ids"] = int(
                len(audio_groups) if audio_groups.ndim else 0
            )
    if misaligned:
        raise RuntimeError(
            f"Paired grounding arrays are not row-aligned: rows={rows}, {misaligned}"
        )
    if arrays["clap"].ndim != 2 or arrays["clap"].shape[1] < 2:
        raise RuntimeError(f"CLAP features must be [N,C>=2], got {arrays['clap'].shape}")
    if (
        arrays["temporal"].ndim != 3
        or arrays["temporal"].shape[2] != TEMPORAL_DIM
    ):
        raise RuntimeError(
            f"Temporal features must be [N,T,{TEMPORAL_DIM}], "
            f"got {arrays['temporal'].shape}"
        )
    if arrays["motion_geometry"].ndim != 2:
        raise RuntimeError("motion_geometry must be a two-dimensional matrix")
    if arrays["bodypart_flow"].ndim != 3:
        raise RuntimeError("bodypart_flow must be [N,B,F]")
    if (
        arrays["gaussian_mean"].ndim != 3
        or arrays["gaussian_covariance"].ndim != 4
        or arrays["gaussian_covariance"].shape[-1]
        != arrays["gaussian_covariance"].shape[-2]
        or arrays["gaussian_mean"].shape[:2]
        != arrays["gaussian_covariance"].shape[:2]
        or arrays["gaussian_mean"].shape[-1]
        != arrays["gaussian_covariance"].shape[-1]
    ):
        raise RuntimeError(
            "Gaussian means/covariances have incompatible shapes: "
            f"{arrays['gaussian_mean'].shape}, "
            f"{arrays['gaussian_covariance'].shape}"
        )
    numeric = (
        "clap",
        "temporal",
        "motion_geometry",
        "bodypart_flow",
        "gaussian_mean",
        "gaussian_covariance",
        "controls",
        "quality",
    )
    for key in numeric:
        if not np.isfinite(np.asarray(payload[key], dtype=np.float64)).all():
            raise RuntimeError(f"Paired grounding field {key} contains NaN or Inf")
    symmetric = 0.5 * (
        arrays["gaussian_covariance"]
        + np.swapaxes(arrays["gaussian_covariance"], -1, -2)
    )
    if float(np.linalg.eigvalsh(symmetric).min()) <= 0.0:
        raise RuntimeError("Gaussian covariance matrices must be strictly SPD")
    if np.any(np.asarray(arrays["quality"], dtype=np.float64) <= 0.0):
        raise RuntimeError("Pair quality weights must be strictly positive")
    return {
        "rows": rows,
        "clap_dim": int(arrays["clap"].shape[1]),
        "temporal_frames": int(arrays["temporal"].shape[1]),
        "temporal_dim": int(arrays["temporal"].shape[2]),
        "motion_geometry_dim": int(arrays["motion_geometry"].shape[1]),
        "bodypart_count": int(arrays["gaussian_mean"].shape[1]),
        "gaussian_dim": int(arrays["gaussian_mean"].shape[2]),
        "control_dim": int(arrays["controls"].shape[1]),
    }


def build_paired_dataset(
    event_db_path: Path,
    manifest_path: Path,
    out_path: Path,
    *,
    model_name: str = "clap",
    cache_dir: Optional[Path] = None,
    temporal_frames: int = DEFAULT_TEMPORAL_FRAMES,
    temporal_source_frames: int = 1024,
    phrase_fps: float = 30.0,
) -> Dict[str, Any]:
    db = _load_npz_mapping(event_db_path)
    required_db = (
        "v46_53_geometry_desc",
        "v46_53_bodypart_flow",
        "v46_53_bodypart_gaussian_mean",
        "v46_53_bodypart_gaussian_covariance",
    )
    missing = [key for key in required_db if key not in db]
    if missing:
        raise RuntimeError(
            "Event-DB lacks mixed-grounder motion fields. Re-run intrinsic "
            f"geometry augmentation; missing={missing}"
        )
    event_uids = event_uids_from_generation_db(db)
    uid_to_index = {str(uid): index for index, uid in enumerate(event_uids)}
    if len(uid_to_index) != len(event_uids):
        raise RuntimeError("Event-DB contains duplicate stable event_uids")
    rows = _load_manifest(manifest_path)
    manifest_dir = manifest_path.resolve().parent

    clap_rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    event_indices: list[int] = []
    pair_tokens: list[str] = []
    audio_group_tokens: list[str] = []
    supervision: list[str] = []
    audio_reports: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows):
        uid = str(row.get("event_uid", "")).strip()
        if not uid:
            raise RuntimeError(f"Pair row {row_number} has no event_uid")
        if uid not in uid_to_index:
            raise RuntimeError(
                f"Pair row {row_number} references unknown event_uid={uid}"
            )
        clap, temporal, audio_report = _row_audio_features(
            row,
            manifest_dir=manifest_dir,
            model_name=model_name,
            cache_dir=cache_dir,
            temporal_frames=temporal_frames,
            temporal_source_frames=temporal_source_frames,
            phrase_fps=phrase_fps,
        )
        clap_rows.append(clap)
        temporal_rows.append(temporal)
        event_indices.append(int(uid_to_index[uid]))
        pair_token = str(
            row.get(
                "pair_id",
                f"{row.get('audio_path', 'features')}::"
                f"{row.get('start_sec', 0.0)}::{row.get('end_sec', 0.0)}",
            )
        )
        pair_tokens.append(pair_token)
        explicit_audio_group = str(row.get("audio_group_id", "")).strip()
        if explicit_audio_group:
            audio_group_token = explicit_audio_group
        elif audio_report.get("mode") == "real_audio_extracted":
            # Keep every crop from one physical music file on the same side of
            # the split.  pair_id remains the finer segment/positive identity.
            audio_group_token = str(audio_report["audio_path"])
        elif audio_report.get("mode") == "precomputed":
            audio_group_token = str(audio_report["path"])
        else:
            # Explicit arrays do not expose a recoverable file identity.
            # Reusing pair_id keeps all declared positives on one side.
            audio_group_token = pair_token
        audio_group_tokens.append(audio_group_token)
        supervision.append(str(row.get("supervision", "unspecified")))
        audio_reports.append(audio_report)

    clap_dims = sorted({int(row.size) for row in clap_rows})
    if len(clap_dims) != 1:
        raise RuntimeError(f"CLAP dimensions differ across pair rows: {clap_dims}")
    indices = np.asarray(event_indices, dtype=np.int64)
    pair_vocabulary = {name: i for i, name in enumerate(sorted(set(pair_tokens)))}
    audio_group_vocabulary = {
        name: i for i, name in enumerate(sorted(set(audio_group_tokens)))
    }
    family_values = np.asarray(
        db.get("event_families", ["unknown"] * len(event_uids)), dtype=object
    )[indices]
    source_values = np.asarray(
        db.get("source_uids", ["unknown"] * len(event_uids)), dtype=object
    )[indices]
    family_vocabulary = {
        name: i for i, name in enumerate(sorted({str(v) for v in family_values}))
    }
    source_vocabulary = {
        name: i for i, name in enumerate(sorted({str(v) for v in source_values}))
    }
    durations = np.asarray(
        db.get("durations", np.ones(len(event_uids), dtype=np.float32) * 2.0),
        dtype=np.float32,
    )[indices]
    quality = np.clip(
        np.asarray(
            db.get(
                "v46_53_combined_quality",
                np.ones(len(event_uids), dtype=np.float32) * 0.5,
            ),
            dtype=np.float32,
        )[indices],
        1.0e-3,
        1.0,
    )
    semantic_confidence = np.clip(
        np.asarray(
            db.get(
                "semantic_confidence",
                np.ones(len(event_uids), dtype=np.float32) * 0.5,
            ),
            dtype=np.float32,
        )[indices],
        0.0,
        1.0,
    )
    position = np.clip(
        np.asarray(
            db.get(
                "event_position_mid", np.ones(len(event_uids), dtype=np.float32) * 0.5
            ),
            dtype=np.float32,
        )[indices],
        0.0,
        1.0,
    )
    controls = np.stack(
        [
            np.clip(durations / 6.0, 0.0, 2.0),
            quality,
            semantic_confidence,
            position,
        ],
        axis=-1,
    ).astype(np.float32)
    event_contract = make_event_db_contract(event_uids.tolist())

    payload: Dict[str, Any] = {
        "schema": np.asarray(SCHEMA, dtype=object),
        "clap": np.stack(clap_rows).astype(np.float32),
        "temporal": np.stack(temporal_rows).astype(np.float32),
        "motion_geometry": np.asarray(
            db["v46_53_geometry_desc"], dtype=np.float32
        )[indices],
        "bodypart_flow": np.asarray(
            db["v46_53_bodypart_flow"], dtype=np.float32
        )[indices, : len(GAUSSIAN_BODY_PARTS)],
        "gaussian_mean": np.asarray(
            db["v46_53_bodypart_gaussian_mean"], dtype=np.float32
        )[indices],
        "gaussian_covariance": np.asarray(
            db["v46_53_bodypart_gaussian_covariance"], dtype=np.float32
        )[indices],
        "controls": controls,
        "quality": quality.astype(np.float32),
        "pair_ids": np.asarray(
            [pair_vocabulary[value] for value in pair_tokens], dtype=np.int64
        ),
        "audio_group_ids": np.asarray(
            [audio_group_vocabulary[value] for value in audio_group_tokens],
            dtype=np.int64,
        ),
        "family_ids": np.asarray(
            [family_vocabulary[str(value)] for value in family_values],
            dtype=np.int64,
        ),
        "source_ids": np.asarray(
            [source_vocabulary[str(value)] for value in source_values],
            dtype=np.int64,
        ),
        "event_indices": indices,
        "event_uids": event_uids[indices],
        "supervision": np.asarray(supervision, dtype=object),
        "event_db_contract_json": np.asarray(
            json.dumps(event_contract, sort_keys=True), dtype=object
        ),
    }
    dimensions = validate_paired_payload(payload)
    metadata = {
        "schema": SCHEMA,
        "event_db": str(event_db_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "event_db_contract": event_contract,
        "dimensions": dimensions,
        "model_name": model_name,
        "unprojected_clap": True,
        "temporal_feature_names": [
            "energy",
            "onset",
            "beat",
            "tempo",
            "arousal",
            "delta_arousal",
            "tension",
            "calmness",
            "novelty",
            "brightness",
            "section_change",
            "accent",
        ],
        "control_names": list(CONTROL_NAMES),
        "gaussian_body_parts": list(GAUSSIAN_BODY_PARTS),
        "gaussian_feature_dim": GAUSSIAN_FEATURE_DIM,
        "supervision_counts": {
            label: supervision.count(label) for label in sorted(set(supervision))
        },
        "identity_groups": {
            "pair_ids": int(len(pair_vocabulary)),
            "audio_files_or_declared_groups": int(
                len(audio_group_vocabulary)
            ),
            "motion_sources": int(len(source_vocabulary)),
        },
        "audio_extraction": audio_reports,
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False), dtype=object
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    report_path = out_path.with_suffix(out_path.suffix + ".json")
    report_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        **metadata,
        "dataset": str(out_path.resolve()),
        "report": str(report_path.resolve()),
        "ok": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build strict real-audio/motion pairs for mixed grounding"
    )
    parser.add_argument("--event_db", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model_name", default="clap")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--temporal_frames", type=int, default=DEFAULT_TEMPORAL_FRAMES)
    parser.add_argument("--temporal_source_frames", type=int, default=1024)
    parser.add_argument("--phrase_fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    report = build_paired_dataset(
        Path(args.event_db),
        Path(args.manifest),
        Path(args.out),
        model_name=str(args.model_name),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        temporal_frames=int(args.temporal_frames),
        temporal_source_frames=int(args.temporal_source_frames),
        phrase_fps=float(args.phrase_fps),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
