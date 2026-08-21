#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoritative metadata/provenance contract for Chang-E official SMPL.

Formal DunhuangDanceRAG experiments use the fitted Chang-E SMPL parameter
sequences directly.  The historical BVH manifest is deliberately not consulted
by this module.

Timebase authority
------------------
The formal source FPS is the explicit ``source_fps`` recorded in the SMPL
manifest.  Embedded NPZ FPS metadata, when present, is retained as a diagnostic
but does not silently override the formal experiment contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    ROOT
    / "assets"
    / "motion"
    / "smpl_official_14"
    / "sources.json"
)

MANIFEST_SCHEMA = "chang_e_official_smpl_manifest_v2"
POSE_LAYOUT = "smplx55_axis_angle_body22_to_smpl24_hands_zero_v1"
COORDINATE_SYSTEM = "y_up"
TRANSLATION_UNITS = "m"
CANONICAL_SKELETON = "smpl24"
HAND_ROTATION_POLICY = "zero_unobserved"
SOURCE_FORMAT = "official_smpl_npz"
OFFICIAL_RELEASE_ID = "chang_e_aligned_smpl_14_v1"
TEST_RELEASE_ID = "unit_test_fixture"
OFFICIAL_SOURCE_IDS = {
    "female_36pose_1",
    "female_36pose_2",
    "female_FeiTian",
    "female_lotus",
    "female_meditation",
    "male_36pose_1",
    "male_36pose_2",
    "male_drum_1",
    "male_drum_2",
    "male_meditation",
    "male_pipa_1",
    "male_pipa_2",
    "male_ribbon",
    "male_ribbon_FenHe",
}

POSE_KEYS: Sequence[str] = (
    "smpl_poses",
    "poses",
    "pose",
    "smpl_pose",
    "body_pose",
    "full_pose",
)

FPS_KEYS: Sequence[str] = (
    "mocap_framerate",
    "fps",
    "frame_rate",
    "framerate",
)

TRANS_KEYS: Sequence[str] = (
    "smpl_trans",
    "trans",
    "transl",
    "translations",
    "root_translation",
    "root_trans",
)


def file_sha256(path: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    digest = hashlib.sha256()

    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def manifest_path(path: str | Path | None = None) -> Path:
    raw = (
        path
        or os.environ.get("CHANG_E_OFFICIAL_SMPL_MANIFEST")
        or DEFAULT_MANIFEST
    )
    return Path(raw).expanduser().resolve()


def manifest_sha256(path: str | Path | None = None) -> str:
    return file_sha256(manifest_path(path))


def _scalar(value: Any) -> Optional[float]:
    if value is None:
        return None

    array = np.asarray(value)

    if array.size == 0:
        return None

    result = float(array.reshape(-1)[0])

    if not math.isfinite(result):
        return None

    return result


def inspect_smpl_source(
    source: str | Path,
    *,
    source_fps: float,
) -> Dict[str, Any]:
    """Inspect one official Chang-E SMPL NPZ without modifying it."""

    path = Path(source).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"Formal Chang-E SMPL source must be .npz: {path}"
        )

    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"Invalid source_fps={source_fps!r}")

    with np.load(path, allow_pickle=True) as payload:
        pose_key = next(
            (key for key in POSE_KEYS if key in payload.files),
            None,
        )

        if pose_key is None:
            raise ValueError(
                f"No SMPL pose array in {path.name}; "
                f"keys={sorted(payload.files)}"
            )

        poses = np.asarray(payload[pose_key])

        if poses.ndim < 2:
            raise ValueError(
                f"Invalid pose shape in {path.name}: {poses.shape}"
            )

        frames = int(poses.shape[0])

        if frames < 2:
            raise ValueError(
                f"SMPL source too short: {path.name}, frames={frames}"
            )

        fps_key = next(
            (key for key in FPS_KEYS if key in payload.files),
            None,
        )

        embedded_fps = (
            _scalar(payload[fps_key])
            if fps_key is not None
            else None
        )

        scaling = (
            _scalar(payload["smpl_scaling"])
            if "smpl_scaling" in payload.files
            else None
        )

        trans_key = next(
            (key for key in TRANS_KEYS if key in payload.files),
            None,
        )
        if trans_key is None:
            raise ValueError(f"No SMPL translation array in {path.name}")
        translation = np.asarray(payload[trans_key])
        if translation.shape != (frames, 3):
            raise ValueError(
                f"Invalid translation shape in {path.name}: {translation.shape}"
            )

        if poses.ndim != 2 or poses.shape[1] != 165:
            raise ValueError(
                f"Chang-E Aligned SMPL must use [T,165] poses: "
                f"{path.name} has {poses.shape}"
            )
        pose55 = poses.reshape(frames, 55, 3)
        unobserved_max_abs = float(
            np.max(np.abs(pose55[:, 22:])) if frames else 0.0
        )
        if unobserved_max_abs > 1.0e-6:
            raise ValueError(
                f"Unobserved hand/face joints are non-zero in {path.name}: "
                f"max_abs={unobserved_max_abs:.6g}"
            )

    return {
        "source": str(path),
        "file": path.name,
        "pose_key": pose_key,
        "pose_shape": [int(value) for value in poses.shape],
        "pose_dim": int(poses.shape[1]),
        "source_joint_count": 55,
        "observed_body_joint_count": 22,
        "unobserved_joint_max_abs": unobserved_max_abs,
        "translation_key": trans_key,
        "translation_shape": [int(value) for value in translation.shape],
        "frames": frames,
        "source_fps": float(source_fps),
        "duration_seconds": float((frames - 1) / source_fps),
        "embedded_fps": embedded_fps,
        "embedded_fps_key": fps_key,
        "smpl_scaling": scaling,
        "sha256": file_sha256(path),
    }


def load_manifest(
    path: str | Path | None = None,
    *,
    required: bool = True,
) -> Optional[Dict[str, Any]]:
    target = manifest_path(path)

    if not target.is_file():
        if required:
            raise FileNotFoundError(
                f"Chang-E official SMPL manifest missing: {target}"
            )
        return None

    payload = json.loads(
        target.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Official SMPL manifest must be a JSON object"
        )

    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"Unsupported SMPL manifest schema "
            f"{payload.get('schema')!r}; "
            f"expected {MANIFEST_SCHEMA!r}"
        )

    release_id = str(payload.get("dataset_release_id", ""))
    if release_id not in {OFFICIAL_RELEASE_ID, TEST_RELEASE_ID}:
        raise ValueError(
            f"Unsupported official SMPL dataset_release_id={release_id!r}"
        )

    required_contract = {
        "source_format": SOURCE_FORMAT,
        "coordinate_system": COORDINATE_SYSTEM,
        "translation_units": TRANSLATION_UNITS,
        "pose_layout": POSE_LAYOUT,
        "canonical_skeleton": CANONICAL_SKELETON,
        "hand_rotation_policy": HAND_ROTATION_POLICY,
    }
    for field, expected in required_contract.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"Official SMPL manifest {field}={payload.get(field)!r}; "
                f"expected {expected!r}"
            )

    rows = payload.get("sources")

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Official SMPL manifest requires non-empty sources"
        )

    if int(payload.get("num_sources", -1)) != len(rows):
        raise ValueError(
            "Official SMPL manifest num_sources mismatch"
        )

    source_ids: set[str] = set()
    files: set[str] = set()
    recording_uids: set[str] = set()

    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"SMPL manifest row {index} is not an object"
            )

        source_id = str(
            raw.get("source_id", "")
        ).strip()

        filename = Path(
            str(raw.get("file", ""))
        ).name

        recording_uid = str(
            raw.get("recording_uid", "")
        ).strip()

        sequence_id = str(raw.get("sequence_id", "")).strip()

        if not source_id:
            raise ValueError(
                f"SMPL manifest row {index} missing source_id"
            )

        if not filename.lower().endswith(".npz"):
            raise ValueError(
                f"SMPL manifest row {source_id} "
                f"does not reference NPZ: {filename}"
            )

        if not recording_uid:
            raise ValueError(
                f"SMPL manifest row {source_id} "
                "missing recording_uid"
            )

        if not sequence_id:
            raise ValueError(
                f"SMPL manifest row {source_id} missing sequence_id"
            )

        if raw.get("dancer_id_status") not in {"verified", "unverified"}:
            raise ValueError(
                f"SMPL manifest row {source_id} has invalid dancer_id_status"
            )
        if raw.get("dancer_id_status") == "verified" and not raw.get("dancer_id"):
            raise ValueError(
                f"SMPL manifest row {source_id} marks an empty dancer_id verified"
            )

        theme_status = str(raw.get("theme_label_status", ""))
        if theme_status not in {"confirmed", "pending_official_confirmation"}:
            raise ValueError(
                f"SMPL manifest row {source_id} has invalid theme_label_status"
            )
        if theme_status != "confirmed" and raw.get("dance_category") != "unknown":
            raise ValueError(
                f"Unconfirmed theme for {source_id} must use dance_category=unknown"
            )

        row_contract = {
            "coordinate_system": COORDINATE_SYSTEM,
            "translation_units": TRANSLATION_UNITS,
            "pose_layout": POSE_LAYOUT,
        }
        for field, expected in row_contract.items():
            if raw.get(field, payload.get(field)) != expected:
                raise ValueError(
                    f"SMPL manifest row {source_id} has unsupported "
                    f"{field}={raw.get(field)!r}"
                )

        if source_id in source_ids:
            raise ValueError(
                f"Duplicate SMPL source_id: {source_id}"
            )

        if filename.lower() in files:
            raise ValueError(
                f"Duplicate SMPL filename: {filename}"
            )

        sha256 = str(
            raw.get("sha256", "")
        ).lower()

        if len(sha256) != 64:
            raise ValueError(
                f"Invalid SHA256 for {source_id}"
            )

        frames = int(raw.get("frames", -1))

        if frames < 2:
            raise ValueError(
                f"Invalid frame count for {source_id}: {frames}"
            )

        source_fps = float(
            raw.get("source_fps", 0.0)
        )

        if (
            not math.isfinite(source_fps)
            or source_fps <= 0.0
        ):
            raise ValueError(
                f"Invalid source_fps for {source_id}: "
                f"{source_fps}"
            )

        expected_duration = (
            float(frames - 1) / source_fps
        )

        duration = float(
            raw.get(
                "duration_seconds",
                expected_duration,
            )
        )

        if not math.isclose(
            duration,
            expected_duration,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                f"Duration/frame/FPS contract mismatch for "
                f"{source_id}: "
                f"duration={duration}, "
                f"expected={expected_duration}"
            )

        source_ids.add(source_id)
        files.add(filename.lower())
        recording_uids.add(recording_uid)

    declared_groups = int(
        payload.get(
            "num_recording_groups",
            len(recording_uids),
        )
    )

    if declared_groups != len(recording_uids):
        raise ValueError(
            "SMPL manifest num_recording_groups mismatch: "
            f"{declared_groups}!={len(recording_uids)}"
        )

    if release_id == OFFICIAL_RELEASE_ID:
        if source_ids != OFFICIAL_SOURCE_IDS:
            raise ValueError(
                "Official Chang-E Aligned SMPL release must contain exactly "
                f"14 sources: missing={sorted(OFFICIAL_SOURCE_IDS - source_ids)}, "
                f"extra={sorted(source_ids - OFFICIAL_SOURCE_IDS)}"
            )
        if declared_groups != 11:
            raise ValueError(
                f"Official Chang-E Aligned SMPL release requires 11 recording groups; "
                f"got {declared_groups}"
            )

    return payload


def manifest_rows(
    manifest: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    rows = manifest.get("sources", [])

    if not isinstance(rows, list):
        raise ValueError(
            "Official SMPL manifest has no sources list"
        )

    return [dict(row) for row in rows]


def match_manifest_entry(
    source: str | Path,
    manifest: Mapping[str, Any],
    *,
    explicit_source_id: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(source)
    stem = path.stem.lower()
    filename = path.name.lower()

    rows = manifest_rows(manifest)

    if explicit_source_id:
        matches = [
            row
            for row in rows
            if str(
                row.get("source_id", "")
            ).lower()
            == explicit_source_id.lower()
        ]
    else:
        matches = [
            row
            for row in rows
            if (
                str(
                    row.get("source_id", "")
                ).lower()
                == stem
                or Path(
                    str(row.get("file", ""))
                ).name.lower()
                == filename
            )
        ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Official SMPL file {path.name!r} "
            f"resolved to {len(matches)} manifest rows"
        )

    return dict(matches[0])


def validate_source(
    source: str | Path,
    *,
    manifest: Mapping[str, Any],
    manifest_file: str | Path | None = None,
    explicit_source_id: Optional[str] = None,
    verify_hash: bool = True,
) -> Dict[str, Any]:
    path = Path(source).expanduser().resolve()

    row = match_manifest_entry(
        path,
        manifest,
        explicit_source_id=explicit_source_id,
    )

    source_fps = float(row["source_fps"])

    actual = inspect_smpl_source(
        path,
        source_fps=source_fps,
    )

    errors: list[str] = []

    if int(row["frames"]) != actual["frames"]:
        errors.append(
            f"frames:{actual['frames']}!={row['frames']}"
        )

    if verify_hash:
        expected_hash = str(
            row["sha256"]
        ).lower()

        if actual["sha256"] != expected_hash:
            errors.append(
                "sha256:"
                f"{actual['sha256']}!="
                f"{expected_hash}"
            )

    if errors:
        raise ValueError(
            f"Official SMPL provenance mismatch "
            f"for {path.name}: "
            + "; ".join(errors)
        )

    embedded_fps = actual["embedded_fps"]

    result = {
        **actual,
        "source_id": str(row["source_id"]),
        "recording_uid": str(row["recording_uid"]),
        "sequence_id": str(row["sequence_id"]),
        "dancer_id": row.get("dancer_id"),
        "dancer_id_status": row.get("dancer_id_status", "unverified"),
        "performer_track_id": row.get(
            "performer_track_id",
            -1,
        ),
        "sequence_index": row.get(
            "sequence_index",
            -1,
        ),
        "performer_group": row.get(
            "performer_group",
            "unknown",
        ),
        "dance_category": row.get(
            "dance_category",
            "unknown",
        ),
        "candidate_dance_category": row.get("candidate_dance_category"),
        "theme_label_status": row.get("theme_label_status"),
        "source_context": row.get("source_context", []),
        "coordinate_system": row.get(
            "coordinate_system", manifest["coordinate_system"]
        ),
        "translation_units": row.get(
            "translation_units", manifest["translation_units"]
        ),
        "pose_layout": row.get("pose_layout", manifest["pose_layout"]),
        "canonical_skeleton": manifest["canonical_skeleton"],
        "hand_rotation_policy": manifest["hand_rotation_policy"],
        "take_id": row.get("take_id"),
        "skeleton_id": (
            "chang_e_official_smpl"
        ),
        "embedded_fps_matches_manifest": (
            None
            if embedded_fps is None
            else bool(
                math.isclose(
                    float(embedded_fps),
                    source_fps,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
            )
        ),
    }

    if manifest_file is not None:
        result["manifest"] = str(
            manifest_path(manifest_file)
        )
        result["manifest_sha256"] = (
            manifest_sha256(manifest_file)
        )

    return result


def semantic_metadata(
    source: str | Path,
    *,
    path: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    manifest = load_manifest(
        path,
        required=False,
    )

    if manifest is None:
        return None

    try:
        row = match_manifest_entry(
            source,
            manifest,
        )
    except RuntimeError:
        return None

    return {
        "source_id": row.get("source_id"),
        "recording_uid": row.get(
            "recording_uid"
        ),
        "sequence_id": row.get("sequence_id"),
        "dancer_id": row.get("dancer_id"),
        "dancer_id_status": row.get("dancer_id_status"),
        "performer_track_id": row.get(
            "performer_track_id"
        ),
        "sequence_index": row.get(
            "sequence_index"
        ),
        "performer_group": row.get(
            "performer_group"
        ),
        "dance_category": row.get(
            "dance_category"
        ),
        "candidate_dance_category": row.get("candidate_dance_category"),
        "theme_label_status": row.get("theme_label_status"),
        "source_context": row.get("source_context", []),
        "coordinate_system": row.get(
            "coordinate_system", manifest.get("coordinate_system")
        ),
        "translation_units": row.get(
            "translation_units", manifest.get("translation_units")
        ),
        "pose_layout": row.get("pose_layout", manifest.get("pose_layout")),
        "take_id": row.get("take_id"),
        "skeleton_id": (
            "chang_e_official_smpl"
        ),
        "source_fps": row.get("source_fps"),
    }
