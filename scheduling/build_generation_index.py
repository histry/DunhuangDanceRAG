#!/usr/bin/env python3
"""Build Scheduler assets from the exact Generation Event-DB.

This replaces the historical snapshot index.  Event order, stable event_uid,
descriptors, endpoint arrays, and the database fingerprint all originate from
one generation database, so Scheduler choices can be resolved without an
index-position join against a different corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    angular_velocity_np,
    rot6d_to_matrix_np,
)
from motion_geometry.smpl24 import (
    ROT6D_END,
    ROT6D_START,
    skeleton_contract,
    skeleton_fingerprint,
)
from support.common import (
    load_motion,
    motion_descriptor_raw,
    motion_mmr_embedding,
    robust_scale,
)
from support.event_identity import (
    event_uids_from_generation_db,
    make_event_db_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_TYPE_MAP = {
    "calm_meditative": "calm_flow",
    "pose_hold": "pose_hold",
    "lyrical_flow": "neutral_flow",
    "instrument_phrase": "arm_flourish",
    "percussive_accent": "high_tension",
    "turning_climax": "high_tension",
    "footwork_flow": "support_shift",
    "aerial_curve": "arm_flourish",
    "build_up": "build_up",
    "release": "release",
}


def _db_dict(path: Path) -> dict[str, Any]:
    source = np.load(path, allow_pickle=True)
    return {key: source[key] for key in source.files}


def _array(db: Mapping[str, Any], key: str, count: int, default: Any) -> np.ndarray:
    if key in db:
        value = np.asarray(db[key])
        if len(value) != count:
            raise RuntimeError(f"Generation DB array {key!r} has {len(value)}, expected {count}")
        return value
    return np.asarray([default] * count)


def resolve_generation_motion(raw_value: Any, db_path: Path) -> Path:
    """Resolve legacy Linux paths after a project is moved to another host."""
    text = str(raw_value).strip().replace("\\", "/")
    raw = Path(text)
    candidates = [raw]
    marker = "/DunhuangDanceRAG/"
    if marker in text:
        candidates.append(PROJECT_ROOT / text.split(marker, 1)[1])
    if not raw.is_absolute():
        candidates.extend((PROJECT_ROOT / raw, db_path.parent / raw))
    candidates.append(db_path.parent / "events" / PurePosixPath(text).name)
    checked: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Cannot resolve generation event {text!r}; checked={checked}")


def _portable_reference(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _semantic_event_type(value: Any) -> str:
    text = str(value or "neutral_flow")
    return EVENT_TYPE_MAP.get(text, text if text in set(EVENT_TYPE_MAP.values()) else "neutral_flow")


def build_generation_index(
    db_path: Path,
    out_json: Path,
    out_npz: Path,
) -> dict[str, Any]:
    db_path = db_path.resolve()
    db = _db_dict(db_path)
    if "paths" not in db:
        raise RuntimeError(f"Generation DB has no paths array: {db_path}")
    count = len(db["paths"])
    raw_skeleton = db.get("skeleton_contract_json")
    if raw_skeleton is None:
        raise RuntimeError(
            "Generation DB has no SMPL24 skeleton contract; rebuild it with events.build_database."
        )
    if isinstance(raw_skeleton, np.ndarray) and raw_skeleton.ndim == 0:
        raw_skeleton = raw_skeleton.item()
    if isinstance(raw_skeleton, bytes):
        raw_skeleton = raw_skeleton.decode("utf-8")
    db_skeleton = json.loads(raw_skeleton) if isinstance(raw_skeleton, str) else dict(raw_skeleton)
    if db_skeleton.get("sha256") != skeleton_fingerprint():
        raise RuntimeError(
            "Generation DB SMPL24 skeleton mismatch: "
            f"db={db_skeleton.get('sha256')!r}, runtime={skeleton_fingerprint()!r}"
        )
    event_uids = event_uids_from_generation_db(db)
    contract = make_event_db_contract(event_uids)

    source_uids = _array(db, "source_uids", count, "unknown")
    source_files = _array(db, "source_files", count, "")
    starts = _array(db, "starts", count, 0).astype(np.int64)
    ends = _array(db, "ends", count, 0).astype(np.int64)
    frames = _array(db, "frames", count, 0).astype(np.int64)
    families = _array(db, "event_families", count, "unknown")
    dance_keys = _array(db, "dance_keys", count, "unknown")
    performers = _array(db, "performer_groups", count, "unknown")
    semantics = _array(db, "aesd_event_semantics", count, "neutral_flow")
    quality = _array(db, "v46_53_combined_quality", count, 0.5).astype(np.float32)
    anatomy_quality = _array(db, "anatomy_quality", count, 0.5).astype(np.float32)
    hard_valid = _array(db, "anatomy_hard_valid", count, True).astype(bool)
    heading_quality = _array(db, "event_heading_quality", count, 0.5).astype(np.float32)
    yaw = _array(db, "event_net_yaw_rad", count, 0.0).astype(np.float32)
    durations_sec = _array(db, "durations", count, 0.0).astype(np.float32)
    if "canonical_fps" in db:
        canonical_fps = np.asarray(db["canonical_fps"], dtype=np.float32)
        if canonical_fps.ndim == 0:
            canonical_fps = np.full(count, float(canonical_fps), dtype=np.float32)
        if len(canonical_fps) != count:
            raise RuntimeError(
                f"Generation DB canonical_fps has {len(canonical_fps)}, expected {count}"
            )
    else:
        canonical_fps = np.full(count, 30.0, dtype=np.float32)
    fps_values = sorted({float(value) for value in canonical_fps})
    if len(fps_values) != 1:
        raise RuntimeError(
            f"One Scheduler index cannot mix canonical frame rates: {fps_values}"
        )
    source_start_seconds = np.asarray(
        db.get("source_start_seconds", starts / canonical_fps), dtype=np.float64
    )
    source_end_seconds = np.asarray(
        db.get("source_end_seconds", ends / canonical_fps), dtype=np.float64
    )

    raw_descriptors: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    entry_pose: list[np.ndarray] = []
    exit_pose: list[np.ndarray] = []
    entry_vel: list[np.ndarray] = []
    exit_vel: list[np.ndarray] = []
    entry_angular_velocity: list[np.ndarray] = []
    exit_angular_velocity: list[np.ndarray] = []
    resolved_motions: list[Path] = []
    items: list[dict[str, Any]] = []

    for index in range(count):
        path = resolve_generation_motion(db["paths"][index], db_path)
        motion = load_motion(path)
        if frames[index] <= 0:
            frames[index] = len(motion)
        if int(frames[index]) != len(motion):
            raise RuntimeError(
                f"Generation event {index} frame mismatch: DB={frames[index]}, file={len(motion)}"
            )
        fps = float(canonical_fps[index])
        raw_descriptors.append(motion_descriptor_raw(motion, fps=fps))
        embeddings.append(motion_mmr_embedding(motion, out_dim=64, fps=fps))
        entry_pose.append(motion[0].astype(np.float32))
        exit_pose.append(motion[-1].astype(np.float32))
        first = np.diff(motion[: min(5, len(motion))], axis=0)
        last = np.diff(motion[max(0, len(motion) - 5) :], axis=0)
        entry_vel.append(((first.mean(axis=0) * fps) if len(first) else np.zeros(151)).astype(np.float32))
        exit_vel.append(((last.mean(axis=0) * fps) if len(last) else np.zeros(151)).astype(np.float32))
        rotation_matrices = rot6d_to_matrix_np(
            motion[:, ROT6D_START:ROT6D_END].reshape(len(motion), 24, 6)
        )
        omega = angular_velocity_np(rotation_matrices, fps=fps)
        entry_angular_velocity.append(
            (omega[: min(4, len(omega))].mean(axis=0) if len(omega) else np.zeros((24, 3))).astype(np.float32)
        )
        exit_angular_velocity.append(
            (omega[max(0, len(omega) - 4):].mean(axis=0) if len(omega) else np.zeros((24, 3))).astype(np.float32)
        )
        resolved_motions.append(path)
        uid = str(event_uids[index])
        items.append(
            {
                "event_uid": uid,
                "event_id": uid,
                "generation_event_index": index,
                "pkl": _portable_reference(path),
                "path": _portable_reference(path),
                "source_uid": str(source_uids[index]),
                "source_file": str(source_files[index]),
                "source_start": int(starts[index]),
                "source_end": int(ends[index]),
                "source_start_seconds": float(source_start_seconds[index]),
                "source_end_seconds": float(source_end_seconds[index]),
                "canonical_fps": fps,
                "length": int(frames[index]),
                "family_id": str(families[index]),
                "dance_key": str(dance_keys[index]),
                "performer_group": str(performers[index]),
                "event_type": _semantic_event_type(semantics[index]),
            }
        )

    raw = np.stack(raw_descriptors).astype(np.float32)
    desc, desc_lo, desc_hi = robust_scale(raw)
    duration_seconds = frames.astype(np.float32) / canonical_fps
    desc[:, 11] = np.clip((duration_seconds - 0.8) / 5.2, 0.0, 1.0)
    combined_quality = np.clip(0.55 * quality + 0.45 * anatomy_quality, 0.0, 1.0)
    safety = np.where(hard_valid, np.clip(0.6 * anatomy_quality + 0.4 * heading_quality, 0.0, 1.0), 0.0)
    turn_angle_deg = np.abs(yaw) * (180.0 / np.pi)
    safe_duration = np.maximum(durations_sec, duration_seconds)
    turn_peak_dps = turn_angle_deg / np.maximum(safe_duration, 1e-3)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "generation_aligned_scheduler_index_v2_multirate",
        "rot6d_layout": CANONICAL_ROT6D_LAYOUT,
        "skeleton_contract": skeleton_contract(),
        "canonical_fps_values": fps_values,
        "natural_duration_units": "frames_at_canonical_fps",
        "velocity_units": {"entry_vel": "channel_units/s", "angular_velocity": "rad/s"},
        "generation_db": str(db_path),
        "event_db_contract": contract,
        "items": items,
    }
    out_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_npz,
        event_uids=event_uids,
        motion_desc=desc,
        motion_desc_raw=raw,
        desc_lo=desc_lo,
        desc_hi=desc_hi,
        mmr_embed=np.stack(embeddings).astype(np.float32),
        entry_pose=np.stack(entry_pose).astype(np.float32),
        exit_pose=np.stack(exit_pose).astype(np.float32),
        entry_vel=np.stack(entry_vel).astype(np.float32),
        exit_vel=np.stack(exit_vel).astype(np.float32),
        entry_angular_velocity_radps=np.stack(entry_angular_velocity).astype(np.float32),
        exit_angular_velocity_radps=np.stack(exit_angular_velocity).astype(np.float32),
        canonical_fps=canonical_fps.astype(np.float32),
        source_start_seconds=source_start_seconds.astype(np.float64),
        source_end_seconds=source_end_seconds.astype(np.float64),
        length=frames.astype(np.int32),
        style_score=combined_quality.astype(np.float32),
        quality_score=combined_quality.astype(np.float32),
        safety_score=safety.astype(np.float32),
        natural_duration=frames.astype(np.float32),
        duration_confidence=np.ones(count, dtype=np.float32),
        v23_duration_used=np.zeros(count, dtype=bool),
        turn_peak_dps=turn_peak_dps.astype(np.float32),
        turn_angle_deg=turn_angle_deg.astype(np.float32),
        event_db_contract_json=np.asarray(json.dumps(contract, sort_keys=True), dtype=object),
    )
    return {
        "ok": True,
        "generation_db": str(db_path),
        "index_json": str(out_json),
        "index_npz": str(out_npz),
        "num_events": count,
        "event_db_contract": contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    report = build_generation_index(Path(args.db), Path(args.out_json), Path(args.out_npz))
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
