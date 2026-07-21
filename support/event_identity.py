"""Stable identity and alignment contracts for motion events.

Event indices are deliberately treated as storage positions, not identities.
The stable identity below is independent of absolute project roots and remains
unchanged when an event database is copied between machines.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

EVENT_UID_SCHEMA = "dunhuang_event_uid_v2_time_interval"
EVENT_DB_CONTRACT_SCHEMA = "dunhuang_event_db_contract_v2"


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else "").strip().replace("\\", "/")
    return text or default


def portable_source_name(value: Any) -> str:
    """Return a root-independent source name for legacy absolute paths."""
    text = _text(value)
    name = PurePosixPath(text).name
    return name or text


def stable_event_uid(
    *,
    source_uid: Any,
    start: Any,
    end: Any,
    frames: Any | None = None,
    source_file: Any = "",
    source_fps: Any = 30.0,
    start_seconds: Any | None = None,
    end_seconds: Any | None = None,
) -> str:
    """Build an identity from source and physical time, never target frames."""
    source = _text(source_uid, default="")
    if not source or source == "unknown":
        source = portable_source_name(source_file)
    fps = float(source_fps)
    if fps <= 0.0:
        raise ValueError("source_fps must be positive")
    start_s = float(start_seconds) if start_seconds is not None else float(start) / fps
    end_s = float(end_seconds) if end_seconds is not None else float(end) / fps
    if end_s < start_s:
        raise ValueError(f"event end precedes start: {start_s} > {end_s}")
    payload = {
        "schema": EVENT_UID_SCHEMA,
        "source_uid": source,
        # Integer microseconds make 30/60 FPS representations of the same
        # source interval hash identically despite floating-point noise.
        "start_us": int(round(start_s * 1_000_000.0)),
        "end_us": int(round(end_s * 1_000_000.0)),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return "evt_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def event_uids_from_generation_db(db: Mapping[str, Any]) -> np.ndarray:
    count = int(len(db.get("paths", [])))
    if "event_uids" in db:
        uids = np.asarray(db["event_uids"], dtype=object)
        if len(uids) != count:
            raise RuntimeError(f"event_uids has {len(uids)} rows, expected {count}")
        return uids

    source_uids = np.asarray(db.get("source_uids", ["unknown"] * count), dtype=object)
    source_files = np.asarray(db.get("source_files", [""] * count), dtype=object)
    starts = np.asarray(db.get("starts", np.zeros(count, dtype=np.int64)))
    ends = np.asarray(db.get("ends", np.zeros(count, dtype=np.int64)))
    frames = np.asarray(db.get("frames", ends - starts))
    start_seconds = db.get("source_start_seconds", db.get("start_seconds"))
    end_seconds = db.get("source_end_seconds", db.get("end_seconds"))
    start_seconds = None if start_seconds is None else np.asarray(start_seconds, dtype=np.float64)
    end_seconds = None if end_seconds is None else np.asarray(end_seconds, dtype=np.float64)
    if start_seconds is not None and start_seconds.ndim == 0:
        start_seconds = np.full(count, float(start_seconds), dtype=np.float64)
    if end_seconds is not None and end_seconds.ndim == 0:
        end_seconds = np.full(count, float(end_seconds), dtype=np.float64)
    if start_seconds is not None and len(start_seconds) != count:
        raise RuntimeError(f"source_start_seconds has {len(start_seconds)} rows, expected {count}")
    if end_seconds is not None and len(end_seconds) != count:
        raise RuntimeError(f"source_end_seconds has {len(end_seconds)} rows, expected {count}")
    raw_fps = db.get("canonical_fps", db.get("fps", 30.0))
    fps_values = np.asarray(raw_fps, dtype=np.float64)
    if fps_values.ndim == 0:
        fps_values = np.full(count, float(fps_values), dtype=np.float64)
    elif len(fps_values) != count:
        raise RuntimeError(f"canonical_fps has {len(fps_values)} rows, expected {count}")
    return np.asarray(
        [
            stable_event_uid(
                source_uid=source_uids[i],
                source_file=source_files[i],
                start=starts[i],
                end=ends[i],
                frames=frames[i],
                source_fps=fps_values[i],
                start_seconds=None if start_seconds is None else start_seconds[i],
                end_seconds=None if end_seconds is None else end_seconds[i],
            )
            for i in range(count)
        ],
        dtype=object,
    )


def event_uid_from_item(item: Mapping[str, Any], position: int = -1) -> str:
    explicit = _text(item.get("event_uid", ""), default="")
    if explicit:
        return explicit
    start = item.get("source_start", item.get("start", 0))
    end = item.get("source_end", item.get("end", start))
    frames = item.get("length", item.get("frames", max(0, int(end) - int(start))))
    return stable_event_uid(
        source_uid=item.get("source_uid", item.get("source_id", "unknown")),
        source_file=item.get("source_file", item.get("pkl", item.get("path", ""))),
        start=start,
        end=end,
        frames=frames,
        source_fps=item.get("canonical_fps", item.get("target_fps", item.get("fps", 30.0))),
        start_seconds=item.get("source_start_seconds", item.get("start_seconds")),
        end_seconds=item.get("source_end_seconds", item.get("end_seconds")),
    )


def ordered_event_fingerprint(event_uids: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for uid in event_uids:
        digest.update(_text(uid).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def make_event_db_contract(event_uids: Sequence[Any]) -> dict[str, Any]:
    normalized = [_text(uid) for uid in event_uids]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("Stable event_uid values must be unique within an event database")
    return {
        "schema": EVENT_DB_CONTRACT_SCHEMA,
        "num_events": len(normalized),
        "ordered_event_uid_sha256": ordered_event_fingerprint(normalized),
    }


def normalize_event_db_contract(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bytes, str)):
        try:
            value = json.loads(value.decode("utf-8") if isinstance(value, bytes) else value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    return {
        "schema": str(value.get("schema", "")),
        "num_events": int(value.get("num_events", -1)),
        "ordered_event_uid_sha256": str(value.get("ordered_event_uid_sha256", "")),
    }


def assert_same_event_db_contract(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
    *,
    context: str,
) -> None:
    lhs = normalize_event_db_contract(expected)
    rhs = normalize_event_db_contract(actual)
    if lhs is None or rhs is None:
        raise RuntimeError(f"{context}: missing event DB identity contract")
    if lhs != rhs:
        raise RuntimeError(f"{context}: event DB contract mismatch: expected={lhs}, actual={rhs}")
