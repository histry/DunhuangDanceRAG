#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.51 Fresh-WAV Audio–Schedule Transaction Contract.

Formal generation must satisfy all of the following:

1. the schedule is rebuilt from the current WAV in a unique run directory;
2. the descriptor contains the current WAV SHA-256 and run identifier;
3. the slot timeline is contiguous, ordered and frame-conserving;
4. total target frames equal round(audio_duration * fps);
5. the raw V26 report names the same current audio;
6. no old MSSD may be silently reused.

This module is importable by both the fresh MSSD builder and the final
V46.51 generation entrypoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "v46_51_fresh_wav_audio_schedule_contract"


def jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(
    path: str | Path,
    *,
    require_exists: bool = True,
    hash_content: bool = True,
) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        if require_exists:
            raise FileNotFoundError(str(p))
        return {
            "path": str(p),
            "exists": False,
        }
    stat = p.stat()
    out = {
        "path": str(p.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        out["sha256"] = sha256_file(p)
    return out


def _audio_info_wave(path: Path) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        sample_rate = int(wf.getframerate())
        sample_frames = int(wf.getnframes())
    return {
        "decoder": "python_wave",
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate": sample_rate,
        "sample_frames": sample_frames,
        "duration_seconds": float(sample_frames / max(sample_rate, 1)),
    }


def _audio_info_soundfile(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        return None
    info = sf.info(str(path))
    return {
        "decoder": "soundfile",
        "channels": int(info.channels),
        "sample_width_bytes": None,
        "sample_rate": int(info.samplerate),
        "sample_frames": int(info.frames),
        "duration_seconds": float(info.duration),
    }


def _audio_info_ffprobe(path: Path) -> Optional[Dict[str, Any]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
        )
        obj = json.loads(result.stdout)
        streams = obj.get("streams", [])
        audio_stream = streams[0] if streams else {}
        duration = float(obj.get("format", {}).get("duration", 0.0))
        sample_rate = int(float(audio_stream.get("sample_rate", 0) or 0))
        channels = int(audio_stream.get("channels", 0) or 0)
        return {
            "decoder": "ffprobe",
            "channels": channels,
            "sample_width_bytes": None,
            "sample_rate": sample_rate,
            "sample_frames": (
                int(round(duration * sample_rate))
                if sample_rate > 0
                else None
            ),
            "duration_seconds": duration,
        }
    except Exception:
        return None


def audio_info(path: str | Path, fps: float = 30.0) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    decoded: Optional[Dict[str, Any]] = None
    if p.suffix.lower() == ".wav":
        try:
            decoded = _audio_info_wave(p)
        except Exception:
            decoded = None
    if decoded is None:
        decoded = _audio_info_soundfile(p)
    if decoded is None:
        decoded = _audio_info_ffprobe(p)
    if decoded is None:
        raise RuntimeError(
            f"Cannot decode audio metadata for {p}; install soundfile or ffprobe."
        )
    duration = float(decoded["duration_seconds"])
    target_frames = int(round(duration * float(fps)))
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "name": p.name,
        "stem": p.stem,
        "suffix": p.suffix.lower(),
        "sha256": sha256_file(p),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "fps": float(fps),
        "target_frames": int(target_frames),
        **decoded,
    }


def extract_slots(obj: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if isinstance(obj, dict):
        for key in ("slots", "segments"):
            value = obj.get(key)
            if isinstance(value, list) and value:
                return [dict(x) for x in value if isinstance(x, dict)], dict(obj)
        if isinstance(obj.get("schedule"), list) and obj["schedule"]:
            return [
                dict(x) for x in obj["schedule"] if isinstance(x, dict)
            ], dict(obj)
    if isinstance(obj, list):
        return [dict(x) for x in obj if isinstance(x, dict)], {}
    return [], {}


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            try:
                value = float(row[key])
                if math.isfinite(value):
                    return value
            except Exception:
                pass
    return None


def normalize_slot_time(
    slot: Mapping[str, Any],
    *,
    index: int,
    fps: float,
    cursor_frame: int,
) -> Dict[str, Any]:
    target_frames_f = _first_number(
        slot,
        (
            "target_frames",
            "allocated_phrase_total",
            "v26_allocated_phrase_total",
            "music_length",
            "frames",
            "num_frames",
        ),
    )
    start_frame_f = _first_number(
        slot,
        ("start_frame", "music_start"),
    )
    end_frame_f = _first_number(
        slot,
        ("end_frame", "music_end"),
    )
    start_sec = _first_number(
        slot,
        ("start_sec", "start", "t0", "audio_start"),
    )
    end_sec = _first_number(
        slot,
        ("end_sec", "end", "t1", "audio_end"),
    )
    duration_sec = _first_number(
        slot,
        ("duration_sec", "duration", "slot_duration"),
    )

    target_frames = (
        int(round(target_frames_f))
        if target_frames_f is not None
        else None
    )
    if start_frame_f is not None:
        start_frame = int(round(start_frame_f))
    elif start_sec is not None:
        start_frame = int(round(start_sec * fps))
    else:
        start_frame = int(cursor_frame)

    if end_frame_f is not None:
        end_frame = int(round(end_frame_f))
    elif end_sec is not None:
        end_frame = int(round(end_sec * fps))
    elif target_frames is not None:
        end_frame = start_frame + int(target_frames)
    elif duration_sec is not None:
        end_frame = start_frame + int(round(duration_sec * fps))
    else:
        raise RuntimeError(f"Slot {index} has no usable duration or endpoint.")

    if target_frames is None:
        target_frames = int(end_frame - start_frame)

    if duration_sec is None:
        duration_sec = float(target_frames / fps)

    return {
        "slot_index": int(index),
        "slot_id": slot.get("slot_id", slot.get("slot", index)),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "target_frames": int(target_frames),
        "start_seconds": float(start_frame / fps),
        "end_seconds": float(end_frame / fps),
        "duration_seconds": float(duration_sec),
        "raw": dict(slot),
    }


def _resolved_same_path(a: Any, b: Any) -> bool:
    try:
        return Path(str(a)).expanduser().resolve() == Path(str(b)).expanduser().resolve()
    except Exception:
        return str(a) == str(b)


def stamp_descriptor(
    descriptor: Dict[str, Any],
    *,
    audio: str | Path,
    fps: float,
    run_id: str,
    run_dir: str | Path,
    raw_schedule_json: str | Path,
    scheduler_command: Sequence[str],
    assets: Mapping[str, Any],
    hash_assets: bool = True,
) -> Dict[str, Any]:
    """Attach immutable V46.51 provenance to a newly rebuilt descriptor."""
    out = dict(descriptor)
    info = audio_info(audio, fps=fps)
    raw_path = Path(raw_schedule_json)
    if not raw_path.is_file():
        raise FileNotFoundError(str(raw_path))

    asset_fingerprints: Dict[str, Any] = {}
    for name, value in assets.items():
        if value is None or str(value).strip() == "":
            asset_fingerprints[str(name)] = {
                "path": "",
                "exists": False,
                "optional": True,
            }
        else:
            asset_fingerprints[str(name)] = file_fingerprint(
                value,
                require_exists=True,
                hash_content=hash_assets,
            )

    transaction = {
        "schema": SCHEMA,
        "schedule_build_mode": "fresh_from_current_wav",
        "schedule_reuse_allowed": False,
        "schedule_run_id": str(run_id),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(Path(run_dir).resolve()),
        "audio": info,
        "raw_schedule": file_fingerprint(
            raw_path,
            require_exists=True,
            hash_content=True,
        ),
        "scheduler_assets": asset_fingerprints,
        "scheduler_command": [str(x) for x in scheduler_command],
        "feature_cache_scope": "run_local_unique",
        "raw_schedule_scope": "run_local_unique",
    }

    provenance = dict(out.get("provenance", {}))
    provenance["v46_51"] = transaction
    out["provenance"] = provenance
    out["audio"] = str(Path(audio).resolve())
    out["audio_sha256"] = info["sha256"]
    out["audio_duration_seconds"] = info["duration_seconds"]
    out["audio_target_frames"] = info["target_frames"]
    out["schedule_run_id"] = str(run_id)
    out["schedule_build_mode"] = "fresh_from_current_wav"
    out["schedule_reuse_allowed"] = False
    out["raw_schedule_json"] = str(raw_path.resolve())
    return out


def audit_contract(
    *,
    audio: str | Path,
    schedule: str | Path | Dict[str, Any],
    fps: float = 30.0,
    required_run_id: Optional[str] = None,
    require_fresh: bool = True,
    max_frame_error: int = 2,
    max_seconds_error: float = 0.10,
    require_raw_report: bool = True,
) -> Dict[str, Any]:
    info = audio_info(audio, fps=fps)
    schedule_path: Optional[Path]
    if isinstance(schedule, dict):
        obj = dict(schedule)
        schedule_path = None
    else:
        schedule_path = Path(schedule)
        obj = load_json(schedule_path)

    slots, meta = extract_slots(obj)
    reasons: List[str] = []
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []

    if not slots:
        reasons.append("schedule_has_no_slots")

    usage = str(meta.get("usage", "")).lower()
    is_final = bool(meta.get("is_final_schedule", False))
    slot_source = str(meta.get("slot_source", ""))
    if usage != "generate_schedule":
        reasons.append(f"usage_not_generate_schedule:{usage}")
    if not is_final:
        reasons.append("is_final_schedule_false")
    if slot_source != "v21_router_v26_planner":
        reasons.append(f"unexpected_slot_source:{slot_source}")

    descriptor_audio = meta.get("audio")
    if descriptor_audio and not _resolved_same_path(descriptor_audio, info["path"]):
        reasons.append(
            f"descriptor_audio_path_mismatch:{descriptor_audio}!={info['path']}"
        )

    transaction = (
        meta.get("provenance", {}).get("v46_51", {})
        if isinstance(meta.get("provenance"), dict)
        else {}
    )
    if require_fresh:
        if not isinstance(transaction, dict) or not transaction:
            reasons.append("missing_v46_51_transaction_provenance")
        else:
            if transaction.get("schedule_build_mode") != "fresh_from_current_wav":
                reasons.append("schedule_build_mode_not_fresh_from_current_wav")
            if bool(transaction.get("schedule_reuse_allowed", True)):
                reasons.append("schedule_reuse_allowed")
            tx_audio = transaction.get("audio", {})
            if not isinstance(tx_audio, dict):
                reasons.append("transaction_audio_missing")
            else:
                if str(tx_audio.get("sha256", "")) != str(info["sha256"]):
                    reasons.append("audio_sha256_mismatch")
                if int(tx_audio.get("target_frames", -1)) != int(info["target_frames"]):
                    reasons.append("transaction_audio_target_frames_mismatch")
                if tx_audio.get("path") and not _resolved_same_path(
                    tx_audio.get("path"),
                    info["path"],
                ):
                    reasons.append("transaction_audio_path_mismatch")
            if required_run_id is not None:
                actual_run_id = str(transaction.get("schedule_run_id", ""))
                if actual_run_id != str(required_run_id):
                    reasons.append(
                        f"schedule_run_id_mismatch:{actual_run_id}!={required_run_id}"
                    )

    cursor = 0
    total_target_frames = 0
    overlap_count = 0
    gap_count = 0
    duration_field_mismatch_count = 0
    nonpositive_count = 0

    for i, slot in enumerate(slots):
        try:
            row = normalize_slot_time(
                slot,
                index=i,
                fps=fps,
                cursor_frame=cursor,
            )
        except Exception as exc:
            reasons.append(f"slot_{i}_time_parse_error:{exc}")
            continue

        start = int(row["start_frame"])
        end = int(row["end_frame"])
        target = int(row["target_frames"])
        frame_extent = end - start
        duration_expected = target / fps
        duration_error = abs(float(row["duration_seconds"]) - duration_expected)

        if target <= 0 or frame_extent <= 0:
            nonpositive_count += 1
        if start < cursor:
            overlap_count += 1
        elif start > cursor:
            gap_count += 1
        if frame_extent != target:
            reasons.append(
                f"slot_{i}_frame_extent_mismatch:{frame_extent}!={target}"
            )
        if duration_error > max(1.0 / fps, max_seconds_error):
            duration_field_mismatch_count += 1

        row["frame_extent"] = frame_extent
        row["duration_expected_seconds"] = duration_expected
        row["duration_error_seconds"] = duration_error
        rows.append(row)
        cursor = end
        total_target_frames += target

    expected_frames = int(info["target_frames"])
    frame_error = int(total_target_frames - expected_frames)
    timeline_end_frame = int(cursor)
    timeline_frame_error = int(timeline_end_frame - expected_frames)
    schedule_seconds = float(total_target_frames / fps)
    seconds_error = float(schedule_seconds - info["duration_seconds"])

    if rows and int(rows[0]["start_frame"]) != 0:
        reasons.append(
            f"timeline_does_not_start_at_zero:{rows[0]['start_frame']}"
        )
    if overlap_count:
        reasons.append(f"overlapping_slots:{overlap_count}")
    if gap_count:
        reasons.append(f"timeline_gaps:{gap_count}")
    if nonpositive_count:
        reasons.append(f"nonpositive_slots:{nonpositive_count}")
    if duration_field_mismatch_count:
        reasons.append(
            f"duration_field_mismatch:{duration_field_mismatch_count}"
        )
    if abs(frame_error) > int(max_frame_error):
        reasons.append(
            f"total_target_frame_error:{frame_error}>limit={max_frame_error}"
        )
    if abs(timeline_frame_error) > int(max_frame_error):
        reasons.append(
            f"timeline_end_frame_error:{timeline_frame_error}>limit={max_frame_error}"
        )
    if abs(seconds_error) > float(max_seconds_error):
        reasons.append(
            f"audio_schedule_seconds_error:{seconds_error:.9f}>limit={max_seconds_error}"
        )

    meta_total = meta.get("total_target_frames")
    if meta_total is not None:
        try:
            if int(meta_total) != total_target_frames:
                reasons.append(
                    f"descriptor_total_target_frames_mismatch:{meta_total}!={total_target_frames}"
                )
        except Exception:
            reasons.append("descriptor_total_target_frames_not_integer")

    raw_report_path = meta.get("raw_schedule_json")
    raw_report_audit: Dict[str, Any] = {}
    if require_raw_report:
        if not raw_report_path:
            reasons.append("missing_raw_schedule_json")
        else:
            rp = Path(str(raw_report_path))
            if not rp.is_file():
                reasons.append(f"raw_schedule_report_missing:{rp}")
            else:
                try:
                    raw = load_json(rp)
                    raw_audio = raw.get("audio") if isinstance(raw, dict) else None
                    raw_schedule = raw.get("schedule", []) if isinstance(raw, dict) else []
                    out_npy = raw.get("out_npy", "") if isinstance(raw, dict) else ""
                    raw_report_audit = {
                        "path": str(rp.resolve()),
                        "sha256": sha256_file(rp),
                        "audio": raw_audio,
                        "schedule_rows": (
                            len(raw_schedule)
                            if isinstance(raw_schedule, list)
                            else 0
                        ),
                        "out_npy": out_npy,
                    }
                    if raw_audio and not _resolved_same_path(raw_audio, info["path"]):
                        reasons.append("raw_report_audio_path_mismatch")
                    if not isinstance(raw_schedule, list) or not raw_schedule:
                        reasons.append("raw_report_schedule_empty")
                    if out_npy:
                        npy_path = Path(str(out_npy))
                        if not npy_path.is_file():
                            warnings.append(f"raw_v26_motion_missing:{npy_path}")
                        else:
                            try:
                                motion = np.load(npy_path, mmap_mode="r")
                                frames = (
                                    int(motion.shape[-2])
                                    if motion.ndim >= 2
                                    else -1
                                )
                                raw_report_audit["out_npy_frames"] = frames
                                if abs(frames - expected_frames) > max_frame_error:
                                    reasons.append(
                                        f"raw_v26_motion_frame_mismatch:{frames}!={expected_frames}"
                                    )
                            except Exception as exc:
                                reasons.append(
                                    f"raw_v26_motion_read_error:{exc}"
                                )
                except Exception as exc:
                    reasons.append(f"raw_schedule_report_read_error:{exc}")

    report = {
        "schema": SCHEMA,
        "ok": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "audio": info,
        "schedule_path": str(schedule_path.resolve()) if schedule_path else None,
        "schedule_sha256": (
            sha256_file(schedule_path)
            if schedule_path and schedule_path.is_file()
            else None
        ),
        "usage": usage,
        "is_final_schedule": is_final,
        "slot_source": slot_source,
        "required_run_id": required_run_id,
        "transaction": transaction,
        "num_slots": int(len(rows)),
        "total_target_frames": int(total_target_frames),
        "expected_audio_target_frames": expected_frames,
        "frame_error": frame_error,
        "timeline_end_frame": timeline_end_frame,
        "timeline_frame_error": timeline_frame_error,
        "schedule_duration_seconds": schedule_seconds,
        "audio_duration_seconds": float(info["duration_seconds"]),
        "seconds_error": seconds_error,
        "overlap_count": int(overlap_count),
        "gap_count": int(gap_count),
        "duration_field_mismatch_count": int(
            duration_field_mismatch_count
        ),
        "nonpositive_slot_count": int(nonpositive_count),
        "raw_report_audit": raw_report_audit,
        "rows": rows,
    }
    return report


def write_rows_csv(report: Mapping[str, Any], path: str | Path) -> None:
    rows = list(report.get("rows", []))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "slot_index",
        "slot_id",
        "start_frame",
        "end_frame",
        "target_frames",
        "frame_extent",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "duration_expected_seconds",
        "duration_error_seconds",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit V46.51 fresh-WAV Audio–Schedule Contract"
    )
    ap.add_argument("--audio", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--required_run_id", default=None)
    ap.add_argument("--max_frame_error", type=int, default=2)
    ap.add_argument("--max_seconds_error", type=float, default=0.10)
    ap.add_argument("--allow_failed", action="store_true")
    ap.add_argument("--allow_nonfresh", action="store_true")
    args = ap.parse_args(argv)

    report = audit_contract(
        audio=args.audio,
        schedule=args.schedule,
        fps=args.fps,
        required_run_id=args.required_run_id,
        require_fresh=not args.allow_nonfresh,
        max_frame_error=args.max_frame_error,
        max_seconds_error=args.max_seconds_error,
        require_raw_report=True,
    )
    save_json(report, args.out)
    if args.csv:
        write_rows_csv(report, args.csv)
    print(
        json.dumps(
            {
                "out": args.out,
                "ok": report["ok"],
                "reasons": report["reasons"],
                "audio_sha256": report["audio"]["sha256"],
                "num_slots": report["num_slots"],
                "total_target_frames": report["total_target_frames"],
                "expected_audio_target_frames": report[
                    "expected_audio_target_frames"
                ],
                "frame_error": report["frame_error"],
                "overlap_count": report["overlap_count"],
                "gap_count": report["gap_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] or args.allow_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
