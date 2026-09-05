"""Music-label-independent hard constraints for generated schedules.

All occupancy metrics use the selected motion Event and its allocated core
frames.  Music-side labels, predictions and semantic probabilities are never
consulted, so calm or ambiguous music cannot silently relax these gates.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Mapping, Sequence


DEFAULT_MAX_POSE_HOLD_RATIO = 0.25
DEFAULT_MAX_SINGLE_SOURCE_RATIO = 0.40
DEFAULT_MIN_UNIQUE_EVENTS = 4
DEFAULT_MIN_CORE_FRAME_RATIO = 0.70
_EPS = 1.0e-9
_CORE_FRAME_RATIO_FAILURE = re.compile(
    r"^core_frame_ratio_below_minimum:-?\d+(?:\.\d+)?<"
    r"-?\d+(?:\.\d+)?$"
)


class ScheduleHardConstraintError(RuntimeError):
    """Raised when a generated schedule violates an intrinsic hard gate."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        reasons = "; ".join(str(x) for x in report.get("reasons", []))
        super().__init__(f"Music-independent schedule hard constraints failed: {reasons}")


def _finite_number(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key not in row:
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _db_text(
    db: Mapping[str, Any], keys: Sequence[str], index: int
) -> str:
    for key in keys:
        values = db.get(key)
        if values is None:
            continue
        try:
            value = str(values[index]).strip()
        except (IndexError, KeyError, TypeError):
            continue
        if value:
            return value
    return ""


def final_selection_constraint_rows(
    db: Mapping[str, Any],
    assembly_report: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Resolve final closed-loop selections into the common audit row schema."""
    rows: list[Dict[str, Any]] = []
    for position, raw in enumerate(assembly_report):
        row = dict(raw)
        try:
            event_index = int(row.get("event_id", -1))
        except (TypeError, ValueError):
            event_index = -1
        source_uid = _db_text(
            db, ("source_uids", "source_groups"), event_index
        )
        recording_uid = _db_text(db, ("recording_uids",), event_index)
        if not recording_uid or recording_uid.lower() == "unknown":
            recording_uid = source_uid
        resolved = {
            **row,
            "slot": int(row.get("slot", position)),
            "allocated_content_len": row.get("core_frames"),
            "allocated_phrase_total": row.get(
                "target_frames", row.get("piece_frames")
            ),
            "event_uid": _db_text(db, ("event_uids",), event_index),
            "source_uid": source_uid,
            "recording_uid": recording_uid,
            "motion_event": _db_text(
                db, ("aesd_event_semantics", "event_types"), event_index
            ),
        }
        rows.append(resolved)
    return rows


def _ratio_limit(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0,1], got {value!r}")
    return number


def audit_schedule_hard_constraints(
    schedule: Sequence[Mapping[str, Any]],
    *,
    max_pose_hold_ratio: float = DEFAULT_MAX_POSE_HOLD_RATIO,
    max_single_source_ratio: float = DEFAULT_MAX_SINGLE_SOURCE_RATIO,
    max_single_recording_ratio: float | None = None,
    min_unique_events: int = DEFAULT_MIN_UNIQUE_EVENTS,
    min_core_frame_ratio: float = DEFAULT_MIN_CORE_FRAME_RATIO,
) -> Dict[str, Any]:
    """Compute and enforce intrinsic schedule diversity/coverage statistics."""
    max_pose_hold_ratio = _ratio_limit(
        "max_pose_hold_ratio", max_pose_hold_ratio
    )
    max_single_source_ratio = _ratio_limit(
        "max_single_source_ratio", max_single_source_ratio
    )
    max_single_recording_ratio = _ratio_limit(
        "max_single_recording_ratio",
        max_single_source_ratio
        if max_single_recording_ratio is None
        else max_single_recording_ratio,
    )
    min_core_frame_ratio = _ratio_limit(
        "min_core_frame_ratio", min_core_frame_ratio
    )
    min_unique_events = int(min_unique_events)
    if min_unique_events < 1:
        raise ValueError(
            f"min_unique_events must be at least 1, got {min_unique_events!r}"
        )

    rows = [dict(row) for row in schedule if isinstance(row, Mapping)]
    reasons: list[str] = []
    source_core_frames: Counter[str] = Counter()
    recording_core_frames: Counter[str] = Counter()
    event_uids: list[str] = []
    total_frames = 0
    core_frames = 0
    pose_hold_core_frames = 0
    missing_core_slots: list[int] = []
    missing_source_slots: list[int] = []
    missing_event_slots: list[int] = []
    missing_motion_event_slots: list[int] = []

    for index, row in enumerate(rows):
        target_value = _finite_number(
            row,
            (
                "allocated_phrase_total",
                "whole_song_allocated_phrase_total",
                "target_frames",
                "music_length",
            ),
        )
        core_value = _finite_number(
            row,
            ("allocated_content_len", "whole_song_allocated_content_len"),
        )
        target = int(round(target_value)) if target_value is not None else 0
        core = int(round(core_value)) if core_value is not None else 0
        if target <= 0:
            reasons.append(f"slot_{index}_invalid_target_frames:{target}")
        if core_value is None:
            missing_core_slots.append(index)
        elif core < 0 or (target > 0 and core > target):
            reasons.append(f"slot_{index}_invalid_core_frames:{core}/{target}")

        source_uid = _text(row, ("source_uid", "whole_song_source_uid"))
        recording_uid = _text(
            row,
            ("recording_uid", "whole_song_recording_uid"),
        ) or source_uid
        event_uid = _text(row, ("event_uid", "whole_song_event_uid"))
        motion_event = _text(row, ("motion_event", "event_type"))
        if not source_uid or source_uid.lower() == "unknown":
            missing_source_slots.append(index)
        if not event_uid:
            missing_event_slots.append(index)
        else:
            event_uids.append(event_uid)
        if not motion_event:
            missing_motion_event_slots.append(index)

        total_frames += max(0, target)
        core_frames += max(0, core)
        if source_uid and source_uid.lower() != "unknown":
            source_core_frames[source_uid] += max(0, core)
        if recording_uid and recording_uid.lower() != "unknown":
            recording_core_frames[recording_uid] += max(0, core)
        if motion_event.strip().lower() == "pose_hold":
            pose_hold_core_frames += max(0, core)

    if not rows:
        reasons.append("schedule_empty")
    if missing_core_slots:
        reasons.append(f"missing_core_frames_slots:{missing_core_slots}")
    if missing_source_slots:
        reasons.append(f"missing_source_uid_slots:{missing_source_slots}")
    if missing_event_slots:
        reasons.append(f"missing_event_uid_slots:{missing_event_slots}")
    if missing_motion_event_slots:
        reasons.append(
            f"missing_motion_event_slots:{missing_motion_event_slots}"
        )

    pose_hold_ratio = (
        float(pose_hold_core_frames / core_frames) if core_frames > 0 else 0.0
    )
    dominant_source = ""
    dominant_source_frames = 0
    if source_core_frames:
        dominant_source, dominant_source_frames = source_core_frames.most_common(1)[0]
    single_source_ratio = (
        float(dominant_source_frames / core_frames) if core_frames > 0 else 0.0
    )
    dominant_recording = ""
    dominant_recording_frames = 0
    if recording_core_frames:
        dominant_recording, dominant_recording_frames = (
            recording_core_frames.most_common(1)[0]
        )
    single_recording_ratio = (
        float(dominant_recording_frames / core_frames)
        if core_frames > 0
        else 0.0
    )
    unique_event_count = len(set(event_uids))
    core_frame_ratio = float(core_frames / total_frames) if total_frames > 0 else 0.0

    if core_frames <= 0:
        reasons.append("core_frames_zero")
    if pose_hold_ratio > max_pose_hold_ratio + _EPS:
        reasons.append(
            "pose_hold_ratio_exceeded:"
            f"{pose_hold_ratio:.6f}>{max_pose_hold_ratio:.6f}"
        )
    if single_source_ratio > max_single_source_ratio + _EPS:
        reasons.append(
            "single_source_ratio_exceeded:"
            f"{single_source_ratio:.6f}>{max_single_source_ratio:.6f}"
            f":source={dominant_source}"
        )
    if single_recording_ratio > max_single_recording_ratio + _EPS:
        reasons.append(
            "single_recording_ratio_exceeded:"
            f"{single_recording_ratio:.6f}>{max_single_recording_ratio:.6f}"
            f":recording={dominant_recording}"
        )
    if unique_event_count < min_unique_events:
        reasons.append(
            "unique_event_count_below_minimum:"
            f"{unique_event_count}<{min_unique_events}"
        )
    if core_frame_ratio + _EPS < min_core_frame_ratio:
        reasons.append(
            "core_frame_ratio_below_minimum:"
            f"{core_frame_ratio:.6f}<{min_core_frame_ratio:.6f}"
        )

    return {
        "schema": "music_independent_schedule_hard_constraints_v2_recording",
        "ok": not reasons,
        "formal_pass": not reasons,
        "diagnostic_bypass_used": False,
        "bypassed_reasons": [],
        "reasons": reasons,
        "music_label_independent": True,
        "occupancy_basis": "allocated_motion_core_frames",
        "limits": {
            "max_pose_hold_ratio": float(max_pose_hold_ratio),
            "max_single_source_ratio": float(max_single_source_ratio),
            "max_single_recording_ratio": float(max_single_recording_ratio),
            "min_unique_events": int(min_unique_events),
            "min_core_frame_ratio": float(min_core_frame_ratio),
        },
        "metrics": {
            "num_slots": int(len(rows)),
            "total_frames": int(total_frames),
            "core_frames": int(core_frames),
            "pose_hold_core_frames": int(pose_hold_core_frames),
            "pose_hold_ratio": float(pose_hold_ratio),
            "dominant_source_uid": dominant_source or None,
            "dominant_source_core_frames": int(dominant_source_frames),
            "single_source_ratio": float(single_source_ratio),
            "dominant_recording_uid": dominant_recording or None,
            "dominant_recording_core_frames": int(dominant_recording_frames),
            "single_recording_ratio": float(single_recording_ratio),
            "unique_event_count": int(unique_event_count),
            "core_frame_ratio": float(core_frame_ratio),
            "source_core_frames": dict(sorted(source_core_frames.items())),
            "recording_core_frames": dict(
                sorted(recording_core_frames.items())
            ),
        },
    }


def assert_schedule_hard_constraints(
    schedule: Sequence[Mapping[str, Any]],
    **limits: Any,
) -> Dict[str, Any]:
    import os

    report = audit_schedule_hard_constraints(schedule, **limits)
    if report["ok"]:
        return report

    # This environment switch is intentionally a diagnostic escape hatch for
    # the core-frame ratio only.  It must never turn an invalid schedule into
    # a formally passing report or hide a second hard-constraint failure.
    bypass_requested = os.environ.get(
        "DISABLE_FINAL_CORE_FRAME_ASSERT", "0"
    ) == "1"
    bypassed_reasons = [
        reason
        for reason in report["reasons"]
        if _CORE_FRAME_RATIO_FAILURE.fullmatch(str(reason))
    ]
    if bypass_requested and bypassed_reasons and len(bypassed_reasons) == len(
        report["reasons"]
    ):
        diagnostic_report = dict(report)
        diagnostic_report["formal_pass"] = False
        diagnostic_report["diagnostic_bypass_used"] = True
        diagnostic_report["bypassed_reasons"] = list(bypassed_reasons)
        print(
            "[WARNING] DISABLE_FINAL_CORE_FRAME_ASSERT=1, "
            "bypassing only core_frame_ratio_below_minimum for diagnostic run; "
            "formal_pass remains false",
            flush=True,
        )
        return diagnostic_report

    failed_report = dict(report)
    failed_report["formal_pass"] = False
    failed_report["diagnostic_bypass_used"] = False
    failed_report["bypassed_reasons"] = []
    raise ScheduleHardConstraintError(failed_report)
