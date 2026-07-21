#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a final V46.38 Music Semantic Slot Descriptor from V21/V26/V23 schedule output.

This script can either reuse an existing `*_v26.schedule_report.json` or invoke
`scheduling.whole_song_scheduler` with trained scheduler checkpoints and then
convert the schedule report into a strict final descriptor.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Make the script runnable from repo root without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduling.descriptor_schema import (  # noqa: E402
    MSSD_SCHEMA_VERSION,
    build_descriptor_object,
    json_load,
    json_save,
    normalize_slot,
)


def _find_report_from_summary(summary_path: Path, audio: Path) -> Optional[Path]:
    if not summary_path.exists():
        return None
    obj = json_load(summary_path)
    results = obj.get("results", {}) if isinstance(obj, dict) else {}
    keys = [audio.stem]
    # Some users pass mp3/wav variants; also accept the only result.
    if len(results) == 1:
        keys.append(next(iter(results.keys())))
    for k in keys:
        val = results.get(k)
        if isinstance(val, dict) and val.get("report"):
            rp = Path(str(val["report"]))
            if not rp.is_absolute():
                rp = summary_path.parent / rp.name if not rp.exists() else rp
            if rp.exists():
                return rp
    return None


def find_existing_report(schedule_dir: Path, audio: Path) -> Optional[Path]:
    direct = schedule_dir / f"{audio.stem}_v26.schedule_report.json"
    if direct.exists():
        return direct
    summary = schedule_dir / "V26_WHOLE_SONG_SUMMARY.json"
    rp = _find_report_from_summary(summary, audio)
    if rp and rp.exists():
        return rp
    hits = sorted(schedule_dir.glob("*_v26.schedule_report.json"))
    if len(hits) == 1:
        return hits[0]
    return None


def run_v26_schedule(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scheduling.whole_song_scheduler",
        "--index_json", args.index_json,
        "--duration_index_npz", args.duration_index_npz,
        "--music", args.audio,
        "--out_dir", args.schedule_dir,
        "--router_ckpt", args.router_ckpt,
        "--v23_ckpt", args.v23_ckpt,
        "--feature_dir", args.feature_dir,
        "--fps", str(args.fps),
        "--min_phrase_seconds", str(args.min_phrase_seconds),
        "--max_phrase_seconds", str(args.max_phrase_seconds),
        "--max_phrases", str(args.max_phrases),
        "--multi_event_phrases", "1",
        "--lock_music_boundaries", "1",
        "--music_dominant_timing", "1",
    ]
    if args.planner_ckpt:
        cmd += ["--planner_ckpt", args.planner_ckpt]
    if args.hierarchy_index_npz:
        cmd += ["--hierarchy_index_npz", args.hierarchy_index_npz]
    print("[V46.38 MSSD SCHEDULE]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))


def convert_report_to_descriptor(report_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    report = json_load(report_path)
    schedule = report.get("schedule", []) if isinstance(report, dict) else []
    if not isinstance(schedule, list) or not schedule:
        raise RuntimeError(f"V26 schedule report has no schedule list: {report_path}")
    fps = float(args.fps)
    slots: List[dict] = []
    feats = []
    cursor = 0.0
    meta = {
        "usage": "generate_schedule",
        "is_final_schedule": True,
        "slot_source": "v21_router_v26_planner",
        "fps": fps,
        "router_ckpt": args.router_ckpt,
        "planner_ckpt": args.planner_ckpt,
        "v23_ckpt": args.v23_ckpt,
        "raw_schedule_json": str(report_path),
        "event_db_contract": dict(report.get("event_db_contract", {})),
        "transition_budget": dict(report.get("transition_budget", {})),
        "schedule_summary_json": str(Path(args.schedule_dir) / "V26_WHOLE_SONG_SUMMARY.json"),
        "provenance": {
            "builder": "scheduling.music_slot_descriptor",
            "source_report": str(report_path),
            "strict_generation_descriptor": True,
        },
    }
    for i, row0 in enumerate(schedule):
        row = dict(row0)
        # V26 rows usually store musical boundaries in frames and allocation fields.
        target = row.get("allocated_phrase_total", row.get("music_length", None))
        if target is None:
            target = max(1, int(round(float(row.get("duration", 4.0)) * fps)))
        target = int(round(float(target)))
        row.setdefault("target_frames", target)
        row.setdefault("start", cursor)
        row.setdefault("end", cursor + target / fps)
        row.setdefault("duration", target / fps)
        row.setdefault("slot_source", "v21_router_v26_planner")
        row.setdefault("music_alignment_label", row.get("music_event", row.get("motion_event", "lyrical_flow")))
        row.setdefault("music_semantic_top_label", row.get("music_alignment_label", "lyrical_flow"))
        slot, feat = normalize_slot(row, meta, i, fps=fps, source_path=str(report_path))
        # Replace timing by cumulative allocated_phrase_total to avoid small report-boundary drifts.
        slot["start"] = slot["start_sec"] = float(cursor)
        slot["end"] = slot["end_sec"] = float(cursor + target / fps)
        slot["duration"] = slot["duration_sec"] = float(target / fps)
        slot["start_frame"] = int(round(cursor * fps))
        slot["end_frame"] = int(round(cursor * fps)) + target
        slot["target_frames"] = target
        cursor += target / fps
        slots.append(slot)
        feats.append(feat)
    obj = build_descriptor_object(args.audio, slots, meta)
    obj["descriptor_schema_version"] = MSSD_SCHEMA_VERSION
    obj["raw_v26_report_summary"] = {
        "frames_from_slots": int(sum(int(s.get("target_frames", 0)) for s in slots)),
        "phrases": int(len(slots)),
        "report_out_npy": str(report.get("out_npy", "")) if isinstance(report, dict) else "",
    }
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--router_ckpt", required=True)
    ap.add_argument("--planner_ckpt", default="")
    ap.add_argument("--v23_ckpt", required=True)
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--duration_index_npz", required=True)
    ap.add_argument("--hierarchy_index_npz", default="")
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--schedule_dir", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min_phrase_seconds", type=float, default=2.5)
    ap.add_argument("--max_phrase_seconds", type=float, default=7.5)
    ap.add_argument("--max_phrases", type=int, default=160)
    ap.add_argument("--force_reschedule", action="store_true")
    args = ap.parse_args(argv)

    audio = Path(args.audio)
    schedule_dir = Path(args.schedule_dir)
    schedule_dir.mkdir(parents=True, exist_ok=True)
    Path(args.feature_dir).mkdir(parents=True, exist_ok=True)

    report = None if args.force_reschedule else find_existing_report(schedule_dir, audio)
    if report is None:
        run_v26_schedule(args)
        report = find_existing_report(schedule_dir, audio)
    if report is None or not report.exists():
        raise RuntimeError(f"Could not obtain V26 schedule report for {audio} in {schedule_dir}")

    desc = convert_report_to_descriptor(report, args)
    json_save(desc, args.out_json)
    print(json.dumps({
        "out_json": args.out_json,
        "descriptor_type": desc.get("descriptor_type"),
        "usage": desc.get("usage"),
        "is_final_schedule": desc.get("is_final_schedule"),
        "slot_source": desc.get("slot_source"),
        "num_slots": desc.get("num_slots"),
        "total_target_frames": desc.get("total_target_frames"),
        "raw_schedule_json": desc.get("raw_schedule_json"),
        "first_slot": desc.get("slots", [{}])[0],
        "last_slot": desc.get("slots", [{}])[-1],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
