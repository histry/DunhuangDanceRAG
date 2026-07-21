#!/usr/bin/env python3
"""Run same-WAV route/action regression without updating model weights."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing.boundary_closed_loop import physical_quality_gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--index_npz", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--planner_ckpt", required=True)
    parser.add_argument("--duration_ckpt", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_slots", type=int, default=40)
    parser.add_argument("--min_unique_ratio", type=float, default=0.80)
    parser.add_argument("--max_source_share", type=float, default=0.40)
    parser.add_argument("--max_transition_fraction", type=float, default=0.20)
    parser.add_argument("--skip_ik", action="store_true")
    parser.add_argument(
        "--allow_legacy_30fps_checkpoints",
        action="store_true",
        help="Parity baseline only: permit old Scheduler checkpoints that predate FPS metadata.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    schedule_dir = out_dir / "schedule"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "scheduling.whole_song_scheduler",
        "--index_json", args.index_json,
        "--duration_index_npz", args.index_npz,
        "--music", args.audio,
        "--out_dir", str(schedule_dir),
        "--router_ckpt", args.router_ckpt,
        "--planner_ckpt", args.planner_ckpt,
        "--v23_ckpt", args.duration_ckpt,
        "--fps", str(args.fps),
        "--deep_music_features", "0",
        "--require_deep_music", "0",
        "--max_single_event_seconds", "5.0",
        "--calm_max_single_event_seconds", "4.5",
        "--min_subphrase_seconds", "2.5",
        "--max_events_per_phrase", "2",
        "--transition_min_frames", "8",
        "--transition_max_frames", "24",
        "--max_transition_fraction", str(args.max_transition_fraction),
    ]
    env = os.environ.copy()
    if args.allow_legacy_30fps_checkpoints:
        if abs(float(args.fps) - 30.0) > 1.0e-6:
            raise RuntimeError("Legacy checkpoint compatibility is restricted to 30 FPS")
        env["DUNHUANG_ALLOW_LEGACY_30FPS_CHECKPOINTS"] = "1"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)

    report_path = schedule_dir / f"{Path(args.audio).stem}_v26.schedule_report.json"
    motion_path = schedule_dir / f"{Path(args.audio).stem}_v26.npy"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    schedule = report["schedule"]
    motion = np.load(motion_path, allow_pickle=True)
    if motion.ndim == 3:
        motion = motion[0]

    import training.motion_models as v46

    cfg = v46.V46Config.from_json(args.config).apply_env()
    if abs(float(cfg.fps) - float(args.fps)) > 1.0e-6:
        raise RuntimeError(
            f"Motion config FPS mismatch: config={cfg.fps}, requested={args.fps}"
        )
    if not args.skip_ik:
        motion, ik_report = v46.true_lower_body_ik(motion.astype(np.float32), cfg)
    else:
        contacts, confidence, floor, _ = v46.derive_contacts_np(motion, cfg)
        motion = motion.copy().astype(np.float32)
        motion[:, :4] = contacts.astype(np.float32)
        ik_report = {
            "enabled": False,
            "contact_recomputed": True,
            "confidence_mean": float(confidence.mean()),
            "floor_y": float(floor),
        }
    action_path = out_dir / "same_wav_no_training_action.npy"
    np.save(action_path, motion.astype(np.float32))
    audit = v46.audit_motion_np(motion, cfg)
    physical = physical_quality_gate(audit)

    event_uids = [str(row.get("event_uid", row.get("event_id"))) for row in schedule]
    sources = [str(row.get("source_uid", "unknown")) for row in schedule]
    unique_ratio = len(set(event_uids)) / max(1, len(event_uids))
    adjacent_repeats = sum(a == b for a, b in zip(event_uids, event_uids[1:]))
    source_counts = Counter(sources)
    source_share = max(source_counts.values(), default=0) / max(1, len(sources))
    transition_fraction = float(report.get("transition_budget", {}).get("actual_fraction", 1.0))
    reasons: list[str] = []
    if len(schedule) > args.max_slots:
        reasons.append("slot_count_too_high")
    if unique_ratio < args.min_unique_ratio:
        reasons.append("event_unique_ratio_too_low")
    if adjacent_repeats:
        reasons.append("adjacent_event_repeat")
    if source_share > args.max_source_share:
        reasons.append("source_share_too_high")
    if transition_fraction > args.max_transition_fraction + 1e-9:
        reasons.append("transition_fraction_too_high")
    if not physical["ok"]:
        reasons.extend(physical["reasons"])

    result = {
        "schema": "same_wav_no_training_regression_v1",
        "ok": not reasons,
        "reasons": reasons,
        "audio": str(Path(args.audio).resolve()),
        "schedule_report": str(report_path),
        "action_motion": str(action_path),
        "event_db_contract": report.get("event_db_contract"),
        "route": {
            "num_slots": len(schedule),
            "unique_events": len(set(event_uids)),
            "unique_ratio": unique_ratio,
            "adjacent_repeats": adjacent_repeats,
            "source_counts": dict(source_counts),
            "max_source_share": source_share,
            "transition_fraction": transition_fraction,
        },
        "ik": ik_report,
        "physical": physical,
    }
    gate_path = out_dir / "regression_gate.json"
    gate_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
