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

from contracts.physical_quality import (
    evaluate_physical_audit,
    evaluate_pretraining_route_audit,
)
from motion_geometry.smpl24 import MOTION_DIM


def evaluate_scheduler_motion_tensor(motion: np.ndarray) -> dict:
    """Validate the raw whole-song motion before IK or metric computation."""

    shape = list(getattr(motion, "shape", ()))
    reasons: list[str] = []
    if motion.ndim != 2:
        reasons.append(f"motion_rank_mismatch:{motion.ndim}!=2")
    elif motion.shape[0] <= 0:
        reasons.append("motion_has_no_frames")
    elif motion.shape[1] != MOTION_DIM:
        reasons.append(f"motion_dim_mismatch:{motion.shape[1]}!={MOTION_DIM}")

    try:
        nonfinite_count = int((~np.isfinite(motion)).sum())
    except (TypeError, ValueError):
        nonfinite_count = -1
    if nonfinite_count != 0:
        reasons.append(f"motion_nonfinite_count:{nonfinite_count}")
    return {
        "schema": "pretraining_scheduler_motion_tensor_contract_v1",
        "ok": not reasons,
        "reasons": reasons,
        "shape": shape,
        "expected_motion_dim": int(MOTION_DIM),
        "nonfinite_count": nonfinite_count,
    }


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
        "--duration_model_ckpt", args.duration_ckpt,
        "--fps", str(args.fps),
        "--max_single_event_seconds", "5.0",
        "--calm_max_single_event_seconds", "4.5",
        "--min_subphrase_seconds", "2.5",
        "--max_events_per_phrase", "2",
        "--transition_min_frames", "8",
        "--transition_max_frames", "24",
        "--max_transition_fraction", str(args.max_transition_fraction),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)

    report_path = schedule_dir / f"{Path(args.audio).stem}.whole_song.schedule_report.json"
    motion_path = schedule_dir / f"{Path(args.audio).stem}.whole_song.npy"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    schedule = report["schedule"]
    motion = np.load(motion_path, allow_pickle=True)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    motion_tensor_contract = evaluate_scheduler_motion_tensor(motion)
    if not motion_tensor_contract["ok"]:
        raise RuntimeError(
            "Invalid same-WAV Scheduler motion tensor: "
            + ",".join(motion_tensor_contract["reasons"])
        )

    import training.motion_models as motion_runtime

    cfg = motion_runtime.MotionGenerationConfig.from_json(args.config).apply_env()
    if abs(float(cfg.fps) - float(args.fps)) > 1.0e-6:
        raise RuntimeError(
            f"Motion config FPS mismatch: config={cfg.fps}, requested={args.fps}"
        )
    if not args.skip_ik:
        motion, ik_report = motion_runtime.true_lower_body_ik(motion.astype(np.float32), cfg)
    else:
        contacts, confidence, floor, _ = motion_runtime.derive_contacts_np(motion, cfg)
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
    audit = motion_runtime.audit_motion_np(motion, cfg)
    # Stage 7E runs before Refiner, diffusion, and boundary closed-loop repair.
    # Keep the strict final-generation gate as an explicit diagnostic, while
    # using a separate catastrophic-only gate to decide whether training may
    # continue.
    physical = evaluate_physical_audit(audit)
    pretraining_physical = evaluate_pretraining_route_audit(audit)

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
    if not pretraining_physical["ok"]:
        reasons.extend(pretraining_physical["reasons"])

    result = {
        "schema": "same_wav_no_training_regression_v2",
        "ok": not reasons,
        "reasons": reasons,
        "audio": str(Path(args.audio).resolve()),
        "schedule_report": str(report_path),
        "action_motion": str(action_path),
        "event_db_contract": report.get("event_db_contract"),
        "motion_tensor_contract": motion_tensor_contract,
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
        "pretraining_physical_gate": pretraining_physical,
        "physical": physical,
        "physical_contract_role": "final_generation_diagnostic_only",
        "final_generation_physical_gate_required_after_motion_repair": True,
    }
    gate_path = out_dir / "regression_gate.json"
    gate_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
