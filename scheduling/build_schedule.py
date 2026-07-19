#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.51 rebuild a strict MSSD transaction from the current WAV.

This script never searches for or reuses an old MSSD.  It creates a unique
run-local feature cache and raw V26 schedule directory, invokes the current
V21/V26/V23 scheduler directly from the supplied WAV, converts the resulting
report to the existing V46.38 MSSD schema, stamps immutable provenance, and
runs the V46.51 Audio–Schedule Contract before returning success.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduling.validate_schedule import (  # noqa: E402
    audit_contract,
    save_json,
    stamp_descriptor,
    write_rows_csv,
)


def bool_text(value: bool) -> str:
    return "1" if bool(value) else "0"


def require_file(path: str | Path, label: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"{label} does not exist: {p}")
    return p.resolve()


def run_checked(cmd: Sequence[str], *, env: Optional[Dict[str, str]] = None) -> None:
    print("[RUN]", shlex.join([str(x) for x in cmd]), flush=True)
    subprocess.run(
        [str(x) for x in cmd],
        check=True,
        env=env,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a fresh current-WAV V21/V26/V23 MSSD transaction"
    )
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--run_id", required=True)

    ap.add_argument("--router_ckpt", required=True)
    ap.add_argument("--planner_ckpt", required=True)
    ap.add_argument("--v23_ckpt", required=True)
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--duration_index_npz", required=True)
    ap.add_argument("--hierarchy_index_npz", default="")
    ap.add_argument("--transition_ckpt", default="")
    ap.add_argument("--transition_diffusion_ckpt", default="")
    ap.add_argument("--start_pose", default="")
    ap.add_argument("--hyperbolic_ckpt", default="")

    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max_seconds", type=float, default=0.0)
    ap.add_argument("--min_phrase_seconds", type=float, default=2.5)
    ap.add_argument("--max_phrase_seconds", type=float, default=7.5)
    ap.add_argument("--max_phrases", type=int, default=160)
    ap.add_argument("--boundary_quantile", type=float, default=0.68)
    ap.add_argument("--beat_snap_seconds", type=float, default=0.35)
    ap.add_argument("--max_single_event_seconds", type=float, default=3.20)
    ap.add_argument("--calm_max_single_event_seconds", type=float, default=2.80)
    ap.add_argument("--min_subphrase_seconds", type=float, default=1.60)
    ap.add_argument("--max_events_per_phrase", type=int, default=4)
    ap.add_argument("--slot_beat_snap_seconds", type=float, default=0.25)

    ap.add_argument("--beam_size", type=int, default=24)
    ap.add_argument("--candidate_top_k", type=int, default=256)
    ap.add_argument("--graph_node_top_k", type=int, default=96)
    ap.add_argument("--graph_edge_weight", type=float, default=0.45)
    ap.add_argument("--graph_hard_prune", action="store_true")
    ap.add_argument("--graph_hard_prune_threshold", type=float, default=1.35)

    ap.add_argument("--deep_music_features", action="store_true")
    ap.add_argument("--deep_music_model", default="clap")
    ap.add_argument("--deep_music_weight", type=float, default=0.25)
    ap.add_argument("--require_deep_music", action="store_true")
    ap.add_argument("--deep_music_min_success", type=float, default=0.80)
    ap.add_argument("--deep_music_cache", default="")

    ap.add_argument("--transition_diffusion", action="store_true")
    ap.add_argument("--transition_diffusion_blend", type=float, default=0.18)
    ap.add_argument("--transition_diffusion_steps", type=int, default=32)

    ap.add_argument("--overwrite_run_dir", action="store_true")
    ap.add_argument("--hash_assets", action="store_true")
    ap.add_argument("--max_frame_error", type=int, default=2)
    ap.add_argument("--max_seconds_error", type=float, default=0.10)
    args = ap.parse_args(argv)

    audio = require_file(args.audio, "audio")
    router_ckpt = require_file(args.router_ckpt, "router checkpoint")
    planner_ckpt = require_file(args.planner_ckpt, "planner checkpoint")
    v23_ckpt = require_file(args.v23_ckpt, "V23 checkpoint")
    index_json = require_file(args.index_json, "V21 index JSON")
    duration_npz = require_file(
        args.duration_index_npz,
        "V26 duration index NPZ",
    )

    hierarchy = (
        require_file(args.hierarchy_index_npz, "hierarchy index NPZ")
        if args.hierarchy_index_npz
        else None
    )
    transition_ckpt = (
        require_file(args.transition_ckpt, "transition checkpoint")
        if args.transition_ckpt
        else None
    )
    transition_diffusion_ckpt = (
        require_file(
            args.transition_diffusion_ckpt,
            "transition diffusion checkpoint",
        )
        if args.transition_diffusion_ckpt
        else None
    )
    start_pose = (
        require_file(args.start_pose, "start pose")
        if args.start_pose
        else None
    )
    hyperbolic_ckpt = (
        require_file(args.hyperbolic_ckpt, "hyperbolic checkpoint")
        if args.hyperbolic_ckpt
        else None
    )

    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.exists():
        if args.overwrite_run_dir:
            shutil.rmtree(run_dir)
        elif any(run_dir.iterdir()):
            raise RuntimeError(
                f"Fresh schedule run_dir must be new and empty: {run_dir}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_schedule_dir = run_dir / "raw_v26_schedule"
    feature_dir = run_dir / "music_features"
    deep_cache = (
        Path(args.deep_music_cache).expanduser().resolve()
        if args.deep_music_cache
        else run_dir / "deep_music_cache"
    )
    raw_schedule_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    deep_cache.mkdir(parents=True, exist_ok=True)

    scheduler_script = require_file(
        ROOT / "tools" / "schedule_v26_whole_song.py",
        "current V26 scheduler",
    )
    descriptor_script = require_file(
        ROOT / "tools" / "v46_38_build_music_semantic_slot_descriptor.py",
        "current V46.38 MSSD converter",
    )

    scheduler_cmd: List[str] = [
        sys.executable,
        str(scheduler_script),
        "--index_json",
        str(index_json),
        "--duration_index_npz",
        str(duration_npz),
        "--music",
        str(audio),
        "--out_dir",
        str(raw_schedule_dir),
        "--router_ckpt",
        str(router_ckpt),
        "--planner_ckpt",
        str(planner_ckpt),
        "--v23_ckpt",
        str(v23_ckpt),
        "--feature_dir",
        str(feature_dir),
        "--deep_music_cache",
        str(deep_cache),
        "--fps",
        str(args.fps),
        "--max_seconds",
        str(args.max_seconds),
        "--min_phrase_seconds",
        str(args.min_phrase_seconds),
        "--max_phrase_seconds",
        str(args.max_phrase_seconds),
        "--boundary_quantile",
        str(args.boundary_quantile),
        "--beat_snap_seconds",
        str(args.beat_snap_seconds),
        "--max_phrases",
        str(args.max_phrases),
        "--multi_event_phrases",
        "1",
        "--lock_music_boundaries",
        "1",
        "--music_dominant_timing",
        "1",
        "--max_single_event_seconds",
        str(args.max_single_event_seconds),
        "--calm_max_single_event_seconds",
        str(args.calm_max_single_event_seconds),
        "--min_subphrase_seconds",
        str(args.min_subphrase_seconds),
        "--max_events_per_phrase",
        str(args.max_events_per_phrase),
        "--slot_beat_snap_seconds",
        str(args.slot_beat_snap_seconds),
        "--beam_size",
        str(args.beam_size),
        "--candidate_top_k",
        str(args.candidate_top_k),
        "--hierarchical_retrieval",
        "1",
        "--graph_scheduler",
        "1",
        "--graph_node_top_k",
        str(args.graph_node_top_k),
        "--graph_edge_weight",
        str(args.graph_edge_weight),
        "--graph_hard_prune",
        bool_text(args.graph_hard_prune),
        "--graph_hard_prune_threshold",
        str(args.graph_hard_prune_threshold),
        "--deep_music_features",
        bool_text(args.deep_music_features),
        "--deep_music_model",
        str(args.deep_music_model),
        "--deep_music_weight",
        str(args.deep_music_weight),
        "--require_deep_music",
        bool_text(args.require_deep_music),
        "--deep_music_min_success",
        str(args.deep_music_min_success),
        "--transition_diffusion",
        bool_text(args.transition_diffusion),
        "--transition_diffusion_blend",
        str(args.transition_diffusion_blend),
        "--transition_diffusion_steps",
        str(args.transition_diffusion_steps),
    ]
    if hierarchy is not None:
        scheduler_cmd += ["--hierarchy_index_npz", str(hierarchy)]
    if transition_ckpt is not None:
        scheduler_cmd += ["--transition_ckpt", str(transition_ckpt)]
    if transition_diffusion_ckpt is not None:
        scheduler_cmd += [
            "--transition_diffusion_ckpt",
            str(transition_diffusion_ckpt),
        ]
    if start_pose is not None:
        scheduler_cmd += ["--start_pose", str(start_pose)]
    if hyperbolic_ckpt is not None:
        scheduler_cmd += ["--hyperbolic_ckpt", str(hyperbolic_ckpt)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else ""
    )
    env["V46_51_SCHEDULE_RUN_ID"] = str(args.run_id)
    run_checked(scheduler_cmd, env=env)

    raw_report = (
        raw_schedule_dir
        / f"{audio.stem}_v26.schedule_report.json"
    )
    if not raw_report.is_file():
        raise RuntimeError(
            f"Fresh V26 scheduler did not produce expected report: {raw_report}"
        )

    temporary_descriptor = run_dir / "descriptor_unstamped.json"
    converter_cmd = [
        sys.executable,
        str(descriptor_script),
        "--audio",
        str(audio),
        "--out_json",
        str(temporary_descriptor),
        "--router_ckpt",
        str(router_ckpt),
        "--planner_ckpt",
        str(planner_ckpt),
        "--v23_ckpt",
        str(v23_ckpt),
        "--index_json",
        str(index_json),
        "--duration_index_npz",
        str(duration_npz),
        "--feature_dir",
        str(feature_dir),
        "--schedule_dir",
        str(raw_schedule_dir),
        "--fps",
        str(args.fps),
        "--min_phrase_seconds",
        str(args.min_phrase_seconds),
        "--max_phrase_seconds",
        str(args.max_phrase_seconds),
        "--max_phrases",
        str(args.max_phrases),
    ]
    if hierarchy is not None:
        converter_cmd += ["--hierarchy_index_npz", str(hierarchy)]
    # Deliberately do not pass --force_reschedule here. The report was just
    # created in this unique run directory by scheduler_cmd.
    run_checked(converter_cmd, env=env)

    descriptor = json.loads(
        temporary_descriptor.read_text(encoding="utf-8")
    )
    assets = {
        "router_ckpt": router_ckpt,
        "planner_ckpt": planner_ckpt,
        "v23_ckpt": v23_ckpt,
        "index_json": index_json,
        "duration_index_npz": duration_npz,
        "hierarchy_index_npz": hierarchy,
        "transition_ckpt": transition_ckpt,
        "transition_diffusion_ckpt": transition_diffusion_ckpt,
        "start_pose": start_pose,
        "hyperbolic_ckpt": hyperbolic_ckpt,
    }
    stamped = stamp_descriptor(
        descriptor,
        audio=audio,
        fps=args.fps,
        run_id=args.run_id,
        run_dir=run_dir,
        raw_schedule_json=raw_report,
        scheduler_command=scheduler_cmd,
        assets=assets,
        hash_assets=bool(args.hash_assets),
    )
    stamped["v46_51_scheduler_policy"] = {
        "fresh_schedule_each_generation": True,
        "old_mssd_reuse": False,
        "unique_feature_cache": True,
        "unique_raw_schedule_dir": True,
        "music_dominant_timing": True,
        "lock_music_boundaries": True,
        "hierarchical_retrieval": True,
        "graph_scheduler": True,
        "deep_music_features": bool(args.deep_music_features),
        "require_deep_music": bool(args.require_deep_music),
    }

    out_json = Path(args.out_json).expanduser().resolve()
    save_json(stamped, out_json)

    contract_report = audit_contract(
        audio=audio,
        schedule=out_json,
        fps=args.fps,
        required_run_id=args.run_id,
        require_fresh=True,
        max_frame_error=args.max_frame_error,
        max_seconds_error=args.max_seconds_error,
        require_raw_report=True,
    )
    contract_path = out_json.with_suffix(
        out_json.suffix + ".contract.json"
    )
    contract_csv = out_json.with_suffix(
        out_json.suffix + ".contract.csv"
    )
    save_json(contract_report, contract_path)
    write_rows_csv(contract_report, contract_csv)
    if not contract_report["ok"]:
        raise RuntimeError(
            "Fresh-WAV schedule contract failed: "
            + "; ".join(contract_report["reasons"])
        )

    build_report = {
        "schema": "v46_51_fresh_wav_mssd_build",
        "ok": True,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "audio": stamped.get("provenance", {})
        .get("v46_51", {})
        .get("audio", {}),
        "out_json": str(out_json),
        "contract_json": str(contract_path),
        "contract_csv": str(contract_csv),
        "raw_schedule_json": str(raw_report),
        "raw_schedule_dir": str(raw_schedule_dir),
        "feature_dir": str(feature_dir),
        "num_slots": contract_report["num_slots"],
        "total_target_frames": contract_report[
            "total_target_frames"
        ],
        "deep_music_features": bool(args.deep_music_features),
        "require_deep_music": bool(args.require_deep_music),
    }
    build_report_path = run_dir / "v46_51_fresh_mssd_build.json"
    save_json(build_report, build_report_path)
    print(json.dumps(build_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
