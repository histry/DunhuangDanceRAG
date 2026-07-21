#!/usr/bin/env python3
"""Build isolated canonical-30 and native-60 data/model branches.

The command is dry-run by default.  Pass ``--execute`` to rebuild caches,
Event-DBs, Scheduler indexes and duration assets.  V45/V46 training remains an
explicit ``--train_v45_v46`` action because it is expensive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_geometry.smpl24 import skeleton_contract

RATE_FIELDS = (
    "window_len",
    "hop_len",
    "min_event_frames",
    "max_event_frames",
    "overlap",
    "ik_chunk",
    "ik_chunk_overlap",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_profile(base: dict[str, Any], fps: float, target: Path) -> dict[str, Any]:
    profile = dict(base)
    scale = float(fps) / 30.0
    profile["fps"] = float(fps)
    for key in RATE_FIELDS:
        if key in base:
            profile[key] = max(1, int(round(float(base[key]) * scale)))
    profile["multirate_contract"] = {
        "schema": "dunhuang_multirate_motion_config_v1",
        "fps": float(fps),
        "base_fps": 30.0,
        "skeleton": skeleton_contract(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def _run(command: list[str], *, execute: bool, env: dict[str, str]) -> None:
    print("[EXEC]" if execute else "[PLAN]", subprocess.list2cmdline(command), flush=True)
    if execute:
        subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def _checkpoint_for_rate(args: argparse.Namespace, fps: int) -> str | None:
    return getattr(args, f"duration_ckpt_{fps}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dirs", nargs="+", required=True, help="BVH/AIST++ roots; each root gets an isolated retarget cache.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--base_config", default="configs/motion_model.json")
    parser.add_argument("--duration_ckpt_30")
    parser.add_argument("--duration_ckpt_60")
    parser.add_argument("--router_ckpt_30")
    parser.add_argument("--router_ckpt_60")
    parser.add_argument("--planner_ckpt_30")
    parser.add_argument("--planner_ckpt_60")
    parser.add_argument(
        "--regression_audio",
        help="One WAV used for the mandatory no-training gate before V45/V46 training.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smpl_scaling_mode", default="canonical_body", choices=("canonical_body", "scale_translation", "inverse_scale_translation"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train_v45_v46", action="store_true")
    parser.add_argument("--refiner_steps", type=int)
    parser.add_argument("--diffusion_steps", type=int)
    args = parser.parse_args(argv)

    if args.execute and args.train_v45_v46 and not args.regression_audio:
        raise RuntimeError(
            "--train_v45_v46 requires --regression_audio so route/action metrics "
            "can gate training for both frame-rate branches."
        )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    python = sys.executable
    intervals = output_root / "canonical30" / "event_intervals.json"
    plan: dict[str, Any] = {
        "schema": "dunhuang_multirate_build_plan_v1",
        "source_dirs": [str(Path(p).resolve()) for p in args.source_dirs],
        "skeleton_contract": skeleton_contract(),
        "branches": {},
    }

    for fps in (30, 60):
        name = "canonical30" if fps == 30 else "native60"
        branch = output_root / name
        config = branch / "motion_model.json"
        profile = _write_profile(base, float(fps), config)
        cache_dirs: list[Path] = []
        commands: list[list[str]] = []
        for index, raw in enumerate(args.source_dirs):
            source = Path(raw).resolve()
            cache = branch / "retarget_cache" / f"source_{index:02d}_{source.name}"
            cache_dirs.append(cache)
            command = [
                python, "-m", "retargeting.build_cache",
                "--in_dir", str(source),
                "--out_dir", str(cache),
                "--target_fps", str(fps),
                "--device", args.device,
                "--smpl_scaling_mode", args.smpl_scaling_mode,
                "--allow_partial",
            ]
            if args.overwrite:
                command.append("--overwrite")
            commands.append(command)

        event_db = branch / "event_db"
        event_command = [
            python, "-m", "events.build_database",
            "--motion_dirs", *[str(path) for path in cache_dirs],
            "--out_db", str(event_db),
            "--config", str(config),
        ]
        if args.overwrite:
            event_command.append("--overwrite")
        if fps == 30:
            event_command.extend(("--canonical_intervals_out", str(intervals)))
        else:
            event_command.extend(("--canonical_intervals_in", str(intervals)))
        commands.append(event_command)

        semantic_db = event_db / "events_aesd.npz"
        commands.append([
            python, "-m", "events.build_semantics",
            "--db", str(event_db / "events.npz"),
            "--out", str(semantic_db),
            "--json", str(event_db / "events_aesd.report.json"),
        ])
        scheduler = branch / "scheduler"
        base_index = scheduler / "event_index.base.npz"
        commands.append([
            python, "-m", "scheduling.build_generation_index",
            "--db", str(semantic_db),
            "--out_json", str(scheduler / "event_index.json"),
            "--out_npz", str(base_index),
            "--report", str(scheduler / "event_index.report.json"),
        ])

        duration_ckpt = _checkpoint_for_rate(args, fps)
        router_ckpt = getattr(args, f"router_ckpt_{fps}")
        planner_ckpt = getattr(args, f"planner_ckpt_{fps}")
        if args.execute and not all((router_ckpt, planner_ckpt, duration_ckpt)):
            raise RuntimeError(
                f"Formal {fps} FPS build requires rate-specific Router, Planner and Duration "
                f"checkpoints (--router_ckpt_{fps}, --planner_ckpt_{fps}, --duration_ckpt_{fps})."
            )
        if duration_ckpt:
            commands.append([
                python, "-m", "scheduling.build_duration_index",
                "--index_json", str(scheduler / "event_index.json"),
                "--index_npz", str(base_index),
                "--v23_checkpoint", str(Path(duration_ckpt).resolve()),
                "--out_npz", str(scheduler / "duration_index.npz"),
                "--out_json", str(scheduler / "duration_index.report.json"),
                "--fps", str(fps),
                "--device", args.device,
            ])
        if args.regression_audio and router_ckpt and planner_ckpt and duration_ckpt:
            commands.append([
                python, "-m", "scripts.run_no_training_regression",
                "--audio", str(Path(args.regression_audio).resolve()),
                "--index_json", str(scheduler / "event_index.json"),
                "--index_npz", str(scheduler / "duration_index.npz"),
                "--router_ckpt", str(Path(router_ckpt).resolve()),
                "--planner_ckpt", str(Path(planner_ckpt).resolve()),
                "--duration_ckpt", str(Path(duration_ckpt).resolve()),
                "--config", str(config),
                "--out_dir", str(branch / "no_training_regression"),
                "--fps", str(fps),
            ])
        if router_ckpt and planner_ckpt and duration_ckpt:
            commands.append([
                python, "-m", "scheduling.build_asset_bundle",
                "--index_json", str(scheduler / "event_index.json"),
                "--index_npz", str(scheduler / "duration_index.npz"),
                "--router_ckpt", str(Path(router_ckpt).resolve()),
                "--planner_ckpt", str(Path(planner_ckpt).resolve()),
                "--duration_ckpt", str(Path(duration_ckpt).resolve()),
                "--fps", str(fps),
                "--out", str(scheduler / "asset_bundle.json"),
            ])

        if args.train_v45_v46:
            refiner = [python, "training/motion_models.py", "--config", str(config), "train-refiner", "--db", str(semantic_db), "--out", str(branch / "checkpoints" / "boundary_refiner.pt")]
            diffusion = [python, "training/motion_models.py", "--config", str(config), "train-diffusion", "--db", str(semantic_db), "--out", str(branch / "checkpoints" / "local_diffusion.pt")]
            if args.refiner_steps:
                refiner.extend(("--steps", str(args.refiner_steps)))
            if args.diffusion_steps:
                diffusion.extend(("--steps", str(args.diffusion_steps)))
            commands.extend((refiner, diffusion))

        plan["branches"][name] = {
            "fps": fps,
            "config": str(config),
            "config_sha256": _sha256(config),
            "profile": profile,
            "event_db": str(semantic_db),
            "scheduler_index": str(scheduler / "event_index.json"),
            "duration_index": str(scheduler / "duration_index.npz"),
            "asset_bundle": str(scheduler / "asset_bundle.json"),
            "commands": commands,
        }
        for command in commands:
            _run(command, execute=args.execute, env=env)

    plan_path = output_root / "multirate_build_manifest.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
