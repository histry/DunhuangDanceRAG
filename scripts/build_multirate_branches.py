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
from support.checkpoint_contracts import (
    assert_checkpoint_fps,
    checkpoint_declared_fps,
)

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


def _load_checkpoint_contract(path: Path) -> dict[str, Any]:
    # Import lazily so dry-run planning remains usable in lightweight
    # environments that do not install the training runtime.
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def _preflight_rate_specific_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    """Fail before costly data builds when a formal asset is absent or stale."""

    if not args.execute:
        return {}

    report: dict[str, Any] = {}
    missing_arguments: list[str] = []
    for fps in (30, 60):
        for role, prefix in (
            ("Router", "router_ckpt"),
            ("Planner", "planner_ckpt"),
            ("Duration", "duration_ckpt"),
        ):
            argument = f"{prefix}_{fps}"
            raw_path = getattr(args, argument, None)
            if not raw_path:
                missing_arguments.append(f"--{argument}")
                continue

            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Formal {fps} FPS {role} checkpoint does not exist: {path}"
                )

            checkpoint = _load_checkpoint_contract(path)
            # A formal rebuild must never inherit the legacy-baseline escape
            # hatch, even if it happens to be present in the parent shell.
            if checkpoint_declared_fps(checkpoint) is None:
                raise RuntimeError(
                    f"Formal {fps} FPS {role} checkpoint {path} has no FPS "
                    "contract; rebuild the rate-specific asset."
                )
            declared = assert_checkpoint_fps(
                checkpoint,
                role=role,
                runtime_fps=float(fps),
                path=str(path),
            )
            report[f"{role.lower()}_{fps}"] = {
                "path": str(path),
                "sha256": _sha256(path),
                "fps": declared,
            }

    if missing_arguments:
        joined = ", ".join(missing_arguments)
        raise RuntimeError(
            "Formal multi-rate build requires all rate-specific Scheduler "
            f"checkpoints before execution; missing: {joined}"
        )
    return report


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
    parser.add_argument("--split_seed", type=int, default=20260718)
    parser.add_argument("--train_ratio", type=float, default=0.67)
    parser.add_argument("--val_ratio", type=float, default=0.165)
    parser.add_argument("--test_ratio", type=float, default=0.165)
    args = parser.parse_args(argv)

    if args.execute and args.train_v45_v46 and not args.regression_audio:
        raise RuntimeError(
            "--train_v45_v46 requires --regression_audio so route/action metrics "
            "can gate training for both frame-rate branches."
        )

    checkpoint_preflight = _preflight_rate_specific_checkpoints(args)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_config_path = Path(args.base_config).expanduser()
    if not base_config_path.is_absolute():
        base_config_path = ROOT / base_config_path
    base_config_path = base_config_path.resolve()
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    python = sys.executable
    intervals_root = output_root / "canonical_intervals"
    plan: dict[str, Any] = {
        "schema": "dunhuang_multirate_build_plan_v2_source_disjoint",
        "source_dirs": [str(Path(p).resolve()) for p in args.source_dirs],
        "base_config": str(base_config_path),
        "skeleton_contract": skeleton_contract(),
        "checkpoint_preflight": checkpoint_preflight,
        "branches": {},
    }

    for fps in (30, 60):
        name = "canonical30" if fps == 30 else "native60"
        branch = output_root / name
        config = branch / "motion_model.json"
        profile = _write_profile(base, float(fps), config)
        branch_env = dict(env)
        branch_env.update({
            "V46_FPS": str(float(fps)),
            "V46_51_FPS": str(float(fps)),
            "V46_49_RETARGET_FPS": str(float(fps)),
            "V46_53_GROUNDER_CKPT": str(
                branch / "checkpoints" / "dual_branch_grounder.pt"
            ),
        })
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

        cache_root = branch / "retarget_cache"
        split_root = branch / "retarget_cache_split"
        split_command = [
            python, "-m", "data_pipeline.split_sources",
            "--cache_root", str(cache_root),
            "--out_root", str(split_root),
            "--seed", str(args.split_seed),
            "--train_ratio", str(args.train_ratio),
            "--val_ratio", str(args.val_ratio),
            "--test_ratio", str(args.test_ratio),
            "--mode", "hardlink",
            "--allow_unknown_performer_group",
        ]
        if args.overwrite:
            split_command.append("--overwrite")
        commands.append(split_command)

        event_db_root = branch / "event_db"
        semantic_dbs: dict[str, Path] = {}
        for split in ("train", "val", "test"):
            event_db = event_db_root / split
            intervals = intervals_root / f"{split}.json"
            event_command = [
                python, "-m", "events.build_pipeline",
                "--motion_dirs", str(split_root / split),
                "--out_db", str(event_db),
                "--config", str(config),
            ]
            if args.overwrite:
                event_command.append("--overwrite")
            if fps == 30:
                event_command.extend((
                    "--canonical_intervals_out", str(intervals)
                ))
            else:
                event_command.extend((
                    "--canonical_intervals_in", str(intervals)
                ))
            commands.append(event_command)

            semantic_db = event_db / "events_aesd.npz"
            semantic_dbs[split] = semantic_db
            commands.append([
                python, "-m", "events.build_semantics",
                "--db", str(event_db / "events.npz"),
                "--out", str(semantic_db),
                "--json", str(event_db / "events_aesd.report.json"),
            ])

        semantic_db = semantic_dbs["train"]
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
            "source_split": str(split_root / "source_split_manifest.json"),
            "event_dbs": {
                split: str(path) for split, path in semantic_dbs.items()
            },
            "training_event_db": str(semantic_db),
            "scheduler_index": str(scheduler / "event_index.json"),
            "duration_index": str(scheduler / "duration_index.npz"),
            "asset_bundle": str(scheduler / "asset_bundle.json"),
            "commands": commands,
        }
        for command in commands:
            _run(command, execute=args.execute, env=branch_env)

    plan_path = output_root / "multirate_build_manifest.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
