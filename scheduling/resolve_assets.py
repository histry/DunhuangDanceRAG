#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve the trained V21/V26/V23 scheduling assets reproducibly.

Explicit environment variables always win.  Otherwise the resolver checks the
known validated project paths in a fixed priority order.  It never selects an
arbitrary newest checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]


CANDIDATES = {
    "index_json": [
        "data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json",
        "output/V21_BEST_SHARED_MULTIMUSIC_BASELINE/reproducibility/v21_shared_event_index.json",
        "data/v21_shared_event_index.json",
    ],
    "duration_index_npz": [
        "data/v26_music_dominant_duration_index.npz",
        "data/v26_duration_augmented_event_index.npz",
    ],
    "router_ckpt": [
        "output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt",
    ],
    "planner_ckpt": [
        "output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt",
    ],
    "v23_ckpt": [
        "checkpoints/v23_release/v23_v2_5/v23_v2_5_seed20260610_best.pt",
        "output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt",
    ],
    "hierarchy_index_npz": [
        "data/v26_hierarchy_index.npz",
        "data/v26_hierarchical_event_index.npz",
    ],
    "start_pose": [
        "data/canonical_dunhuang_start_pose.npy",
    ],
}


ENV_NAMES = {
    "index_json": "V46_51_INDEX_JSON",
    "duration_index_npz": "V46_51_DURATION_INDEX_NPZ",
    "router_ckpt": "V46_51_ROUTER_CKPT",
    "planner_ckpt": "V46_51_PLANNER_CKPT",
    "v23_ckpt": "V46_51_V23_CKPT",
    "hierarchy_index_npz": "V46_51_HIERARCHY_INDEX_NPZ",
    "start_pose": "V46_51_START_POSE",
}


REQUIRED = {
    "index_json",
    "duration_index_npz",
    "router_ckpt",
    "planner_ckpt",
    "v23_ckpt",
}


def _resolve_pointer_file(path: Path) -> Optional[Path]:
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip().splitlines()[0]
    except Exception:
        return None
    if not value:
        return None
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    return target.resolve() if target.is_file() else None


def resolve_one(key: str) -> Tuple[str, str]:
    env_name = ENV_NAMES[key]
    explicit = os.environ.get(env_name, "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            return str(p.resolve()), f"environment:{env_name}"
        if key in REQUIRED:
            raise FileNotFoundError(
                f"{env_name} points to a missing file: {p}"
            )
        return "", f"optional_environment_missing:{env_name}"

    pointer_candidates = []
    if key == "router_ckpt":
        pointer_candidates = [
            ROOT
            / "output/v21_music_router_985songs_20260605_154801/BEST_ROUTER_CKPT.txt",
        ]
    elif key == "planner_ckpt":
        pointer_candidates = [
            ROOT
            / "output/v26_music_dominant_whole_song_planner_985/BEST_V26_PLANNER_CKPT.txt",
        ]

    for pointer in pointer_candidates:
        target = _resolve_pointer_file(pointer)
        if target is not None:
            return str(target), f"pointer:{pointer}"

    for rel in CANDIDATES[key]:
        p = ROOT / rel
        if p.is_file():
            return str(p.resolve()), f"validated_candidate:{rel}"

    if key in REQUIRED:
        raise FileNotFoundError(
            f"Cannot resolve required scheduler asset {key}. "
            f"Set {env_name}. Checked: {CANDIDATES[key]}"
        )
    return "", "optional_not_found"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_env", required=True)
    args = ap.parse_args(argv)

    assets: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    for key in ENV_NAMES:
        value, source = resolve_one(key)
        assets[key] = value
        sources[key] = source

    report = {
        "schema": "v46_51_scheduler_asset_resolution",
        "root": str(ROOT),
        "assets": assets,
        "resolution_sources": sources,
        "required_assets": sorted(REQUIRED),
        "optional_assets": sorted(set(ENV_NAMES) - REQUIRED),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shell_names = {
        "index_json": "V46_51_RESOLVED_INDEX_JSON",
        "duration_index_npz": "V46_51_RESOLVED_DURATION_INDEX_NPZ",
        "router_ckpt": "V46_51_RESOLVED_ROUTER_CKPT",
        "planner_ckpt": "V46_51_RESOLVED_PLANNER_CKPT",
        "v23_ckpt": "V46_51_RESOLVED_V23_CKPT",
        "hierarchy_index_npz": "V46_51_RESOLVED_HIERARCHY_INDEX_NPZ",
        "start_pose": "V46_51_RESOLVED_START_POSE",
    }
    lines = [
        "# Generated by scheduling/resolve_assets.py",
    ]
    for key, shell_name in shell_names.items():
        lines.append(
            f"export {shell_name}={shlex.quote(assets[key])}"
        )
    out_env = Path(args.out_env)
    out_env.parent.mkdir(parents=True, exist_ok=True)
    out_env.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
