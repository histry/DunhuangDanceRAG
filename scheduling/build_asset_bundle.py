#!/usr/bin/env python3
"""Create an auditable Router/Planner/Duration asset bundle manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from motion_geometry.rotations import (
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    normalize_rot6d_layout,
)
from scheduling.index_io import load_shared_index
from support.scheduler_checkpoint_contracts import (
    assert_scheduler_checkpoint_contract,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_summary(
    path: Path,
    role: str,
    expected_fps: float,
    event_db_contract: dict,
    index_json: Path,
    index_npz: Path,
) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    state = raw.get("state_dict", raw.get("model_state_dict", {}))
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint has no model state: {path}")
    config = raw.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError(f"Checkpoint config is not a mapping: {path}")
    contract = assert_scheduler_checkpoint_contract(
        raw,
        role=role,
        runtime_fps=expected_fps,
        event_db_contract=event_db_contract,
        index_json=index_json,
        index_npz=index_npz,
        path=str(path),
        allow_legacy_30fps=False,
    )
    if contract is None:  # pragma: no cover - prohibited above
        raise RuntimeError(f"Formal Scheduler contract missing: {path}")
    declared_fps = float(contract["fps"])
    declared_layout = raw.get(
        "rot6d_layout",
        raw.get("config", {}).get("rot6d_layout"),
    )
    if role == "duration":
        if declared_layout is None:
            raise RuntimeError(
                f"Duration checkpoint has no explicit Rot6D layout contract: {path}"
            )
        effective_layout = normalize_rot6d_layout(str(declared_layout))
        if effective_layout != ROT6D_LAYOUT_PYTORCH3D_ROW:
            raise RuntimeError(
                f"Duration checkpoint top-level Rot6D layout is incompatible: "
                f"{declared_layout!r}, path={path}"
            )
        if effective_layout != str(contract.get("model_rot6d_layout")):
            raise RuntimeError(
                f"Duration checkpoint Rot6D layout disagrees with its Scheduler "
                f"contract: top_level={effective_layout!r}, "
                f"contract={contract.get('model_rot6d_layout')!r}, path={path}"
            )
        rotation_policy = (
            "checkpoint must explicitly declare pytorch3d_row; scheduler "
            "adapts it at the canonical-column boundary"
        )
    else:
        effective_layout = "not_applicable"
        rotation_policy = "descriptor/state model does not decode Rot6D"
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "version": str(raw.get("version", "unknown")),
        "fps": float(declared_fps),
        "declared_rot6d_layout": declared_layout,
        "effective_rot6d_layout": effective_layout,
        "rotation_contract_policy": rotation_policy,
        "num_state_tensors": len(state),
        "scheduler_contract": contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--index_npz", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--planner_ckpt", required=True)
    parser.add_argument("--duration_ckpt", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    metadata, arrays, _items = load_shared_index(args.index_json, args.index_npz)
    arrays.close()
    index_rates = [float(value) for value in metadata.get("canonical_fps_values", [])]
    if index_rates != [float(args.fps)]:
        raise RuntimeError(
            f"Scheduler index FPS mismatch: index={index_rates}, requested={[float(args.fps)]}"
        )
    paths = {
        "router": Path(args.router_ckpt),
        "planner": Path(args.planner_ckpt),
        "duration": Path(args.duration_ckpt),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
    report = {
        "schema": "scheduler_asset_bundle_v1",
        "ok": True,
        "fps": float(args.fps),
        "skeleton_contract": metadata["skeleton_contract"],
        "event_db_contract": metadata["event_db_contract"],
        "index_json": str(Path(args.index_json).resolve()),
        "index_npz": str(Path(args.index_npz).resolve()),
        "index_json_sha256": sha256_file(Path(args.index_json)),
        "index_npz_sha256": sha256_file(Path(args.index_npz)),
        "checkpoints": {
            name: checkpoint_summary(
                path,
                name,
                float(args.fps),
                metadata["event_db_contract"],
                Path(args.index_json),
                Path(args.index_npz),
            )
            for name, path in paths.items()
        },
        "asset_bundle_rebuilt": True,
        "checkpoint_policy": (
            "Router/Duration/Planner are trained from the current ordered "
            "Generation Event-DB. Only the historical Router music encoder "
            "may be imported as a frozen semantic prior. A previously produced "
            "formal checkpoint is accepted only when all content hashes match."
        ),
        "required_lifecycle": [
            "build_generation_aligned_scheduler_index",
            "train_router_from_music_encoder_prior",
            "train_duration_with_explicit_native_rot6d_layout",
            "train_whole_song_planner",
            "validate_scheduler_asset_bundle",
            "same_wav_no_training_regression",
            "train_v45",
            "train_v46",
        ],
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
