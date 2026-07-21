"""Fail-closed contracts shared by trainable Scheduler models.

Router, Duration and Planner checkpoints are only comparable when they were
trained against the same ordered Generation Event-DB and frame rate.  This
module keeps that provenance next to the model weights instead of relying on a
filename or a mutable environment variable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    normalize_rot6d_layout,
)
from motion_geometry.smpl24 import skeleton_contract
from support.checkpoint_contracts import assert_checkpoint_fps
from support.event_identity import (
    assert_same_event_db_contract,
    normalize_event_db_contract,
)


SCHEDULER_CHECKPOINT_CONTRACT_SCHEMA = "dunhuang_scheduler_checkpoint_contract_v1"
SCHEDULER_ROLES = {"router", "duration", "planner"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_scheduler_dataset_contract(
    dataset: str | Path,
    *,
    role: str,
    fps: float,
    event_db_contract: Mapping[str, Any],
) -> None:
    """Reject stale training data before a checkpoint can be produced."""

    path = Path(dataset).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role} training dataset: {path}")
    with np.load(path, allow_pickle=True) as payload:
        if "fps" not in payload.files:
            raise RuntimeError(f"{role} training dataset has no FPS contract: {path}")
        dataset_fps = float(np.asarray(payload["fps"]).item())
        if abs(dataset_fps - float(fps)) > 1.0e-6:
            raise RuntimeError(
                f"{role} training dataset FPS mismatch: dataset={dataset_fps}, "
                f"requested={fps}, path={path}"
            )
        if "event_db_contract_json" not in payload.files:
            raise RuntimeError(
                f"{role} training dataset has no Event-DB identity contract: {path}"
            )
        raw_contract = np.asarray(payload["event_db_contract_json"]).item()
    try:
        dataset_contract = json.loads(str(raw_contract))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{role} training dataset has malformed Event-DB contract: {path}"
        ) from exc
    assert_same_event_db_contract(
        event_db_contract,
        dataset_contract,
        context=f"{role} training dataset/Generation index",
    )


def scheduler_training_contract(
    *,
    role: str,
    fps: float,
    index_metadata: Mapping[str, Any],
    index_json: str | Path,
    index_npz: str | Path,
    dataset: str | Path,
    model_rot6d_layout: str | None = None,
    music_prior: str | Path | None = None,
    upstream_checkpoints: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Create the immutable training provenance stored in a checkpoint."""

    role = str(role).strip().lower()
    if role not in SCHEDULER_ROLES:
        raise ValueError(f"Unsupported Scheduler role: {role!r}")
    rate = float(fps)
    rates = [float(value) for value in index_metadata.get("canonical_fps_values", [])]
    if rates != [rate]:
        raise RuntimeError(
            f"{role} training index FPS mismatch: index={rates}, requested={[rate]}"
        )
    event_contract = normalize_event_db_contract(index_metadata.get("event_db_contract"))
    if event_contract is None:
        raise RuntimeError(f"{role} training index has no Event-DB identity contract")
    declared_skeleton = index_metadata.get("skeleton_contract")
    runtime_skeleton = skeleton_contract()
    if not isinstance(declared_skeleton, Mapping):
        raise RuntimeError(f"{role} training index has no SMPL24 skeleton contract")
    if str(declared_skeleton.get("sha256", "")) != str(runtime_skeleton["sha256"]):
        raise RuntimeError(
            f"{role} training skeleton mismatch: index={declared_skeleton.get('sha256')!r}, "
            f"runtime={runtime_skeleton['sha256']!r}"
        )

    layout: str | None
    if role == "duration":
        layout = normalize_rot6d_layout(
            model_rot6d_layout or ROT6D_LAYOUT_PYTORCH3D_ROW
        )
        if layout != ROT6D_LAYOUT_PYTORCH3D_ROW:
            raise RuntimeError(
                "DurationPredictor must be trained in its native pytorch3d_row layout"
            )
    else:
        layout = None

    dataset_path = Path(dataset).resolve()
    index_json_path = Path(index_json).resolve()
    index_npz_path = Path(index_npz).resolve()
    for label, path in (
        ("training dataset", dataset_path),
        ("Scheduler index JSON", index_json_path),
        ("Scheduler index NPZ", index_npz_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    assert_scheduler_dataset_contract(
        dataset_path,
        role=role,
        fps=rate,
        event_db_contract=event_contract,
    )

    prior_path = Path(music_prior).resolve() if music_prior else None
    if prior_path is not None and not prior_path.is_file():
        raise FileNotFoundError(f"Missing music prior: {prior_path}")
    upstream: dict[str, dict[str, str]] = {}
    for name, raw_path in sorted((upstream_checkpoints or {}).items()):
        checkpoint_path = Path(raw_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing upstream {name} checkpoint for {role}: {checkpoint_path}"
            )
        upstream[str(name)] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        }

    return {
        "schema": SCHEDULER_CHECKPOINT_CONTRACT_SCHEMA,
        "role": role,
        "fps": rate,
        "event_db_contract": event_contract,
        "skeleton_schema": str(runtime_skeleton["schema"]),
        "skeleton_sha256": str(runtime_skeleton["sha256"]),
        "index_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
        "model_rot6d_layout": layout,
        "index_json": str(index_json_path),
        "index_json_sha256": sha256_file(index_json_path),
        "index_npz": str(index_npz_path),
        "index_npz_sha256": sha256_file(index_npz_path),
        "training_dataset": str(dataset_path),
        "training_dataset_sha256": sha256_file(dataset_path),
        "music_prior": str(prior_path) if prior_path is not None else None,
        "music_prior_sha256": sha256_file(prior_path) if prior_path is not None else None,
        "upstream_checkpoints": upstream,
    }


def assert_scheduler_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    role: str,
    runtime_fps: float,
    event_db_contract: Mapping[str, Any] | None = None,
    index_json: str | Path | None = None,
    index_npz: str | Path | None = None,
    path: str = "",
    allow_legacy_30fps: bool | None = None,
) -> dict[str, Any] | None:
    """Validate a formal checkpoint, retaining an explicit legacy parity path."""

    role = str(role).strip().lower()
    if role not in SCHEDULER_ROLES:
        raise ValueError(f"Unsupported Scheduler role: {role!r}")
    assert_checkpoint_fps(
        checkpoint,
        role=role.capitalize(),
        runtime_fps=float(runtime_fps),
        path=path,
    )
    contract = checkpoint.get("scheduler_contract")
    if contract is None:
        legacy = (
            bool(allow_legacy_30fps)
            if allow_legacy_30fps is not None
            else os.environ.get("DUNHUANG_ALLOW_LEGACY_30FPS_CHECKPOINTS", "0") == "1"
        )
        if legacy and abs(float(runtime_fps) - 30.0) <= 1.0e-6:
            return None
        raise RuntimeError(
            f"{role} checkpoint {path} has no formal Scheduler training contract"
        )
    if not isinstance(contract, Mapping):
        raise RuntimeError(f"{role} checkpoint {path} has a malformed Scheduler contract")
    if str(contract.get("schema")) != SCHEDULER_CHECKPOINT_CONTRACT_SCHEMA:
        raise RuntimeError(
            f"{role} checkpoint {path} has unsupported contract schema: "
            f"{contract.get('schema')!r}"
        )
    if str(contract.get("role")) != role:
        raise RuntimeError(
            f"Scheduler checkpoint role mismatch: expected={role!r}, "
            f"actual={contract.get('role')!r}, path={path}"
        )
    if abs(float(contract.get("fps", -1.0)) - float(runtime_fps)) > 1.0e-6:
        raise RuntimeError(
            f"{role} Scheduler contract FPS mismatch: contract={contract.get('fps')}, "
            f"runtime={runtime_fps}, path={path}"
        )
    runtime_skeleton = skeleton_contract()
    if str(contract.get("skeleton_sha256", "")) != str(runtime_skeleton["sha256"]):
        raise RuntimeError(f"{role} checkpoint SMPL24 skeleton contract mismatch: {path}")
    if role == "duration":
        layout = normalize_rot6d_layout(str(contract.get("model_rot6d_layout", "")))
        if layout != ROT6D_LAYOUT_PYTORCH3D_ROW:
            raise RuntimeError(
                f"Duration checkpoint must declare pytorch3d_row model layout: {path}"
            )
    if event_db_contract is not None:
        assert_same_event_db_contract(
            event_db_contract,
            contract.get("event_db_contract"),
            context=f"{role} checkpoint/Generation index",
        )
    for label, runtime_path, hash_key in (
        ("index JSON", index_json, "index_json_sha256"),
        ("index NPZ", index_npz, "index_npz_sha256"),
    ):
        if runtime_path is None:
            continue
        expected_hash = str(contract.get(hash_key, ""))
        if not expected_hash:
            raise RuntimeError(
                f"{role} checkpoint {path} has no {label} content hash"
            )
        actual_hash = sha256_file(runtime_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"{role} checkpoint {label} hash mismatch: "
                f"checkpoint={expected_hash}, runtime={actual_hash}, path={path}"
            )
    return dict(contract)
