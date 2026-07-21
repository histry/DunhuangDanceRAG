"""Shared fail-closed contracts for frame-rate-specific checkpoints."""
from __future__ import annotations

import math
import os
from typing import Any, Mapping


LEGACY_30FPS_ENV = "DUNHUANG_ALLOW_LEGACY_30FPS_CHECKPOINTS"


def checkpoint_declared_fps(checkpoint: Mapping[str, Any]) -> Any:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint is not a mapping")
    config = checkpoint.get("config", {})
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise RuntimeError("Checkpoint config is not a mapping")
    return config.get("fps", checkpoint.get("fps"))


def assert_checkpoint_fps(
    checkpoint: Mapping[str, Any],
    *,
    role: str,
    runtime_fps: float,
    path: str = "",
    legacy_env: str = LEGACY_30FPS_ENV,
) -> float:
    """Validate a checkpoint's physical sampling-rate contract.

    Missing metadata is accepted only for an explicitly requested, read-only
    30 FPS legacy parity run. Formal 30/60 FPS experiments must declare FPS.
    """
    runtime = float(runtime_fps)
    if not math.isfinite(runtime) or runtime <= 0.0:
        raise ValueError(f"runtime_fps must be finite and positive, got {runtime_fps!r}")
    declared = checkpoint_declared_fps(checkpoint)
    label = f"{role} checkpoint" + (f" {path}" if path else "")
    if declared is None:
        legacy_ok = (
            abs(runtime - 30.0) <= 1.0e-6
            and os.environ.get(legacy_env, "0") == "1"
        )
        if legacy_ok:
            return 30.0
        raise RuntimeError(
            f"{label} has no FPS contract. Rebuild the rate-specific asset. "
            f"For a read-only 30 FPS parity baseline only, set {legacy_env}=1."
        )
    try:
        value = float(declared)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} has invalid FPS metadata: {declared!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{label} has invalid FPS metadata: {declared!r}")
    if abs(value - runtime) > 1.0e-6:
        raise RuntimeError(
            f"{label} FPS mismatch: checkpoint={value}, runtime={runtime}"
        )
    return value
