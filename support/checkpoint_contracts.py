"""Shared fail-closed contracts for frame-rate-specific checkpoints."""
from __future__ import annotations

import math
from typing import Any, Mapping


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
) -> float:
    """Validate a checkpoint's physical sampling-rate contract.

    Missing metadata is rejected for every current-protocol experiment.
    """
    runtime = float(runtime_fps)
    if not math.isfinite(runtime) or runtime <= 0.0:
        raise ValueError(f"runtime_fps must be finite and positive, got {runtime_fps!r}")
    declared = checkpoint_declared_fps(checkpoint)
    label = f"{role} checkpoint" + (f" {path}" if path else "")
    if declared is None:
        raise RuntimeError(
            f"{label} has no FPS contract. Rebuild the rate-specific asset."
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
