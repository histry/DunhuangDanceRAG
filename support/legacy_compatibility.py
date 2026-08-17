"""Read-only translation for artifacts created before semantic renaming.

New code must emit only the semantic identifiers defined on the right-hand
side of these maps.  Historical tokens remain isolated here so old schedules
and checkpoints can still be inspected or reused without leaking release
numbers back into the runtime API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


_MOTION_CHECKPOINT_ROLES = {
    "v44_contrastive": "semantic_retriever",
    "v45_refiner": "boundary_refiner",
    "v46_diffusion": "motion_diffusion",
}

_MOTION_CHECKPOINT_VERSION_PREFIXES = {
    "v45_product_manifold_79d": "product_manifold_boundary_refiner",
    "v45_refiner": "boundary_refiner",
    "v46_reference_tangent_diffusion_79d": "reference_tangent_motion_diffusion",
    "v46_conditional_residual_diffusion": "conditional_residual_motion_diffusion",
}

_TRANSITION_ARCHITECTURES = {
    "v32_continuous_c2_contact_inr_latent_diffusion": (
        "contact_transition_continuous_c2_contact_inr_latent_diffusion"
    ),
    "v34_continuous_c3_contact_inr_latent_diffusion": (
        "boundary_transition_continuous_c3_contact_inr_latent_diffusion"
    ),
    "v32_c2_baseline": "contact_transition_c2_baseline",
}

_HISTORICAL_SCHEDULER_TOKENS = ("v21", "v23", "v26")
_SEMANTIC_SCHEDULER_TOKENS = (
    "music_router",
    "duration_model",
    "whole_song_planner",
    "router",
    "planner",
    "pretrained",
)


def normalize_motion_checkpoint_role(value: Any) -> str:
    role = str(value)
    return _MOTION_CHECKPOINT_ROLES.get(role, role)


def normalize_motion_checkpoint_version(value: Any) -> str:
    version = str(value)
    for old_prefix, semantic_prefix in _MOTION_CHECKPOINT_VERSION_PREFIXES.items():
        if version.startswith(old_prefix):
            return semantic_prefix + version[len(old_prefix) :]
    return version


def normalize_transition_architecture(value: Any) -> str:
    architecture = str(value)
    return _TRANSITION_ARCHITECTURES.get(architecture, architecture)


def has_scheduler_provenance(value: Any) -> bool:
    source = str(value).lower()
    tokens = _SEMANTIC_SCHEDULER_TOKENS + _HISTORICAL_SCHEDULER_TOKENS
    return any(token in source for token in tokens)


def historical_whole_song_summary(schedule_dir: str | Path) -> Path:
    """Return the former scheduler summary path for read-only discovery."""
    return Path(schedule_dir) / "V26_WHOLE_SONG_SUMMARY.json"


def historical_whole_song_reports(
    schedule_dir: str | Path,
    audio_stem: str,
) -> tuple[Path, ...]:
    """Return former report paths without making them part of the public API."""
    root = Path(schedule_dir)
    direct = root / f"{audio_stem}_v26.schedule_report.json"
    others = tuple(sorted(root.glob("*_v26.schedule_report.json")))
    return (direct, *tuple(path for path in others if path != direct))
