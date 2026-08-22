"""Shared sequence contract for the formal CTSR-Weak music Router.

Training, Planner construction, and fresh-audio inference must resample the
same structure-derived phrase windows.  Keeping this operation in one module
prevents the historical fixed-three-slot training/inference mismatch.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


TEMPORAL_ROUTER_ARCHITECTURE = "ctsr_weak_temporal_v1"
TEMPORAL_ROUTER_SEQUENCE_SCHEMA = "librosa_12d_phrase_sequence_v1"
TEMPORAL_ROUTER_SUPERVISION_SOURCE = "semantic_ot_teacher"
FORMAL_PLANNER_CONTRACT = {
    "schema": "ctsr_continuous_whole_song_planner_v2",
    "router_architecture": TEMPORAL_ROUTER_ARCHITECTURE,
    "categorical_event_head": False,
    "song_disjoint_validation": True,
    "is_ground_truth": False,
}


def resample_feature_sequence(
    features: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    """Resample a finite ``[T,12]`` Librosa stream without collapsing time."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 12 or len(values) < 1:
        raise ValueError(f"Expected non-empty [T,12] music features, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Music phrase sequence contains NaN/Inf")
    count = int(target_frames)
    if count < 2:
        raise ValueError("Temporal Router target_frames must be at least 2")
    if len(values) == count:
        return values.copy()
    if len(values) == 1:
        return np.repeat(values, count, axis=0).astype(np.float32)
    source_time = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return np.stack(
        [np.interp(target_time, source_time, values[:, dim]) for dim in range(12)],
        axis=-1,
    ).astype(np.float32)


def phrase_feature_sequences(
    whole_song_features: np.ndarray,
    phrases: Sequence[Any],
    target_frames: int,
) -> np.ndarray:
    """Return sequence tensors for the exact final phrase/event-slot bounds."""

    features = np.asarray(whole_song_features, dtype=np.float32)
    rows: list[np.ndarray] = []
    for phrase in phrases:
        start = int(phrase.start)
        end = int(phrase.end)
        if start < 0 or end > len(features) or end <= start:
            raise ValueError(
                f"Invalid phrase bounds [{start}, {end}) for {len(features)} frames"
            )
        rows.append(resample_feature_sequence(features[start:end], target_frames))
    if not rows:
        return np.zeros((0, int(target_frames), 12), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def scientific_supervision_contract() -> dict[str, Any]:
    """Machine-readable evidence boundary stored in every formal checkpoint."""

    return {
        "architecture": TEMPORAL_ROUTER_ARCHITECTURE,
        "sequence_schema": TEMPORAL_ROUTER_SEQUENCE_SCHEMA,
        "supervision_source": TEMPORAL_ROUTER_SUPERVISION_SOURCE,
        "is_ground_truth": False,
        "paired_audio_motion": False,
        "human_training_labels": 0,
        "external_pretrained_model": False,
        "cultural_semantics_claimed": False,
        "observable_music_controls_are_proxies": True,
    }


def assert_formal_router_scientific_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when a formal asset is legacy, paired, or externally pretrained."""

    contract = checkpoint.get("scientific_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Formal Router checkpoint has no scientific_contract")
    expected = scientific_supervision_contract()
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"Formal Router scientific contract mismatch for {key}: "
                f"expected={value!r}, actual={contract.get(key)!r}"
            )
    return dict(contract)


def assert_formal_planner_scientific_contract(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Reject a Planner outside the current continuous-control protocol."""

    contract = checkpoint.get("formal_planner_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Formal Planner checkpoint has no formal_planner_contract")
    for key, value in FORMAL_PLANNER_CONTRACT.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"Formal Planner contract mismatch for {key}: "
                f"expected={value!r}, actual={contract.get(key)!r}"
            )
    return dict(contract)
