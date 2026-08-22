"""Candidate retrieval primitives used by whole-song schedulers."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


LOCAL_ACTION_LABELS = (
    "pose_hold",
    "locomotion",
    "turn_spin",
    "jump_aerial",
    "floorwork",
    "upper_body_gesture",
    "rhythmic_accent",
    "transition",
    "unknown",
)


def aggregate_action_compatibility(
    event_probabilities: np.ndarray,
    items: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Aggregate event probabilities into motion-derived local-action probabilities."""

    probabilities = np.asarray(event_probabilities, dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(items):
        raise ValueError(
            f"Event probabilities/items mismatch: {probabilities.shape}, {len(items)}"
        )
    action_matrix = np.zeros((len(items), len(LOCAL_ACTION_LABELS)), dtype=np.float32)
    for event_index, item in enumerate(items):
        scores = item.get("local_action_scores")
        if isinstance(scores, dict):
            for label, raw_score in scores.items():
                if str(label) not in LOCAL_ACTION_LABELS:
                    continue
                try:
                    score = max(0.0, float(raw_score))
                except (TypeError, ValueError):
                    continue
                action_matrix[event_index, LOCAL_ACTION_LABELS.index(str(label))] += score
        if float(action_matrix[event_index].sum()) <= 0.0:
            labels = item.get("local_action_labels", ["unknown"])
            if not isinstance(labels, (list, tuple)):
                labels = ["unknown"]
            valid = [str(label) for label in labels if str(label) in LOCAL_ACTION_LABELS]
            if not valid:
                valid = ["unknown"]
            for label in valid:
                action_matrix[event_index, LOCAL_ACTION_LABELS.index(label)] = 1.0
        action_matrix[event_index] /= max(float(action_matrix[event_index].sum()), 1.0e-12)
    action_probability = probabilities @ action_matrix
    action_probability /= np.maximum(
        action_probability.sum(axis=1, keepdims=True), 1.0e-12
    )
    return action_probability.astype(np.float32)


def _softmax_rows(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max(float(temperature), 1.0e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    probability = np.exp(scaled)
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1.0e-12)
    return probability.astype(np.float32)


def precompute_music_routing(
    router,
    queries: Sequence[np.ndarray],
    motion_desc: np.ndarray,
    device: torch.device,
    *,
    phrase_sequences: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return similarities plus auditable weak probabilities/uncertainty."""

    query_matrix = np.stack(queries).astype(np.float32)
    descriptors = np.asarray(motion_desc, dtype=np.float32)
    if router is None:
        raise RuntimeError("Formal scheduling requires a trained CTSR-Weak Router")
    architecture = str(getattr(router, "architecture", ""))
    supervision_source = str(getattr(router, "supervision_source", ""))
    if architecture != "ctsr_weak_temporal_v1":
        raise RuntimeError(
            f"Formal scheduling rejects Router architecture={architecture!r}"
        )
    if supervision_source != "semantic_ot_teacher":
        raise RuntimeError(
            f"Formal scheduling rejects Router supervision={supervision_source!r}"
        )
    if phrase_sequences is None:
        raise RuntimeError(
            "CTSR-Weak temporal inference requires complete phrase sequences; "
            "an aggregated 12D query is not an accepted formal input"
        )
    music_values = np.asarray(phrase_sequences, dtype=np.float32)
    if (
        music_values.ndim != 3
        or music_values.shape[0] != len(query_matrix)
        or music_values.shape[2] != 12
    ):
        raise RuntimeError(f"Invalid CTSR-Weak phrase sequences: {music_values.shape}")

    with torch.no_grad():
        query_tensor = torch.from_numpy(music_values).to(device)
        descriptor_tensor = torch.from_numpy(descriptors).to(device)
        query_embedding = router.encode_music(query_tensor)
        motion_embedding = router.encode_motion(descriptor_tensor)
        similarity_tensor = query_embedding @ motion_embedding.t()
    similarity = similarity_tensor.detach().cpu().numpy().astype(np.float32)
    temperature = float(getattr(router, "inference_temperature", 0.12))
    feature_mean = getattr(router, "feature_mean", None)
    feature_std = getattr(router, "feature_std", None)
    if feature_mean and feature_std:
        phrase_mean = music_values.mean(axis=1)
        mean = np.asarray(feature_mean, dtype=np.float32).reshape(1, 12)
        std = np.maximum(
            np.asarray(feature_std, dtype=np.float32).reshape(1, 12), 1.0e-5
        )
        excess = np.maximum(np.abs((phrase_mean - mean) / std) - 2.0, 0.0)
        ood = (1.0 - np.exp(-excess.mean(axis=1))).astype(np.float32)
    else:
        ood = np.zeros((len(query_matrix),), dtype=np.float32)

    probabilities = _softmax_rows(similarity, temperature)
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1.0e-12)), axis=1
    ) / np.log(float(max(2, probabilities.shape[1])))
    return {
        "similarity": similarity,
        "probabilities": probabilities,
        "entropy": entropy.astype(np.float32),
        "confidence": (1.0 - entropy).astype(np.float32),
        "ood": ood,
        "architecture": architecture,
        "supervision_source": supervision_source,
        "is_ground_truth": False,
        "inference_temperature": temperature,
    }
