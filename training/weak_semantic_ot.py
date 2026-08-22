"""Label-free semantic OT teacher for unpaired music and motion corpora.

The teacher matches only observable temporal controls to motion kinematics.
It deliberately assigns zero weight to body-region dimensions that cannot be
inferred from music alone and never treats the transport plan as ground truth.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from scheduling.music_event_calibration import phrase_statistics


TEACHER_SCHEMA = "ctsr_weak_sparse_semantic_ot_teacher_v1"

# Motion descriptor dimensions 1:4 describe upper/torso/lower activity.  Music
# alone does not identify which body region should move, so their formal cross-
# modal weights are exactly zero instead of being filled from a dance-theme or
# filename rule.
DEFAULT_COST_WEIGHTS = np.asarray(
    [1.00, 0.00, 0.00, 0.00, 0.55, 0.80, 0.85, 0.70, 0.70, 0.80, 0.60, 0.35],
    dtype=np.float32,
)


def observable_music_target(sequence: np.ndarray, duration_seconds: float) -> np.ndarray:
    """Map observable/proxy music statistics into the 12D motion-control space."""

    values = np.asarray(sequence, dtype=np.float32)
    stats = phrase_statistics(values)
    positive_trend = max(
        float(stats["arousal_trend"]), float(stats["tension_trend"]), 0.0
    )
    negative_trend = max(
        -float(stats["arousal_trend"]), -float(stats["tension_trend"]), 0.0
    )
    return np.asarray(
        [
            np.clip(stats["arousal"], 0.0, 1.0),
            0.5,
            0.5,
            0.5,
            np.clip(stats["tension"], 0.0, 1.0),
            np.clip(stats["calm"], 0.0, 1.0),
            np.clip(0.60 * stats["section_mean"] + 0.40 * stats["beat"], 0.0, 1.0),
            np.clip(positive_trend * 6.0, 0.0, 1.0),
            np.clip(negative_trend * 6.0, 0.0, 1.0),
            np.clip(stats["accent_mean"], 0.0, 1.0),
            np.clip(stats["novelty"], 0.0, 1.0),
            # Match scheduling/build_generation_index.py exactly: motion
            # duration is normalized from 0.8 to 6.0 seconds.
            np.clip((float(duration_seconds) - 0.8) / 5.2, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def weighted_control_cost(
    music_targets: np.ndarray,
    motion_descriptors: np.ndarray,
    weights: np.ndarray = DEFAULT_COST_WEIGHTS,
) -> np.ndarray:
    """Return finite weighted L1 costs without categorical action assumptions."""

    music = np.asarray(music_targets, dtype=np.float32)
    motion = np.asarray(motion_descriptors, dtype=np.float32)
    weight = np.asarray(weights, dtype=np.float32).reshape(-1)
    if music.ndim != 2 or motion.ndim != 2 or music.shape[1] != 12 or motion.shape[1] != 12:
        raise ValueError(
            f"Semantic OT expects music=[P,12], motion=[E,12], got {music.shape}, {motion.shape}"
        )
    if weight.shape != (12,) or np.any(weight < 0.0) or float(weight.sum()) <= 0.0:
        raise ValueError(f"Invalid semantic OT cost weights: {weight}")
    if not np.isfinite(music).all() or not np.isfinite(motion).all():
        raise ValueError("Semantic OT inputs contain NaN/Inf")
    difference = np.abs(music[:, None, :] - motion[None, :, :])
    return (difference * weight[None, None, :]).sum(axis=-1) / float(weight.sum())


def _balanced_event_marginal(groups: Sequence[str]) -> np.ndarray:
    normalized = [str(value or "unknown") for value in groups]
    counts = Counter(normalized)
    weights = np.asarray([1.0 / counts[value] for value in normalized], dtype=np.float64)
    return weights / max(float(weights.sum()), 1.0e-12)


def sparse_sinkhorn_teacher(
    cost: np.ndarray,
    event_groups: Sequence[str],
    *,
    top_k: int = 64,
    epsilon: float = 0.12,
    max_iterations: int = 200,
    tolerance: float = 1.0e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a sparse, group-balanced soft teacher for one unpaired song.

    The returned rows are compatibility distributions.  They are pseudo-
    supervision produced by a declared transport teacher, not paired labels.
    """

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"Semantic OT cost must be [phrases,events], got {matrix.shape}")
    if len(event_groups) != matrix.shape[1]:
        raise ValueError("event_groups must align with Semantic OT event columns")
    if not np.isfinite(matrix).all():
        raise ValueError("Semantic OT cost contains NaN/Inf")
    temperature = float(epsilon)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Semantic OT epsilon must be finite and positive")

    phrase_count, event_count = matrix.shape
    keep = max(2, min(int(top_k), event_count))
    top = np.argpartition(matrix, keep - 1, axis=1)[:, :keep]
    mask = np.zeros_like(matrix, dtype=bool)
    mask[np.arange(phrase_count)[:, None], top] = True
    active = np.flatnonzero(mask.any(axis=0))
    active_cost = matrix[:, active]
    active_mask = mask[:, active]
    shifted = active_cost - active_cost.min(axis=1, keepdims=True)
    kernel = np.exp(-shifted / temperature)
    # A tiny positive floor keeps the balanced scaling problem connected while
    # preserving an effectively sparse teacher.  Rows are sparsified again
    # after the marginal audit.
    kernel = np.where(active_mask, kernel, 1.0e-12)

    source = np.full(phrase_count, 1.0 / phrase_count, dtype=np.float64)
    full_target = _balanced_event_marginal(event_groups)
    target = full_target[active]
    target /= max(float(target.sum()), 1.0e-12)
    left = np.ones_like(source)
    right = np.ones_like(target)
    converged = False
    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        left = source / np.maximum(kernel @ right, 1.0e-18)
        right = target / np.maximum(kernel.T @ left, 1.0e-18)
        if iterations == 1 or iterations % 10 == 0:
            transport_probe = left[:, None] * kernel * right[None, :]
            row_error = float(np.max(np.abs(transport_probe.sum(axis=1) - source)))
            column_error = float(np.max(np.abs(transport_probe.sum(axis=0) - target)))
            if max(row_error, column_error) <= float(tolerance):
                converged = True
                break

    transport = left[:, None] * kernel * right[None, :]
    row_error = float(np.max(np.abs(transport.sum(axis=1) - source)))
    column_error = float(np.max(np.abs(transport.sum(axis=0) - target)))
    probabilities_active = transport / np.maximum(
        transport.sum(axis=1, keepdims=True), 1.0e-18
    )
    probabilities_active = np.where(active_mask, probabilities_active, 0.0)
    probabilities_active /= np.maximum(
        probabilities_active.sum(axis=1, keepdims=True), 1.0e-18
    )
    probabilities = np.zeros_like(matrix, dtype=np.float64)
    probabilities[:, active] = probabilities_active
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1.0e-12)), axis=1
    ) / np.log(float(event_count))
    report = {
        "schema": TEACHER_SCHEMA,
        "supervision_source": "semantic_ot_teacher",
        "is_ground_truth": False,
        "paired_audio_motion": False,
        "num_phrases": int(phrase_count),
        "num_events": int(event_count),
        "active_events": int(len(active)),
        "top_k": int(keep),
        "epsilon": temperature,
        "iterations": int(iterations),
        "converged": bool(converged),
        "row_marginal_error": row_error,
        "column_marginal_error": column_error,
        "mean_teacher_entropy": float(np.mean(entropy)),
        "cost_weights": DEFAULT_COST_WEIGHTS.tolist(),
        "body_region_music_weights_zero": True,
    }
    return probabilities.astype(np.float32), report
