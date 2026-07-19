#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hierarchical Event-RAG features and graph scheduling costs for V26/V27.

This module keeps the existing whole-song scheduler intact, but upgrades its
candidate scoring from flat feature matching to:

1. hierarchical retrieval: match coarse body-state first, then local event
   details in a small Poincare-ball style embedding;
2. graph scheduling: treat candidate events as nodes and transition feasibility
   as edge cost, so beam search becomes explicit graph-path inference.

The implementation is deliberately deterministic and index-compatible.  If a
prebuilt hierarchy index is provided, it is used; otherwise features are derived
from the existing V21/V26 duration index arrays.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


EVENT_GROUPS = {
    "pose_hold": 0,
    "calm_flow": 1,
    "release": 1,
    "neutral_flow": 2,
    "build_up": 3,
    "high_tension": 3,
    "arm_flourish": 4,
    "support_shift": 5,
}

MUSIC_TO_GROUP = {
    "calm_flow": 1,
    "release": 1,
    "neutral_flow": 2,
    "build_up": 3,
    "climax": 3,
    "accent": 4,
    "section_change": 5,
}


def _array_names(arrays: Any) -> set[str]:
    return set(arrays.files) if hasattr(arrays, "files") else set(arrays.keys())


def _get_array(arrays: Any, name: str, default: np.ndarray) -> np.ndarray:
    names = _array_names(arrays)
    if name in names:
        return np.asarray(arrays[name])
    return np.asarray(default)


def _normalize01(x: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if lo is None:
        lo = float(np.nanpercentile(arr, 5)) if arr.size else 0.0
    if hi is None:
        hi = float(np.nanpercentile(arr, 95)) if arr.size else 1.0
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _event_group(event_type: str) -> int:
    return int(EVENT_GROUPS.get(str(event_type), 2))


def _one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    out = np.zeros((len(indices), depth), dtype=np.float32)
    out[np.arange(len(indices)), np.clip(indices.astype(np.int64), 0, depth - 1)] = 1.0
    return out


def _l2_unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def expmap0(tangent: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    """Map Euclidean tangent vectors at the origin into a Poincare ball.

    exp_0^c(v) = tanh(sqrt(c)||v||) v / (sqrt(c)||v||)
    """
    c = max(float(curvature), 1e-6)
    sqrt_c = float(np.sqrt(c))
    tangent = np.asarray(tangent, dtype=np.float32)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-8)
    scale = np.tanh(sqrt_c * norms_safe) / (sqrt_c * norms_safe)
    mapped = scale * tangent
    max_norm = (1.0 - 1e-5) / sqrt_c
    mapped_norms = np.linalg.norm(mapped, axis=1, keepdims=True)
    shrink = np.minimum(1.0, max_norm / np.maximum(mapped_norms, 1e-8))
    return (mapped * shrink).astype(np.float32)


def tangent_to_poincare(
    raw_vectors: np.ndarray,
    radius_ratio: np.ndarray,
    curvature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project Euclidean hierarchy features through tangent space.

    ``radius_ratio`` is in [0, 1).  Small radii encode coarse/global body states
    near the origin; large radii encode specific local gestures near the ball
    boundary.  This keeps the hierarchy interpretation explicit instead of only
    rescaling Euclidean vectors inside a unit ball.
    """
    c = max(float(curvature), 1e-6)
    sqrt_c = float(np.sqrt(c))
    unit = _l2_unit(raw_vectors)
    radius_ratio = np.clip(np.asarray(radius_ratio, dtype=np.float32), 0.05, 0.95).reshape(-1, 1)
    tangent_norm = np.arctanh(radius_ratio) / sqrt_c
    tangent = unit * tangent_norm
    embed = expmap0(tangent, curvature=c)
    return embed.astype(np.float32), tangent.astype(np.float32)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def learned_tangent_to_poincare(
    raw_vectors: np.ndarray,
    radius_ratio: np.ndarray,
    checkpoint_path: str | Path,
    curvature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use a learned hyperbolic encoder checkpoint when available.

    The checkpoint is trained by ``train_v27_hyperbolic_hierarchy.py`` with a
    hierarchy-aware contrastive loss.  If loading fails, raise a clear error so
    the caller can decide whether to fall back to deterministic expmap.
    """
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    config = ckpt.get("config", {})
    in_dim = int(config.get("in_dim", raw_vectors.shape[1]))
    hidden_dim = int(config.get("hidden_dim", 128))
    out_dim = int(config.get("out_dim", raw_vectors.shape[1]))
    if in_dim != int(raw_vectors.shape[1]):
        raise RuntimeError(f"Hyperbolic checkpoint in_dim={in_dim} does not match raw dim={raw_vectors.shape[1]}")

    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, out_dim),
    )
    if any(str(k).startswith("net.") for k in state.keys()):
        state = {str(k)[4:]: v for k, v in state.items() if str(k).startswith("net.")}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        tangent = model(torch.from_numpy(np.asarray(raw_vectors, dtype=np.float32))).cpu().numpy().astype(np.float32)
    # Keep the learned direction but use the hierarchy radius as explicit
    # coarse-to-fine control.  This prevents learned embeddings from collapsing
    # to a Euclidean cluster while still letting contrastive learning shape the
    # angular topology.
    unit = _l2_unit(tangent)
    c = max(float(curvature), 1e-6)
    tangent_norm = np.arctanh(np.clip(np.asarray(radius_ratio, dtype=np.float32), 0.05, 0.95)).reshape(-1, 1) / np.sqrt(c)
    tangent = unit * tangent_norm
    embed = expmap0(tangent, curvature=c)
    return embed.astype(np.float32), tangent.astype(np.float32)


def _hierarchy_curvature(hierarchy: Mapping[str, np.ndarray] | None = None) -> float:
    if not hierarchy:
        return 1.0
    value = hierarchy.get("hierarchy_curvature")
    if value is None:
        return 1.0
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else 1.0


def poincare_distance_matrix(query: np.ndarray, points: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    """Distance from one query point to many Poincare-ball points."""
    c = max(float(curvature), 1e-6)
    sqrt_c = float(np.sqrt(c))
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    p = np.asarray(points, dtype=np.float32)
    q_norm = np.sum(q * q, axis=1).reshape(-1)[0]
    p_norm = np.sum(p * p, axis=1)
    diff = np.sum((p - q) * (p - q), axis=1)
    denom = np.maximum((1.0 - c * q_norm) * (1.0 - c * p_norm), 1e-6)
    z = 1.0 + 2.0 * c * diff / denom
    return (np.arccosh(np.maximum(z, 1.0 + 1e-6)) / sqrt_c).astype(np.float32)


def poincare_pair_distance(a: np.ndarray, b: np.ndarray, curvature: float = 1.0) -> float:
    return float(
        poincare_distance_matrix(
            np.asarray(a, dtype=np.float32),
            np.asarray(b, dtype=np.float32)[None],
            curvature=curvature,
        )[0]
    )


def build_hierarchy_features(
    arrays: Any,
    items: Sequence[Mapping[str, Any]],
    hyperbolic_ckpt: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    n = len(items)
    natural = _get_array(arrays, "natural_duration", np.full((n,), 41.0, dtype=np.float32)).astype(np.float32)
    style = _get_array(arrays, "style_score", np.full((n,), 0.5, dtype=np.float32)).astype(np.float32)
    quality = _get_array(arrays, "quality_score", np.full((n,), 0.5, dtype=np.float32)).astype(np.float32)
    safety = _get_array(arrays, "safety_score", np.full((n,), 0.5, dtype=np.float32)).astype(np.float32)
    motion_desc = _get_array(arrays, "motion_desc", np.zeros((n, 4), dtype=np.float32)).astype(np.float32)
    if motion_desc.ndim == 1:
        motion_desc = motion_desc.reshape(n, 1)
    activity = motion_desc[:, 0] if motion_desc.shape[1] else np.zeros((n,), dtype=np.float32)
    activity01 = _normalize01(activity)

    turn_peak = _get_array(arrays, "turn_peak_dps", np.zeros((n,), dtype=np.float32)).astype(np.float32)
    turn_angle = _get_array(arrays, "turn_angle_deg", np.zeros((n,), dtype=np.float32)).astype(np.float32)
    turn01 = np.clip(0.55 * _normalize01(turn_peak, 0.0, 720.0) + 0.45 * _normalize01(turn_angle, 0.0, 420.0), 0.0, 1.0)
    duration01 = _normalize01(natural, 24.0, 96.0)

    event_types = [str(item.get("event_type", "neutral_flow")) for item in items]
    body_code = np.asarray([_event_group(x) for x in event_types], dtype=np.int32)
    center_code = np.asarray(
        [
            2 if event_types[i] == "support_shift" else (1 if turn01[i] > 0.35 else 0)
            for i in range(n)
        ],
        dtype=np.int32,
    )
    gesture_code = np.asarray(
        [
            2 if event_types[i] == "arm_flourish" else (1 if activity01[i] > 0.55 else 0)
            for i in range(n)
        ],
        dtype=np.int32,
    )

    coarse = _one_hot(body_code, 6)
    continuous = np.stack([activity01, turn01, duration01, style, quality, safety], axis=1).astype(np.float32)
    raw = np.concatenate([coarse, continuous], axis=1)
    semantic_proxy = _normalize_rows(raw)
    specificity = np.clip(0.30 * activity01 + 0.28 * turn01 + 0.22 * style + 0.20 * duration01, 0.0, 1.0)
    radius = 0.18 + 0.72 * specificity
    curvature = 1.0
    encoder_mode = "deterministic_expmap"
    if hyperbolic_ckpt and Path(str(hyperbolic_ckpt)).is_file():
        embed, tangent = learned_tangent_to_poincare(raw, radius, hyperbolic_ckpt, curvature=curvature)
        encoder_mode = "learned_contrastive_expmap"
    else:
        embed, tangent = tangent_to_poincare(raw, radius, curvature=curvature)

    return {
        "hierarchy_raw": raw.astype(np.float32),
        "hierarchy_embed": embed.astype(np.float32),
        "hierarchy_tangent": tangent.astype(np.float32),
        "hierarchy_radius": radius.astype(np.float32),
        "hierarchy_curvature": np.asarray([curvature], dtype=np.float32),
        "hierarchy_encoder_mode": np.asarray([encoder_mode], dtype=object),
        "semantic_proxy": semantic_proxy.astype(np.float32),
        "body_code": body_code,
        "center_code": center_code,
        "gesture_code": gesture_code,
        "activity01": activity01.astype(np.float32),
        "turn01": turn01.astype(np.float32),
        "duration01": duration01.astype(np.float32),
        "specificity": specificity.astype(np.float32),
    }


def load_or_build_hierarchy(
    arrays: Any,
    items: Sequence[Mapping[str, Any]],
    hierarchy_index_npz: str | Path | None = None,
    hyperbolic_ckpt: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    path = Path(str(hierarchy_index_npz)) if hierarchy_index_npz else None
    if path and path.is_file():
        loaded = np.load(path, allow_pickle=True)
        required = {"hierarchy_embed", "body_code", "activity01", "turn01", "duration01"}
        missing = required.difference(set(loaded.files))
        if missing:
            raise RuntimeError(f"Hierarchy index {path} is missing arrays: {sorted(missing)}")
        out = {name: np.asarray(loaded[name]) for name in loaded.files}
        if len(out["hierarchy_embed"]) != len(items):
            raise RuntimeError(
                f"Hierarchy index length {len(out['hierarchy_embed'])} does not match event index length {len(items)}"
            )
        return out
    return build_hierarchy_features(arrays, items, hyperbolic_ckpt=hyperbolic_ckpt)


def build_slot_query(
    phrase: Any,
    predicted_event: str,
    target_natural: float,
    desired_activity: float,
    music_semantic: np.ndarray | None = None,
    deep_music_weight: float = 0.0,
) -> Dict[str, Any]:
    music_event = str(getattr(phrase, "music_event", "neutral_flow"))
    group = int(MUSIC_TO_GROUP.get(music_event, EVENT_GROUPS.get(str(predicted_event), 2)))
    energy = float(getattr(phrase, "energy", 0.5))
    onset = float(getattr(phrase, "onset", 0.0))
    beat = float(getattr(phrase, "beat_density", 0.0))
    tension = float(getattr(phrase, "tension", 0.0))
    calm = float(getattr(phrase, "calmness", 0.0))
    boundary = float(getattr(phrase, "boundary_accent_strength", 0.0))
    semantic = None
    semantic_activity = 0.0
    semantic_turn = 0.0
    semantic_group_bias = np.zeros((6,), dtype=np.float32)
    if music_semantic is not None:
        semantic = np.asarray(music_semantic, dtype=np.float32).reshape(-1)
        if semantic.size >= 12:
            semantic_group_bias = semantic[:6].astype(np.float32)
            semantic_activity = float(semantic[6])
            semantic_turn = float(semantic[7])
    deep_w = float(np.clip(deep_music_weight, 0.0, 1.0))
    activity_rule = float(np.clip(0.45 * desired_activity + 0.25 * energy + 0.18 * beat + 0.12 * onset - 0.20 * calm, 0.0, 1.0))
    turn_rule = float(np.clip(0.45 * tension + 0.25 * boundary + 0.20 * beat + (0.22 if music_event in {"climax", "section_change"} else 0.0), 0.0, 1.0))
    activity = float(np.clip((1.0 - deep_w) * activity_rule + deep_w * semantic_activity, 0.0, 1.0))
    turn = float(np.clip((1.0 - deep_w) * turn_rule + deep_w * semantic_turn, 0.0, 1.0))
    duration01 = float(np.clip((target_natural - 24.0) / 72.0, 0.0, 1.0))
    coarse = np.zeros((6,), dtype=np.float32)
    coarse[np.clip(group, 0, 5)] = 1.0
    if deep_w > 0.0 and np.linalg.norm(semantic_group_bias) > 1e-6:
        semantic_group_bias = semantic_group_bias / max(float(np.sum(np.abs(semantic_group_bias))), 1e-6)
        coarse = (1.0 - deep_w) * coarse + deep_w * np.clip(semantic_group_bias, 0.0, 1.0)
        if float(coarse.sum()) > 1e-6:
            coarse = coarse / float(coarse.sum())
    raw = np.concatenate(
        [
            coarse,
            np.asarray([activity, turn, duration01, 0.72, 0.68, 0.70], dtype=np.float32),
        ]
    )
    radius = 0.18 + 0.72 * np.clip(0.34 * activity + 0.30 * turn + 0.20 * duration01 + 0.16 * boundary, 0.0, 1.0)
    curvature = 1.0
    embed, tangent = tangent_to_poincare(raw[None], np.asarray([radius], dtype=np.float32), curvature=curvature)
    semantic_proxy = _normalize_rows(raw[None])[0]
    return {
        "music_event": music_event,
        "group": group,
        "activity": activity,
        "turn": turn,
        "duration01": duration01,
        "boundary_strength": boundary,
        "embed": embed[0].astype(np.float32),
        "tangent": tangent[0].astype(np.float32),
        "radius": float(radius),
        "curvature": float(curvature),
        "semantic_proxy": semantic_proxy.astype(np.float32),
        "deep_music_weight": float(deep_w),
    }


def hierarchical_node_scores(
    hierarchy: Dict[str, np.ndarray],
    query: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    embed = np.asarray(hierarchy["hierarchy_embed"], dtype=np.float32)
    body = np.asarray(hierarchy["body_code"], dtype=np.int32)
    activity = np.asarray(hierarchy["activity01"], dtype=np.float32)
    turn = np.asarray(hierarchy["turn01"], dtype=np.float32)
    duration = np.asarray(hierarchy["duration01"], dtype=np.float32)
    curvature = _hierarchy_curvature(hierarchy)
    dist = poincare_distance_matrix(np.asarray(query["embed"], dtype=np.float32), embed, curvature=curvature)
    # Convert distance to a bounded positive score.  Very close hierarchical
    # matches approach 1; distant points approach 0.
    hyper_score = np.exp(-0.55 * dist).astype(np.float32)
    group_gap = np.abs(body.astype(np.float32) - float(query["group"])) / 5.0
    coarse_score = (1.0 - np.clip(group_gap, 0.0, 1.0)).astype(np.float32)
    exact_group = (body == int(query["group"])).astype(np.float32)
    activity_score = (1.0 - np.minimum(np.abs(activity - float(query["activity"])), 1.0)).astype(np.float32)
    turn_score = (1.0 - np.minimum(np.abs(turn - float(query["turn"])), 1.0)).astype(np.float32)
    duration_score = (1.0 - np.minimum(np.abs(duration - float(query["duration01"])), 1.0)).astype(np.float32)
    semantic_score = np.zeros_like(hyper_score, dtype=np.float32)
    if "semantic_proxy" in hierarchy and "semantic_proxy" in query:
        event_sem = _normalize_rows(np.asarray(hierarchy["semantic_proxy"], dtype=np.float32))
        query_sem = _normalize_rows(np.asarray(query["semantic_proxy"], dtype=np.float32).reshape(1, -1))[0]
        semantic_score = np.clip(0.5 + 0.5 * (event_sem @ query_sem), 0.0, 1.0).astype(np.float32)
    score = (
        0.34 * hyper_score
        + 0.24 * coarse_score
        + 0.14 * exact_group
        + 0.10 * activity_score
        + 0.08 * turn_score
        + 0.06 * duration_score
        + 0.04 * semantic_score
    ).astype(np.float32)
    return score, {
        "hierarchy_hyper_score": hyper_score,
        "hierarchy_coarse_score": coarse_score,
        "hierarchy_exact_group": exact_group,
        "hierarchy_activity_score": activity_score,
        "hierarchy_turn_score": turn_score,
        "hierarchy_duration_score": duration_score,
        "hierarchy_semantic_score": semantic_score,
        "hierarchy_distance": dist.astype(np.float32),
    }


def graph_edge_penalty(
    hierarchy: Dict[str, np.ndarray],
    prev_idx: int,
    idx: int,
    phrase: Any,
    prev_prev_idx: int | None = None,
) -> Tuple[float, Dict[str, Any]]:
    embed = np.asarray(hierarchy["hierarchy_embed"], dtype=np.float32)
    body = np.asarray(hierarchy["body_code"], dtype=np.int32)
    activity = np.asarray(hierarchy["activity01"], dtype=np.float32)
    turn = np.asarray(hierarchy["turn01"], dtype=np.float32)

    boundary_strength = float(getattr(phrase, "boundary_accent_strength", 0.0))
    music_event = str(getattr(phrase, "music_event", "neutral_flow"))
    tension = float(getattr(phrase, "tension", 0.0))
    calm = float(getattr(phrase, "calmness", 0.0))
    reset_allow = np.clip(0.22 + 0.55 * boundary_strength + (0.35 if music_event == "section_change" else 0.0), 0.0, 1.0)

    curvature = _hierarchy_curvature(hierarchy)
    hdist = poincare_pair_distance(embed[prev_idx], embed[idx], curvature=curvature)
    coarse_jump = float(abs(int(body[idx]) - int(body[prev_idx])) / 5.0)
    activity_jump = float(abs(float(activity[idx]) - float(activity[prev_idx])))
    turn_jump = float(abs(float(turn[idx]) - float(turn[prev_idx])))

    reset_penalty = max(0.0, coarse_jump - reset_allow)
    activity_allow = np.clip(0.28 + 0.35 * boundary_strength + 0.22 * tension - 0.16 * calm, 0.12, 0.85)
    activity_penalty = max(0.0, activity_jump - float(activity_allow))

    trend_penalty = 0.0
    if prev_prev_idx is not None:
        prev_delta = float(activity[prev_idx] - activity[prev_prev_idx])
        new_delta = float(activity[idx] - activity[prev_idx])
        # Abrupt sign flips are acceptable near strong boundaries, but should
        # be discouraged inside one smooth phrase.
        sign_flip = 1.0 if prev_delta * new_delta < -0.035 else 0.0
        trend_penalty = sign_flip * max(0.0, 0.55 - reset_allow)

    # Normalize hdist into a soft 0-1 range.
    hierarchy_jump_penalty = max(0.0, min(hdist / 4.5, 2.0) - 0.45 * reset_allow)
    penalty = (
        0.36 * hierarchy_jump_penalty
        + 0.28 * reset_penalty
        + 0.22 * activity_penalty
        + 0.14 * trend_penalty
        + 0.08 * turn_jump
    )
    meta = {
        "graph_hierarchy_distance": float(hdist),
        "graph_coarse_jump": float(coarse_jump),
        "graph_activity_jump": float(activity_jump),
        "graph_turn_jump": float(turn_jump),
        "graph_reset_allow": float(reset_allow),
        "graph_reset_penalty": float(reset_penalty),
        "graph_activity_penalty": float(activity_penalty),
        "graph_trend_penalty": float(trend_penalty),
        "graph_hierarchy_jump_penalty": float(hierarchy_jump_penalty),
        "graph_edge_penalty": float(penalty),
    }
    return float(penalty), meta
