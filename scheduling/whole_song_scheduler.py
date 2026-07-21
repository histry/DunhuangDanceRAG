#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Music-dominant whole-song ChoreoRAG scheduler.

Main change from the previous V26:
- music controls phrase speed and transition intent;
- natural duration is a feasibility/calibration constraint;
- boundary dynamics defines a physical minimum transition length;
- exact whole-song alignment is still enforced without hidden pad/trim.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    matrix_to_rot6d_np,
    relative_rotvec_np,
    rot6d_to_matrix_np,
    so3_exp_np,
    so3_geodesic_np,
    tangent_blend_np,
)

from model.music_motion_router import load_router_checkpoint
from model.duration_predictor import load_duration_checkpoint
from model.whole_song_planner import load_planner_checkpoint
from scheduling.index_io import load_shared_index, resolve_event_motion_path
from scheduling.retrieval import precompute_music_similarity
from scheduling.transition_builder import (
    load_optional_transition,
    refine_transition,
)
from support.scheduler_common import (
    CONTACT,
    EVENT_TYPES,
    ROOT_X,
    ROOT_Z,
    ROT,
    apply_start_anchor,
    event_compatibility,
    json_safe,
    load_motion,
    make_geodesic_transition,
    motion_boundary_metrics,
    transition_cost_from_arrays,
)
from scheduling.event_resampling import resample_event
from scheduling.duration_alignment import allocate_whole_song_durations
from scheduling.hierarchical_graph_scheduler import (
    build_slot_query,
    graph_edge_penalty as hierarchical_graph_edge_penalty,
    hierarchical_node_scores,
    load_or_build_hierarchy,
)
from scheduling.music_phrase_segmentation import (
    MusicPhrase,
    segment_music_phrases,
    split_music_phrases_for_events,
    whole_song_features,
)
from scheduling.deep_music_features import phrase_semantic_matrix
from scheduling.transition_diffusion import load_transition_diffusion, sample_transition_diffusion
from motion_geometry.heading import ROOT_ROT6D, root_yaw_np, yaw_speed_dps_np


@dataclass
class CandidateState:
    score: float
    selected: List[int]
    transition_lengths: List[int]
    parts: List[Dict[str, Any]]


def _bool_arg(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def planner_predictions(
    phrases: Sequence[MusicPhrase],
    planner_bundle,
    device: torch.device,
    fps: float,
) -> Dict[str, np.ndarray]:
    k = len(phrases)
    if planner_bundle is None:
        event_map = {
            "calm_flow": "calm_flow",
            "release": "release",
            "build_up": "build_up",
            "climax": "high_tension",
            "accent": "arm_flourish",
            "section_change": "support_shift",
            "neutral_flow": "neutral_flow",
        }
        event_ids = np.asarray(
            [EVENT_TYPES.index(event_map.get(p.music_event, "neutral_flow")) for p in phrases],
            dtype=np.int64,
        )
        # Music-dominant fallback: phrase length is the first duration prior;
        # natural duration will constrain this later in the global allocator.
        minimum = max(1, int(round(0.4 * float(fps))))
        durations = np.asarray([max(minimum, p.length - (0 if i == 0 else p.transition_base_frames)) for i, p in enumerate(phrases)], dtype=np.float32)
        transitions = np.asarray([0] + [2] * max(0, k - 1), dtype=np.int64)
        activity = np.asarray([float(np.asarray(p.query)[0]) for p in phrases], dtype=np.float32)
        return {
            "event_ids": event_ids,
            "durations": durations,
            "transition_class": transitions,
            "activity": activity,
            "mode": np.asarray(["music_rule"], dtype=object),
        }

    model = planner_bundle["model"]
    features = np.stack([np.asarray(p.planner_feature, dtype=np.float32) for p in phrases])[None]
    with torch.no_grad():
        output = model(torch.from_numpy(features).to(device))
    return {
        "event_ids": output["event_logits"][0].argmax(-1).cpu().numpy().astype(np.int64),
        "durations": output["duration_frames"][0].cpu().numpy().astype(np.float32),
        "transition_class": output["transition_logits"][0].argmax(-1).cpu().numpy().astype(np.int64),
        "activity": output["activity"][0].cpu().numpy().astype(np.float32),
        "mode": np.asarray(["learned"], dtype=object),
    }


def boundary_metrics(prev: np.ndarray, nxt: np.ndarray, fps: float = 30.0) -> Dict[str, float]:
    return motion_boundary_metrics(prev, nxt, fps=fps)


def smootherstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def dampen_event_edges(motion: np.ndarray, edge_frames: int, strength: float) -> np.ndarray:
    """Blend event edges toward low-velocity ease curves.

    V23 preserves the event's internal monotonic timing, but whole-song stitching
    can still expose high outgoing/incoming velocity at event boundaries.  This
    local C2-style edge damping leaves the event center untouched and only
    regularizes the first/last few frames before transitions are built.
    """
    x = np.asarray(motion, dtype=np.float32).copy()
    n = min(max(0, int(edge_frames)), max(0, (len(x) - 3) // 2))
    s = float(np.clip(strength, 0.0, 1.0))
    if n <= 1 or s <= 0.0:
        return x

    left_start = x[0].copy()
    left_end = x[n + 1].copy()
    left_start_rot = rot6d_to_matrix_np(left_start[ROT].reshape(24, 6))
    left_end_rot = rot6d_to_matrix_np(left_end[ROT].reshape(24, 6))
    for i in range(1, n + 1):
        u = i / float(n + 1)
        eased = smootherstep01(u)
        target = (1.0 - eased) * left_start + eased * left_end
        weight = s * (1.0 - eased)
        target_rot = tangent_blend_np(
            left_start_rot,
            left_end_rot,
            np.full((24,), eased, dtype=np.float32),
        )
        current_rot = rot6d_to_matrix_np(x[i, ROT].reshape(24, 6))
        x[i, ROT] = matrix_to_rot6d_np(
            tangent_blend_np(
                current_rot,
                target_rot,
                np.full((24,), weight, dtype=np.float32),
            )
        ).reshape(-1)
        x[i, 5] = (1.0 - weight) * x[i, 5] + weight * target[5]

    right_start_index = len(x) - n - 2
    right_end_index = len(x) - 1
    right_start = x[right_start_index].copy()
    right_end = x[right_end_index].copy()
    right_start_rot = rot6d_to_matrix_np(right_start[ROT].reshape(24, 6))
    right_end_rot = rot6d_to_matrix_np(right_end[ROT].reshape(24, 6))
    span = max(right_end_index - right_start_index, 1)
    for idx in range(right_start_index + 1, right_end_index):
        u = (idx - right_start_index) / float(span)
        eased = smootherstep01(u)
        target = (1.0 - eased) * right_start + eased * right_end
        weight = s * eased
        target_rot = tangent_blend_np(
            right_start_rot,
            right_end_rot,
            np.full((24,), eased, dtype=np.float32),
        )
        current_rot = rot6d_to_matrix_np(x[idx, ROT].reshape(24, 6))
        x[idx, ROT] = matrix_to_rot6d_np(
            tangent_blend_np(
                current_rot,
                target_rot,
                np.full((24,), weight, dtype=np.float32),
            )
        ).reshape(-1)
        x[idx, 5] = (1.0 - weight) * x[idx, 5] + weight * target[5]

    x[:, ROOT_X] = 0.0
    x[:, ROOT_Z] = 0.0
    return x.astype(np.float32)


def root_geodesic6d(start_frame: np.ndarray, end_frame: np.ndarray, length: int) -> np.ndarray:
    """Full SO(3) shortest-path interpolation for root rotation.

    The previous yaw-only fix suppressed heading spikes but discarded root
    pitch/roll, which created pose jumps.  This keeps the full root orientation
    and interpolates along the geodesic between the two endpoint rotations.
    """
    k = max(0, int(length))
    if k == 0:
        return np.zeros((0, 6), dtype=np.float32)
    roots = np.stack(
        [
            np.asarray(start_frame, dtype=np.float32)[ROOT_ROT6D],
            np.asarray(end_frame, dtype=np.float32)[ROOT_ROT6D],
        ],
        axis=0,
    )
    alphas = np.asarray([smootherstep01((i + 1) / float(k + 1)) for i in range(k)], dtype=np.float32)
    matrices = rot6d_to_matrix_np(roots)
    tangent = relative_rotvec_np(matrices[0], matrices[1])
    interpolation = matrices[0][None] @ so3_exp_np(alphas[:, None] * tangent[None])
    return matrix_to_rot6d_np(interpolation).astype(np.float32)


def enforce_yaw_safe_transition(transition: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    x = np.asarray(transition, dtype=np.float32).copy()
    if len(x) == 0:
        return x
    # Preserve full root orientation while still avoiding the 6D linear
    # interpolation long-path artifact that produced transition yaw spikes.
    x[:, ROOT_ROT6D] = root_geodesic6d(prev[-1], nxt[0], len(x))
    x[:, ROOT_X] = 0.0
    x[:, ROOT_Z] = 0.0
    return x.astype(np.float32)


def music_transition_frames(phrase: MusicPhrase, args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    base = int(phrase.transition_base_frames)
    frames_24 = int(round(24.0 * float(args.fps) / 30.0))
    frames_18 = int(round(18.0 * float(args.fps) / 30.0))
    if phrase.transition_profile == "accent_cut":
        base = min(base, frames_24)
    elif phrase.transition_profile in {"calm_sustain", "section_sustain"}:
        base = max(base, frames_24)
    elif phrase.transition_profile == "tense_drive":
        base = int(round(0.65 * base + 0.35 * frames_18))
    base = int(np.clip(base, args.transition_min_frames, args.transition_max_frames))
    return base, {
        "music_transition_frames": base,
        "transition_profile": phrase.transition_profile,
        "boundary_accent_strength": float(phrase.boundary_accent_strength),
        "speed_factor": float(phrase.speed_factor),
        "energy": float(phrase.energy),
        "onset": float(phrase.onset),
        "beat_density": float(phrase.beat_density),
        "tension": float(phrase.tension),
        "calmness": float(phrase.calmness),
    }


def physical_min_transition_frames(metrics: Dict[str, float], args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    pose = float(metrics.get("pose_jump", 0.0))
    vel = float(metrics.get("angular_velocity_jump_radps", 0.0))
    acc = float(metrics.get("angular_acceleration_jump_radps2", 0.0))
    contact = float(metrics.get("contact_jump", 0.0))
    yaw_gap = float(metrics.get("yaw_gap_deg", 0.0))
    extra = (
        args.physical_pose_frames * min(pose / max(args.pose_jump_reference, 1e-6), 2.0)
        + args.physical_velocity_frames * min(vel / max(args.velocity_jump_reference_radps, 1e-6), 2.0)
        + args.physical_acceleration_frames * min(acc / max(args.acceleration_jump_reference_radps2, 1e-6), 2.0)
        + args.physical_contact_frames * contact
    )
    yaw_frames = int(math.ceil(
        args.yaw_transition_safety_factor
        * yaw_gap
        * float(args.fps)
        / max(float(args.transition_yaw_limit_dps), 1.0)
    ))
    frames = int(round(max(args.transition_min_frames + extra, yaw_frames)))
    frames = int(np.clip(frames, args.transition_min_frames, args.transition_max_frames))
    return frames, {
        "physical_min_frames": frames,
        "pose_jump": pose,
        "angular_velocity_jump_radps": vel,
        "angular_acceleration_jump_radps2": acc,
        "contact_jump": contact,
        "yaw_gap_deg": yaw_gap,
        "yaw_required_frames": yaw_frames,
    }


def dynamic_transition_len(
    prev_motion: np.ndarray,
    next_motion: np.ndarray,
    phrase: MusicPhrase,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    metrics = boundary_metrics(prev_motion, next_motion, fps=float(args.fps))
    music_len, music_meta = music_transition_frames(phrase, args)
    physical_len, physical_meta = physical_min_transition_frames(metrics, args)
    chosen = max(music_len, physical_len)
    if phrase.transition_profile == "accent_cut" and physical_len <= music_len:
        chosen = min(chosen, int(round(24.0 * float(args.fps) / 30.0)))
    chosen = int(np.clip(chosen, args.transition_min_frames, args.transition_max_frames))
    slot_budget_cap = max(0, int(phrase.length) - int(args.min_content_frames))
    slot_budget_capped_from = None
    if getattr(args, "lock_music_boundaries", False) and chosen > slot_budget_cap:
        slot_budget_capped_from = int(chosen)
        chosen = int(slot_budget_cap)
    meta = {
        **music_meta,
        **physical_meta,
        "chosen_transition_frames": chosen,
        "slot_budget_cap": int(slot_budget_cap),
        "slot_budget_capped_from": slot_budget_capped_from,
        "dominant_reason": (
            "slot_budget"
            if slot_budget_capped_from is not None
            else ("physical" if physical_len > music_len else "music")
        ),
    }
    return chosen, meta


def planner_bundle_lengths(path: str, fps: float) -> Tuple[int, ...]:
    if not path:
        return tuple(int(round(x * float(fps) / 30.0)) for x in (12, 16, 20, 24, 30, 36, 42, 48))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    checkpoint_fps = config.get("fps")
    if checkpoint_fps is None:
        legacy_ok = (
            abs(float(fps) - 30.0) < 1.0e-6
            and os.environ.get("DUNHUANG_ALLOW_LEGACY_30FPS_CHECKPOINTS", "0") == "1"
        )
        if not legacy_ok:
            raise RuntimeError(
                f"Planner checkpoint {path} has no FPS contract. Rebuild it for {fps} FPS. "
                "Legacy weights are allowed only for the explicit 30 FPS parity baseline."
            )
    elif abs(float(checkpoint_fps) - float(fps)) > 1.0e-6:
        raise RuntimeError(
            f"Planner checkpoint FPS mismatch: checkpoint={checkpoint_fps}, runtime={fps}"
        )
    fallback = tuple(int(round(x * float(fps) / 30.0)) for x in (12, 16, 20, 24, 30, 36, 42, 48))
    return tuple(int(x) for x in config.get("transition_lengths", fallback))


def validate_checkpoint_fps(path: str, role: str, fps: float) -> None:
    """Reject Scheduler checkpoints trained under an unknown or other frame rate."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{role} checkpoint is not a mapping: {path}")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError(f"{role} checkpoint config is not a mapping: {path}")
    declared = config.get("fps", checkpoint.get("fps"))
    if declared is None:
        legacy_ok = (
            abs(float(fps) - 30.0) < 1.0e-6
            and os.environ.get("DUNHUANG_ALLOW_LEGACY_30FPS_CHECKPOINTS", "0") == "1"
        )
        if legacy_ok:
            return
        raise RuntimeError(
            f"{role} checkpoint has no FPS contract: {path}. Rebuild the rate-specific asset."
        )
    if abs(float(declared) - float(fps)) > 1.0e-6:
        raise RuntimeError(
            f"{role} checkpoint FPS mismatch: checkpoint={declared}, runtime={fps}, path={path}"
        )


def choose_events(
    phrases: Sequence[MusicPhrase],
    phrase_semantics: np.ndarray,
    predictions: Dict[str, np.ndarray],
    arrays,
    hierarchy,
    items: List[Dict[str, Any]],
    router,
    motions: Sequence[np.ndarray],
    transition_bundle,
    device: torch.device,
    args: argparse.Namespace,
) -> CandidateState:
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    mmr_embed = np.asarray(arrays["mmr_embed"], dtype=np.float32)
    style = np.asarray(arrays["style_score"], dtype=np.float32)
    quality = np.asarray(arrays["quality_score"], dtype=np.float32)
    safety = np.asarray(arrays["safety_score"], dtype=np.float32)
    natural = np.asarray(arrays["natural_duration"], dtype=np.float32)
    array_names = set(arrays.files) if hasattr(arrays, "files") else set(arrays.keys())
    turn_peak_dps = (
        np.asarray(arrays["turn_peak_dps"], dtype=np.float32)
        if "turn_peak_dps" in array_names
        else np.zeros_like(natural, dtype=np.float32)
    )
    turn_angle_deg = (
        np.asarray(arrays["turn_angle_deg"], dtype=np.float32)
        if "turn_angle_deg" in array_names
        else np.zeros_like(natural, dtype=np.float32)
    )
    entry_pose = np.asarray(arrays["entry_pose"], dtype=np.float32)
    exit_pose = np.asarray(arrays["exit_pose"], dtype=np.float32)
    entry_vel = np.asarray(arrays["entry_vel"], dtype=np.float32)
    exit_vel = np.asarray(arrays["exit_vel"], dtype=np.float32)
    event_types = [str(item.get("event_type", "neutral_flow")) for item in items]
    families = [str(item.get("family_id", "")) for item in items]
    queries = [np.asarray(p.query, dtype=np.float32) for p in phrases]
    similarities = precompute_music_similarity(router, queries, motion_desc, device)
    transition_choices = planner_bundle_lengths(args.planner_ckpt, fps=float(args.fps))

    beam = [CandidateState(0.0, [], [], [])]
    for slot, phrase in enumerate(phrases):
        predicted_event = EVENT_TYPES[int(predictions["event_ids"][slot])]
        predicted_duration = float(predictions["durations"][slot])
        desired_activity = float(predictions["activity"][slot])
        compat = np.asarray(
            [
                0.60 * event_compatibility(phrase.music_event, event)
                + 0.40 * (1.0 if event == predicted_event else event_compatibility(predicted_event, event))
                for event in event_types
            ],
            dtype=np.float32,
        )
        transition_guess = 0 if slot == 0 else int(phrase.transition_base_frames)
        slot_content_target = max(
            float(args.min_content_frames),
            float(phrase.length - min(transition_guess, max(0, phrase.length - args.min_content_frames))),
        )
        # A faster phrase can compress a longer natural action into the slot;
        # a calmer phrase can stretch a shorter one, but the target remains
        # anchored to this slot's music length rather than to natural duration.
        target_natural = max(float(args.min_content_frames), slot_content_target * max(float(phrase.speed_factor), 1e-6))
        duration_match = 1.0 - np.minimum(
            np.abs(natural - target_natural) / max(target_natural, 1.0),
            1.0,
        )
        planner_duration_match = 1.0 - np.minimum(
            np.abs(natural - predicted_duration) / max(predicted_duration, 1.0),
            1.0,
        )
        activity_match = 1.0 - np.minimum(np.abs(motion_desc[:, 0] - desired_activity), 1.0)
        low_activity = np.clip(
            (float(args.anti_static_activity_threshold) - motion_desc[:, 0])
            / max(float(args.anti_static_activity_threshold), 1e-6),
            0.0,
            1.0,
        )
        long_slot_pressure = np.clip(
            (slot_content_target - float(args.anti_static_min_content_frames))
            / max(float(args.max_single_event_seconds * args.fps) - float(args.anti_static_min_content_frames), 1.0),
            0.0,
            1.0,
        )
        music_motion_need = np.clip(
            0.42 * float(phrase.energy)
            + 0.26 * float(phrase.beat_density)
            + 0.20 * float(phrase.onset)
            + 0.12 * float(phrase.tension)
            - 0.22 * float(phrase.calmness),
            0.0,
            1.0,
        )
        anti_static_penalty = low_activity * max(float(long_slot_pressure), float(music_motion_need))
        turn_soft = float(args.turn_peak_soft_dps)
        turn_hard = max(float(args.turn_peak_hard_dps), turn_soft + 1.0)
        turn_over = np.clip((turn_peak_dps - turn_soft) / (turn_hard - turn_soft), 0.0, 1.0)
        turn_angle_over = np.clip((turn_angle_deg - args.turn_angle_soft_deg) / max(args.turn_angle_hard_deg - args.turn_angle_soft_deg, 1.0), 0.0, 1.0)
        turn_penalty = 0.75 * turn_over + 0.25 * turn_angle_over
        hierarchy_score = np.zeros_like(style, dtype=np.float32)
        hierarchy_components: Dict[str, np.ndarray] = {}
        hierarchy_query: Dict[str, Any] = {}
        if args.hierarchical_retrieval:
            hierarchy_query = build_slot_query(
                phrase,
                predicted_event=predicted_event,
                target_natural=target_natural,
                desired_activity=desired_activity,
                music_semantic=phrase_semantics[slot] if len(phrase_semantics) > slot else None,
                deep_music_weight=args.deep_music_weight if args.deep_music_features else 0.0,
            )
            hierarchy_score, hierarchy_components = hierarchical_node_scores(hierarchy, hierarchy_query)
        base = (
            args.style_weight * style
            + args.quality_weight * quality
            + args.safety_weight * safety
            + args.music_weight * similarities[slot]
            + args.event_weight * compat
            + args.duration_weight * duration_match
            + args.planner_duration_weight * planner_duration_match
            + args.activity_weight * activity_match
            + args.hierarchy_weight * hierarchy_score
            - args.anti_static_weight * anti_static_penalty
            - args.turn_peak_penalty_weight * turn_penalty
        )
        node_top_k = int(args.candidate_top_k)
        if args.graph_scheduler and int(args.graph_node_top_k) > 0:
            node_top_k = min(node_top_k, int(args.graph_node_top_k))
        shortlist = np.argsort(base)[::-1][: min(node_top_k, len(items))]
        expanded: List[CandidateState] = []
        for state in beam:
            for raw_idx in shortlist:
                idx = int(raw_idx)
                if idx in state.selected:
                    continue
                family = families[idx]
                same_family = sum(1 for previous in state.selected if families[previous] == family)
                candidate_source = str(
                    items[idx].get("source_uid", items[idx].get("source_id", "unknown"))
                )
                same_source = sum(
                    1
                    for previous in state.selected
                    if str(items[previous].get("source_uid", items[previous].get("source_id", "unknown")))
                    == candidate_source
                )
                source_run = 0
                for previous in reversed(state.selected):
                    previous_source = str(
                        items[previous].get(
                            "source_uid", items[previous].get("source_id", "unknown")
                        )
                    )
                    if previous_source != candidate_source:
                        break
                    source_run += 1
                if source_run >= int(args.max_source_run):
                    continue
                projected_slots = len(state.selected) + 1
                projected_source_share = (same_source + 1) / max(1, projected_slots)
                if (
                    projected_slots >= int(args.min_source_share_slots)
                    and projected_source_share > float(args.max_source_share)
                ):
                    continue
                if args.hard_family_unique and same_family > 0:
                    continue

                transition_len = 0
                transition_cost = 0.0
                boundary_velocity_penalty = 0.0
                boundary_acceleration_penalty = 0.0
                graph_edge_cost = 0.0
                graph_edge_meta: Dict[str, Any] = {}
                transition_meta: Dict[str, Any] = {}
                if state.selected:
                    previous = state.selected[-1]
                    transition_cost = transition_cost_from_arrays(
                        exit_pose[previous],
                        exit_vel[previous],
                        entry_pose[idx],
                        entry_vel[idx],
                    )
                    candidate_boundary = boundary_metrics(
                        motions[previous], motions[idx], fps=float(args.fps)
                    )
                    boundary_velocity_penalty = min(
                        candidate_boundary["angular_velocity_jump_radps"]
                        / max(args.velocity_jump_reference_radps, 1e-6),
                        args.boundary_penalty_cap,
                    )
                    boundary_acceleration_penalty = min(
                        candidate_boundary["angular_acceleration_jump_radps2"]
                        / max(args.acceleration_jump_reference_radps2, 1e-6),
                        args.boundary_penalty_cap,
                    )
                    if args.music_dominant_timing:
                        transition_len, transition_meta = dynamic_transition_len(
                            motions[previous],
                            motions[idx],
                            phrase,
                            args,
                        )
                        transition_meta = {**transition_meta, "candidate_boundary": candidate_boundary}
                    else:
                        class_index = int(predictions["transition_class"][slot])
                        transition_len = int(transition_choices[min(class_index, len(transition_choices) - 1)])
                        transition_meta = {"chosen_transition_frames": transition_len, "dominant_reason": "planner_class"}
                    if args.graph_scheduler:
                        prev_prev = state.selected[-2] if len(state.selected) >= 2 else None
                        graph_edge_cost, graph_edge_meta = hierarchical_graph_edge_penalty(
                            hierarchy,
                            previous,
                            idx,
                            phrase,
                            prev_prev_idx=prev_prev,
                        )
                        if args.graph_hard_prune and graph_edge_cost > args.graph_hard_prune_threshold:
                            continue
                mmr = 0.0
                if state.selected:
                    mmr = max(float(mmr_embed[idx] @ mmr_embed[previous]) for previous in state.selected)
                score = (
                    state.score
                    + float(base[idx])
                    - args.transition_weight * transition_cost
                    - args.boundary_velocity_penalty_weight * boundary_velocity_penalty
                    - args.boundary_acceleration_penalty_weight * boundary_acceleration_penalty
                    - args.graph_edge_weight * graph_edge_cost
                    - args.mmr_weight * mmr
                    - args.family_repeat_weight * same_family
                    - args.source_repeat_weight * same_source
                )
                part = {
                    "slot": slot,
                    "music_start": phrase.start,
                    "music_end": phrase.end,
                    "music_length": phrase.length,
                    "music_event": phrase.music_event,
                    "music_speed_factor": float(phrase.speed_factor),
                    "music_transition_profile": phrase.transition_profile,
                    "boundary_accent_strength": float(phrase.boundary_accent_strength),
                    "predicted_motion_event": predicted_event,
                    "predicted_duration": predicted_duration,
                    "event_index": idx,
                    "event_uid": str(items[idx]["event_uid"]),
                    "event_id": str(items[idx].get("event_id", idx)),
                    "source_uid": candidate_source,
                    "projected_source_share": float(projected_source_share),
                    "family_id": family,
                    "motion_event": event_types[idx],
                    "natural_duration": float(natural[idx]),
                    "slot_content_target": float(slot_content_target),
                    "target_natural_duration": float(target_natural),
                    "transition_len": int(transition_len),
                    "transition_meta": transition_meta,
                    "style": float(style[idx]),
                    "quality": float(quality[idx]),
                    "safety": float(safety[idx]),
                    "music_similarity": float(similarities[slot, idx]),
                    "event_compatibility": float(compat[idx]),
                    "duration_match": float(duration_match[idx]),
                    "planner_duration_match": float(planner_duration_match[idx]),
                    "activity_match": float(activity_match[idx]),
                    "anti_static_penalty": float(anti_static_penalty[idx]),
                    "turn_peak_dps": float(turn_peak_dps[idx]),
                    "turn_angle_deg": float(turn_angle_deg[idx]),
                    "turn_penalty": float(turn_penalty[idx]),
                    "candidate_top_k": int(args.candidate_top_k),
                    "graph_node_top_k": int(node_top_k),
                    "hierarchy_enabled": bool(args.hierarchical_retrieval),
                    "hierarchy_query_group": int(hierarchy_query.get("group", -1)) if hierarchy_query else -1,
                    "hierarchy_score": float(hierarchy_score[idx]) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_hyper_score": float(hierarchy_components.get("hierarchy_hyper_score", np.zeros_like(style))[idx]) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_coarse_score": float(hierarchy_components.get("hierarchy_coarse_score", np.zeros_like(style))[idx]) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_activity_score": float(hierarchy_components.get("hierarchy_activity_score", np.zeros_like(style))[idx]) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_turn_score": float(hierarchy_components.get("hierarchy_turn_score", np.zeros_like(style))[idx]) if args.hierarchical_retrieval else 0.0,
                    "hierarchy_semantic_score": float(hierarchy_components.get("hierarchy_semantic_score", np.zeros_like(style))[idx]) if args.hierarchical_retrieval else 0.0,
                    "transition_cost": float(transition_cost),
                    "boundary_velocity_penalty": float(boundary_velocity_penalty),
                    "boundary_acceleration_penalty": float(boundary_acceleration_penalty),
                    "graph_scheduler_enabled": bool(args.graph_scheduler),
                    "graph_edge_cost": float(graph_edge_cost),
                    "graph_edge_meta": graph_edge_meta,
                    "mmr_penalty": float(mmr),
                    "score": float(score),
                }
                expanded.append(
                    CandidateState(
                        score=score,
                        selected=state.selected + [idx],
                        transition_lengths=state.transition_lengths + [transition_len],
                        parts=state.parts + [part],
                    )
                )
        if not expanded:
            raise RuntimeError(
                f"No V26 candidate for phrase {slot}. Increase candidate_top_k/graph_node_top_k or relax hard pruning."
            )
        expanded.sort(key=lambda state: state.score, reverse=True)
        beam = expanded[: args.beam_size]
    return beam[0]


def cap_transition_budget(
    transition_lengths: Sequence[int],
    *,
    total_frames: int,
    max_fraction: float,
    minimum_nonzero: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Cap total transition coverage while preserving every real boundary."""
    values = [max(0, int(value)) for value in transition_lengths]
    if values:
        values[0] = 0
    before = int(sum(values))
    budget = max(0, int(math.floor(float(total_frames) * float(max_fraction))))
    active = [index for index, value in enumerate(values) if index > 0 and value > 0]
    floor = max(1, int(minimum_nonzero))
    if active and floor * len(active) > budget:
        floor = max(1, budget // len(active))
    while sum(values) > budget:
        reducible = [index for index in active if values[index] > floor]
        if not reducible:
            break
        index = max(reducible, key=lambda i: values[i])
        values[index] -= 1
    return values, {
        "before_frames": before,
        "after_frames": int(sum(values)),
        "total_frames": int(total_frames),
        "max_fraction": float(max_fraction),
        "actual_fraction": float(sum(values) / max(1, int(total_frames))),
        "minimum_nonzero_frames": int(floor),
        "capped": bool(values != [max(0, int(x)) if i else 0 for i, x in enumerate(transition_lengths)]),
    }


def generate_one(
    audio_path: Path,
    arrays,
    hierarchy,
    items,
    motions,
    router,
    transition_bundle,
    v23_bundle,
    planner_bundle,
    device,
    args,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    features, audio_meta = whole_song_features(
        audio_path,
        fps=args.fps,
        cache_dir=args.feature_dir,
        max_seconds=args.max_seconds,
    )
    source_phrases, segmentation = segment_music_phrases(
        features,
        fps=args.fps,
        min_phrase_seconds=args.min_phrase_seconds,
        max_phrase_seconds=args.max_phrase_seconds,
        boundary_quantile=args.boundary_quantile,
        beat_snap_seconds=args.beat_snap_seconds,
    )
    phrases, slot_expansion = split_music_phrases_for_events(
        features,
        source_phrases,
        fps=args.fps,
        enabled=args.multi_event_phrases,
        max_slot_seconds=args.max_single_event_seconds,
        min_slot_seconds=args.min_subphrase_seconds,
        max_events_per_phrase=args.max_events_per_phrase,
        beat_snap_seconds=args.slot_beat_snap_seconds,
        calm_max_slot_seconds=args.calm_max_single_event_seconds,
    )
    if len(phrases) > args.max_phrases:
        raise RuntimeError(
            f"{audio_path}: detected {len(phrases)} event slots, above --max_phrases={args.max_phrases}. "
            "Increase max_phrases or max_single_event_seconds."
        )
    phrase_semantics, semantic_meta = phrase_semantic_matrix(
        audio_path,
        phrases,
        enabled=bool(args.deep_music_features),
        model_name=str(args.deep_music_model),
        cache_dir=args.deep_music_cache or args.feature_dir,
        require_deep=bool(args.require_deep_music),
        min_deep_success=float(args.deep_music_min_success),
        fps=float(args.fps),
    )
    predictions = planner_predictions(phrases, planner_bundle, device, fps=float(args.fps))
    selected_state = choose_events(
        phrases,
        phrase_semantics,
        predictions,
        arrays,
        hierarchy,
        items,
        router,
        motions,
        transition_bundle,
        device,
        args,
    )

    phrase_lengths = [phrase.length for phrase in phrases]
    natural_durations = [part["natural_duration"] for part in selected_state.parts]
    planner_durations = [float(x) for x in predictions["durations"]]
    event_types = [part["motion_event"] for part in selected_state.parts]
    music_events = [phrase.music_event for phrase in phrases]
    transition_lengths = list(selected_state.transition_lengths)
    transition_lengths[0] = 0
    if args.lock_music_boundaries:
        for i, phrase in enumerate(phrases):
            cap = 0 if i == 0 else max(0, int(phrase.length) - int(args.min_content_frames))
            if int(transition_lengths[i]) > cap:
                previous = int(transition_lengths[i])
                transition_lengths[i] = int(cap)
                selected_state.parts[i]["transition_len"] = int(cap)
                meta = dict(selected_state.parts[i].get("transition_meta", {}))
                meta["pre_allocation_slot_budget_cap"] = int(cap)
                meta["pre_allocation_capped_from"] = previous
                meta["dominant_reason"] = "slot_budget"
                selected_state.parts[i]["transition_meta"] = meta
    transition_lengths, transition_budget = cap_transition_budget(
        transition_lengths,
        total_frames=len(features),
        max_fraction=args.max_transition_fraction,
        minimum_nonzero=args.transition_budget_min_frames,
    )
    for index, value in enumerate(transition_lengths):
        selected_state.parts[index]["transition_len"] = int(value)
        meta = dict(selected_state.parts[index].get("transition_meta", {}))
        meta["global_transition_budget"] = transition_budget
        selected_state.parts[index]["transition_meta"] = meta
    music_speed_factors = [phrase.speed_factor for phrase in phrases]
    music_content_targets = [max(args.min_content_frames, phrase.length - transition_lengths[i]) for i, phrase in enumerate(phrases)]
    allocation = allocate_whole_song_durations(
        phrase_lengths=phrase_lengths,
        natural_durations=natural_durations,
        planner_durations=planner_durations,
        event_types=event_types,
        music_events=music_events,
        transition_lengths=transition_lengths,
        total_frames=len(features),
        music_weight=args.global_music_weight,
        natural_weight=args.global_natural_weight,
        planner_weight=args.global_planner_weight,
        min_content_frames=args.min_content_frames,
        min_warp=args.min_time_warp,
        max_warp=args.max_time_warp,
        music_speed_factors=music_speed_factors,
        music_content_targets=music_content_targets,
        allow_music_bound_override=args.allow_music_bound_override,
        lock_music_boundaries=args.lock_music_boundaries,
    )

    contents: List[np.ndarray] = []
    resampling_reports: List[Dict[str, Any]] = []
    for idx, target_len in zip(selected_state.selected, allocation["content_lengths"]):
        content, report = resample_event(
            motions[idx],
            int(target_len),
            v23_bundle,
            device,
            fps=float(args.fps),
            min_turn_angle=args.v23_min_turn_angle,
            min_peak_dps=args.v23_min_peak_dps,
        )
        content[:, ROOT_X] = 0.0
        content[:, ROOT_Z] = 0.0
        content = dampen_event_edges(content, args.edge_damping_frames, args.edge_damping_strength)
        contents.append(content)
        resampling_reports.append(report)

    pieces: List[np.ndarray] = []
    boundary_reports: List[Dict[str, Any]] = []
    for slot, content in enumerate(contents):
        if slot > 0:
            k = int(transition_lengths[slot])
            rough = make_geodesic_transition(contents[slot - 1], content, k)
            transition = refine_transition(
                transition_bundle,
                rough,
                contents[slot - 1][-1],
                content[0],
                np.asarray(phrases[slot].query, dtype=np.float32),
                device,
            )
            transition = enforce_yaw_safe_transition(transition, contents[slot - 1], content)
            if args.transition_diffusion and args.transition_diffusion_ckpt:
                transition, diffusion_meta = sample_transition_diffusion(
                    args.transition_diffusion_bundle,
                    contents[slot - 1][-1],
                    content[0],
                    k,
                    np.asarray(phrases[slot].query, dtype=np.float32),
                    rough=transition,
                    device=device,
                    blend=args.transition_diffusion_blend,
                    steps=args.transition_diffusion_steps,
                )
                transition = enforce_yaw_safe_transition(transition, contents[slot - 1], content)
            else:
                diffusion_meta = {"enabled": False}
            metrics = boundary_metrics(
                contents[slot - 1], content, fps=float(args.fps)
            )
            metrics["transition_len"] = k
            metrics["transition_meta"] = selected_state.parts[slot].get("transition_meta", {})
            metrics["transition_diffusion"] = diffusion_meta
            boundary_reports.append(metrics)
            pieces.append(transition)
        pieces.append(content)

    motion = np.concatenate(pieces, axis=0).astype(np.float32)
    if len(motion) != len(features):
        raise AssertionError(
            f"V26 output length mismatch: generated={len(motion)} music_frames={len(features)}. "
            "No pad/trim fallback is permitted."
        )
    motion[:, ROOT_X] = 0.0
    motion[:, ROOT_Z] = 0.0
    if args.start_pose:
        start_path = Path(args.start_pose)
        if start_path.is_file():
            motion = apply_start_anchor(
                motion,
                np.load(start_path).astype(np.float32).reshape(-1),
                args.start_anchor_blend,
            )

    report = {
        "version": "v26_music_dominant_whole_song_choreorag",
        "audio": str(audio_path),
        "audio_meta": audio_meta,
        "rotation_contract": {
            "motion_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
            "event_index_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
            "duration_checkpoint_rot6d_layout": v23_bundle.get("rot6d_layout"),
            "transition_checkpoint_rot6d_layout": (
                transition_bundle.get("rot6d_layout")
                if transition_bundle is not None
                else None
            ),
            "transition_diffusion_checkpoint_rot6d_layout": (
                args.transition_diffusion_bundle.get("rot6d_layout")
                if args.transition_diffusion_bundle is not None
                else None
            ),
        },
        "planner_mode": str(predictions["mode"][0]),
        "music_semantic": semantic_meta,
        "segmentation": {
            **segmentation,
            "source_num_phrases": len(source_phrases),
            "source_boundaries": [int(source_phrases[0].start)] + [int(p.end) for p in source_phrases] if source_phrases else [],
            "event_slot_expansion": slot_expansion,
            "effective_num_slots": len(phrases),
            "effective_slot_boundaries": [int(phrases[0].start)] + [int(p.end) for p in phrases] if phrases else [],
        },
        "allocation": allocation,
        "event_db_contract": dict(getattr(args, "event_db_contract", {})),
        "transition_budget": transition_budget,
        "score": selected_state.score,
        "schedule": [],
        "boundary_metrics": boundary_reports,
        "timing_policy": {
            "hierarchical_retrieval": bool(args.hierarchical_retrieval),
            "graph_scheduler": bool(args.graph_scheduler),
            "hierarchy_index_npz": str(args.hierarchy_index_npz),
            "hierarchy_weight": float(args.hierarchy_weight),
            "deep_music_features": bool(args.deep_music_features),
            "deep_music_model": str(args.deep_music_model),
            "deep_music_weight": float(args.deep_music_weight),
            "require_deep_music": bool(args.require_deep_music),
            "deep_music_min_success": float(args.deep_music_min_success),
            "graph_node_top_k": int(args.graph_node_top_k),
            "graph_edge_weight": float(args.graph_edge_weight),
            "graph_hard_prune": bool(args.graph_hard_prune),
            "graph_hard_prune_threshold": float(args.graph_hard_prune_threshold),
            "music_dominant_timing": bool(args.music_dominant_timing),
            "transition_min_frames": int(args.transition_min_frames),
            "transition_max_frames": int(args.transition_max_frames),
            "global_music_weight": float(args.global_music_weight),
            "global_natural_weight": float(args.global_natural_weight),
            "global_planner_weight": float(args.global_planner_weight),
            "turn_peak_penalty_weight": float(args.turn_peak_penalty_weight),
            "boundary_velocity_penalty_weight": float(args.boundary_velocity_penalty_weight),
            "boundary_acceleration_penalty_weight": float(args.boundary_acceleration_penalty_weight),
            "edge_damping_frames": int(args.edge_damping_frames),
            "edge_damping_strength": float(args.edge_damping_strength),
            "multi_event_phrases": bool(args.multi_event_phrases),
            "lock_music_boundaries": bool(args.lock_music_boundaries),
            "max_single_event_seconds": float(args.max_single_event_seconds),
            "calm_max_single_event_seconds": float(args.calm_max_single_event_seconds),
            "anti_static_weight": float(args.anti_static_weight),
            "transition_diffusion": bool(args.transition_diffusion),
            "transition_diffusion_ckpt": str(args.transition_diffusion_ckpt),
            "transition_diffusion_blend": float(args.transition_diffusion_blend),
            "transition_diffusion_steps": int(args.transition_diffusion_steps),
        },
    }
    for slot, part in enumerate(selected_state.parts):
        merged = dict(part)
        if slot < len(slot_expansion.get("slot_meta", [])):
            merged["slot_meta"] = slot_expansion["slot_meta"][slot]
        merged["allocated_content_len"] = int(allocation["content_lengths"][slot])
        merged["allocated_phrase_total"] = int(allocation["phrase_total_lengths"][slot])
        merged["time_warp_ratio"] = float(allocation["warp_ratios"][slot])
        merged["resampling"] = resampling_reports[slot]
        report["schedule"].append(merged)
    return motion, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--music", action="append", default=[])
    parser.add_argument("--music_glob", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--v23_ckpt", required=True)
    parser.add_argument("--planner_ckpt", default="")
    parser.add_argument("--transition_ckpt", default="")
    parser.add_argument("--transition_diffusion_ckpt", default="")
    parser.add_argument("--hierarchy_index_npz", default="")
    parser.add_argument("--hyperbolic_ckpt", default="")
    parser.add_argument("--feature_dir", default="")
    parser.add_argument("--deep_music_cache", default="")
    parser.add_argument("--start_pose", default="")
    parser.add_argument("--start_anchor_blend", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--frame_parameters_fps",
        type=float,
        default=30.0,
        help="Reference rate for CLI values named *_frames; they are scaled to --fps.",
    )
    parser.add_argument("--max_seconds", type=float, default=0.0)
    parser.add_argument("--min_phrase_seconds", type=float, default=2.5)
    parser.add_argument("--max_phrase_seconds", type=float, default=7.5)
    parser.add_argument("--boundary_quantile", type=float, default=0.68)
    parser.add_argument("--beat_snap_seconds", type=float, default=0.35)
    parser.add_argument("--max_phrases", type=int, default=96)
    parser.add_argument("--multi_event_phrases", type=_bool_arg, default=True)
    parser.add_argument("--lock_music_boundaries", type=_bool_arg, default=True)
    parser.add_argument("--max_single_event_seconds", type=float, default=5.00)
    parser.add_argument("--calm_max_single_event_seconds", type=float, default=4.50)
    parser.add_argument("--min_subphrase_seconds", type=float, default=2.50)
    parser.add_argument("--max_events_per_phrase", type=int, default=2)
    parser.add_argument("--slot_beat_snap_seconds", type=float, default=0.25)
    parser.add_argument("--beam_size", type=int, default=24)
    parser.add_argument("--candidate_top_k", type=int, default=256)
    parser.add_argument("--style_weight", type=float, default=1.35)
    parser.add_argument("--quality_weight", type=float, default=0.65)
    parser.add_argument("--safety_weight", type=float, default=0.35)
    parser.add_argument("--music_weight", type=float, default=0.90)
    parser.add_argument("--event_weight", type=float, default=0.70)
    parser.add_argument("--duration_weight", type=float, default=0.45)
    parser.add_argument("--planner_duration_weight", type=float, default=0.15)
    parser.add_argument("--activity_weight", type=float, default=0.25)
    parser.add_argument("--hierarchical_retrieval", type=_bool_arg, default=True)
    parser.add_argument("--hierarchy_weight", type=float, default=0.55)
    parser.add_argument("--deep_music_features", type=_bool_arg, default=False)
    parser.add_argument("--deep_music_model", default="clap")
    parser.add_argument("--deep_music_weight", type=float, default=0.25)
    parser.add_argument("--require_deep_music", type=_bool_arg, default=False)
    parser.add_argument("--deep_music_min_success", type=float, default=0.80)
    parser.add_argument("--graph_scheduler", type=_bool_arg, default=True)
    parser.add_argument("--graph_node_top_k", type=int, default=96)
    parser.add_argument("--graph_edge_weight", type=float, default=0.45)
    parser.add_argument("--graph_hard_prune", type=_bool_arg, default=False)
    parser.add_argument("--graph_hard_prune_threshold", type=float, default=1.35)
    parser.add_argument("--anti_static_weight", type=float, default=0.45)
    parser.add_argument("--anti_static_activity_threshold", type=float, default=0.030)
    parser.add_argument("--anti_static_min_content_frames", type=int, default=60)
    parser.add_argument("--transition_weight", type=float, default=0.60)
    parser.add_argument("--boundary_velocity_penalty_weight", type=float, default=0.35)
    parser.add_argument("--boundary_acceleration_penalty_weight", type=float, default=0.35)
    parser.add_argument("--boundary_penalty_cap", type=float, default=4.0)
    parser.add_argument("--turn_peak_soft_dps", type=float, default=360.0)
    parser.add_argument("--turn_peak_hard_dps", type=float, default=720.0)
    parser.add_argument("--turn_angle_soft_deg", type=float, default=220.0)
    parser.add_argument("--turn_angle_hard_deg", type=float, default=420.0)
    parser.add_argument("--turn_peak_penalty_weight", type=float, default=0.75)
    parser.add_argument("--edge_damping_frames", type=int, default=10)
    parser.add_argument("--edge_damping_strength", type=float, default=0.65)
    parser.add_argument("--mmr_weight", type=float, default=0.40)
    parser.add_argument("--family_repeat_weight", type=float, default=0.58)
    parser.add_argument("--source_repeat_weight", type=float, default=0.18)
    parser.add_argument("--max_source_run", type=int, default=2)
    parser.add_argument("--max_source_share", type=float, default=0.40)
    parser.add_argument("--min_source_share_slots", type=int, default=6)
    parser.add_argument("--hard_family_unique", action="store_true")
    parser.add_argument("--global_music_weight", type=float, default=1.60)
    parser.add_argument("--global_natural_weight", type=float, default=0.85)
    parser.add_argument("--global_planner_weight", type=float, default=0.75)
    parser.add_argument("--min_content_frames", type=int, default=12)
    parser.add_argument("--min_time_warp", type=float, default=0.70)
    parser.add_argument("--max_time_warp", type=float, default=1.50)
    parser.add_argument("--allow_music_bound_override", type=_bool_arg, default=True)
    parser.add_argument("--music_dominant_timing", type=_bool_arg, default=True)
    parser.add_argument("--transition_min_frames", type=int, default=8)
    parser.add_argument("--transition_max_frames", type=int, default=24)
    parser.add_argument("--max_transition_fraction", type=float, default=0.20)
    parser.add_argument("--transition_budget_min_frames", type=int, default=6)
    parser.add_argument("--transition_diffusion", type=_bool_arg, default=False)
    parser.add_argument("--transition_diffusion_blend", type=float, default=0.45)
    parser.add_argument("--transition_diffusion_steps", type=int, default=12)
    parser.add_argument("--transition_yaw_limit_dps", type=float, default=220.0)
    parser.add_argument("--yaw_transition_safety_factor", type=float, default=1.90)
    parser.add_argument("--pose_jump_reference", type=float, default=0.120)
    parser.add_argument("--velocity_jump_reference_radps", type=float, default=0.30)
    parser.add_argument("--acceleration_jump_reference_radps2", type=float, default=16.20)
    parser.add_argument("--physical_pose_frames", type=float, default=8.0)
    parser.add_argument("--physical_velocity_frames", type=float, default=10.0)
    parser.add_argument("--physical_acceleration_frames", type=float, default=8.0)
    parser.add_argument("--physical_contact_frames", type=float, default=8.0)
    parser.add_argument("--v23_min_turn_angle", type=float, default=10.0)
    parser.add_argument("--v23_min_peak_dps", type=float, default=14.0)
    args = parser.parse_args()
    if args.fps <= 0.0 or args.frame_parameters_fps <= 0.0:
        raise ValueError("fps and frame_parameters_fps must be positive")
    frame_scale = float(args.fps) / float(args.frame_parameters_fps)
    for name in (
        "start_anchor_blend",
        "anti_static_min_content_frames",
        "edge_damping_frames",
        "min_content_frames",
        "transition_min_frames",
        "transition_max_frames",
        "transition_budget_min_frames",
    ):
        setattr(args, name, max(1, int(round(float(getattr(args, name)) * frame_scale))))
    for name in (
        "physical_pose_frames",
        "physical_velocity_frames",
        "physical_acceleration_frames",
        "physical_contact_frames",
    ):
        setattr(args, name, float(getattr(args, name)) * frame_scale)

    paths = [Path(x) for x in args.music]
    if args.music_glob:
        paths.extend(Path(x) for x in sorted(glob.glob(args.music_glob)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise RuntimeError("Provide --music or --music_glob")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.feature_dir:
        args.feature_dir = str(out_dir / "music_features")

    index_json = Path(args.index_json).resolve()
    metadata, arrays, items = load_shared_index(
        index_json,
        Path(args.duration_index_npz),
    )
    index_rates = [float(value) for value in metadata.get("canonical_fps_values", [])]
    legacy_30_ok = (
        not index_rates
        and abs(float(args.fps) - 30.0) < 1.0e-6
        and os.environ.get("DUNHUANG_ALLOW_LEGACY_30FPS_INDEX", "0") == "1"
    )
    if not legacy_30_ok and index_rates != [float(args.fps)]:
        raise RuntimeError(
            "Scheduler FPS contract mismatch: "
            f"index={index_rates!r}, runtime={[float(args.fps)]!r}. "
            "Use the rate-specific Event-DB, Scheduler index and duration assets."
        )
    args.event_db_contract = dict(metadata["event_db_contract"])
    if "natural_duration" not in arrays.files:
        raise RuntimeError(
            "duration_index_npz lacks natural_duration. Run scheduling/build_duration_index.py first."
        )
    hierarchy = load_or_build_hierarchy(arrays, items, args.hierarchy_index_npz, hyperbolic_ckpt=args.hyperbolic_ckpt)
    motions = [
        load_motion(resolve_event_motion_path(item, index_json, metadata=metadata))
        for item in items
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_checkpoint_fps(args.router_ckpt, "Router", float(args.fps))
    validate_checkpoint_fps(args.v23_ckpt, "Duration", float(args.fps))
    if args.planner_ckpt:
        validate_checkpoint_fps(args.planner_ckpt, "Planner", float(args.fps))
    router = load_router_checkpoint(args.router_ckpt, device=device)
    transition_bundle = load_optional_transition(args.transition_ckpt, device)
    args.transition_diffusion_bundle = load_transition_diffusion(args.transition_diffusion_ckpt, device) if args.transition_diffusion and args.transition_diffusion_ckpt else None
    v23_bundle = load_duration_checkpoint(args.v23_ckpt, device=device)
    planner_bundle = load_planner_checkpoint(args.planner_ckpt, device=device) if args.planner_ckpt else None

    summary = {
        "version": "v26_music_dominant_whole_song_choreorag",
        "rotation_contract": {
            "motion_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
            "event_index_rot6d_layout": metadata["rot6d_layout"],
            "duration_checkpoint_rot6d_layout": v23_bundle.get("rot6d_layout"),
            "transition_checkpoint_rot6d_layout": (
                transition_bundle.get("rot6d_layout")
                if transition_bundle is not None
                else None
            ),
            "transition_diffusion_checkpoint_rot6d_layout": (
                args.transition_diffusion_bundle.get("rot6d_layout")
                if args.transition_diffusion_bundle is not None
                else None
            ),
        },
        "planner_ckpt": args.planner_ckpt,
        "router_ckpt": args.router_ckpt,
        "v23_ckpt": args.v23_ckpt,
        "transition_ckpt": args.transition_ckpt,
        "event_db_contract": dict(metadata["event_db_contract"]),
        "results": {},
    }
    for path in paths:
        motion, report = generate_one(
            path,
            arrays,
            hierarchy,
            items,
            motions,
            router,
            transition_bundle,
            v23_bundle,
            planner_bundle,
            device,
            args,
        )
        key = path.stem
        npy_path = out_dir / f"{key}_v26.npy"
        report_path = out_dir / f"{key}_v26.schedule_report.json"
        np.save(npy_path, motion[None].astype(np.float32))
        report["out_npy"] = str(npy_path)
        report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
        summary["results"][key] = {
            "npy": str(npy_path),
            "report": str(report_path),
            "frames": int(len(motion)),
            "phrases": len(report["schedule"]),
            "event_ids": [row["event_id"] for row in report["schedule"]],
            "families": [row["family_id"] for row in report["schedule"]],
        }
        print(f"[SAVED] {key}: frames={len(motion)} phrases={len(report['schedule'])}")

    summary_path = out_dir / "V26_WHOLE_SONG_SUMMARY.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
