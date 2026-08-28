#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Music-dominant whole-song ChoreoRAG scheduler.

Main change from the previous Whole-Song Planner:
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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    matrix_to_rot6d_np,
    relative_rotvec_np,
    rot6d_to_matrix_np,
    so3_exp_np,
    tangent_blend_np,
)

from model.music_motion_router import load_router_checkpoint
from model.duration_predictor import load_duration_checkpoint
from model.whole_song_planner import load_planner_checkpoint
from scheduling.index_io import load_shared_index, resolve_event_motion_path
from scheduling.retrieval import (
    LOCAL_ACTION_LABELS,
    aggregate_action_compatibility,
    precompute_music_routing,
)
from scheduling.transition_builder import (
    load_optional_transition,
    refine_transition,
)
from support.scheduler_common import (
    CONTACT,
    ROOT_X,
    ROOT_Z,
    ROT,
    apply_start_anchor,
    intrinsic_transition_cost_from_arrays,
    posture_state_distance,
    json_safe,
    load_motion,
    motion_boundary_metrics,
)
from support.motion_geometry import (
    canonicalize_event_root_np,
    compose_event_root_xz_np,
    event_endpoint_geometry_np,
    make_so3_transition,
    project_motion_rotations_np,
    project_transition_floor_np,
    recompute_transition_contacts_np,
)
from motion_geometry.inbetween import duration_displacement, INBETWEEN_PROTOCOL
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
from scheduling.temporal_router_contract import (
    assert_formal_planner_scientific_contract,
    assert_formal_router_scientific_contract,
    phrase_feature_sequences,
)
from scheduling.schedule_hard_constraints import (
    DEFAULT_MAX_POSE_HOLD_RATIO,
    DEFAULT_MAX_SINGLE_SOURCE_RATIO,
    DEFAULT_MIN_CORE_FRAME_RATIO,
    DEFAULT_MIN_UNIQUE_EVENTS,
    assert_schedule_hard_constraints,
)
from support.scheduler_checkpoint_contracts import assert_scheduler_checkpoint_contract
from motion_geometry.heading import ROOT_ROT6D


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
    if planner_bundle is None:
        raise RuntimeError("Current-protocol scheduling requires a Planner checkpoint")

    model = planner_bundle["model"]
    features = np.stack([np.asarray(p.planner_feature, dtype=np.float32) for p in phrases])[None]
    with torch.no_grad():
        output = model(torch.from_numpy(features).to(device))
    return {
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

    Duration Model preserves the event's internal monotonic timing, but whole-song stitching
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


def enforce_yaw_safe_transition(
    transition: np.ndarray,
    prev: np.ndarray,
    nxt: np.ndarray,
    *,
    canonical_root: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(transition, dtype=np.float32).copy()
    if len(x) == 0:
        return x
    # Learned refiners may edit limbs, but the canonical Hermite root path is a
    # hard contract.  Restoring it prevents checkpoint-era zero-XZ behavior and
    # preserves C1 root translation/orientation.
    if canonical_root is not None:
        reference = np.asarray(canonical_root, dtype=np.float32)
        if reference.shape != x.shape:
            raise ValueError(
                "canonical_root and transition shapes differ: "
                f"{reference.shape} != {x.shape}"
            )
        x[:, ROOT_X:ROOT_Z + 1] = reference[:, ROOT_X:ROOT_Z + 1]
        x[:, ROOT_ROT6D] = reference[:, ROOT_ROT6D]
    else:
        root_matrix = rot6d_to_matrix_np(x[:, ROOT_ROT6D])
        x[:, ROOT_ROT6D] = matrix_to_rot6d_np(root_matrix)
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
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    checkpoint_fps = config.get("fps")
    if checkpoint_fps is None:
        raise RuntimeError(
            f"Planner checkpoint {path} has no FPS contract. Rebuild it for {fps} FPS."
        )
    elif abs(float(checkpoint_fps) - float(fps)) > 1.0e-6:
        raise RuntimeError(
            f"Planner checkpoint FPS mismatch: checkpoint={checkpoint_fps}, runtime={fps}"
        )
    fallback = tuple(int(round(x * float(fps) / 30.0)) for x in (12, 16, 20, 24, 30, 36, 42, 48))
    return tuple(int(x) for x in config.get("transition_lengths", fallback))


def validate_scheduler_checkpoint(
    path: str,
    role: str,
    fps: float,
    event_db_contract: Dict[str, Any],
    index_json: str,
    index_npz: str,
) -> None:
    """Reject checkpoints trained against another rate or ordered Event-DB."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert_scheduler_checkpoint_contract(
        checkpoint,
        role=role.lower(),
        runtime_fps=float(fps),
        event_db_contract=event_db_contract,
        index_json=index_json,
        index_npz=index_npz,
        path=str(path),
    )


def choose_events(
    phrases: Sequence[MusicPhrase],
    phrase_sequences: np.ndarray,
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
    endpoint_geometry_keys = {
        "event_floor_y_m",
        "entry_floor_relative_m",
        "exit_floor_relative_m",
        "entry_root_height_m",
        "exit_root_height_m",
    }
    if endpoint_geometry_keys.issubset(array_names):
        endpoint_geometry = [
            {
                "floor_y_m": float(arrays["event_floor_y_m"][index]),
                "entry_floor_relative_m": float(
                    arrays["entry_floor_relative_m"][index]
                ),
                "exit_floor_relative_m": float(
                    arrays["exit_floor_relative_m"][index]
                ),
                "entry_root_height_m": float(
                    arrays["entry_root_height_m"][index]
                ),
                "exit_root_height_m": float(
                    arrays["exit_root_height_m"][index]
                ),
            }
            for index in range(len(motions))
        ]
    else:
        endpoint_geometry = [
            event_endpoint_geometry_np(
                motion,
                floor_quantile=float(args.event_floor_quantile),
                window_frames=max(
                    2, int(round(5.0 * float(args.fps) / 30.0))
                ),
            )
            for motion in motions
        ]
    event_types = [str(item.get("event_type", "neutral_flow")) for item in items]
    families = [str(item.get("family_id", "")) for item in items]
    queries = [np.asarray(p.query, dtype=np.float32) for p in phrases]
    routing = precompute_music_routing(
        router,
        queries,
        motion_desc,
        device,
        phrase_sequences=phrase_sequences,
    )
    similarities = np.asarray(routing["similarity"], dtype=np.float32)
    compatibility_probabilities = np.asarray(
        routing["probabilities"], dtype=np.float32
    )
    action_compatibility = aggregate_action_compatibility(
        compatibility_probabilities, items
    )
    transition_choices = planner_bundle_lengths(args.planner_ckpt, fps=float(args.fps))

    beam = [CandidateState(0.0, [], [], [])]
    for slot, phrase in enumerate(phrases):
        predicted_duration = float(predictions["durations"][slot])
        desired_activity = float(predictions["activity"][slot])
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
                target_natural=target_natural,
                desired_activity=desired_activity,
                action_compatibility=action_compatibility[slot],
            )
            hierarchy_score, hierarchy_components = hierarchical_node_scores(hierarchy, hierarchy_query)
        base = (
            args.style_weight * style
            + args.quality_weight * quality
            + args.safety_weight * safety
            + args.music_weight * similarities[slot]
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
                candidate_recording = str(
                    items[idx].get("recording_uid", candidate_source)
                )
                same_recording = sum(
                    1
                    for previous in state.selected
                    if str(
                        items[previous].get(
                            "recording_uid",
                            items[previous].get(
                                "source_uid",
                                items[previous].get("source_id", "unknown"),
                            ),
                        )
                    ) == candidate_recording
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
                projected_recording_share = (same_recording + 1) / max(
                    1, projected_slots
                )
                if (
                    projected_slots >= int(args.min_source_share_slots)
                    and projected_source_share > float(args.max_source_share)
                ):
                    continue
                if (
                    projected_slots >= int(args.min_source_share_slots)
                    and projected_recording_share > float(args.max_recording_share)
                ):
                    continue
                if args.hard_family_unique and same_family > 0:
                    continue

                transition_len = 0
                transition_cost = 0.0
                boundary_velocity_penalty = 0.0
                boundary_acceleration_penalty = 0.0
                physical_edge_cost = 0.0
                physical_edge_meta: Dict[str, Any] = {}
                graph_edge_cost = 0.0
                graph_edge_meta: Dict[str, Any] = {}
                transition_meta: Dict[str, Any] = {}
                if state.selected:
                    previous = state.selected[-1]
                    transition_cost = intrinsic_transition_cost_from_arrays(
                        exit_pose[previous],
                        entry_pose[idx],
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
                    previous_geometry = endpoint_geometry[previous]
                    candidate_geometry = endpoint_geometry[idx]
                    floor_gap = abs(
                        float(previous_geometry["exit_floor_relative_m"])
                        - float(candidate_geometry["entry_floor_relative_m"])
                    )
                    root_height_gap = abs(
                        float(previous_geometry["exit_root_height_m"])
                        - float(candidate_geometry["entry_root_height_m"])
                    )
                    posture_gap = posture_state_distance(
                        str(items[previous].get("posture_exit", "unknown")),
                        str(items[idx].get("posture_entry", "unknown")),
                    )
                    contact_gap = float(candidate_boundary["contact_jump"])
                    root_velocity_gap = float(
                        candidate_boundary["root_velocity_jump_mps"]
                    )
                    physical_edge_cost = (
                        0.34
                        * min(
                            root_height_gap
                            / max(float(args.root_height_gap_reference_m), 1e-6),
                            float(args.boundary_penalty_cap),
                        )
                        + 0.20 * min(float(posture_gap) / 2.0, 2.0)
                        + 0.24
                        * min(
                            floor_gap
                            / max(float(args.floor_gap_reference_m), 1e-6),
                            float(args.boundary_penalty_cap),
                        )
                        + 0.18 * min(contact_gap, 1.0)
                        + 0.24
                        * min(
                            root_velocity_gap
                            / max(
                                float(args.root_velocity_jump_reference_mps),
                                1e-6,
                            ),
                            float(args.boundary_penalty_cap),
                        )
                    )
                    physical_edge_meta = {
                        "posture_root_height_gap_m": float(root_height_gap),
                        "posture_state_gap": int(posture_gap),
                        "posture_exit": str(
                            items[previous].get("posture_exit", "unknown")
                        ),
                        "posture_entry": str(
                            items[idx].get("posture_entry", "unknown")
                        ),
                        "floor_gap_m": float(floor_gap),
                        "contact_gap": float(contact_gap),
                        "root_velocity_jump_mps": float(root_velocity_gap),
                        "physical_edge_cost": float(physical_edge_cost),
                    }
                    strong_reset = float(phrase.boundary_accent_strength) >= float(
                        args.physical_edge_reset_accent
                    )
                    if bool(args.physical_edge_hard_prune):
                        reset_multiplier = 1.35 if strong_reset else 1.0
                        if (
                            root_height_gap
                            > float(args.root_height_gap_hard_m)
                            * reset_multiplier
                            or posture_gap
                            > int(args.posture_state_gap_hard)
                            or floor_gap
                            > float(args.floor_gap_hard_m)
                            * reset_multiplier
                            or root_velocity_gap
                            > float(args.root_velocity_jump_hard_mps)
                            * reset_multiplier
                            or contact_gap
                            > float(args.contact_gap_hard)
                            * reset_multiplier
                        ):
                            continue
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
                    - args.physical_edge_weight * physical_edge_cost
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
                    "target_motion_density": float(
                        phrase.target_motion_density
                    ),
                    "target_motion_density_source": str(
                        phrase.target_motion_density_source
                    ),
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
                    "router_compatibility_probability": float(
                        compatibility_probabilities[slot, idx]
                    ),
                    "router_uncertainty": float(routing["entropy"][slot]),
                    "router_confidence": float(routing["confidence"][slot]),
                    "router_ood": float(routing["ood"][slot]),
                    "router_architecture": str(routing["architecture"]),
                    "router_supervision_source": str(
                        routing["supervision_source"]
                    ),
                    "router_compatibility_is_ground_truth": False,
                    "action_compatibility_probs": {
                        label: float(action_compatibility[slot, action_index])
                        for action_index, label in enumerate(LOCAL_ACTION_LABELS)
                    },
                    "action_compatibility_top": str(
                        LOCAL_ACTION_LABELS[
                            int(np.argmax(action_compatibility[slot]))
                        ]
                    ),
                    "action_compatibility_supervision": "semantic_ot_teacher_x_weak_motion_kinematics",
                    "action_compatibility_is_ground_truth": False,
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
                    "hierarchy_semantic_contract": str(
                        hierarchy_query.get("semantic_contract", "disabled")
                    ),
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
                    "physical_edge_cost": float(physical_edge_cost),
                    "physical_edge_meta": physical_edge_meta,
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
                f"No Whole-Song Planner candidate for phrase {slot}. Increase candidate_top_k/graph_node_top_k or relax hard pruning."
            )
        # Preserve formal alternatives from the same beam prefix.  Boundary
        # closed-loop repair may reselect only among these siblings, which have
        # already passed this Scheduler's semantic, identity, source-share and
        # physical-edge constraints.  It must never rebuild alternatives with
        # the historical MSSD/AESD categorical retriever.
        siblings: Dict[tuple[int, ...], List[CandidateState]] = {}
        for candidate_state in expanded:
            prefix = tuple(candidate_state.selected[:-1])
            siblings.setdefault(prefix, []).append(candidate_state)
        formal_candidate_limit = max(
            1, int(getattr(args, "formal_candidate_top_k", 48))
        )
        for candidate_state in expanded:
            prefix = tuple(candidate_state.selected[:-1])
            ranked = sorted(
                siblings[prefix], key=lambda value: value.score, reverse=True
            )
            selected_index = int(candidate_state.selected[-1])
            ordered_states = [candidate_state] + [
                value
                for value in ranked
                if int(value.selected[-1]) != selected_index
            ]
            ordered_states = ordered_states[:formal_candidate_limit]
            candidate_state.parts[-1]["formal_candidate_event_uids"] = [
                str(items[int(value.selected[-1])]["event_uid"])
                for value in ordered_states
            ]
            candidate_state.parts[-1][
                "formal_candidate_router_probabilities"
            ] = [
                float(
                    value.parts[-1]["router_compatibility_probability"]
                )
                for value in ordered_states
            ]
            candidate_state.parts[-1]["formal_candidate_scheduler_scores"] = [
                float(value.score) for value in ordered_states
            ]
            candidate_state.parts[-1]["formal_candidate_contract"] = (
                "ctsr_weak_scheduler_siblings_v1"
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
    duration_model_bundle,
    planner_bundle,
    device,
    args,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    features, audio_meta = whole_song_features(
        audio_path,
        fps=args.fps,
        cache_dir=args.feature_dir,
        max_seconds=args.max_seconds,
        require_rhythm=bool(args.require_rhythm_features),
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
    router_sequence_frames = int(getattr(router, "sequence_frames", 0))
    if router_sequence_frames < 2:
        raise RuntimeError("CTSR-Weak checkpoint has no valid sequence_frames contract")
    temporal_sequences = phrase_feature_sequences(
        features, phrases, router_sequence_frames
    )
    if len(phrases) > args.max_phrases:
        raise RuntimeError(
            f"{audio_path}: detected {len(phrases)} event slots, above --max_phrases={args.max_phrases}. "
            "Increase max_phrases or max_single_event_seconds."
        )
    # CTSR probabilities are the only music--motion semantic evidence.  The
    # hierarchy receives no second externally pretrained embedding.
    semantic_meta = {
        "schema": "librosa12d_ctsr_router_only_v1",
        "external_pretrained_model": False,
        "used_for_routing": False,
    }
    predictions = planner_predictions(phrases, planner_bundle, device, fps=float(args.fps))
    selected_state = choose_events(
        phrases,
        temporal_sequences,
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

    schedule_rows: List[Dict[str, Any]] = []
    for slot, part in enumerate(selected_state.parts):
        merged = dict(part)
        if slot < len(slot_expansion.get("slot_meta", [])):
            merged["slot_meta"] = slot_expansion["slot_meta"][slot]
        merged["allocated_content_len"] = int(allocation["content_lengths"][slot])
        merged["allocated_phrase_total"] = int(
            allocation["phrase_total_lengths"][slot]
        )
        merged["time_warp_ratio"] = float(allocation["warp_ratios"][slot])
        schedule_rows.append(merged)
    hard_constraint_report = assert_schedule_hard_constraints(
        schedule_rows,
        max_pose_hold_ratio=float(args.max_pose_hold_ratio),
        max_single_source_ratio=float(args.max_source_share),
        max_single_recording_ratio=float(args.max_recording_share),
        min_unique_events=int(args.min_unique_events),
        min_core_frame_ratio=float(args.min_core_frame_ratio),
    )

    contents: List[np.ndarray] = []
    resampling_reports: List[Dict[str, Any]] = []
    stage_cursor_xz = np.zeros((2,), dtype=np.float32)
    for slot, (idx, target_len) in enumerate(zip(selected_state.selected, allocation["content_lengths"])):
        content, report = resample_event(
            motions[idx],
            int(target_len),
            duration_model_bundle,
            device,
            fps=float(args.fps),
            min_turn_angle=args.duration_model_min_turn_angle,
            min_peak_dps=args.duration_model_min_peak_dps,
        )
        # Edge damping changes Root-Y and joint rotations, so floor
        # canonicalization must be the final event-local geometry operation
        # before stage composition and transition construction.
        content = dampen_event_edges(
            content,
            args.edge_damping_frames,
            args.edge_damping_strength,
        )
        content, root_contract = canonicalize_event_root_np(
            content,
            target_floor_y=float(args.stage_floor_y),
            floor_quantile=float(args.event_floor_quantile),
            max_floor_penetration_m=float(
                args.event_max_floor_penetration_m
            ),
        )
        landing = np.zeros(2)
        landing_report = {}
        if contents:
            landing,landing_report = duration_displacement(contents[-1],content,int(transition_lengths[slot]),
                float(args.fps),float(args.transition_root_horizontal_speed_cap_mps))
        content, stage_contract = compose_event_root_xz_np(
            content,
            stage_cursor_xz + landing,
        )
        stage_cursor_xz = content[-1, [ROOT_X, ROOT_Z]].astype(np.float32)
        contents.append(content)
        resampling_reports.append(
            {
                **report,
                "root_trajectory_contract": {
                    **root_contract,
                    **stage_contract,
                    "policy": INBETWEEN_PROTOCOL,
                    "landing_displacement_xz_m":landing.tolist(),
                    "landing":landing_report,
                },
            }
        )

    pieces: List[np.ndarray] = []
    boundary_reports: List[Dict[str, Any]] = []
    for slot, content in enumerate(contents):
        if slot > 0:
            k = int(transition_lengths[slot])
            rough = make_so3_transition(
                contents[slot - 1],
                content,
                k,
                fps=float(args.fps),
                angular_speed_cap_radps=float(
                    args.transition_angular_speed_cap_radps
                ),
                root_horizontal_speed_cap_mps=float(
                    args.transition_root_horizontal_speed_cap_mps
                ),
                root_vertical_speed_cap_mps=float(
                    args.transition_root_vertical_speed_cap_mps
                ),
                root_tangent_margin_m=float(args.transition_root_tangent_margin_m),
            )
            transition = refine_transition(
                transition_bundle,
                rough,
                contents[slot - 1][-1],
                content[0],
                np.asarray(phrases[slot].query, dtype=np.float32),
                device,
            )
            transition = enforce_yaw_safe_transition(
                transition,
                contents[slot - 1],
                content,
                canonical_root=rough,
            )
            # Learned transition heads operate in Euclidean 6D coordinates.
            # Canonicalize every joint once before FK floor/contact audits so
            # the saved bridge and the audited bridge are the same rotations.
            transition = project_motion_rotations_np(transition)
            transition, floor_projection = project_transition_floor_np(
                transition,
                target_floor_y=float(args.stage_floor_y),
                clearance_m=float(args.transition_floor_clearance_m),
                smoothing_frames=int(args.transition_floor_smoothing_frames),
            )
            transition, contact_rebuild = recompute_transition_contacts_np(
                transition,
                fps=float(args.fps),
                floor_y=float(args.stage_floor_y),
                left_contact=contents[slot - 1][-1, CONTACT],
                right_contact=content[0, CONTACT],
                ramp_seconds=float(args.transition_contact_ramp_seconds),
            )
            metrics = boundary_metrics(
                contents[slot - 1], content, fps=float(args.fps)
            )
            metrics["transition_len"] = k
            metrics["transition_meta"] = selected_state.parts[slot].get("transition_meta", {})
            metrics["transition_root_contract"] = {
                "mode": "endpoint_velocity_aware_so3_root_hermite",
                "preserves_root_xz": True,
                "stage_floor_y_m": float(args.stage_floor_y),
                "floor_projection": floor_projection,
                "contact_rebuild": contact_rebuild,
            }
            boundary_reports.append(metrics)
            pieces.append(transition)
        pieces.append(content)

    motion = np.concatenate(pieces, axis=0).astype(np.float32)
    if len(motion) != len(features):
        raise AssertionError(
            f"Whole-Song Planner output length mismatch: generated={len(motion)} music_frames={len(features)}. "
            "No pad/trim fallback is permitted."
        )
    if args.start_pose:
        start_path = Path(args.start_pose)
        if start_path.is_file():
            motion = apply_start_anchor(
                motion,
                np.load(start_path).astype(np.float32).reshape(-1),
                args.start_anchor_blend,
            )

    report = {
        "version": "whole_song_music_dominant_whole_song_choreorag",
        "audio": str(audio_path),
        "audio_meta": audio_meta,
        "rotation_contract": {
            "motion_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
            "event_index_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
            "duration_checkpoint_rot6d_layout": duration_model_bundle.get("rot6d_layout"),
            "transition_checkpoint_rot6d_layout": (
                transition_bundle.get("rot6d_layout")
                if transition_bundle is not None
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
        "schedule": schedule_rows,
        "music_independent_hard_constraints": hard_constraint_report,
        "boundary_metrics": boundary_reports,
        "timing_policy": {
            "hierarchical_retrieval": bool(args.hierarchical_retrieval),
            "graph_scheduler": bool(args.graph_scheduler),
            "hierarchy_index_npz": str(args.hierarchy_index_npz),
            "hierarchy_semantic_contract": str(
                getattr(args, "hierarchy_semantic_contract", "unspecified")
            ),
            "hierarchy_weight": float(args.hierarchy_weight),
            "require_rhythm_features": bool(args.require_rhythm_features),
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
            "physical_edge_weight": float(args.physical_edge_weight),
            "physical_edge_hard_prune": bool(args.physical_edge_hard_prune),
            "root_height_gap_reference_m": float(args.root_height_gap_reference_m),
            "root_height_gap_hard_m": float(args.root_height_gap_hard_m),
            "posture_state_gap_hard": int(args.posture_state_gap_hard),
            "floor_gap_reference_m": float(args.floor_gap_reference_m),
            "floor_gap_hard_m": float(args.floor_gap_hard_m),
            "root_velocity_jump_reference_mps": float(
                args.root_velocity_jump_reference_mps
            ),
            "edge_damping_frames": int(args.edge_damping_frames),
            "edge_damping_strength": float(args.edge_damping_strength),
            "root_trajectory_contract": {
                "policy": (
                    "event_first_xz_localization_then_previous_endpoint_composition"
                ),
                "start_anchor_policy": "single_global_xz_translation",
                "stage_floor_y_m": float(args.stage_floor_y),
                "event_floor_quantile": float(args.event_floor_quantile),
                "transition_mode": "endpoint_velocity_aware_so3_root_hermite",
                "transition_floor_clearance_m": float(
                    args.transition_floor_clearance_m
                ),
                "transition_contact_ramp_seconds": float(
                    args.transition_contact_ramp_seconds
                ),
            },
            "multi_event_phrases": bool(args.multi_event_phrases),
            "lock_music_boundaries": bool(args.lock_music_boundaries),
            "max_single_event_seconds": float(args.max_single_event_seconds),
            "calm_max_single_event_seconds": float(args.calm_max_single_event_seconds),
            "anti_static_weight": float(args.anti_static_weight),
            "max_pose_hold_ratio": float(args.max_pose_hold_ratio),
            "max_single_source_ratio": float(args.max_source_share),
            "max_single_recording_ratio": float(args.max_recording_share),
            "min_unique_events": int(args.min_unique_events),
            "min_core_frame_ratio": float(args.min_core_frame_ratio),
        },
    }
    for slot, resampling in enumerate(resampling_reports):
        report["schedule"][slot]["resampling"] = resampling
    return motion, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--music", action="append", default=[])
    parser.add_argument("--music_glob", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--duration_model_ckpt", required=True)
    parser.add_argument("--planner_ckpt", required=True)
    parser.add_argument("--transition_ckpt", default="")
    parser.add_argument("--hierarchy_index_npz", default="")
    parser.add_argument("--hyperbolic_ckpt", default="")
    parser.add_argument("--feature_dir", default="")
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
    parser.add_argument(
        "--formal_candidate_top_k",
        type=int,
        default=48,
        help="Audited CTSR Scheduler sibling candidates retained for boundary repair.",
    )
    parser.add_argument("--style_weight", type=float, default=1.35)
    parser.add_argument("--quality_weight", type=float, default=0.65)
    parser.add_argument("--safety_weight", type=float, default=0.35)
    parser.add_argument("--music_weight", type=float, default=0.90)
    parser.add_argument("--duration_weight", type=float, default=0.45)
    parser.add_argument("--planner_duration_weight", type=float, default=0.15)
    parser.add_argument("--activity_weight", type=float, default=0.25)
    parser.add_argument("--hierarchical_retrieval", type=_bool_arg, default=True)
    parser.add_argument("--hierarchy_weight", type=float, default=0.55)
    parser.add_argument("--require_rhythm_features", type=_bool_arg, default=False)
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
    parser.add_argument("--physical_edge_weight", type=float, default=0.55)
    parser.add_argument("--physical_edge_hard_prune", type=_bool_arg, default=True)
    parser.add_argument("--physical_edge_reset_accent", type=float, default=0.82)
    parser.add_argument("--root_height_gap_reference_m", type=float, default=0.18)
    parser.add_argument("--root_height_gap_hard_m", type=float, default=0.55)
    parser.add_argument("--posture_state_gap_hard", type=int, default=2)
    parser.add_argument("--floor_gap_reference_m", type=float, default=0.08)
    parser.add_argument("--floor_gap_hard_m", type=float, default=0.20)
    parser.add_argument(
        "--root_velocity_jump_reference_mps", type=float, default=0.80
    )
    parser.add_argument("--root_velocity_jump_hard_mps", type=float, default=2.0)
    parser.add_argument("--contact_gap_hard", type=float, default=0.75)
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
    parser.add_argument(
        "--max_source_share",
        type=float,
        default=DEFAULT_MAX_SINGLE_SOURCE_RATIO,
    )
    parser.add_argument(
        "--max_recording_share",
        type=float,
        default=DEFAULT_MAX_SINGLE_SOURCE_RATIO,
    )
    parser.add_argument("--min_source_share_slots", type=int, default=6)
    parser.add_argument(
        "--max_pose_hold_ratio",
        type=float,
        default=DEFAULT_MAX_POSE_HOLD_RATIO,
    )
    parser.add_argument(
        "--min_unique_events", type=int, default=DEFAULT_MIN_UNIQUE_EVENTS
    )
    parser.add_argument(
        "--min_core_frame_ratio",
        type=float,
        default=DEFAULT_MIN_CORE_FRAME_RATIO,
    )
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
    parser.add_argument("--stage_floor_y", type=float, default=0.0)
    parser.add_argument("--event_floor_quantile", type=float, default=5.0)
    parser.add_argument("--event_max_floor_penetration_m", type=float, default=0.005)
    parser.add_argument(
        "--transition_angular_speed_cap_radps", type=float, default=8.0
    )
    parser.add_argument(
        "--transition_root_horizontal_speed_cap_mps", type=float, default=1.5
    )
    parser.add_argument(
        "--transition_root_vertical_speed_cap_mps", type=float, default=0.9
    )
    parser.add_argument(
        "--transition_root_tangent_margin_m", type=float, default=0.12
    )
    parser.add_argument(
        "--transition_floor_clearance_m", type=float, default=0.002
    )
    parser.add_argument(
        "--transition_floor_smoothing_frames", type=int, default=5
    )
    parser.add_argument(
        "--transition_contact_ramp_seconds", type=float, default=4.0 / 30.0
    )
    parser.add_argument("--transition_yaw_limit_dps", type=float, default=220.0)
    parser.add_argument("--yaw_transition_safety_factor", type=float, default=1.90)
    parser.add_argument("--pose_jump_reference", type=float, default=0.120)
    parser.add_argument("--velocity_jump_reference_radps", type=float, default=0.30)
    parser.add_argument("--acceleration_jump_reference_radps2", type=float, default=16.20)
    parser.add_argument("--physical_pose_frames", type=float, default=8.0)
    parser.add_argument("--physical_velocity_frames", type=float, default=10.0)
    parser.add_argument("--physical_acceleration_frames", type=float, default=8.0)
    parser.add_argument("--physical_contact_frames", type=float, default=8.0)
    parser.add_argument("--duration_model_min_turn_angle", type=float, default=10.0)
    parser.add_argument("--duration_model_min_peak_dps", type=float, default=14.0)
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
        "transition_floor_smoothing_frames",
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
    if index_rates != [float(args.fps)]:
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
    hierarchy = load_or_build_hierarchy(
        arrays,
        items,
        args.hierarchy_index_npz,
        hyperbolic_ckpt=args.hyperbolic_ckpt,
    )
    args.hierarchy_semantic_contract = str(
        np.asarray(
            hierarchy.get(
                "hierarchy_semantic_contract", np.asarray(["unspecified"])
            )
        ).reshape(-1)[0]
    )
    motions = [
        load_motion(resolve_event_motion_path(item, index_json, metadata=metadata))
        for item in items
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_scheduler_checkpoint(
        args.router_ckpt,
        "Router",
        float(args.fps),
        metadata["event_db_contract"],
        args.index_json,
        args.duration_index_npz,
    )
    validate_scheduler_checkpoint(
        args.duration_model_ckpt,
        "Duration",
        float(args.fps),
        metadata["event_db_contract"],
        args.index_json,
        args.duration_index_npz,
    )
    validate_scheduler_checkpoint(
        args.planner_ckpt,
        "Planner",
        float(args.fps),
        metadata["event_db_contract"],
        args.index_json,
        args.duration_index_npz,
    )
    router = load_router_checkpoint(args.router_ckpt, device=device)
    if str(getattr(router, "architecture", "")) != "ctsr_weak_temporal_v1":
        raise RuntimeError(
            "Formal fresh-audio scheduling requires architecture=ctsr_weak_temporal_v1"
        )
    raw_router_checkpoint = torch.load(
        args.router_ckpt, map_location="cpu", weights_only=False
    )
    if not isinstance(raw_router_checkpoint, dict):
        raise RuntimeError("Formal Router checkpoint is not a mapping")
    assert_formal_router_scientific_contract(raw_router_checkpoint)
    raw_planner_checkpoint = torch.load(
        args.planner_ckpt, map_location="cpu", weights_only=False
    )
    if not isinstance(raw_planner_checkpoint, dict):
        raise RuntimeError("Formal Planner checkpoint is not a mapping")
    assert_formal_planner_scientific_contract(raw_planner_checkpoint)
    local_action_contract = metadata.get("local_action_contract")
    if not isinstance(local_action_contract, dict):
        raise RuntimeError(
            "Formal CTSR-Weak requires a Generation index with local_action_contract"
        )
    if (
        bool(local_action_contract.get("is_ground_truth", True))
        or bool(
            local_action_contract.get(
                "dance_theme_used_as_local_action_truth", True
            )
        )
        or not bool(local_action_contract.get("multi_label", False))
    ):
        raise RuntimeError(
            f"Invalid formal local-action evidence contract: {local_action_contract}"
        )
    transition_bundle = load_optional_transition(
        args.transition_ckpt,
        device,
        fps=float(args.fps),
    )
    duration_model_bundle = load_duration_checkpoint(args.duration_model_ckpt, device=device)
    planner_bundle = load_planner_checkpoint(args.planner_ckpt, device=device)

    summary = {
        "version": "whole_song_music_dominant_whole_song_choreorag",
        "rotation_contract": {
            "motion_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
            "event_index_rot6d_layout": metadata["rot6d_layout"],
            "duration_checkpoint_rot6d_layout": duration_model_bundle.get("rot6d_layout"),
            "transition_checkpoint_rot6d_layout": (
                transition_bundle.get("rot6d_layout")
                if transition_bundle is not None
                else None
            ),
        },
        "planner_ckpt": args.planner_ckpt,
        "router_ckpt": args.router_ckpt,
        "router_contract": {
            "architecture": str(getattr(router, "architecture", "")),
            "supervision_source": str(
                getattr(router, "supervision_source", "unknown")
            ),
            "is_ground_truth": False,
            "categorical_event_compatibility": False,
        },
        "duration_model_ckpt": args.duration_model_ckpt,
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
            duration_model_bundle,
            planner_bundle,
            device,
            args,
        )
        key = path.stem
        npy_path = out_dir / f"{key}.whole_song.npy"
        report_path = out_dir / f"{key}.whole_song.schedule_report.json"
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

    summary_path = out_dir / "WHOLE_SONG_SUMMARY.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
