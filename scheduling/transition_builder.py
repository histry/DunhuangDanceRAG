"""Deterministic and learned boundary-transition helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.transition_model import load_transition_checkpoint
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    convert_motion_rot6d_layout_np,
    normalize_rot6d_layout,
)
from support.scheduler_common import CONTACT, ROOT_Y, ROT


def load_optional_transition(path: str, device: torch.device):
    if not path:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    bundle = load_transition_checkpoint(checkpoint_path, device=device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    bundle["dpn_lo"] = np.asarray(
        checkpoint.get("dpn_lo", np.zeros((20,), dtype=np.float32)),
        dtype=np.float32,
    )
    bundle["dpn_hi"] = np.asarray(
        checkpoint.get("dpn_hi", np.ones((20,), dtype=np.float32)),
        dtype=np.float32,
    )
    bundle["rot6d_layout"] = normalize_rot6d_layout(
        bundle.get("rot6d_layout", ROT6D_LAYOUT_PYTORCH3D_ROW)
    )
    return bundle


def _to_checkpoint_layout(motion: np.ndarray, transition_bundle) -> np.ndarray:
    return convert_motion_rot6d_layout_np(
        motion,
        CANONICAL_ROT6D_LAYOUT,
        transition_bundle["rot6d_layout"],
    )


def _from_checkpoint_layout(motion: np.ndarray, transition_bundle) -> np.ndarray:
    return convert_motion_rot6d_layout_np(
        motion,
        transition_bundle["rot6d_layout"],
        CANONICAL_ROT6D_LAYOUT,
    )


def transition_feature(
    previous_motion: np.ndarray,
    next_motion: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    pose_difference = previous_motion[-1] - next_motion[0]
    previous_velocity = (
        previous_motion[-1] - previous_motion[-2]
        if len(previous_motion) > 1
        else np.zeros((151,), dtype=np.float32)
    )
    next_velocity = (
        next_motion[1] - next_motion[0]
        if len(next_motion) > 1
        else np.zeros((151,), dtype=np.float32)
    )
    boundary = np.asarray(
        [
            np.linalg.norm(pose_difference[ROT]) / np.sqrt(144.0),
            np.linalg.norm(previous_velocity[ROT] - next_velocity[ROT]) / np.sqrt(144.0),
            abs(float(pose_difference[ROOT_Y])),
            float(np.abs(pose_difference[CONTACT]).mean()),
            float(
                np.linalg.norm(previous_motion[-1, ROT] - previous_motion[0, ROT])
                / np.sqrt(144.0)
            ),
            float(
                np.linalg.norm(next_motion[-1, ROT] - next_motion[0, ROT])
                / np.sqrt(144.0)
            ),
            float(len(previous_motion) / 72.0),
            float(len(next_motion) / 72.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [boundary, np.asarray(query, dtype=np.float32).reshape(12)],
        axis=0,
    )


def rule_transition_length(
    music_event: str,
    next_event: str,
    query: np.ndarray,
) -> int:
    energy = float(query[0])
    if music_event in {"accent", "climax"} or next_event in {
        "high_tension",
        "arm_flourish",
    }:
        return 6 if energy > 0.65 else 8
    if music_event == "section_change" or next_event == "support_shift":
        return 12
    if music_event in {"calm_flow", "release"}:
        return 14
    return 10


def predict_transition_length(
    transition_bundle,
    previous_motion: np.ndarray,
    next_motion: np.ndarray,
    query: np.ndarray,
    music_event: str,
    next_event: str,
    device: torch.device,
) -> int:
    if transition_bundle is None:
        return rule_transition_length(music_event, next_event, query)
    feature = transition_feature(
        _to_checkpoint_layout(previous_motion, transition_bundle),
        _to_checkpoint_layout(next_motion, transition_bundle),
        query,
    )
    lower = transition_bundle["dpn_lo"]
    upper = transition_bundle["dpn_hi"]
    feature = np.clip((feature - lower) / (upper - lower + 1e-8), 0.0, 1.0)
    with torch.no_grad():
        logits = transition_bundle["dpn"](
            torch.from_numpy(feature.astype(np.float32)[None]).to(device)
        )
        index = int(logits.argmax(dim=-1).item())
    return int(transition_bundle["transition_lengths"][index])


def refine_transition(
    transition_bundle,
    rough: np.ndarray,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    query: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if transition_bundle is None or len(rough) < 2:
        return rough
    checkpoint_rough = _to_checkpoint_layout(rough, transition_bundle)
    checkpoint_start = _to_checkpoint_layout(start_pose, transition_bundle)
    checkpoint_end = _to_checkpoint_layout(end_pose, transition_bundle)
    with torch.no_grad():
        prediction = transition_bundle["refiner"](
            torch.from_numpy(checkpoint_rough[None]).to(device),
            torch.from_numpy(checkpoint_start[None]).to(device),
            torch.from_numpy(checkpoint_end[None]).to(device),
            torch.from_numpy(query[None]).to(device),
            torch.ones((1, len(rough)), dtype=torch.float32, device=device),
        )[0]
    return _from_checkpoint_layout(
        prediction.detach().cpu().numpy().astype(np.float32),
        transition_bundle,
    )
