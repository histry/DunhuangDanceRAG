#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V21 transition duration predictor and endpoint-conditioned local refiner."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    normalize_rot6d_layout,
)


TRANSITION_LENGTHS = (4, 6, 8, 10, 12, 16)
NATIVE_ROT6D_LAYOUT = ROT6D_LAYOUT_PYTORCH3D_ROW


class V21TransitionDurationPredictor(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 192, num_classes: int = len(TRANSITION_LENGTHS), dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class V21EndpointTransitionRefiner(nn.Module):
    """Predict a small residual over a rough variable-length transition.

    The network preserves contacts and root X/Z by default. It refines root Y and
    joint rotations only, matching the conservative strategy that worked best in
    the V17B experiments.
    """

    def __init__(
        self,
        motion_dim: int = 151,
        music_dim: int = 12,
        hidden_dim: int = 256,
        residual_scale: float = 0.18,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.music_dim = int(music_dim)
        self.residual_scale = float(residual_scale)
        in_dim = motion_dim * 3 + music_dim + 1
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                )
                for _ in range(4)
            ]
        )
        self.out_proj = nn.Conv1d(hidden_dim, motion_dim, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        rough: torch.Tensor,
        start_pose: torch.Tensor,
        end_pose: torch.Tensor,
        music: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # rough [B,K,151], start/end [B,151], music [B,12] or [B,K,12]
        b, k, d = rough.shape
        if start_pose.ndim != 2 or end_pose.ndim != 2:
            raise ValueError("start_pose/end_pose must be [B,151]")
        start = start_pose[:, None, :].expand(-1, k, -1)
        end = end_pose[:, None, :].expand(-1, k, -1)
        if music.ndim == 2:
            music = music[:, None, :].expand(-1, k, -1)
        if music.shape[1] != k:
            music = F.interpolate(music.transpose(1, 2), size=k, mode="linear", align_corners=False).transpose(1, 2)
        t = torch.linspace(0.0, 1.0, k, device=rough.device, dtype=rough.dtype)[None, :, None].expand(b, -1, -1)
        x = torch.cat([rough, start, end, music, t], dim=-1).transpose(1, 2)
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
        residual = self.out_proj(h).transpose(1, 2) * self.residual_scale
        out = rough + residual
        # Conservative contract: do not rewrite contacts or root X/Z.
        out[..., 0:4] = rough[..., 0:4]
        out[..., 4] = rough[..., 4]
        out[..., 6] = rough[..., 6]
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[..., None]
            out = torch.where(mask > 0.5, out, rough)
        # Exact endpoints.
        out[:, 0] = start_pose
        out[:, -1] = end_pose
        return out


def load_transition_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    checkpoint_layout = normalize_rot6d_layout(
        checkpoint.get("rot6d_layout", config.get("rot6d_layout", NATIVE_ROT6D_LAYOUT))
    )
    if checkpoint_layout != NATIVE_ROT6D_LAYOUT:
        raise RuntimeError(
            "This historical transition architecture expects PyTorch3D-row "
            f"Rot6D, but checkpoint declares {checkpoint_layout!r}."
        )
    dpn = V21TransitionDurationPredictor(
        input_dim=int(config.get("dpn_input_dim", 20)),
        hidden_dim=int(config.get("dpn_hidden_dim", 192)),
        num_classes=len(config.get("transition_lengths", TRANSITION_LENGTHS)),
        dropout=float(config.get("dropout", 0.1)),
    )
    refiner = V21EndpointTransitionRefiner(
        motion_dim=int(config.get("motion_dim", 151)),
        music_dim=int(config.get("music_dim", 12)),
        hidden_dim=int(config.get("refiner_hidden_dim", 256)),
        residual_scale=float(config.get("residual_scale", 0.18)),
    )
    dpn.load_state_dict(checkpoint["dpn_state_dict"], strict=True)
    refiner.load_state_dict(checkpoint["refiner_state_dict"], strict=True)
    dpn.to(device).eval()
    refiner.to(device).eval()
    return {
        "dpn": dpn,
        "refiner": refiner,
        "config": config,
        "transition_lengths": tuple(config.get("transition_lengths", TRANSITION_LENGTHS)),
        "rot6d_layout": checkpoint_layout,
        "canonical_rot6d_layout": CANONICAL_ROT6D_LAYOUT,
    }


# Version-free class names for new code. Historical checkpoint state keys are
# intentionally unchanged.
TransitionDurationPredictor = V21TransitionDurationPredictor
EndpointTransitionRefiner = V21EndpointTransitionRefiner
