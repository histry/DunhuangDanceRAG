#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current-protocol phrase planner for whole-song choreography.

The planner predicts continuous duration/activity and transition class only.
Local-action compatibility belongs to the temporal Router and formal candidate
sets; a categorical dance-event head would duplicate the archived BVH task.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

MUSIC_DOMINANT_TRANSITION_LENGTHS: tuple[int, ...] = (12, 16, 20, 24, 30, 36, 42, 48)
PLANNER_ARCHITECTURE = "ctsr_continuous_planner_v2"


class WholeSongPlanner(nn.Module):
    """Predict natural duration, transition class, and activity per phrase."""

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.15,
        transition_lengths: tuple[int, ...] = MUSIC_DOMINANT_TRANSITION_LENGTHS,
        fps: float = 30.0,
        min_duration_seconds: float = 8.0 / 30.0,
        max_duration_seconds: float = 20.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.transition_lengths = tuple(int(x) for x in transition_lengths)
        self.fps = float(fps)
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be finite and positive, got {fps!r}")
        self.duration_min_frames = float(min_duration_seconds) * self.fps
        self.duration_max_frames = float(max_duration_seconds) * self.fps
        if self.duration_min_frames <= 0.0 or self.duration_max_frames <= self.duration_min_frames:
            raise ValueError(
                "Planner duration limits must be positive and strictly ordered"
            )
        self.input_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.log_duration_head = nn.Linear(hidden_dim, 1)
        self.transition_head = nn.Linear(hidden_dim, len(self.transition_lengths))
        self.activity_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must be [B,K,{self.feature_dim}], got {tuple(features.shape)}")
        hidden = self.input_projection(features)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        log_duration = self.log_duration_head(hidden).squeeze(-1)
        return {
            "log_duration": log_duration,
            "duration_frames": torch.exp(log_duration).clamp(
                self.duration_min_frames, self.duration_max_frames
            ),
            "transition_logits": self.transition_head(hidden),
            "activity": torch.sigmoid(self.activity_head(hidden).squeeze(-1)),
            "hidden": hidden,
        }


def load_whole_song_planner_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if str(checkpoint.get("architecture", "")) != PLANNER_ARCHITECTURE:
        raise RuntimeError(
            "Planner checkpoint is not a current-protocol asset; historical "
            "checkpoints remain available only through the archive tag"
        )
    config = dict(checkpoint.get("config", {}))
    model = WholeSongPlanner(
        feature_dim=int(config.get("feature_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.15)),
        transition_lengths=tuple(config.get("transition_lengths", MUSIC_DOMINANT_TRANSITION_LENGTHS)),
        fps=float(config.get("fps", 30.0)),
        min_duration_seconds=float(config.get("min_duration_seconds", 8.0 / 30.0)),
        max_duration_seconds=float(config.get("max_duration_seconds", 20.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return {"model": model, "config": config, "checkpoint": checkpoint}

def load_planner_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    """Load a current-protocol planner checkpoint."""
    return load_whole_song_planner_checkpoint(path, device=device)
