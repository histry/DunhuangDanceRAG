#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phrase-sequence planner for whole-song cultural dance choreography.

This replacement keeps the original planner architecture intact, but changes
the default transition vocabulary from short fixed cuts to music/physics-ready
durations. Old checkpoints remain loadable because their saved config still
overrides this fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from support.common import EVENT_TYPES


MUSIC_DOMINANT_TRANSITION_LENGTHS: tuple[int, ...] = (12, 16, 20, 24, 30, 36, 42, 48)


class V26WholeSongPlanner(nn.Module):
    """Predict motion-event type, natural duration and transition class per phrase."""

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.15,
        num_event_types: int = len(EVENT_TYPES),
        transition_lengths: tuple[int, ...] = MUSIC_DOMINANT_TRANSITION_LENGTHS,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_event_types = int(num_event_types)
        self.transition_lengths = tuple(int(x) for x in transition_lengths)
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
        self.event_head = nn.Linear(hidden_dim, num_event_types)
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
            "event_logits": self.event_head(hidden),
            "log_duration": log_duration,
            "duration_frames": torch.exp(log_duration).clamp(8.0, 600.0),
            "transition_logits": self.transition_head(hidden),
            "activity": torch.sigmoid(self.activity_head(hidden).squeeze(-1)),
            "hidden": hidden,
        }


def load_v26_planner_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    model = V26WholeSongPlanner(
        feature_dim=int(config.get("feature_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 4)),
        num_heads=int(config.get("num_heads", 4)),
        dropout=float(config.get("dropout", 0.15)),
        num_event_types=int(config.get("num_event_types", len(EVENT_TYPES))),
        transition_lengths=tuple(config.get("transition_lengths", MUSIC_DOMINANT_TRANSITION_LENGTHS)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return {"model": model, "config": config, "checkpoint": checkpoint}


# Public, version-free API. The historical class alias remains only for
# checkpoint compatibility and is not used in public file names.
WholeSongPlanner = V26WholeSongPlanner
