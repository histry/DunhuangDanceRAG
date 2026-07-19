#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional trainable dual encoder for music-query / motion-event matching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int = 12, hidden_dim: int = 128, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class V21MusicMotionRouter(nn.Module):
    def __init__(
        self,
        music_dim: int = 12,
        motion_dim: int = 12,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.music_dim = int(music_dim)
        self.motion_dim = int(motion_dim)
        self.music_encoder = MLPEncoder(music_dim, hidden_dim, latent_dim, dropout)
        self.motion_encoder = MLPEncoder(motion_dim, hidden_dim, latent_dim, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / init_temperature)))))

    def encode_music(self, x: torch.Tensor) -> torch.Tensor:
        return self.music_encoder(x)

    def encode_motion(self, x: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(x)

    def forward(self, music: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        m = self.encode_music(music)
        e = self.encode_motion(motion)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (m @ e.transpose(-1, -2))


def load_router_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> V21MusicMotionRouter:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config: Dict[str, Any] = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    model = V21MusicMotionRouter(
        music_dim=int(config.get("music_dim", 12)),
        motion_dim=int(config.get("motion_dim", 12)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 64)),
        dropout=float(config.get("dropout", 0.1)),
    )
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# Public, version-free API. The historical class alias remains only for
# checkpoint compatibility and is not used in public file names.
MusicMotionRouter = V21MusicMotionRouter
