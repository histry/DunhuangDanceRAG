"""Non-temporal Router baseline trained on the CTSR weak-teacher dataset."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE = "ctsr_mean_pool_mlp_baseline_v1"


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(value), dim=-1)


class MeanPoolMusicMotionRouter(nn.Module):
    """A capacity-controlled ablation with no temporal ordering mechanism."""

    def __init__(
        self,
        music_dim: int = 12,
        motion_dim: int = 12,
        hidden_dim: int = 128,
        latent_dim: int = 96,
        dropout: float = 0.1,
        init_temperature: float = 0.12,
    ) -> None:
        super().__init__()
        self.music_dim = int(music_dim)
        self.motion_dim = int(motion_dim)
        self.music_encoder = _Encoder(music_dim, hidden_dim, latent_dim, dropout)
        self.motion_encoder = _Encoder(motion_dim, hidden_dim, latent_dim, dropout)
        self.logit_scale = nn.Parameter(
            torch.tensor(float(math.log(1.0 / init_temperature)))
        )
        self.architecture = ARCHITECTURE
        self.supervision_source = "semantic_ot_teacher"
        self.sequence_frames = 0
        self.inference_temperature = float(init_temperature)
        self.feature_mean: list[float] | None = None
        self.feature_std: list[float] | None = None

    def encode_music(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim == 3:
            sequence = sequence.mean(dim=1)
        if sequence.ndim != 2 or sequence.shape[-1] != self.music_dim:
            raise ValueError(
                f"Mean-pool baseline expects [B,T,{self.music_dim}] or "
                f"[B,{self.music_dim}], got {tuple(sequence.shape)}"
            )
        return self.music_encoder(sequence)

    def encode_motion(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(descriptor)

    def forward(self, music: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        music_embedding = self.encode_music(music)
        motion_embedding = self.encode_motion(motion)
        return self.logit_scale.exp().clamp(max=100.0) * (
            music_embedding @ motion_embedding.transpose(-1, -2)
        )
