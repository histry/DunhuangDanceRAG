"""Temporal music-motion Router for the unpaired CTSR-Weak formal path."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=int(dilation),
            dilation=int(dilation),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        hidden = self.norm(values).transpose(1, 2)
        hidden = self.conv(hidden).transpose(1, 2)
        hidden = self.proj(F.silu(hidden))
        return residual + self.dropout(hidden)


class TemporalMusicEncoder(nn.Module):
    """Dilated TCN + Transformer encoder over complete Librosa phrase streams."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        latent_dim: int = 96,
        dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        self.input_dim = int(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.tcn = nn.ModuleList(
            [TemporalResidualBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4, 8)]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.attention_pool = nn.Linear(hidden_dim, 1)
        self.latent_projection = nn.Linear(hidden_dim, latent_dim)
        self.frame_decoder = nn.Linear(hidden_dim, input_dim)

    @staticmethod
    def _sinusoidal_position(length: int, width: int, device, dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(width, 1))
        )
        encoding = torch.zeros((length, width), device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * frequency)
        if width > 1:
            encoding[:, 1::2] = torch.cos(position * frequency[: encoding[:, 1::2].shape[1]])
        return encoding

    def frame_features(self, sequence: torch.Tensor) -> torch.Tensor:
        values = sequence
        if values.ndim == 2:
            values = values.unsqueeze(1)
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"Temporal music encoder expects [B,T,{self.input_dim}], got {tuple(values.shape)}"
            )
        hidden = self.input_projection(values)
        hidden = hidden + self._sinusoidal_position(
            hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype
        ).unsqueeze(0)
        for block in self.tcn:
            hidden = block(hidden)
        return self.output_norm(self.transformer(hidden))

    def pooled_embedding(self, frame_features: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.attention_pool(frame_features).squeeze(-1), dim=1)
        pooled = torch.sum(frame_features * attention.unsqueeze(-1), dim=1)
        return F.normalize(self.latent_projection(pooled), dim=-1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.pooled_embedding(self.frame_features(sequence))


class MotionControlEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(values), dim=-1)


class TemporalMusicMotionRouter(nn.Module):
    def __init__(
        self,
        music_dim: int = 12,
        motion_dim: int = 12,
        hidden_dim: int = 128,
        latent_dim: int = 96,
        dropout: float = 0.1,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        init_temperature: float = 0.12,
    ):
        super().__init__()
        self.music_dim = int(music_dim)
        self.motion_dim = int(motion_dim)
        self.music_encoder = TemporalMusicEncoder(
            input_dim=music_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )
        self.motion_encoder = MotionControlEncoder(
            motion_dim, hidden_dim, latent_dim, dropout
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(float(math.log(1.0 / init_temperature)))
        )
        self.temporal_order_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.architecture = "ctsr_weak_temporal_v1"
        self.supervision_source = "semantic_ot_teacher"
        self.sequence_frames = 0
        self.inference_temperature = float(init_temperature)
        self.feature_mean: list[float] | None = None
        self.feature_std: list[float] | None = None

    def encode_music(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.music_encoder(sequence)

    def encode_motion(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(descriptor)

    def reconstruct_music(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.music_encoder.frame_decoder(self.music_encoder.frame_features(sequence))

    def temporal_order_logits(
        self, first_embedding: torch.Tensor, second_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.temporal_order_head(
            torch.cat([first_embedding, second_embedding], dim=-1)
        )

    def forward(self, music: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        music_embedding = self.encode_music(music)
        motion_embedding = self.encode_motion(motion)
        return self.logit_scale.exp().clamp(max=100.0) * (
            music_embedding @ motion_embedding.transpose(-1, -2)
        )
