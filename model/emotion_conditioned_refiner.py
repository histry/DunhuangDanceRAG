#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion-conditioned Seam Residual Refiner

设计目标：
1. 输入 V16C/V17 调度得到的稳定 150 帧 motion；
2. 输入逐帧音乐情感语义条件 [energy, onset, beat, arousal, tension, calmness 等]；
3. 只预测小残差，优先修接缝附近，避免重写整段动作；
4. 强制保留 contacts 和 root X/Z，避免原地约束被破坏。

输入：
    motion:    [B, T, 151]
    music:     [B, T, 8]
    seam_mask: [B, T, 1]，接缝附近为 1，其他为 0

输出：
    refined motion [B, T, 151]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / max(dim, 1)))
        pe[:, 0::2] = torch.sin(position * div)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 2, dim, kernel_size=kernel_size, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        # LayerNorm must be applied before transpose, because it normalizes the last dim C.
        h = self.norm(x)
        y = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return x + y


class EmotionConditionedSeamRefiner(nn.Module):
    def __init__(
        self,
        motion_dim: int = 151,
        music_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_scale: float = 0.20,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.music_dim = int(music_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)

        in_dim = motion_dim + music_dim + 1  # + seam mask
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos = SinusoidalPositionEncoding(hidden_dim, max_len=1024)

        self.local_blocks = nn.ModuleList([
            TemporalConvBlock(hidden_dim, dropout=dropout, kernel_size=5)
            for _ in range(max(1, num_layers // 2))
        ])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, num_layers - len(self.local_blocks)))

        self.music_film = nn.Sequential(
            nn.Linear(music_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, motion_dim)

        # channel gate: 默认不改 contacts 与 root X/Z，主要改 root_y + rotations。
        gate = torch.ones(motion_dim)
        gate[0:4] = 0.0
        gate[ROOT_X] = 0.0
        gate[ROOT_Z] = 0.0
        gate[ROOT_Y] = 0.35
        gate[7:] = 1.0
        self.register_buffer("channel_gate", gate.view(1, 1, motion_dim), persistent=False)

    def forward(
        self,
        motion: torch.Tensor,
        music: Optional[torch.Tensor] = None,
        seam_mask: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ):
        if motion.ndim != 3:
            raise ValueError(f"motion must be [B,T,C], got {tuple(motion.shape)}")
        B, T, C = motion.shape
        if C != self.motion_dim:
            raise ValueError(f"motion last dim should be {self.motion_dim}, got {C}")

        if music is None:
            music = torch.zeros(B, T, self.music_dim, device=motion.device, dtype=motion.dtype)
        if music.shape[:2] != motion.shape[:2]:
            raise ValueError(f"music shape {tuple(music.shape)} incompatible with motion {tuple(motion.shape)}")
        if music.shape[-1] != self.music_dim:
            raise ValueError(f"music last dim should be {self.music_dim}, got {music.shape[-1]}")

        if seam_mask is None:
            seam_mask = torch.ones(B, T, 1, device=motion.device, dtype=motion.dtype)
        if seam_mask.ndim == 2:
            seam_mask = seam_mask.unsqueeze(-1)
        seam_mask = seam_mask.to(device=motion.device, dtype=motion.dtype)

        x = torch.cat([motion, music, seam_mask], dim=-1)
        h = self.input_proj(x)
        h = self.pos(h)

        # 全局音乐条件通过 FiLM 调制，表达音乐情感语义对整段动作的影响。
        global_music = music.mean(dim=1)
        gamma_beta = self.music_film(global_music).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

        for block in self.local_blocks:
            h = block(h)
        h = self.global_encoder(h)

        raw_residual = torch.tanh(self.out(self.out_norm(h))) * self.residual_scale

        # seam-aware gate：接缝附近允许较强细化，非接缝只允许极小幅度 residual。
        local_gate = 0.15 + 0.85 * seam_mask
        residual = raw_residual * self.channel_gate.to(motion.dtype) * local_gate

        refined = motion + residual

        # 硬约束：contacts 与 root X/Z 保持输入，符合原地生成主线。
        refined = refined.clone()
        refined[:, :, 0:4] = motion[:, :, 0:4]
        refined[:, :, ROOT_X] = motion[:, :, ROOT_X]
        refined[:, :, ROOT_Z] = motion[:, :, ROOT_Z]

        if return_residual:
            return refined, residual
        return refined
