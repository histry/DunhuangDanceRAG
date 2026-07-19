#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Endpoint-conditioned transition refiner for V20.

The model refines a short rough transition between an exit pose and an entry pose.
It is small by design: it should repair only the boundary transition, not rewrite
retrieved Dunhuang motion events.
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
    def __init__(self, dim: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / max(dim, 1)))
        pe[:, 0::2] = torch.sin(position * div)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]].to(device=x.device, dtype=x.dtype)


class EndpointTransitionRefiner(nn.Module):
    def __init__(self, motion_dim: int = 151, music_dim: int = 12, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4, dropout: float = 0.1, residual_scale: float = 0.18):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.music_dim = int(music_dim)
        self.residual_scale = float(residual_scale)
        # rough motion + exit pose + entry pose + music event + valid mask
        in_dim = motion_dim + motion_dim * 2 + music_dim + 1
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos = SinusoidalPositionEncoding(hidden_dim, max_len=256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, motion_dim)
        gate = torch.ones(motion_dim)
        gate[0:4] = 0.0
        gate[ROOT_X] = 0.0
        gate[ROOT_Z] = 0.0
        gate[ROOT_Y] = 0.35
        gate[7:] = 1.0
        self.register_buffer("channel_gate", gate.view(1, 1, motion_dim), persistent=False)

    def forward(self, rough: torch.Tensor, exit_pose: torch.Tensor, entry_pose: torch.Tensor, music_event: Optional[torch.Tensor] = None, valid_mask: Optional[torch.Tensor] = None, return_residual: bool = False):
        if rough.ndim != 3 or rough.shape[-1] != self.motion_dim:
            raise ValueError(f"rough must be [B,K,{self.motion_dim}], got {tuple(rough.shape)}")
        B, K, C = rough.shape
        if exit_pose.ndim == 2:
            exit_pose = exit_pose[:, None].expand(-1, K, -1)
        if entry_pose.ndim == 2:
            entry_pose = entry_pose[:, None].expand(-1, K, -1)
        if music_event is None:
            music_event = torch.zeros(B, K, self.music_dim, device=rough.device, dtype=rough.dtype)
        elif music_event.ndim == 2:
            music_event = music_event[:, None].expand(-1, K, -1)
        if valid_mask is None:
            valid_mask = torch.ones(B, K, 1, device=rough.device, dtype=rough.dtype)
        elif valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        music_event = music_event.to(device=rough.device, dtype=rough.dtype)
        valid_mask = valid_mask.to(device=rough.device, dtype=rough.dtype)
        x = torch.cat([rough, exit_pose.to(rough.dtype), entry_pose.to(rough.dtype), music_event, valid_mask], dim=-1)
        h = self.pos(self.input_proj(x))
        h = self.encoder(h)
        raw = torch.tanh(self.out(self.out_norm(h))) * self.residual_scale
        residual = raw * self.channel_gate.to(dtype=rough.dtype) * valid_mask
        refined = rough + residual
        refined = refined.clone()
        refined[:, :, 0:4] = rough[:, :, 0:4]
        refined[:, :, ROOT_X] = rough[:, :, ROOT_X]
        refined[:, :, ROOT_Z] = rough[:, :, ROOT_Z]
        # First/last frame should stay close to endpoints if included by caller.
        if return_residual:
            return refined, residual
        return refined
