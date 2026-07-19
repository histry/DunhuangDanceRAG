#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned turn-aware temporal pace refiner for V22.

The model reconstructs a natural-speed Dunhuang turn from a synthetically
compressed (too-fast) turn window.  It is deliberately conservative:
contacts and root X/Z are copied exactly, edits are restricted by a soft mask,
and the output residual head is zero-initialized.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


class FiLMResidualBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 4),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.cond(cond).chunk(4, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + gamma1[..., None]) + beta1[..., None]
        h = self.conv1(F.silu(h))
        h = self.dropout(h)
        h = self.norm2(h)
        h = h * (1.0 + gamma2[..., None]) + beta2[..., None]
        h = self.conv2(F.silu(h))
        return x + h


class V22TurnPaceRefiner(nn.Module):
    """Conservative local residual model for too-fast turn events.

    Inputs:
      motion      [B,T,151] compressed/too-fast motion window
      edit_mask   [B,T] soft edit region
      condition   [B,C] 12D phrase query + turn statistics

    Output:
      refined motion with exact contacts and root X/Z preservation.
    """

    def __init__(
        self,
        motion_dim: int = 151,
        condition_dim: int = 17,
        hidden_dim: int = 256,
        residual_scale: float = 0.22,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)

        # motion + first-order velocity + mask + normalized time
        input_dim = motion_dim * 2 + 2
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        dilations = (1, 2, 4, 8, 16, 8, 4, 2)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock1D(hidden_dim, hidden_dim, d, dropout=dropout) for d in dilations]
        )
        self.output_norm = nn.GroupNorm(8, hidden_dim)
        self.output_projection = nn.Conv1d(hidden_dim, motion_dim, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        motion: torch.Tensor,
        edit_mask: torch.Tensor,
        condition: torch.Tensor,
        strength: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[-1] != self.motion_dim:
            raise ValueError(f"motion must be [B,T,{self.motion_dim}], got {tuple(motion.shape)}")
        if edit_mask.ndim == 3 and edit_mask.shape[-1] == 1:
            edit_mask = edit_mask[..., 0]
        if edit_mask.ndim != 2:
            raise ValueError(f"edit_mask must be [B,T], got {tuple(edit_mask.shape)}")
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must be [B,{self.condition_dim}], got {tuple(condition.shape)}"
            )

        b, t, _ = motion.shape
        velocity = torch.zeros_like(motion)
        if t > 1:
            velocity[:, 1:] = motion[:, 1:] - motion[:, :-1]
            velocity[:, 0] = velocity[:, 1]
        time = torch.linspace(0.0, 1.0, t, device=motion.device, dtype=motion.dtype)
        time = time[None, :, None].expand(b, -1, -1)
        mask_channel = edit_mask[..., None].to(dtype=motion.dtype)
        x = torch.cat([motion, velocity, mask_channel, time], dim=-1).transpose(1, 2)

        h = self.input_projection(x)
        cond = self.condition_projection(condition)
        for block in self.blocks:
            h = block(h, cond)
        residual = self.output_projection(F.silu(self.output_norm(h))).transpose(1, 2)

        if not torch.is_tensor(strength):
            strength_tensor = torch.tensor(float(strength), device=motion.device, dtype=motion.dtype)
        else:
            strength_tensor = strength.to(device=motion.device, dtype=motion.dtype)
        while strength_tensor.ndim < 3:
            strength_tensor = strength_tensor.unsqueeze(-1)

        residual = residual * self.residual_scale * strength_tensor
        out = motion + residual * mask_channel

        # Hard conservative contracts without in-place autograd edits.
        predicted_root = out[..., 4:7]
        conservative_root = torch.stack(
            [motion[..., ROOT_X], predicted_root[..., 1], motion[..., ROOT_Z]],
            dim=-1,
        )

        # Project all predicted 6D rotations back to valid rotation matrices.
        rot = out[..., ROT].reshape(b, t, 24, 6)
        out_rot = matrix_to_rotation_6d(rotation_6d_to_matrix(rot)).reshape(b, t, 144)
        out = torch.cat([motion[..., CONTACT], conservative_root, out_rot], dim=-1)

        # Exact window endpoints, preventing drift at splice boundaries.
        endpoint_mask = torch.zeros((1, t, 1), device=motion.device, dtype=motion.dtype)
        endpoint_mask[:, 0] = 1.0
        endpoint_mask[:, -1] = 1.0
        out = out * (1.0 - endpoint_mask) + motion * endpoint_mask
        return out


def load_turn_pace_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    model = V22TurnPaceRefiner(
        motion_dim=int(config.get("motion_dim", 151)),
        condition_dim=int(config.get("condition_dim", 17)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        residual_scale=float(config.get("residual_scale", 0.22)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return {
        "model": model,
        "config": config,
        "condition_lo": checkpoint.get("condition_lo"),
        "condition_hi": checkpoint.get("condition_hi"),
        "epoch": checkpoint.get("epoch", -1),
        "val_loss": checkpoint.get("val_loss", float("inf")),
    }
