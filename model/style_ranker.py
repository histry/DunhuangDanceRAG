#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight pairwise Dunhuang-style ranker over 64D motion embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


class V21StyleRanker(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_style_ranker(path: str | Path, device: torch.device | str = "cpu") -> V21StyleRanker:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config: Dict[str, Any] = dict(ckpt.get("config", {}))
    model = V21StyleRanker(
        input_dim=int(config.get("input_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.to(device).eval()
    return model
