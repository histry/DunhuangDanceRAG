#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transition Duration Predictor (DPN-lite) for V20 Event-Graph ChoreoRAG."""
from __future__ import annotations

import torch
import torch.nn as nn

TRANSITION_BINS = [4, 6, 8, 10, 12, 16, 20]


class TransitionDurationPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4, dropout: float = 0.1, num_bins: int = len(TRANSITION_BINS)):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(dim, num_bins)
        self.regressor = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x.float())
        logits = self.classifier(h)
        length_raw = self.regressor(h).squeeze(-1)
        return {"logits": logits, "length_raw": length_raw}

    @torch.no_grad()
    def predict_length(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward(x)
        idx = out["logits"].argmax(dim=-1)
        bins = torch.as_tensor(TRANSITION_BINS, device=x.device, dtype=torch.float32)
        return bins[idx]
