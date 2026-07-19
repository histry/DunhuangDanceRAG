#!/usr/bin/env python3
"""
V16D Phrase Refiner 模型
输入：45~150帧 motion unit + optional onset embedding
输出：修正后的 motion
"""
import torch
import torch.nn as nn

class PhraseRefiner(nn.Module):
    def __init__(self, input_dim=151, hidden_dim=256, num_layers=3, use_onset=False, onset_dim=16):
        super().__init__()
        self.use_onset = use_onset
        input_total = input_dim + (onset_dim if use_onset else 0)
        self.encoder = nn.Sequential(
            nn.Linear(input_total, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers-1)]
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)
    def forward(self, x, onset=None):
        # x: [B,T,151], onset: [B,T,onset_dim]
        if self.use_onset and onset is not None:
            x = torch.cat([x, onset], dim=-1)
        B,T,_ = x.shape
        x_flat = x.view(B*T, -1)
        h = self.encoder(x_flat)
        out = self.decoder(h)
        return out.view(B,T,-1)
