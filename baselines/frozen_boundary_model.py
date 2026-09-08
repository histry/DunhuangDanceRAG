"""Frozen context Transformer baseline; no checkpoint training in this module.

This is a repository baseline, not a claimed reproduction of an external paper.
Inputs include observed context, analytic bridge and mask, never hidden targets.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ContextTransformer(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        self.input = nn.Linear(153, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim*4,
                                       dropout=0.0, batch_first=True), layers)
        self.output = nn.Linear(hidden_dim,147)

    def forward(self, bridge, mask):
        phase = torch.linspace(-1,1,bridge.shape[1],device=bridge.device)
        phase = phase[None,:,None].expand(bridge.shape[0],-1,-1)
        feature = torch.cat((bridge,mask[:,:,None],phase),dim=-1)
        return self.output(self.encoder(self.input(feature)))


def infer(path, bridge, span, recording_uid):
    from motion_geometry.heading import project_rot6d_np
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['schema']!='paper1_frozen_context_transformer_v1':
        raise ValueError('Unsupported boundary Transformer checkpoint')
    if not recording_uid or recording_uid in checkpoint['training_recording_uids']:
        raise ValueError('Missing heldout recording identity or training overlap')
    model = ContextTransformer(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    a,b = span
    mask = np.zeros(len(bridge),dtype=np.float32)
    mask[a:b] = 1
    with torch.inference_mode():
        delta = model(torch.as_tensor(bridge[None],dtype=torch.float32),
                      torch.as_tensor(mask[None]))[0].numpy()
    candidate = bridge.copy()
    candidate[a:b,4:] += delta[a:b]
    # Projection is part of this baseline, preceding all independent audits.
    candidate[a:b] = project_rot6d_np(candidate[a:b])
    return candidate
