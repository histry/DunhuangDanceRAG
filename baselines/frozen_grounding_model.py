"""Inference architecture for explicitly supplied frozen dual encoders.

The same projection architecture can represent a frozen contrastive baseline.
Its training objective is checkpoint provenance, never inferred from weights.
No optimizer, training loop or synthetic positive labels are provided here.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DualEncoder(nn.Module):
    def __init__(self, music_dim, motion_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.music = nn.Sequential(nn.Linear(music_dim,hidden_dim),nn.GELU(),
                                   nn.Linear(hidden_dim,embedding_dim))
        self.motion = nn.Sequential(nn.Linear(motion_dim,hidden_dim),nn.GELU(),
                                    nn.Linear(hidden_dim,embedding_dim))

    def forward(self, music, motion):
        return F.normalize(self.music(music),dim=-1) @ F.normalize(self.motion(motion),dim=-1).T


def frozen_scores(path, music, motion, query_songs):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['schema']!='paper2_frozen_dual_encoder_v1':
        raise ValueError('Unsupported dual encoder checkpoint')
    if checkpoint['training_objective']!='contrastive':
        raise ValueError('Contrastive baseline requires explicit objective provenance')
    if set(query_songs)&set(checkpoint['training_song_uids']):
        raise ValueError('Frozen encoder training/evaluation songs overlap')
    model = DualEncoder(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    with torch.inference_mode():
        score = model(torch.as_tensor(music,dtype=torch.float32),
                      torch.as_tensor(motion,dtype=torch.float32))
    result = score.cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite dual encoder output')
    return result
