"""
Music-Motion Retrieval (MMR) encoders for EDGE.

This module implements a CLIP-style dual tower:
  AudioEncoder(audio_feature [B,T,803]) -> z_audio [B,D]
  MotionEncoder(motion [B,T,151])       -> z_motion [B,D]

Use train_mmr.py to train on paired AIST++ clips first, then use
adapt_mmr_motion_dunhuang.py for motion-domain adaptation on Dunhuang BVH.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MMRConfig:
    audio_dim: int = 803
    motion_dim: int = 151
    hidden_dim: int = 512
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_len: int = 512


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(f"sequence length {x.shape[1]} exceeds max_len {self.pe.shape[1]}")
        return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        out_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = SinusoidalPositionalEncoding(hidden_dim, max_len=max_len)
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
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,C]
        mask: optional [B,T] where True means valid token.
        """
        h = self.input_proj(x)
        h = self.pos(h)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            w = mask.to(dtype=h.dtype).unsqueeze(-1)
            pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)


class MusicMotionRetrievalModel(nn.Module):
    def __init__(self, config: Optional[MMRConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = MMRConfig(**kwargs)
        self.config = config
        self.audio_encoder = TemporalTransformerEncoder(
            input_dim=config.audio_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            out_dim=config.embed_dim,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        self.motion_encoder = TemporalTransformerEncoder(
            input_dim=config.motion_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            out_dim=config.embed_dim,
            dropout=config.dropout,
            max_len=config.max_len,
        )
        # log(1 / 0.07), CLIP-style initialization.
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_audio(self, audio: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.audio_encoder(audio, mask=mask)

    def encode_motion(self, motion: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.motion_encoder(motion, mask=mask)

    def forward(self, audio: torch.Tensor, motion: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_audio = self.encode_audio(audio)
        z_motion = self.encode_motion(motion)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * z_audio @ z_motion.t()
        return {
            "z_audio": z_audio,
            "z_motion": z_motion,
            "logits": logits,
            "logit_scale": logit_scale,
        }


def save_mmr_checkpoint(path: str, model: MusicMotionRetrievalModel, extra: Optional[Dict[str, Any]] = None):
    payload = {
        "model": model.state_dict(),
        "config": asdict(model.config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_mmr_model(path: str, device: str | torch.device = "cpu") -> MusicMotionRetrievalModel:
    ckpt = torch.load(path, map_location=device)
    config_dict = ckpt.get("config") or ckpt.get("args") or {}
    # Keep only MMRConfig fields.
    allowed = {f.name for f in MMRConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in config_dict.items() if k in allowed}
    model = MusicMotionRetrievalModel(MMRConfig(**filtered)).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
