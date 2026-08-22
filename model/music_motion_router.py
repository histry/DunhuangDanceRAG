#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict loader for formal and current-protocol Router checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def load_router_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config: Dict[str, Any] = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    architecture = str(
        config.get(
            "architecture",
            checkpoint.get("architecture", "")
            if isinstance(checkpoint, dict)
            else "",
        )
    )
    if architecture == "ctsr_weak_temporal_v1":
        from model.temporal_music_motion_router import TemporalMusicMotionRouter

        model = TemporalMusicMotionRouter(
            music_dim=int(config.get("music_dim", 12)),
            motion_dim=int(config.get("motion_dim", 12)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            latent_dim=int(config.get("latent_dim", 96)),
            dropout=float(config.get("dropout", 0.1)),
            transformer_layers=int(config.get("transformer_layers", 2)),
            transformer_heads=int(config.get("transformer_heads", 4)),
            init_temperature=float(config.get("init_temperature", 0.12)),
        )
        model.sequence_frames = int(config.get("sequence_frames", 64))
        model.inference_temperature = float(
            config.get("inference_temperature", config.get("init_temperature", 0.12))
        )
        model.feature_mean = list(config.get("feature_mean", [])) or None
        model.feature_std = list(config.get("feature_std", [])) or None
    elif architecture == "ctsr_mean_pool_mlp_baseline_v1":
        from model.current_protocol_router_baseline import MeanPoolMusicMotionRouter

        model = MeanPoolMusicMotionRouter(
            music_dim=int(config.get("music_dim", 12)),
            motion_dim=int(config.get("motion_dim", 12)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            latent_dim=int(config.get("latent_dim", 96)),
            dropout=float(config.get("dropout", 0.1)),
            init_temperature=float(config.get("init_temperature", 0.12)),
        )
        model.inference_temperature = float(
            config.get("inference_temperature", config.get("init_temperature", 0.12))
        )
        model.feature_mean = list(config.get("feature_mean", [])) or None
        model.feature_std = list(config.get("feature_std", [])) or None
    else:
        raise RuntimeError(
            f"Unsupported Router architecture {architecture!r}; historical "
            "checkpoints are available only from the archive Git tag"
        )
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
