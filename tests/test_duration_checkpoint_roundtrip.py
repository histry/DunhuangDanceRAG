#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
from pathlib import Path

import torch

import model.duration_predictor as duration_module
from model.duration_predictor import (
    NATIVE_ROT6D_LAYOUT,
    MonotonicDurationModel,
    load_duration_checkpoint,
)


def _config() -> dict[str, object]:
    return {
        "motion_dim": 151,
        "condition_dim": 3,
        "hidden_dim": 8,
        "dropout": 0.0,
        "duration_edges": [2, 5, 9],
        "window_len": 8,
        "duration_dilations": [1],
        "tau_dilations": [1],
        "slow_feature_span": 2,
        "ordinal_blend": 0.82,
        "fps": 30.0,
        "rot6d_layout": NATIVE_ROT6D_LAYOUT,
    }


def test_public_duration_loader_has_one_definition() -> None:
    source_path = Path(duration_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "load_duration_checkpoint"
    ]
    assert len(definitions) == 1


def test_training_format_checkpoint_roundtrip_and_forward(tmp_path) -> None:
    config = _config()
    model = MonotonicDurationModel(
        **{key: value for key, value in config.items() if key != "rot6d_layout"}
    )
    checkpoint_path = tmp_path / "duration_predictor.pt"
    torch.save(
        {
            "version": "formal_monotonic_duration_v1",
            "model_state_dict": model.state_dict(),
            "config": config,
            "fps": 30.0,
            "rot6d_layout": NATIVE_ROT6D_LAYOUT,
            "epoch": 1,
            "stage": "joint",
        },
        checkpoint_path,
    )

    bundle = load_duration_checkpoint(checkpoint_path, device="cpu")
    loaded = bundle["model"]
    motion = torch.zeros((1, 8, 151), dtype=torch.float32)
    identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
    motion[..., 7:151] = identity.repeat(24)
    output = loaded(
        motion,
        torch.ones((1, 8), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
    )

    assert bundle["rot6d_layout"] == NATIVE_ROT6D_LAYOUT
    assert output["tau"].shape == (1, 8)
    assert torch.isfinite(output["duration_frames"]).all()
    assert torch.isfinite(output["tau"]).all()
