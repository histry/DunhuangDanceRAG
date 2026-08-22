#!/usr/bin/env python3
"""Train the non-temporal Router baseline on the exact CTSR weak dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.current_protocol_router_baseline import (
    ARCHITECTURE,
    MeanPoolMusicMotionRouter,
)
from scheduling.index_io import load_shared_index
from support.scheduler_checkpoint_contracts import scheduler_training_contract
from training.temporal_music_router import DATASET_SCHEMA
from training.weak_semantic_ot import TEACHER_SCHEMA


CHECKPOINT_VERSION = "smpl14_ctsr_mean_pool_router_baseline_v1"


def _song_disjoint_split(
    song_uids: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(map(str, song_uids.tolist()))), dtype=object)
    if len(unique) < 2:
        raise RuntimeError("Router baseline requires at least two songs")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    val_count = max(1, min(len(unique) - 1, int(round(len(unique) * val_ratio))))
    validation = set(map(str, unique[:val_count].tolist()))
    val = np.asarray(
        [index for index, value in enumerate(song_uids) if str(value) in validation],
        dtype=np.int64,
    )
    train = np.asarray(
        [index for index, value in enumerate(song_uids) if str(value) not in validation],
        dtype=np.int64,
    )
    if not len(train) or not len(val):
        raise RuntimeError("Song-disjoint baseline split is empty")
    return train, val


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(args.data, allow_pickle=True) as payload:
        schema = str(np.asarray(payload["dataset_schema"]).item())
        teacher_schema = str(np.asarray(payload["teacher_schema"]).item())
        sequences = np.asarray(payload["music_sequences"], dtype=np.float32)
        teacher = np.asarray(payload["teacher_probabilities"], dtype=np.float32)
        motion_desc = np.asarray(payload["motion_desc"], dtype=np.float32)
        song_uids = np.asarray(payload["song_uids"], dtype=object)
    if schema != DATASET_SCHEMA or teacher_schema != TEACHER_SCHEMA:
        raise RuntimeError("Baseline must consume the formal CTSR weak-teacher dataset")
    if sequences.ndim != 3 or sequences.shape[-1] != 12:
        raise RuntimeError(f"Invalid baseline sequence shape: {sequences.shape}")
    if teacher.shape != (len(sequences), len(motion_desc)):
        raise RuntimeError("Baseline teacher/event dimensions do not match")
    if not np.isfinite(sequences).all() or not np.isfinite(teacher).all():
        raise RuntimeError("Baseline dataset contains NaN/Inf")

    train_ids, val_ids = _song_disjoint_split(
        song_uids, float(args.val_ratio), int(args.seed)
    )
    sequence_tensor = torch.from_numpy(sequences)
    teacher_tensor = torch.from_numpy(teacher)

    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        ids = torch.from_numpy(indices)
        return DataLoader(
            TensorDataset(
                sequence_tensor.index_select(0, ids),
                teacher_tensor.index_select(0, ids),
            ),
            batch_size=int(args.batch_size),
            shuffle=shuffle,
            num_workers=int(args.num_workers),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeanPoolMusicMotionRouter(
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        dropout=float(args.dropout),
        init_temperature=float(args.init_temperature),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    motion = torch.from_numpy(motion_desc).to(device)

    def epoch(loader: DataLoader, update: bool) -> float:
        model.train(update)
        total = 0.0
        count = 0
        for sequence, target in loader:
            sequence = sequence.to(device)
            target = target.to(device)
            with torch.set_grad_enabled(update):
                logits = model(sequence, motion)
                loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                if update:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total += float(loss.detach()) * len(sequence)
            count += len(sequence)
        return total / max(count, 1)

    metadata, arrays, _ = load_shared_index(args.index_json, args.index_npz)
    arrays.close()
    contract = scheduler_training_contract(
        role="router",
        fps=float(args.fps),
        index_metadata=metadata,
        index_json=args.index_json,
        index_npz=args.index_npz,
        dataset=args.data,
    )
    feature_mean = sequences.mean(axis=(0, 1)).astype(np.float32)
    feature_std = np.maximum(sequences.std(axis=(0, 1)), 1.0e-5).astype(np.float32)
    config = {
        "architecture": ARCHITECTURE,
        "music_dim": 12,
        "motion_dim": 12,
        "hidden_dim": int(args.hidden_dim),
        "latent_dim": int(args.latent_dim),
        "dropout": float(args.dropout),
        "init_temperature": float(args.init_temperature),
        "inference_temperature": float(args.init_temperature),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "fps": float(args.fps),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train_ids, True)
    val_loader = make_loader(val_ids, False)
    best = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    for epoch_id in range(1, int(args.epochs) + 1):
        train_loss = epoch(train_loader, True)
        val_loss = epoch(val_loader, False)
        history.append(
            {"epoch": epoch_id, "train_loss": train_loss, "validation_loss": val_loss}
        )
        if val_loss < best - 1.0e-7:
            best = val_loss
            patience = 0
            torch.save(
                {
                    "version": CHECKPOINT_VERSION,
                    "architecture": ARCHITECTURE,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fps": float(args.fps),
                    "scheduler_contract": contract,
                    "baseline_contract": {
                        "schema": "smpl14_ctsr_router_baseline_contract_v1",
                        "same_dataset_schema": DATASET_SCHEMA,
                        "same_teacher_schema": TEACHER_SCHEMA,
                        "same_song_disjoint_split_policy": True,
                        "external_pretrained_model": False,
                        "human_training_labels": 0,
                        "temporal_order_model": False,
                        "formal_model": False,
                    },
                },
                out,
            )
        else:
            patience += 1
        if patience >= int(args.patience):
            break
    out.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "checkpoint": str(out), "best_val_loss": best}))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--index_npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init_temperature", type=float, default=0.12)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args(argv)

if __name__ == "__main__":
    raise SystemExit(train(parse_args()))
