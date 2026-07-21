"""Build and train the FPS/Event-DB-specific music-motion Router."""
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

from model.music_motion_router import MusicMotionRouter
from scheduling.audio_features import extract_audio_features
from scheduling.index_io import load_shared_index
from scheduling.music_event_calibration import build_phrase_query
from scheduling.music_phrase_segmentation import audio_duration_seconds
from support.common import event_compatibility
from training.music_corpus import audio_sha256, discover_training_audio
from support.scheduler_checkpoint_contracts import scheduler_training_contract


def _bool(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _music_features(path: Path, cache_dir: Path, num_frames: int) -> np.ndarray:
    fingerprint = audio_sha256(path)
    cached = cache_dir / f"{path.stem}.{fingerprint[:16]}.{num_frames}.npy"
    if cached.is_file():
        features = np.load(cached)
    else:
        features, _metadata = extract_audio_features(path, num_frames=num_frames)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, np.asarray(features, dtype=np.float32))
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 12 or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid music features for {path}: {features.shape}")
    return features[:, :12]


def build_dataset(args: argparse.Namespace) -> int:
    metadata, arrays, items = load_shared_index(args.index_json, args.index_npz)
    try:
        rates = [float(value) for value in metadata.get("canonical_fps_values", [])]
        if rates != [float(args.fps)]:
            raise RuntimeError(
                f"Router dataset FPS mismatch: index={rates}, requested={[float(args.fps)]}"
            )
        motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
        style = np.asarray(arrays["style_score"], dtype=np.float32)
        quality = np.asarray(arrays["quality_score"], dtype=np.float32)
        safety = np.asarray(arrays["safety_score"], dtype=np.float32)
    finally:
        arrays.close()

    if motion_desc.ndim != 2 or motion_desc.shape[1] != 12:
        raise RuntimeError(f"Router requires 12D motion descriptors, got {motion_desc.shape}")
    event_types = [str(item.get("event_type", "neutral_flow")) for item in items]
    event_uids = [str(item["event_uid"]) for item in items]
    base_quality = 0.50 * style + 0.30 * quality + 0.20 * safety
    rng = np.random.default_rng(args.seed)
    cache_dir = Path(args.cache_dir).resolve()
    audio_paths = discover_training_audio(args.music_dirs)

    music_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    song_uids: list[str] = []
    positive_uids: list[str] = []
    negative_uids: list[str] = []
    labels: list[str] = []

    for song_index, audio_path in enumerate(audio_paths):
        features = _music_features(audio_path, cache_dir, args.num_frames)
        effective_fps = len(features) / max(audio_duration_seconds(audio_path), 1.0e-6)
        boundaries = np.linspace(0, len(features), args.phrases + 1).astype(np.int64)
        song_uid = "aud_" + audio_sha256(audio_path)[:24]
        for slot in range(args.phrases):
            start = int(boundaries[slot])
            end = int(boundaries[slot + 1])
            if end <= start:
                continue
            query, music_event = build_phrase_query(
                features[start:end], start, end, fps=effective_fps
            )
            compatibility = np.asarray(
                [event_compatibility(music_event, event) for event in event_types],
                dtype=np.float32,
            )
            distance = np.linalg.norm(motion_desc - query[None], axis=1)
            score = base_quality + 0.70 * compatibility - 0.55 * distance
            positive_indices = np.argsort(score)[::-1][: max(1, args.positives_per_phrase)]
            positive_set = {int(index) for index in positive_indices}
            hard_order = np.argsort(
                base_quality - 0.55 * compatibility - 0.25 * distance
            )[::-1]
            hard_pool = [int(index) for index in hard_order[: min(512, len(hard_order))] if int(index) not in positive_set]
            if not hard_pool:
                continue
            for positive_index in positive_indices:
                for _ in range(max(1, args.negatives_per_positive)):
                    negative_index = int(rng.choice(hard_pool))
                    music_rows.append(query.astype(np.float32))
                    positive_rows.append(motion_desc[int(positive_index)])
                    negative_rows.append(motion_desc[negative_index])
                    song_uids.append(song_uid)
                    positive_uids.append(event_uids[int(positive_index)])
                    negative_uids.append(event_uids[negative_index])
                    labels.append(f"{song_uid}:slot{slot}:{music_event}")
        if (song_index + 1) % 50 == 0 or song_index + 1 == len(audio_paths):
            print(f"[Router data] {song_index + 1}/{len(audio_paths)} songs", flush=True)

    if not music_rows:
        raise RuntimeError("Router dataset is empty")
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        music=np.stack(music_rows).astype(np.float32),
        positive=np.stack(positive_rows).astype(np.float32),
        negative=np.stack(negative_rows).astype(np.float32),
        song_uids=np.asarray(song_uids, dtype=object),
        positive_event_uids=np.asarray(positive_uids, dtype=object),
        negative_event_uids=np.asarray(negative_uids, dtype=object),
        labels=np.asarray(labels, dtype=object),
        fps=np.asarray(float(args.fps), dtype=np.float32),
        event_db_contract_json=np.asarray(
            json.dumps(metadata["event_db_contract"], sort_keys=True), dtype=object
        ),
    )
    report = {
        "schema": "dunhuang_router_dataset_v1",
        "dataset": str(target),
        "num_samples": len(music_rows),
        "num_songs": len(audio_paths),
        "fps": float(args.fps),
        "event_db_contract": metadata["event_db_contract"],
    }
    target.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def _group_split(groups: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted({str(value) for value in groups}), dtype=object)
    if len(unique) < 2:
        raise RuntimeError("Router training requires at least two distinct songs")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val_count = max(1, min(len(unique) - 1, int(round(len(unique) * val_ratio))))
    validation = {str(value) for value in unique[:val_count]}
    val_indices = np.asarray([i for i, value in enumerate(groups) if str(value) in validation], dtype=np.int64)
    train_indices = np.asarray([i for i, value in enumerate(groups) if str(value) not in validation], dtype=np.int64)
    return train_indices, val_indices


def _load_music_prior(model: MusicMotionRouter, checkpoint_path: Path) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError(f"Music prior has no model state: {checkpoint_path}")
    prior = {
        key.removeprefix("music_encoder."): value
        for key, value in state.items()
        if str(key).startswith("music_encoder.")
    }
    if not prior:
        raise RuntimeError(
            f"Music prior does not contain a Router music_encoder: {checkpoint_path}"
        )
    model.music_encoder.load_state_dict(prior, strict=True)
    return sorted(prior)


def train_model(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.data, allow_pickle=True) as data:
        music = np.asarray(data["music"], dtype=np.float32)
        positive = np.asarray(data["positive"], dtype=np.float32)
        negative = np.asarray(data["negative"], dtype=np.float32)
        song_uids = np.asarray(data["song_uids"], dtype=object)
        positive_uids = np.asarray(data["positive_event_uids"], dtype=object)
        pair_labels = np.asarray(data["labels"], dtype=object)
    if not (
        len(music)
        == len(positive)
        == len(negative)
        == len(song_uids)
        == len(positive_uids)
        == len(pair_labels)
    ):
        raise RuntimeError("Router dataset arrays are not aligned")
    label_to_id = {
        label: index
        for index, label in enumerate(sorted({str(value) for value in pair_labels}))
    }
    pair_ids = np.asarray(
        [label_to_id[str(value)] for value in pair_labels], dtype=np.int64
    )
    train_indices, val_indices = _group_split(song_uids, args.val_ratio, args.seed)

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        index_tensor = torch.from_numpy(indices.astype(np.int64, copy=False))
        dataset = TensorDataset(
            torch.from_numpy(music).index_select(0, index_tensor),
            torch.from_numpy(positive).index_select(0, index_tensor),
            torch.from_numpy(negative).index_select(0, index_tensor),
            torch.from_numpy(pair_ids).index_select(0, index_tensor),
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = loader(train_indices, True)
    val_loader = loader(val_indices, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_path = Path(args.music_prior_ckpt).resolve() if args.music_prior_ckpt else None
    prior_config: dict[str, Any] = {}
    if prior_path is not None:
        prior_checkpoint = torch.load(
            prior_path, map_location="cpu", weights_only=False
        )
        if not isinstance(prior_checkpoint, dict):
            raise RuntimeError(f"Music prior is not a checkpoint mapping: {prior_path}")
        prior_config = dict(prior_checkpoint.get("config", {}))
        prior_music_dim = int(prior_config.get("music_dim", music.shape[1]))
        if prior_music_dim != int(music.shape[1]):
            raise RuntimeError(
                f"Music prior input mismatch: prior={prior_music_dim}, "
                f"dataset={music.shape[1]}"
            )
    effective_hidden_dim = int(prior_config.get("hidden_dim", args.hidden_dim))
    effective_latent_dim = int(prior_config.get("latent_dim", args.latent_dim))
    effective_dropout = float(prior_config.get("dropout", args.dropout))
    model = MusicMotionRouter(
        music_dim=music.shape[1],
        motion_dim=positive.shape[1],
        hidden_dim=effective_hidden_dim,
        latent_dim=effective_latent_dim,
        dropout=effective_dropout,
    )
    imported_keys: list[str] = []
    if prior_path is not None:
        imported_keys = _load_music_prior(model, prior_path)
    freeze_music = _bool(args.freeze_music_encoder)
    if freeze_music:
        if prior_path is None:
            raise RuntimeError("--freeze_music_encoder requires --music_prior_ckpt")
        for parameter in model.music_encoder.parameters():
            parameter.requires_grad = False
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def run_epoch(data_loader: DataLoader, training: bool) -> float:
        model.train(training)
        if freeze_music:
            model.music_encoder.eval()
        total = 0.0
        count = 0
        for music_batch, positive_batch, negative_batch, pair_id_batch in data_loader:
            music_batch = music_batch.to(device, non_blocking=True)
            positive_batch = positive_batch.to(device, non_blocking=True)
            negative_batch = negative_batch.to(device, non_blocking=True)
            pair_id_batch = pair_id_batch.to(device, non_blocking=True)
            with torch.set_grad_enabled(training):
                music_embedding = model.encode_music(music_batch)
                positive_embedding = model.encode_motion(positive_batch)
                negative_embedding = model.encode_motion(negative_batch)
                scale = model.logit_scale.exp().clamp(max=100.0)
                logits = scale * (music_embedding @ positive_embedding.t())
                # All top-k positives proposed for the same music phrase are
                # positives.  Treating them as one-to-one pairs would create
                # contradictory false negatives inside every batch.
                positive_mask = pair_id_batch[:, None] == pair_id_batch[None, :]
                negative_infinity = torch.finfo(logits.dtype).min
                row_positive = torch.logsumexp(
                    logits.masked_fill(~positive_mask, negative_infinity), dim=1
                )
                column_positive = torch.logsumexp(
                    logits.t().masked_fill(~positive_mask.t(), negative_infinity), dim=1
                )
                contrastive = 0.5 * (
                    (torch.logsumexp(logits, dim=1) - row_positive).mean()
                    + (torch.logsumexp(logits.t(), dim=1) - column_positive).mean()
                )
                margin = F.softplus(
                    (music_embedding * negative_embedding).sum(-1)
                    - (music_embedding * positive_embedding).sum(-1)
                    + args.margin
                ).mean()
                loss = contrastive + args.margin_weight * margin
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total += float(loss.detach()) * len(music_batch)
            count += len(music_batch)
        return total / max(count, 1)

    metadata, arrays, _items = load_shared_index(args.index_json, args.index_npz)
    arrays.close()
    contract = scheduler_training_contract(
        role="router",
        fps=args.fps,
        index_metadata=metadata,
        index_json=args.index_json,
        index_npz=args.index_npz,
        dataset=args.data,
        music_prior=prior_path,
    )
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "music_dim": int(music.shape[1]),
        "motion_dim": int(positive.shape[1]),
        "hidden_dim": effective_hidden_dim,
        "latent_dim": effective_latent_dim,
        "dropout": effective_dropout,
        "fps": float(args.fps),
        "music_encoder_frozen": freeze_music,
    }
    best = float("inf")
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best - 1.0e-7:
            best = val_loss
            patience = 0
            torch.save(
                {
                    "version": "formal_music_motion_router_v1",
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fps": float(args.fps),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "scheduler_contract": contract,
                    "music_prior_policy": {
                        "checkpoint": str(prior_path) if prior_path else None,
                        "imported_keys": imported_keys,
                        "music_encoder_frozen": freeze_music,
                        "motion_encoder_retrained": True,
                    },
                },
                output,
            )
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Router] epoch={epoch} train={train_loss:.6f} "
                f"val={val_loss:.6f} best={best:.6f}",
                flush=True,
            )
        if patience >= args.patience:
            break
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"[PASS] Router checkpoint: {output}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-dataset")
    build.add_argument("--index_json", required=True)
    build.add_argument("--index_npz", required=True)
    build.add_argument("--music_dirs", nargs="+", required=True)
    build.add_argument("--cache_dir", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--fps", type=float, required=True)
    build.add_argument("--num_frames", type=int, default=150)
    build.add_argument("--phrases", type=int, default=3)
    build.add_argument("--positives_per_phrase", type=int, default=4)
    build.add_argument("--negatives_per_positive", type=int, default=2)
    build.add_argument("--seed", type=int, default=20260722)

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--index_json", required=True)
    train.add_argument("--index_npz", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--music_prior_ckpt", default="")
    train.add_argument("--freeze_music_encoder", default="1")
    train.add_argument("--fps", type=float, required=True)
    train.add_argument("--epochs", type=int, default=250)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--lr", type=float, default=2.0e-4)
    train.add_argument("--weight_decay", type=float, default=1.0e-4)
    train.add_argument("--hidden_dim", type=int, default=128)
    train.add_argument("--latent_dim", type=int, default=64)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--margin", type=float, default=0.10)
    train.add_argument("--margin_weight", type=float, default=0.5)
    train.add_argument("--val_ratio", type=float, default=0.10)
    train.add_argument("--patience", type=int, default=40)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return build_dataset(args) if args.command == "build-dataset" else train_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
