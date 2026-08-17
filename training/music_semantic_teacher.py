#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and apply a weak music-semantic teacher from the historical Router.

Only the historical ``music_encoder`` branch is imported.  The historical
motion encoder, duration model and planner are never reused.  A lightweight
8-class semantic head is trained with song-disjoint weak labels generated from
music structure descriptors, then exports MSSD sidecars for semantic optimal
transport.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from events.semantic_descriptor import MUSIC_SEMANTIC_LABELS, canonical_music_label
from model.music_motion_router import MusicMotionRouter
from scheduling.audio_features import extract_audio_features
from scheduling.music_event_calibration import phrase_statistics
from scheduling.music_phrase_segmentation import (
    audio_duration_seconds,
    segment_music_phrases,
)
from training.music_corpus import audio_sha256, discover_training_audio


SCHEMA = "dunhuang_music_semantic_teacher_v1"
DATASET_SCHEMA = "dunhuang_music_semantic_teacher_dataset_v1"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _music_features(path: Path, cache_dir: Path, num_frames: int) -> np.ndarray:
    fingerprint = audio_sha256(path)
    cached = cache_dir / f"{path.stem}.{fingerprint[:16]}.{num_frames}.npy"
    if cached.is_file():
        features = np.load(cached)
    else:
        features, metadata = extract_audio_features(path, num_frames=num_frames)
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, np.asarray(features, dtype=np.float32))
        cached.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 12:
        raise RuntimeError(f"Invalid music features for {path}: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"Music features contain NaN/Inf: {path}")
    return features[:, :12]


def _label_id(music_event: str) -> int:
    label = canonical_music_label(music_event)
    return MUSIC_SEMANTIC_LABELS.index(label)


def _softmax_numpy(logits: np.ndarray, temperature: float = 0.55) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values / max(float(temperature), 0.05)
    values -= float(np.max(values))
    probabilities = np.exp(values)
    probabilities /= max(float(probabilities.sum()), 1.0e-12)
    return probabilities.astype(np.float32)


def weak_semantic_distribution(window: np.ndarray, music_event: str) -> np.ndarray:
    """Build an eight-class soft target from observable music evidence.

    These are compatibility targets, not action labels.  Every class receives a
    finite probability, while the dominant mass follows energy, calmness,
    rhythmic accent, brightness, novelty and phrase trend.
    """
    statistics = phrase_statistics(np.asarray(window, dtype=np.float32))
    energy = float(np.clip(statistics["energy"], 0.0, 1.0))
    onset = float(np.clip(statistics["onset"], 0.0, 1.0))
    beat = float(np.clip(statistics["beat_density"], 0.0, 1.0))
    arousal = float(np.clip(statistics["arousal"], 0.0, 1.0))
    tension = float(np.clip(statistics["tension"], 0.0, 1.0))
    calm = float(np.clip(statistics["calm"], 0.0, 1.0))
    novelty = float(np.clip(statistics["novelty"], 0.0, 1.0))
    brightness = float(np.clip(statistics["brightness"], 0.0, 1.0))
    accent = float(np.clip(statistics["accent_mean"], 0.0, 1.0))
    positive_trend = float(
        np.clip(max(statistics["arousal_trend"], statistics["tension_trend"], 0.0) * 5.0, 0.0, 1.0)
    )
    negative_trend = float(
        np.clip(max(-statistics["arousal_trend"], -statistics["tension_trend"], 0.0) * 5.0, 0.0, 1.0)
    )
    moderate = 1.0 - min(1.0, abs(arousal - 0.52) * 2.0)
    sustained = float(np.clip(1.0 - 0.55 * onset - 0.45 * beat, 0.0, 1.0))
    scores = np.asarray(
        [
            1.35 * calm + 0.45 * (1.0 - arousal) + 0.25 * negative_trend,
            0.95 * calm + 0.65 * sustained + 0.35 * (1.0 - novelty),
            0.75 * moderate + 0.55 * sustained + 0.35 * (1.0 - tension),
            0.65 * brightness + 0.50 * accent + 0.40 * moderate + 0.20 * novelty,
            0.90 * onset + 0.85 * beat + 0.75 * accent + 0.30 * energy,
            0.85 * arousal + 0.80 * tension + 0.55 * positive_trend + 0.35 * novelty,
            0.80 * beat + 0.55 * onset + 0.45 * moderate + 0.30 * energy,
            0.65 * brightness + 0.60 * novelty + 0.45 * positive_trend + 0.35 * sustained,
        ],
        dtype=np.float32,
    )
    event_hint = canonical_music_label(music_event)
    if event_hint in MUSIC_SEMANTIC_LABELS:
        scores[MUSIC_SEMANTIC_LABELS.index(event_hint)] += 0.35
    return _softmax_numpy(scores, temperature=0.55)


def _prior_adjust_rows(probabilities: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(probabilities, dtype=np.float64)
    raw = np.maximum(raw, 1.0e-8)
    raw /= raw.sum(axis=1, keepdims=True)
    prior = raw.mean(axis=0)
    prior /= max(float(prior.sum()), 1.0e-8)
    logits = np.log(raw) - float(np.clip(alpha, 0.0, 1.0)) * np.log(prior[None] + 1.0e-8)
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted.astype(np.float32), prior.astype(np.float32)


def _segment_song(
    features: np.ndarray,
    effective_fps: float,
    *,
    min_phrase_seconds: float,
    max_phrase_seconds: float,
    boundary_quantile: float,
    beat_snap_seconds: float,
):
    phrases, report = segment_music_phrases(
        features,
        fps=float(effective_fps),
        min_phrase_seconds=float(min_phrase_seconds),
        max_phrase_seconds=float(max_phrase_seconds),
        boundary_quantile=float(boundary_quantile),
        beat_snap_seconds=float(beat_snap_seconds),
    )
    if not phrases:
        raise RuntimeError("Music structure segmentation produced no phrases")
    return phrases, report


def load_audio_manifest(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("songs", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Audio manifest is empty: {path}")
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("audio_path"):
            raise RuntimeError("Each audio-manifest row requires audio_path")
        audio = Path(str(row["audio_path"])).expanduser()
        if not audio.is_absolute():
            audio = path.parent / audio
        audio = audio.resolve()
        if not audio.is_file():
            raise FileNotFoundError(str(audio))
        result.append(audio)
    return result


def build_dataset(
    audio_paths: Sequence[Path],
    cache_dir: Path,
    out_path: Path,
    *,
    num_frames: int = 1800,
    min_phrase_seconds: float = 1.6,
    max_phrase_seconds: float = 7.5,
    boundary_quantile: float = 0.68,
    beat_snap_seconds: float = 0.35,
    weak_prior_alpha: float = 0.25,
) -> Dict[str, Any]:
    audio_paths = [Path(path).resolve() for path in audio_paths]
    if not audio_paths:
        raise RuntimeError("No training music discovered")
    query_rows = []
    probability_rows = []
    song_rows = []
    phrase_rows = []
    start_rows = []
    end_rows = []
    audio_rows = []
    segmentation_reports = []
    for song_index, audio_path in enumerate(audio_paths):
        features = _music_features(audio_path, cache_dir, int(num_frames))
        duration = max(audio_duration_seconds(audio_path), 1.0e-6)
        effective_fps = len(features) / duration
        phrases, segmentation = _segment_song(
            features,
            effective_fps,
            min_phrase_seconds=float(min_phrase_seconds),
            max_phrase_seconds=float(max_phrase_seconds),
            boundary_quantile=float(boundary_quantile),
            beat_snap_seconds=float(beat_snap_seconds),
        )
        song_uid = "aud_" + audio_sha256(audio_path)[:24]
        segmentation_reports.append(
            {
                "song_uid": song_uid,
                "audio": str(audio_path),
                "num_phrases": len(phrases),
                "boundaries": [int(phrases[0].start)] + [int(row.end) for row in phrases],
                "segmentation": segmentation,
            }
        )
        for phrase in phrases:
            start = int(phrase.start)
            end = int(phrase.end)
            if end <= start:
                continue
            query = np.asarray(phrase.query, dtype=np.float32)
            probabilities = weak_semantic_distribution(
                features[start:end], str(phrase.music_event)
            )
            query_rows.append(query)
            probability_rows.append(probabilities)
            song_rows.append(song_uid)
            phrase_rows.append(f"{song_uid}::phrase_{int(phrase.index):05d}")
            start_rows.append(float(start / effective_fps))
            end_rows.append(float(end / effective_fps))
            audio_rows.append(str(audio_path))
        if (song_index + 1) % 50 == 0 or song_index + 1 == len(audio_paths):
            print(
                f"[Music semantic teacher data] {song_index + 1}/{len(audio_paths)} songs",
                flush=True,
            )
    queries = np.stack(query_rows).astype(np.float32)
    raw_probabilities = np.stack(probability_rows).astype(np.float32)
    weak_probabilities, empirical_prior = _prior_adjust_rows(
        raw_probabilities, float(weak_prior_alpha)
    )
    labels = weak_probabilities.argmax(axis=1).astype(np.int64)
    counts = np.bincount(labels, minlength=len(MUSIC_SEMANTIC_LABELS))
    payload = {
        "schema": np.asarray(DATASET_SCHEMA, dtype=object),
        "music_query": queries,
        "weak_probabilities": weak_probabilities,
        "raw_weak_probabilities": raw_probabilities,
        "weak_label": labels,
        "weak_class_prior": empirical_prior,
        "song_uids": np.asarray(song_rows, dtype=object),
        "phrase_uids": np.asarray(phrase_rows, dtype=object),
        "start_sec": np.asarray(start_rows, dtype=np.float32),
        "end_sec": np.asarray(end_rows, dtype=np.float32),
        "audio_paths": np.asarray(audio_rows, dtype=object),
        "label_names": np.asarray(MUSIC_SEMANTIC_LABELS, dtype=object),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    report = {
        "schema": DATASET_SCHEMA,
        "dataset": str(out_path.resolve()),
        "num_songs": int(len(audio_paths)),
        "num_phrases": int(len(queries)),
        "phrase_intervals_are_structure_segmented": True,
        "segmentation_parameters": {
            "min_phrase_seconds": float(min_phrase_seconds),
            "max_phrase_seconds": float(max_phrase_seconds),
            "boundary_quantile": float(boundary_quantile),
            "beat_snap_seconds": float(beat_snap_seconds),
        },
        "weak_prior_alpha": float(weak_prior_alpha),
        "raw_soft_class_mass": {
            label: float(raw_probabilities[:, index].sum())
            for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
        },
        "calibrated_soft_class_mass": {
            label: float(weak_probabilities[:, index].sum())
            for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
        },
        "top_label_histogram": {
            label: int(counts[index])
            for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
        },
        "weak_supervision": True,
        "targets_are_music_action_compatibility_not_action_annotations": True,
        "ground_truth_music_motion_pairing": False,
        "segmentation": segmentation_reports,
        "ok": True,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _group_split(
    groups: np.ndarray, validation_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted({str(value) for value in groups}), dtype=object)
    if len(unique) < 2:
        raise RuntimeError("Teacher training requires at least two songs")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique)
    validation_count = max(
        1,
        min(
            len(unique) - 1,
            int(round(len(unique) * float(validation_ratio))),
        ),
    )
    validation_songs = {str(value) for value in unique[:validation_count]}
    validation = np.asarray(
        [
            index
            for index, value in enumerate(groups)
            if str(value) in validation_songs
        ],
        dtype=np.int64,
    )
    training = np.asarray(
        [
            index
            for index, value in enumerate(groups)
            if str(value) not in validation_songs
        ],
        dtype=np.int64,
    )
    return training, validation


def _load_music_encoder(
    model: MusicMotionRouter, checkpoint_path: Path
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Invalid historical Router checkpoint: {checkpoint_path}")
    prior = {
        str(key).removeprefix("music_encoder."): value
        for key, value in state.items()
        if str(key).startswith("music_encoder.")
    }
    if not prior:
        raise RuntimeError(
            f"Checkpoint contains no music_encoder branch: {checkpoint_path}"
        )
    model.music_encoder.load_state_dict(prior, strict=True)
    return dict(checkpoint.get("config", {})) if isinstance(checkpoint, Mapping) else {}


class SemanticTeacher(nn.Module):
    def __init__(
        self,
        music_encoder: nn.Module,
        latent_dim: int,
        class_count: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.music_encoder = music_encoder
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(int(latent_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(latent_dim), int(class_count)),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.semantic_head(self.music_encoder(query))


def train_teacher(
    dataset_path: Path,
    historical_router_checkpoint: Path,
    out_path: Path,
    *,
    epochs: int = 80,
    batch_size: int = 256,
    validation_ratio: float = 0.20,
    seed: int = 20260724,
    learning_rate: float = 2.0e-4,
    weight_decay: float = 1.0e-4,
    freeze_music_encoder: bool = True,
    patience: int = 15,
) -> Dict[str, Any]:
    with np.load(dataset_path, allow_pickle=True) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema != DATASET_SCHEMA:
            raise RuntimeError(f"Unsupported teacher dataset schema: {schema}")
        queries = np.asarray(data["music_query"], dtype=np.float32)
        labels = np.asarray(data["weak_label"], dtype=np.int64)
        weak_probabilities = np.asarray(data["weak_probabilities"], dtype=np.float32)
        song_uids = np.asarray(data["song_uids"], dtype=object)
    training_indices, validation_indices = _group_split(
        song_uids, float(validation_ratio), int(seed)
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(
        historical_router_checkpoint, map_location="cpu", weights_only=False
    )
    prior_config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, Mapping) else {}
    music_dim = int(prior_config.get("music_dim", queries.shape[1]))
    hidden_dim = int(prior_config.get("hidden_dim", 128))
    latent_dim = int(prior_config.get("latent_dim", 64))
    dropout = float(prior_config.get("dropout", 0.1))
    if music_dim != queries.shape[1]:
        raise RuntimeError(
            f"Historical music encoder input mismatch: {music_dim} vs {queries.shape[1]}"
        )
    router = MusicMotionRouter(
        music_dim=music_dim,
        motion_dim=12,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    _load_music_encoder(router, historical_router_checkpoint)
    teacher = SemanticTeacher(
        router.music_encoder,
        latent_dim,
        len(MUSIC_SEMANTIC_LABELS),
        dropout=dropout,
    )
    if freeze_music_encoder:
        for parameter in teacher.music_encoder.parameters():
            parameter.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in teacher.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    counts = weak_probabilities[training_indices].sum(axis=0).astype(np.float64)
    class_weight = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    class_weight = class_weight / class_weight.mean()
    class_weight_tensor = torch.from_numpy(class_weight.astype(np.float32)).to(device)

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        index_tensor = torch.from_numpy(indices.astype(np.int64, copy=False))
        dataset = TensorDataset(
            torch.from_numpy(queries).index_select(0, index_tensor),
            torch.from_numpy(labels).index_select(0, index_tensor),
            torch.from_numpy(weak_probabilities).index_select(0, index_tensor),
        )
        return DataLoader(
            dataset,
            batch_size=min(int(batch_size), len(dataset)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    training_loader = loader(training_indices, True)
    validation_loader = loader(validation_indices, False)

    def run_epoch(data_loader: DataLoader, training: bool) -> Dict[str, float]:
        teacher.train(training)
        if freeze_music_encoder:
            teacher.music_encoder.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for query_batch, label_batch, target_batch in data_loader:
            query_batch = query_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            with torch.set_grad_enabled(training):
                logits = teacher(query_batch)
                weighted_target = target_batch * class_weight_tensor[None]
                weighted_target = weighted_target / weighted_target.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0e-8)
                loss = -(
                    weighted_target * F.log_softmax(logits, dim=-1)
                ).sum(dim=-1).mean()
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 2.0)
                    optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(query_batch)
            total_correct += int((logits.argmax(dim=-1) == label_batch).sum().cpu())
            total_count += len(query_batch)
        return {
            "loss": total_loss / max(total_count, 1),
            "weak_target_top1_agreement": total_correct / max(total_count, 1),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_epoch(training_loader, True)
        validation_metrics = run_epoch(validation_loader, False)
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        if validation_metrics["loss"] < best_validation - 1.0e-7:
            best_validation = validation_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema": SCHEMA,
                    "model_state_dict": teacher.state_dict(),
                    "config": {
                        "music_dim": music_dim,
                        "hidden_dim": hidden_dim,
                        "latent_dim": latent_dim,
                        "dropout": dropout,
                        "num_classes": len(MUSIC_SEMANTIC_LABELS),
                        "label_names": MUSIC_SEMANTIC_LABELS,
                    },
                    "historical_music_prior": str(
                        historical_router_checkpoint.resolve()
                    ),
                    "imported_branch": "music_encoder",
                    "historical_motion_encoder_reused": False,
                    "freeze_music_encoder": bool(freeze_music_encoder),
                    "training_indices": training_indices,
                    "validation_indices": validation_indices,
                    "epoch": epoch,
                    "validation_loss": validation_metrics["loss"],
                    "weak_supervision": True,
                },
                out_path,
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            print("[MUSIC SEMANTIC TEACHER] " + json.dumps(row), flush=True)
        if stale >= int(patience):
            break
    report = {
        "schema": SCHEMA,
        "checkpoint": str(out_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "historical_music_prior": str(historical_router_checkpoint.resolve()),
        "imported_branch": "music_encoder",
        "historical_motion_encoder_reused": False,
        "train_songs": int(len(set(map(str, song_uids[training_indices])))),
        "validation_songs": int(len(set(map(str, song_uids[validation_indices])))),
        "song_disjoint": True,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "history": history,
        "metric_interpretation": {
            "weak_target_agreement_is_not_human_annotation_accuracy": True
        },
        "ok": True,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def load_teacher(path: Path, device: torch.device) -> SemanticTeacher:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise RuntimeError(f"Not a {SCHEMA} checkpoint: {path}")
    config = dict(checkpoint["config"])
    router = MusicMotionRouter(
        music_dim=int(config["music_dim"]),
        motion_dim=12,
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        dropout=float(config["dropout"]),
    )
    teacher = SemanticTeacher(
        router.music_encoder,
        int(config["latent_dim"]),
        int(config["num_classes"]),
        dropout=float(config["dropout"]),
    )
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return teacher.to(device).eval()


def infer_mssd(
    checkpoint_path: Path,
    audio_paths: Sequence[Path],
    out_dir: Path,
    *,
    cache_dir: Path,
    num_frames: int = 1800,
    min_phrase_seconds: float = 1.6,
    max_phrase_seconds: float = 7.5,
    boundary_quantile: float = 0.68,
    beat_snap_seconds: float = 0.35,
    fps: float = 30.0,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(checkpoint_path, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for audio_path in audio_paths:
        audio_path = audio_path.resolve()
        features = _music_features(audio_path, cache_dir, int(num_frames))
        duration = max(audio_duration_seconds(audio_path), 1.0e-6)
        effective_fps = len(features) / duration
        phrases, segmentation = _segment_song(
            features,
            effective_fps,
            min_phrase_seconds=float(min_phrase_seconds),
            max_phrase_seconds=float(max_phrase_seconds),
            boundary_quantile=float(boundary_quantile),
            beat_snap_seconds=float(beat_snap_seconds),
        )
        query_rows = [np.asarray(phrase.query, dtype=np.float32) for phrase in phrases]
        with torch.no_grad():
            logits = teacher(torch.from_numpy(np.stack(query_rows)).to(device))
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        slots = []
        for probability, phrase in zip(probabilities, phrases):
            probability = np.asarray(probability, dtype=np.float64)
            probability /= max(float(probability.sum()), 1.0e-8)
            order = np.argsort(-probability, kind="stable")
            entropy = float(
                -(probability * np.log(probability + 1.0e-8)).sum()
                / math.log(len(probability))
            )
            margin = float(probability[order[0]] - probability[order[1]])
            start = int(phrase.start)
            end = int(phrase.end)
            slots.append(
                {
                    "slot_id": int(phrase.index),
                    "start_sec": float(start / effective_fps),
                    "end_sec": float(end / effective_fps),
                    "duration_sec": float((end - start) / effective_fps),
                    "target_frames": int(round((end - start) / effective_fps * fps)),
                    "music_semantic_top_label": MUSIC_SEMANTIC_LABELS[int(order[0])],
                    "music_semantic_probs": {
                        label: float(probability[index])
                        for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
                    },
                    "teacher_entropy": entropy,
                    "teacher_margin": margin,
                    "teacher_confidence": float(
                        np.clip(0.5 * (1.0 - entropy) + 0.5 * margin, 0.05, 1.0)
                    ),
                    "boundary_confidence": float(phrase.boundary_confidence),
                    "source_music_event": str(phrase.music_event),
                    "usage": "train_semantic",
                    "is_final_schedule": False,
                    "slot_source": "pretrained_music_semantic_teacher",
                }
            )
        song_uid = "aud_" + audio_sha256(audio_path)[:24]
        payload = {
            "descriptor_type": "music_semantic_slot_descriptor",
            "descriptor_schema_version": "v46_38_mssd_aesd_routing_descriptor",
            "usage": "train_semantic",
            "is_final_schedule": False,
            "slot_source": "pretrained_music_semantic_teacher",
            "audio": str(audio_path),
            "song_uid": song_uid,
            "fps": float(fps),
            "num_slots": len(slots),
            "slots": slots,
            "segments": slots,
            "teacher_checkpoint": str(checkpoint_path.resolve()),
            "weak_supervision": True,
            "phrase_intervals_are_structure_segmented": True,
            "segmentation": segmentation,
        }
        target = out_dir / f"{audio_path.stem}.{song_uid[4:16]}.mssd.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        reports.append({"audio": str(audio_path), "mssd": str(target), "slots": len(slots)})
    report = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint_path.resolve()),
        "num_songs": len(reports),
        "outputs": reports,
        "phrase_intervals_are_structure_segmented": True,
        "ok": True,
    }
    (out_dir / "music_semantic_teacher_inference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset")
    build.add_argument("--music_dirs", nargs="*", default=[])
    build.add_argument("--music_manifest", default="")
    build.add_argument("--cache_dir", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--num_frames", type=int, default=1800)
    build.add_argument("--phrases_per_song", type=int, default=0, help=argparse.SUPPRESS)
    build.add_argument("--min_phrase_seconds", type=float, default=1.6)
    build.add_argument("--max_phrase_seconds", type=float, default=7.5)
    build.add_argument("--boundary_quantile", type=float, default=0.68)
    build.add_argument("--beat_snap_seconds", type=float, default=0.35)
    build.add_argument("--weak_prior_alpha", type=float, default=0.25)

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--music_prior_ckpt", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--validation_ratio", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=20260724)
    train.add_argument("--learning_rate", type=float, default=2.0e-4)
    train.add_argument("--weight_decay", type=float, default=1.0e-4)
    train.add_argument("--freeze_music_encoder", default="1")
    train.add_argument("--patience", type=int, default=15)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--music_dirs", nargs="*", default=[])
    infer.add_argument("--music_manifest", default="")
    infer.add_argument("--cache_dir", required=True)
    infer.add_argument("--out_dir", required=True)
    infer.add_argument("--num_frames", type=int, default=1800)
    infer.add_argument("--phrases_per_song", type=int, default=0, help=argparse.SUPPRESS)
    infer.add_argument("--min_phrase_seconds", type=float, default=1.6)
    infer.add_argument("--max_phrase_seconds", type=float, default=7.5)
    infer.add_argument("--boundary_quantile", type=float, default=0.68)
    infer.add_argument("--beat_snap_seconds", type=float, default=0.35)
    infer.add_argument("--fps", type=float, default=30.0)

    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        audio_paths = (
            load_audio_manifest(Path(args.music_manifest))
            if args.music_manifest
            else [Path(value) for value in discover_training_audio(args.music_dirs)]
        )
        report = build_dataset(
            audio_paths,
            Path(args.cache_dir),
            Path(args.out),
            num_frames=int(args.num_frames),
            min_phrase_seconds=float(args.min_phrase_seconds),
            max_phrase_seconds=float(args.max_phrase_seconds),
            boundary_quantile=float(args.boundary_quantile),
            beat_snap_seconds=float(args.beat_snap_seconds),
            weak_prior_alpha=float(args.weak_prior_alpha),
        )
    elif args.command == "train":
        report = train_teacher(
            Path(args.data),
            Path(args.music_prior_ckpt),
            Path(args.out),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            validation_ratio=float(args.validation_ratio),
            seed=int(args.seed),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            freeze_music_encoder=_bool(args.freeze_music_encoder),
            patience=int(args.patience),
        )
    else:
        audio_paths = (
            load_audio_manifest(Path(args.music_manifest))
            if args.music_manifest
            else [Path(value) for value in discover_training_audio(args.music_dirs)]
        )
        report = infer_mssd(
            Path(args.checkpoint),
            audio_paths,
            Path(args.out_dir),
            cache_dir=Path(args.cache_dir),
            num_frames=int(args.num_frames),
            min_phrase_seconds=float(args.min_phrase_seconds),
            max_phrase_seconds=float(args.max_phrase_seconds),
            boundary_quantile=float(args.boundary_quantile),
            beat_snap_seconds=float(args.beat_snap_seconds),
            fps=float(args.fps),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
