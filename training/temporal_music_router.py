#!/usr/bin/env python3
"""Build and train the formal zero-label CTSR-Weak temporal Router.

There are no paired Chang-E audio-motion samples.  Music is trained from
Librosa 12D phrase sequences and a declared sparse Semantic-OT weak teacher;
the transport plan is never represented as ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model.temporal_music_motion_router import TemporalMusicMotionRouter
from scheduling.index_io import load_shared_index
from scheduling.music_phrase_segmentation import (
    segment_music_phrases,
    split_music_phrases_for_events,
    whole_song_features,
)
from scheduling.temporal_router_contract import (
    TEMPORAL_ROUTER_ARCHITECTURE,
    phrase_feature_sequences,
    scientific_supervision_contract,
)
from support.scheduler_checkpoint_contracts import scheduler_training_contract
from training.music_corpus import (
    assert_content_disjoint,
    audio_sha256,
    discover_training_audio,
)
from training.weak_semantic_ot import (
    TEACHER_SCHEMA,
    observable_music_target,
    sparse_sinkhorn_teacher,
    weighted_control_cost,
)


DATASET_SCHEMA = "ctsr_weak_temporal_router_dataset_v1"
CHECKPOINT_VERSION = "formal_ctsr_weak_temporal_router_v1"


def _bool(value: str | int | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _group_split(
    groups: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted({str(value) for value in groups}), dtype=object)
    if len(unique) < 2:
        raise RuntimeError("Temporal Router training requires at least two songs")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val_count = max(1, min(len(unique) - 1, int(round(len(unique) * val_ratio))))
    validation = {str(value) for value in unique[:val_count]}
    val_indices = np.asarray(
        [index for index, value in enumerate(groups) if str(value) in validation],
        dtype=np.int64,
    )
    train_indices = np.asarray(
        [index for index, value in enumerate(groups) if str(value) not in validation],
        dtype=np.int64,
    )
    return train_indices, val_indices


def _corpus_fingerprint(song_uids: Sequence[str]) -> str:
    payload = "\n".join(sorted({str(value) for value in song_uids})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_dataset(args: argparse.Namespace) -> int:
    metadata, arrays, items = load_shared_index(args.index_json, args.index_npz)
    try:
        rates = [float(value) for value in metadata.get("canonical_fps_values", [])]
        if rates != [float(args.fps)]:
            raise RuntimeError(
                f"Temporal Router dataset FPS mismatch: index={rates}, requested={[float(args.fps)]}"
            )
        motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    finally:
        arrays.close()
    if motion_desc.ndim != 2 or motion_desc.shape[1] != 12:
        raise RuntimeError(f"Temporal Router requires 12D motion descriptors, got {motion_desc.shape}")

    balance_key = str(args.teacher_balance_key)
    event_groups = [
        str(item.get(balance_key, item.get("recording_uid", item.get("source_uid", "unknown"))))
        for item in items
    ]
    event_uids = [str(item["event_uid"]) for item in items]
    audio_paths = discover_training_audio(args.music_dirs)
    if int(args.expected_num_songs) > 0 and len(audio_paths) != int(args.expected_num_songs):
        raise RuntimeError(
            "Formal CTSR-Weak must extract every expected song exactly once; "
            f"unique_content_songs={len(audio_paths)}, expected={args.expected_num_songs}"
        )
    heldout_report = assert_content_disjoint(audio_paths, args.heldout_audio)
    if not _bool(args.require_librosa_backend):
        raise RuntimeError("Formal CTSR-Weak requires --require_librosa_backend=1")

    all_sequences: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_teacher: list[np.ndarray] = []
    song_uids: list[str] = []
    phrase_starts: list[int] = []
    phrase_ends: list[int] = []
    neighbor_indices: list[int] = []
    neighbor_directions: list[int] = []
    feature_reports: list[dict[str, Any]] = []
    transport_reports: list[dict[str, Any]] = []

    for song_index, audio_path in enumerate(audio_paths):
        features, feature_metadata = whole_song_features(
            audio_path,
            fps=float(args.fps),
            cache_dir=args.cache_dir,
            max_seconds=float(args.max_seconds),
            require_rhythm=bool(_bool(args.require_rhythm_features)),
            require_librosa=True,
        )
        extractor = dict(feature_metadata.get("extractor", {}))
        if extractor.get("backend") != "librosa" or not extractor.get("backend_version"):
            raise RuntimeError(
                f"Formal CTSR-Weak received unproven audio backend for {audio_path}: {extractor}"
            )
        feature_reports.append(
            {
                "audio": str(audio_path.resolve()),
                "audio_sha256": str(
                    feature_metadata.get("audio_sha256") or audio_sha256(audio_path)
                ),
                "backend": extractor.get("backend"),
                "backend_version": extractor.get("backend_version"),
                "num_frames": int(len(features)),
            }
        )
        phrases, segmentation = segment_music_phrases(
            features,
            fps=float(args.fps),
            min_phrase_seconds=float(args.min_phrase_seconds),
            max_phrase_seconds=float(args.max_phrase_seconds),
            boundary_quantile=float(args.boundary_quantile),
            beat_snap_seconds=float(args.beat_snap_seconds),
        )
        phrases, expansion = split_music_phrases_for_events(
            features,
            phrases,
            fps=float(args.fps),
            enabled=True,
            max_slot_seconds=float(args.max_slot_seconds),
            min_slot_seconds=float(args.min_slot_seconds),
            max_events_per_phrase=int(args.max_events_per_phrase),
            beat_snap_seconds=float(args.slot_beat_snap_seconds),
            calm_max_slot_seconds=float(args.calm_max_slot_seconds),
        )
        if not phrases:
            raise RuntimeError(f"No structure-derived phrases were found for {audio_path}")
        sequences = phrase_feature_sequences(
            features, phrases, int(args.sequence_frames)
        )
        targets = np.stack(
            [
                observable_music_target(
                    features[int(phrase.start) : int(phrase.end)],
                    float(phrase.length) / float(args.fps),
                )
                for phrase in phrases
            ]
        ).astype(np.float32)
        cost = weighted_control_cost(targets, motion_desc)
        teacher, transport_report = sparse_sinkhorn_teacher(
            cost,
            event_groups,
            top_k=int(args.teacher_top_k),
            epsilon=float(args.teacher_epsilon),
            max_iterations=int(args.teacher_max_iterations),
            tolerance=float(args.teacher_tolerance),
        )
        maximum_allowed_error = max(
            1.0e-4, 10.0 * float(args.teacher_tolerance)
        )
        if (
            not bool(transport_report["converged"])
            or float(transport_report["row_marginal_error"]) > maximum_allowed_error
            or float(transport_report["column_marginal_error"]) > maximum_allowed_error
        ):
            raise RuntimeError(
                "Semantic OT teacher failed its marginal contract for "
                f"{audio_path}: {transport_report}"
            )
        if not np.allclose(teacher.sum(axis=1), 1.0, atol=1.0e-5):
            raise RuntimeError(f"Semantic OT teacher rows are not normalized for {audio_path}")
        uid = "aud_" + audio_sha256(audio_path)[:24]
        offset = len(all_sequences)
        for local_index, phrase in enumerate(phrases):
            all_sequences.append(sequences[local_index])
            all_targets.append(targets[local_index])
            all_teacher.append(teacher[local_index])
            song_uids.append(uid)
            phrase_starts.append(int(phrase.start))
            phrase_ends.append(int(phrase.end))
            if len(phrases) == 1:
                neighbor_indices.append(offset)
                neighbor_directions.append(0)
            elif local_index + 1 < len(phrases):
                neighbor_indices.append(offset + local_index + 1)
                neighbor_directions.append(1)
            else:
                neighbor_indices.append(offset + local_index - 1)
                neighbor_directions.append(-1)
        transport_reports.append(
            {
                "song_uid": uid,
                "audio": str(audio_path.resolve()),
                "segmentation_num_phrases": int(segmentation["num_phrases"]),
                "event_slot_num_phrases": int(len(phrases)),
                "slot_expansion": expansion,
                **transport_report,
            }
        )
        if (song_index + 1) % 25 == 0 or song_index + 1 == len(audio_paths):
            print(
                f"[CTSR-Weak data] {song_index + 1}/{len(audio_paths)} songs; "
                f"phrases={len(all_sequences)}",
                flush=True,
            )

    if not all_sequences:
        raise RuntimeError("CTSR-Weak dataset is empty")
    sequence_array = np.stack(all_sequences).astype(np.float32)
    target_array = np.stack(all_targets).astype(np.float32)
    teacher_array = np.stack(all_teacher).astype(np.float32)
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        music_sequences=sequence_array,
        observable_targets=target_array,
        teacher_probabilities=teacher_array,
        motion_desc=motion_desc,
        event_uids=np.asarray(event_uids, dtype=object),
        event_groups=np.asarray(event_groups, dtype=object),
        song_uids=np.asarray(song_uids, dtype=object),
        phrase_starts=np.asarray(phrase_starts, dtype=np.int64),
        phrase_ends=np.asarray(phrase_ends, dtype=np.int64),
        neighbor_indices=np.asarray(neighbor_indices, dtype=np.int64),
        neighbor_directions=np.asarray(neighbor_directions, dtype=np.int8),
        fps=np.asarray(float(args.fps), dtype=np.float32),
        event_db_contract_json=np.asarray(
            json.dumps(metadata["event_db_contract"], sort_keys=True), dtype=object
        ),
        dataset_schema=np.asarray(DATASET_SCHEMA, dtype=object),
        teacher_schema=np.asarray(TEACHER_SCHEMA, dtype=object),
    )
    backend_counts: dict[str, int] = {}
    backend_versions: dict[str, int] = {}
    for row in feature_reports:
        backend_counts[str(row["backend"])] = backend_counts.get(str(row["backend"]), 0) + 1
        version = str(row["backend_version"])
        backend_versions[version] = backend_versions.get(version, 0) + 1
    report = {
        "schema": DATASET_SCHEMA,
        "ok": True,
        "dataset": str(target),
        "num_songs": int(len(audio_paths)),
        "num_phrases": int(len(sequence_array)),
        "num_events": int(len(event_uids)),
        "sequence_shape": list(sequence_array.shape),
        "fps": float(args.fps),
        "event_db_contract": metadata["event_db_contract"],
        "content_disjoint": heldout_report,
        "training_corpus_sha256": _corpus_fingerprint(song_uids),
        "backend_counts": backend_counts,
        "backend_version_counts": backend_versions,
        "scientific_contract": scientific_supervision_contract(),
        "segmentation_contract": {
            "shared_with_inference": True,
            "min_phrase_seconds": float(args.min_phrase_seconds),
            "max_phrase_seconds": float(args.max_phrase_seconds),
            "boundary_quantile": float(args.boundary_quantile),
            "beat_snap_seconds": float(args.beat_snap_seconds),
            "max_slot_seconds": float(args.max_slot_seconds),
            "min_slot_seconds": float(args.min_slot_seconds),
            "max_events_per_phrase": int(args.max_events_per_phrase),
            "sequence_frames": int(args.sequence_frames),
        },
        "teacher_contract": {
            "schema": TEACHER_SCHEMA,
            "balance_key": balance_key,
            "top_k": int(args.teacher_top_k),
            "epsilon": float(args.teacher_epsilon),
            "max_iterations": int(args.teacher_max_iterations),
            "tolerance": float(args.teacher_tolerance),
            "fail_closed_on_nonconvergence": True,
            "is_ground_truth": False,
        },
        "transport_summary": {
            "songs_converged": int(sum(bool(row["converged"]) for row in transport_reports)),
            "songs_total": int(len(transport_reports)),
            "max_row_marginal_error": float(max(row["row_marginal_error"] for row in transport_reports)),
            "max_column_marginal_error": float(max(row["column_marginal_error"] for row in transport_reports)),
            "mean_teacher_entropy": float(np.mean([row["mean_teacher_entropy"] for row in transport_reports])),
        },
        "feature_cache_entries": feature_reports,
        "transport_reports": transport_reports,
    }
    target.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key not in {"feature_cache_entries", "transport_reports"}}, indent=2))
    return 0


def train_model(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with np.load(args.data, allow_pickle=True) as data:
        sequences = np.asarray(data["music_sequences"], dtype=np.float32)
        teacher = np.asarray(data["teacher_probabilities"], dtype=np.float32)
        motion_desc = np.asarray(data["motion_desc"], dtype=np.float32)
        song_uids = np.asarray(data["song_uids"], dtype=object)
        neighbor_indices = np.asarray(data["neighbor_indices"], dtype=np.int64)
        neighbor_directions = np.asarray(data["neighbor_directions"], dtype=np.int8)
        dataset_schema = str(np.asarray(data["dataset_schema"]).item())
    if dataset_schema != DATASET_SCHEMA:
        raise RuntimeError(f"Unsupported CTSR-Weak dataset schema: {dataset_schema!r}")
    if sequences.ndim != 3 or sequences.shape[1:] != (int(args.sequence_frames), 12):
        raise RuntimeError(f"Invalid CTSR-Weak music sequence shape: {sequences.shape}")
    if teacher.shape != (len(sequences), len(motion_desc)):
        raise RuntimeError(
            f"Teacher/event mismatch: teacher={teacher.shape}, motion={motion_desc.shape}"
        )
    if not np.isfinite(sequences).all() or not np.isfinite(teacher).all():
        raise RuntimeError("CTSR-Weak dataset contains NaN/Inf")
    if np.any(neighbor_indices < 0) or np.any(neighbor_indices >= len(sequences)):
        raise RuntimeError("CTSR-Weak neighbor indices are out of range")
    if neighbor_directions.shape != (len(sequences),) or not set(
        np.unique(neighbor_directions).tolist()
    ).issubset({-1, 0, 1}):
        raise RuntimeError("CTSR-Weak temporal neighbor directions are invalid")

    train_indices, val_indices = _group_split(song_uids, args.val_ratio, args.seed)
    sequence_tensor = torch.from_numpy(sequences)
    teacher_tensor = torch.from_numpy(teacher)
    neighbor_tensor = torch.from_numpy(sequences[neighbor_indices])
    direction_tensor = torch.from_numpy(neighbor_directions.astype(np.int64))

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        selected = torch.from_numpy(indices.astype(np.int64, copy=False))
        dataset = TensorDataset(
            sequence_tensor.index_select(0, selected),
            teacher_tensor.index_select(0, selected),
            neighbor_tensor.index_select(0, selected),
            direction_tensor.index_select(0, selected),
        )
        return DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=shuffle,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalMusicMotionRouter(
        music_dim=12,
        motion_dim=12,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        dropout=float(args.dropout),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        init_temperature=float(args.init_temperature),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    motion_tensor = torch.from_numpy(motion_desc).to(device)
    mask_ratio = float(args.mask_ratio)
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be between zero and one")

    def run_epoch(data_loader: DataLoader, training: bool) -> dict[str, float]:
        model.train(training)
        totals = {"loss": 0.0, "teacher": 0.0, "reconstruction": 0.0, "consistency": 0.0, "temporal_order": 0.0}
        count = 0
        for sequence_batch, teacher_batch, neighbor_batch, direction_batch in data_loader:
            sequence_batch = sequence_batch.to(device, non_blocking=True)
            teacher_batch = teacher_batch.to(device, non_blocking=True)
            neighbor_batch = neighbor_batch.to(device, non_blocking=True)
            direction_batch = direction_batch.to(device, non_blocking=True)
            if training:
                mask = torch.rand(sequence_batch.shape[:2], device=device) < mask_ratio
            else:
                period = max(2, int(round(1.0 / mask_ratio)))
                position = torch.arange(sequence_batch.shape[1], device=device)
                mask = (position % period == 0).unsqueeze(0).expand(sequence_batch.shape[0], -1)
            masked = sequence_batch.clone()
            masked[mask] = 0.0
            with torch.set_grad_enabled(training):
                masked_frames = model.music_encoder.frame_features(masked)
                masked_embedding = model.music_encoder.pooled_embedding(masked_frames)
                reconstruction = model.music_encoder.frame_decoder(masked_frames)
                clean_embedding = model.encode_music(sequence_batch)
                neighbor_embedding = model.encode_music(neighbor_batch)
                motion_embedding = model.encode_motion(motion_tensor)
                logits = model.logit_scale.exp().clamp(max=100.0) * (
                    masked_embedding @ motion_embedding.t()
                )
                teacher_loss = -(
                    teacher_batch * F.log_softmax(logits, dim=-1)
                ).sum(dim=-1).mean()
                reconstruction_loss = F.smooth_l1_loss(
                    reconstruction[mask], sequence_batch[mask]
                )
                consistency_loss = (
                    1.0 - (masked_embedding * clean_embedding.detach()).sum(dim=-1)
                ).mean()
                valid_order = direction_batch != 0
                if bool(valid_order.any()):
                    valid_direction = direction_batch[valid_order]
                    valid_clean = clean_embedding[valid_order]
                    valid_neighbor = neighbor_embedding[valid_order]
                    forward_first = torch.where(
                        (valid_direction > 0).unsqueeze(1), valid_clean, valid_neighbor
                    )
                    forward_second = torch.where(
                        (valid_direction > 0).unsqueeze(1), valid_neighbor, valid_clean
                    )
                    forward_logits = model.temporal_order_logits(
                        forward_first, forward_second
                    )
                    reverse_logits = model.temporal_order_logits(
                        forward_second, forward_first
                    )
                    temporal_order_loss = 0.5 * (
                        F.cross_entropy(
                            forward_logits,
                            torch.ones(
                                len(forward_logits), dtype=torch.long, device=device
                            ),
                        )
                        + F.cross_entropy(
                            reverse_logits,
                            torch.zeros(
                                len(reverse_logits), dtype=torch.long, device=device
                            ),
                        )
                    )
                else:
                    temporal_order_loss = model.temporal_order_head[0].weight.sum() * 0.0
                loss = (
                    teacher_loss
                    + float(args.reconstruction_weight) * reconstruction_loss
                    + float(args.consistency_weight) * consistency_loss
                    + float(args.temporal_order_weight) * temporal_order_loss
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            batch_size = len(sequence_batch)
            for key, value in (
                ("loss", loss),
                ("teacher", teacher_loss),
                ("reconstruction", reconstruction_loss),
                ("consistency", consistency_loss),
                ("temporal_order", temporal_order_loss),
            ):
                totals[key] += float(value.detach()) * batch_size
            count += batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    metadata, arrays, _items = load_shared_index(args.index_json, args.index_npz)
    arrays.close()
    scheduler_contract = scheduler_training_contract(
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
        "architecture": TEMPORAL_ROUTER_ARCHITECTURE,
        "music_dim": 12,
        "motion_dim": 12,
        "hidden_dim": int(args.hidden_dim),
        "latent_dim": int(args.latent_dim),
        "dropout": float(args.dropout),
        "transformer_layers": int(args.transformer_layers),
        "transformer_heads": int(args.transformer_heads),
        "sequence_frames": int(args.sequence_frames),
        "init_temperature": float(args.init_temperature),
        "inference_temperature": float(args.inference_temperature),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "fps": float(args.fps),
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    train_loader = loader(train_indices, True)
    val_loader = loader(val_indices, False)
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(train_loader, True)
        val_metrics = run_epoch(val_loader, False)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        if val_metrics["loss"] < best - 1.0e-7:
            best = val_metrics["loss"]
            patience = 0
            torch.save(
                {
                    "version": CHECKPOINT_VERSION,
                    "architecture": TEMPORAL_ROUTER_ARCHITECTURE,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "fps": float(args.fps),
                    "epoch": int(epoch),
                    "val_loss": float(best),
                    "scheduler_contract": scheduler_contract,
                    "scientific_contract": scientific_supervision_contract(),
                    "training_contract": {
                        "dataset_schema": DATASET_SCHEMA,
                        "teacher_schema": TEACHER_SCHEMA,
                        "song_disjoint_validation": True,
                        "external_pretrained_model": False,
                        "music_encoder_initialized_from_scratch": True,
                        "human_training_labels": 0,
                    },
                },
                output,
            )
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[CTSR-Weak] epoch={epoch} train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} best={best:.6f}",
                flush=True,
            )
        if patience >= int(args.patience):
            break
    output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    output.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "schema": "ctsr_weak_router_training_provenance_v1",
                "checkpoint": str(output),
                "dataset": str(Path(args.data).resolve()),
                "scientific_contract": scientific_supervision_contract(),
                "best_validation_loss": float(best),
                "epochs_completed": int(len(history)),
                "song_disjoint_validation": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PASS] CTSR-Weak Router checkpoint: {output}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-dataset")
    build.add_argument("--index_json", required=True)
    build.add_argument("--index_npz", required=True)
    build.add_argument("--music_dirs", nargs="+", required=True)
    build.add_argument("--heldout_audio", nargs="*", default=[])
    build.add_argument("--cache_dir", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--fps", type=float, required=True)
    build.add_argument("--expected_num_songs", type=int, default=0)
    build.add_argument("--max_seconds", type=float, default=0.0)
    build.add_argument("--min_phrase_seconds", type=float, default=2.5)
    build.add_argument("--max_phrase_seconds", type=float, default=7.5)
    build.add_argument("--boundary_quantile", type=float, default=0.68)
    build.add_argument("--beat_snap_seconds", type=float, default=0.35)
    build.add_argument("--max_slot_seconds", type=float, default=5.0)
    build.add_argument("--calm_max_slot_seconds", type=float, default=4.5)
    build.add_argument("--min_slot_seconds", type=float, default=2.5)
    build.add_argument("--max_events_per_phrase", type=int, default=2)
    build.add_argument("--slot_beat_snap_seconds", type=float, default=0.25)
    build.add_argument("--sequence_frames", type=int, default=64)
    build.add_argument("--teacher_top_k", type=int, default=64)
    build.add_argument("--teacher_epsilon", type=float, default=0.12)
    build.add_argument("--teacher_max_iterations", type=int, default=5000)
    build.add_argument("--teacher_tolerance", type=float, default=1.0e-5)
    build.add_argument("--teacher_balance_key", default="recording_uid")
    build.add_argument("--require_librosa_backend", default="1")
    build.add_argument("--require_rhythm_features", default="1")

    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--index_json", required=True)
    train.add_argument("--index_npz", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--fps", type=float, required=True)
    train.add_argument("--sequence_frames", type=int, default=64)
    train.add_argument("--epochs", type=int, default=250)
    train.add_argument("--batch_size", type=int, default=64)
    train.add_argument("--lr", type=float, default=2.0e-4)
    train.add_argument("--weight_decay", type=float, default=1.0e-4)
    train.add_argument("--hidden_dim", type=int, default=128)
    train.add_argument("--latent_dim", type=int, default=96)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--transformer_layers", type=int, default=2)
    train.add_argument("--transformer_heads", type=int, default=4)
    train.add_argument("--init_temperature", type=float, default=0.12)
    train.add_argument("--inference_temperature", type=float, default=0.12)
    train.add_argument("--mask_ratio", type=float, default=0.20)
    train.add_argument("--reconstruction_weight", type=float, default=0.35)
    train.add_argument("--consistency_weight", type=float, default=0.10)
    train.add_argument("--temporal_order_weight", type=float, default=0.05)
    train.add_argument("--val_ratio", type=float, default=0.10)
    train.add_argument("--patience", type=int, default=40)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build-dataset":
        return build_dataset(args)
    if args.command == "train":
        return train_model(args)
    raise RuntimeError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
