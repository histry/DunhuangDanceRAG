#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the mixed-curvature grounder from sparse semantic-OT supervision.

The trainer consumes separate train and validation datasets produced by
``grounding.semantic_optimal_transport``.  This is deliberate: music is split
by whole song and motion is split by ``source_uid`` *before* constructing OT,
so no pseudo-pair identity graph is allowed to reconnect the partitions.

The resulting checkpoint uses the existing mixed-grounder model schema and is
therefore compatible with ``python -m grounding.mixed_curvature embed``.  Its
metadata explicitly records weak supervision and never claims paired ground
truth or retrieval accuracy against human annotations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None
    F = None
    DataLoader = None
    Dataset = object

from grounding.manifold_ops import (
    EPS,
    gaussian_wasserstein_distance_sq_torch,
)
from grounding.mixed_curvature import (
    SCHEMA as MIXED_CHECKPOINT_SCHEMA,
    MixedCurvatureGrounder,
    MixedGrounderConfig,
    _env_bool,
    _hierarchy_loss,
    apply_normalization,
    fit_train_normalization,
)
from grounding.paired_data import validate_paired_payload
from grounding.semantic_optimal_transport import SCHEMA as OT_DATASET_SCHEMA


TRAINING_SCHEMA = "dunhuang_semantic_ot_mixed_grounder_training_v1"


def load_semantic_ot_dataset(path: Path) -> Tuple[Dict[str, Any], Dict[str, int]]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    schema = str(np.asarray(payload.get("schema", "")).reshape(-1)[0])
    if schema != OT_DATASET_SCHEMA:
        raise RuntimeError(
            f"Expected {OT_DATASET_SCHEMA}, got {schema!r}: {path}"
        )
    dimensions = validate_paired_payload(payload)
    required = (
        "phrase_ids",
        "song_ids",
        "song_tokens",
        "teacher_pair_weight",
        "teacher_music_probs",
        "teacher_action_probs",
        "teacher_js_divergence",
        "teacher_entropy",
        "teacher_margin",
        "is_ground_truth_pair",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Semantic-OT dataset misses fields: {missing}")
    rows = dimensions["rows"]
    for key in required:
        value = np.asarray(payload[key])
        if value.ndim == 0 or len(value) != rows:
            raise RuntimeError(f"Semantic-OT field {key} is not row-aligned")
    if np.any(np.asarray(payload["is_ground_truth_pair"], dtype=bool)):
        raise RuntimeError(
            "Semantic-OT training refuses datasets that claim ground-truth pairs"
        )
    if set(map(str, np.asarray(payload.get("supervision", []), dtype=object))) != {
        "semantic_ot_teacher"
    }:
        raise RuntimeError("Unexpected supervision label in semantic-OT dataset")
    _phrase_layout(payload)
    return payload, dimensions


def _phrase_layout(payload: Mapping[str, Any]) -> Tuple[np.ndarray, int]:
    phrase_ids = np.asarray(payload["phrase_ids"], dtype=np.int64).reshape(-1)
    unique, counts = np.unique(phrase_ids, return_counts=True)
    if len(unique) < 1:
        raise RuntimeError("Semantic-OT dataset has no phrases")
    if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise RuntimeError("phrase_ids must be contiguous and start at zero")
    if len(np.unique(counts)) != 1:
        raise RuntimeError(
            "Every phrase must have the same candidate count for grouped training"
        )
    candidate_count = int(counts[0])
    expected = np.repeat(unique, candidate_count)
    if not np.array_equal(phrase_ids, expected):
        raise RuntimeError(
            "Rows must be contiguous by phrase; rebuild the semantic-OT dataset"
        )
    weights = np.asarray(payload["teacher_pair_weight"], dtype=np.float64).reshape(
        len(unique), candidate_count
    )
    if np.any(weights < 0.0) or not np.isfinite(weights).all():
        raise RuntimeError("teacher_pair_weight contains invalid values")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5):
        raise RuntimeError("Teacher candidate weights must sum to one per phrase")
    return unique, candidate_count


def _normalize_payload(
    payload: Mapping[str, Any], normalization: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    normalized = apply_normalization(payload, normalization)
    for key in (
        "phrase_ids",
        "song_ids",
        "song_tokens",
        "teacher_pair_weight",
        "teacher_music_probs",
        "teacher_action_probs",
        "teacher_js_divergence",
        "teacher_entropy",
        "teacher_margin",
        "candidate_rank",
    ):
        normalized[key] = np.asarray(payload[key])
    return normalized


if torch is not None:

    class PhraseGroupDataset(Dataset):
        def __init__(self, payload: Mapping[str, np.ndarray]):
            self.payload = payload
            self.phrase_ids, self.candidate_count = _phrase_layout(payload)
            self.order = (
                "clap",
                "temporal",
                "motion_geometry",
                "bodypart_flow",
                "gaussian_mean",
                "gaussian_covariance",
                "controls",
                "quality",
                "family_ids",
                "source_ids",
                "event_indices",
                "teacher_pair_weight",
                "teacher_music_probs",
                "teacher_action_probs",
                "teacher_js_divergence",
                "teacher_entropy",
                "teacher_margin",
            )

        def __len__(self) -> int:
            return int(len(self.phrase_ids))

        def __getitem__(self, index: int):
            start = int(index) * self.candidate_count
            end = start + self.candidate_count
            tensors = []
            for key in self.order:
                value = np.asarray(self.payload[key][start:end])
                tensors.append(torch.from_numpy(value))
            return tuple(tensors)


def _soft_transport_bidirectional_loss(
    logits: "torch.Tensor",
    target: "torch.Tensor",
    phrase_confidence: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(EPS)
    row_loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    phrase_confidence = phrase_confidence.clamp(0.05, 1.0)
    audio_to_motion = (
        row_loss * phrase_confidence
    ).sum() / phrase_confidence.sum().clamp_min(EPS)

    reverse_target = target.transpose(0, 1)
    reverse_target = reverse_target / reverse_target.sum(
        dim=-1, keepdim=True
    ).clamp_min(EPS)
    reverse_row_loss = -(
        reverse_target * F.log_softmax(logits.transpose(0, 1), dim=-1)
    ).sum(dim=-1)
    reverse_confidence = reverse_target.max(dim=-1).values.clamp(0.05, 1.0)
    motion_to_audio = (
        reverse_row_loss * reverse_confidence
    ).sum() / reverse_confidence.sum().clamp_min(EPS)
    return 0.5 * (audio_to_motion + motion_to_audio), audio_to_motion, motion_to_audio


def _semantic_cross_phrase_target(
    music_probabilities: "torch.Tensor",
    action_probabilities: "torch.Tensor",
    temperature: float,
) -> "torch.Tensor":
    music = music_probabilities.clamp_min(EPS)
    action = action_probabilities.clamp_min(EPS)
    music = music / music.sum(dim=-1, keepdim=True).clamp_min(EPS)
    action = action / action.sum(dim=-1, keepdim=True).clamp_min(EPS)
    midpoint = 0.5 * (music[:, None, :] + action[None, :, :])
    divergence = 0.5 * (
        (
            music[:, None, :]
            * (torch.log(music[:, None, :]) - torch.log(midpoint))
        ).sum(dim=-1)
        + (
            action[None, :, :]
            * (torch.log(action[None, :, :]) - torch.log(midpoint))
        ).sum(dim=-1)
    )
    return torch.softmax(-divergence / max(float(temperature), 0.05), dim=-1)


def _expected_calibration_loss(
    distance: "torch.Tensor",
    variance: "torch.Tensor",
    target: "torch.Tensor",
    phrase_confidence: "torch.Tensor",
) -> "torch.Tensor":
    nll = distance / (2.0 * variance) + 0.5 * torch.log(variance)
    expected = (target * nll).sum(dim=-1)
    confidence = phrase_confidence.clamp(0.05, 1.0)
    return (expected * confidence).sum() / confidence.sum().clamp_min(EPS)



def _weighted_gaussian_anchor(
    gaussian_per_part: "torch.Tensor",
    candidate_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Reduce body-part Gaussian distances before OT candidate weighting."""
    if gaussian_per_part.ndim < 1:
        raise RuntimeError(
            "Gaussian anchor distance must contain a candidate dimension"
        )

    weights = candidate_weight.reshape(-1).clamp_min(1.0e-6)

    if gaussian_per_part.shape[0] != weights.shape[0]:
        raise RuntimeError(
            "Gaussian candidate count does not match OT candidate weights: "
            f"{tuple(gaussian_per_part.shape)} vs {tuple(weights.shape)}"
        )

    # gaussian_per_part is normally [num_candidates, num_body_parts].
    # Aggregate body-part geometry first, producing one loss per candidate.
    gaussian_per_row = gaussian_per_part.reshape(
        weights.shape[0], -1
    ).mean(dim=-1)

    return (
        gaussian_per_row * weights
    ).sum() / weights.sum().clamp_min(EPS)

def _batch_loss(
    model: "MixedCurvatureGrounder",
    batch: Sequence["torch.Tensor"],
    *,
    training: bool,
    hierarchy_weight: float,
    gaussian_anchor_weight: float,
    control_weight: float,
    uncertainty_weight: float,
    source_weight: float,
    metric_balance_weight: float,
    hierarchy_margin: float,
    cross_phrase_target_weight: float,
    cross_phrase_temperature: float,
) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
    (
        clap_group,
        temporal_group,
        geometry_group,
        bodypart_group,
        gaussian_mean_group,
        gaussian_covariance_group,
        controls_group,
        quality_group,
        family_group,
        source_group,
        _event_indices,
        pair_weight_group,
        music_probs_group,
        action_probs_group,
        js_group,
        entropy_group,
        margin_group,
    ) = batch
    batch_size, candidate_count = pair_weight_group.shape
    clap = clap_group[:, 0]
    temporal = temporal_group[:, 0]
    geometry = geometry_group.reshape(batch_size * candidate_count, *geometry_group.shape[2:])
    bodypart = bodypart_group.reshape(batch_size * candidate_count, *bodypart_group.shape[2:])
    gaussian_mean = gaussian_mean_group.reshape(
        batch_size * candidate_count, *gaussian_mean_group.shape[2:]
    )
    gaussian_covariance = gaussian_covariance_group.reshape(
        batch_size * candidate_count, *gaussian_covariance_group.shape[2:]
    )
    controls = controls_group.reshape(batch_size * candidate_count, -1)
    quality = quality_group.reshape(batch_size * candidate_count)
    family_ids = family_group.reshape(batch_size * candidate_count)
    source_ids = source_group.reshape(batch_size * candidate_count)
    pair_weights = pair_weight_group.float()
    local_target = torch.zeros(
        (batch_size, batch_size * candidate_count),
        dtype=pair_weights.dtype,
        device=pair_weights.device,
    )
    for row in range(batch_size):
        start = row * candidate_count
        local_target[row, start : start + candidate_count] = pair_weights[row]
    semantic_target = _semantic_cross_phrase_target(
        music_probs_group[:, 0].float(),
        action_probs_group.reshape(
            batch_size * candidate_count, action_probs_group.shape[-1]
        ).float(),
        float(cross_phrase_temperature),
    )
    cross_weight = float(np.clip(cross_phrase_target_weight, 0.0, 0.5))
    target = (1.0 - cross_weight) * local_target + cross_weight * semantic_target
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(EPS)
    phrase_confidence = (
        0.50 * (1.0 - entropy_group[:, 0].float()).clamp(0.0, 1.0)
        + 0.50 * margin_group[:, 0].float().clamp(0.0, 1.0)
    ).clamp(0.05, 1.0)

    audio = model.encode_audio(clap, temporal)
    motion = model.encode_motion(geometry, bodypart)
    logits, distance, variance = model.pairwise_logits(audio, motion)
    contrastive, audio_to_motion, motion_to_audio = _soft_transport_bidirectional_loss(
        logits, target, phrase_confidence
    )
    hierarchy = _hierarchy_loss(
        motion, family_ids, model.curvature, hierarchy_margin
    )
    gaussian_per_part = gaussian_wasserstein_distance_sq_torch(
        motion["gaussian_mean"],
        motion["gaussian_covariance"],
        gaussian_mean,
        gaussian_covariance,
        model.config.minimum_covariance,
    )
    candidate_weight = pair_weights.reshape(-1).clamp_min(1.0e-6)
    gaussian_anchor = _weighted_gaussian_anchor(
        gaussian_per_part,
        candidate_weight,
    )
    phrase_controls = (pair_weights[..., None] * controls_group).sum(dim=1)
    motion_control_rows = F.smooth_l1_loss(
        motion["euclidean"], controls, reduction="none"
    ).mean(dim=-1)
    control = 0.5 * (
        F.smooth_l1_loss(audio["euclidean"], phrase_controls)
        + (motion_control_rows * candidate_weight).sum()
        / candidate_weight.sum().clamp_min(EPS)
    )
    uncertainty = _expected_calibration_loss(
        distance, variance, target, phrase_confidence
    )
    source_logits = model.source_logits(motion, scale=1.0) if training else None
    source = (
        F.cross_entropy(source_logits, source_ids, reduction="none")
        if source_logits is not None
        else contrastive.new_zeros((len(source_ids),))
    )
    if source.ndim > 0:
        source = (source * candidate_weight).sum() / candidate_weight.sum().clamp_min(EPS)
    metric_balance = -torch.log(
        model.metric_weights * len(model.metric_weights)
    ).mean()
    teacher_js = (
        js_group.reshape(-1).float() * candidate_weight
    ).sum() / candidate_weight.sum().clamp_min(EPS)
    total = (
        contrastive
        + float(hierarchy_weight) * hierarchy
        + float(gaussian_anchor_weight) * gaussian_anchor
        + float(control_weight) * control
        + float(uncertainty_weight) * uncertainty
        + float(source_weight) * source
        + float(metric_balance_weight) * metric_balance
    )
    return total, {
        "soft_distillation": contrastive,
        "audio_to_motion_distillation": audio_to_motion,
        "motion_to_audio_distillation": motion_to_audio,
        "hierarchy": hierarchy,
        "gaussian_anchor": gaussian_anchor,
        "control": control,
        "uncertainty": uncertainty,
        "source_adversarial": source,
        "metric_balance": metric_balance,
        "teacher_js_divergence": teacher_js,
    }


def _prediction_audit(
    model: "MixedCurvatureGrounder",
    loader: "DataLoader",
    device: "torch.device",
) -> Dict[str, float]:
    model.eval()
    weighted_top1 = 0.0
    weight_total = 0.0
    cross_entropy_total = 0.0
    phrase_total = 0
    source_coverage: list[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = tuple(value.to(device, non_blocking=True) for value in raw_batch)
            clap_group = batch[0]
            temporal_group = batch[1]
            geometry_group = batch[2]
            bodypart_group = batch[3]
            source_group = batch[9]
            pair_weights = batch[11].float()
            batch_size, candidate_count = pair_weights.shape
            audio = model.encode_audio(clap_group[:, 0], temporal_group[:, 0])
            motion = model.encode_motion(
                geometry_group.reshape(batch_size * candidate_count, -1),
                bodypart_group.reshape(
                    batch_size * candidate_count, *bodypart_group.shape[2:]
                ),
            )
            logits, _, _ = model.pairwise_logits(audio, motion)
            local_logits = torch.stack(
                [
                    logits[row, row * candidate_count : (row + 1) * candidate_count]
                    for row in range(batch_size)
                ],
                dim=0,
            )
            target = pair_weights / pair_weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
            cross_entropy_total += float(
                (-(target * F.log_softmax(local_logits, dim=-1)).sum(dim=-1)).sum().cpu()
            )
            predicted = local_logits.argmax(dim=-1)
            teacher_top = target.argmax(dim=-1)
            teacher_confidence = target.max(dim=-1).values
            weighted_top1 += float(
                ((predicted == teacher_top).float() * teacher_confidence).sum().cpu()
            )
            weight_total += float(teacher_confidence.sum().cpu())
            for row in range(batch_size):
                top_count = min(5, candidate_count)
                top = torch.topk(local_logits[row], top_count).indices
                sources = source_group[row].index_select(0, top)
                source_coverage.append(float(len(torch.unique(sources))) / top_count)
            phrase_total += batch_size
    return {
        "teacher_transport_cross_entropy": cross_entropy_total / max(phrase_total, 1),
        "teacher_weighted_top1_agreement": weighted_top1 / max(weight_total, 1.0e-8),
        "top5_source_coverage": float(np.mean(source_coverage)) if source_coverage else 0.0,
        "phrases": int(phrase_total),
        "ground_truth_retrieval_metric": False,
    }


def train_semantic_ot_grounder(
    train_data_path: Path,
    validation_data_path: Path,
    out_path: Path,
    *,
    epochs: int = 120,
    batch_phrases: int = 12,
    seed: int = 20260724,
    learning_rate: float = 2.0e-4,
    weight_decay: float = 1.0e-4,
    patience: int = 20,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for semantic-OT grounder training")
    train_payload, dimensions = load_semantic_ot_dataset(train_data_path)
    validation_payload, validation_dimensions = load_semantic_ot_dataset(
        validation_data_path
    )
    for key in (
        "clap_dim",
        "temporal_frames",
        "temporal_dim",
        "motion_geometry_dim",
        "bodypart_count",
        "gaussian_dim",
        "control_dim",
    ):
        if dimensions[key] != validation_dimensions[key]:
            raise RuntimeError(
                f"Train/validation dimension mismatch for {key}: "
                f"{dimensions[key]} vs {validation_dimensions[key]}"
            )
    train_songs = set(map(str, np.asarray(train_payload["song_tokens"], dtype=object)))
    val_songs = set(map(str, np.asarray(validation_payload["song_tokens"], dtype=object)))
    if train_songs.intersection(val_songs):
        raise RuntimeError("Train and validation phrase identities overlap")
    train_event_contract = str(np.asarray(train_payload["event_db_contract_json"]).item())
    val_event_contract = str(np.asarray(validation_payload["event_db_contract_json"]).item())
    if train_event_contract == val_event_contract:
        raise RuntimeError(
            "Train and validation semantic-OT datasets reference the same Event-DB; "
            "use source-disjoint motion splits"
        )

    train_indices = np.arange(dimensions["rows"], dtype=np.int64)
    normalization = fit_train_normalization(train_payload, train_indices)
    train_normalized = _normalize_payload(train_payload, normalization)
    validation_normalized = _normalize_payload(validation_payload, normalization)
    config = MixedGrounderConfig(
        clap_dim=dimensions["clap_dim"],
        temporal_dim=dimensions["temporal_dim"],
        motion_geometry_dim=dimensions["motion_geometry_dim"],
        bodypart_count=dimensions["bodypart_count"],
        bodypart_feature_dim=int(np.asarray(train_payload["bodypart_flow"]).shape[-1]),
        gaussian_dim=dimensions["gaussian_dim"],
        control_dim=dimensions["control_dim"],
        num_sources=int(np.max(train_payload["source_ids"])) + 1,
        hidden_dim=int(os.environ.get("V46_53_MIXED_HIDDEN", 192)),
        lorentz_dim=int(os.environ.get("V46_53_MIXED_LORENTZ_DIM", 16)),
        sphere_dim=int(os.environ.get("V46_53_MIXED_SPHERE_DIM", 96)),
        dropout=float(os.environ.get("V46_53_MIXED_DROPOUT", 0.10)),
        minimum_covariance=float(os.environ.get("V46_53_MIXED_COV_EPS", 1.0e-4)),
        initial_curvature=float(os.environ.get("V46_53_MIXED_CURVATURE", 1.0)),
        initial_temperature=float(os.environ.get("V46_53_MIXED_TEMPERATURE", 0.08)),
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and _env_bool("V46_53_MIXED_GROUNDER_CUDA", True)
        else "cpu"
    )
    model = MixedCurvatureGrounder(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    train_dataset = PhraseGroupDataset(train_normalized)
    validation_dataset = PhraseGroupDataset(validation_normalized)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(int(batch_phrases), len(train_dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(int(batch_phrases), len(validation_dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_weights = {
        "hierarchy_weight": float(os.environ.get("V46_53_MIXED_HIERARCHY_W", 0.20)),
        "gaussian_anchor_weight": float(os.environ.get("V46_53_MIXED_GAUSSIAN_W", 0.25)),
        "control_weight": float(os.environ.get("V46_53_MIXED_CONTROL_W", 0.10)),
        "uncertainty_weight": float(os.environ.get("V46_53_MIXED_UNCERTAINTY_W", 0.05)),
        "source_weight": float(os.environ.get("V46_53_MIXED_SOURCE_W", 0.05)),
        "metric_balance_weight": float(os.environ.get("V46_53_MIXED_METRIC_BALANCE_W", 0.01)),
        "hierarchy_margin": float(os.environ.get("V46_53_MIXED_HIERARCHY_MARGIN", 1.25)),
        "cross_phrase_target_weight": float(
            os.environ.get("SEMANTIC_OT_CROSS_PHRASE_TARGET_W", 0.15)
        ),
        "cross_phrase_temperature": float(
            os.environ.get("SEMANTIC_OT_CROSS_PHRASE_TEMPERATURE", 0.25)
        ),
    }

    def run_epoch(data_loader: "DataLoader", training: bool) -> Dict[str, float]:
        model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        for raw_batch in data_loader:
            batch = tuple(value.to(device, non_blocking=True) for value in raw_batch)
            with torch.set_grad_enabled(training):
                loss, pieces = _batch_loss(
                    model, batch, training=training, **loss_weights
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Semantic-OT grounder loss became non-finite")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
            batch_count = int(len(batch[0]))
            count += batch_count
            for name, value in {"loss": loss, **pieces}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_count
        return {name: value / max(count, 1) for name, value in totals.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = run_epoch(train_loader, True)
        validation_metrics = run_epoch(validation_loader, False)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "curvature": float(model.curvature.detach().cpu()),
            "metric_weights": model.metric_weights.detach().cpu().tolist(),
        }
        history.append(row)
        current = validation_metrics["loss"]
        if current < best_validation - 1.0e-7:
            best_validation = current
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema": MIXED_CHECKPOINT_SCHEMA,
                    "training_schema": TRAINING_SCHEMA,
                    "training_supervision": "semantic_optimal_transport",
                    "is_ground_truth_pair": False,
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "normalization": normalization,
                    "paired_dataset": str(train_data_path.resolve()),
                    "validation_dataset": str(validation_data_path.resolve()),
                    "event_db_contract_json": train_event_contract,
                    "validation_event_db_contract_json": val_event_contract,
                    "training_indices": train_indices,
                    "validation_indices": np.arange(
                        validation_dimensions["rows"], dtype=np.int64
                    ),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "validation_loss": float(current),
                    "loss_weights": loss_weights,
                    "split_contract": {
                        "music_song_disjoint": True,
                        "motion_source_disjoint": True,
                        "ot_built_after_split": True,
                    },
                },
                out_path,
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            print("[SEMANTIC OT GROUNDER] " + json.dumps(row), flush=True)
        if stale >= int(patience):
            break

    checkpoint = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    validation_audit = _prediction_audit(model, validation_loader, device)
    report = {
        "schema": TRAINING_SCHEMA,
        "model_checkpoint_schema": MIXED_CHECKPOINT_SCHEMA,
        "training_supervision": "semantic_optimal_transport",
        "is_ground_truth_pair": False,
        "train_dataset": str(train_data_path.resolve()),
        "validation_dataset": str(validation_data_path.resolve()),
        "checkpoint": str(out_path.resolve()),
        "device": str(device),
        "dimensions": dimensions,
        "config": asdict(config),
        "normalization": "training-motion-sources-only",
        "split_contract": {
            "music_song_disjoint": True,
            "motion_source_disjoint": True,
            "ot_built_after_split": True,
            "train_validation_contracts_differ": True,
        },
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "validation_teacher_audit": validation_audit,
        "history": history,
        "metric_interpretation": {
            "teacher_agreement_is_not_human_ground_truth_retrieval": True,
            "report_r_at_k_as_ground_truth": False,
        },
        "ok": True,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train mixed-curvature grounding from semantic optimal transport"
    )
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--validation_data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_phrases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args(argv)
    report = train_semantic_ot_grounder(
        Path(args.train_data),
        Path(args.validation_data),
        Path(args.out),
        epochs=int(args.epochs),
        batch_phrases=int(args.batch_phrases),
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
