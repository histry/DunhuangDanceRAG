#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real-audio mixed-curvature Gaussian-Wasserstein grounder.

The production V46.53 grounder remains available for historical checkpoints.
This module implements the research architecture as an opt-in, schema-versioned
path with strict paired-data and train-only-normalization contracts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - data validation remains available
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

from grounding.manifold_ops import (
    EPS,
    gaussian_wasserstein_distance_sq_torch,
    lorentz_distance_sq_torch,
    lorentz_project_torch,
    mixed_product_distance_sq_torch,
    sphere_project_torch,
)
from grounding.paired_data import _resample_sequence, validate_paired_payload
from support.event_identity import event_uids_from_generation_db


SCHEMA = "v46_53_mixed_curvature_gaussian_grounder_v1"
EMBED_SCHEMA = "v46_53_mixed_curvature_event_factors_v1"


def _env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_torch_checkpoint(path: Path) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load mixed-grounder checkpoints")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA:
        raise RuntimeError(f"Not a {SCHEMA} checkpoint: {path}")
    return checkpoint


def load_paired_dataset(path: Path) -> Tuple[Dict[str, Any], Dict[str, int]]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    schema = str(np.asarray(payload.get("schema", "")).reshape(-1)[0])
    if schema != "v46_53_real_audio_motion_paired_grounding_v1":
        raise RuntimeError(f"Unsupported paired-grounding schema {schema!r}: {path}")
    dimensions = validate_paired_payload(payload)
    return payload, dimensions


def source_disjoint_split(
    source_ids: np.ndarray,
    validation_ratio: float,
    seed: int,
    *,
    pair_ids: Optional[np.ndarray] = None,
    audio_group_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split rows without crossing any declared identity group.

    Rows form an identity graph: sharing a motion source, a positive-pair
    identity, or an exact audio-segment identity joins them into one connected
    component.  Components, rather than individual sources, are assigned to
    train/validation so an observed audio query cannot leak across the split.

    ``pair_ids=None`` preserves source-only behavior for legacy callers.
    Research training always supplies pair and audio identities.
    """

    groups = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    if groups.size < 2:
        raise RuntimeError(
            "Mixed-grounder validation requires at least two rows"
        )
    if (
        pair_ids is None
        and audio_group_ids is None
        and len(np.unique(groups)) < 2
    ):
        raise RuntimeError(
            "Mixed-grounder validation requires at least two distinct motion sources"
        )
    identities: list[tuple[str, np.ndarray]] = [("source", groups)]
    for name, raw in (
        ("pair", pair_ids),
        ("audio", audio_group_ids),
    ):
        if raw is None:
            continue
        values = np.asarray(raw, dtype=np.int64).reshape(-1)
        if values.shape != groups.shape:
            raise ValueError(
                f"{name}_ids shape {values.shape} does not match "
                f"source_ids {groups.shape}"
            )
        identities.append((name, values))

    parent = np.arange(len(groups), dtype=np.int64)
    rank = np.zeros(len(groups), dtype=np.int8)

    def find(index: int) -> int:
        cursor = int(index)
        while int(parent[cursor]) != cursor:
            parent[cursor] = parent[int(parent[cursor])]
            cursor = int(parent[cursor])
        return cursor

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if int(rank[root_left]) < int(rank[root_right]):
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if int(rank[root_left]) == int(rank[root_right]):
            rank[root_left] += 1

    first_seen: dict[tuple[str, int], int] = {}
    for identity_name, values in identities:
        for row_index, raw_value in enumerate(values):
            token = (identity_name, int(raw_value))
            if token in first_seen:
                union(row_index, first_seen[token])
            else:
                first_seen[token] = row_index

    component_rows: dict[int, list[int]] = {}
    for row_index in range(len(groups)):
        component_rows.setdefault(find(row_index), []).append(row_index)
    components = [
        np.asarray(rows, dtype=np.int64)
        for rows in component_rows.values()
    ]
    if len(components) < 2:
        raise RuntimeError(
            "Source/pair/audio identity graph has only one connected component; "
            "a leakage-free validation split is impossible"
        )
    ratio = float(validation_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError("validation_ratio must lie strictly between zero and one")
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(components), dtype=np.int64)
    rng.shuffle(order)
    target_rows = int(
        np.clip(round(len(groups) * ratio), 1, len(groups) - 1)
    )
    randomized_order = order.tolist()
    first_component = min(
        randomized_order,
        key=lambda index: abs(len(components[index]) - target_rows),
    )
    selected: list[int] = [int(first_component)]
    selected_rows = int(len(components[first_component]))
    for component_index in randomized_order:
        if int(component_index) == int(first_component):
            continue
        size = int(len(components[component_index]))
        if selected_rows + size >= len(groups):
            continue
        if abs(selected_rows + size - target_rows) < abs(
            selected_rows - target_rows
        ):
            selected.append(int(component_index))
            selected_rows += size
    validation = np.sort(
        np.concatenate([components[index] for index in selected])
    ).astype(np.int64)
    validation_set = set(map(int, validation.tolist()))
    training = np.asarray(
        [
            row_index
            for row_index in range(len(groups))
            if row_index not in validation_set
        ],
        dtype=np.int64,
    )
    if len(training) == 0 or len(validation) == 0:
        raise RuntimeError(
            "Source/pair/audio-disjoint split produced an empty partition"
        )
    for identity_name, values in identities:
        overlap = set(map(int, values[training])).intersection(
            set(map(int, values[validation]))
        )
        if overlap:
            raise AssertionError(
                f"{identity_name} identities crossed the split: "
                f"{sorted(overlap)[:16]}"
            )
    return training, validation


def _mean_std(value: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(value, dtype=np.float64)[indices]
    mean = selected.mean(axis=0, keepdims=True)
    std = selected.std(axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_train_normalization(
    payload: Mapping[str, Any], training_indices: np.ndarray
) -> Dict[str, np.ndarray]:
    indices = np.asarray(training_indices, dtype=np.int64)
    geometry_mean, geometry_std = _mean_std(payload["motion_geometry"], indices)
    bodypart_mean, bodypart_std = _mean_std(payload["bodypart_flow"], indices)
    # Temporal normalization is feature-wise, not frame-position-specific.
    temporal_selected = np.asarray(payload["temporal"], dtype=np.float64)[indices]
    temporal_mean = temporal_selected.mean(axis=(0, 1), keepdims=True)
    temporal_std = np.maximum(
        temporal_selected.std(axis=(0, 1), keepdims=True), 1.0e-5
    )
    gaussian = np.asarray(payload["gaussian_mean"], dtype=np.float64)[indices]
    gaussian_center = gaussian.mean(axis=(0, 1), keepdims=True)
    gaussian_scale = np.maximum(
        gaussian.std(axis=(0, 1), keepdims=True), 1.0e-4
    )
    return {
        "geometry_mean": geometry_mean,
        "geometry_std": geometry_std,
        "bodypart_mean": bodypart_mean,
        "bodypart_std": bodypart_std,
        "temporal_mean": temporal_mean.astype(np.float32),
        "temporal_std": temporal_std.astype(np.float32),
        "gaussian_center": gaussian_center.astype(np.float32),
        "gaussian_scale": gaussian_scale.astype(np.float32),
    }


def apply_normalization(
    payload: Mapping[str, Any], normalization: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    geometry = (
        np.asarray(payload["motion_geometry"], dtype=np.float32)
        - np.asarray(normalization["geometry_mean"], dtype=np.float32)
    ) / np.asarray(normalization["geometry_std"], dtype=np.float32)
    bodypart = (
        np.asarray(payload["bodypart_flow"], dtype=np.float32)
        - np.asarray(normalization["bodypart_mean"], dtype=np.float32)
    ) / np.asarray(normalization["bodypart_std"], dtype=np.float32)
    temporal = (
        np.asarray(payload["temporal"], dtype=np.float32)
        - np.asarray(normalization["temporal_mean"], dtype=np.float32)
    ) / np.asarray(normalization["temporal_std"], dtype=np.float32)
    gaussian_scale = np.asarray(
        normalization["gaussian_scale"], dtype=np.float32
    )
    gaussian_mean = (
        np.asarray(payload["gaussian_mean"], dtype=np.float32)
        - np.asarray(normalization["gaussian_center"], dtype=np.float32)
    ) / gaussian_scale
    scale_vector = gaussian_scale.reshape(-1)
    covariance_scale = scale_vector[:, None] * scale_vector[None, :]
    gaussian_covariance = np.asarray(
        payload["gaussian_covariance"], dtype=np.float32
    ) / covariance_scale[None, None]
    result = {
        "clap": np.asarray(payload["clap"], dtype=np.float32),
        "temporal": temporal.astype(np.float32),
        "motion_geometry": geometry.astype(np.float32),
        "bodypart_flow": bodypart.astype(np.float32),
        "gaussian_mean": gaussian_mean.astype(np.float32),
        "gaussian_covariance": gaussian_covariance.astype(np.float32),
        "controls": np.asarray(payload["controls"], dtype=np.float32),
        "quality": np.asarray(payload["quality"], dtype=np.float32),
        "pair_ids": np.asarray(payload["pair_ids"], dtype=np.int64),
        "family_ids": np.asarray(payload["family_ids"], dtype=np.int64),
        "source_ids": np.asarray(payload["source_ids"], dtype=np.int64),
        "event_indices": np.asarray(payload["event_indices"], dtype=np.int64),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Normalized mixed-grounder dataset contains NaN or Inf")
    return result


@dataclass(frozen=True)
class MixedGrounderConfig:
    clap_dim: int
    temporal_dim: int
    motion_geometry_dim: int
    bodypart_count: int
    bodypart_feature_dim: int
    gaussian_dim: int
    control_dim: int
    num_sources: int
    hidden_dim: int = 192
    lorentz_dim: int = 16
    sphere_dim: int = 96
    dropout: float = 0.10
    minimum_covariance: float = 1.0e-4
    initial_curvature: float = 1.0
    initial_temperature: float = 0.08

    def __post_init__(self) -> None:
        integer_fields = (
            "clap_dim",
            "temporal_dim",
            "motion_geometry_dim",
            "bodypart_count",
            "bodypart_feature_dim",
            "gaussian_dim",
            "control_dim",
            "hidden_dim",
            "lorentz_dim",
            "sphere_dim",
        )
        invalid = [name for name in integer_fields if int(getattr(self, name)) <= 0]
        if invalid:
            raise ValueError(f"Mixed-grounder dimensions must be positive: {invalid}")
        if self.hidden_dim < 16 or self.hidden_dim % 16 != 0:
            raise ValueError(
                "hidden_dim must be at least 16 and divisible by 16 so "
                "temporal GroupNorm has a valid channel contract"
            )
        if self.num_sources < 1:
            raise ValueError("num_sources must be positive")
        if self.minimum_covariance <= 0.0:
            raise ValueError("minimum_covariance must be positive")
        if self.initial_curvature <= 0.0 or self.initial_temperature <= 0.0:
            raise ValueError("initial curvature and temperature must be positive")


if nn is not None:

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.input = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )
            self.residual = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)

        def forward(self, value: "torch.Tensor") -> "torch.Tensor":
            hidden = self.input(value)
            return self.output_norm(hidden + self.residual(hidden))


    class AudioTower(nn.Module):
        def __init__(self, config: MixedGrounderConfig):
            super().__init__()
            hidden = config.hidden_dim
            self.clap = ResidualMLP(config.clap_dim, hidden, config.dropout)
            self.temporal = nn.Sequential(
                nn.Conv1d(config.temporal_dim, hidden // 2, 5, padding=2),
                nn.GroupNorm(8, hidden // 2),
                nn.GELU(),
                nn.Conv1d(hidden // 2, hidden // 2, 5, padding=2),
                nn.GroupNorm(8, hidden // 2),
                nn.GELU(),
            )
            self.fusion = ResidualMLP(hidden * 2, hidden, config.dropout)

        def forward(
            self, clap: "torch.Tensor", temporal: "torch.Tensor"
        ) -> "torch.Tensor":
            clap_hidden = self.clap(clap)
            temporal_hidden = self.temporal(temporal.transpose(1, 2))
            pooled = torch.cat(
                [
                    temporal_hidden.mean(dim=-1),
                    temporal_hidden.amax(dim=-1),
                ],
                dim=-1,
            )
            return self.fusion(torch.cat([clap_hidden, pooled], dim=-1))


    class MotionTower(nn.Module):
        def __init__(self, config: MixedGrounderConfig):
            super().__init__()
            hidden = config.hidden_dim
            self.geometry = ResidualMLP(
                config.motion_geometry_dim, hidden, config.dropout
            )
            self.part = ResidualMLP(
                config.bodypart_feature_dim, hidden, config.dropout
            )
            self.part_attention = nn.Linear(hidden, 1)
            self.fusion = ResidualMLP(hidden * 2, hidden, config.dropout)

        def forward(
            self, geometry: "torch.Tensor", bodypart: "torch.Tensor"
        ) -> "torch.Tensor":
            geometry_hidden = self.geometry(geometry)
            part_hidden = self.part(bodypart)
            attention = torch.softmax(self.part_attention(part_hidden), dim=1)
            part_summary = (part_hidden * attention).sum(dim=1)
            return self.fusion(
                torch.cat([geometry_hidden, part_summary], dim=-1)
            )


    class FactorHeads(nn.Module):
        def __init__(self, config: MixedGrounderConfig):
            super().__init__()
            hidden = config.hidden_dim
            self.bodypart_count = config.bodypart_count
            self.gaussian_dim = config.gaussian_dim
            triangular = config.gaussian_dim * (config.gaussian_dim + 1) // 2
            self.lorentz_spatial = nn.Linear(hidden, config.lorentz_dim)
            self.sphere = nn.Linear(hidden, config.sphere_dim)
            self.gaussian_mean = nn.Linear(
                hidden, config.bodypart_count * config.gaussian_dim
            )
            self.gaussian_cholesky = nn.Linear(
                hidden, config.bodypart_count * triangular
            )
            self.euclidean = nn.Linear(hidden, config.control_dim)
            self.uncertainty = nn.Linear(hidden, 1)
            self.minimum_covariance = float(config.minimum_covariance)
            self.register_buffer(
                "tril_rows",
                torch.tril_indices(
                    config.gaussian_dim, config.gaussian_dim
                )[0],
                persistent=False,
            )
            self.register_buffer(
                "tril_columns",
                torch.tril_indices(
                    config.gaussian_dim, config.gaussian_dim
                )[1],
                persistent=False,
            )

        def _covariance(self, hidden: "torch.Tensor") -> "torch.Tensor":
            batch = hidden.shape[0]
            raw = self.gaussian_cholesky(hidden).reshape(
                batch, self.bodypart_count, -1
            )
            lower = hidden.new_zeros(
                (
                    batch,
                    self.bodypart_count,
                    self.gaussian_dim,
                    self.gaussian_dim,
                )
            )
            lower[
                ...,
                self.tril_rows,
                self.tril_columns,
            ] = raw
            diagonal = torch.diagonal(lower, dim1=-2, dim2=-1)
            positive_diagonal = F.softplus(diagonal) + math.sqrt(
                self.minimum_covariance
            )
            lower = lower - torch.diag_embed(diagonal) + torch.diag_embed(
                positive_diagonal
            )
            covariance = lower @ lower.transpose(-1, -2)
            identity = torch.eye(
                self.gaussian_dim, dtype=hidden.dtype, device=hidden.device
            )
            return covariance + self.minimum_covariance * identity

        def forward(
            self, hidden: "torch.Tensor", curvature: "torch.Tensor"
        ) -> Dict[str, "torch.Tensor"]:
            gaussian_mean = self.gaussian_mean(hidden).reshape(
                hidden.shape[0], self.bodypart_count, self.gaussian_dim
            )
            return {
                "lorentz": lorentz_project_torch(
                    self.lorentz_spatial(hidden), curvature
                ),
                "sphere": sphere_project_torch(self.sphere(hidden)),
                "gaussian_mean": gaussian_mean,
                "gaussian_covariance": self._covariance(hidden),
                "euclidean": self.euclidean(hidden),
                "uncertainty": F.softplus(self.uncertainty(hidden).squeeze(-1))
                + 1.0e-3,
                "latent": hidden,
            }


    class _GradientReversal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value: "torch.Tensor", scale: float):
            ctx.scale = float(scale)
            return value.view_as(value)

        @staticmethod
        def backward(ctx, gradient: "torch.Tensor"):
            return -ctx.scale * gradient, None


    class MixedCurvatureGrounder(nn.Module):
        def __init__(self, config: MixedGrounderConfig):
            super().__init__()
            self.config = config
            self.audio_tower = AudioTower(config)
            self.motion_tower = MotionTower(config)
            self.audio_heads = FactorHeads(config)
            self.motion_heads = FactorHeads(config)
            initial_raw_curvature = math.log(
                max(math.exp(config.initial_curvature) - 1.0, 1.0e-6)
            )
            self.raw_curvature = nn.Parameter(
                torch.tensor(initial_raw_curvature, dtype=torch.float32)
            )
            self.raw_metric_weights = nn.Parameter(torch.zeros(4))
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / config.initial_temperature))
            )
            self.source_classifier = (
                nn.Linear(config.hidden_dim, config.num_sources)
                if config.num_sources > 1
                else None
            )

        @property
        def curvature(self) -> "torch.Tensor":
            return (F.softplus(self.raw_curvature) + 1.0e-3).clamp(max=10.0)

        @property
        def metric_weights(self) -> "torch.Tensor":
            # Global weights preserve a valid fixed product metric.
            return torch.softmax(self.raw_metric_weights, dim=-1)

        def encode_audio(
            self, clap: "torch.Tensor", temporal: "torch.Tensor"
        ) -> Dict[str, "torch.Tensor"]:
            hidden = self.audio_tower(clap, temporal)
            return self.audio_heads(hidden, self.curvature)

        def encode_motion(
            self, geometry: "torch.Tensor", bodypart: "torch.Tensor"
        ) -> Dict[str, "torch.Tensor"]:
            hidden = self.motion_tower(geometry, bodypart)
            return self.motion_heads(hidden, self.curvature)

        def pairwise_distance_sq(
            self,
            audio_factors: Mapping[str, "torch.Tensor"],
            motion_factors: Mapping[str, "torch.Tensor"],
        ) -> "torch.Tensor":
            factor_keys = (
                "lorentz",
                "sphere",
                "gaussian_mean",
                "gaussian_covariance",
                "euclidean",
            )
            left = {key: audio_factors[key][:, None] for key in factor_keys}
            right = {key: motion_factors[key][None, :] for key in factor_keys}
            return mixed_product_distance_sq_torch(
                left,
                right,
                self.metric_weights,
                curvature=self.curvature,
                minimum_eigenvalue=self.config.minimum_covariance,
            )

        def pairwise_logits(
            self,
            audio_factors: Mapping[str, "torch.Tensor"],
            motion_factors: Mapping[str, "torch.Tensor"],
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            distance = self.pairwise_distance_sq(audio_factors, motion_factors)
            variance = (
                audio_factors["uncertainty"][:, None].square()
                + motion_factors["uncertainty"][None, :].square()
                + 1.0e-5
            )
            calibrated_energy = distance / (2.0 * variance) + 0.5 * torch.log(
                variance
            )
            logits = -self.logit_scale.exp().clamp(max=100.0) * calibrated_energy
            return logits, distance, variance

        def source_logits(
            self, motion_factors: Mapping[str, "torch.Tensor"], scale: float
        ) -> Optional["torch.Tensor"]:
            if self.source_classifier is None:
                return None
            reversed_hidden = _GradientReversal.apply(
                motion_factors["latent"], float(scale)
            )
            return self.source_classifier(reversed_hidden)

else:  # pragma: no cover
    MixedCurvatureGrounder = object


def _multi_positive_bidirectional_loss(
    logits: "torch.Tensor",
    pair_ids: "torch.Tensor",
    quality: "torch.Tensor",
) -> "torch.Tensor":
    positive = pair_ids[:, None] == pair_ids[None, :]
    negative_infinity = torch.finfo(logits.dtype).min

    def directional(
        values: "torch.Tensor",
        mask: "torch.Tensor",
        weights: "torch.Tensor",
    ) -> "torch.Tensor":
        positive_lse = torch.logsumexp(
            values.masked_fill(~mask, negative_infinity), dim=1
        )
        all_lse = torch.logsumexp(values, dim=1)
        loss = all_lse - positive_lse
        normalized = weights.clamp(0.05, 1.0)
        return (loss * normalized).sum() / normalized.sum().clamp_min(EPS)

    return 0.5 * (
        directional(logits, positive, quality)
        + directional(logits.transpose(0, 1), positive.transpose(0, 1), quality)
    )


def _hierarchy_loss(
    factors: Mapping[str, "torch.Tensor"],
    family_ids: "torch.Tensor",
    curvature: "torch.Tensor",
    margin: float,
) -> "torch.Tensor":
    points = factors["lorentz"]
    distance = lorentz_distance_sq_torch(
        points[:, None], points[None, :], curvature
    ).sqrt()
    off_diagonal = ~torch.eye(
        len(points), dtype=torch.bool, device=points.device
    )
    same = (family_ids[:, None] == family_ids[None, :]) & off_diagonal
    different = (family_ids[:, None] != family_ids[None, :]) & off_diagonal
    positive = distance[same].mean() if same.any() else distance.sum() * 0.0
    negative = (
        F.relu(float(margin) - distance[different]).mean()
        if different.any()
        else distance.sum() * 0.0
    )
    return positive + 0.35 * negative


def _pair_calibration_loss(
    distance: "torch.Tensor",
    variance: "torch.Tensor",
    pair_ids: "torch.Tensor",
    quality: "torch.Tensor",
) -> "torch.Tensor":
    positive = pair_ids[:, None] == pair_ids[None, :]
    nll = distance / (2.0 * variance) + 0.5 * torch.log(variance)
    pair_quality = torch.sqrt(
        quality[:, None].clamp_min(1.0e-3)
        * quality[None, :].clamp_min(1.0e-3)
    )
    return (nll[positive] * pair_quality[positive]).sum() / pair_quality[
        positive
    ].sum().clamp_min(EPS)


def _batch_loss(
    model: "MixedCurvatureGrounder",
    batch: Sequence["torch.Tensor"],
    *,
    hierarchy_weight: float,
    gaussian_anchor_weight: float,
    control_weight: float,
    uncertainty_weight: float,
    source_weight: float,
    metric_balance_weight: float,
    hierarchy_margin: float,
) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
    (
        clap,
        temporal,
        geometry,
        bodypart,
        gaussian_mean,
        gaussian_covariance,
        controls,
        quality,
        pair_ids,
        family_ids,
        source_ids,
        _event_indices,
    ) = batch
    audio = model.encode_audio(clap, temporal)
    motion = model.encode_motion(geometry, bodypart)
    logits, distance, variance = model.pairwise_logits(audio, motion)
    contrastive = _multi_positive_bidirectional_loss(logits, pair_ids, quality)
    hierarchy = _hierarchy_loss(
        motion, family_ids, model.curvature, hierarchy_margin
    )
    gaussian_anchor = gaussian_wasserstein_distance_sq_torch(
        motion["gaussian_mean"],
        motion["gaussian_covariance"],
        gaussian_mean,
        gaussian_covariance,
        model.config.minimum_covariance,
    ).mean()
    control = 0.5 * (
        F.smooth_l1_loss(audio["euclidean"], controls)
        + F.smooth_l1_loss(motion["euclidean"], controls)
    )
    uncertainty = _pair_calibration_loss(
        distance, variance, pair_ids, quality
    )
    source_logits = model.source_logits(motion, scale=1.0)
    source = (
        F.cross_entropy(source_logits, source_ids)
        if source_logits is not None
        else contrastive.new_zeros(())
    )
    # A small KL-to-uniform penalty prevents a formally present factor from
    # collapsing to a numerically zero metric weight without evidence.
    metric_balance = -torch.log(
        model.metric_weights * len(model.metric_weights)
    ).mean()
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
        "contrastive": contrastive,
        "hierarchy": hierarchy,
        "gaussian_anchor": gaussian_anchor,
        "control": control,
        "uncertainty": uncertainty,
        "source_adversarial": source,
        "metric_balance": metric_balance,
    }


def _tensor_dataset(normalized: Mapping[str, np.ndarray]) -> "TensorDataset":
    if torch is None:
        raise RuntimeError("PyTorch is required to train the mixed grounder")
    order = (
        "clap",
        "temporal",
        "motion_geometry",
        "bodypart_flow",
        "gaussian_mean",
        "gaussian_covariance",
        "controls",
        "quality",
        "pair_ids",
        "family_ids",
        "source_ids",
        "event_indices",
    )
    tensors = []
    for key in order:
        value = np.asarray(normalized[key])
        tensor = torch.from_numpy(value)
        tensors.append(tensor)
    return TensorDataset(*tensors)


def retrieval_metrics(
    scores: np.ndarray,
    query_pair_ids: np.ndarray,
    candidate_pair_ids: np.ndarray,
) -> Dict[str, float]:
    """Compute multi-positive retrieval metrics without one-to-one assumptions."""

    values = np.asarray(scores, dtype=np.float64)
    query_ids = np.asarray(query_pair_ids, dtype=np.int64).reshape(-1)
    candidate_ids = np.asarray(candidate_pair_ids, dtype=np.int64).reshape(-1)
    if values.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError(
            f"score matrix shape {values.shape} is incompatible with "
            f"{len(query_ids)} queries and {len(candidate_ids)} candidates"
        )
    reciprocal_ranks: list[float] = []
    average_precisions: list[float] = []
    recalls = {1: [], 5: [], 10: []}
    for row, pair_id in zip(values, query_ids):
        relevant = candidate_ids == pair_id
        if not relevant.any():
            continue
        order = np.argsort(row, kind="stable")[::-1]
        ordered_relevant = relevant[order]
        positive_ranks = np.flatnonzero(ordered_relevant) + 1
        reciprocal_ranks.append(1.0 / float(positive_ranks[0]))
        precision_at_positive = np.arange(
            1, len(positive_ranks) + 1, dtype=np.float64
        ) / positive_ranks
        average_precisions.append(float(precision_at_positive.mean()))
        for cutoff in recalls:
            recalls[cutoff].append(
                float(ordered_relevant[: min(cutoff, len(order))].any())
            )
    if not reciprocal_ranks:
        raise RuntimeError("Retrieval evaluation has no query with a positive candidate")
    return {
        "R@1": float(np.mean(recalls[1])),
        "R@5": float(np.mean(recalls[5])),
        "R@10": float(np.mean(recalls[10])),
        "MRR": float(np.mean(reciprocal_ranks)),
        "mAP": float(np.mean(average_precisions)),
        "queries": int(len(reciprocal_ranks)),
    }


def _full_retrieval_evaluation(
    model: "MixedCurvatureGrounder",
    normalized: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: "torch.device",
    batch_size: int,
) -> Dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    factor_names = (
        "lorentz",
        "sphere",
        "gaussian_mean",
        "gaussian_covariance",
        "euclidean",
        "uncertainty",
    )
    audio_chunks: Dict[str, list["torch.Tensor"]] = {
        key: [] for key in factor_names
    }
    motion_chunks: Dict[str, list["torch.Tensor"]] = {
        key: [] for key in factor_names
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(selected), int(batch_size)):
            batch_indices = selected[start : start + int(batch_size)]
            audio = model.encode_audio(
                torch.from_numpy(normalized["clap"][batch_indices]).to(device),
                torch.from_numpy(normalized["temporal"][batch_indices]).to(device),
            )
            motion = model.encode_motion(
                torch.from_numpy(
                    normalized["motion_geometry"][batch_indices]
                ).to(device),
                torch.from_numpy(
                    normalized["bodypart_flow"][batch_indices]
                ).to(device),
            )
            for key in factor_names:
                audio_chunks[key].append(audio[key])
                motion_chunks[key].append(motion[key])
        audio_all = {
            key: torch.cat(value, dim=0) for key, value in audio_chunks.items()
        }
        motion_all = {
            key: torch.cat(value, dim=0) for key, value in motion_chunks.items()
        }

        audio_to_motion_rows: list[np.ndarray] = []
        motion_to_audio_rows: list[np.ndarray] = []
        for start in range(0, len(selected), int(batch_size)):
            end = min(len(selected), start + int(batch_size))
            audio_block = {
                key: value[start:end] for key, value in audio_all.items()
            }
            logits, _, _ = model.pairwise_logits(audio_block, motion_all)
            audio_to_motion_rows.append(logits.cpu().numpy())

            motion_block = {
                key: value[start:end] for key, value in motion_all.items()
            }
            reverse_logits, _, _ = model.pairwise_logits(
                audio_all, motion_block
            )
            motion_to_audio_rows.append(reverse_logits.transpose(0, 1).cpu().numpy())
    audio_to_motion = np.concatenate(audio_to_motion_rows, axis=0)
    motion_to_audio = np.concatenate(motion_to_audio_rows, axis=0)
    pair_ids = np.asarray(normalized["pair_ids"], dtype=np.int64)[selected]
    return {
        "audio_to_motion": retrieval_metrics(
            audio_to_motion, pair_ids, pair_ids
        ),
        "motion_to_audio": retrieval_metrics(
            motion_to_audio, pair_ids, pair_ids
        ),
        "rows": int(len(selected)),
    }


def train_mixed_grounder(
    paired_dataset_path: Path,
    out_path: Path,
    *,
    epochs: int = 120,
    batch_size: int = 96,
    seed: int = 20260724,
    validation_ratio: float = 0.20,
    learning_rate: float = 2.0e-4,
    weight_decay: float = 1.0e-4,
    patience: int = 20,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to train mixed-curvature grounding")
    payload, dimensions = load_paired_dataset(paired_dataset_path)
    training_indices, validation_indices = source_disjoint_split(
        payload["source_ids"],
        validation_ratio,
        seed,
        pair_ids=payload["pair_ids"],
        audio_group_ids=payload.get("audio_group_ids", payload["pair_ids"]),
    )
    normalization = fit_train_normalization(payload, training_indices)
    normalized = apply_normalization(payload, normalization)
    config = MixedGrounderConfig(
        clap_dim=dimensions["clap_dim"],
        temporal_dim=dimensions["temporal_dim"],
        motion_geometry_dim=dimensions["motion_geometry_dim"],
        bodypart_count=dimensions["bodypart_count"],
        bodypart_feature_dim=int(np.asarray(payload["bodypart_flow"]).shape[-1]),
        gaussian_dim=dimensions["gaussian_dim"],
        control_dim=dimensions["control_dim"],
        num_sources=int(np.max(payload["source_ids"])) + 1,
        hidden_dim=int(os.environ.get("V46_53_MIXED_HIDDEN", 192)),
        lorentz_dim=int(os.environ.get("V46_53_MIXED_LORENTZ_DIM", 16)),
        sphere_dim=int(os.environ.get("V46_53_MIXED_SPHERE_DIM", 96)),
        dropout=float(os.environ.get("V46_53_MIXED_DROPOUT", 0.10)),
        minimum_covariance=float(
            os.environ.get("V46_53_MIXED_COV_EPS", 1.0e-4)
        ),
        initial_curvature=float(
            os.environ.get("V46_53_MIXED_CURVATURE", 1.0)
        ),
        initial_temperature=float(
            os.environ.get("V46_53_MIXED_TEMPERATURE", 0.08)
        ),
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
    dataset = _tensor_dataset(normalized)

    def loader(indices: np.ndarray, shuffle: bool) -> "DataLoader":
        subset = torch.utils.data.Subset(dataset, indices.tolist())
        return DataLoader(
            subset,
            batch_size=min(int(batch_size), len(subset)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    train_loader = loader(training_indices, True)
    validation_loader = loader(validation_indices, False)
    loss_weights = {
        "hierarchy_weight": float(
            os.environ.get("V46_53_MIXED_HIERARCHY_W", 0.20)
        ),
        "gaussian_anchor_weight": float(
            os.environ.get("V46_53_MIXED_GAUSSIAN_W", 0.25)
        ),
        "control_weight": float(
            os.environ.get("V46_53_MIXED_CONTROL_W", 0.10)
        ),
        "uncertainty_weight": float(
            os.environ.get("V46_53_MIXED_UNCERTAINTY_W", 0.05)
        ),
        "source_weight": float(
            os.environ.get("V46_53_MIXED_SOURCE_W", 0.05)
        ),
        "metric_balance_weight": float(
            os.environ.get("V46_53_MIXED_METRIC_BALANCE_W", 0.01)
        ),
        "hierarchy_margin": float(
            os.environ.get("V46_53_MIXED_HIERARCHY_MARGIN", 1.25)
        ),
    }

    def run_epoch(data_loader: "DataLoader", training: bool) -> Dict[str, float]:
        model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        for raw_batch in data_loader:
            batch = tuple(value.to(device, non_blocking=True) for value in raw_batch)
            with torch.set_grad_enabled(training):
                loss, pieces = _batch_loss(model, batch, **loss_weights)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Mixed-grounder loss became non-finite; "
                        "inspect covariance conditioning and input normalization"
                    )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
            current_batch = len(batch[0])
            count += current_batch
            values = {"loss": loss, **pieces}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().cpu()
                ) * current_batch
        return {name: value / max(count, 1) for name, value in totals.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    best_epoch = 0
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
                    "schema": SCHEMA,
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "normalization": normalization,
                    "paired_dataset": str(paired_dataset_path.resolve()),
                    "event_db_contract_json": str(
                        np.asarray(payload["event_db_contract_json"]).item()
                    ),
                    "training_indices": training_indices,
                    "validation_indices": validation_indices,
                    "seed": int(seed),
                    "epoch": epoch,
                    "validation_loss": current,
                    "loss_weights": loss_weights,
                },
                out_path,
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                "[V46.53 MIXED] "
                + json.dumps(row, ensure_ascii=False),
                flush=True,
            )
        if stale >= int(patience):
            break

    best_checkpoint = _load_torch_checkpoint(out_path)
    best_model = _model_from_checkpoint(best_checkpoint, device)
    retrieval = _full_retrieval_evaluation(
        best_model,
        normalized,
        validation_indices,
        device=device,
        batch_size=max(16, int(batch_size)),
    )

    report = {
        "schema": SCHEMA,
        "dataset": str(paired_dataset_path.resolve()),
        "checkpoint": str(out_path.resolve()),
        "device": str(device),
        "dimensions": dimensions,
        "config": asdict(config),
        "normalization": "training-sources-only",
        "split": {
            "train_rows": int(len(training_indices)),
            "validation_rows": int(len(validation_indices)),
            "train_sources": sorted(
                set(map(int, np.asarray(payload["source_ids"])[training_indices]))
            ),
            "validation_sources": sorted(
                set(map(int, np.asarray(payload["source_ids"])[validation_indices]))
            ),
            "train_pair_ids": sorted(
                set(map(int, np.asarray(payload["pair_ids"])[training_indices]))
            ),
            "validation_pair_ids": sorted(
                set(map(int, np.asarray(payload["pair_ids"])[validation_indices]))
            ),
            "train_audio_groups": sorted(
                set(
                    map(
                        int,
                        np.asarray(
                            payload.get("audio_group_ids", payload["pair_ids"])
                        )[training_indices],
                    )
                )
            ),
            "validation_audio_groups": sorted(
                set(
                    map(
                        int,
                        np.asarray(
                            payload.get("audio_group_ids", payload["pair_ids"])
                        )[validation_indices],
                    )
                )
            ),
            "identity_components_disjoint": True,
        },
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation),
        "validation_retrieval": retrieval,
        "history": history,
        "ok": True,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _model_from_checkpoint(
    checkpoint: Mapping[str, Any], device: "torch.device"
) -> "MixedCurvatureGrounder":
    config = MixedGrounderConfig(**dict(checkpoint["config"]))
    model = MixedCurvatureGrounder(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model


def _motion_inputs_for_checkpoint(
    db: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    config = MixedGrounderConfig(**dict(checkpoint["config"]))
    for key in ("v46_53_geometry_desc", "v46_53_bodypart_flow"):
        if key not in db:
            raise RuntimeError(f"Event-DB lacks mixed-grounder input {key}")
    geometry = np.asarray(db["v46_53_geometry_desc"], dtype=np.float32)
    bodypart = np.asarray(db["v46_53_bodypart_flow"], dtype=np.float32)[
        :, : config.bodypart_count
    ]
    if geometry.shape[1] != config.motion_geometry_dim:
        raise RuntimeError(
            "Mixed-grounder geometry dimension mismatch: "
            f"db={geometry.shape}, checkpoint={config.motion_geometry_dim}"
        )
    if bodypart.shape[1:] != (
        config.bodypart_count,
        config.bodypart_feature_dim,
    ):
        raise RuntimeError(
            "Mixed-grounder body-part dimension mismatch: "
            f"db={bodypart.shape}, checkpoint="
            f"{(config.bodypart_count, config.bodypart_feature_dim)}"
        )
    normalization = checkpoint["normalization"]
    geometry = (
        geometry - np.asarray(normalization["geometry_mean"], dtype=np.float32)
    ) / np.asarray(normalization["geometry_std"], dtype=np.float32)
    bodypart = (
        bodypart - np.asarray(normalization["bodypart_mean"], dtype=np.float32)
    ) / np.asarray(normalization["bodypart_std"], dtype=np.float32)
    if not np.isfinite(geometry).all() or not np.isfinite(bodypart).all():
        raise RuntimeError("Mixed-grounder inference inputs contain NaN or Inf")
    return geometry.astype(np.float32), bodypart.astype(np.float32)


def _factor_numpy(
    factors: Mapping[str, "torch.Tensor"],
) -> Dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in factors.items()
        if key != "latent"
    }


def embed_database_mixed(
    db_path: Path, checkpoint_path: Path, batch_size: int = 256
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for mixed-grounder embedding")
    with np.load(db_path, allow_pickle=True) as data:
        db = {key: data[key] for key in data.files}
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and _env_bool("V46_53_MIXED_GROUNDER_INFER_CUDA", False)
        else "cpu"
    )
    model = _model_from_checkpoint(checkpoint, device)
    geometry, bodypart = _motion_inputs_for_checkpoint(db, checkpoint)
    collected: Dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(geometry), int(batch_size)):
            end = min(len(geometry), start + int(batch_size))
            factors = _factor_numpy(
                model.encode_motion(
                    torch.from_numpy(geometry[start:end]).to(device),
                    torch.from_numpy(bodypart[start:end]).to(device),
                )
            )
            for key, value in factors.items():
                collected.setdefault(key, []).append(value)
    factors_np = {
        key: np.concatenate(value, axis=0) for key, value in collected.items()
    }
    payload = dict(db)
    payload.update(
        {
            "v46_53_mixed_grounding_schema_version": np.asarray(
                EMBED_SCHEMA, dtype=object
            ),
            "v46_53_mixed_lorentz": factors_np["lorentz"],
            "v46_53_mixed_sphere": factors_np["sphere"],
            "v46_53_mixed_gaussian_mean": factors_np["gaussian_mean"],
            "v46_53_mixed_gaussian_covariance": factors_np[
                "gaussian_covariance"
            ],
            "v46_53_mixed_euclidean": factors_np["euclidean"],
            "v46_53_mixed_uncertainty": factors_np["uncertainty"],
            "v46_53_mixed_curvature": np.asarray(
                float(model.curvature.detach().cpu()), dtype=np.float32
            ),
            "v46_53_mixed_metric_weights": model.metric_weights.detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        }
    )
    backup = db_path.with_name(db_path.stem + ".pre_v46_53_mixed_grounding.npz")
    if not backup.exists():
        shutil.copy2(db_path, backup)
    np.savez_compressed(db_path, **payload)
    report = {
        "schema": EMBED_SCHEMA,
        "db": str(db_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "events": int(len(geometry)),
        "device": str(device),
        "curvature": float(model.curvature.detach().cpu()),
        "metric_weights": model.metric_weights.detach().cpu().tolist(),
        "normalization": "training-sources-only",
        "ok": True,
    }
    db_path.with_name(db_path.stem + ".v46_53_mixed_grounding.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


class MixedGroundingRuntime:
    """Runtime scorer that refuses to synthesize missing real-audio features."""

    def __init__(self, db: Mapping[str, Any], checkpoint_path: Path):
        if torch is None:
            raise RuntimeError("PyTorch is required for mixed-grounder runtime")
        self.checkpoint = _load_torch_checkpoint(checkpoint_path)
        self.config = MixedGrounderConfig(**dict(self.checkpoint["config"]))
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            and _env_bool("V46_53_MIXED_GROUNDER_INFER_CUDA", False)
            else "cpu"
        )
        self.model = _model_from_checkpoint(self.checkpoint, self.device)
        required = (
            "v46_53_mixed_lorentz",
            "v46_53_mixed_sphere",
            "v46_53_mixed_gaussian_mean",
            "v46_53_mixed_gaussian_covariance",
            "v46_53_mixed_euclidean",
            "v46_53_mixed_uncertainty",
        )
        if all(key in db for key in required):
            factors = {
                "lorentz": np.asarray(db[required[0]], dtype=np.float32),
                "sphere": np.asarray(db[required[1]], dtype=np.float32),
                "gaussian_mean": np.asarray(db[required[2]], dtype=np.float32),
                "gaussian_covariance": np.asarray(
                    db[required[3]], dtype=np.float32
                ),
                "euclidean": np.asarray(db[required[4]], dtype=np.float32),
                "uncertainty": np.asarray(db[required[5]], dtype=np.float32),
            }
        else:
            geometry, bodypart = _motion_inputs_for_checkpoint(db, self.checkpoint)
            with torch.no_grad():
                factors = _factor_numpy(
                    self.model.encode_motion(
                        torch.from_numpy(geometry).to(self.device),
                        torch.from_numpy(bodypart).to(self.device),
                    )
                )
        self.event_factors = {
            key: torch.from_numpy(value).to(self.device)
            for key, value in factors.items()
        }

    def _audio_input(
        self, slot: Mapping[str, Any]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        clap = None
        for key in (
            "clap_embedding",
            "clap_features",
            "deep_music_embedding",
            "v46_53_clap_embedding",
        ):
            if key in slot:
                clap = np.asarray(slot[key], dtype=np.float32).reshape(-1)
                break
        temporal = None
        for key in (
            "temporal_features",
            "music_temporal_features",
            "v46_53_temporal_features",
        ):
            if key in slot:
                temporal = np.asarray(slot[key], dtype=np.float32)
                break
        if clap is None or temporal is None:
            return None
        if clap.shape != (self.config.clap_dim,):
            raise RuntimeError(
                f"Runtime CLAP dimension mismatch: {clap.shape}, "
                f"expected={(self.config.clap_dim,)}"
            )
        temporal = _resample_sequence(temporal, temporal.shape[0])
        if temporal.shape[1] != self.config.temporal_dim:
            raise RuntimeError(
                f"Runtime temporal dimension mismatch: {temporal.shape}"
            )
        normalization = self.checkpoint["normalization"]
        temporal = (
            temporal[None]
            - np.asarray(normalization["temporal_mean"], dtype=np.float32)
        ) / np.asarray(normalization["temporal_std"], dtype=np.float32)
        clap = clap / max(float(np.linalg.norm(clap)), 1.0e-8)
        return clap[None].astype(np.float32), temporal.astype(np.float32)

    def score(self, slot: Mapping[str, Any], event_id: int) -> Optional[float]:
        audio_input = self._audio_input(slot)
        if audio_input is None:
            return None
        index = int(event_id)
        if index < 0 or index >= len(self.event_factors["lorentz"]):
            raise IndexError(f"event_id out of range: {index}")
        clap, temporal = audio_input
        with torch.no_grad():
            audio = self.model.encode_audio(
                torch.from_numpy(clap).to(self.device),
                torch.from_numpy(temporal).to(self.device),
            )
            event = {
                key: value[index : index + 1]
                for key, value in self.event_factors.items()
            }
            distance = self.model.pairwise_distance_sq(audio, event)[0, 0]
            variance = (
                audio["uncertainty"][0].square()
                + event["uncertainty"][0].square()
                + 1.0e-5
            )
            score = torch.exp(-distance / (2.0 * variance))
        return float(score.clamp(0.0, 1.0).cpu())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch_size", type=int, default=96)
    train.add_argument("--seed", type=int, default=20260724)
    train.add_argument("--validation_ratio", type=float, default=0.20)
    train.add_argument("--lr", type=float, default=2.0e-4)
    train.add_argument("--weight_decay", type=float, default=1.0e-4)
    train.add_argument("--patience", type=int, default=20)
    embed = subparsers.add_parser("embed")
    embed.add_argument("--db", required=True)
    embed.add_argument("--checkpoint", required=True)
    embed.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.command == "train":
        report = train_mixed_grounder(
            Path(args.data),
            Path(args.out),
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            validation_ratio=args.validation_ratio,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
    else:
        report = embed_database_mixed(
            Path(args.db), Path(args.checkpoint), batch_size=args.batch_size
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
