"""Lightweight post-generation repairability models.

The predictor is deliberately not a safety authority.  It consumes only
inference-visible, pre-generation features and returns a ranking hint.  The
closed-loop generator must still run the real generator and authoritative
Guard before committing a candidate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn


REPAIRABILITY_CHECKPOINT_SCHEMA = "repairability_predictor_checkpoint_v1"

# Stable order is part of the checkpoint contract.  Features are intentionally
# limited to values available after cheap bridge simulation and before neural
# generation.  No post-generation or Guard value may enter this vector.
FEATURE_NAMES: tuple[str, ...] = (
    "router_probability",
    "original_rank_fraction",
    "duration_mismatch_ratio",
    "pre_safe",
    "pre_risk_score",
    "transition_ratio",
    "core_warp_delta",
    "entry_velocity",
    "exit_velocity",
    "entry_acceleration",
    "exit_acceleration",
    "joint_jerk",
    "angular_jerk",
    "boundary_joint_jerk_max",
    "boundary_angular_jerk_max",
    "foot_slip",
    "foot_slip_p95",
    "foot_penetration",
    "foot_penetration_max_m",
    "contact_switch",
    "max_rotation_step_rad",
    "entry_rotation_step_rad",
    "exit_rotation_step_rad",
    "entry_fk_jump",
    "exit_fk_jump",
    "entry_fk_jump_max_m",
    "exit_fk_jump_max_m",
    "high_frequency",
    "predicted_contact_rate",
    "kinematic_contact_rate",
)

VIOLATION_FAMILIES: tuple[str, ...] = (
    "velocity",
    "acceleration",
    "jerk",
    "rotation",
    "fk",
    "foot_slip",
    "penetration",
    "contact",
    "physical_other",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def feature_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Return the exact checkpoint-ordered finite feature vector."""

    return np.asarray(
        [_finite_float(features.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float32,
    )


def proposal_feature_mapping(
    *,
    router_probability: float,
    original_rank: int,
    pool_size: int,
    target_frames: int,
    transition_frames: int,
    core_frames: int,
    core_warp: float,
    pre_safe: bool,
    pre_risk_score: float,
    risk: Mapping[str, Any],
) -> Dict[str, float]:
    """Build inference-visible features shared by runtime and Outcome Bank."""

    target = max(1, int(target_frames))
    pool_denom = max(1, int(pool_size) - 1)
    warp = max(1.0e-6, abs(_finite_float(core_warp, 1.0)))
    estimated_source_frames = float(max(0, int(core_frames))) / warp
    duration_mismatch = abs(estimated_source_frames - target) / float(target)
    result: Dict[str, float] = {
        "router_probability": _finite_float(router_probability),
        "original_rank_fraction": float(max(0, int(original_rank))) / float(pool_denom),
        "duration_mismatch_ratio": float(duration_mismatch),
        "pre_safe": 1.0 if bool(pre_safe) else 0.0,
        "pre_risk_score": _finite_float(pre_risk_score),
        "transition_ratio": float(max(0, int(transition_frames))) / float(target),
        "core_warp_delta": abs(_finite_float(core_warp, 1.0) - 1.0),
    }
    for name in FEATURE_NAMES:
        if name not in result:
            result[name] = _finite_float(risk.get(name, 0.0))
    return result


class RepairabilityPredictor(nn.Module):
    """Linear or small MLP with safe, risk, and optional violation heads."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        *,
        architecture: str = "mlp",
        hidden_dims: Sequence[int] = (64, 64, 32),
        dropout: float = 0.10,
        violation_dim: int = len(VIOLATION_FAMILIES),
    ) -> None:
        super().__init__()
        if architecture not in {"linear", "mlp"}:
            raise ValueError(f"unsupported repairability architecture={architecture!r}")
        self.architecture = str(architecture)
        self.violation_dim = int(violation_dim)
        if architecture == "linear":
            self.encoder = nn.Identity()
            encoded_dim = int(input_dim)
        else:
            dims = [int(input_dim), *[int(value) for value in hidden_dims]]
            layers: list[nn.Module] = []
            for index in range(len(dims) - 1):
                if index == 0:
                    layers.append(nn.LayerNorm(dims[index]))
                layers.extend(
                    [
                        nn.Linear(dims[index], dims[index + 1]),
                        nn.SiLU(),
                    ]
                )
                if index < len(dims) - 2 and float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
            self.encoder = nn.Sequential(*layers)
            encoded_dim = dims[-1]
        self.safe_head = nn.Linear(encoded_dim, 1)
        self.risk_head = nn.Linear(encoded_dim, 1)
        self.violation_head = (
            nn.Linear(encoded_dim, self.violation_dim)
            if self.violation_dim > 0
            else None
        )

    def forward(self, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encoder(values)
        result = {
            "safe_logit": self.safe_head(encoded).squeeze(-1),
            "post_risk_normalized": self.risk_head(encoded).squeeze(-1),
        }
        if self.violation_head is not None:
            result["violation_logits"] = self.violation_head(encoded)
        return result


@dataclass(frozen=True)
class RepairabilityPrediction:
    safe_probability: float
    expected_post_risk: float
    utility: float
    violation_probabilities: Mapping[str, float]
    finite: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "safe_probability": float(self.safe_probability),
            "expected_post_risk": float(self.expected_post_risk),
            "utility": float(self.utility),
            "violation_probabilities": dict(self.violation_probabilities),
            "finite": bool(self.finite),
        }


class RepairabilityRanker:
    """Frozen calibrated predictor used only as a candidate ranking hint."""

    def __init__(self, checkpoint: Mapping[str, Any], *, device: str = "cpu") -> None:
        if checkpoint.get("schema") != REPAIRABILITY_CHECKPOINT_SCHEMA:
            raise RuntimeError("unsupported repairability checkpoint schema")
        names = tuple(str(value) for value in checkpoint.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise RuntimeError("repairability checkpoint feature schema mismatch")
        families = tuple(
            str(value) for value in checkpoint.get("violation_families", ())
        )
        if families and families != VIOLATION_FAMILIES:
            raise RuntimeError("repairability checkpoint violation schema mismatch")
        model_config = dict(checkpoint.get("model_config", {}))
        self.device = torch.device(device)
        self.model = RepairabilityPredictor(
            input_dim=len(FEATURE_NAMES),
            architecture=str(model_config.get("architecture", "mlp")),
            hidden_dims=tuple(model_config.get("hidden_dims", (64, 64, 32))),
            dropout=float(model_config.get("dropout", 0.10)),
            violation_dim=len(VIOLATION_FAMILIES),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        if self.feature_mean.shape != (len(FEATURE_NAMES),) or self.feature_std.shape != (
            len(FEATURE_NAMES),
        ):
            raise RuntimeError("repairability checkpoint normalization shape mismatch")
        if not np.isfinite(self.feature_mean).all() or not np.isfinite(
            self.feature_std
        ).all():
            raise RuntimeError("repairability checkpoint normalization is non-finite")
        self.feature_std = np.maximum(self.feature_std, 1.0e-6)
        self.risk_mean = _finite_float(checkpoint.get("post_risk_mean", 0.0))
        self.risk_std = max(
            1.0e-6, _finite_float(checkpoint.get("post_risk_std", 1.0), 1.0)
        )
        self.temperature = max(
            1.0e-3, _finite_float(checkpoint.get("calibration_temperature", 1.0), 1.0)
        )
        self.risk_transform = str(
            checkpoint.get("post_risk_transform", "identity")
        )
        if self.risk_transform not in {"identity", "log1p"}:
            raise RuntimeError("unsupported repairability post-risk transform")
        selection = dict(checkpoint.get("selection", {}))
        self.risk_weight = max(0.0, _finite_float(selection.get("risk_weight", 0.25), 0.25))
        self.minimum_probability_spread = max(
            0.0,
            _finite_float(selection.get("minimum_probability_spread", 0.02), 0.02),
        )
        self.maximum_abs_z = max(
            1.0, _finite_float(selection.get("maximum_abs_z", 8.0), 8.0)
        )
        self.checkpoint_metadata = {
            "training_fingerprint": checkpoint.get("training_fingerprint"),
            "architecture": model_config.get("architecture", "mlp"),
            "calibration_temperature": self.temperature,
            "risk_weight": self.risk_weight,
            "minimum_probability_spread": self.minimum_probability_spread,
            "maximum_abs_z": self.maximum_abs_z,
            "rank_authorized": bool(
                dict(checkpoint.get("promotion", {})).get("rank_authorized", False)
            ),
        }

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "RepairabilityRanker":
        try:
            checkpoint = torch.load(
                Path(path), map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch versions before the weights_only argument.
            checkpoint = torch.load(Path(path), map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("repairability checkpoint must be a mapping")
        return cls(checkpoint, device=device)

    def predict(self, features: Mapping[str, Any]) -> RepairabilityPrediction:
        raw = feature_vector(features)
        normalized = (raw - self.feature_mean) / self.feature_std
        with torch.no_grad():
            tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
            output = self.model(tensor)
            safe_logit = float(output["safe_logit"].detach().cpu().item())
            risk_normalized = float(
                output["post_risk_normalized"].detach().cpu().item()
            )
            violation_logits = output.get("violation_logits")
        safe_probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, safe_logit / self.temperature))))
        expected_risk_transformed = self.risk_mean + self.risk_std * risk_normalized
        expected_risk = (
            math.expm1(min(40.0, expected_risk_transformed))
            if self.risk_transform == "log1p"
            else expected_risk_transformed
        )
        expected_risk = max(0.0, expected_risk)
        utility = math.log(max(safe_probability, 1.0e-8)) - self.risk_weight * math.log1p(expected_risk)
        violations: Dict[str, float] = {}
        if violation_logits is not None:
            logits = violation_logits.detach().cpu().numpy().reshape(-1)
            violations = {
                name: float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(value))))))
                for name, value in zip(VIOLATION_FAMILIES, logits)
            }
        finite = bool(
            np.isfinite(raw).all()
            and np.isfinite(normalized).all()
            and float(np.max(np.abs(normalized))) <= self.maximum_abs_z
            and math.isfinite(safe_probability)
            and math.isfinite(expected_risk)
            and math.isfinite(utility)
        )
        return RepairabilityPrediction(
            safe_probability=float(safe_probability),
            expected_post_risk=float(expected_risk),
            utility=float(utility),
            violation_probabilities=violations,
            finite=finite,
        )

    def should_abstain(self, predictions: Sequence[RepairabilityPrediction]) -> bool:
        finite = [item for item in predictions if item.finite]
        if len(finite) != len(predictions) or len(finite) < 2:
            return True
        probabilities = [item.safe_probability for item in finite]
        return bool(
            max(probabilities) - min(probabilities)
            < self.minimum_probability_spread
        )
