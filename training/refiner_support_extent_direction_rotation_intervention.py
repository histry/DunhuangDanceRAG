"""SECDR: support-extent conditioned direction rotation intervention.

This is the second and final pre-registered Phase 3 intervention.  The
completed A0 ProductManifoldTemporalRefiner and the completed RCSP adapter are
loaded from their explicit frozen lineage and remain immutable.  A zero
initialized, bias-free root 3x3 / joint 72x72 tangent rotator changes only the
direction of the already projected RCSP geometric correction.  The production
decoder, objective, gate, and inference path are reused unchanged.

The intervention is diagnostic-only.  It does not select a checkpoint, tune a
variant, modify production code, publish a model, or authorize Pilot.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from motion_geometry.boundary_observables import boundary_metrics_torch, observable_gate
from training import motion_models as m
from training import refiner_bctr_reporting_correction as correction
from training import refiner_boundary_crossing_temporal_reduction_intervention as bctr
from training import refiner_cross_width_normalization_audit as phase2
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_safe_start_diagnostics as safe
from training import refiner_temporal_action_alignment_audit as alignment
from training.refiner_optimizer import (
    checked_refiner_step,
    record_update,
    validate_update_summary,
)


SCHEMA = "refiner_support_extent_conditioned_direction_rotation_intervention_v1"
FROZEN_PHASE21_COMMIT = "c461ba44689103cd0690488267e3bd42507ad7ab"
FROZEN_BCTR_COMMIT = "b0cd4437cfb0144046b1408397cc5dad72471cf9"
IMPLEMENTATION_PARENT_COMMIT = FROZEN_BCTR_COMMIT
PRIMARY_CASES = 32
FINAL_CASES = 64
CASES_PER_GROUP = 8
TRAIN_EXPECTED_CROSS_CASES = 96
STEPS = 400
GEOMETRY_DIM = 75
ROOT_DIM = 3
JOINT_DIM = 72
ROTATOR_PARAMETER_COUNT = ROOT_DIM * ROOT_DIM + JOINT_DIM * JOINT_DIM
ROTATOR_EPS = 1.0e-12
MAJOR_GAP_FRACTION = 0.50
PARITY_ATOL = phase2.PARITY_ATOL
PARITY_RTOL = phase2.PARITY_RTOL
WIDTHS = (10, 28)
PRIMARY_ROLE = "cross_event"
CONTROL_ROLE = "single_recording"
GROUP_ORDER = (
    "seen/cross_event/10",
    "seen/cross_event/28",
    "new_position/cross_event/10",
    "new_position/cross_event/28",
)
SUMMARY_SCOPES = (
    "overall",
    "seen",
    "new",
    "seen/cross_event/10",
    "seen/cross_event/28",
    "new_position/cross_event/10",
    "new_position/cross_event/28",
    "width10",
    "width28",
)


def _finite(value: Any, label: str) -> float:
    result = float(value.detach()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite {label}")
    return result


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _tensor_max_error(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    if left.shape != right.shape:
        raise ValueError(f"{label} shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return _finite((left.detach() - right.detach()).abs().max(), label)


def _median(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _bool_exact(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise ValueError(f"{label} must be exactly {expected}")


def _state_value(row: Mapping[str, Any], name: str) -> Any:
    value = row.get(name)
    if value is None:
        raise ValueError(f"SECDR state row is missing {name}")
    return value


def _validate_bctr_report(
    path: Path,
    phase21_path: Path,
    phase21_hash: str,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen BCTR report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    report_hash = _file_sha256(path)
    if report.get("schema") != bctr.SCHEMA:
        raise ValueError("frozen BCTR schema mismatch")
    _bool_exact(report.get("completed"), True, "frozen BCTR completed")
    provenance = report.get("provenance", {})
    if provenance.get("runtime_commit") != FROZEN_BCTR_COMMIT:
        raise ValueError("frozen BCTR runtime commit mismatch")
    if Path(provenance.get("phase21_report", "")).resolve() != phase21_path:
        raise ValueError("BCTR does not reference the explicit Phase 2.1 report")
    if provenance.get("phase21_report_sha256") != phase21_hash:
        raise ValueError("BCTR Phase 2.1 report hash mismatch")
    if provenance.get("source") != str(Path(provenance.get("source", "")).resolve()):
        raise ValueError("BCTR source path is not canonical")
    if report.get("primary_cohort", {}).get("cases") != PRIMARY_CASES:
        raise ValueError("BCTR primary cohort is not exactly 32 cases")
    case_rows = report.get("case_level", [])
    if not isinstance(case_rows, list) or len(case_rows) != PRIMARY_CASES:
        raise ValueError("BCTR case-level report is not exactly 32 cases")
    identities = [row.get("identity") for row in case_rows]
    if len(set(identities)) != PRIMARY_CASES:
        raise ValueError("BCTR case identities are not unique")
    required_false = (
        "parameter_update_performed",
        "production_model_modified",
        "production_inference_modified",
        "scientific_acceptance",
        "publish_allowed",
        "pilot_allowed",
    )
    for field in required_false:
        _bool_exact(report.get(field), False, f"BCTR {field}")
    if report.get("optimizer_steps") != 0:
        raise ValueError("BCTR optimizer_steps must be zero")
    _bool_exact(report.get("no_further_metric_search"), True, "BCTR no_further_metric_search")
    decision = report.get("decision", {})
    if decision.get("result") != correction.EXPECTED_DECISION:
        raise ValueError("BCTR decision is not the frozen NOT_SUPPORTED result")
    if decision.get("next_action") != correction.EXPECTED_NEXT_ACTION:
        raise ValueError("BCTR next action mismatch")
    if decision.get("split_supported") != {"seen": False, "new": False}:
        raise ValueError("BCTR split support must be false for both splits")
    if decision.get("overall_supported") is not False:
        raise ValueError("BCTR overall support must be false")
    if decision.get("no_further_metric_search") is not True:
        raise ValueError("BCTR decision did not stop metric search")
    for scope in ("overall", "seen", "new"):
        summary = report.get("summaries", {}).get(scope, {})
        if summary.get("split_supported") is not False:
            raise ValueError(f"BCTR {scope} split support mismatch")
    integrity = report.get("state_integrity", {})
    for field in required_false:
        _bool_exact(integrity.get(field), False, f"BCTR state_integrity.{field}")
    if integrity.get("optimizer_steps") != 0:
        raise ValueError("BCTR state integrity optimizer_steps must be zero")
    _bool_exact(report.get("parity", {}).get("current_metric_parity_verified"), True, "BCTR current metric parity")
    _bool_exact(report.get("parity", {}).get("model_output_unchanged"), True, "BCTR model output parity")
    return report, report_hash


def _validate_correction_report(
    path: Path,
    bctr_path: Path,
    bctr_hash: str,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"BCTR correction report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    report_hash = _file_sha256(path)
    if report.get("schema") != correction.SCHEMA or report.get("completed") is not True:
        raise ValueError("BCTR correction schema/completed mismatch")
    provenance = report.get("provenance", {})
    if Path(provenance.get("source_bctr_report", "")).resolve() != bctr_path:
        raise ValueError("BCTR correction source path mismatch")
    if provenance.get("source_bctr_report_sha256") != bctr_hash:
        raise ValueError("BCTR correction source hash mismatch")
    if provenance.get("source_decision") != correction.EXPECTED_DECISION:
        raise ValueError("BCTR correction source decision mismatch")
    for field in (
        "source_report_modified",
        "measurements_changed",
        "decision_inputs_changed",
        "scientific_classification_changed",
    ):
        _bool_exact(report.get("correction", {}).get(field), False, f"BCTR correction {field}")
        _bool_exact(report.get(field), False, f"BCTR correction top-level {field}")
    _bool_exact(report.get("correction", {}).get("recomputed_decision_same"), True, "BCTR correction decision identity")
    _bool_exact(report.get("decision", {}).get("recomputed_same_as_source"), True, "BCTR correction recomputed decision")
    if report.get("decision", {}).get("result") != correction.EXPECTED_DECISION:
        raise ValueError("BCTR correction decision result mismatch")
    if report.get("decision", {}).get("next_action") != correction.EXPECTED_NEXT_ACTION:
        raise ValueError("BCTR correction next action mismatch")
    for field in ("scientific_acceptance", "publish_allowed", "pilot_allowed"):
        _bool_exact(report.get(field), False, f"BCTR correction {field}")
    summaries = report.get("corrected_summaries", {})
    for scope in correction.SCOPES:
        if scope not in summaries:
            raise ValueError(f"BCTR correction missing scope {scope}")
        for field in (
            "newly_rescued_cases",
            "width10_newly_rescued_cases",
            "width28_newly_rescued_cases",
        ):
            if field not in summaries[scope]:
                raise ValueError(f"BCTR correction missing {scope}.{field}")
    return report, report_hash


def support_extent_fraction(joint_weight: torch.Tensor, root_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return authoritative binary support, active-frame count and s=N_active/T."""
    support = rcsp.binary_geometry_support(joint_weight, root_weight)
    active = support.any(dim=-1)
    active_count = active.sum(dim=-1).to(support.dtype)
    frames = int(support.shape[1])
    if bool((active_count <= 0).any()):
        raise ValueError("authoritative geometric support has no active frame")
    fraction = active_count / float(frames)
    if not bool(torch.isfinite(fraction).all()) or bool((fraction <= 0).any()) or bool((fraction > 1).any()):
        raise FloatingPointError("support extent fraction is invalid")
    return support, active_count.to(torch.long), fraction


def calibrate_support_extent(batch: Mapping[str, torch.Tensor], cfg: Any) -> dict[str, Any]:
    """Calibrate q only on the fixed TRAIN transaction-0 cross-event repair cases."""
    if "group" not in batch:
        raise ValueError("SECDR TRAIN calibration requires explicit group ids")
    cross = (batch["group"] == 1) | (batch["group"] == 3)
    count = int(cross.sum().item())
    if count != TRAIN_EXPECTED_CROSS_CASES:
        raise ValueError(f"SECDR TRAIN cross-event calibration requires 96 cases, got {count}")
    joint, root, _contact = m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg
    )
    _support, active, fraction = support_extent_fraction(joint[cross], root[cross])
    s_min = _finite(fraction.min(), "support extent s_min")
    s_max = _finite(fraction.max(), "support extent s_max")
    if not s_max > s_min:
        raise ValueError("support extent calibration requires s_max > s_min")
    q = support_extent_q(fraction, s_min, s_max)
    if bool((q < 0).any()) or bool((q > 1).any()):
        raise RuntimeError("support extent q is outside [0,1]")
    order = torch.argsort(fraction)
    if bool((q[order][1:] < q[order][:-1]).any()):
        raise RuntimeError("support extent q is not monotonic")
    values = fraction.detach().cpu().numpy().astype(np.float64)
    return {
        "cases": count,
        "active_support_frames": [int(value) for value in active.detach().cpu().tolist()],
        "s_values": [float(value) for value in values.tolist()],
        "s_min": s_min,
        "s_max": s_max,
        "q_min": _finite(q.min(), "q_min"),
        "q_max": _finite(q.max(), "q_max"),
        "q_monotonic": True,
        "distribution": {
            "min": float(values.min()),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(values.max()),
        },
    }


def support_extent_q(s: torch.Tensor | float, s_min: float, s_max: float) -> torch.Tensor | float:
    if not math.isfinite(float(s_min)) or not math.isfinite(float(s_max)) or not s_max > s_min:
        raise ValueError("support extent calibration range is invalid")
    if torch.is_tensor(s):
        return ((s - float(s_min)) / float(s_max - s_min)).clamp(0.0, 1.0)
    return min(1.0, max(0.0, (float(s) - float(s_min)) / float(s_max - s_min)))


def effective_conditioner_q(
    role_id: torch.Tensor,
    fraction: torch.Tensor,
    s_min: float,
    s_max: float,
) -> torch.Tensor:
    if role_id.shape != fraction.shape or role_id.ndim != 1:
        raise ValueError("role ids and support extent fractions must be rank-1 and aligned")
    if role_id.dtype not in (torch.int32, torch.int64):
        raise ValueError("role ids must be integer tensors")
    if bool(((role_id < 0) | (role_id > 1)).any()):
        raise ValueError("role ids must be 0=single_recording or 1=cross_event")
    q = support_extent_q(fraction, s_min, s_max)
    return torch.where(role_id == rcsp.ROLE_MAPPING[CONTROL_ROLE], torch.zeros_like(q), q)


class TangentDirectionRotator(nn.Module):
    """Bias-free root/joint tangent maps with exact 5193 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.root = nn.Linear(ROOT_DIM, ROOT_DIM, bias=False)
        self.joint = nn.Linear(JOINT_DIM, JOINT_DIM, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.root.weight)
        nn.init.zeros_(self.joint.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _rotate_block(
        action: torch.Tensor,
        support: torch.Tensor,
        q: torch.Tensor,
        layer: nn.Linear,
    ) -> torch.Tensor:
        action = action * support
        u = F.linear(action, layer.weight)
        u = u * support
        action_norm_sq = action.square().sum(dim=-1, keepdim=True)
        u_perp = u - action * (action * u).sum(dim=-1, keepdim=True) / (action_norm_sq + ROTATOR_EPS)
        v = action + q[:, None, None] * u_perp
        v_norm = v.norm(dim=-1, keepdim=True)
        action_norm = action.norm(dim=-1, keepdim=True)
        rotated = v * action_norm / v_norm.clamp_min(ROTATOR_EPS)
        rotated = torch.where(action_norm <= ROTATOR_EPS, action, rotated)
        update_zero = u.abs().sum(dim=-1, keepdim=True) == 0.0
        rotated = torch.where(update_zero, action, rotated)
        # q=0 is an exact bypass, including single_recording controls and the
        # lower calibration endpoint.  This also avoids numerical drift in
        # the zero-initialized parity contract.
        return torch.where(q[:, None, None] > 0.0, rotated, action)

    def forward(
        self,
        action: torch.Tensor,
        support: torch.Tensor,
        q: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if action.ndim != 3 or action.shape[-1] != GEOMETRY_DIM:
            raise ValueError("SECDR action must have shape [B,T,75]")
        if support.shape != action.shape or q.shape != (action.shape[0],):
            raise ValueError("SECDR action/support/q layouts differ")
        root_support = support[..., :ROOT_DIM]
        joint_support = support[..., ROOT_DIM:]
        root = self._rotate_block(action[..., :ROOT_DIM], root_support, q, self.root)
        joint = self._rotate_block(action[..., ROOT_DIM:], joint_support, q, self.joint)
        rotated = torch.cat((root, joint), dim=-1)
        rotated = rotated * support
        return rotated, {
            "input_action": action,
            "rotated_action": rotated,
            "binary_support": support,
            "q": q,
        }


class SECDRModel(nn.Module):
    """Route explicit roles and insert SECDR before the unchanged decoder."""

    def __init__(self, rcsp_model: rcsp.FrozenBaseRCSPModel, s_min: float, s_max: float) -> None:
        super().__init__()
        self.rcsp = rcsp_model
        self.base = rcsp_model.base
        self.rotator = TangentDirectionRotator().to(next(self.base.parameters()).device)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.out = nn.Identity()
        self._active_route = None
        self._last_details = None
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.rcsp.adapter.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.base.eval()
        self.rcsp.eval()
        self.validate_parameter_scope()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.rcsp.base.eval()
        self.rcsp.adapter.eval()
        return self

    @property
    def last_details(self) -> dict[str, Any]:
        if self._last_details is None:
            raise RuntimeError("no SECDR forward details are available")
        return self._last_details

    def clear_last_details(self) -> None:
        self._last_details = None
        self.rcsp.clear_last_details()

    def validate_parameter_scope(self) -> dict[str, Any]:
        base = list(self.base.named_parameters())
        adapter = list(self.rcsp.adapter.named_parameters())
        rotator = list(self.rotator.named_parameters())
        if not base or not adapter or not rotator:
            raise RuntimeError("SECDR parameter scopes are incomplete")
        if any(parameter.requires_grad or parameter.grad is not None for _, parameter in base + adapter):
            raise RuntimeError("frozen base/RCSP parameter is trainable or has gradient residue")
        if any(not parameter.requires_grad for _, parameter in rotator):
            raise RuntimeError("SECDR rotator parameter is unexpectedly frozen")
        if self.rotator.parameter_count != ROTATOR_PARAMETER_COUNT:
            raise RuntimeError("SECDR rotator parameter count is not 5193")
        if self.rotator.root.bias is not None or self.rotator.joint.bias is not None:
            raise RuntimeError("SECDR rotator must be bias-free")
        return {
            "base_parameters": sum(parameter.numel() for _, parameter in base),
            "rcsp_adapter_parameters": sum(parameter.numel() for _, parameter in adapter),
            "rotator_parameters": self.rotator.parameter_count,
            "trainable_parameter_names": [f"rotator.{name}" for name, _ in rotator],
            "optimizer_parameter_scope": ["rotator.root.weight", "rotator.joint.weight"],
        }

    def validate_zero_initialization(self) -> dict[str, Any]:
        exact = all(bool((parameter.detach() == 0).all()) for parameter in self.rotator.parameters())
        if not exact:
            raise RuntimeError("SECDR rotator is not exactly zero initialized")
        return {
            "root_weight_exactly_zero": bool((self.rotator.root.weight.detach() == 0).all()),
            "joint_weight_exactly_zero": bool((self.rotator.joint.weight.detach() == 0).all()),
            "all_parameters_exactly_zero": exact,
            "parameter_count": self.rotator.parameter_count,
        }

    @contextmanager
    def route(
        self,
        role_id: torch.Tensor,
        support: torch.Tensor,
        q: torch.Tensor,
        *,
        capture_details: bool = False,
        mode: str = "secdr",
    ):
        if self._active_route is not None:
            raise RuntimeError("nested SECDR routes are forbidden")
        if mode not in {"rcsp", "secdr"}:
            raise ValueError("unknown SECDR route mode")
        self._active_route = (role_id, support, q, bool(capture_details), mode)
        self._last_details = None
        try:
            yield
        finally:
            self._active_route = None

    def forward(self, x, cond, seam_mask, joint_mask):
        if self._active_route is None:
            raise RuntimeError("SECDR forward requires explicit role/support routing")
        role_id, support, q, capture_details, mode = self._active_route
        root_weight = support[..., :ROOT_DIM].any(dim=-1, keepdim=True).to(support.dtype)
        joint_weight = support[..., ROOT_DIM:].reshape(support.shape[:-1] + (24, 3)).any(dim=-1).to(support.dtype)
        raw_rcsp = self.rcsp.forward_explicit(
            x,
            cond,
            seam_mask,
            joint_mask,
            role_id,
            joint_weight,
            root_weight,
            capture_details=True,
        )
        details = self.rcsp.last_details
        action = details["adapter_projected"]
        if mode == "rcsp":
            raw_output = raw_rcsp.detach().requires_grad_(True)
            rotated = action
        else:
            rotated, rotation_details = self.rotator(action, support, q)
            raw_output = torch.cat(
                (raw_rcsp[..., :4], details["raw_base"][..., 4:] + rotated), dim=-1
            )
            if not torch.equal(raw_output[..., :4], raw_rcsp[..., :4]):
                raise RuntimeError("SECDR changed frozen contact channels")
            # ``rotation_details`` contains only the correction, never the
            # full base geometry.  Keep it for the report and clear it after
            # the consuming batch.
            _ = rotation_details
        # Keep the output-head hook layout identical to the production
        # Conv1d head: the alignment audit captures [B,C,T] and transposes it
        # back to the public [B,T,C] refiner layout.
        output = self.out(raw_output.transpose(1, 2)).transpose(1, 2)
        if capture_details:
            self._last_details = {
                "role_id": role_id.detach(),
                "binary_support": support.detach(),
                "q": q.detach(),
                "raw_base": details["raw_base"].detach(),
                "raw_rcsp": raw_rcsp.detach(),
                "rcsp_action": action.detach(),
                "secdr_action": rotated.detach(),
                "raw_secdr": output,
                "mode": mode,
            }
        else:
            self._last_details = None
        return output


def _route_values(
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
    calibration: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    role_id, joint_weight, root_weight = rcsp._route_values(batch, cfg)
    support, active, fraction = support_extent_fraction(joint_weight, root_weight)
    q = effective_conditioner_q(
        role_id,
        fraction,
        float(calibration["s_min"]),
        float(calibration["s_max"]),
    )
    return role_id, joint_weight, root_weight, support, active, q


def attach_train_role_ids(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return rcsp.attach_train_role_ids(batch)


def secdr_batch_outputs(
    model: SECDRModel,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
    calibration: Mapping[str, Any],
    *,
    trace: dict[str, Any] | None = None,
    capture_details: bool = False,
    mode: str = "secdr",
):
    role_id, joint_weight, root_weight, support, _active, q = _route_values(batch, cfg, calibration)
    with model.route(role_id, support, q, capture_details=capture_details, mode=mode):
        return m._refiner_batch_outputs(model, batch, cfg, trace=trace)


def secdr_batch_objectives(model, batch, cfg, calibration, *, capture_details=False):
    role_id, _joint_weight, _root_weight, support, _active, q = _route_values(batch, cfg, calibration)
    with model.route(role_id, support, q, capture_details=capture_details, mode="secdr"):
        return m._refiner_batch_objectives(model, batch, cfg)


def secdr_guarded_total_batch_loss(model, batch, cfg, calibration):
    role_id, _joint_weight, _root_weight, support, _active, q = _route_values(batch, cfg, calibration)
    with model.route(role_id, support, q, capture_details=False, mode="secdr"):
        return m._refiner_guarded_total_batch_loss(model, batch, cfg, require_all_groups=False)


def _cross_train_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "group" not in batch:
        raise ValueError("SECDR training requires explicit group metadata")
    selected = (batch["group"] == 1) | (batch["group"] == 3)
    if int(selected.sum()) != TRAIN_EXPECTED_CROSS_CASES:
        raise ValueError("SECDR cross-event TRAIN case count is not 96")
    return {key: value[selected] for key, value in batch.items()}


def _update_norms(model: SECDRModel, before: Mapping[str, torch.Tensor]) -> dict[str, float]:
    values = [
        (parameter.detach().double() - before[name].double()).square().sum()
        for name, parameter in model.rotator.named_parameters()
    ]
    return {"total": math.sqrt(sum(float(value) for value in values))}


def _rotator_norms(model: SECDRModel) -> dict[str, float]:
    root = _finite(model.rotator.root.weight.detach().double().norm(), "root rotator norm")
    joint = _finite(model.rotator.joint.weight.detach().double().norm(), "joint rotator norm")
    return {"root": root, "joint": joint, "total": math.sqrt(root * root + joint * joint)}


def _compact_update(update: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol", "optimizer_update_accepted", "direction", "step_scale", "reason",
        "loss_before", "loss_after", "minimum_loss_decrease", "armijo_factor",
        "trial_evaluations", "loss_rejected_trials", "nonfinite_trials",
        "insufficient_decrease_trials", "used_gradient_rescue", "adam_directional_derivative",
        "group_guard_enabled", "group_guard_before", "group_guard_after",
        "group_guard_rejected_trials", "group_guard_last_violations", "gradient_unscale",
    )
    return {key: update.get(key) for key in keys}


def train_rotator(model: SECDRModel, train_batch: Mapping[str, torch.Tensor], cfg: Any, calibration: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    updates_path = destination / "updates.jsonl"
    updates_path.touch(exist_ok=False)
    optimizer = torch.optim.AdamW(model.rotator.parameters(), lr=cfg.lr, weight_decay=1.0e-4)
    summary: dict[str, Any] = {}
    started = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        repair, clean, terms, _identity_terms = secdr_batch_objectives(
            model, train_batch, cfg, calibration, capture_details=True
        )
        loss = repair + cfg.product_refiner_clean_identity_weight * clean
        model.clear_last_details()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.validate_parameter_scope()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.rotator.named_parameters()
        }
        clip_norm = float(torch.nn.utils.clip_grad_norm_(model.rotator.parameters(), 1.0, error_if_nonfinite=True))
        update = checked_refiner_step(
            optimizer,
            loss,
            lambda: secdr_guarded_total_batch_loss(model, train_batch, cfg, calibration),
            gradient_unscale=max(1.0, clip_norm + 1.0e-6),
            group_guard_before=m._refiner_group_repair_losses(terms, require_all=False),
            group_guard_relative_tolerance=cfg.product_refiner_group_guard_relative_tolerance,
            group_guard_absolute_tolerance=cfg.product_refiner_group_guard_absolute_tolerance,
        )
        record_update(summary, update)
        updates = _update_norms(model, before)
        if not update["optimizer_update_accepted"] and updates["total"] != 0.0:
            raise RuntimeError("rolled-back SECDR step changed rotator parameters")
        row = {
            "step": step,
            "state_position": "after_checked_step",
            "transaction_index": 0,
            "cases": int(train_batch["clean"].shape[0]),
            "cross_event_cases": TRAIN_EXPECTED_CROSS_CASES,
            "training_objective_before": _finite(loss, "SECDR training objective"),
            "optimizer": _compact_update(update),
            "accepted": bool(update["optimizer_update_accepted"]),
            "rolled_back": not bool(update["optimizer_update_accepted"]),
            "rotator_parameter_norm": _rotator_norms(model),
            "rotator_update_norm": updates,
        }
        with updates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if step in (1, 2, 5, 10, 25, 50, 100, 200, 300, 400):
            print(json.dumps({
                "stage": "secdr_rotator_training",
                "step": step,
                "accepted": row["accepted"],
                "objective": row["training_objective_before"],
                "rotator_parameter_norm": row["rotator_parameter_norm"],
                "rotator_update_norm": row["rotator_update_norm"],
                "elapsed_seconds": time.perf_counter() - started,
            }, allow_nan=False), flush=True)
    validate_update_summary(summary, STEPS)
    return {
        "optimizer_summary": summary,
        "optimizer_steps": STEPS,
        "accepted_steps": int(summary["accepted_steps"]),
        "rollback_steps": int(summary["retained_steps"]),
        "final_rotator_parameter_norm": _rotator_norms(model),
        "train_transaction_index": 0,
        "train_cases": int(train_batch["clean"].shape[0]),
        "cross_event_cases": TRAIN_EXPECTED_CROSS_CASES,
        "updates_artifact": {
            "path": str(updates_path),
            "sha256": _file_sha256(updates_path),
            "rows": STEPS,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _base_outputs(base, batch, cfg, *, trace=None):
    with alignment._capture_model_output(base) as captured:
        prediction, identity = m._refiner_batch_outputs(base, batch, cfg, trace=trace)
    if len(captured) != 1:
        raise RuntimeError("BASE output capture requires exactly one forward")
    return prediction, identity, captured[0].transpose(1, 2).detach()


def _block_norms(action: torch.Tensor) -> dict[str, Any]:
    root = action[..., :3].double().norm(dim=-1)
    joint = action[..., 3:].double().norm(dim=-1)
    return {
        "root_max": _finite(root.max(), "root action norm"),
        "root_mean": _finite(root.mean(), "root action norm mean"),
        "joint_max": _finite(joint.max(), "joint action norm"),
        "joint_mean": _finite(joint.mean(), "joint action norm mean"),
    }


def _block_norm_error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_root = left[..., :3].double().norm(dim=-1)
    right_root = right[..., :3].double().norm(dim=-1)
    left_joint = left[..., 3:].double().norm(dim=-1)
    right_joint = right[..., 3:].double().norm(dim=-1)
    return {
        "root_max_abs_error": _finite((left_root - right_root).abs().max(), "root norm preservation error"),
        "joint_max_abs_error": _finite((left_joint - right_joint).abs().max(), "joint norm preservation error"),
    }


def _vector_cosine(left: torch.Tensor, right: torch.Tensor) -> list[float | None]:
    a = left.detach().double().reshape(left.shape[0], -1)
    b = right.detach().double().reshape(right.shape[0], -1)
    an, bn = a.norm(dim=1), b.norm(dim=1)
    result: list[float | None] = []
    for index in range(a.shape[0]):
        if float(an[index]) == 0.0 or float(bn[index]) == 0.0:
            result.append(None)
        else:
            result.append(max(-1.0, min(1.0, _finite((a[index] * b[index]).sum(), "rotation cosine") / _finite(an[index] * bn[index], "rotation cosine denominator"))))
    return result


def _state_payload(row: Mapping[str, Any], gate_margin: float | None = None) -> dict[str, Any]:
    return {
        "M": row["temporal_metric"],
        "G": row["temporal_repair_gain"],
        "endpoint_metric": row["endpoint_metric"],
        "gate_margin": row["temporal_repair_gain"] if gate_margin is None else gate_margin,
        "temporal_pass": row["temporal_gate_pass"],
        "endpoint_acceptance": row["endpoint_gate_pass"],
        "jerk_non_regression": row["observable"]["jerk_non_regression"],
        "overall_acceptance": row["all_diagnostic_conditions"],
        "physical_pass": row["physical_pass"],
        "geometry_pass": row["geometry_pass"],
        "clean_pass": row["clean_pass"],
        "observable": row["observable"],
        "physical": row["physical"],
        "geometry": row["geometry"],
        "clean_identity": row["clean_identity"],
    }


def _current_metric_parity(
    identity: str,
    base_row: Mapping[str, Any],
    rcsp_row: Mapping[str, Any],
    bctr_row: Mapping[str, Any],
) -> dict[str, Any]:
    current = bctr_row["current"]
    observed = {
        "M_before": base_row["observable"]["before"]["temporal_energy"],
        "M_base": base_row["temporal_metric"],
        "M_rcsp": rcsp_row["temporal_metric"],
        "G_base": base_row["temporal_repair_gain"],
        "G_rcsp": rcsp_row["temporal_repair_gain"],
    }
    expected = {key: current[key] for key in observed}
    errors = {key: abs(float(observed[key]) - float(expected[key])) for key in observed}
    maximum = max(errors.values(), default=0.0)
    pass_fields = {
        "temporal_pass_base": base_row["temporal_gate_pass"] == current["temporal_pass_base"],
        "temporal_pass_rcsp": rcsp_row["temporal_gate_pass"] == current["temporal_pass_rcsp"],
        "endpoint_base": base_row["endpoint_gate_pass"] == current["endpoint_acceptance_base"],
        "endpoint_rcsp": rcsp_row["endpoint_gate_pass"] == current["endpoint_acceptance_rcsp"],
        "jerk_base": base_row["observable"]["jerk_non_regression"] == current["jerk_non_regression_base"],
        "jerk_rcsp": rcsp_row["observable"]["jerk_non_regression"] == current["jerk_non_regression_rcsp"],
    }
    return {
        "identity": identity,
        "verified": bool(maximum <= PARITY_ATOL and all(pass_fields.values())),
        "max_abs_error": maximum,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "values": {key: {"recomputed": observed[key], "bctr": expected[key]} for key in observed},
        "errors": errors,
        "boolean_fields": pass_fields,
    }


def _zero_parity(
    base,
    rcsp_model,
    secdr_model,
    batch,
    cfg,
    calibration,
) -> dict[str, Any]:
    base_trace: dict[str, Any] = {}
    rcsp_trace: dict[str, Any] = {}
    secdr_trace: dict[str, Any] = {}
    with torch.no_grad():
        base_prediction, base_identity, base_raw = _base_outputs(base, batch, cfg)
        rcsp_prediction, rcsp_identity = rcsp.rcsp_batch_outputs(
            rcsp_model, batch, cfg, trace=rcsp_trace, capture_details=True
        )
        rcsp_details = rcsp_model.last_details
        secdr_prediction, secdr_identity = secdr_batch_outputs(
            secdr_model, batch, cfg, calibration, trace=secdr_trace, capture_details=True, mode="secdr"
        )
        secdr_details = secdr_model.last_details
    fields = {
        "raw_rcsp_vs_secdr": _tensor_max_error(rcsp_details["raw_adapted"], secdr_details["raw_secdr"], "zero raw RCSP/SECDR"),
        "projected_action_vs_rotated_action": _tensor_max_error(rcsp_details["adapter_projected"], secdr_details["secdr_action"], "zero projected action"),
        "decoded_repair": _tensor_max_error(rcsp_prediction, secdr_prediction, "zero decoded repair"),
        "decoded_clean": _tensor_max_error(rcsp_identity, secdr_identity, "zero decoded clean"),
        "final_tangent": _tensor_max_error(rcsp_trace["repair"]["after_cap"], secdr_trace["repair"]["after_cap"], "zero final tangent"),
        "contact": _tensor_max_error(rcsp_details["raw_adapted"][..., :4], secdr_details["raw_secdr"][..., :4], "zero contact"),
    }
    _, rcsp_terms = m._observable_refiner_objective(rcsp_prediction, batch["bad"], batch["seam"], cfg, reduction="none")
    _, secdr_terms = m._observable_refiner_objective(secdr_prediction, batch["bad"], batch["seam"], cfg, reduction="none")
    fields["temporal_observable"] = _tensor_max_error(rcsp_terms["temporal_scientific_deficit"], secdr_terms["temporal_scientific_deficit"], "zero temporal observable")
    fields["endpoint_observable"] = _tensor_max_error(rcsp_terms["endpoint_scientific_deficit"], secdr_terms["endpoint_scientific_deficit"], "zero endpoint observable")
    metric_joints = m._observable_boundary_joints_torch(torch.cat((rcsp_prediction, secdr_prediction)))
    metric_values = boundary_metrics_torch(
        metric_joints,
        torch.cat((batch["seam"], batch["seam"])),
        cfg.fps,
    )
    count = int(batch["clean"].shape[0])
    rcsp_metric = {key: value[:count] for key, value in metric_values.items()}
    secdr_metric = {key: value[count:] for key, value in metric_values.items()}
    fields["current_temporal_metric"] = _tensor_max_error(
        rcsp_metric["temporal_energy"], secdr_metric["temporal_energy"], "zero current temporal metric"
    )
    fields["current_endpoint_metric"] = _tensor_max_error(
        rcsp_metric["endpoint_velocity_jump_mps"], secdr_metric["endpoint_velocity_jump_mps"], "zero current endpoint metric"
    )
    exact = all(value == 0.0 for value in fields.values())
    if not exact:
        raise RuntimeError("zero-initialized SECDR parity failed")
    secdr_model.clear_last_details()
    rcsp_model.clear_last_details()
    return {"cases": int(batch["clean"].shape[0]), "verified": True, "exact": exact, "max_abs_errors": fields}


def _alignment_point(model: SECDRModel, batch, cfg, calibration) -> dict[str, Any]:
    role_id, _joint_weight, _root_weight, support, _active, q = _route_values(batch, cfg, calibration)
    with model.route(role_id, support, q, capture_details=True, mode="rcsp"):
        return alignment.production_current_point(model, batch, cfg)


def _evaluate_chunk(
    base,
    rcsp_model,
    secdr_model,
    batch,
    metadata,
    cfg,
    calibration,
    bctr_map,
):
    count = len(metadata)
    base_trace: dict[str, Any] = {}
    rcsp_trace: dict[str, Any] = {}
    secdr_trace: dict[str, Any] = {}
    with torch.no_grad():
        base_prediction, base_identity, _base_raw = _base_outputs(
            base, batch, cfg, trace=base_trace
        )
        rcsp_prediction, rcsp_identity = rcsp.rcsp_batch_outputs(
            rcsp_model, batch, cfg, trace=rcsp_trace, capture_details=True
        )
        rcsp_details = rcsp_model.last_details
        secdr_prediction, secdr_identity = secdr_batch_outputs(
            secdr_model, batch, cfg, calibration, trace=secdr_trace, capture_details=True, mode="secdr"
        )
        secdr_details = secdr_model.last_details
    _, base_terms = m._observable_refiner_objective(base_prediction, batch["bad"], batch["seam"], cfg, reduction="none")
    _, rcsp_terms = m._observable_refiner_objective(rcsp_prediction, batch["bad"], batch["seam"], cfg, reduction="none")
    _, secdr_terms = m._observable_refiner_objective(secdr_prediction, batch["bad"], batch["seam"], cfg, reduction="none")
    point = _alignment_point(secdr_model, batch, cfg, calibration)
    gradient = point["gradients"]["temporal"]
    rcsp_action = rcsp_details["adapter_projected"][:count].detach().cpu()
    secdr_action = secdr_details["secdr_action"][:count].detach().cpu()
    gradient = gradient[:count]
    rcsp_alignment = alignment.alignment_stats(rcsp_action, gradient)
    secdr_alignment = alignment.alignment_stats(secdr_action, gradient)
    route_support = secdr_details["binary_support"][:count]
    active = route_support.any(dim=-1).sum(dim=-1).detach().cpu()
    fraction = active.double() / float(route_support.shape[1])
    q = secdr_details["q"][:count].detach().cpu()
    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        identity = phase2._identity_key(meta)
        base_row = rcsp._case_row(meta, base_prediction, base_identity, batch, index, base_terms, cfg)
        rcsp_row = rcsp._case_row(meta, rcsp_prediction, rcsp_identity, batch, index, rcsp_terms, cfg)
        secdr_row = rcsp._case_row(meta, secdr_prediction, secdr_identity, batch, index, secdr_terms, cfg)
        before_metric = float(base_row["observable"]["before"]["temporal_energy"])
        current = {
            "M_before": before_metric,
            "M_base": base_row["temporal_metric"],
            "M_rcsp": rcsp_row["temporal_metric"],
            "M_secdr": secdr_row["temporal_metric"],
            "G_base": base_row["temporal_repair_gain"],
            "G_rcsp": rcsp_row["temporal_repair_gain"],
            "G_secdr": secdr_row["temporal_repair_gain"],
            "gate_margin_base": base_row["temporal_repair_gain"] - float(cfg.checkpoint_validation_min_temporal_repair_gain),
            "gate_margin_rcsp": rcsp_row["temporal_repair_gain"] - float(cfg.checkpoint_validation_min_temporal_repair_gain),
            "gate_margin_secdr": secdr_row["temporal_repair_gain"] - float(cfg.checkpoint_validation_min_temporal_repair_gain),
            "temporal_pass_base": base_row["temporal_gate_pass"],
            "temporal_pass_rcsp": rcsp_row["temporal_gate_pass"],
            "temporal_pass_secdr": secdr_row["temporal_gate_pass"],
            "endpoint_acceptance_base": base_row["endpoint_gate_pass"],
            "endpoint_acceptance_rcsp": rcsp_row["endpoint_gate_pass"],
            "endpoint_acceptance_secdr": secdr_row["endpoint_gate_pass"],
            "jerk_non_regression_base": base_row["observable"]["jerk_non_regression"],
            "jerk_non_regression_rcsp": rcsp_row["observable"]["jerk_non_regression"],
            "jerk_non_regression_secdr": secdr_row["observable"]["jerk_non_regression"],
            "overall_acceptance_base": base_row["all_diagnostic_conditions"],
            "overall_acceptance_rcsp": rcsp_row["all_diagnostic_conditions"],
            "overall_acceptance_secdr": secdr_row["all_diagnostic_conditions"],
        }
        before_tangent = base_trace["repair"]["after_cap"][index]
        rcsp_tangent = rcsp_trace["repair"]["after_cap"][index]
        secdr_tangent = secdr_trace["repair"]["after_cap"][index]
        action_norm_rcsp = _finite((rcsp_tangent - before_tangent).double().reshape(-1).norm(), "RCSP applied action norm")
        action_norm_secdr = _finite((secdr_tangent - before_tangent).double().reshape(-1).norm(), "SECDR applied action norm")
        delta_rcsp = float(current["G_rcsp"] - current["G_base"])
        delta_secdr = float(current["G_secdr"] - current["G_base"])
        safety_non_regression = all(
            not bool(rcsp_row[field]) or bool(secdr_row[field])
            for field in ("physical_pass", "geometry_pass", "clean_pass")
        )
        state = {
            "BASE": _state_payload(base_row, current["gate_margin_base"]),
            "RCSP": _state_payload(rcsp_row, current["gate_margin_rcsp"]),
            "SECDR": _state_payload(secdr_row, current["gate_margin_secdr"]),
        }
        row = {
            **meta,
            "identity": identity,
            "observed_group_pairing": "UNPAIRED",
            "support_extent_fraction": _finite(fraction[index], "support extent fraction"),
            "active_support_frames": int(active[index]),
            "conditioner_q": _finite(q[index], "conditioner q"),
            "M_before": current["M_before"],
            "M_base": current["M_base"],
            "M_rcsp": current["M_rcsp"],
            "M_secdr": current["M_secdr"],
            "G_base": current["G_base"],
            "G_rcsp": current["G_rcsp"],
            "G_secdr": current["G_secdr"],
            "gate_margin_base": current["gate_margin_base"],
            "gate_margin_rcsp": current["gate_margin_rcsp"],
            "gate_margin_secdr": current["gate_margin_secdr"],
            "temporal_pass_base": current["temporal_pass_base"],
            "temporal_pass_rcsp": current["temporal_pass_rcsp"],
            "temporal_pass_secdr": current["temporal_pass_secdr"],
            "endpoint_acceptance_base": current["endpoint_acceptance_base"],
            "endpoint_acceptance_rcsp": current["endpoint_acceptance_rcsp"],
            "endpoint_acceptance_secdr": current["endpoint_acceptance_secdr"],
            "jerk_non_regression_base": current["jerk_non_regression_base"],
            "jerk_non_regression_rcsp": current["jerk_non_regression_rcsp"],
            "jerk_non_regression_secdr": current["jerk_non_regression_secdr"],
            "overall_acceptance_base": current["overall_acceptance_base"],
            "overall_acceptance_rcsp": current["overall_acceptance_rcsp"],
            "overall_acceptance_secdr": current["overall_acceptance_secdr"],
            "applied_action_norm_rcsp": action_norm_rcsp,
            "applied_action_norm_secdr": action_norm_secdr,
            "relative_temporal_gain_rcsp": delta_rcsp,
            "relative_temporal_gain_secdr": delta_secdr,
            "gain_per_action_norm_rcsp": _ratio(delta_rcsp, action_norm_rcsp),
            "gain_per_action_norm_secdr": _ratio(delta_secdr, action_norm_secdr),
            "temporal_alignment_rcsp": rcsp_alignment[index],
            "temporal_alignment_secdr": secdr_alignment[index],
            "rotation_cosine_rcsp_secdr": _vector_cosine(rcsp_action[index:index + 1], secdr_action[index:index + 1])[0],
            "rotation_one_minus_cosine": (
                None if _vector_cosine(rcsp_action[index:index + 1], secdr_action[index:index + 1])[0] is None
                else 1.0 - _vector_cosine(rcsp_action[index:index + 1], secdr_action[index:index + 1])[0]
            ),
            "predecoder_action_norms": {
                "RCSP": _block_norms(rcsp_action[index:index + 1]),
                "SECDR": _block_norms(secdr_action[index:index + 1]),
            },
            "predecoder_norm_preservation_error": _block_norm_error(rcsp_action[index:index + 1], secdr_action[index:index + 1]),
            "binary_support_identical": bool(torch.equal(rcsp_details["binary_support"][index], secdr_details["binary_support"][index])),
            "contact_identical": bool(torch.equal(rcsp_details["raw_adapted"][index, ..., :4], secdr_details["raw_secdr"][index, ..., :4])),
            "single_recording_control": meta["role"] == CONTROL_ROLE,
            "single_output_exact": bool(
                torch.equal(rcsp_prediction[index], secdr_prediction[index])
                and torch.equal(rcsp_identity[index], secdr_identity[index])
                and torch.equal(rcsp_trace["repair"]["after_cap"][index], secdr_trace["repair"]["after_cap"][index])
            ),
            "safety_non_regression": safety_non_regression,
            "states": state,
        }
        if meta["role"] == PRIMARY_ROLE:
            parity = _current_metric_parity(identity, base_row, rcsp_row, bctr_map[identity])
            if not parity["verified"]:
                raise RuntimeError(f"current metric parity failed for {identity}")
            row["current_metric_parity"] = parity
            parity_rows.append(parity)
        else:
            row["current_metric_parity"] = None
        rows.append(row)
    secdr_model.clear_last_details()
    rcsp_model.clear_last_details()
    return rows, parity_rows


def _scope_rows(rows: list[Mapping[str, Any]], scope: str) -> list[Mapping[str, Any]]:
    if scope == "overall":
        return list(rows)
    if scope == "seen":
        return [row for row in rows if row["split"] == "seen"]
    if scope == "new":
        return [row for row in rows if row["split"] == "new_position"]
    if scope == "width10":
        return [row for row in rows if int(row["width"]) == 10]
    if scope == "width28":
        return [row for row in rows if int(row["width"]) == 28]
    if scope in GROUP_ORDER:
        return [row for row in rows if f"{row['split']}/{row['role']}/{int(row['width'])}" == scope]
    raise ValueError(f"unknown SECDR summary scope: {scope}")


def _summary(rows: list[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"SECDR summary scope is empty: {scope}")
    def values(field: str):
        return [row.get(field) for row in rows]
    cosine_rcsp = [row["temporal_alignment_rcsp"].get("cosine_to_negative_gradient") for row in rows]
    cosine_secdr = [row["temporal_alignment_secdr"].get("cosine_to_negative_gradient") for row in rows]
    return {
        "scope": scope,
        "cases": len(rows),
        "median_G_rcsp": _median(values("G_rcsp")),
        "median_G_secdr": _median(values("G_secdr")),
        "median_delta_G_secdr_minus_rcsp": _median([float(row["G_secdr"] - row["G_rcsp"]) for row in rows]),
        "temporal_pass_count_base": sum(bool(row["temporal_pass_base"]) for row in rows),
        "temporal_pass_count_rcsp": sum(bool(row["temporal_pass_rcsp"]) for row in rows),
        "temporal_pass_count_secdr": sum(bool(row["temporal_pass_secdr"]) for row in rows),
        "endpoint_acceptance_pass_count_base": sum(bool(row["endpoint_acceptance_base"]) for row in rows),
        "endpoint_acceptance_pass_count_rcsp": sum(bool(row["endpoint_acceptance_rcsp"]) for row in rows),
        "endpoint_acceptance_pass_count_secdr": sum(bool(row["endpoint_acceptance_secdr"]) for row in rows),
        "overall_acceptance_count_base": sum(bool(row["overall_acceptance_base"]) for row in rows),
        "overall_acceptance_count_rcsp": sum(bool(row["overall_acceptance_rcsp"]) for row in rows),
        "overall_acceptance_count_secdr": sum(bool(row["overall_acceptance_secdr"]) for row in rows),
        "median_temporal_alignment_cosine_rcsp": _median(cosine_rcsp),
        "median_temporal_alignment_cosine_secdr": _median(cosine_secdr),
        "defined_temporal_alignment_cosine_cases_rcsp": sum(value is not None for value in cosine_rcsp),
        "defined_temporal_alignment_cosine_cases_secdr": sum(value is not None for value in cosine_secdr),
        "median_applied_action_norm_rcsp": _median(values("applied_action_norm_rcsp")),
        "median_applied_action_norm_secdr": _median(values("applied_action_norm_secdr")),
        "median_gain_per_action_norm_rcsp": _median(values("gain_per_action_norm_rcsp")),
        "median_gain_per_action_norm_secdr": _median(values("gain_per_action_norm_secdr")),
        "safety_non_regression_all_cases": all(bool(row["safety_non_regression"]) for row in rows),
        "binary_support_identical_all_cases": all(bool(row["binary_support_identical"]) for row in rows),
    }


def make_summaries(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(rows) != PRIMARY_CASES:
        raise ValueError("SECDR scientific summaries require exactly 32 primary rows")
    return {scope: _summary(_scope_rows(rows, scope), scope) for scope in SUMMARY_SCOPES}


def width_gap(rows: list[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    selected = _scope_rows(rows, scope)
    width10 = [row for row in selected if int(row["width"]) == 10]
    width28 = [row for row in selected if int(row["width"]) == 28]
    current = _median([row["G_rcsp"] for row in width28])
    current10 = _median([row["G_rcsp"] for row in width10])
    secdr = _median([row["G_secdr"] for row in width28])
    secdr10 = _median([row["G_secdr"] for row in width10])
    gap_current = None if current is None or current10 is None else float(current - current10)
    gap_secdr = None if secdr is None or secdr10 is None else float(secdr - secdr10)
    shrink = None if gap_current is None or gap_current == 0.0 or gap_secdr is None else 1.0 - abs(gap_secdr) / abs(gap_current)
    return {
        "scope": scope,
        "cases": len(selected),
        "median_G_rcsp_width10": current10,
        "median_G_rcsp_width28": current,
        "median_G_secdr_width10": secdr10,
        "median_G_secdr_width28": secdr,
        "gap_current": gap_current,
        "gap_secdr": gap_secdr,
        "gap_shrink_fraction": shrink,
        "major_gap_fraction": MAJOR_GAP_FRACTION,
        "gap_reduced_at_fixed_fraction": bool(gap_current is not None and gap_secdr is not None and abs(gap_secdr) <= MAJOR_GAP_FRACTION * abs(gap_current)),
    }


def _efficacy(rows: list[Mapping[str, Any]], split: str) -> dict[str, Any]:
    selected = _scope_rows(rows, split)
    gap = width_gap(rows, split)
    w10 = [row for row in selected if int(row["width"]) == 10]
    w28 = [row for row in selected if int(row["width"]) == 28]
    med_rcsp_28 = _median(row["G_rcsp"] for row in w28)
    med_secdr_28 = _median(row["G_secdr"] for row in w28)
    med_rcsp_10 = _median(row["G_rcsp"] for row in w10)
    med_secdr_10 = _median(row["G_secdr"] for row in w10)
    safety = all(bool(row["safety_non_regression"]) for row in selected)
    conditions = {
        "gap_reduced": gap["gap_reduced_at_fixed_fraction"],
        "width28_gain_strictly_improved": bool(med_secdr_28 is not None and med_rcsp_28 is not None and med_secdr_28 > med_rcsp_28),
        "width10_gain_non_degraded": bool(med_secdr_10 is not None and med_rcsp_10 is not None and med_secdr_10 >= med_rcsp_10),
        "width10_temporal_pass_non_decreased": sum(bool(row["temporal_pass_secdr"]) for row in w10) >= sum(bool(row["temporal_pass_rcsp"]) for row in w10),
        "width10_endpoint_non_decreased": sum(bool(row["endpoint_acceptance_secdr"]) for row in w10) >= sum(bool(row["endpoint_acceptance_rcsp"]) for row in w10),
        "width28_endpoint_non_decreased": sum(bool(row["endpoint_acceptance_secdr"]) for row in w28) >= sum(bool(row["endpoint_acceptance_rcsp"]) for row in w28),
        "safety_non_regression": safety,
        "original_jerk_non_regression": all(bool(row["jerk_non_regression_secdr"]) for row in selected),
    }
    return {
        "split": split,
        "cases": len(selected),
        "conditions": conditions,
        "supported": all(conditions.values()),
        "width_gap": gap,
    }


def _mechanism(rows: list[Mapping[str, Any]], split: str, rotator_norm: Mapping[str, float]) -> dict[str, Any]:
    selected = [row for row in _scope_rows(rows, split) if int(row["width"]) == 28]
    paired = [
        row for row in selected
        if row["temporal_alignment_rcsp"].get("cosine_to_negative_gradient") is not None
        and row["temporal_alignment_secdr"].get("cosine_to_negative_gradient") is not None
        and row.get("gain_per_action_norm_rcsp") is not None
        and row.get("gain_per_action_norm_secdr") is not None
    ]
    med_rcsp_cos = _median(row["temporal_alignment_rcsp"]["cosine_to_negative_gradient"] for row in paired)
    med_secdr_cos = _median(row["temporal_alignment_secdr"]["cosine_to_negative_gradient"] for row in paired)
    med_rcsp_eff = _median(row["gain_per_action_norm_rcsp"] for row in paired)
    med_secdr_eff = _median(row["gain_per_action_norm_secdr"] for row in paired)
    result = {
        "split": split,
        "width": 28,
        "cases": len(selected),
        "defined_paired_cases": len(paired),
        "median_cosine_rcsp": med_rcsp_cos,
        "median_cosine_secdr": med_secdr_cos,
        "median_gain_per_action_norm_rcsp": med_rcsp_eff,
        "median_gain_per_action_norm_secdr": med_secdr_eff,
        "cosine_strictly_improved": bool(med_secdr_cos is not None and med_rcsp_cos is not None and med_secdr_cos > med_rcsp_cos),
        "efficiency_strictly_improved": bool(med_secdr_eff is not None and med_rcsp_eff is not None and med_secdr_eff > med_rcsp_eff),
        "rotator_parameter_norm_positive": bool(rotator_norm["total"] > 0.0),
        "at_least_one_defined_rotation": any(row.get("rotation_one_minus_cosine") is not None and row["rotation_one_minus_cosine"] > 0.0 for row in selected),
        "root_joint_norm_preservation": all(
            float(row["predecoder_norm_preservation_error"]["root_max_abs_error"]) <= PARITY_ATOL
            and float(row["predecoder_norm_preservation_error"]["joint_max_abs_error"]) <= PARITY_ATOL
            for row in selected
        ),
    }
    result["supported"] = all(result[key] for key in (
        "cosine_strictly_improved",
        "efficiency_strictly_improved",
        "rotator_parameter_norm_positive",
        "at_least_one_defined_rotation",
        "root_joint_norm_preservation",
    ))
    return result


def adjudicate(efficacy: Mapping[str, Mapping[str, Any]], mechanism: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    efficacy_seen = bool(efficacy["seen"]["supported"])
    efficacy_new = bool(efficacy["new"]["supported"])
    mechanism_seen = bool(mechanism["seen"]["supported"])
    mechanism_new = bool(mechanism["new"]["supported"])
    efficacy_both = efficacy_seen and efficacy_new
    mechanism_both = mechanism_seen and mechanism_new
    if efficacy_both and mechanism_both:
        result = "WIDTH_CONDITIONED_DIRECTION_INTERVENTION_SUPPORTED"
        next_action = "freeze_secdr_candidate_and_enter_joint_evidence_synthesis"
    elif (efficacy_seen and mechanism_seen) != (efficacy_new and mechanism_new):
        result = "PARTIAL_WIDTH_CONDITIONED_DIRECTION_INTERVENTION"
        next_action = "retain_partial_direction_evidence_and_enter_joint_evidence_synthesis"
    elif mechanism_both:
        result = "DIRECTION_MECHANISM_WITHOUT_SUFFICIENT_EFFICACY"
        next_action = "reject_as_solution_and_enter_joint_evidence_synthesis"
    else:
        result = "WIDTH_CONDITIONED_DIRECTION_INTERVENTION_NOT_SUPPORTED"
        next_action = "reject_secdr_and_enter_joint_evidence_synthesis"
    return {
        "result": result,
        "next_action": next_action,
        "efficacy_supported": {"seen": efficacy_seen, "new": efficacy_new},
        "mechanism_supported": {"seen": mechanism_seen, "new": mechanism_new},
        "no_further_intervention_search": True,
        "causal_root_cause_proven": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
    }


def _load_models(lineage_paths, upstream, trajectory_report, source, state, bank, cfg, device):
    checkpoint = m._trusted_torch_load(lineage_paths["trajectory"] / "diagnostic_latest.pt", map_location="cpu")
    base = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    base.eval()
    base_hash = safe.state_hash(base.state_dict())
    if base_hash != trajectory_report["final_state_sha256"]:
        raise RuntimeError("loaded A0 base differs from immutable trajectory state")
    rcsp_model = rcsp.FrozenBaseRCSPModel(base)
    adapter_checkpoint = upstream["rcsp"]["adapter_checkpoint"]
    rcsp_model.adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"], strict=True)
    for parameter in rcsp_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    rcsp_model.eval()
    adapter_hash = safe.state_hash(rcsp_model.adapter.state_dict())
    expected_adapter_hash = upstream["rcsp"]["report"]["parameter_update_scope"]["adapter_state_sha256"]
    if adapter_hash != expected_adapter_hash:
        raise RuntimeError("loaded RCSP adapter differs from immutable Phase 2.1 adapter")
    return base, rcsp_model, base_hash, adapter_hash


def run(args: argparse.Namespace) -> int:
    phase21_path = Path(args.phase21_report).resolve()
    bctr_path = Path(args.bctr_report).resolve()
    correction_path = Path(args.bctr_correction_report).resolve()
    output = Path(args.output_dir).resolve()
    phase21_report, phase21_hash, lineage_paths, upstream = bctr._validate_phase21_lineage(phase21_path)
    if phase21_report.get("provenance", {}).get("runtime_commit") != FROZEN_PHASE21_COMMIT:
        raise ValueError("Phase 2.1 runtime commit is not frozen")
    bctr_report, bctr_hash = _validate_bctr_report(bctr_path, phase21_path, phase21_hash)
    correction_report, correction_hash = _validate_correction_report(correction_path, bctr_path, bctr_hash)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("SECDR output directory must be a fresh empty directory")
    immutable = bctr._immutable_paths(lineage_paths, phase21_path)
    immutable.update({"bctr/report.json": bctr_path, "bctr_correction/report.json": correction_path})
    if any(output == path or output.is_relative_to(path) for path in immutable.values()):
        raise FileExistsError("SECDR output overlaps immutable lineage input")
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir = output / "result"
    result_dir.mkdir(exist_ok=False)
    failure_path = result_dir / "failure.json"
    before_files = {name: _file_sha256(path) for name, path in immutable.items()}
    implementation_paths = {
        "secdR.py": Path(__file__).resolve(),
        "motion_models.py": Path(m.__file__).resolve(),
        "boundary_observables.py": Path(__import__("motion_geometry.boundary_observables", fromlist=["__name__"]).__file__).resolve(),
        "product_manifold.py": Path(__import__("motion_geometry.product_manifold", fromlist=["__name__"]).__file__).resolve(),
        "rcsp.py": Path(rcsp.__file__).resolve(),
        "alignment.py": Path(alignment.__file__).resolve(),
        "optimizer.py": Path(__file__).with_name("refiner_optimizer.py").resolve(),
    }
    implementation_before = {name: _file_sha256(path) for name, path in implementation_paths.items()}
    state = bank = cfg = source_metadata = None
    base = rcsp_model = secdr_model = None
    try:
        trajectory, trajectory_paths, trajectory_hashes, trajectory_report, _experiment, _checkpoint = failure._load_trajectory(
            lineage_paths["trajectory"], failure.TRAJECTORY_COMMIT
        )
        state, bank, cfg, source_metadata = group_audit.load_frozen_source(
            lineage_paths["source"], group_audit.LEGACY_COMMIT,
            legacy_core_strength=args.legacy_core_strength,
            legacy_transition_strength=args.legacy_transition_strength,
        )
        if _experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
            raise ValueError("trajectory does not reference the supplied frozen source")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        cfg = dataclasses.replace(cfg, device=str(device))
        with group_audit.frozen_environment(state["fingerprint"], source_metadata["decoder_strengths"]):
            train_full = group_audit.materialize_transaction(bank, cfg, 0)
            train_full = {key: value.to(device) for key, value in train_full.items()}
            train_full = attach_train_role_ids(train_full)
            train_batch = _cross_train_batch(train_full)
            calibration = calibrate_support_extent(train_full, cfg)
            base, rcsp_model, base_hash, adapter_hash = _load_models(
                lineage_paths, upstream, trajectory_report, lineage_paths["source"], state, bank, cfg, device
            )
            secdr_model = SECDRModel(rcsp_model, calibration["s_min"], calibration["s_max"]).to(device)
            zero_init = secdr_model.validate_zero_initialization()
            train_parity = _zero_parity(base, rcsp_model, secdr_model, train_batch, cfg, calibration)
            probe, probe_hash = safe.load_probe(lineage_paths["source"], state, bank, cfg)
            final_batch, final_metadata = alignment.combine_final_banks(failure.final_banks(bank, probe, cfg))
            final_batch = rcsp._move_batch(final_batch, device)
            final_batch["role_id"] = rcsp.role_ids_from_metadata(final_metadata, device)
            phase2._validate_fixed_metadata(final_metadata)
            final_parity = _zero_parity(base, rcsp_model, secdr_model, final_batch, cfg, calibration)
            training = train_rotator(secdr_model, train_batch, cfg, calibration, output)
            for parameter in secdr_model.rotator.parameters():
                parameter.grad = None
            if safe.state_hash(base.state_dict()) != base_hash:
                raise RuntimeError("SECDR training changed frozen base")
            if safe.state_hash(rcsp_model.adapter.state_dict()) != adapter_hash:
                raise RuntimeError("SECDR training changed frozen RCSP adapter")
            bctr_map = {phase2._identity_key(row): row for row in bctr_report["case_level"]}
            all_rows: list[dict[str, Any]] = []
            parity_rows: list[dict[str, Any]] = []
            for start in range(0, FINAL_CASES, rcsp.FINAL_CHUNK_SIZE):
                stop = start + rcsp.FINAL_CHUNK_SIZE
                chunk = {key: value[start:stop] for key, value in final_batch.items()}
                metadata = final_metadata[start:stop]
                rows, parity = _evaluate_chunk(
                    base, rcsp_model, secdr_model, chunk, metadata, cfg, calibration, bctr_map
                )
                all_rows.extend(rows)
                parity_rows.extend(parity)
            if len(all_rows) != FINAL_CASES:
                raise RuntimeError("SECDR fixed-final evaluation did not contain 64 cases")
            primary_rows = [row for row in all_rows if row["role"] == PRIMARY_ROLE]
            control_rows = [row for row in all_rows if row["role"] == CONTROL_ROLE]
            if len(primary_rows) != PRIMARY_CASES or len(control_rows) != PRIMARY_CASES:
                raise RuntimeError("SECDR primary/control cohort count mismatch")
            if len(parity_rows) != PRIMARY_CASES or not all(row["verified"] for row in parity_rows):
                raise RuntimeError("SECDR current metric parity failed closed")
            single_parity = all(
                row["single_recording_control"] and row["q"] == 0.0
                and row["binary_support_identical"] and row["contact_identical"]
                and row["single_output_exact"]
                for row in all_rows if row["role"] == CONTROL_ROLE
            )
            if not single_parity:
                raise RuntimeError("SECDR single-recording control parity failed")
            summaries = make_summaries(primary_rows)
            gaps = {scope: width_gap(primary_rows, scope) for scope in ("overall", "seen", "new")}
            efficacy = {split: _efficacy(primary_rows, split) for split in ("seen", "new")}
            rotator_norm = _rotator_norms(secdr_model)
            mechanism = {split: _mechanism(primary_rows, split, rotator_norm) for split in ("seen", "new")}
            decision = adjudicate(efficacy, mechanism)
            source_after = {name: _file_sha256(path) for name, path in immutable.items()}
            if before_files != source_after:
                raise RuntimeError("immutable SECDR input changed during intervention")
            if _file_sha256(lineage_paths["source"] / "probe_bank.pt") != probe_hash:
                raise RuntimeError("probe artifact changed during SECDR intervention")
            base_after = safe.state_hash(base.state_dict())
            adapter_after = safe.state_hash(rcsp_model.adapter.state_dict())
            secdr_model.validate_parameter_scope()
            if base_after != base_hash or adapter_after != adapter_hash:
                raise RuntimeError("frozen base/RCSP state changed during SECDR intervention")
            grads_none = all(parameter.grad is None for parameter in base.parameters()) and all(
                parameter.grad is None for parameter in rcsp_model.adapter.parameters()
            )
            if not grads_none:
                raise RuntimeError("frozen base/RCSP gradient residue detected")
        implementation_after = {name: _file_sha256(path) for name, path in implementation_paths.items()}
        if implementation_before != implementation_after:
            raise RuntimeError("implementation source changed during SECDR intervention")
        integrity = {
            "implementation_parent_commit": IMPLEMENTATION_PARENT_COMMIT,
            "base_state_sha256_before": base_hash,
            "base_state_sha256_after": base_after,
            "rcsp_adapter_state_sha256_before": adapter_hash,
            "rcsp_adapter_state_sha256_after": adapter_after,
            "rotator_state_sha256_after": safe.state_hash(secdr_model.rotator.state_dict()),
            "base_unchanged": base_after == base_hash,
            "rcsp_adapter_unchanged": adapter_after == adapter_hash,
            "frozen_input_artifacts_unchanged": before_files == source_after,
            "production_source_unchanged": implementation_before == implementation_after,
            "base_and_rcsp_gradients_none": grads_none,
            "only_rotator_parameters_trainable": True,
            "optimizer_parameter_scope_exact": True,
            "optimizer_steps": STEPS,
            "optimizer_constructed": True,
            "parameter_update_performed": True,
            "checkpoint_selection_performed": False,
            "scale_selection_performed": False,
            "architecture_selection_performed": False,
            "production_model_modified": False,
            "production_inference_modified": False,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
        }
        report = {
            "schema": SCHEMA,
            "completed": True,
            "provenance": {
                "runtime_commit": runtime_commit,
                "expected_main_commit": args.expected_main_commit,
                "implementation_parent_commit": IMPLEMENTATION_PARENT_COMMIT,
                "phase21_runtime_commit": FROZEN_PHASE21_COMMIT,
                "phase21_report": str(phase21_path),
                "phase21_report_sha256": phase21_hash,
                "bctr_runtime_commit": FROZEN_BCTR_COMMIT,
                "bctr_report": str(bctr_path),
                "bctr_report_sha256": bctr_hash,
                "bctr_correction_report": str(correction_path),
                "bctr_correction_report_sha256": correction_hash,
                "source": str(lineage_paths["source"]),
                "trajectory": str(trajectory),
                "rcsp_directory": str(lineage_paths["rcsp_directory"]),
                "adapter_checkpoint": str(lineage_paths["adapter_checkpoint"]),
                "phase1_report": str(lineage_paths["phase1_report"]),
                "single_decomposition_report": str(lineage_paths["single_decomposition_report"]),
                "parameter_attribution_report": str(lineage_paths["parameter_attribution_report"]),
                "immutable_input_sha256": before_files,
                "implementation_sha256_before": implementation_before,
                "implementation_sha256_after": implementation_after,
            },
            "lineage": {
                "no_latest_artifact_search": True,
                "phase21_schema": phase21_report["schema"],
                "phase21_completed": True,
                "phase21_primary_cases": PRIMARY_CASES,
                "phase21_report_path": str(phase21_path),
                "bctr_schema": bctr_report["schema"],
                "bctr_completed": True,
                "bctr_primary_cases": PRIMARY_CASES,
                "bctr_report_path": str(bctr_path),
                "bctr_correction_schema": correction.SCHEMA,
                "bctr_correction_report_path": str(correction_path),
                "adapter_checkpoint_path_read_from_phase21_lineage": str(lineage_paths["adapter_checkpoint"]),
            },
            "frozen_conclusions": {
                "single_recording_direction_alignment_bottleneck": True,
                "cross_event_width_effect_and_direction_effectiveness": True,
                "normalized_spreading_unsupported": True,
                "bctr_unsupported": True,
                "metric_search_stopped": True,
                "only_direction_intervention": True,
            },
            "intervention": {
                "name": "Support-Extent Conditioned Direction Rotation",
                "acronym": "SECDR",
                "changed_variable": "direction of the frozen RCSP projected geometric correction",
                "support_extent_conditioned": True,
                "width_conditioning_in_forward": False,
                "single_recording_bypass": True,
                "decoder_unchanged": True,
                "production_model_modified": False,
                "production_inference_modified": False,
                "bctr_recomputed": False,
                "bctr_used_for_candidate_evaluation": False,
                "production_temporal_metric_changed": False,
                "gate_threshold_changed": False,
            },
            "conditioner": {
                "source": "production _refiner_decode_masks effective root_weight/joint_weight",
                "binary_support": "weight > 0",
                "active_support_frames": "any binary root/joint geometric support > 0 per frame",
                "fraction": "s=N_active/T",
                "calibration": calibration,
                "q_formula": "clip((s-s_min)/(s_max-s_min),0,1)",
                "q_role_contract": "explicit role_id; single_recording q_effective=0; cross_event q=q(s)",
                "width_used": False,
                "support_category_used": False,
            },
            "rotator": {
                "class": "TangentDirectionRotator",
                "root_map": "bias-free 3x3",
                "joint_map": "bias-free 72x72",
                "bias": False,
                "parameter_count": ROTATOR_PARAMETER_COUNT,
                "initialization": zero_init,
                "epsilon_numerical_only": ROTATOR_EPS,
                "zero_action_behavior": "return original block when norm <= eps",
                "orthogonal_residual": "u_perp=u-a*(a^T u)/(||a||^2+eps)",
                "norm_preservation": "a_prime=v*||a||/max(||v||,eps)",
                "support_reapplied": True,
                "no_mlp_conv_attention_and_width_conditioner": True,
            },
            "training": {
                "data_source": "frozen TRAIN transaction 0 only",
                "transaction_index": 0,
                "full_transaction_cases": int(train_full["clean"].shape[0]),
                "cross_event_cases": TRAIN_EXPECTED_CROSS_CASES,
                "single_recording_used": False,
                "new_position_used": False,
                "final64_used_for_training": False,
                "trainable_scope": "rotator.root.weight and rotator.joint.weight only",
                "base_frozen": True,
                "rcsp_adapter_frozen": True,
                "decoder_frozen": True,
                "objective": "authoritative RCSP training_total and existing observable/endpoint/physical/safety components",
                "loss_weights_changed": False,
                "thresholds_changed": False,
                "direction_cosine_loss_added": False,
                "width_loss_added": False,
                "optimizer": "authoritative AdamW + checked Armijo + rollback",
                "steps": STEPS,
                "accepted_steps": training["accepted_steps"],
                "rollback_steps": training["rollback_steps"],
                "accepted_plus_rollback_equals_steps": training["accepted_steps"] + training["rollback_steps"] == STEPS,
                "attempt_accounting": training["optimizer_summary"],
                "updates_artifact": training["updates_artifact"],
            },
            "initial_parity": {
                "zero_initialized": True,
                "train_cross_event_transaction_0": train_parity,
                "fixed_final_64": final_parity,
                "current_temporal_metric_and_endpoint_metric_verified": True,
                "contact_raw_correction_projected_action_final_tangent_decoded_observable_current_temporal_endpoint": True,
            },
            "cohort": {
                "primary_cases": PRIMARY_CASES,
                "groups": {group: CASES_PER_GROUP for group in GROUP_ORDER},
                "role": PRIMARY_ROLE,
                "widths": list(WIDTHS),
                "observed_pairing": "UNPAIRED",
                "no_fake_width_pair": True,
            },
            "control": {
                "role": CONTROL_ROLE,
                "cases": len(control_rows),
                "single_recording_only": True,
                "exact_rcsp_parity": single_parity,
                "q_effective_zero": True,
                "not_in_primary_summaries": True,
            },
            "case_level": primary_rows,
            "excluded_case_level": control_rows,
            "fixed_final_64": all_rows,
            "summaries": summaries,
            "width_gap": gaps,
            "efficacy": efficacy,
            "mechanism": mechanism,
            "decision": decision,
            "state_integrity": integrity,
            "optimizer_steps": STEPS,
            "parameter_update_performed": True,
            "production_model_modified": False,
            "production_inference_modified": False,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
            "no_further_intervention_search": True,
            "causal_root_cause_proven": False,
            "next_action": decision["next_action"],
        }
        _exclusive_json(result_dir / "report.json", report)
        print(json.dumps({
            "stage": "refiner_support_extent_direction_rotation_intervention_complete",
            "report": str(result_dir / "report.json"),
            "primary_cases": PRIMARY_CASES,
            "current_metric_parity_verified": True,
            "single_recording_control_parity_verified": single_parity,
            "optimizer_steps": STEPS,
            "accepted_steps": training["accepted_steps"],
            "rollback_steps": training["rollback_steps"],
            "decision": decision["result"],
            "production_model_modified": False,
            "scientific_acceptance": False,
            "pilot_allowed": False,
        }, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(failure_path, {
                "schema": SCHEMA,
                "completed": False,
                "error": {"type": type(error).__name__, "message": str(error)},
                "optimizer_steps": 0,
                "production_model_modified": False,
                "production_inference_modified": False,
                "scientific_acceptance": False,
                "publish_allowed": False,
                "pilot_allowed": False,
            })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase21-report", required=True)
    parser.add_argument("--bctr-report", required=True)
    parser.add_argument("--bctr-correction-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
