"""RPA-LRTA: role-phase-anatomy conditioned low-rank tangent adaptation.

This is a research-method candidate layered on the frozen A0
ProductManifoldTemporalRefiner and frozen RCSP adapter.  It adds only a
conditioned 75D geometric residual before the unchanged production decoder.

The candidate deliberately does NOT modify motion generation, the production
Refiner, support/confidence masks, smoothing, taper, caps, SO(3) retraction,
contact semantics, observable metrics, gates, or loss weights.

Model:
    delta_total(t) = delta_RCSP(t) + E(p_t) * delta_RPA(t)

    E(p) = 64 p^3 (1-p)^3

For anatomy branch a in {ROOT, BODY, EXTREMITY}:
    z_a(t) = V_a h_t
    delta_a(t) = U_a [ g_a(role, p_t, duration) * z_a(t) ]

Ranks are fixed:
    ROOT=2, BODY=8, EXTREMITY=4.

The output is projected by the authoritative RCSP binary geometry support and
then passed once through the unchanged production decoder.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from motion_geometry.boundary_observables import (
    boundary_features_torch,
    boundary_metrics_torch,
)
from motion_geometry.physical import EXTREMITY_JOINTS as AUTHORITATIVE_EXTREMITY_JOINTS
from training import motion_models as m
from training import refiner_boundary_crossing_temporal_reduction_intervention as bctr
from training import refiner_cross_width_normalization_audit as phase2
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_safe_start_diagnostics as safe
from training import refiner_support_extent_direction_rotation_intervention as secdr
from training import refiner_temporal_action_alignment_audit as alignment
from training.refiner_optimizer import (
    checked_refiner_step,
    record_update,
    validate_update_summary,
)

SCHEMA = "refiner_role_phase_anatomy_low_rank_tangent_adaptation_experiment_v2"

# The patch supplied with this file is adapted to this exact latest-main base.
IMPLEMENTATION_PARENT_COMMIT = "b59fbdbf29e8b44fb9758ca7894da92cc3eb3db1"

STEPS = 400
TERMINATION_PROTOCOL = "deterministic_no_descent_fixed_point_v1"
GEOMETRY_DIM = 75
ROOT_DIM = 3
JOINT_DIM = 72

ROLE_MAPPING = dict(rcsp.ROLE_MAPPING)
ROLE_NAMES = dict(rcsp.ROLE_NAMES)

ROOT_RANK = 2
BODY_RANK = 8
EXTREMITY_RANK = 4
CONDITION_DIM = 4
CONDITION_HIDDEN = 32
GATE_DIM = ROOT_RANK + BODY_RANK + EXTREMITY_RANK

RPA_INIT_SEED = 20260903
GRADIENT_NUMERICAL_TOL = 1.0e-14
PARITY_ATOL = phase2.PARITY_ATOL
PARITY_RTOL = phase2.PARITY_RTOL

EXPECTED_TRAIN_CASES = 192
EXPECTED_CASES_PER_GROUP = 48
FINAL_CASES = 64
FINAL_GROUP_CASES = 8
WIDTHS = (10, 28)

EXTREMITY_JOINTS = tuple(int(value) for value in AUTHORITATIVE_EXTREMITY_JOINTS)
BODY_JOINTS = tuple(
    index for index in range(m.NUM_JOINTS) if index not in set(EXTREMITY_JOINTS)
)
EXPECTED_EXTREMITY_JOINTS = (7, 8, 10, 11, 20, 21, 22, 23)
EXPECTED_BODY_JOINTS = (0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)

ROOT_OUTPUT_DIM = ROOT_DIM
BODY_OUTPUT_DIM = len(BODY_JOINTS) * 3
EXTREMITY_OUTPUT_DIM = len(EXTREMITY_JOINTS) * 3

LEGACY_CORE_STRENGTH = 0.02
LEGACY_TRANSITION_STRENGTH = 1.0


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


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _median(values: Iterable[Any]) -> float | None:
    selected = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(selected)) if selected else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _tensor_max_error(
    left: torch.Tensor, right: torch.Tensor, label: str
) -> float:
    if left.shape != right.shape:
        raise ValueError(
            f"{label} shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    return _finite((left.detach() - right.detach()).abs().max(), label)


def _state_hash(module: nn.Module) -> str:
    return safe.state_hash(module.state_dict())


def validate_anatomy_partition() -> dict[str, Any]:
    if EXTREMITY_JOINTS != EXPECTED_EXTREMITY_JOINTS:
        raise RuntimeError(
            "authoritative EXTREMITY_JOINTS changed; RPA-LRTA requires review"
        )
    if BODY_JOINTS != EXPECTED_BODY_JOINTS:
        raise RuntimeError("BODY_JOINTS complement changed unexpectedly")
    if set(BODY_JOINTS) & set(EXTREMITY_JOINTS):
        raise RuntimeError("BODY and EXTREMITY anatomy partitions overlap")
    if set(BODY_JOINTS) | set(EXTREMITY_JOINTS) != set(range(m.NUM_JOINTS)):
        raise RuntimeError("BODY/EXTREMITY anatomy partition does not cover 24 joints")
    if ROOT_OUTPUT_DIM + BODY_OUTPUT_DIM + EXTREMITY_OUTPUT_DIM != GEOMETRY_DIM:
        raise RuntimeError("anatomy output dimensions do not cover 75D geometry")
    return {
        "root_translation_coordinates": [0, 1, 2],
        "body_joints": list(BODY_JOINTS),
        "extremity_joints": list(EXTREMITY_JOINTS),
        "root_dimensions": ROOT_OUTPUT_DIM,
        "body_dimensions": BODY_OUTPUT_DIM,
        "extremity_dimensions": EXTREMITY_OUTPUT_DIM,
        "total_geometry_dimensions": GEOMETRY_DIM,
        "skeleton_joint0_is_body_rotation": True,
    }


def endpoint_envelope(phase: torch.Tensor) -> torch.Tensor:
    """Endpoint-safe envelope E(p)=64*p^3*(1-p)^3.

    The phase is the authoritative normalized phase emitted by
    boundary_features_torch.  No width-specific branch is allowed.
    """
    if phase.ndim == 2:
        phase = phase.unsqueeze(-1)
    if phase.ndim != 3 or phase.shape[-1] != 1:
        raise ValueError("phase must have shape [B,T] or [B,T,1]")
    if not bool(torch.isfinite(phase).all()):
        raise FloatingPointError("nonfinite transition phase")
    if bool((phase < -1.0e-7).any()) or bool((phase > 1.0 + 1.0e-7).any()):
        raise ValueError("authoritative transition phase is outside [0,1]")
    p = phase.clamp(0.0, 1.0)
    result = 64.0 * p.pow(3) * (1.0 - p).pow(3)
    if bool((result < -1.0e-7).any()) or bool((result > 1.0 + 1.0e-6).any()):
        raise RuntimeError("RPA endpoint envelope escaped [0,1]")
    return result


def authoritative_phase_duration(
    motion: torch.Tensor,
    seam: torch.Tensor,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Reuse production phase and verify continuous duration against FK features."""
    if motion.ndim != 3 or motion.shape[-1] != m.EDGE_DIM:
        raise ValueError(f"motion must be [B,T,{m.EDGE_DIM}]")
    if seam.ndim == 2:
        seam = seam.unsqueeze(-1)
    if seam.ndim != 3 or seam.shape[:2] != motion.shape[:2]:
        raise ValueError("seam is not aligned to RPA motion")
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps must be finite and positive")

    observed = boundary_features_torch(motion, seam)
    phase = observed[..., 0:1]

    core = seam.amax(dim=-1) >= 0.5
    duration_per_case = core.sum(dim=1, keepdim=True).to(motion.dtype) / float(fps)
    duration = duration_per_case[:, None, :].expand(-1, motion.shape[1], -1)

    fk_features = m._refiner_fk_dynamics_features(motion, seam, fps)
    fk_duration = fk_features[..., -1:]
    fk_active = fk_duration != 0.0
    if bool(fk_active.any()):
        parity_error = _finite(
            (fk_duration[fk_active] - duration[fk_active]).abs().max(),
            "duration parity with existing FK dynamics",
        )
    else:
        parity_error = 0.0
    if parity_error != 0.0:
        raise RuntimeError("RPA duration differs from authoritative FK duration channel")

    return phase, duration, {
        "phase_source": "boundary_features_torch(...)[...,0:1]",
        "duration_formula": "core_frames / fps",
        "duration_fk_channel_parity_max_abs_error": parity_error,
        "width_metadata_used": False,
    }


class RolePhaseAnatomyLowRankTangentAdapter(nn.Module):
    """4692-parameter low-rank tangent adapter for hidden=256.

    The conditioner receives only:
        explicit role one-hot (2)
        normalized phase (1)
        continuous duration seconds (1)

    It never receives width=10/28 or a width category.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.root_down = nn.Linear(self.hidden_dim, ROOT_RANK, bias=False)
        self.root_up = nn.Linear(ROOT_RANK, ROOT_OUTPUT_DIM, bias=False)

        self.body_down = nn.Linear(self.hidden_dim, BODY_RANK, bias=False)
        self.body_up = nn.Linear(BODY_RANK, BODY_OUTPUT_DIM, bias=False)

        self.extremity_down = nn.Linear(
            self.hidden_dim, EXTREMITY_RANK, bias=False
        )
        self.extremity_up = nn.Linear(
            EXTREMITY_RANK, EXTREMITY_OUTPUT_DIM, bias=False
        )

        self.conditioner = nn.Sequential(
            nn.Linear(CONDITION_DIM, CONDITION_HIDDEN),
            nn.SiLU(),
            nn.Linear(CONDITION_HIDDEN, GATE_DIM),
        )
        self.reset_parameters()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_parameters(self) -> None:
        # Fixed local RNG: deterministic Kaiming without changing global RNG state.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(RPA_INIT_SEED)
        for layer in (
            self.root_down,
            self.body_down,
            self.extremity_down,
            self.conditioner[0],
        ):
            nn.init.kaiming_uniform_(
                layer.weight, a=math.sqrt(5.0), generator=generator
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for layer in (self.root_up, self.body_up, self.extremity_up):
            nn.init.zeros_(layer.weight)

        final = self.conditioner[2]
        nn.init.zeros_(final.weight)
        nn.init.ones_(final.bias)

    def _condition(
        self,
        role_id: torch.Tensor,
        phase: torch.Tensor,
        duration: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if role_id.ndim != 1 or role_id.dtype not in (torch.int32, torch.int64):
            raise ValueError("role_id must be rank-1 integer tensor")
        if bool(((role_id < 0) | (role_id > 1)).any()):
            raise ValueError("role_id must be 0=single_recording or 1=cross_event")
        if phase.ndim == 2:
            phase = phase.unsqueeze(-1)
        if duration.ndim == 2:
            duration = duration.unsqueeze(-1)
        if phase.shape != duration.shape:
            raise ValueError("phase and duration layouts differ")
        if phase.shape[0] != role_id.shape[0]:
            raise ValueError("role/phase batch size mismatch")

        role = F.one_hot(role_id.long(), num_classes=2).to(
            device=phase.device, dtype=dtype
        )
        role = role[:, None, :].expand(-1, phase.shape[1], -1)
        condition = torch.cat(
            [role, phase.to(dtype), duration.to(dtype)], dim=-1
        )
        if condition.shape[-1] != CONDITION_DIM:
            raise RuntimeError("RPA conditioner layout mismatch")
        return condition

    @staticmethod
    def _scatter_joint_blocks(
        root: torch.Tensor,
        body: torch.Tensor,
        extremity: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames = root.shape[:2]
        joints = root.new_zeros((batch, frames, m.NUM_JOINTS, 3))
        joints[..., list(BODY_JOINTS), :] = body.reshape(
            batch, frames, len(BODY_JOINTS), 3
        )
        joints[..., list(EXTREMITY_JOINTS), :] = extremity.reshape(
            batch, frames, len(EXTREMITY_JOINTS), 3
        )
        return torch.cat((root, joints.reshape(batch, frames, JOINT_DIM)), dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        role_id: torch.Tensor,
        phase: torch.Tensor,
        duration: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 3 or hidden.shape[1] != self.hidden_dim:
            raise ValueError("hidden must be [B,H,T]")
        batch, _hidden, frames = hidden.shape
        if phase.ndim == 2:
            phase = phase.unsqueeze(-1)
        if duration.ndim == 2:
            duration = duration.unsqueeze(-1)
        if phase.shape != (batch, frames, 1):
            raise ValueError("phase must align to hidden time axis")
        if duration.shape != phase.shape:
            raise ValueError("duration must align to phase")

        hidden_bt = hidden.transpose(1, 2)
        condition = self._condition(
            role_id, phase, duration, dtype=hidden_bt.dtype
        )
        gates = self.conditioner(condition)
        root_gate, body_gate, extremity_gate = torch.split(
            gates, (ROOT_RANK, BODY_RANK, EXTREMITY_RANK), dim=-1
        )

        root_latent = self.root_down(hidden_bt)
        body_latent = self.body_down(hidden_bt)
        extremity_latent = self.extremity_down(hidden_bt)

        root = self.root_up(root_gate * root_latent)
        body = self.body_up(body_gate * body_latent)
        extremity = self.extremity_up(extremity_gate * extremity_latent)

        raw = self._scatter_joint_blocks(root, body, extremity)
        if raw.shape != (batch, frames, GEOMETRY_DIM):
            raise RuntimeError("RPA output is not [B,T,75]")
        return raw, {
            "raw": raw,
            "root_raw": root,
            "body_raw": body,
            "extremity_raw": extremity,
            "root_gate": root_gate,
            "body_gate": body_gate,
            "extremity_gate": extremity_gate,
            "condition": condition,
        }

    def validate_initialization(self) -> dict[str, Any]:
        up_exact = {
            "root_up_zero": bool((self.root_up.weight.detach() == 0).all()),
            "body_up_zero": bool((self.body_up.weight.detach() == 0).all()),
            "extremity_up_zero": bool(
                (self.extremity_up.weight.detach() == 0).all()
            ),
        }
        final = self.conditioner[2]
        result = {
            **up_exact,
            "conditioner_final_weight_zero": bool(
                (final.weight.detach() == 0).all()
            ),
            "conditioner_final_bias_one": bool(
                (final.bias.detach() == 1).all()
            ),
            "parameter_count": self.parameter_count,
            "init_seed": RPA_INIT_SEED,
        }
        if not all(
            result[key]
            for key in (
                "root_up_zero",
                "body_up_zero",
                "extremity_up_zero",
                "conditioner_final_weight_zero",
                "conditioner_final_bias_one",
            )
        ):
            raise RuntimeError("RPA zero-start initialization contract failed")
        return result


class FrozenRCSPRPARefiner(nn.Module):
    """Frozen A0+RCSP plus trainable RPA residual before production decoding."""

    def __init__(self, rcsp_model: rcsp.FrozenBaseRCSPModel):
        super().__init__()
        self.rcsp = rcsp_model
        self.base = rcsp_model.base
        self.adapter = RolePhaseAnatomyLowRankTangentAdapter(
            self.base.out.in_channels
        )
        reference = next(self.base.parameters())
        self.adapter.to(device=reference.device, dtype=reference.dtype)

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.rcsp.adapter.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

        self.base.eval()
        self.rcsp.eval()
        self.out = nn.Identity()
        self._active_route = None
        self._last_details = None
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
            raise RuntimeError("no captured RPA forward details")
        return self._last_details

    def clear_last_details(self) -> None:
        self._last_details = None
        self.rcsp.clear_last_details()

    def validate_parameter_scope(self) -> dict[str, Any]:
        base = list(self.base.named_parameters())
        rcsp_parameters = list(self.rcsp.adapter.named_parameters())
        rpa_parameters = list(self.adapter.named_parameters())
        if not base or not rcsp_parameters or not rpa_parameters:
            raise RuntimeError("RPA parameter scopes are incomplete")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for _, parameter in base + rcsp_parameters
        ):
            raise RuntimeError("frozen base/RCSP is trainable or has gradient residue")
        if any(not parameter.requires_grad for _, parameter in rpa_parameters):
            raise RuntimeError("an RPA parameter is unexpectedly frozen")

        expected = expected_parameter_count(self.base.out.in_channels)
        if self.adapter.parameter_count != expected:
            raise RuntimeError(
                f"RPA parameter count mismatch: "
                f"{self.adapter.parameter_count} != {expected}"
            )
        if self.base.out.in_channels == 256 and expected != 4692:
            raise RuntimeError("hidden=256 RPA parameter budget must be 4692")
        return {
            "base_parameters": sum(p.numel() for _, p in base),
            "rcsp_parameters": sum(p.numel() for _, p in rcsp_parameters),
            "rpa_parameters": self.adapter.parameter_count,
            "hidden_dim": int(self.base.out.in_channels),
            "optimizer_parameter_scope": [
                f"adapter.{name}" for name, _ in rpa_parameters
            ],
        }

    @contextmanager
    def route(
        self,
        role_id: torch.Tensor,
        joint_weight: torch.Tensor,
        root_weight: torch.Tensor,
        *,
        capture_details: bool = False,
        mode: str = "rpa",
    ):
        if self._active_route is not None:
            raise RuntimeError("nested RPA routing is forbidden")
        if mode not in {"rcsp", "rpa"}:
            raise ValueError("mode must be rcsp or rpa")
        self._active_route = (
            role_id,
            joint_weight,
            root_weight,
            bool(capture_details),
            mode,
        )
        self._last_details = None
        try:
            yield
        finally:
            self._active_route = None

    def forward(self, x, cond, seam_mask, joint_mask):
        if self._active_route is None:
            raise RuntimeError("RPA forward requires explicit role/support route")

        role_id, joint_weight, root_weight, capture_details, mode = (
            self._active_route
        )

        # Frozen base output and hidden feature entering the production output head.
        raw_base, hidden = self.rcsp._base_forward(
            x, cond, seam_mask, joint_mask
        )
        rcsp_delta, rcsp_details = self.rcsp.adapter(
            hidden, role_id, joint_weight, root_weight
        )
        support = rcsp_details["binary_support"]

        phase, duration, duration_details = authoritative_phase_duration(
            x, seam_mask, self.base.fps
        )
        rpa_raw, rpa_details = self.adapter(
            hidden, role_id, phase, duration
        )
        rpa_projected = rpa_raw * support
        envelope = endpoint_envelope(phase)
        rpa_safe = rpa_projected * envelope

        if mode == "rcsp":
            effective_rpa = torch.zeros_like(rpa_safe)
        else:
            effective_rpa = rpa_safe

        total_delta = rcsp_delta + effective_rpa
        raw_output = torch.cat(
            (
                raw_base[..., :4],
                raw_base[..., 4:] + total_delta,
            ),
            dim=-1,
        )
        if not torch.equal(raw_output[..., :4], raw_base[..., :4]):
            raise RuntimeError("RPA changed frozen contact channels")

        # Keep the same hook layout as the production Conv1d output head.
        output = self.out(raw_output.transpose(1, 2)).transpose(1, 2)

        if capture_details:
            self._last_details = {
                "role_id": role_id.detach(),
                "raw_base": raw_base.detach(),
                "raw_rcsp": torch.cat(
                    (
                        raw_base[..., :4],
                        raw_base[..., 4:] + rcsp_delta,
                    ),
                    dim=-1,
                ).detach(),
                "raw_rpa": output.detach(),
                "rcsp_action": rcsp_delta.detach(),
                "rpa_raw": rpa_raw.detach(),
                "rpa_projected": rpa_projected.detach(),
                "rpa_safe": rpa_safe.detach(),
                "total_action_delta": total_delta.detach(),
                "binary_support": support.detach(),
                "phase": phase.detach(),
                "duration": duration.detach(),
                "envelope": envelope.detach(),
                "root_raw": rpa_details["root_raw"].detach(),
                "body_raw": rpa_details["body_raw"].detach(),
                "extremity_raw": rpa_details["extremity_raw"].detach(),
                "root_gate": rpa_details["root_gate"].detach(),
                "body_gate": rpa_details["body_gate"].detach(),
                "extremity_gate": rpa_details["extremity_gate"].detach(),
                "duration_details": duration_details,
                "mode": mode,
            }
        else:
            self._last_details = None
        return output


def expected_parameter_count(hidden_dim: int) -> int:
    hidden_dim = int(hidden_dim)
    down = (ROOT_RANK + BODY_RANK + EXTREMITY_RANK) * hidden_dim
    up = (
        ROOT_OUTPUT_DIM * ROOT_RANK
        + BODY_OUTPUT_DIM * BODY_RANK
        + EXTREMITY_OUTPUT_DIM * EXTREMITY_RANK
    )
    conditioner = (
        CONDITION_DIM * CONDITION_HIDDEN
        + CONDITION_HIDDEN
        + CONDITION_HIDDEN * GATE_DIM
        + GATE_DIM
    )
    return down + up + conditioner


def attach_train_role_ids(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return rcsp.attach_train_role_ids(batch)


def _route_values(
    batch: Mapping[str, torch.Tensor], cfg: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return rcsp._route_values(batch, cfg)


def rpa_batch_outputs(
    model: FrozenRCSPRPARefiner,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
    *,
    trace: dict[str, Any] | None = None,
    capture_details: bool = False,
    mode: str = "rpa",
):
    role_id, joint_weight, root_weight = _route_values(batch, cfg)
    with model.route(
        role_id,
        joint_weight,
        root_weight,
        capture_details=capture_details,
        mode=mode,
    ):
        return m._refiner_batch_outputs(model, batch, cfg, trace=trace)


def rpa_batch_objectives(
    model: FrozenRCSPRPARefiner,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
    *,
    capture_details: bool = False,
):
    role_id, joint_weight, root_weight = _route_values(batch, cfg)
    with model.route(
        role_id,
        joint_weight,
        root_weight,
        capture_details=capture_details,
        mode="rpa",
    ):
        return m._refiner_batch_objectives(model, batch, cfg)


def rpa_guarded_total_batch_loss(
    model: FrozenRCSPRPARefiner,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
):
    role_id, joint_weight, root_weight = _route_values(batch, cfg)
    with model.route(
        role_id,
        joint_weight,
        root_weight,
        capture_details=False,
        mode="rpa",
    ):
        return m._refiner_guarded_total_batch_loss(
            model, batch, cfg, require_all_groups=True
        )


def validate_train_contract(batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    count = int(batch["clean"].shape[0])
    if count != EXPECTED_TRAIN_CASES:
        raise ValueError(
            f"RPA requires frozen TRAIN transaction0 with 192 cases, got {count}"
        )
    if "group" not in batch or "role_id" not in batch:
        raise ValueError("RPA TRAIN batch lacks explicit group/role metadata")
    groups = {
        int(group): int((batch["group"] == group).sum().item())
        for group in range(4)
    }
    if groups != {0: 48, 1: 48, 2: 48, 3: 48}:
        raise ValueError(f"RPA TRAIN group counts differ from 48/group: {groups}")
    roles = {
        role: int((batch["role_id"] == role_id).sum().item())
        for role, role_id in ROLE_MAPPING.items()
    }
    if roles != {"single_recording": 96, "cross_event": 96}:
        raise ValueError(f"RPA TRAIN role counts are wrong: {roles}")
    return {
        "transaction_index": 0,
        "cases": count,
        "groups": groups,
        "roles": roles,
        "new_position_used": False,
        "fixed_final64_used_for_training": False,
        "width_metadata_used_in_forward": False,
    }


def zero_start_trainability_preflight(
    model: FrozenRCSPRPARefiner,
    train_batch: Mapping[str, torch.Tensor],
    cfg: Any,
) -> dict[str, Any]:
    """Read-only authoritative-objective gradient probe before AdamW."""
    model.validate_parameter_scope()
    before_hash = _state_hash(model.adapter)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.adapter.named_parameters()
    }
    for parameter in model.adapter.parameters():
        parameter.grad = None

    repair, clean, _terms, _identity_terms = rpa_batch_objectives(
        model, train_batch, cfg, capture_details=False
    )
    loss = repair + cfg.product_refiner_clean_identity_weight * clean
    loss_value = _finite(loss, "RPA zero-start preflight objective")
    backward_error = None
    if not loss.requires_grad:
        backward_error = "authoritative RPA objective does not require grad"
    else:
        loss.backward()

    named = dict(model.adapter.named_parameters())
    required = {
        "root_up": named["root_up.weight"].grad,
        "body_up": named["body_up.weight"].grad,
        "extremity_up": named["extremity_up.weight"].grad,
    }
    norms: dict[str, float | None] = {}
    all_finite = True
    for name, gradient in required.items():
        if gradient is None:
            norms[name] = 0.0
            all_finite = False
            continue
        finite = bool(torch.isfinite(gradient).all())
        all_finite = all_finite and finite
        norms[name] = (
            _finite(gradient.detach().double().norm(), f"{name} gradient norm")
            if finite
            else None
        )
    total = (
        math.sqrt(
            sum(float(value) ** 2 for value in norms.values() if value is not None)
        )
        if all_finite
        else None
    )
    any_nonzero = bool(
        total is not None and total > GRADIENT_NUMERICAL_TOL
    )
    unchanged = before_hash == _state_hash(model.adapter) and all(
        torch.equal(parameter.detach(), before[name])
        for name, parameter in model.adapter.named_parameters()
    )

    for parameter in model.adapter.parameters():
        parameter.grad = None
    cleared = all(parameter.grad is None for parameter in model.adapter.parameters())

    passed = bool(
        backward_error is None
        and all_finite
        and any_nonzero
        and unchanged
        and cleared
    )
    return {
        "scope": "TRAIN transaction 0 all 192 cases",
        "cases": int(train_batch["clean"].shape[0]),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_update_attempted": False,
        "loss": loss_value,
        "root_up_gradient_norm": norms["root_up"],
        "body_up_gradient_norm": norms["body_up"],
        "extremity_up_gradient_norm": norms["extremity_up"],
        "total_up_gradient_norm": total,
        "numerical_zero_tolerance": GRADIENT_NUMERICAL_TOL,
        "all_finite": all_finite,
        "any_gradient_nonzero": any_nonzero,
        "parameters_unchanged_after_probe": unchanged,
        "gradients_cleared_after_probe": cleared,
        "backward_error": backward_error,
        "passed": passed,
    }


def _parameter_norm(module: nn.Module) -> float:
    values = [
        parameter.detach().double().reshape(-1)
        for parameter in module.parameters()
    ]
    return _finite(torch.cat(values).norm(), "RPA parameter norm")


def _update_norm(
    module: nn.Module, before: Mapping[str, torch.Tensor]
) -> float:
    squares = [
        (
            parameter.detach().double()
            - before[name].detach().double()
        ).square().sum()
        for name, parameter in module.named_parameters()
    ]
    return math.sqrt(sum(float(value) for value in squares))



def _canonical_tree_sha256(value: Any) -> str:
    """Stable hash for optimizer/gradient/rejection trees.

    This is provenance/termination bookkeeping only.  It never changes the
    optimizer proposal, loss, acceptance gate, or model output.
    """
    digest = hashlib.sha256()

    def visit(node: Any) -> None:
        if torch.is_tensor(node):
            tensor = node.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(node, Mapping):
            digest.update(b"M")
            for key in sorted(node, key=lambda item: repr(item)):
                visit(key)
                visit(node[key])
            digest.update(b"m")
            return
        if isinstance(node, (list, tuple)):
            digest.update(b"L" if isinstance(node, list) else b"Q")
            for item in node:
                visit(item)
            digest.update(b"l")
            return
        if node is None:
            digest.update(b"N")
            return
        if isinstance(node, bool):
            digest.update(b"B1" if node else b"B0")
            return
        if isinstance(node, int):
            digest.update(b"I")
            digest.update(str(node).encode())
            return
        if isinstance(node, float):
            if not math.isfinite(node):
                raise FloatingPointError("nonfinite value in deterministic hash")
            digest.update(b"F")
            digest.update(node.hex().encode())
            return
        if isinstance(node, str):
            digest.update(b"S")
            digest.update(node.encode("utf-8"))
            return
        raise TypeError(
            f"unsupported deterministic-hash node type: {type(node).__name__}"
        )

    visit(value)
    return digest.hexdigest()


def _named_gradient_sha256(module: nn.Module) -> str:
    gradients: dict[str, torch.Tensor | None] = {}
    for name, parameter in module.named_parameters():
        gradients[name] = (
            None
            if parameter.grad is None
            else parameter.grad.detach().cpu().contiguous()
        )
    return _canonical_tree_sha256(gradients)


class DeterministicNoDescentFixedPointDetector:
    """Confirm one repeated, fully rolled-back no-descent optimizer state.

    No patience hyperparameter is used.  The first eligible rejection is only
    recorded.  The immediately following attempt is executed normally.  A
    fixed point is declared only if the complete pre-step state and complete
    rejection signature repeat exactly after the optimizer rollback.
    """

    def __init__(self):
        self.pending: dict[str, Any] | None = None

    @staticmethod
    def _eligible(
        update: Mapping[str, Any],
        *,
        adapter_rollback_exact: bool,
        optimizer_rollback_exact: bool,
    ) -> bool:
        return bool(
            not update["optimizer_update_accepted"]
            and update.get("reason") == "bounded_search_no_descent"
            and adapter_rollback_exact
            and optimizer_rollback_exact
        )

    @staticmethod
    def _payload(
        *,
        loss_before: float,
        adapter_state_sha256: str,
        optimizer_state_sha256: str,
        gradient_sha256: str,
        update: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "loss_before": float(loss_before),
            "adapter_state_sha256": adapter_state_sha256,
            "optimizer_state_sha256": optimizer_state_sha256,
            "gradient_sha256": gradient_sha256,
            "reason": update.get("reason"),
            "direction": update.get("direction"),
            "step_scale": float(update.get("step_scale", 0.0)),
            "trial_evaluations": int(update.get("trial_evaluations", 0)),
            "loss_rejected_trials": int(update.get("loss_rejected_trials", 0)),
            "nonfinite_trials": int(update.get("nonfinite_trials", 0)),
            "insufficient_decrease_trials": int(
                update.get("insufficient_decrease_trials", 0)
            ),
            "group_guard_rejected_trials": int(
                update.get("group_guard_rejected_trials", 0)
            ),
            "used_gradient_rescue": bool(
                update.get("used_gradient_rescue", False)
            ),
            "adam_directional_derivative": update.get(
                "adam_directional_derivative"
            ),
            "minimum_loss_decrease": float(
                update.get("minimum_loss_decrease", 0.0)
            ),
            "group_guard_before": update.get("group_guard_before"),
            "group_guard_last_violations": update.get(
                "group_guard_last_violations"
            ),
            # Hash the full trial list so "same rejection" cannot mean merely
            # the same summary counts.
            "trials_sha256": _canonical_tree_sha256(
                update.get("trials", [])
            ),
        }

    def observe(
        self,
        *,
        step: int,
        loss_before: float,
        adapter_state_before_sha256: str,
        adapter_state_after_sha256: str,
        optimizer_state_before_sha256: str,
        optimizer_state_after_sha256: str,
        gradient_sha256: str,
        update: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        adapter_rollback_exact = (
            adapter_state_after_sha256 == adapter_state_before_sha256
        )
        optimizer_rollback_exact = (
            optimizer_state_after_sha256 == optimizer_state_before_sha256
        )
        if not self._eligible(
            update,
            adapter_rollback_exact=adapter_rollback_exact,
            optimizer_rollback_exact=optimizer_rollback_exact,
        ):
            self.pending = None
            return None

        payload = self._payload(
            loss_before=loss_before,
            adapter_state_sha256=adapter_state_before_sha256,
            optimizer_state_sha256=optimizer_state_before_sha256,
            gradient_sha256=gradient_sha256,
            update=update,
        )
        signature = _canonical_tree_sha256(payload)
        current = {
            "step": int(step),
            "signature_sha256": signature,
            "payload": payload,
            "adapter_rollback_exact": adapter_rollback_exact,
            "optimizer_rollback_exact": optimizer_rollback_exact,
        }

        if (
            self.pending is not None
            and self.pending["step"] == int(step) - 1
            and self.pending["signature_sha256"] == signature
        ):
            result = {
                "protocol": TERMINATION_PROTOCOL,
                "confirmed": True,
                "first_rejected_step": int(self.pending["step"]),
                "confirmation_step": int(step),
                "signature_sha256": signature,
                "state_repeated_exactly": True,
                "optimizer_state_repeated_exactly": True,
                "gradient_repeated_exactly": True,
                "loss_repeated_exactly": True,
                "trial_rejection_repeated_exactly": True,
                "confirmation_attempt_executed": True,
                "patience_threshold_used": False,
                "heuristic_early_stopping": False,
                "scientific_acceptance_changed": False,
                "optimizer_acceptance_rule_changed": False,
            }
            self.pending = current
            return result

        self.pending = current
        return None


def _compact_update(update: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "protocol",
        "optimizer_update_accepted",
        "direction",
        "step_scale",
        "reason",
        "loss_before",
        "loss_after",
        "minimum_loss_decrease",
        "armijo_factor",
        "trial_evaluations",
        "loss_rejected_trials",
        "nonfinite_trials",
        "insufficient_decrease_trials",
        "used_gradient_rescue",
        "adam_directional_derivative",
        "group_guard_enabled",
        "group_guard_before",
        "group_guard_after",
        "group_guard_rejected_trials",
        "group_guard_last_violations",
        "gradient_unscale",
    )
    return {key: update.get(key) for key in keys}


def train_rpa(
    model: FrozenRCSPRPARefiner,
    train_batch: Mapping[str, torch.Tensor],
    cfg: Any,
    destination: Path,
) -> dict[str, Any]:
    """Up to 400 checked attempts, with exact deterministic fixed-point exit.

    The scientific budget remains 400.  The optimizer, Armijo rule, gradient
    rescue, subgroup guard, LR, loss, and batch are unchanged.  We stop before
    exhausting the budget only after one full confirmatory retry reproduces the
    exact same fully rolled-back ``bounded_search_no_descent`` state.
    """
    updates_path = destination / "updates.jsonl"
    updates_path.touch(exist_ok=False)
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.adapter.named_parameters()
    }
    initial_hash = _state_hash(model.adapter)

    optimizer = torch.optim.AdamW(
        model.adapter.parameters(), lr=cfg.lr, weight_decay=1.0e-4
    )
    summary: dict[str, Any] = {}
    detector = DeterministicNoDescentFixedPointDetector()
    fixed_point: dict[str, Any] | None = None
    last_accepted_step = 0
    started = time.perf_counter()
    model.train()

    for step in range(1, STEPS + 1):
        repair, clean, terms, _identity_terms = rpa_batch_objectives(
            model, train_batch, cfg, capture_details=False
        )
        loss = repair + cfg.product_refiner_clean_identity_weight * clean
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.validate_parameter_scope()

        if any(
            parameter.grad is None
            for parameter in model.adapter.parameters()
        ):
            raise RuntimeError("an RPA parameter received no gradient tensor")

        before = {
            name: parameter.detach().clone()
            for name, parameter in model.adapter.named_parameters()
        }
        clip_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.adapter.parameters(), 1.0, error_if_nonfinite=True
            )
        )

        adapter_state_before_sha256 = _state_hash(model.adapter)
        optimizer_state_before_sha256 = _canonical_tree_sha256(
            optimizer.state_dict()
        )
        gradient_sha256 = _named_gradient_sha256(model.adapter)
        loss_before = _finite(loss, "RPA training objective")

        update = checked_refiner_step(
            optimizer,
            loss,
            lambda: rpa_guarded_total_batch_loss(
                model, train_batch, cfg
            ),
            gradient_unscale=max(1.0, clip_norm + 1.0e-6),
            group_guard_before=m._refiner_group_repair_losses(
                terms, require_all=True
            ),
            group_guard_relative_tolerance=(
                cfg.product_refiner_group_guard_relative_tolerance
            ),
            group_guard_absolute_tolerance=(
                cfg.product_refiner_group_guard_absolute_tolerance
            ),
        )
        record_update(summary, update)

        displacement = _update_norm(model.adapter, before)
        if (
            not bool(update["optimizer_update_accepted"])
            and displacement != 0.0
        ):
            raise RuntimeError("rolled-back RPA step changed parameters")
        if bool(update["optimizer_update_accepted"]):
            last_accepted_step = step

        adapter_state_after_sha256 = _state_hash(model.adapter)
        optimizer_state_after_sha256 = _canonical_tree_sha256(
            optimizer.state_dict()
        )
        fixed_point = detector.observe(
            step=step,
            loss_before=loss_before,
            adapter_state_before_sha256=adapter_state_before_sha256,
            adapter_state_after_sha256=adapter_state_after_sha256,
            optimizer_state_before_sha256=optimizer_state_before_sha256,
            optimizer_state_after_sha256=optimizer_state_after_sha256,
            gradient_sha256=gradient_sha256,
            update=update,
        )

        row = {
            "step": step,
            "state_position": "after_checked_step",
            "transaction_index": 0,
            "cases": int(train_batch["clean"].shape[0]),
            "training_objective_before": loss_before,
            "optimizer": _compact_update(update),
            "accepted": bool(update["optimizer_update_accepted"]),
            "rolled_back": not bool(update["optimizer_update_accepted"]),
            "rpa_parameter_norm": _parameter_norm(model.adapter),
            "rpa_update_norm": displacement,
            "adapter_state_before_sha256": adapter_state_before_sha256,
            "adapter_state_after_sha256": adapter_state_after_sha256,
            "optimizer_state_before_sha256": optimizer_state_before_sha256,
            "optimizer_state_after_sha256": optimizer_state_after_sha256,
            "gradient_sha256": gradient_sha256,
            "deterministic_fixed_point_confirmed": bool(
                fixed_point is not None
            ),
        }
        with updates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

        should_print = (
            step in (1, 2, 5, 10, 25, 50, 100, 200, 300, 400)
            or not row["accepted"]
            or fixed_point is not None
        )
        if should_print:
            print(
                json.dumps(
                    {
                        "stage": "rpa_lrta_training",
                        "step": step,
                        "accepted": row["accepted"],
                        "objective": row["training_objective_before"],
                        "parameter_norm": row["rpa_parameter_norm"],
                        "update_norm": row["rpa_update_norm"],
                        "trial_evaluations": int(
                            update.get("trial_evaluations", 0)
                        ),
                        "reason": update.get("reason"),
                        "fixed_point_confirmed": bool(
                            fixed_point is not None
                        ),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )

        if fixed_point is not None:
            break

    actual_attempts = int(summary.get("attempted_steps", 0))
    if actual_attempts <= 0 or actual_attempts > STEPS:
        raise RuntimeError("invalid RPA attempted-step accounting")
    validate_update_summary(summary, actual_attempts)

    if actual_attempts < STEPS and fixed_point is None:
        raise RuntimeError(
            "RPA stopped before its fixed budget without a confirmed fixed point"
        )
    if fixed_point is not None:
        if actual_attempts != int(fixed_point["confirmation_step"]):
            raise RuntimeError(
                "fixed-point confirmation does not match attempted-step count"
            )
        termination_reason = "DETERMINISTIC_NO_DESCENT_FIXED_POINT"
    else:
        termination_reason = "ATTEMPT_BUDGET_EXHAUSTED"

    displacement = _update_norm(model.adapter, initial)
    final_hash = _state_hash(model.adapter)
    retained = bool(displacement > GRADIENT_NUMERICAL_TOL)
    return {
        "optimizer": "AdamW + checked_refiner_step + Armijo + rollback",
        "optimizer_constructed": True,
        "attempt_budget": STEPS,
        "optimizer_steps": actual_attempts,
        "attempted_steps": actual_attempts,
        "accepted_steps": int(summary["accepted_steps"]),
        # refiner_optimizer legacy key "retained_steps" counts rejected/rollback.
        "rollback_steps": int(summary["retained_steps"]),
        "accepted_plus_rollback_equals_attempts": (
            int(summary["accepted_steps"])
            + int(summary["retained_steps"])
            == actual_attempts
        ),
        "remaining_budget_not_executed": STEPS - actual_attempts,
        "termination_protocol": TERMINATION_PROTOCOL,
        "termination_reason": termination_reason,
        "deterministic_fixed_point": fixed_point,
        "fixed_point_confirmed": bool(fixed_point is not None),
        "last_accepted_step": int(last_accepted_step),
        "final_state_definition": (
            "state_after_last_accepted_step; subsequent confirmed no-descent "
            "attempts were fully rolled back"
            if fixed_point is not None
            else "state_after_attempt_budget_exhausted"
        ),
        "termination_is_heuristic_early_stopping": False,
        "patience_threshold_used": False,
        "optimizer_acceptance_rule_changed": False,
        "optimizer_search_rule_changed": False,
        "training_objective_changed": False,
        "parameter_update_attempted": True,
        "retained_parameter_update_performed": retained,
        "parameter_update_performed": retained,
        "parameter_displacement_norm": displacement,
        "initial_adapter_state_sha256": initial_hash,
        "final_adapter_state_sha256": final_hash,
        "final_parameter_norm": _parameter_norm(model.adapter),
        "optimizer_summary": summary,
        "updates_artifact": {
            "path": str(updates_path),
            "sha256": _file_sha256(updates_path),
            "rows": actual_attempts,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _base_outputs(base, batch, cfg, *, trace=None):
    return secdr._base_outputs(base, batch, cfg, trace=trace)


def _zero_parity(
    rcsp_model: rcsp.FrozenBaseRCSPModel,
    rpa_model: FrozenRCSPRPARefiner,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
) -> dict[str, Any]:
    rcsp_trace: dict[str, Any] = {}
    rpa_trace: dict[str, Any] = {}
    with torch.no_grad():
        rcsp_prediction, rcsp_identity = rcsp.rcsp_batch_outputs(
            rcsp_model,
            batch,
            cfg,
            trace=rcsp_trace,
            capture_details=True,
        )
        rcsp_details = rcsp_model.last_details
        rpa_prediction, rpa_identity = rpa_batch_outputs(
            rpa_model,
            batch,
            cfg,
            trace=rpa_trace,
            capture_details=True,
            mode="rpa",
        )
        rpa_details = rpa_model.last_details

    fields = {
        "raw_rcsp_vs_rpa": _tensor_max_error(
            rcsp_details["raw_adapted"],
            rpa_details["raw_rpa"],
            "zero raw RCSP/RPA",
        ),
        "rpa_residual": _finite(
            rpa_details["rpa_safe"].abs().max(),
            "zero RPA residual",
        ),
        "decoded_repair": _tensor_max_error(
            rcsp_prediction, rpa_prediction, "zero decoded repair"
        ),
        "decoded_clean": _tensor_max_error(
            rcsp_identity, rpa_identity, "zero decoded clean"
        ),
        "final_tangent": _tensor_max_error(
            rcsp_trace["repair"]["after_cap"],
            rpa_trace["repair"]["after_cap"],
            "zero final tangent",
        ),
        "contact_raw": _tensor_max_error(
            rcsp_details["raw_adapted"][..., :4],
            rpa_details["raw_rpa"][..., :4],
            "zero raw contact",
        ),
    }

    _, rcsp_terms = m._observable_refiner_objective(
        rcsp_prediction,
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )
    _, rpa_terms = m._observable_refiner_objective(
        rpa_prediction,
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )
    fields["temporal_observable"] = _tensor_max_error(
        rcsp_terms["temporal_scientific_deficit"],
        rpa_terms["temporal_scientific_deficit"],
        "zero temporal observable",
    )
    fields["endpoint_observable"] = _tensor_max_error(
        rcsp_terms["endpoint_scientific_deficit"],
        rpa_terms["endpoint_scientific_deficit"],
        "zero endpoint observable",
    )

    metric_joints = m._observable_boundary_joints_torch(
        torch.cat((rcsp_prediction, rpa_prediction))
    )
    metric = boundary_metrics_torch(
        metric_joints,
        torch.cat((batch["seam"], batch["seam"])),
        cfg.fps,
    )
    count = int(batch["clean"].shape[0])
    fields["current_temporal_metric"] = _tensor_max_error(
        metric["temporal_energy"][:count],
        metric["temporal_energy"][count:],
        "zero current temporal metric",
    )
    fields["current_endpoint_metric"] = _tensor_max_error(
        metric["endpoint_velocity_jump_mps"][:count],
        metric["endpoint_velocity_jump_mps"][count:],
        "zero current endpoint metric",
    )

    exact = all(value == 0.0 for value in fields.values())
    if not exact:
        raise RuntimeError(f"zero-initialized RPA parity failed: {fields}")

    rpa_model.clear_last_details()
    rcsp_model.clear_last_details()
    return {
        "cases": int(batch["clean"].shape[0]),
        "verified": True,
        "exact": True,
        "max_abs_errors": fields,
    }


def _anatomy_coordinate_indices() -> dict[str, tuple[int, ...]]:
    body = tuple(
        coordinate
        for joint in BODY_JOINTS
        for coordinate in range(3 + 3 * joint, 3 + 3 * joint + 3)
    )
    extremity = tuple(
        coordinate
        for joint in EXTREMITY_JOINTS
        for coordinate in range(3 + 3 * joint, 3 + 3 * joint + 3)
    )
    return {
        "root": (0, 1, 2),
        "body": body,
        "extremity": extremity,
    }


ANATOMY_COORDINATES = _anatomy_coordinate_indices()


def _cosine(action: torch.Tensor, negative_gradient: torch.Tensor) -> float | None:
    a = action.detach().double().reshape(-1)
    target = negative_gradient.detach().double().reshape(-1)
    denominator = float(a.norm() * target.norm())
    if denominator == 0.0:
        return None
    return max(
        -1.0,
        min(
            1.0,
            _finite(
                torch.dot(a, target) / denominator,
                "RPA direction cosine",
            ),
        ),
    )


def _alignment_point(
    model: FrozenRCSPRPARefiner,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
) -> dict[str, Any]:
    role_id, joint_weight, root_weight = _route_values(batch, cfg)
    with model.route(
        role_id,
        joint_weight,
        root_weight,
        capture_details=False,
        mode="rcsp",
    ):
        return alignment.production_current_point(model, batch, cfg)


def _anatomy_phase_rows(
    rpa_residual: torch.Tensor,
    gradient: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, Any]:
    """Nine descriptive anatomy×phase blocks; never an optimization input."""
    active = support.any(dim=-1)
    result: dict[str, Any] = {}
    for anatomy, coordinates in ANATOMY_COORDINATES.items():
        coordinate_index = torch.as_tensor(
            coordinates, dtype=torch.long, device=rpa_residual.device
        )
        for case_index in range(rpa_residual.shape[0]):
            identity = str(case_index)
            active_indices = torch.nonzero(
                active[case_index], as_tuple=False
            ).flatten()
            chunks = torch.tensor_split(active_indices, 3)
            for phase_name, frame_ids in zip(
                ("early", "center", "late"), chunks
            ):
                key = f"{identity}/{anatomy}/{phase_name}"
                if frame_ids.numel() == 0:
                    result[key] = {
                        "action_norm": 0.0,
                        "signed_temporal_contribution": 0.0,
                        "cosine": None,
                        "frames": 0,
                    }
                    continue
                action = rpa_residual[case_index].index_select(
                    0, frame_ids
                ).index_select(-1, coordinate_index)
                grad = gradient[case_index].index_select(
                    0, frame_ids
                ).index_select(-1, coordinate_index)
                target = -grad
                result[key] = {
                    "action_norm": _finite(
                        action.double().norm(),
                        "anatomy phase action norm",
                    ),
                    "signed_temporal_contribution": _finite(
                        (action.double() * target.double()).sum(),
                        "anatomy phase signed contribution",
                    ),
                    "cosine": _cosine(action, target),
                    "frames": int(frame_ids.numel()),
                }
    return result


def _evaluate_chunk(
    base,
    rcsp_model,
    rpa_model,
    batch,
    metadata,
    cfg,
):
    count = len(metadata)
    base_trace: dict[str, Any] = {}
    rcsp_trace: dict[str, Any] = {}
    rpa_trace: dict[str, Any] = {}

    with torch.no_grad():
        base_prediction, base_identity, _ = _base_outputs(
            base, batch, cfg, trace=base_trace
        )
        rcsp_prediction, rcsp_identity = rcsp.rcsp_batch_outputs(
            rcsp_model,
            batch,
            cfg,
            trace=rcsp_trace,
            capture_details=True,
        )
        rcsp_details = rcsp_model.last_details

        rpa_prediction, rpa_identity = rpa_batch_outputs(
            rpa_model,
            batch,
            cfg,
            trace=rpa_trace,
            capture_details=True,
            mode="rpa",
        )
        rpa_details = rpa_model.last_details

    _, base_terms = m._observable_refiner_objective(
        base_prediction,
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )
    _, rcsp_terms = m._observable_refiner_objective(
        rcsp_prediction,
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )
    _, rpa_terms = m._observable_refiner_objective(
        rpa_prediction,
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )

    point = _alignment_point(rpa_model, batch, cfg)
    gradient = point["gradients"]["temporal"][:count].detach().cpu()

    rcsp_total_action = rcsp_details["raw_adapted"][
        :count, ..., 4:
    ].detach().cpu()
    rpa_total_action = rpa_details["raw_rpa"][
        :count, ..., 4:
    ].detach().cpu()
    rpa_residual = rpa_details["rpa_safe"][:count].detach().cpu()
    support = rpa_details["binary_support"][:count].detach().cpu()
    phase = rpa_details["phase"][:count].detach().cpu()
    envelope = rpa_details["envelope"][:count].detach().cpu()
    duration = rpa_details["duration"][:count].detach().cpu()

    rcsp_alignment = alignment.alignment_stats(
        rcsp_total_action, gradient
    )
    rpa_alignment = alignment.alignment_stats(
        rpa_total_action, gradient
    )

    anatomy_phase = _anatomy_phase_rows(
        rpa_residual, gradient, support
    )

    rows = []
    anatomy = ANATOMY_COORDINATES
    for index, meta in enumerate(metadata):
        base_row = rcsp._case_row(
            meta,
            base_prediction,
            base_identity,
            batch,
            index,
            base_terms,
            cfg,
        )
        rcsp_row = rcsp._case_row(
            meta,
            rcsp_prediction,
            rcsp_identity,
            batch,
            index,
            rcsp_terms,
            cfg,
        )
        rpa_row = rcsp._case_row(
            meta,
            rpa_prediction,
            rpa_identity,
            batch,
            index,
            rpa_terms,
            cfg,
        )

        residual = rpa_residual[index]
        total_residual_norm = _finite(
            residual.double().reshape(-1).norm(),
            "RPA residual norm",
        )

        anatomy_norms = {}
        for name, coords in anatomy.items():
            coord = torch.as_tensor(coords, dtype=torch.long)
            value = residual.index_select(-1, coord)
            anatomy_norms[name] = _finite(
                value.double().reshape(-1).norm(),
                f"{name} RPA residual norm",
            )
        anatomy_fractions = {
            name: _ratio(value, total_residual_norm)
            for name, value in anatomy_norms.items()
        }

        base_tangent = base_trace["repair"]["after_cap"][index]
        rcsp_tangent = rcsp_trace["repair"]["after_cap"][index]
        rpa_tangent = rpa_trace["repair"]["after_cap"][index]
        rcsp_applied_norm = _finite(
            (rcsp_tangent - base_tangent).double().reshape(-1).norm(),
            "RCSP applied action norm",
        )
        rpa_applied_norm = _finite(
            (rpa_tangent - base_tangent).double().reshape(-1).norm(),
            "RPA applied action norm",
        )

        projected_outside = (
            rpa_residual[index] * (1.0 - support[index])
        )
        projected_outside_max = _finite(
            projected_outside.abs().max(),
            "RPA outside-support residual",
        )
        contact_raw_identical = bool(
            torch.equal(
                rcsp_details["raw_adapted"][index, ..., :4],
                rpa_details["raw_rpa"][index, ..., :4],
            )
        )
        contact_decoded_identical = bool(
            torch.equal(
                rcsp_prediction[index, ..., :4],
                rpa_prediction[index, ..., :4],
            )
        )

        temporal_regression = bool(
            rcsp_row["temporal_gate_pass"]
            and not rpa_row["temporal_gate_pass"]
        )
        endpoint_regression = bool(
            rcsp_row["endpoint_gate_pass"]
            and not rpa_row["endpoint_gate_pass"]
        )
        physical_regression = bool(
            rcsp_row["physical_pass"] and not rpa_row["physical_pass"]
        )
        geometry_regression = bool(
            rcsp_row["geometry_pass"] and not rpa_row["geometry_pass"]
        )
        clean_regression = bool(
            rcsp_row["clean_pass"] and not rpa_row["clean_pass"]
        )

        phase_case = phase[index, ..., 0]
        envelope_case = envelope[index, ..., 0]
        active_phase = phase_case[support[index].any(dim=-1)]
        row_phase_min = (
            _finite(active_phase.min(), "active phase min")
            if active_phase.numel()
            else None
        )
        row_phase_max = (
            _finite(active_phase.max(), "active phase max")
            if active_phase.numel()
            else None
        )

        block_rows = {}
        prefix = f"{index}/"
        for key, value in anatomy_phase.items():
            if key.startswith(prefix):
                block_rows[key[len(prefix):]] = value

        rows.append(
            {
                **meta,
                "identity": phase2._identity_key(meta),
                "BASE": base_row,
                "RCSP": rcsp_row,
                "RPA_LRTA": rpa_row,
                "temporal_newly_rescued_vs_rcsp": bool(
                    not rcsp_row["temporal_gate_pass"]
                    and rpa_row["temporal_gate_pass"]
                ),
                "temporal_regression_vs_rcsp": temporal_regression,
                "endpoint_newly_rescued_vs_rcsp": bool(
                    not rcsp_row["endpoint_gate_pass"]
                    and rpa_row["endpoint_gate_pass"]
                ),
                "endpoint_regression_vs_rcsp": endpoint_regression,
                "physical_regression_vs_rcsp": physical_regression,
                "geometry_regression_vs_rcsp": geometry_regression,
                "clean_identity_regression_vs_rcsp": clean_regression,
                "support_regression": projected_outside_max != 0.0,
                "contact_regression": not (
                    contact_raw_identical
                    and contact_decoded_identical
                ),
                "contact_raw_identical": contact_raw_identical,
                "contact_decoded_identical": contact_decoded_identical,
                "rpa_residual_outside_support_max": projected_outside_max,
                "rpa_residual_norm": total_residual_norm,
                "rpa_anatomy_norms": anatomy_norms,
                "rpa_anatomy_fractions": anatomy_fractions,
                "applied_action_norm_rcsp": rcsp_applied_norm,
                "applied_action_norm_rpa": rpa_applied_norm,
                "temporal_alignment_rcsp": rcsp_alignment[index],
                "temporal_alignment_rpa": rpa_alignment[index],
                "phase_min_active": row_phase_min,
                "phase_max_active": row_phase_max,
                "envelope_min": _finite(
                    envelope_case.min(), "envelope min"
                ),
                "envelope_max": _finite(
                    envelope_case.max(), "envelope max"
                ),
                "duration_seconds": _finite(
                    duration[index, 0, 0], "duration seconds"
                ),
                "anatomy_phase": block_rows,
            }
        )

    rcsp_model.clear_last_details()
    rpa_model.clear_last_details()
    return rows


def _scope_rows(
    rows: list[Mapping[str, Any]], scope: str
) -> list[Mapping[str, Any]]:
    if scope == "overall":
        return list(rows)
    if scope == "single_recording":
        return [row for row in rows if row["role"] == "single_recording"]
    if scope == "cross_event":
        return [row for row in rows if row["role"] == "cross_event"]
    if scope == "width10":
        return [row for row in rows if int(row["width"]) == 10]
    if scope == "width28":
        return [row for row in rows if int(row["width"]) == 28]
    if scope == "seen":
        return [row for row in rows if row["split"] == "seen"]
    if scope == "new_position":
        return [row for row in rows if row["split"] == "new_position"]

    parts = scope.split("/")
    selected = list(rows)
    for part in parts:
        if part in ("seen", "new_position"):
            selected = [row for row in selected if row["split"] == part]
        elif part in ("single_recording", "cross_event"):
            selected = [row for row in selected if row["role"] == part]
        elif part in ("10", "28"):
            selected = [row for row in selected if int(row["width"]) == int(part)]
        else:
            raise ValueError(f"unknown summary scope component: {part}")
    return selected


SUMMARY_SCOPES = (
    "overall",
    "single_recording",
    "cross_event",
    "width10",
    "width28",
    "seen",
    "new_position",
    "seen/single_recording/10",
    "seen/single_recording/28",
    "new_position/single_recording/10",
    "new_position/single_recording/28",
    "seen/cross_event/10",
    "seen/cross_event/28",
    "new_position/cross_event/10",
    "new_position/cross_event/28",
    "single_recording/10",
    "single_recording/28",
    "cross_event/10",
    "cross_event/28",
)


def _summary(rows: list[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"empty RPA summary scope: {scope}")

    def state_count(state: str, key: str) -> int:
        return sum(bool(row[state][key]) for row in rows)

    rcsp_cos = [
        row["temporal_alignment_rcsp"].get("cosine_to_negative_gradient")
        for row in rows
    ]
    rpa_cos = [
        row["temporal_alignment_rpa"].get("cosine_to_negative_gradient")
        for row in rows
    ]
    return {
        "scope": scope,
        "cases": len(rows),
        "BASE_temporal_pass": state_count("BASE", "temporal_gate_pass"),
        "RCSP_temporal_pass": state_count("RCSP", "temporal_gate_pass"),
        "RPA_temporal_pass": state_count("RPA_LRTA", "temporal_gate_pass"),
        "BASE_endpoint_pass": state_count("BASE", "endpoint_gate_pass"),
        "RCSP_endpoint_pass": state_count("RCSP", "endpoint_gate_pass"),
        "RPA_endpoint_pass": state_count("RPA_LRTA", "endpoint_gate_pass"),
        "temporal_newly_rescued_vs_rcsp": sum(
            bool(row["temporal_newly_rescued_vs_rcsp"]) for row in rows
        ),
        "temporal_regressions_vs_rcsp": sum(
            bool(row["temporal_regression_vs_rcsp"]) for row in rows
        ),
        "endpoint_newly_rescued_vs_rcsp": sum(
            bool(row["endpoint_newly_rescued_vs_rcsp"]) for row in rows
        ),
        "endpoint_regressions_vs_rcsp": sum(
            bool(row["endpoint_regression_vs_rcsp"]) for row in rows
        ),
        "median_temporal_deficit_BASE": _median(
            row["BASE"]["temporal_scientific_deficit"] for row in rows
        ),
        "median_temporal_deficit_RCSP": _median(
            row["RCSP"]["temporal_scientific_deficit"] for row in rows
        ),
        "median_temporal_deficit_RPA": _median(
            row["RPA_LRTA"]["temporal_scientific_deficit"] for row in rows
        ),
        "median_endpoint_deficit_BASE": _median(
            row["BASE"]["endpoint_scientific_deficit"] for row in rows
        ),
        "median_endpoint_deficit_RCSP": _median(
            row["RCSP"]["endpoint_scientific_deficit"] for row in rows
        ),
        "median_endpoint_deficit_RPA": _median(
            row["RPA_LRTA"]["endpoint_scientific_deficit"] for row in rows
        ),
        "median_applied_action_norm_RCSP": _median(
            row["applied_action_norm_rcsp"] for row in rows
        ),
        "median_applied_action_norm_RPA": _median(
            row["applied_action_norm_rpa"] for row in rows
        ),
        "median_rpa_residual_norm": _median(
            row["rpa_residual_norm"] for row in rows
        ),
        "median_direction_cosine_RCSP": _median(rcsp_cos),
        "median_direction_cosine_RPA": _median(rpa_cos),
        "defined_direction_cosines_RCSP": sum(value is not None for value in rcsp_cos),
        "defined_direction_cosines_RPA": sum(value is not None for value in rpa_cos),
        "physical_regressions": sum(
            bool(row["physical_regression_vs_rcsp"]) for row in rows
        ),
        "geometry_regressions": sum(
            bool(row["geometry_regression_vs_rcsp"]) for row in rows
        ),
        "clean_identity_regressions": sum(
            bool(row["clean_identity_regression_vs_rcsp"]) for row in rows
        ),
        "support_regressions": sum(
            bool(row["support_regression"]) for row in rows
        ),
        "contact_regressions": sum(
            bool(row["contact_regression"]) for row in rows
        ),
    }


def make_summaries(
    rows: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(rows) != FINAL_CASES:
        raise ValueError("RPA summaries require exactly final64")
    return {
        scope: _summary(_scope_rows(rows, scope), scope)
        for scope in SUMMARY_SCOPES
    }


def adjudicate(
    rows: list[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    seen_single = summaries["seen/single_recording/10"]["cases"] + summaries[
        "seen/single_recording/28"
    ]["cases"]
    new_single = summaries["new_position/single_recording/10"]["cases"] + summaries[
        "new_position/single_recording/28"
    ]["cases"]
    if seen_single != 16 or new_single != 16:
        raise RuntimeError("single-recording decision cohorts must be 16+16")

    seen_single_rows = _scope_rows(rows, "seen/single_recording")
    new_single_rows = _scope_rows(rows, "new_position/single_recording")
    seen_cross28_rows = _scope_rows(rows, "seen/cross_event/28")
    new_cross28_rows = _scope_rows(rows, "new_position/cross_event/28")
    all_single = _scope_rows(rows, "single_recording")
    all_cross28 = _scope_rows(rows, "cross_event/28")

    def temporal_pass(rows_, state):
        return sum(bool(row[state]["temporal_gate_pass"]) for row in rows_)

    def median_deficit(rows_, state):
        return _median(
            row[state]["temporal_scientific_deficit"] for row in rows_
        )

    def median_cosine(rows_, field):
        return _median(
            row[field].get("cosine_to_negative_gradient") for row in rows_
        )

    conditions = {
        "A_single_seen_rescue": (
            temporal_pass(seen_single_rows, "RPA_LRTA")
            > temporal_pass(seen_single_rows, "RCSP")
        ),
        "B_single_new_rescue": (
            temporal_pass(new_single_rows, "RPA_LRTA")
            > temporal_pass(new_single_rows, "RCSP")
        ),
        "C_cross28_seen_effectiveness": (
            median_deficit(seen_cross28_rows, "RPA_LRTA")
            < median_deficit(seen_cross28_rows, "RCSP")
        ),
        "D_cross28_new_effectiveness": (
            median_deficit(new_cross28_rows, "RPA_LRTA")
            < median_deficit(new_cross28_rows, "RCSP")
        ),
        "E_no_temporal_regression": not any(
            bool(row["temporal_regression_vs_rcsp"]) for row in rows
        ),
        "F_no_endpoint_regression": not any(
            bool(row["endpoint_regression_vs_rcsp"]) for row in rows
        ),
        "G_no_safety_regression": not any(
            bool(row[field])
            for row in rows
            for field in (
                "physical_regression_vs_rcsp",
                "geometry_regression_vs_rcsp",
                "clean_identity_regression_vs_rcsp",
                "support_regression",
                "contact_regression",
            )
        ),
        "H_single_direction_improved": (
            median_cosine(all_single, "temporal_alignment_rpa")
            is not None
            and median_cosine(all_single, "temporal_alignment_rcsp")
            is not None
            and median_cosine(all_single, "temporal_alignment_rpa")
            > median_cosine(all_single, "temporal_alignment_rcsp")
        ),
        "I_cross28_direction_improved": (
            median_cosine(all_cross28, "temporal_alignment_rpa")
            is not None
            and median_cosine(all_cross28, "temporal_alignment_rcsp")
            is not None
            and median_cosine(all_cross28, "temporal_alignment_rpa")
            > median_cosine(all_cross28, "temporal_alignment_rcsp")
        ),
    }

    total_rcsp_pass = temporal_pass(rows, "RCSP")
    total_rpa_pass = temporal_pass(rows, "RPA_LRTA")
    total_rescues = sum(
        bool(row["temporal_newly_rescued_vs_rcsp"]) for row in rows
    )
    net_gate_improvement = bool(
        total_rpa_pass > total_rcsp_pass
        and conditions["E_no_temporal_regression"]
    )

    if all(conditions.values()):
        result = "RPA_LRTA_CANDIDATE_ADVANCE_REVIEW"
        next_action = "request_separate_pilot_authorization"
    elif conditions["G_no_safety_regression"] and net_gate_improvement:
        result = "RPA_LRTA_PARTIAL_DIAGNOSTIC_SUCCESS"
        next_action = "freeze_rpa_lrta_result_and_review_evidence"
    elif (
        conditions["H_single_direction_improved"]
        or conditions["I_cross28_direction_improved"]
    ) and total_rescues == 0:
        result = "RPA_LRTA_MECHANISM_ONLY"
        next_action = "freeze_rpa_lrta_result_and_review_evidence"
    else:
        result = "RPA_LRTA_NOT_SUPPORTED"
        next_action = (
            "reject_rpa_lrta_candidate_without_additional_architecture_search"
        )

    return {
        "result": result,
        "conditions": conditions,
        "total_temporal_pass_RCSP": total_rcsp_pass,
        "total_temporal_pass_RPA": total_rpa_pass,
        "total_temporal_newly_rescued_vs_RCSP": total_rescues,
        "net_gate_improvement": net_gate_improvement,
        "safety_regression_blocks_advance": True,
        "next_action": next_action,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
    }


def _load_models(
    lineage_paths,
    upstream,
    trajectory_report,
    source,
    state,
    bank,
    cfg,
    device,
):
    return secdr._load_models(
        lineage_paths,
        upstream,
        trajectory_report,
        source,
        state,
        bank,
        cfg,
        device,
    )


def run(args: argparse.Namespace) -> int:
    validate_anatomy_partition()

    phase21_path = Path(args.phase21_report).resolve()
    bctr_path = Path(args.bctr_report).resolve()
    output = Path(args.output_dir).resolve()

    _phase21_report, phase21_hash, lineage_paths, upstream = (
        bctr._validate_phase21_lineage(phase21_path)
    )
    _bctr_report, bctr_hash = secdr._validate_bctr_report(
        bctr_path, phase21_path, phase21_hash
    )

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        raise FileExistsError(
            "RPA output directory must be a fresh empty directory"
        )

    immutable = bctr._immutable_paths(lineage_paths, phase21_path)
    immutable["bctr/report.json"] = bctr_path
    if any(
        output == path or output.is_relative_to(path)
        for path in immutable.values()
    ):
        raise FileExistsError("RPA output overlaps frozen lineage input")

    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError(
            "runtime commit does not match --expected-main-commit"
        )

    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir = output / "result"
    result_dir.mkdir(exist_ok=False)
    failure_path = result_dir / "failure.json"

    immutable_before = {
        name: _file_sha256(path) for name, path in immutable.items()
    }

    implementation_paths = {
        "rpa_lrta.py": Path(__file__).resolve(),
        "motion_models.py": Path(m.__file__).resolve(),
        "boundary_observables.py": Path(
            __import__(
                "motion_geometry.boundary_observables",
                fromlist=["__name__"],
            ).__file__
        ).resolve(),
        "product_manifold.py": Path(
            __import__(
                "motion_geometry.product_manifold",
                fromlist=["__name__"],
            ).__file__
        ).resolve(),
        "physical.py": Path(
            __import__(
                "motion_geometry.physical",
                fromlist=["__name__"],
            ).__file__
        ).resolve(),
        "rcsp.py": Path(rcsp.__file__).resolve(),
        "alignment.py": Path(alignment.__file__).resolve(),
        "optimizer.py": Path(__file__).with_name(
            "refiner_optimizer.py"
        ).resolve(),
    }
    implementation_before = {
        name: _file_sha256(path)
        for name, path in implementation_paths.items()
    }

    zero_start = None
    try:
        (
            trajectory,
            _trajectory_paths,
            _trajectory_hashes,
            trajectory_report,
            experiment,
            _checkpoint,
        ) = failure._load_trajectory(
            lineage_paths["trajectory"],
            failure.TRAJECTORY_COMMIT,
        )
        state, bank, cfg, source_metadata = (
            group_audit.load_frozen_source(
                lineage_paths["source"],
                group_audit.LEGACY_COMMIT,
                legacy_core_strength=args.legacy_core_strength,
                legacy_transition_strength=args.legacy_transition_strength,
            )
        )
        if (
            experiment.get("source", {}).get("source_sha256")
            != source_metadata["source_sha256"]
        ):
            raise ValueError(
                "trajectory does not reference supplied frozen source"
            )

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; no silent CPU fallback"
            )
        cfg = dataclasses.replace(cfg, device=str(device))

        with group_audit.frozen_environment(
            state["fingerprint"],
            source_metadata["decoder_strengths"],
        ):
            train_batch = group_audit.materialize_transaction(
                bank, cfg, 0
            )
            train_batch = {
                key: value.to(device)
                for key, value in train_batch.items()
            }
            train_batch = attach_train_role_ids(train_batch)
            train_contract = validate_train_contract(train_batch)

            (
                base,
                rcsp_model,
                base_hash,
                rcsp_hash,
            ) = _load_models(
                lineage_paths,
                upstream,
                trajectory_report,
                lineage_paths["source"],
                state,
                bank,
                cfg,
                device,
            )

            rpa_model = FrozenRCSPRPARefiner(rcsp_model).to(device)
            architecture = {
                "hidden_dim": int(base.out.in_channels),
                "parameter_count": rpa_model.adapter.parameter_count,
                "expected_parameter_count": expected_parameter_count(
                    base.out.in_channels
                ),
                "root_rank": ROOT_RANK,
                "body_rank": BODY_RANK,
                "extremity_rank": EXTREMITY_RANK,
                "condition_dim": CONDITION_DIM,
                "condition_hidden": CONDITION_HIDDEN,
                "gate_dim": GATE_DIM,
                "zero_start": rpa_model.adapter.validate_initialization(),
            }
            if (
                base.out.in_channels == 256
                and architecture["parameter_count"] != 4692
            ):
                raise RuntimeError(
                    "hidden=256 RPA must contain exactly 4692 parameters"
                )

            train_parity = _zero_parity(
                rcsp_model, rpa_model, train_batch, cfg
            )
            zero_start = zero_start_trainability_preflight(
                rpa_model, train_batch, cfg
            )
            if not zero_start["passed"]:
                raise RuntimeError(
                    "RPA zero-start trainability preflight failed closed"
                )

            probe, probe_hash = safe.load_probe(
                lineage_paths["source"], state, bank, cfg
            )
            final_batch, final_metadata = alignment.combine_final_banks(
                failure.final_banks(bank, probe, cfg)
            )
            final_batch = rcsp._move_batch(final_batch, device)
            final_batch["role_id"] = rcsp.role_ids_from_metadata(
                final_metadata, device
            )
            phase2._validate_fixed_metadata(final_metadata)
            if (
                len(final_metadata) != FINAL_CASES
                or int(final_batch["clean"].shape[0]) != FINAL_CASES
            ):
                raise RuntimeError("RPA final evaluation is not final64")

            final_parity = _zero_parity(
                rcsp_model, rpa_model, final_batch, cfg
            )

            adapter_initial_hash = _state_hash(rpa_model.adapter)
            training = train_rpa(
                rpa_model, train_batch, cfg, result_dir
            )

            for parameter in rpa_model.adapter.parameters():
                parameter.grad = None

            if _state_hash(base) != base_hash:
                raise RuntimeError("RPA training changed frozen base")
            if _state_hash(rcsp_model.adapter) != rcsp_hash:
                raise RuntimeError("RPA training changed frozen RCSP")

            all_rows: list[dict[str, Any]] = []
            for start in range(
                0, FINAL_CASES, rcsp.FINAL_CHUNK_SIZE
            ):
                stop = start + rcsp.FINAL_CHUNK_SIZE
                chunk = {
                    key: value[start:stop]
                    for key, value in final_batch.items()
                }
                metadata = final_metadata[start:stop]
                all_rows.extend(
                    _evaluate_chunk(
                        base,
                        rcsp_model,
                        rpa_model,
                        chunk,
                        metadata,
                        cfg,
                    )
                )

            if len(all_rows) != FINAL_CASES:
                raise RuntimeError(
                    "RPA evaluation did not produce exactly 64 rows"
                )

            # Exact final cohort: eight split/role/width groups × eight.
            group_counts = {}
            for split in ("seen", "new_position"):
                for role in ("single_recording", "cross_event"):
                    for width in WIDTHS:
                        name = f"{split}/{role}/{width}"
                        count = sum(
                            row["split"] == split
                            and row["role"] == role
                            and int(row["width"]) == width
                            for row in all_rows
                        )
                        group_counts[name] = count
                        if count != FINAL_GROUP_CASES:
                            raise RuntimeError(
                                f"RPA final group {name} has {count}, not 8"
                            )

            summaries = make_summaries(all_rows)
            decision = adjudicate(all_rows, summaries)

            # Fixed final adapter state from the frozen budget/fixed-point contract; this is not checkpoint selection.
            adapter_path = result_dir / "rpa_adapter_final.pt"
            if adapter_path.exists():
                raise FileExistsError(
                    "RPA adapter artifact already exists"
                )
            torch.save(
                {
                    "schema": SCHEMA,
                    "step": int(training["attempted_steps"]),
                    "attempt_budget": STEPS,
                    "termination_reason": training["termination_reason"],
                    "runtime_commit": runtime_commit,
                    "parent_commit": IMPLEMENTATION_PARENT_COMMIT,
                    "adapter_state_dict": {
                        key: value.detach().cpu()
                        for key, value in rpa_model.adapter.state_dict().items()
                    },
                    "parameter_count": rpa_model.adapter.parameter_count,
                    "fixed_final_state_not_selected": True,
                },
                adapter_path,
            )

            immutable_after = {
                name: _file_sha256(path)
                for name, path in immutable.items()
            }
            if immutable_before != immutable_after:
                raise RuntimeError(
                    "a frozen RPA input artifact changed during run"
                )
            if (
                _file_sha256(
                    lineage_paths["source"] / "probe_bank.pt"
                )
                != probe_hash
            ):
                raise RuntimeError("probe bank changed during RPA run")

            implementation_after = {
                name: _file_sha256(path)
                for name, path in implementation_paths.items()
            }
            if implementation_before != implementation_after:
                raise RuntimeError(
                    "implementation source changed during RPA run"
                )

            base_after = _state_hash(base)
            rcsp_after = _state_hash(rcsp_model.adapter)
            rpa_after = _state_hash(rpa_model.adapter)
            base_rcsp_grads_none = all(
                parameter.grad is None
                for parameter in base.parameters()
            ) and all(
                parameter.grad is None
                for parameter in rcsp_model.adapter.parameters()
            )
            if not base_rcsp_grads_none:
                raise RuntimeError(
                    "frozen base/RCSP contains gradient residue"
                )

            mechanism_summary = {
                "single_direction_RCSP": summaries[
                    "single_recording"
                ]["median_direction_cosine_RCSP"],
                "single_direction_RPA": summaries[
                    "single_recording"
                ]["median_direction_cosine_RPA"],
                "cross28_direction_RCSP": summaries[
                    "cross_event/28"
                ]["median_direction_cosine_RCSP"],
                "cross28_direction_RPA": summaries[
                    "cross_event/28"
                ]["median_direction_cosine_RPA"],
                "single_direction_improved": decision[
                    "conditions"
                ]["H_single_direction_improved"],
                "cross28_direction_improved": decision[
                    "conditions"
                ]["I_cross28_direction_improved"],
                "anatomy_phase_is_descriptive_only": True,
            }

            report = {
                "schema": SCHEMA,
                "completed": True,
                "provenance": {
                    "runtime_commit": runtime_commit,
                    "expected_main_commit": args.expected_main_commit,
                    "parent_commit": IMPLEMENTATION_PARENT_COMMIT,
                    "phase21_source": str(phase21_path),
                    "phase21_sha256": phase21_hash,
                    "bctr_source": str(bctr_path),
                    "bctr_sha256": bctr_hash,
                    "source": str(lineage_paths["source"]),
                    "trajectory": str(trajectory),
                    "rcsp_report": str(
                        lineage_paths["rcsp_directory"] / "report.json"
                    ),
                    "rcsp_adapter_checkpoint": str(
                        lineage_paths["adapter_checkpoint"]
                    ),
                    "phase1_report": str(
                        lineage_paths["phase1_report"]
                    ),
                    "single_decomposition_report": str(
                        lineage_paths["single_decomposition_report"]
                    ),
                    "parameter_attribution_report": str(
                        lineage_paths["parameter_attribution_report"]
                    ),
                    "immutable_input_sha256": immutable_before,
                    "implementation_sha256_before": implementation_before,
                    "implementation_sha256_after": implementation_after,
                    "no_latest_artifact_search": True,
                },
                "experiment_scope": {
                    "type": "new_research_method_candidate",
                    "method": (
                        "Role-Phase-Anatomy Conditioned "
                        "Low-Rank Tangent Adaptation"
                    ),
                    "acronym": "RPA-LRTA",
                    "prior_experiment_decisions_modified": False,
                    "prior_scientific_classification_reinterpreted": False,
                    "generator_redesigned": False,
                    "implementation_correction": {
                        "type": "DETERMINISTIC_NO_DESCENT_FIXED_POINT",
                        "parent_runtime_commit": IMPLEMENTATION_PARENT_COMMIT,
                        "scientific_method_changed": False,
                        "architecture_changed": False,
                        "objective_changed": False,
                        "optimizer_acceptance_changed": False,
                        "optimizer_search_changed": False,
                        "learning_rate_changed": False,
                        "rank_changed": False,
                        "case_cohort_changed": False,
                    },
                },
                "architecture": architecture,
                "role_definition": {
                    "mapping": ROLE_MAPPING,
                    "condition": "explicit one-hot only",
                    "width_used_to_infer_role": False,
                },
                "phase_definition": {
                    "source": "boundary_features_torch(...)[...,0:1]",
                    "continuous": True,
                    "early_center_late_used_in_forward": False,
                },
                "duration_definition": {
                    "formula": "core_frames/fps",
                    "core": "seam>=0.5",
                    "continuous_seconds": True,
                    "parity_with_existing_fk_duration": True,
                    "width_metadata_used_in_forward": False,
                },
                "anatomy_partition": validate_anatomy_partition(),
                "rank_definition": {
                    "root": ROOT_RANK,
                    "body": BODY_RANK,
                    "extremity": EXTREMITY_RANK,
                    "rank_search_performed": False,
                },
                "conditioner_definition": {
                    "input": [
                        "role_onehot_0",
                        "role_onehot_1",
                        "phase",
                        "duration_seconds",
                    ],
                    "architecture": "4->32->14",
                    "activation": "SiLU",
                    "root_gate": ROOT_RANK,
                    "body_gate": BODY_RANK,
                    "extremity_gate": EXTREMITY_RANK,
                    "width_conditioning": False,
                },
                "envelope_definition": {
                    "formula": "64*p^3*(1-p)^3",
                    "applied_only_to_new_rpa_residual": True,
                    "rcsp_residual_rescaled": False,
                    "endpoint_guaranteed_by_envelope": False,
                },
                "training": {
                    **train_contract,
                    **training,
                    "objective": (
                        "unchanged authoritative RCSP training_total/"
                        "observable/endpoint/physical/clean terms"
                    ),
                    "loss_weights_changed": False,
                    "thresholds_changed": False,
                    "optimizer_scope": "RPA adapter only",
                    "base_frozen": True,
                    "rcsp_frozen": True,
                    "checkpoint_selection_performed": False,
                    "alpha_search_performed": False,
                    "architecture_search_performed": False,
                    "rank_search_performed": False,
                },
                "zero_start_trainability_preflight": zero_start,
                "initial_parity": {
                    "train_transaction_0": train_parity,
                    "fixed_final_64": final_parity,
                },
                "fixed_final_case_count": FINAL_CASES,
                "fixed_final_groups": group_counts,
                "case_level": all_rows,
                "summaries": summaries,
                "mechanism_summary": mechanism_summary,
                "decision": decision,
                "artifacts": {
                    "rpa_adapter_final": {
                        "path": str(adapter_path),
                        "sha256": _file_sha256(adapter_path),
                        "selected_checkpoint": False,
                    },
                    "updates": training["updates_artifact"],
                },
                "state_integrity": {
                    "base_state_sha256_before": base_hash,
                    "base_state_sha256_after": base_after,
                    "rcsp_adapter_sha256_before": rcsp_hash,
                    "rcsp_adapter_sha256_after": rcsp_after,
                    "rpa_adapter_sha256_initial": adapter_initial_hash,
                    "rpa_adapter_sha256_final": rpa_after,
                    "base_unchanged": base_after == base_hash,
                    "rcsp_unchanged": rcsp_after == rcsp_hash,
                    "rpa_changed_iff_retained_update": (
                        (rpa_after != adapter_initial_hash)
                        == bool(
                            training[
                                "retained_parameter_update_performed"
                            ]
                        )
                    ),
                    "frozen_inputs_unchanged": (
                        immutable_before == immutable_after
                    ),
                    "production_source_unchanged": (
                        implementation_before == implementation_after
                    ),
                    "base_and_rcsp_gradients_none": base_rcsp_grads_none,
                    "only_rpa_parameters_trainable": True,
                },
                "scientific_acceptance": False,
                "publish_allowed": False,
                "pilot_allowed": False,
                "production_model_modified": False,
                "production_inference_modified": False,
                "causal_root_cause_proven": False,
                "next_action": decision["next_action"],
            }
            _exclusive_json(result_dir / "report.json", report)

            print(
                json.dumps(
                    {
                        "stage": "rpa_lrta_complete",
                        "report": str(result_dir / "report.json"),
                        "fixed_final_cases": FINAL_CASES,
                        "accepted_steps": training["accepted_steps"],
                        "rollback_steps": training["rollback_steps"],
                        "decision": decision["result"],
                        "production_model_modified": False,
                        "production_inference_modified": False,
                        "pilot_allowed": False,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )
            return 0

    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "schema": SCHEMA,
                    "completed": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "zero_start_trainability_preflight": zero_start,
                    "optimizer_steps": 0,
                    "parameter_update_performed": False,
                    "production_model_modified": False,
                    "production_inference_modified": False,
                    "scientific_acceptance": False,
                    "publish_allowed": False,
                    "pilot_allowed": False,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase21-report", required=True)
    parser.add_argument("--bctr-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--legacy-core-strength", type=float, required=True
    )
    parser.add_argument(
        "--legacy-transition-strength", type=float, required=True
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
