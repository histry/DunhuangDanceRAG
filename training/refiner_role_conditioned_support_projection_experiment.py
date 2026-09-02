"""Diagnostic-only role-conditioned support-projected Refiner adapter.

The completed 400-step A0 Refiner is immutable.  Two zero-initialized 1x1
convolutions learn a 75D geometric correction from the frozen feature entering
the production output head.  Explicit role ids route cases to the single or
cross adapter.  The correction is projected onto binary support derived from
the production decoder masks, then the unchanged production decoder applies
its soft confidence, smoothing, taper, caps, and retraction exactly once.

This experiment never changes production inference, selects a checkpoint or
scale, publishes a model, or authorizes Pilot.
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

import numpy as np
import torch
from torch import nn

from training import motion_models as m
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_safe_start_diagnostics as safe
from training import refiner_temporal_action_alignment_audit as alignment
from training import refiner_temporal_scale_response_audit as scale_response
from training.refiner_optimizer import (
    checked_refiner_step,
    record_update,
    validate_update_summary,
)


SCHEMA = "refiner_role_conditioned_support_projection_experiment_v2"
MODEL_VERSION = "rcsp_adapter_diagnostic_only_v1"
STEPS = 400
GEOMETRY_DIM = 75
ROLE_MAPPING = {"single_recording": 0, "cross_event": 1}
ROLE_NAMES = {value: key for key, value in ROLE_MAPPING.items()}
GROUP_ROLE_IDS = torch.tensor((0, 0, 1, 1), dtype=torch.long)
FINAL_BLOCK_ORDER = (
    ("seen", "single_recording"),
    ("seen", "cross_event"),
    ("new_position", "single_recording"),
    ("new_position", "cross_event"),
)
FINAL_BLOCK_SIZE = 16
FINAL_CHUNK_SIZE = 8
SUPPORT_PROJECTION_PROTOCOL = (
    "production_effective_root_joint_weight_gt_zero_binary_projection_v1"
)
HIDDEN_CAPTURE_PROTOCOL = "temporary_forward_hook_on_frozen_base_out_input_v1"
PARAMETER_SCOPE = "single_adapter_and_cross_adapter_only"


def _finite(value, label):
    result = float(value.detach()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite {label}")
    return result


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if float(denominator) != 0.0 else None


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _tensor_max_error(left, right):
    if left.shape != right.shape:
        raise ValueError(f"parity shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return _finite((left.detach() - right.detach()).abs().max(), "parity error")


def role_ids_from_train_groups(group):
    """Map the existing four-group TRAIN contract to explicit role ids."""
    if group.ndim != 1 or group.dtype not in (torch.int32, torch.int64):
        raise ValueError("TRAIN group ids must be a rank-1 integer tensor")
    if bool(((group < 0) | (group >= len(GROUP_ROLE_IDS))).any()):
        raise ValueError("TRAIN group id is outside the fixed four-group contract")
    mapping = GROUP_ROLE_IDS.to(group.device)
    return mapping[group.long()]


def attach_train_role_ids(batch):
    if "group" not in batch:
        raise ValueError("TRAIN batch requires explicit group ids; width inference is forbidden")
    result = dict(batch)
    result["role_id"] = role_ids_from_train_groups(batch["group"])
    return result


def role_ids_from_metadata(metadata, device):
    values = []
    for row in metadata:
        role = row.get("role")
        if role not in ROLE_MAPPING:
            raise ValueError(f"unknown explicit final role: {role!r}")
        values.append(ROLE_MAPPING[role])
    return torch.tensor(values, dtype=torch.long, device=device)


def binary_geometry_support(joint_weight, root_weight):
    if joint_weight.ndim != 3 or joint_weight.shape[-1] != m.NUM_JOINTS:
        raise ValueError("joint weight must have shape [B,T,24]")
    if root_weight.ndim == 2:
        root_weight = root_weight.unsqueeze(-1)
    if root_weight.shape != joint_weight.shape[:-1] + (1,):
        raise ValueError("root and joint support layouts differ")
    root = (root_weight > 0).to(joint_weight.dtype).expand(
        joint_weight.shape[:-1] + (3,)
    )
    joint = (joint_weight > 0).to(joint_weight.dtype)[..., None].expand(
        joint_weight.shape + (3,)
    ).reshape(joint_weight.shape[:-1] + (m.NUM_JOINTS * 3,))
    support = torch.cat((root, joint), dim=-1)
    if support.shape[-1] != GEOMETRY_DIM:
        raise RuntimeError("binary support does not cover exactly 75 geometric coordinates")
    return support


class RoleConditionedSupportProjectedAdapter(nn.Module):
    """Two independent role heads with no width conditioning or soft scaling."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim < 1:
            raise ValueError("adapter hidden dimension must be positive")
        self.single_adapter = nn.Conv1d(self.hidden_dim, GEOMETRY_DIM, 1)
        self.cross_adapter = nn.Conv1d(self.hidden_dim, GEOMETRY_DIM, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.single_adapter, self.cross_adapter):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, hidden, role_id, joint_weight, root_weight):
        if hidden.ndim != 3 or hidden.shape[1] != self.hidden_dim:
            raise ValueError("hidden feature must have shape [B,H,T]")
        if role_id.shape != (hidden.shape[0],) or role_id.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("role_id must be an explicit rank-1 integer tensor")
        if bool(((role_id < 0) | (role_id > 1)).any()):
            raise ValueError("role_id values must be 0=single_recording or 1=cross_event")
        if joint_weight.shape[:2] != (hidden.shape[0], hidden.shape[2]):
            raise ValueError("decoder support is not aligned with hidden features")

        single = self.single_adapter(hidden).transpose(1, 2)
        cross = self.cross_adapter(hidden).transpose(1, 2)
        selected = torch.where(role_id[:, None, None] == 0, single, cross)
        support = binary_geometry_support(joint_weight, root_weight).to(selected.dtype)
        projected = selected * support
        return projected, {
            "adapter_raw": selected,
            "adapter_projected": projected,
            "binary_support": support,
        }


class FrozenBaseRCSPModel(nn.Module):
    """Diagnostic wrapper; the production model and decoder remain unchanged."""

    def __init__(self, base_model):
        super().__init__()
        if not isinstance(base_model, m.ProductManifoldTemporalRefiner):
            raise TypeError("RCSP requires ProductManifoldTemporalRefiner as the frozen base")
        self.base = base_model
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.base.eval()
        self.adapter = RoleConditionedSupportProjectedAdapter(self.base.out.in_channels)
        reference = next(self.base.parameters())
        self.adapter.to(device=reference.device, dtype=reference.dtype)
        self._active_route = None
        self._last_details = None

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @property
    def last_details(self):
        if self._last_details is None:
            raise RuntimeError("no captured RCSP forward details are available")
        return self._last_details

    def clear_last_details(self):
        self._last_details = None

    @contextmanager
    def route(self, role_id, joint_weight, root_weight, *, capture_details=False):
        if self._active_route is not None:
            raise RuntimeError("nested RCSP routing contexts are forbidden")
        self._active_route = (role_id, joint_weight, root_weight, bool(capture_details))
        self._last_details = None
        try:
            yield
        finally:
            self._active_route = None

    def _base_forward(self, x, cond, seam_mask, joint_mask):
        captured = []

        def hook(_module, inputs, _output):
            if captured:
                raise RuntimeError("RCSP base forward invoked output head more than once")
            captured.append(inputs[0].detach())

        handle = self.base.out.register_forward_hook(hook)
        try:
            self.base.eval()
            with torch.no_grad():
                raw = self.base(x, cond, seam_mask, joint_mask)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("RCSP did not capture the frozen output-head input")
        return raw.detach(), captured[0]

    def forward_explicit(
        self,
        x,
        cond,
        seam_mask,
        joint_mask,
        role_id,
        joint_weight,
        root_weight,
        *,
        capture_details=False,
    ):
        raw_base, hidden = self._base_forward(x, cond, seam_mask, joint_mask)
        delta, details = self.adapter(hidden, role_id, joint_weight, root_weight)
        raw_adapted = torch.cat(
            (raw_base[..., :4], raw_base[..., 4:] + delta), dim=-1
        )
        if not torch.equal(raw_adapted[..., :4], raw_base[..., :4]):
            raise RuntimeError("RCSP changed frozen contact channels")
        if capture_details:
            self._last_details = {
                **{key: value.detach() for key, value in details.items()},
                "role_id": role_id.detach(),
                "raw_base": raw_base.detach(),
                "raw_adapted": raw_adapted.detach(),
                "adapted_total_geometry": raw_adapted[..., 4:].detach(),
            }
        else:
            self._last_details = None
        return raw_adapted

    def forward(self, x, cond, seam_mask, joint_mask):
        if self._active_route is None:
            raise RuntimeError("RCSP forward requires explicit role and decoder support")
        role_id, joint_weight, root_weight, capture = self._active_route
        return self.forward_explicit(
            x,
            cond,
            seam_mask,
            joint_mask,
            role_id,
            joint_weight,
            root_weight,
            capture_details=capture,
        )


def _route_values(batch, cfg):
    if "role_id" not in batch:
        raise ValueError("RCSP batch is missing explicit role_id")
    count = int(batch["clean"].shape[0])
    role_id = batch["role_id"]
    if role_id.shape != (count,):
        raise ValueError("role_id does not match the RCSP batch")
    repair = m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg
    )
    clean = m._refiner_decode_masks(
        batch["clean_joint"],
        batch["clean_root"],
        batch["clean_contact"],
        batch["seam"],
        cfg,
    )
    return (
        torch.cat((role_id, role_id)),
        torch.cat((repair[0], clean[0])),
        torch.cat((repair[1], clean[1])),
    )


def rcsp_batch_outputs(model, batch, cfg, *, trace=None, capture_details=False):
    role_id, joint_weight, root_weight = _route_values(batch, cfg)
    with model.route(
        role_id, joint_weight, root_weight, capture_details=capture_details
    ):
        return m._refiner_batch_outputs(model, batch, cfg, trace=trace)


def rcsp_batch_objectives(model, batch, cfg, *, capture_details=False):
    role_id, joint_weight, root_weight = _route_values(batch, cfg)
    with model.route(
        role_id, joint_weight, root_weight, capture_details=capture_details
    ):
        return m._refiner_batch_objectives(model, batch, cfg)


def rcsp_guarded_total_batch_loss(model, batch, cfg):
    role_id, joint_weight, root_weight = _route_values(batch, cfg)
    with model.route(role_id, joint_weight, root_weight, capture_details=False):
        return m._refiner_guarded_total_batch_loss(
            model, batch, cfg, require_all_groups=True
        )


def _named_adapter_parameters(model):
    return dict(model.adapter.named_parameters())


def validate_parameter_scope(model):
    base = list(model.base.named_parameters())
    adapter = list(model.adapter.named_parameters())
    if not base or not adapter:
        raise RuntimeError("RCSP parameter scopes are incomplete")
    if any(parameter.requires_grad or parameter.grad is not None for _, parameter in base):
        raise RuntimeError("frozen base parameter is trainable or has gradient residue")
    if any(not parameter.requires_grad for _, parameter in adapter):
        raise RuntimeError("RCSP adapter parameter is unexpectedly frozen")
    total = sum(parameter.numel() for _, parameter in adapter)
    expected = 2 * GEOMETRY_DIM * (model.base.out.in_channels + 1)
    if total != expected:
        raise RuntimeError("RCSP adapter parameter count mismatch")
    return {
        "base_parameters": sum(parameter.numel() for _, parameter in base),
        "base_trainable_parameters": 0,
        "adapter_parameters": total,
        "single_adapter_parameters": sum(
            parameter.numel()
            for name, parameter in adapter
            if name.startswith("single_adapter.")
        ),
        "cross_adapter_parameters": sum(
            parameter.numel()
            for name, parameter in adapter
            if name.startswith("cross_adapter.")
        ),
        "trainable_parameter_names": [f"adapter.{name}" for name, _ in adapter],
    }


def validate_zero_initialization(model):
    parameters = _named_adapter_parameters(model)
    exact = all(bool((parameter.detach() == 0).all()) for parameter in parameters.values())
    if not exact:
        raise RuntimeError("RCSP adapters are not exactly zero initialized")
    return {
        "weight": 0.0,
        "bias": 0.0,
        "all_parameters_exactly_zero": True,
        "delta_single_exactly_zero": True,
        "delta_cross_exactly_zero": True,
    }


def _parameter_norms(model):
    result = {}
    for role in ROLE_MAPPING:
        prefix = "single_adapter." if role == "single_recording" else "cross_adapter."
        values = [
            parameter.detach().double().reshape(-1)
            for name, parameter in model.adapter.named_parameters()
            if name.startswith(prefix)
        ]
        result[role] = _finite(torch.cat(values).norm(), f"{role} parameter norm")
    result["total"] = math.sqrt(sum(value * value for value in result.values()))
    return result


def _update_norms(model, before):
    result = {}
    current = _named_adapter_parameters(model)
    for role in ROLE_MAPPING:
        prefix = "single_adapter." if role == "single_recording" else "cross_adapter."
        squares = [
            (current[name].detach().double() - before[name].double()).square().sum()
            for name in current
            if name.startswith(prefix)
        ]
        result[role] = math.sqrt(sum(float(value) for value in squares))
    result["total"] = math.sqrt(sum(value * value for value in result.values()))
    return result


def _adapter_output_norms(details, repair_cases):
    rows = {}
    role_ids = details["role_id"][:repair_cases]
    for role, role_id in ROLE_MAPPING.items():
        selected = role_ids == role_id
        if not bool(selected.any()):
            raise RuntimeError(f"TRAIN transaction omitted {role}")
        raw = details["adapter_raw"][:repair_cases][selected].double()
        projected = details["adapter_projected"][:repair_cases][selected].double()
        rows[role] = {
            "cases": int(selected.sum()),
            "raw_l2_norm": _finite(raw.norm(), f"{role} raw adapter output"),
            "projected_l2_norm": _finite(
                projected.norm(), f"{role} projected adapter output"
            ),
            "projected_rms": _finite(
                projected.square().mean().sqrt(), f"{role} projected adapter rms"
            ),
        }
    return rows


def _compact_update(update):
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


def _move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_adapters(model, bank, cfg, destination):
    """Exactly 400 checked updates on the frozen TRAIN reservoir."""
    destination = Path(destination)
    updates_path = destination / "updates.jsonl"
    updates_path.touch(exist_ok=False)
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(), lr=cfg.lr, weight_decay=1.0e-4
    )
    summary = {}
    started = time.perf_counter()
    model.train()
    for step in range(1, STEPS + 1):
        transaction = (step - 1) % len(bank["transaction_schedule"])
        batch = attach_train_role_ids(
            _move_batch(group_audit.materialize_transaction(bank, cfg, transaction), cfg.device)
        )
        repair, clean, terms, _ = rcsp_batch_objectives(
            model, batch, cfg, capture_details=True
        )
        loss = repair + cfg.product_refiner_clean_identity_weight * clean
        output_norms = _adapter_output_norms(model.last_details, len(batch["clean"]))
        model.clear_last_details()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        named = _named_adapter_parameters(model)
        if any(parameter.grad is None for parameter in named.values()):
            raise RuntimeError("an RCSP adapter parameter did not receive a gradient")
        before = {name: parameter.detach().clone() for name, parameter in named.items()}
        clip_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.adapter.parameters(), 1.0, error_if_nonfinite=True
            )
        )
        update = checked_refiner_step(
            optimizer,
            loss,
            lambda: rcsp_guarded_total_batch_loss(model, batch, cfg),
            gradient_unscale=max(1.0, clip_norm + 1.0e-6),
            group_guard_before=m._refiner_group_repair_losses(terms, require_all=True),
            group_guard_relative_tolerance=cfg.product_refiner_group_guard_relative_tolerance,
            group_guard_absolute_tolerance=cfg.product_refiner_group_guard_absolute_tolerance,
        )
        record_update(summary, update)
        updates = _update_norms(model, before)
        if not update["optimizer_update_accepted"] and any(
            value != 0.0 for value in updates.values()
        ):
            raise RuntimeError("rolled-back RCSP step changed adapter parameters")
        row = {
            "step": step,
            "state_position": "after_checked_step",
            "transaction_index": transaction,
            "context_indices": list(bank["transaction_schedule"][transaction]),
            "cases": int(batch["clean"].shape[0]),
            "cases_per_role": {
                role: int((batch["role_id"] == role_id).sum())
                for role, role_id in ROLE_MAPPING.items()
            },
            "training_objective_before": _finite(loss, "training objective"),
            "optimizer": _compact_update(update),
            "accepted": bool(update["optimizer_update_accepted"]),
            "rolled_back": not bool(update["optimizer_update_accepted"]),
            "adapter_parameter_norm": _parameter_norms(model),
            "adapter_update_norm": updates,
            "adapter_output_norm": output_norms,
        }
        with updates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if step in (1, 2, 5, 10, 25, 50, 100, 200, 300, 400):
            print(
                json.dumps(
                    {
                        "stage": "rcsp_adapter_training",
                        "step": step,
                        "accepted": row["accepted"],
                        "objective": row["training_objective_before"],
                        "parameter_norm": row["adapter_parameter_norm"],
                        "update_norm": row["adapter_update_norm"],
                        "output_norm": row["adapter_output_norm"],
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
    validate_update_summary(summary, STEPS)
    return {
        "optimizer_summary": summary,
        "accepted_steps": int(summary["accepted_steps"]),
        "rollback_steps": int(summary["retained_steps"]),
        "final_adapter_parameter_norm": _parameter_norms(model),
        "updates_artifact": {
            "path": str(updates_path),
            "sha256": group_audit.file_sha256(updates_path),
            "rows": STEPS,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _base_batch_outputs(base, batch, cfg):
    with alignment._capture_model_output(base) as captured:
        prediction, identity = m._refiner_batch_outputs(base, batch, cfg)
    if len(captured) != 1:
        raise RuntimeError("base parity requires exactly one frozen model forward")
    return prediction, identity, captured[0].transpose(1, 2).detach()


def train_initial_parity(base, model, batch, cfg):
    """Exact TRAIN raw/decoded and temporal/endpoint parity at step zero."""
    batch = attach_train_role_ids(batch)
    base.eval()
    model.eval()
    with torch.no_grad():
        base_prediction, base_identity, base_raw = _base_batch_outputs(base, batch, cfg)
        adapted_prediction, adapted_identity = rcsp_batch_outputs(
            model, batch, cfg, capture_details=True
        )
        details = model.last_details
        _, base_terms = m._observable_refiner_objective(
            base_prediction, batch["bad"], batch["seam"], cfg, reduction="none"
        )
        _, adapted_terms = m._observable_refiner_objective(
            adapted_prediction, batch["bad"], batch["seam"], cfg, reduction="none"
        )
    fields = ("temporal_scientific_deficit", "endpoint_scientific_deficit")
    result = {
        "cases": int(batch["clean"].shape[0]),
        "raw_max_abs_difference": _tensor_max_error(base_raw, details["raw_adapted"]),
        "base_raw_vs_wrapper_base_max_abs_difference": _tensor_max_error(
            base_raw, details["raw_base"]
        ),
        "decoded_motion_max_abs_difference": _tensor_max_error(
            base_prediction, adapted_prediction
        ),
        "clean_decoded_motion_max_abs_difference": _tensor_max_error(
            base_identity, adapted_identity
        ),
        "scientific_metric_max_abs_difference": {
            field: _tensor_max_error(base_terms[field], adapted_terms[field])
            for field in fields
        },
        "contact_channels_exactly_unchanged": bool(
            torch.equal(details["raw_adapted"][..., :4], details["raw_base"][..., :4])
        ),
        "adapter_projected_delta_exactly_zero": bool(
            (details["adapter_projected"] == 0).all()
        ),
    }
    model.clear_last_details()
    if (
        any(
            result[key] != 0.0
            for key in (
                "raw_max_abs_difference",
                "base_raw_vs_wrapper_base_max_abs_difference",
                "decoded_motion_max_abs_difference",
                "clean_decoded_motion_max_abs_difference",
            )
        )
        or any(value != 0.0 for value in result["scientific_metric_max_abs_difference"].values())
        or not result["contact_channels_exactly_unchanged"]
        or not result["adapter_projected_delta_exactly_zero"]
    ):
        raise RuntimeError("zero-initialized RCSP TRAIN parity failed")
    result["verified"] = True
    result["physical_geometry_clean_identity_parity_basis"] = (
        "exact decoded repair and clean tensors"
    )
    return result


def _case_row(meta, prediction, identity, batch, index, terms, cfg):
    pred = prediction[index].detach().cpu().numpy()
    ident = identity[index].detach().cpu().numpy()
    reference = batch["bad"][index].detach().cpu().numpy()
    clean = batch["clean"][index].detach().cpu().numpy()
    seam = batch["seam"][index].detach().cpu().numpy()
    details = scale_response._physical_case(
        reference, pred, clean, ident, seam, cfg
    )
    observable = details.pop("observable")
    temporal_pass = bool(observable["temporal_accepted"])
    endpoint_pass = bool(observable["endpoint_accepted"])
    physical_pass = bool(details["physical"]["accepted"])
    geometry_pass = bool(details["geometry"]["accepted"])
    clean_pass = bool(details["clean_identity"]["accepted"])
    return {
        **meta,
        "temporal_scientific_deficit": _finite(
            terms["temporal_scientific_deficit"][index], "temporal deficit"
        ),
        "endpoint_scientific_deficit": _finite(
            terms["endpoint_scientific_deficit"][index], "endpoint deficit"
        ),
        "temporal_metric": _finite(
            observable["after"]["temporal_energy"], "temporal metric"
        ),
        "endpoint_metric": _finite(
            observable["after"]["endpoint_velocity_jump_mps"], "endpoint metric"
        ),
        "temporal_repair_gain": _finite(observable["temporal_gain"], "temporal gain"),
        "endpoint_repair_gain": _finite(observable["endpoint_gain"], "endpoint gain"),
        "temporal_gate_pass": temporal_pass,
        "endpoint_gate_pass": endpoint_pass,
        "physical_pass": physical_pass,
        "geometry_pass": geometry_pass,
        "clean_pass": clean_pass,
        "all_diagnostic_conditions": bool(
            temporal_pass and endpoint_pass and physical_pass and geometry_pass and clean_pass
        ),
        "observable": observable,
        "physical": details["physical"],
        "geometry": details["geometry"],
        "clean_identity": details["clean_identity"],
    }


def _support_case_rows(details, metadata, repair_cases):
    raw = details["adapter_raw"][:repair_cases].double()
    projected = details["adapter_projected"][:repair_cases].double()
    support = details["binary_support"][:repair_cases].double()
    rows = []
    for index, meta in enumerate(metadata):
        raw_case, projected_case, support_case = raw[index], projected[index], support[index]
        outside = raw_case * (1.0 - support_case)
        projected_outside = projected_case * (1.0 - support_case)
        raw_norm = _finite(raw_case.norm(), "adapter raw norm")
        projected_norm = _finite(projected_case.norm(), "adapter projected norm")
        rows.append(
            {
                **meta,
                "adapter_raw_norm": raw_norm,
                "adapter_projected_norm": projected_norm,
                "projection_retention_ratio": _ratio(projected_norm, raw_norm),
                "adapter_energy_outside_support_before_projection": _finite(
                    outside.square().sum(), "outside-support adapter energy"
                ),
                "projected_outside_support_max": _finite(
                    projected_outside.abs().max(), "projected outside-support max"
                ),
            }
        )
    return rows


def _forward_evaluation_batch(model, batch, cfg, *, capture_details=False):
    if isinstance(model, FrozenBaseRCSPModel):
        prediction, identity = rcsp_batch_outputs(
            model, batch, cfg, capture_details=capture_details
        )
        raw = model.last_details["raw_adapted"] if capture_details else None
        details = model.last_details if capture_details else None
        return prediction, identity, raw, details
    prediction, identity, raw = _base_batch_outputs(model, batch, cfg)
    return prediction, identity, raw, None


def evaluate_fixed_final(model, batch, metadata, cfg, *, parity_base=None):
    if int(batch["clean"].shape[0]) != 64 or len(metadata) != 64:
        raise ValueError("RCSP fixed final evaluation requires exactly 64 cases")
    rows, support_rows = [], []
    parity = {
        "raw_max_abs_difference": 0.0,
        "decoded_motion_max_abs_difference": 0.0,
        "clean_decoded_motion_max_abs_difference": 0.0,
    }
    offset = 0
    model.eval()
    if parity_base is not None:
        parity_base.eval()
    with torch.no_grad():
        for split, role in FINAL_BLOCK_ORDER:
            block = metadata[offset : offset + FINAL_BLOCK_SIZE]
            if any(row["split"] != split or row["role"] != role for row in block):
                raise ValueError("fixed final block order mismatch")
            for local in range(0, FINAL_BLOCK_SIZE, FINAL_CHUNK_SIZE):
                start, stop = offset + local, offset + local + FINAL_CHUNK_SIZE
                part_meta = metadata[start:stop]
                part = {key: value[start:stop] for key, value in batch.items()}
                part["role_id"] = role_ids_from_metadata(part_meta, part["clean"].device)
                prediction, identity, raw, details = _forward_evaluation_batch(
                    model, part, cfg, capture_details=isinstance(model, FrozenBaseRCSPModel)
                )
                if parity_base is not None:
                    base_prediction, base_identity, base_raw, _ = _forward_evaluation_batch(
                        parity_base, part, cfg
                    )
                    parity["raw_max_abs_difference"] = max(
                        parity["raw_max_abs_difference"], _tensor_max_error(base_raw, raw)
                    )
                    parity["decoded_motion_max_abs_difference"] = max(
                        parity["decoded_motion_max_abs_difference"],
                        _tensor_max_error(base_prediction, prediction),
                    )
                    parity["clean_decoded_motion_max_abs_difference"] = max(
                        parity["clean_decoded_motion_max_abs_difference"],
                        _tensor_max_error(base_identity, identity),
                    )
                _, terms = m._observable_refiner_objective(
                    prediction, part["bad"], part["seam"], cfg, reduction="none"
                )
                rows.extend(
                    _case_row(meta, prediction, identity, part, index, terms, cfg)
                    for index, meta in enumerate(part_meta)
                )
                if details is not None:
                    support_rows.extend(
                        _support_case_rows(details, part_meta, len(part_meta))
                    )
                    model.clear_last_details()
            offset += FINAL_BLOCK_SIZE
    if len(rows) != 64:
        raise RuntimeError("fixed final evaluation did not produce 64 rows")
    if support_rows and max(row["projected_outside_support_max"] for row in support_rows) != 0.0:
        raise RuntimeError("RCSP projected correction escaped binary decoder support")
    parity["verified"] = parity_base is not None and all(value == 0.0 for key, value in parity.items()
                                                          if key != "verified")
    return {"case_level": rows, "support_projection_case_level": support_rows,
            "step_zero_base_parity": parity}


def _trajectory_parity_rows(rows):
    result = []
    for row in rows:
        result.append(
            {
                "split": row["split"],
                "role": row["role"],
                "width": row["width"],
                "bank_case_index": row["bank_case_index"],
                "responses": {
                    "1.00": {
                        "authoritative_observable": {
                            "temporal_metric": row["temporal_metric"],
                            "endpoint_metric": row["endpoint_metric"],
                            "temporal_repair_gain": row["temporal_repair_gain"],
                            "endpoint_repair_gain": row["endpoint_repair_gain"],
                            "temporal_gate_pass": row["temporal_gate_pass"],
                            "endpoint_gate_pass": row["endpoint_gate_pass"],
                        },
                        "geometry": {
                            "accepted": row["geometry_pass"],
                            "reference_fidelity": row["geometry"]["reference_fidelity"],
                        },
                        "physical": {
                            "accepted": row["physical_pass"],
                            "reasons": row["physical"]["reasons"],
                            "authoritative_gate": row["physical"]["authoritative_gate"],
                        },
                        "clean_identity": {
                            "accepted": row["clean_pass"],
                            "product_log_l1": row["clean_identity"]["product_log_l1"],
                            "contact_l1": row["clean_identity"]["contact_l1"],
                        },
                    }
                },
            }
        )
    return result


def summarize_rows(rows):
    if not rows:
        raise ValueError("cannot summarize an empty fixed-final group")

    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    return {
        "cases": len(rows),
        "temporal_gate_pass_cases": sum(row["temporal_gate_pass"] for row in rows),
        "endpoint_gate_pass_cases": sum(row["endpoint_gate_pass"] for row in rows),
        "physical_pass_cases": sum(row["physical_pass"] for row in rows),
        "geometry_pass_cases": sum(row["geometry_pass"] for row in rows),
        "clean_pass_cases": sum(row["clean_pass"] for row in rows),
        "all_diagnostic_conditions_cases": sum(
            row["all_diagnostic_conditions"] for row in rows
        ),
        "temporal_scientific_deficit_mean": mean("temporal_scientific_deficit"),
        "endpoint_scientific_deficit_mean": mean("endpoint_scientific_deficit"),
        "temporal_repair_gain_mean": mean("temporal_repair_gain"),
        "endpoint_repair_gain_mean": mean("endpoint_repair_gain"),
    }


def fixed_final_summary(rows):
    result = {"overall": summarize_rows(rows), "groups": {}}
    for split, role in FINAL_BLOCK_ORDER:
        for width in (10, 28):
            name = f"{split}/{role}/{width}"
            selected = [
                row
                for row in rows
                if row["split"] == split and row["role"] == role and row["width"] == width
            ]
            result["groups"][name] = summarize_rows(selected)
    return result


def baseline_comparison(base_summary, rcsp_summary):
    fields = (
        "temporal_scientific_deficit_mean",
        "endpoint_scientific_deficit_mean",
        "temporal_gate_pass_cases",
        "endpoint_gate_pass_cases",
        "physical_pass_cases",
        "geometry_pass_cases",
        "clean_pass_cases",
        "all_diagnostic_conditions_cases",
    )

    def compare(base, rcsp):
        return {
            "BASE": base,
            "RCSP": rcsp,
            "delta_rcsp_minus_base": {field: rcsp[field] - base[field] for field in fields},
            "deficit_delta_sign": "negative_means_improvement",
        }

    return {
        "overall": compare(base_summary["overall"], rcsp_summary["overall"]),
        "groups": {
            name: compare(base_summary["groups"][name], rcsp_summary["groups"][name])
            for name in base_summary["groups"]
        },
        "baseline_source": "recomputed_from_immutable_trajectory_final_checkpoint",
        "baseline_retrained": False,
    }


def _cosine_to_negative_gradient(action, gradient):
    a, target = action.double().reshape(-1), -gradient.double().reshape(-1)
    denominator = float(a.norm() * target.norm())
    if denominator == 0.0:
        return None
    return _finite(torch.dot(a, target) / denominator, "direction cosine")


def _median(values):
    finite = [float(value) for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def _direction_summary(rows):
    return {
        "cases": len(rows),
        "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median": _median(
            row["projected_adapter_delta_vs_negative_temporal_gradient_cosine"] for row in rows
        ),
        "adapted_total_action_vs_negative_temporal_gradient_cosine_median": _median(
            row["adapted_total_action_vs_negative_temporal_gradient_cosine"] for row in rows
        ),
        "defined_projected_adapter_cosines": sum(
            row["projected_adapter_delta_vs_negative_temporal_gradient_cosine"] is not None
            for row in rows
        ),
        "defined_total_action_cosines": sum(
            row["adapted_total_action_vs_negative_temporal_gradient_cosine"] is not None
            for row in rows
        ),
    }


def direction_alignment(model, batch, metadata, cfg):
    """One read-only final-step temporal raw-action VJP; never an optimizer input."""
    rows = []
    model.eval()
    for parameter in model.adapter.parameters():
        parameter.grad = None
    offset = 0
    with torch.enable_grad():
        for split, role in FINAL_BLOCK_ORDER:
            block = metadata[offset : offset + FINAL_BLOCK_SIZE]
            if len(block) != FINAL_BLOCK_SIZE or any(
                row["split"] != split or row["role"] != role for row in block
            ):
                raise ValueError("direction diagnostic fixed-final block order mismatch")
            for local in range(0, FINAL_BLOCK_SIZE, FINAL_CHUNK_SIZE):
                start, stop = offset + local, offset + local + FINAL_CHUNK_SIZE
                part_meta = metadata[start:stop]
                part = {key: value[start:stop] for key, value in batch.items()}
                role_id = role_ids_from_metadata(part_meta, part["bad"].device)
                masks = m._refiner_decode_masks(
                    part["joint"], part["root"], part["contact"], part["seam"], cfg
                )
                raw = model.forward_explicit(
                    part["bad"],
                    part["cond"],
                    part["seam"],
                    part["joint"],
                    role_id,
                    masks[0],
                    masks[1],
                    capture_details=True,
                )
                prediction = m._decode_product_refiner_output(
                    part["bad"], raw, *masks, cfg
                )
                terms = alignment._scientific_terms(
                    prediction, part["bad"], part["seam"], cfg
                )
                gradient = torch.autograd.grad(
                    terms["temporal"].sum(), raw, allow_unused=True
                )[0]
                gradient = torch.zeros_like(raw) if gradient is None else gradient
                details = model.last_details
                projected = details["adapter_projected"]
                total = raw[..., 4:].detach()
                for index, meta in enumerate(part_meta):
                    rows.append(
                        {
                            **meta,
                            "projected_adapter_delta_vs_negative_temporal_gradient_cosine": (
                                _cosine_to_negative_gradient(
                                    projected[index], gradient[index, ..., 4:]
                                )
                            ),
                            "adapted_total_action_vs_negative_temporal_gradient_cosine": (
                                _cosine_to_negative_gradient(
                                    total[index], gradient[index, ..., 4:]
                                )
                            ),
                            "temporal_gradient_norm": _finite(
                                gradient[index, ..., 4:].double().norm(),
                                "temporal raw gradient norm",
                            ),
                        }
                    )
                model.clear_last_details()
            offset += FINAL_BLOCK_SIZE
    if any(parameter.grad is not None for parameter in model.adapter.parameters()):
        raise RuntimeError("read-only direction diagnostic populated adapter .grad")
    scopes = {"overall": rows}
    for role in ROLE_MAPPING:
        scopes[f"role:{role}"] = [row for row in rows if row["role"] == role]
    for width in (10, 28):
        scopes[f"width:{width}"] = [row for row in rows if row["width"] == width]
    for split, role in FINAL_BLOCK_ORDER:
        for width in (10, 28):
            name = f"{split}/{role}/{width}"
            scopes[f"group:{name}"] = [
                row
                for row in rows
                if row["split"] == split
                and row["role"] == role
                and row["width"] == width
            ]
    return {
        "case_level": rows,
        "summary": {name: _direction_summary(values) for name, values in scopes.items()},
        "objective": "temporal_scientific_deficit",
        "gradient_target": "raw_geometric_tangent_75D",
        "read_only_final_step_400": True,
        "used_for_optimizer_update": False,
        "gradient_surgery_performed": False,
    }


def support_projection_summary(rows):
    if len(rows) != 64:
        raise ValueError("support projection summary requires all 64 fixed-final cases")

    def summarize(values):
        return {
            "cases": len(values),
            "adapter_raw_norm_median": _median(row["adapter_raw_norm"] for row in values),
            "adapter_projected_norm_median": _median(
                row["adapter_projected_norm"] for row in values
            ),
            "projection_retention_ratio_median": _median(
                row["projection_retention_ratio"] for row in values
            ),
            "adapter_energy_outside_support_before_projection_median": _median(
                row["adapter_energy_outside_support_before_projection"] for row in values
            ),
            "projected_outside_support_max": max(
                row["projected_outside_support_max"] for row in values
            ),
        }

    scopes = {"overall": rows}
    for role in ROLE_MAPPING:
        scopes[f"role:{role}"] = [row for row in rows if row["role"] == role]
    for width in (10, 28):
        scopes[f"width:{width}"] = [row for row in rows if row["width"] == width]
    for split, role in FINAL_BLOCK_ORDER:
        for width in (10, 28):
            name = f"group:{split}/{role}/{width}"
            scopes[name] = [
                row
                for row in rows
                if row["split"] == split
                and row["role"] == role
                and row["width"] == width
            ]
    result = {name: summarize(values) for name, values in scopes.items()}
    if result["overall"]["projected_outside_support_max"] != 0.0:
        raise RuntimeError("support projection summary found an escaped correction")
    return {
        "case_level": rows,
        "summary": result,
        "energy_definition": "squared_l2_norm_before_projection_outside_binary_support",
        "selection_metric": False,
    }


def scientific_answers(base_summary, rcsp_summary, comparison):
    base_overall, rcsp_overall = base_summary["overall"], rcsp_summary["overall"]
    base_single = summarize_rows(
        [row for row in base_summary["case_level"] if row["role"] == "single_recording"]
    ) if "case_level" in base_summary else None
    rcsp_single = summarize_rows(
        [row for row in rcsp_summary["case_level"] if row["role"] == "single_recording"]
    ) if "case_level" in rcsp_summary else None
    base_cross = summarize_rows(
        [row for row in base_summary["case_level"] if row["role"] == "cross_event"]
    ) if "case_level" in base_summary else None
    rcsp_cross = summarize_rows(
        [row for row in rcsp_summary["case_level"] if row["role"] == "cross_event"]
    ) if "case_level" in rcsp_summary else None
    if any(value is None for value in (base_single, rcsp_single, base_cross, rcsp_cross)):
        raise ValueError("scientific answer requires fixed-final case rows")
    safety_fields = ("physical_pass_cases", "geometry_pass_cases", "clean_pass_cases")
    safety_regression = any(
        rcsp_overall[field] < base_overall[field] for field in safety_fields
    ) or any(
        row["RCSP"][field] < row["BASE"][field]
        for row in comparison["groups"].values()
        for field in safety_fields
    )
    single_pass_increased = (
        rcsp_single["temporal_gate_pass_cases"] > base_single["temporal_gate_pass_cases"]
    )
    single_deficit_improved = (
        rcsp_single["temporal_scientific_deficit_mean"]
        < base_single["temporal_scientific_deficit_mean"]
    )
    cross_improved = (
        rcsp_cross["temporal_gate_pass_cases"] > base_cross["temporal_gate_pass_cases"]
        or rcsp_cross["temporal_scientific_deficit_mean"]
        < base_cross["temporal_scientific_deficit_mean"]
    )
    group_improved = {
        name: (
            row["RCSP"]["temporal_gate_pass_cases"]
            > row["BASE"]["temporal_gate_pass_cases"]
            or row["RCSP"]["temporal_scientific_deficit_mean"]
            < row["BASE"]["temporal_scientific_deficit_mean"]
        )
        for name, row in comparison["groups"].items()
    }
    group_gate_rescue = {
        name: (
            row["RCSP"]["temporal_gate_pass_cases"]
            > row["BASE"]["temporal_gate_pass_cases"]
        )
        for name, row in comparison["groups"].items()
    }
    group_gate_delta = {
        name: int(row["delta_rcsp_minus_base"]["temporal_gate_pass_cases"])
        for name, row in comparison["groups"].items()
    }
    group_relative_deficit_improvement = {}
    for name, row in comparison["groups"].items():
        base_value = float(row["BASE"]["temporal_scientific_deficit_mean"])
        rcsp_value = float(row["RCSP"]["temporal_scientific_deficit_mean"])
        group_relative_deficit_improvement[name] = (
            (base_value - rcsp_value) / base_value if base_value != 0.0 else None
        )

    role_gate_delta = {
        role: sum(
            value for name, value in group_gate_delta.items() if f"/{role}/" in name
        )
        for role in ROLE_MAPPING
    }
    width_gate_delta = {
        str(width): sum(
            value for name, value in group_gate_delta.items() if name.endswith(f"/{width}")
        )
        for width in (10, 28)
    }
    if any(value < 0 for value in width_gate_delta.values()):
        width_pattern = "TEMPORAL_GATE_REGRESSION_PRESENT"
    elif width_gate_delta["10"] > 0 and width_gate_delta["28"] == 0:
        width_pattern = "WIDTH_10_ONLY"
    elif width_gate_delta["28"] > 0 and width_gate_delta["10"] == 0:
        width_pattern = "WIDTH_28_ONLY"
    elif width_gate_delta["10"] > 0 and width_gate_delta["28"] > 0:
        width_pattern = "BOTH_WIDTHS"
    else:
        width_pattern = "NO_TEMPORAL_GATE_RESCUE"

    if safety_regression:
        classification = "REJECTED_SAFETY_REGRESSION"
    elif single_pass_increased and single_deficit_improved:
        classification = "SUPPORTED_BY_DIAGNOSTIC_EXPERIMENT"
    elif width_pattern in ("WIDTH_10_ONLY", "WIDTH_28_ONLY"):
        classification = "ROLE_CONDITIONING_USEFUL_BUT_WIDTH_DEPENDENT_MECHANISM_REMAINS"
    elif cross_improved and not single_pass_increased:
        classification = "ROLE_CONDITIONING_ALONE_INSUFFICIENT"
    elif not any(group_improved.values()):
        classification = "NOT_SUPPORTED_BY_DIAGNOSTIC_EXPERIMENT"
    else:
        classification = "MIXED_DESCRIPTIVE_RESPONSE"
    return {
        "role_conditioned_direction_rescue": classification,
        "overall_temporal_pass_count_increased": (
            rcsp_overall["temporal_gate_pass_cases"]
            > base_overall["temporal_gate_pass_cases"]
        ),
        "any_single_recording_case_crossed_temporal_gate": (
            rcsp_single["temporal_gate_pass_cases"] > 0
        ),
        "single_temporal_deficit_improved": single_deficit_improved,
        "cross_performance_preserved_or_improved": (
            rcsp_cross["temporal_gate_pass_cases"]
            >= base_cross["temporal_gate_pass_cases"]
            and rcsp_cross["temporal_scientific_deficit_mean"]
            <= base_cross["temporal_scientific_deficit_mean"]
        ),
        "physical_geometry_or_clean_regression": safety_regression,
        "group_descriptive_improvement": group_improved,
        "group_temporal_gate_rescue": group_gate_rescue,
        "temporal_gate_pass_delta_by_group": group_gate_delta,
        "temporal_gate_pass_delta_by_role": role_gate_delta,
        "temporal_gate_pass_delta_by_width": width_gate_delta,
        "temporal_gate_rescue_width_pattern": width_pattern,
        "relative_temporal_deficit_improvement_by_group": (
            group_relative_deficit_improvement
        ),
        "claim_boundary": (
            "Descriptive fixed-step diagnostic comparison only. It does not prove a root cause, "
            "select a checkpoint or scale, change production architecture, or authorize Pilot."
        ),
    }


def _save_adapter_checkpoint(path, model, base_checkpoint, base_state_hash):
    payload = {
        "schema": SCHEMA,
        "version": MODEL_VERSION,
        "completed_steps": STEPS,
        "base_checkpoint": str(base_checkpoint),
        "base_state_sha256": base_state_hash,
        "adapter_state_dict": {
            name: value.detach().cpu() for name, value in model.adapter.state_dict().items()
        },
        "parameter_update_scope": PARAMETER_SCOPE,
        "formal_checkpoint": False,
        "production_model_modified": False,
        "checkpoint_selection_performed": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "resume_allowed": False,
    }
    m._atomic_torch_save(payload, path)


def run(args):
    source = Path(args.state_dir).resolve()
    output = Path(args.output_dir).resolve()
    traj_dir, traj_paths, traj_hashes, traj_report, experiment, checkpoint = (
        failure._load_trajectory(args.trajectory_dir, args.expected_trajectory_commit)
    )
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(traj_dir):
        raise FileExistsError("RCSP output must be a fresh directory outside immutable inputs")
    state, bank, cfg, source_metadata = group_audit.load_frozen_source(
        source,
        group_audit.LEGACY_COMMIT,
        legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength,
    )
    if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
        raise ValueError("trajectory does not reference the supplied frozen source")
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    source_paths = {
        name: source / name
        for name in ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise FileNotFoundError("frozen source is incomplete, including probe_bank.pt")
    source_hashes = {name: group_audit.file_sha256(path) for name, path in source_paths.items()}
    probe_hash = state.get("probe_bank_artifact", {}).get("sha256")
    if probe_hash != traj_report.get("probe_sha256") or source_hashes["probe_bank.pt"] != probe_hash:
        raise ValueError("trajectory/source probe lineage mismatch")
    train_banks = safe.train_banks(bank, cfg)
    output.mkdir(parents=True, exist_ok=False)
    failure_path = output / "failure.json"
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices), group_audit.frozen_environment(
            state["fingerprint"], source_metadata["decoder_strengths"]
        ):
            base = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
            base.load_state_dict(checkpoint["model_state_dict"], strict=True)
            base.eval()
            base_state_hash = safe.state_hash(base.state_dict())
            if base_state_hash != traj_report["final_state_sha256"]:
                raise RuntimeError("loaded base does not match the immutable trajectory final state")
            model = FrozenBaseRCSPModel(base)
            parameter_scope = validate_parameter_scope(model)
            initialization = validate_zero_initialization(model)

            initial_train_banks = {}
            for bank_name, cpu_batch in train_banks:
                parity_batch = attach_train_role_ids(_move_batch(cpu_batch, device))
                initial_train_banks[bank_name] = train_initial_parity(
                    base, model, parity_batch, cfg
                )
            train_batch = attach_train_role_ids(
                _move_batch(group_audit.materialize_transaction(bank, cfg, 0), device)
            )
            initial_train_transaction = train_initial_parity(
                base, model, train_batch, cfg
            )
            initial_train = {
                "verified": initial_train_transaction["verified"]
                and all(row["verified"] for row in initial_train_banks.values()),
                "transaction_0": initial_train_transaction,
                "all_frozen_reservoir_banks": initial_train_banks,
                "reservoir_banks_checked": len(initial_train_banks),
                "reservoir_cases_checked": sum(
                    row["cases"] for row in initial_train_banks.values()
                ),
            }
            probe, loaded_probe_hash = safe.load_probe(source, state, bank, cfg)
            final_batch, final_metadata = alignment.combine_final_banks(
                failure.final_banks(bank, probe, cfg)
            )
            final_batch = _move_batch(final_batch, device)
            base_final = evaluate_fixed_final(base, final_batch, final_metadata, cfg)
            trajectory_parity = scale_response.validate_alpha_one_final_metrics(
                _trajectory_parity_rows(base_final["case_level"]), traj_report["final"]
            )
            initial_rcsp = evaluate_fixed_final(
                model, final_batch, final_metadata, cfg, parity_base=base
            )
            initial_case_hash_equal = _canonical_hash(base_final["case_level"]) == _canonical_hash(
                initial_rcsp["case_level"]
            )
            initial_final = {
                **initial_rcsp["step_zero_base_parity"],
                "cases": 64,
                "case_level_metrics_gates_hash_equal": initial_case_hash_equal,
                "trajectory_final_metric_parity": trajectory_parity,
            }
            if not initial_final["verified"] or not initial_case_hash_equal:
                raise RuntimeError("zero-initialized RCSP fixed-final parity failed")

            trained = train_adapters(model, bank, cfg, output)
            if safe.state_hash(base.state_dict()) != base_state_hash:
                raise RuntimeError("RCSP training changed the frozen base model")
            validate_parameter_scope(model)
            final_rcsp = evaluate_fixed_final(model, final_batch, final_metadata, cfg)
            direction = direction_alignment(model, final_batch, final_metadata, cfg)
            support = support_projection_summary(
                final_rcsp["support_projection_case_level"]
            )
            base_summary = fixed_final_summary(base_final["case_level"])
            rcsp_summary = fixed_final_summary(final_rcsp["case_level"])
            comparison = baseline_comparison(base_summary, rcsp_summary)
            answer_base = {**base_summary, "case_level": base_final["case_level"]}
            answer_rcsp = {**rcsp_summary, "case_level": final_rcsp["case_level"]}
            answers = scientific_answers(answer_base, answer_rcsp, comparison)

            checkpoint_path = output / "adapter_final.pt"
            _save_adapter_checkpoint(
                checkpoint_path,
                model,
                traj_paths["diagnostic_latest.pt"],
                base_state_hash,
            )
            adapter_state_hash = safe.state_hash(model.adapter.state_dict())
            if loaded_probe_hash != probe_hash:
                raise RuntimeError("probe changed during RCSP experiment")
            if safe.state_hash(base.state_dict()) != base_state_hash:
                raise RuntimeError("final RCSP diagnostics changed the frozen base model")

        for name, digest in source_hashes.items():
            if group_audit.file_sha256(source_paths[name]) != digest:
                raise RuntimeError("frozen source changed during RCSP experiment")
        for name, digest in traj_hashes.items():
            if group_audit.file_sha256(traj_paths[name]) != digest:
                raise RuntimeError("immutable trajectory changed during RCSP experiment")

        report = {
            "schema": SCHEMA,
            "completed": True,
            "provenance": {
                "runtime_commit": runtime_commit,
                "source": source_metadata,
                "source_sha256_including_probe": source_hashes,
                "trajectory_commit": args.expected_trajectory_commit,
                "trajectory_directory": str(traj_dir),
                "trajectory_sha256": traj_hashes,
                "trajectory_final_state_sha256": traj_report["final_state_sha256"],
                "probe_sha256": probe_hash,
                "implementation_sha256": {
                    Path(path).name: group_audit.file_sha256(path)
                    for path in (
                        __file__,
                        m.__file__,
                        failure.__file__,
                        group_audit.__file__,
                        safe.__file__,
                        alignment.__file__,
                        scale_response.__file__,
                        Path(__file__).with_name("refiner_optimizer.py"),
                    )
                },
            },
            "base_checkpoint": {
                "path": str(traj_paths["diagnostic_latest.pt"]),
                "sha256": traj_hashes["diagnostic_latest.pt"],
                "state_sha256": base_state_hash,
                "completed_steps": 400,
                "retrained": False,
            },
            "base_model_frozen": True,
            "adapter_only_training": True,
            "adapter_initialization": initialization,
            "adapter_architecture": {
                "single_adapter": f"Conv1d({model.base.out.in_channels},75,kernel_size=1)",
                "cross_adapter": f"Conv1d({model.base.out.in_channels},75,kernel_size=1)",
                "hidden_feature_source": HIDDEN_CAPTURE_PROTOCOL,
                "width_conditioning": False,
                "attention": False,
                "bottleneck": False,
                "parameter_count": parameter_scope["adapter_parameters"],
            },
            "role_mapping": ROLE_MAPPING,
            "role_id_source": {
                "train": "existing_explicit_four_group_contract",
                "fixed_final": "explicit_split_role_bank_metadata",
                "width_used_for_role": False,
                "case_position_used_for_role": False,
            },
            "support_projection_protocol": {
                "protocol": SUPPORT_PROJECTION_PROTOCOL,
                "source": "production _refiner_decode_masks effective root_weight and joint_weight",
                "operation": "weight_gt_zero_only",
                "support_expansion": False,
                "soft_confidence_applied_to_adapter_before_decoder": False,
                "production_decoder_applies_soft_confidence_once": True,
            },
            "optimizer": {
                "family": "AdamW",
                "learning_rate": cfg.lr,
                "weight_decay": 1.0e-4,
                "gradient_clip_norm": 1.0,
                "checked_step_protocol": m.REFINER_UPDATE_PROTOCOL,
                "group_guard_relative_tolerance": cfg.product_refiner_group_guard_relative_tolerance,
                "group_guard_absolute_tolerance": cfg.product_refiner_group_guard_absolute_tolerance,
                "hyperparameter_search": False,
            },
            "optimizer_steps": STEPS,
            "accepted_steps": trained["accepted_steps"],
            "rollback_steps": trained["rollback_steps"],
            "initial_parity": {"train": initial_train, "fixed_final_64": initial_final},
            "train_summary": trained,
            "fixed_final_64": {
                "BASE": {"summary": base_summary, "case_level": base_final["case_level"]},
                "RCSP": {"summary": rcsp_summary, "case_level": final_rcsp["case_level"]},
                "evaluation_step": 400,
                "alpha": 1.0,
                "alpha_sweep_performed": False,
            },
            "baseline_comparison": comparison,
            "direction_alignment": direction,
            "support_projection_stats": support,
            "scientific_answers": answers,
            "parameter_update_scope": {
                **parameter_scope,
                "scope": PARAMETER_SCOPE,
                "base_state_unchanged": True,
                "adapter_state_sha256": adapter_state_hash,
                "adapter_checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": group_audit.file_sha256(checkpoint_path),
                },
            },
            "train_data_contract": {
                "frozen_reservoir": True,
                "transaction_schedule_unchanged": True,
                "held_out_new_position_used_for_training": False,
                "probe_used_for_step_zero_parity_and_fixed_final_only": True,
                "seed_search": False,
            },
            "checkpoint_selection_performed": False,
            "scale_selection_performed": False,
            "production_model_modified": False,
            "production_inference_modified": False,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
            "next_action": "review_rcsp_fixed_step_diagnostic_no_pilot",
        }
        report_path = output / "report.json"
        _exclusive_json(report_path, report)
        for name in (
            "seen/single_recording/10",
            "seen/single_recording/28",
            "new_position/single_recording/10",
            "new_position/single_recording/28",
            "seen/cross_event/10",
            "seen/cross_event/28",
            "new_position/cross_event/10",
            "new_position/cross_event/28",
        ):
            print(
                json.dumps(
                    {"stage": "rcsp_fixed_final_group", "group": name, **comparison["groups"][name]},
                    allow_nan=False,
                ),
                flush=True,
            )
        print("SCIENTIFIC ANSWERS", flush=True)
        print(json.dumps(answers, ensure_ascii=False, allow_nan=False), flush=True)
        print(
            json.dumps(
                {
                    "stage": "refiner_role_conditioned_support_projection_experiment_complete",
                    "report": str(report_path),
                    "optimizer_steps": STEPS,
                    "accepted_steps": trained["accepted_steps"],
                    "rollback_steps": trained["rollback_steps"],
                    "production_model_modified": False,
                    "scientific_acceptance": False,
                    "publish_allowed": False,
                    "pilot_allowed": False,
                },
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
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "production_model_modified": False,
                    "scientific_acceptance": False,
                    "publish_allowed": False,
                    "pilot_allowed": False,
                },
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument(
        "--expected-trajectory-commit", default=failure.TRAJECTORY_COMMIT
    )
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
