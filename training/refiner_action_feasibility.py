"""Observable action-feasibility evaluator and bounded local solver.

This module is a development diagnostic, not a training path.  It operates on
the existing product-manifold decoder and the repository's independent
boundary/physical audits.  It never accepts a hidden clean interior and never
mutates a production model.
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the server environment supplies torch
    torch = None

from motion_geometry.product_manifold import (
    PRODUCT_STATE_DIM,
    product_log_np,
    product_log_torch,
)
from training import motion_models as m

PROTOCOL_VERSION = "refiner_action_feasibility_dev_v3"
DECODER_PROTOCOL = "product_refiner_true_decoder_confidence_smoothing_taper_cap_v1"
METRIC_PROTOCOL = "observable_boundary_stage_physical_fixed_support_fidelity_v1"
ACTION_DIM = PRODUCT_STATE_DIM - 4
STATUS_VERIFIED_FEASIBLE = "VERIFIED_FEASIBLE"
STATUS_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
STATUS_NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
STATUS_INVALID_INPUT = "INVALID_INPUT"
VALID_STATUSES = frozenset(
    {
        STATUS_VERIFIED_FEASIBLE,
        STATUS_BUDGET_EXHAUSTED,
        STATUS_NUMERICAL_FAILURE,
        STATUS_INVALID_INPUT,
    }
)


@dataclasses.dataclass(frozen=True)
class FeasibilitySolverConfig:
    """Frozen solver controls written to the run manifest before execution."""

    max_iterations: int = 24
    initial_trust_radius: float = 0.25
    minimum_trust_radius: float = 0.015625
    trust_radius_shrink: float = 0.5
    backtracking_steps: int = 4
    minimum_edit_factors: tuple[float, ...] = (0.5, 0.25, 0.0)
    gradient_norm_floor: float = 1.0e-12
    comparison_tolerance: float = 1.0e-7
    finite_difference_probe_radius: float = 0.03125

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not 0 < self.minimum_trust_radius <= self.initial_trust_radius:
            raise ValueError("invalid trust-region radii")
        if not 0 < self.trust_radius_shrink < 1:
            raise ValueError("trust_radius_shrink must be in (0,1)")
        if self.backtracking_steps < 1:
            raise ValueError("backtracking_steps must be positive")
        if any(not 0.0 <= factor < 1.0 for factor in self.minimum_edit_factors):
            raise ValueError("minimum_edit_factors must be in [0,1)")
        if not math.isfinite(self.finite_difference_probe_radius) or self.finite_difference_probe_radius <= 0.0:
            raise ValueError("finite_difference_probe_radius must be positive and finite")


@dataclasses.dataclass
class ActionFeasibilityCase:
    """One explicitly registered repair case.

    ``reference`` is the observed repair input/bridge.  There is deliberately
    no clean target field: hidden clean data cannot enter deployment-triggered
    evaluation or solver acceptance.
    """

    case_id: str
    role: str
    width: int
    position_stratum: str
    split: str
    reference: np.ndarray
    seam: np.ndarray
    joint_mask: np.ndarray
    root_mask: np.ndarray
    cfg: Any
    boundary_role: str = ""
    contact_mask: np.ndarray | None = None
    condition: np.ndarray | None = None
    source_uid: str = ""
    recording_uid: str = ""
    left_source_uid: str = ""
    right_source_uid: str = ""
    left_recording_uid: str = ""
    right_recording_uid: str = ""
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.boundary_role:
            self.boundary_role = str(self.role)
        self.reference = np.asarray(self.reference, dtype=np.float32)
        self.seam = np.asarray(self.seam, dtype=np.float32)
        self.joint_mask = np.asarray(self.joint_mask, dtype=np.float32)
        self.root_mask = np.asarray(self.root_mask, dtype=np.float32)
        if self.contact_mask is None:
            self.contact_mask = np.zeros((len(self.reference), 4), dtype=np.float32)
        else:
            self.contact_mask = np.asarray(self.contact_mask, dtype=np.float32)
        if self.reference.ndim != 2 or self.reference.shape[1] != m.EDGE_DIM:
            raise ValueError(f"reference must have shape [T,{m.EDGE_DIM}]")
        if len(self.reference) < 4:
            raise ValueError("action-feasibility cases require at least four frames")
        if self.seam.shape not in {(len(self.reference),), (len(self.reference), 1)}:
            raise ValueError("seam must have shape [T] or [T,1]")
        if self.joint_mask.shape != (len(self.reference), m.NUM_JOINTS):
            raise ValueError("joint_mask must have shape [T,24]")
        if self.root_mask.shape not in {(len(self.reference),), (len(self.reference), 1)}:
            raise ValueError("root_mask must have shape [T] or [T,1]")
        if self.contact_mask.shape != (len(self.reference), 4):
            raise ValueError("contact_mask must have shape [T,4]")
        if not np.allclose(self.contact_mask, 0.0, atol=0.0, rtol=0.0):
            raise ValueError("contact_mask must be exactly zero in the geometry-only protocol")
        if self.condition is not None:
            self.condition = np.asarray(self.condition, dtype=np.float32)
            if self.condition.ndim != 2 or self.condition.shape[0] != len(self.reference):
                raise ValueError("condition must have shape [T,C]")
        for label, value in (("reference", self.reference), ("seam", self.seam),
                             ("joint_mask", self.joint_mask), ("root_mask", self.root_mask),
                             ("contact_mask", self.contact_mask)):
            if not np.isfinite(value).all():
                raise ValueError(f"{label} contains non-finite values")
        if self.condition is not None and not np.isfinite(self.condition).all():
            raise ValueError("condition contains non-finite values")
        if int(self.width) < 1:
            raise ValueError("width must be positive")
        if not str(self.split).strip():
            raise ValueError("split is required for leakage checks")

    @property
    def frames(self) -> int:
        return int(self.reference.shape[0])

    def manifest_identity(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "role": self.role,
            "boundary_role": self.boundary_role,
            "width": int(self.width),
            "position_stratum": self.position_stratum,
            "split": self.split,
            "source_uid": self.source_uid,
            "recording_uid": self.recording_uid,
            "left_source_uid": self.left_source_uid,
            "right_source_uid": self.right_source_uid,
            "left_recording_uid": self.left_recording_uid,
            "right_recording_uid": self.right_recording_uid,
        }


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for the true decoder and solver")
    return torch


def _mask_tensors(case: ActionFeasibilityCase, device: Any, dtype: Any) -> tuple[Any, Any, Any]:
    t = _require_torch()
    joint = t.as_tensor(case.joint_mask, device=device, dtype=dtype)[None]
    root = t.as_tensor(case.root_mask, device=device, dtype=dtype)[None]
    contact = t.as_tensor(case.contact_mask, device=device, dtype=dtype)[None]
    return joint, root, contact


def _normalised_action_norm_torch(action: Any, cfg: Any) -> Any:
    t = _require_torch()
    root_cap = max(float(cfg.product_refiner_root_cap_m), 1.0e-12)
    rotation_cap = max(float(cfg.product_refiner_rotation_cap_rad), 1.0e-12)
    root = action[..., :3] / root_cap
    joint = action[..., 3:] / rotation_cap
    return t.linalg.vector_norm(t.cat([root.reshape(-1), joint.reshape(-1)]))


def normalized_raw_action_norm(action: np.ndarray, cfg: Any) -> float:
    """Report raw action size normalized by configured root/rotation caps.

    This is not an FK displacement metric; decoded FK edit is reported
    separately by :func:`evaluate_action_candidate`.
    """
    value = np.asarray(action, dtype=np.float64)
    if value.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must end in {ACTION_DIM}")
    root_cap = max(float(cfg.product_refiner_root_cap_m), 1.0e-12)
    rotation_cap = max(float(cfg.product_refiner_rotation_cap_rad), 1.0e-12)
    root = value[..., :3] / root_cap
    joint = value[..., 3:] / rotation_cap
    return float(np.linalg.norm(np.concatenate([root.reshape(-1), joint.reshape(-1)])))


def decode_geometry_action_torch(
    reference: Any,
    raw_action: Any,
    joint_mask: Any,
    root_mask: Any,
    contact_mask: Any,
    cfg: Any,
    *,
    trace: dict[str, Any] | None = None,
) -> Any:
    """Decode a 75D action through the authoritative production decoder."""
    t = _require_torch()
    if reference.shape[-1] != m.EDGE_DIM:
        raise ValueError(f"reference must end in {m.EDGE_DIM}")
    if raw_action.shape[-1] != ACTION_DIM:
        raise ValueError(f"raw_action must end in {ACTION_DIM}")
    if reference.ndim == 2:
        reference = reference[None]
    if raw_action.ndim == 2:
        raw_action = raw_action[None]
    output = t.cat([t.zeros(raw_action.shape[:-1] + (4,), device=raw_action.device, dtype=raw_action.dtype), raw_action], dim=-1)
    if joint_mask.ndim == 2:
        joint_mask = joint_mask[None]
    if root_mask.ndim == 1 or (root_mask.ndim == 2 and root_mask.shape[-1] == 1 and reference.ndim == 3):
        root_mask = root_mask[None] if root_mask.ndim == 1 else root_mask
    if contact_mask.ndim == 2:
        contact_mask = contact_mask[None]
    # contact_mask is intentionally all zero in normal repair cases.  Keeping
    # it explicit makes a contact residual impossible to smuggle into the
    # geometry solver.
    return m._decode_product_refiner_output(
        reference,
        output,
        joint_mask,
        root_mask,
        contact_mask,
        cfg,
        trace=trace,
    )


def decode_geometry_action(case: ActionFeasibilityCase, raw_action: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Numpy wrapper around the true decoder plus cap/edit diagnostics."""
    t = _require_torch()
    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (case.frames, ACTION_DIM):
        raise ValueError(f"raw action must have shape {(case.frames, ACTION_DIM)}")
    if not np.isfinite(action).all():
        raise ValueError("raw action contains non-finite values")
    device = getattr(case.cfg, "device", "cpu")
    if str(device).startswith("cuda") and not t.cuda.is_available():
        device = "cpu"
    trace: dict[str, Any] = {}
    with t.enable_grad():
        reference = t.as_tensor(case.reference, dtype=t.float32, device=device)[None]
        raw = t.as_tensor(action, dtype=t.float32, device=device)[None]
        joint, root, contact = _mask_tensors(case, device, t.float32)
        decoded = decode_geometry_action_torch(reference, raw, joint, root, contact, case.cfg, trace=trace)
    candidate = decoded[0].detach().cpu().numpy().astype(np.float32)
    def trace_array(name: str, fallback: np.ndarray) -> np.ndarray:
        value = trace.get(name, fallback)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    applied = trace_array("after_cap", np.zeros_like(action))[0]
    pre_cap = trace_array("after_taper", trace.get("after_mask", applied))[0]
    pre_root = np.linalg.norm(pre_cap[..., :3], axis=-1)
    pre_joint = np.linalg.norm(pre_cap[..., 3:].reshape(case.frames, m.NUM_JOINTS, 3), axis=-1)
    root_cap = float(case.cfg.product_refiner_root_cap_m)
    rotation_cap = float(case.cfg.product_refiner_rotation_cap_rad)
    cap_tol = 1.0e-6
    saturation = {
        "root_fraction": float(np.mean(pre_root > root_cap + cap_tol)),
        "rotation_fraction": float(np.mean(pre_joint > rotation_cap + cap_tol)),
        "root_frames": int(np.sum(pre_root > root_cap + cap_tol)),
        "rotation_joint_frames": int(np.sum(pre_joint > rotation_cap + cap_tol)),
    }
    decoded_edit = product_log_np(case.reference, candidate)
    active = (np.asarray(case.root_mask).reshape(case.frames, -1).max(axis=-1) > 0)
    active |= np.asarray(case.joint_mask).max(axis=-1) > 0
    inactive_edit = np.linalg.norm(decoded_edit[~active], axis=-1) if np.any(~active) else np.zeros(0)
    fk_edit = np.linalg.norm(m.fk_24_np(candidate) - m.fk_24_np(case.reference), axis=-1)
    detail = {
        "raw_action_norm_normalized": normalized_raw_action_norm(action, case.cfg),
        "decoded_edit_norm_p95": float(np.percentile(np.linalg.norm(decoded_edit, axis=-1), 95)),
        "decoded_edit_norm_max": float(np.max(np.linalg.norm(decoded_edit, axis=-1))),
        "decoded_fk_edit_m_p95": float(np.percentile(fk_edit, 95)),
                "decoded_fk_edit_m_max": float(np.max(fk_edit)),
        "cap_saturation": saturation,
        "support_outside_edit_max": float(np.max(inactive_edit)) if inactive_edit.size else 0.0,
        "contact_residual_max": float(np.max(np.abs(candidate[..., :4] - case.reference[..., :4]))),
        "decoder_protocol": DECODER_PROTOCOL,
    }
    return candidate, detail


def _failed_gate(schema: str, reason: str) -> dict[str, Any]:
    return {"schema": schema, "accepted": False, "reasons": [reason]}


def _observable_excesses(observable: Mapping[str, Any], cfg: Any) -> dict[str, float]:
    """Compute authoritative-scale endpoint, temporal, and jerk excesses."""
    before = observable.get("before", {})
    after = observable.get("after", {})
    if not before or not after:
        return {"endpoint": float("inf"), "temporal": float("inf"), "jerk": float("inf")}
    try:
        endpoint_limit = float(before["endpoint_velocity_jump_mps"]) * (
            1.0 - float(cfg.checkpoint_validation_min_endpoint_repair_gain)
        )
        temporal_limit = float(before["temporal_energy"]) * (
            1.0 - float(cfg.checkpoint_validation_min_temporal_repair_gain)
        )
        endpoint_denominator = max(abs(endpoint_limit), 1.0e-6)
        temporal_denominator = max(abs(temporal_limit), 1.0e-6)
        endpoint = max(
            0.0,
            float(after["endpoint_velocity_jump_mps"]) - endpoint_limit,
        ) / endpoint_denominator
        temporal = max(
            0.0,
            float(after["temporal_energy"]) - temporal_limit,
        ) / temporal_denominator
        jerk_limit = float(before["seam_jerk_mps3"]) * 1.02 + 1.0e-6
        jerk = max(0.0, float(after["seam_jerk_mps3"]) - jerk_limit)
        return {
            "endpoint": float(endpoint),
            "temporal": float(temporal),
            "jerk": float(jerk / max(abs(jerk_limit), 1.0)),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"endpoint": float("inf"), "temporal": float("inf"), "jerk": float("inf")}


def _proxy_residual(observable: Mapping[str, Any], cfg: Any) -> tuple[int, float]:
    """Return a solver residual on the same scale as the observable gate."""
    excesses = _observable_excesses(observable, cfg)
    failures = int(not bool(observable.get("endpoint_accepted", False))) + int(
        not bool(observable.get("temporal_accepted", False))
    )
    residual = sum(excesses.values())
    return failures, float(residual)


def evaluate_action_candidate(case: ActionFeasibilityCase, raw_action: np.ndarray, *, label: str = "candidate") -> dict[str, Any]:
    """Run the complete independent acceptance contract for one action."""
    started = time.perf_counter()
    action = np.asarray(raw_action, dtype=np.float32)
    if action.shape != (case.frames, ACTION_DIM) or not np.isfinite(action).all():
        return {
            "label": label,
            "joint_pass": False,
            "failure_reasons": ["invalid_action_shape_or_nonfinite"],
            "invalid_input": True,
            "hidden_clean_used": False,
        }
    try:
        candidate, action_detail = decode_geometry_action(case, action)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "label": label,
            "joint_pass": False,
            "failure_reasons": [f"decoder_failure:{type(exc).__name__}"],
            "invalid_input": False,
            "numerical_failure": True,
            "hidden_clean_used": False,
        }
    try:
        reference_audit = m._safe_validation_audit(case.reference, case.cfg, role="feasibility_reference", support_policy="source_observation")
        candidate_audit = m._safe_validation_audit(candidate, case.cfg, role="feasibility_candidate", support_policy="source_observation")
        physical_stage = m.evaluate_stage_candidate(
            reference_audit,
            candidate_audit,
            require_repair_gain=False,
            ignored_layers=("long_horizon_root_drift",),
        )
        fidelity_stage = m.evaluate_stage_reference_fidelity(reference_audit, candidate_audit)
        absolute_physical = m.evaluate_physical_audit(candidate_audit)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        physical_stage = _failed_gate("stage_physical_quality_v1", f"physical_audit_error:{type(exc).__name__}")
        fidelity_stage = _failed_gate("stage_reference_fidelity_v1", f"fidelity_audit_error:{type(exc).__name__}")
        absolute_physical = {"ok": False, "reasons": [f"absolute_physical_audit_error:{type(exc).__name__}"]}
        reference_audit = {"schema": "invalid"}
        candidate_audit = {"schema": "invalid"}
    try:
        observable = m._observable_boundary_audit(candidate, case.reference, case.seam, case.cfg)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        observable = _failed_gate(m.BOUNDARY_PROTOCOL, f"observable_audit_error:{type(exc).__name__}")
        observable["hidden_clean_used"] = False
    try:
        fixed_support = m._fixed_support_stage_gate(
            case.reference,
            candidate,
            case.cfg,
            before_audit=reference_audit,
            after_audit=candidate_audit,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        fixed_support = _failed_gate("fixed_reference_support_v1", f"fixed_support_audit_error:{type(exc).__name__}")
        fixed_support["independent_support_diagnostic"] = {"accepted": False, "reasons": [str(exc)]}
    support_outside = float(action_detail["support_outside_edit_max"]) <= 1.0e-6
    contact_fixed = float(action_detail["contact_residual_max"]) <= 1.0e-6
    endpoint_pass = bool(observable.get("endpoint_accepted", False))
    temporal_pass = bool(observable.get("temporal_accepted", False))
    jerk_pass = bool(observable.get("jerk_non_regression", False))
    fidelity_pass = bool(observable.get("reference_fidelity_accepted", False) and fidelity_stage.get("accepted", False))
    physical_pass = bool(physical_stage.get("accepted", False))
    fixed_support_pass = bool(fixed_support.get("accepted", False))
    observable_pass = bool(observable.get("accepted", False))
    hidden_clean_used = bool(observable.get("hidden_clean_used", False))
    finite_pass = bool(np.isfinite(candidate).all())
    reasons: list[str] = []
    if not finite_pass:
        reasons.append("candidate_nonfinite")
    if not contact_fixed:
        reasons.append("contact_residual_not_fixed_zero")
    if not support_outside:
        reasons.append("support_outside_edit")
    if not observable_pass:
        reasons.extend(f"observable:{r}" for r in observable.get("reasons", []))
    if not physical_pass:
        reasons.extend(f"physical:{r}" for r in physical_stage.get("reasons", []))
    if not fixed_support_pass:
        reasons.extend(f"fixed_support:{r}" for r in fixed_support.get("reasons", []))
    if not fidelity_pass:
        reasons.extend(f"fidelity:{r}" for r in fidelity_stage.get("reasons", []))
        if not observable.get("reference_fidelity_accepted", False):
            reasons.append("fidelity:observable_reference_fidelity_rejected")
    if hidden_clean_used:
        reasons.append("hidden_clean_used")
    observable_excesses = _observable_excesses(observable, case.cfg)
    failure_count, residual = _proxy_residual(observable, case.cfg)
    reasons = list(dict.fromkeys(reasons))
    joint_pass = bool(
        endpoint_pass
        and temporal_pass
        and jerk_pass
        and physical_pass
        and fidelity_pass
        and finite_pass
        and fixed_support_pass
        and contact_fixed
        and support_outside
        and not hidden_clean_used
    )
    return {
        "label": label,
        "endpoint_pass": endpoint_pass,
        "temporal_pass": temporal_pass,
        "jerk_pass": jerk_pass,
        "physical_pass": physical_pass,
        "fidelity_pass": fidelity_pass,
        "finite_pass": finite_pass,
        "joint_pass": joint_pass,
        "failure_reasons": reasons,
        "observable_boundary": observable,
        "physical_stage": physical_stage,
        "fixed_reference_support": fixed_support,
        "reference_fidelity": fidelity_stage,
        "absolute_physical_diagnostic": absolute_physical,
        "reference_audit": reference_audit,
        "candidate_audit": candidate_audit,
        "action": action_detail,
        "proxy_failure_count": int(failure_count + int(not physical_pass) + int(not fixed_support_pass) + int(not fidelity_pass)),
        "proxy_residual": float(residual),
        "solver_observable_excess": observable_excesses,
        "hidden_clean_used": hidden_clean_used,
        "invalid_input": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _proxy_loss_components(
    case: ActionFeasibilityCase, raw_action: Any
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return differentiable objective components and their scalar diagnostics."""
    t = _require_torch()
    device, dtype = raw_action.device, raw_action.dtype
    reference = t.as_tensor(case.reference, device=device, dtype=dtype)[None]
    seam = t.as_tensor(case.seam, device=device, dtype=dtype)
    if seam.ndim == 1:
        seam = seam[None, :, None]
    else:
        seam = seam[None] if seam.ndim == 2 else seam
    joint, root, contact = _mask_tensors(case, device, dtype)
    candidate = decode_geometry_action_torch(reference, raw_action, joint, root, contact, case.cfg)
    joints = m.fk_24_torch(t.cat([reference, candidate], dim=0))
    metrics = m.boundary_metrics_torch(t.cat([joints[:1], joints[1:]], dim=0), t.cat([seam, seam], dim=0), case.cfg.fps)
    before = {key: metrics[key][0].detach() for key in metrics if key != "valid"}
    after = {key: metrics[key][1] for key in metrics if key != "valid"}
    endpoint_limit = before["endpoint_velocity_jump_mps"] * (1.0 - float(case.cfg.checkpoint_validation_min_endpoint_repair_gain))
    temporal_limit = before["temporal_energy"] * (1.0 - float(case.cfg.checkpoint_validation_min_temporal_repair_gain))
    jerk_limit = before["seam_jerk_mps3"] * 1.02 + 1.0e-6
    endpoint_excess = t.relu(after["endpoint_velocity_jump_mps"] - endpoint_limit) / endpoint_limit.abs().clamp_min(1.0e-6)
    temporal_excess = t.relu(after["temporal_energy"] - temporal_limit) / temporal_limit.abs().clamp_min(1.0e-6)
    jerk_excess = t.relu(after["seam_jerk_mps3"] - jerk_limit) / jerk_limit.abs().clamp_min(1.0)
    fidelity = t.mean(t.abs(product_log_torch(reference, candidate))) / max(float(case.cfg.checkpoint_validation_max_refiner_product_log_l1), 1.0e-6)
    losses = {
        "endpoint": endpoint_excess,
        "temporal": temporal_excess,
        "joint": endpoint_excess + temporal_excess + jerk_excess + 0.05 * fidelity,
    }
    values = {
        "endpoint_excess": float(endpoint_excess.detach().cpu()),
        "temporal_excess": float(temporal_excess.detach().cpu()),
        "jerk_excess": float(jerk_excess.detach().cpu()),
        "fidelity_proxy": float(fidelity.detach().cpu()),
    }
    return losses, values


def _proxy_loss(
    case: ActionFeasibilityCase,
    raw_action: Any,
    *,
    objective: str = "joint",
) -> tuple[Any, dict[str, Any]]:
    """Return one differentiable proxy objective and component diagnostics."""
    losses, values = _proxy_loss_components(case, raw_action)
    if objective not in losses:
        raise ValueError(f"unknown feasibility proxy objective: {objective}")
    return losses[objective], {**values, "objective": objective}


def _solver_key(evaluation: Mapping[str, Any], action: np.ndarray) -> tuple[int, float, float]:
    return (
        int(evaluation.get("proxy_failure_count", 10**6)),
        float(evaluation.get("proxy_residual", float("inf"))),
        normalized_raw_action_norm(action, evaluation["_cfg"]),
    )


def _solver_key_payload(key: tuple[int, float, float]) -> dict[str, Any]:
    """Make a JSON-safe, named representation of the candidate comparator."""
    return {
        "failure_count": int(key[0]),
        "proxy_residual": float(key[1]) if math.isfinite(key[1]) else None,
        "action_norm_normalized": float(key[2]) if math.isfinite(key[2]) else None,
    }


def _hard_constraint_failures(
    evaluation: Mapping[str, Any], *, preserve_endpoint: bool = False
) -> list[str]:
    """Return constraints that must remain satisfied during restoration."""
    failures: list[str] = []
    if not bool(evaluation.get("physical_pass", False)):
        failures.extend(
            f"physical:{reason}"
            for reason in (evaluation.get("physical_stage", {}) or {}).get("reasons", [])
        )
        if not any(reason.startswith("physical:") for reason in failures):
            failures.append("physical:stage_not_accepted")
    fixed_support = evaluation.get("fixed_reference_support", {}) or {}
    if not bool(fixed_support.get("accepted", False)):
        failures.extend(
            f"fixed_support:{reason}"
            for reason in fixed_support.get("reasons", [])
        )
        if not any(reason.startswith("fixed_support:") for reason in failures):
            failures.append("fixed_support:stage_not_accepted")
    if not bool(evaluation.get("fidelity_pass", False)):
        failures.extend(
            f"fidelity:{reason}"
            for reason in (evaluation.get("reference_fidelity", {}) or {}).get("reasons", [])
        )
        if not any(reason.startswith("fidelity:") for reason in failures):
            failures.append("fidelity:stage_not_accepted")
    if not bool(evaluation.get("finite_pass", False)):
        failures.append("candidate_nonfinite")
    action = evaluation.get("action", {}) or {}
    if float(action.get("contact_residual_max", float("inf"))) > 1.0e-6:
        failures.append("contact_residual_not_fixed_zero")
    if float(action.get("support_outside_edit_max", float("inf"))) > 1.0e-6:
        failures.append("support_outside_edit")
    if bool(evaluation.get("hidden_clean_used", False)):
        failures.append("hidden_clean_used")
    if preserve_endpoint and not bool(evaluation.get("endpoint_pass", False)):
        failures.append("observable:endpoint_not_preserved")
    return list(dict.fromkeys(failures))


def _solver_stage(evaluation: Mapping[str, Any]) -> str:
    """Select the active restoration stage from the authoritative gate."""
    if not bool(evaluation.get("endpoint_pass", False)):
        return "endpoint"
    if not bool(evaluation.get("temporal_pass", False)):
        return "temporal"
    return "joint"


def _stage_key(
    evaluation: Mapping[str, Any], action: np.ndarray, stage: str
) -> tuple[int, float, float, float, float, float]:
    """Compare candidates while preserving hard constraints and stage order."""
    preserve_endpoint = stage == "temporal"
    hard_failures = _hard_constraint_failures(
        evaluation, preserve_endpoint=preserve_endpoint
    )
    excess = evaluation.get("solver_observable_excess", {}) or {}
    endpoint = float(excess.get("endpoint", float("inf")))
    temporal = float(excess.get("temporal", float("inf")))
    jerk = float(excess.get("jerk", float("inf")))
    residual = float(evaluation.get("proxy_residual", float("inf")))
    if stage == "temporal":
        primary, secondary = temporal, endpoint
    else:
        primary, secondary = endpoint, temporal
    return (
        int(bool(hard_failures)),
        primary,
        secondary,
        jerk,
        residual,
        normalized_raw_action_norm(action, evaluation["_cfg"]),
    )


def _stage_key_payload(key: tuple[int, float, float, float, float, float]) -> dict[str, Any]:
    return {
        "hard_constraint_violation": bool(key[0]),
        "primary_excess": float(key[1]) if math.isfinite(key[1]) else None,
        "secondary_excess": float(key[2]) if math.isfinite(key[2]) else None,
        "jerk_excess": float(key[3]) if math.isfinite(key[3]) else None,
        "proxy_residual": float(key[4]) if math.isfinite(key[4]) else None,
        "action_norm_normalized": float(key[5]) if math.isfinite(key[5]) else None,
    }


def _stage_key_rejection_reason(
    current_key: tuple[int, float, float, float, float, float],
    trial_key: tuple[int, float, float, float, float, float],
    hard_failures: list[str],
) -> str:
    if hard_failures:
        return "hard_constraint_violation"
    if trial_key[1] > current_key[1]:
        return "stage_primary_excess_not_reduced"
    if trial_key[1] == current_key[1] and trial_key[2] > current_key[2]:
        return "stage_secondary_excess_not_reduced"
    if trial_key >= current_key:
        return "stage_key_not_strictly_better"
    return "stage_key_improved"


def _solver_objective_order(stage: str) -> tuple[str, ...]:
    if stage == "endpoint":
        return ("endpoint", "joint", "temporal")
    if stage == "temporal":
        return ("temporal", "joint", "endpoint")
    return ("joint", "endpoint", "temporal")


def _finite_difference_reachability(
    case: ActionFeasibilityCase,
    current_action: np.ndarray,
    directions: Mapping[str, np.ndarray],
    stage: str,
    probe_radius: float,
) -> list[dict[str, Any]]:
    """Probe both signs of each objective direction without changing state."""
    probes: list[dict[str, Any]] = []
    for objective, direction in directions.items():
        for sign in (-1, 1):
            probe_action = current_action + (
                float(sign) * float(probe_radius) * direction
            ).astype(np.float32)
            evaluation = evaluate_action_candidate(
                case,
                probe_action,
                label=f"finite_difference_{objective}_{sign:+d}",
            )
            evaluation["_cfg"] = case.cfg
            observable = evaluation.get("observable_boundary", {}) or {}
            fixed_support = evaluation.get("fixed_reference_support", {}) or {}
            hard_failures = _hard_constraint_failures(
                evaluation, preserve_endpoint=stage == "temporal"
            )
            observable_improved = any(
                float(observable.get(name)) > 0.0
                for name in ("endpoint_gain", "temporal_gain")
                if observable.get(name) is not None
            )
            observable_pass = bool(
                evaluation.get("endpoint_pass", False)
                and evaluation.get("temporal_pass", False)
            )
            physical_pass = bool(
                evaluation.get("physical_pass", False)
                and fixed_support.get("accepted", False)
                and evaluation.get("fidelity_pass", False)
            )
            probes.append(
                {
                    "stage": stage,
                    "objective": objective,
                    "sign": int(sign),
                    "probe_radius": float(probe_radius),
                    "action_delta_norm_normalized": normalized_raw_action_norm(
                        probe_action - current_action, case.cfg
                    ),
                    "endpoint_gain": (
                        observable.get("endpoint_gain")
                    ),
                    "temporal_gain": (
                        observable.get("temporal_gain")
                    ),
                    "endpoint_pass": bool(evaluation.get("endpoint_pass", False)),
                    "temporal_pass": bool(evaluation.get("temporal_pass", False)),
                    "physical_pass": bool(evaluation.get("physical_pass", False)),
                    "fixed_support_pass": bool(fixed_support.get("accepted", False)),
                    "fidelity_pass": bool(evaluation.get("fidelity_pass", False)),
                    "observable_improved": bool(observable_improved),
                    "observable_physical_feasible": bool(
                        observable_pass and physical_pass and not hard_failures
                    ),
                    "hard_constraint_failures": hard_failures,
                    "failure_reasons": list(evaluation.get("failure_reasons", [])),
                }
            )
    return probes


def _reachability_diagnosis(probes: list[Mapping[str, Any]]) -> str:
    if any(bool(probe.get("observable_physical_feasible", False)) for probe in probes):
        return "search_direction_or_acceptance_mismatch"
    if any(
        bool(probe.get("observable_improved", False))
        and bool(probe.get("hard_constraint_failures"))
        for probe in probes
    ):
        return "physical_constraint_blocks_observable_repair"
    return "local_unreachable_under_finite_difference_probe"


def _solver_trial_payload(
    evaluation: Mapping[str, Any],
    *,
    stage: str,
    objective: str,
    current_stage_key: tuple[int, float, float, float, float, float],
    trial_stage_key: tuple[int, float, float, float, float, float],
    backtrack: int,
    current_key: tuple[int, float, float],
    trial_key: tuple[int, float, float],
    action_delta_norm: float,
) -> dict[str, Any]:
    """Capture compact authoritative evidence for one trial action."""
    observable = evaluation.get("observable_boundary", {}) or {}
    action_detail = evaluation.get("action", {}) or {}
    absolute_physical = evaluation.get("absolute_physical_diagnostic", {}) or {}
    fixed_support = evaluation.get("fixed_reference_support", {}) or {}
    hard_failures = _hard_constraint_failures(
        evaluation, preserve_endpoint=stage == "temporal"
    )
    payload = {
        "stage": stage,
        "objective": objective,
        "backtrack": int(backtrack),
        "solver_key": _solver_key_payload(trial_key),
        "current_solver_key": _solver_key_payload(current_key),
        "stage_key": _stage_key_payload(trial_stage_key),
        "current_stage_key": _stage_key_payload(current_stage_key),
        "key_improved": bool(trial_stage_key < current_stage_key),
        "hard_constraint_failures": hard_failures,
        "action_delta_norm_normalized": float(action_delta_norm),
        "joint_pass": bool(evaluation.get("joint_pass", False)),
        "failure_reasons": list(evaluation.get("failure_reasons", [])),
        "endpoint_pass": bool(evaluation.get("endpoint_pass", False)),
        "temporal_pass": bool(evaluation.get("temporal_pass", False)),
        "jerk_pass": bool(evaluation.get("jerk_pass", False)),
        "physical_pass": bool(evaluation.get("physical_pass", False)),
        "fixed_support_pass": bool(fixed_support.get("accepted", False)),
        "fidelity_pass": bool(evaluation.get("fidelity_pass", False)),
        "absolute_physical_ok": bool(absolute_physical.get("ok", False)),
        "absolute_physical_reasons": list(absolute_physical.get("reasons", [])),
        "support_outside_edit_max": action_detail.get("support_outside_edit_max"),
        "contact_residual_max": action_detail.get("contact_residual_max"),
        "proxy_failure_count": int(evaluation.get("proxy_failure_count", 10**6)),
        "proxy_residual": (
            float(evaluation["proxy_residual"])
            if math.isfinite(float(evaluation.get("proxy_residual", float("inf"))))
            else None
        ),
        "endpoint_gain": observable.get("endpoint_gain"),
        "temporal_gain": observable.get("temporal_gain"),
        "endpoint_accepted": bool(observable.get("endpoint_accepted", False)),
        "temporal_accepted": bool(observable.get("temporal_accepted", False)),
    }
    if trial_stage_key >= current_stage_key:
        payload["rejected_reason"] = _stage_key_rejection_reason(
            current_stage_key, trial_stage_key, hard_failures
        )
    return payload


@dataclasses.dataclass
class SolverResult:
    status: str
    returned_action: np.ndarray
    returned_motion: np.ndarray
    rollback: bool
    initial_evaluation: dict[str, Any]
    final_evaluation: dict[str, Any]
    iterations: list[dict[str, Any]]
    detail: dict[str, Any]


def solve_action_feasibility(
    case: ActionFeasibilityCase,
    *,
    initial_action: np.ndarray | None = None,
    solver_config: FeasibilitySolverConfig | None = None,
) -> SolverResult:
    """Run bounded restoration, then minimum-edit search, with rollback."""
    controls = solver_config or FeasibilitySolverConfig()
    controls.validate()
    zero = np.zeros((case.frames, ACTION_DIM), dtype=np.float32)
    start_action = zero if initial_action is None else np.asarray(initial_action, dtype=np.float32).copy()
    if start_action.shape != zero.shape or not np.isfinite(start_action).all():
        invalid = {"joint_pass": False, "failure_reasons": ["invalid_initial_action"], "_cfg": case.cfg}
        return SolverResult(STATUS_INVALID_INPUT, zero, case.reference.copy(), True, invalid, invalid, [], {"rollback_reason": "invalid_initial_action"})
    initial = evaluate_action_candidate(case, start_action, label="initial")
    initial["_cfg"] = case.cfg
    if initial.get("joint_pass", False):
        current_action = start_action
        current_eval = initial
        iterations: list[dict[str, Any]] = []
        for index, factor in enumerate(controls.minimum_edit_factors):
            trial_action = current_action * float(factor)
            trial = evaluate_action_candidate(case, trial_action, label=f"minimum_edit_{index}")
            trial["_cfg"] = case.cfg
            iterations.append({"phase": "minimum_edit", "index": index, "factor": float(factor), "joint_pass": bool(trial.get("joint_pass", False))})
            if trial.get("joint_pass", False) and normalized_raw_action_norm(trial_action, case.cfg) < normalized_raw_action_norm(current_action, case.cfg):
                current_action, current_eval = trial_action, trial
        current_eval.pop("_cfg", None)
        return SolverResult(STATUS_VERIFIED_FEASIBLE, current_action, decode_geometry_action(case, current_action)[0], False, initial, current_eval, iterations, {"minimum_edit_completed": True})

    t = _require_torch()
    current_action = start_action.copy()
    current_eval = initial
    iterations = []
    radius = float(controls.initial_trust_radius)
    status = STATUS_BUDGET_EXHAUSTED
    detail: dict[str, Any] = {
        "feasibility_restoration": True,
        "minimum_edit_completed": False,
        "hard_constraint_gradient_coverage": [
            "observable_boundary_proxy_directions",
        ],
        "hard_constraint_enforcement": [
            "physical_stage_hard_filter",
            "fixed_support_stage_hard_filter",
            "fidelity_stage_hard_filter",
        ],
        "finite_difference_probe_radius": float(
            controls.finite_difference_probe_radius
        ),
    }
    last_reachability_diagnosis = None
    for step_index in range(controls.max_iterations):
        try:
            device = getattr(case.cfg, "device", "cpu")
            if str(device).startswith("cuda") and not t.cuda.is_available():
                device = "cpu"
            stage = _solver_stage(current_eval)
            action_tensor = (
                t.as_tensor(current_action, dtype=t.float64, device=device)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            losses, proxy_components = _proxy_loss_components(case, action_tensor)
            objective_order = _solver_objective_order(stage)
            directions: dict[str, np.ndarray] = {}
            direction_diagnostics: list[dict[str, Any]] = []
            for objective in objective_order:
                gradient = t.autograd.grad(
                    losses[objective],
                    action_tensor,
                    allow_unused=True,
                    retain_graph=True,
                )[0]
                if gradient is None:
                    direction_diagnostics.append(
                        {
                            "objective": objective,
                            "available": False,
                            "reason": "gradient_unavailable",
                        }
                    )
                    continue
                gradient_norm = float(t.linalg.vector_norm(gradient).detach().cpu())
                if not math.isfinite(gradient_norm) or gradient_norm <= controls.gradient_norm_floor:
                    direction_diagnostics.append(
                        {
                            "objective": objective,
                            "available": False,
                            "gradient_norm": gradient_norm,
                            "reason": "zero_constraint_gradient",
                        }
                    )
                    continue
                direction = -gradient / max(gradient_norm, controls.gradient_norm_floor)
                direction_norm = float(
                    _normalised_action_norm_torch(direction, case.cfg)
                    .detach()
                    .cpu()
                )
                if not math.isfinite(direction_norm) or direction_norm <= controls.gradient_norm_floor:
                    direction_diagnostics.append(
                        {
                            "objective": objective,
                            "available": False,
                            "gradient_norm": gradient_norm,
                            "reason": "zero_normalized_gradient_direction",
                        }
                    )
                    continue
                direction = direction / direction_norm
                directions[objective] = direction.detach().cpu().numpy().astype(np.float32)
                direction_diagnostics.append(
                    {
                        "objective": objective,
                        "available": True,
                        "gradient_norm": gradient_norm,
                        "direction_norm": direction_norm,
                    }
                )
            if not directions:
                status = STATUS_NUMERICAL_FAILURE
                detail["reason"] = "no_available_proxy_direction"
                break

            probe_radius = min(
                float(radius), float(controls.finite_difference_probe_radius)
            )
            reachability = _finite_difference_reachability(
                case,
                current_action,
                directions,
                stage,
                probe_radius,
            )
            reachability_diagnosis = _reachability_diagnosis(reachability)
            last_reachability_diagnosis = reachability_diagnosis
            current_stage_key = _stage_key(current_eval, current_action, stage)
            current_solver_key = _solver_key(current_eval, current_action)
            trial_diagnostics: list[dict[str, Any]] = []
            best = None
            for objective in objective_order:
                direction = directions.get(objective)
                if direction is None:
                    continue
                direction_meta = next(
                    item
                    for item in direction_diagnostics
                    if item["objective"] == objective
                )
                for backtrack in range(controls.backtracking_steps):
                    factor = 0.5 ** backtrack
                    trial_action = current_action + (
                        direction * (radius * factor)
                    ).astype(np.float32)
                    trial = evaluate_action_candidate(
                        case,
                        trial_action,
                        label=f"iteration_{step_index}_{objective}_trial_{backtrack}",
                    )
                    trial["_cfg"] = case.cfg
                    trial_key = _solver_key(trial, trial_action)
                    trial_stage_key = _stage_key(trial, trial_action, stage)
                    trial_payload = _solver_trial_payload(
                        trial,
                        stage=stage,
                        objective=objective,
                        current_stage_key=current_stage_key,
                        trial_stage_key=trial_stage_key,
                        backtrack=backtrack,
                        current_key=current_solver_key,
                        trial_key=trial_key,
                        action_delta_norm=normalized_raw_action_norm(
                            trial_action - current_action, case.cfg
                        ),
                    )
                    trial_payload["gradient_norm"] = direction_meta.get(
                        "gradient_norm"
                    )
                    trial_diagnostics.append(trial_payload)
                    hard_failures = _hard_constraint_failures(
                        trial, preserve_endpoint=stage == "temporal"
                    )
                    if (
                        not hard_failures
                        and trial_stage_key < current_stage_key
                        and (best is None or trial_stage_key < best[0])
                    ):
                        best = (
                            trial_stage_key,
                            trial_action,
                            trial,
                            objective,
                            backtrack,
                        )

            accepted = best is not None
            record = {
                "iteration": int(step_index),
                "stage": stage,
                "objective_order": list(objective_order),
                "radius": float(radius),
                "proxy_loss": float(losses["joint"].detach().cpu()),
                "proxy_components": proxy_components,
                "gradient_directions": direction_diagnostics,
                "finite_difference_reachability": reachability,
                "reachability_diagnosis": reachability_diagnosis,
                "accepted": bool(accepted),
                "current_solver_key": _solver_key_payload(current_solver_key),
                "current_stage_key": _stage_key_payload(current_stage_key),
                "trial_diagnostics": trial_diagnostics,
            }
            if accepted and best is not None:
                (
                    selected_stage_key,
                    current_action,
                    current_eval,
                    selected_objective,
                    backtrack,
                ) = best
                radius = min(
                    float(controls.initial_trust_radius),
                    radius / max(controls.trust_radius_shrink, 1.0e-6),
                )
                record["selected_objective"] = selected_objective
                record["selected_backtrack"] = int(backtrack)
                record["selected_stage_key"] = _stage_key_payload(selected_stage_key)
                record["selected_solver_key"] = _solver_key_payload(
                    _solver_key(current_eval, current_action)
                )
                record["joint_pass"] = bool(current_eval.get("joint_pass", False))
                if current_eval.get("joint_pass", False):
                    # A feasible candidate is subjected to the same fixed
                    # minimum-edit schedule as an initially feasible point.
                    for index, factor in enumerate(controls.minimum_edit_factors):
                        trial_action = current_action * float(factor)
                        trial = evaluate_action_candidate(
                            case,
                            trial_action,
                            label=f"minimum_edit_{step_index}_{index}",
                        )
                        trial["_cfg"] = case.cfg
                        iterations.append(
                            {
                                "phase": "minimum_edit",
                                "index": index,
                                "factor": float(factor),
                                "joint_pass": bool(trial.get("joint_pass", False)),
                            }
                        )
                        if (
                            trial.get("joint_pass", False)
                            and normalized_raw_action_norm(trial_action, case.cfg)
                            < normalized_raw_action_norm(current_action, case.cfg)
                        ):
                            current_action, current_eval = trial_action, trial
                    detail["minimum_edit_completed"] = True
                    status = STATUS_VERIFIED_FEASIBLE
                    iterations.append(record)
                    break
            else:
                radius *= controls.trust_radius_shrink
                record["radius_after_rejection"] = float(radius)
                record["rejection_reason"] = reachability_diagnosis
                detail["last_reachability_diagnosis"] = reachability_diagnosis
            iterations.append(record)
            if radius < controls.minimum_trust_radius:
                detail["reason"] = "trust_region_exhausted"
                break
        except (ValueError, RuntimeError, FloatingPointError, torch.linalg.LinAlgError) as exc:
            status = STATUS_NUMERICAL_FAILURE
            detail["reason"] = f"numerical_failure:{type(exc).__name__}"
            break

    if status != STATUS_VERIFIED_FEASIBLE:
        detail.setdefault("reason", "budget_exhausted")
        detail.setdefault(
            "reachability_diagnosis",
            last_reachability_diagnosis or "not_evaluated",
        )
        returned_action = zero
        returned_motion = case.reference.copy()
        rollback = True
        current_eval = dict(current_eval)
        current_eval.pop("_cfg", None)
        initial_public = dict(initial)
        initial_public.pop("_cfg", None)
        return SolverResult(status, returned_action, returned_motion, rollback, initial_public, current_eval, iterations, detail)
    current_eval = dict(current_eval)
    current_eval.pop("_cfg", None)
    initial_public = dict(initial)
    initial_public.pop("_cfg", None)
    return SolverResult(status, current_action, decode_geometry_action(case, current_action)[0], False, initial_public, current_eval, iterations, detail)


def evaluate_clean_identity(case: ActionFeasibilityCase) -> dict[str, Any]:
    """Evaluate authentic-clean handling separately, without solver trigger."""
    zero = np.zeros((case.frames, ACTION_DIM), dtype=np.float32)
    result = evaluate_action_candidate(case, zero, label="authentic_clean_identity")
    result["deployment_trigger_used"] = False
    result["hidden_clean_used"] = False
    result["path"] = "separate_identity_diagnostic"
    return result
