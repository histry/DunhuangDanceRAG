"""V15.15g fixed-budget Euclidean/Riemannian correction ablation.

The Adapter is a frozen warm start.  Every method uses the same immutable
Anchor, ownership mask, exact 1e-4 tangent sphere, differentiable constraint
bundle and final raw/Projector/Guard audit.  Correction outputs are detached;
validation successes and failures are never recycled into training data.

The activation-aware modes add identity as an explicit zero-action candidate.
V15.15g1 freezes an observable severity envelope from train-split single
controls, then selects among Adapter, Euclidean and Riemannian candidates only
when the Anchor lies outside that envelope.  Five differentiable signed-margin
terms from the fixed-Guard metric implementation prevent jerk/window/boundary
regression. Offline role/group metadata is attached only after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from motion_geometry.product_manifold import (
    product_exp_torch,
    product_log_torch,
)
from training import motion_models as m
from training import refiner_case_local_full_tangent_oracle as oracle
from training import refiner_observable_adapter_probe as adapter
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_15g_fixed_budget_manifold_correction_ablation_v1"
ACTIVATION_AWARE_SCHEMA = (
    "refiner_v15_15g_activation_aware_manifold_selection_probe_v1"
)
G1_SCHEMA = (
    "refiner_v15_15g1_observable_severity_guard_aligned_correction_v1"
)
HARD_NEGATIVE_SCHEMA = "refiner_v15_15g_guard_rejected_direction_bank_v1"
TEACHER_SCHEMA = adapter.V15_15E_TEACHER_SCHEMA
METHODS = ("adapter", "euclidean_projected", "riemannian_retraction")
IMPLICIT_BACKWARD_PROTOCOL = (
    "future_only_implicit_kkt_adjoint_after_stop_gradient_gate"
)
ACTIVATION_PHYSICAL_PROXY_KEYS = (
    "jerk_safety_excess",
    "root_vertical_safety_excess",
    "support_excess",
    "penetration_excess",
    "observable_trust_excess",
    "outside",
    "contact",
)
G1_GUARD_PROXY_TERMS = {
    "joint_jerk_p95": "repair_joint_jerk_mps3_p95_signed_margin",
    "joint_jerk_window_p95": (
        "repair_joint_jerk_window_p95_max_mps3_signed_margin"
    ),
    "extremity_jerk_p95": (
        "repair_extremity_jerk_mps3_p95_signed_margin"
    ),
    "extremity_jerk_window_p95": (
        "repair_extremity_jerk_window_p95_max_mps3_signed_margin"
    ),
    "boundary": "boundary_jerk_signed_margin",
}
G1_SEVERITY_CHANNELS = (
    "endpoint_gap",
    "temporal_gap",
    "root_gap",
    "root_velocity_gap",
    "yaw_gap",
    "support_mismatch",
    "phase_coverage",
    "phase_edge_density",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def negative_contrastive_penalty(predicted, rejected, *, margin=0.0):
    """Repel a prediction from a rejected train-only direction.

    The caller owns split enforcement.  V15.15g never calls this function on
    validation evidence; it records the definition for a later train-only
    online-teacher experiment.
    """
    cosine = m.torch.nn.functional.cosine_similarity(
        predicted.reshape(1, -1),
        rejected.reshape(1, -1),
        dim=-1,
        eps=1.0e-12,
    ).mean()
    return m.torch.relu(cosine - float(margin))


def _owned_case_mask(ownership, tangent, case_index):
    selected = m.torch.zeros_like(ownership, dtype=m.torch.bool)
    selected[int(case_index)] = True
    return (selected & ownership).expand_as(tangent)


def _normalize_exact_radius(tangent, mask, target_rms):
    masked = tangent.masked_fill(~mask, 0.0)
    active = masked[mask]
    if active.numel() == 0:
        raise RuntimeError("empty correction ownership support")
    norm = m.torch.linalg.vector_norm(active)
    if not bool(m.torch.isfinite(norm)) or float(norm.detach()) <= 1.0e-12:
        return masked, False
    scale = (
        float(target_rms) * math.sqrt(float(active.numel()))
        / norm.detach()
    )
    result = (masked * scale).masked_fill(~mask, 0.0)
    return result, True


def _rms(value, mask):
    active = value[mask]
    if active.numel() == 0:
        return math.nan
    return float(m.torch.sqrt(active.square().mean()).detach())


def _constraint_loss(
    model,
    batch,
    cfg,
    baseline,
    identity,
    candidate,
    case_index,
    group,
    baseline_case,
    contract,
):
    limits, scales = oracle._fixed_guard_limits(
        contract["initial_anchor"],
        contract["relative_tolerance"],
        contract["absolute_tolerance"],
        group,
    )
    _, _, constraints = oracle._constraint_bundle(
        model,
        batch,
        cfg,
        candidate,
        identity,
        case_index,
        group,
        baseline_case,
        limits,
        scales,
    )
    vector = m.torch.stack([constraints[key] for key in sorted(constraints)])
    positive = m.torch.relu(vector)
    return positive.square().sum(), {
        key: float(constraints[key].detach()) for key in sorted(constraints)
    }


def _label_free_activation_objective(
    batch,
    cfg,
    baseline,
    candidate,
    case_index,
):
    """Return a per-case observable objective with no role/group input."""
    _, terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    index = int(case_index)
    components = {
        "endpoint_scientific_deficit": terms[
            "endpoint_scientific_deficit"
        ][index],
        "temporal_scientific_deficit": terms[
            "temporal_scientific_deficit"
        ][index],
    }
    for key in ACTIVATION_PHYSICAL_PROXY_KEYS:
        components[key] = terms[key][index]
    loss = m.torch.stack([
        m.torch.clamp_min(value, 0.0)
        for value in components.values()
    ]).sum()
    physical = m.torch.stack([
        m.torch.clamp_min(components[key], 0.0)
        for key in ACTIVATION_PHYSICAL_PROXY_KEYS
    ]).sum()
    return loss, physical, {
        key: float(value.detach()) for key, value in components.items()
    }


def _observable_severity(sample):
    """Summarize serialized Adapter observables without using role labels."""
    features = sample["observable_condition"].detach().to(
        dtype=m.torch.float64, device="cpu"
    )
    ownership = sample["ownership"].detach().to(
        dtype=m.torch.bool, device="cpu"
    )
    if features.ndim != 2 or features.shape[-1] != (
        m.REFINER_ADAPTER_OBSERVABLE_DIM + m.REFINER_ADAPTER_PHASE_DIM
    ):
        raise RuntimeError("observable severity feature layout mismatch")
    active = ownership[..., 0] if ownership.ndim == 2 else ownership
    if active.ndim != 1 or active.shape[0] != features.shape[0]:
        raise RuntimeError("observable severity ownership layout mismatch")
    if not bool(active.any()):
        raise RuntimeError("observable severity has empty ownership")
    owned = features[active]
    support_mismatch = (
        owned[:, 5:9] - owned[:, 9:13]
    ).abs().amax()
    phase_start = m.REFINER_ADAPTER_OBSERVABLE_DIM
    phase_coverage = features[:, phase_start].mean()
    phase_edge_density = owned[:, phase_start + 1].abs().mean()
    values = {
        "endpoint_gap": owned[:, 0].amax(),
        "temporal_gap": owned[:, 1].amax(),
        "root_gap": owned[:, 2].amax(),
        "root_velocity_gap": owned[:, 3].amax(),
        "yaw_gap": owned[:, 4].amax(),
        "support_mismatch": support_mismatch,
        "phase_coverage": phase_coverage,
        "phase_edge_density": phase_edge_density,
    }
    return {key: float(values[key]) for key in G1_SEVERITY_CHANNELS}


def _freeze_single_severity_envelope(
    train_teacher,
    *,
    margin_fraction,
    absolute_margin,
):
    """Freeze an observable envelope from train-split single controls only."""
    controls = [
        sample for sample in train_teacher["samples"]
        if sample.get("teacher_kind") == "identity_control"
    ]
    if not controls:
        raise RuntimeError("train bank has no single controls for envelope")
    rows = {
        str(sample["case_uid"]): _observable_severity(sample)
        for sample in controls
    }
    maxima = {
        key: max(row[key] for row in rows.values())
        for key in G1_SEVERITY_CHANNELS
    }
    limits = {
        key: max(
            maxima[key] * (1.0 + float(margin_fraction)),
            maxima[key] + float(absolute_margin),
        )
        for key in G1_SEVERITY_CHANNELS
    }
    return {
        "schema": "refiner_v15_15g1_single_severity_envelope_v1",
        "calibration_split": "train",
        "calibration_role": "identity_control",
        "calibration_role_used_offline_only": True,
        "inference_role_label_consumed": False,
        "control_count": len(controls),
        "channels": list(G1_SEVERITY_CHANNELS),
        "maximum_by_channel": maxima,
        "limit_by_channel": limits,
        "margin_fraction": float(margin_fraction),
        "absolute_margin": float(absolute_margin),
        "case_severity": rows,
    }


def _severity_status(sample, envelope, scale_floor):
    values = _observable_severity(sample)
    limits = envelope["limit_by_channel"]
    excess = {
        key: values[key] - float(limits[key])
        for key in G1_SEVERITY_CHANNELS
    }
    normalized = {
        key: excess[key] / max(abs(float(limits[key])), float(scale_floor))
        for key in G1_SEVERITY_CHANNELS
    }
    return {
        "values": values,
        "limits": dict(limits),
        "excess": excess,
        "normalized_excess": normalized,
        "maximum_normalized_excess": max(normalized.values()),
        "outside_frozen_single_envelope": any(
            value > 0.0 for value in excess.values()
        ),
    }


def _guard_proxy_values(batch, cfg, baseline, candidate, case_index):
    _, terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    index = int(case_index)
    return {
        name: terms[key][index]
        for name, key in G1_GUARD_PROXY_TERMS.items()
    }


def _guard_aligned_correction_objective(
    batch,
    cfg,
    baseline,
    candidate,
    case_index,
    anchor_proxy,
    tolerance,
    scale_floor,
):
    _, terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    index = int(case_index)
    scientific = {
        "endpoint_scientific_deficit": terms[
            "endpoint_scientific_deficit"
        ][index],
        "temporal_scientific_deficit": terms[
            "temporal_scientific_deficit"
        ][index],
    }
    candidate_proxy = {
        name: terms[key][index]
        for name, key in G1_GUARD_PROXY_TERMS.items()
    }
    proxy_delta = {
        name: candidate_proxy[name] - anchor_proxy[name]
        for name in G1_GUARD_PROXY_TERMS
    }
    proxy_excess = {
        name: m.torch.relu(delta - float(tolerance))
        / anchor_proxy[name].abs().clamp_min(float(scale_floor))
        for name, delta in proxy_delta.items()
    }
    loss = m.torch.stack(list(scientific.values())).sum()
    loss = loss + m.torch.stack(list(proxy_excess.values())).square().sum()
    diagnostics = {
        **{
            key: float(value.detach())
            for key, value in scientific.items()
        },
        "guard_proxy_anchor": {
            key: float(value.detach()) for key, value in anchor_proxy.items()
        },
        "guard_proxy_candidate": {
            key: float(value.detach())
            for key, value in candidate_proxy.items()
        },
        "guard_proxy_delta": {
            key: float(value.detach()) for key, value in proxy_delta.items()
        },
        "guard_proxy_normalized_excess": {
            key: float(value.detach())
            for key, value in proxy_excess.items()
        },
    }
    return loss, diagnostics


def _bounded_direction(direction, mask, maximum_rms):
    masked = direction.masked_fill(~mask, 0.0)
    active = masked[mask]
    norm = m.torch.linalg.vector_norm(active)
    if not bool(m.torch.isfinite(norm)) or float(norm.detach()) <= 1.0e-20:
        return m.torch.zeros_like(masked), False
    rms = norm / math.sqrt(float(active.numel()))
    scale = min(1.0, float(maximum_rms) / max(float(rms.detach()), 1.0e-20))
    return masked * scale, True


def _correct_case(
    *,
    method,
    steps,
    initial_tangent,
    ownership,
    baseline,
    identity,
    batch,
    cfg,
    case_index,
    group,
    baseline_case,
    contract,
    target_rms,
    step_size,
    trust_fraction,
    activation_aware=False,
    guard_aligned=False,
    guard_proxy_tolerance=1.0e-6,
    guard_proxy_scale_floor=1.0e-6,
):
    mask = _owned_case_mask(ownership, initial_tangent, case_index)
    current, normalized = _normalize_exact_radius(
        initial_tangent, mask, target_rms
    )
    history = []
    zero_start = bool(not normalized and activation_aware)
    numeric_failure = bool(not normalized and not activation_aware)
    started = time.perf_counter()
    if numeric_failure:
        return current.detach(), {
            "method": method,
            "steps_requested": int(steps),
            "steps_completed": 0,
            "numeric_failure": True,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
        }
    if zero_start:
        current = m.torch.zeros_like(initial_tangent).masked_fill(
            ~mask, 0.0
        )
    anchor_proxy = None
    if guard_aligned:
        anchor_proxy = _guard_proxy_values(
            batch,
            cfg,
            baseline,
            baseline,
            case_index,
        )

    for iteration in range(int(steps)):
        if method == "euclidean_projected":
            variable = current.detach().requires_grad_(True)
            candidate = product_exp_torch(
                baseline, variable.masked_fill(~mask, 0.0)
            )
        elif method == "riemannian_retraction":
            current_motion = product_exp_torch(baseline, current.detach())
            variable = m.torch.zeros_like(current).requires_grad_(True)
            candidate = product_exp_torch(
                current_motion, variable.masked_fill(~mask, 0.0)
            )
        else:
            raise ValueError(f"unsupported correction method: {method}")

        if guard_aligned:
            loss, constraints = _guard_aligned_correction_objective(
                batch,
                cfg,
                baseline,
                candidate,
                case_index,
                anchor_proxy,
                guard_proxy_tolerance,
                guard_proxy_scale_floor,
            )
        elif activation_aware:
            loss, _, constraints = _label_free_activation_objective(
                batch,
                cfg,
                baseline,
                candidate,
                case_index,
            )
        else:
            loss, constraints = _constraint_loss(
                model,
                batch,
                cfg,
                baseline,
                identity,
                candidate,
                case_index,
                group,
                baseline_case,
                contract,
            )
        gradient = m.torch.autograd.grad(
            loss, variable, allow_unused=True
        )[0]
        if gradient is None or not bool(m.torch.isfinite(gradient).all()):
            numeric_failure = True
            break
        direction, usable = _bounded_direction(
            -gradient.detach(),
            mask,
            float(target_rms) * float(trust_fraction),
        )
        if not usable:
            history.append({
                "iteration": iteration,
                "loss_before": float(loss.detach()),
                "accepted": False,
                "reason": "zero_or_nonfinite_correction_gradient",
                "constraints": constraints,
            })
            break

        accepted = False
        accepted_loss = float(loss.detach())
        accepted_scale = 0.0
        accepted_tangent = current
        for backtrack in range(6):
            alpha = float(step_size) * (0.5 ** backtrack)
            if method == "euclidean_projected":
                trial_raw = current + alpha * direction
            else:
                trial_motion = product_exp_torch(
                    current_motion, alpha * direction
                )
                trial_raw = product_log_torch(baseline, trial_motion)
            trial, trial_ok = _normalize_exact_radius(
                trial_raw, mask, target_rms
            )
            if not trial_ok:
                continue
            trial_candidate = product_exp_torch(baseline, trial)
            if guard_aligned:
                trial_loss, _ = _guard_aligned_correction_objective(
                    batch,
                    cfg,
                    baseline,
                    trial_candidate,
                    case_index,
                    anchor_proxy,
                    guard_proxy_tolerance,
                    guard_proxy_scale_floor,
                )
            elif activation_aware:
                trial_loss, _, _ = _label_free_activation_objective(
                    batch,
                    cfg,
                    baseline,
                    trial_candidate,
                    case_index,
                )
            else:
                trial_loss, _ = _constraint_loss(
                    model,
                    batch,
                    cfg,
                    baseline,
                    identity,
                    trial_candidate,
                    case_index,
                    group,
                    baseline_case,
                    contract,
                )
            if (
                bool(m.torch.isfinite(trial_loss))
                and float(trial_loss.detach())
                <= float(loss.detach()) + 1.0e-12
            ):
                accepted = True
                accepted_loss = float(trial_loss.detach())
                accepted_scale = alpha
                accepted_tangent = trial.detach()
                break
        history.append({
            "iteration": iteration,
            "loss_before": float(loss.detach()),
            "loss_after": accepted_loss,
            "accepted": accepted,
            "step_scale": accepted_scale,
            "constraints": constraints,
            "exact_radius_rms": _rms(accepted_tangent, mask),
        })
        if not accepted:
            break
        current = accepted_tangent

    return current.detach(), {
        "method": method,
        "steps_requested": int(steps),
        "steps_completed": len(history),
        "accepted_steps": sum(int(row["accepted"]) for row in history),
        "numeric_failure": numeric_failure,
        "zero_adapter_start": zero_start,
        "role_or_group_consumed_by_correction": bool(
            not activation_aware and not guard_aligned
        ),
        "guard_aligned_correction": bool(guard_aligned),
        "final_exact_radius_rms": _rms(current, mask),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _transaction_domains(teacher, batch, baseline, identity, model, cfg):
    domains = {}
    for transaction_id, metadata in teacher["transaction_schedules"].items():
        start = int(metadata["global_case_offset"])
        stop = start + int(metadata["case_count"])
        local_batch = adapter._slice_batch(batch, start, stop)
        local_baseline = baseline[start:stop]
        local_identity = identity[start:stop]
        domains[transaction_id] = {
            "slice": (start, stop),
            "batch": local_batch,
            "baseline": local_baseline,
            "identity": local_identity,
            "baseline_guard": projected_probe._float_guard(
                projected_probe._guard_values_for_prediction(
                    model,
                    local_batch,
                    cfg,
                    local_baseline,
                    local_identity,
                )
            ),
            "contract": teacher["transaction_guard_contracts"][
                transaction_id
            ],
        }
    return domains


def _normalize_all_sample_tangents(
    tangent,
    ownership,
    samples,
    target_rms,
):
    """Normalize every enumerated case without consulting its role label."""
    result = m.torch.zeros_like(tangent)
    diagnostics = {}
    visited = set()
    for sample in samples:
        case_index = int(sample["case_index"])
        if case_index in visited:
            raise RuntimeError(f"duplicate activation case {case_index}")
        visited.add(case_index)
        mask = _owned_case_mask(ownership, tangent, case_index)
        local, resolved = _normalize_exact_radius(
            tangent, mask, target_rms
        )
        result = result + local
        uid = str(sample.get("case_uid", case_index))
        diagnostics[uid] = {
            "input_rms": _rms(tangent, mask),
            "normalized_rms": _rms(local, mask),
            "radius_equality_resolved": bool(resolved),
        }
    return result, diagnostics


def _sample_problem(sample, domains, ownership):
    transaction_id = str(sample["transaction_id"])
    domain = domains[transaction_id]
    local_case = int(sample["local_case_index"])
    global_case = int(sample["case_index"])
    case_batch = adapter._slice_batch(
        domain["batch"], local_case, local_case + 1
    )
    return {
        "transaction_id": transaction_id,
        "local_case_index": local_case,
        "global_case_index": global_case,
        "batch": case_batch,
        "baseline": domain["baseline"][local_case:local_case + 1],
        "identity": domain["identity"][local_case:local_case + 1],
        "ownership": ownership[global_case:global_case + 1],
    }


def _activation_aware_selection(
    *,
    variants,
    correction_reports,
    samples,
    domains,
    ownership,
    baseline_terms,
    cfg,
    target_rms,
    proxy_tolerance,
    relative_improvement,
):
    """Select identity or repair using observables only, then attach labels."""
    selected = m.torch.zeros_like(next(iter(variants.values())))
    decisions = {}
    method_counts = Counter()
    for sample in samples:
        uid = str(sample.get("case_uid", sample["case_index"]))
        problem = _sample_problem(
            sample,
            domains,
            ownership,
        )
        case_index = problem["global_case_index"]
        baseline_case = {
            key: float(baseline_terms[key][case_index].detach())
            for key in ("endpoint", "temporal")
        }
        identity_loss, identity_physical, identity_components = (
            _label_free_activation_objective(
                problem["batch"],
                cfg,
                problem["baseline"],
                problem["baseline"],
                0,
            )
        )
        identity_value = float(identity_loss.detach())
        required_reduction = max(
            abs(identity_value) * float(relative_improvement),
            1.0e-8,
        )
        candidates = {}
        eligible = []
        for method, tangent in variants.items():
            local = tangent[case_index:case_index + 1]
            mask = problem["ownership"].expand_as(local)
            scoped = local.masked_fill(~mask, 0.0)
            active_rms = _rms(scoped, mask)
            radius_resolved = bool(
                math.isfinite(active_rms)
                and abs(active_rms - float(target_rms))
                <= max(1.0e-12, float(target_rms) * 1.0e-6)
            )
            candidate = product_exp_torch(problem["baseline"], scoped)
            loss, physical, components = _label_free_activation_objective(
                problem["batch"],
                cfg,
                problem["baseline"],
                candidate,
                0,
            )
            case_terms = oracle.case_probe._case_terms(
                candidate,
                problem["batch"],
                cfg,
            )
            candidate_case = {
                key: float(case_terms[key][0].detach())
                for key in ("endpoint", "temporal")
            }
            scientific = oracle._case_scientific_status(
                candidate_case, baseline_case
            )
            numeric_failure = bool(
                correction_reports.get(method, {}).get(uid, {}).get(
                    "numeric_failure", False
                )
            )
            loss_value = float(loss.detach())
            physical_value = float(physical.detach())
            observable_improvement = bool(
                math.isfinite(loss_value)
                and loss_value <= identity_value - required_reduction
            )
            proxy_safe = bool(
                math.isfinite(physical_value)
                and physical_value <= float(proxy_tolerance)
            )
            accepted = bool(
                radius_resolved
                and scientific["passed"]
                and observable_improvement
                and proxy_safe
                and not numeric_failure
            )
            candidates[method] = {
                "objective": loss_value,
                "identity_objective": identity_value,
                "required_objective_reduction": required_reduction,
                "physical_proxy_excess": physical_value,
                "proxy_components": components,
                "endpoint_delta": scientific["endpoint_delta"],
                "temporal_delta": scientific["temporal_delta"],
                "scientific_passed": bool(scientific["passed"]),
                "radius_rms": active_rms,
                "radius_equality_resolved": radius_resolved,
                "numeric_failure": numeric_failure,
                "eligible": accepted,
                "eligible_after_adapter_incumbent": accepted,
            }
            if accepted:
                eligible.append((loss_value, method, scoped))
        if eligible:
            _, selected_method, selected_tangent = min(
                eligible, key=lambda row: (row[0], row[1])
            )
            selected[case_index:case_index + 1] = selected_tangent
        else:
            selected_method = "identity"
        method_counts.update([selected_method])
        decisions[uid] = {
            "selected_method": selected_method,
            "selected_nonzero": bool(selected_method != "identity"),
            "selection": {
                "identity_objective": identity_value,
                "identity_physical_proxy_excess": float(
                    identity_physical.detach()
                ),
                "identity_proxy_components": identity_components,
                "candidates": candidates,
            },
            "evaluation_only": {
                "teacher_kind": sample.get("teacher_kind"),
                "audit_group": sample.get("audit_group"),
            },
        }
    return selected, decisions, dict(method_counts)


def _g1_observable_guard_selection(
    *,
    variants,
    correction_reports,
    samples,
    domains,
    ownership,
    baseline_terms,
    cfg,
    target_rms,
    severity_envelope,
    severity_scale_floor,
    guard_proxy_tolerance,
    guard_proxy_scale_floor,
):
    """Select a repair from severity and differentiable Guard proxies only."""
    selected = m.torch.zeros_like(next(iter(variants.values())))
    decisions = {}
    method_counts = Counter()
    for sample in samples:
        uid = str(sample.get("case_uid", sample["case_index"]))
        problem = _sample_problem(sample, domains, ownership)
        case_index = problem["global_case_index"]
        severity = _severity_status(
            sample,
            severity_envelope,
            severity_scale_floor,
        )
        baseline_case = {
            key: float(baseline_terms[key][case_index].detach())
            for key in ("endpoint", "temporal")
        }
        anchor_proxy = _guard_proxy_values(
            problem["batch"],
            cfg,
            problem["baseline"],
            problem["baseline"],
            0,
        )
        candidates = {}
        eligible = []
        for method, tangent in variants.items():
            local = tangent[case_index:case_index + 1]
            mask = problem["ownership"].expand_as(local)
            outside = local.masked_fill(mask, 0.0)
            outside_scope_abs_max = (
                float(outside.abs().amax().detach())
                if outside.numel() else 0.0
            )
            scoped = local.masked_fill(~mask, 0.0)
            active_rms = _rms(scoped, mask)
            radius_resolved = bool(
                math.isfinite(active_rms)
                and abs(active_rms - float(target_rms))
                <= max(1.0e-12, float(target_rms) * 1.0e-6)
            )
            candidate = product_exp_torch(problem["baseline"], scoped)
            objective, components = _guard_aligned_correction_objective(
                problem["batch"],
                cfg,
                problem["baseline"],
                candidate,
                0,
                anchor_proxy,
                guard_proxy_tolerance,
                guard_proxy_scale_floor,
            )
            candidate_proxy = _guard_proxy_values(
                problem["batch"],
                cfg,
                problem["baseline"],
                candidate,
                0,
            )
            proxy_delta = {
                name: float(
                    (candidate_proxy[name] - anchor_proxy[name]).detach()
                )
                for name in G1_GUARD_PROXY_TERMS
            }
            normalized_proxy_delta = {
                name: proxy_delta[name] / max(
                    abs(float(anchor_proxy[name].detach())),
                    float(guard_proxy_scale_floor),
                )
                for name in G1_GUARD_PROXY_TERMS
            }
            worst_proxy_delta = max(normalized_proxy_delta.values())
            proxy_nonregression = all(
                value <= float(guard_proxy_tolerance)
                for value in proxy_delta.values()
            )
            case_terms = oracle.case_probe._case_terms(
                candidate,
                problem["batch"],
                cfg,
            )
            candidate_case = {
                key: float(case_terms[key][0].detach())
                for key in ("endpoint", "temporal")
            }
            scientific = oracle._case_scientific_status(
                candidate_case, baseline_case
            )
            numeric_failure = bool(
                correction_reports.get(method, {}).get(uid, {}).get(
                    "numeric_failure", False
                )
            )
            objective_value = float(objective.detach())
            accepted = bool(
                severity["outside_frozen_single_envelope"]
                and scientific["passed"]
                and proxy_nonregression
                and radius_resolved
                and outside_scope_abs_max == 0.0
                and math.isfinite(objective_value)
                and not numeric_failure
            )
            candidates[method] = {
                "objective": objective_value,
                "severity_gate_passed": bool(
                    severity["outside_frozen_single_envelope"]
                ),
                "guard_proxy_nonregression": proxy_nonregression,
                "guard_proxy_delta": proxy_delta,
                "guard_proxy_normalized_delta": normalized_proxy_delta,
                "guard_proxy_worst_normalized_delta": worst_proxy_delta,
                "proxy_components": components,
                "endpoint_delta": scientific["endpoint_delta"],
                "temporal_delta": scientific["temporal_delta"],
                "scientific_passed": bool(scientific["passed"]),
                "radius_rms": active_rms,
                "radius_equality_resolved": radius_resolved,
                "outside_scope_abs_max": outside_scope_abs_max,
                "scope_safe": outside_scope_abs_max == 0.0,
                "numeric_failure": numeric_failure,
                "eligible": accepted,
            }
            if accepted:
                eligible.append((
                    worst_proxy_delta,
                    objective_value,
                    method,
                    scoped,
                ))
        adapter_candidate = candidates.get("adapter")
        if adapter_candidate and adapter_candidate["eligible"]:
            retained = []
            for row in eligible:
                method = row[2]
                candidate = candidates[method]
                guard_dominates_adapter = all(
                    candidate["guard_proxy_delta"][name]
                    <= adapter_candidate["guard_proxy_delta"][name]
                    + float(guard_proxy_tolerance)
                    for name in G1_GUARD_PROXY_TERMS
                )
                science_dominates_adapter = bool(
                    candidate["objective"]
                    <= adapter_candidate["objective"] + 1.0e-12
                )
                incumbent_safe = bool(
                    method == "adapter"
                    or (
                        guard_dominates_adapter
                        and science_dominates_adapter
                    )
                )
                candidate["guard_dominates_adapter"] = (
                    guard_dominates_adapter
                )
                candidate["science_dominates_adapter"] = (
                    science_dominates_adapter
                )
                candidate["eligible_after_adapter_incumbent"] = (
                    incumbent_safe
                )
                if incumbent_safe:
                    retained.append(row)
            eligible = retained
        if eligible:
            _, _, selected_method, selected_tangent = min(
                eligible,
                key=lambda row: (row[0], row[1], row[2]),
            )
            selected[case_index:case_index + 1] = selected_tangent
        else:
            selected_method = "identity"
        method_counts.update([selected_method])
        decisions[uid] = {
            "selected_method": selected_method,
            "selected_nonzero": bool(selected_method != "identity"),
            "selection": {
                "anchor_severity": severity,
                "activation_condition": (
                    "anchor_outside_frozen_train_single_envelope"
                ),
                "guard_proxy_terms": dict(G1_GUARD_PROXY_TERMS),
                "candidates": candidates,
            },
            "evaluation_only": {
                "teacher_kind": sample.get("teacher_kind"),
                "audit_group": sample.get("audit_group"),
            },
        }
    return selected, decisions, dict(method_counts)


def _variant_summary(audits, correction_reports, elapsed):
    raw_by_group = Counter()
    projected_by_group = Counter()
    blockers = Counter()
    for row in audits:
        if row["raw_audit"]["passed"]:
            raw_by_group.update([row["group"]])
        if row["effective_projected_candidate"]:
            projected_by_group.update([row["group"]])
        blockers.update(row["raw_audit"].get("fixed_guard_blockers", []))
    return {
        "raw_pass_count_by_group": dict(raw_by_group),
        "projected_pass_count_by_group": dict(projected_by_group),
        "fixed_guard_blocker_counts": dict(blockers),
        "numeric_failure_count": sum(
            int(row.get("numeric_failure", False))
            for row in correction_reports.values()
        ),
        "elapsed_seconds": float(elapsed),
        "correction_by_case": correction_reports,
        "exact_audits": audits,
    }


def run(args):
    started = time.perf_counter()
    activation_enabled = bool(
        args.activation_aware or args.activation_aware_g1
    )
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    teacher_path = Path(args.validation_teacher_bank).resolve()
    state_path = Path(args.adapter_state).resolve()
    teacher = m.torch.load(
        teacher_path, map_location="cpu", weights_only=False
    )
    if teacher.get("schema") != TEACHER_SCHEMA:
        raise RuntimeError("V15.15g requires a V15.15e bank")
    if teacher.get("split") != "validation":
        raise RuntimeError("V15.15g correction ablation requires validation")
    if not teacher.get("teacher_bank_ready"):
        raise RuntimeError("validation bank lacks a complete cross route")
    if teacher.get("train_validation_case_overlap"):
        raise RuntimeError("validation bank reports train case overlap")
    if teacher.get("train_validation_source_case_overlap"):
        raise RuntimeError("validation bank reports source-case overlap")

    train_teacher = None
    severity_envelope = None
    severity_envelope_path = None
    train_teacher_path = None
    if args.activation_aware_g1:
        if not args.train_teacher_bank:
            raise RuntimeError("V15.15g1 requires --train-teacher-bank")
        train_teacher_path = Path(args.train_teacher_bank).resolve()
        train_teacher = m.torch.load(
            train_teacher_path, map_location="cpu", weights_only=False
        )
        if train_teacher.get("schema") != TEACHER_SCHEMA:
            raise RuntimeError("V15.15g1 train bank schema mismatch")
        if train_teacher.get("split") != "train":
            raise RuntimeError("V15.15g1 envelope bank must be train split")
        if not train_teacher.get("teacher_bank_ready"):
            raise RuntimeError("V15.15g1 train bank is not ready")
        if train_teacher.get("source_diagnostic") != teacher.get(
            "source_diagnostic"
        ):
            raise RuntimeError("train/validation source diagnostic mismatch")
        for key in (
            "split_manifest_file_sha256",
            "split_manifest_content_sha256",
        ):
            if train_teacher.get(key) != teacher.get(key):
                raise RuntimeError(f"train/validation {key} mismatch")
        severity_envelope = _freeze_single_severity_envelope(
            train_teacher,
            margin_fraction=float(args.severity_envelope_margin_fraction),
            absolute_margin=float(args.severity_envelope_absolute_margin),
        )
        severity_envelope.update({
            "train_teacher_bank": str(train_teacher_path),
            "train_teacher_bank_sha256": _file_sha256(train_teacher_path),
            "split_manifest_file_sha256": teacher.get(
                "split_manifest_file_sha256"
            ),
            "split_manifest_content_sha256": teacher.get(
                "split_manifest_content_sha256"
            ),
        })
        severity_envelope_path = (
            destination / "observable_severity_envelope.json"
        )
        severity_envelope_path.write_text(
            json.dumps(severity_envelope, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    cfg.product_refiner_observable_adapter = True
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.15g correction ablation requires CUDA")
    batch = adapter._to_device(teacher["batch"], device)
    model, _, _gate_mode = adapter._load_source_model(
        Path(teacher["source_diagnostic"]),
        cfg,
        device,
        adapter_state_path=state_path,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = m.torch.load(state_path, map_location="cpu", weights_only=False)
    model.observable_adapter_gate_floor.copy_(m.torch.as_tensor(
        float(state["gate_floor"]),
        dtype=model.observable_adapter_gate_floor.dtype,
        device=device,
    ))
    baseline = teacher["baseline_prediction"].to(device)
    identity = teacher["baseline_identity"].to(device)
    baseline_terms = oracle.case_probe._case_terms(baseline, batch, cfg)
    domains = _transaction_domains(
        teacher, batch, baseline, identity, model, cfg
    )
    _, trace = adapter._adapter_batch_outputs(model, batch, cfg)
    ownership = trace["ownership"].to(m.torch.bool)
    initial = trace["decoder_consistent_tangent"].masked_fill(
        ~ownership.expand_as(trace["decoder_consistent_tangent"]), 0.0
    )
    samples = teacher["samples"]
    if activation_enabled:
        initial, radius = _normalize_all_sample_tangents(
            initial,
            ownership,
            samples,
            float(args.target_rms),
        )
    else:
        initial, radius = adapter._safe_exact_radius_tangent(
            initial,
            ownership,
            samples,
            float(args.target_rms),
            eps=1.0e-8,
        )

    variants = {}
    variant_tangents = {}
    variant_correction_reports = {}
    hard_negatives = []
    budgets = tuple(int(value) for value in args.steps)
    specifications = [("adapter", 0)] + [
        (method, budget)
        for method in METHODS[1:]
        for budget in budgets
    ]
    workspace_floor = max(1.0e-8, float(args.target_rms) * 0.01)
    for method, budget in specifications:
        variant_started = time.perf_counter()
        tangent = initial.detach().clone()
        correction_reports = {}
        if method != "adapter":
            tangent.zero_()
            for sample in samples:
                if (
                    not activation_enabled
                    and sample["teacher_kind"] != "exact_projected_direction"
                ):
                    continue
                transaction_id = str(sample["transaction_id"])
                domain = domains[transaction_id]
                start, stop = domain["slice"]
                local_case = int(sample["local_case_index"])
                if activation_enabled:
                    group = None
                    local_initial = initial[
                        int(sample["case_index"]):
                        int(sample["case_index"]) + 1
                    ]
                    local_ownership = ownership[
                        int(sample["case_index"]):
                        int(sample["case_index"]) + 1
                    ]
                    local_baseline = domain["baseline"][
                        local_case:local_case + 1
                    ]
                    local_identity = domain["identity"][
                        local_case:local_case + 1
                    ]
                    local_batch = adapter._slice_batch(
                        domain["batch"], local_case, local_case + 1
                    )
                    correction_case = 0
                    baseline_case = None
                    local_contract = None
                else:
                    group = str(sample["audit_group"])
                    local_initial = initial[start:stop]
                    local_ownership = ownership[start:stop]
                    local_baseline = domain["baseline"]
                    local_identity = domain["identity"]
                    local_batch = domain["batch"]
                    correction_case = local_case
                    baseline_case = {
                        key: float(
                            baseline_terms[key][int(sample["case_index"])]
                        )
                        for key in ("endpoint", "temporal")
                    }
                    local_contract = domain["contract"]
                corrected, correction_report = _correct_case(
                    method=method,
                    steps=budget,
                    initial_tangent=local_initial,
                    ownership=local_ownership,
                    baseline=local_baseline,
                    identity=local_identity,
                    batch=local_batch,
                    cfg=cfg,
                    case_index=correction_case,
                    group=group,
                    baseline_case=baseline_case,
                    contract=local_contract,
                    target_rms=float(args.target_rms),
                    step_size=float(args.step_size),
                    trust_fraction=float(args.trust_fraction),
                    activation_aware=activation_enabled,
                    guard_aligned=bool(args.activation_aware_g1),
                    guard_proxy_tolerance=float(
                        args.guard_proxy_nonregression_tolerance
                    ),
                    guard_proxy_scale_floor=float(
                        args.guard_proxy_scale_floor
                    ),
                )
                if activation_enabled:
                    global_case = int(sample["case_index"])
                    tangent[global_case:global_case + 1] = corrected
                else:
                    mask = _owned_case_mask(
                        local_ownership, corrected, local_case
                    )
                    tangent[start:stop] += corrected.masked_fill(~mask, 0.0)
                correction_reports[str(sample["case_uid"])] = (
                    correction_report
                )
        audits = adapter._audit_step(
            model,
            batch,
            cfg,
            tangent.detach(),
            tangent.detach(),
            trace["c2_taper"].detach(),
            ownership.detach(),
            baseline,
            identity,
            None,
            {},
            baseline_terms,
            samples,
            float(args.target_rms),
            workspace_floor,
            args,
            transaction_domains=domains,
        )
        key = method if method == "adapter" else f"{method}_k{budget}"
        variant_tangents[key] = tangent.detach().clone()
        variant_correction_reports[key] = correction_reports
        variants[key] = _variant_summary(
            audits,
            correction_reports,
            time.perf_counter() - variant_started,
        )
        audit_by_uid = {str(row["case_uid"]): row for row in audits}
        for sample in samples:
            if sample["teacher_kind"] != "exact_projected_direction":
                continue
            uid = str(sample["case_uid"])
            audit = audit_by_uid[uid]
            if audit["raw_audit"]["passed"]:
                continue
            hard_negatives.append({
                "case_uid": uid,
                "transaction_id": str(sample["transaction_id"]),
                "local_case_index": int(sample["local_case_index"]),
                "group": str(sample["audit_group"]),
                "method": method,
                "steps": int(budget),
                "fixed_guard_blockers": list(
                    audit["raw_audit"].get("fixed_guard_blockers", [])
                ),
                "teacher_eligible": False,
                "negative_contrastive_definition": (
                    "relu(cos(predicted,rejected)-margin)"
                ),
            })
        print(json.dumps({
            "stage": "v15_15g_correction_variant",
            "variant": key,
            "raw_pass_count_by_group": variants[key][
                "raw_pass_count_by_group"
            ],
            "projected_pass_count_by_group": variants[key][
                "projected_pass_count_by_group"
            ],
            "numeric_failure_count": variants[key][
                "numeric_failure_count"
            ],
        }), flush=True)

    activation_summary = None
    activation_aware_supported = False
    if activation_enabled:
        selection_started = time.perf_counter()
        if args.activation_aware_g1:
            (
                selected_tangent,
                activation_decisions,
                selected_method_counts,
            ) = _g1_observable_guard_selection(
                variants=variant_tangents,
                correction_reports=variant_correction_reports,
                samples=samples,
                domains=domains,
                ownership=ownership,
                baseline_terms=baseline_terms,
                cfg=cfg,
                target_rms=float(args.target_rms),
                severity_envelope=severity_envelope,
                severity_scale_floor=float(args.severity_scale_floor),
                guard_proxy_tolerance=float(
                    args.guard_proxy_nonregression_tolerance
                ),
                guard_proxy_scale_floor=float(
                    args.guard_proxy_scale_floor
                ),
            )
        else:
            (
                selected_tangent,
                activation_decisions,
                selected_method_counts,
            ) = _activation_aware_selection(
                variants=variant_tangents,
                correction_reports=variant_correction_reports,
                samples=samples,
                domains=domains,
                ownership=ownership,
                baseline_terms=baseline_terms,
                cfg=cfg,
                target_rms=float(args.target_rms),
                proxy_tolerance=float(args.activation_proxy_tolerance),
                relative_improvement=float(
                    args.activation_relative_improvement
                ),
            )
        selected_audits = adapter._audit_step(
            model,
            batch,
            cfg,
            selected_tangent.detach(),
            selected_tangent.detach(),
            trace["c2_taper"].detach(),
            ownership.detach(),
            baseline,
            identity,
            None,
            {},
            baseline_terms,
            samples,
            float(args.target_rms),
            workspace_floor,
            args,
            transaction_domains=domains,
        )
        selected_summary = _variant_summary(
            selected_audits,
            {},
            time.perf_counter() - selection_started,
        )
        variants["activation_aware_selected"] = selected_summary
        audit_by_uid = {
            str(row["case_uid"]): row for row in selected_audits
        }
        cross_samples = [
            sample for sample in samples
            if sample["teacher_kind"] == "exact_projected_direction"
        ]
        identity_samples = [
            sample for sample in samples
            if sample["teacher_kind"] == "identity_control"
        ]
        required_by_group = Counter(
            str(sample["audit_group"]) for sample in cross_samples
        )
        selected_projected_by_group = Counter(
            str(row["group"])
            for row in selected_audits
            if row["effective_projected_candidate"]
        )
        cross_closure_complete = bool(
            cross_samples
            and all(
                uid in audit_by_uid
                and audit_by_uid[uid]["raw_audit"]["passed"]
                and audit_by_uid[uid]["projector_result"] is not None
                and audit_by_uid[uid]["effective_projected_candidate"]
                for uid in (
                    str(sample["case_uid"])
                    for sample in cross_samples
                )
            )
        )
        identity_rms_by_uid = {}
        for sample in identity_samples:
            case_index = int(sample["case_index"])
            mask = _owned_case_mask(
                ownership, selected_tangent, case_index
            )
            identity_rms_by_uid[str(sample["case_uid"])] = _rms(
                selected_tangent, mask
            )
        identity_rms_max = max(identity_rms_by_uid.values(), default=0.0)
        false_activation_uids = sorted(
            str(sample["case_uid"])
            for sample in identity_samples
            if activation_decisions[str(sample["case_uid"])][
                "selected_nonzero"
            ]
        )
        missed_cross_uids = sorted(
            str(sample["case_uid"])
            for sample in cross_samples
            if not activation_decisions[str(sample["case_uid"])][
                "selected_nonzero"
            ]
        )
        selected_proxy_nonregression_complete = bool(
            not args.activation_aware_g1
            or all(
                not decision["selected_nonzero"]
                or decision["selection"]["candidates"][
                    decision["selected_method"]
                ]["guard_proxy_nonregression"]
                for decision in activation_decisions.values()
            )
        )
        selected_severity_condition_complete = bool(
            not args.activation_aware_g1
            or all(
                not decision["selected_nonzero"]
                or decision["selection"]["anchor_severity"][
                    "outside_frozen_single_envelope"
                ]
                for decision in activation_decisions.values()
            )
        )
        correction_numeric_failures = sum(
            int(report.get("numeric_failure", False))
            for reports in variant_correction_reports.values()
            for report in reports.values()
        )
        if args.activation_aware_g1:
            decision_numeric_complete = all(
                math.isfinite(float(decision["selection"][
                    "anchor_severity"
                ]["maximum_normalized_excess"]))
                and all(
                    math.isfinite(float(candidate["objective"]))
                    and math.isfinite(float(candidate[
                        "outside_scope_abs_max"
                    ]))
                    and all(
                        math.isfinite(float(value))
                        for value in candidate[
                            "guard_proxy_delta"
                        ].values()
                    )
                    for candidate in decision["selection"][
                        "candidates"
                    ].values()
                )
                for decision in activation_decisions.values()
            )
        else:
            decision_numeric_complete = all(
                math.isfinite(float(decision["selection"][
                    "identity_objective"
                ]))
                and all(
                    math.isfinite(float(candidate["objective"]))
                    and math.isfinite(float(candidate[
                        "physical_proxy_excess"
                    ]))
                    for candidate in decision["selection"][
                        "candidates"
                    ].values()
                )
                for decision in activation_decisions.values()
            )
        scope_safe = bool(
            selected_audits
            and all(
                float(row["raw_audit"]["scope"][
                    "outside_case_group_or_ownership_abs_max"
                ]) == 0.0
                for row in selected_audits
            )
        )
        group_coverage_complete = all(
            selected_projected_by_group.get(group, 0) >= count
            for group, count in required_by_group.items()
        )
        single_identity_safe = bool(
            not false_activation_uids
            and identity_rms_max <= 1.0e-7
        )
        numeric_audit_complete = bool(
            decision_numeric_complete
            and correction_numeric_failures == 0
            and selected_audits
        )
        activation_aware_supported = bool(
            cross_closure_complete
            and group_coverage_complete
            and single_identity_safe
            and scope_safe
            and numeric_audit_complete
            and selected_proxy_nonregression_complete
            and selected_severity_condition_complete
        )
        activation_summary = {
            "selection_protocol": (
                "severity_then_guard_proxy_lexicographic_v1"
                if args.activation_aware_g1
                else "identity_or_lowest_observable_anchor_objective_v1"
            ),
            "identity_is_explicit_zero_candidate": True,
            "nonidentity_candidate_radius_rms": float(args.target_rms),
            "activation_proxy_tolerance": float(
                args.activation_proxy_tolerance
            ) if not args.activation_aware_g1 else None,
            "activation_relative_improvement": (
                float(args.activation_relative_improvement)
                if not args.activation_aware_g1 else None
            ),
            "observable_severity_envelope": (
                str(severity_envelope_path)
                if severity_envelope_path is not None else None
            ),
            "observable_severity_envelope_sha256": (
                _file_sha256(severity_envelope_path)
                if severity_envelope_path is not None else None
            ),
            "severity_envelope_calibration_split": (
                "train" if args.activation_aware_g1 else None
            ),
            "severity_envelope_role_used_offline_only": bool(
                args.activation_aware_g1
            ),
            "guard_aligned_proxy_terms": (
                dict(G1_GUARD_PROXY_TERMS)
                if args.activation_aware_g1 else None
            ),
            "guard_proxy_nonregression_tolerance": (
                float(args.guard_proxy_nonregression_tolerance)
                if args.activation_aware_g1 else None
            ),
            "observable_severity_channels": (
                list(G1_SEVERITY_CHANNELS)
                if args.activation_aware_g1 else None
            ),
            "activation_decision_role_label_consumed": False,
            "activation_decision_teacher_kind_consumed": False,
            "activation_decision_group_consumed": False,
            "activation_decision_hidden_clean_consumed": False,
            "activation_decision_fixed_guard_consumed": False,
            "fixed_guard_evaluation_only": True,
            "selected_method_counts": selected_method_counts,
            "required_projected_count_by_group": dict(
                required_by_group
            ),
            "selected_projected_count_by_group": dict(
                selected_projected_by_group
            ),
            "false_activation_case_uids": false_activation_uids,
            "missed_cross_case_uids": missed_cross_uids,
            "single_control_applied_tangent_rms_max": identity_rms_max,
            "single_control_applied_tangent_rms_by_case": (
                identity_rms_by_uid
            ),
            "correction_numeric_failure_count": (
                correction_numeric_failures
            ),
            "cross_exact_closure_complete": cross_closure_complete,
            "group_coverage_complete": group_coverage_complete,
            "single_identity_safe": single_identity_safe,
            "scope_safe": scope_safe,
            "numeric_audit_complete": numeric_audit_complete,
            "selected_guard_proxy_nonregression_complete": (
                selected_proxy_nonregression_complete
            ),
            "selected_severity_condition_complete": (
                selected_severity_condition_complete
            ),
            "activation_aware_supported": activation_aware_supported,
            "decisions": activation_decisions,
        }

    base_cross_long = variants["adapter"][
        "projected_pass_count_by_group"
    ].get("cross_long", 0)
    supported_budgets = []
    for budget in budgets:
        euclidean = variants[f"euclidean_projected_k{budget}"]
        riemannian = variants[f"riemannian_retraction_k{budget}"]
        riemannian_cross_long = riemannian[
            "projected_pass_count_by_group"
        ].get("cross_long", 0)
        euclidean_cross_long = euclidean[
            "projected_pass_count_by_group"
        ].get("cross_long", 0)
        if (
            riemannian_cross_long > base_cross_long
            and riemannian_cross_long > euclidean_cross_long
            and riemannian["numeric_failure_count"] == 0
        ):
            supported_budgets.append(int(budget))
    riemannian_supported = bool(supported_budgets)
    hard_negative_path = destination / "validation_hard_negatives.pt"
    m.torch.save({
        "schema": HARD_NEGATIVE_SCHEMA,
        "split": "validation",
        "training_allowed": False,
        "teacher_eligible": False,
        "directions_serialized": False,
        "records": hard_negatives,
    }, hard_negative_path)
    report = {
        "schema": (
            G1_SCHEMA
            if args.activation_aware_g1
            else ACTIVATION_AWARE_SCHEMA
            if args.activation_aware
            else SCHEMA
        ),
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "train_teacher_bank": (
            str(train_teacher_path) if train_teacher_path else None
        ),
        "validation_teacher_bank": str(teacher_path),
        "adapter_state": str(state_path),
        "observable_adapter_gate_mode": _gate_mode,
        "adapter_role": "learner_warm_start",
        "correction_gradient_protocol": "stop_gradient",
        "future_end_to_end_gradient_protocol": IMPLICIT_BACKWARD_PROTOCOL,
        "unrolled_backward_allowed": False,
        "target_rms": float(args.target_rms),
        "euclidean_baseline_protocol": (
            "tangent_gradient_then_exact_sphere_projection_then_retraction"
        ),
        "riemannian_protocol": (
            "moving_product_tangent_retraction_then_original_anchor_sphere_projection"
        ),
        "same_checkpoint_cases_radius_and_budget": True,
        "validation_recycled_as_teacher": False,
        "pseudo_teachers_generated": False,
        "activation_aware": activation_enabled,
        "observable_severity_guard_aligned": bool(
            args.activation_aware_g1
        ),
        "activation_severity_source": (
            "frozen_train_single_observable_envelope"
            if args.activation_aware_g1 else None
        ),
        "correction_guard_proxy_source": (
            "observable_refiner_objective_same_source_signed_margins"
            if args.activation_aware_g1 else None
        ),
        "complete_fixed_guard_used_for_candidate_selection": False,
        "complete_fixed_guard_used_for_final_acceptance": True,
        "inference_role_label_consumed": False,
        "inference_teacher_kind_consumed": False,
        "inference_validation_label_consumed": False,
        "inference_hidden_clean_consumed": False,
        "activation_aware_summary": activation_summary,
        "activation_aware_supported": activation_aware_supported,
        "exact_radius_normalization_by_case": radius,
        "variants": variants,
        "hard_negative_artifact": str(hard_negative_path),
        "hard_negative_training_allowed": False,
        "riemannian_correction_supported": riemannian_supported,
        "riemannian_supported_budgets": supported_budgets,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "numeric_audit_complete": bool(
            all(row["exact_audits"] for row in variants.values())
            and (
                not activation_enabled
                or activation_summary["numeric_audit_complete"]
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = destination / "fixed_budget_correction.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": (
            "v15_15g1_observable_guard_aligned_complete"
            if args.activation_aware_g1
            else "v15_15g_activation_aware_complete"
            if args.activation_aware
            else "v15_15g_fixed_budget_correction_complete"
        ),
        "report": str(report_path),
        "riemannian_correction_supported": riemannian_supported,
        "activation_aware_supported": activation_aware_supported,
        "numeric_audit_complete": report["numeric_audit_complete"],
    }), flush=True)
    if activation_enabled:
        return 0 if activation_aware_supported else 2
    return 0 if report["numeric_audit_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-teacher-bank", required=True)
    parser.add_argument("--train-teacher-bank")
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--steps", type=int, nargs="+", default=(2, 3, 5))
    parser.add_argument("--target-rms", type=float, default=1.0e-4)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--trust-fraction", type=float, default=0.5)
    parser.add_argument("--activation-aware", action="store_true")
    parser.add_argument("--activation-aware-g1", action="store_true")
    parser.add_argument(
        "--activation-proxy-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--activation-relative-improvement", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--severity-envelope-margin-fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--severity-envelope-absolute-margin", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--severity-scale-floor", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--guard-proxy-nonregression-tolerance",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--guard-proxy-scale-floor", type=float, default=1.0e-6
    )
    parser.add_argument("--ik-iterations", type=int, default=6)
    parser.add_argument("--damping", type=float, default=1.0e-4)
    parser.add_argument("--jacobian-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--stiffness-ceiling", type=float, default=1.0e4)
    parser.add_argument(
        "--acceleration-regularization", type=float, default=1.0e-2
    )
    parser.add_argument("--jerk-regularization", type=float, default=1.0e-3)
    args = parser.parse_args()
    if args.target_rms != 1.0e-4:
        parser.error("--target-rms must remain exactly 1e-4")
    if tuple(sorted(set(args.steps))) != (2, 3, 5):
        parser.error("--steps must contain exactly 2 3 5")
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if not 0.0 < args.trust_fraction <= 1.0:
        parser.error("--trust-fraction must be in (0, 1]")
    if args.activation_proxy_tolerance < 0.0:
        parser.error("--activation-proxy-tolerance must be non-negative")
    if args.activation_aware and args.activation_aware_g1:
        parser.error("choose one activation-aware protocol")
    if args.activation_aware_g1 and not args.train_teacher_bank:
        parser.error("--activation-aware-g1 requires --train-teacher-bank")
    if args.severity_envelope_margin_fraction < 0.0:
        parser.error("severity envelope margin fraction must be non-negative")
    if args.severity_envelope_absolute_margin < 0.0:
        parser.error("severity envelope absolute margin must be non-negative")
    if args.severity_scale_floor <= 0.0:
        parser.error("severity scale floor must be positive")
    if args.guard_proxy_nonregression_tolerance < 0.0:
        parser.error("guard proxy tolerance must be non-negative")
    if args.guard_proxy_scale_floor <= 0.0:
        parser.error("guard proxy scale floor must be positive")
    if not 0.0 < args.activation_relative_improvement < 1.0:
        parser.error(
            "--activation-relative-improvement must be in (0, 1)"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
