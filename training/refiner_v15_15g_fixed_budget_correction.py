"""V15.15g fixed-budget Euclidean/Riemannian correction ablation.

The Adapter is a frozen warm start.  Every method uses the same immutable
Anchor, ownership mask, exact 1e-4 tangent sphere, differentiable constraint
bundle and final raw/Projector/Guard audit.  Correction outputs are detached;
validation successes and failures are never recycled into training data.

The activation-aware mode adds identity as an explicit zero-action candidate
and selects among identity, Adapter, Euclidean and Riemannian candidates using
only observable Anchor-relative objectives.  Offline role/group metadata is
attached only after selection for exact closure reporting.
"""
from __future__ import annotations

import argparse
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

        if activation_aware:
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
            if activation_aware:
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
            not activation_aware
        ),
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
    if args.activation_aware:
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
                    not args.activation_aware
                    and sample["teacher_kind"] != "exact_projected_direction"
                ):
                    continue
                transaction_id = str(sample["transaction_id"])
                domain = domains[transaction_id]
                start, stop = domain["slice"]
                local_case = int(sample["local_case_index"])
                if args.activation_aware:
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
                    activation_aware=bool(args.activation_aware),
                )
                if args.activation_aware:
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
    if args.activation_aware:
        selection_started = time.perf_counter()
        selected_tangent, activation_decisions, selected_method_counts = (
            _activation_aware_selection(
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
        correction_numeric_failures = sum(
            int(report.get("numeric_failure", False))
            for reports in variant_correction_reports.values()
            for report in reports.values()
        )
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
        )
        activation_summary = {
            "selection_protocol": (
                "identity_or_lowest_observable_anchor_objective_v1"
            ),
            "identity_is_explicit_zero_candidate": True,
            "nonidentity_candidate_radius_rms": float(args.target_rms),
            "activation_proxy_tolerance": float(
                args.activation_proxy_tolerance
            ),
            "activation_relative_improvement": float(
                args.activation_relative_improvement
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
            ACTIVATION_AWARE_SCHEMA if args.activation_aware else SCHEMA
        ),
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
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
        "activation_aware": bool(args.activation_aware),
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
                not args.activation_aware
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
            "v15_15g_activation_aware_complete"
            if args.activation_aware
            else "v15_15g_fixed_budget_correction_complete"
        ),
        "report": str(report_path),
        "riemannian_correction_supported": riemannian_supported,
        "activation_aware_supported": activation_aware_supported,
        "numeric_audit_complete": report["numeric_audit_complete"],
    }), flush=True)
    if args.activation_aware:
        return 0 if activation_aware_supported else 2
    return 0 if report["numeric_audit_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-teacher-bank", required=True)
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--steps", type=int, nargs="+", default=(2, 3, 5))
    parser.add_argument("--target-rms", type=float, default=1.0e-4)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--trust-fraction", type=float, default=0.5)
    parser.add_argument("--activation-aware", action="store_true")
    parser.add_argument(
        "--activation-proxy-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--activation-relative-improvement", type=float, default=1.0e-3
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
    if not 0.0 < args.activation_relative_improvement < 1.0:
        parser.error(
            "--activation-relative-improvement must be in (0, 1)"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
