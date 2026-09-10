"""Development-only V15.14h full-tangent nonlinear feasibility oracle.

The oracle searches the complete owned ``[frame, 75]`` product tangent for
four frozen cases.  Every evaluated candidate lies on the resolved 1e-4
case-local output sphere and receives the unchanged fixed-bank exact closure.
It does not train, promote, publish, replay, or generate motion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from motion_geometry.product_manifold import product_exp_torch, product_log_torch
from training import motion_models as m
from training import refiner_case_local_finite_radius_cone_probe as case_probe
from training import refiner_exact_closure_tangent_probe as closure_probe
from training import refiner_group_local_nullspace_cone_probe as group_probe
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_14h_case_local_full_tangent_nonlinear_oracle_v1"
PROTOCOL = "fixed_radius_full_tangent_augmented_lagrangian_multistart_v1"
TARGET_CASES = {
    10: "single_short",
    14: "single_short",
    16: "cross_short",
    29: "cross_long",
}
PRIMARY_CASES = (16, 29)
START_MODES = (
    "endpoint",
    "temporal",
    "endpoint_temporal",
    "witness_correction",
    "orthogonal_0",
    "orthogonal_1",
    "orthogonal_2",
    "orthogonal_3",
)
DEFAULT_TARGET_RMS = 1.0e-4
DEFAULT_ITERATIONS = 60
KINETIC_PRIOR_WEIGHT = 1.0e-6
RADIUS_TOLERANCE = 1.0e-3


def _mapped_tangent(local_direction, weighted):
    return weighted * local_direction.unsqueeze(0)


def _case_tensor_rms(tangent, seam, case_index):
    selected = tangent[int(case_index):int(case_index) + 1]
    selected_seam = seam[int(case_index):int(case_index) + 1]
    active = (selected_seam >= 0.5).expand_as(selected)
    values = selected[active]
    if values.numel() == 0:
        return tangent.sum() * 0.0
    return m.torch.sqrt(m.torch.mean(values.square()).clamp_min(1.0e-24))


def _normalize_local_direction(local_direction, weighted, seam, case_index):
    tangent = _mapped_tangent(local_direction, weighted)
    rms = _case_tensor_rms(tangent, seam, case_index)
    if not bool(m.torch.isfinite(rms)) or float(rms.detach()) <= 1.0e-14:
        return m.torch.zeros_like(local_direction), float(rms.detach())
    return local_direction / rms.detach(), float(rms.detach())


def _candidate_on_sphere(
    baseline,
    local_direction,
    weighted,
    seam,
    case_index,
    target_rms,
):
    unscaled = _mapped_tangent(local_direction, weighted)
    direction_rms = _case_tensor_rms(unscaled, seam, case_index)
    unit = unscaled / direction_rms.clamp_min(1.0e-12)
    tangent = float(target_rms) * unit
    candidate = product_exp_torch(baseline, tangent)
    sphere_equality = direction_rms.square() - 1.0
    return candidate, tangent, direction_rms, sphere_equality


def _direction_gradient(scalar, local_edit, *, retain_graph=True):
    gradient = m.torch.autograd.grad(
        scalar,
        local_edit,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if gradient is None:
        return m.torch.zeros_like(local_edit)
    return -gradient.detach()


def _direction_inner(left, right, weighted, seam, case_index):
    left_tangent = _mapped_tangent(left, weighted)
    right_tangent = _mapped_tangent(right, weighted)
    selected = m.torch.zeros_like(seam, dtype=m.torch.bool)
    selected[int(case_index)] = seam[int(case_index)] >= 0.5
    active = selected.expand_as(left_tangent)
    return (left_tangent[active] * right_tangent[active]).sum()


def _orthogonalize(candidate, existing, weighted, seam, case_index):
    result = candidate.clone()
    for direction in existing:
        denominator = _direction_inner(
            direction, direction, weighted, seam, case_index
        )
        if float(denominator.detach()) <= 1.0e-20:
            continue
        coefficient = _direction_inner(
            result, direction, weighted, seam, case_index
        ) / denominator
        result = result - coefficient * direction
    return result


def _deterministic_direction(template, seed):
    index = m.torch.arange(
        template.numel(),
        dtype=template.dtype,
        device=template.device,
    ).reshape_as(template)
    frequency = float(seed + 1)
    return (
        m.torch.sin((index + 1.0) * (0.61803398875 * frequency))
        + 0.5 * m.torch.cos((index + 1.0) * (0.41421356237 + frequency))
    )


def _build_multistarts(baseline, batch, cfg, case_index):
    _, weighted, taper_frames = case_probe._case_activity(
        batch, case_index, cfg
    )
    local_edit = m.torch.zeros(
        baseline.shape[1:-1] + (75,),
        dtype=baseline.dtype,
        device=baseline.device,
        requires_grad=True,
    )
    candidate = product_exp_torch(
        baseline, _mapped_tangent(local_edit, weighted)
    )
    primitives = case_probe._witness_primitives(candidate, batch, cfg)
    witnesses = case_probe._discover_witnesses(
        primitives, batch, case_index, topk=4
    )
    values, _ = case_probe._case_value_tensors(
        candidate, batch, cfg, case_index, witnesses
    )
    endpoint = _direction_gradient(values["endpoint"], local_edit)
    temporal = _direction_gradient(values["temporal"], local_edit)
    mixed = endpoint + temporal
    witness_terms = []
    for witness in witnesses:
        value = values[case_probe._witness_key(witness)]
        scale = value.detach().abs().clamp_min(1.0)
        witness_terms.append(value / scale)
    witness_scalar = (
        m.torch.stack(witness_terms).sum()
        if witness_terms else values["endpoint"] + values["temporal"]
    )
    witness_direction = _direction_gradient(
        witness_scalar, local_edit, retain_graph=False
    )
    raw = {
        "endpoint": endpoint,
        "temporal": temporal,
        "endpoint_temporal": mixed,
        "witness_correction": witness_direction,
    }
    starts = {}
    basis = []
    rows = []
    for name in START_MODES:
        direction = raw.get(name)
        if direction is None:
            seed = int(name.rsplit("_", 1)[1])
            direction = _deterministic_direction(local_edit.detach(), seed)
            direction = _orthogonalize(
                direction, basis, weighted, batch["seam"], case_index
            )
        normalized, before = _normalize_local_direction(
            direction, weighted, batch["seam"], case_index
        )
        available = bool(math.isfinite(before) and before > 1.0e-12)
        if available:
            starts[name] = normalized.detach()
            basis.append(normalized.detach())
        rows.append({
            "start": name,
            "available": available,
            "pre_normalization_case_tangent_rms": before,
        })
    return starts, rows, witnesses, weighted, taper_frames


def _fixed_guard_limits(anchor, relative, absolute, group_name):
    limits = {}
    scales = {}
    for key, value in anchor.items():
        if not key.startswith(f"{group_name}."):
            continue
        allowance = max(
            abs(float(value)) * float(relative[key]),
            float(absolute[key]),
        )
        limit = float(value) + allowance
        limits[key] = limit
        scales[key] = max(abs(limit), abs(float(value)), allowance, 1.0e-6)
    return limits, scales


def _constraint_bundle(
    model,
    batch,
    cfg,
    candidate,
    identity,
    case_index,
    group_name,
    baseline_case,
    guard_limits,
    guard_scales,
):
    case_terms = case_probe._case_terms(candidate, batch, cfg)
    case_values = {
        "endpoint": case_terms["endpoint"][int(case_index)],
        "temporal": case_terms["temporal"][int(case_index)],
    }
    constraints = {
        "case.endpoint_nonregression": (
            case_values["endpoint"] - baseline_case["endpoint"]
        ) / max(abs(float(baseline_case["endpoint"])), 1.0e-6),
        "case.temporal_nonregression": (
            case_values["temporal"] - baseline_case["temporal"]
        ) / max(abs(float(baseline_case["temporal"])), 1.0e-6),
    }
    guard = projected_probe._guard_values_for_prediction(
        model, batch, cfg, candidate, identity
    )
    for key, limit in guard_limits.items():
        constraints[f"guard.{key}"] = (
            guard[key] - float(limit)
        ) / float(guard_scales[key])
    return case_values, guard, constraints


def _case_scientific_status(candidate_values, baseline_case):
    delta = {
        key: float(candidate_values[key]) - float(baseline_case[key])
        for key in ("endpoint", "temporal")
    }
    return case_probe._scientific_status(delta, baseline_case)


def _workspace_displacement(baseline, candidate, case_index):
    with m.torch.no_grad():
        before = m.fk_24_torch(
            baseline[int(case_index):int(case_index) + 1]
        )
        after = m.fk_24_torch(
            candidate[int(case_index):int(case_index) + 1]
        )
        displacement = m.torch.linalg.vector_norm(after - before, dim=-1)
    return {
        "workspace_displacement_mean": float(displacement.mean()),
        "workspace_displacement_max": float(displacement.max()),
        "workspace_displacement_definition": "fk24_joint_position_m",
    }


def _kinetic_prior(tangent, case_index, target_rms):
    local = tangent[int(case_index)] / max(float(target_rms), 1.0e-12)
    velocity = m.torch.diff(local, dim=0)
    acceleration = m.torch.diff(local, n=2, dim=0)
    velocity_term = velocity.square().mean() if velocity.numel() else local.sum() * 0
    acceleration_term = (
        acceleration.square().mean()
        if acceleration.numel() else local.sum() * 0
    )
    return velocity_term + 0.25 * acceleration_term


def _exact_audit(
    model,
    batch,
    cfg,
    baseline,
    identity,
    baseline_guard,
    guard_anchor,
    guard_relative,
    guard_absolute,
    candidate,
    case_index,
    baseline_case,
    target_rms,
    workspace_floor,
):
    with m.torch.no_grad():
        case_terms = case_probe._case_terms(candidate, batch, cfg)
        candidate_case = {
            key: float(case_terms[key][int(case_index)])
            for key in ("endpoint", "temporal")
        }
        case_scientific = _case_scientific_status(
            candidate_case, baseline_case
        )
        (
            guard_values,
            guard_passed,
            blockers,
            guard_details,
            observable_delta,
            scientific,
        ) = group_probe._audit_prediction(
            model,
            batch,
            cfg,
            candidate,
            identity,
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
        radius = projected_probe._motion_edit_rms(
            baseline[int(case_index):int(case_index) + 1],
            candidate[int(case_index):int(case_index) + 1],
            batch["seam"][int(case_index):int(case_index) + 1],
        )
        tangent = product_log_torch(baseline, candidate)
        scope = case_probe._case_scope(
            group_probe._product_action(tangent), batch, case_index
        )
        workspace = _workspace_displacement(
            baseline, candidate, case_index
        )
    radius_resolved = bool(
        math.isfinite(radius)
        and abs(radius / float(target_rms) - 1.0) <= RADIUS_TOLERANCE
    )
    workspace_resolved = bool(
        workspace["workspace_displacement_max"] > float(workspace_floor)
    )
    passed = bool(
        radius_resolved
        and workspace_resolved
        and scope["scope_safe"]
        and case_scientific["passed"]
        and guard_passed
        and scientific["raw_exact_common_descent"]
    )
    return {
        "passed": passed,
        "case_scientific": case_scientific,
        "fixed_guard_passed": guard_passed,
        "fixed_guard_blockers": blockers,
        "fixed_guard_values": guard_values,
        "fixed_guard_details": guard_details,
        "observable_residual_delta": observable_delta,
        "full_scientific_status": scientific,
        "achieved_case_output_tangent_rms": radius,
        "radius_equality_resolved": radius_resolved,
        "workspace_observable_resolved": workspace_resolved,
        "workspace": workspace,
        "scope": scope,
    }


def _optimize_start(
    *,
    model,
    batch,
    cfg,
    baseline,
    identity,
    baseline_guard,
    guard_anchor,
    guard_relative,
    guard_absolute,
    case_index,
    group_name,
    start_name,
    start_direction,
    weighted,
    baseline_case,
    guard_limits,
    guard_scales,
    target_rms,
    iterations,
    learning_rate,
    initial_penalty,
    workspace_floor,
):
    local = m.torch.nn.Parameter(start_direction.clone())
    optimizer = m.torch.optim.Adam([local], lr=float(learning_rate))
    constraint_names = [
        "case.endpoint_nonregression",
        "case.temporal_nonregression",
        *(f"guard.{key}" for key in guard_limits),
    ]
    multipliers = m.torch.zeros(
        len(constraint_names), dtype=baseline.dtype, device=baseline.device
    )
    equality_multiplier = m.torch.zeros(
        (), dtype=baseline.dtype, device=baseline.device
    )
    penalty = float(initial_penalty)
    equality_penalty = float(initial_penalty)
    history = []
    best_candidate = None
    best_score = math.inf
    blocker_counts = Counter()
    previous_violation = math.inf
    started = time.perf_counter()

    for step in range(int(iterations) + 1):
        optimizer.zero_grad(set_to_none=True)
        candidate, tangent, direction_rms, sphere_equality = (
            _candidate_on_sphere(
                baseline,
                local,
                weighted,
                batch["seam"],
                case_index,
                target_rms,
            )
        )
        case_values, _, constraints = _constraint_bundle(
            model,
            batch,
            cfg,
            candidate,
            identity,
            case_index,
            group_name,
            baseline_case,
            guard_limits,
            guard_scales,
        )
        constraint_vector = m.torch.stack([
            constraints[name] for name in constraint_names
        ])
        positive = m.torch.relu(constraint_vector)
        endpoint_scale = max(abs(float(baseline_case["endpoint"])), 1.0e-6)
        temporal_scale = max(abs(float(baseline_case["temporal"])), 1.0e-6)
        if start_name == "endpoint":
            weights = (1.0, 0.0)
        elif start_name == "temporal":
            weights = (0.0, 1.0)
        else:
            weights = (0.5, 0.5)
        scientific_objective = (
            weights[0] * case_values["endpoint"] / endpoint_scale
            + weights[1] * case_values["temporal"] / temporal_scale
        )
        kinetic = _kinetic_prior(tangent, case_index, target_rms)
        loss = scientific_objective
        loss = loss + KINETIC_PRIOR_WEIGHT * kinetic
        loss = loss + (multipliers * positive).sum()
        loss = loss + 0.5 * penalty * positive.square().sum()
        loss = loss + equality_multiplier * sphere_equality
        loss = loss + 0.5 * equality_penalty * sphere_equality.square()

        audit = _exact_audit(
            model,
            batch,
            cfg,
            baseline,
            identity,
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
            candidate.detach(),
            case_index,
            baseline_case,
            target_rms,
            workspace_floor,
        )
        blocker_counts.update(audit["fixed_guard_blockers"])
        score = (
            float(audit["case_scientific"]["endpoint_delta"])
            + float(audit["case_scientific"]["temporal_delta"])
        )
        if audit["passed"] and score < best_score:
            best_score = score
            best_candidate = candidate.detach().clone()
        maximum_violation = (
            float(positive.max().detach()) if positive.numel() else 0.0
        )
        row = {
            "step": int(step),
            "loss": float(loss.detach()),
            "scientific_objective": float(scientific_objective.detach()),
            "kinetic_prior": float(kinetic.detach()),
            "sphere_equality_constraint": float(sphere_equality.detach()),
            "unscaled_direction_rms": float(direction_rms.detach()),
            "normalized_constraint_residuals": {
                name: float(constraints[name].detach())
                for name in constraint_names
            },
            "maximum_positive_constraint_violation": maximum_violation,
            "alm_penalty": penalty,
            "exact_audit": audit,
        }
        history.append(row)
        if step % 10 == 0 or audit["passed"]:
            print(json.dumps({
                "stage": "v15_14h_oracle_step",
                "case_index": int(case_index),
                "group": group_name,
                "start": start_name,
                "step": int(step),
                "passed": audit["passed"],
                "max_constraint_violation": maximum_violation,
                "guard_blockers": audit["fixed_guard_blockers"],
            }), flush=True)
        if audit["passed"]:
            break
        if step == int(iterations):
            break
        loss.backward()
        gradient_norm = float(
            m.torch.nn.utils.clip_grad_norm_([local], max_norm=10.0)
        )
        row["gradient_norm_before_clip"] = gradient_norm
        optimizer.step()
        with m.torch.no_grad():
            if not bool(m.torch.isfinite(local).all()):
                local.copy_(start_direction)
                optimizer.state.clear()
                row["nonfinite_update_reset"] = True
            multipliers.copy_(
                m.torch.clamp(
                    multipliers + penalty * constraint_vector.detach(),
                    min=0.0,
                )
            )
            equality_multiplier.add_(
                equality_penalty * sphere_equality.detach()
            )
        if (step + 1) % 10 == 0:
            if maximum_violation > 0.75 * previous_violation:
                penalty = min(penalty * 2.0, 1.0e6)
                equality_penalty = min(equality_penalty * 2.0, 1.0e6)
            previous_violation = min(previous_violation, maximum_violation)
            row["best_maximum_violation"] = previous_violation

    return best_candidate, {
        "case_index": int(case_index),
        "group": group_name,
        "start": start_name,
        "iterations_completed": len(history) - 1,
        "raw_exact_candidate_found": bool(best_candidate is not None),
        "best_scientific_delta_sum": (
            best_score if best_candidate is not None else None
        ),
        "fixed_guard_blocker_counts": dict(blocker_counts),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _project_raw_candidate(
    *,
    model,
    batch,
    cfg,
    baseline,
    identity,
    baseline_guard,
    guard_anchor,
    guard_relative,
    guard_absolute,
    raw,
    case_index,
    baseline_case,
    target_rms,
    workspace_floor,
    args,
):
    full_projected, projector_report = (
        projected_probe.weighted_dls_contact_project_torch(
            batch["bad"],
            baseline,
            raw,
            batch["seam"],
            cfg,
            iterations=args.ik_iterations,
            damping=args.damping,
            jacobian_epsilon=args.jacobian_epsilon,
            stiffness_ceiling=args.stiffness_ceiling,
            acceleration_regularization=args.acceleration_regularization,
            jerk_regularization=args.jerk_regularization,
            scientific_context={
                "model": model,
                "batch": batch,
                "identity": identity,
            },
        )
    )
    trials = []
    accepted = None
    accepted_factor = None
    for factor in closure_probe.PROJECTION_FACTORS:
        candidate = closure_probe._interpolate_projector_correction(
            raw, full_projected, factor
        )
        case_selector = m.torch.zeros(
            candidate.shape[0], dtype=m.torch.bool, device=candidate.device
        )
        case_selector[int(case_index)] = True
        candidate = m.torch.where(
            case_selector[:, None, None], candidate, baseline
        )
        action = group_probe._product_action(
            product_log_torch(baseline, candidate)
        )
        candidate, radius_scale, radius = case_probe._candidate_from_action(
            baseline,
            action,
            batch["seam"],
            target_rms,
            case_index,
        )
        audit = _exact_audit(
            model,
            batch,
            cfg,
            baseline,
            identity,
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
            candidate,
            case_index,
            baseline_case,
            target_rms,
            workspace_floor,
        )
        trials.append({
            "factor": factor,
            "fixed_radius_rematerialization_scale": radius_scale,
            "fixed_radius_before_exact_audit": radius,
            "audit": audit,
        })
        if audit["passed"] and projector_report["scope_safe"]:
            accepted = candidate.detach()
            accepted_factor = factor
            break
    return accepted, {
        "projector": projector_report,
        "projection_trials": trials,
        "projection_backtracking_factor": accepted_factor,
        "effective_projected_candidate": bool(accepted is not None),
    }


def run(args):
    started = time.perf_counter()
    source = Path(args.source_diagnostic_dir)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(
        (source / "diagnostic_report.json").read_text(encoding="utf-8-sig")
    )
    if source_report.get("published") or source_report.get("diagnostic_ready"):
        raise RuntimeError("V15.14h requires an unpublished failed diagnostic")
    state = m.torch.load(
        source / "diagnostic_state.pt", map_location="cpu", weights_only=False
    )
    artifact = m.torch.load(
        source / "fit_bank.pt", map_location="cpu", weights_only=False
    )
    if state.get("formal_checkpoint") or artifact.get("formal_checkpoint"):
        raise RuntimeError("formal checkpoint input is forbidden")

    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.14h full-tangent oracle requires CUDA")
    batch, _, schedule = projected_probe._materialize_first_transaction(
        artifact, device
    )
    if max(TARGET_CASES) >= int(batch["bad"].shape[0]):
        raise RuntimeError("fixed oracle case indices are absent from the bank")
    for case_index, expected_group in TARGET_CASES.items():
        observed_group = m.REFINER_GROUP_LABELS[
            int(batch["group"][case_index].detach())
        ]
        if observed_group != expected_group:
            raise RuntimeError(
                f"case {case_index} expected {expected_group}, got {observed_group}"
            )

    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.train()
    model.requires_grad_(False)
    baseline, identity = projected_probe._model_prediction(model, batch, cfg)
    baseline_guard = projected_probe._float_guard(
        projected_probe._guard_values_for_prediction(
            model, batch, cfg, baseline, identity
        )
    )
    contract = source_report["group_guard_contract"]
    guard_anchor = contract["initial_anchor"]
    guard_relative = contract["relative_tolerance"]
    guard_absolute = contract["absolute_tolerance"]
    baseline_guard_passed, baseline_blockers, baseline_guard_details = (
        projected_probe._audit_guard_candidate(
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
    )
    baseline_case_terms = case_probe._case_terms(baseline, batch, cfg)
    workspace_floor = max(1.0e-8, float(args.target_rms) * 0.01)

    case_reports = []
    raw_best = {}
    raw_start_counts = Counter()
    projected_counts = Counter()
    all_scope = []
    for case_index, group_name in TARGET_CASES.items():
        starts, start_rows, witnesses, weighted, taper_frames = (
            _build_multistarts(baseline, batch, cfg, case_index)
        )
        baseline_case = {
            key: float(baseline_case_terms[key][case_index].detach())
            for key in ("endpoint", "temporal")
        }
        guard_limits, guard_scales = _fixed_guard_limits(
            guard_anchor,
            guard_relative,
            guard_absolute,
            group_name,
        )
        trials = []
        best_candidate = None
        best_score = math.inf
        for start_name in START_MODES:
            if start_name not in starts:
                trials.append({
                    "case_index": int(case_index),
                    "group": group_name,
                    "start": start_name,
                    "raw_exact_candidate_found": False,
                    "reason": "unavailable_nonzero_start_direction",
                    "history": [],
                })
                continue
            candidate, trial = _optimize_start(
                model=model,
                batch=batch,
                cfg=cfg,
                baseline=baseline,
                identity=identity,
                baseline_guard=baseline_guard,
                guard_anchor=guard_anchor,
                guard_relative=guard_relative,
                guard_absolute=guard_absolute,
                case_index=case_index,
                group_name=group_name,
                start_name=start_name,
                start_direction=starts[start_name],
                weighted=weighted,
                baseline_case=baseline_case,
                guard_limits=guard_limits,
                guard_scales=guard_scales,
                target_rms=float(args.target_rms),
                iterations=int(args.iterations),
                learning_rate=float(args.learning_rate),
                initial_penalty=float(args.initial_penalty),
                workspace_floor=workspace_floor,
            )
            trials.append(trial)
            if candidate is not None:
                raw_start_counts.update([group_name])
                score = float(trial["best_scientific_delta_sum"])
                if score < best_score:
                    best_score = score
                    best_candidate = candidate
        projection = None
        projected = None
        if best_candidate is not None:
            raw_best[case_index] = best_candidate
            np.save(
                destination / f"raw_case_{case_index}.npy",
                best_candidate.detach().cpu().numpy(),
            )
            projected, projection = _project_raw_candidate(
                model=model,
                batch=batch,
                cfg=cfg,
                baseline=baseline,
                identity=identity,
                baseline_guard=baseline_guard,
                guard_anchor=guard_anchor,
                guard_relative=guard_relative,
                guard_absolute=guard_absolute,
                raw=best_candidate,
                case_index=case_index,
                baseline_case=baseline_case,
                target_rms=float(args.target_rms),
                workspace_floor=workspace_floor,
                args=args,
            )
            if projected is not None:
                projected_counts.update([group_name])
                np.save(
                    destination / f"projected_case_{case_index}.npy",
                    projected.detach().cpu().numpy(),
                )
        for trial in trials:
            for row in trial.get("history", []):
                scope = row.get("exact_audit", {}).get("scope")
                if scope:
                    all_scope.append(scope)
        case_reports.append({
            "case_index": int(case_index),
            "group": group_name,
            "role": "primary" if case_index in PRIMARY_CASES else "control",
            "baseline_case_scientific": baseline_case,
            "taper_frames": taper_frames,
            "start_directions": start_rows,
            "witnesses": witnesses,
            "guard_limits": guard_limits,
            "guard_scales": guard_scales,
            "start_trials": trials,
            "raw_exact_candidate_found": bool(best_candidate is not None),
            "projector_result": projection,
            "effective_projected_candidate": bool(projected is not None),
        })

    outside_max = max(
        (
            float(row["outside_case_group_or_ownership_abs_max"])
            for row in all_scope
        ),
        default=0.0,
    )
    expected_trials = len(TARGET_CASES) * len(START_MODES)
    completed_trials = sum(
        len(case["start_trials"]) for case in case_reports
    )
    available_trials = sum(
        int(row["available"])
        for case in case_reports
        for row in case["start_directions"]
    )
    audited_trials = sum(
        bool(trial.get("history"))
        for case in case_reports
        for trial in case["start_trials"]
    )
    numeric_complete = bool(
        completed_trials == expected_trials
        and audited_trials == available_trials
        and all(
            "exact_audit" in row
            for case in case_reports
            for trial in case["start_trials"]
            for row in trial.get("history", [])
        )
    )
    raw_counts = Counter(TARGET_CASES[index] for index in raw_best)
    ghost_candidate_count = sum(
        not row["exact_audit"]["workspace_observable_resolved"]
        for case in case_reports
        for trial in case["start_trials"]
        for row in trial.get("history", [])
    )
    primary_raw = all(case_index in raw_best for case_index in PRIMARY_CASES)
    route_supported = bool(
        projected_counts["cross_short"] > 0
        and projected_counts["cross_long"] > 0
        and outside_max == 0.0
        and numeric_complete
    )
    if route_supported:
        status = "full_tangent_projected_cross_feasibility_exists"
    elif primary_raw:
        status = "full_tangent_raw_cross_feasible_projector_failed"
    else:
        status = "full_tangent_multistart_no_raw_feasible_at_required_radius"

    result = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "development_only": True,
        "training_started": False,
        "formal_checkpoint": False,
        "publish_allowed": False,
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "source_diagnostic": str(source.resolve()),
        "source_schema": source_report.get("schema"),
        "transaction_context_indices": list(schedule),
        "selected_cases": TARGET_CASES,
        "primary_cases": list(PRIMARY_CASES),
        "start_modes": list(START_MODES),
        "target_output_tangent_rms": float(args.target_rms),
        "smaller_radius_evidence_allowed": False,
        "sphere_equality_constraint_active": True,
        "optimizer": "augmented_lagrangian_adam_on_fixed_radius_sphere",
        "iterations_per_start": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "initial_alm_penalty": float(args.initial_penalty),
        "kinetic_prior_weight": KINETIC_PRIOR_WEIGHT,
        "kinetic_prior": "tangent_velocity_l2_plus_quarter_acceleration_l2",
        "workspace_displacement_definition": "fk24_joint_position_m",
        "workspace_displacement_numeric_floor": workspace_floor,
        "baseline_fixed_guard_passed": baseline_guard_passed,
        "baseline_fixed_guard_blockers": baseline_blockers,
        "baseline_fixed_guard_details": baseline_guard_details,
        "raw_exact_candidate_count_by_group": dict(raw_counts),
        "raw_exact_candidate_start_count_by_group": dict(raw_start_counts),
        "effective_projected_candidate_count_by_group": dict(
            projected_counts
        ),
        "cross_short_effective_projected_candidate_count": int(
            projected_counts["cross_short"]
        ),
        "cross_long_effective_projected_candidate_count": int(
            projected_counts["cross_long"]
        ),
        "primary_cross_raw_feasible": primary_raw,
        "routing_architecture_pivot_supported": route_supported,
        "outside_case_group_or_ownership_abs_max": outside_max,
        "scope_safe": bool(outside_max == 0.0),
        "start_trial_count_expected": expected_trials,
        "start_trial_count": completed_trials,
        "available_start_trial_count": available_trials,
        "numerically_audited_start_trial_count": audited_trials,
        "ghost_workspace_candidate_count": ghost_candidate_count,
        "numeric_audit_complete": numeric_complete,
        "feasibility_status": status,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "case_reports": case_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = destination / "case_local_full_tangent_oracle.report.json"
    m.save_json(result, report_path)
    print(json.dumps({
        "stage": "refiner_v15_14h_full_tangent_oracle_complete",
        "report": str(report_path.resolve()),
        "raw_exact_candidate_count_by_group": dict(raw_counts),
        "raw_exact_candidate_start_count_by_group": dict(raw_start_counts),
        "effective_projected_candidate_count_by_group": dict(
            projected_counts
        ),
        "routing_architecture_pivot_supported": route_supported,
        "feasibility_status": status,
        "scope_safe": result["scope_safe"],
        "numeric_audit_complete": numeric_complete,
    }), flush=True)
    return 0 if route_supported else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-diagnostic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--target-rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--learning-rate", type=float, default=2.0e-2)
    parser.add_argument("--initial-penalty", type=float, default=10.0)
    parser.add_argument("--ik-iterations", type=int, default=6)
    parser.add_argument("--damping", type=float, default=1.0e-4)
    parser.add_argument("--jacobian-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--stiffness-ceiling", type=float, default=1.0e4)
    parser.add_argument(
        "--acceleration-regularization", type=float, default=1.0e-2
    )
    parser.add_argument("--jerk-regularization", type=float, default=1.0e-3)
    args = parser.parse_args()
    if args.target_rms != DEFAULT_TARGET_RMS:
        parser.error(f"--target-rms must remain exactly {DEFAULT_TARGET_RMS:g}")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.learning_rate <= 0.0 or args.initial_penalty <= 0.0:
        parser.error("optimizer values must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
