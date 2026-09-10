"""Development-only V15.14e group-local exact-closure cone probe.

Directions live in a [case, frame, 79] product-action tensor.  Contact action
channels are identically zero and the remaining 75 channels are a legitimate
product-manifold tangent.  Endpoint and temporal directions are isolated by
fixed-bank group, ownership, and C2 seam activity before an exact one-sided
finite-difference hard-Guard Jacobian projects their two-dimensional subgroup
span into its active non-regression nullspace.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from motion_geometry.product_manifold import product_exp_torch
from training import motion_models as m
from training import refiner_exact_closure_tangent_probe as closure_probe
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_14e_group_local_boundary_nullspace_cone_v1"
PROTOCOL = "fixed_bank_group_local_output_tangent_fd_nullspace_cone_v1"
DEFAULT_FD_EPSILON = 1.0e-6
DEFAULT_TARGET_RMS = 1.0e-4
PAIR_GRID_DENOMINATOR = 64
CONTACT_CHANNELS = 4
HARD_SUFFIXES = (
    ".joint_jerk_p95",
    ".joint_jerk_max",
    ".joint_jerk_window_p95",
    ".extremity_jerk_p95",
    ".extremity_jerk_window_p95",
    ".foot_skate_p95",
    ".foot_skate_max",
    ".support_drift_p95",
    ".support_drift_max",
    ".penetration",
    ".fixed_support",
    ".boundary",
    ".fidelity_geometry",
    ".fidelity_contact",
    ".fidelity_temporal",
    ".fidelity_support",
)


def _product_action(tangent):
    contact = m.torch.zeros(
        tangent.shape[:-1] + (CONTACT_CHANNELS,),
        dtype=tangent.dtype,
        device=tangent.device,
    )
    return m.torch.cat((contact, tangent), dim=-1)


def _tangent_from_action(action):
    if action.shape[-1] != 79:
        raise ValueError("product action must end in 79 channels")
    if bool((action[..., :CONTACT_CHANNELS] != 0).any()):
        raise ValueError("contact action channels must remain exactly zero")
    return action[..., CONTACT_CHANNELS:]


def _active_tangent_rms(tangent, seam):
    active = (seam >= 0.5).expand_as(tangent)
    values = tangent[active]
    if values.numel() == 0:
        return 0.0
    return float(m.torch.sqrt(m.torch.mean(values.square())).detach())


def _normalize_action(action, seam):
    tangent = _tangent_from_action(action)
    rms = _active_tangent_rms(tangent, seam)
    if not math.isfinite(rms) or rms <= 1.0e-14:
        return m.torch.zeros_like(action), rms
    return action / rms, rms


def _candidate_at_action_rms(baseline, action, seam, target):
    tangent = _tangent_from_action(action)
    scale = float(target)
    candidate = baseline
    achieved = 0.0
    for _ in range(6):
        candidate = product_exp_torch(baseline, scale * tangent)
        achieved = projected_probe._motion_edit_rms(
            baseline, candidate, seam
        )
        if not math.isfinite(achieved) or achieved <= 1.0e-16:
            scale *= 10.0
            continue
        if achieved >= target and abs(achieved / target - 1.0) <= 1.0e-3:
            break
        scale *= (float(target) / achieved) * 1.0001
    return candidate.detach(), float(scale), float(achieved)


def _audit_prediction(
    model,
    batch,
    cfg,
    prediction,
    identity,
    baseline_guard,
    guard_anchor,
    guard_relative,
    guard_absolute,
):
    values = projected_probe._float_guard(
        projected_probe._guard_values_for_prediction(
            model, batch, cfg, prediction, identity
        )
    )
    passed, blockers, details = projected_probe._audit_guard_candidate(
        values,
        guard_anchor,
        guard_relative,
        guard_absolute,
    )
    delta = {
        key: values[key] - baseline_guard[key]
        for key in closure_probe._observable_values(baseline_guard)
    }
    scientific = closure_probe._scientific_delta_status(delta, details)
    return values, passed, blockers, details, delta, scientific


def _group_activity(batch, group_index, cfg):
    group = (batch["group"] == int(group_index))[:, None, None]
    taper_frames = max(
        1,
        int(getattr(cfg, "product_refiner_residual_taper_frames", 3)),
    )
    owned, c2 = projected_probe._ownership_c2_activity(
        batch["seam"], taper_frames
    )
    activity = group & owned[..., None]
    weighted = activity.to(batch["bad"].dtype) * c2[..., None]
    return activity, weighted, taper_frames


def _group_local_scientific_directions(
    model,
    batch,
    cfg,
    baseline,
    identity,
):
    """Create endpoint/temporal output directions on one fixed closure graph."""
    edit = m.torch.zeros(
        baseline.shape[:-1] + (75,),
        dtype=baseline.dtype,
        device=baseline.device,
        requires_grad=True,
    )
    candidate = product_exp_torch(baseline, edit)
    guard_tensors = projected_probe._guard_values_for_prediction(
        model, batch, cfg, candidate, identity
    )
    observables = closure_probe._observable_values(guard_tensors)
    expected = 2 * len(m.REFINER_GROUP_LABELS)
    if len(observables) != expected:
        raise RuntimeError(
            f"V15.14e expected {expected} observables, "
            f"found {len(observables)}"
        )

    rows = []
    directions = {}
    for group_index, group_name in enumerate(m.REFINER_GROUP_LABELS):
        activity, weighted, taper_frames = _group_activity(
            batch, group_index, cfg
        )
        for objective_name in ("endpoint", "temporal"):
            key = f"{group_name}.observable_{objective_name}_0p03"
            gradient = m.torch.autograd.grad(
                observables[key],
                edit,
                retain_graph=True,
            )[0]
            tangent = -gradient * weighted
            action, pre_normalization_rms = _normalize_action(
                _product_action(tangent.detach()), batch["seam"]
            )
            source = f"{group_name}.{objective_name}"
            directions[source] = action
            outside = ~activity.expand_as(tangent)
            outside_max = (
                float(tangent[outside].abs().max().detach())
                if bool(outside.any()) else 0.0
            )
            rows.append({
                "direction_source": source,
                "group": group_name,
                "objective": objective_name,
                "product_action_shape": list(action.shape),
                "product_action_channels": 79,
                "contact_action_channels": 4,
                "manifold_tangent_channels": 75,
                "contact_action_abs_max": float(
                    action[..., :CONTACT_CHANNELS].abs().max().detach()
                ),
                "outside_group_or_ownership_abs_max": outside_max,
                "pre_normalization_active_tangent_rms": (
                    pre_normalization_rms
                ),
                "normalized_active_tangent_rms": _active_tangent_rms(
                    _tangent_from_action(action), batch["seam"]
                ),
                "c2_taper_frames": taper_frames,
            })
    return directions, rows


def _group_local_constraint_corrections(
    model,
    batch,
    cfg,
    baseline,
    identity,
    active_rows_by_group,
):
    """Build a structured output basis used to realize the FD nullspace."""
    edit = m.torch.zeros(
        baseline.shape[:-1] + (75,),
        dtype=baseline.dtype,
        device=baseline.device,
        requires_grad=True,
    )
    candidate = product_exp_torch(baseline, edit)
    guard_tensors = projected_probe._guard_values_for_prediction(
        model, batch, cfg, candidate, identity
    )
    directions = {}
    rows = []
    for group_index, group_name in enumerate(m.REFINER_GROUP_LABELS):
        _, weighted, taper_frames = _group_activity(batch, group_index, cfg)
        for active in active_rows_by_group[group_name]:
            key = active["constraint"]
            value = guard_tensors[key]
            if value.requires_grad:
                gradient = m.torch.autograd.grad(
                    value,
                    edit,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            else:
                gradient = None
            if gradient is None:
                tangent = m.torch.zeros_like(edit)
            else:
                tangent = -gradient * weighted
            action, pre_normalization_rms = _normalize_action(
                _product_action(tangent.detach()), batch["seam"]
            )
            source = f"{group_name}.hard_correction:{key}"
            available = bool(pre_normalization_rms > 1.0e-12)
            if available:
                directions[source] = action
            rows.append({
                "direction_source": source,
                "group": group_name,
                "constraint": key,
                "available": available,
                "pre_normalization_active_tangent_rms": (
                    pre_normalization_rms
                ),
                "c2_taper_frames": taper_frames,
            })
    return directions, rows


def _one_sided_guard_fd(
    model,
    batch,
    cfg,
    baseline,
    identity,
    baseline_guard,
    directions,
    epsilon,
):
    rows = []
    derivatives = {}
    for source, action in directions.items():
        candidate, action_scale, achieved = _candidate_at_action_rms(
            baseline, action, batch["seam"], epsilon
        )
        values = projected_probe._float_guard(
            projected_probe._guard_values_for_prediction(
                model, batch, cfg, candidate, identity
            )
        )
        delta = {
            key: values[key] - baseline_guard[key]
            for key in baseline_guard
        }
        derivative = {
            key: value / float(epsilon)
            for key, value in delta.items()
        }
        derivatives[source] = derivative
        rows.append({
            "direction_source": source,
            "epsilon_output_tangent_rms": float(epsilon),
            "diagnostic_only": True,
            "eligible_as_learning_step": False,
            "action_scale": action_scale,
            "achieved_output_tangent_rms": achieved,
            "guard_delta": delta,
            "one_sided_exact_guard_directional_derivative": derivative,
        })
    return rows, derivatives


def _is_group_hard_constraint(key, group_name):
    return key.startswith(f"{group_name}.") and key.endswith(HARD_SUFFIXES)


def _active_constraint_rows(
    group_name,
    source_names,
    derivatives,
    baseline_details,
):
    rows = []
    for key, detail in baseline_details.items():
        if not _is_group_hard_constraint(key, group_name):
            continue
        values = np.asarray(
            [derivatives[source][key] for source in source_names],
            dtype=np.float64,
        )
        allowance = max(
            0.0,
            float(detail["absolute_limit"]) - float(detail["fixed_anchor"]),
        )
        numeric = float(detail["numeric_tolerance"])
        active_band = max(10.0 * numeric, 0.25 * allowance)
        near_limit = float(detail["remaining_margin"]) <= active_band
        derivative_scale = max(1.0, float(np.max(np.abs(values))))
        threatening = bool(np.any(values > 1.0e-8 * derivative_scale))
        if near_limit or threatening:
            rows.append({
                "constraint": key,
                "one_sided_fd_jacobian_row": values.tolist(),
                "fixed_anchor": float(detail["fixed_anchor"]),
                "current_value": float(detail["candidate"]),
                "absolute_limit": float(detail["absolute_limit"]),
                "remaining_margin": float(detail["remaining_margin"]),
                "numeric_tolerance": numeric,
                "active_band": active_band,
                "near_limit": near_limit,
                "directionally_threatening": threatening,
            })
    return rows


def _nullspace_project_group(
    group_name,
    directions,
    derivatives,
    active_rows,
    seam,
):
    scientific_sources = [
        f"{group_name}.endpoint",
        f"{group_name}.temporal",
    ]
    correction_sources = [
        f"{group_name}.hard_correction:{row['constraint']}"
        for row in active_rows
        if f"{group_name}.hard_correction:{row['constraint']}" in directions
    ]
    sources = scientific_sources + correction_sources
    if active_rows:
        jacobian = m.torch.as_tensor(
            [
                [derivatives[source][row["constraint"]] for source in sources]
                for row in active_rows
            ],
            dtype=m.torch.float64,
            device=seam.device,
        )
        row_norm = m.torch.linalg.vector_norm(jacobian, dim=1)
        nonzero = row_norm > 1.0e-12
        normalized = jacobian[nonzero] / row_norm[nonzero, None]
    else:
        jacobian = m.torch.zeros(
            (0, len(sources)), dtype=m.torch.float64, device=seam.device
        )
        normalized = jacobian

    if normalized.numel():
        _, singular_values, vh = m.torch.linalg.svd(
            normalized, full_matrices=True
        )
        threshold = max(normalized.shape) * 1.0e-10 * max(
            1.0, float(singular_values.max().detach())
        )
        rank = int((singular_values > threshold).sum().detach())
        row_basis = vh[:rank]
        projector = (
            m.torch.eye(
                len(sources), dtype=m.torch.float64, device=seam.device
            )
            - row_basis.transpose(0, 1) @ row_basis
        )
    else:
        singular_values = m.torch.zeros(
            (0,), dtype=m.torch.float64, device=seam.device
        )
        rank = 0
        projector = m.torch.eye(
            len(sources), dtype=m.torch.float64, device=seam.device
        )

    projected = {}
    direction_rows = []
    base_actions = [directions[source] for source in sources]
    for source_index, source in enumerate(scientific_sources):
        coefficients = projector[:, source_index]
        action = sum(
            float(coefficient.detach()) * base_action
            for coefficient, base_action in zip(coefficients, base_actions)
        )
        action, before_rms = _normalize_action(action, seam)
        available = bool(before_rms > 1.0e-12)
        projected[source] = action
        predicted_hard = {
            row["constraint"]: float(
                np.dot(
                    np.asarray([
                        derivatives[basis_source][row["constraint"]]
                        for basis_source in sources
                    ]),
                    coefficients.detach().cpu().numpy(),
                )
            )
            for row in active_rows
        }
        direction_rows.append({
            "direction_source": source,
            "projection_coefficients_in_endpoint_temporal_span": [
                float(value.detach()) for value in coefficients
            ],
            "pre_normalization_projected_rms": before_rms,
            "available": available,
            "predicted_active_hard_directional_derivatives": predicted_hard,
        })
    return projected, {
        "group": group_name,
        "scientific_source_directions": scientific_sources,
        "nullspace_basis_directions": sources,
        "finite_difference_jacobian_domain": (
            "same_fixed_bank_exact_guard_metrics"
        ),
        "active_constraint_rows": [
            {
                **row,
                "nullspace_fd_jacobian_row": [
                    derivatives[source][row["constraint"]]
                    for source in sources
                ],
            }
            for row in active_rows
        ],
        "active_constraint_count": len(active_rows),
        "nullspace_basis_dimension": len(sources),
        "jacobian_rank": rank,
        "nullspace_dimension": len(sources) - rank,
        "singular_values": [
            float(value.detach()) for value in singular_values
        ],
        "nullspace_projector": projector.detach().cpu().tolist(),
        "projected_directions": direction_rows,
    }


def _group_combination_grid(group_name, projected_directions):
    endpoint = projected_directions[f"{group_name}.endpoint"]
    temporal = projected_directions[f"{group_name}.temporal"]
    yield [f"{group_name}.endpoint"], [1.0], endpoint
    yield [f"{group_name}.temporal"], [1.0], temporal
    for endpoint_weight in range(1, PAIR_GRID_DENOMINATOR):
        left = endpoint_weight / PAIR_GRID_DENOMINATOR
        right = 1.0 - left
        yield (
            [f"{group_name}.endpoint", f"{group_name}.temporal"],
            [left, right],
            left * endpoint + right * temporal,
        )


def _action_scope_report(action, batch, group_index):
    tangent = _tangent_from_action(action)
    selected = (batch["group"] == int(group_index))[:, None, None]
    owned = batch["seam"] >= 0.5
    permitted = selected & owned
    outside = ~permitted.expand_as(tangent)
    outside_max = (
        float(tangent[outside].abs().max().detach())
        if bool(outside.any()) else 0.0
    )
    return {
        "outside_group_or_ownership_abs_max": outside_max,
        "scope_safe": bool(outside_max == 0.0),
        "contact_action_abs_max": float(
            action[..., :CONTACT_CHANNELS].abs().max().detach()
        ),
    }


def run(args):
    started = time.perf_counter()
    source = Path(args.source_diagnostic_dir)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = source / "diagnostic_report.json"
    state_path = source / "diagnostic_state.pt"
    fit_path = source / "fit_bank.pt"
    for path in (report_path, state_path, fit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if source_report.get("published") or source_report.get("diagnostic_ready"):
        raise RuntimeError("V15.14e requires an unpublished failed diagnostic")
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.14e exact-closure cone probe requires CUDA")
    state = m.torch.load(state_path, map_location="cpu", weights_only=False)
    artifact = m.torch.load(fit_path, map_location="cpu", weights_only=False)
    if state.get("formal_checkpoint") or artifact.get("formal_checkpoint"):
        raise RuntimeError("formal checkpoint input is forbidden")

    anchor_batch, _, schedule = projected_probe._materialize_first_transaction(
        artifact, device
    )
    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.train()
    baseline, identity = projected_probe._model_prediction(
        model, anchor_batch, cfg
    )
    baseline_guard = projected_probe._float_guard(
        projected_probe._guard_values_for_prediction(
            model, anchor_batch, cfg, baseline, identity
        )
    )
    contract = source_report["group_guard_contract"]
    guard_anchor = contract["initial_anchor"]
    guard_relative = contract["relative_tolerance"]
    guard_absolute = contract["absolute_tolerance"]
    if set(guard_anchor) != set(baseline_guard):
        raise RuntimeError("source fixed Guard and reconstructed bank differ")
    baseline_passed, baseline_blockers, baseline_details = (
        projected_probe._audit_guard_candidate(
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
    )

    base_directions, base_direction_rows = (
        _group_local_scientific_directions(
            model, anchor_batch, cfg, baseline, identity
        )
    )
    fd_rows, fd_derivatives = _one_sided_guard_fd(
        model,
        anchor_batch,
        cfg,
        baseline,
        identity,
        baseline_guard,
        base_directions,
        float(args.fd_epsilon),
    )
    active_rows_by_group = {}
    for group_name in m.REFINER_GROUP_LABELS:
        scientific_sources = [
            f"{group_name}.endpoint",
            f"{group_name}.temporal",
        ]
        active_rows_by_group[group_name] = _active_constraint_rows(
            group_name,
            scientific_sources,
            fd_derivatives,
            baseline_details,
        )
    correction_directions, correction_direction_rows = (
        _group_local_constraint_corrections(
            model,
            anchor_batch,
            cfg,
            baseline,
            identity,
            active_rows_by_group,
        )
    )
    correction_fd_rows, correction_fd_derivatives = _one_sided_guard_fd(
        model,
        anchor_batch,
        cfg,
        baseline,
        identity,
        baseline_guard,
        correction_directions,
        float(args.fd_epsilon),
    )
    nullspace_basis_directions = {
        **base_directions,
        **correction_directions,
    }
    nullspace_fd_derivatives = {
        **fd_derivatives,
        **correction_fd_derivatives,
    }
    projected_directions = {}
    nullspace_reports = []
    for group_name in m.REFINER_GROUP_LABELS:
        group_directions, group_report = _nullspace_project_group(
            group_name,
            nullspace_basis_directions,
            nullspace_fd_derivatives,
            active_rows_by_group[group_name],
            anchor_batch["seam"],
        )
        projected_directions.update(group_directions)
        nullspace_reports.append(group_report)

    projected_fd_rows, projected_fd_derivatives = _one_sided_guard_fd(
        model,
        anchor_batch,
        cfg,
        baseline,
        identity,
        baseline_guard,
        projected_directions,
        float(args.fd_epsilon),
    )

    target = float(args.target_rms)
    if target < DEFAULT_TARGET_RMS:
        raise ValueError(
            f"target_rms must be at least {DEFAULT_TARGET_RMS:g}"
        )
    candidates = []
    raw_blocker_counts = Counter()
    projected_blocker_counts = Counter()
    raw_pass_count = 0
    projected_pass_count = 0
    selected_index = None
    selected_score = math.inf
    selected_motion = None

    for group_index, group_name in enumerate(m.REFINER_GROUP_LABELS):
        for sources, coefficients, unnormalized_action in (
            _group_combination_grid(group_name, projected_directions)
        ):
            candidate_started = time.perf_counter()
            action, combination_rms = _normalize_action(
                unnormalized_action, anchor_batch["seam"]
            )
            scope = _action_scope_report(action, anchor_batch, group_index)
            available = bool(combination_rms > 1.0e-12 and scope["scope_safe"])
            if available:
                raw, action_scale, achieved = _candidate_at_action_rms(
                    baseline,
                    action,
                    anchor_batch["seam"],
                    target,
                )
                (
                    raw_values,
                    raw_guard_passed,
                    raw_blockers,
                    raw_details,
                    raw_delta,
                    raw_scientific,
                ) = _audit_prediction(
                    model,
                    anchor_batch,
                    cfg,
                    raw,
                    identity,
                    baseline_guard,
                    guard_anchor,
                    guard_relative,
                    guard_absolute,
                )
            else:
                raw = baseline
                action_scale = 0.0
                achieved = 0.0
                raw_values = baseline_guard
                raw_guard_passed = False
                raw_blockers = ["empty_boundary_nullspace_direction"]
                raw_details = baseline_details
                raw_delta = {
                    key: 0.0
                    for key in closure_probe._observable_values(baseline_guard)
                }
                raw_scientific = closure_probe._scientific_delta_status(
                    raw_delta, baseline_details
                )
            raw_blocker_counts.update(raw_blockers)
            resolved_radius = bool(achieved >= DEFAULT_TARGET_RMS)
            raw_valid = bool(
                available
                and resolved_radius
                and raw_guard_passed
                and raw_scientific["raw_exact_common_descent"]
            )
            raw_pass_count += int(raw_valid)

            projection_trials = []
            projector_report = None
            accepted_factor = None
            accepted_prediction = None
            accepted_delta = None
            accepted_scientific = None
            if raw_valid:
                full_projected, projector_report = (
                    projected_probe.weighted_dls_contact_project_torch(
                        anchor_batch["bad"],
                        baseline,
                        raw,
                        anchor_batch["seam"],
                        cfg,
                        iterations=args.ik_iterations,
                        damping=args.damping,
                        jacobian_epsilon=args.jacobian_epsilon,
                        stiffness_ceiling=args.stiffness_ceiling,
                        acceleration_regularization=(
                            args.acceleration_regularization
                        ),
                        jerk_regularization=args.jerk_regularization,
                        scientific_context={
                            "model": model,
                            "batch": anchor_batch,
                            "identity": identity,
                        },
                    )
                )
                for factor in closure_probe.PROJECTION_FACTORS:
                    projected = closure_probe._interpolate_projector_correction(
                        raw, full_projected, factor
                    )
                    (
                        values,
                        passed,
                        blockers,
                        details,
                        delta,
                        scientific,
                    ) = _audit_prediction(
                        model,
                        anchor_batch,
                        cfg,
                        projected,
                        identity,
                        baseline_guard,
                        guard_anchor,
                        guard_relative,
                        guard_absolute,
                    )
                    projected_blocker_counts.update(blockers)
                    accepted = bool(
                        passed
                        and scientific["raw_exact_common_descent"]
                        and projector_report["scope_safe"]
                    )
                    projection_trials.append({
                        "factor": factor,
                        "fixed_exact_guard_passed": passed,
                        "guard_blockers": blockers,
                        "observable_residual_delta": delta,
                        "projected_scientific_nonregression": scientific[
                            "all_nonpositive_within_numeric_tolerance"
                        ],
                        "at_least_one_resolved_strict_descent": scientific[
                            "at_least_one_resolved_strict_descent"
                        ],
                        "accepted": accepted,
                    })
                    if accepted:
                        accepted_factor = factor
                        accepted_prediction = projected
                        accepted_delta = delta
                        accepted_scientific = scientific
                        break
            effective = accepted_prediction is not None
            projected_pass_count += int(effective)
            row = {
                "group": group_name,
                "direction_sources": sources,
                "nonnegative_coefficients": coefficients,
                "coefficient_sum": float(sum(coefficients)),
                "combination_pre_normalization_rms": combination_rms,
                "direction_available": available,
                "action_scope": scope,
                "target_output_tangent_rms": target,
                "achieved_output_tangent_rms": achieved,
                "resolved_required_radius": resolved_radius,
                "action_scale": action_scale,
                "raw_guard_values": raw_values,
                "raw_fixed_exact_guard_passed": raw_guard_passed,
                "raw_guard_blockers": raw_blockers,
                "raw_guard_metrics": raw_details,
                "raw_observable_residual_delta": raw_delta,
                "raw_exact_common_descent": raw_scientific[
                    "raw_exact_common_descent"
                ],
                "raw_candidate_admitted_to_projector": raw_valid,
                "resolution_limited_under_exact_closure": raw_scientific[
                    "resolution_limited_under_exact_closure"
                ],
                "source_one_sided_fd_derivatives": {
                    source_name: projected_fd_derivatives[source_name]
                    for source_name in sources
                },
                "projector": projector_report,
                "projection_trials": projection_trials,
                "projection_backtracking_factor": accepted_factor,
                "projected_scientific_nonregression": bool(
                    accepted_scientific
                    and accepted_scientific[
                        "all_nonpositive_within_numeric_tolerance"
                    ]
                ),
                "projected_observable_residual_delta": accepted_delta,
                "effective_projected_candidate": effective,
                "elapsed_seconds": time.perf_counter() - candidate_started,
            }
            candidates.append(row)
            if effective:
                score = float(sum(accepted_delta.values()))
                if score < selected_score:
                    selected_score = score
                    selected_index = len(candidates) - 1
                    selected_motion = accepted_prediction.detach().cpu().numpy()

    if selected_motion is not None:
        np.save(destination / "selected_projected_candidate.npy", selected_motion)
    if raw_pass_count == 0:
        feasibility_status = (
            "no_group_local_boundary_nullspace_descent_at_required_radius"
        )
    elif projected_pass_count == 0:
        feasibility_status = "group_local_raw_descent_projector_failed"
    else:
        feasibility_status = "group_local_projected_descent_exists"

    expected_per_group = 2 + (PAIR_GRID_DENOMINATOR - 1)
    expected_count = len(m.REFINER_GROUP_LABELS) * expected_per_group
    result = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "development_only": True,
        "training_started": False,
        "formal_checkpoint": False,
        "publish_allowed": False,
        "source_diagnostic": str(source.resolve()),
        "source_schema": source_report.get("schema"),
        "source_completed_steps": source_report.get("completed_steps"),
        "transaction_context_indices": list(schedule),
        "baseline_fixed_exact_guard_passed": baseline_passed,
        "baseline_guard_blockers": baseline_blockers,
        "baseline_guard_metrics": baseline_details,
        "baseline_guard_values": baseline_guard,
        "local_action_shape": list(baseline.shape[:-1]) + [79],
        "local_action_product_channels": 79,
        "local_action_contact_channels_frozen": 4,
        "local_action_manifold_tangent_channels": 75,
        "group_local_base_directions": base_direction_rows,
        "base_direction_one_sided_exact_guard_fd": fd_rows,
        "group_local_hard_constraint_correction_directions": (
            correction_direction_rows
        ),
        "hard_constraint_correction_one_sided_exact_guard_fd": (
            correction_fd_rows
        ),
        "hard_constraint_fd_jacobian": nullspace_reports,
        "projected_direction_one_sided_exact_guard_fd": projected_fd_rows,
        "diagnostic_fd_epsilon_output_tangent_rms": float(args.fd_epsilon),
        "diagnostic_fd_scales_eligible_as_learning_steps": False,
        "combination_scope": "within_group_endpoint_temporal_only",
        "coefficient_constraint": "nonnegative_sum_equal_1",
        "pair_grid_denominator": PAIR_GRID_DENOMINATOR,
        "minimum_required_output_tangent_rms": DEFAULT_TARGET_RMS,
        "target_output_tangent_rms": target,
        "expected_candidate_count": expected_count,
        "candidate_count": len(candidates),
        "raw_exact_common_descent_count": raw_pass_count,
        "effective_projected_candidate_count": projected_pass_count,
        "selected_candidate_index": selected_index,
        "projected_direction_exists": selected_index is not None,
        "feasibility_status": feasibility_status,
        "raw_guard_blocker_counts": dict(raw_blocker_counts),
        "projected_guard_blocker_counts": dict(projected_blocker_counts),
        "projection_backtracking_factors": list(
            closure_probe.PROJECTION_FACTORS
        ),
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "candidates": candidates,
        "scope_safe": all(
            row["action_scope"]["scope_safe"]
            and (
                row["projector"] is None
                or row["projector"]["scope_safe"]
            )
            for row in candidates
        ),
        "numeric_audit_complete": bool(len(candidates) == expected_count),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = destination / "group_local_nullspace_cone.report.json"
    m.save_json(result, report)
    print(json.dumps({
        "stage": "refiner_v15_14e_group_local_nullspace_cone_complete",
        "report": str(report.resolve()),
        "candidates": len(candidates),
        "raw_exact_common_descent": raw_pass_count,
        "effective_projected_candidates": projected_pass_count,
        "feasibility_status": feasibility_status,
        "scope_safe": result["scope_safe"],
        "numeric_audit_complete": result["numeric_audit_complete"],
    }), flush=True)
    return 0 if selected_index is not None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-diagnostic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--fd-epsilon", type=float, default=DEFAULT_FD_EPSILON)
    parser.add_argument("--target-rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument("--ik-iterations", type=int, default=6)
    parser.add_argument("--damping", type=float, default=1.0e-4)
    parser.add_argument("--jacobian-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--stiffness-ceiling", type=float, default=1.0e4)
    parser.add_argument(
        "--acceleration-regularization", type=float, default=1.0e-2
    )
    parser.add_argument("--jerk-regularization", type=float, default=1.0e-3)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
