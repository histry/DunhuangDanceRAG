"""V15.15g1f3 second-order models on the owned exact-radius sphere.

This module contains only the numerical kernel shared by train calibration and
formal inference.  It never reads split, role, ``single``/``cross`` or teacher
labels.  The caller supplies three scalar metric functions evaluated through
the real geodesic -> product retraction -> FK -> metric graph.

The Hessian is never materialised in the ambient motion coordinates.  A
deterministic basis is formed from the three projected first derivatives and
directional Hessian-vector products are polarised into a matrix of dimension at
most three.  Because every directional derivative is taken with respect to the
geodesic angle itself, the second derivative includes the sphere-geodesic
acceleration term; it is not merely ``q.T @ H @ q`` in a flat coordinate.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from training import motion_models as m


SECOND_ORDER_STATES = (
    "second_order_closure_succeeded",
    "insufficient_second_order_predicted_progress",
    "second_order_finite_radius_model_mismatch",
    "active_set_transition_model_mismatch",
    "nonfinite_or_unverified_curvature",
    "second_order_solver_failure",
)


_UNIT_GRID_CACHE = {}
HVP_RECOVERY_EPSILON_RADIANS = (1.0e-4, 3.0e-4)
HVP_RECOVERY_RELATIVE_TOLERANCE = 0.25
HVP_RECOVERY_ABSOLUTE_TOLERANCE = 1.0e-6
SECOND_ORDER_SQP_REFINEMENT_STARTS = 768
SECOND_ORDER_SQP_REFINEMENT_ITERATIONS = 64
SECOND_ORDER_SQP_SMOOTHING = (1.0, 8.0, 64.0, 512.0)
SECOND_ORDER_SQP_LINE_SEARCH_RADIANS = (
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
    0.00048828125,
    0.000244140625,
    0.0001220703125,
    0.00006103515625,
    0.000030517578125,
)


def _inner(left, right, mask):
    return (left[mask] * right[mask]).sum()


def _unit_owned_sphere_tangent(vector, current, mask, *, floor):
    scoped = vector.masked_fill(~mask, 0.0)
    radial = current.masked_fill(~mask, 0.0)
    radial_norm_sq = _inner(radial, radial, mask)
    if not bool(m.torch.isfinite(radial_norm_sq)) or float(
        radial_norm_sq.detach()
    ) <= float(floor) ** 2:
        return None
    scoped = scoped - (_inner(scoped, radial, mask) / radial_norm_sq) * radial
    scoped = scoped.masked_fill(~mask, 0.0)
    norm = m.torch.linalg.vector_norm(scoped[mask])
    if not bool(m.torch.isfinite(norm)) or float(norm.detach()) <= float(floor):
        return None
    return scoped / norm


def build_owned_tangent_basis(
    *,
    current,
    mask,
    taper,
    gradients: Mapping[str, object],
    dimension: int,
    floor: float,
):
    """Build a deterministic physical basis in the ownership-sphere tangent.

    Gradients arrive in the free ``z`` coordinate.  The physical path is
    ``p = taper * N_scope(z)``; consequently the physical covector is obtained
    with the inverse taper on editable coordinates before the sphere projector.
    """
    dtype = m.torch.float64
    current64 = current.detach().to(dtype)
    taper64 = taper.detach().to(dtype)
    mask = mask.to(m.torch.bool)
    active = mask & (taper64.abs() > float(floor))
    rows = []
    source_names = []
    for name in ("shadow", "endpoint", "temporal"):
        gradient = gradients.get(name)
        if gradient is None:
            continue
        gradient64 = gradient.detach().to(dtype)
        if not bool(m.torch.isfinite(gradient64).all()):
            return None, {
                "status": "nonfinite_or_unverified_curvature",
                "reason": f"nonfinite_{name}_gradient",
            }
        physical = m.torch.zeros_like(gradient64)
        physical[active] = gradient64[active] / taper64[active]
        unit = _unit_owned_sphere_tangent(
            physical, current64, active, floor=float(floor)
        )
        if unit is None:
            continue
        for existing in rows:
            unit = unit - _inner(unit, existing, active) * existing
        unit = _unit_owned_sphere_tangent(
            unit, current64, active, floor=float(floor)
        )
        if unit is None:
            continue
        rows.append(unit)
        source_names.append(name)
        if len(rows) >= int(dimension):
            break
    if not rows:
        return None, {
            "status": "insufficient_second_order_predicted_progress",
            "reason": "zero_science_gradient",
            "basis_dimension": 0,
        }
    return m.torch.stack(rows), {
        "status": "second_order_basis_ready",
        "basis_dimension": len(rows),
        "basis_sources": source_names,
        "ambient_hessian_materialized": False,
        "scope_null_space_projection": "exact_boolean_ownership_mask",
        "sphere_tangent_projection": True,
    }


def differentiable_geodesic_trial(current, unit_direction, mask, theta):
    """Exact sphere geodesic without detach, used by the curvature graph."""
    radial = current.masked_fill(~mask, 0.0)
    direction = unit_direction.masked_fill(~mask, 0.0)
    radius = m.torch.linalg.vector_norm(radial[mask])
    return (
        m.torch.cos(theta) * radial
        + m.torch.sin(theta) * radius * direction
    ).masked_fill(~mask, 0.0)


def _directional_first_derivative(
    *,
    current,
    mask,
    unit_direction,
    metric_builder,
    name,
    theta_radians,
):
    theta = current.new_tensor(
        float(theta_radians), dtype=m.torch.float64, requires_grad=True
    )
    trial = differentiable_geodesic_trial(
        current, unit_direction, mask, theta
    )
    value = metric_builder(trial)[name]
    if value.numel() != 1 or not bool(m.torch.isfinite(value).all()):
        return None
    first = m.torch.autograd.grad(
        value,
        theta,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )[0]
    if first is None:
        first = m.torch.zeros_like(theta)
    if not bool(m.torch.isfinite(first).all()):
        return None
    return value.detach(), first.detach()


def _recover_directional_hvp(
    *,
    current,
    mask,
    unit_direction,
    metric_builder,
    name,
):
    """Recover a nonfinite autograd HvP from verified float64 first derivatives.

    This is used only when the scalar value and first derivative at the
    expansion point are finite but PyTorch's second backward encounters a
    dormant zero-norm branch. Two symmetric radii must agree before the
    direction is admitted to the curvature model.
    """
    estimates = []
    rows = []
    for epsilon in HVP_RECOVERY_EPSILON_RADIANS:
        positive = _directional_first_derivative(
            current=current,
            mask=mask,
            unit_direction=unit_direction,
            metric_builder=metric_builder,
            name=name,
            theta_radians=epsilon,
        )
        negative = _directional_first_derivative(
            current=current,
            mask=mask,
            unit_direction=unit_direction,
            metric_builder=metric_builder,
            name=name,
            theta_radians=-epsilon,
        )
        if positive is None or negative is None:
            return None, {
                "status": "nonfinite_or_unverified_curvature",
                "second_order_state": "nonfinite_or_unverified_curvature",
                "metric": name,
                "reason": "nonfinite_hvp_recovery_first_derivative",
                "epsilon_radians": float(epsilon),
            }
        estimate = (positive[1] - negative[1]) / (2.0 * float(epsilon))
        if not bool(m.torch.isfinite(estimate).all()):
            return None, {
                "status": "nonfinite_or_unverified_curvature",
                "second_order_state": "nonfinite_or_unverified_curvature",
                "metric": name,
                "reason": "nonfinite_hvp_recovery_estimate",
                "epsilon_radians": float(epsilon),
            }
        estimates.append(estimate)
        rows.append({
            "epsilon_radians": float(epsilon),
            "positive_first_derivative": float(positive[1]),
            "negative_first_derivative": float(negative[1]),
            "second_derivative_estimate": float(estimate),
        })

    difference = (estimates[0] - estimates[1]).abs()
    scale = m.torch.stack([
        estimates[0].abs(),
        estimates[1].abs(),
        estimates[0].new_tensor(1.0),
    ]).amax()
    tolerance = (
        float(HVP_RECOVERY_ABSOLUTE_TOLERANCE)
        + float(HVP_RECOVERY_RELATIVE_TOLERANCE) * scale
    )
    if not bool(difference <= tolerance):
        return None, {
            "status": "nonfinite_or_unverified_curvature",
            "second_order_state": "nonfinite_or_unverified_curvature",
            "metric": name,
            "reason": "inconsistent_hvp_recovery_epsilon_ladder",
            "epsilon_ladder": rows,
            "estimate_difference": float(difference),
            "allowed_difference": float(tolerance),
        }

    small = float(HVP_RECOVERY_EPSILON_RADIANS[0])
    large = float(HVP_RECOVERY_EPSILON_RADIANS[1])
    recovered = (
        large * large * estimates[0] - small * small * estimates[1]
    ) / (large * large - small * small)
    if not bool(m.torch.isfinite(recovered).all()):
        return None, {
            "status": "nonfinite_or_unverified_curvature",
            "second_order_state": "nonfinite_or_unverified_curvature",
            "metric": name,
            "reason": "nonfinite_richardson_hvp_recovery",
            "epsilon_ladder": rows,
        }
    return recovered.detach(), {
        "status": "directional_curvature_verified",
        "metric": name,
        "recovery": "symmetric_first_derivative_hvp_epsilon_ladder",
        "epsilon_ladder": rows,
        "relative_tolerance": float(HVP_RECOVERY_RELATIVE_TOLERANCE),
        "absolute_tolerance": float(HVP_RECOVERY_ABSOLUTE_TOLERANCE),
        "active_branch_stability_verified": True,
        "richardson_second_derivative": float(recovered),
    }


def _directional_jet(
    *,
    current,
    mask,
    unit_direction,
    metric_builder: Callable[[object], Mapping[str, object]],
    names: Sequence[str],
):
    """Return F(0), dF/dtheta and d2F/dtheta2 for one real path.

    The audit identifies the exact metric and derivative order that failed.
    Callers may then remove that direction from the verified low-dimensional
    curvature subspace; a nonfinite derivative is never replaced by zero.
    """
    theta = current.new_zeros((), dtype=m.torch.float64, requires_grad=True)
    trial = differentiable_geodesic_trial(current, unit_direction, mask, theta)
    metrics = metric_builder(trial)
    result = {}
    recovered_hvps = []
    for index, name in enumerate(names):
        value = metrics[name]
        if value.numel() != 1 or not bool(m.torch.isfinite(value).all()):
            return None, {
                "status": "nonfinite_or_unverified_curvature",
                "second_order_state": "nonfinite_or_unverified_curvature",
                "metric": name,
                "derivative_order": 0,
                "reason": f"nonfinite_{name}_value",
            }
        if value.requires_grad:
            first = m.torch.autograd.grad(
                value,
                theta,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            first = None
        if first is None:
            first = m.torch.zeros_like(theta)
        if not bool(m.torch.isfinite(first).all()):
            return None, {
                "status": "nonfinite_or_unverified_curvature",
                "second_order_state": "nonfinite_or_unverified_curvature",
                "metric": name,
                "derivative_order": 1,
                "reason": f"nonfinite_{name}_first_derivative",
            }
        if first.requires_grad:
            second = m.torch.autograd.grad(
                first,
                theta,
                retain_graph=index < len(names) - 1,
                allow_unused=True,
            )[0]
        else:
            second = None
        if second is None:
            second = m.torch.zeros_like(theta)
        if not bool(m.torch.isfinite(second).all()):
            second, recovery_audit = _recover_directional_hvp(
                current=current,
                mask=mask,
                unit_direction=unit_direction,
                metric_builder=metric_builder,
                name=name,
            )
            if second is None:
                return None, {
                    **recovery_audit,
                    "derivative_order": 2,
                    "autograd_reason":
                        f"nonfinite_{name}_second_derivative",
                }
            recovered_hvps.append(recovery_audit)
        result[name] = (
            value.detach(),
            first.detach(),
            second.detach(),
        )
    return result, {
        "status": "directional_curvature_verified",
        "derivative_order": 2,
        "autograd_hvp_recovery": recovered_hvps,
    }


def build_second_order_models(
    *,
    current,
    mask,
    basis,
    metric_builder: Callable[[object], Mapping[str, object]],
    names: Sequence[str] = ("shadow", "endpoint", "temporal"),
):
    """Build a Hessian in the largest deterministic verified subspace.

    Norm-based physical metrics can have undefined second derivatives on an
    exactly zero vector. Such a direction is excluded before the model is
    built. No NaN is consumed or replaced, and failure is returned if no
    verified direction remains. Pair failures deterministically remove the
    later basis vector and rebuild the model.
    """
    requested_count = int(basis.shape[0])
    verified_indices = []
    jets_by_index = {}
    jet_audits_by_index = {}
    dropped = []
    for index in range(requested_count):
        jet, jet_audit = _directional_jet(
            current=current,
            mask=mask,
            unit_direction=basis[index],
            metric_builder=metric_builder,
            names=names,
        )
        if jet is None:
            dropped.append({
                "basis_index": int(index),
                "kind": "basis_direction",
                **jet_audit,
            })
            continue
        verified_indices.append(index)
        jets_by_index[index] = jet
        jet_audits_by_index[index] = jet_audit

    if not verified_indices:
        return None, {
            "status": "nonfinite_or_unverified_curvature",
            "second_order_state": "nonfinite_or_unverified_curvature",
            "reason": "no_verified_basis_direction",
            "requested_basis_dimension": requested_count,
            "verified_basis_dimension": 0,
            "dropped_directions": dropped,
        }

    pair_jets = {}
    pair_audits = {}
    while len(verified_indices) > 1:
        failed_pair = None
        pair_jets = {}
        pair_audits = {}
        for left_position, left_index in enumerate(verified_indices):
            for right_index in verified_indices[left_position + 1:]:
                direction = (
                    basis[left_index] + basis[right_index]
                ) / math.sqrt(2.0)
                jet, jet_audit = _directional_jet(
                    current=current,
                    mask=mask,
                    unit_direction=direction,
                    metric_builder=metric_builder,
                    names=names,
                )
                if jet is None:
                    failed_pair = (left_index, right_index, jet_audit)
                    break
                pair_jets[(left_index, right_index)] = jet
                pair_audits[(left_index, right_index)] = jet_audit
            if failed_pair is not None:
                break
        if failed_pair is None:
            break
        left_index, right_index, jet_audit = failed_pair
        dropped.append({
            "basis_index": int(right_index),
            "paired_with_basis_index": int(left_index),
            "kind": "pair_direction",
            **jet_audit,
        })
        verified_indices.remove(right_index)

    count = len(verified_indices)
    verified_basis = basis.index_select(
        0,
        m.torch.as_tensor(
            verified_indices, dtype=m.torch.long, device=basis.device
        ),
    )
    diagonal = [
        {name: jets_by_index[index][name][2] for name in names}
        for index in verified_indices
    ]
    first_columns = [
        {name: jets_by_index[index][name][1] for name in names}
        for index in verified_indices
    ]
    base_values: Optional[Dict[str, object]] = {
        name: jets_by_index[verified_indices[0]][name][0]
        for name in names
    }

    hessians = {
        name: m.torch.zeros(
            (count, count), dtype=m.torch.float64, device=current.device
        )
        for name in names
    }
    for index in range(count):
        for name in names:
            hessians[name][index, index] = diagonal[index][name]
    pair_hvp_count = 0
    for left in range(count):
        for right in range(left + 1, count):
            source_pair = (verified_indices[left], verified_indices[right])
            jet = pair_jets[source_pair]
            pair_hvp_count += 1
            for name in names:
                cross = (
                    jet[name][2]
                    - 0.5 * diagonal[left][name]
                    - 0.5 * diagonal[right][name]
                )
                hessians[name][left, right] = cross
                hessians[name][right, left] = cross

    first = {
        name: m.torch.stack([column[name] for column in first_columns])
        for name in names
    }
    finite = all(
        bool(m.torch.isfinite(first[name]).all())
        and bool(m.torch.isfinite(hessians[name]).all())
        for name in names
    )
    if not finite:
        return None, {
            "status": "nonfinite_or_unverified_curvature",
            "second_order_state": "nonfinite_or_unverified_curvature",
            "reason": "nonfinite_low_dimensional_model",
            "requested_basis_dimension": requested_count,
            "verified_basis_dimension": count,
            "dropped_directions": dropped,
        }
    return {
        name: {
            "value": base_values[name],
            "first": first[name],
            "hessian": hessians[name],
        }
        for name in names
    }, {
        "status": "second_order_curvature_verified",
        "dtype": "float64",
        "path": "exact_geodesic_product_retraction_fk_metric",
        "geodesic_acceleration_included": True,
        "ambient_hessian_materialized": False,
        "basis_dimension": count,
        "requested_basis_dimension": requested_count,
        "verified_basis_indices": [int(index) for index in verified_indices],
        "verified_basis": verified_basis,
        "curvature_subspace_reduced": count < requested_count,
        "dropped_directions": dropped,
        "verified_direction_audits": [
            {
                "basis_index": int(index),
                **jet_audits_by_index[index],
            }
            for index in verified_indices
        ],
        "verified_pair_audits": [
            {
                "left_basis_index": int(left),
                "right_basis_index": int(right),
                **pair_audits[(left, right)],
            }
            for left, right in sorted(pair_audits)
        ],
        "unverified_directions_used": False,
        "nonfinite_basis_policy": "deterministic_verified_subspace_reduction",
        "directional_hvp_count": count + pair_hvp_count,
        "polarization_used_for_cross_terms": True,
    }


def _deterministic_unit_grid(dimension: int, levels: int, *, device):
    levels = int(levels)
    if dimension <= 0 or levels < 3 or levels % 2 == 0:
        raise ValueError("second-order grid requires dimension>0 and odd levels>=3")
    key = (str(device), int(dimension), levels)
    cached = _UNIT_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    coordinates = m.torch.linspace(
        -1.0, 1.0, levels, dtype=m.torch.float64, device=device
    )
    rows = m.torch.cartesian_prod(
        *([coordinates] * int(dimension))
    ).reshape(-1, int(dimension))
    norms = m.torch.linalg.vector_norm(rows, dim=1)
    rows = rows[norms > 1.0e-15]
    norms = m.torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    result = rows / norms
    _UNIT_GRID_CACHE[key] = result
    return result


def _quadratic_changes(models, directions, theta):
    """Evaluate every frozen second-order constraint on device."""
    names = ("shadow", "endpoint", "temporal")
    first = m.torch.stack([models[name]["first"] for name in names])
    hessian = m.torch.stack([models[name]["hessian"] for name in names])
    linear = directions @ first.transpose(0, 1)
    quadratic = m.torch.einsum(
        "nd,kde,ne->nk", directions, hessian, directions
    )
    return theta * linear + 0.5 * theta * theta * quadratic


def _constraint_scales(models, theta, required, floor):
    """Normalize by each signed boundary gap, not metric dynamic range.

    Dynamic-range normalization can declare a large positive Guard violation
    numerically small merely because that Guard has high curvature.  Gap
    normalization makes zero the equally authoritative feasibility boundary
    for all three constraints.  ``models`` and ``theta`` remain explicit in
    the signature so the frozen solver interface records its inputs.
    """
    del models, theta
    return required.abs().clamp_min(max(float(floor), 1.0e-12))


def _continuous_joint_sqp_refinement(
    *,
    models,
    theta,
    required,
    scales,
    coarse_directions,
    feasibility_tolerance,
):
    """Refine the best coarse seeds on the coefficient unit sphere.

    The three low-dimensional quadratic constraints are differentiated
    analytically.  A deterministic Riemannian active-constraint iteration and
    fixed angular line search minimize their worst normalized residual.  This
    closes narrow feasible cones that a Cartesian direction grid can miss,
    while all candidate evaluation and selection remain on the motion device.
    """
    coarse_changes = _quadratic_changes(models, coarse_directions, theta)
    coarse_residual = (
        coarse_changes + required.unsqueeze(0) - float(feasibility_tolerance)
    ) / scales.unsqueeze(0)
    coarse_worst = coarse_residual.amax(dim=1)
    start_count = min(
        int(SECOND_ORDER_SQP_REFINEMENT_STARTS),
        int(coarse_directions.shape[0]),
    )
    start_indices = m.torch.argsort(
        coarse_worst, stable=True
    )[:start_count]
    current = coarse_directions.index_select(0, start_indices).clone()
    line_search = current.new_tensor(
        SECOND_ORDER_SQP_LINE_SEARCH_RADIANS
    )
    names = ("shadow", "endpoint", "temporal")
    first = m.torch.stack([models[name]["first"] for name in names])
    hessian = m.torch.stack([models[name]["hessian"] for name in names])

    iteration_count = 0
    for smoothing in SECOND_ORDER_SQP_SMOOTHING:
        stage_iterations = int(SECOND_ORDER_SQP_REFINEMENT_ITERATIONS) // len(
            SECOND_ORDER_SQP_SMOOTHING
        )
        for _ in range(stage_iterations):
            iteration_count += 1
            changes = _quadratic_changes(models, current, theta)
            residual = (
                changes
                + required.unsqueeze(0)
                - float(feasibility_tolerance)
            ) / scales.unsqueeze(0)
            weights = m.torch.softmax(
                float(smoothing) * residual, dim=1
            )
            constraint_gradients = (
                theta * first.unsqueeze(0)
                + theta
                * theta
                * m.torch.einsum("kde,ne->nkd", hessian, current)
            ) / scales.reshape(1, -1, 1)
            gradient = (
                weights.unsqueeze(-1) * constraint_gradients
            ).sum(dim=1)
            gradient = gradient - (
                gradient * current
            ).sum(dim=1, keepdim=True) * current
            gradient_norm = m.torch.linalg.vector_norm(
                gradient, dim=1, keepdim=True
            )
            movable = gradient_norm.squeeze(1) > 1.0e-15
            unit_gradient = gradient / gradient_norm.clamp_min(1.0e-30)

            trials = (
                m.torch.cos(line_search).reshape(1, -1, 1)
                * current.unsqueeze(1)
                - m.torch.sin(line_search).reshape(1, -1, 1)
                * unit_gradient.unsqueeze(1)
            )
            trial_changes = _quadratic_changes(
                models, trials.reshape(-1, current.shape[1]), theta
            ).reshape(current.shape[0], line_search.numel(), -1)
            trial_residual = (
                trial_changes
                + required.reshape(1, 1, -1)
                - float(feasibility_tolerance)
            ) / scales.reshape(1, 1, -1)
            trial_objective = m.torch.logsumexp(
                float(smoothing) * trial_residual, dim=2
            ) / float(smoothing)
            current_objective = m.torch.logsumexp(
                float(smoothing) * residual, dim=1
            ) / float(smoothing)
            combined = m.torch.cat(
                [current_objective.unsqueeze(1), trial_objective], dim=1
            )
            selected = combined.argmin(dim=1)
            selected_trial = (selected - 1).clamp_min(0)
            replacement = trials[
                m.torch.arange(current.shape[0], device=current.device),
                selected_trial,
            ]
            improve = movable & (selected > 0)
            current = m.torch.where(
                improve.unsqueeze(1), replacement, current
            )

    refined_changes = _quadratic_changes(models, current, theta)
    refined_residual = (
        refined_changes + required.unsqueeze(0) - float(feasibility_tolerance)
    ) / scales.unsqueeze(0)
    return current, refined_changes, {
        "continuous_sqp_start_count": start_count,
        "continuous_sqp_iteration_count": iteration_count,
        "continuous_sqp_smoothing": [
            float(value) for value in SECOND_ORDER_SQP_SMOOTHING
        ],
        "continuous_sqp_line_search_radians": [
            float(value) for value in SECOND_ORDER_SQP_LINE_SEARCH_RADIANS
        ],
        "best_continuous_sqp_normalized_residual": float(
            refined_residual.amax(dim=1).amin().detach()
        ),
    }


def solve_angle_subproblem(
    *,
    models,
    theta_radians: float,
    required_reduction: Mapping[str, float],
    grid_levels: int,
    feasibility_tolerance: float,
):
    """Solve the frozen-angle joint quadratic model on the unit sphere."""
    names = ("shadow", "endpoint", "temporal")
    dimension = int(models["shadow"]["first"].numel())
    coarse_directions = _deterministic_unit_grid(
        dimension, int(grid_levels), device=models["shadow"]["first"].device
    )
    theta = float(theta_radians)
    required = coarse_directions.new_tensor([
        float(required_reduction[name]) for name in names
    ])
    scales = _constraint_scales(
        models, theta, required, float(feasibility_tolerance)
    )
    coarse_changes = _quadratic_changes(models, coarse_directions, theta)
    coarse_feasible = (
        coarse_changes
        <= -required.unsqueeze(0) + float(feasibility_tolerance)
    ).all(dim=1)
    if bool(coarse_feasible.any()):
        directions = coarse_directions
        changes = coarse_changes
        refinement_audit = {
            "continuous_sqp_executed": False,
            "continuous_sqp_skip_reason": "coarse_joint_candidate_feasible",
            "continuous_sqp_start_count": 0,
            "continuous_sqp_iteration_count": 0,
        }
    else:
        refined_directions, refined_changes, refinement_audit = (
            _continuous_joint_sqp_refinement(
                models=models,
                theta=theta,
                required=required,
                scales=scales,
                coarse_directions=coarse_directions,
                feasibility_tolerance=float(feasibility_tolerance),
            )
        )
        refinement_audit["continuous_sqp_executed"] = True
        directions = m.torch.cat(
            [coarse_directions, refined_directions], dim=0
        )
        changes = m.torch.cat([coarse_changes, refined_changes], dim=0)
    feasible = (
        changes
        <= -required.unsqueeze(0) + float(feasibility_tolerance)
    ).all(dim=1)
    required_audit = {
        name: float(required_reduction[name]) for name in names
    }
    if not bool(feasible.any()):
        best_progress = {
            name: float(changes[:, index].amin().detach())
            for index, name in enumerate(names)
        }
        normalized_residual = (
            changes
            + required.unsqueeze(0)
            - float(feasibility_tolerance)
        ) / scales.unsqueeze(0)
        nearest = normalized_residual.amax(dim=1).argmin()
        return None, {
            "solver_status": "insufficient_second_order_predicted_progress",
            "second_order_state": "insufficient_second_order_predicted_progress",
            "theta_radians": theta,
            "candidate_count": int(directions.shape[0]),
            "coarse_candidate_count": int(coarse_directions.shape[0]),
            "feasible_candidate_count": 0,
            "best_predicted_change_by_term": best_progress,
            "nearest_joint_predicted_change_by_term": {
                name: float(changes[nearest, index].detach())
                for index, name in enumerate(names)
            },
            "nearest_joint_maximum_normalized_residual": float(
                normalized_residual[nearest].amax().detach()
            ),
            "required_reduction_by_term": required_audit,
            "constraint_scale_by_term": {
                name: float(scales[index].detach())
                for index, name in enumerate(names)
            },
            **refinement_audit,
        }
    indices = m.torch.nonzero(feasible, as_tuple=False).reshape(-1)
    feasible_rows = directions.index_select(0, indices)
    feasible_changes = changes.index_select(0, indices)
    shadow, endpoint, temporal = (
        feasible_changes[:, index] for index in range(3)
    )
    # Deterministic lexicographic ordering: strongest worst-constraint closure,
    # then shadow, endpoint, temporal, and finally the grid order.
    normalized = (
        feasible_changes + required.unsqueeze(0)
    ) / scales.unsqueeze(0)
    worst = normalized.amax(dim=1)
    # Resolve the exact lexicographic order on-device.  This avoids one host
    # synchronization per feasible row while retaining deterministic ties.
    remaining = m.torch.arange(
        indices.numel(), dtype=m.torch.long, device=indices.device
    )
    for values in (worst, shadow, endpoint, temporal):
        selected_values = values.index_select(0, remaining)
        minimum = selected_values.amin()
        remaining = remaining[selected_values == minimum]
    chosen_local_tensor = remaining.amin()
    chosen_local = int(chosen_local_tensor.detach())
    chosen_global_tensor = indices[chosen_local_tensor]
    chosen_global = int(chosen_global_tensor.detach())
    coefficients = feasible_rows[chosen_local_tensor]
    selected_changes = {
        name: float(changes[chosen_global, index].detach())
        for index, name in enumerate(names)
    }
    active = [
        name
        for name in names
        if abs(selected_changes[name] + float(required_reduction[name]))
        <= max(float(feasibility_tolerance), 1.0e-12)
    ]
    return coefficients, {
        "solver_status": "second_order_angle_subproblem_solved",
        "second_order_state": None,
        "theta_radians": theta,
        "candidate_count": int(directions.shape[0]),
        "coarse_candidate_count": int(coarse_directions.shape[0]),
        "feasible_candidate_count": int(indices.numel()),
        "selected_candidate_index": chosen_global,
        "selected_active_constraints": active,
        "predicted_change_by_term": selected_changes,
        "required_reduction_by_term": required_audit,
        "constraint_scale_by_term": {
            name: float(scales[index].detach())
            for index, name in enumerate(names)
        },
        **refinement_audit,
    }


def prepare_second_order_subproblem(
    *,
    current,
    mask,
    taper,
    gradients,
    metric_builder,
    basis_dimension,
    direction_norm_floor,
):
    """Build one curvature model for reuse by every frozen angle."""
    basis, basis_audit = build_owned_tangent_basis(
        current=current,
        mask=mask,
        taper=taper,
        gradients=gradients,
        dimension=int(basis_dimension),
        floor=float(direction_norm_floor),
    )
    if basis is None:
        return None, basis_audit
    current64 = current.detach().to(m.torch.float64)
    models, curvature_audit = build_second_order_models(
        current=current64,
        mask=mask,
        basis=basis,
        metric_builder=metric_builder,
    )
    if models is None:
        return None, {**basis_audit, **curvature_audit}
    verified_basis = curvature_audit.pop("verified_basis")
    verified_indices = curvature_audit["verified_basis_indices"]
    basis_sources = list(basis_audit.get("basis_sources") or [])
    curvature_audit["verified_basis_sources"] = [
        basis_sources[index] for index in verified_indices
    ]
    serial_models = {
        name: {
            "value": float(model["value"]),
            "first_derivative_in_basis": [
                float(value) for value in model["first"].detach().cpu()
            ],
            "hessian_in_basis": [
                [float(value) for value in row]
                for row in model["hessian"].detach().cpu()
            ],
        }
        for name, model in models.items()
    }
    return {
        "basis": verified_basis,
        "models": models,
        "current64": current64,
        "mask": mask,
        "direction_norm_floor": float(direction_norm_floor),
    }, {
        **basis_audit,
        **curvature_audit,
        "low_dimensional_models": serial_models,
        "second_order_hessian_used": True,
        "full_hessian_formed": False,
        "model_reused_across_frozen_angles": True,
    }


def solve_prepared_second_order_angle(
    *,
    prepared,
    theta_radians,
    required_reduction,
    grid_levels,
    feasibility_tolerance,
):
    """Solve one angle using a prepared on-device curvature model."""
    coefficients, solver_audit = solve_angle_subproblem(
        models=prepared["models"],
        theta_radians=float(theta_radians),
        required_reduction=required_reduction,
        grid_levels=int(grid_levels),
        feasibility_tolerance=float(feasibility_tolerance),
    )
    audit = {
        **solver_audit,
        "second_order_hessian_used": True,
        "full_hessian_formed": False,
        "curvature_model_reused": True,
    }
    if coefficients is None:
        return None, audit
    basis = prepared["basis"]
    current64 = prepared["current64"]
    mask = prepared["mask"]
    floor = prepared["direction_norm_floor"]
    unit = m.torch.einsum("i,i...->...", coefficients, basis)
    unit = _unit_owned_sphere_tangent(
        unit, current64, mask, floor=float(floor)
    )
    if unit is None or not bool(m.torch.isfinite(unit).all()):
        return None, {
            **audit,
            "solver_status": "second_order_solver_failure",
            "second_order_state": "second_order_solver_failure",
            "reason": "selected_direction_normalization_failed",
        }
    radius = m.torch.linalg.vector_norm(current64[mask])
    direction = (radius * unit).to(basis.dtype).masked_fill(~mask, 0.0)
    audit["selected_basis_coefficients"] = [
        float(value) for value in coefficients.detach().cpu()
    ]
    audit["geodesic_direction_norm"] = float(
        m.torch.linalg.vector_norm(direction[mask]).detach()
    )
    audit["geodesic_radial_inner_product"] = float(
        _inner(direction, current64, mask).detach()
    )
    return direction.detach(), audit


def second_order_direction_for_angle(
    *,
    current,
    mask,
    taper,
    gradients,
    metric_builder,
    theta_radians,
    required_reduction,
    basis_dimension,
    grid_levels,
    direction_norm_floor,
    feasibility_tolerance,
):
    """Build and solve one frozen-angle g1f3 joint subproblem."""
    prepared, preparation_audit = prepare_second_order_subproblem(
        current=current,
        mask=mask,
        taper=taper,
        gradients=gradients,
        metric_builder=metric_builder,
        basis_dimension=int(basis_dimension),
        direction_norm_floor=float(direction_norm_floor),
    )
    if prepared is None:
        return None, {
            **preparation_audit,
            "theta_radians": float(theta_radians),
        }
    direction, solver_audit = solve_prepared_second_order_angle(
        prepared=prepared,
        theta_radians=float(theta_radians),
        required_reduction=required_reduction,
        grid_levels=int(grid_levels),
        feasibility_tolerance=float(feasibility_tolerance),
    )
    audit = {
        **preparation_audit,
        **solver_audit,
    }
    if direction is None:
        return None, audit
    return direction.to(current.dtype).detach(), audit
