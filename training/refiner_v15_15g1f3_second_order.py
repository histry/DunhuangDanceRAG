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


def _directional_jet(
    *,
    current,
    mask,
    unit_direction,
    metric_builder: Callable[[object], Mapping[str, object]],
    names: Sequence[str],
):
    """Return F(0), dF/dtheta and d2F/dtheta2 for one real path."""
    theta = current.new_zeros((), dtype=m.torch.float64, requires_grad=True)
    trial = differentiable_geodesic_trial(current, unit_direction, mask, theta)
    metrics = metric_builder(trial)
    result = {}
    for index, name in enumerate(names):
        value = metrics[name]
        if value.numel() != 1 or not bool(m.torch.isfinite(value).all()):
            return None
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
            return None
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
            return None
        result[name] = (
            value.detach(),
            first.detach(),
            second.detach(),
        )
    return result


def build_second_order_models(
    *,
    current,
    mask,
    basis,
    metric_builder: Callable[[object], Mapping[str, object]],
    names: Sequence[str] = ("shadow", "endpoint", "temporal"),
):
    """Build low-dimensional Hessians exclusively from directional HvPs."""
    count = int(basis.shape[0])
    diagonal = []
    first_columns = []
    base_values: Optional[Dict[str, object]] = None
    for index in range(count):
        jet = _directional_jet(
            current=current,
            mask=mask,
            unit_direction=basis[index],
            metric_builder=metric_builder,
            names=names,
        )
        if jet is None:
            return None, {
                "status": "nonfinite_or_unverified_curvature",
                "reason": f"basis_hvp_{index}_nonfinite",
            }
        if base_values is None:
            base_values = {name: jet[name][0] for name in names}
        first_columns.append({name: jet[name][1] for name in names})
        diagonal.append({name: jet[name][2] for name in names})

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
            direction = (basis[left] + basis[right]) / math.sqrt(2.0)
            jet = _directional_jet(
                current=current,
                mask=mask,
                unit_direction=direction,
                metric_builder=metric_builder,
                names=names,
            )
            if jet is None:
                return None, {
                    "status": "nonfinite_or_unverified_curvature",
                    "reason": f"pair_hvp_{left}_{right}_nonfinite",
                }
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
            "reason": "nonfinite_low_dimensional_model",
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
    directions = _deterministic_unit_grid(
        dimension, int(grid_levels), device=models["shadow"]["first"].device
    )
    theta = float(theta_radians)
    predicted = {}
    feasible = m.torch.ones(
        directions.shape[0], dtype=m.torch.bool, device=directions.device
    )
    for name in names:
        first = models[name]["first"]
        hessian = models[name]["hessian"]
        linear = directions @ first
        quadratic = m.torch.einsum("ni,ij,nj->n", directions, hessian, directions)
        change = theta * linear + 0.5 * theta * theta * quadratic
        predicted[name] = change
        feasible &= change <= (
            -float(required_reduction[name]) + float(feasibility_tolerance)
        )
    if not bool(feasible.any()):
        best_progress = {
            name: float(predicted[name].amin().detach()) for name in names
        }
        return None, {
            "solver_status": "insufficient_second_order_predicted_progress",
            "second_order_state": "insufficient_second_order_predicted_progress",
            "theta_radians": theta,
            "candidate_count": int(directions.shape[0]),
            "feasible_candidate_count": 0,
            "best_predicted_change_by_term": best_progress,
        }
    indices = m.torch.nonzero(feasible, as_tuple=False).reshape(-1)
    feasible_rows = directions.index_select(0, indices)
    shadow = predicted["shadow"].index_select(0, indices)
    endpoint = predicted["endpoint"].index_select(0, indices)
    temporal = predicted["temporal"].index_select(0, indices)
    # Deterministic lexicographic ordering: strongest worst-constraint closure,
    # then shadow, endpoint, temporal, and finally the grid order.
    normalized = m.torch.stack(
        [
            shadow / max(float(required_reduction["shadow"]), 1.0e-30),
            endpoint / max(float(required_reduction["endpoint"]), 1.0e-30),
            temporal / max(float(required_reduction["temporal"]), 1.0e-30),
        ],
        dim=1,
    )
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
    changes = {
        name: float(predicted[name][chosen_global].detach()) for name in names
    }
    active = [
        name
        for name in names
        if abs(changes[name] + float(required_reduction[name]))
        <= max(float(feasibility_tolerance), 1.0e-12)
    ]
    return coefficients, {
        "solver_status": "second_order_angle_subproblem_solved",
        "second_order_state": None,
        "theta_radians": theta,
        "candidate_count": int(directions.shape[0]),
        "feasible_candidate_count": int(indices.numel()),
        "selected_grid_index": chosen_global,
        "selected_active_constraints": active,
        "predicted_change_by_term": changes,
        "required_reduction_by_term": {
            name: float(required_reduction[name]) for name in names
        },
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
        "basis": basis,
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
