"""Development-only V15.14g case-local finite-radius inequality probe.

Every action lives in the fixed bank's ``[case, frame, 79]`` product tangent
and is exactly zero outside one case and its owned C2-tapered seam.  A small
one-sided finite difference initializes the local cone, but candidate decisions
are made only at the resolved 1e-4 output radius.  Failed finite-radius trials
add case/frame/joint witness cuts and refresh their Jacobian before the next
signed hard-correction step.  No training or checkpoint mutation occurs.
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
from training import refiner_group_local_nullspace_cone_probe as group_probe
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_14g_case_local_finite_radius_sequential_cone_v1"
PROTOCOL = "fixed_bank_case_local_witness_cut_sequential_inequality_v1"
DEFAULT_FD_EPSILON = 1.0e-6
DEFAULT_TARGET_RMS = 1.0e-4
DEFAULT_MAX_ITERATIONS = 6
DEFAULT_TOPK_WITNESSES = 4
ACTIVE_FRACTION = 0.10
LINEAR_TOLERANCE = 1.0e-8
STRICT_DESCENT_FLOOR = 1.0e-7
SCIENCE_SEEDS = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))

CASE_HARD_TERMS = {
    "joint_jerk_p95": "repair_joint_jerk_mps3_p95_signed_margin",
    "joint_jerk_max": "repair_joint_jerk_mps3_max_signed_margin",
    "joint_jerk_window_p95": (
        "repair_joint_jerk_window_p95_max_mps3_signed_margin"
    ),
    "extremity_jerk_p95": (
        "repair_extremity_jerk_mps3_p95_signed_margin"
    ),
    "extremity_jerk_window_p95": (
        "repair_extremity_jerk_window_p95_max_mps3_signed_margin"
    ),
    "foot_skate_p95": "repair_foot_skate_mps_p95_signed_margin",
    "foot_skate_max": "repair_foot_skate_mps_max_signed_margin",
    "support_drift_p95": (
        "repair_foot_support_drift_m_p95_signed_margin"
    ),
    "support_drift_max": (
        "repair_foot_support_drift_m_max_signed_margin"
    ),
    "penetration": "repair_foot_penetration_min_m_signed_margin",
    "boundary": "boundary_jerk_signed_margin",
}

WITNESS_GUARD_SUFFIX = {
    "joint_jerk": "joint_jerk_max",
    "boundary_jerk": "boundary",
    "penetration": "penetration",
    "foot_skate": "foot_skate_max",
    "support_drift": "support_drift_max",
}


def _case_activity(batch, case_index, cfg):
    taper_frames = max(
        1,
        int(getattr(cfg, "product_refiner_residual_taper_frames", 3)),
    )
    owned, c2 = projected_probe._ownership_c2_activity(
        batch["seam"], taper_frames
    )
    selected = m.torch.zeros_like(owned)
    selected[int(case_index)] = True
    active = selected & owned
    weighted = active[..., None].to(batch["bad"].dtype) * c2[..., None]
    return active, weighted, taper_frames


def _case_scope(action, batch, case_index):
    tangent = group_probe._tangent_from_action(action)
    owned = batch["seam"] >= 0.5
    selected = m.torch.zeros_like(owned)
    selected[int(case_index)] = True
    permitted = (selected & owned).expand_as(tangent)
    outside = ~permitted
    outside_max = (
        float(tangent[outside].abs().max().detach())
        if bool(outside.any()) else 0.0
    )
    contact_max = float(action[..., :4].abs().max().detach())
    return {
        "case_index": int(case_index),
        "outside_case_group_or_ownership_abs_max": outside_max,
        "contact_action_abs_max": contact_max,
        "scope_safe": bool(outside_max == 0.0 and contact_max == 0.0),
    }


def _case_terms(prediction, batch, cfg):
    _, terms = m._observable_refiner_objective(
        prediction,
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )
    values = {
        "endpoint": terms["endpoint_scientific_deficit"],
        "temporal": terms["temporal_scientific_deficit"],
    }
    for suffix, term in CASE_HARD_TERMS.items():
        values[suffix] = terms[term]
    values["fixed_support"] = m.torch.stack([
        values["foot_skate_p95"],
        values["foot_skate_max"],
        values["support_drift_p95"],
        values["support_drift_max"],
        values["penetration"],
    ]).max(dim=0).values
    return values


def _witness_primitives(prediction, batch, cfg):
    joints = m._observable_boundary_joints_torch(prediction)
    reference_joints = m._observable_boundary_joints_torch(
        batch["bad"].detach()
    )
    fps = float(cfg.fps)
    jerk = m.torch.linalg.vector_norm(
        m.torch.diff(joints.to(m.torch.float64), n=3, dim=1) * fps**3,
        dim=-1,
    )
    foot_ids = list(m.DEFAULT_FOOT_JOINTS)
    feet = joints[..., foot_ids, :]
    reference_feet = reference_joints[..., foot_ids, :]
    floor = m.torch.quantile(
        reference_feet[..., 1].flatten(1), 0.05, dim=1
    ).detach()
    penetration = -(feet[..., 1] - floor[:, None, None])
    _, _, static = m._reference_support_statistics_torch(
        joints,
        reference_joints,
        batch["bad"][..., :4],
        cfg,
    )
    foot_xz = feet[..., (0, 2)].to(m.torch.float64)
    speed = m.torch.nn.functional.pad(
        m.torch.linalg.vector_norm(
            m.torch.diff(foot_xz, dim=1), dim=-1
        ) * fps,
        (0, 0, 1, 0),
    )
    frame_index = m.torch.arange(
        static.shape[1], device=static.device
    )[None, :, None].expand_as(static)
    starts = static & ~m.torch.nn.functional.pad(
        static[:, :-1], (0, 0, 1, 0), value=False
    )
    anchors = m.torch.where(
        starts, frame_index, m.torch.zeros_like(frame_index)
    ).cummax(1).values
    anchor_xz = foot_xz.gather(
        1, anchors[..., None].expand(-1, -1, -1, 2)
    )
    drift = m.torch.linalg.vector_norm(foot_xz - anchor_xz, dim=-1)
    return {
        "joints": joints,
        "joint_jerk": jerk,
        "penetration": penetration,
        "foot_skate": speed,
        "support_drift": drift,
        "static_support": static,
    }


def _rank_witnesses(
    values,
    *,
    case_index,
    kind,
    valid=None,
    topk=DEFAULT_TOPK_WITNESSES,
    frame_offset=0,
    frame_limit=None,
):
    case_values = values[int(case_index)].detach().double()
    if valid is None:
        valid_case = m.torch.ones_like(case_values, dtype=m.torch.bool)
    else:
        valid_case = valid[int(case_index)].expand_as(case_values)
    if frame_limit is not None:
        indices = m.torch.arange(
            case_values.shape[0], device=case_values.device
        )
        valid_case = valid_case & frame_limit(indices)[:, None]
    flat = case_values.flatten()
    valid_flat = valid_case.flatten()
    positions = m.torch.nonzero(valid_flat, as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    selected_values = flat[positions]
    maximum = selected_values.max()
    active_threshold = maximum - ACTIVE_FRACTION * maximum.abs().clamp_min(
        1.0e-12
    )
    active = positions[selected_values >= active_threshold]
    ordered = positions[m.torch.argsort(selected_values, descending=True)]
    keep = []
    for index in m.torch.cat((active, ordered[: int(topk)])).tolist():
        if index not in keep:
            keep.append(int(index))
        if len(keep) >= max(int(topk), 2 * int(topk)):
            break
    width = int(case_values.shape[1])
    rows = []
    for flat_index in keep:
        frame = flat_index // width
        element = flat_index % width
        rows.append({
            "kind": kind,
            "frame": int(frame + frame_offset),
            "sample_index": int(frame),
            "joint_or_foot": int(element),
            "baseline_value": float(case_values[frame, element]),
        })
    return rows


def _discover_witnesses(primitives, batch, case_index, topk):
    seam = batch["seam"][int(case_index), :, 0] >= 0.5
    active_frames = m.torch.nonzero(seam, as_tuple=False).flatten()
    if active_frames.numel() == 0:
        return []
    left = int(active_frames[0])
    right = int(active_frames[-1])
    witnesses = []
    witnesses.extend(_rank_witnesses(
        primitives["joint_jerk"],
        case_index=case_index,
        kind="joint_jerk",
        topk=topk,
        frame_offset=2,
    ))
    for side, boundary in (("left", left), ("right", right)):
        rows = _rank_witnesses(
            primitives["joint_jerk"],
            case_index=case_index,
            kind="boundary_jerk",
            topk=max(1, int(topk) // 2),
            frame_offset=2,
            frame_limit=lambda index, b=boundary: (
                (index + 2 - b).abs() <= 3
            ),
        )
        for row in rows:
            row["boundary_side"] = side
        witnesses.extend(rows)
    witnesses.extend(_rank_witnesses(
        primitives["penetration"],
        case_index=case_index,
        kind="penetration",
        topk=topk,
    ))
    witnesses.extend(_rank_witnesses(
        primitives["foot_skate"],
        case_index=case_index,
        kind="foot_skate",
        valid=primitives["static_support"],
        topk=topk,
    ))
    witnesses.extend(_rank_witnesses(
        primitives["support_drift"],
        case_index=case_index,
        kind="support_drift",
        valid=primitives["static_support"],
        topk=topk,
    ))
    unique = {}
    for witness in witnesses:
        key = (
            witness["kind"],
            witness["sample_index"],
            witness["joint_or_foot"],
            witness.get("boundary_side"),
        )
        unique[key] = witness
    return list(unique.values())


def _witness_key(witness):
    side = witness.get("boundary_side", "none")
    return (
        f"witness:{witness['kind']}:frame={witness['frame']}:"
        f"element={witness['joint_or_foot']}:side={side}"
    )


def _witness_value(primitives, case_index, witness):
    source = (
        "joint_jerk"
        if witness["kind"] == "boundary_jerk"
        else witness["kind"]
    )
    return primitives[source][
        int(case_index),
        int(witness["sample_index"]),
        int(witness["joint_or_foot"]),
    ]


def _case_value_tensors(prediction, batch, cfg, case_index, witnesses):
    terms = _case_terms(prediction, batch, cfg)
    values = {
        key: tensor[int(case_index)] for key, tensor in terms.items()
    }
    primitives = _witness_primitives(prediction, batch, cfg)
    for witness in witnesses:
        values[_witness_key(witness)] = _witness_value(
            primitives, case_index, witness
        )
    return values, primitives


def _float_values(values):
    return {key: float(value.detach()) for key, value in values.items()}


def _case_active_tangent_rms(tangent, seam, case_index):
    selected_tangent = tangent[int(case_index):int(case_index) + 1]
    selected_seam = seam[int(case_index):int(case_index) + 1]
    active = (selected_seam >= 0.5).expand_as(selected_tangent)
    values = selected_tangent[active]
    if values.numel() == 0:
        return 0.0
    return float(m.torch.sqrt(m.torch.mean(values.square())).detach())


def _normalize_case_action(action, seam, case_index):
    tangent = group_probe._tangent_from_action(action)
    rms = _case_active_tangent_rms(tangent, seam, case_index)
    if not math.isfinite(rms) or rms <= 1.0e-14:
        return m.torch.zeros_like(action), rms
    return action / rms, rms


def _action_from_scalar(scalar, edit, weighted, seam, case_index):
    gradient = m.torch.autograd.grad(
        scalar, edit, retain_graph=True, allow_unused=True
    )[0]
    if gradient is None:
        tangent = m.torch.zeros_like(edit)
    else:
        tangent = -gradient * weighted
    return _normalize_case_action(
        group_probe._product_action(tangent.detach()), seam, case_index
    )


def _build_case_basis(
    baseline,
    batch,
    cfg,
    case_index,
    witnesses,
):
    edit = m.torch.zeros(
        baseline.shape[:-1] + (75,),
        dtype=baseline.dtype,
        device=baseline.device,
        requires_grad=True,
    )
    candidate = product_exp_torch(baseline, edit)
    values, primitives = _case_value_tensors(
        candidate, batch, cfg, case_index, witnesses
    )
    _, weighted, taper_frames = _case_activity(batch, case_index, cfg)
    directions = {}
    rows = []
    for objective in ("endpoint", "temporal"):
        action, before_rms = _action_from_scalar(
            values[objective], edit, weighted, batch["seam"], case_index
        )
        source = f"case={case_index}.{objective}"
        if before_rms > 1.0e-12:
            directions[source] = action
        rows.append({
            "direction_source": source,
            "kind": "scientific",
            "objective": objective,
            "available": bool(before_rms > 1.0e-12),
            "pre_normalization_rms": before_rms,
        })
    for witness in witnesses:
        key = _witness_key(witness)
        action, before_rms = _action_from_scalar(
            values[key], edit, weighted, batch["seam"], case_index
        )
        source = f"case={case_index}.hard_correction:{key}"
        if before_rms > 1.0e-12:
            directions[source] = action
        rows.append({
            "direction_source": source,
            "kind": "hard_correction",
            "witness": witness,
            "available": bool(before_rms > 1.0e-12),
            "pre_normalization_rms": before_rms,
        })
    del primitives
    return directions, rows, taper_frames


def _candidate_from_action(reference, action, seam, target, case_index):
    tangent = group_probe._tangent_from_action(action)
    scale = float(target)
    candidate = reference
    achieved = 0.0
    case_slice = slice(int(case_index), int(case_index) + 1)
    for _ in range(6):
        candidate = product_exp_torch(reference, scale * tangent)
        achieved = projected_probe._motion_edit_rms(
            reference[case_slice],
            candidate[case_slice],
            seam[case_slice],
        )
        if not math.isfinite(achieved) or achieved <= 1.0e-16:
            scale *= 10.0
            continue
        if achieved >= target and abs(achieved / target - 1.0) <= 1.0e-3:
            break
        scale *= (float(target) / achieved) * 1.0001
    return candidate.detach(), float(scale), float(achieved)


def _perturb(reference, action, seam, epsilon, case_index):
    candidate, _, _ = _candidate_from_action(
        reference, action, seam, epsilon, case_index
    )
    return candidate


def _fd_matrix(
    reference,
    batch,
    cfg,
    case_index,
    witnesses,
    directions,
    keys,
    epsilon,
):
    reference_tensors, _ = _case_value_tensors(
        reference, batch, cfg, case_index, witnesses
    )
    reference_values = _float_values(reference_tensors)
    matrix = np.zeros((len(keys), len(directions)), dtype=np.float64)
    fd_rows = []
    for column, (source, action) in enumerate(directions.items()):
        trial = _perturb(
            reference, action, batch["seam"], epsilon, case_index
        )
        trial_tensors, _ = _case_value_tensors(
            trial, batch, cfg, case_index, witnesses
        )
        trial_values = _float_values(trial_tensors)
        derivative = {}
        for row, key in enumerate(keys):
            value = (
                trial_values[key] - reference_values[key]
            ) / float(epsilon)
            matrix[row, column] = value
            derivative[key] = value
        fd_rows.append({
            "direction_source": source,
            "epsilon_output_tangent_rms": float(epsilon),
            "diagnostic_only": True,
            "eligible_as_learning_step": False,
            "directional_derivative": derivative,
        })
    return matrix, reference_values, fd_rows


def _jacobian_diagnostics(matrix):
    if matrix.size == 0:
        return {
            "shape": list(matrix.shape),
            "effective_rank": 0,
            "rank_deficient": False,
            "nullity": 0,
            "condition_number": None,
            "singular_values": [],
        }
    singular = np.linalg.svd(matrix, compute_uv=False)
    maximum = float(singular[0]) if singular.size else 0.0
    threshold = max(1.0e-12, maximum * 1.0e-8)
    effective = singular[singular > threshold]
    condition = (
        float(maximum / effective[-1]) if effective.size else math.inf
    )
    return {
        "shape": list(matrix.shape),
        "effective_rank": int(effective.size),
        "rank_deficient": bool(
            effective.size < min(matrix.shape)
        ),
        "nullity": int(max(0, matrix.shape[1] - effective.size)),
        "condition_number": condition,
        "singular_values": [float(value) for value in singular],
        "effective_rank_threshold": threshold,
    }


def _project_halfspaces(offset, matrix, *, iterations=512):
    """Find a deterministic signed correction satisfying offset+A*x <= 0."""
    offset = np.asarray(offset, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] != offset.size:
        raise ValueError("halfspace matrix and offset differ")
    if matrix.shape[1] == 0:
        feasible = bool(np.all(offset <= LINEAR_TOLERANCE))
        return np.zeros((0,), dtype=np.float64), feasible, 0
    row_scale = np.maximum(1.0, np.max(np.abs(matrix), axis=1))
    normalized_a = matrix / row_scale[:, None]
    normalized_b = offset / row_scale
    value = np.zeros((matrix.shape[1],), dtype=np.float64)
    completed = 0
    for completed in range(1, int(iterations) + 1):
        changed = False
        for row, base in zip(normalized_a, normalized_b):
            violation = float(base + row @ value)
            if violation <= LINEAR_TOLERANCE:
                continue
            denominator = float(row @ row)
            if denominator <= 1.0e-20:
                continue
            value -= (violation / denominator) * row
            changed = True
        if not changed:
            break
    feasible = bool(
        np.all(normalized_b + normalized_a @ value <= LINEAR_TOLERANCE)
    )
    return value, feasible, completed


def _constraint_allowances(
    group_name,
    baseline_values,
    baseline_guard_details,
    witnesses,
):
    allowed = {}
    allowed["endpoint"] = baseline_values["endpoint"]
    allowed["temporal"] = baseline_values["temporal"]
    metadata = {
        "endpoint": {"cut_priority": "scientific_hard_nonregression"},
        "temporal": {"cut_priority": "scientific_hard_nonregression"},
    }
    for suffix in CASE_HARD_TERMS:
        guard_key = f"{group_name}.{suffix}"
        detail = baseline_guard_details[guard_key]
        margin = max(0.0, float(detail["remaining_margin"]))
        allowed[suffix] = baseline_values[suffix] + margin
        metadata[suffix] = {
            "cut_priority": (
                "soft_within_fixed_absolute_limit"
                if "jerk" in suffix else "hard_fixed_limit"
            ),
            "group_guard_key": guard_key,
            "remaining_fixed_margin": margin,
        }
    fixed_key = f"{group_name}.fixed_support"
    fixed_detail = baseline_guard_details[fixed_key]
    fixed_margin = max(0.0, float(fixed_detail["remaining_margin"]))
    allowed["fixed_support"] = (
        baseline_values["fixed_support"] + fixed_margin
    )
    metadata["fixed_support"] = {
        "cut_priority": "hard_fixed_limit",
        "group_guard_key": fixed_key,
        "remaining_fixed_margin": fixed_margin,
    }
    for witness in witnesses:
        key = _witness_key(witness)
        suffix = WITNESS_GUARD_SUFFIX[witness["kind"]]
        guard_key = f"{group_name}.{suffix}"
        detail = baseline_guard_details[guard_key]
        margin = max(0.0, float(detail["remaining_margin"]))
        allowed[key] = baseline_values[key] + margin
        metadata[key] = {
            "cut_priority": (
                "soft_within_fixed_absolute_limit"
                if "jerk" in witness["kind"] else "hard_fixed_limit"
            ),
            "group_guard_key": guard_key,
            "remaining_fixed_margin": margin,
            "witness": witness,
        }
    return allowed, metadata


def _scientific_status(delta, baseline_values):
    tolerances = {
        key: max(1.0e-12, abs(float(baseline_values[key])) * 1.0e-9)
        for key in ("endpoint", "temporal")
    }
    nonregression = all(
        float(delta[key]) <= tolerances[key]
        for key in ("endpoint", "temporal")
    )
    strict = any(
        float(delta[key]) < -max(tolerances[key], STRICT_DESCENT_FLOOR)
        for key in ("endpoint", "temporal")
    )
    return {
        "endpoint_delta": float(delta["endpoint"]),
        "temporal_delta": float(delta["temporal"]),
        "nonregression": bool(nonregression),
        "strict_descent": bool(strict),
        "passed": bool(nonregression and strict),
        "numeric_tolerance": tolerances,
    }


def _merge_witnesses(existing, discovered):
    merged = {(_witness_key(row)): row for row in existing}
    for row in discovered:
        merged.setdefault(_witness_key(row), row)
    return list(merged.values())


def _combine_actions(directions, sources, coefficients):
    if not sources:
        raise ValueError("cannot combine an empty action basis")
    result = m.torch.zeros_like(directions[sources[0]])
    for source, coefficient in zip(sources, coefficients):
        result = result + float(coefficient) * directions[source]
    return result


def _case_trial(
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
    baseline_guard_details,
    case_index,
    group_name,
    science_weights,
    initial_witnesses,
    fd_epsilon,
    target_rms,
    max_iterations,
    topk,
):
    started = time.perf_counter()
    witnesses = list(initial_witnesses)
    directions, direction_rows, taper_frames = _build_case_basis(
        baseline, batch, cfg, case_index, witnesses
    )
    endpoint_source = f"case={case_index}.endpoint"
    temporal_source = f"case={case_index}.temporal"
    science_sources = [endpoint_source, temporal_source]
    correction_sources = [
        source for source in directions if ".hard_correction:" in source
    ]
    missing_science = [
        source for source in science_sources if source not in directions
    ]
    scope_rows = {
        source: _case_scope(action, batch, case_index)
        for source, action in directions.items()
    }
    scope_safe = bool(
        directions
        and all(row["scope_safe"] for row in scope_rows.values())
    )
    if missing_science or not scope_safe:
        return None, {
            "case_index": int(case_index),
            "group": group_name,
            "scientific_weights": list(science_weights),
            "case_local_linear_feasible": False,
            "case_local_exact_finite_radius_passed": False,
            "reason": (
                "missing_case_local_scientific_direction"
                if missing_science else "case_local_scope_leakage"
            ),
            "missing_scientific_directions": missing_science,
            "direction_rows": direction_rows,
            "action_scope": scope_rows,
            "active_witness_frames": sorted({
                int(witness["frame"]) for witness in witnesses
            }),
            "active_witness_joints": sorted({
                int(witness["joint_or_foot"])
                for witness in witnesses if "jerk" in witness["kind"]
            }),
            "witnesses": witnesses,
            "elapsed_seconds": time.perf_counter() - started,
        }

    constraint_keys = [
        "endpoint", "temporal", *CASE_HARD_TERMS, "fixed_support"
    ] + [_witness_key(row) for row in witnesses]
    initial_constraint_keys = list(constraint_keys)
    initial_matrix, baseline_values, initial_fd_rows = _fd_matrix(
        baseline,
        batch,
        cfg,
        case_index,
        witnesses,
        directions,
        constraint_keys,
        fd_epsilon,
    )
    allowed, cut_metadata = _constraint_allowances(
        group_name,
        baseline_values,
        baseline_guard_details,
        witnesses,
    )
    source_names = list(directions)
    science_columns = [source_names.index(source) for source in science_sources]
    correction_columns = [
        source_names.index(source) for source in correction_sources
    ]
    science_vector = np.asarray(science_weights, dtype=np.float64)
    initial_offset = initial_matrix[:, science_columns] @ science_vector
    initial_correction_matrix = initial_matrix[:, correction_columns]
    correction, linear_feasible, projection_iterations = _project_halfspaces(
        initial_offset, initial_correction_matrix
    )
    initial_directional = initial_offset + initial_correction_matrix @ correction
    scientific_rows = [
        constraint_keys.index("endpoint"),
        constraint_keys.index("temporal"),
    ]
    linear_scientific_strict = bool(
        np.any(initial_directional[scientific_rows] < -STRICT_DESCENT_FLOOR)
    )
    linear_feasible = bool(linear_feasible and linear_scientific_strict)
    jacobian_history = [{
        "iteration": 0,
        **_jacobian_diagnostics(initial_matrix),
        "projection_iterations": projection_iterations,
        "constraint_keys": constraint_keys,
    }]
    if not linear_feasible:
        return None, {
            "case_index": int(case_index),
            "group": group_name,
            "scientific_weights": list(science_weights),
            "case_local_linear_feasible": False,
            "case_local_exact_finite_radius_passed": False,
            "reason": "no_case_local_linearized_inequality_direction",
            "initial_directional_derivatives": {
                key: float(value)
                for key, value in zip(constraint_keys, initial_directional)
            },
            "initial_fd_rows": initial_fd_rows,
            "jacobian_history": jacobian_history,
            "direction_rows": direction_rows,
            "action_scope": scope_rows,
            "active_witness_frames": sorted({
                int(witness["frame"]) for witness in witnesses
            }),
            "active_witness_joints": sorted({
                int(witness["joint_or_foot"])
                for witness in witnesses if "jerk" in witness["kind"]
            }),
            "witnesses": witnesses,
            "elapsed_seconds": time.perf_counter() - started,
        }

    coefficients = np.concatenate((science_vector, correction))
    ordered_sources = science_sources + correction_sources
    iteration_rows = []
    accepted_prediction = None
    accepted_score = math.inf
    final_reason = "finite_radius_iterations_exhausted"
    final_scope = None
    for iteration in range(1, int(max_iterations) + 1):
        action = _combine_actions(
            directions, ordered_sources, coefficients
        )
        action, pre_normalization_rms = _normalize_case_action(
            action, batch["seam"], case_index
        )
        final_scope = _case_scope(action, batch, case_index)
        if pre_normalization_rms <= 1.0e-12 or not final_scope["scope_safe"]:
            final_reason = (
                "empty_case_local_sequential_direction"
                if pre_normalization_rms <= 1.0e-12
                else "case_local_scope_leakage"
            )
            break
        candidate, action_scale, achieved = _candidate_from_action(
            baseline, action, batch["seam"], target_rms, case_index
        )
        candidate_values_t, candidate_primitives = _case_value_tensors(
            candidate, batch, cfg, case_index, witnesses
        )
        candidate_values = _float_values(candidate_values_t)
        delta = {
            key: candidate_values[key] - baseline_values[key]
            for key in candidate_values
        }
        scientific = _scientific_status(delta, baseline_values)
        constraint_residual = {
            key: candidate_values[key] - allowed[key]
            for key in constraint_keys
        }
        case_constraints_passed = bool(all(
            value <= LINEAR_TOLERANCE
            for key, value in constraint_residual.items()
            if key not in {"endpoint", "temporal"}
        ))
        (
            _,
            full_passed,
            full_blockers,
            _,
            full_delta,
            full_scientific,
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
        resolved = bool(achieved >= float(target_rms))
        exact_passed = bool(
            resolved
            and final_scope["scope_safe"]
            and scientific["passed"]
            and case_constraints_passed
            and full_passed
        )
        row = {
            "iteration": iteration,
            "action_scale": action_scale,
            "achieved_output_tangent_rms": achieved,
            "resolved_required_radius": resolved,
            "pre_normalization_direction_rms": pre_normalization_rms,
            "signed_hard_correction_coefficients": {
                source: float(value)
                for source, value in zip(correction_sources, coefficients[2:])
            },
            "case_scientific_delta": scientific,
            "case_constraint_residual": constraint_residual,
            "case_constraints_passed": case_constraints_passed,
            "full_fixed_guard_passed": full_passed,
            "full_guard_blockers": full_blockers,
            "full_observable_residual_delta": full_delta,
            "full_scientific_status": full_scientific,
            "case_local_exact_finite_radius_passed": exact_passed,
            "active_witness_frames": sorted({
                int(witness["frame"]) for witness in witnesses
            }),
            "active_witness_joints": sorted({
                int(witness["joint_or_foot"])
                for witness in witnesses
                if "jerk" in witness["kind"]
            }),
        }
        iteration_rows.append(row)
        if exact_passed:
            accepted_prediction = candidate
            accepted_score = float(scientific["endpoint_delta"])
            accepted_score += float(scientific["temporal_delta"])
            final_reason = "case_local_exact_finite_radius_descent_exists"
            break

        discovered = _discover_witnesses(
            candidate_primitives, batch, case_index, topk
        )
        merged = _merge_witnesses(witnesses, discovered)
        newly_active = [
            witness for witness in merged
            if _witness_key(witness) not in {
                _witness_key(old) for old in witnesses
            }
        ]
        row["new_active_witnesses"] = newly_active
        if newly_active:
            # New witnesses become cuts immediately.  Keep the correction
            # basis fixed so every coefficient remains in the same baseline
            # tangent coordinate system throughout the sequential solve.
            witnesses = merged
            constraint_keys = [
                "endpoint", "temporal", *CASE_HARD_TERMS, "fixed_support"
            ] + [_witness_key(item) for item in witnesses]
            baseline_tensors, _ = _case_value_tensors(
                baseline, batch, cfg, case_index, witnesses
            )
            baseline_values = _float_values(baseline_tensors)
            allowed, cut_metadata = _constraint_allowances(
                group_name,
                baseline_values,
                baseline_guard_details,
                witnesses,
            )

        current_tensors, _ = _case_value_tensors(
            candidate, batch, cfg, case_index, witnesses
        )
        current_values = _float_values(current_tensors)
        correction_directions = {
            source: directions[source] for source in correction_sources
        }
        current_matrix, _, current_fd_rows = _fd_matrix(
            candidate,
            batch,
            cfg,
            case_index,
            witnesses,
            correction_directions,
            constraint_keys,
            fd_epsilon,
        )
        residual_vector = np.asarray([
            current_values[key] - allowed[key] for key in constraint_keys
        ], dtype=np.float64)
        correction_step, step_feasible, step_iterations = _project_halfspaces(
            residual_vector, current_matrix
        )
        predicted_delta = current_matrix @ correction_step
        row["fresh_cut_fd_rows"] = current_fd_rows
        row["linear_predicted_constraint_delta"] = {
            key: float(value)
            for key, value in zip(constraint_keys, predicted_delta)
        }
        row["fresh_cut_solver_feasible"] = step_feasible
        row["fresh_cut_projection_iterations"] = step_iterations
        jacobian = _jacobian_diagnostics(current_matrix)
        jacobian["iteration"] = iteration
        jacobian["constraint_keys"] = constraint_keys
        jacobian_history.append(jacobian)
        if not step_feasible:
            final_reason = "sequential_cut_feasible_region_empty"
            break
        if correction_step.size == 0 or float(
            np.linalg.norm(correction_step)
        ) <= 1.0e-12:
            final_reason = "sequential_cut_correction_stalled"
            break
        coefficients[2:] += correction_step

        next_action = _combine_actions(
            directions, ordered_sources, coefficients
        )
        next_action, _ = _normalize_case_action(
            next_action, batch["seam"], case_index
        )
        next_candidate, _, _ = _candidate_from_action(
            baseline,
            next_action,
            batch["seam"],
            target_rms,
            case_index,
        )
        next_tensors, _ = _case_value_tensors(
            next_candidate, batch, cfg, case_index, witnesses
        )
        next_values = _float_values(next_tensors)
        actual_delta = np.asarray([
            next_values[key] - current_values[key]
            for key in constraint_keys
        ], dtype=np.float64)
        row["actual_constraint_delta_after_rematerialization"] = {
            key: float(value)
            for key, value in zip(constraint_keys, actual_delta)
        }
        row["predicted_vs_actual_constraint_delta"] = {
            key: {
                "predicted": float(predicted),
                "actual": float(actual),
            }
            for key, predicted, actual in zip(
                constraint_keys, predicted_delta, actual_delta
            )
        }
        row["constraint_curvature_ratio"] = {
            key: (
                float(actual / predicted)
                if abs(float(predicted)) > 1.0e-12 else None
            )
            for key, predicted, actual in zip(
                constraint_keys, predicted_delta, actual_delta
            )
        }

    result = {
        "case_index": int(case_index),
        "group": group_name,
        "scientific_weights": [float(value) for value in science_weights],
        "case_local_linear_feasible": True,
        "case_local_exact_finite_radius_passed": bool(
            accepted_prediction is not None
        ),
        "reason": final_reason,
        "accepted_score": (
            accepted_score if accepted_prediction is not None else None
        ),
        "iterations": iteration_rows,
        "jacobian_history": jacobian_history,
        "initial_fd_rows": initial_fd_rows,
        "initial_directional_derivatives": {
            key: float(value)
            for key, value in zip(initial_constraint_keys, initial_directional)
        },
        "cut_metadata": cut_metadata,
        "active_witness_frames": sorted({
            int(witness["frame"]) for witness in witnesses
        }),
        "active_witness_joints": sorted({
            int(witness["joint_or_foot"])
            for witness in witnesses if "jerk" in witness["kind"]
        }),
        "witnesses": witnesses,
        "c2_taper_frames": taper_frames,
        "action_scope": final_scope or scope_rows,
        "direction_rows": direction_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return accepted_prediction, result


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
        raise RuntimeError("V15.14g requires an unpublished failed diagnostic")
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.14g finite-radius probe requires CUDA")
    state = m.torch.load(state_path, map_location="cpu", weights_only=False)
    artifact = m.torch.load(fit_path, map_location="cpu", weights_only=False)
    if state.get("formal_checkpoint") or artifact.get("formal_checkpoint"):
        raise RuntimeError("formal checkpoint input is forbidden")

    batch, _, schedule = projected_probe._materialize_first_transaction(
        artifact, device
    )
    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.train()
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
    baseline_passed, baseline_blockers, baseline_details = (
        projected_probe._audit_guard_candidate(
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
    )
    baseline_primitives = _witness_primitives(baseline, batch, cfg)

    case_rows = []
    best_by_case = {}
    reason_counts = Counter()
    linear_feasible_by_group = Counter()
    exact_trial_pass_by_group = Counter()
    violating_case_indices = set()
    for case_index in range(int(baseline.shape[0])):
        group_index = int(batch["group"][case_index].detach())
        group_name = m.REFINER_GROUP_LABELS[group_index]
        witnesses = _discover_witnesses(
            baseline_primitives,
            batch,
            case_index,
            int(args.topk_witnesses),
        )
        for science_weights in SCIENCE_SEEDS:
            prediction, row = _case_trial(
                model=model,
                batch=batch,
                cfg=cfg,
                baseline=baseline,
                identity=identity,
                baseline_guard=baseline_guard,
                guard_anchor=guard_anchor,
                guard_relative=guard_relative,
                guard_absolute=guard_absolute,
                baseline_guard_details=baseline_details,
                case_index=case_index,
                group_name=group_name,
                science_weights=science_weights,
                initial_witnesses=witnesses,
                fd_epsilon=float(args.fd_epsilon),
                target_rms=float(args.target_rms),
                max_iterations=int(args.max_iterations),
                topk=int(args.topk_witnesses),
            )
            case_rows.append(row)
            reason_counts.update([row["reason"]])
            if row["case_local_linear_feasible"]:
                linear_feasible_by_group.update([group_name])
            if prediction is not None:
                exact_trial_pass_by_group.update([group_name])
                previous = best_by_case.get(case_index)
                if previous is None or row["accepted_score"] < previous[1]:
                    best_by_case[case_index] = (prediction, row["accepted_score"])
        if case_index not in best_by_case:
            violating_case_indices.add(case_index)

    group_rows = []
    effective_by_group = Counter()
    group_aggregate_fail = Counter()
    improvement_masked = Counter()
    selected_motion = None
    selected_score = math.inf
    for group_index, group_name in enumerate(m.REFINER_GROUP_LABELS):
        case_indices = m.torch.nonzero(
            batch["group"] == group_index, as_tuple=False
        ).flatten().tolist()
        passing = [index for index in case_indices if index in best_by_case]
        combined = baseline.clone()
        for index in passing:
            combined[index] = best_by_case[index][0][index]
        (
            _,
            guard_passed,
            blockers,
            _,
            delta,
            scientific,
        ) = group_probe._audit_prediction(
            model,
            batch,
            cfg,
            combined,
            identity,
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
        endpoint_key = f"{group_name}.observable_endpoint_0p03"
        temporal_key = f"{group_name}.observable_temporal_0p03"
        group_strict = bool(
            delta[endpoint_key] < -STRICT_DESCENT_FLOOR
            or delta[temporal_key] < -STRICT_DESCENT_FLOOR
        )
        raw_group_passed = bool(
            passing
            and guard_passed
            and scientific["raw_exact_common_descent"]
            and group_strict
        )
        case_masked = bool(
            case_indices
            and len(passing) == len(case_indices)
            and not group_strict
        )
        if passing and not raw_group_passed:
            group_aggregate_fail.update([group_name])
        if case_masked:
            improvement_masked.update([group_name])

        projection_trials = []
        projector_report = None
        accepted_factor = None
        accepted_prediction = None
        if raw_group_passed:
            full_projected, projector_report = (
                projected_probe.weighted_dls_contact_project_torch(
                    batch["bad"],
                    baseline,
                    combined,
                    batch["seam"],
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
                        "batch": batch,
                        "identity": identity,
                    },
                )
            )
            for factor in closure_probe.PROJECTION_FACTORS:
                projected = closure_probe._interpolate_projector_correction(
                    combined, full_projected, factor
                )
                (
                    _,
                    projected_passed,
                    projected_blockers,
                    _,
                    projected_delta,
                    projected_scientific,
                ) = group_probe._audit_prediction(
                    model,
                    batch,
                    cfg,
                    projected,
                    identity,
                    baseline_guard,
                    guard_anchor,
                    guard_relative,
                    guard_absolute,
                )
                accepted = bool(
                    projected_passed
                    and projected_scientific["raw_exact_common_descent"]
                    and projector_report["scope_safe"]
                )
                projection_trials.append({
                    "factor": factor,
                    "fixed_exact_guard_passed": projected_passed,
                    "guard_blockers": projected_blockers,
                    "observable_residual_delta": projected_delta,
                    "projected_scientific_nonregression": (
                        projected_scientific[
                            "all_nonpositive_within_numeric_tolerance"
                        ]
                    ),
                    "accepted": accepted,
                })
                if accepted:
                    accepted_factor = factor
                    accepted_prediction = projected
                    effective_by_group.update([group_name])
                    score = float(sum(projected_delta.values()))
                    if score < selected_score:
                        selected_score = score
                        selected_motion = projected.detach().cpu().numpy()
                    break
        group_rows.append({
            "group": group_name,
            "case_indices": case_indices,
            "case_local_pass_indices": passing,
            "all_cases_passed": bool(len(passing) == len(case_indices)),
            "raw_group_fixed_guard_passed": guard_passed,
            "raw_group_guard_blockers": blockers,
            "raw_group_observable_delta": delta,
            "raw_group_scientific_status": scientific,
            "raw_group_candidate_passed": raw_group_passed,
            "case_local_pass_group_aggregate_fail": bool(
                passing and not raw_group_passed
            ),
            "case_improvement_masked_by_group_metric": case_masked,
            "projector": projector_report,
            "projection_trials": projection_trials,
            "projection_backtracking_factor": accepted_factor,
            "effective_projected_candidate": bool(
                accepted_prediction is not None
            ),
        })

    if selected_motion is not None:
        np.save(destination / "selected_projected_candidate.npy", selected_motion)
    outside_max = max(
        (
            float(scope["outside_case_group_or_ownership_abs_max"])
            for row in case_rows
            for scope in (
                [row["action_scope"]]
                if isinstance(row.get("action_scope"), dict)
                and "outside_case_group_or_ownership_abs_max"
                in row["action_scope"]
                else list((row.get("action_scope") or {}).values())
            )
        ),
        default=0.0,
    )
    cross_short = int(effective_by_group["cross_short"])
    cross_long = int(effective_by_group["cross_long"])
    linear_case_indices = {
        int(row["case_index"])
        for row in case_rows
        if row.get("case_local_linear_feasible")
    }
    linear_case_by_group = Counter(
        m.REFINER_GROUP_LABELS[
            int(batch["group"][case_index].detach())
        ]
        for case_index in linear_case_indices
    )
    exact_case_pass_by_group = Counter(
        m.REFINER_GROUP_LABELS[
            int(batch["group"][case_index].detach())
        ]
        for case_index in best_by_case
    )
    if cross_short > 0 and cross_long > 0:
        status = "case_local_finite_radius_projected_descent_exists"
    elif any(exact_case_pass_by_group.values()):
        status = "case_local_exact_pass_but_group_or_projector_failed"
    else:
        status = "no_case_local_exact_closure_descent_at_required_radius"
    expected_trials = int(baseline.shape[0]) * len(SCIENCE_SEEDS)
    witness_frames_by_case = {
        str(case_index): sorted({
            int(frame)
            for row in case_rows
            if int(row["case_index"]) == case_index
            for frame in row.get("active_witness_frames", [])
        })
        for case_index in range(int(baseline.shape[0]))
    }
    witness_joints_by_case = {
        str(case_index): sorted({
            int(joint)
            for row in case_rows
            if int(row["case_index"]) == case_index
            for joint in row.get("active_witness_joints", [])
        })
        for case_index in range(int(baseline.shape[0]))
    }
    predicted_vs_actual = [
        {
            "case_index": int(row["case_index"]),
            "group": row["group"],
            "scientific_weights": row["scientific_weights"],
            "iteration": int(iteration["iteration"]),
            "values": iteration["predicted_vs_actual_constraint_delta"],
        }
        for row in case_rows
        for iteration in row.get("iterations", [])
        if "predicted_vs_actual_constraint_delta" in iteration
    ]
    curvature_ratios = [
        {
            "case_index": int(row["case_index"]),
            "group": row["group"],
            "scientific_weights": row["scientific_weights"],
            "iteration": int(iteration["iteration"]),
            "values": iteration["constraint_curvature_ratio"],
        }
        for row in case_rows
        for iteration in row.get("iterations", [])
        if "constraint_curvature_ratio" in iteration
    ]
    scope_safe = bool(outside_max == 0.0)
    numeric_audit_complete = bool(len(case_rows) == expected_trials)
    result = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "development_only": True,
        "training_started": False,
        "formal_checkpoint": False,
        "publish_allowed": False,
        "source_diagnostic": str(source.resolve()),
        "source_schema": source_report.get("schema"),
        "transaction_context_indices": list(schedule),
        "baseline_fixed_exact_guard_passed": baseline_passed,
        "baseline_guard_blockers": baseline_blockers,
        "target_output_tangent_rms": float(args.target_rms),
        "minimum_required_output_tangent_rms": DEFAULT_TARGET_RMS,
        "diagnostic_fd_epsilon_output_tangent_rms": float(args.fd_epsilon),
        "diagnostic_fd_scales_eligible_as_learning_steps": False,
        "max_sequential_iterations": int(args.max_iterations),
        "epsilon_active_fraction": ACTIVE_FRACTION,
        "topk_witnesses": int(args.topk_witnesses),
        "case_trial_count_expected": expected_trials,
        "case_trial_count": len(case_rows),
        "case_local_linear_feasible": bool(
            linear_case_indices
        ),
        "case_local_linear_feasible_by_group": dict(
            linear_case_by_group
        ),
        "case_local_linear_feasible_trials_by_group": dict(
            linear_feasible_by_group
        ),
        "case_local_exact_finite_radius_passed": bool(best_by_case),
        "case_local_exact_finite_radius_passed_by_group": dict(
            exact_case_pass_by_group
        ),
        "case_local_exact_finite_radius_passed_trials_by_group": dict(
            exact_trial_pass_by_group
        ),
        "violating_case_indices": sorted(violating_case_indices),
        "case_trial_reason_counts": dict(reason_counts),
        "active_witness_frames": witness_frames_by_case,
        "active_witness_joints": witness_joints_by_case,
        "predicted_vs_actual_constraint_delta": predicted_vs_actual,
        "constraint_curvature_ratio": curvature_ratios,
        "case_local_pass_group_aggregate_fail": dict(group_aggregate_fail),
        "case_improvement_masked_by_group_metric": dict(improvement_masked),
        "effective_projected_candidate_by_group": dict(effective_by_group),
        "cross_short_effective_projected_candidate_count": cross_short,
        "cross_long_effective_projected_candidate_count": cross_long,
        "outside_case_group_or_ownership_abs_max": outside_max,
        "scope_safe": scope_safe,
        "numeric_audit_complete": numeric_audit_complete,
        "routing_architecture_pivot_supported": bool(
            cross_short > 0
            and cross_long > 0
            and scope_safe
            and numeric_audit_complete
        ),
        "feasibility_status": status,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "case_trials": case_rows,
        "group_composition_audits": group_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = destination / "case_local_finite_radius_cone.report.json"
    m.save_json(result, report)
    print(json.dumps({
        "stage": "refiner_v15_14g_case_local_finite_radius_cone_complete",
        "report": str(report.resolve()),
        "case_local_linear_feasible_by_group": dict(
            linear_case_by_group
        ),
        "case_local_exact_finite_radius_passed_by_group": dict(
            exact_case_pass_by_group
        ),
        "effective_projected_candidate_by_group": dict(effective_by_group),
        "case_local_pass_group_aggregate_fail": dict(group_aggregate_fail),
        "case_improvement_masked_by_group_metric": dict(improvement_masked),
        "feasibility_status": status,
        "scope_safe": result["scope_safe"],
        "numeric_audit_complete": result["numeric_audit_complete"],
    }), flush=True)
    return 0 if cross_short > 0 and cross_long > 0 else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-diagnostic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--fd-epsilon", type=float, default=DEFAULT_FD_EPSILON)
    parser.add_argument("--target-rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    parser.add_argument(
        "--topk-witnesses", type=int, default=DEFAULT_TOPK_WITNESSES
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
    if args.target_rms < DEFAULT_TARGET_RMS:
        parser.error(f"--target-rms must be >= {DEFAULT_TARGET_RMS:g}")
    if args.max_iterations < 1 or args.max_iterations > 6:
        parser.error("--max-iterations must be in [1, 6]")
    if args.topk_witnesses < 1:
        parser.error("--topk-witnesses must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
