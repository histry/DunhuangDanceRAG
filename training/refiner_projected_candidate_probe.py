"""Development-only V15.14 projected-candidate feasibility probe.

This module does not train, promote, or publish a Refiner.  It recovers the
unconstrained eight-subgroup MGDA direction from a completed diagnostic,
constructs finite output-amplitude candidates, projects static-support feet
with a weighted damped-least-squares IK solve, and then evaluates the same
immutable fixed-bank Guard used by the diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from motion_geometry.product_manifold import product_exp_torch, product_log_torch
from motion_geometry.physical import ContactStateThresholds
from training import motion_models as m
from training import refiner_bridge_diagnostics as diagnostic


SCHEMA = "refiner_v15_14b_identity_active_set_projected_candidate_probe_v1"
PROJECTOR_PROTOCOL = (
    "identity_preserving_active_set_c2_weighted_temporal_dls_solve_v1"
)
DEFAULT_TARGET_RMS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)
IK_JOINTS = (0, 1, 2, 4, 5, 7, 8)


def _to_device_tree(value, device):
    if m.torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device_tree(item, device) for key, item in value.items()}
    return value


def _transaction_identity(transaction_index, selected_context_indices):
    selected = tuple(int(value) for value in selected_context_indices)
    payload = json.dumps(
        {
            "transaction_index": int(transaction_index),
            "context_indices": list(selected),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"txn_{int(transaction_index):04d}_{digest}"


def _materialize_transaction(artifact, device, transaction_index):
    anchor = _to_device_tree(artifact["anchor"], device)
    schedule = artifact["transaction_schedule"]
    if not schedule:
        raise RuntimeError("fit-bank artifact contains no transaction schedule")
    index = int(transaction_index)
    if not 0 <= index < len(schedule):
        raise IndexError(
            f"transaction index {index} outside [0, {len(schedule)})"
        )
    selected = tuple(int(value) for value in schedule[index])
    if len(selected) != diagnostic.FIT_CONTEXT_COUNT:
        raise RuntimeError("fit-bank transaction is not rotating-C5")
    batch = anchor
    for index in selected:
        context = _to_device_tree(
            artifact["context_reservoir"][str(index)],
            device,
        )
        batch = diagnostic._concat_fit_batches(batch, context)
    return anchor, batch, selected


def _materialize_first_transaction(artifact, device):
    return _materialize_transaction(artifact, device, 0)


def _ownership_c2_activity(seam, taper_frames):
    """Return hard ownership plus a C2 activity used inside DLS stiffness."""
    if seam.ndim != 3 or seam.shape[-1] != 1:
        raise ValueError("seam must have shape [B,T,1]")
    owned = seam[..., 0] >= 0.5
    activity = m.torch.zeros_like(seam[..., 0])
    radius = max(1, int(taper_frames))
    for batch_index in range(owned.shape[0]):
        indices = m.torch.nonzero(owned[batch_index], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        left = int(indices[0].detach())
        right = int(indices[-1].detach())
        frames = m.torch.arange(
            left,
            right + 1,
            device=seam.device,
            dtype=seam.dtype,
        )
        edge_distance = m.torch.minimum(frames - left, right - frames)
        x = (edge_distance / float(radius)).clamp(0.0, 1.0)
        c2 = x.pow(3) * (10.0 - 15.0 * x + 6.0 * x.pow(2))
        activity[batch_index, left:right + 1] = c2
    return owned, activity


def _reference_static_support(reference, cfg):
    joints = m.fk_24_torch(reference)
    feet = joints[..., list(m.DEFAULT_FOOT_JOINTS), :]
    xz_speed = m.torch.zeros(
        feet.shape[:-1],
        dtype=feet.dtype,
        device=feet.device,
    )
    if feet.shape[1] > 1:
        xz_speed[:, 1:] = m.torch.linalg.vector_norm(
            m.torch.diff(feet[..., (0, 2)], dim=1),
            dim=-1,
        ) * float(cfg.fps)
    floor = m.torch.quantile(feet[..., 1].flatten(1), 0.05, dim=1)
    height = feet[..., 1] <= floor[:, None, None] + 0.055
    median_frames = max(1, int(round(float(cfg.fps) / 12.0)))
    if median_frames % 2 == 0:
        median_frames += 1
    height = m._median_bool_filter_torch(height, median_frames)
    declared = reference[..., m.CONTACT] > 0.5
    speed_limit = ContactStateThresholds.from_environment().static_support_speed_mps
    static = (height | declared) & (xz_speed <= float(speed_limit))

    frame_index = m.torch.arange(
        static.shape[1], device=static.device, dtype=m.torch.long
    )[None, :, None].expand_as(static)
    starts = static & ~m.F.pad(
        static[:, :-1], (0, 0, 1, 0), value=False
    )
    anchor_index = m.torch.where(
        starts, frame_index, m.torch.zeros_like(frame_index)
    ).cummax(1).values
    anchors = feet.gather(
        1,
        anchor_index[..., None].expand(-1, -1, -1, 3),
    ).detach()
    return (
        static.detach(),
        anchors,
        feet.detach(),
        anchor_index.detach(),
        floor.detach(),
    )


def _foot_speed(feet, fps):
    speed = m.torch.zeros(
        feet.shape[:-1],
        dtype=feet.dtype,
        device=feet.device,
    )
    if feet.shape[1] > 1:
        speed[:, 1:] = m.torch.linalg.vector_norm(
            m.torch.diff(feet[..., (0, 2)], dim=1),
            dim=-1,
        ) * float(fps)
    return speed


def _support_drift(feet, anchor_index):
    xz = feet[..., (0, 2)]
    anchor = xz.gather(
        1,
        anchor_index[..., None].expand(-1, -1, -1, 2),
    )
    return m.torch.linalg.vector_norm(xz - anchor, dim=-1)


def _active_constraint_masks(
    baseline_feet,
    candidate_feet,
    static,
    anchor_index,
    floor,
    cfg,
    *,
    numeric_tolerance=1.0e-8,
):
    baseline_speed = _foot_speed(baseline_feet, cfg.fps)
    candidate_speed = _foot_speed(candidate_feet, cfg.fps)
    baseline_drift = _support_drift(baseline_feet, anchor_index)
    candidate_drift = _support_drift(candidate_feet, anchor_index)
    baseline_height = baseline_feet[..., 1] - floor[:, None, None]
    candidate_height = candidate_feet[..., 1] - floor[:, None, None]
    tolerance = float(numeric_tolerance)
    masks = {
        "foot_skate": static & (
            candidate_speed > baseline_speed + tolerance
        ),
        "support_drift": static & (
            candidate_drift > baseline_drift + tolerance
        ),
        "penetration": candidate_height < baseline_height - tolerance,
    }
    return masks, {
        "baseline_foot_skate_mps": baseline_speed,
        "candidate_foot_skate_mps": candidate_speed,
        "baseline_support_drift_m": baseline_drift,
        "candidate_support_drift_m": candidate_drift,
        "baseline_penetration_height_m": baseline_height,
        "candidate_penetration_height_m": candidate_height,
    }


def _ik_tangent_from_solution(solution, reference):
    tangent = reference.new_zeros(reference.shape[:-1] + (75,))
    tangent[..., :3] = solution[..., :3]
    offset = 3
    for joint_id in IK_JOINTS:
        tangent[..., 3 + 3 * joint_id:3 + 3 * (joint_id + 1)] = (
            solution[..., offset:offset + 3]
        )
        offset += 3
    return tangent


def _foot_jacobian(motion, epsilon):
    columns = []
    variable_count = 3 + 3 * len(IK_JOINTS)
    for column in range(variable_count):
        positive = motion.new_zeros(motion.shape[:-1] + (75,))
        negative = motion.new_zeros(motion.shape[:-1] + (75,))
        if column < 3:
            tangent_index = column
        else:
            local = column - 3
            joint_id = IK_JOINTS[local // 3]
            tangent_index = 3 + 3 * joint_id + local % 3
        positive[..., tangent_index] = float(epsilon)
        negative[..., tangent_index] = -float(epsilon)
        plus = m.fk_24_torch(product_exp_torch(motion, positive))
        minus = m.fk_24_torch(product_exp_torch(motion, negative))
        plus = plus[..., list(m.DEFAULT_FOOT_JOINTS), :]
        minus = minus[..., list(m.DEFAULT_FOOT_JOINTS), :]
        columns.append(((plus - minus) / (2.0 * float(epsilon))).flatten(-2))
    return m.torch.stack(columns, dim=-1)


def _masked_residual_summary(error, active):
    norm = m.torch.linalg.vector_norm(error, dim=-1)
    values = norm[active]
    if values.numel() == 0:
        return {"rms_m": 0.0, "max_m": 0.0, "samples": 0}
    return {
        "rms_m": float(m.torch.sqrt(m.torch.mean(values.square())).detach()),
        "max_m": float(values.max().detach()),
        "samples": int(values.numel()),
    }


def _scientific_motion_tangent_gradients(
    model,
    batch,
    cfg,
    motion,
    identity,
):
    """Differentiate the fixed-bank observable Guard in motion tangent space."""
    with m.torch.enable_grad():
        zero = m.torch.zeros(
            motion.shape[:-1] + (75,),
            dtype=motion.dtype,
            device=motion.device,
            requires_grad=True,
        )
        probe = product_exp_torch(motion.detach(), zero)
        values = _guard_values_for_prediction(
            model,
            batch,
            cfg,
            probe,
            identity.detach(),
        )
        selected = {
            key: value
            for key, value in values.items()
            if ".observable_" in key
        }
        gradients = {}
        for index, (key, value) in enumerate(selected.items()):
            gradient = m.torch.autograd.grad(
                value,
                zero,
                retain_graph=index + 1 < len(selected),
                allow_unused=False,
            )[0]
            gradients[key] = gradient.detach()
    return gradients


def _project_scientific_nonregression(
    tangent,
    gradients,
    owned,
    *,
    max_passes=16,
):
    """Project an IK tangent into observable non-regression halfspaces."""
    allowed = m.torch.zeros_like(tangent)
    allowed[..., :3] = owned[..., None]
    for joint_id in IK_JOINTS:
        start = 3 + 3 * joint_id
        allowed[..., start:start + 3] = owned[..., None]
    normals = {
        key: gradient * allowed
        for key, gradient in gradients.items()
    }
    projected = tangent
    before = {
        key: float((normal.double() * projected.double()).sum().detach())
        for key, normal in normals.items()
    }
    passes = 0
    for passes in range(1, int(max_passes) + 1):
        changed = False
        for normal in normals.values():
            dot = (normal.double() * projected.double()).sum()
            norm_square = normal.double().square().sum().clamp_min(1.0e-24)
            tolerance = 1.0e-10 * max(1.0, abs(float(dot.detach())))
            if float(dot.detach()) > tolerance:
                projected = projected - (dot / norm_square).to(
                    projected.dtype
                ) * normal
                changed = True
        if not changed:
            break
    after = {
        key: float((normal.double() * projected.double()).sum().detach())
        for key, normal in normals.items()
    }
    scale = max(1.0, *(abs(value) for value in after.values()))
    tolerance = 1.0e-10 * scale
    return projected, {
        "enabled": True,
        "protocol": "fixed_bank_observable_tangent_halfspace_projection_v1",
        "active_constraints": list(normals),
        "active_constraint_count": len(normals),
        "projection_passes": int(passes),
        "directional_derivatives_before": before,
        "directional_derivatives_after": after,
        "all_directional_derivatives_nonpositive": all(
            value <= tolerance for value in after.values()
        ),
        "derivative_tolerance": tolerance,
    }


def weighted_dls_contact_project_torch(
    reference,
    baseline,
    candidate,
    seam,
    cfg,
    *,
    iterations=6,
    damping=1.0e-4,
    jacobian_epsilon=1.0e-4,
    stiffness_ceiling=1.0e4,
    acceleration_regularization=1.0e-2,
    jerk_regularization=1.0e-3,
    scientific_context=None,
):
    """Project only candidate-induced constraint regressions.

    The baseline is an exact fixed point: if a candidate introduces no new
    skate, support-drift or penetration regression, the original tensor is
    returned without retraction.  Active residual rows target the baseline
    foot trajectory rather than a segment-start world anchor. Ownership/C2
    stiffness and D2/D3 temporal regularizers are part of the coupled normal
    equation; no solved update is tapered afterward.
    """
    if (
        reference.shape != baseline.shape
        or reference.shape != candidate.shape
        or reference.ndim != 3
    ):
        raise ValueError("reference, baseline and candidate must share [B,T,151]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not damping > 0.0 or not jacobian_epsilon > 0.0:
        raise ValueError("DLS damping and Jacobian epsilon must be positive")
    if stiffness_ceiling < 1.0:
        raise ValueError("stiffness ceiling must be at least one")
    if acceleration_regularization < 0.0 or jerk_regularization < 0.0:
        raise ValueError("temporal regularization weights must be non-negative")

    static, _, _, anchor_index, floor = _reference_static_support(reference, cfg)
    baseline_feet = m.fk_24_torch(baseline)[
        ..., list(m.DEFAULT_FOOT_JOINTS), :
    ].detach()
    taper_frames = max(
        1,
        int(getattr(cfg, "product_refiner_residual_taper_frames", 3)),
    )
    owned, activity = _ownership_c2_activity(seam, taper_frames)
    stiffness = 1.0 + (float(stiffness_ceiling) - 1.0) * (
        1.0 - activity
    ).square()
    variable_count = 3 + 3 * len(IK_JOINTS)
    stiffness = stiffness[..., None].expand(-1, -1, variable_count)
    motion = candidate
    iteration_residuals = []
    initial_constraint_counts = None
    scientific_projection_history = []

    def difference_matrix(width, order, dtype, device):
        if width <= order:
            return m.torch.zeros((0, width), dtype=dtype, device=device)
        eye = m.torch.eye(width, dtype=dtype, device=device)
        return m.torch.diff(eye, n=order, dim=0)

    def coupled_solve(frame_matrix, frame_rhs, owned_mask):
        solution = frame_rhs.new_zeros(frame_rhs.shape)
        solve_dtype = m.torch.float64
        identity = m.torch.eye(
            variable_count,
            dtype=solve_dtype,
            device=frame_rhs.device,
        )
        for batch_index in range(frame_rhs.shape[0]):
            indices = m.torch.nonzero(
                owned_mask[batch_index], as_tuple=False
            ).flatten()
            if indices.numel() == 0:
                continue
            left = int(indices[0].detach())
            right = int(indices[-1].detach()) + 1
            width = right - left
            blocks = [
                frame_matrix[batch_index, frame].to(solve_dtype)
                for frame in range(left, right)
            ]
            matrix = m.torch.block_diag(*blocks)
            d2 = difference_matrix(width, 2, solve_dtype, frame_rhs.device)
            d3 = difference_matrix(width, 3, solve_dtype, frame_rhs.device)
            if d2.numel():
                temporal = (d2.transpose(0, 1) @ d2).contiguous()
                matrix = matrix + float(acceleration_regularization) * (
                    m.torch.kron(temporal, identity)
                )
            if d3.numel():
                temporal = (d3.transpose(0, 1) @ d3).contiguous()
                matrix = matrix + float(jerk_regularization) * (
                    m.torch.kron(temporal, identity)
                )
            rhs = frame_rhs[batch_index, left:right].to(solve_dtype).reshape(-1)
            solved = m.torch.linalg.solve(matrix, rhs).to(frame_rhs.dtype)
            solution[batch_index, left:right] = solved.reshape(
                width, variable_count
            )
        return solution

    for iteration in range(int(iterations)):
        feet = m.fk_24_torch(motion)[..., list(m.DEFAULT_FOOT_JOINTS), :]
        masks, _ = _active_constraint_masks(
            baseline_feet,
            feet,
            static,
            anchor_index,
            floor,
            cfg,
        )
        masks = {
            name: value & owned[..., None]
            for name, value in masks.items()
        }
        active = m.torch.stack(list(masks.values()), dim=0).any(dim=0)
        counts = {
            name: int(value.sum().detach())
            for name, value in masks.items()
        }
        if initial_constraint_counts is None:
            initial_constraint_counts = counts
        error = baseline_feet - feet
        summary = _masked_residual_summary(error, active)
        summary["iteration"] = iteration
        summary["active_constraint_counts"] = counts
        iteration_residuals.append(summary)

        if not bool(active.any()):
            break

        jacobian = _foot_jacobian(motion, jacobian_epsilon)
        horizontal = masks["foot_skate"] | masks["support_drift"]
        coordinate_active = m.torch.stack(
            [horizontal, masks["penetration"], horizontal],
            dim=-1,
        )
        row_mask = coordinate_active.flatten(-2)
        solve_dtype = m.torch.float64
        weighted_jacobian = jacobian.to(solve_dtype) * row_mask[..., None]
        weighted_error = error.flatten(-2).to(solve_dtype) * row_mask
        transpose = weighted_jacobian.transpose(-2, -1)
        matrix = transpose @ weighted_jacobian
        matrix = matrix + m.torch.diag_embed(
            float(damping) * stiffness.to(solve_dtype)
        )
        rhs = (transpose @ weighted_error[..., None]).squeeze(-1)
        case_active = active.any(dim=(1, 2))
        solution = coupled_solve(matrix, rhs, owned & case_active[:, None])
        tangent = _ik_tangent_from_solution(solution, motion)
        if scientific_context is not None:
            gradients = _scientific_motion_tangent_gradients(
                scientific_context["model"],
                scientific_context["batch"],
                cfg,
                motion,
                scientific_context["identity"],
            )
            tangent, scientific_projection = (
                _project_scientific_nonregression(
                    tangent,
                    gradients,
                    owned & case_active[:, None],
                )
            )
            scientific_projection["iteration"] = iteration
            scientific_projection_history.append(scientific_projection)
        updated = product_exp_torch(motion, tangent)
        motion = m.torch.where(
            case_active[:, None, None],
            updated,
            motion,
        ).detach()

    feet = m.fk_24_torch(motion)[..., list(m.DEFAULT_FOOT_JOINTS), :]
    final_masks, _ = _active_constraint_masks(
        baseline_feet,
        feet,
        static,
        anchor_index,
        floor,
        cfg,
    )
    final_masks = {
        name: value & owned[..., None]
        for name, value in final_masks.items()
    }
    active = m.torch.stack(list(final_masks.values()), dim=0).any(dim=0)
    error = baseline_feet - feet
    final_summary = _masked_residual_summary(error, active)
    final_summary["iterations"] = int(iterations)
    by_foot = {}
    for index, joint_id in enumerate(m.DEFAULT_FOOT_JOINTS):
        by_foot[str(int(joint_id))] = _masked_residual_summary(
            error[..., index, :], active[..., index]
        )
    by_side = {
        "left_foot": _masked_residual_summary(
            error[..., (0, 2), :], active[..., (0, 2)]
        ),
        "right_foot": _masked_residual_summary(
            error[..., (1, 3), :], active[..., (1, 3)]
        ),
    }
    outside = ~owned[..., None]
    outside_delta = (motion - candidate).abs() * outside
    scope_max = float(outside_delta.max().detach())
    return motion, {
        "protocol": PROJECTOR_PROTOCOL,
        "linear_solver": "torch.linalg.solve",
        "pseudoinverse_used": False,
        "inverse_used": False,
        "iterations": int(iterations),
        "damping": float(damping),
        "jacobian_epsilon": float(jacobian_epsilon),
        "stiffness_ceiling": float(stiffness_ceiling),
        "acceleration_regularization": float(acceleration_regularization),
        "jerk_regularization": float(jerk_regularization),
        "taper_injected_in_normal_equation": True,
        "temporal_regularization_in_normal_equation": True,
        "post_solve_taper_multiplication": False,
        "scientific_nonregression_projection_enabled": (
            scientific_context is not None
        ),
        "scientific_nonregression_projection_history": (
            scientific_projection_history
        ),
        "identity_preserving_target": "baseline_candidate_induced_violation",
        "support_mask_source": "observed_reference_only",
        "static_support_samples": int(static.sum().detach()),
        "initial_active_constraint_counts": initial_constraint_counts or {
            "foot_skate": 0,
            "support_drift": 0,
            "penetration": 0,
        },
        "final_active_constraint_counts": {
            name: int(value.sum().detach())
            for name, value in final_masks.items()
        },
        "iteration_residuals": iteration_residuals,
        "ik_residual_after_n_iters": final_summary,
        "ik_residual_after_n_iters_by_side": by_side,
        "ik_residual_after_n_iters_by_foot_joint": by_foot,
        "ownership_outside_change_abs_max": scope_max,
        "scope_safe": bool(scope_max == 0.0),
    }


def _model_prediction(model, batch, cfg):
    with m.torch.no_grad():
        return m._refiner_batch_outputs(model, batch, cfg)


def _motion_edit_rms(reference, candidate, seam):
    tangent = product_log_torch(reference, candidate)
    active = (seam >= 0.5).expand_as(tangent)
    values = tangent[active]
    if values.numel() == 0:
        return 0.0
    return float(m.torch.sqrt(m.torch.mean(values.square())).detach())


def _set_parameter_candidate(parameters, base, direction, scale):
    with m.torch.no_grad():
        for parameter, initial, update in zip(parameters, base, direction):
            parameter.copy_(initial + float(scale) * update)


def _candidate_at_output_rms(
    model,
    batch,
    cfg,
    baseline,
    parameters,
    base,
    direction,
    target,
):
    scale = 1.0
    measured = 0.0
    prediction = baseline
    for _ in range(8):
        _set_parameter_candidate(parameters, base, direction, scale)
        prediction, _ = _model_prediction(model, batch, cfg)
        measured = _motion_edit_rms(baseline, prediction, batch["seam"])
        if not math.isfinite(measured) or measured <= 1.0e-16:
            scale *= 10.0
            continue
        ratio = float(target) / measured
        if abs(math.log(max(ratio, 1.0e-12))) <= 0.03:
            break
        scale *= min(10.0, max(0.1, ratio))
    return prediction.detach(), float(scale), float(measured)


def _guard_values_for_prediction(model, batch, cfg, prediction, identity):
    groups = {}
    _, _, terms, _ = m._refiner_batch_objectives(
        model,
        batch,
        cfg,
        group_objectives=groups,
        prediction_override=prediction,
        identity_override=identity,
    )
    return diagnostic._diagnostic_group_guard_values(terms, groups)


def _float_guard(values):
    return {key: float(value.detach()) for key, value in values.items()}


def _audit_guard_candidate(values, anchor, relative, absolute):
    blockers = []
    details = {}
    for key, candidate in values.items():
        reference = float(anchor[key])
        allowance = max(
            abs(reference) * float(relative[key]),
            float(absolute[key]),
        )
        allowed = reference + allowance
        numeric = max(1.0e-12, abs(allowed) * 1.0e-9, allowance * 1.0e-6)
        passed = float(candidate) <= allowed + numeric
        if not passed:
            blockers.append(key)
        details[key] = {
            "fixed_anchor": reference,
            "candidate": float(candidate),
            "absolute_limit": allowed,
            "remaining_margin": allowed - float(candidate),
            "delta_from_anchor": float(candidate) - reference,
            "numeric_tolerance": numeric,
            "passed": bool(passed),
        }
    return not blockers, blockers, details


def _physical_summary(values):
    return {
        key: {
            "mean": float(value.mean().detach()),
            "max": float(value.max().detach()),
        }
        for key, value in values.items()
        if m.torch.is_tensor(value) and value.ndim == 1
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
        raise RuntimeError("V15.14 accepts only an unpublished failed diagnostic")
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.14 projected-candidate probe requires CUDA")
    state = m.torch.load(state_path, map_location="cpu", weights_only=False)
    artifact = m.torch.load(fit_path, map_location="cpu", weights_only=False)
    if state.get("formal_checkpoint") or artifact.get("formal_checkpoint"):
        raise RuntimeError("formal checkpoint input is forbidden")
    anchor_batch, train_batch, schedule = _materialize_first_transaction(
        artifact, device
    )
    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.train()

    repair, protection, terms, _ = m._refiner_batch_objectives(
        model, train_batch, cfg
    )
    total = repair + float(cfg.product_refiner_clean_identity_weight) * protection
    model.zero_grad(set_to_none=True)
    mgda = diagnostic._pareto_common_descent_backward(model, total, terms, cfg)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    direction = [
        m.torch.zeros_like(parameter)
        if parameter.grad is None
        else -parameter.grad.detach().clone()
        for parameter in parameters
    ]
    base_parameters = [parameter.detach().clone() for parameter in parameters]
    if not mgda.get("common_descent_exists"):
        raise RuntimeError("unconstrained subgroup MGDA has no common descent")

    baseline, identity = _model_prediction(model, anchor_batch, cfg)
    baseline_guard = _float_guard(
        _guard_values_for_prediction(
            model, anchor_batch, cfg, baseline, identity
        )
    )
    contract = source_report["group_guard_contract"]
    guard_anchor = contract["initial_anchor"]
    guard_relative = contract["relative_tolerance"]
    guard_absolute = contract["absolute_tolerance"]
    if set(guard_anchor) != set(baseline_guard):
        raise RuntimeError("source fixed Guard and reconstructed bank differ")
    baseline_guard_passed, baseline_blockers, baseline_guard_details = (
        _audit_guard_candidate(
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
    )

    identity_projected, identity_projector = weighted_dls_contact_project_torch(
        anchor_batch["bad"],
        baseline,
        baseline,
        anchor_batch["seam"],
        cfg,
        iterations=args.ik_iterations,
        damping=args.damping,
        jacobian_epsilon=args.jacobian_epsilon,
        stiffness_ceiling=args.stiffness_ceiling,
        acceleration_regularization=args.acceleration_regularization,
        jerk_regularization=args.jerk_regularization,
    )
    identity_delta_rms = _motion_edit_rms(
        baseline, identity_projected, anchor_batch["seam"]
    )
    identity_delta_abs_max = float(
        (identity_projected - baseline).abs().max().detach()
    )
    identity_control_passed = bool(
        identity_delta_rms <= 1.0e-12
        and identity_delta_abs_max <= 1.0e-12
    )

    targets = tuple(float(value) for value in args.target_rms.split(","))
    rows = []
    blocker_counts = Counter()
    raw_blocker_counts = Counter()
    selected = None
    for target in targets:
        candidate_started = time.perf_counter()
        raw, parameter_scale, achieved = _candidate_at_output_rms(
            model,
            anchor_batch,
            cfg,
            baseline,
            parameters,
            base_parameters,
            direction,
            target,
        )
        raw_physical = m.batch_physical_audit_torch(raw, cfg)
        raw_guard_values = _float_guard(
            _guard_values_for_prediction(
                model, anchor_batch, cfg, raw, identity
            )
        )
        raw_guard_passed, raw_blockers, raw_guard_details = (
            _audit_guard_candidate(
                raw_guard_values,
                guard_anchor,
                guard_relative,
                guard_absolute,
            )
        )
        raw_blocker_counts.update(raw_blockers)
        raw_observable_delta = {
            key: raw_guard_values[key] - baseline_guard[key]
            for key in raw_guard_values
            if ".observable_" in key
        }
        raw_strict_observable_descent = any(
            value < -1.0e-7 for value in raw_observable_delta.values()
        )
        projected, ik = weighted_dls_contact_project_torch(
            anchor_batch["bad"],
            baseline,
            raw,
            anchor_batch["seam"],
            cfg,
            iterations=args.ik_iterations,
            damping=args.damping,
            jacobian_epsilon=args.jacobian_epsilon,
            stiffness_ceiling=args.stiffness_ceiling,
            acceleration_regularization=args.acceleration_regularization,
            jerk_regularization=args.jerk_regularization,
        )
        projected_physical = m.batch_physical_audit_torch(projected, cfg)
        guard_values = _float_guard(
            _guard_values_for_prediction(
                model, anchor_batch, cfg, projected, identity
            )
        )
        guard_passed, blockers, guard_details = _audit_guard_candidate(
            guard_values,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
        blocker_counts.update(blockers)
        observable_delta = {
            key: guard_values[key] - baseline_guard[key]
            for key in guard_values
            if ".observable_" in key
        }
        strict_contact_or_trajectory_descent = any(
            value < -1.0e-7 for value in observable_delta.values()
        )
        effective = bool(
            identity_control_passed
            and raw_strict_observable_descent
            and guard_passed
            and strict_contact_or_trajectory_descent
            and ik["scope_safe"]
        )
        row = {
            "target_output_tangent_rms": target,
            "achieved_output_tangent_rms": achieved,
            "parameter_direction_scale": parameter_scale,
            "projected_edit_tangent_rms": _motion_edit_rms(
                baseline, projected, anchor_batch["seam"]
            ),
            "projector": ik,
            "gpu_batch_audit_before_projection": _physical_summary(raw_physical),
            "gpu_batch_audit_after_projection": _physical_summary(
                projected_physical
            ),
            "raw_fixed_exact_guard_passed": raw_guard_passed,
            "raw_strict_observable_descent": raw_strict_observable_descent,
            "raw_observable_residual_delta": raw_observable_delta,
            "raw_guard_blockers": raw_blockers,
            "raw_guard_metrics": raw_guard_details,
            "fixed_exact_guard_passed": guard_passed,
            "strict_observable_descent": strict_contact_or_trajectory_descent,
            "effective_projected_candidate": effective,
            "observable_residual_delta": observable_delta,
            "guard_blockers": blockers,
            "guard_metrics": guard_details,
            "elapsed_seconds": time.perf_counter() - candidate_started,
        }
        rows.append(row)
        if effective and selected is None:
            selected = len(rows) - 1
            np.save(
                destination / "selected_projected_candidate.npy",
                projected.detach().cpu().numpy(),
            )

    _set_parameter_candidate(
        parameters, base_parameters, direction, 0.0
    )
    result = {
        "schema": SCHEMA,
        "development_only": True,
        "training_started": False,
        "formal_checkpoint": False,
        "publish_allowed": False,
        "source_diagnostic": str(source.resolve()),
        "source_schema": source_report.get("schema"),
        "source_completed_steps": source_report.get("completed_steps"),
        "transaction_context_indices": list(schedule),
        "projector_protocol": PROJECTOR_PROTOCOL,
        "projector_identity_control": {
            "passed": identity_control_passed,
            "edit_tangent_rms": identity_delta_rms,
            "state_abs_max": identity_delta_abs_max,
            "projector": identity_projector,
            "failure_reason": (
                None
                if identity_control_passed
                else "projector_not_identity_at_anchor"
            ),
        },
        "unconstrained_subgroup_mgda": mgda,
        "candidate_count": len(rows),
        "effective_projected_candidate_count": sum(
            int(row["effective_projected_candidate"]) for row in rows
        ),
        "selected_candidate_index": selected,
        "projected_direction_exists": selected is not None,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "guard_blocker_counts": dict(blocker_counts),
        "raw_guard_blocker_counts": dict(raw_blocker_counts),
        "baseline_fixed_exact_guard_passed": baseline_guard_passed,
        "baseline_guard_blockers": baseline_blockers,
        "baseline_guard_metrics": baseline_guard_details,
        "baseline_guard_values": baseline_guard,
        "candidates": rows,
        "scope_safe": all(row["projector"]["scope_safe"] for row in rows),
        "numeric_audit_complete": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    m.save_json(result, destination / "projected_candidate.report.json")
    print(json.dumps({
        "stage": "refiner_v15_14_projected_candidate_probe_complete",
        "report": str((destination / "projected_candidate.report.json").resolve()),
        "candidates": len(rows),
        "effective_projected_candidates": result[
            "effective_projected_candidate_count"
        ],
        "scope_safe": result["scope_safe"],
    }), flush=True)
    return 0 if selected is not None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-diagnostic-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument(
        "--target-rms",
        default=",".join(str(value) for value in DEFAULT_TARGET_RMS),
    )
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
