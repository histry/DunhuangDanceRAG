"""Development-only V15.14 projected-candidate feasibility probe.

This module does not train, promote, or publish a Refiner.  It recovers the
unconstrained eight-subgroup MGDA direction from a completed diagnostic,
constructs finite output-amplitude candidates, projects static-support feet
with a weighted damped-least-squares IK solve, and then evaluates the same
immutable fixed-bank Guard used by the diagnostic.
"""
from __future__ import annotations

import argparse
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


SCHEMA = "refiner_v15_14_weighted_dls_projected_candidate_probe_v1"
PROJECTOR_PROTOCOL = "ownership_c2_stiffness_weighted_dls_solve_v1"
DEFAULT_TARGET_RMS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)
IK_JOINTS = (0, 1, 2, 4, 5, 7, 8)


def _to_device_tree(value, device):
    if m.torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device_tree(item, device) for key, item in value.items()}
    return value


def _materialize_first_transaction(artifact, device):
    anchor = _to_device_tree(artifact["anchor"], device)
    schedule = artifact["transaction_schedule"]
    if not schedule:
        raise RuntimeError("fit-bank artifact contains no transaction schedule")
    selected = tuple(int(value) for value in schedule[0])
    if len(selected) != diagnostic.FIT_CONTEXT_COUNT:
        raise RuntimeError("first fit-bank transaction is not rotating-C5")
    batch = anchor
    for index in selected:
        context = _to_device_tree(
            artifact["context_reservoir"][str(index)],
            device,
        )
        batch = diagnostic._concat_fit_batches(batch, context)
    return anchor, batch, selected


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
    return static.detach(), anchors, feet.detach()


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


def weighted_dls_contact_project_torch(
    reference,
    candidate,
    seam,
    cfg,
    *,
    iterations=6,
    damping=1.0e-4,
    jacobian_epsilon=1.0e-4,
    stiffness_ceiling=1.0e4,
):
    """Project support anchors with ownership/C2 stiffness inside the solve.

    The taper is never multiplied into a solved joint update.  Frozen frames
    have no active residual rows, while the smooth ownership activity enters
    the diagonal DLS stiffness.  ``torch.linalg.solve`` is retained so this
    operator can later be unrolled without replacing its linear algebra.
    """
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("reference and candidate must share [B,T,151]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not damping > 0.0 or not jacobian_epsilon > 0.0:
        raise ValueError("DLS damping and Jacobian epsilon must be positive")
    if stiffness_ceiling < 1.0:
        raise ValueError("stiffness ceiling must be at least one")

    static, anchors, _ = _reference_static_support(reference, cfg)
    taper_frames = max(
        1,
        int(getattr(cfg, "product_refiner_residual_taper_frames", 3)),
    )
    owned, activity = _ownership_c2_activity(seam, taper_frames)
    active = static & owned[..., None]
    stiffness = 1.0 + (float(stiffness_ceiling) - 1.0) * (
        1.0 - activity
    ).square()
    variable_count = 3 + 3 * len(IK_JOINTS)
    stiffness = stiffness[..., None].expand(-1, -1, variable_count)
    motion = candidate
    iteration_residuals = []

    for iteration in range(int(iterations)):
        feet = m.fk_24_torch(motion)[..., list(m.DEFAULT_FOOT_JOINTS), :]
        error = anchors - feet
        summary = _masked_residual_summary(error, active)
        summary["iteration"] = iteration
        iteration_residuals.append(summary)

        jacobian = _foot_jacobian(motion, jacobian_epsilon)
        row_mask = active[..., None].expand(-1, -1, -1, 3).flatten(-2)
        solve_dtype = m.torch.float64
        weighted_jacobian = jacobian.to(solve_dtype) * row_mask[..., None]
        weighted_error = error.flatten(-2).to(solve_dtype) * row_mask
        transpose = weighted_jacobian.transpose(-2, -1)
        matrix = transpose @ weighted_jacobian
        matrix = matrix + m.torch.diag_embed(
            float(damping) * stiffness.to(solve_dtype)
        )
        rhs = (transpose @ weighted_error[..., None]).squeeze(-1)
        solution = m.torch.linalg.solve(matrix, rhs).to(motion.dtype)
        tangent = _ik_tangent_from_solution(solution, motion)
        motion = product_exp_torch(motion, tangent)

    feet = m.fk_24_torch(motion)[..., list(m.DEFAULT_FOOT_JOINTS), :]
    error = anchors - feet
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
        "taper_injected_in_normal_equation": True,
        "post_solve_taper_multiplication": False,
        "static_support_samples": int(active.sum().detach()),
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

    targets = tuple(float(value) for value in args.target_rms.split(","))
    rows = []
    blocker_counts = Counter()
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
        projected, ik = weighted_dls_contact_project_torch(
            anchor_batch["bad"],
            raw,
            anchor_batch["seam"],
            cfg,
            iterations=args.ik_iterations,
            damping=args.damping,
            jacobian_epsilon=args.jacobian_epsilon,
            stiffness_ceiling=args.stiffness_ceiling,
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
            guard_passed
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
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
