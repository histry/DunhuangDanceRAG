"""Development-only V15.14c exact-closure-consistent tangent probe.

The probe compares the training-transaction MGDA derivative with central
finite differences of the immutable fixed-bank observable Guard.  A sign
mismatch replaces that direction with an eight-objective MGDA direction built
on the exact same fixed bank and signed residual tensors used by closure.
Only finite raw candidates that pass every unchanged exact Guard component and
give a numerically resolved eight-objective common descent enter the contact
projector.  Projector strength is then backtracked under full exact closure.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from motion_geometry.product_manifold import product_exp_torch, product_log_torch
from training import motion_models as m
from training import refiner_bridge_diagnostics as diagnostic
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_14c_exact_closure_consistent_tangent_probe_v1"
PROTOCOL = "fixed_bank_exact_closure_fd_mgda_projector_backtracking_v1"
DEFAULT_FD_EPSILON = (1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5)
DEFAULT_TARGET_RMS = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3)
PROJECTION_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
STRICT_DELTA_FLOOR = 1.0e-7


def _observable_values(values):
    return {
        key: value
        for key, value in values.items()
        if ".observable_" in key
    }


def _initial_to_guard_name(name):
    group, objective = name.rsplit(".", 1)
    if objective == "endpoint":
        return f"{group}.observable_endpoint_0p03"
    if objective == "temporal":
        return f"{group}.observable_temporal_0p03"
    raise ValueError(f"unknown scientific objective {name!r}")


def _exact_closure_mgda_backward(model, batch, cfg):
    """Build MGDA from exact fixed-bank observable signed residuals."""
    prediction, identity = m._refiner_batch_outputs(model, batch, cfg)
    guard_values = projected_probe._guard_values_for_prediction(
        model,
        batch,
        cfg,
        prediction,
        identity,
    )
    objectives = _observable_values(guard_values)
    expected = 2 * len(m.REFINER_GROUP_LABELS)
    if len(objectives) != expected:
        raise RuntimeError(
            f"exact closure MGDA expected {expected} observables, "
            f"found {len(objectives)}"
        )
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count < 1:
        raise RuntimeError("exact closure MGDA found no trainable parameters")

    names = list(objectives)
    gradients = []
    for objective in objectives.values():
        raw = m.torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients.append([
            m.torch.zeros_like(parameter) if value is None else value
            for parameter, value in zip(parameters, raw)
        ])

    count = prediction.new_tensor(float(parameter_count), dtype=m.torch.float64)
    epsilon = prediction.new_tensor(1.0e-24, dtype=m.torch.float64)
    norm_squares = m.torch.stack([
        diagnostic._tuple_dot(gradient, gradient)
        for gradient in gradients
    ])
    rms_norms = (norm_squares / count).clamp_min(0.0).sqrt()
    active = [
        index for index, value in enumerate(rms_norms)
        if float(value.detach()) > 1.0e-12
    ]
    normalized = [
        [
            value / rms_norms[index].clamp_min(epsilon).to(value.dtype)
            for value in gradients[index]
        ]
        for index in active
    ]
    cosine = []
    for left_index, left in enumerate(gradients):
        row = []
        for right_index, right in enumerate(gradients):
            denominator = (
                norm_squares[left_index].clamp_min(epsilon).sqrt()
                * norm_squares[right_index].clamp_min(epsilon).sqrt()
            )
            value = diagnostic._tuple_dot(left, right) / denominator
            if left_index not in active or right_index not in active:
                value = value.new_zeros(())
            row.append(float(value.detach()))
        cosine.append(row)

    full_weights = prediction.new_zeros(
        (len(names),), dtype=m.torch.float64
    )
    iterations = 0
    duality_gap = 0.0
    if normalized:
        gram = m.torch.stack([
            m.torch.stack([
                diagnostic._tuple_dot(left, right) / count
                for right in normalized
            ])
            for left in normalized
        ])
        weights, iterations, duality_gap = (
            diagnostic._deterministic_mgda_weights(gram)
        )
        for offset, index in enumerate(active):
            full_weights[index] = weights[offset]
        normalized_common = [
            sum(
                weights[index].to(parts[0].dtype) * parts[parameter_index]
                for index, parts in enumerate(normalized)
            )
            for parameter_index in range(len(parameters))
        ]
        minimum_norm = float(
            (
                diagnostic._tuple_dot(normalized_common, normalized_common)
                / count
            ).clamp_min(0.0).sqrt().detach()
        )
        target_rms = m.torch.stack([rms_norms[index] for index in active]).median()
        common = [
            value * target_rms.to(value.dtype)
            for value in normalized_common
        ]
    else:
        minimum_norm = 0.0
        target_rms = prediction.new_zeros((), dtype=m.torch.float64)
        common = [m.torch.zeros_like(parameter) for parameter in parameters]

    directional = {
        name: float(-diagnostic._tuple_dot(common, gradient).detach())
        for name, gradient in zip(names, gradients)
    }
    scale = max(1.0, *(abs(value) for value in directional.values()))
    tolerance = 1.0e-10 * scale
    nonpositive = all(value <= tolerance for value in directional.values())
    one_strict = any(value < -tolerance for value in directional.values())
    common_exists = bool(
        minimum_norm > diagnostic.MGDA_COMMON_DESCENT_RMS_EPSILON
        and nonpositive
        and one_strict
    )
    if not common_exists:
        common = [m.torch.zeros_like(parameter) for parameter in parameters]
        directional = {name: 0.0 for name in names}
    model.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, common):
        parameter.grad = gradient.detach().clone()

    return {
        "protocol": "fixed_bank_exact_observable_signed_residual_mgda_v1",
        "task_names": names,
        "fixed_bank_used": True,
        "guard_value_domain": (
            "exact_stage_signed_residual_or_observable_fidelity_metric"
        ),
        "gradient_cosine_matrix": cosine,
        "gradient_norms_before_normalization": {
            name: float(value.detach())
            for name, value in zip(names, rms_norms)
        },
        "mgda_weights": {
            name: float(value.detach())
            for name, value in zip(names, full_weights)
        },
        "mgda_min_norm": minimum_norm,
        "mgda_iterations": int(iterations),
        "mgda_duality_gap": float(duality_gap),
        "gradient_rms_rescale": float(target_rms.detach()),
        "directional_derivatives": directional,
        "all_directional_derivatives_nonpositive": nonpositive,
        "at_least_one_directional_derivative_strict": one_strict,
        "common_descent_exists": common_exists,
        "reason": (
            "fixed_bank_exact_closure_common_descent"
            if common_exists
            else "pareto_stationary_or_no_exact_closure_common_descent"
        ),
    }


def _parameter_direction(model):
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    direction = [
        m.torch.zeros_like(parameter)
        if parameter.grad is None
        else -parameter.grad.detach().clone()
        for parameter in parameters
    ]
    return parameters, direction


def _central_exact_fd(
    model,
    batch,
    cfg,
    baseline,
    identity,
    baseline_guard,
    parameters,
    base_parameters,
    direction,
    epsilon_values,
):
    rows = []
    for epsilon in epsilon_values:
        plus, plus_scale, plus_achieved = (
            projected_probe._candidate_at_output_rms(
                model,
                batch,
                cfg,
                baseline,
                parameters,
                base_parameters,
                direction,
                epsilon,
            )
        )
        negative_direction = [-value for value in direction]
        minus, minus_scale, minus_achieved = (
            projected_probe._candidate_at_output_rms(
                model,
                batch,
                cfg,
                baseline,
                parameters,
                base_parameters,
                negative_direction,
                epsilon,
            )
        )
        plus_values = projected_probe._float_guard(
            projected_probe._guard_values_for_prediction(
                model, batch, cfg, plus, identity
            )
        )
        minus_values = projected_probe._float_guard(
            projected_probe._guard_values_for_prediction(
                model, batch, cfg, minus, identity
            )
        )
        keys = list(_observable_values(baseline_guard))
        derivative = {
            key: (plus_values[key] - minus_values[key]) / (2.0 * epsilon)
            for key in keys
        }
        achieved_denominator = plus_achieved + minus_achieved
        achieved_derivative = {
            key: (
                (plus_values[key] - minus_values[key]) / achieved_denominator
                if achieved_denominator > 0.0 else None
            )
            for key in keys
        }
        rows.append({
            "epsilon_output_tangent_rms": float(epsilon),
            "diagnostic_only": True,
            "eligible_as_learning_step": False,
            "plus_parameter_scale": plus_scale,
            "minus_parameter_scale": -minus_scale,
            "plus_achieved_output_tangent_rms": plus_achieved,
            "minus_achieved_output_tangent_rms": minus_achieved,
            "central_derivative_nominal_epsilon": derivative,
            "central_derivative_achieved_rms": achieved_derivative,
            "plus_delta": {
                key: plus_values[key] - baseline_guard[key]
                for key in keys
            },
            "minus_delta": {
                key: minus_values[key] - baseline_guard[key]
                for key in keys
            },
        })
    projected_probe._set_parameter_candidate(
        parameters, base_parameters, direction, 0.0
    )
    representative = {
        key: statistics.median([
            row["central_derivative_nominal_epsilon"][key]
            for row in rows
        ])
        for key in _observable_values(baseline_guard)
    }
    return rows, representative


def _sign(value, tolerance):
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _sign_agreement(initial_mgda, exact_fd):
    result = {}
    for name, derivative in initial_mgda["directional_derivatives"].items():
        key = _initial_to_guard_name(name)
        autograd_tolerance = 1.0e-10 * max(1.0, abs(float(derivative)))
        fd_value = float(exact_fd[key])
        fd_tolerance = 1.0e-8 * max(1.0, abs(fd_value))
        autograd_sign = _sign(float(derivative), autograd_tolerance)
        exact_sign = _sign(fd_value, fd_tolerance)
        result[key] = {
            "training_transaction_autograd_directional_derivative": (
                float(derivative)
            ),
            "fixed_bank_exact_fd_directional_derivative": fd_value,
            "autograd_sign": autograd_sign,
            "exact_fd_sign": exact_sign,
            "agrees": bool(autograd_sign == exact_sign),
        }
    return {
        "by_objective": result,
        "all_agree": all(row["agrees"] for row in result.values()),
        "mismatch_count": sum(not row["agrees"] for row in result.values()),
    }


def _scientific_delta_status(delta, guard_details):
    tolerances = {
        key: float(guard_details[key]["numeric_tolerance"])
        for key in delta
    }
    all_nonpositive = all(
        float(value) <= tolerances[key]
        for key, value in delta.items()
    )
    strict = any(
        float(value) < -max(STRICT_DELTA_FLOOR, tolerances[key])
        for key, value in delta.items()
    )
    any_negative = any(float(value) < 0.0 for value in delta.values())
    return {
        "all_nonpositive_within_numeric_tolerance": all_nonpositive,
        "at_least_one_resolved_strict_descent": strict,
        "raw_exact_common_descent": bool(all_nonpositive and strict),
        "resolution_limited_under_exact_closure": bool(
            all_nonpositive and any_negative and not strict
        ),
        "numeric_tolerance": tolerances,
    }


def _interpolate_projector_correction(raw, full_projected, factor):
    correction = product_log_torch(raw, full_projected)
    return product_exp_torch(raw, float(factor) * correction)


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
        raise RuntimeError("V15.14c requires an unpublished failed diagnostic")
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.14c exact-closure probe requires CUDA")
    state = m.torch.load(state_path, map_location="cpu", weights_only=False)
    artifact = m.torch.load(fit_path, map_location="cpu", weights_only=False)
    if state.get("formal_checkpoint") or artifact.get("formal_checkpoint"):
        raise RuntimeError("formal checkpoint input is forbidden")
    anchor_batch, train_batch, schedule = (
        projected_probe._materialize_first_transaction(artifact, device)
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
    initial_mgda = diagnostic._pareto_common_descent_backward(
        model, total, terms, cfg
    )
    parameters, initial_direction = _parameter_direction(model)
    base_parameters = [parameter.detach().clone() for parameter in parameters]
    if not initial_mgda.get("common_descent_exists"):
        raise RuntimeError("training-transaction MGDA has no common descent")

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
    baseline_guard_passed, baseline_blockers, baseline_details = (
        projected_probe._audit_guard_candidate(
            baseline_guard,
            guard_anchor,
            guard_relative,
            guard_absolute,
        )
    )

    fd_epsilon = tuple(float(value) for value in args.fd_epsilon.split(","))
    initial_fd_rows, initial_exact_fd = _central_exact_fd(
        model,
        anchor_batch,
        cfg,
        baseline,
        identity,
        baseline_guard,
        parameters,
        base_parameters,
        initial_direction,
        fd_epsilon,
    )
    agreement = _sign_agreement(initial_mgda, initial_exact_fd)

    exact_closure_mgda = None
    direction = initial_direction
    direction_source = "training_transaction_subgroup_mgda"
    if not agreement["all_agree"]:
        projected_probe._set_parameter_candidate(
            parameters, base_parameters, initial_direction, 0.0
        )
        model.zero_grad(set_to_none=True)
        exact_closure_mgda = _exact_closure_mgda_backward(
            model, anchor_batch, cfg
        )
        parameters, direction = _parameter_direction(model)
        direction_source = "fixed_bank_exact_observable_signed_residual_mgda"

    effective_common_exists = bool(
        (exact_closure_mgda or initial_mgda).get("common_descent_exists")
    )
    effective_fd_rows = []
    effective_exact_fd = {}
    if effective_common_exists:
        effective_fd_rows, effective_exact_fd = _central_exact_fd(
            model,
            anchor_batch,
            cfg,
            baseline,
            identity,
            baseline_guard,
            parameters,
            base_parameters,
            direction,
            fd_epsilon,
        )

    effective_mgda = exact_closure_mgda or initial_mgda
    effective_derivatives = {
        _initial_to_guard_name(key): value
        for key, value in effective_mgda["directional_derivatives"].items()
    } if exact_closure_mgda is None else dict(
        effective_mgda["directional_derivatives"]
    )

    targets = tuple(float(value) for value in args.target_rms.split(","))
    rows = []
    raw_blocker_counts = Counter()
    projected_blocker_counts = Counter()
    selected = None
    if effective_common_exists:
        for target in targets:
            candidate_started = time.perf_counter()
            raw, parameter_scale, achieved = (
                projected_probe._candidate_at_output_rms(
                    model,
                    anchor_batch,
                    cfg,
                    baseline,
                    parameters,
                    base_parameters,
                    direction,
                    target,
                )
            )
            raw_values = projected_probe._float_guard(
                projected_probe._guard_values_for_prediction(
                    model, anchor_batch, cfg, raw, identity
                )
            )
            raw_guard_passed, raw_blockers, raw_details = (
                projected_probe._audit_guard_candidate(
                    raw_values,
                    guard_anchor,
                    guard_relative,
                    guard_absolute,
                )
            )
            raw_blocker_counts.update(raw_blockers)
            raw_delta = {
                key: raw_values[key] - baseline_guard[key]
                for key in _observable_values(baseline_guard)
            }
            raw_status = _scientific_delta_status(raw_delta, raw_details)
            raw_admitted = bool(
                raw_guard_passed and raw_status["raw_exact_common_descent"]
            )

            projection_trials = []
            selected_factor = None
            selected_motion = None
            selected_values = None
            selected_details = None
            selected_blockers = None
            selected_delta = None
            selected_status = None
            projector_report = None
            if raw_admitted:
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
                for factor in PROJECTION_FACTORS:
                    candidate = _interpolate_projector_correction(
                        raw, full_projected, factor
                    )
                    values = projected_probe._float_guard(
                        projected_probe._guard_values_for_prediction(
                            model, anchor_batch, cfg, candidate, identity
                        )
                    )
                    passed, blockers, details = (
                        projected_probe._audit_guard_candidate(
                            values,
                            guard_anchor,
                            guard_relative,
                            guard_absolute,
                        )
                    )
                    projected_blocker_counts.update(blockers)
                    delta = {
                        key: values[key] - baseline_guard[key]
                        for key in _observable_values(baseline_guard)
                    }
                    status = _scientific_delta_status(delta, details)
                    accepted = bool(
                        passed
                        and status["raw_exact_common_descent"]
                        and projector_report["scope_safe"]
                    )
                    projection_trials.append({
                        "factor": factor,
                        "fixed_exact_guard_passed": passed,
                        "guard_blockers": blockers,
                        "observable_residual_delta": delta,
                        "projected_scientific_nonregression": status[
                            "all_nonpositive_within_numeric_tolerance"
                        ],
                        "at_least_one_resolved_strict_descent": status[
                            "at_least_one_resolved_strict_descent"
                        ],
                        "resolution_limited_under_exact_closure": status[
                            "resolution_limited_under_exact_closure"
                        ],
                        "accepted": accepted,
                    })
                    if accepted:
                        selected_factor = factor
                        selected_motion = candidate
                        selected_values = values
                        selected_details = details
                        selected_blockers = blockers
                        selected_delta = delta
                        selected_status = status
                        break

            objective_rows = {}
            for key in _observable_values(baseline_guard):
                objective_rows[key] = {
                    "training_transaction_autograd_directional_derivative": (
                        agreement["by_objective"][key][
                            "training_transaction_autograd_directional_derivative"
                        ]
                    ),
                    "effective_directional_derivative": (
                        effective_derivatives.get(key)
                    ),
                    "exact_fd_directional_derivative": (
                        effective_exact_fd.get(key)
                    ),
                    "autograd_linear_predicted_delta": (
                        effective_derivatives.get(key, 0.0) * parameter_scale
                    ),
                    "exact_fd_linear_predicted_delta": (
                        effective_exact_fd.get(key, 0.0) * target
                    ),
                    "raw_delta": raw_delta[key],
                    "projected_delta": (
                        None if selected_delta is None else selected_delta[key]
                    ),
                }

            effective = selected_motion is not None
            row = {
                "target_output_tangent_rms": target,
                "achieved_output_tangent_rms": achieved,
                "parameter_direction_scale": parameter_scale,
                "direction_source": direction_source,
                "subobjective_deltas": objective_rows,
                "raw_fixed_exact_guard_passed": raw_guard_passed,
                "raw_guard_blockers": raw_blockers,
                "raw_guard_metrics": raw_details,
                "raw_observable_residual_delta": raw_delta,
                "raw_exact_common_descent": raw_status[
                    "raw_exact_common_descent"
                ],
                "raw_candidate_admitted_to_projector": raw_admitted,
                "resolution_limited_under_exact_closure": raw_status[
                    "resolution_limited_under_exact_closure"
                ],
                "projector": projector_report,
                "projection_trials": projection_trials,
                "projection_backtracking_factor": selected_factor,
                "projected_scientific_nonregression": bool(
                    selected_status
                    and selected_status[
                        "all_nonpositive_within_numeric_tolerance"
                    ]
                ),
                "projected_fixed_exact_guard_passed": bool(
                    selected_motion is not None
                ),
                "projected_guard_blockers": selected_blockers,
                "projected_guard_metrics": selected_details,
                "projected_guard_values": selected_values,
                "projected_observable_residual_delta": selected_delta,
                "effective_projected_candidate": effective,
                "elapsed_seconds": time.perf_counter() - candidate_started,
            }
            rows.append(row)
            if effective and selected is None:
                selected = len(rows) - 1
                np.save(
                    destination / "selected_projected_candidate.npy",
                    selected_motion.detach().cpu().numpy(),
                )

    projected_probe._set_parameter_candidate(
        parameters, base_parameters, direction, 0.0
    )
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
        "baseline_fixed_exact_guard_passed": baseline_guard_passed,
        "baseline_guard_blockers": baseline_blockers,
        "baseline_guard_metrics": baseline_details,
        "baseline_guard_values": baseline_guard,
        "training_transaction_autograd_mgda": initial_mgda,
        "training_transaction_exact_fd_samples": initial_fd_rows,
        "autograd_vs_exact_fd_sign_agreement": agreement,
        "direction_replaced_with_exact_closure_mgda": bool(
            not agreement["all_agree"]
        ),
        "exact_closure_mgda": exact_closure_mgda,
        "effective_direction_source": direction_source,
        "effective_common_descent_exists": effective_common_exists,
        "exact_fd_samples": effective_fd_rows,
        "exact_fd_directional_derivatives": effective_exact_fd,
        "candidate_count": len(rows),
        "effective_projected_candidate_count": sum(
            int(row["effective_projected_candidate"]) for row in rows
        ),
        "selected_candidate_index": selected,
        "projected_direction_exists": selected is not None,
        "raw_guard_blocker_counts": dict(raw_blocker_counts),
        "projected_guard_blocker_counts": dict(projected_blocker_counts),
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "projection_backtracking_factors": list(PROJECTION_FACTORS),
        "diagnostic_fd_scales_eligible_as_learning_steps": False,
        "candidates": rows,
        "scope_safe": all(
            row["projector"] is None or row["projector"]["scope_safe"]
            for row in rows
        ),
        "numeric_audit_complete": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = destination / "exact_closure_tangent.report.json"
    m.save_json(result, report)
    print(json.dumps({
        "stage": "refiner_v15_14c_exact_closure_tangent_probe_complete",
        "report": str(report.resolve()),
        "autograd_exact_fd_sign_mismatches": agreement["mismatch_count"],
        "direction_replaced": result[
            "direction_replaced_with_exact_closure_mgda"
        ],
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
        "--fd-epsilon",
        default=",".join(str(value) for value in DEFAULT_FD_EPSILON),
    )
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
