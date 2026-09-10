"""Development-only V15.14d one-sided exact-closure subspace cone probe.

The probe keeps the eight fixed-bank endpoint/temporal negative gradients as
independent directions.  One-sided exact-closure finite differences describe
their local action, while every one-, two-, or three-direction nonnegative
combination is decided by a resolved-radius exact closure audit.  Only a raw
combination that passes the immutable Guard and gives an eight-objective
common descent is admitted to the scientific contact projector.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from training import motion_models as m
from training import refiner_bridge_diagnostics as diagnostic
from training import refiner_exact_closure_tangent_probe as closure_probe
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_14d_one_sided_exact_closure_subspace_cone_v1"
PROTOCOL = "fixed_bank_one_sided_fd_sparse_nonnegative_exact_closure_cone_v1"
DEFAULT_FD_EPSILON = 1.0e-6
DEFAULT_TARGET_RMS = 1.0e-4
MAX_COMBINATION_SIZE = 3
PAIR_GRID_DENOMINATOR = 64
TRIPLE_GRID_DENOMINATOR = 32


def _restore(parameters, base_parameters):
    projected_probe._set_parameter_candidate(
        parameters,
        base_parameters,
        [m.torch.zeros_like(value) for value in parameters],
        0.0,
    )


def _candidate_at_required_output_rms(
    model,
    batch,
    cfg,
    baseline,
    parameters,
    base_parameters,
    direction,
    target,
):
    """Calibrate an actual candidate on or just above the requested radius."""
    candidate, scale, achieved = projected_probe._candidate_at_output_rms(
        model,
        batch,
        cfg,
        baseline,
        parameters,
        base_parameters,
        direction,
        target,
    )
    for _ in range(4):
        if achieved >= target or achieved <= 1.0e-16:
            break
        scale *= (float(target) / achieved) * 1.001
        projected_probe._set_parameter_candidate(
            parameters, base_parameters, direction, scale
        )
        candidate, _ = projected_probe._model_prediction(model, batch, cfg)
        achieved = projected_probe._motion_edit_rms(
            baseline, candidate, batch["seam"]
        )
    return candidate.detach(), float(scale), float(achieved)


def _independent_exact_observable_directions(model, batch, cfg):
    """Return eight RMS-normalized negative exact-Guard gradients."""
    prediction, identity = m._refiner_batch_outputs(model, batch, cfg)
    guard_tensors = projected_probe._guard_values_for_prediction(
        model,
        batch,
        cfg,
        prediction,
        identity,
    )
    objectives = closure_probe._observable_values(guard_tensors)
    expected = 2 * len(m.REFINER_GROUP_LABELS)
    if len(objectives) != expected:
        raise RuntimeError(
            f"V15.14d expected {expected} exact observables, "
            f"found {len(objectives)}"
        )
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count < 1:
        raise RuntimeError("V15.14d found no trainable parameters")

    names = list(objectives)
    raw_gradients = []
    for objective in objectives.values():
        values = m.torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        raw_gradients.append([
            m.torch.zeros_like(parameter) if value is None else value
            for parameter, value in zip(parameters, values)
        ])

    count = prediction.new_tensor(float(parameter_count), dtype=m.torch.float64)
    epsilon = prediction.new_tensor(1.0e-24, dtype=m.torch.float64)
    norm_squares = m.torch.stack([
        diagnostic._tuple_dot(gradient, gradient)
        for gradient in raw_gradients
    ])
    rms_norms = (norm_squares / count).clamp_min(0.0).sqrt()
    if any(float(value.detach()) <= 1.0e-12 for value in rms_norms):
        inactive = [
            name for name, value in zip(names, rms_norms)
            if float(value.detach()) <= 1.0e-12
        ]
        raise RuntimeError(f"zero exact observable gradients: {inactive}")

    directions = [
        [
            -value / rms_norms[index].clamp_min(epsilon).to(value.dtype)
            for value in gradient
        ]
        for index, gradient in enumerate(raw_gradients)
    ]
    autograd_matrix = []
    for direction in directions:
        autograd_matrix.append({
            name: float(diagnostic._tuple_dot(gradient, direction).detach())
            for name, gradient in zip(names, raw_gradients)
        })
    model.zero_grad(set_to_none=True)
    return {
        "names": names,
        "parameters": parameters,
        "directions": directions,
        "gradient_rms_norms": {
            name: float(value.detach())
            for name, value in zip(names, rms_norms)
        },
        "autograd_directional_derivative_matrix": {
            source: row for source, row in zip(names, autograd_matrix)
        },
        "guard_value_domain": (
            "exact_stage_signed_residual_or_observable_fidelity_metric"
        ),
        "fixed_bank_used": True,
        "rms_normalized": True,
    }


def _one_sided_fd_basis(
    model,
    batch,
    cfg,
    baseline,
    identity,
    baseline_guard,
    parameters,
    base_parameters,
    names,
    directions,
    epsilon,
):
    """Measure R(x + eps d) - R(x) for each independent direction."""
    objective_names = list(closure_probe._observable_values(baseline_guard))
    rows = []
    matrix = []
    for source, direction in zip(names, directions):
        candidate, parameter_scale, achieved = (
            _candidate_at_required_output_rms(
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
        values = projected_probe._float_guard(
            projected_probe._guard_values_for_prediction(
                model, batch, cfg, candidate, identity
            )
        )
        delta = {
            key: values[key] - baseline_guard[key]
            for key in objective_names
        }
        derivative = {
            key: delta[key] / float(epsilon)
            for key in objective_names
        }
        achieved_derivative = {
            key: delta[key] / achieved if achieved > 0.0 else None
            for key in objective_names
        }
        matrix.append([derivative[key] for key in objective_names])
        rows.append({
            "direction_source": source,
            "epsilon_output_tangent_rms": float(epsilon),
            "diagnostic_only": True,
            "eligible_as_learning_step": False,
            "parameter_direction_scale": parameter_scale,
            "achieved_output_tangent_rms": achieved,
            "observable_residual_delta": delta,
            "one_sided_exact_fd_directional_derivative": derivative,
            "one_sided_exact_fd_derivative_by_achieved_rms": (
                achieved_derivative
            ),
        })
    _restore(parameters, base_parameters)
    return rows, objective_names, np.asarray(matrix, dtype=np.float64)


def _coefficient_grid(size):
    if size == 1:
        yield np.ones((1,), dtype=np.float64)
        return
    denominator = (
        PAIR_GRID_DENOMINATOR if size == 2 else TRIPLE_GRID_DENOMINATOR
    )
    if size == 2:
        for left in range(1, denominator):
            yield np.asarray(
                [left, denominator - left], dtype=np.float64
            ) / denominator
        return
    for first in range(1, denominator - 1):
        for second in range(1, denominator - first):
            third = denominator - first - second
            if third > 0:
                yield np.asarray(
                    [first, second, third], dtype=np.float64
                ) / denominator


def _solve_subset_coefficients(derivative_matrix, indices):
    """Choose deterministic simplex coefficients from exact one-sided FD."""
    local = derivative_matrix[np.asarray(indices, dtype=np.int64)]
    best = None
    best_key = None
    for coefficients in _coefficient_grid(len(indices)):
        predicted = coefficients @ local
        scale = max(1.0, float(np.max(np.abs(predicted))))
        tolerance = 1.0e-8 * scale
        all_nonpositive = bool(np.all(predicted <= tolerance))
        one_strict = bool(np.any(predicted < -tolerance))
        feasible = all_nonpositive and one_strict
        key = (
            0 if feasible else 1,
            max(0.0, float(np.max(predicted))),
            float(np.maximum(predicted, 0.0).sum()),
            float(predicted.sum()),
            float(np.square(coefficients).sum()),
            tuple(float(value) for value in coefficients),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (coefficients.copy(), predicted.copy(), feasible)
    if best is None:
        raise RuntimeError(f"empty coefficient grid for subset {indices}")
    return best


def _combined_direction(directions, indices, coefficients):
    return [
        sum(
            float(coefficient) * directions[index][parameter_index]
            for index, coefficient in zip(indices, coefficients)
        )
        for parameter_index in range(len(directions[0]))
    ]


def _observable_delta(values, baseline_guard):
    return {
        key: values[key] - baseline_guard[key]
        for key in closure_probe._observable_values(baseline_guard)
    }


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
    delta = _observable_delta(values, baseline_guard)
    scientific = closure_probe._scientific_delta_status(delta, details)
    return values, passed, blockers, details, delta, scientific


def _candidate_score(delta):
    return float(sum(float(value) for value in delta.values()))


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
        raise RuntimeError("V15.14d requires an unpublished failed diagnostic")
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.14d exact-closure cone probe requires CUDA")
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

    basis = _independent_exact_observable_directions(
        model, anchor_batch, cfg
    )
    parameters = basis.pop("parameters")
    directions = basis.pop("directions")
    base_parameters = [parameter.detach().clone() for parameter in parameters]
    fd_rows, objective_names, derivative_matrix = _one_sided_fd_basis(
        model,
        anchor_batch,
        cfg,
        baseline,
        identity,
        baseline_guard,
        parameters,
        base_parameters,
        basis["names"],
        directions,
        float(args.fd_epsilon),
    )

    target = float(args.target_rms)
    if target < DEFAULT_TARGET_RMS:
        raise ValueError(
            f"target_rms must be at least {DEFAULT_TARGET_RMS:g}"
        )
    combinations = []
    raw_blocker_counts = Counter()
    projected_blocker_counts = Counter()
    raw_pass_count = 0
    projected_pass_count = 0
    selected_motion = None
    selected_index = None
    selected_score = math.inf

    for size in range(1, MAX_COMBINATION_SIZE + 1):
        for indices in itertools.combinations(range(len(directions)), size):
            candidate_started = time.perf_counter()
            coefficients, predicted_array, predicted_feasible = (
                _solve_subset_coefficients(derivative_matrix, indices)
            )
            direction = _combined_direction(
                directions, indices, coefficients
            )
            raw, parameter_scale, achieved = (
                _candidate_at_required_output_rms(
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
            raw_blocker_counts.update(raw_blockers)
            resolved_radius = bool(
                achieved >= DEFAULT_TARGET_RMS
                and target >= DEFAULT_TARGET_RMS
            )
            raw_valid = bool(
                resolved_radius
                and raw_guard_passed
                and raw_scientific["raw_exact_common_descent"]
            )
            raw_pass_count += int(raw_valid)

            projection_trials = []
            projector_report = None
            accepted_factor = None
            accepted_prediction = None
            accepted_delta = None
            accepted_details = None
            accepted_values = None
            accepted_scientific = None
            if raw_valid:
                _restore(parameters, base_parameters)
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
                        accepted_details = details
                        accepted_values = values
                        accepted_scientific = scientific
                        break
            effective = accepted_prediction is not None
            projected_pass_count += int(effective)
            predicted = {
                key: float(value)
                for key, value in zip(objective_names, predicted_array)
            }
            subobjective_deltas = {
                key: {
                    "one_sided_fd_predicted_derivative": predicted[key],
                    "one_sided_fd_predicted_delta_at_target_rms": (
                        predicted[key] * target
                    ),
                    "raw_delta": raw_delta[key],
                    "projected_delta": (
                        None if accepted_delta is None
                        else accepted_delta[key]
                    ),
                }
                for key in objective_names
            }
            row = {
                "combination_size": size,
                "selected_direction_sources": [
                    basis["names"][index] for index in indices
                ],
                "direction_coefficients": [
                    float(value) for value in coefficients
                ],
                "coefficient_sum": float(coefficients.sum()),
                "coefficients_nonnegative": bool(
                    np.all(coefficients >= 0.0)
                ),
                "one_sided_fd_predicted_directional_derivatives": predicted,
                "one_sided_fd_predicted_common_descent": predicted_feasible,
                "target_output_tangent_rms": target,
                "achieved_output_tangent_rms": achieved,
                "resolved_required_radius": resolved_radius,
                "parameter_direction_scale": parameter_scale,
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
                "projector": projector_report,
                "projection_trials": projection_trials,
                "projection_backtracking_factor": accepted_factor,
                "projected_scientific_nonregression": bool(
                    accepted_scientific
                    and accepted_scientific[
                        "all_nonpositive_within_numeric_tolerance"
                    ]
                ),
                "projected_fixed_exact_guard_passed": effective,
                "projected_guard_metrics": accepted_details,
                "projected_guard_values": accepted_values,
                "projected_observable_residual_delta": accepted_delta,
                "effective_projected_candidate": effective,
                "subobjective_deltas": subobjective_deltas,
                "elapsed_seconds": time.perf_counter() - candidate_started,
            }
            combinations.append(row)
            if effective:
                score = _candidate_score(accepted_delta)
                if score < selected_score:
                    selected_score = score
                    selected_index = len(combinations) - 1
                    selected_motion = accepted_prediction.detach().cpu().numpy()

    _restore(parameters, base_parameters)
    if selected_motion is not None:
        np.save(destination / "selected_projected_candidate.npy", selected_motion)

    if raw_pass_count == 0:
        feasibility_status = (
            "no_resolved_exact_closure_common_descent_at_required_radius"
        )
    elif projected_pass_count == 0:
        feasibility_status = "resolved_raw_common_descent_projector_failed"
    else:
        feasibility_status = (
            "resolved_exact_closure_projected_direction_exists"
        )
    expected_combinations = sum(
        math.comb(len(directions), size)
        for size in range(1, MAX_COMBINATION_SIZE + 1)
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
        "baseline_fixed_exact_guard_passed": baseline_passed,
        "baseline_guard_blockers": baseline_blockers,
        "baseline_guard_metrics": baseline_details,
        "baseline_guard_values": baseline_guard,
        "independent_direction_count": len(directions),
        "independent_direction_sources": basis["names"],
        "independent_direction_basis": basis,
        "one_sided_fd_epsilon_output_tangent_rms": float(args.fd_epsilon),
        "one_sided_exact_fd": fd_rows,
        "diagnostic_fd_scales_eligible_as_learning_steps": False,
        "maximum_combination_size": MAX_COMBINATION_SIZE,
        "coefficient_constraint": "nonnegative_sum_le_1",
        "coefficient_solver": {
            "protocol": "deterministic_simplex_grid_minimax_fd_derivative_v1",
            "pair_grid_denominator": PAIR_GRID_DENOMINATOR,
            "triple_grid_denominator": TRIPLE_GRID_DENOMINATOR,
        },
        "minimum_required_output_tangent_rms": DEFAULT_TARGET_RMS,
        "target_output_tangent_rms": target,
        "expected_combination_count": expected_combinations,
        "combination_count": len(combinations),
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
        "combinations": combinations,
        "scope_safe": all(
            row["projector"] is None or row["projector"]["scope_safe"]
            for row in combinations
        ),
        "numeric_audit_complete": bool(
            len(combinations) == expected_combinations
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = destination / "one_sided_subspace_cone.report.json"
    m.save_json(result, report)
    print(json.dumps({
        "stage": "refiner_v15_14d_one_sided_subspace_cone_probe_complete",
        "report": str(report.resolve()),
        "independent_directions": len(directions),
        "combinations": len(combinations),
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
