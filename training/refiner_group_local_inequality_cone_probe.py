"""Development-only V15.14f group-local hard-inequality cone probe.

The fixed-bank endpoint, temporal, and hard-constraint correction actions from
V15.14e remain group-local output-space directions.  For each nonnegative
endpoint/temporal mixture, a deterministic active-set solve finds signed hard
correction coefficients satisfying the one-sided exact-closure inequalities.
Only nonzero directions are normalized to the required resolved radius and
decided by the immutable full Guard before the scientific projector may run.
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
from training import refiner_exact_closure_tangent_probe as closure_probe
from training import refiner_group_local_nullspace_cone_probe as group_probe
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_14f_group_local_one_sided_hard_inequality_cone_v1"
PROTOCOL = "fixed_bank_group_local_fd_active_set_hard_inequality_cone_v1"
DEFAULT_FD_EPSILON = 1.0e-6
DEFAULT_TARGET_RMS = 1.0e-4
PAIR_GRID_DENOMINATOR = 64
LINEAR_INEQUALITY_TOLERANCE = 1.0e-8
LINEAR_STRICT_DESCENT_FLOOR = 1.0e-7


def _scientific_weight_grid(group_name):
    yield [f"{group_name}.endpoint"], [1.0, 0.0]
    yield [f"{group_name}.temporal"], [0.0, 1.0]
    for endpoint_weight in range(1, PAIR_GRID_DENOMINATOR):
        left = endpoint_weight / PAIR_GRID_DENOMINATOR
        yield (
            [f"{group_name}.endpoint", f"{group_name}.temporal"],
            [left, 1.0 - left],
        )


def _least_norm_equalities(matrix, target):
    if matrix.size == 0:
        return np.zeros((matrix.shape[1],), dtype=np.float64)
    gram = matrix @ matrix.T
    multipliers = np.linalg.lstsq(gram, target, rcond=1.0e-12)[0]
    return matrix.T @ multipliers


def _solve_signed_corrections(
    group_name,
    scientific_weights,
    scientific_sources,
    correction_sources,
    active_rows,
    derivatives,
):
    """Project a fixed nonnegative science mixture into the FD inequality cone."""
    endpoint_key = f"{group_name}.observable_endpoint_0p03"
    temporal_key = f"{group_name}.observable_temporal_0p03"
    constraint_keys = [
        row["constraint"] for row in active_rows
    ] + [endpoint_key, temporal_key]
    sources = scientific_sources + correction_sources
    matrix = np.asarray([
        [derivatives[source][key] for source in sources]
        for key in constraint_keys
    ], dtype=np.float64)
    row_scales = np.maximum(1.0, np.max(np.abs(matrix), axis=1))
    normalized_matrix = matrix / row_scales[:, None]
    science = np.asarray(scientific_weights, dtype=np.float64)
    science_part = normalized_matrix[:, :2] @ science
    correction_matrix = normalized_matrix[:, 2:]
    tolerance = LINEAR_INEQUALITY_TOLERANCE
    strict_floor = LINEAR_STRICT_DESCENT_FLOOR

    candidates = []
    correction_count = correction_matrix.shape[1]
    if correction_count == 0:
        candidates.append((np.zeros((0,), dtype=np.float64), ()))
    else:
        candidates.append((
            np.zeros((correction_count,), dtype=np.float64),
            (),
        ))
        max_active = min(correction_count, len(constraint_keys))
        for active_count in range(1, max_active + 1):
            for active_indices in itertools.combinations(
                range(len(constraint_keys)), active_count
            ):
                selected = np.asarray(active_indices, dtype=np.int64)
                correction = _least_norm_equalities(
                    correction_matrix[selected],
                    -science_part[selected],
                )
                candidates.append((correction, active_indices))

    feasible = []
    for correction, active_indices in candidates:
        coefficients = np.concatenate((science, correction))
        directional = matrix @ coefficients
        normalized_directional = normalized_matrix @ coefficients
        hard_directional = normalized_directional[:-2]
        scientific_directional = normalized_directional[-2:]
        hard_nonregression = bool(
            np.all(hard_directional <= tolerance)
        )
        scientific_nonregression = bool(
            np.all(scientific_directional <= tolerance)
        )
        scientific_strict = bool(
            np.any(scientific_directional < -strict_floor)
        )
        if hard_nonregression and scientific_nonregression and scientific_strict:
            feasible.append((
                float(np.linalg.norm(correction)),
                float(np.max(directional)),
                tuple(active_indices),
                coefficients,
                directional,
            ))

    if not feasible:
        return {
            "feasible": False,
            "reason": "no_linearized_group_local_hard_inequality_direction",
            "sources": sources,
            "scientific_sources": scientific_sources,
            "correction_sources": correction_sources,
            "scientific_coefficients_nonnegative": bool(
                np.all(science >= 0.0)
            ),
            "scientific_coefficient_sum": float(science.sum()),
            "constraint_keys": constraint_keys,
            "one_sided_fd_matrix": matrix.tolist(),
            "fd_row_scales": {
                key: float(value)
                for key, value in zip(constraint_keys, row_scales)
            },
            "candidate_active_sets_evaluated": len(candidates),
            "normalized_linear_tolerance": tolerance,
            "normalized_strict_descent_floor": strict_floor,
        }

    feasible.sort(key=lambda row: (row[0], row[1], row[2]))
    _, _, active_indices, coefficients, directional = feasible[0]
    normalized_directional = directional / row_scales
    by_constraint = {
        key: float(value)
        for key, value in zip(constraint_keys, directional)
    }
    return {
        "feasible": True,
        "reason": "group_local_hard_inequality_direction_exists",
        "sources": sources,
        "scientific_sources": scientific_sources,
        "correction_sources": correction_sources,
        "coefficients": [float(value) for value in coefficients],
        "scientific_coefficients": [
            float(value) for value in coefficients[:2]
        ],
        "signed_hard_correction_coefficients": [
            float(value) for value in coefficients[2:]
        ],
        "scientific_coefficients_nonnegative": bool(
            np.all(coefficients[:2] >= 0.0)
        ),
        "scientific_coefficient_sum": float(coefficients[:2].sum()),
        "active_set_constraints": [
            constraint_keys[index] for index in active_indices
        ],
        "constraint_keys": constraint_keys,
        "one_sided_fd_matrix": matrix.tolist(),
        "fd_row_scales": {
            key: float(value)
            for key, value in zip(constraint_keys, row_scales)
        },
        "directional_derivatives": by_constraint,
        "hard_directional_derivatives": {
            key: by_constraint[key] for key in constraint_keys[:-2]
        },
        "scientific_directional_derivatives": {
            endpoint_key: by_constraint[endpoint_key],
            temporal_key: by_constraint[temporal_key],
        },
        "all_hard_directional_derivatives_nonpositive": bool(
            np.all(normalized_directional[:-2] <= tolerance)
        ),
        "all_scientific_directional_derivatives_nonpositive": bool(
            np.all(normalized_directional[-2:] <= tolerance)
        ),
        "at_least_one_scientific_directional_derivative_strict": bool(
            np.any(normalized_directional[-2:] < -strict_floor)
        ),
        "candidate_active_sets_evaluated": len(candidates),
        "feasible_active_set_count": len(feasible),
        "normalized_linear_tolerance": tolerance,
        "normalized_strict_descent_floor": strict_floor,
    }


def _combine_actions(directions, sources, coefficients):
    return sum(
        float(coefficient) * directions[source]
        for source, coefficient in zip(sources, coefficients)
    )


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
        raise RuntimeError("V15.14f requires an unpublished failed diagnostic")
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.14f exact-closure cone probe requires CUDA")
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

    scientific_directions, scientific_direction_rows = (
        group_probe._group_local_scientific_directions(
            model, anchor_batch, cfg, baseline, identity
        )
    )
    scientific_fd_rows, scientific_fd_derivatives = (
        group_probe._one_sided_guard_fd(
            model,
            anchor_batch,
            cfg,
            baseline,
            identity,
            baseline_guard,
            scientific_directions,
            float(args.fd_epsilon),
        )
    )
    active_rows_by_group = {}
    for group_name in m.REFINER_GROUP_LABELS:
        scientific_sources = [
            f"{group_name}.endpoint",
            f"{group_name}.temporal",
        ]
        active_rows_by_group[group_name] = (
            group_probe._active_constraint_rows(
                group_name,
                scientific_sources,
                scientific_fd_derivatives,
                baseline_details,
            )
        )
    correction_directions, correction_direction_rows = (
        group_probe._group_local_constraint_corrections(
            model,
            anchor_batch,
            cfg,
            baseline,
            identity,
            active_rows_by_group,
        )
    )
    correction_fd_rows, correction_fd_derivatives = (
        group_probe._one_sided_guard_fd(
            model,
            anchor_batch,
            cfg,
            baseline,
            identity,
            baseline_guard,
            correction_directions,
            float(args.fd_epsilon),
        )
    )
    directions = {**scientific_directions, **correction_directions}
    derivatives = {
        **scientific_fd_derivatives,
        **correction_fd_derivatives,
    }

    target = float(args.target_rms)
    if target < DEFAULT_TARGET_RMS:
        raise ValueError(
            f"target_rms must be at least {DEFAULT_TARGET_RMS:g}"
        )
    rows = []
    raw_blocker_counts = Counter()
    projected_blocker_counts = Counter()
    solver_failure_counts = Counter()
    linear_feasible_by_group = Counter()
    resolved_audit_by_group = Counter()
    raw_pass_by_group = Counter()
    projected_pass_by_group = Counter()
    selected_index = None
    selected_score = math.inf
    selected_motion = None

    for group_index, group_name in enumerate(m.REFINER_GROUP_LABELS):
        scientific_sources = [
            f"{group_name}.endpoint",
            f"{group_name}.temporal",
        ]
        correction_sources = [
            f"{group_name}.hard_correction:{active['constraint']}"
            for active in active_rows_by_group[group_name]
            if f"{group_name}.hard_correction:{active['constraint']}"
            in directions
        ]
        for selected_science_sources, science_weights in (
            _scientific_weight_grid(group_name)
        ):
            candidate_started = time.perf_counter()
            solve = _solve_signed_corrections(
                group_name,
                science_weights,
                scientific_sources,
                correction_sources,
                active_rows_by_group[group_name],
                derivatives,
            )
            if solve["feasible"]:
                action = _combine_actions(
                    directions,
                    solve["sources"],
                    solve["coefficients"],
                )
                action, pre_normalization_rms = group_probe._normalize_action(
                    action, anchor_batch["seam"]
                )
            else:
                action = m.torch.zeros_like(
                    next(iter(scientific_directions.values()))
                )
                pre_normalization_rms = 0.0
            scope = group_probe._action_scope_report(
                action, anchor_batch, group_index
            )
            if not solve["feasible"]:
                unavailable_reason = solve["reason"]
            elif pre_normalization_rms <= 1.0e-12:
                unavailable_reason = (
                    "empty_group_local_hard_inequality_direction"
                )
            elif not scope["scope_safe"]:
                unavailable_reason = "group_local_action_scope_leakage"
            else:
                unavailable_reason = None
                linear_feasible_by_group.update([group_name])
            direction_available = bool(
                unavailable_reason is None
            )
            if unavailable_reason is not None:
                solver_failure_counts.update([unavailable_reason])
            if direction_available:
                raw, action_scale, achieved = (
                    group_probe._candidate_at_action_rms(
                        baseline,
                        action,
                        anchor_batch["seam"],
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
                ) = group_probe._audit_prediction(
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
                raw_blockers = [unavailable_reason]
                raw_details = baseline_details
                raw_delta = {
                    key: 0.0
                    for key in closure_probe._observable_values(baseline_guard)
                }
                raw_scientific = closure_probe._scientific_delta_status(
                    raw_delta, baseline_details
                )
            if direction_available:
                raw_blocker_counts.update(raw_blockers)
            resolved_radius = bool(achieved >= DEFAULT_TARGET_RMS)
            if direction_available and resolved_radius:
                resolved_audit_by_group.update([group_name])
            raw_valid = bool(
                direction_available
                and resolved_radius
                and raw_guard_passed
                and raw_scientific["raw_exact_common_descent"]
            )
            if raw_valid:
                raw_pass_by_group.update([group_name])

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
                    ) = group_probe._audit_prediction(
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
            if effective:
                projected_pass_by_group.update([group_name])
            row = {
                "group": group_name,
                "selected_scientific_sources": selected_science_sources,
                "requested_scientific_coefficients": science_weights,
                "inequality_solve": solve,
                "direction_available": direction_available,
                "direction_unavailable_reason": unavailable_reason,
                "pre_normalization_direction_rms": pre_normalization_rms,
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
            rows.append(row)
            if effective:
                score = float(sum(accepted_delta.values()))
                if score < selected_score:
                    selected_score = score
                    selected_index = len(rows) - 1
                    selected_motion = accepted_prediction.detach().cpu().numpy()

    if selected_motion is not None:
        np.save(destination / "selected_projected_candidate.npy", selected_motion)
    raw_count = sum(raw_pass_by_group.values())
    projected_count = sum(projected_pass_by_group.values())
    if raw_count == 0:
        feasibility_status = (
            "no_group_local_hard_inequality_descent_at_required_radius"
        )
    elif projected_count == 0:
        feasibility_status = "group_local_inequality_raw_descent_projector_failed"
    else:
        feasibility_status = "group_local_inequality_projected_descent_exists"

    expected_per_group = 2 + (PAIR_GRID_DENOMINATOR - 1)
    expected_count = len(m.REFINER_GROUP_LABELS) * expected_per_group
    outside_max = max(
        (
            float(row["action_scope"][
                "outside_group_or_ownership_abs_max"
            ])
            for row in rows
        ),
        default=0.0,
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
        "local_action_shape": list(baseline.shape[:-1]) + [79],
        "local_action_product_channels": 79,
        "local_action_contact_channels_frozen": 4,
        "local_action_manifold_tangent_channels": 75,
        "group_local_scientific_directions": scientific_direction_rows,
        "scientific_direction_one_sided_exact_guard_fd": scientific_fd_rows,
        "active_hard_constraints_by_group": active_rows_by_group,
        "group_local_hard_constraint_correction_directions": (
            correction_direction_rows
        ),
        "hard_constraint_correction_one_sided_exact_guard_fd": (
            correction_fd_rows
        ),
        "inequality_protocol": {
            "hard_constraints": "one_sided_fd_directional_derivative_le_0",
            "endpoint": "one_sided_fd_directional_derivative_le_0",
            "temporal": "one_sided_fd_directional_derivative_le_0",
            "scientific_strict_descent_required": True,
            "scientific_coefficients": "nonnegative_sum_equal_1",
            "hard_correction_coefficients": "signed_minimum_l2_norm",
            "solver": "deterministic_active_set_enumeration",
        },
        "diagnostic_fd_epsilon_output_tangent_rms": float(args.fd_epsilon),
        "diagnostic_fd_scales_eligible_as_learning_steps": False,
        "minimum_required_output_tangent_rms": DEFAULT_TARGET_RMS,
        "target_output_tangent_rms": target,
        "expected_candidate_count": expected_count,
        "candidate_count": len(rows),
        "solver_failure_counts": dict(solver_failure_counts),
        "linear_feasible_direction_count": int(
            sum(linear_feasible_by_group.values())
        ),
        "linear_feasible_direction_by_group": dict(
            linear_feasible_by_group
        ),
        "resolved_candidate_audit_count": int(
            sum(resolved_audit_by_group.values())
        ),
        "resolved_candidate_audit_by_group": dict(
            resolved_audit_by_group
        ),
        "raw_exact_common_descent_count": raw_count,
        "raw_exact_common_descent_by_group": dict(raw_pass_by_group),
        "effective_projected_candidate_count": projected_count,
        "effective_projected_candidate_by_group": dict(
            projected_pass_by_group
        ),
        "cross_short_effective_projected_candidate_count": int(
            projected_pass_by_group["cross_short"]
        ),
        "cross_long_effective_projected_candidate_count": int(
            projected_pass_by_group["cross_long"]
        ),
        "selected_candidate_index": selected_index,
        "projected_direction_exists": selected_index is not None,
        "feasibility_status": feasibility_status,
        "raw_guard_blocker_counts": dict(raw_blocker_counts),
        "projected_guard_blocker_counts": dict(projected_blocker_counts),
        "outside_group_or_ownership_abs_max": outside_max,
        "routing_architecture_pivot_supported": bool(
            baseline_passed
            and projected_pass_by_group["cross_short"] > 0
            and projected_pass_by_group["cross_long"] > 0
            and outside_max == 0.0
        ),
        "projection_backtracking_factors": list(
            closure_probe.PROJECTION_FACTORS
        ),
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "candidates": rows,
        "scope_safe": all(
            row["action_scope"]["scope_safe"]
            and (
                row["projector"] is None
                or row["projector"]["scope_safe"]
            )
            for row in rows
        ),
        "numeric_audit_complete": bool(len(rows) == expected_count),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["routing_architecture_pivot_supported"] = bool(
        result["routing_architecture_pivot_supported"]
        and result["numeric_audit_complete"]
        and result["scope_safe"]
    )
    report = destination / "group_local_inequality_cone.report.json"
    m.save_json(result, report)
    print(json.dumps({
        "stage": "refiner_v15_14f_group_local_inequality_cone_complete",
        "report": str(report.resolve()),
        "candidates": len(rows),
        "raw_exact_common_descent_by_group": dict(raw_pass_by_group),
        "effective_projected_candidate_by_group": dict(
            projected_pass_by_group
        ),
        "feasibility_status": feasibility_status,
        "routing_architecture_pivot_supported": result[
            "routing_architecture_pivot_supported"
        ],
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
