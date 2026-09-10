"""Transactional, bounded same-minibatch descent for the physical Refiner.

Gradient clipping does not bound Adam's parameter update. High-order FK losses
can increase sharply even along a first-order descent direction. Check actual
loss values instead of assuming a clipped gradient makes an update safe.
This is optimization acceptance, NOT per-case physical or scientific acceptance.
The closure must use the SAME fixed batch and deterministic model (no dropout,
mutable running statistics, resampling, validation data or random degradation).
"""
from __future__ import annotations

import copy
import math

import torch


REFINER_UPDATE_PROTOCOL = "exact_guard_constrained_fixed_anchor_armijo_v9"
MAX_BACKTRACK_TRIALS = 12  # per direction; at most 24 extra forward evaluations
ARMIJO_FACTOR = 1.0e-4
MIN_RELATIVE_DECREASE = 1.0e-8  # optimization progress, NOT a motion-quality gate
_SCALE_KEY = "refiner_trial_scale"  # persisted by optimizer.state_dict()


def checked_refiner_step(
    optimizer,
    loss,
    closure,
    *,
    max_trials=MAX_BACKTRACK_TRIALS,
    gradient_unscale=1.0,
    group_guard_before=None,
    group_guard_reference=None,
    group_guard_relative_tolerance=0.0,
    group_guard_absolute_tolerance=0.0,
    group_guard_metric_metadata=None,
    group_guard_directional_derivative=None,
    required_guard_improvement_keys=(),
    minimum_effective_scale=0.0,
):
    """Transactional Armijo step with optional subgroup non-regression.

    With ``group_guard_before`` disabled this is the V11 same-batch optimizer.
    With the guard enabled, ``closure`` MUST return ``(loss, group_losses)``.
    A trial is accepted only when the scalar Armijo condition passes AND every
    named subgroup/component guard stays within its fixed reference
    relative/absolute allowance. ``group_guard_before`` records the actual
    fixed-bank metrics before this transaction. ``group_guard_reference`` may
    hold a persistent anchor/best-so-far envelope; when omitted, the former
    pre-update behavior is retained for callers without a fixed bank.
    Parameters and the complete optimizer state are restored on rejection.
    """
    if not 1 <= int(max_trials) <= MAX_BACKTRACK_TRIALS:
        raise ValueError(f"max_trials must be in [1,{MAX_BACKTRACK_TRIALS}]")
    if not math.isfinite(gradient_unscale) or gradient_unscale < 1.0:
        raise ValueError("gradient_unscale must be finite and >= 1")
    minimum_effective_scale = float(minimum_effective_scale)
    if not math.isfinite(minimum_effective_scale) or not (
        0.0 <= minimum_effective_scale < 1.0
    ):
        raise ValueError("minimum_effective_scale must be finite in [0,1)")
    raw_relative_tolerance = group_guard_relative_tolerance
    raw_absolute_tolerance = group_guard_absolute_tolerance

    def tolerance_map(value, keys, name):
        """Resolve one scalar or an exact per-guard absolute allowance map."""
        if hasattr(value, "items"):
            resolved = {str(k): float(v) for k, v in value.items()}
            if set(resolved) != set(keys):
                raise ValueError(f"{name} keys differ from group guard metrics")
        else:
            scalar_value = float(value)
            if not math.isfinite(scalar_value) or scalar_value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            resolved = {key: scalar_value for key in keys}
        if not all(math.isfinite(v) and v >= 0.0 for v in resolved.values()):
            raise ValueError(f"{name} must be finite and non-negative")
        return resolved

    def scalar(value):
        if torch.is_tensor(value):
            return float(value.detach())
        return float(value)

    guard_enabled = group_guard_before is not None
    guard_before = {}
    guard_reference = {}
    guard_relative_tolerance = {}
    guard_absolute_tolerance = {}
    guard_metadata = {}
    if guard_enabled:
        if not hasattr(group_guard_before, "items") or not group_guard_before:
            raise ValueError("group_guard_before must be a non-empty mapping")
        guard_before = {str(k): scalar(v) for k, v in group_guard_before.items()}
        if not all(math.isfinite(v) for v in guard_before.values()):
            raise FloatingPointError("nonfinite subgroup loss before optimizer update")
        raw_reference = (
            group_guard_before
            if group_guard_reference is None
            else group_guard_reference
        )
        if not hasattr(raw_reference, "items") or not raw_reference:
            raise ValueError("group_guard_reference must be a non-empty mapping")
        guard_reference = {
            str(k): scalar(v) for k, v in raw_reference.items()
        }
        if set(guard_reference) != set(guard_before):
            raise ValueError("group guard reference keys differ from current metrics")
        if not all(math.isfinite(v) for v in guard_reference.values()):
            raise FloatingPointError("nonfinite subgroup guard reference")
        guard_relative_tolerance = tolerance_map(
            group_guard_relative_tolerance,
            guard_before,
            "group_guard_relative_tolerance",
        )
        guard_absolute_tolerance = tolerance_map(
            group_guard_absolute_tolerance,
            guard_before,
            "group_guard_absolute_tolerance",
        )
        if group_guard_metric_metadata is None:
            guard_metadata = {key: {} for key in guard_before}
        else:
            if set(group_guard_metric_metadata) != set(guard_before):
                raise ValueError("group guard metadata keys differ from metrics")
            guard_metadata = {
                str(key): dict(value)
                for key, value in group_guard_metric_metadata.items()
            }
        required_guard_improvement_keys = tuple(
            str(key) for key in required_guard_improvement_keys
        )
        if not set(required_guard_improvement_keys).issubset(guard_before):
            raise ValueError("required Guard improvement key is missing")
        if (
            group_guard_directional_derivative is not None
            and not callable(group_guard_directional_derivative)
        ):
            raise TypeError("group_guard_directional_derivative must be callable")
    else:
        if group_guard_metric_metadata is not None:
            raise ValueError("group guard metadata requires group_guard_before")
        if group_guard_directional_derivative is not None:
            raise ValueError("Guard derivative callback requires group_guard_before")
        if required_guard_improvement_keys:
            raise ValueError("required Guard improvements require group_guard_before")
        # Validate disabled scalar callers too. A mapping has no meaning without
        # named guard metrics and is rejected rather than silently ignored.
        if hasattr(group_guard_relative_tolerance, "items") or hasattr(
            group_guard_absolute_tolerance, "items"
        ):
            raise ValueError("per-guard tolerances require group_guard_before")
        tolerance_map(
            group_guard_relative_tolerance,
            (),
            "group_guard_relative_tolerance",
        )
        tolerance_map(
            group_guard_absolute_tolerance,
            (),
            "group_guard_absolute_tolerance",
        )

    before = float(loss.detach())
    if not math.isfinite(before):
        raise FloatingPointError("nonfinite Refiner loss before optimizer update")
    parameters, rates = [], []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameters.append(parameter)
                rates.append(float(group["lr"]))
    if not parameters:
        raise RuntimeError("Refiner optimizer has no gradients")
    gradients = [p.grad.detach().clone() for p in parameters]
    if not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
        raise FloatingPointError("nonfinite Refiner gradient before optimizer update")
    minimum_decrease = max(abs(before), torch.finfo(loss.dtype).tiny) * max(
        MIN_RELATIVE_DECREASE, 8.0 * torch.finfo(loss.dtype).eps
    )
    report = {
        "protocol": REFINER_UPDATE_PROTOCOL,
        "loss_before": before,
        "loss_after": before,
        "optimizer_update_accepted": False,
        "direction": "none",
        "step_scale": 0.0,
        "trial_evaluations": 0,
        "loss_rejected_trials": 0,
        "nonfinite_trials": 0,
        "nonfinite_parameter_trials": 0,
        "resolution_limited_trials": 0,
        "first_trial_loss": None,
        "adam_directional_derivative": None,
        "used_gradient_rescue": False,
        "max_trials_per_direction": int(max_trials),
        "scientific_acceptance": False,
        "minimum_loss_decrease": minimum_decrease,
        "armijo_factor": ARMIJO_FACTOR,
        "gradient_unscale": float(gradient_unscale),
        "insufficient_decrease_trials": 0,
        "group_guard_enabled": guard_enabled,
        "group_guard_relative_tolerance": (
            guard_relative_tolerance
            if hasattr(raw_relative_tolerance, "items")
            else float(raw_relative_tolerance)
        ),
        "group_guard_absolute_tolerance": (
            guard_absolute_tolerance
            if hasattr(raw_absolute_tolerance, "items")
            else float(raw_absolute_tolerance)
        ),
        "group_guard_before": guard_before,
        "group_guard_reference": guard_reference,
        "group_guard_reference_is_persistent": bool(
            guard_enabled and group_guard_reference is not None
        ),
        "group_guard_after": None,
        "group_guard_rejected_trials": 0,
        "group_guard_last_violations": {},
        "minimum_audited_scale": None,
        "minimum_accepted_scale": None,
        "minimum_acceptable_scale": None,
        "minimum_effective_scale": minimum_effective_scale,
        "resolution_limited_under_exact_guard": False,
        "guard_metric_metadata": guard_metadata,
        "required_guard_improvement_keys": list(
            required_guard_improvement_keys
        ),
        "trials": [],
    }
    maximum_gradient = torch.stack([g.abs().max() for g in gradients]).max()
    if float(maximum_gradient) == 0:
        report["reason"] = "zero_gradient"
        return report
    original = [p.detach().clone() for p in parameters]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())

    def restore():
        with torch.no_grad():
            for parameter, value in zip(parameters, original):
                parameter.copy_(value)
        optimizer.load_state_dict(saved_optimizer)

    def derivative(direction):
        return float(
            torch.stack(
                [
                    (g.double() * delta.double()).sum()
                    for g, delta in zip(gradients, direction)
                ]
            ).sum()
        ) * gradient_unscale

    def evaluate_closure():
        result = closure()
        if guard_enabled:
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    "guarded Refiner closure must return (loss, subgroup_losses)"
                )
            value, raw_groups = result
            if not hasattr(raw_groups, "items"):
                raise RuntimeError("guarded Refiner subgroup output must be a mapping")
            groups = {str(k): scalar(v) for k, v in raw_groups.items()}
            if set(groups) != set(guard_before):
                raise RuntimeError("Refiner subgroup guard keys changed during trial")
            return float(value.detach()), groups
        if isinstance(result, tuple):
            result = result[0]
        return float(result.detach()), None

    def subgroup_violations(candidate_groups):
        if not guard_enabled:
            return {}
        violations = {}
        for key, baseline in guard_reference.items():
            candidate = candidate_groups[key]
            allowance = max(
                abs(baseline) * guard_relative_tolerance[key],
                guard_absolute_tolerance[key],
            )
            allowed = baseline + allowance
            if not math.isfinite(candidate) or candidate > allowed:
                violations[key] = {
                    "before": guard_before[key],
                    "reference": baseline,
                    "candidate": candidate if math.isfinite(candidate) else None,
                    "allowed": allowed,
                }
        return violations

    def search(direction, scale, name):
        slope = derivative(direction)
        if not math.isfinite(slope) or slope >= 0:
            return False
        predicted_derivatives = (
            group_guard_directional_derivative(direction)
            if group_guard_directional_derivative is not None
            else {}
        )
        with torch.no_grad():
            for _ in range(int(max_trials)):
                changed = False
                for parameter, value, delta in zip(parameters, original, direction):
                    candidate = value + scale * delta
                    changed = changed or not torch.equal(candidate, value)
                    parameter.copy_(candidate)
                if not changed:
                    report["resolution_limited_trials"] += 1
                    break
                if not bool(
                    torch.stack([torch.isfinite(p).all() for p in parameters]).all()
                ):
                    report["nonfinite_parameter_trials"] += 1
                    scale *= 0.5
                    continue

                candidate_loss, candidate_groups = evaluate_closure()
                report["trial_evaluations"] += 1
                report["minimum_audited_scale"] = (
                    scale
                    if report["minimum_audited_scale"] is None
                    else min(report["minimum_audited_scale"], scale)
                )
                required = max(minimum_decrease, -ARMIJO_FACTOR * scale * slope)
                loss_ok = math.isfinite(candidate_loss) and before - candidate_loss >= required
                violations = subgroup_violations(candidate_groups) if loss_ok else {}
                guard_ok = not violations
                residual_delta = (
                    {
                        key: candidate_groups[key] - guard_reference[key]
                        for key in guard_reference
                    }
                    if guard_enabled
                    else {}
                )
                current_delta = (
                    {
                        key: candidate_groups[key] - guard_before[key]
                        for key in guard_before
                    }
                    if guard_enabled
                    else {}
                )
                metric_audit = {}
                if guard_enabled:
                    for key in guard_before:
                        baseline = guard_reference[key]
                        allowance = max(
                            abs(baseline) * guard_relative_tolerance[key],
                            guard_absolute_tolerance[key],
                        )
                        allowed = baseline + allowance
                        predicted_directional = predicted_derivatives.get(key)
                        closure_secant_directional = (
                            current_delta[key] / scale
                        )
                        metric_audit[key] = {
                            **guard_metadata[key],
                            "fixed_anchor": baseline,
                            "current": guard_before[key],
                            "candidate": candidate_groups[key],
                            "allowed": allowed,
                            "absolute_upper_limit": allowed,
                            "remaining_margin_before": allowed - guard_before[key],
                            "remaining_margin_candidate": (
                                allowed - candidate_groups[key]
                            ),
                            "directional_derivative": (
                                predicted_directional
                                if predicted_directional is not None
                                else closure_secant_directional
                            ),
                            "directional_derivative_source": (
                                "exact_guard_autograd"
                                if predicted_directional is not None
                                else "real_closure_secant"
                            ),
                            "closure_secant_directional_derivative": (
                                closure_secant_directional
                            ),
                            "linear_predicted_delta": (
                                scale * predicted_directional
                                if predicted_directional is not None
                                else None
                            ),
                            "actual_residual_delta": current_delta[key],
                        }
                improvement_deltas = {
                    key: guard_before[key] - candidate_groups[key]
                    for key in required_guard_improvement_keys
                }
                measurable_improvement = (
                    not required_guard_improvement_keys
                    or any(
                        improvement > max(
                            1.0e-12,
                            abs(guard_before[key]) * 1.0e-9,
                            guard_absolute_tolerance[key] * 1.0e-6,
                        )
                        for key, improvement in improvement_deltas.items()
                    )
                )
                scale_effective = scale > minimum_effective_scale
                report["trials"].append(
                    {
                        "direction": name,
                        "scale": scale,
                        "loss": candidate_loss if math.isfinite(candidate_loss) else None,
                        "required_decrease": required,
                        "directional_derivative": slope,
                        "group_guard_passed": guard_ok if guard_enabled and loss_ok else None,
                        "group_guard_violations": violations,
                        "group_guard_residual_delta": residual_delta,
                        "group_guard_current_delta": current_delta,
                        "group_guard_blocking_reasons": sorted(violations),
                        "group_guard_metric_audit": metric_audit,
                        "guard_improvement_deltas": improvement_deltas,
                        "measurable_guard_improvement": measurable_improvement,
                        "effective_step_scale": scale_effective,
                    }
                )
                if report["trial_evaluations"] == 1:
                    report["first_trial_loss"] = (
                        candidate_loss if math.isfinite(candidate_loss) else None
                    )
                if (
                    loss_ok
                    and guard_ok
                    and measurable_improvement
                    and scale_effective
                ):
                    report.update(
                        loss_after=candidate_loss,
                        optimizer_update_accepted=True,
                        direction=name,
                        step_scale=scale,
                        reason=(
                            "full_cycle_feasibility_guard_loss_decreased"
                            if guard_enabled
                            else "same_batch_loss_decreased"
                        ),
                        group_guard_after=(candidate_groups if guard_enabled else None),
                        minimum_accepted_scale=scale,
                        minimum_acceptable_scale=scale,
                    )
                    return True
                if loss_ok and guard_ok and (
                    not measurable_improvement or not scale_effective
                ):
                    report["resolution_limited_trials"] += 1
                    report["resolution_limited_under_exact_guard"] = True
                    break
                if loss_ok and violations:
                    report["group_guard_rejected_trials"] += 1
                    report["group_guard_last_violations"] = violations
                report["loss_rejected_trials"] += 1
                report["nonfinite_trials"] += int(not math.isfinite(candidate_loss))
                report["insufficient_decrease_trials"] += int(
                    math.isfinite(candidate_loss) and candidate_loss < before and not loss_ok
                )
                curvature = candidate_loss - before - scale * slope
                proposal = (
                    -slope * scale * scale / (2.0 * curvature)
                    if math.isfinite(curvature) and curvature > 0
                    else scale * 0.5
                )
                scale = min(scale * 0.5, max(scale * 0.01, proposal))
        return False

    try:
        optimizer.step()
        direction = [p.detach() - value for p, value in zip(parameters, original)]
        slope = derivative(direction)
        report["adam_directional_derivative"] = slope if math.isfinite(slope) else None
        saved_scales = [
            float(group.get(_SCALE_KEY, 1.0)) for group in optimizer.param_groups
        ]
        if any(
            not math.isfinite(scale) or not 0 < scale <= 1 for scale in saved_scales
        ):
            raise ValueError("invalid persisted Refiner trial scale")
        accepted = search(direction, 1.0, "adam")
        if not accepted:
            report["used_gradient_rescue"] = True
            direction = [
                -rate * g / maximum_gradient for rate, g in zip(rates, gradients)
            ]
            accepted = search(direction, 1.0, "current_gradient")
            if accepted:
                optimizer.state.clear()
        if accepted:
            for group in optimizer.param_groups:
                group[_SCALE_KEY] = report["step_scale"]
        else:
            restore()
            report["reason"] = (
                "resolution_limited_under_exact_guard"
                if report["resolution_limited_under_exact_guard"]
                else "bounded_search_no_descent"
            )
    except BaseException:
        restore()
        raise
    return report

def record_update(summary, update):
    """Accumulate EVERY attempted training step, not only printed samples."""
    summary["protocol"] = REFINER_UPDATE_PROTOCOL
    counts = {
        "attempted_steps": 1,
        "accepted_steps": int(update["optimizer_update_accepted"]),
        "retained_steps": int(not update["optimizer_update_accepted"]),
        "gradient_rescue_steps": int(update["used_gradient_rescue"]),
        "trial_evaluations": int(update["trial_evaluations"]),
        "nonfinite_trials": int(update["nonfinite_trials"]),
        "insufficient_decrease_trials": int(update.get("insufficient_decrease_trials", 0)),
        "group_guard_rejected_trials": int(update.get("group_guard_rejected_trials", 0)),
        "resolution_limited_steps": int(
            update.get("resolution_limited_under_exact_guard", False)
        ),
        "accepted_non_descent_steps": int(update["optimizer_update_accepted"] and
                                           update["loss_after"] >= update["loss_before"]),
    }
    for name, value in counts.items():
        summary[name] = summary.get(name, 0) + value
    reasons = dict(summary.get("group_guard_rejection_reasons", {}))
    categories = dict(summary.get("group_guard_rejection_categories", {}))
    for trial in update.get("trials", []):
        for reason in trial.get("group_guard_blocking_reasons", []):
            reasons[reason] = reasons.get(reason, 0) + 1
            category = (
                update.get("guard_metric_metadata", {})
                .get(reason, {})
                .get("category", "unclassified")
            )
            categories[category] = categories.get(category, 0) + 1
    summary["group_guard_rejection_reasons"] = dict(
        sorted(reasons.items())
    )
    summary["group_guard_rejection_categories"] = dict(
        sorted(categories.items())
    )
    audited_scale = update.get("minimum_audited_scale")
    if audited_scale is not None:
        current = summary.get("minimum_audited_scale")
        summary["minimum_audited_scale"] = (
            float(audited_scale)
            if current is None
            else min(float(current), float(audited_scale))
        )
    accepted_scale = update.get("minimum_accepted_scale")
    if accepted_scale is not None:
        current = summary.get("minimum_accepted_scale")
        summary["minimum_accepted_scale"] = (
            float(accepted_scale)
            if current is None
            else min(float(current), float(accepted_scale))
        )
        accepted_scales = list(summary.get("accepted_step_scales", []))
        accepted_scales.append(float(update["step_scale"]))
        summary["accepted_step_scales"] = accepted_scales
        ordered = sorted(accepted_scales)
        midpoint = len(ordered) // 2
        median = (
            ordered[midpoint]
            if len(ordered) % 2
            else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
        )
        summary["accepted_step_scale_distribution"] = {
            "count": len(ordered),
            "minimum": ordered[0],
            "median": median,
            "maximum": ordered[-1],
        }


def validate_update_summary(summary, expected_steps):
    """Require complete optimization accounting, not a claim of repair quality."""
    if summary.get("protocol") != REFINER_UPDATE_PROTOCOL:
        raise RuntimeError("missing or mismatched Refiner optimizer update protocol")
    for key in ("attempted_steps", "accepted_steps", "retained_steps",
                "trial_evaluations", "accepted_non_descent_steps"):
        if type(summary.get(key)) is not int or summary[key] < 0:
            raise RuntimeError(f"incomplete Refiner optimizer accounting: {key}")
    if (summary["attempted_steps"] != expected_steps
            or summary["accepted_steps"] + summary["retained_steps"] != expected_steps
            or summary["trial_evaluations"] < summary["accepted_steps"]
            or summary["accepted_non_descent_steps"] != 0):
        raise RuntimeError("inconsistent or non-descent Refiner optimizer updates")
