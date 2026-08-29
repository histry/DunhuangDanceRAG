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


REFINER_UPDATE_PROTOCOL = "full_cycle_component_guard_pareto_rescue_v7"
MAX_BACKTRACK_TRIALS = 12  # per direction; at most 24 extra forward evaluations
ARMIJO_FACTOR = 1.0e-4
MIN_RELATIVE_DECREASE = 1.0e-8  # optimization progress, NOT a motion-quality gate
_SCALE_KEY = "refiner_trial_scale"  # persisted by optimizer.state_dict()


PARETO_MAX_ITERATIONS = 96
PARETO_WEIGHT_EPS = 1.0e-8
PARETO_NORM_EPS = 1.0e-10
PARETO_COMMON_DOT_EPS = 1.0e-8


def _minimum_norm_simplex_weights(
    gram,
    *,
    max_iterations=PARETO_MAX_ITERATIONS,
):
    """Minimum-norm convex combination from a small gradient Gram matrix.

    Deterministic Frank-Wolfe is sufficient here because the Refiner guard
    contributes at most 12 subgroup/component objectives plus the global loss.
    """
    gram = torch.as_tensor(
        gram,
        dtype=torch.float64,
        device="cpu",
    )

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Pareto Gram matrix must be square")

    count = int(gram.shape[0])

    if count < 1:
        raise ValueError("Pareto Gram matrix cannot be empty")

    if not bool(torch.isfinite(gram).all()):
        raise FloatingPointError("nonfinite Pareto Gram matrix")

    # Remove tiny numerical asymmetry from independently accumulated dots.
    gram = 0.5 * (gram + gram.T)

    alpha = torch.zeros(count, dtype=torch.float64)

    # Deterministic initial vertex: smallest individual gradient norm.
    alpha[int(torch.argmin(torch.diag(gram)))] = 1.0

    for _ in range(int(max_iterations)):
        gram_alpha = gram @ alpha

        # Frank-Wolfe linear minimization oracle on the simplex.
        vertex = int(torch.argmin(gram_alpha))

        direction = -alpha.clone()
        direction[vertex] += 1.0

        gram_direction = gram @ direction

        denominator = float(direction @ gram_direction)

        if denominator <= 1.0e-18:
            break

        numerator = -float(direction @ gram_alpha)

        gamma = max(
            0.0,
            min(1.0, numerator / denominator),
        )

        if gamma <= 1.0e-12:
            break

        alpha = alpha + gamma * direction

    alpha = alpha.clamp_min(0.0)

    total = float(alpha.sum())

    if not total > 0.0:
        raise RuntimeError("invalid Pareto simplex weights")

    return alpha / total


def checked_refiner_step(
    optimizer,
    loss,
    closure,
    *,
    max_trials=MAX_BACKTRACK_TRIALS,
    gradient_unscale=1.0,
    group_guard_before=None,
    group_guard_relative_tolerance=0.0,
    group_guard_absolute_tolerance=0.0,
):
    """Transactional Armijo step with optional subgroup non-regression.

    With ``group_guard_before`` disabled this is the V11 same-batch optimizer.
    With the guard enabled, ``closure`` MUST return ``(loss, group_losses)``.
    A trial is accepted only when the scalar Armijo condition passes AND every
    named subgroup/component guard stays within its pre-update
    relative/absolute allowance.
    Parameters and the complete optimizer state are restored on rejection.
    """
    if not 1 <= int(max_trials) <= MAX_BACKTRACK_TRIALS:
        raise ValueError(f"max_trials must be in [1,{MAX_BACKTRACK_TRIALS}]")
    if not math.isfinite(gradient_unscale) or gradient_unscale < 1.0:
        raise ValueError("gradient_unscale must be finite and >= 1")
    for value, name in (
        (group_guard_relative_tolerance, "group_guard_relative_tolerance"),
        (group_guard_absolute_tolerance, "group_guard_absolute_tolerance"),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    def scalar(value):
        if torch.is_tensor(value):
            return float(value.detach())
        return float(value)

    guard_enabled = group_guard_before is not None
    guard_before = {}
    if guard_enabled:
        if not hasattr(group_guard_before, "items") or not group_guard_before:
            raise ValueError("group_guard_before must be a non-empty mapping")
        guard_before = {str(k): scalar(v) for k, v in group_guard_before.items()}
        if not all(math.isfinite(v) for v in guard_before.values()):
            raise FloatingPointError("nonfinite subgroup loss before optimizer update")

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
        "group_guard_relative_tolerance": float(group_guard_relative_tolerance),
        "group_guard_absolute_tolerance": float(group_guard_absolute_tolerance),
        "group_guard_before": guard_before,
        "group_guard_after": None,
        "group_guard_rejected_trials": 0,
        "group_guard_last_violations": {},
        "pareto_rescue_attempted": False,
        "pareto_rescue_feasible": None,
        "pareto_rescue_accepted": False,
        "pareto_rescue_objectives": [],
        "pareto_rescue_coefficients": {},
        "pareto_rescue_min_norm_sq": None,
        "pareto_rescue_worst_normalized_dot": None,
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
        for key, baseline in guard_before.items():
            candidate = candidate_groups[key]
            allowance = max(
                abs(baseline) * float(group_guard_relative_tolerance),
                float(group_guard_absolute_tolerance),
            )
            allowed = baseline + allowance
            if not math.isfinite(candidate) or candidate > allowed:
                violations[key] = {
                    "before": baseline,
                    "candidate": candidate if math.isfinite(candidate) else None,
                    "allowed": allowed,
                }
        return violations

    def pareto_common_descent_direction():
        """Construct a first-order common descent direction for all guards.

        The direction is only a proposal. The existing full-cycle Armijo and
        component guard still make the final transactional decision.
        """
        if not guard_enabled:
            return None

        report["pareto_rescue_attempted"] = True

        # Adam/backtracking may currently leave parameters at a rejected
        # candidate. Pareto gradients must be evaluated at the exact original
        # transaction state.
        restore()

        result = closure()

        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "Pareto Refiner closure must return (loss, subgroup_losses)"
            )

        total_loss, raw_groups = result

        if not torch.is_tensor(total_loss):
            raise RuntimeError(
                "Pareto Refiner total loss must be a tensor"
            )

        if not hasattr(raw_groups, "items"):
            raise RuntimeError(
                "Pareto Refiner subgroup output must be a mapping"
            )

        if set(str(k) for k in raw_groups) != set(guard_before):
            raise RuntimeError(
                "Refiner subgroup guard keys changed during Pareto rescue"
            )

        # Confirm that the differentiable rescue closure is the exact same
        # deterministic transaction as the pre-update guard snapshot.
        for key, tensor_value in raw_groups.items():
            key = str(key)
            observed = scalar(tensor_value)
            baseline = guard_before[key]
            tolerance = max(
                1.0e-10,
                abs(baseline) * 1.0e-8,
            )

            if (
                not math.isfinite(observed)
                or abs(observed - baseline) > tolerance
            ):
                raise RuntimeError(
                    f"Pareto rescue closure changed guard baseline: {key}"
                )

        objectives = [
            ("global_total", total_loss),
            *[
                (str(key), raw_groups[key])
                for key in sorted(raw_groups, key=str)
            ],
        ]

        normalized_rows = []
        active_names = []

        for name, objective in objectives:
            if not torch.is_tensor(objective):
                raise RuntimeError(
                    f"Pareto objective {name!r} is not a tensor"
                )

            if not objective.requires_grad:
                continue

            raw = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )

            row = [
                (
                    gradient.detach()
                    if gradient is not None
                    else torch.zeros_like(parameter)
                )
                for parameter, gradient in zip(parameters, raw)
            ]

            norm_sq = sum(
                float(
                    gradient.double().square().sum()
                )
                for gradient in row
            )

            if not math.isfinite(norm_sq):
                raise FloatingPointError(
                    f"nonfinite Pareto gradient norm for {name}"
                )

            if norm_sq <= 1.0e-24:
                # Zero-gradient constraint cannot define a useful first-order
                # half-space. The post-update exact guard still protects it.
                continue

            norm = math.sqrt(norm_sq)

            normalized_rows.append([
                gradient / norm
                for gradient in row
            ])
            active_names.append(name)

        report["pareto_rescue_objectives"] = list(active_names)

        if len(normalized_rows) < 2:
            report["pareto_rescue_feasible"] = False
            return None

        count = len(normalized_rows)

        gram = torch.empty(
            (count, count),
            dtype=torch.float64,
        )

        for i in range(count):
            for j in range(i, count):
                dot = sum(
                    float(
                        (
                            left.double()
                            * right.double()
                        ).sum()
                    )
                    for left, right in zip(
                        normalized_rows[i],
                        normalized_rows[j],
                    )
                )

                gram[i, j] = dot
                gram[j, i] = dot

        weights = _minimum_norm_simplex_weights(gram)

        gram_weights = gram @ weights

        min_norm_sq = float(
            weights @ gram_weights
        )

        worst_dot = float(
            torch.min(gram_weights)
        )

        report["pareto_rescue_min_norm_sq"] = min_norm_sq
        report["pareto_rescue_worst_normalized_dot"] = worst_dot

        report["pareto_rescue_coefficients"] = {
            name: float(weight)
            for name, weight in zip(active_names, weights)
            if float(weight) > PARETO_WEIGHT_EPS
        }

        # For the minimum-norm convex combination g*, a strict common
        # descent direction -g* requires every normalized objective gradient
        # to have positive dot product with g*.
        feasible = (
            math.isfinite(min_norm_sq)
            and math.isfinite(worst_dot)
            and min_norm_sq > PARETO_NORM_EPS
            and worst_dot > PARETO_COMMON_DOT_EPS
        )

        report["pareto_rescue_feasible"] = bool(feasible)

        if not feasible:
            return None

        mixed = []

        for parameter_index, parameter in enumerate(parameters):
            value = torch.zeros_like(parameter)

            for weight, row in zip(weights, normalized_rows):
                value = value + float(weight) * row[parameter_index]

            mixed.append(value)

        maximum = max(
            float(value.abs().max())
            for value in mixed
        )

        if not math.isfinite(maximum) or maximum <= 0.0:
            report["pareto_rescue_feasible"] = False
            return None

        positive_rates = [
            rate for rate in rates
            if math.isfinite(rate) and rate > 0.0
        ]

        if not positive_rates:
            raise RuntimeError(
                "Refiner optimizer has no positive learning rate"
            )

        # A single positive scalar preserves every common-descent sign.
        # Backtracking remains responsible for selecting the actual step.
        base_rate = min(positive_rates)

        direction = [
            -base_rate * value / maximum
            for value in mixed
        ]

        # Fail closed if numerical reconstruction lost global descent.
        slope = derivative(direction)

        if not math.isfinite(slope) or slope >= 0.0:
            report["pareto_rescue_feasible"] = False
            return None

        return direction


    def search(direction, scale, name):
        slope = derivative(direction)
        if not math.isfinite(slope) or slope >= 0:
            return False
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
                required = max(minimum_decrease, -ARMIJO_FACTOR * scale * slope)
                loss_ok = math.isfinite(candidate_loss) and before - candidate_loss >= required
                violations = subgroup_violations(candidate_groups) if loss_ok else {}
                guard_ok = not violations
                report["trials"].append(
                    {
                        "direction": name,
                        "scale": scale,
                        "loss": candidate_loss if math.isfinite(candidate_loss) else None,
                        "required_decrease": required,
                        "directional_derivative": slope,
                        "group_guard_passed": guard_ok if guard_enabled and loss_ok else None,
                        "group_guard_violations": violations,
                    }
                )
                if report["trial_evaluations"] == 1:
                    report["first_trial_loss"] = (
                        candidate_loss if math.isfinite(candidate_loss) else None
                    )
                if loss_ok and guard_ok:
                    report.update(
                        loss_after=candidate_loss,
                        optimizer_update_accepted=True,
                        direction=name,
                        step_scale=scale,
                        reason=(
                            "full_cycle_component_guard_loss_decreased"
                            if guard_enabled
                            else "same_batch_loss_decreased"
                        ),
                        group_guard_after=(candidate_groups if guard_enabled else None),
                    )
                    return True
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

            pareto_direction = pareto_common_descent_direction()

            if pareto_direction is not None:
                accepted = search(
                    pareto_direction,
                    1.0,
                    "pareto_common_descent",
                )

                report["pareto_rescue_accepted"] = bool(accepted)

                if accepted:
                    # This update did not follow Adam's moment direction.
                    optimizer.state.clear()

            if not accepted:
                # Preserve the V13 current-gradient rescue as the final
                # fallback. Start again from the exact transaction state.
                restore()

                direction = [
                    -rate * g / maximum_gradient
                    for rate, g in zip(rates, gradients)
                ]

                accepted = search(
                    direction,
                    1.0,
                    "current_gradient",
                )

                if accepted:
                    optimizer.state.clear()
        if accepted:
            for group in optimizer.param_groups:
                group[_SCALE_KEY] = report["step_scale"]
        else:
            restore()
            report["reason"] = "bounded_search_no_descent"
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
        "pareto_rescue_steps": int(bool(update.get("pareto_rescue_attempted", False))),
        "pareto_feasible_steps": int(bool(update.get("pareto_rescue_feasible", False))),
        "pareto_accepted_steps": int(bool(update.get("pareto_rescue_accepted", False))),
        "accepted_non_descent_steps": int(update["optimizer_update_accepted"] and
                                           update["loss_after"] >= update["loss_before"]),
    }
    for name, value in counts.items():
        summary[name] = summary.get(name, 0) + value


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
