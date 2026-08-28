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


REFINER_UPDATE_PROTOCOL = "same_batch_descent_backtracking_v1"
MAX_BACKTRACK_TRIALS = 12  # per direction; at most 24 extra forward evaluations
_SCALE_KEY = "refiner_trial_scale"  # persisted by optimizer.state_dict()


def checked_refiner_step(optimizer, loss, closure, *, max_trials=MAX_BACKTRACK_TRIALS):
    """After backward/gradient clipping, accept only a finite strict decrease.

    Try the Adam proposal first, then current steepest descent if momentum is
    uphill or its finite search fails. Backtracking never samples a new batch.
    An accepted scaled Adam proposal retains its gradient moments. An accepted
    steepest-descent rescue resets stale moments. A rejected/exceptional update
    restores BOTH parameters and complete optimizer state, including scale.
    """
    if not 1 <= int(max_trials) <= MAX_BACKTRACK_TRIALS:
        raise ValueError(f"max_trials must be in [1,{MAX_BACKTRACK_TRIALS}]")
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
    report = {
        "protocol": REFINER_UPDATE_PROTOCOL, "loss_before": before,
        "loss_after": before, "optimizer_update_accepted": False,
        "direction": "none", "step_scale": 0.0, "trial_evaluations": 0,
        "loss_rejected_trials": 0, "nonfinite_trials": 0,
        "nonfinite_parameter_trials": 0,
        "resolution_limited_trials": 0, "first_trial_loss": None,
        "adam_directional_derivative": None, "used_gradient_rescue": False,
        "max_trials_per_direction": int(max_trials), "scientific_acceptance": False,
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
        return float(torch.stack([(g * delta).sum() for g, delta in zip(gradients, direction)]).sum())

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
                if not bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()):
                    report["nonfinite_parameter_trials"] += 1
                    scale *= .5
                    continue
                candidate_loss = float(closure().detach())
                report["trial_evaluations"] += 1
                if report["trial_evaluations"] == 1:
                    report["first_trial_loss"] = candidate_loss if math.isfinite(candidate_loss) else None
                if math.isfinite(candidate_loss) and candidate_loss < before:
                    report.update(loss_after=candidate_loss, optimizer_update_accepted=True,
                                  direction=name, step_scale=scale, reason="same_batch_loss_decreased")
                    return True
                report["loss_rejected_trials"] += 1
                report["nonfinite_trials"] += int(not math.isfinite(candidate_loss))
                scale *= .5
        return False

    try:
        optimizer.step()
        direction = [p.detach() - value for p, value in zip(parameters, original)]
        slope = derivative(direction)
        report["adam_directional_derivative"] = slope if math.isfinite(slope) else None
        prior_scale = min(float(group.get(_SCALE_KEY, 1.0)) for group in optimizer.param_groups)
        if not math.isfinite(prior_scale) or not 0 < prior_scale <= 1:
            raise ValueError("invalid persisted Refiner trial scale")
        accepted = search(direction, min(1.0, 2.0 * prior_scale), "adam")
        if not accepted:
            report["used_gradient_rescue"] = True
            direction = [-rate * g / maximum_gradient for rate, g in zip(rates, gradients)]
            accepted = search(direction, 1.0, "current_gradient")
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
