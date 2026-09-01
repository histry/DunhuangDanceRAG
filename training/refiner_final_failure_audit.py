"""Read-only attribution of the fixed A0 final failure and contact path.

This audit never constructs an optimizer, changes a model parameter, selects a
checkpoint, or authorizes Pilot.  It consumes the already fixed step-400 A0
trajectory, the original frozen TRAIN source and its immutable probe artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from training import motion_models as m
from training import refiner_bridge_diagnostics as d
from training import refiner_group_gradient_audit as a
from training import refiner_safe_start_diagnostics as safe
from training import refiner_zero_start_trajectory as trajectory
from training.bridge_feasibility import group_decisions


SCHEMA = "refiner_final_failure_contact_audit_v1"
TRAJECTORY_COMMIT = "b2d71e1fa92cb2a6723810060722c0edea7a3a99"
PHYSICAL_KEYS = (
    "joint_jerk_mps3_p95",
    "joint_jerk_mps3_max",
    "joint_jerk_window_p95_max_mps3",
    "extremity_jerk_mps3_p95",
    "extremity_jerk_window_p95_max_mps3",
    "foot_support_drift_m_p95",
    "foot_support_drift_m_max",
    "foot_penetration_min_m",
)
HEAD_BLOCKS = (("contact", 0, 4), ("root", 4, 7), ("joint", 7, 79))


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"nonfinite {label}")
    return value


def tensor_stats(value):
    value = value.detach().double()
    numel = value.numel()
    norm = float(value.norm())
    return {
        "numel": numel,
        "norm": norm,
        "rms": norm / math.sqrt(numel) if numel else 0.0,
        "abs_max": float(value.abs().max()) if numel else 0.0,
        "zero_fraction": float((value == 0).double().mean()) if numel else None,
        "exactly_zero": bool((value == 0).all()),
    }


def mask_stats(value):
    value = value.detach().double()
    numel = value.numel()
    nonzero = int(torch.count_nonzero(value))
    return {
        "numel": numel,
        "nonzero_count": nonzero,
        "nonzero_fraction": nonzero / numel if numel else None,
        "mean": float(value.mean()) if numel else None,
        "rms": float(value.square().mean().sqrt()) if numel else None,
        "max": float(value.max()) if numel else None,
        "exactly_zero": nonzero == 0,
    }


def _block_stats(weight, bias):
    result = {}
    for name, lo, hi in HEAD_BLOCKS:
        result[name] = {
            "weight": tensor_stats(weight[lo:hi]),
            "bias": tensor_stats(bias[lo:hi]),
            "combined": tensor_stats(torch.cat((weight[lo:hi].reshape(-1), bias[lo:hi].reshape(-1)))),
            "row_range": [lo, hi],
        }
    return result


def _gradient_or_zero(objective, targets, *, retain_graph):
    if not objective.requires_grad:
        return [torch.zeros_like(value) for value in targets]
    values = torch.autograd.grad(objective, targets, retain_graph=retain_graph, allow_unused=True)
    return [torch.zeros_like(target) if value is None else value for target, value in zip(targets, values)]


@contextmanager
def _capture_output(model):
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output)

    handle = model.out.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def preserve_model_runtime(model):
    """Restore mode, existing gradients and hook registries after an audit."""
    modes = {module: module.training for module in model.modules()}
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters()}
    hook_ids = {module: (tuple(module._forward_hooks), tuple(module._forward_pre_hooks),
                         tuple(module._backward_hooks)) for module in model.modules()}
    try:
        yield
    finally:
        for module, mode in modes.items():
            module.training = mode
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
    for module, expected in hook_ids.items():
        observed = (tuple(module._forward_hooks), tuple(module._forward_pre_hooks),
                    tuple(module._backward_hooks))
        if observed != expected:
            raise RuntimeError("read-only audit changed model hooks")
    for name, parameter in model.named_parameters():
        expected = gradients[name]
        if expected is None and parameter.grad is not None:
            raise RuntimeError("read-only audit left parameter.grad residue")
        if expected is not None and not torch.equal(expected, parameter.grad):
            raise RuntimeError("read-only audit changed an existing parameter.grad")


def true_contact_gradients(model, batch, cfg):
    """Ordinary autograd for the three current scalar TRAIN objectives."""
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    modes = {module: module.training for module in model.modules()}
    old_grads = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters()}
    result = {}
    try:
        model.eval()
        with torch.enable_grad(), _capture_output(model) as captured:
            repair, clean, terms, identity_terms = m._refiner_batch_objectives(model, batch, cfg)
            total = repair + cfg.product_refiner_clean_identity_weight * clean
            if len(captured) != 1:
                raise RuntimeError("contact gradient audit requires exactly one model forward")
            raw_output = captured[0]
            targets = [model.out.weight, model.out.bias, raw_output]
            objectives = (("repair_objective", repair), ("clean_identity_objective", clean),
                          ("training_total", total))
            for index, (name, objective) in enumerate(objectives):
                weight, bias, output = _gradient_or_zero(
                    objective, targets, retain_graph=index + 1 < len(objectives))
                count = batch["clean"].shape[0]
                result[name] = {
                    "value": _finite(objective.detach(), name),
                    "head_parameter_gradients": _block_stats(weight, bias),
                    "raw_output_gradients": {
                        "repair_branch": {block: tensor_stats(output[:count, ..., lo:hi])
                                          for block, lo, hi in HEAD_BLOCKS},
                        "clean_branch": {block: tensor_stats(output[count:, ..., lo:hi])
                                         for block, lo, hi in HEAD_BLOCKS},
                    },
                }
            result["current_terms"] = {
                "repair": {key: _finite(value.detach(), key) for key, value in terms.items()},
                "clean_identity": {key: _finite(value.detach(), key)
                                   for key, value in identity_terms.items()},
            }
    finally:
        for module, mode in modes.items():
            module.training = mode
        for name, parameter in model.named_parameters():
            parameter.grad = old_grads[name]
    if any(not torch.equal(state[name], value) for name, value in model.state_dict().items()):
        raise RuntimeError("true-gradient contact audit changed model state")
    for name, parameter in model.named_parameters():
        expected = old_grads[name]
        if expected is None and parameter.grad is not None:
            raise RuntimeError("contact audit left parameter.grad residue")
        if expected is not None and not torch.equal(expected, parameter.grad):
            raise RuntimeError("contact audit changed an existing parameter.grad")
    return result


def _branch_masks(batch, cfg, branch):
    prefix = "" if branch == "repair" else "clean_"
    return m._refiner_decode_masks(
        batch[prefix + "joint"], batch[prefix + "root"], batch[prefix + "contact"],
        batch["seam"], cfg)


def _group_mask_rows(batch, cfg, *, source, role=None):
    rows = []
    for branch in ("repair", "clean"):
        effective_joint, effective_root, effective = _branch_masks(batch, cfg, branch)
        source_mask = batch[("" if branch == "repair" else "clean_") + "contact"]
        if "group" in batch:
            groups = [(a.GROUPS[index], batch["group"] == index) for index in range(4)]
        else:
            widths = (batch["seam"][..., 0] >= .5).sum(1)
            groups = [(f"{role}/10", widths == 10), (f"{role}/28", widths == 28)]
        for group, selected in groups:
            if not bool(selected.any()):
                continue
            rows.append({
                "source": source, "branch": branch, "group": group,
                "cases": int(selected.sum()),
                "source_risk_mask": mask_stats(source_mask[selected]),
                "effective_decoder_mask": mask_stats(effective[selected]),
                "effective_root_mask": mask_stats(effective_root[selected]),
                "effective_joint_mask": mask_stats(effective_joint[selected]),
            })
    return rows


def final_banks(bank, probe, cfg):
    banks = {("seen", role): {key: value[offset:offset + 16].to(cfg.device)
                              for key, value in bank["anchor"].items() if key != "group"}
             for role, offset in (("single_recording", 0), ("cross_event", 16))}
    banks.update({("new_position", role): {key: value.to(cfg.device) for key, value in part.items()}
                  for role, part in probe.items()})
    return banks


def contact_mask_audit(train_batch, banks, cfg):
    rows = _group_mask_rows(train_batch, cfg, source="train_transaction_0")
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            rows.extend(_group_mask_rows(banks[(split, role)], cfg, source=split, role=role))
    return {
        "rows": rows,
        "all_effective_masks_exactly_zero": all(row["effective_decoder_mask"]["exactly_zero"] for row in rows),
        "note": "Effective decoder mask equals the frozen risk mask times the unchanged core/transition strength.",
    }


def _jacobian_group_rows(output_gradient, effective_mask, batch, branch, *, source, role=None):
    rows = []
    if "group" in batch:
        groups = [(a.GROUPS[index], batch["group"] == index) for index in range(4)]
    else:
        widths = (batch["seam"][..., 0] >= .5).sum(1)
        groups = [(f"{role}/10", widths == 10), (f"{role}/28", widths == 28)]
    for group, selected in groups:
        values = output_gradient[selected, ..., :4]
        active = effective_mask[selected].expand_as(values) != 0
        rows.append({
            "source": source, "branch": branch, "group": group, "cases": int(selected.sum()),
            "all_positions": tensor_stats(values),
            "effective_mask": mask_stats(effective_mask[selected]),
            "active_positions": tensor_stats(values[active]) if bool(active.any()) else tensor_stats(values[:0]),
        })
    return rows


def _actual_decoder_jacobian_rows(model, batch, cfg, *, source, role=None):
    count = batch["clean"].shape[0]
    with torch.no_grad():
        raw = model(
            torch.cat((batch["bad"], batch["clean"])),
            torch.cat((batch["cond"], batch.get("clean_cond", batch["cond"]))),
            torch.cat((batch["seam"], batch["seam"])),
            torch.cat((batch["joint"], batch["clean_joint"])),
        )
    rows = []
    for branch, reference, part in (("repair", batch["bad"], raw[:count]),
                                    ("clean", batch["clean"], raw[count:])):
        output = part.detach().requires_grad_(True)
        masks = _branch_masks(batch, cfg, branch)
        decoded = m._decode_product_refiner_output(reference, output, *masks, cfg)
        gradient = torch.autograd.grad(decoded[..., :4].sum(), output)[0]
        rows.extend(_jacobian_group_rows(
            gradient, masks[2], batch, branch, source=source, role=role))
    return rows


def contact_decoder_jacobian(model, batch, cfg, *, banks=None):
    """Actual decoder VJPs plus zero/nonzero-mask controls and finite difference."""
    rows = _actual_decoder_jacobian_rows(
        model, batch, cfg, source="train_transaction_0")
    if banks is not None:
        for split in ("seen", "new_position"):
            for role in ("single_recording", "cross_event"):
                rows.extend(_actual_decoder_jacobian_rows(
                    model, banks[(split, role)], cfg, source=split, role=role))

    reference = batch["bad"][:1].detach().clone()
    reference[..., :4] = 0.5
    output = torch.zeros((1, reference.shape[1], m.PRODUCT_STATE_DIM),
                         dtype=reference.dtype, device=reference.device, requires_grad=True)
    joint = torch.zeros((1, reference.shape[1], m.NUM_JOINTS), device=reference.device)
    root = torch.zeros((1, reference.shape[1], 1), device=reference.device)

    def control(contact):
        decoded = m._decode_product_refiner_output(reference, output, joint, root, contact, cfg)
        return torch.autograd.grad(decoded[..., :4].sum(), output, retain_graph=True)[0][..., :4]

    zero = control(torch.zeros_like(root))
    nonzero = control(torch.ones_like(root))
    frame = reference.shape[1] // 2
    epsilon = 1e-3
    with torch.no_grad():
        plus = output.detach().clone()
        minus = output.detach().clone()
        plus[0, frame, 0] += epsilon
        minus[0, frame, 0] -= epsilon
        mask = torch.ones_like(root)
        plus_value = m._decode_product_refiner_output(reference, plus, joint, root, mask, cfg)[0, frame, 0]
        minus_value = m._decode_product_refiner_output(reference, minus, joint, root, mask, cfg)[0, frame, 0]
    finite_difference = float((plus_value - minus_value) / (2 * epsilon))
    analytic = float(nonzero[0, frame, 0])
    return {
        "actual_paths": rows,
        "zero_mask_control": tensor_stats(zero),
        "nonzero_mask_control": tensor_stats(nonzero),
        "finite_difference_control": {
            "epsilon": epsilon, "analytic": analytic, "finite_difference": finite_difference,
            "absolute_error": abs(analytic - finite_difference),
            "autograd_finite_difference_agree": math.isclose(analytic, finite_difference, rel_tol=2e-3, abs_tol=2e-5),
        },
        "control_note": "Synthetic masks only test decoder connectivity; they do not alter the audited model, objective, or artifacts.",
    }


def _physical_values(audit):
    return {key: float(audit[key]) if key in audit and np.isfinite(audit[key]) else None
            for key in PHYSICAL_KEYS}


def _clean_gate(prediction, clean, cfg):
    accumulator = m._new_validation_physical_accumulator()
    m._record_validation_clean_identity_prediction(accumulator, prediction, clean, cfg)
    return accumulator["clean_identity_gates"][-1]


def _hidden_clean_geometry(reference, prediction, clean, seam, cfg):
    local = np.asarray(seam).reshape(-1) >= .5
    before = np.abs(m.product_log_np(clean, reference))[local]
    after = np.abs(m.product_log_np(clean, prediction))[local]
    before_value, after_value = float(before.mean()), float(after.mean())
    gain = ((before_value - after_value) / before_value if before_value > 1e-6
            else (1.0 if after_value <= 1e-6 else -1.0))
    threshold = float(cfg.checkpoint_validation_min_geometry_repair_gain)
    return {"available": True, "before": before_value, "after": after_value,
            "absolute_improvement": before_value - after_value, "relative_improvement": gain,
            "required_relative_improvement": threshold, "accepted": math.isfinite(gain) and gain >= threshold,
            "hidden_clean_used": True}


def case_failure_attribution(model, banks, cfg):
    rows = []
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for split in ("seen", "new_position"):
            for role in ("single_recording", "cross_event"):
                batch = banks[(split, role)]
                with torch.no_grad():
                    prediction, identity = m._refiner_batch_outputs(model, batch, cfg)
                arrays = [value.detach().cpu().numpy() for value in
                          (prediction, identity, batch["bad"], batch["clean"], batch["seam"])]
                for index, (pred, ident, reference, clean, seam) in enumerate(zip(*arrays)):
                    before_audit = m._safe_validation_audit(
                        reference, cfg, role="final_attribution_before", support_policy="source_observation")
                    after_audit = m._safe_validation_audit(
                        pred, cfg, role="final_attribution_after", support_policy="source_observation")
                    physical = m._fixed_support_stage_gate(
                        reference, pred, cfg, before_audit=before_audit, after_audit=after_audit)
                    observable = m._observable_boundary_audit(pred, reference, seam, cfg)
                    fidelity = observable["reference_fidelity"]
                    clean_gate = _clean_gate(ident, clean, cfg)
                    width = int(np.sum(seam >= .5))
                    geometry = (_hidden_clean_geometry(reference, pred, clean, seam, cfg)
                                if role == "single_recording" else {
                                    "available": False, "hidden_clean_used": False,
                                    "reason": "cross_event protocol forbids hidden-clean repair scoring"})
                    endpoint_before = float(observable["before"]["endpoint_velocity_jump_mps"])
                    endpoint_after = float(observable["after"]["endpoint_velocity_jump_mps"])
                    temporal_before = float(observable["before"]["temporal_energy"])
                    temporal_after = float(observable["after"]["temporal_energy"])
                    reasons = []
                    if not observable["endpoint_accepted"]:
                        reasons.append("endpoint/observable_endpoint_not_improved")
                    if not observable["temporal_accepted"]:
                        reasons.append("temporal/observable_temporal_not_improved")
                    reasons.extend("physical/" + str(reason) for reason in physical.get("reasons", []))
                    if not observable["reference_fidelity_accepted"]:
                        reasons.append("geometry/reference_fidelity_budget_exceeded")
                    if geometry.get("available") and not geometry["accepted"]:
                        reasons.append("geometry/no_meaningful_hidden_clean_repair_gain")
                    if role == "single_recording" and not clean_gate["accepted"]:
                        clean_reasons = list(clean_gate.get("reasons", [])) or ["rejected"]
                        reasons.extend("clean_identity/" + str(reason) for reason in clean_reasons)
                    rows.append({
                        "split": split, "role": role, "width": width, "group": f"{role}/{width}",
                        "case_index": index,
                        "endpoint": {"before": endpoint_before, "after": endpoint_after,
                            "absolute_improvement": endpoint_before - endpoint_after,
                            "relative_improvement": float(observable["endpoint_gain"]),
                            "required_relative_improvement": float(cfg.checkpoint_validation_min_endpoint_repair_gain),
                            "accepted": bool(observable["endpoint_accepted"])},
                        "temporal": {"before": temporal_before, "after": temporal_after,
                            "absolute_improvement": temporal_before - temporal_after,
                            "relative_improvement": float(observable["temporal_gain"]),
                            "required_relative_improvement": float(cfg.checkpoint_validation_min_temporal_repair_gain),
                            "jerk_before": float(observable["before"]["seam_jerk_mps3"]),
                            "jerk_after": float(observable["after"]["seam_jerk_mps3"]),
                            "jerk_required_max": float(observable["before"]["seam_jerk_mps3"] * 1.02 + 1e-6),
                            "jerk_non_regression": bool(observable["jerk_non_regression"]),
                            "accepted": bool(observable["temporal_accepted"])},
                        "geometry": {"repair_toward_hidden_clean": geometry,
                            "reference_fidelity": {"before": {"fk_p95_m": 0.0, "fk_max_m": 0.0,
                                                               "product_log_l1": 0.0},
                                "after": fidelity,
                                "required_max": {"fk_p95_m": float(cfg.checkpoint_validation_max_fk_p95_m),
                                    "fk_max_m": float(cfg.checkpoint_validation_max_fk_max_m),
                                    "product_log_l1": float(cfg.checkpoint_validation_max_refiner_product_log_l1)},
                                "accepted": bool(observable["reference_fidelity_accepted"]),
                                "hidden_clean_used": False}},
                        "physical": {"before": _physical_values(before_audit),
                                     "after": _physical_values(after_audit),
                                     "accepted": bool(physical["accepted"]),
                                     "reasons": list(physical.get("reasons", []))},
                        "clean_identity": {"evaluated": True,
                            "checkpoint_criterion_for_role": role == "single_recording",
                            "accepted": bool(clean_gate["accepted"]),
                            "product_log_l1": float(clean_gate["identity_detail"]["product_log_l1"]),
                            "maximum_product_log_l1": float(clean_gate["identity_detail"]["maximum_product_log_l1"]),
                            "contact_l1": float(clean_gate["identity_detail"]["contact_l1"]),
                            "maximum_contact_l1": float(clean_gate["identity_detail"]["maximum_contact_l1"]),
                            "reasons": list(clean_gate.get("reasons", []))},
                        "failure_reasons": list(dict.fromkeys(reasons)),
                    })
    finally:
        for module, mode in modes.items():
            module.training = mode
    return rows


def _mean(rows, path):
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def summarize_cases(cases, evaluated):
    tables, by_split, by_group = {}, Counter(), Counter()
    for split in ("seen", "new_position"):
        tables[split] = {}
        for group in ("single_recording/10", "single_recording/28", "cross_event/10", "cross_event/28"):
            selected = [row for row in cases if row["split"] == split and row["group"] == group]
            if len(selected) != 8:
                raise RuntimeError("final attribution requires four eight-case groups in both splits")
            original = evaluated["group_decisions"][split][group]
            required, temporal_required = original["required"], original["temporal_required"]
            reasons = Counter(reason for row in selected for reason in row["failure_reasons"])
            row = {
                "split": split, "group": group, "cases": 8, "passed": bool(original["passed"]),
                "endpoint_gate": {"passed_cases": sum(x["endpoint"]["accepted"] for x in selected),
                                  "required_cases": required},
                "temporal_gate": {"passed_cases": sum(x["temporal"]["accepted"] for x in selected),
                                  "required_cases": temporal_required},
                "geometry_gate": {"passed_cases": sum(x["geometry"]["reference_fidelity"]["accepted"] for x in selected),
                                  "required_cases": required,
                                  "hidden_clean_repair_cases": sum(bool(x["geometry"]["repair_toward_hidden_clean"].get("accepted"))
                                                                   for x in selected
                                                                   if x["geometry"]["repair_toward_hidden_clean"].get("available"))},
                "physical_gate": {"passed_cases": sum(x["physical"]["accepted"] for x in selected),
                                  "required_cases": required},
                "clean_identity_gate": {"passed_cases": sum(x["clean_identity"]["accepted"] for x in selected),
                                        "required_cases": math.ceil(8 * float(evaluated["decisions"][split]["thresholds"]["min_clean_identity_rate"])),
                                        "checkpoint_criterion_for_role": group.startswith("single_recording/")},
                "exact_failure_reasons": dict(sorted(reasons.items())),
                "metrics": {
                    "endpoint": {key: _mean(selected, ("endpoint", key)) for key in
                                 ("before", "after", "absolute_improvement", "relative_improvement", "required_relative_improvement")},
                    "temporal": {key: _mean(selected, ("temporal", key)) for key in
                                 ("before", "after", "absolute_improvement", "relative_improvement", "required_relative_improvement")},
                    "hidden_clean_geometry": {key: _mean(selected, ("geometry", "repair_toward_hidden_clean", key))
                                              for key in ("before", "after", "absolute_improvement",
                                                          "relative_improvement", "required_relative_improvement")},
                    "clean_identity_product_log_l1": _mean(selected, ("clean_identity", "product_log_l1")),
                    "clean_identity_contact_l1": _mean(selected, ("clean_identity", "contact_l1")),
                    "physical": {key: {"before": _mean(selected, ("physical", "before", key)),
                                       "after": _mean(selected, ("physical", "after", key))}
                                 for key in PHYSICAL_KEYS},
                },
            }
            tables[split][group] = row
            for reason, count in reasons.items():
                by_split[(reason, split)] += count
                by_group[(reason, f"{split}/{group}")] += count
    return {
        "group_table": tables,
        "failure_reason_by_split": {reason: {split: by_split[(reason, split)]
                                               for split in ("seen", "new_position")}
                                    for reason in sorted({key[0] for key in by_split})},
        "failure_reason_by_group": {reason: {group: by_group[(reason, group)]
                                               for group in sorted({key[1] for key in by_group})}
                                    for reason in sorted({key[0] for key in by_group})},
    }


def answer_scientific_questions(cases, group_table):
    def aggregate(selected):
        return {
            "cases": len(selected),
            "endpoint_failed_cases": sum(not row["endpoint"]["accepted"] for row in selected),
            "temporal_failed_cases": sum(not row["temporal"]["accepted"] for row in selected),
            "physical_failed_cases": sum(not row["physical"]["accepted"] for row in selected),
            "geometry_fidelity_failed_cases": sum(
                not row["geometry"]["reference_fidelity"]["accepted"] for row in selected),
            "clean_identity_failed_cases": sum(
                row["clean_identity"].get(
                    "checkpoint_criterion_for_role", row["role"] == "single_recording")
                and not row["clean_identity"]["accepted"] for row in selected),
        }

    overall = aggregate(cases)
    endpoint, temporal = overall["endpoint_failed_cases"], overall["temporal_failed_cases"]
    primary = ("endpoint" if endpoint > temporal else "temporal" if temporal > endpoint
               else "endpoint_and_temporal_tied")
    axes = {
        "split": {value: aggregate([row for row in cases if row["split"] == value])
                  for value in ("seen", "new_position")},
        "role": {value: aggregate([row for row in cases if row["role"] == value])
                 for value in ("single_recording", "cross_event")},
        "width": {str(value): aggregate([row for row in cases if row["width"] == value])
                  for value in (10, 28)},
    }
    axes["split_group_passes"] = {
        split: {"passed_groups": sum(row["passed"] for row in groups.values()),
                "total_groups": len(groups)} for split, groups in group_table.items()}
    return {
        "primary_endpoint_or_temporal": {"answer": primary,
            "endpoint_failed_cases": endpoint, "temporal_failed_cases": temporal},
        "physical_safety_failure": {"answer": overall["physical_failed_cases"] > 0,
                                    "failed_cases": overall["physical_failed_cases"]},
        "clean_identity_failure": {"answer": overall["clean_identity_failed_cases"] > 0,
                                   "failed_checkpoint_criterion_cases": overall["clean_identity_failed_cases"]},
        "seen_vs_new_position": axes["split"],
        "single_vs_cross": axes["role"],
        "width_10_vs_28": axes["width"],
        "group_passes_by_split": axes["split_group_passes"],
        "interpretation_rule": "Differences are descriptive counts from the fixed 64 cases; no significance claim is made.",
    }


def trajectory_contact_history(path):
    rows, expected = [], 1
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("step") != expected:
                raise ValueError("trajectory updates are incomplete or out of order")
            contact = row["head"]["blocks"]["contact"]
            rows.append({"step": expected, "retained": bool(row["optimizer"]["retained"]),
                         "rolled_back": bool(row["optimizer"]["rolled_back"]),
                         "parameter_norm": float(contact["parameter_norm"]),
                         "actual_update_norm": float(contact["actual_update_norm"])})
            expected += 1
    if len(rows) != trajectory.STEPS:
        raise ValueError("trajectory history must contain exactly 400 steps")
    return {
        "steps": len(rows), "all_steps_retained": all(row["retained"] for row in rows),
        "rollback_steps": sum(row["rolled_back"] for row in rows),
        "contact_parameter_exact_zero_all_steps": all(row["parameter_norm"] == 0 for row in rows),
        "contact_actual_update_exact_zero_all_steps": all(row["actual_update_norm"] == 0 for row in rows),
        "contact_gradient_recorded_per_block": False,
        "gradient_evidence_boundary": (
            "updates.jsonl records contact parameter/update blocks, not historical contact-row gradients; "
            "the audit therefore reports the fixed-final true gradient separately."),
        "first_step": rows[0], "last_step": rows[-1],
    }


def _load_trajectory(directory, expected_commit):
    directory = Path(directory).resolve()
    paths = {name: directory / name for name in
             ("report.json", "experiment.json", "diagnostic_latest.pt", "updates.jsonl")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("trajectory directory is incomplete")
    hashes = {name: a.file_sha256(path) for name, path in paths.items()}
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    experiment = json.loads(paths["experiment.json"].read_text(encoding="utf-8"))
    optimizer = report.get("trajectory", {}).get("optimizer_steps", {})
    if (report.get("schema") != trajectory.SCHEMA or report.get("completed") is not True
            or report.get("diagnostic_completed") is not True or report.get("completed_steps") != 400
            or report.get("optimizer_steps") != 400 or report.get("probe_loaded") is not True
            or optimizer.get("attempted") != 400 or optimizer.get("accepted") != 400
            or optimizer.get("retained") != 400 or optimizer.get("rolled_back") != 0
            or report.get("final", {}).get("diagnostic_gates_passed") is not False
            or report.get("fresh_initialization") is not True
            or report.get("source_weights_used_for_initialization") is not False
            or report.get("historical_comparison_is_descriptive_only") is not True
            or report.get("scientific_acceptance") is not False or report.get("publish_allowed") is not False
            or report.get("pilot_allowed") is not False):
        raise ValueError("requires the complete fixed, nonpublishing A0 trajectory")
    if experiment != report.get("experiment") or _canonical_hash(experiment) != report.get("experiment_sha256"):
        raise ValueError("trajectory experiment hash/content mismatch")
    if experiment.get("runtime_commit") != expected_commit:
        raise ValueError("unexpected trajectory runtime commit")
    checkpoint = m._trusted_torch_load(paths["diagnostic_latest.pt"], map_location="cpu")
    if (checkpoint.get("schema") != trajectory.SCHEMA or checkpoint.get("version") != trajectory.MODEL_VERSION
            or checkpoint.get("completed_steps") != 400 or checkpoint.get("target_steps") != 400
            or checkpoint.get("experiment_sha256") != report["experiment_sha256"]
            or checkpoint.get("formal_checkpoint") is not False or checkpoint.get("publish_allowed") is not False
            or checkpoint.get("pilot_allowed") is not False or checkpoint.get("resume_allowed") is not False
            or hashes["diagnostic_latest.pt"] != report.get("final_checkpoint_sha256")
            or safe.state_hash(checkpoint["model_state_dict"]) != report.get("final_state_sha256")):
        raise ValueError("trajectory final checkpoint provenance/hash mismatch")
    return directory, paths, hashes, report, experiment, checkpoint


def _historical_comparison(source_report, cfg, current):
    historical = {}
    raw = source_report.get("final", {})
    for split in ("seen", "new_position"):
        if split in raw:
            historical[split] = group_decisions(raw[split], cfg)
    return {
        "historical_comparison_is_descriptive_only": True,
        "historical_groups": historical,
        "fresh_A0_groups": current["group_decisions"],
        "claim_boundary": "The runs do not have matched initialization; no statistical superiority claim is allowed.",
    }


def _alignment(history, gradients, cases, final_evaluation):
    failed = Counter()
    for row in cases:
        failed.update(reason.split("/", 1)[0] for reason in row["failure_reasons"])
    start, end = history.get("step_1_objective", {}), history.get("step_400_objective", {})
    change = {}
    for key in ("repair", "clean", "training_total"):
        if key in start and key in end:
            before, after = float(start[key]), float(end[key])
            change[key] = {"step_1_before": before, "step_400_before": after,
                           "absolute_change": after - before,
                           "relative_change": ((after - before) / abs(before) if before else None)}
    return {
        "training_objective": {
            "step_1_before": history.get("step_1_objective"),
            "step_400_before": history.get("step_400_objective"),
            "fixed_budget_change_not_monotonicity_claim": change,
            "fixed_final_current": {name: row["value"] for name, row in gradients.items()
                                    if name in {"repair_objective", "clean_identity_objective", "training_total"}},
            "optimized_quantities": [
                "group-balanced upper-tail endpoint and temporal scientific deficits",
                "input-relative support/jerk/root safety surrogates",
                "reference trust/contact/minimum-edit terms",
                "weighted clean-input identity objective",
            ],
        },
        "scientific_gate": {
            "diagnostic_gates_passed": bool(final_evaluation["diagnostic_gates_passed"]),
            "failed_case_reason_categories": dict(sorted(failed.items())),
            "required_quantities": [
                "per-group observable endpoint pass rate",
                "per-group observable temporal gain with jerk non-regression",
                "input-relative physical non-regression and reference fidelity",
                "clean-input identity rate",
            ],
        },
        "objective_blind_spot": {
            "present": not final_evaluation["diagnostic_gates_passed"],
            "interpretation": (
                "All checked steps can descend the scalar TRAIN transaction while discrete held-out pass-rate "
                "requirements remain unmet. The case table identifies which requirement, without changing the objective."),
        },
    }


def run(args):
    output = Path(args.output).resolve()
    source = Path(args.state_dir).resolve()
    traj_dir, traj_paths, traj_hashes, traj_report, experiment, checkpoint = _load_trajectory(
        args.trajectory_dir, args.expected_trajectory_commit)
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(traj_dir):
        raise FileExistsError("write a create-only report outside both immutable input directories")

    state, bank, cfg, source_metadata = a.load_frozen_source(
        source, a.LEGACY_COMMIT, legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength)
    if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
        raise ValueError("trajectory does not reference the supplied frozen source")
    if traj_report.get("probe_sha256") != state.get("probe_bank_artifact", {}).get("sha256"):
        raise ValueError("trajectory probe provenance does not match frozen source")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    source_report_path = source / "diagnostic_report.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    source_input_hashes = {name: a.file_sha256(source / name) for name in
                           ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")}
    history = trajectory_contact_history(traj_paths["updates.jsonl"])
    update_rows = [json.loads(line) for line in traj_paths["updates.jsonl"].read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    history["step_1_objective"] = update_rows[0]["objective_before"]
    history["step_400_objective"] = update_rows[-1]["objective_before"]
    del update_rows

    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices), a.frozen_environment(
            state["fingerprint"], source_metadata["decoder_strengths"]):
        model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        initial_state_hash = safe.state_hash(model.state_dict())
        train_batch = {key: value.to(device) for key, value in
                       a.materialize_transaction(bank, cfg, 0).items()}
        probe, probe_hash = safe.load_probe(source, state, bank, cfg)
        banks = final_banks(bank, probe, cfg)
        with preserve_model_runtime(model):
            final_evaluation = safe.evaluate_final(model, bank, probe, cfg)
            if (final_evaluation["group_decisions"] != traj_report["final"]["group_decisions"]
                    or final_evaluation["diagnostic_gates_passed"] !=
                    traj_report["final"]["diagnostic_gates_passed"]):
                raise RuntimeError("fixed-final reevaluation does not match trajectory report")
            cases = case_failure_attribution(model, banks, cfg)
            summarized = summarize_cases(cases, final_evaluation)
            masks = contact_mask_audit(train_batch, banks, cfg)
            gradients = true_contact_gradients(model, train_batch, cfg)
            jacobian = contact_decoder_jacobian(model, train_batch, cfg, banks=banks)
        final_state_hash = safe.state_hash(model.state_dict())
    if initial_state_hash != final_state_hash or final_state_hash != traj_report["final_state_sha256"]:
        raise RuntimeError("read-only final failure audit changed the model state")
    if probe_hash != traj_report["probe_sha256"]:
        raise RuntimeError("probe hash changed during read-only audit")

    actual_derivative_zero = all(row["all_positions"]["exactly_zero"]
                                 for row in jacobian["actual_paths"])
    final_contact_gradient_zero = all(
        objective["head_parameter_gradients"]["contact"]["combined"]["exactly_zero"]
        for name, objective in gradients.items() if name in
        {"repair_objective", "clean_identity_objective", "training_total"})
    synthetic_decoder_nonzero = not jacobian["nonzero_mask_control"]["exactly_zero"]
    if masks["all_effective_masks_exactly_zero"] and actual_derivative_zero and synthetic_decoder_nonzero:
        cause = "mask_zero"
    elif not actual_derivative_zero and final_contact_gradient_zero:
        cause = "objective_zero_gradient"
    elif not synthetic_decoder_nonzero:
        cause = "decoder_zero_jacobian"
    else:
        cause = "mixed_or_not_identified"

    report = {
        "schema": SCHEMA, "completed": True,
        "provenance": {
            "trajectory_commit": args.expected_trajectory_commit,
            "trajectory_directory": str(traj_dir), "trajectory_sha256": traj_hashes,
            "trajectory_experiment_sha256": traj_report["experiment_sha256"],
            "trajectory_final_state_sha256": traj_report["final_state_sha256"],
            "source": source_metadata, "source_sha256_including_probe": source_input_hashes,
            "probe_sha256": probe_hash, "runtime_commit": m._training_code_revision(),
            "implementation_sha256": {Path(path).name: a.file_sha256(path) for path in
                (__file__, m.__file__, d.__file__, safe.__file__, trajectory.__file__)},
        },
        "final_scientific_failure": {
            **summarized, "case_level": cases,
            "diagnostic_gates_passed": final_evaluation["diagnostic_gates_passed"],
            "split_decisions": final_evaluation["decisions"],
            "scientific_answers": answer_scientific_questions(cases, summarized["group_table"]),
        },
        "contact_connectivity": {
            "mask_audit": masks, "true_objective_gradients": gradients,
            "decoder_jacobian_vjp": jacobian, "trajectory_history": history,
            "exact_zero_origin": {
                "classification": cause,
                "optimizer_rollback": history["rollback_steps"] > 0,
                "all_400_contact_updates_exact_zero": history["contact_actual_update_exact_zero_all_steps"],
                "final_contact_parameter_gradients_exact_zero": final_contact_gradient_zero,
                "actual_decoder_contact_derivative_exact_zero": actual_derivative_zero,
                "synthetic_nonzero_mask_decoder_derivative_nonzero": synthetic_decoder_nonzero,
                "root_joint_contrast": gradients["training_total"]["head_parameter_gradients"],
            },
        },
        "objective_vs_scientific_gate_alignment": _alignment(history, gradients, cases, final_evaluation),
        "historical_v15_4_1": _historical_comparison(source_report, cfg, final_evaluation),
        "historical_comparison_is_descriptive_only": True,
        "optimizer": None, "optimizer_steps": 0, "model_state_unchanged": True,
        "probe_used_for_fixed_final_evaluation_only": True,
        "checkpoint_selection_performed": False,
        "scientific_acceptance": False, "publish_allowed": False, "pilot_allowed": False,
        "next_action": "review_failure_and_contact_connectivity_no_training_or_pilot",
    }

    for name, digest in traj_hashes.items():
        if a.file_sha256(traj_paths[name]) != digest:
            raise RuntimeError("trajectory artifact changed during read-only audit")
    for name, digest in source_input_hashes.items():
        if a.file_sha256(source / name) != digest:
            raise RuntimeError("frozen source artifact changed during read-only audit")
    _exclusive_json(output, report)
    for split, groups in summarized["group_table"].items():
        for group, row in groups.items():
            print(json.dumps({"stage": "final_failure_group", "split": split, "group": group,
                              "cases": row["cases"], "passed": row["passed"],
                              "endpoint": row["endpoint_gate"], "temporal": row["temporal_gate"],
                              "geometry": row["geometry_gate"], "physical": row["physical_gate"],
                              "clean_identity": row["clean_identity_gate"]}, allow_nan=False), flush=True)
    print(json.dumps({"stage": "refiner_final_failure_audit_complete", "output": str(output),
                      "contact_zero_origin": cause, "optimizer_steps": 0,
                      "scientific_acceptance": False, "publish_allowed": False,
                      "pilot_allowed": False}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-trajectory-commit", default=TRAJECTORY_COMMIT)
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
