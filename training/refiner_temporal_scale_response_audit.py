"""Read-only finite-scale response audit for the fixed Refiner action.

The seven alpha values are preregistered constants.  They are counterfactual
response points, never candidates for selection, deployment, or Pilot.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

from motion_geometry.boundary_observables import boundary_metrics_torch, observable_gate
from training import motion_models as m
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_safe_start_diagnostics as safe
from training import refiner_temporal_action_alignment_audit as alignment


SCHEMA = "refiner_temporal_scale_response_audit_v1"
REVIEWED_ALIGNMENT_COMMIT = "5557f78398f94e448c61e8d14bbf25ac0d5ee373"
ALPHAS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
ALPHA_KEYS = tuple(f"{value:.2f}" for value in ALPHAS)
FD_H = 1.0e-3
TRACE_STAGES = ("raw", "after_mask", "after_smoothing", "after_taper", "after_cap", "applied")
PARITY_ATOL_CPU = 1.0e-7
PARITY_ATOL_CUDA = 2.0e-6
FINAL_METRIC_RTOL = 2.0e-5
FINAL_METRIC_ATOL = 2.0e-6
DECODER_PROTOCOL = "soft_scale_then_cap_applied_tangent_v1"


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"nonfinite {label}")
    return value


def ratio(numerator, denominator):
    numerator, denominator = _finite(numerator, "ratio numerator"), _finite(denominator, "ratio denominator")
    return numerator / denominator if denominator != 0.0 else None


def tensor_stats(value):
    value = value.detach().double().reshape(-1)
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise FloatingPointError("empty or nonfinite response tensor")
    norm = _finite(value.norm(), "response norm")
    return {"numel": value.numel(), "l2_norm": norm,
            "rms": norm / math.sqrt(value.numel()), "abs_max": _finite(value.abs().max(), "abs max")}


def scale_raw_output(raw_output, alpha):
    """Keep contact logits unchanged and scale only raw 75D geometry."""
    if raw_output.shape[-1] != m.PRODUCT_STATE_DIM:
        raise ValueError("raw Refiner output must be 79D")
    alpha = torch.as_tensor(alpha, dtype=raw_output.dtype, device=raw_output.device)
    if alpha.ndim != 0 or not bool(torch.isfinite(alpha)):
        raise ValueError("alpha must be one finite scalar")
    return torch.cat((raw_output[..., :4], alpha * raw_output[..., 4:]), dim=-1)


def _effective_masks(batch, cfg):
    joint, root, contact = m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)
    geometry = torch.cat((root.expand(root.shape[:-1] + (3,)),
        joint[..., None].expand(joint.shape + (3,)).reshape(joint.shape[:-1] + (72,))), -1)
    return joint, root, contact, geometry


def _stage_stats(value, support):
    all_frames = tensor_stats(value)
    active = value[support]
    return {"all_geometry": all_frames,
            "decoder_support": tensor_stats(active) if active.numel() else None}


def decoder_case_stats(trace, geometry_support):
    required = set(TRACE_STAGES) | {"root_mask", "joint_mask", "root_cap_m", "rotation_cap_rad"}
    if not required <= set(trace):
        raise ValueError("production decoder trace is incomplete")
    cases = trace["raw"].shape[0]
    rows = []
    for index in range(cases):
        support = geometry_support[index] > 0
        stages = {name: _stage_stats(trace[name][index], support) for name in TRACE_STAGES}
        raw = stages["raw"]["decoder_support"]
        masked = stages["after_mask"]["decoder_support"]
        applied = stages["applied"]["decoder_support"]
        rows.append({
            "stage_order": list(TRACE_STAGES), "stages": stages,
            "attenuation": {
                "mask_attenuation_l2_ratio": ratio(masked["l2_norm"], raw["l2_norm"]),
                "postprocessing_attenuation_l2_ratio": ratio(applied["l2_norm"], masked["l2_norm"]),
                "total_attenuation_l2_ratio": ratio(applied["l2_norm"], raw["l2_norm"]),
                "applied_raw_rms_ratio": ratio(applied["rms"], raw["rms"]),
            },
        })
    return rows


def cap_saturation_stats(trace):
    pre = trace["after_taper"].detach()
    root_pre = pre[..., :3]
    joints_pre = pre[..., 3:].reshape(pre.shape[:-1] + (24, 3))
    root_eligible = trace["root_mask"].detach()[..., 0] > 0
    joint_eligible = trace["joint_mask"].detach() > 0
    root_limit = float(trace["root_cap_m"])
    joint_limit = float(trace["rotation_cap_rad"])
    root_sat = (torch.linalg.vector_norm(root_pre, dim=-1) > root_limit) & root_eligible
    joint_sat = (torch.linalg.vector_norm(joints_pre, dim=-1) > joint_limit) & joint_eligible
    extremities = list(alignment.EXTREMITY_JOINTS)
    body = [index for index in range(24) if index not in extremities]

    def fraction(mask, eligible):
        denominator = int(eligible.sum())
        return float(mask.sum()) / denominator if denominator else None

    rows = []
    for index in range(pre.shape[0]):
        blocks = {
            "root": fraction(root_sat[index], root_eligible[index]),
            "body": fraction(joint_sat[index, ..., body], joint_eligible[index, ..., body]),
            "extremity": fraction(joint_sat[index, ..., extremities], joint_eligible[index, ..., extremities]),
        }
        any_frame = root_sat[index] | joint_sat[index].any(-1)
        rows.append({
            "root_cap_m": root_limit, "rotation_cap_rad": joint_limit,
            "root_cap_saturation_fraction": blocks["root"],
            "joint_rotation_cap_saturation_fraction": fraction(joint_sat[index], joint_eligible[index]),
            "block_saturation_fraction": blocks,
            "frames_with_any_saturation": int(any_frame.sum()),
            "frame_saturation_fraction": float(any_frame.float().mean()),
            "case_has_any_saturation": bool(any_frame.any()),
        })
    return rows


def _cpu_trace(trace):
    return {key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in trace.items()}


def _torch_observable_rows(prediction, reference, seam, cfg):
    count = prediction.shape[0]
    with torch.no_grad():
        joints = m._observable_boundary_joints_torch(torch.cat((reference, prediction)))
        metrics = boundary_metrics_torch(joints, torch.cat((seam, seam)), cfg.fps)
    before, after = ({key: value[:count].detach().cpu() for key, value in metrics.items()},
                     {key: value[count:].detach().cpu() for key, value in metrics.items()})
    rows = []
    for index in range(count):
        before_row = {key: (bool(value[index]) if key == "valid" else _finite(value[index], key))
                      for key, value in before.items()}
        after_row = {key: (bool(value[index]) if key == "valid" else _finite(value[index], key))
                     for key, value in after.items()}
        rows.append(observable_gate(before_row, after_row, cfg))
    return rows


def _physical_case(reference, prediction, clean, clean_prediction, seam, cfg):
    before = m._safe_validation_audit(reference, cfg, role="scale_response_before",
                                      support_policy="source_observation")
    after = m._safe_validation_audit(prediction, cfg, role="scale_response_after",
                                     support_policy="source_observation")
    physical = m._fixed_support_stage_gate(reference, prediction, cfg,
                                           before_audit=before, after_audit=after)
    observable = m._observable_boundary_audit(prediction, reference, seam, cfg)
    clean_gate = failure._clean_gate(clean_prediction, clean, cfg)
    return {
        "observable": observable,
        "physical": {"before": failure._physical_values(before),
                     "after": failure._physical_values(after),
                     "accepted": bool(physical["accepted"]),
                     "reasons": list(physical.get("reasons", [])),
                     "authoritative_gate": physical},
        "geometry": {"reference_fidelity": observable["reference_fidelity"],
                     "accepted": bool(observable["reference_fidelity_accepted"]),
                     "required_max": {"fk_p95_m": float(cfg.checkpoint_validation_max_fk_p95_m),
                                      "fk_max_m": float(cfg.checkpoint_validation_max_fk_max_m),
                                      "product_log_l1": float(
                                          cfg.checkpoint_validation_max_refiner_product_log_l1)}},
        "clean_identity": {"accepted": bool(clean_gate["accepted"]),
                           "product_log_l1": _finite(clean_gate["identity_detail"]["product_log_l1"], "identity"),
                           "maximum_product_log_l1": _finite(
                               clean_gate["identity_detail"]["maximum_product_log_l1"], "identity max"),
                           "contact_l1": _finite(clean_gate["identity_detail"]["contact_l1"], "contact identity"),
                           "maximum_contact_l1": _finite(
                               clean_gate["identity_detail"]["maximum_contact_l1"], "contact max"),
                           "reasons": list(clean_gate.get("reasons", []))},
    }


def _scope_masks(metadata, source):
    scopes = {"overall": torch.ones(len(metadata), dtype=torch.bool)}
    if source == "train":
        for group in group_audit.GROUPS:
            scopes[f"group:{group}"] = torch.tensor([row["group"] == group for row in metadata])
    else:
        for split in ("seen", "new_position"):
            scopes[f"split:{split}"] = torch.tensor([row["split"] == split for row in metadata])
        for role in ("single_recording", "cross_event"):
            scopes[f"role:{role}"] = torch.tensor([row["role"] == role for row in metadata])
        for width in (10, 28):
            scopes[f"width:{width}"] = torch.tensor([row["width"] == width for row in metadata])
        for split in ("seen", "new_position"):
            for role in ("single_recording", "cross_event"):
                for width in (10, 28):
                    key = f"group:{split}/{role}/{width}"
                    scopes[key] = torch.tensor([row["split"] == split and row["role"] == role
                                                and row["width"] == width for row in metadata])
    return {name: mask for name, mask in scopes.items() if bool(mask.any())}


def _objective_derivatives(terms, alpha, scopes):
    targets = []
    for objective, key in (("temporal", "temporal_scientific_deficit"),
                           ("endpoint", "endpoint_scientific_deficit")):
        for scope, cpu_mask in scopes.items():
            mask = cpu_mask.to(terms[key].device)
            targets.append((objective, scope, terms[key][mask].mean()))
    result = {objective: {} for objective in ("temporal", "endpoint")}
    for index, (objective, scope, value) in enumerate(targets):
        derivative = torch.autograd.grad(value, alpha, retain_graph=index + 1 < len(targets),
                                         allow_unused=True)[0]
        result[objective][scope] = 0.0 if derivative is None else _finite(derivative.detach(), "dL/dalpha")
    return result


def capture_fixed_action(model, batch, cfg):
    """Exactly one model forward supplies immutable repair and clean actions."""
    count = batch["clean"].shape[0]
    with torch.no_grad(), alignment._capture_model_output(model) as captured:
        production_prediction, production_identity = m._refiner_batch_outputs(model, batch, cfg)
    if len(captured) != 1:
        raise RuntimeError("scale response requires exactly one model forward")
    raw = captured[0].transpose(1, 2).detach()
    if raw.shape[0] != 2 * count or raw.shape[-1] != 79:
        raise ValueError("captured repair/clean raw action shape mismatch")
    return {"repair": raw[:count], "clean": raw[count:],
            "production_prediction": production_prediction.detach(),
            "production_identity": production_identity.detach(),
            "model_forward_calls": 1}


def _alpha_response(raw, batch, cfg, alpha_value, metadata, source):
    alpha = raw["repair"].new_tensor(alpha_value, requires_grad=True)
    repair_output = scale_raw_output(raw["repair"], alpha)
    clean_output = scale_raw_output(raw["clean"], alpha)
    repair_masks = m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)
    clean_masks = m._refiner_decode_masks(
        batch["clean_joint"], batch["clean_root"], batch["clean_contact"], batch["seam"], cfg)
    repair_trace, clean_trace = {}, {}
    prediction = m._decode_product_refiner_output(
        batch["bad"], repair_output, *repair_masks, cfg, trace=repair_trace)
    identity = m._decode_product_refiner_output(
        batch["clean"], clean_output, *clean_masks, cfg, trace=clean_trace)
    _loss, terms = m._observable_refiner_objective(
        prediction, batch["bad"], batch["seam"], cfg, reduction="none")
    derivatives = _objective_derivatives(terms, alpha, _scope_masks(metadata, source))
    observables = _torch_observable_rows(prediction.detach(), batch["bad"], batch["seam"], cfg)
    _, _, _, geometry_mask = _effective_masks(batch, cfg)
    _, _, _, clean_geometry_mask = _effective_masks({
        **batch, "joint": batch["clean_joint"], "root": batch["clean_root"],
        "contact": batch["clean_contact"]}, cfg)
    repair_trace_cpu, clean_trace_cpu = _cpu_trace(repair_trace), _cpu_trace(clean_trace)
    decoder = decoder_case_stats(repair_trace_cpu, geometry_mask.detach().cpu())
    clean_decoder = decoder_case_stats(clean_trace_cpu, clean_geometry_mask.detach().cpu())
    saturation = cap_saturation_stats(repair_trace_cpu)
    clean_saturation = cap_saturation_stats(clean_trace_cpu)
    rows = []
    predicted_np = prediction.detach().cpu().numpy() if source == "final" else None
    identity_np = identity.detach().cpu().numpy() if source == "final" else None
    if source == "final":
        bad_np, clean_np, seam_np = (batch[key].detach().cpu().numpy()
                                     for key in ("bad", "clean", "seam"))
    for index, meta in enumerate(metadata):
        authoritative = observables[index]
        row = {
            **meta, "alpha": float(alpha_value),
            "objectives": {
                "temporal_scientific_deficit": _finite(
                    terms["temporal_scientific_deficit"][index].detach(), "temporal deficit"),
                "endpoint_scientific_deficit": _finite(
                    terms["endpoint_scientific_deficit"][index].detach(), "endpoint deficit"),
            },
            "authoritative_observable": {
                "temporal_metric": _finite(authoritative["after"]["temporal_energy"], "temporal metric"),
                "endpoint_metric": _finite(authoritative["after"]["endpoint_velocity_jump_mps"], "endpoint metric"),
                "temporal_repair_gain": _finite(authoritative["temporal_gain"], "temporal gain"),
                "endpoint_repair_gain": _finite(authoritative["endpoint_gain"], "endpoint gain"),
                "temporal_gate_pass": bool(authoritative["temporal_accepted"]),
                "endpoint_gate_pass": bool(authoritative["endpoint_accepted"]),
                "jerk_non_regression": bool(authoritative["jerk_non_regression"]),
                "reasons": list(authoritative["reasons"]),
            },
            "decoder": decoder[index], "cap_saturation": saturation[index],
            "clean_decoder": clean_decoder[index], "clean_cap_saturation": clean_saturation[index],
        }
        if source == "final":
            details = _physical_case(bad_np[index], predicted_np[index], clean_np[index],
                                     identity_np[index], seam_np[index], cfg)
            # Use the exact NumPy authoritative gate for final reporting.
            obs = details.pop("observable")
            row["authoritative_observable"] = {
                "temporal_metric": _finite(obs["after"]["temporal_energy"], "temporal metric"),
                "endpoint_metric": _finite(obs["after"]["endpoint_velocity_jump_mps"], "endpoint metric"),
                "temporal_repair_gain": _finite(obs["temporal_gain"], "temporal gain"),
                "endpoint_repair_gain": _finite(obs["endpoint_gain"], "endpoint gain"),
                "temporal_gate_pass": bool(obs["temporal_accepted"]),
                "endpoint_gate_pass": bool(obs["endpoint_accepted"]),
                "jerk_non_regression": bool(obs["jerk_non_regression"]),
                "before": obs["before"], "after": obs["after"], "reasons": list(obs["reasons"]),
            }
            row.update(details)
        rows.append(row)
    contracts = {
        "contact_channels_unchanged": bool(torch.equal(repair_output[..., :4].detach(), raw["repair"][..., :4])),
        "scaled_geometry_matches_alpha": bool(torch.equal(
            repair_output[..., 4:].detach(), float(alpha_value) * raw["repair"][..., 4:])),
        "geometric_applied_abs_max": _finite(repair_trace_cpu["applied"].abs().max(), "applied max"),
    }
    result = {"rows": rows, "derivatives": derivatives, "contracts": contracts}
    if alpha_value == 1.0:
        tolerance = PARITY_ATOL_CUDA if prediction.is_cuda else PARITY_ATOL_CPU
        result["alpha_one_parity"] = {
            "prediction_max_abs_error": _finite(
                (prediction.detach() - raw["production_prediction"]).abs().max(), "prediction parity"),
            "clean_prediction_max_abs_error": _finite(
                (identity.detach() - raw["production_identity"]).abs().max(), "identity parity"),
            "rtol": 0.0, "atol": tolerance,
        }
        if max(result["alpha_one_parity"]["prediction_max_abs_error"],
               result["alpha_one_parity"]["clean_prediction_max_abs_error"]) > tolerance:
            raise RuntimeError("alpha=1 does not reproduce production prediction")
    del prediction, identity, terms, alpha, repair_output, clean_output
    return result


def _relative_change(before, after):
    return (before - after) / abs(before) if before != 0.0 else None


def audit_batch(model, batch, cfg, metadata, source):
    model.eval()
    raw = capture_fixed_action(model, batch, cfg)
    responses = {}
    for alpha_value, key in zip(ALPHAS, ALPHA_KEYS):
        responses[key] = _alpha_response(raw, batch, cfg, alpha_value, metadata, source)
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.empty_cache()
    if not all(response["contracts"]["contact_channels_unchanged"] and
               response["contracts"]["scaled_geometry_matches_alpha"]
               for response in responses.values()):
        raise RuntimeError("alpha scaling changed contact or violated geometric scaling")
    if responses["0.00"]["contracts"]["geometric_applied_abs_max"] != 0.0:
        raise RuntimeError("alpha=0 is not the exact zero geometric edit")
    rows = []
    for index, meta in enumerate(metadata):
        alpha_rows = {key: responses[key]["rows"][index] for key in ALPHA_KEYS}
        zero, one = alpha_rows["0.00"], alpha_rows["1.00"]
        for key, row in alpha_rows.items():
            for objective in ("temporal", "endpoint"):
                field = f"{objective}_scientific_deficit"
                value = row["objectives"][field]
                row["objectives"][f"relative_{objective}_deficit_improvement_vs_alpha_0"] = (
                    _relative_change(zero["objectives"][field], value))
                row["objectives"][f"relative_{objective}_deficit_improvement_vs_alpha_1"] = (
                    _relative_change(one["objectives"][field], value))
                metric = row["authoritative_observable"][f"{objective}_metric"]
                row["authoritative_observable"][f"relative_{objective}_metric_improvement_vs_alpha_0"] = (
                    _relative_change(zero["authoritative_observable"][f"{objective}_metric"], metric))
                row["authoritative_observable"][f"relative_{objective}_metric_improvement_vs_alpha_1"] = (
                    _relative_change(one["authoritative_observable"][f"{objective}_metric"], metric))
        rows.append({**meta, "responses": alpha_rows})
    return {"case_level": rows,
            "derivatives": {key: responses[key]["derivatives"] for key in ALPHA_KEYS},
            "alpha_one_parity": responses["1.00"]["alpha_one_parity"],
            "alpha_zero_contract": responses["0.00"]["contracts"],
            "model_forward_calls": raw["model_forward_calls"]}


def _close(actual, expected):
    return math.isclose(float(actual), float(expected), rel_tol=FINAL_METRIC_RTOL,
                        abs_tol=FINAL_METRIC_ATOL)


def validate_alpha_one_final_metrics(rows, trajectory_final):
    expected = {}
    for split in ("seen", "new_position"):
        metrics = trajectory_final["metrics"][split]
        for role, values in (("single_recording", metrics["windows"]),
                             ("cross_event", metrics["cross_event"]["windows"])):
            for row in values:
                expected[(split, role, int(row["case_index"]))] = row
    if len(expected) != 64:
        raise ValueError("trajectory final metric layout is not 64 cases")
    max_error = 0.0
    for row in rows:
        response = row["responses"]["1.00"]
        reference = expected[(row["split"], row["role"], row["bank_case_index"])]
        actual_obs, expected_obs = response["authoritative_observable"], reference["observable"]
        pairs = ((actual_obs["temporal_metric"], expected_obs["after"]["temporal_energy"]),
                 (actual_obs["endpoint_metric"], expected_obs["after"]["endpoint_velocity_jump_mps"]),
                 (actual_obs["temporal_repair_gain"], expected_obs["temporal_gain"]),
                 (actual_obs["endpoint_repair_gain"], expected_obs["endpoint_gain"]))
        for actual, target in pairs:
            max_error = max(max_error, abs(float(actual) - float(target)))
            if not _close(actual, target):
                raise RuntimeError("alpha=1 authoritative final metric parity failed")
        if (actual_obs["temporal_gate_pass"] != bool(expected_obs["temporal_accepted"])
                or actual_obs["endpoint_gate_pass"] != bool(expected_obs["endpoint_accepted"])
                or response["geometry"]["accepted"] != bool(expected_obs["reference_fidelity_accepted"])):
            raise RuntimeError("alpha=1 final observable/geometry gate parity failed")
        expected_physical = (expected_obs["physical_non_regression"] if row["role"] == "single_recording"
                             else reference["safety"])
        if response["physical"]["accepted"] != bool(expected_physical["accepted"]):
            raise RuntimeError("alpha=1 final physical gate parity failed")
        if list(response["physical"]["reasons"]) != list(expected_physical.get("reasons", [])):
            raise RuntimeError("alpha=1 final physical reason parity failed")
        actual_support = response["physical"]["authoritative_gate"].get("support_comparison", {})
        expected_support = expected_physical.get("support_comparison", {})
        for phase in ("before", "after"):
            for key, target in expected_support.get(phase, {}).items():
                if key not in actual_support.get(phase, {}) or not _close(
                        actual_support[phase][key], target):
                    raise RuntimeError("alpha=1 final physical metric parity failed")
        actual_fidelity = response["geometry"]["reference_fidelity"]
        expected_fidelity = expected_obs["reference_fidelity"]
        for key in ("fk_p95_m", "fk_max_m", "product_log_l1"):
            if not _close(actual_fidelity[key], expected_fidelity[key]):
                raise RuntimeError("alpha=1 final geometry metric parity failed")
        if row["role"] == "single_recording":
            expected_identity = reference["clean_identity"]
            if response["clean_identity"]["accepted"] != bool(expected_identity["accepted"]):
                raise RuntimeError("alpha=1 final clean identity parity failed")
            detail = expected_identity.get("identity_detail", expected_identity)
            for key in ("product_log_l1", "contact_l1"):
                if key in detail and not _close(response["clean_identity"][key], detail[key]):
                    raise RuntimeError("alpha=1 final clean identity metric parity failed")
    return {"verified": True, "cases": 64, "max_absolute_metric_error": max_error,
            "rtol": FINAL_METRIC_RTOL, "atol": FINAL_METRIC_ATOL}


def _mean(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def _aggregate_decoder(rows):
    result = {"stages": {}, "attenuation": {}}
    for stage in TRACE_STAGES:
        stats = [row["decoder"]["stages"][stage]["decoder_support"] for row in rows]
        result["stages"][stage] = {
            "l2_norm_median": _median([value["l2_norm"] for value in stats]),
            "rms_median": _median([value["rms"] for value in stats]),
            "abs_max": max(value["abs_max"] for value in stats),
        }
    for key in ("mask_attenuation_l2_ratio", "postprocessing_attenuation_l2_ratio",
                "total_attenuation_l2_ratio", "applied_raw_rms_ratio"):
        values = [row["decoder"]["attenuation"][key] for row in rows
                  if row["decoder"]["attenuation"][key] is not None]
        result["attenuation"][key + "_median"] = _median(values)
        result["attenuation"][key + "_defined_cases"] = len(values)
    return result


def _aggregate_cap(rows):
    result = {"cases": len(rows),
              "cases_with_any_saturation": sum(row["cap_saturation"]["case_has_any_saturation"] for row in rows),
              "frames_with_any_saturation": sum(row["cap_saturation"]["frames_with_any_saturation"] for row in rows)}
    for key in ("root_cap_saturation_fraction", "joint_rotation_cap_saturation_fraction",
                "frame_saturation_fraction"):
        values = [row["cap_saturation"][key] for row in rows if row["cap_saturation"][key] is not None]
        result[key + "_median"] = _median(values)
    result["block_saturation_fraction_median"] = {
        block: _median([row["cap_saturation"]["block_saturation_fraction"][block] for row in rows
                        if row["cap_saturation"]["block_saturation_fraction"][block] is not None])
        for block in ("root", "body", "extremity")}
    return result


def _rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(ALPHAS) or len(y) != len(ALPHAS) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("response correlation requires seven finite fixed-grid points")
    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denominator) if denominator else None
    return {"pearson": pearson(x, y), "spearman": pearson(_rank(x), _rank(y)),
            "points": len(ALPHAS), "descriptive_only": True,
            "statistical_significance_claim": False}


def _curve_shape(points):
    values = [row["temporal_scientific_deficit_mean"] for row in points]
    derivatives = [row["dL_temporal_dalpha"] for row in points]
    tolerance = 1.0e-12
    monotonic = all(right <= left + tolerance * max(1.0, abs(left))
                    for left, right in zip(values[:-1], values[1:]))
    signs = [(-1 if value < 0 else 1 if value > 0 else 0) for value in derivatives]
    nonzero = [value for value in signs if value]
    turning = any(left != right for left, right in zip(nonzero[:-1], nonzero[1:]))
    return {"monotonic_improvement_over_grid": monotonic,
            "local_turning_detected": turning,
            "derivative_signs": signs,
            "temporal_gate_crossing_alphas": [row["alpha"] for row in points
                                               if row["temporal_gate_pass_cases"] > 0]}


def response_curve(case_rows, derivatives, scope, selected):
    points = []
    for alpha_value, key in zip(ALPHAS, ALPHA_KEYS):
        values = [row["responses"][key] for row in case_rows if selected(row)]
        if not values:
            raise ValueError("empty response curve group")
        temporal = [row["objectives"]["temporal_scientific_deficit"] for row in values]
        endpoint = [row["objectives"]["endpoint_scientific_deficit"] for row in values]
        observable = [row["authoritative_observable"] for row in values]
        point = {
            "alpha": alpha_value, "cases": len(values),
            "temporal_scientific_deficit_mean": _mean(temporal),
            "endpoint_scientific_deficit_mean": _mean(endpoint),
            "temporal_metric_mean": _mean([row["temporal_metric"] for row in observable]),
            "endpoint_metric_mean": _mean([row["endpoint_metric"] for row in observable]),
            "temporal_repair_gain_mean": _mean([row["temporal_repair_gain"] for row in observable]),
            "endpoint_repair_gain_mean": _mean([row["endpoint_repair_gain"] for row in observable]),
            "relative_temporal_deficit_improvement_vs_alpha_0_mean": _mean([
                row["objectives"]["relative_temporal_deficit_improvement_vs_alpha_0"]
                for row in values if row["objectives"][
                    "relative_temporal_deficit_improvement_vs_alpha_0"] is not None]),
            "relative_temporal_deficit_improvement_vs_alpha_1_mean": _mean([
                row["objectives"]["relative_temporal_deficit_improvement_vs_alpha_1"]
                for row in values if row["objectives"][
                    "relative_temporal_deficit_improvement_vs_alpha_1"] is not None]),
            "relative_endpoint_deficit_improvement_vs_alpha_0_mean": _mean([
                row["objectives"]["relative_endpoint_deficit_improvement_vs_alpha_0"]
                for row in values if row["objectives"][
                    "relative_endpoint_deficit_improvement_vs_alpha_0"] is not None]),
            "relative_endpoint_deficit_improvement_vs_alpha_1_mean": _mean([
                row["objectives"]["relative_endpoint_deficit_improvement_vs_alpha_1"]
                for row in values if row["objectives"][
                    "relative_endpoint_deficit_improvement_vs_alpha_1"] is not None]),
            "temporal_gate_pass_cases": sum(row["temporal_gate_pass"] for row in observable),
            "endpoint_gate_pass_cases": sum(row["endpoint_gate_pass"] for row in observable),
            "dL_temporal_dalpha": derivatives[key]["temporal"][scope],
            "dL_endpoint_dalpha": derivatives[key]["endpoint"][scope],
            "decoder_attenuation": _aggregate_decoder(values),
            "cap_saturation": _aggregate_cap(values),
        }
        if "physical" in values[0]:
            point.update({
                "physical_pass_cases": sum(row["physical"]["accepted"] for row in values),
                "geometry_pass_cases": sum(row["geometry"]["accepted"] for row in values),
                "clean_pass_cases": sum(row["clean_identity"]["accepted"] for row in values),
                "temporal_and_physical_cases": sum(row["authoritative_observable"]["temporal_gate_pass"]
                    and row["physical"]["accepted"] for row in values),
                "temporal_and_geometry_cases": sum(row["authoritative_observable"]["temporal_gate_pass"]
                    and row["geometry"]["accepted"] for row in values),
                "temporal_and_clean_cases": sum(row["authoritative_observable"]["temporal_gate_pass"]
                    and row["clean_identity"]["accepted"] for row in values),
                "all_diagnostic_conditions_cases": sum(row["authoritative_observable"]["temporal_gate_pass"]
                    and row["authoritative_observable"]["endpoint_gate_pass"]
                    and row["physical"]["accepted"] and row["geometry"]["accepted"]
                    and row["clean_identity"]["accepted"] for row in values),
            })
        points.append(point)
    shape = _curve_shape(points)
    shape["correlations"] = {
        "temporal_objective_vs_temporal_metric": correlation(
            [row["temporal_scientific_deficit_mean"] for row in points],
            [row["temporal_metric_mean"] for row in points]),
        "applied_tangent_norm_vs_temporal_repair_gain": correlation(
            [row["decoder_attenuation"]["stages"]["applied"]["l2_norm_median"] for row in points],
            [row["temporal_repair_gain_mean"] for row in points]),
        "saturation_vs_temporal_repair_gain": correlation(
            [row["cap_saturation"]["frame_saturation_fraction_median"] or 0.0 for row in points],
            [row["temporal_repair_gain_mean"] for row in points]),
    }
    shape["objective_gate_response_decoupling"] = any(
        right["temporal_scientific_deficit_mean"] < left["temporal_scientific_deficit_mean"]
        and right["temporal_repair_gain_mean"] <= left["temporal_repair_gain_mean"]
        for left, right in zip(points[:-1], points[1:]))
    return {"points": points, "response_shape": shape,
            "counterfactual_only": True, "scale_selection_performed": False}


def build_curves(train, final):
    train_curves = {group: response_curve(
        train["case_level"], train["derivatives"], f"group:{group}",
        lambda row, group=group: row["group"] == group) for group in group_audit.GROUPS}
    final_curves = {}
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            for width in (10, 28):
                name = f"{split}/{role}/{width}"
                final_curves[name] = response_curve(
                    final["case_level"], final["derivatives"], f"group:{name}",
                    lambda row, split=split, role=role, width=width:
                        row["split"] == split and row["role"] == role and row["width"] == width)
    aggregates = {"overall": response_curve(final["case_level"], final["derivatives"], "overall", lambda _row: True)}
    for role in ("single_recording", "cross_event"):
        aggregates[f"role:{role}"] = response_curve(
            final["case_level"], final["derivatives"], f"role:{role}",
            lambda row, role=role: row["role"] == role)
    return train_curves, final_curves, aggregates


def _classification(value):
    return "supported_by_fixed_response_audit" if value else "not_supported_by_fixed_response_audit"


def scientific_answers(final_curves, aggregates):
    overall = aggregates["overall"]
    points = {row["alpha"]: row for row in overall["points"]}
    above = [points[value] for value in ALPHAS if value > 1.0]
    under = (all(row["dL_temporal_dalpha"] < 0 for row in above)
             and all(right["temporal_scientific_deficit_mean"] <= left["temporal_scientific_deficit_mean"]
                     for left, right in zip([points[1.0], *above[:-1]], above))
             and points[1.0]["temporal_gate_pass_cases"] < points[1.0]["cases"]
             and all(row["physical_pass_cases"] == row["cases"] for row in above)
             and points[2.0]["cap_saturation"]["cases_with_any_saturation"]
                 <= points[1.0]["cap_saturation"]["cases_with_any_saturation"])
    raw_growth = ratio(points[2.0]["decoder_attenuation"]["stages"]["raw"]["l2_norm_median"],
                       points[1.0]["decoder_attenuation"]["stages"]["raw"]["l2_norm_median"])
    applied_growth = ratio(points[2.0]["decoder_attenuation"]["stages"]["applied"]["l2_norm_median"],
                           points[1.0]["decoder_attenuation"]["stages"]["applied"]["l2_norm_median"])
    saturation = (raw_growth is not None and applied_growth is not None and raw_growth > applied_growth
                  and points[2.0]["cap_saturation"]["cases_with_any_saturation"]
                      > points[1.0]["cap_saturation"]["cases_with_any_saturation"]
                  and points[2.0]["temporal_scientific_deficit_mean"]
                      >= points[1.5]["temporal_scientific_deficit_mean"])
    overshoot_groups = [name for name, curve in final_curves.items()
                        if curve["response_shape"]["local_turning_detected"]]
    mismatch_groups = [name for name, curve in final_curves.items()
                       if curve["response_shape"]["objective_gate_response_decoupling"]]
    return {
        "finite_scale_under_actuation_supported": {"classification": _classification(under),
            "evidence": {"alpha_above_one_derivatives": [row["dL_temporal_dalpha"] for row in above],
                         "alpha_one_temporal_pass_cases": points[1.0]["temporal_gate_pass_cases"],
                         "alpha_two_temporal_pass_cases": points[2.0]["temporal_gate_pass_cases"]}},
        "decoder_saturation_supported": {"classification": _classification(saturation),
            "evidence": {"raw_norm_growth_1_to_2": raw_growth,
                         "applied_norm_growth_1_to_2": applied_growth,
                         "saturated_cases_alpha_1": points[1.0]["cap_saturation"]["cases_with_any_saturation"],
                         "saturated_cases_alpha_2": points[2.0]["cap_saturation"]["cases_with_any_saturation"]}},
        "nonlinear_overshoot_supported": {"classification": _classification(bool(overshoot_groups)),
                                           "groups_with_fixed_grid_turning": overshoot_groups},
        "objective_gate_scale_mismatch_supported": {"classification": _classification(bool(mismatch_groups)),
                                                      "groups_with_sign_decoupling": mismatch_groups},
        "new_position_single_recording_28": final_curves["new_position/single_recording/28"],
        "single_vs_cross_control": {
            "single_recording": aggregates["role:single_recording"],
            "cross_event": aggregates["role:cross_event"],
            "descriptive_only": True,
        },
        "claim_boundary": (
            "All classifications are descriptive support from the preregistered response grid. They do not prove "
            "a root cause, select an alpha, alter inference, recommend an architecture, or authorize Pilot."),
    }


def load_alignment_report(path, traj_hashes, traj_report, source_hashes, probe_hash):
    path = Path(path).resolve()
    digest = group_audit.file_sha256(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    provenance = report.get("provenance", {})
    answers = report.get("scientific_answers", {})
    required_flags = (report.get("schema") == alignment.SCHEMA
        and provenance.get("runtime_commit") == REVIEWED_ALIGNMENT_COMMIT
        and report.get("optimizer_steps") == 0
        and report.get("parameter_update_performed") is False
        and report.get("model_state_unchanged") is True
        and report.get("scientific_acceptance") is False
        and report.get("publish_allowed") is False and report.get("pilot_allowed") is False)
    if not required_flags:
        raise ValueError("alignment report contract/runtime/flags mismatch")
    if (provenance.get("trajectory_sha256") != traj_hashes
            or provenance.get("trajectory_final_state_sha256") != traj_report.get("final_state_sha256")
            or provenance.get("source_sha256_including_probe") != source_hashes
            or provenance.get("probe_sha256") != probe_hash):
        raise ValueError("alignment/source/trajectory/probe lineage mismatch")
    expected_answers = {
        "train_temporal_gradient_present": True,
        "final_temporal_gradient_present": True,
        "final_model_action_vs_zero_origin_temporal_descent": "mostly_aligned_with_negative_gradient",
        "current_temporal_action_scaling": "increasing_current_action_scale_is_mostly_local_descent",
        "endpoint_control_at_zero_origin": "mostly_aligned_with_negative_gradient",
    }
    if any(answers.get(key, {}).get("answer") != value for key, value in expected_answers.items()):
        raise ValueError("alignment report does not contain the reviewed server conclusions")
    outside = answers.get("gradient_outside_decoder_support", {})
    if outside.get("all_exact_zero") is not True or outside.get("nonzero_measurements") != 0:
        raise ValueError("alignment decoder-support conclusion mismatch")
    return report, {"path": str(path), "sha256": digest,
                    "runtime_commit": provenance["runtime_commit"], "schema": report["schema"]}


def synthetic_finite_difference(objective, alpha, *, h=FD_H):
    if h != FD_H or alpha not in ALPHAS:
        raise ValueError("finite difference uses fixed h and preregistered alpha")
    variable = torch.tensor(float(alpha), dtype=torch.float64, requires_grad=True)
    value = objective(variable)
    derivative = torch.autograd.grad(value, variable)[0]
    with torch.no_grad():
        finite = (objective(variable.detach() + h) - objective(variable.detach() - h)) / (2 * h)
    return {"alpha": alpha, "h": h, "autograd": _finite(derivative, "synthetic derivative"),
            "finite_difference": _finite(finite, "synthetic finite difference"),
            "absolute_error": abs(float(derivative) - float(finite))}


def run(args):
    output, source = Path(args.output).resolve(), Path(args.state_dir).resolve()
    traj_dir, traj_paths, traj_hashes, traj_report, experiment, checkpoint = failure._load_trajectory(
        args.trajectory_dir, args.expected_trajectory_commit)
    alignment_path = Path(args.alignment_report).resolve()
    if (output.exists() or output.is_relative_to(source) or output.is_relative_to(traj_dir)
            or output.is_relative_to(alignment_path.parent)):
        raise FileExistsError("write a create-only report outside all immutable input directories")
    state, bank, cfg, source_metadata = group_audit.load_frozen_source(
        source, group_audit.LEGACY_COMMIT, legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength)
    if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
        raise ValueError("trajectory does not reference the supplied source")
    source_hashes = {name: group_audit.file_sha256(source / name) for name in
                     ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")}
    probe_hash = state.get("probe_bank_artifact", {}).get("sha256")
    if probe_hash != traj_report.get("probe_sha256"):
        raise ValueError("trajectory/source probe lineage mismatch")
    _alignment_report, alignment_metadata = load_alignment_report(
        alignment_path, traj_hashes, traj_report, source_hashes, probe_hash)
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices), group_audit.frozen_environment(
            state["fingerprint"], source_metadata["decoder_strengths"]):
        model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        before_hash = safe.state_hash(model.state_dict())
        train_batch = {key: value.to(device) for key, value in
                       group_audit.materialize_transaction(bank, cfg, 0).items()}
        probe, loaded_probe_hash = safe.load_probe(source, state, bank, cfg)
        final_batch, final_metadata = alignment.combine_final_banks(failure.final_banks(bank, probe, cfg))
        final_batch = {key: value.to(device) for key, value in final_batch.items()}
        with failure.preserve_model_runtime(model), torch.enable_grad():
            train = audit_batch(model, train_batch, cfg, alignment.train_metadata(train_batch), "train")
            final = audit_batch(model, final_batch, cfg, final_metadata, "final")
        after_hash = safe.state_hash(model.state_dict())
    if before_hash != after_hash or after_hash != traj_report["final_state_sha256"]:
        raise RuntimeError("read-only scale-response audit changed model state")
    if loaded_probe_hash != probe_hash:
        raise RuntimeError("probe changed during scale-response audit")
    final_parity = validate_alpha_one_final_metrics(final["case_level"], traj_report["final"])
    train_curves, final_curves, aggregate_curves = build_curves(train, final)
    answers = scientific_answers(final_curves, aggregate_curves)
    report = {
        "schema": SCHEMA, "completed": True,
        "provenance": {
            "runtime_commit": runtime_commit, "source": source_metadata,
            "source_sha256_including_probe": source_hashes,
            "trajectory_commit": args.expected_trajectory_commit,
            "trajectory_directory": str(traj_dir), "trajectory_sha256": traj_hashes,
            "trajectory_final_state_sha256": traj_report["final_state_sha256"],
            "probe_sha256": probe_hash, "alignment_audit": alignment_metadata,
            "decoder_protocol": DECODER_PROTOCOL,
            "gradient_protocol": m.REFINER_TANGENT_GRADIENT_PROTOCOL,
            "objective_protocol": m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL,
            "legacy_decoder_strengths": source_metadata["decoder_strengths"],
            "implementation_sha256": {Path(path).name: group_audit.file_sha256(path) for path in
                (__file__, m.__file__, alignment.__file__, failure.__file__, group_audit.__file__, safe.__file__)},
        },
        "fixed_alpha_grid": {"values": list(ALPHAS), "keys": list(ALPHA_KEYS),
                             "preregistered": True, "adaptive_extension_allowed": False,
                             "counterfactual_response_not_tuning": True},
        "train_transaction_0": train,
        "fixed_final_64": {**final, "alpha_one_final_metric_parity": final_parity},
        "group_response_curves": {"train": train_curves, "fixed_final": final_curves,
                                  "fixed_final_aggregates": aggregate_curves},
        "decoder_attenuation": {name: {key: row["decoder_attenuation"] for key, row in
            ((point["alpha"], point) for point in curve["points"])} for name, curve in final_curves.items()},
        "cap_saturation": {name: {key: row["cap_saturation"] for key, row in
            ((point["alpha"], point) for point in curve["points"])} for name, curve in final_curves.items()},
        "objective_gate_response": {name: {"response_shape": curve["response_shape"],
                                            "descriptive_only": True}
                                    for name, curve in final_curves.items()},
        "scientific_answers": answers,
        "finite_difference_controls": {"fixed_h": FD_H, "all_grid_points_tested_synthetically": True,
                                       "probe_used_for_h": False},
        "optimizer": None, "optimizer_steps": 0, "parameter_update_performed": False,
        "model_state_unchanged": True, "checkpoint_selection_performed": False,
        "scale_selection_performed": False, "scientific_acceptance": False,
        "publish_allowed": False, "pilot_allowed": False,
        "claim_boundary": (
            "Alpha values other than one are counterfactual mechanism evidence only. No scale is selected, no "
            "inference behavior is changed, and Pilot remains forbidden."),
    }
    for name, digest in traj_hashes.items():
        if group_audit.file_sha256(traj_paths[name]) != digest:
            raise RuntimeError("trajectory artifact changed during read-only audit")
    for name, digest in source_hashes.items():
        if group_audit.file_sha256(source / name) != digest:
            raise RuntimeError("source artifact changed during read-only audit")
    if group_audit.file_sha256(alignment_path) != alignment_metadata["sha256"]:
        raise RuntimeError("alignment report changed during read-only audit")
    failure._exclusive_json(output, report)
    print(json.dumps({"stage": "refiner_temporal_scale_response_audit_complete",
                      "output": str(output), "fixed_alpha_grid": list(ALPHAS),
                      "optimizer_steps": 0, "scale_selection_performed": False,
                      "scientific_acceptance": False, "publish_allowed": False,
                      "pilot_allowed": False}, allow_nan=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--alignment-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--expected-trajectory-commit", default=failure.TRAJECTORY_COMMIT)
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
