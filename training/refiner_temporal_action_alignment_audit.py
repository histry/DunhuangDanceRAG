"""Read-only alignment audit for the fixed Refiner geometric action.

The audit compares the final model action with the negative temporal and
endpoint scientific-deficit gradients at the exact zero origin and at the
current output.  It never constructs an optimizer, changes the forward
decoder, selects a checkpoint, or authorizes Pilot.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import json
import math
from pathlib import Path

import torch

from motion_geometry.physical import EXTREMITY_JOINTS
from training import motion_models as m
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_safe_start_diagnostics as safe


SCHEMA = "refiner_temporal_action_alignment_audit_v1"
TRAJECTORY_COMMIT = failure.TRAJECTORY_COMMIT
REVIEWED_MAIN_BASELINE = "2dc0fcb9606cda7ad44fc3ae9fea15ef13fca1e6"
EXPECTED_TANGENT_PROTOCOL = "soft_confidence_true_chain_rule_v2"
OBJECTIVES = ("temporal", "endpoint")
POINTS = ("zero_origin", "current_output")
SPACES = (
    "raw_all_geometry",
    "raw_supported_geometry",
    "soft_masked_supported_geometry",
)
EXPECTED_FINAL_FAILURES = {
    "cases": 64,
    "temporal_failed": 64,
    "endpoint_failed": 50,
    "physical_failed": 0,
    "reference_fidelity_failed": 0,
    "clean_identity_failed": 0,
}
PARITY_ATOL_CPU = 1.0e-7
PARITY_ATOL_CUDA = 2.0e-6


def _geometry_blocks():
    extremities = tuple(int(index) for index in EXTREMITY_JOINTS)
    expected = (7, 8, 10, 11, 20, 21, 22, 23)
    if extremities != expected:
        raise ValueError(f"canonical extremity joints changed: {extremities!r}")
    if len(set(extremities)) != len(extremities) or any(not 0 <= j < 24 for j in extremities):
        raise ValueError("invalid canonical extremity joint set")
    body = tuple(j for j in range(24) if j not in extremities)
    result = {"root_translation": tuple(range(3))}
    for name, joints in (("body_joints", body), ("extremity_joints", extremities)):
        result[name] = tuple(index for joint in joints for index in range(3 + 3 * joint, 6 + 3 * joint))
    covered = set().union(*(set(value) for value in result.values()))
    if covered != set(range(75)) or sum(len(value) for value in result.values()) != 75:
        raise ValueError("root/body/extremity geometry blocks do not partition 75D action")
    return result


GEOMETRY_BLOCKS = _geometry_blocks()


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f"nonfinite {label}")
    return value


def _quantile(values, q):
    if not values:
        return None
    return _finite(torch.quantile(torch.tensor(values, dtype=torch.float64), q), "quantile")


def _median(values):
    return _quantile(values, 0.5)


def _capture_model_output(model):
    class Capture:
        def __init__(self):
            self.values = []

        def __enter__(self):
            self.handle = model.out.register_forward_hook(
                lambda _module, _inputs, output: self.values.append(output))
            return self.values

        def __exit__(self, _kind, _value, _traceback):
            self.handle.remove()

    return Capture()


def _effective_geometry_mask(batch, cfg):
    joint, root, _contact = m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)
    return torch.cat((
        root.expand(root.shape[:-1] + (3,)),
        joint[..., None].expand(joint.shape + (3,)).reshape(joint.shape[:-1] + (72,)),
    ), dim=-1).clamp(0.0, 1.0)


def _space_vectors(action, gradient, soft_mask, space):
    if action.shape != gradient.shape or action.shape != soft_mask.shape or action.shape[-1] != 75:
        raise ValueError("alignment vectors must have identical [case,frame,75] shapes")
    support = soft_mask > 0
    if space == "raw_all_geometry":
        return action, gradient
    if space == "raw_supported_geometry":
        return action * support, gradient * support
    if space == "soft_masked_supported_geometry":
        return action * soft_mask, gradient * soft_mask
    raise ValueError(f"unknown alignment space: {space}")


def alignment_stats(action, gradient):
    """Per-case alignment with ``-gradient``; zero vectors have null cosine."""
    if action.shape != gradient.shape or action.ndim < 2:
        raise ValueError("action and gradient shapes must match with a case axis")
    a = action.detach().double().reshape(action.shape[0], -1)
    g = gradient.detach().double().reshape(gradient.shape[0], -1)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(g).all()):
        raise FloatingPointError("nonfinite alignment vector")
    an, gn = a.norm(dim=1), g.norm(dim=1)
    dot = (g * a).sum(dim=1)
    rows = []
    for index in range(a.shape[0]):
        action_norm = _finite(an[index], "action norm")
        gradient_norm = _finite(gn[index], "gradient norm")
        directional = _finite(dot[index], "directional derivative")
        cosine = None
        if action_norm != 0.0 and gradient_norm != 0.0:
            cosine = max(-1.0, min(1.0, -directional / (action_norm * gradient_norm)))
        rows.append({
            "action_norm": action_norm,
            "gradient_norm": gradient_norm,
            "cosine_to_negative_gradient": cosine,
            "directional_derivative": directional,
            "local_descent": directional < 0.0,
            "local_ascent": directional > 0.0,
            "local_flat": directional == 0.0,
            "exact_zero_action": action_norm == 0.0,
            "exact_zero_gradient": gradient_norm == 0.0,
        })
    return rows


def _objective_alignment(action, gradient, soft_mask, values):
    if values.shape != (action.shape[0],):
        raise ValueError("scientific deficit must be one scalar per case")
    by_space = {}
    by_block = {name: {} for name in GEOMETRY_BLOCKS}
    for space in SPACES:
        a_value, g_value = _space_vectors(action, gradient, soft_mask, space)
        by_space[space] = alignment_stats(a_value, g_value)
        for name, indices in GEOMETRY_BLOCKS.items():
            by_block[name][space] = alignment_stats(a_value[..., indices], g_value[..., indices])
    outside = gradient * (soft_mask == 0)
    outside_stats = alignment_stats(torch.zeros_like(outside), outside)
    rows = []
    for case in range(action.shape[0]):
        rows.append({
            "objective_value": _finite(values[case].detach(), "objective value"),
            "spaces": {space: by_space[space][case] for space in SPACES},
            "blocks": {name: {space: by_block[name][space][case] for space in SPACES}
                       for name in GEOMETRY_BLOCKS},
            "gradient_outside_decoder_support": {
                "norm": outside_stats[case]["gradient_norm"],
                "exactly_zero": outside_stats[case]["exact_zero_gradient"],
            },
        })
    return rows


def _scientific_terms(prediction, reference, seam, cfg):
    _loss, terms = m._observable_refiner_objective(
        prediction, reference, seam, cfg, reduction="none")
    return {
        "temporal": terms["temporal_scientific_deficit"],
        "endpoint": terms["endpoint_scientific_deficit"],
    }


def _gradients(terms, target):
    result = {}
    for index, name in enumerate(OBJECTIVES):
        value = terms[name].sum()
        if value.requires_grad:
            gradient = torch.autograd.grad(
                value, target, retain_graph=index + 1 < len(OBJECTIVES), allow_unused=True)[0]
        else:
            gradient = None
        result[name] = torch.zeros_like(target) if gradient is None else gradient
    return result


def production_current_point(model, batch, cfg):
    """One production forward, strict manual decoder parity, then true VJPs."""
    count = int(batch["clean"].shape[0])
    with _capture_model_output(model) as captured:
        prediction, _identity = m._refiner_batch_outputs(model, batch, cfg)
    if len(captured) != 1:
        raise RuntimeError("alignment audit requires exactly one production model forward")
    raw_all = captured[0].transpose(1, 2)
    if raw_all.shape != (2 * count, batch["bad"].shape[1], m.PRODUCT_STATE_DIM):
        raise ValueError("captured production raw output shape mismatch")
    masks = m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)
    manual = m._decode_product_refiner_output(batch["bad"], raw_all[:count], *masks, cfg)
    error = (prediction - manual).detach().abs()
    maximum = _finite(error.max(), "production parity error")
    tolerance = PARITY_ATOL_CUDA if prediction.is_cuda else PARITY_ATOL_CPU
    if maximum > tolerance:
        raise RuntimeError(
            f"production/manual decoder parity failed: max_abs={maximum} tolerance={tolerance}")
    terms = _scientific_terms(prediction, batch["bad"], batch["seam"], cfg)
    gradients_cf = _gradients(terms, captured[0])
    action = raw_all[:count, ..., 4:].detach()
    result = {
        "action": action.cpu(),
        "values": {name: value.detach().cpu() for name, value in terms.items()},
        "gradients": {name: gradient.transpose(1, 2)[:count, ..., 4:].detach().cpu()
                      for name, gradient in gradients_cf.items()},
        "production_forward_parity": {
            "verified": True,
            "max_abs_error": maximum,
            "rtol": 0.0,
            "atol": tolerance,
            "model_forward_calls": 1,
        },
    }
    del prediction, _identity, manual, terms, gradients_cf, raw_all, captured
    return result


def zero_origin_point(action, batch, cfg):
    """Decode exact-zero geometric action with the identical production decoder."""
    count, frames = action.shape[:2]
    output = batch["bad"].new_zeros((count, frames, m.PRODUCT_STATE_DIM), requires_grad=True)
    masks = m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)
    prediction = m._decode_product_refiner_output(batch["bad"], output, *masks, cfg)
    terms = _scientific_terms(prediction, batch["bad"], batch["seam"], cfg)
    gradients = _gradients(terms, output)
    result = {
        "action": action.detach().cpu(),
        "values": {name: value.detach().cpu() for name, value in terms.items()},
        "gradients": {name: gradient[..., 4:].detach().cpu()
                      for name, gradient in gradients.items()},
        "decoded_origin_exact_identity": bool(torch.equal(prediction.detach(), batch["bad"])),
        "raw_geometry_origin_exact_zero": bool((output[..., 4:].detach() == 0).all()),
    }
    if not result["decoded_origin_exact_identity"] or not result["raw_geometry_origin_exact_zero"]:
        raise RuntimeError("zero-origin decoder is not the exact production identity")
    del output, prediction, terms, gradients
    return result


def audit_batch(model, batch, cfg, metadata):
    """Audit one complete batch; current graph is released before zero graph."""
    model.eval()
    soft_mask = _effective_geometry_mask(batch, cfg).detach().cpu()
    current = production_current_point(model, batch, cfg)
    action = current["action"]
    current_rows = {
        name: _objective_alignment(action, current["gradients"][name], soft_mask, current["values"][name])
        for name in OBJECTIVES
    }
    parity = current["production_forward_parity"]
    del current
    if torch.cuda.is_available() and torch.device(cfg.device).type == "cuda":
        torch.cuda.empty_cache()
    zero = zero_origin_point(action.to(batch["bad"].device), batch, cfg)
    zero_rows = {
        name: _objective_alignment(action, zero["gradients"][name], soft_mask, zero["values"][name])
        for name in OBJECTIVES
    }
    origin = {key: zero[key] for key in
              ("decoded_origin_exact_identity", "raw_geometry_origin_exact_zero")}
    del zero
    rows = []
    for index, fields in enumerate(metadata):
        if fields["case_index"] < 0:
            raise ValueError("invalid case index")
        rows.append({
            **fields,
            "active_geometry_fraction": _finite((soft_mask[index] > 0).double().mean(),
                                                  "active geometry fraction"),
            "zero_origin": {name: zero_rows[name][index] for name in OBJECTIVES},
            "current_output": {name: current_rows[name][index] for name in OBJECTIVES},
        })
    return {
        "rows": rows,
        "production_forward_parity": parity,
        "zero_origin_contract": origin,
        "spaces": {
            "raw_all_geometry": "raw 75D action and raw-action gradient",
            "raw_supported_geometry": "both vectors restricted to effective decoder support > 0",
            "soft_masked_supported_geometry": (
                "both vectors multiplied by the actual effective soft confidence; this is a diagnostic "
                "weighted-coordinate comparison, not the raw alpha directional derivative"),
        },
    }


def train_metadata(batch):
    if batch["group"].shape != (192,) or any(int((batch["group"] == i).sum()) != 48 for i in range(4)):
        raise ValueError("TRAIN transaction 0 must contain four complete 48-case groups")
    counters = Counter()
    rows = []
    for label in batch["group"].detach().cpu().tolist():
        group = group_audit.GROUPS[int(label)]
        role, width_name = group.rsplit("_", 1)
        width = 10 if width_name == "short" else 28
        rows.append({"split": "train", "role": role, "width": width, "group": group,
                     "case_index": counters[group]})
        counters[group] += 1
    return rows


def combine_final_banks(banks):
    parts, metadata = [], []
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            part = banks[(split, role)]
            if int(part["clean"].shape[0]) != 16:
                raise ValueError("each fixed final split/role bank must contain 16 cases")
            parts.append(part)
            counters = Counter()
            widths = (part["seam"][..., 0] >= 0.5).sum(1).detach().cpu().tolist()
            for original_index, width in enumerate(widths):
                width = int(width)
                if width not in (10, 28):
                    raise ValueError("fixed final seam width must be 10 or 28")
                group = f"{role}/{width}"
                metadata.append({"split": split, "role": role, "width": width, "group": group,
                                 "case_index": counters[group], "bank_case_index": original_index})
                counters[group] += 1
            if counters[f"{role}/10"] != 8 or counters[f"{role}/28"] != 8:
                raise ValueError("each fixed final split/role bank must contain eight cases per width")
    keys = set(parts[0])
    if any(set(part) != keys for part in parts[1:]):
        raise ValueError("fixed final bank fields differ")
    combined = {key: torch.cat([part[key] for part in parts]) for key in keys}
    if combined["clean"].shape[0] != 64 or len(metadata) != 64:
        raise ValueError("fixed final audit must contain exactly 64 cases")
    return combined, metadata


def validate_confirmed_final_failures(final_report):
    metrics = final_report.get("metrics", {})
    if set(metrics) != {"seen", "new_position"}:
        raise ValueError("trajectory final report lacks both fixed final splits")
    counts = Counter(cases=0, temporal_failed=0, endpoint_failed=0,
                     physical_failed=0, reference_fidelity_failed=0,
                     clean_identity_failed=0)
    for split in ("seen", "new_position"):
        single = metrics[split].get("windows", [])
        cross = metrics[split].get("cross_event", {}).get("windows", [])
        if len(single) != 16 or len(cross) != 16:
            raise ValueError("trajectory final report must contain 16 single and 16 cross cases per split")
        for role, rows in (("single_recording", single), ("cross_event", cross)):
            for row in rows:
                observable = row["observable"]
                counts["cases"] += 1
                counts["temporal_failed"] += not bool(observable["temporal_accepted"])
                counts["endpoint_failed"] += not bool(observable["endpoint_accepted"])
                physical = (observable.get("physical_non_regression") if role == "single_recording"
                            else row.get("safety"))
                if not isinstance(physical, dict) or "accepted" not in physical:
                    raise ValueError("trajectory final physical gate is incomplete")
                counts["physical_failed"] += not bool(physical["accepted"])
                counts["reference_fidelity_failed"] += not bool(
                    observable["reference_fidelity_accepted"])
                if role == "single_recording":
                    identity = row.get("clean_identity")
                    if not isinstance(identity, dict) or "accepted" not in identity:
                        raise ValueError("trajectory final clean identity gate is incomplete")
                    counts["clean_identity_failed"] += not bool(identity["accepted"])
    result = dict(counts)
    if result != EXPECTED_FINAL_FAILURES:
        raise RuntimeError(f"fixed-final failure facts changed: {result!r}")
    result["clean_identity_cases"] = 32
    result["confirmed_against_fixed_trajectory_report"] = True
    return result


def _summarize_metric(records):
    cosines = [record["cosine_to_negative_gradient"] for record in records
               if record["cosine_to_negative_gradient"] is not None]
    directional = [record["directional_derivative"] for record in records]
    action_norms = [record["action_norm"] for record in records]
    gradient_norms = [record["gradient_norm"] for record in records]
    return {
        "cases": len(records),
        "defined_cosine_cases": len(cosines),
        "exact_zero_gradient_cases": sum(record["exact_zero_gradient"] for record in records),
        "exact_zero_action_cases": sum(record["exact_zero_action"] for record in records),
        "cosine_median": _median(cosines),
        "cosine_min": min(cosines) if cosines else None,
        "cosine_max": max(cosines) if cosines else None,
        "positive_cosine_cases": sum(value > 0 for value in cosines),
        "negative_cosine_cases": sum(value < 0 for value in cosines),
        "zero_cosine_cases": sum(value == 0 for value in cosines),
        "local_descent_cases": sum(record["local_descent"] for record in records),
        "local_ascent_cases": sum(record["local_ascent"] for record in records),
        "local_flat_cases": sum(record["local_flat"] for record in records),
        "directional_derivative_median": _median(directional),
        "action_norm_median": _median(action_norms),
        "gradient_norm_median": _median(gradient_norms),
    }


def summarize(rows, group_fields):
    grouped = {}
    for row in rows:
        key = "/".join(str(row[field]) for field in group_fields)
        grouped.setdefault(key, []).append(row)
    result = {}
    for key, selected in grouped.items():
        record = {"cases": len(selected), "points": {}}
        for point in POINTS:
            record["points"][point] = {}
            for objective in OBJECTIVES:
                objective_rows = [row[point][objective] for row in selected]
                spaces = {
                    space: _summarize_metric([value["spaces"][space] for value in objective_rows])
                    for space in SPACES
                }
                blocks = {name: {
                    space: _summarize_metric([value["blocks"][name][space] for value in objective_rows])
                    for space in SPACES} for name in GEOMETRY_BLOCKS}
                outside = [value["gradient_outside_decoder_support"]["norm"] for value in objective_rows]
                record["points"][point][objective] = {
                    "spaces": spaces,
                    "soft_masked_cosine_median": spaces["soft_masked_supported_geometry"]["cosine_median"],
                    "blocks": blocks,
                    "gradient_outside_decoder_support": {
                        "exact_zero_cases": sum(value == 0 for value in outside),
                        "norm_median": _median(outside),
                        "norm_max": max(outside),
                    },
                }
        result[key] = record
    return result


def _direction_answer(summary):
    defined = summary["defined_cosine_cases"]
    if defined == 0:
        return "undefined_exact_zero_gradient_or_action"
    positive, negative = summary["positive_cosine_cases"], summary["negative_cosine_cases"]
    if positive > defined / 2:
        return "mostly_aligned_with_negative_gradient"
    if negative > defined / 2:
        return "mostly_opposed_to_negative_gradient"
    return "mixed_or_tied"


def _scaling_answer(summary):
    cases = summary["cases"]
    if summary["local_descent_cases"] > cases / 2:
        return "increasing_current_action_scale_is_mostly_local_descent"
    if summary["local_ascent_cases"] > cases / 2:
        return "increasing_current_action_scale_is_mostly_local_ascent"
    return "mixed_or_locally_flat"


def scientific_answers(train_rows, final_rows):
    def aggregate(rows, point, objective, space="raw_supported_geometry", block=None):
        if block is None:
            values = [row[point][objective]["spaces"][space] for row in rows]
        else:
            values = [row[point][objective]["blocks"][block][space] for row in rows]
        return _summarize_metric(values)

    train_gradient = aggregate(train_rows, "zero_origin", "temporal")
    final_zero = aggregate(final_rows, "zero_origin", "temporal")
    final_current = aggregate(final_rows, "current_output", "temporal", "raw_all_geometry")
    endpoint = aggregate(final_rows, "zero_origin", "endpoint")
    outside = [row[point][objective]["gradient_outside_decoder_support"]["norm"]
               for row in train_rows + final_rows for point in POINTS for objective in OBJECTIVES]
    return {
        "train_temporal_gradient_present": {
            "answer": train_gradient["exact_zero_gradient_cases"] < train_gradient["cases"],
            "exact_zero_gradient_cases": train_gradient["exact_zero_gradient_cases"],
            "cases": train_gradient["cases"],
        },
        "final_temporal_gradient_present": {
            "answer": final_zero["exact_zero_gradient_cases"] < final_zero["cases"],
            "exact_zero_gradient_cases": final_zero["exact_zero_gradient_cases"],
            "cases": final_zero["cases"],
        },
        "final_model_action_vs_zero_origin_temporal_descent": {
            "answer": _direction_answer(final_zero), **final_zero,
        },
        "current_temporal_action_scaling": {"answer": _scaling_answer(final_current), **final_current},
        "endpoint_control_at_zero_origin": {"answer": _direction_answer(endpoint), **endpoint},
        "temporal_zero_origin_blocks": {
            block: {"answer": _direction_answer(value), **value}
            for block in GEOMETRY_BLOCKS
            for value in [aggregate(final_rows, "zero_origin", "temporal", block=block)]
        },
        "gradient_outside_decoder_support": {
            "all_exact_zero": all(value == 0 for value in outside),
            "nonzero_measurements": sum(value != 0 for value in outside),
            "measurements": len(outside),
            "norm_max": max(outside),
        },
        "interpretation_boundary": (
            "Negative zero-origin cosine supports a local directional mismatch only. Positive cosine with fixed "
            "failure can instead indicate magnitude, nonlinearity, decoder attenuation, or objective/gate scale "
            "mismatch. Exact-zero gradients indicate a local dead zone or conditioning issue. This audit does not "
            "recommend an architecture or tune a scale."),
    }


def finite_difference_directional_check(objective, action, alpha, *, h=1.0e-3):
    """Fixed-h central difference used by synthetic unit controls."""
    if h != 1.0e-3 or alpha not in (0.0, 1.0):
        raise ValueError("finite-difference control uses fixed h=1e-3 at alpha 0 or 1")
    direction = action.detach()
    variable = (direction * alpha).detach().requires_grad_(True)
    value = objective(variable)
    if value.ndim != 0 or not value.requires_grad:
        raise ValueError("finite-difference objective must be a differentiable scalar")
    gradient = torch.autograd.grad(value, variable)[0]
    autograd_value = _finite((gradient * direction).sum(), "autograd directional derivative")
    with torch.no_grad():
        finite_difference = _finite(
            (objective(direction * (alpha + h)) - objective(direction * (alpha - h))) / (2 * h),
            "finite-difference directional derivative")
    error = abs(autograd_value - finite_difference)
    return {"alpha": alpha, "h": h, "autograd": autograd_value,
            "finite_difference": finite_difference, "absolute_error": error}


def run(args):
    output = Path(args.output).resolve()
    source = Path(args.state_dir).resolve()
    traj_dir, traj_paths, traj_hashes, traj_report, experiment, checkpoint = failure._load_trajectory(
        args.trajectory_dir, args.expected_trajectory_commit)
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(traj_dir):
        raise FileExistsError("write a create-only report outside both immutable input directories")
    state, bank, cfg, source_metadata = group_audit.load_frozen_source(
        source, group_audit.LEGACY_COMMIT,
        legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength)
    if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
        raise ValueError("trajectory does not reference the supplied frozen source")
    probe_descriptor = state.get("probe_bank_artifact", {})
    if traj_report.get("probe_sha256") != probe_descriptor.get("sha256"):
        raise ValueError("trajectory probe provenance does not match frozen source")
    if m.REFINER_TANGENT_GRADIENT_PROTOCOL != EXPECTED_TANGENT_PROTOCOL:
        raise ValueError("temporal alignment audit requires the true soft-confidence chain rule")
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    fixed_failure = validate_confirmed_final_failures(traj_report["final"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    source_hashes = {name: group_audit.file_sha256(source / name) for name in
                     ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")}
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=cuda_devices), group_audit.frozen_environment(
            state["fingerprint"], source_metadata["decoder_strengths"]):
        model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        before_hash = safe.state_hash(model.state_dict())
        train_batch = {key: value.to(device) for key, value in
                       group_audit.materialize_transaction(bank, cfg, 0).items()}
        probe, probe_hash = safe.load_probe(source, state, bank, cfg)
        final_banks = failure.final_banks(bank, probe, cfg)
        final_batch, final_meta = combine_final_banks(final_banks)
        final_batch = {key: value.to(device) for key, value in final_batch.items()}
        with failure.preserve_model_runtime(model), torch.enable_grad():
            train = audit_batch(model, train_batch, cfg, train_metadata(train_batch))
            final = audit_batch(model, final_batch, cfg, final_meta)
        after_hash = safe.state_hash(model.state_dict())
    if before_hash != after_hash or after_hash != traj_report["final_state_sha256"]:
        raise RuntimeError("read-only temporal alignment audit changed model state")
    if probe_hash != traj_report["probe_sha256"]:
        raise RuntimeError("probe hash changed during read-only audit")
    train_summary = summarize(train["rows"], ("group",))
    final_summary = summarize(final["rows"], ("split", "group"))
    answers = scientific_answers(train["rows"], final["rows"])
    report = {
        "schema": SCHEMA,
        "completed": True,
        "provenance": {
            "reviewed_main_baseline": REVIEWED_MAIN_BASELINE,
            "runtime_commit": runtime_commit,
            "trajectory_commit": args.expected_trajectory_commit,
            "trajectory_directory": str(traj_dir),
            "trajectory_sha256": traj_hashes,
            "trajectory_experiment_sha256": traj_report["experiment_sha256"],
            "trajectory_final_state_sha256": traj_report["final_state_sha256"],
            "source": source_metadata,
            "source_sha256_including_probe": source_hashes,
            "probe_sha256": probe_hash,
            "implementation_sha256": {Path(path).name: group_audit.file_sha256(path) for path in
                (__file__, m.__file__, failure.__file__, group_audit.__file__, safe.__file__)},
        },
        "contract": {
            "action_layout": {"full_output_dim": 79, "contact_excluded": [0, 4],
                              "geometry_in_full_output": [4, 79], "geometry_dim": 75,
                              "blocks": {key: list(value) for key, value in GEOMETRY_BLOCKS.items()},
                              "canonical_extremity_joints": list(EXTREMITY_JOINTS)},
            "train": {"transaction_index": 0, "cases": 192, "cases_per_group": 48,
                      "single_full_transaction_forward": True},
            "fixed_final": {"cases": 64, "cases_per_split_role_width": 8,
                            "checkpoint_selection": False, "single_combined_forward": True},
            "gradient_points": list(POINTS), "objectives": list(OBJECTIVES),
            "spaces": train["spaces"],
        },
        "confirmed_fixed_final_failure": fixed_failure,
        "train_transaction_0": {"case_level": train["rows"], "group_summaries": train_summary,
                                "production_forward_parity": train["production_forward_parity"],
                                "zero_origin_contract": train["zero_origin_contract"]},
        "fixed_final_64": {"case_level": final["rows"], "group_summaries": final_summary,
                           "production_forward_parity": final["production_forward_parity"],
                           "zero_origin_contract": final["zero_origin_contract"]},
        "scientific_answers": answers,
        "finite_difference_controls": {
            "performed_in_unit_tests": True,
            "fixed_h": 1.0e-3,
            "points": [0.0, 1.0],
            "objectives": list(OBJECTIVES),
            "probe_used_for_step_size_selection": False,
        },
        "optimizer": None,
        "optimizer_steps": 0,
        "parameter_update_performed": False,
        "model_state_unchanged": True,
        "checkpoint_selection": False,
        "checkpoint_selection_performed": False,
        "probe_used_for_fixed_final_evaluation_only": True,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "claim_boundary": (
            "Local action-gradient alignment is attribution evidence at one fixed checkpoint. It neither proves "
            "a multi-head mechanism nor authorizes architecture changes, scale tuning, training, or Pilot."),
    }
    for name, digest in traj_hashes.items():
        if group_audit.file_sha256(traj_paths[name]) != digest:
            raise RuntimeError("trajectory artifact changed during read-only audit")
    for name, digest in source_hashes.items():
        if group_audit.file_sha256(source / name) != digest:
            raise RuntimeError("frozen source artifact changed during read-only audit")
    failure._exclusive_json(output, report)
    for name, answer in answers.items():
        print(json.dumps({"stage": "scientific_answer", "question": name, "answer": answer},
                         allow_nan=False), flush=True)
    for name, row in final_summary.items():
        print(json.dumps({"stage": "final_group", "group": name, "summary": row},
                         allow_nan=False), flush=True)
    print(json.dumps({"stage": "refiner_temporal_action_alignment_audit_complete",
                      "output": str(output), "optimizer_steps": 0,
                      "scientific_acceptance": False, "publish_allowed": False,
                      "pilot_allowed": False}, allow_nan=False), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--expected-trajectory-commit", default=TRAJECTORY_COMMIT)
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
