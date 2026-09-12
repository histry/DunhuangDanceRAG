"""V15.15c/d exact-radius constraint-aware Adapter development probe.

The probe freezes the V15.13 shared Refiner, fits only the zero-initialized
75D Adapter against frozen V15.14h projected directions, and performs exact
fixed-Guard audits. V15.15d adds case-isolated, allowance-scaled Guard
restoration and cross-group balanced optimization. It cannot publish a formal
checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from torch.utils.checkpoint import checkpoint as activation_checkpoint

from motion_geometry.product_manifold import product_exp_torch, product_log_torch
from training import motion_models as m
from training import refiner_case_local_full_tangent_oracle as oracle
from training import refiner_group_local_nullspace_cone_probe as group_probe
from training import refiner_projected_candidate_probe as projected_probe


V15_15C_SCHEMA = "refiner_v15_15c_exact_radius_constraint_adapter_probe_v3"
V15_15D_SCHEMA = (
    "refiner_v15_15d_case_isolated_fixed_guard_restoration_probe_v1"
)
V15_15E_TEACHER_SCHEMA = (
    "refiner_v15_15e_multi_transaction_observable_adapter_teacher_bank_v1"
)
EXACT_RADIUS_NORMALIZATION_EPS = 1.0e-8
FORMAL_READINESS_CASES = (16, 18, 20, 29)
CROSS_GROUPS = ("cross_short", "cross_long")


def _to_device(value, device):
    if m.torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _slice_batch(batch, start, stop):
    return {
        key: value[int(start):int(stop)]
        for key, value in batch.items()
    }


def _adapter_batch_outputs(model, batch, cfg):
    adapter_trace = {}
    outputs = model(
        batch["bad"],
        batch["cond"],
        batch["seam"],
        batch["joint"],
        adapter_trace=adapter_trace,
    )
    repair_masks = m._refiner_decode_masks(
        batch["joint"],
        batch["root"],
        batch["contact"],
        batch["seam"],
        cfg,
    )
    prediction = m._decode_product_refiner_output(
        batch["bad"], outputs, *repair_masks, cfg
    )
    adapter_output = m.torch.cat(
        [
            m.torch.zeros_like(outputs[..., :4]),
            adapter_trace["tangent"],
        ],
        dim=-1,
    )
    shared_prediction = m._decode_product_refiner_output(
        batch["bad"], outputs - adapter_output, *repair_masks, cfg
    )
    adapter_trace["decoder_consistent_tangent"] = product_log_torch(
        shared_prediction,
        prediction,
    )
    return prediction, adapter_trace


def _load_source_model(source, cfg, device, adapter_state_path=None):
    state = m.torch.load(
        source / "diagnostic_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
        observable_adapter=True,
        residual_taper_frames=int(cfg.product_refiner_residual_taper_frames),
    ).to(device)
    incompatible = model.load_state_dict(
        state["model_state_dict"], strict=False
    )
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed = (
        "observable_adapter_net.",
        "observable_adapter_gate.",
        "observable_adapter_gate_floor",
    )
    invalid_missing = [
        key for key in missing if not key.startswith(allowed)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError({
            "invalid_missing_keys": invalid_missing,
            "unexpected_keys": unexpected,
        })
    if adapter_state_path is not None:
        payload = m.torch.load(
            adapter_state_path,
            map_location="cpu",
            weights_only=False,
        )
        adapter_state = payload.get("adapter_state_dict")
        if not isinstance(adapter_state, dict) or not adapter_state:
            raise RuntimeError("adapter state does not contain Adapter weights")
        invalid_keys = [
            key for key in adapter_state
            if not key.startswith("observable_adapter_")
        ]
        if invalid_keys:
            raise RuntimeError({"invalid_adapter_state_keys": invalid_keys})
        current_state = model.state_dict()
        for key, value in adapter_state.items():
            if key not in current_state:
                raise RuntimeError({"unknown_adapter_state_key": key})
            if current_state[key].shape != value.shape:
                raise RuntimeError({
                    "adapter_state_shape_mismatch": key,
                    "expected": tuple(current_state[key].shape),
                    "actual": tuple(value.shape),
                })
            current_state[key] = value
        model.load_state_dict(current_state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.observable_adapter_net,
        model.observable_adapter_gate,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return model, missing


def _case_vectors(adapter_tangent, teacher, ownership, case_index):
    active = ownership[int(case_index)].expand_as(
        adapter_tangent[int(case_index)]
    )
    predicted = adapter_tangent[int(case_index)][active]
    target = teacher.to(adapter_tangent.device)[active]
    return predicted, target


def _fixed_guard_penalty(
    model,
    batch,
    cfg,
    prediction,
    identity,
    anchor,
    relative,
    absolute,
):
    values = projected_probe._guard_values_for_prediction(
        model, batch, cfg, prediction, identity
    )
    terms = []
    for key, value in values.items():
        allowance = max(
            abs(float(anchor[key])) * float(relative[key]),
            float(absolute[key]),
        )
        limit = float(anchor[key]) + allowance
        scale = max(abs(limit), abs(float(anchor[key])), allowance, 1.0e-6)
        terms.append(m.torch.relu((value - limit) / scale).square())
    return m.torch.stack(terms).mean(), values


def _fixed_guard_restoration_terms(
    values,
    anchor,
    relative,
    absolute,
    safety_fraction,
):
    """Return an allowance-scaled sum over every unsafe Guard component.

    The final fixed Guard is unchanged.  ``safety_fraction`` only moves the
    differentiable training target inside the existing fixed allowance.  A
    sum is intentional: every simultaneously active metric receives a
    gradient, instead of the current maximum component monopolizing it.
    """
    terms = []
    details = {}
    for key in sorted(values):
        anchor_value = float(anchor[key])
        allowance = max(
            abs(anchor_value) * float(relative[key]),
            float(absolute[key]),
        )
        if not math.isfinite(allowance) or allowance <= 0.0:
            raise RuntimeError({
                "invalid_fixed_guard_allowance": key,
                "allowance": allowance,
            })
        final_limit = anchor_value + allowance
        training_limit = anchor_value + float(safety_fraction) * allowance
        normalized = (values[key] - training_limit) / allowance
        excess = m.torch.relu(normalized)
        terms.append(excess)
        details[key] = {
            "fixed_anchor": anchor_value,
            "absolute_allowance": allowance,
            "final_absolute_limit": final_limit,
            "training_safety_limit": training_limit,
            "candidate": float(values[key].detach()),
            "normalized_training_excess": float(excess.detach()),
            "active": bool(float(excess.detach()) > 0.0),
        }
    if not terms:
        raise RuntimeError("empty fixed Guard value set")
    return m.torch.stack(terms).sum(), details


def _case_isolated_fixed_guard_restoration(
    model,
    batch,
    cfg,
    baseline,
    identity,
    training_tangent,
    ownership,
    case_index,
    anchor,
    relative,
    absolute,
    safety_fraction,
):
    permitted = _owned_case_mask(
        ownership, training_tangent, int(case_index)
    )
    guard_keys = tuple(sorted(anchor))

    def checkpointed_guard_values(tangent):
        """Recompute one case's FK/Guard graph during backward."""
        isolated_tangent = tangent.masked_fill(~permitted, 0.0)
        isolated_prediction = product_exp_torch(
            baseline.detach(), isolated_tangent
        )
        result = projected_probe._guard_values_for_prediction(
            model, batch, cfg, isolated_prediction, identity
        )
        missing = [key for key in guard_keys if key not in result]
        extra = sorted(set(result) - set(guard_keys))
        if missing or extra:
            raise RuntimeError({
                "fixed_guard_checkpoint_key_mismatch": {
                    "missing": missing,
                    "extra": extra,
                },
            })
        return tuple(result[key] for key in guard_keys)

    # A multi-transaction bank can contain dozens of cross teachers. Keeping
    # one float64 FK graph per isolated candidate makes peak memory grow with
    # teacher count. Recompute each graph during backward while preserving the
    # exact objective and gradient.
    checkpointed = activation_checkpoint(
        checkpointed_guard_values,
        training_tangent,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    if m.torch.is_tensor(checkpointed):
        checkpointed = (checkpointed,)
    values = dict(zip(guard_keys, checkpointed))
    loss, details = _fixed_guard_restoration_terms(
        values,
        anchor,
        relative,
        absolute,
        safety_fraction,
    )
    isolated_tangent = training_tangent.masked_fill(~permitted, 0.0)
    outside = ~permitted
    outside_max = (
        float(isolated_tangent[outside].abs().max().detach())
        if bool(outside.any()) else 0.0
    )
    return loss, {
        "case_index": int(case_index),
        "aggregation": "sum_normalized_relu",
        "normalization_scale": "fixed_anchor_absolute_allowance",
        "training_safety_fraction": float(safety_fraction),
        "activation_checkpointed": True,
        "checkpoint_reentrant": False,
        "outside_case_or_ownership_abs_max": outside_max,
        "active_metric_count": sum(
            int(row["active"]) for row in details.values()
        ),
        "normalized_excess_sum": float(loss.detach()),
        "metrics": details,
    }


def _continuous_direction_weight(
    guard_pressure,
    nonregression_pressure,
    floor,
    decay,
):
    pressure = (
        guard_pressure.detach() + nonregression_pressure.detach()
    ).clamp_min(0.0)
    return float(floor) + (1.0 - float(floor)) * m.torch.exp(
        -float(decay) * pressure
    )


def _balanced_cross_group_mean(values_by_group):
    missing = [
        group for group in CROSS_GROUPS
        if not values_by_group.get(group)
    ]
    if missing:
        raise RuntimeError({
            "missing_cross_group_training_objectives": missing,
        })
    group_means = [
        m.torch.stack(values_by_group[group]).mean()
        for group in CROSS_GROUPS
    ]
    return m.torch.stack(group_means).mean(), {
        group: float(value.detach())
        for group, value in zip(CROSS_GROUPS, group_means)
    }


def _stratified_cross_group_mean(values_by_group, weights_by_group):
    """Give each cross group equal mass and each transaction equal mass."""
    missing = [
        group for group in CROSS_GROUPS
        if not values_by_group.get(group)
    ]
    if missing:
        raise RuntimeError({
            "missing_cross_group_training_objectives": missing,
        })
    group_means = []
    details = {}
    for group in CROSS_GROUPS:
        values = m.torch.stack(values_by_group[group])
        weights = values.new_tensor(weights_by_group[group])
        if values.numel() != weights.numel():
            if values.numel() % weights.numel() != 0:
                raise RuntimeError("stratified weight/value count mismatch")
            weights = weights.repeat_interleave(
                values.numel() // weights.numel()
            )
        if not bool((weights > 0.0).all()):
            raise RuntimeError("stratified weights must be positive")
        normalized = weights / weights.sum()
        mean = (values * normalized).sum()
        group_means.append(mean)
        details[group] = float(mean.detach())
    return m.torch.stack(group_means).mean(), details


def _owned_case_mask(ownership, tangent, case_index):
    selected = m.torch.zeros_like(ownership, dtype=m.torch.bool)
    selected[int(case_index)] = True
    return (selected & ownership).expand_as(tangent)


def _masked_rms(value, mask):
    selected = value[mask]
    if selected.numel() == 0:
        return value.sum() * 0.0
    return m.torch.sqrt(selected.square().mean().clamp_min(1.0e-24))


def _safe_exact_radius_tangent(
    tangent,
    ownership,
    samples,
    target_rms,
    *,
    eps=EXACT_RADIUS_NORMALIZATION_EPS,
):
    """Normalize each cross case on its applied ownership support.

    A detached lower branch keeps the derivative finite at the zero
    initialized Adapter. Above that safety floor the ordinary vector norm is
    used, so every resolved proposal lies exactly on the requested RMS sphere.
    ``torch.linalg.vector_norm`` intentionally is not passed a nonexistent
    ``eps`` argument.
    """
    result = m.torch.zeros_like(tangent)
    diagnostics = {}
    visited = set()
    for sample in samples:
        if sample["teacher_kind"] != "exact_projected_direction":
            continue
        case_index = int(sample["case_index"])
        if case_index in visited:
            raise RuntimeError(f"duplicate Adapter teacher case {case_index}")
        visited.add(case_index)
        permitted = _owned_case_mask(ownership, tangent, case_index)
        active = tangent[permitted]
        if active.numel() == 0:
            raise RuntimeError(f"empty ownership support for case {case_index}")
        count = active.new_tensor(float(active.numel()))
        raw_norm = m.torch.linalg.vector_norm(active)
        norm_floor = float(eps) * m.torch.sqrt(count)
        stable_norm = m.torch.where(
            raw_norm >= norm_floor,
            raw_norm,
            norm_floor.detach(),
        )
        scale = float(target_rms) * m.torch.sqrt(count) / stable_norm
        local = tangent * permitted.to(tangent.dtype) * scale
        result = result + local
        normalized_rms = _masked_rms(local, permitted)
        case_uid = str(sample.get("case_uid", case_index))
        diagnostics[case_uid] = {
            "normalization_eps": float(eps),
            "active_coordinate_count": int(active.numel()),
            "input_rms": float(_masked_rms(tangent, permitted).detach()),
            "normalization_scale": float(scale.detach()),
            "normalization_floor_active": bool(
                float(raw_norm.detach()) < float(norm_floor.detach())
            ),
            "normalized_tangent_rms": float(normalized_rms.detach()),
            "radius_equality_resolved": bool(
                abs(float(normalized_rms.detach()) - float(target_rms))
                <= max(1.0e-12, float(target_rms) * 1.0e-6)
            ),
        }
    return result, diagnostics


def _isolated_fixed_radius_candidate(
    baseline,
    decoder_adapter_tangent,
    pre_taper_adapter_tangent,
    c2_taper,
    ownership,
    batch,
    case_index,
    target_rms,
):
    permitted = _owned_case_mask(
        ownership, decoder_adapter_tangent, case_index
    )
    outside = ~permitted
    outside_before = (
        float(decoder_adapter_tangent[outside].abs().max().detach())
        if bool(outside.any()) else 0.0
    )
    selected = decoder_adapter_tangent.masked_fill(~permitted, 0.0)
    # The fixed-radius contract is defined on the actual decoder-consistent
    # tangent after tapering.  Pre-taper RMS is reported separately so an
    # unexpectedly large taper compensation cannot be hidden by normalization.
    pre_taper_permitted = _owned_case_mask(
        ownership, pre_taper_adapter_tangent, case_index
    )
    pre_taper_rms = _masked_rms(
        pre_taper_adapter_tangent, pre_taper_permitted
    )
    tapered_model_rms = _masked_rms(selected, permitted)
    taper_values = c2_taper.expand_as(pre_taper_adapter_tangent)
    tapered_pre_decode = (
        pre_taper_adapter_tangent * taper_values
    ).masked_fill(~pre_taper_permitted, 0.0)
    tapered_pre_decode_rms = _masked_rms(
        tapered_pre_decode, pre_taper_permitted
    )
    action = group_probe._product_action(selected)
    candidate, scale, achieved = oracle.case_probe._candidate_from_action(
        baseline,
        action,
        batch["seam"],
        target_rms,
        case_index,
    )
    outside_after = (
        float(selected[outside].abs().max().detach())
        if bool(outside.any()) else 0.0
    )
    inflation = (
        float(tapered_model_rms.detach() / pre_taper_rms.detach())
        if float(pre_taper_rms.detach()) > 1.0e-12 else None
    )
    return candidate, {
        "outside_scope_abs_max_before_final_mask": outside_before,
        "outside_scope_masked_action_abs_max": outside_after,
        "scope_audit_reference": "fixed_teacher_bank_baseline_prediction",
        "scope_audit_space": "79d_product_tangent_action",
        "pre_taper_owned_rms": float(pre_taper_rms.detach()),
        "tapered_pre_decode_owned_rms": float(
            tapered_pre_decode_rms.detach()
        ),
        "decoder_consistent_owned_rms_before_radius_normalization": float(
            tapered_model_rms.detach()
        ),
        "decoder_to_pre_taper_rms_ratio": inflation,
        "fixed_radius_scale": float(scale),
        "fixed_radius_achieved_rms": float(achieved),
    }


def _audit_step(
    model,
    batch,
    cfg,
    decoder_adapter_tangent,
    pre_taper_adapter_tangent,
    c2_taper,
    ownership,
    baseline,
    baseline_identity,
    baseline_guard,
    contract,
    baseline_terms,
    samples,
    target_rms,
    workspace_floor,
    args,
    transaction_domains=None,
):
    rows = []
    for sample in samples:
        if sample["teacher_kind"] != "exact_projected_direction":
            continue
        global_case_index = int(sample["case_index"])
        case_index = global_case_index
        case_uid = str(sample.get("case_uid", case_index))
        group_name = str(sample["audit_group"])
        local_batch = batch
        local_decoder_tangent = decoder_adapter_tangent
        local_pre_taper_tangent = pre_taper_adapter_tangent
        local_c2_taper = c2_taper
        local_ownership = ownership
        local_baseline = baseline
        local_identity = baseline_identity
        local_baseline_guard = baseline_guard
        local_contract = contract
        if transaction_domains is not None:
            transaction_id = str(sample["transaction_id"])
            domain = transaction_domains[transaction_id]
            case_index = int(sample["local_case_index"])
            start, stop = domain["slice"]
            local_batch = domain["batch"]
            local_decoder_tangent = decoder_adapter_tangent[start:stop]
            local_pre_taper_tangent = pre_taper_adapter_tangent[start:stop]
            local_c2_taper = c2_taper[start:stop]
            local_ownership = ownership[start:stop]
            local_baseline = domain["baseline"]
            local_identity = domain["identity"]
            local_baseline_guard = domain["baseline_guard"]
            local_contract = domain["contract"]
        candidate, scope_diagnostics = _isolated_fixed_radius_candidate(
            local_baseline,
            local_decoder_tangent,
            local_pre_taper_tangent,
            local_c2_taper,
            local_ownership,
            local_batch,
            case_index,
            target_rms,
        )
        baseline_case = {
            key: float(baseline_terms[key][global_case_index].detach())
            for key in ("endpoint", "temporal")
        }
        raw_audit = oracle._exact_audit(
            model,
            local_batch,
            cfg,
            local_baseline,
            local_identity,
            local_baseline_guard,
            local_contract["initial_anchor"],
            local_contract["relative_tolerance"],
            local_contract["absolute_tolerance"],
            candidate,
            case_index,
            baseline_case,
            target_rms,
            workspace_floor,
        )
        scope_diagnostics[
            "outside_scope_abs_max_after_final_mask"
        ] = float(
            raw_audit["scope"][
                "outside_case_group_or_ownership_abs_max"
            ]
        )
        raw_audit.update(scope_diagnostics)
        projection = None
        if raw_audit["passed"]:
            _, projection = oracle._project_raw_candidate(
                model=model,
                batch=local_batch,
                cfg=cfg,
                baseline=local_baseline,
                identity=local_identity,
                baseline_guard=local_baseline_guard,
                guard_anchor=local_contract["initial_anchor"],
                guard_relative=local_contract["relative_tolerance"],
                guard_absolute=local_contract["absolute_tolerance"],
                raw=candidate,
                case_index=case_index,
                baseline_case=baseline_case,
                target_rms=target_rms,
                workspace_floor=workspace_floor,
                args=args,
            )
        rows.append({
            "case_index": global_case_index,
            "local_case_index": int(
                sample.get("local_case_index", case_index)
            ),
            "transaction_id": sample.get("transaction_id"),
            "case_uid": case_uid,
            "group": group_name,
            "raw_audit": raw_audit,
            "projector_stop_gradient": True,
            "projector_result": projection,
            "effective_projected_candidate": bool(
                projection
                and projection["effective_projected_candidate"]
            ),
        })
    return rows


def run(args):
    started = time.perf_counter()
    teacher_path = Path(args.teacher_bank)
    teacher = m.torch.load(
        teacher_path, map_location="cpu", weights_only=False
    )
    if teacher.get("schema") not in {
        "refiner_v15_15_observable_adapter_teacher_bank_v1",
        V15_15E_TEACHER_SCHEMA,
    }:
        raise RuntimeError("unsupported V15.15 teacher bank")
    if teacher.get("formal_training_allowed") is not False:
        raise RuntimeError("teacher bank is not development-only")
    multi_transaction_teacher = bool(
        teacher.get("schema") == V15_15E_TEACHER_SCHEMA
    )
    if multi_transaction_teacher and not teacher.get("teacher_bank_ready"):
        raise RuntimeError("multi-transaction teacher split lacks both groups")
    if multi_transaction_teacher and not args.case_isolated_guard_restoration:
        raise RuntimeError(
            "multi-transaction teachers require case-isolated Guard domains"
        )
    source = Path(teacher["source_diagnostic"])
    source_report = json.loads(
        (source / "diagnostic_report.json").read_text(encoding="utf-8-sig")
    )
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    cfg.product_refiner_observable_adapter = True
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.15 Adapter probe requires CUDA")
    batch = _to_device(teacher["batch"], device)
    samples = teacher["samples"]
    adapter_state_path = (
        Path(args.adapter_state) if args.adapter_state else None
    )
    if args.audit_only and adapter_state_path is None:
        raise RuntimeError("--audit-only requires --adapter-state")
    if args.audit_only and int(args.steps) != 0:
        raise RuntimeError("--audit-only requires --steps 0")
    model, missing = _load_source_model(
        source,
        cfg,
        device,
        adapter_state_path=adapter_state_path,
    )
    if not args.preserve_adapter_gate_floor:
        model.observable_adapter_gate_floor.copy_(m.torch.as_tensor(
            float(teacher["observable_adapter_gate_floor"]),
            dtype=model.observable_adapter_gate_floor.dtype,
            device=device,
        ))
    model.train(not args.audit_only)
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = m.torch.optim.AdamW(
        parameters, lr=float(args.learning_rate), weight_decay=1.0e-4
    )
    optimizer_state_resumed = False
    if args.resume_optimizer:
        if adapter_state_path is None:
            raise RuntimeError("--resume-optimizer requires --adapter-state")
        resume_payload = m.torch.load(
            adapter_state_path,
            map_location="cpu",
            weights_only=False,
        )
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict) or not optimizer_state:
            raise RuntimeError(
                "adapter state does not contain resumable optimizer state"
            )
        optimizer.load_state_dict(optimizer_state)
        optimizer_state_resumed = True
    with m.torch.no_grad():
        # Reuse the exact frozen V15.13 outputs captured by the teacher bank.
        # Re-running the newly Adapter-enabled model here would make the
        # anchor depend on probe architecture state and would also execute an
        # unnecessary clean forward pass.
        baseline = teacher["baseline_prediction"].to(device)
        baseline_identity = teacher["baseline_identity"].to(device)
        baseline_guard = (
            None
            if multi_transaction_teacher
            else projected_probe._float_guard(
                projected_probe._guard_values_for_prediction(
                    model, batch, cfg, baseline, baseline_identity
                )
            )
        )
        baseline_terms = oracle.case_probe._case_terms(
            baseline, batch, cfg
        )
    contract = source_report["group_guard_contract"]
    transaction_domains = None
    if multi_transaction_teacher:
        transaction_domains = {}
        schedules = teacher["transaction_schedules"]
        contracts = teacher["transaction_guard_contracts"]
        for transaction_id, metadata in schedules.items():
            start = int(metadata["global_case_offset"])
            stop = start + int(metadata["case_count"])
            local_batch = _slice_batch(batch, start, stop)
            local_baseline = baseline[start:stop]
            local_identity = baseline_identity[start:stop]
            transaction_domains[transaction_id] = {
                "slice": (start, stop),
                "batch": local_batch,
                "baseline": local_baseline,
                "identity": local_identity,
                "baseline_guard": projected_probe._float_guard(
                    projected_probe._guard_values_for_prediction(
                        model,
                        local_batch,
                        cfg,
                        local_baseline,
                        local_identity,
                    )
                ),
                "contract": contracts[transaction_id],
            }
    workspace_floor = max(1.0e-8, float(args.target_rms) * 0.01)
    history = []
    blocker_counts = Counter()
    final_audits = []

    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        _, adapter_trace = _adapter_batch_outputs(
            model, batch, cfg
        )
        adapter_tangent = adapter_trace["tangent"]
        ownership = adapter_trace["ownership"].to(m.torch.bool)
        decoder_adapter_tangent = adapter_trace[
            "decoder_consistent_tangent"
        ]
        permitted = ownership.expand_as(decoder_adapter_tangent)
        applied_adapter_tangent = decoder_adapter_tangent.masked_fill(
            ~permitted,
            0.0,
        )
        if args.exact_radius_training:
            training_tangent, radius_normalization = (
                _safe_exact_radius_tangent(
                    applied_adapter_tangent,
                    ownership,
                    samples,
                    float(args.target_rms),
                    eps=float(args.normalization_eps),
                )
            )
        else:
            training_tangent = applied_adapter_tangent
            radius_normalization = {}
        # Scientific losses and Guard penalties see the same fixed Anchor and
        # exact-scope tangent that the exact closure auditor receives. V15.15c
        # additionally evaluates every cross case on the required 1e-4 sphere.
        prediction = product_exp_torch(
            baseline.detach(), training_tangent
        )
        case_terms = oracle.case_probe._case_terms(prediction, batch, cfg)
        direction_losses = []
        magnitude_losses = []
        scientific_losses = []
        nonregression_losses = []
        direction_losses_by_group = {
            group: [] for group in CROSS_GROUPS
        }
        magnitude_losses_by_group = {
            group: [] for group in CROSS_GROUPS
        }
        scientific_losses_by_group = {
            group: [] for group in CROSS_GROUPS
        }
        nonregression_losses_by_group = {
            group: [] for group in CROSS_GROUPS
        }
        guard_losses_by_group = {
            group: [] for group in CROSS_GROUPS
        }
        stratified_weights_by_group = {
            group: [] for group in CROSS_GROUPS
        }
        case20_temporal_losses = []
        control_losses = []
        exact_radius_constraints = {}
        case_isolated_guard = {}
        direction_weight_by_case = {}
        amplitude = {}
        gate = {}
        for sample in samples:
            case_index = int(sample["case_index"])
            local_case_index = int(
                sample.get("local_case_index", case_index)
            )
            case_uid = str(sample.get("case_uid", case_index))
            group_name = str(sample["audit_group"])
            predicted, target = _case_vectors(
                applied_adapter_tangent,
                sample["teacher_tangent"],
                ownership,
                case_index,
            )
            raw_predicted, _ = _case_vectors(
                adapter_tangent,
                sample["teacher_tangent"],
                ownership,
                case_index,
            )
            predicted_rms = m.torch.sqrt(
                predicted.square().mean().clamp_min(1.0e-24)
            ) if predicted.numel() else adapter_tangent.sum() * 0.0
            target_rms = m.torch.sqrt(
                target.square().mean().clamp_min(1.0e-24)
            ) if target.numel() else adapter_tangent.sum() * 0.0
            raw_rms = m.torch.sqrt(
                raw_predicted.square().mean().clamp_min(1.0e-24)
            ) if raw_predicted.numel() else adapter_tangent.sum() * 0.0
            amplitude[case_uid] = {
                "raw_adapter_tangent_rms": float(raw_rms.detach()),
                "applied_adapter_tangent_rms": float(predicted_rms.detach()),
            }
            active = ownership[case_index, :, 0]
            gate[case_uid] = float(
                adapter_trace["gate"][case_index, active].mean().detach()
            ) if bool(active.any()) else 0.0
            if sample["teacher_kind"] == "identity_control":
                control_losses.append(predicted.square().mean())
                continue
            stratified_weight = float(
                sample.get("stratified_sampling_weight", 1.0)
            )
            stratified_weights_by_group[group_name].append(
                stratified_weight
            )
            cosine = m.torch.nn.functional.cosine_similarity(
                predicted.reshape(1, -1),
                target.reshape(1, -1),
                dim=-1,
                eps=1.0e-12,
            ).mean()
            magnitude_losses.append(
                m.torch.relu(predicted_rms - 1.25 * target_rms).square()
            )
            endpoint_scale = max(
                abs(float(baseline_terms["endpoint"][case_index])), 1.0e-6
            )
            temporal_scale = max(
                abs(float(baseline_terms["temporal"][case_index])), 1.0e-6
            )
            endpoint_anchor = baseline_terms["endpoint"][case_index].detach()
            temporal_anchor = baseline_terms["temporal"][case_index].detach()
            endpoint_tolerance = max(
                1.0e-12, abs(float(endpoint_anchor)) * 1.0e-9
            )
            temporal_tolerance = max(
                1.0e-12, abs(float(temporal_anchor)) * 1.0e-9
            )
            endpoint_delta = (
                case_terms["endpoint"][case_index] - endpoint_anchor
            )
            temporal_delta = (
                case_terms["temporal"][case_index] - temporal_anchor
            )
            endpoint_nonregression = m.torch.relu(
                endpoint_delta - endpoint_tolerance
            ) / endpoint_scale
            temporal_nonregression = m.torch.relu(
                temporal_delta - temporal_tolerance
            ) / temporal_scale
            nonregression_pressure = (
                endpoint_nonregression + temporal_nonregression
            )
            case_guard_pressure = nonregression_pressure * 0.0
            if args.case_isolated_guard_restoration:
                guard_batch = batch
                guard_baseline = baseline
                guard_identity = baseline_identity
                guard_tangent = training_tangent
                guard_ownership = ownership
                guard_case_index = case_index
                guard_contract = contract
                if transaction_domains is not None:
                    domain = transaction_domains[str(
                        sample["transaction_id"]
                    )]
                    start, stop = domain["slice"]
                    guard_batch = domain["batch"]
                    guard_baseline = domain["baseline"]
                    guard_identity = domain["identity"]
                    guard_tangent = training_tangent[start:stop]
                    guard_ownership = ownership[start:stop]
                    guard_case_index = local_case_index
                    guard_contract = domain["contract"]
                case_guard_pressure, guard_diagnostics = (
                    _case_isolated_fixed_guard_restoration(
                        model,
                        guard_batch,
                        cfg,
                        guard_baseline,
                        guard_identity,
                        guard_tangent,
                        guard_ownership,
                        guard_case_index,
                        guard_contract["initial_anchor"],
                        guard_contract["relative_tolerance"],
                        guard_contract["absolute_tolerance"],
                        float(args.guard_safety_fraction),
                    )
                )
                guard_losses_by_group[group_name].append(
                    case_guard_pressure
                )
                case_isolated_guard[case_uid] = guard_diagnostics
                direction_weight_tensor = _continuous_direction_weight(
                    case_guard_pressure,
                    nonregression_pressure,
                    float(args.guard_direction_floor),
                    float(args.guard_direction_decay),
                )
                direction_weight = float(direction_weight_tensor.detach())
                direction_losses_by_group[group_name].append(
                    direction_weight_tensor * (1.0 - cosine)
                )
                magnitude_losses_by_group[group_name].append(
                    m.torch.relu(
                        predicted_rms - 1.25 * target_rms
                    ).square()
                )
                scientific_losses_by_group[group_name].extend([
                    case_terms["endpoint"][case_index] / endpoint_scale,
                    case_terms["temporal"][case_index] / temporal_scale,
                ])
                nonregression_losses_by_group[group_name].extend([
                    endpoint_nonregression,
                    temporal_nonregression,
                ])
            else:
                nonregression_losses.extend([
                    endpoint_nonregression,
                    temporal_nonregression,
                ])
                violates_nonregression = bool(
                    float(nonregression_pressure.detach()) > 0.0
                )
                direction_weight = (
                    float(args.violating_direction_weight)
                    if violates_nonregression else 1.0
                )
                direction_losses.append(
                    direction_weight * (1.0 - cosine)
                )
                scientific_losses.extend([
                    case_terms["endpoint"][case_index] / endpoint_scale,
                    case_terms["temporal"][case_index] / temporal_scale,
                ])
            if local_case_index == 20:
                case20_temporal_losses.append(temporal_nonregression)
            direction_weight_by_case[case_uid] = direction_weight
            exact_radius_constraints[case_uid] = {
                "endpoint_anchor": float(endpoint_anchor),
                "endpoint_candidate": float(
                    case_terms["endpoint"][case_index].detach()
                ),
                "endpoint_delta": float(endpoint_delta.detach()),
                "endpoint_numeric_tolerance": endpoint_tolerance,
                "endpoint_nonregression": bool(
                    float(endpoint_delta.detach()) <= endpoint_tolerance
                ),
                "endpoint_nonregression_hinge": float(
                    endpoint_nonregression.detach()
                ),
                "temporal_anchor": float(temporal_anchor),
                "temporal_candidate": float(
                    case_terms["temporal"][case_index].detach()
                ),
                "temporal_delta": float(temporal_delta.detach()),
                "temporal_numeric_tolerance": temporal_tolerance,
                "temporal_nonregression": bool(
                    float(temporal_delta.detach()) <= temporal_tolerance
                ),
                "temporal_nonregression_hinge": float(
                    temporal_nonregression.detach()
                ),
                "direction_distillation_weight": direction_weight,
                "case_isolated_fixed_guard_pressure": float(
                    case_guard_pressure.detach()
                ),
            }
        group_balanced_components = {}
        if args.case_isolated_guard_restoration:
            if teacher.get("schema") == V15_15E_TEACHER_SCHEMA:
                direction_loss, group_balanced_components["direction"] = (
                    _stratified_cross_group_mean(
                        direction_losses_by_group,
                        stratified_weights_by_group,
                    )
                )
                magnitude_loss, group_balanced_components["magnitude"] = (
                    _stratified_cross_group_mean(
                        magnitude_losses_by_group,
                        stratified_weights_by_group,
                    )
                )
                scientific_loss, group_balanced_components["scientific"] = (
                    _stratified_cross_group_mean(
                        scientific_losses_by_group,
                        stratified_weights_by_group,
                    )
                )
                nonregression_loss, group_balanced_components[
                    "nonregression"
                ] = _stratified_cross_group_mean(
                    nonregression_losses_by_group,
                    stratified_weights_by_group,
                )
                guard_loss, group_balanced_components["fixed_guard"] = (
                    _stratified_cross_group_mean(
                        guard_losses_by_group,
                        stratified_weights_by_group,
                    )
                )
            else:
                direction_loss, group_balanced_components["direction"] = (
                    _balanced_cross_group_mean(direction_losses_by_group)
                )
                magnitude_loss, group_balanced_components["magnitude"] = (
                    _balanced_cross_group_mean(magnitude_losses_by_group)
                )
                scientific_loss, group_balanced_components["scientific"] = (
                    _balanced_cross_group_mean(scientific_losses_by_group)
                )
                nonregression_loss, group_balanced_components[
                    "nonregression"
                ] = _balanced_cross_group_mean(
                    nonregression_losses_by_group
                )
                guard_loss, group_balanced_components["fixed_guard"] = (
                    _balanced_cross_group_mean(guard_losses_by_group)
                )
        else:
            direction_loss = m.torch.stack(direction_losses).mean()
            magnitude_loss = m.torch.stack(magnitude_losses).mean()
            scientific_loss = m.torch.stack(scientific_losses).mean()
            nonregression_loss = m.torch.stack(
                nonregression_losses
            ).mean()
            guard_loss, _ = _fixed_guard_penalty(
                model,
                batch,
                cfg,
                prediction,
                baseline_identity,
                contract["initial_anchor"],
                contract["relative_tolerance"],
                contract["absolute_tolerance"],
            )
        case20_temporal_loss = (
            m.torch.stack(case20_temporal_losses).mean()
            if case20_temporal_losses
            else applied_adapter_tangent.sum() * 0.0
        )
        control_loss = (
            m.torch.stack(control_losses).mean()
            if control_losses else applied_adapter_tangent.sum() * 0.0
        )
        loss = (
            direction_loss
            + 0.10 * scientific_loss
            + float(args.nonregression_weight) * nonregression_loss
            + float(args.case20_temporal_weight) * case20_temporal_loss
            + 10.0 * magnitude_loss
            + 10.0 * control_loss
            + float(args.guard_restoration_weight) * guard_loss
        )
        audits = []
        if step % int(args.eval_every) == 0 or step == int(args.steps):
            audits = _audit_step(
                model,
                batch,
                cfg,
                applied_adapter_tangent.detach(),
                adapter_trace["owned_pre_taper_tangent"].detach(),
                adapter_trace["c2_taper"].detach(),
                ownership.detach(),
                baseline,
                baseline_identity,
                baseline_guard,
                contract,
                baseline_terms,
                samples,
                float(args.target_rms),
                workspace_floor,
                args,
                transaction_domains=transaction_domains,
            )
            final_audits = audits
            for row in audits:
                blocker_counts.update(
                    row["raw_audit"]["fixed_guard_blockers"]
                )
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "directional_cosine_loss": float(direction_loss.detach()),
            "scientific_loss": float(scientific_loss.detach()),
            "exact_radius_nonregression_loss": float(
                nonregression_loss.detach()
            ),
            "case20_temporal_signed_residual_loss": float(
                case20_temporal_loss.detach()
            ),
            "magnitude_upper_bound_loss": float(magnitude_loss.detach()),
            "identity_control_loss": float(control_loss.detach()),
            "differentiable_fixed_guard_excess": float(guard_loss.detach()),
            "case_isolated_fixed_guard_restoration": bool(
                args.case_isolated_guard_restoration
            ),
            "case_isolated_fixed_guard_by_case": case_isolated_guard,
            "continuous_direction_weight_by_case": (
                direction_weight_by_case
            ),
            "cross_group_balanced_loss_components": (
                group_balanced_components
            ),
            "adapter_output_rms_by_case": amplitude,
            "adapter_gate_mean_by_case": gate,
            "exact_radius_normalization_by_case": radius_normalization,
            "exact_radius_constraints_by_case": exact_radius_constraints,
            "exact_audits": audits,
        }
        history.append(row)
        print(json.dumps({
            "stage": "v15_15_adapter_probe_step",
            "step": step,
            "loss": row["loss"],
            "directional_cosine_loss": row["directional_cosine_loss"],
            "exact_radius_nonregression_loss": row[
                "exact_radius_nonregression_loss"
            ],
            "case20_temporal_signed_residual_loss": row[
                "case20_temporal_signed_residual_loss"
            ],
            "identity_control_loss": row["identity_control_loss"],
            "differentiable_fixed_guard_excess": row[
                "differentiable_fixed_guard_excess"
            ],
            "minimum_direction_distillation_weight": min(
                row["continuous_direction_weight_by_case"].values(),
                default=1.0,
            ),
            "projected_passes": sum(
                audit["effective_projected_candidate"] for audit in audits
            ),
        }), flush=True)
        if step == int(args.steps):
            break
        loss.backward()
        row["gradient_norm_before_clip"] = float(
            m.torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=float(args.gradient_clip)
            )
        )
        optimizer.step()

    projected_by_group = Counter(
        row["group"]
        for row in final_audits
        if row["effective_projected_candidate"]
    )
    control_indices = [
        str(sample.get("case_uid", sample["case_index"]))
        for sample in samples
        if sample["teacher_kind"] == "identity_control"
    ]
    final_amplitude = history[-1]["adapter_output_rms_by_case"]
    final_gate = history[-1]["adapter_gate_mean_by_case"]
    single_control_gate_max = max(
        (float(final_gate[index]) for index in control_indices),
        default=0.0,
    )
    single_control_applied_rms_max = max(
        (
            float(final_amplitude[index]["applied_adapter_tangent_rms"])
            for index in control_indices
        ),
        default=0.0,
    )
    single_conservative_path_preserved = bool(
        single_control_gate_max == 0.0
        and single_control_applied_rms_max <= 1.0e-7
    )
    route_ready = bool(
        projected_by_group["cross_short"] > 0
        and projected_by_group["cross_long"] > 0
        and single_conservative_path_preserved
    )
    final_scope_safe = bool(
        final_audits
        and all(
            row["raw_audit"]["scope"]["scope_safe"]
            for row in final_audits
        )
    )
    outside_after_max = max(
        (
            float(row["raw_audit"].get(
                "outside_scope_abs_max_after_final_mask", 0.0
            ))
            for row in final_audits
        ),
        default=0.0,
    )
    route_ready = bool(route_ready and final_scope_safe)
    def readiness_status(audit):
        raw = audit["raw_audit"]
        scientific = raw["case_scientific"]
        tolerances = scientific["numeric_tolerance"]
        return {
            "present": True,
            "scope_safe": bool(raw["scope"]["scope_safe"]),
            "endpoint_nonregression": bool(
                float(scientific["endpoint_delta"])
                <= float(tolerances["endpoint"])
            ),
            "temporal_nonregression": bool(
                float(scientific["temporal_delta"])
                <= float(tolerances["temporal"])
            ),
            "fixed_guard_passed": bool(raw["fixed_guard_passed"]),
            "fixed_guard_blockers": list(
                raw.get("fixed_guard_blockers", [])
            ),
            "raw_passed": bool(raw["passed"]),
            "projector_invoked": audit["projector_result"] is not None,
            "effective_projected_candidate": bool(
                audit["effective_projected_candidate"]
            ),
        }

    audit_by_case = {
        int(row["case_index"]): row for row in final_audits
    }
    case_uid_by_index = {
        int(sample["case_index"]): str(
            sample.get("case_uid", sample["case_index"])
        )
        for sample in samples
        if sample["teacher_kind"] == "exact_projected_direction"
    }
    required_case_status = {}
    required_cases = (
        tuple(
            int(sample["case_index"])
            for sample in samples
            if sample["teacher_kind"] == "exact_projected_direction"
        )
        if multi_transaction_teacher else FORMAL_READINESS_CASES
    )
    for case_index in required_cases:
        audit = audit_by_case.get(case_index)
        if audit is None:
            missing_key = (
                case_uid_by_index.get(case_index, str(case_index))
                if multi_transaction_teacher else str(case_index)
            )
            required_case_status[missing_key] = {
                "present": False,
                "scope_safe": False,
                "endpoint_nonregression": False,
                "temporal_nonregression": False,
                "fixed_guard_passed": False,
                "fixed_guard_blockers": ["missing_required_case"],
                "raw_passed": False,
                "projector_invoked": False,
                "effective_projected_candidate": False,
            }
            continue
        key = (
            str(audit.get("case_uid", case_index))
            if multi_transaction_teacher else str(case_index)
        )
        required_case_status[key] = readiness_status(audit)
    formal_readiness_criteria = {
        "required_cases_present": all(
            row["present"] for row in required_case_status.values()
        ),
        "required_cases_scope_safe": all(
            row["scope_safe"] for row in required_case_status.values()
        ),
        "required_cases_endpoint_nonregression": all(
            row["endpoint_nonregression"]
            for row in required_case_status.values()
        ),
        "required_cases_temporal_nonregression": all(
            row["temporal_nonregression"]
            for row in required_case_status.values()
        ),
        "required_cases_fixed_guard_passed": all(
            row["fixed_guard_passed"]
            for row in required_case_status.values()
        ),
        "required_cases_raw_passed": all(
            row["raw_passed"] for row in required_case_status.values()
        ),
        "required_cases_projector_invoked": all(
            row["projector_invoked"]
            for row in required_case_status.values()
        ),
        "cross_short_projected_candidate_exists": bool(
            projected_by_group["cross_short"] > 0
        ),
        "cross_long_projected_candidate_exists": bool(
            projected_by_group["cross_long"] > 0
        ),
        "single_control_gate_is_zero": bool(
            single_control_gate_max == 0.0
        ),
        "single_control_tangent_is_conservative": bool(
            single_control_applied_rms_max <= 1.0e-7
        ),
        "numeric_audit_complete": bool(final_audits),
        "fixed_guard_thresholds_unchanged": True,
        "train_validation_case_disjoint": not bool(
            teacher.get("train_validation_case_overlap", [])
        ),
        "train_validation_source_case_disjoint": not bool(
            teacher.get("train_validation_source_case_overlap", [])
        ),
    }
    if not multi_transaction_teacher:
        formal_readiness_criteria.update({
            "case_16_penetration_passed": bool(
                required_case_status["16"]["present"]
                and "cross_short.penetration" not in required_case_status[
                    "16"
                ]["fixed_guard_blockers"]
            ),
            "case_29_support_drift_max_passed": bool(
                required_case_status["29"]["present"]
                and "cross_long.support_drift_max"
                not in required_case_status["29"]["fixed_guard_blockers"]
            ),
        })
    ready_for_formal_adapter_training = bool(
        args.exact_radius_training
        and all(formal_readiness_criteria.values())
    )
    if args.exact_radius_training:
        route_ready = ready_for_formal_adapter_training
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "observable_adapter_probe_state.pt"
    schema = (
        V15_15D_SCHEMA
        if args.case_isolated_guard_restoration
        else V15_15C_SCHEMA
    )
    m.torch.save({
        "schema": schema,
        "formal_checkpoint": False,
        "formal_training_allowed": False,
        "adapter_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if key.startswith("observable_adapter_")
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "gate_floor": float(model.observable_adapter_gate_floor),
    }, state_path)
    report = {
        "schema": schema,
        "development_only": True,
        "formal_checkpoint": False,
        "formal_training_allowed": False,
        "publish_allowed": False,
        "audit_only": bool(args.audit_only),
        "exact_radius_training": bool(args.exact_radius_training),
        "exact_radius_normalization_eps": float(args.normalization_eps),
        "nonregression_weight": float(args.nonregression_weight),
        "case20_temporal_weight": float(args.case20_temporal_weight),
        "violating_direction_weight": float(
            args.violating_direction_weight
        ),
        "case_isolated_fixed_guard_restoration": bool(
            args.case_isolated_guard_restoration
        ),
        "fixed_guard_training_aggregation": (
            "sum_normalized_relu"
            if args.case_isolated_guard_restoration else "mean_squared_relu"
        ),
        "fixed_guard_training_scale": (
            "fixed_anchor_absolute_allowance"
            if args.case_isolated_guard_restoration
            else "metric_absolute_magnitude"
        ),
        "fixed_guard_activation_checkpointing": bool(
            args.case_isolated_guard_restoration
        ),
        "fixed_guard_checkpoint_reentrant": False,
        "guard_safety_fraction": float(args.guard_safety_fraction),
        "guard_restoration_weight": float(
            args.guard_restoration_weight
        ),
        "guard_direction_floor": float(args.guard_direction_floor),
        "guard_direction_decay": float(args.guard_direction_decay),
        "cross_group_balanced_training": bool(
            args.case_isolated_guard_restoration
        ),
        "resumed_adapter_state": (
            str(adapter_state_path.resolve())
            if adapter_state_path is not None else None
        ),
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "teacher_bank": str(teacher_path.resolve()),
        "teacher_bank_schema": teacher.get("schema"),
        "teacher_bank_split": teacher.get("split"),
        "teacher_sampling_weight_protocol": teacher.get(
            "sampling_weight_protocol"
        ),
        "source_diagnostic": str(source.resolve()),
        "teacher_sample_count": len(samples),
        "projected_teacher_count_by_group": teacher.get(
            "projected_teacher_count_by_group", {}
        ),
        "identity_control_count": sum(
            sample["teacher_kind"] == "identity_control"
            for sample in samples
        ),
        "steps": int(args.steps),
        "role_label_consumed_at_inference": False,
        "hidden_clean_consumed_by_adapter": False,
        "hidden_clean_used_for_fixed_guard_audit_only": True,
        "continuous_soft_gate": True,
        "single_control_deadzone_floor": float(
            model.observable_adapter_gate_floor
        ),
        "adapter_zero_initialized": adapter_state_path is None,
        "projector_gradient_protocol": "stop_gradient_initial_probe",
        "gradient_clip": float(args.gradient_clip),
        "optimizer_state_resumed": optimizer_state_resumed,
        "adapter_gate_floor_source": (
            "resumed_training_state"
            if args.preserve_adapter_gate_floor
            else "current_teacher_bank_calibration"
        ),
        "source_checkpoint_missing_adapter_keys": missing,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "fixed_guard_blocker_counts": dict(blocker_counts),
        "single_control_gate_max": single_control_gate_max,
        "single_control_applied_tangent_rms_max": (
            single_control_applied_rms_max
        ),
        "single_conservative_path_preserved": (
            single_conservative_path_preserved
        ),
        "adapter_gate_mean_by_case": final_gate,
        "outside_scope_abs_max_after_final_mask": outside_after_max,
        "scope_safe": final_scope_safe,
        "numeric_audit_complete": bool(final_audits),
        "effective_projected_candidate_count_by_group": dict(
            projected_by_group
        ),
        "ready_for_expanded_adapter_training": route_ready,
        "required_case_status": required_case_status,
        "formal_readiness_criteria": formal_readiness_criteria,
        "ready_for_formal_adapter_training": (
            ready_for_formal_adapter_training
        ),
        "probe_state": str(state_path.resolve()),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = destination / "observable_adapter_probe.report.json"
    m.save_json(report, report_path)
    print(json.dumps({
        "stage": "v15_15_observable_adapter_probe_complete",
        "report": str(report_path.resolve()),
        "ready_for_expanded_adapter_training": route_ready,
        "ready_for_formal_adapter_training": (
            ready_for_formal_adapter_training
        ),
        "effective_projected_candidate_count_by_group": dict(
            projected_by_group
        ),
    }), flush=True)
    return 0 if route_ready else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--target-rms", type=float, default=1.0e-4)
    parser.add_argument("--adapter-state")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument(
        "--preserve-adapter-gate-floor", action="store_true"
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--exact-radius-training", action="store_true")
    parser.add_argument(
        "--normalization-eps",
        type=float,
        default=EXACT_RADIUS_NORMALIZATION_EPS,
    )
    parser.add_argument("--nonregression-weight", type=float, default=25.0)
    parser.add_argument("--case20-temporal-weight", type=float, default=25.0)
    parser.add_argument(
        "--violating-direction-weight", type=float, default=0.10
    )
    parser.add_argument(
        "--case-isolated-guard-restoration", action="store_true"
    )
    parser.add_argument(
        "--guard-safety-fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--guard-restoration-weight", type=float, default=10.0
    )
    parser.add_argument(
        "--guard-direction-floor", type=float, default=0.10
    )
    parser.add_argument(
        "--guard-direction-decay", type=float, default=1.0
    )
    parser.add_argument("--ik-iterations", type=int, default=6)
    parser.add_argument("--damping", type=float, default=1.0e-4)
    parser.add_argument("--jacobian-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--stiffness-ceiling", type=float, default=1.0e4)
    parser.add_argument(
        "--acceleration-regularization", type=float, default=1.0e-2
    )
    parser.add_argument("--jerk-regularization", type=float, default=1.0e-3)
    args = parser.parse_args()
    if args.target_rms != 1.0e-4:
        parser.error("--target-rms must remain exactly 1e-4")
    if args.eval_every < 1:
        parser.error("evaluation interval must be positive")
    if not 0.0 < args.normalization_eps <= 1.0e-6:
        parser.error("normalization epsilon must be in (0, 1e-6]")
    if args.nonregression_weight <= 0.0:
        parser.error("nonregression weight must be positive")
    if args.case20_temporal_weight <= 0.0:
        parser.error("case-20 temporal weight must be positive")
    if not 0.0 <= args.violating_direction_weight <= 1.0:
        parser.error("violating direction weight must be in [0, 1]")
    if not 0.0 <= args.guard_safety_fraction < 1.0:
        parser.error("Guard safety fraction must be in [0, 1)")
    if args.guard_restoration_weight <= 0.0:
        parser.error("Guard restoration weight must be positive")
    if not 0.0 <= args.guard_direction_floor <= 1.0:
        parser.error("Guard direction floor must be in [0, 1]")
    if args.guard_direction_decay <= 0.0:
        parser.error("Guard direction decay must be positive")
    if args.case_isolated_guard_restoration and not args.exact_radius_training:
        parser.error(
            "case-isolated Guard restoration requires exact-radius training"
        )
    if args.audit_only:
        if args.steps != 0:
            parser.error("--audit-only requires --steps 0")
    elif args.steps < 1:
        parser.error("probe steps must be positive")
    if args.resume_optimizer and args.audit_only:
        parser.error("audit-only runs cannot resume the optimizer")
    if args.preserve_adapter_gate_floor and not args.adapter_state:
        parser.error(
            "--preserve-adapter-gate-floor requires --adapter-state"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
