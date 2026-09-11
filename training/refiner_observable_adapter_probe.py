"""V15.15 short observable-conditioned Adapter distillation probe.

The probe freezes the V15.13 shared Refiner, fits only the zero-initialized
75D Adapter against frozen V15.14h projected directions, and performs exact
fixed-Guard audits. It cannot publish a formal checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from motion_geometry.product_manifold import product_exp_torch, product_log_torch
from training import motion_models as m
from training import refiner_case_local_full_tangent_oracle as oracle
from training import refiner_group_local_nullspace_cone_probe as group_probe
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_15b_exact_scope_observable_cross_adapter_probe_v2"


def _to_device(value, device):
    if m.torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


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


def _owned_case_mask(ownership, tangent, case_index):
    selected = m.torch.zeros_like(ownership, dtype=m.torch.bool)
    selected[int(case_index)] = True
    return (selected & ownership).expand_as(tangent)


def _masked_rms(value, mask):
    selected = value[mask]
    if selected.numel() == 0:
        return value.sum() * 0.0
    return m.torch.sqrt(selected.square().mean().clamp_min(1.0e-24))


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
):
    rows = []
    for sample in samples:
        if sample["teacher_kind"] != "exact_projected_direction":
            continue
        case_index = int(sample["case_index"])
        group_name = str(sample["audit_group"])
        candidate, scope_diagnostics = _isolated_fixed_radius_candidate(
            baseline,
            decoder_adapter_tangent,
            pre_taper_adapter_tangent,
            c2_taper,
            ownership,
            batch,
            case_index,
            target_rms,
        )
        baseline_case = {
            key: float(baseline_terms[key][case_index].detach())
            for key in ("endpoint", "temporal")
        }
        raw_audit = oracle._exact_audit(
            model,
            batch,
            cfg,
            baseline,
            baseline_identity,
            baseline_guard,
            contract["initial_anchor"],
            contract["relative_tolerance"],
            contract["absolute_tolerance"],
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
                batch=batch,
                cfg=cfg,
                baseline=baseline,
                identity=baseline_identity,
                baseline_guard=baseline_guard,
                guard_anchor=contract["initial_anchor"],
                guard_relative=contract["relative_tolerance"],
                guard_absolute=contract["absolute_tolerance"],
                raw=candidate,
                case_index=case_index,
                baseline_case=baseline_case,
                target_rms=target_rms,
                workspace_floor=workspace_floor,
                args=args,
            )
        rows.append({
            "case_index": case_index,
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
    if teacher.get("schema") != (
        "refiner_v15_15_observable_adapter_teacher_bank_v1"
    ):
        raise RuntimeError("unsupported V15.15 teacher bank")
    if teacher.get("formal_training_allowed") is not False:
        raise RuntimeError("teacher bank is not development-only")
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
    with m.torch.no_grad():
        # Reuse the exact frozen V15.13 outputs captured by the teacher bank.
        # Re-running the newly Adapter-enabled model here would make the
        # anchor depend on probe architecture state and would also execute an
        # unnecessary clean forward pass.
        baseline = teacher["baseline_prediction"].to(device)
        baseline_identity = teacher["baseline_identity"].to(device)
        baseline_guard = projected_probe._float_guard(
            projected_probe._guard_values_for_prediction(
                model, batch, cfg, baseline, baseline_identity
            )
        )
        baseline_terms = oracle.case_probe._case_terms(
            baseline, batch, cfg
        )
    contract = source_report["group_guard_contract"]
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
        # Scientific losses and Guard penalties see the same fixed Anchor and
        # exact-scope tangent that the exact closure auditor receives.
        prediction = product_exp_torch(
            baseline.detach(), applied_adapter_tangent
        )
        case_terms = oracle.case_probe._case_terms(prediction, batch, cfg)
        direction_losses = []
        magnitude_losses = []
        scientific_losses = []
        control_losses = []
        amplitude = {}
        gate = {}
        for sample in samples:
            case_index = int(sample["case_index"])
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
            amplitude[str(case_index)] = {
                "raw_adapter_tangent_rms": float(raw_rms.detach()),
                "applied_adapter_tangent_rms": float(predicted_rms.detach()),
            }
            active = ownership[case_index, :, 0]
            gate[str(case_index)] = float(
                adapter_trace["gate"][case_index, active].mean().detach()
            ) if bool(active.any()) else 0.0
            if sample["teacher_kind"] == "identity_control":
                control_losses.append(predicted.square().mean())
                continue
            cosine = m.torch.nn.functional.cosine_similarity(
                predicted.reshape(1, -1),
                target.reshape(1, -1),
                dim=-1,
                eps=1.0e-12,
            ).mean()
            direction_losses.append(1.0 - cosine)
            magnitude_losses.append(
                m.torch.relu(predicted_rms - 1.25 * target_rms).square()
            )
            endpoint_scale = max(
                abs(float(baseline_terms["endpoint"][case_index])), 1.0e-6
            )
            temporal_scale = max(
                abs(float(baseline_terms["temporal"][case_index])), 1.0e-6
            )
            scientific_losses.extend([
                case_terms["endpoint"][case_index] / endpoint_scale,
                case_terms["temporal"][case_index] / temporal_scale,
            ])
        direction_loss = m.torch.stack(direction_losses).mean()
        magnitude_loss = m.torch.stack(magnitude_losses).mean()
        scientific_loss = m.torch.stack(scientific_losses).mean()
        control_loss = (
            m.torch.stack(control_losses).mean()
            if control_losses else applied_adapter_tangent.sum() * 0.0
        )
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
        loss = (
            direction_loss
            + 0.10 * scientific_loss
            + 10.0 * magnitude_loss
            + 10.0 * control_loss
            + 10.0 * guard_loss
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
            "magnitude_upper_bound_loss": float(magnitude_loss.detach()),
            "identity_control_loss": float(control_loss.detach()),
            "differentiable_fixed_guard_excess": float(guard_loss.detach()),
            "adapter_output_rms_by_case": amplitude,
            "adapter_gate_mean_by_case": gate,
            "exact_audits": audits,
        }
        history.append(row)
        print(json.dumps({
            "stage": "v15_15_adapter_probe_step",
            "step": step,
            "loss": row["loss"],
            "directional_cosine_loss": row["directional_cosine_loss"],
            "identity_control_loss": row["identity_control_loss"],
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
        str(sample["case_index"])
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
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "observable_adapter_probe_state.pt"
    m.torch.save({
        "schema": SCHEMA,
        "formal_checkpoint": False,
        "formal_training_allowed": False,
        "adapter_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if key.startswith("observable_adapter_")
        },
        "gate_floor": float(model.observable_adapter_gate_floor),
    }, state_path)
    report = {
        "schema": SCHEMA,
        "development_only": True,
        "formal_checkpoint": False,
        "formal_training_allowed": False,
        "publish_allowed": False,
        "audit_only": bool(args.audit_only),
        "resumed_adapter_state": (
            str(adapter_state_path.resolve())
            if adapter_state_path is not None else None
        ),
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "teacher_bank": str(teacher_path.resolve()),
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
    parser.add_argument("--audit-only", action="store_true")
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
    if args.audit_only:
        if args.steps != 0:
            parser.error("--audit-only requires --steps 0")
    elif args.steps < 1:
        parser.error("probe steps must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
