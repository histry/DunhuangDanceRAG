"""Inference-only Adapter + second-order repair composite transaction.

The module consumes observables only.  It has no split, single/cross, teacher,
or case-ID input and implements one fail-closed transaction over the complete
motion: conformal activation, Adapter incumbent, bounded g1f3 repair,
Projector, authoritative Guard, then atomic commit or identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from contracts.physical_quality import evaluate_stage_candidate
from motion_geometry.product_manifold import product_exp_torch, product_log_torch
from training import motion_models as m
from training import refiner_observable_adapter_probe as adapter
from training import refiner_v15_15g_fixed_budget_correction as g1f
from training import refiner_v15_15g1f3_second_order as second_order


MODEL_SCHEMA = "v15_15h_adapter_second_order_repair_composite_v1"
CONTRACT_SCHEMA = "v15_15h_adapter_second_order_repair_composite_contract_v1"
_CACHE = {}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_composite(model_path, contract_path, cfg):
    key = (str(Path(model_path).resolve()), str(Path(contract_path).resolve()))
    if key in _CACHE:
        return _CACHE[key]
    contract = json.loads(
        Path(contract_path).read_text(encoding="utf-8-sig")
    )
    _require(contract.get("schema") == CONTRACT_SCHEMA,
             "V15.15h composite contract schema mismatch")
    expected_commit = os.environ.get("EXPECTED_COMMIT")
    if expected_commit:
        _require(contract.get("implementation_commit") == expected_commit,
                 "V15.15h implementation commit mismatch")
    _require(contract.get("model", {}).get("sha256") == _sha256(model_path),
             "V15.15h composite model hash mismatch")
    fixed = contract.get("fixed_runtime_contract") or {}
    _require(fixed.get("correction_budgets") == [2, 3, 5],
             "V15.15h correction budgets changed")
    _require(len(fixed.get("angular_line_search_radians") or []) == 12,
             "V15.15h angular ladder changed")
    _require(float(fixed.get("target_rms", 0.0)) == 1.0e-4,
             "V15.15h radius changed")
    _require(fixed.get("curvature_dtype") == "float64",
             "V15.15h curvature dtype changed")
    _require(fixed.get("geodesic_acceleration_included") is True,
             "V15.15h geodesic acceleration is absent")
    _require(fixed.get("second_order_model_builds_per_iteration") == 1,
             "V15.15h curvature model is rebuilt per angle")
    _require(fixed.get("second_order_grid_execution_device") ==
             "same_cuda_device_as_motion",
             "V15.15h candidate grid is not device-resident")
    _require(fixed.get("second_order_joint_subproblem_solver") ==
             "deterministic_device_resident_riemannian_continuous_sqp",
             "V15.15h continuous joint SQP is absent")
    _require(fixed.get("second_order_sqp_refinement_starts") ==
             second_order.SECOND_ORDER_SQP_REFINEMENT_STARTS,
             "V15.15h SQP start count changed")
    _require(fixed.get("second_order_sqp_refinement_iterations") ==
             second_order.SECOND_ORDER_SQP_REFINEMENT_ITERATIONS,
             "V15.15h SQP iteration count changed")
    _require(fixed.get("second_order_sqp_smoothing") == [
        float(value) for value in second_order.SECOND_ORDER_SQP_SMOOTHING
    ], "V15.15h SQP smoothing schedule changed")
    _require(fixed.get("second_order_sqp_constraint_scaling") ==
             "absolute_signed_boundary_gap_floor_1e-12",
             "V15.15h SQP constraint scaling changed")
    _require(fixed.get("second_order_budget_semantics") ==
             "remaining_joint_closure_gap_divided_by_remaining_steps",
             "V15.15h second-order budget is not multi-step")
    _require(fixed.get("second_order_intermediate_acceptance") ==
             "authoritative_remaining_gap_share_or_joint_gap_filter_progress",
             "V15.15h intermediate acceptance changed")
    _require(fixed.get("second_order_infeasible_joint_policy") ==
             "deterministic_full_remaining_gap_minimum_normalized_"
             "residual_restoration",
             "V15.15h infeasible-joint policy changed")
    _require(fixed.get("second_order_restoration_target") ==
             "complete_remaining_closure_gap",
             "V15.15h restoration is not terminal-gap directed")
    _require(fixed.get("second_order_restoration_acceptance") ==
             "authoritative_safe_boundary_and_positive_gap_merit_decrease",
             "V15.15h restoration acceptance changed")
    _require(fixed.get(
        "second_order_restoration_final_step_allowed"
    ) is False, "V15.15h restoration may replace final closure")
    _require(fixed.get("second_order_zero_start_seed_policy") ==
             "observable_ungated_frozen_adapter_decoder_direction",
             "V15.15h zero-start seed policy changed")
    _require(fixed.get(
        "second_order_zero_start_seed_teacher_or_label_consumed"
    ) is False, "V15.15h zero-start seed consumed offline evidence")
    _require(fixed.get("second_order_sqp_line_search_radians") == [
        float(value)
        for value in second_order.SECOND_ORDER_SQP_LINE_SEARCH_RADIANS
    ], "V15.15h SQP line search changed")
    _require(fixed.get("finite_gap_required_reduction_formula") ==
             "current_delta+strict_limit+safety_margin",
             "V15.15h finite-gap formula changed")
    _require(fixed.get(
        "finite_gap_already_safe_term_may_use_safe_slack"
    ) is True, "V15.15h safe science slack is unavailable")
    _require(fixed.get("second_order_host_candidate_sorting") is False,
             "V15.15h host candidate sorting is forbidden")
    _require(fixed.get("second_order_nonfinite_basis_policy") ==
             "deterministic_verified_subspace_reduction",
             "V15.15h nonfinite-basis policy changed")
    _require(fixed.get("second_order_unverified_directions_used") is False,
             "V15.15h unverified curvature directions are forbidden")
    recovery = fixed.get("second_order_hvp_recovery") or {}
    _require(recovery.get("trigger") ==
             "nonfinite_autograd_second_derivative_only",
             "V15.15h HvP recovery trigger changed")
    _require(recovery.get("method") ==
             "symmetric_first_derivative_hvp_epsilon_ladder",
             "V15.15h HvP recovery method changed")
    _require(recovery.get("epsilon_radians") == [
        float(value) for value in second_order.HVP_RECOVERY_EPSILON_RADIANS
    ], "V15.15h HvP recovery epsilon ladder changed")
    _require(float(recovery.get("relative_tolerance", -1.0)) ==
             float(second_order.HVP_RECOVERY_RELATIVE_TOLERANCE),
             "V15.15h HvP recovery relative tolerance changed")
    _require(float(recovery.get("absolute_tolerance", -1.0)) ==
             float(second_order.HVP_RECOVERY_ABSOLUTE_TOLERANCE),
             "V15.15h HvP recovery absolute tolerance changed")
    _require(recovery.get("requires_consistent_estimates") is True,
             "V15.15h HvP recovery verification is absent")
    _require(fixed.get(
        "second_order_prediction_active_guard_terms_frozen_across_"
        "curvature_evaluations"
    ) is True, "V15.15h prediction active Guard terms are not frozen")
    _require(fixed.get("atomic_commit_or_identity") is True,
             "V15.15h atomic policy is absent")
    _require(fixed.get("runtime_case_labels_consumed") is False,
             "V15.15h runtime labels are forbidden")
    _require(float(fixed.get("minimum_repair_gain", 0.0)) == 0.03,
             "V15.15h 0.03 gate changed")

    payload = m.torch.load(model_path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == MODEL_SCHEMA,
             "V15.15h composite model schema mismatch")
    _require(payload.get("implementation_commit") ==
             contract.get("implementation_commit"),
             "V15.15h model/contract commit mismatch")
    source_hashes = payload.get("source_hashes") or {}
    for contract_key, source_key in (
        ("base_refiner_checkpoint", "base_refiner_checkpoint_sha256"),
        ("adapter_state", "adapter_state_sha256"),
        ("conformal_envelope", "conformal_envelope_sha256"),
        ("g1f3_frozen_contract", "g1f3_frozen_contract_sha256"),
    ):
        _require(
            contract.get(contract_key, {}).get("sha256")
            == source_hashes.get(source_key),
            f"V15.15h source hash mismatch: {contract_key}",
        )
    runtime = payload.get("runtime_refiner") or {}
    gate_mode = runtime.get("observable_adapter_gate_mode")
    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
        observable_adapter=True,
        observable_adapter_learned_gate_residual=(
            gate_mode == adapter.LEARNED_GATE_MODE
        ),
        residual_taper_frames=int(cfg.product_refiner_residual_taper_frames),
    ).to(m.torch.device(cfg.device))
    incompatible = model.load_state_dict(
        runtime["base_model_state_dict"], strict=False
    )
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith("observable_adapter_")
    ]
    _require(not invalid_missing and not incompatible.unexpected_keys,
             "base Refiner state is incompatible with composite runtime")
    state = model.state_dict()
    for name, value in runtime["adapter_state_dict"].items():
        _require(name in state and tuple(state[name].shape) == tuple(value.shape),
                 f"Adapter runtime parameter mismatch: {name}")
        state[name] = value
    model.load_state_dict(state, strict=True)
    gate_floor = runtime.get("observable_adapter_gate_floor")
    if gate_floor is not None:
        model.observable_adapter_gate_floor.copy_(m.torch.as_tensor(
            float(gate_floor),
            dtype=model.observable_adapter_gate_floor.dtype,
            device=model.observable_adapter_gate_floor.device,
        ))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loaded = (payload, contract, model)
    _CACHE[key] = loaded
    return loaded


def _batch(motion, cond, seam, cfg, device):
    bad = m.torch.as_tensor(
        np.asarray(motion)[None].copy(), dtype=m.torch.float32, device=device
    )
    condition = m.torch.as_tensor(
        np.asarray(cond)[None].copy(), dtype=m.torch.float32, device=device
    )
    seam_tensor = m.torch.as_tensor(
        np.asarray(seam)[None].copy(), dtype=m.torch.float32, device=device
    )
    if seam_tensor.ndim == 2:
        seam_tensor = seam_tensor.unsqueeze(-1)
    joint, root, contact = m._risk_masks_for_batch_torch(
        bad, seam_tensor, cfg
    )
    return {
        "bad": bad,
        "cond": condition,
        "seam": seam_tensor,
        "joint": joint,
        "root": root,
        "contact": contact,
    }


def _severity(model_input, cfg):
    features, _, _, ownership = m._refiner_observable_adapter_features(
        model_input["bad"],
        model_input["seam"],
        float(cfg.fps),
        int(cfg.product_refiner_residual_taper_frames),
    )
    sample = {
        "observable_condition": features[0].detach().cpu(),
        "ownership": ownership[0].detach().cpu(),
    }
    return sample, ownership


def _science_terms(candidate, baseline, seam, cfg):
    _, terms = m._observable_refiner_objective(
        candidate, baseline.detach(), seam, cfg, reduction="none"
    )
    return {
        "endpoint": terms["endpoint_scientific_deficit"][0],
        "temporal": terms["temporal_scientific_deficit"][0],
        "guard_terms": {
            name: terms[key][0] for name, key in g1f.G1_GUARD_PROXY_TERMS.items()
        },
    }


def _project_and_guard(candidate, snapshot, cfg, audit_fn, limits, policy):
    projected, projector = m.enforce_edge151_contract_np(
        np.asarray(candidate, dtype=np.float32),
        cfg,
        source_hint="v15_15h_composite_projector",
        derive_contact=True,
        project_rot=True,
    )
    before_audit = dict(audit_fn(snapshot))
    candidate_audit = dict(audit_fn(projected))
    decision = evaluate_stage_candidate(
        before_audit,
        candidate_audit,
        limits=limits,
        policy=policy,
        require_repair_gain=True,
    )
    return projected, {
        "projector": projector,
        "before_audit": before_audit,
        "candidate_audit": candidate_audit,
        "full_transaction_guard": decision,
    }


def _apply_one_transaction(
    motion,
    cond,
    seam_mask,
    model_path,
    contract_path,
    cfg,
    *,
    audit_fn,
    limits,
    policy,
    transaction_index,
):
    """Execute one complete-motion composite transaction."""
    snapshot = np.asarray(motion, dtype=np.float32).copy()
    report = {
        "schema": "v15_15h_full_transaction_inference_receipt_v1",
        "stage": "refiner_composite",
        "transaction_index": int(transaction_index),
        "selection": "identity",
        "accepted": False,
        "rolled_back": True,
        "runtime_case_labels_consumed": False,
        "runtime_case_whitelist_used": False,
        "pseudo_teacher_generated": False,
        "budgets_attempted": [],
        "attempts": [],
    }
    try:
        payload, contract, model = _load_composite(model_path, contract_path, cfg)
        fixed = contract["fixed_runtime_contract"]
        _require(abs(float(policy.minimum_repair_gain) - 0.03) <= 1.0e-15,
                 "runtime StageAcceptancePolicy minimum repair gain is not 0.03")
        device = next(model.parameters()).device
        batch = _batch(snapshot, cond, seam_mask, cfg, device)
        sample, ownership = _severity(batch, cfg)
        conformal = g1f._discriminative_conformal_status(
            sample, payload["conformal_envelope"]
        )
        report["observable_severity"] = conformal
        if not conformal["activation_supported_by_observables"]:
            report["reason"] = (
                "conformal_uncertain_identity"
                if conformal["severity_abstained"]
                else "observable_identity"
            )
            report["selected_audit"] = dict(audit_fn(snapshot))
            return snapshot, report

        with m.torch.no_grad():
            adapter_candidate, trace = adapter._adapter_batch_outputs(
                model, batch, cfg
            )
        mask = ownership.expand_as(trace["decoder_consistent_tangent"])
        gated_initial, normalized = g1f._normalize_exact_radius(
            trace["decoder_consistent_tangent"], mask, 1.0e-4
        )
        baseline = batch["bad"]
        if not normalized:
            initial, ungated_normalized = g1f._normalize_exact_radius(
                trace["decoder_consistent_ungated_tangent"], mask, 1.0e-4
            )
            report["second_order_zero_start_seed"] = {
                "policy": "observable_ungated_frozen_adapter_decoder_direction",
                "used": bool(ungated_normalized),
                "teacher_or_label_consumed": False,
            }
            report["adapter_incumbent"] = {
                "available": False,
                "reason": "gated_adapter_zero_direction",
            }
            if not ungated_normalized:
                report["reason"] = "zero_gradient_abstention"
                report["selected_audit"] = dict(audit_fn(snapshot))
                return snapshot, report
        else:
            initial = gated_initial
            report["second_order_zero_start_seed"] = {
                "policy": "observable_ungated_frozen_adapter_decoder_direction",
                "used": False,
                "teacher_or_label_consumed": False,
            }
            adapter_exact = product_exp_torch(baseline, initial)
            adapter_np = adapter_exact[0].detach().cpu().numpy()
            adapter_projected, adapter_guard = _project_and_guard(
                adapter_np, snapshot, cfg, audit_fn, limits, policy
            )
            adapter_tensor = m.torch.as_tensor(
                adapter_projected[None], dtype=m.torch.float32, device=device
            )
            adapter_science = _science_terms(
                adapter_tensor, baseline, batch["seam"], cfg
            )
            adapter_scope = float(
                product_log_torch(baseline, adapter_tensor)
                .masked_fill(mask, 0.0)
                .abs()
                .amax()
                .detach()
            )
            adapter_science_closed = all(
                float(adapter_science[name].detach()) <= 1.0e-12
                for name in ("endpoint", "temporal")
            )
            report["adapter_incumbent"] = {
                **adapter_guard,
                "scientific_closed": adapter_science_closed,
                "scope_leakage_abs_max": adapter_scope,
            }
            if (
                adapter_guard["full_transaction_guard"]["accepted"]
                and adapter_science_closed
                and adapter_scope == 0.0
            ):
                final_audit = dict(audit_fn(adapter_projected))
                report.update({
                    "selection": "adapter",
                    "adapter_incumbent_locked": True,
                    "accepted": True,
                    "rolled_back": False,
                    "reason": "adapter_full_guard_closure_locked",
                    "selected_audit": final_audit,
                    "post_commit_full_transaction_reaudit": final_audit,
                })
                return adapter_projected.astype(np.float32), report

        current = initial.detach()
        taper = trace["c2_taper"].expand_as(current).detach()
        angular = [float(value) for value in fixed["angular_line_search_radians"]]
        basis_dimension = int(fixed["second_order_basis_dimension"])
        grid_levels = int(fixed["second_order_grid_levels"])
        feasibility_tolerance = float(
            fixed["second_order_feasibility_tolerance"]
        )

        for budget in (2, 3, 5):
            report["budgets_attempted"].append(budget)
            budget_current = current.clone()
            for iteration in range(budget):
                variable = m.torch.zeros_like(budget_current).requires_grad_(True)
                local_physical = (budget_current + taper * variable).masked_fill(
                    ~mask, 0.0
                )
                expansion = product_exp_torch(baseline, local_physical)
                expansion_terms = _science_terms(
                    expansion, baseline, batch["seam"], cfg
                )
                active_name = max(
                    sorted(expansion_terms["guard_terms"]),
                    key=lambda name: float(
                        expansion_terms["guard_terms"][name].detach()
                    ),
                )
                scalar_terms = {
                    "shadow": expansion_terms["guard_terms"][active_name],
                    "endpoint": expansion_terms["endpoint"],
                    "temporal": expansion_terms["temporal"],
                }
                gradients = {}
                for index, name in enumerate(("shadow", "endpoint", "temporal")):
                    gradients[name] = m.torch.autograd.grad(
                        scalar_terms[name], variable,
                        retain_graph=index < 2, allow_unused=True,
                    )[0]
                if any(value is None for value in gradients.values()):
                    report["attempts"].append({
                        "budget": budget,
                        "iteration": iteration,
                        "state": "nonfinite_or_unverified_curvature",
                        "reason": "missing_metric_gradient",
                    })
                    break
                frozen_active_name = active_name

                def metric_builder(trial64):
                    candidate64 = product_exp_torch(
                        baseline.to(m.torch.float64), trial64
                    )
                    values = _science_terms(
                        candidate64,
                        baseline.to(m.torch.float64),
                        batch["seam"].to(m.torch.float64),
                        cfg,
                    )
                    return {
                        "shadow": values["guard_terms"][frozen_active_name],
                        "endpoint": values["endpoint"],
                        "temporal": values["temporal"],
                    }

                remaining_steps = int(budget) - int(iteration)
                required = {
                    name: (
                        float(value.detach()) + feasibility_tolerance
                    ) / float(remaining_steps)
                    for name, value in scalar_terms.items()
                }
                remaining_closure_gap = {
                    name: float(value.detach()) + feasibility_tolerance
                    for name, value in scalar_terms.items()
                }
                try:
                    prepared_model, preparation_audit = (
                        second_order.prepare_second_order_subproblem(
                            current=budget_current,
                            mask=mask,
                            taper=taper,
                            gradients=gradients,
                            metric_builder=metric_builder,
                            basis_dimension=basis_dimension,
                            direction_norm_floor=1.0e-8,
                        )
                    )
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    prepared_model = None
                    preparation_audit = {
                        "second_order_state": "second_order_solver_failure",
                        "exception": repr(exc),
                        "model_reused_across_frozen_angles": True,
                    }
                if prepared_model is None:
                    preparation_state = (
                        preparation_audit.get("second_order_state")
                        or preparation_audit.get("status")
                        or "nonfinite_or_unverified_curvature"
                    )
                    report["attempts"].append({
                        "budget": budget,
                        "iteration": iteration,
                        "state": preparation_state,
                        "second_order_model_preparation": preparation_audit,
                    })
                    break
                accepted_step = False
                for angle_index, theta in enumerate(angular):
                    try:
                        direction, solver = second_order.solve_prepared_second_order_angle(
                            prepared=prepared_model,
                            theta_radians=theta,
                            required_reduction=required,
                            grid_levels=grid_levels,
                            feasibility_tolerance=feasibility_tolerance,
                            permit_restoration_candidate=True,
                            restoration_required_reduction=(
                                remaining_closure_gap
                            ),
                        )
                        if direction is not None:
                            direction = direction.to(budget_current.dtype)
                    except (RuntimeError, ValueError, FloatingPointError) as exc:
                        direction = None
                        solver = {
                            "second_order_state": "second_order_solver_failure",
                            "exception": repr(exc),
                        }
                    attempt = {
                        "budget": budget,
                        "iteration": iteration,
                        "angle_index": angle_index,
                        "theta_radians": theta,
                        "prediction_active_guard_term": frozen_active_name,
                        "remaining_steps_including_current": remaining_steps,
                        "required_step_reduction_by_term": dict(required),
                        "complete_remaining_closure_gap_by_term": dict(
                            remaining_closure_gap
                        ),
                        "solver": solver,
                        "second_order_model_preparation": (
                            preparation_audit if angle_index == 0 else None
                        ),
                    }
                    if direction is None:
                        attempt["state"] = (
                            solver.get("second_order_state")
                            or solver.get("status")
                            or "insufficient_second_order_predicted_progress"
                        )
                        report["attempts"].append(attempt)
                        continue
                    trial, valid, geodesic = g1f._exact_radius_geodesic_update(
                        budget_current,
                        direction,
                        mask,
                        theta,
                        1.0e-8,
                        direction_is_sphere_tangent=True,
                    )
                    attempt["geodesic"] = geodesic
                    if not valid:
                        attempt["state"] = "nonfinite_or_unverified_curvature"
                        report["attempts"].append(attempt)
                        continue
                    trial_tensor = product_exp_torch(baseline, trial)
                    trial_terms = _science_terms(
                        trial_tensor, baseline, batch["seam"], cfg
                    )
                    authoritative_active = max(
                        sorted(trial_terms["guard_terms"]),
                        key=lambda name: float(
                            trial_terms["guard_terms"][name].detach()
                        ),
                    )
                    active_transition = authoritative_active != frozen_active_name
                    trial_np = trial_tensor[0].detach().cpu().numpy()
                    projected, guard = _project_and_guard(
                        trial_np, snapshot, cfg, audit_fn, limits, policy
                    )
                    projected_tensor = m.torch.as_tensor(
                        projected[None], dtype=m.torch.float32, device=device
                    )
                    projected_terms = _science_terms(
                        projected_tensor, baseline, batch["seam"], cfg
                    )
                    scope_leakage = float(
                        product_log_torch(baseline, projected_tensor)
                        .masked_fill(mask, 0.0)
                        .abs()
                        .amax()
                        .detach()
                    )
                    science_closed = all(
                        float(projected_terms[name].detach()) <= 1.0e-12
                        for name in ("endpoint", "temporal")
                    )
                    full_closed = bool(
                        guard["full_transaction_guard"]["accepted"]
                        and science_closed
                        and scope_leakage == 0.0
                    )
                    trial_scalar = {
                        "shadow": max(
                            float(value.detach())
                            for value in trial_terms["guard_terms"].values()
                        ),
                        "endpoint": float(trial_terms["endpoint"].detach()),
                        "temporal": float(trial_terms["temporal"].detach()),
                    }
                    expansion_scalar = {
                        name: float(value.detach())
                        for name, value in scalar_terms.items()
                    }
                    current_signed_gap = {
                        name: expansion_scalar[name] + feasibility_tolerance
                        for name in expansion_scalar
                    }
                    trial_signed_gap = {
                        name: trial_scalar[name] + feasibility_tolerance
                        for name in trial_scalar
                    }
                    gap_scale = {
                        name: max(abs(value), 1.0e-12)
                        for name, value in current_signed_gap.items()
                    }
                    current_gap_merit = sum(
                        (max(value, 0.0) / gap_scale[name]) ** 2
                        for name, value in current_signed_gap.items()
                    )
                    trial_gap_merit = sum(
                        (max(value, 0.0) / gap_scale[name]) ** 2
                        for name, value in trial_signed_gap.items()
                    )
                    actual_change = {
                        name: trial_scalar[name] - expansion_scalar[name]
                        for name in trial_scalar
                    }
                    quota_progress = bool(all(
                        math.isfinite(actual_change[name])
                        and actual_change[name]
                        <= -required[name] + feasibility_tolerance
                        for name in trial_scalar
                    ))
                    filter_boundary_ok = all(
                        math.isfinite(trial_signed_gap[name])
                        and (
                            current_signed_gap[name] > 0.0
                            or trial_signed_gap[name]
                            <= feasibility_tolerance
                        )
                        for name in trial_scalar
                    )
                    filter_progress = bool(
                        filter_boundary_ok
                        and math.isfinite(trial_gap_merit)
                        and trial_gap_merit
                        < current_gap_merit - feasibility_tolerance
                    )
                    actual_progress = bool(
                        full_closed
                        or (
                            remaining_steps > 1
                            and (quota_progress or filter_progress)
                        )
                    )
                    progress_mode = (
                        "authoritative_full_closure"
                        if full_closed
                        else "remaining_gap_share"
                        if quota_progress
                        else "authoritative_joint_gap_filter"
                        if filter_progress
                        else None
                    )
                    attempt.update({
                        "authoritative_active_guard_term": authoritative_active,
                        "active_set_transition": active_transition,
                        "projector_and_full_guard": guard,
                        "science_closed": science_closed,
                        "scope_leakage_abs_max": scope_leakage,
                        "authoritative_metric_before": expansion_scalar,
                        "authoritative_metric_after": trial_scalar,
                        "authoritative_metric_change": actual_change,
                        "current_signed_closure_gap_by_term": current_signed_gap,
                        "trial_signed_closure_gap_by_term": trial_signed_gap,
                        "current_positive_closure_gap_merit": current_gap_merit,
                        "trial_positive_closure_gap_merit": trial_gap_merit,
                        "authoritative_filter_safe_boundary": filter_boundary_ok,
                        "authoritative_filter_locked_safe_terms": sorted(
                            name
                            for name, value in current_signed_gap.items()
                            if value <= 0.0
                        ),
                        "authoritative_filter_progress": filter_progress,
                        "authoritative_progress": actual_progress,
                        "authoritative_progress_mode": progress_mode,
                        "second_order_prediction_passed": bool(
                            solver.get("joint_predicted_feasible", True)
                        ),
                        "second_order_restoration_candidate": bool(
                            solver.get("restoration_candidate", False)
                        ),
                        "state": (
                            "second_order_closure_succeeded"
                            if full_closed
                            else "second_order_trial_succeeded"
                            if actual_progress
                            else "active_set_transition_model_mismatch"
                            if active_transition
                            else "second_order_finite_radius_model_mismatch"
                        ),
                    })
                    report["attempts"].append(attempt)
                    if full_closed:
                        final_audit = dict(audit_fn(projected))
                        report.update({
                            "selection": f"second_order_k{budget}",
                            "adapter_incumbent_locked": False,
                            "accepted": True,
                            "rolled_back": False,
                            "reason": "second_order_closure_succeeded",
                            "selected_audit": final_audit,
                            "post_commit_full_transaction_reaudit": final_audit,
                        })
                        return projected.astype(np.float32), report
                    if actual_progress:
                        budget_current = trial.detach()
                        accepted_step = True
                        break
                if not accepted_step:
                    break

        report["reason"] = (
            report["attempts"][-1]["state"]
            if report["attempts"] else "second_order_solver_failure"
        )
        report["selected_audit"] = dict(audit_fn(snapshot))
        return snapshot, report
    except Exception as exc:
        report["reason"] = "second_order_solver_failure"
        report["exception"] = repr(exc)
        report["selected_audit"] = dict(audit_fn(snapshot))
        return snapshot, report


def apply_composite_refiner(
    motion,
    cond,
    seam_mask,
    model_path,
    contract_path,
    cfg,
    *,
    audit_fn,
    limits,
    policy,
):
    """Apply each contiguous boundary transaction with a full-motion Guard."""
    _load_composite(model_path, contract_path, cfg)
    _require(abs(float(policy.minimum_repair_gain) - 0.03) <= 1.0e-15,
             "runtime StageAcceptancePolicy minimum repair gain is not 0.03")
    selected = np.asarray(motion, dtype=np.float32).copy()
    seam = np.asarray(seam_mask, dtype=np.float32)
    active = seam[..., 0] >= 0.5 if seam.ndim == 2 else seam >= 0.5
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    regions = [(int(changes[i]), int(changes[i + 1]))
               for i in range(0, len(changes), 2)]
    transactions = []
    for transaction_index, (start, stop) in enumerate(regions):
        transaction_mask = np.zeros((len(selected), 1), dtype=np.float32)
        transaction_mask[start:stop, 0] = 1.0
        selected, receipt = _apply_one_transaction(
            selected,
            cond,
            transaction_mask,
            model_path,
            contract_path,
            cfg,
            audit_fn=audit_fn,
            limits=limits,
            policy=policy,
            transaction_index=transaction_index,
        )
        receipt["ownership_span"] = [start, stop]
        transactions.append(receipt)
    accepted_count = sum(int(row.get("accepted", False)) for row in transactions)
    return selected, {
        "schema": "v15_15h_full_motion_composite_receipt_v1",
        "stage": "refiner_composite",
        "transaction_count": len(transactions),
        "accepted_transaction_count": accepted_count,
        "accepted": accepted_count > 0,
        "rolled_back": accepted_count == 0,
        "selection": "composite_transactions" if accepted_count else "identity",
        "reason": (
            "one_or_more_atomic_transactions_committed"
            if accepted_count else "all_transactions_identity"
        ),
        "runtime_case_labels_consumed": False,
        "runtime_case_whitelist_used": False,
        "transactions": transactions,
        "selected_audit": dict(audit_fn(selected)),
        "post_commit_full_transaction_reaudit": dict(audit_fn(selected)),
    }
