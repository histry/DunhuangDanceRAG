"""V15.15g fixed-budget Euclidean/Riemannian correction ablation.

The Adapter is a frozen warm start.  Every method uses the same immutable
Anchor, ownership mask, exact 1e-4 tangent sphere, differentiable constraint
bundle and final raw/Projector/Guard audit.  Correction outputs are detached;
validation successes and failures are never recycled into training data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from motion_geometry.product_manifold import (
    product_exp_torch,
    product_log_torch,
)
from training import motion_models as m
from training import refiner_case_local_full_tangent_oracle as oracle
from training import refiner_observable_adapter_probe as adapter
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_15g_fixed_budget_manifold_correction_ablation_v1"
HARD_NEGATIVE_SCHEMA = "refiner_v15_15g_guard_rejected_direction_bank_v1"
TEACHER_SCHEMA = adapter.V15_15E_TEACHER_SCHEMA
METHODS = ("adapter", "euclidean_projected", "riemannian_retraction")
IMPLICIT_BACKWARD_PROTOCOL = (
    "future_only_implicit_kkt_adjoint_after_stop_gradient_gate"
)


def negative_contrastive_penalty(predicted, rejected, *, margin=0.0):
    """Repel a prediction from a rejected train-only direction.

    The caller owns split enforcement.  V15.15g never calls this function on
    validation evidence; it records the definition for a later train-only
    online-teacher experiment.
    """
    cosine = m.torch.nn.functional.cosine_similarity(
        predicted.reshape(1, -1),
        rejected.reshape(1, -1),
        dim=-1,
        eps=1.0e-12,
    ).mean()
    return m.torch.relu(cosine - float(margin))


def _owned_case_mask(ownership, tangent, case_index):
    selected = m.torch.zeros_like(ownership, dtype=m.torch.bool)
    selected[int(case_index)] = True
    return (selected & ownership).expand_as(tangent)


def _normalize_exact_radius(tangent, mask, target_rms):
    masked = tangent.masked_fill(~mask, 0.0)
    active = masked[mask]
    if active.numel() == 0:
        raise RuntimeError("empty correction ownership support")
    norm = m.torch.linalg.vector_norm(active)
    if not bool(m.torch.isfinite(norm)) or float(norm.detach()) <= 1.0e-12:
        return masked, False
    scale = (
        float(target_rms) * math.sqrt(float(active.numel()))
        / norm.detach()
    )
    result = (masked * scale).masked_fill(~mask, 0.0)
    return result, True


def _rms(value, mask):
    active = value[mask]
    if active.numel() == 0:
        return math.nan
    return float(m.torch.sqrt(active.square().mean()).detach())


def _constraint_loss(
    model,
    batch,
    cfg,
    baseline,
    identity,
    candidate,
    case_index,
    group,
    baseline_case,
    contract,
):
    limits, scales = oracle._fixed_guard_limits(
        contract["initial_anchor"],
        contract["relative_tolerance"],
        contract["absolute_tolerance"],
        group,
    )
    _, _, constraints = oracle._constraint_bundle(
        model,
        batch,
        cfg,
        candidate,
        identity,
        case_index,
        group,
        baseline_case,
        limits,
        scales,
    )
    vector = m.torch.stack([constraints[key] for key in sorted(constraints)])
    positive = m.torch.relu(vector)
    return positive.square().sum(), {
        key: float(constraints[key].detach()) for key in sorted(constraints)
    }


def _bounded_direction(direction, mask, maximum_rms):
    masked = direction.masked_fill(~mask, 0.0)
    active = masked[mask]
    norm = m.torch.linalg.vector_norm(active)
    if not bool(m.torch.isfinite(norm)) or float(norm.detach()) <= 1.0e-20:
        return m.torch.zeros_like(masked), False
    rms = norm / math.sqrt(float(active.numel()))
    scale = min(1.0, float(maximum_rms) / max(float(rms.detach()), 1.0e-20))
    return masked * scale, True


def _correct_case(
    *,
    method,
    steps,
    initial_tangent,
    ownership,
    baseline,
    identity,
    batch,
    cfg,
    case_index,
    group,
    baseline_case,
    contract,
    target_rms,
    step_size,
    trust_fraction,
):
    mask = _owned_case_mask(ownership, initial_tangent, case_index)
    current, normalized = _normalize_exact_radius(
        initial_tangent, mask, target_rms
    )
    history = []
    numeric_failure = not normalized
    started = time.perf_counter()
    if numeric_failure:
        return current.detach(), {
            "method": method,
            "steps_requested": int(steps),
            "steps_completed": 0,
            "numeric_failure": True,
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
        }

    for iteration in range(int(steps)):
        if method == "euclidean_projected":
            variable = current.detach().requires_grad_(True)
            candidate = product_exp_torch(
                baseline, variable.masked_fill(~mask, 0.0)
            )
        elif method == "riemannian_retraction":
            current_motion = product_exp_torch(baseline, current.detach())
            variable = m.torch.zeros_like(current).requires_grad_(True)
            candidate = product_exp_torch(
                current_motion, variable.masked_fill(~mask, 0.0)
            )
        else:
            raise ValueError(f"unsupported correction method: {method}")

        loss, constraints = _constraint_loss(
            model,
            batch,
            cfg,
            baseline,
            identity,
            candidate,
            case_index,
            group,
            baseline_case,
            contract,
        )
        gradient = m.torch.autograd.grad(
            loss, variable, allow_unused=True
        )[0]
        if gradient is None or not bool(m.torch.isfinite(gradient).all()):
            numeric_failure = True
            break
        direction, usable = _bounded_direction(
            -gradient.detach(),
            mask,
            float(target_rms) * float(trust_fraction),
        )
        if not usable:
            history.append({
                "iteration": iteration,
                "loss_before": float(loss.detach()),
                "accepted": False,
                "reason": "zero_or_nonfinite_correction_gradient",
                "constraints": constraints,
            })
            break

        accepted = False
        accepted_loss = float(loss.detach())
        accepted_scale = 0.0
        accepted_tangent = current
        for backtrack in range(6):
            alpha = float(step_size) * (0.5 ** backtrack)
            if method == "euclidean_projected":
                trial_raw = current + alpha * direction
            else:
                trial_motion = product_exp_torch(
                    current_motion, alpha * direction
                )
                trial_raw = product_log_torch(baseline, trial_motion)
            trial, trial_ok = _normalize_exact_radius(
                trial_raw, mask, target_rms
            )
            if not trial_ok:
                continue
            trial_candidate = product_exp_torch(baseline, trial)
            trial_loss, _ = _constraint_loss(
                model,
                batch,
                cfg,
                baseline,
                identity,
                trial_candidate,
                case_index,
                group,
                baseline_case,
                contract,
            )
            if (
                bool(m.torch.isfinite(trial_loss))
                and float(trial_loss.detach())
                <= float(loss.detach()) + 1.0e-12
            ):
                accepted = True
                accepted_loss = float(trial_loss.detach())
                accepted_scale = alpha
                accepted_tangent = trial.detach()
                break
        history.append({
            "iteration": iteration,
            "loss_before": float(loss.detach()),
            "loss_after": accepted_loss,
            "accepted": accepted,
            "step_scale": accepted_scale,
            "constraints": constraints,
            "exact_radius_rms": _rms(accepted_tangent, mask),
        })
        if not accepted:
            break
        current = accepted_tangent

    return current.detach(), {
        "method": method,
        "steps_requested": int(steps),
        "steps_completed": len(history),
        "accepted_steps": sum(int(row["accepted"]) for row in history),
        "numeric_failure": numeric_failure,
        "final_exact_radius_rms": _rms(current, mask),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _transaction_domains(teacher, batch, baseline, identity, model, cfg):
    domains = {}
    for transaction_id, metadata in teacher["transaction_schedules"].items():
        start = int(metadata["global_case_offset"])
        stop = start + int(metadata["case_count"])
        local_batch = adapter._slice_batch(batch, start, stop)
        local_baseline = baseline[start:stop]
        local_identity = identity[start:stop]
        domains[transaction_id] = {
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
            "contract": teacher["transaction_guard_contracts"][
                transaction_id
            ],
        }
    return domains


def _variant_summary(audits, correction_reports, elapsed):
    raw_by_group = Counter()
    projected_by_group = Counter()
    blockers = Counter()
    for row in audits:
        if row["raw_audit"]["passed"]:
            raw_by_group.update([row["group"]])
        if row["effective_projected_candidate"]:
            projected_by_group.update([row["group"]])
        blockers.update(row["raw_audit"].get("fixed_guard_blockers", []))
    return {
        "raw_pass_count_by_group": dict(raw_by_group),
        "projected_pass_count_by_group": dict(projected_by_group),
        "fixed_guard_blocker_counts": dict(blockers),
        "numeric_failure_count": sum(
            int(row.get("numeric_failure", False))
            for row in correction_reports.values()
        ),
        "elapsed_seconds": float(elapsed),
        "correction_by_case": correction_reports,
        "exact_audits": audits,
    }


def run(args):
    started = time.perf_counter()
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    teacher_path = Path(args.validation_teacher_bank).resolve()
    state_path = Path(args.adapter_state).resolve()
    teacher = m.torch.load(
        teacher_path, map_location="cpu", weights_only=False
    )
    if teacher.get("schema") != TEACHER_SCHEMA:
        raise RuntimeError("V15.15g requires a V15.15e bank")
    if teacher.get("split") != "validation":
        raise RuntimeError("V15.15g correction ablation requires validation")
    if not teacher.get("teacher_bank_ready"):
        raise RuntimeError("validation bank lacks a complete cross route")
    if teacher.get("train_validation_case_overlap"):
        raise RuntimeError("validation bank reports train case overlap")
    if teacher.get("train_validation_source_case_overlap"):
        raise RuntimeError("validation bank reports source-case overlap")

    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    cfg.product_refiner_observable_adapter = True
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.15g correction ablation requires CUDA")
    batch = adapter._to_device(teacher["batch"], device)
    model, _ = adapter._load_source_model(
        Path(teacher["source_diagnostic"]),
        cfg,
        device,
        adapter_state_path=state_path,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = m.torch.load(state_path, map_location="cpu", weights_only=False)
    model.observable_adapter_gate_floor.copy_(m.torch.as_tensor(
        float(state["gate_floor"]),
        dtype=model.observable_adapter_gate_floor.dtype,
        device=device,
    ))
    baseline = teacher["baseline_prediction"].to(device)
    identity = teacher["baseline_identity"].to(device)
    baseline_terms = oracle.case_probe._case_terms(baseline, batch, cfg)
    domains = _transaction_domains(
        teacher, batch, baseline, identity, model, cfg
    )
    _, trace = adapter._adapter_batch_outputs(model, batch, cfg)
    ownership = trace["ownership"].to(m.torch.bool)
    initial = trace["decoder_consistent_tangent"].masked_fill(
        ~ownership.expand_as(trace["decoder_consistent_tangent"]), 0.0
    )
    samples = teacher["samples"]
    initial, radius = adapter._safe_exact_radius_tangent(
        initial,
        ownership,
        samples,
        float(args.target_rms),
        eps=1.0e-8,
    )

    variants = {}
    hard_negatives = []
    budgets = tuple(int(value) for value in args.steps)
    specifications = [("adapter", 0)] + [
        (method, budget)
        for method in METHODS[1:]
        for budget in budgets
    ]
    workspace_floor = max(1.0e-8, float(args.target_rms) * 0.01)
    for method, budget in specifications:
        variant_started = time.perf_counter()
        tangent = initial.detach().clone()
        correction_reports = {}
        if method != "adapter":
            tangent.zero_()
            for sample in samples:
                if sample["teacher_kind"] != "exact_projected_direction":
                    continue
                transaction_id = str(sample["transaction_id"])
                domain = domains[transaction_id]
                start, stop = domain["slice"]
                local_case = int(sample["local_case_index"])
                group = str(sample["audit_group"])
                local_initial = initial[start:stop]
                local_ownership = ownership[start:stop]
                baseline_case = {
                    key: float(
                        baseline_terms[key][int(sample["case_index"])]
                    )
                    for key in ("endpoint", "temporal")
                }
                corrected, correction_report = _correct_case(
                    method=method,
                    steps=budget,
                    initial_tangent=local_initial,
                    ownership=local_ownership,
                    baseline=domain["baseline"],
                    identity=domain["identity"],
                    batch=domain["batch"],
                    cfg=cfg,
                    case_index=local_case,
                    group=group,
                    baseline_case=baseline_case,
                    contract=domain["contract"],
                    target_rms=float(args.target_rms),
                    step_size=float(args.step_size),
                    trust_fraction=float(args.trust_fraction),
                )
                mask = _owned_case_mask(
                    local_ownership, corrected, local_case
                )
                tangent[start:stop] += corrected.masked_fill(~mask, 0.0)
                correction_reports[str(sample["case_uid"])] = (
                    correction_report
                )
        audits = adapter._audit_step(
            model,
            batch,
            cfg,
            tangent.detach(),
            tangent.detach(),
            trace["c2_taper"].detach(),
            ownership.detach(),
            baseline,
            identity,
            None,
            {},
            baseline_terms,
            samples,
            float(args.target_rms),
            workspace_floor,
            args,
            transaction_domains=domains,
        )
        key = method if method == "adapter" else f"{method}_k{budget}"
        variants[key] = _variant_summary(
            audits,
            correction_reports,
            time.perf_counter() - variant_started,
        )
        audit_by_uid = {str(row["case_uid"]): row for row in audits}
        for sample in samples:
            if sample["teacher_kind"] != "exact_projected_direction":
                continue
            uid = str(sample["case_uid"])
            audit = audit_by_uid[uid]
            if audit["raw_audit"]["passed"]:
                continue
            hard_negatives.append({
                "case_uid": uid,
                "transaction_id": str(sample["transaction_id"]),
                "local_case_index": int(sample["local_case_index"]),
                "group": str(sample["audit_group"]),
                "method": method,
                "steps": int(budget),
                "fixed_guard_blockers": list(
                    audit["raw_audit"].get("fixed_guard_blockers", [])
                ),
                "teacher_eligible": False,
                "negative_contrastive_definition": (
                    "relu(cos(predicted,rejected)-margin)"
                ),
            })
        print(json.dumps({
            "stage": "v15_15g_correction_variant",
            "variant": key,
            "raw_pass_count_by_group": variants[key][
                "raw_pass_count_by_group"
            ],
            "projected_pass_count_by_group": variants[key][
                "projected_pass_count_by_group"
            ],
            "numeric_failure_count": variants[key][
                "numeric_failure_count"
            ],
        }), flush=True)

    base_cross_long = variants["adapter"][
        "projected_pass_count_by_group"
    ].get("cross_long", 0)
    supported_budgets = []
    for budget in budgets:
        euclidean = variants[f"euclidean_projected_k{budget}"]
        riemannian = variants[f"riemannian_retraction_k{budget}"]
        riemannian_cross_long = riemannian[
            "projected_pass_count_by_group"
        ].get("cross_long", 0)
        euclidean_cross_long = euclidean[
            "projected_pass_count_by_group"
        ].get("cross_long", 0)
        if (
            riemannian_cross_long > base_cross_long
            and riemannian_cross_long > euclidean_cross_long
            and riemannian["numeric_failure_count"] == 0
        ):
            supported_budgets.append(int(budget))
    riemannian_supported = bool(supported_budgets)
    hard_negative_path = destination / "validation_hard_negatives.pt"
    m.torch.save({
        "schema": HARD_NEGATIVE_SCHEMA,
        "split": "validation",
        "training_allowed": False,
        "teacher_eligible": False,
        "directions_serialized": False,
        "records": hard_negatives,
    }, hard_negative_path)
    report = {
        "schema": SCHEMA,
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "validation_teacher_bank": str(teacher_path),
        "adapter_state": str(state_path),
        "adapter_role": "learner_warm_start",
        "correction_gradient_protocol": "stop_gradient",
        "future_end_to_end_gradient_protocol": IMPLICIT_BACKWARD_PROTOCOL,
        "unrolled_backward_allowed": False,
        "target_rms": float(args.target_rms),
        "euclidean_baseline_protocol": (
            "tangent_gradient_then_exact_sphere_projection_then_retraction"
        ),
        "riemannian_protocol": (
            "moving_product_tangent_retraction_then_original_anchor_sphere_projection"
        ),
        "same_checkpoint_cases_radius_and_budget": True,
        "validation_recycled_as_teacher": False,
        "pseudo_teachers_generated": False,
        "exact_radius_normalization_by_case": radius,
        "variants": variants,
        "hard_negative_artifact": str(hard_negative_path),
        "hard_negative_training_allowed": False,
        "riemannian_correction_supported": riemannian_supported,
        "riemannian_supported_budgets": supported_budgets,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "numeric_audit_complete": all(
            row["exact_audits"] for row in variants.values()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = destination / "fixed_budget_correction.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "v15_15g_fixed_budget_correction_complete",
        "report": str(report_path),
        "riemannian_correction_supported": riemannian_supported,
        "numeric_audit_complete": report["numeric_audit_complete"],
    }), flush=True)
    return 0 if report["numeric_audit_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-teacher-bank", required=True)
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--steps", type=int, nargs="+", default=(2, 3, 5))
    parser.add_argument("--target-rms", type=float, default=1.0e-4)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--trust-fraction", type=float, default=0.5)
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
    if tuple(sorted(set(args.steps))) != (2, 3, 5):
        parser.error("--steps must contain exactly 2 3 5")
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if not 0.0 < args.trust_fraction <= 1.0:
        parser.error("--trust-fraction must be in (0, 1]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
