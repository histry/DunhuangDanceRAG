"""V15.15g fixed-budget Euclidean/Riemannian correction ablation.

The Adapter is a frozen warm start.  Every method uses the same immutable
Anchor, ownership mask, exact 1e-4 tangent sphere, differentiable constraint
bundle and final raw/Projector/Guard audit.  Correction outputs are detached;
validation successes and failures are never recycled into training data.

The activation-aware modes add identity as an explicit zero-action candidate.
V15.15g1 freezes an observable severity envelope from train-split single
controls, then selects among Adapter, Euclidean and Riemannian candidates only
when the Anchor lies outside that envelope.  Five differentiable signed-margin
terms from the fixed-Guard metric implementation prevent jerk/window/boundary
regression. Offline role/group metadata is attached only after selection.

V15.15g1b interprets those fields directly as contract signed margins, uses a
Guard-first projected active set, and calibrates a multivariate observable
severity score by leaving out each train transaction.  Its uncertainty band
falls back to identity.  The correction budgets and starts remain unchanged.

V15.15g1c separates the per-case physical margins from the authoritative
fixed-bank Guard contract.  Activation uses a train-only, transaction-held-out
single/cross discriminative conformal model, while every activated candidate
is checked on its complete transaction against the frozen Guard anchor before
selection.  A feasible Adapter is locked as the incumbent; correction is
considered only when that incumbent fails exact raw closure.

V15.15g1d keeps that activation and selection contract, but repairs a candidate
inside its complete transaction.  Hard fixed-Guard shadows choose the active
terms and accept line-search trials; train-frozen LogSumExp relaxations provide
the gradient direction without changing the authoritative Guard thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
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
from training import refiner_bridge_diagnostics as diagnostic
from training import refiner_case_local_full_tangent_oracle as oracle
from training import refiner_observable_adapter_probe as adapter
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_15g_fixed_budget_manifold_correction_ablation_v1"
ACTIVATION_AWARE_SCHEMA = (
    "refiner_v15_15g_activation_aware_manifold_selection_probe_v1"
)
G1_SCHEMA = (
    "refiner_v15_15g1_observable_severity_guard_aligned_correction_v1"
)
G1B_SCHEMA = (
    "refiner_v15_15g1b_contract_margin_transaction_conformal_repair_v1"
)
G1C_SCHEMA = (
    "refiner_v15_15g1c_fixed_guard_shadow_incumbent_lock_"
    "discriminative_conformal_v1"
)
G1D_SCHEMA = (
    "refiner_v15_15g1d_full_transaction_shadow_gradient_repair_v1"
)
HARD_NEGATIVE_SCHEMA = "refiner_v15_15g_guard_rejected_direction_bank_v1"
TEACHER_SCHEMA = adapter.V15_15E_TEACHER_SCHEMA
METHODS = ("adapter", "euclidean_projected", "riemannian_retraction")
IMPLICIT_BACKWARD_PROTOCOL = (
    "future_only_implicit_kkt_adjoint_after_stop_gradient_gate"
)
ACTIVATION_PHYSICAL_PROXY_KEYS = (
    "jerk_safety_excess",
    "root_vertical_safety_excess",
    "support_excess",
    "penetration_excess",
    "observable_trust_excess",
    "outside",
    "contact",
)
G1_GUARD_PROXY_TERMS = {
    "joint_jerk_p95": "repair_joint_jerk_mps3_p95_signed_margin",
    "joint_jerk_window_p95": (
        "repair_joint_jerk_window_p95_max_mps3_signed_margin"
    ),
    "extremity_jerk_p95": (
        "repair_extremity_jerk_mps3_p95_signed_margin"
    ),
    "extremity_jerk_window_p95": (
        "repair_extremity_jerk_window_p95_max_mps3_signed_margin"
    ),
    "boundary": "boundary_jerk_signed_margin",
}
FULL_GUARD_PHYSICAL_CASE_TERMS = {
    "joint_jerk_p95": "repair_joint_jerk_mps3_p95_signed_margin",
    "joint_jerk_max": "repair_joint_jerk_mps3_max_signed_margin",
    "joint_jerk_window_p95": (
        "repair_joint_jerk_window_p95_max_mps3_signed_margin"
    ),
    "extremity_jerk_p95": (
        "repair_extremity_jerk_mps3_p95_signed_margin"
    ),
    "extremity_jerk_window_p95": (
        "repair_extremity_jerk_window_p95_max_mps3_signed_margin"
    ),
    "foot_skate_p95": "repair_foot_skate_mps_p95_signed_margin",
    "foot_skate_max": "repair_foot_skate_mps_max_signed_margin",
    "support_drift_p95": (
        "repair_foot_support_drift_m_p95_signed_margin"
    ),
    "support_drift_max": (
        "repair_foot_support_drift_m_max_signed_margin"
    ),
    "penetration": "repair_foot_penetration_min_m_signed_margin",
    "boundary": "boundary_jerk_signed_margin",
}
G1_SEVERITY_CHANNELS = (
    "endpoint_gap",
    "temporal_gap",
    "root_gap",
    "root_velocity_gap",
    "yaw_gap",
    "support_mismatch",
    "phase_coverage",
    "phase_edge_density",
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _label_free_activation_objective(
    batch,
    cfg,
    baseline,
    candidate,
    case_index,
):
    """Return a per-case observable objective with no role/group input."""
    _, terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    index = int(case_index)
    components = {
        "endpoint_scientific_deficit": terms[
            "endpoint_scientific_deficit"
        ][index],
        "temporal_scientific_deficit": terms[
            "temporal_scientific_deficit"
        ][index],
    }
    for key in ACTIVATION_PHYSICAL_PROXY_KEYS:
        components[key] = terms[key][index]
    loss = m.torch.stack([
        m.torch.clamp_min(value, 0.0)
        for value in components.values()
    ]).sum()
    physical = m.torch.stack([
        m.torch.clamp_min(components[key], 0.0)
        for key in ACTIVATION_PHYSICAL_PROXY_KEYS
    ]).sum()
    return loss, physical, {
        key: float(value.detach()) for key, value in components.items()
    }


def _observable_severity(sample):
    """Summarize serialized Adapter observables without using role labels."""
    features = sample["observable_condition"].detach().to(
        dtype=m.torch.float64, device="cpu"
    )
    ownership = sample["ownership"].detach().to(
        dtype=m.torch.bool, device="cpu"
    )
    if features.ndim != 2 or features.shape[-1] != (
        m.REFINER_ADAPTER_OBSERVABLE_DIM + m.REFINER_ADAPTER_PHASE_DIM
    ):
        raise RuntimeError("observable severity feature layout mismatch")
    active = ownership[..., 0] if ownership.ndim == 2 else ownership
    if active.ndim != 1 or active.shape[0] != features.shape[0]:
        raise RuntimeError("observable severity ownership layout mismatch")
    if not bool(active.any()):
        raise RuntimeError("observable severity has empty ownership")
    owned = features[active]
    support_mismatch = (
        owned[:, 5:9] - owned[:, 9:13]
    ).abs().amax()
    phase_start = m.REFINER_ADAPTER_OBSERVABLE_DIM
    phase_coverage = features[:, phase_start].mean()
    phase_edge_density = owned[:, phase_start + 1].abs().mean()
    values = {
        "endpoint_gap": owned[:, 0].amax(),
        "temporal_gap": owned[:, 1].amax(),
        "root_gap": owned[:, 2].amax(),
        "root_velocity_gap": owned[:, 3].amax(),
        "yaw_gap": owned[:, 4].amax(),
        "support_mismatch": support_mismatch,
        "phase_coverage": phase_coverage,
        "phase_edge_density": phase_edge_density,
    }
    return {key: float(values[key]) for key in G1_SEVERITY_CHANNELS}


def _freeze_single_severity_envelope(
    train_teacher,
    *,
    margin_fraction,
    absolute_margin,
):
    """Freeze an observable envelope from train-split single controls only."""
    controls = [
        sample for sample in train_teacher["samples"]
        if sample.get("teacher_kind") == "identity_control"
    ]
    if not controls:
        raise RuntimeError("train bank has no single controls for envelope")
    rows = {
        str(sample["case_uid"]): _observable_severity(sample)
        for sample in controls
    }
    maxima = {
        key: max(row[key] for row in rows.values())
        for key in G1_SEVERITY_CHANNELS
    }
    limits = {
        key: max(
            maxima[key] * (1.0 + float(margin_fraction)),
            maxima[key] + float(absolute_margin),
        )
        for key in G1_SEVERITY_CHANNELS
    }
    return {
        "schema": "refiner_v15_15g1_single_severity_envelope_v1",
        "calibration_split": "train",
        "calibration_role": "identity_control",
        "calibration_role_used_offline_only": True,
        "inference_role_label_consumed": False,
        "control_count": len(controls),
        "channels": list(G1_SEVERITY_CHANNELS),
        "maximum_by_channel": maxima,
        "limit_by_channel": limits,
        "margin_fraction": float(margin_fraction),
        "absolute_margin": float(absolute_margin),
        "case_severity": rows,
    }


def _severity_status(sample, envelope, scale_floor):
    values = _observable_severity(sample)
    limits = envelope["limit_by_channel"]
    excess = {
        key: values[key] - float(limits[key])
        for key in G1_SEVERITY_CHANNELS
    }
    normalized = {
        key: excess[key] / max(abs(float(limits[key])), float(scale_floor))
        for key in G1_SEVERITY_CHANNELS
    }
    return {
        "values": values,
        "limits": dict(limits),
        "excess": excess,
        "normalized_excess": normalized,
        "maximum_normalized_excess": max(normalized.values()),
        "outside_frozen_single_envelope": any(
            value > 0.0 for value in excess.values()
        ),
    }


def _fit_multivariate_severity_model(rows, shrinkage, scale_floor):
    matrix = m.torch.tensor(
        [[row[key] for key in G1_SEVERITY_CHANNELS] for row in rows],
        dtype=m.torch.float64,
    )
    if matrix.shape[0] < 2:
        raise RuntimeError("multivariate severity model needs two controls")
    center = matrix.median(dim=0).values
    absolute_deviation = (matrix - center).abs().median(dim=0).values
    mad_scale = 1.4826 * absolute_deviation
    standard_scale = matrix.std(dim=0, unbiased=False)
    scale = m.torch.where(
        mad_scale > float(scale_floor),
        mad_scale,
        standard_scale.clamp_min(float(scale_floor)),
    )
    standardized = (matrix - center) / scale
    covariance = (
        standardized.transpose(0, 1) @ standardized
        / float(max(1, matrix.shape[0] - 1))
    )
    dimension = covariance.shape[0]
    covariance = (
        (1.0 - float(shrinkage)) * covariance
        + float(shrinkage)
        * m.torch.eye(dimension, dtype=m.torch.float64)
    )
    precision = m.torch.linalg.pinv(covariance, hermitian=True)
    if not bool(m.torch.isfinite(precision).all()):
        raise RuntimeError("nonfinite multivariate severity precision")
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "precision": precision.tolist(),
        "shrinkage": float(shrinkage),
        "scale_floor": float(scale_floor),
        "sample_count": int(matrix.shape[0]),
    }


def _multivariate_severity_score(values, model):
    vector = m.torch.tensor(
        [values[key] for key in G1_SEVERITY_CHANNELS],
        dtype=m.torch.float64,
    )
    center = m.torch.tensor(model["center"], dtype=m.torch.float64)
    scale = m.torch.tensor(model["scale"], dtype=m.torch.float64)
    precision = m.torch.tensor(
        model["precision"], dtype=m.torch.float64
    )
    standardized = (vector - center) / scale
    squared = standardized @ precision @ standardized
    return math.sqrt(max(0.0, float(squared)))


def _freeze_transaction_conformal_envelope(
    train_teacher,
    *,
    shrinkage,
    scale_floor,
    uncertainty_fraction,
    absolute_margin,
):
    controls = [
        sample for sample in train_teacher["samples"]
        if sample.get("teacher_kind") == "identity_control"
    ]
    by_transaction = {}
    rows_by_uid = {}
    for sample in controls:
        uid = str(sample["case_uid"])
        transaction_id = str(sample["transaction_id"])
        severity = _observable_severity(sample)
        rows_by_uid[uid] = severity
        by_transaction.setdefault(transaction_id, []).append((uid, severity))
    if len(by_transaction) < 2:
        raise RuntimeError(
            "transaction-conformal envelope needs two train transactions"
        )
    fold_scores = {}
    fold_maxima = {}
    for held_out_transaction, held_out_rows in by_transaction.items():
        fitting_rows = [
            severity
            for transaction_id, transaction_rows in by_transaction.items()
            if transaction_id != held_out_transaction
            for _, severity in transaction_rows
        ]
        model = _fit_multivariate_severity_model(
            fitting_rows,
            shrinkage,
            scale_floor,
        )
        scores = {
            uid: _multivariate_severity_score(severity, model)
            for uid, severity in held_out_rows
        }
        fold_scores[held_out_transaction] = scores
        fold_maxima[held_out_transaction] = max(scores.values())
    conformal_threshold = max(fold_maxima.values())
    activation_threshold = max(
        conformal_threshold * (1.0 + float(uncertainty_fraction)),
        conformal_threshold + float(absolute_margin),
    )
    final_model = _fit_multivariate_severity_model(
        list(rows_by_uid.values()),
        shrinkage,
        scale_floor,
    )
    return {
        "schema": (
            "refiner_v15_15g1b_transaction_conformal_single_envelope_v1"
        ),
        "calibration_split": "train",
        "calibration_role": "identity_control",
        "calibration_role_used_offline_only": True,
        "inference_role_label_consumed": False,
        "channels": list(G1_SEVERITY_CHANNELS),
        "control_count": len(controls),
        "transaction_count": len(by_transaction),
        "leave_one_transaction_out": True,
        "fold_score_by_case": fold_scores,
        "fold_maximum_score": fold_maxima,
        "conformal_threshold": conformal_threshold,
        "activation_threshold": activation_threshold,
        "uncertainty_fraction": float(uncertainty_fraction),
        "absolute_margin": float(absolute_margin),
        "final_model": final_model,
    }


def _conformal_severity_status(sample, envelope):
    values = _observable_severity(sample)
    score = _multivariate_severity_score(values, envelope["final_model"])
    conformal_threshold = float(envelope["conformal_threshold"])
    activation_threshold = float(envelope["activation_threshold"])
    abstained = bool(
        conformal_threshold < score <= activation_threshold
    )
    return {
        "values": values,
        "severity_conformal_score": score,
        "severity_conformal_threshold": conformal_threshold,
        "severity_activation_threshold": activation_threshold,
        "severity_abstained": abstained,
        "conformal_fallback": abstained,
        "outside_frozen_single_envelope": bool(
            score > activation_threshold
        ),
    }


def _freeze_discriminative_transaction_conformal(
    train_teacher,
    *,
    shrinkage,
    scale_floor,
    uncertainty_fraction,
    absolute_margin,
):
    """Freeze a train-only two-sided single/cross conformal classifier.

    Role labels are consumed only here.  Each held-out transaction is scored
    by single and cross Mahalanobis models fitted without that transaction.
    Runtime classification receives only the observable severity vector.
    """
    labelled = {"single": [], "cross": []}
    for sample in train_teacher["samples"]:
        kind = sample.get("teacher_kind")
        if kind == "identity_control":
            label = "single"
        elif kind == "exact_projected_direction":
            label = "cross"
        else:
            continue
        labelled[label].append({
            "case_uid": str(sample["case_uid"]),
            "transaction_id": str(sample["transaction_id"]),
            "severity": _observable_severity(sample),
        })
    for label, rows in labelled.items():
        transactions = {row["transaction_id"] for row in rows}
        if len(rows) < 2 or len(transactions) < 2:
            raise RuntimeError(
                "discriminative conformal calibration requires two "
                f"{label} transactions"
            )

    transactions = sorted({
        row["transaction_id"]
        for rows in labelled.values()
        for row in rows
    })
    fold_scores = {}
    same_class_distances = {"single": [], "cross": []}
    discriminants = {"single": [], "cross": []}
    for held_out_transaction in transactions:
        fit_rows = {
            label: [
                row["severity"]
                for row in rows
                if row["transaction_id"] != held_out_transaction
            ]
            for label, rows in labelled.items()
        }
        if any(len(rows) < 2 for rows in fit_rows.values()):
            raise RuntimeError(
                "transaction-held-out discriminative fold lacks calibration"
            )
        models = {
            label: _fit_multivariate_severity_model(
                rows, shrinkage, scale_floor
            )
            for label, rows in fit_rows.items()
        }
        held_out_scores = {}
        for label, rows in labelled.items():
            for row in rows:
                if row["transaction_id"] != held_out_transaction:
                    continue
                single_distance = _multivariate_severity_score(
                    row["severity"], models["single"]
                )
                cross_distance = _multivariate_severity_score(
                    row["severity"], models["cross"]
                )
                discriminant = single_distance - cross_distance
                held_out_scores[row["case_uid"]] = {
                    "offline_label": label,
                    "single_distance": single_distance,
                    "cross_distance": cross_distance,
                    "cross_discriminant": discriminant,
                }
                same_class_distances[label].append(
                    single_distance if label == "single" else cross_distance
                )
                discriminants[label].append(discriminant)
        if held_out_scores:
            fold_scores[held_out_transaction] = held_out_scores

    single_distance_threshold = max(same_class_distances["single"])
    cross_distance_threshold = max(same_class_distances["cross"])
    single_discriminant_upper = max(discriminants["single"])
    cross_discriminant_lower = min(discriminants["cross"])
    discriminant_scale = max(
        1.0,
        abs(single_discriminant_upper),
        abs(cross_discriminant_lower),
    )
    uncertainty_margin = max(
        discriminant_scale * float(uncertainty_fraction),
        float(absolute_margin),
    )
    activation_discriminant_threshold = max(
        cross_discriminant_lower,
        single_discriminant_upper + uncertainty_margin,
    )
    final_models = {
        label: _fit_multivariate_severity_model(
            [row["severity"] for row in rows], shrinkage, scale_floor
        )
        for label, rows in labelled.items()
    }
    return {
        "schema": (
            "refiner_v15_15g1c_train_transaction_discriminative_"
            "conformal_v1"
        ),
        "calibration_split": "train",
        "calibration_labels": {
            "single": "identity_control",
            "cross": "exact_projected_direction",
        },
        "calibration_labels_used_offline_only": True,
        "inference_role_label_consumed": False,
        "inference_group_label_consumed": False,
        "inference_hidden_clean_consumed": False,
        "channels": list(G1_SEVERITY_CHANNELS),
        "class_counts": {
            label: len(rows) for label, rows in labelled.items()
        },
        "transaction_count": len(transactions),
        "leave_one_transaction_out": True,
        "fold_score_by_case": fold_scores,
        "single_distance_threshold": single_distance_threshold,
        "cross_distance_threshold": cross_distance_threshold,
        "single_discriminant_upper": single_discriminant_upper,
        "cross_discriminant_lower": cross_discriminant_lower,
        "activation_discriminant_threshold": (
            activation_discriminant_threshold
        ),
        "uncertainty_margin": uncertainty_margin,
        "uncertainty_fraction": float(uncertainty_fraction),
        "absolute_margin": float(absolute_margin),
        "class_overlap_in_calibration": bool(
            cross_discriminant_lower <= single_discriminant_upper
        ),
        "final_models": final_models,
    }


def _discriminative_conformal_status(sample, envelope):
    """Classify from observable severity only; ambiguity is identity-safe."""
    values = _observable_severity(sample)
    single_distance = _multivariate_severity_score(
        values, envelope["final_models"]["single"]
    )
    cross_distance = _multivariate_severity_score(
        values, envelope["final_models"]["cross"]
    )
    discriminant = single_distance - cross_distance
    single_inlier = bool(
        single_distance <= float(envelope["single_distance_threshold"])
    )
    cross_inlier = bool(
        cross_distance <= float(envelope["cross_distance_threshold"])
    )
    separated_cross = bool(
        discriminant
        > float(envelope["activation_discriminant_threshold"])
    )
    activated = bool(cross_inlier and not single_inlier and separated_cross)
    confident_identity = bool(
        single_inlier
        and discriminant <= float(envelope["single_discriminant_upper"])
    )
    abstained = bool(not activated and not confident_identity)
    return {
        "values": values,
        "single_conformal_distance": single_distance,
        "cross_conformal_distance": cross_distance,
        "single_distance_threshold": float(
            envelope["single_distance_threshold"]
        ),
        "cross_distance_threshold": float(
            envelope["cross_distance_threshold"]
        ),
        "cross_discriminant": discriminant,
        "single_discriminant_upper": float(
            envelope["single_discriminant_upper"]
        ),
        "activation_discriminant_threshold": float(
            envelope["activation_discriminant_threshold"]
        ),
        "single_inlier": single_inlier,
        "cross_inlier": cross_inlier,
        "activation_supported_by_observables": activated,
        "severity_abstained": abstained,
        "conformal_fallback": abstained,
        "outside_frozen_single_envelope": activated,
    }


def _guard_proxy_values(batch, cfg, baseline, candidate, case_index):
    _, terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    index = int(case_index)
    return {
        name: terms[key][index]
        for name, key in G1_GUARD_PROXY_TERMS.items()
    }


def _fixed_guard_shadow_from_details(details, relative, absolute):
    """Reproduce the exact fixed-Guard formula and retain its components."""
    shadow = {}
    audit = {}
    for name, row in details.items():
        fixed_anchor = float(row["fixed_anchor"])
        relative_tolerance = float(relative[name])
        absolute_tolerance = float(absolute[name])
        allowance = max(
            abs(fixed_anchor) * relative_tolerance,
            absolute_tolerance,
        )
        absolute_limit = fixed_anchor + allowance
        recorded_limit = float(row["absolute_limit"])
        numeric_tolerance = float(row["numeric_tolerance"])
        margin = (
            float(row["candidate"])
            - absolute_limit
            - numeric_tolerance
        )
        shadow[name] = margin
        audit[name] = {
            "group_guard_value": float(row["candidate"]),
            "fixed_anchor": fixed_anchor,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "fixed_anchor_allowance": allowance,
            "absolute_limit": absolute_limit,
            "recorded_absolute_limit": recorded_limit,
            "absolute_limit_matches_exact_guard": bool(
                math.isclose(
                    absolute_limit,
                    recorded_limit,
                    rel_tol=0.0,
                    abs_tol=max(1.0e-15, abs(recorded_limit) * 1.0e-12),
                )
            ),
            "numeric_tolerance": numeric_tolerance,
            "fixed_guard_shadow_margin": margin,
            "passed": bool(margin <= 0.0),
        }
    return shadow, audit


def _fixed_guard_limit(contract, name):
    """Return the exact immutable Guard limit and comparison tolerance."""
    anchor = float(contract["initial_anchor"][name])
    relative = float(contract["relative_tolerance"][name])
    absolute = float(contract["absolute_tolerance"][name])
    allowance = max(abs(anchor) * relative, absolute)
    allowed = anchor + allowance
    numeric = max(1.0e-12, abs(allowed) * 1.0e-9, allowance * 1.0e-6)
    return {
        "fixed_anchor": anchor,
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "fixed_anchor_allowance": allowance,
        "absolute_limit": allowed,
        "numeric_tolerance": numeric,
    }


def _median(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _freeze_train_full_shadow_repair_contract(
    train_teacher,
    *,
    lse_allowance_fraction,
    lse_temperature_floor,
    projection_damping,
    minimum_reduction_floor,
):
    """Freeze every g1d numerical choice from train transactions only."""
    if train_teacher.get("split") != "train":
        raise RuntimeError("g1d shadow repair calibration requires train split")
    contracts = train_teacher.get("transaction_guard_contracts") or {}
    if not contracts:
        raise RuntimeError("g1d train bank lacks transaction Guard contracts")
    allowances = {}
    numerics = {}
    transaction_ids = sorted(str(key) for key in contracts)
    for transaction_id in transaction_ids:
        contract = contracts[transaction_id]
        for name in sorted(contract["initial_anchor"]):
            limit = _fixed_guard_limit(contract, name)
            allowances.setdefault(name, []).append(
                limit["fixed_anchor_allowance"]
            )
            numerics.setdefault(name, []).append(limit["numeric_tolerance"])
    temperatures = {
        name: max(
            float(lse_temperature_floor),
            float(lse_allowance_fraction) * _median(values),
        )
        for name, values in allowances.items()
    }
    minimum_reductions = {
        name: max(float(minimum_reduction_floor), _median(numerics[name]))
        for name in sorted(numerics)
    }
    return {
        "schema": "refiner_v15_15g1d_train_frozen_shadow_repair_contract_v1",
        "calibration_split": "train",
        "validation_consumed_for_calibration": False,
        "transaction_ids": transaction_ids,
        "transaction_count": len(transaction_ids),
        "group_aggregation_gradient_relaxation": (
            "logsumexp_train_frozen_temperature"
        ),
        "authoritative_forward_and_acceptance_aggregation": (
            "exact_fixed_guard_hard_group_aggregation"
        ),
        "p95_active_index_frozen_within_line_search": True,
        "lse_allowance_fraction": float(lse_allowance_fraction),
        "lse_temperature_floor": float(lse_temperature_floor),
        "lse_temperature_by_guard_term": temperatures,
        "minimum_shadow_reduction_by_guard_term": minimum_reductions,
        "projection_damping": float(projection_damping),
        "line_search_acceptance": [
            "full_transaction_fixed_guard_shadow_strictly_decreases",
            "endpoint_and_temporal_strict_descent_pass",
            "owned_tangent_radius_rms_equals_1e-4",
            "outside_scope_abs_max_equals_0",
        ],
    }


def _full_transaction_fixed_guard_shadows(
    model,
    batch,
    cfg,
    candidate,
    identity,
    contract,
):
    """Compute authoritative differentiable hard shadows on one transaction."""
    groups = {}
    _, _, terms, _ = m._refiner_batch_objectives(
        model,
        batch,
        cfg,
        group_objectives=groups,
        prediction_override=candidate,
        identity_override=identity,
    )
    values = diagnostic._diagnostic_group_guard_values(terms, groups)
    expected = set(contract["initial_anchor"])
    if set(values) != expected:
        missing = sorted(expected - set(values))
        unexpected = sorted(set(values) - expected)
        raise RuntimeError(
            "full-transaction Guard term mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    shadows = {}
    limits = {}
    for name, value in values.items():
        limit = _fixed_guard_limit(contract, name)
        shadows[name] = (
            value
            - float(limit["absolute_limit"])
            - float(limit["numeric_tolerance"])
        )
        limits[name] = limit
    return shadows, values, limits


def _smooth_logsumexp(values, temperature):
    if values.numel() == 0:
        raise RuntimeError("cannot aggregate an empty Guard group")
    if values.numel() == 1:
        return values.reshape(-1)[0]
    temperature = float(temperature)
    flat = values.reshape(-1)
    return temperature * m.torch.logsumexp(flat / temperature, dim=0)


def _smooth_group_guard_value(
    name,
    *,
    exact_value,
    case_terms,
    batch,
    temperature,
):
    """Relax only the cross-case hard max; exact Guard remains authoritative."""
    label, suffix = name.split(".", 1)
    if suffix in FULL_GUARD_PHYSICAL_CASE_TERMS:
        group_index = m.REFINER_GROUP_LABELS.index(label)
        selected = batch["group"] == group_index
        values = case_terms[FULL_GUARD_PHYSICAL_CASE_TERMS[suffix]][selected]
        return _smooth_logsumexp(values, temperature)
    if suffix == "fixed_support":
        parts = [
            _smooth_group_guard_value(
                f"{label}.{part}",
                exact_value=exact_value,
                case_terms=case_terms,
                batch=batch,
                temperature=temperature,
            )
            for part in (
                "foot_skate_p95",
                "foot_skate_max",
                "support_drift_p95",
                "support_drift_max",
                "penetration",
            )
        ]
        return _smooth_logsumexp(m.torch.stack(parts), temperature)
    return exact_value


def _g1d_shadow_objective(
    *,
    model,
    batch,
    cfg,
    baseline,
    identity,
    candidate,
    contract,
    local_case,
    baseline_case,
    local_tangent,
    local_mask,
    train_repair_contract,
    temporal_smoothness_weight,
):
    hard_shadows, hard_values, limits = (
        _full_transaction_fixed_guard_shadows(
            model, batch, cfg, candidate, identity, contract
        )
    )
    hard_float = {
        name: float(value.detach()) for name, value in hard_shadows.items()
    }
    maximum_hard = max(hard_float.values())
    active_names = sorted(
        name for name, value in hard_float.items() if value > 0.0
    )
    _, case_terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    science_terms = oracle.case_probe._case_terms(candidate, batch, cfg)
    science_tensors = {
        key: science_terms[key][int(local_case)]
        for key in ("endpoint", "temporal")
    }
    scientific = oracle._case_scientific_status(
        {key: float(value.detach()) for key, value in science_tensors.items()},
        baseline_case,
    )
    smooth_shadows = {}
    for name in active_names:
        temperature = float(
            train_repair_contract[
                "lse_temperature_by_guard_term"
            ][name]
        )
        value = _smooth_group_guard_value(
            name,
            exact_value=hard_values[name],
            case_terms=case_terms,
            batch=batch,
            temperature=temperature,
        )
        smooth_shadows[name] = (
            value
            - float(limits[name]["absolute_limit"])
            - float(limits[name]["numeric_tolerance"])
        )
    smoothness = _tangent_temporal_smoothness(local_tangent, local_mask)
    if active_names:
        primary = "full_transaction_fixed_guard_shadow"
        primary_loss = m.torch.stack(
            [m.torch.relu(smooth_shadows[name]) for name in active_names]
        ).sum()
    else:
        primary = "endpoint_temporal"
        primary_loss = (
            case_terms["endpoint_scientific_deficit"][int(local_case)]
            + case_terms["temporal_scientific_deficit"][int(local_case)]
        )
    loss = primary_loss + float(temporal_smoothness_weight) * smoothness
    diagnostics = {
        "primary_objective": primary,
        "active_full_shadow_terms": active_names,
        "full_transaction_fixed_guard_shadow_margin_by_term": hard_float,
        "maximum_full_transaction_fixed_guard_shadow_margin": maximum_hard,
        "smooth_full_shadow_margin_by_active_term": {
            name: float(value.detach())
            for name, value in smooth_shadows.items()
        },
        "p95_active_index_frozen_within_line_search": True,
        "endpoint_delta": scientific["endpoint_delta"],
        "temporal_delta": scientific["temporal_delta"],
        "scientific_passed": bool(scientific["passed"]),
        "scientific_numeric_tolerance": scientific.get(
            "numeric_tolerance", {}
        ),
        "temporal_smoothness": float(smoothness.detach()),
    }
    return loss, primary_loss, diagnostics, science_tensors


def _case_isolated_transaction_tangent(baseline, scoped, local_case):
    """Place one 75D product tangent into its complete transaction."""
    transaction_tangent = scoped.new_zeros(
        (int(baseline.shape[0]), *scoped.shape[1:])
    )
    transaction_tangent[int(local_case):int(local_case) + 1] = scoped
    return transaction_tangent


def _g1c_candidate_evidence(
    *,
    model,
    sample,
    tangent,
    domains,
    ownership,
    baseline_terms,
    cfg,
    target_rms,
    workspace_floor,
):
    """Audit one generated candidate in its complete frozen transaction."""
    transaction_id = str(sample["transaction_id"])
    domain = domains[transaction_id]
    global_case = int(sample["case_index"])
    local_case = int(sample["local_case_index"])
    local = tangent[global_case:global_case + 1]
    mask = ownership[global_case:global_case + 1].expand_as(local)
    scoped = local.masked_fill(~mask, 0.0)
    transaction_tangent = _case_isolated_transaction_tangent(
        domain["baseline"], scoped, local_case
    )
    candidate = product_exp_torch(
        domain["baseline"], transaction_tangent
    )
    baseline_case = {
        key: float(baseline_terms[key][global_case].detach())
        for key in ("endpoint", "temporal")
    }
    raw = oracle._exact_audit(
        model,
        domain["batch"],
        cfg,
        domain["baseline"],
        domain["identity"],
        domain["baseline_guard"],
        domain["contract"]["initial_anchor"],
        domain["contract"]["relative_tolerance"],
        domain["contract"]["absolute_tolerance"],
        candidate,
        local_case,
        baseline_case,
        target_rms,
        workspace_floor,
    )
    case_candidate = candidate[local_case:local_case + 1]
    case_baseline = domain["baseline"][local_case:local_case + 1]
    case_batch = adapter._slice_batch(
        domain["batch"], local_case, local_case + 1
    )
    with m.torch.no_grad():
        physical = _guard_proxy_values(
            case_batch,
            cfg,
            case_baseline,
            case_candidate,
            0,
        )
    case_physical_margins = {
        name: float(value.detach()) for name, value in physical.items()
    }
    fixed_guard_shadow, fixed_guard_shadow_audit = (
        _fixed_guard_shadow_from_details(
            raw["fixed_guard_details"],
            domain["contract"]["relative_tolerance"],
            domain["contract"]["absolute_tolerance"],
        )
    )
    shadow_passed = bool(
        fixed_guard_shadow
        and all(value <= 0.0 for value in fixed_guard_shadow.values())
    )
    shadow_consistent = bool(
        shadow_passed == bool(raw["fixed_guard_passed"])
        and all(
            row["absolute_limit_matches_exact_guard"]
            for row in fixed_guard_shadow_audit.values()
        )
    )
    maximum_shadow = max(fixed_guard_shadow.values(), default=math.inf)
    return {
        "case_physical_signed_margin_by_term": case_physical_margins,
        "maximum_positive_case_physical_signed_margin": max(
            0.0,
            max(case_physical_margins.values(), default=math.inf),
        ),
        "full_transaction_fixed_guard_shadow_margin_by_term": (
            fixed_guard_shadow
        ),
        "full_transaction_fixed_guard_shadow_detail_by_term": (
            fixed_guard_shadow_audit
        ),
        "maximum_full_transaction_fixed_guard_shadow_margin": (
            maximum_shadow
        ),
        "maximum_positive_full_transaction_fixed_guard_shadow_margin": max(
            0.0, maximum_shadow
        ),
        "fixed_guard_shadow_passed": shadow_passed,
        "fixed_guard_shadow_matches_exact_guard": shadow_consistent,
        "exact_raw_closure_passed": bool(raw["passed"]),
        "fixed_guard_passed": bool(raw["fixed_guard_passed"]),
        "fixed_guard_blockers": list(raw["fixed_guard_blockers"]),
        "case_scientific": dict(raw["case_scientific"]),
        "radius_rms": float(raw["achieved_case_output_tangent_rms"]),
        "radius_equality_resolved": bool(raw["radius_equality_resolved"]),
        "workspace_observable_resolved": bool(
            raw["workspace_observable_resolved"]
        ),
        "scope_safe": bool(raw["scope"]["scope_safe"]),
        "outside_scope_abs_max": float(
            raw["scope"]["outside_case_group_or_ownership_abs_max"]
        ),
    }


def _guard_aligned_correction_objective(
    batch,
    cfg,
    baseline,
    candidate,
    case_index,
    anchor_proxy,
    tolerance,
    scale_floor,
):
    _, terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    index = int(case_index)
    scientific = {
        "endpoint_scientific_deficit": terms[
            "endpoint_scientific_deficit"
        ][index],
        "temporal_scientific_deficit": terms[
            "temporal_scientific_deficit"
        ][index],
    }
    candidate_proxy = {
        name: terms[key][index]
        for name, key in G1_GUARD_PROXY_TERMS.items()
    }
    proxy_delta = {
        name: candidate_proxy[name] - anchor_proxy[name]
        for name in G1_GUARD_PROXY_TERMS
    }
    proxy_excess = {
        name: m.torch.relu(delta - float(tolerance))
        / anchor_proxy[name].abs().clamp_min(float(scale_floor))
        for name, delta in proxy_delta.items()
    }
    loss = m.torch.stack(list(scientific.values())).sum()
    loss = loss + m.torch.stack(list(proxy_excess.values())).square().sum()
    diagnostics = {
        **{
            key: float(value.detach())
            for key, value in scientific.items()
        },
        "guard_proxy_anchor": {
            key: float(value.detach()) for key, value in anchor_proxy.items()
        },
        "guard_proxy_candidate": {
            key: float(value.detach())
            for key, value in candidate_proxy.items()
        },
        "guard_proxy_delta": {
            key: float(value.detach()) for key, value in proxy_delta.items()
        },
        "guard_proxy_normalized_excess": {
            key: float(value.detach())
            for key, value in proxy_excess.items()
        },
    }
    return loss, diagnostics


def _tangent_temporal_smoothness(tangent, mask):
    """Penalize local high-frequency correction without changing the Guard."""
    scoped = tangent.masked_fill(~mask, 0.0)
    frame_active = mask.any(dim=-1)
    penalties = []
    if scoped.shape[1] >= 3:
        second = scoped[:, 2:] - 2.0 * scoped[:, 1:-1] + scoped[:, :-2]
        valid = (
            frame_active[:, 2:]
            & frame_active[:, 1:-1]
            & frame_active[:, :-2]
        )
        if bool(valid.any()):
            penalties.append(second[valid].square().mean())
    if scoped.shape[1] >= 4:
        third = (
            scoped[:, 3:]
            - 3.0 * scoped[:, 2:-1]
            + 3.0 * scoped[:, 1:-2]
            - scoped[:, :-3]
        )
        valid = (
            frame_active[:, 3:]
            & frame_active[:, 2:-1]
            & frame_active[:, 1:-2]
            & frame_active[:, :-3]
        )
        if bool(valid.any()):
            penalties.append(third[valid].square().mean())
    if not penalties:
        return scoped.square().sum() * 0.0
    return m.torch.stack(penalties).sum()


def _contract_margin_active_set_objective(
    batch,
    cfg,
    baseline,
    candidate,
    case_index,
    baseline_case,
    tangent,
    mask,
    tolerance,
    smooth_max_temperature,
    temporal_smoothness_weight,
):
    """Use absolute contract margins, with Guard-first lexicographic repair."""
    _, terms = m._observable_refiner_objective(
        candidate,
        baseline.detach(),
        batch["seam"],
        cfg,
        reduction="none",
    )
    index = int(case_index)
    scientific_deficits = {
        "endpoint_scientific_deficit": terms[
            "endpoint_scientific_deficit"
        ][index],
        "temporal_scientific_deficit": terms[
            "temporal_scientific_deficit"
        ][index],
    }
    candidate_proxy = {
        name: terms[key][index]
        for name, key in G1_GUARD_PROXY_TERMS.items()
    }
    proxy_vector = m.torch.stack(list(candidate_proxy.values()))
    positive = m.torch.relu(proxy_vector - float(tolerance))
    maximum_positive = m.torch.relu(proxy_vector).amax()
    guard_active = bool(float(positive.amax().detach()) > 0.0)
    temperature = float(smooth_max_temperature)
    smooth_guard_violation = temperature * m.torch.logsumexp(
        positive / temperature,
        dim=0,
    )
    case_terms = oracle.case_probe._case_terms(candidate, batch, cfg)
    candidate_case_tensors = {
        key: case_terms[key][index] for key in ("endpoint", "temporal")
    }
    candidate_case = {
        key: float(value.detach())
        for key, value in candidate_case_tensors.items()
    }
    scientific = oracle._case_scientific_status(
        candidate_case,
        baseline_case,
    )
    smoothness = _tangent_temporal_smoothness(tangent, mask)
    if guard_active:
        loss = (
            smooth_guard_violation
            + float(temporal_smoothness_weight) * smoothness
        )
        primary_objective = "guard_violation"
    else:
        loss = m.torch.stack(list(scientific_deficits.values())).sum()
        primary_objective = "endpoint_temporal"
    diagnostics = {
        "primary_objective": primary_objective,
        "candidate_guard_signed_margin_by_term": {
            key: float(value.detach())
            for key, value in candidate_proxy.items()
        },
        "maximum_positive_guard_signed_margin": float(
            maximum_positive.detach()
        ),
        "smooth_guard_violation": float(smooth_guard_violation.detach()),
        "temporal_smoothness": float(smoothness.detach()),
        "endpoint_delta": scientific["endpoint_delta"],
        "temporal_delta": scientific["temporal_delta"],
        "scientific_passed": bool(scientific["passed"]),
        "scientific_numeric_tolerance": scientific.get(
            "numeric_tolerance", {}
        ),
    }
    return loss, diagnostics, candidate_case_tensors


def _project_guard_direction(
    direction,
    *,
    current,
    mask,
    feasibility_gradients,
    damping=0.0,
):
    """Project a Guard descent direction onto radius/science tangent cones."""
    result = direction.masked_fill(~mask, 0.0)
    radial = current.detach().masked_fill(~mask, 0.0)
    constraints = [
        gradient.detach().masked_fill(~mask, 0.0)
        for gradient in feasibility_gradients
        if gradient is not None
    ]
    # Two alternating passes keep the radial and feasibility projections
    # consistent without changing the fixed 2/3/5 correction budget.
    for _ in range(2):
        radial_norm = (
            radial[mask].square().sum() + float(damping)
        ).clamp_min(1.0e-20)
        radial_dot = (result[mask] * radial[mask]).sum()
        result = result - (radial_dot / radial_norm) * radial
        for gradient in constraints:
            directional_change = (result[mask] * gradient[mask]).sum()
            if float(directional_change.detach()) > 0.0:
                denominator = (
                    gradient[mask].square().sum() + float(damping)
                ).clamp_min(1.0e-20)
                result = result - (
                    directional_change / denominator
                ) * gradient
    return result.masked_fill(~mask, 0.0)


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
    model,
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
    activation_aware=False,
    guard_aligned=False,
    contract_margin_active_set=False,
    guard_proxy_tolerance=1.0e-6,
    guard_proxy_scale_floor=1.0e-6,
    guard_smooth_max_temperature=1.0e-3,
    correction_temporal_smoothness_weight=0.05,
    guard_minimum_reduction=1.0e-12,
):
    mask = _owned_case_mask(ownership, initial_tangent, case_index)
    current, normalized = _normalize_exact_radius(
        initial_tangent, mask, target_rms
    )
    history = []
    zero_start = bool(not normalized and activation_aware)
    numeric_failure = bool(not normalized and not activation_aware)
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
    if zero_start:
        current = m.torch.zeros_like(initial_tangent).masked_fill(
            ~mask, 0.0
        )
    anchor_proxy = None
    if guard_aligned:
        anchor_proxy = _guard_proxy_values(
            batch,
            cfg,
            baseline,
            baseline,
            case_index,
        )

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

        science_tensors = None
        if contract_margin_active_set:
            total_tangent = product_log_torch(baseline, candidate)
            loss, constraints, science_tensors = (
                _contract_margin_active_set_objective(
                    batch,
                    cfg,
                    baseline,
                    candidate,
                    case_index,
                    baseline_case,
                    total_tangent,
                    mask,
                    guard_proxy_tolerance,
                    guard_smooth_max_temperature,
                    correction_temporal_smoothness_weight,
                )
            )
        elif guard_aligned:
            loss, constraints = _guard_aligned_correction_objective(
                batch,
                cfg,
                baseline,
                candidate,
                case_index,
                anchor_proxy,
                guard_proxy_tolerance,
                guard_proxy_scale_floor,
            )
        elif activation_aware:
            loss, _, constraints = _label_free_activation_objective(
                batch,
                cfg,
                baseline,
                candidate,
                case_index,
            )
        else:
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
            loss,
            variable,
            allow_unused=True,
            retain_graph=bool(
                contract_margin_active_set
                and constraints["primary_objective"] == "guard_violation"
            ),
        )[0]
        if gradient is None or not bool(m.torch.isfinite(gradient).all()):
            numeric_failure = True
            break
        raw_direction = -gradient.detach()
        if (
            contract_margin_active_set
            and constraints["primary_objective"] == "guard_violation"
        ):
            feasibility_gradients = []
            for position, key in enumerate(("endpoint", "temporal")):
                feasibility_gradients.append(m.torch.autograd.grad(
                    science_tensors[key],
                    variable,
                    allow_unused=True,
                    retain_graph=position == 0,
                )[0])
            raw_direction = _project_guard_direction(
                raw_direction,
                current=current,
                mask=mask,
                feasibility_gradients=feasibility_gradients,
            )
        direction, usable = _bounded_direction(
            raw_direction,
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
        accepted_constraints = constraints
        accepted_guard_margin_reduction = 0.0
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
            trial_outside_scope_abs_max = float(
                trial.masked_fill(mask, 0.0).abs().amax().detach()
            )
            if contract_margin_active_set:
                trial_loss, trial_constraints, _ = (
                    _contract_margin_active_set_objective(
                        batch,
                        cfg,
                        baseline,
                        trial_candidate,
                        case_index,
                        baseline_case,
                        trial,
                        mask,
                        guard_proxy_tolerance,
                        guard_smooth_max_temperature,
                        correction_temporal_smoothness_weight,
                    )
                )
            elif guard_aligned:
                trial_loss, _ = _guard_aligned_correction_objective(
                    batch,
                    cfg,
                    baseline,
                    trial_candidate,
                    case_index,
                    anchor_proxy,
                    guard_proxy_tolerance,
                    guard_proxy_scale_floor,
                )
            elif activation_aware:
                trial_loss, _, _ = _label_free_activation_objective(
                    batch,
                    cfg,
                    baseline,
                    trial_candidate,
                    case_index,
                )
            else:
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
            if contract_margin_active_set:
                current_margin = float(
                    constraints["maximum_positive_guard_signed_margin"]
                )
                trial_margin = float(
                    trial_constraints[
                        "maximum_positive_guard_signed_margin"
                    ]
                )
                if constraints["primary_objective"] == "guard_violation":
                    guard_improved = bool(
                        trial_margin
                        <= current_margin - float(guard_minimum_reduction)
                    )
                    accept_trial = bool(
                        bool(m.torch.isfinite(trial_loss))
                        and trial_constraints["scientific_passed"]
                        and guard_improved
                        and trial_outside_scope_abs_max == 0.0
                    )
                else:
                    guard_improved = bool(
                        trial_margin <= float(guard_proxy_tolerance)
                    )
                    accept_trial = bool(
                        bool(m.torch.isfinite(trial_loss))
                        and trial_constraints["scientific_passed"]
                        and guard_improved
                        and trial_outside_scope_abs_max == 0.0
                        and float(trial_loss.detach())
                        <= float(loss.detach()) + 1.0e-12
                    )
            else:
                accept_trial = bool(
                    bool(m.torch.isfinite(trial_loss))
                    and float(trial_loss.detach())
                    <= float(loss.detach()) + 1.0e-12
                )
            if accept_trial:
                accepted = True
                accepted_loss = float(trial_loss.detach())
                accepted_scale = alpha
                accepted_tangent = trial.detach()
                if contract_margin_active_set:
                    accepted_constraints = trial_constraints
                    accepted_guard_margin_reduction = (
                        current_margin - trial_margin
                    )
                break
        history.append({
            "iteration": iteration,
            "loss_before": float(loss.detach()),
            "loss_after": accepted_loss,
            "accepted": accepted,
            "step_scale": accepted_scale,
            "constraints": constraints,
            "accepted_constraints": accepted_constraints,
            "accepted_guard_margin_reduction": (
                accepted_guard_margin_reduction
            ),
            "outside_scope_abs_max": (
                0.0 if accepted else None
            ),
            "exact_radius_rms": _rms(accepted_tangent, mask),
        })
        if not accepted:
            break
        current = accepted_tangent

    accepted_steps = sum(int(row["accepted"]) for row in history)
    total_guard_margin_reduction = sum(
        float(row.get("accepted_guard_margin_reduction", 0.0))
        for row in history
    )
    return current.detach(), {
        "method": method,
        "steps_requested": int(steps),
        "steps_completed": len(history),
        "accepted_steps": accepted_steps,
        "correction_accepted_steps": accepted_steps,
        "accepted_guard_margin_reduction": total_guard_margin_reduction,
        "numeric_failure": numeric_failure,
        "zero_adapter_start": zero_start,
        "role_or_group_consumed_by_correction": bool(
            not activation_aware and not guard_aligned
        ),
        "guard_aligned_correction": bool(guard_aligned),
        "contract_margin_active_set": bool(contract_margin_active_set),
        "scientific_feasibility_projection": bool(
            contract_margin_active_set
        ),
        "tangent_second_third_difference_regularization": bool(
            contract_margin_active_set
        ),
        "final_exact_radius_rms": _rms(current, mask),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _g1d_rejection_reason(
    *,
    radius_ok,
    scope_ok,
    science_ok,
    shadow_ok,
):
    if not radius_ok:
        return "exact_radius_normalization_failed"
    if not scope_ok:
        return "scope_leakage"
    if not science_ok:
        return "radius_normalization_broke_science_cone"
    if not shadow_ok:
        return "full_shadow_not_strictly_reduced"
    return None


def _correct_case_full_transaction_shadow(
    *,
    model,
    method,
    steps,
    initial_tangent,
    ownership,
    c2_taper,
    baseline,
    identity,
    batch,
    cfg,
    local_case,
    baseline_case,
    contract,
    train_repair_contract,
    target_rms,
    step_size,
    trust_fraction,
    temporal_smoothness_weight,
):
    """Repair one local tangent using its complete transaction Guard shadow."""
    mask = _owned_case_mask(ownership, initial_tangent, 0)
    taper = c2_taper.expand_as(initial_tangent).detach()
    current, normalized = _normalize_exact_radius(
        initial_tangent, mask, target_rms
    )
    zero_start = not normalized
    if zero_start:
        current = m.torch.zeros_like(initial_tangent).masked_fill(~mask, 0.0)
    history = []
    numeric_failure = False
    started = time.perf_counter()
    damping = float(train_repair_contract["projection_damping"])

    for iteration in range(int(steps)):
        local_baseline = baseline[int(local_case):int(local_case) + 1]
        if method == "euclidean_projected":
            variable = m.torch.zeros_like(current).requires_grad_(True)
            local_total = current.detach() + variable.masked_fill(
                ~mask, 0.0
            ) * taper
            local_candidate = product_exp_torch(
                local_baseline, local_total
            )
        elif method == "riemannian_retraction":
            current_motion = product_exp_torch(local_baseline, current.detach())
            variable = m.torch.zeros_like(current).requires_grad_(True)
            local_candidate = product_exp_torch(
                current_motion,
                variable.masked_fill(~mask, 0.0) * taper,
            )
        else:
            raise ValueError(f"unsupported correction method: {method}")
        local_total = product_log_torch(local_baseline, local_candidate)
        local_total = local_total.masked_fill(~mask, 0.0)
        transaction_tangent = _case_isolated_transaction_tangent(
            baseline, local_total, local_case
        )
        candidate = product_exp_torch(baseline, transaction_tangent)
        loss, primary_loss, constraints, science_tensors = (
            _g1d_shadow_objective(
                model=model,
                batch=batch,
                cfg=cfg,
                baseline=baseline,
                identity=identity,
                candidate=candidate,
                contract=contract,
                local_case=local_case,
                baseline_case=baseline_case,
                local_tangent=local_total,
                local_mask=mask,
                train_repair_contract=train_repair_contract,
                temporal_smoothness_weight=temporal_smoothness_weight,
            )
        )
        primary_gradient = m.torch.autograd.grad(
            primary_loss, variable, allow_unused=True, retain_graph=True
        )[0]
        gradient = m.torch.autograd.grad(
            loss, variable, allow_unused=True, retain_graph=True
        )[0]
        if (
            gradient is None
            or not bool(m.torch.isfinite(gradient).all())
            or primary_gradient is None
            or not bool(m.torch.isfinite(primary_gradient).all())
        ):
            numeric_failure = True
            history.append({
                "iteration": iteration,
                "accepted": False,
                "step_rejection_reason": "shadow_gradient_zero_or_nonfinite",
                "constraints": constraints,
                "full_shadow_gradient_norm": math.nan,
                "line_search_rejections": [],
            })
            break
        raw_direction = (-gradient.detach()).masked_fill(~mask, 0.0)
        full_shadow_gradient_norm = float(
            m.torch.linalg.vector_norm(
                primary_gradient.detach().masked_fill(~mask, 0.0)[mask]
            ).detach()
        )
        if (
            constraints["primary_objective"]
            == "full_transaction_fixed_guard_shadow"
            and full_shadow_gradient_norm <= 1.0e-20
        ):
            history.append({
                "iteration": iteration,
                "loss_before": float(loss.detach()),
                "accepted": False,
                "step_rejection_reason": "shadow_gradient_zero_or_nonfinite",
                "constraints": constraints,
                "full_shadow_gradient_norm": full_shadow_gradient_norm,
                "line_search_rejections": [],
            })
            break
        if constraints["primary_objective"] == (
            "full_transaction_fixed_guard_shadow"
        ):
            feasibility_gradients = [
                m.torch.autograd.grad(
                    science_tensors[key],
                    variable,
                    allow_unused=True,
                    retain_graph=position == 0,
                )[0]
                for position, key in enumerate(("endpoint", "temporal"))
            ]
            raw_direction = _project_guard_direction(
                raw_direction,
                current=current,
                mask=mask,
                feasibility_gradients=feasibility_gradients,
                damping=damping,
            )
            raw_direction = raw_direction.masked_fill(~mask, 0.0)
        direction, usable = _bounded_direction(
            raw_direction,
            mask,
            float(target_rms) * float(trust_fraction),
        )
        if not usable:
            history.append({
                "iteration": iteration,
                "loss_before": float(loss.detach()),
                "accepted": False,
                "step_rejection_reason": "shadow_gradient_zero_or_nonfinite",
                "constraints": constraints,
                "full_shadow_gradient_norm": full_shadow_gradient_norm,
                "line_search_rejections": [],
            })
            break

        current_shadow = float(
            constraints[
                "maximum_full_transaction_fixed_guard_shadow_margin"
            ]
        )
        active_names = constraints["active_full_shadow_terms"]
        minimum_reduction = max(
            [
                float(train_repair_contract[
                    "minimum_shadow_reduction_by_guard_term"
                ][name])
                for name in active_names
            ]
            or [float(min(
                train_repair_contract[
                    "minimum_shadow_reduction_by_guard_term"
                ].values()
            ))]
        )
        accepted = False
        accepted_tangent = current
        accepted_scale = 0.0
        accepted_shadow = current_shadow
        rejection_rows = []
        for backtrack in range(6):
            alpha = float(step_size) * (0.5 ** backtrack)
            if method == "euclidean_projected":
                trial_raw = current + alpha * direction
            else:
                trial_motion = product_exp_torch(
                    current_motion, alpha * direction
                )
                trial_raw = product_log_torch(local_baseline, trial_motion)
            trial, trial_ok = _normalize_exact_radius(
                trial_raw, mask, target_rms
            )
            if not trial_ok:
                rejection_rows.append({
                    "backtrack": backtrack,
                    "step_scale": alpha,
                    "reason": "exact_radius_normalization_failed",
                })
                continue
            radius_rms = _rms(trial, mask)
            radius_ok = bool(
                math.isfinite(radius_rms)
                and abs(radius_rms - float(target_rms))
                <= max(1.0e-12, float(target_rms) * 1.0e-6)
            )
            outside_scope = float(
                trial.masked_fill(mask, 0.0).abs().amax().detach()
            )
            scope_ok = outside_scope == 0.0
            trial_transaction_tangent = _case_isolated_transaction_tangent(
                baseline, trial, local_case
            )
            trial_candidate = product_exp_torch(
                baseline, trial_transaction_tangent
            )
            with m.torch.no_grad():
                trial_shadows, _, _ = _full_transaction_fixed_guard_shadows(
                    model,
                    batch,
                    cfg,
                    trial_candidate,
                    identity,
                    contract,
                )
                trial_shadow = max(
                    float(value.detach()) for value in trial_shadows.values()
                )
                trial_case_terms = oracle.case_probe._case_terms(
                    trial_candidate, batch, cfg
                )
            trial_scientific = oracle._case_scientific_status(
                {
                    key: float(
                        trial_case_terms[key][int(local_case)].detach()
                    )
                    for key in ("endpoint", "temporal")
                },
                baseline_case,
            )
            science_ok = bool(trial_scientific["passed"])
            shadow_ok = bool(
                trial_shadow <= current_shadow - minimum_reduction
            )
            rejection = _g1d_rejection_reason(
                radius_ok=radius_ok,
                scope_ok=scope_ok,
                science_ok=science_ok,
                shadow_ok=shadow_ok,
            )
            if rejection is None:
                accepted = True
                accepted_tangent = trial.detach()
                accepted_scale = alpha
                accepted_shadow = trial_shadow
                break
            rejection_rows.append({
                "backtrack": backtrack,
                "step_scale": alpha,
                "reason": rejection,
                "trial_full_shadow": trial_shadow,
                "trial_radius_rms": radius_rms,
                "trial_outside_scope_abs_max": outside_scope,
                "trial_scientific_passed": science_ok,
            })
        step_rejection_reason = None
        if not accepted:
            step_rejection_reason = (
                rejection_rows[-1]["reason"]
                if rejection_rows
                else "line_search_step_reduced_to_zero"
            )
        reduction = current_shadow - accepted_shadow if accepted else 0.0
        history.append({
            "iteration": iteration,
            "loss_before": float(loss.detach()),
            "accepted": accepted,
            "step_scale": accepted_scale,
            "constraints": constraints,
            "full_shadow_gradient_norm": full_shadow_gradient_norm,
            "full_shadow_before": current_shadow,
            "full_shadow_after": accepted_shadow,
            "full_shadow_reduction": reduction,
            "step_rejection_reason": step_rejection_reason,
            "line_search_exhausted": bool(not accepted),
            "line_search_terminal_reason": (
                "line_search_step_reduced_to_zero" if not accepted else None
            ),
            "line_search_rejections": rejection_rows,
            "exact_radius_rms": _rms(accepted_tangent, mask),
            "outside_scope_abs_max": 0.0 if accepted else None,
        })
        if not accepted:
            break
        current = accepted_tangent

    accepted_steps = sum(int(row.get("accepted", False)) for row in history)
    reductions = [float(row.get("full_shadow_reduction", 0.0)) for row in history]
    gradient_norms = [
        float(row["full_shadow_gradient_norm"])
        for row in history
        if math.isfinite(float(row.get("full_shadow_gradient_norm", math.nan)))
    ]
    final_rejection = next(
        (
            row["step_rejection_reason"]
            for row in reversed(history)
            if row.get("step_rejection_reason")
        ),
        None,
    )
    return current.detach(), {
        "method": method,
        "steps_requested": int(steps),
        "steps_completed": len(history),
        "accepted_steps": accepted_steps,
        "correction_accepted_steps": accepted_steps,
        "numeric_failure": numeric_failure,
        "zero_adapter_start": zero_start,
        "full_transaction_shadow_gradient_repair": True,
        "hard_shadow_authoritative_for_active_set_and_acceptance": True,
        "smooth_shadow_used_for_gradient_only": True,
        "gradient_scope": "current_case_c2_tapered_ownership_only",
        "c2_taper_in_autograd_forward_parameterization": True,
        "scientific_feasibility_projection": True,
        "projection_damping": damping,
        "full_shadow_gradient_norm": max(gradient_norms, default=0.0),
        "full_shadow_reduction": sum(reductions),
        "step_rejection_reason": final_rejection,
        "final_exact_radius_rms": _rms(current, mask),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _g1c_name_case_physical_diagnostics(report):
    """Remove ambiguous Guard names from g1c per-case proxy diagnostics."""
    replacements = {
        "candidate_guard_signed_margin_by_term": (
            "candidate_case_physical_signed_margin_by_term"
        ),
        "maximum_positive_guard_signed_margin": (
            "maximum_positive_case_physical_signed_margin"
        ),
        "accepted_guard_margin_reduction": (
            "accepted_case_physical_margin_reduction"
        ),
    }

    def visit(value):
        if isinstance(value, dict):
            renamed = {}
            for key, item in value.items():
                renamed[replacements.get(key, key)] = visit(item)
            return renamed
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(report)


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


def _normalize_all_sample_tangents(
    tangent,
    ownership,
    samples,
    target_rms,
):
    """Normalize every enumerated case without consulting its role label."""
    result = m.torch.zeros_like(tangent)
    diagnostics = {}
    visited = set()
    for sample in samples:
        case_index = int(sample["case_index"])
        if case_index in visited:
            raise RuntimeError(f"duplicate activation case {case_index}")
        visited.add(case_index)
        mask = _owned_case_mask(ownership, tangent, case_index)
        local, resolved = _normalize_exact_radius(
            tangent, mask, target_rms
        )
        result = result + local
        uid = str(sample.get("case_uid", case_index))
        diagnostics[uid] = {
            "input_rms": _rms(tangent, mask),
            "normalized_rms": _rms(local, mask),
            "radius_equality_resolved": bool(resolved),
        }
    return result, diagnostics


def _sample_problem(sample, domains, ownership):
    transaction_id = str(sample["transaction_id"])
    domain = domains[transaction_id]
    local_case = int(sample["local_case_index"])
    global_case = int(sample["case_index"])
    case_batch = adapter._slice_batch(
        domain["batch"], local_case, local_case + 1
    )
    return {
        "transaction_id": transaction_id,
        "local_case_index": local_case,
        "global_case_index": global_case,
        "batch": case_batch,
        "baseline": domain["baseline"][local_case:local_case + 1],
        "identity": domain["identity"][local_case:local_case + 1],
        "ownership": ownership[global_case:global_case + 1],
    }


def _activation_aware_selection(
    *,
    variants,
    correction_reports,
    samples,
    domains,
    ownership,
    baseline_terms,
    cfg,
    target_rms,
    proxy_tolerance,
    relative_improvement,
):
    """Select identity or repair using observables only, then attach labels."""
    selected = m.torch.zeros_like(next(iter(variants.values())))
    decisions = {}
    method_counts = Counter()
    for sample in samples:
        uid = str(sample.get("case_uid", sample["case_index"]))
        problem = _sample_problem(
            sample,
            domains,
            ownership,
        )
        case_index = problem["global_case_index"]
        baseline_case = {
            key: float(baseline_terms[key][case_index].detach())
            for key in ("endpoint", "temporal")
        }
        identity_loss, identity_physical, identity_components = (
            _label_free_activation_objective(
                problem["batch"],
                cfg,
                problem["baseline"],
                problem["baseline"],
                0,
            )
        )
        identity_value = float(identity_loss.detach())
        required_reduction = max(
            abs(identity_value) * float(relative_improvement),
            1.0e-8,
        )
        candidates = {}
        eligible = []
        for method, tangent in variants.items():
            local = tangent[case_index:case_index + 1]
            mask = problem["ownership"].expand_as(local)
            scoped = local.masked_fill(~mask, 0.0)
            active_rms = _rms(scoped, mask)
            radius_resolved = bool(
                math.isfinite(active_rms)
                and abs(active_rms - float(target_rms))
                <= max(1.0e-12, float(target_rms) * 1.0e-6)
            )
            candidate = product_exp_torch(problem["baseline"], scoped)
            loss, physical, components = _label_free_activation_objective(
                problem["batch"],
                cfg,
                problem["baseline"],
                candidate,
                0,
            )
            case_terms = oracle.case_probe._case_terms(
                candidate,
                problem["batch"],
                cfg,
            )
            candidate_case = {
                key: float(case_terms[key][0].detach())
                for key in ("endpoint", "temporal")
            }
            scientific = oracle._case_scientific_status(
                candidate_case, baseline_case
            )
            numeric_failure = bool(
                correction_reports.get(method, {}).get(uid, {}).get(
                    "numeric_failure", False
                )
            )
            loss_value = float(loss.detach())
            physical_value = float(physical.detach())
            observable_improvement = bool(
                math.isfinite(loss_value)
                and loss_value <= identity_value - required_reduction
            )
            proxy_safe = bool(
                math.isfinite(physical_value)
                and physical_value <= float(proxy_tolerance)
            )
            accepted = bool(
                radius_resolved
                and scientific["passed"]
                and observable_improvement
                and proxy_safe
                and not numeric_failure
            )
            candidates[method] = {
                "objective": loss_value,
                "identity_objective": identity_value,
                "required_objective_reduction": required_reduction,
                "physical_proxy_excess": physical_value,
                "proxy_components": components,
                "endpoint_delta": scientific["endpoint_delta"],
                "temporal_delta": scientific["temporal_delta"],
                "scientific_passed": bool(scientific["passed"]),
                "radius_rms": active_rms,
                "radius_equality_resolved": radius_resolved,
                "numeric_failure": numeric_failure,
                "eligible": accepted,
                "eligible_after_adapter_incumbent": accepted,
            }
            if accepted:
                eligible.append((loss_value, method, scoped))
        if eligible:
            _, selected_method, selected_tangent = min(
                eligible, key=lambda row: (row[0], row[1])
            )
            selected[case_index:case_index + 1] = selected_tangent
        else:
            selected_method = "identity"
        method_counts.update([selected_method])
        decisions[uid] = {
            "selected_method": selected_method,
            "selected_nonzero": bool(selected_method != "identity"),
            "selection": {
                "identity_objective": identity_value,
                "identity_physical_proxy_excess": float(
                    identity_physical.detach()
                ),
                "identity_proxy_components": identity_components,
                "candidates": candidates,
            },
            "evaluation_only": {
                "teacher_kind": sample.get("teacher_kind"),
                "audit_group": sample.get("audit_group"),
            },
        }
    return selected, decisions, dict(method_counts)


def _g1_observable_guard_selection(
    *,
    variants,
    correction_reports,
    samples,
    domains,
    ownership,
    baseline_terms,
    cfg,
    target_rms,
    severity_envelope,
    severity_scale_floor,
    guard_proxy_tolerance,
    guard_proxy_scale_floor,
):
    """Select a repair from severity and differentiable Guard proxies only."""
    selected = m.torch.zeros_like(next(iter(variants.values())))
    decisions = {}
    method_counts = Counter()
    for sample in samples:
        uid = str(sample.get("case_uid", sample["case_index"]))
        problem = _sample_problem(sample, domains, ownership)
        case_index = problem["global_case_index"]
        severity = _severity_status(
            sample,
            severity_envelope,
            severity_scale_floor,
        )
        baseline_case = {
            key: float(baseline_terms[key][case_index].detach())
            for key in ("endpoint", "temporal")
        }
        anchor_proxy = _guard_proxy_values(
            problem["batch"],
            cfg,
            problem["baseline"],
            problem["baseline"],
            0,
        )
        candidates = {}
        eligible = []
        for method, tangent in variants.items():
            local = tangent[case_index:case_index + 1]
            mask = problem["ownership"].expand_as(local)
            outside = local.masked_fill(mask, 0.0)
            outside_scope_abs_max = (
                float(outside.abs().amax().detach())
                if outside.numel() else 0.0
            )
            scoped = local.masked_fill(~mask, 0.0)
            active_rms = _rms(scoped, mask)
            radius_resolved = bool(
                math.isfinite(active_rms)
                and abs(active_rms - float(target_rms))
                <= max(1.0e-12, float(target_rms) * 1.0e-6)
            )
            candidate = product_exp_torch(problem["baseline"], scoped)
            objective, components = _guard_aligned_correction_objective(
                problem["batch"],
                cfg,
                problem["baseline"],
                candidate,
                0,
                anchor_proxy,
                guard_proxy_tolerance,
                guard_proxy_scale_floor,
            )
            candidate_proxy = _guard_proxy_values(
                problem["batch"],
                cfg,
                problem["baseline"],
                candidate,
                0,
            )
            proxy_delta = {
                name: float(
                    (candidate_proxy[name] - anchor_proxy[name]).detach()
                )
                for name in G1_GUARD_PROXY_TERMS
            }
            normalized_proxy_delta = {
                name: proxy_delta[name] / max(
                    abs(float(anchor_proxy[name].detach())),
                    float(guard_proxy_scale_floor),
                )
                for name in G1_GUARD_PROXY_TERMS
            }
            worst_proxy_delta = max(normalized_proxy_delta.values())
            proxy_nonregression = all(
                value <= float(guard_proxy_tolerance)
                for value in proxy_delta.values()
            )
            case_terms = oracle.case_probe._case_terms(
                candidate,
                problem["batch"],
                cfg,
            )
            candidate_case = {
                key: float(case_terms[key][0].detach())
                for key in ("endpoint", "temporal")
            }
            scientific = oracle._case_scientific_status(
                candidate_case, baseline_case
            )
            numeric_failure = bool(
                correction_reports.get(method, {}).get(uid, {}).get(
                    "numeric_failure", False
                )
            )
            objective_value = float(objective.detach())
            accepted = bool(
                severity["outside_frozen_single_envelope"]
                and scientific["passed"]
                and proxy_nonregression
                and radius_resolved
                and outside_scope_abs_max == 0.0
                and math.isfinite(objective_value)
                and not numeric_failure
            )
            candidates[method] = {
                "objective": objective_value,
                "severity_gate_passed": bool(
                    severity["outside_frozen_single_envelope"]
                ),
                "guard_proxy_nonregression": proxy_nonregression,
                "guard_proxy_delta": proxy_delta,
                "guard_proxy_normalized_delta": normalized_proxy_delta,
                "guard_proxy_worst_normalized_delta": worst_proxy_delta,
                "proxy_components": components,
                "endpoint_delta": scientific["endpoint_delta"],
                "temporal_delta": scientific["temporal_delta"],
                "scientific_passed": bool(scientific["passed"]),
                "radius_rms": active_rms,
                "radius_equality_resolved": radius_resolved,
                "outside_scope_abs_max": outside_scope_abs_max,
                "scope_safe": outside_scope_abs_max == 0.0,
                "numeric_failure": numeric_failure,
                "eligible": accepted,
            }
            if accepted:
                eligible.append((
                    worst_proxy_delta,
                    objective_value,
                    method,
                    scoped,
                ))
        adapter_candidate = candidates.get("adapter")
        if adapter_candidate and adapter_candidate["eligible"]:
            retained = []
            for row in eligible:
                method = row[2]
                candidate = candidates[method]
                guard_dominates_adapter = all(
                    candidate["guard_proxy_delta"][name]
                    <= adapter_candidate["guard_proxy_delta"][name]
                    + float(guard_proxy_tolerance)
                    for name in G1_GUARD_PROXY_TERMS
                )
                science_dominates_adapter = bool(
                    candidate["objective"]
                    <= adapter_candidate["objective"] + 1.0e-12
                )
                incumbent_safe = bool(
                    method == "adapter"
                    or (
                        guard_dominates_adapter
                        and science_dominates_adapter
                    )
                )
                candidate["guard_dominates_adapter"] = (
                    guard_dominates_adapter
                )
                candidate["science_dominates_adapter"] = (
                    science_dominates_adapter
                )
                candidate["eligible_after_adapter_incumbent"] = (
                    incumbent_safe
                )
                if incumbent_safe:
                    retained.append(row)
            eligible = retained
        if eligible:
            _, _, selected_method, selected_tangent = min(
                eligible,
                key=lambda row: (row[0], row[1], row[2]),
            )
            selected[case_index:case_index + 1] = selected_tangent
        else:
            selected_method = "identity"
        method_counts.update([selected_method])
        decisions[uid] = {
            "selected_method": selected_method,
            "selected_nonzero": bool(selected_method != "identity"),
            "selection": {
                "anchor_severity": severity,
                "activation_condition": (
                    "anchor_outside_frozen_train_single_envelope"
                ),
                "guard_proxy_terms": dict(G1_GUARD_PROXY_TERMS),
                "candidates": candidates,
            },
            "evaluation_only": {
                "teacher_kind": sample.get("teacher_kind"),
                "audit_group": sample.get("audit_group"),
            },
        }
    return selected, decisions, dict(method_counts)


def _g1b_contract_margin_selection(
    *,
    variants,
    correction_reports,
    samples,
    domains,
    ownership,
    baseline_terms,
    cfg,
    target_rms,
    severity_envelope,
    guard_proxy_tolerance,
    guard_safe_interior_margin,
):
    """Select with absolute Guard margins and a conformal identity fallback."""
    selected = m.torch.zeros_like(next(iter(variants.values())))
    decisions = {}
    method_counts = Counter()
    for sample in samples:
        uid = str(sample.get("case_uid", sample["case_index"]))
        problem = _sample_problem(sample, domains, ownership)
        case_index = problem["global_case_index"]
        severity = _conformal_severity_status(sample, severity_envelope)
        baseline_case = {
            key: float(baseline_terms[key][case_index].detach())
            for key in ("endpoint", "temporal")
        }
        candidates = {}
        eligible = []
        for method, tangent in variants.items():
            local = tangent[case_index:case_index + 1]
            mask = problem["ownership"].expand_as(local)
            outside = local.masked_fill(mask, 0.0)
            outside_scope_abs_max = (
                float(outside.abs().amax().detach())
                if outside.numel() else 0.0
            )
            scoped = local.masked_fill(~mask, 0.0)
            active_rms = _rms(scoped, mask)
            radius_resolved = bool(
                math.isfinite(active_rms)
                and abs(active_rms - float(target_rms))
                <= max(1.0e-12, float(target_rms) * 1.0e-6)
            )
            candidate = product_exp_torch(problem["baseline"], scoped)
            candidate_proxy = _guard_proxy_values(
                problem["batch"],
                cfg,
                problem["baseline"],
                candidate,
                0,
            )
            signed_margins = {
                name: float(value.detach())
                for name, value in candidate_proxy.items()
            }
            maximum_signed_margin = max(signed_margins.values())
            maximum_positive_margin = max(0.0, maximum_signed_margin)
            proxy_passed = all(
                value <= float(guard_proxy_tolerance)
                for value in signed_margins.values()
            )
            case_terms = oracle.case_probe._case_terms(
                candidate,
                problem["batch"],
                cfg,
            )
            candidate_case = {
                key: float(case_terms[key][0].detach())
                for key in ("endpoint", "temporal")
            }
            scientific = oracle._case_scientific_status(
                candidate_case,
                baseline_case,
            )
            scientific_score = sum(
                float(scientific[f"{key}_delta"])
                / max(abs(float(baseline_case[key])), 1.0e-12)
                for key in ("endpoint", "temporal")
            )
            correction = correction_reports.get(method, {}).get(uid, {})
            numeric_failure = bool(
                correction.get("numeric_failure", False)
            )
            accepted = bool(
                severity["outside_frozen_single_envelope"]
                and not severity["severity_abstained"]
                and scientific["passed"]
                and proxy_passed
                and radius_resolved
                and outside_scope_abs_max == 0.0
                and math.isfinite(scientific_score)
                and not numeric_failure
            )
            safe_interior = bool(
                maximum_signed_margin
                <= -float(guard_safe_interior_margin)
            )
            candidates[method] = {
                "objective": scientific_score,
                "scientific_descent_score": scientific_score,
                "severity_gate_passed": bool(
                    severity["outside_frozen_single_envelope"]
                ),
                "severity_abstained": bool(
                    severity["severity_abstained"]
                ),
                "guard_proxy_nonregression": proxy_passed,
                "proxy_passed": proxy_passed,
                "candidate_guard_signed_margin_by_term": signed_margins,
                "maximum_positive_guard_signed_margin": (
                    maximum_positive_margin
                ),
                "maximum_guard_signed_margin": maximum_signed_margin,
                "guard_safe_interior": safe_interior,
                "endpoint_delta": scientific["endpoint_delta"],
                "temporal_delta": scientific["temporal_delta"],
                "scientific_numeric_tolerance": scientific.get(
                    "numeric_tolerance", {}
                ),
                "scientific_passed": bool(scientific["passed"]),
                "radius_rms": active_rms,
                "radius_equality_resolved": radius_resolved,
                "outside_scope_abs_max": outside_scope_abs_max,
                "scope_safe": outside_scope_abs_max == 0.0,
                "numeric_failure": numeric_failure,
                "accepted_guard_margin_reduction": float(
                    correction.get("accepted_guard_margin_reduction", 0.0)
                ),
                "correction_accepted_steps": int(
                    correction.get("correction_accepted_steps", 0)
                ),
                "eligible": accepted,
                "eligible_after_adapter_incumbent": accepted,
            }
            if accepted:
                if safe_interior:
                    rank = (0, scientific_score, maximum_signed_margin, method)
                else:
                    rank = (1, maximum_signed_margin, scientific_score, method)
                eligible.append((rank, method, scoped))

        adapter_candidate = candidates.get("adapter")
        if adapter_candidate and adapter_candidate["eligible"]:
            retained = []
            for row in eligible:
                method = row[1]
                candidate = candidates[method]
                both_safely_interior = bool(
                    adapter_candidate["guard_safe_interior"]
                    and candidate["guard_safe_interior"]
                )
                if both_safely_interior:
                    guard_dominates_adapter = True
                    science_dominates_adapter = bool(
                        candidate["scientific_descent_score"]
                        <= adapter_candidate["scientific_descent_score"]
                        + 1.0e-12
                    )
                    dominance_protocol = (
                        "safe_interior_guard_equivalence_then_science"
                    )
                else:
                    guard_dominates_adapter = all(
                        candidate[
                            "candidate_guard_signed_margin_by_term"
                        ][name]
                        <= adapter_candidate[
                            "candidate_guard_signed_margin_by_term"
                        ][name] + float(guard_proxy_tolerance)
                        for name in G1_GUARD_PROXY_TERMS
                    )
                    endpoint_tolerance = float(
                        adapter_candidate["scientific_numeric_tolerance"].get(
                            "endpoint", 1.0e-12
                        )
                    )
                    temporal_tolerance = float(
                        adapter_candidate["scientific_numeric_tolerance"].get(
                            "temporal", 1.0e-12
                        )
                    )
                    science_dominates_adapter = bool(
                        candidate["endpoint_delta"]
                        <= adapter_candidate["endpoint_delta"]
                        + endpoint_tolerance
                        and candidate["temporal_delta"]
                        <= adapter_candidate["temporal_delta"]
                        + temporal_tolerance
                    )
                    dominance_protocol = "guard_and_science_pareto"
                incumbent_safe = bool(
                    method == "adapter"
                    or (
                        guard_dominates_adapter
                        and science_dominates_adapter
                    )
                )
                candidate["guard_dominates_adapter"] = (
                    guard_dominates_adapter
                )
                candidate["science_dominates_adapter"] = (
                    science_dominates_adapter
                )
                candidate["adapter_dominance_protocol"] = (
                    dominance_protocol
                )
                candidate["eligible_after_adapter_incumbent"] = (
                    incumbent_safe
                )
                if incumbent_safe:
                    retained.append(row)
            eligible = retained

        if eligible:
            _, selected_method, selected_tangent = min(
                eligible,
                key=lambda row: row[0],
            )
            selected[case_index:case_index + 1] = selected_tangent
        else:
            selected_method = "identity"
        conformal_fallback = bool(
            severity["severity_abstained"]
            and selected_method == "identity"
        )
        method_counts.update([selected_method])
        decisions[uid] = {
            "selected_method": selected_method,
            "selected_nonzero": bool(selected_method != "identity"),
            "conformal_fallback": conformal_fallback,
            "selection": {
                "anchor_severity": severity,
                "severity_conformal_score": severity[
                    "severity_conformal_score"
                ],
                "severity_conformal_threshold": severity[
                    "severity_conformal_threshold"
                ],
                "severity_abstained": severity["severity_abstained"],
                "activation_condition": (
                    "train_transaction_conformal_score_above_"
                    "frozen_uncertainty_band"
                ),
                "guard_proxy_terms": dict(G1_GUARD_PROXY_TERMS),
                "guard_proxy_contract": (
                    "candidate_signed_margin_le_numeric_tolerance"
                ),
                "candidates": candidates,
            },
            "evaluation_only": {
                "teacher_kind": sample.get("teacher_kind"),
                "audit_group": sample.get("audit_group"),
            },
        }
    return selected, decisions, dict(method_counts)


def _g1c_fixed_guard_shadow_selection(
    *,
    model,
    variants,
    correction_reports,
    samples,
    domains,
    ownership,
    baseline_terms,
    cfg,
    target_rms,
    workspace_floor,
    severity_envelope,
):
    """Select with observable activation and authoritative Guard shadows.

    The discriminative gate consumes no runtime role/group labels.  Once it
    activates, each already-generated candidate is reconstructed inside the
    complete transaction and compared with the exact immutable Guard contract.
    A raw-feasible Adapter is retained without ranking against corrections.
    """
    selected = m.torch.zeros_like(next(iter(variants.values())))
    decisions = {}
    method_counts = Counter()
    for sample in samples:
        uid = str(sample.get("case_uid", sample["case_index"]))
        case_index = int(sample["case_index"])
        severity = _discriminative_conformal_status(
            sample, severity_envelope
        )
        candidates = {}
        selected_method = "identity"
        incumbent_locked = False
        if severity["activation_supported_by_observables"]:
            for method, tangent in variants.items():
                evidence = _g1c_candidate_evidence(
                    model=model,
                    sample=sample,
                    tangent=tangent,
                    domains=domains,
                    ownership=ownership,
                    baseline_terms=baseline_terms,
                    cfg=cfg,
                    target_rms=target_rms,
                    workspace_floor=workspace_floor,
                )
                correction = correction_reports.get(method, {}).get(uid, {})
                numeric_failure = bool(
                    correction.get("numeric_failure", False)
                )
                scientific = evidence["case_scientific"]
                baseline_case = {
                    key: float(baseline_terms[key][case_index].detach())
                    for key in ("endpoint", "temporal")
                }
                scientific_score = sum(
                    float(scientific[f"{key}_delta"])
                    / max(abs(baseline_case[key]), 1.0e-12)
                    for key in ("endpoint", "temporal")
                )
                eligible = bool(
                    evidence["fixed_guard_shadow_passed"]
                    and evidence["fixed_guard_shadow_matches_exact_guard"]
                    and evidence["fixed_guard_passed"]
                    and scientific["passed"]
                    and evidence["radius_equality_resolved"]
                    and evidence["workspace_observable_resolved"]
                    and evidence["scope_safe"]
                    and evidence["outside_scope_abs_max"] == 0.0
                    and math.isfinite(scientific_score)
                    and not numeric_failure
                )
                candidates[method] = {
                    **evidence,
                    "scientific_descent_score": scientific_score,
                    "numeric_failure": numeric_failure,
                    "correction_accepted_steps": int(
                        correction.get("correction_accepted_steps", 0)
                    ),
                    "accepted_case_physical_margin_reduction": float(
                        correction.get(
                            "accepted_case_physical_margin_reduction", 0.0
                        )
                    ),
                    "full_shadow_gradient_norm": float(
                        correction.get("full_shadow_gradient_norm", 0.0)
                    ),
                    "full_shadow_reduction": float(
                        correction.get("full_shadow_reduction", 0.0)
                    ),
                    "step_rejection_reason": correction.get(
                        "step_rejection_reason"
                    ),
                    "eligible": eligible,
                }

            adapter_candidate = candidates.get("adapter")
            if (
                adapter_candidate is not None
                and adapter_candidate["eligible"]
                and adapter_candidate["exact_raw_closure_passed"]
            ):
                selected_method = "adapter"
                incumbent_locked = True
            else:
                eligible_corrections = [
                    (method, candidate)
                    for method, candidate in candidates.items()
                    if method != "adapter" and candidate["eligible"]
                ]
                if eligible_corrections:
                    selected_method, _ = min(
                        eligible_corrections,
                        key=lambda row: (
                            row[1][
                                "maximum_full_transaction_fixed_guard_"
                                "shadow_margin"
                            ],
                            row[1]["scientific_descent_score"],
                            row[0],
                        ),
                    )
            if selected_method != "identity":
                local = variants[selected_method][
                    case_index:case_index + 1
                ]
                mask = ownership[
                    case_index:case_index + 1
                ].expand_as(local)
                selected[case_index:case_index + 1] = local.masked_fill(
                    ~mask, 0.0
                )

        fallback_reason = None
        if selected_method == "identity":
            if severity["severity_abstained"]:
                fallback_reason = "discriminative_conformal_uncertain"
            elif not severity["activation_supported_by_observables"]:
                fallback_reason = "observable_classifier_single"
            else:
                fallback_reason = "no_exact_guard_shadow_candidate"
        method_counts.update([selected_method])
        decisions[uid] = {
            "selected_method": selected_method,
            "selected_nonzero": bool(selected_method != "identity"),
            "adapter_incumbent_locked": incumbent_locked,
            "conformal_fallback": bool(severity["severity_abstained"]),
            "identity_fallback_reason": fallback_reason,
            "selection": {
                "anchor_severity": severity,
                "activation_condition": (
                    "train_transaction_discriminative_conformal_"
                    "cross_only"
                ),
                "case_physical_margin_contract": (
                    "per_case_stage_relative_diagnostic_only"
                ),
                "fixed_guard_shadow_contract": (
                    "complete_transaction_group_guard_value_minus_"
                    "frozen_anchor_allowance_and_numeric_tolerance"
                ),
                "candidates": candidates,
            },
            "evaluation_only": {
                "teacher_kind": sample.get("teacher_kind"),
                "audit_group": sample.get("audit_group"),
            },
        }
    return selected, decisions, dict(method_counts)


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
    g1c_family = bool(
        args.activation_aware_g1c or args.activation_aware_g1d
    )
    g1_family = bool(
        args.activation_aware_g1
        or args.activation_aware_g1b
        or g1c_family
    )
    activation_enabled = bool(
        args.activation_aware or g1_family
    )
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

    train_teacher = None
    severity_envelope = None
    severity_envelope_path = None
    train_shadow_contract = None
    train_shadow_contract_path = None
    train_teacher_path = None
    if g1_family:
        if not args.train_teacher_bank:
            raise RuntimeError(
                "V15.15g1/g1b/g1c/g1d requires --train-teacher-bank"
            )
        train_teacher_path = Path(args.train_teacher_bank).resolve()
        train_teacher = m.torch.load(
            train_teacher_path, map_location="cpu", weights_only=False
        )
        if train_teacher.get("schema") != TEACHER_SCHEMA:
            raise RuntimeError(
                "V15.15g1/g1b/g1c/g1d train bank schema mismatch"
            )
        if train_teacher.get("split") != "train":
            raise RuntimeError(
                "V15.15g1/g1b/g1c/g1d envelope bank must be train split"
            )
        if not train_teacher.get("teacher_bank_ready"):
            raise RuntimeError(
                "V15.15g1/g1b/g1c/g1d train bank is not ready"
            )
        if train_teacher.get("source_diagnostic") != teacher.get(
            "source_diagnostic"
        ):
            raise RuntimeError("train/validation source diagnostic mismatch")
        for key in (
            "split_manifest_file_sha256",
            "split_manifest_content_sha256",
        ):
            if train_teacher.get(key) != teacher.get(key):
                raise RuntimeError(f"train/validation {key} mismatch")
        if g1c_family:
            severity_envelope = _freeze_discriminative_transaction_conformal(
                train_teacher,
                shrinkage=float(args.severity_conformal_shrinkage),
                scale_floor=float(args.severity_scale_floor),
                uncertainty_fraction=float(
                    args.severity_conformal_uncertainty_fraction
                ),
                absolute_margin=float(
                    args.severity_envelope_absolute_margin
                ),
            )
        elif args.activation_aware_g1b:
            severity_envelope = _freeze_transaction_conformal_envelope(
                train_teacher,
                shrinkage=float(args.severity_conformal_shrinkage),
                scale_floor=float(args.severity_scale_floor),
                uncertainty_fraction=float(
                    args.severity_conformal_uncertainty_fraction
                ),
                absolute_margin=float(
                    args.severity_envelope_absolute_margin
                ),
            )
        else:
            severity_envelope = _freeze_single_severity_envelope(
                train_teacher,
                margin_fraction=float(
                    args.severity_envelope_margin_fraction
                ),
                absolute_margin=float(
                    args.severity_envelope_absolute_margin
                ),
            )
        severity_envelope.update({
            "train_teacher_bank": str(train_teacher_path),
            "train_teacher_bank_sha256": _file_sha256(train_teacher_path),
            "split_manifest_file_sha256": teacher.get(
                "split_manifest_file_sha256"
            ),
            "split_manifest_content_sha256": teacher.get(
                "split_manifest_content_sha256"
            ),
        })
        severity_envelope_path = destination / (
            "discriminative_transaction_conformal_severity.json"
            if g1c_family
            else "transaction_conformal_severity_envelope.json"
            if args.activation_aware_g1b
            else "observable_severity_envelope.json"
        )
        severity_envelope_path.write_text(
            json.dumps(severity_envelope, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        if args.activation_aware_g1d:
            train_shadow_contract = _freeze_train_full_shadow_repair_contract(
                train_teacher,
                lse_allowance_fraction=float(
                    args.full_shadow_lse_allowance_fraction
                ),
                lse_temperature_floor=float(
                    args.full_shadow_lse_temperature_floor
                ),
                projection_damping=float(args.full_shadow_projection_damping),
                minimum_reduction_floor=float(args.guard_minimum_reduction),
            )
            train_shadow_contract.update({
                "train_teacher_bank": str(train_teacher_path),
                "train_teacher_bank_sha256": _file_sha256(train_teacher_path),
                "validation_teacher_bank_consumed": False,
                "held_out_validation_evaluation_passes": 1,
            })
            train_shadow_contract_path = (
                destination / "train_frozen_full_shadow_repair_contract.json"
            )
            train_shadow_contract_path.write_text(
                json.dumps(
                    train_shadow_contract, ensure_ascii=False, indent=2
                ) + "\n",
                encoding="utf-8",
            )

    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    cfg.product_refiner_observable_adapter = True
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.15g correction ablation requires CUDA")
    batch = adapter._to_device(teacher["batch"], device)
    model, _, _gate_mode = adapter._load_source_model(
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
    if activation_enabled:
        initial, radius = _normalize_all_sample_tangents(
            initial,
            ownership,
            samples,
            float(args.target_rms),
        )
    else:
        initial, radius = adapter._safe_exact_radius_tangent(
            initial,
            ownership,
            samples,
            float(args.target_rms),
            eps=1.0e-8,
        )

    variants = {}
    variant_tangents = {}
    variant_correction_reports = {}
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
                if (
                    not activation_enabled
                    and sample["teacher_kind"] != "exact_projected_direction"
                ):
                    continue
                transaction_id = str(sample["transaction_id"])
                domain = domains[transaction_id]
                start, stop = domain["slice"]
                local_case = int(sample["local_case_index"])
                if activation_enabled:
                    group = None
                    global_case = int(sample["case_index"])
                    local_initial = initial[global_case:global_case + 1]
                    local_ownership = ownership[global_case:global_case + 1]
                    local_c2_taper = trace["c2_taper"][
                        global_case:global_case + 1
                    ]
                    local_baseline = domain["baseline"][
                        local_case:local_case + 1
                    ]
                    local_identity = domain["identity"][
                        local_case:local_case + 1
                    ]
                    local_batch = adapter._slice_batch(
                        domain["batch"], local_case, local_case + 1
                    )
                    correction_case = 0
                    baseline_case = {
                        key: float(
                            baseline_terms[key][int(sample["case_index"])]
                        )
                        for key in ("endpoint", "temporal")
                    }
                    local_contract = None
                else:
                    group = str(sample["audit_group"])
                    local_initial = initial[start:stop]
                    local_ownership = ownership[start:stop]
                    local_baseline = domain["baseline"]
                    local_identity = domain["identity"]
                    local_batch = domain["batch"]
                    correction_case = local_case
                    baseline_case = {
                        key: float(
                            baseline_terms[key][int(sample["case_index"])]
                        )
                        for key in ("endpoint", "temporal")
                    }
                    local_contract = domain["contract"]
                if args.activation_aware_g1d:
                    corrected, correction_report = (
                        _correct_case_full_transaction_shadow(
                            model=model,
                            method=method,
                            steps=budget,
                            initial_tangent=local_initial,
                            ownership=local_ownership,
                            c2_taper=local_c2_taper,
                            baseline=domain["baseline"],
                            identity=domain["identity"],
                            batch=domain["batch"],
                            cfg=cfg,
                            local_case=local_case,
                            baseline_case=baseline_case,
                            contract=domain["contract"],
                            train_repair_contract=train_shadow_contract,
                            target_rms=float(args.target_rms),
                            step_size=float(args.step_size),
                            trust_fraction=float(args.trust_fraction),
                            temporal_smoothness_weight=float(
                                args.correction_temporal_smoothness_weight
                            ),
                        )
                    )
                else:
                    corrected, correction_report = _correct_case(
                        model=model,
                        method=method,
                        steps=budget,
                        initial_tangent=local_initial,
                        ownership=local_ownership,
                        baseline=local_baseline,
                        identity=local_identity,
                        batch=local_batch,
                        cfg=cfg,
                        case_index=correction_case,
                        group=group,
                        baseline_case=baseline_case,
                        contract=local_contract,
                        target_rms=float(args.target_rms),
                        step_size=float(args.step_size),
                        trust_fraction=float(args.trust_fraction),
                        activation_aware=activation_enabled,
                        guard_aligned=g1_family,
                        contract_margin_active_set=bool(
                            args.activation_aware_g1b
                            or args.activation_aware_g1c
                        ),
                        guard_proxy_tolerance=float(
                            args.guard_proxy_nonregression_tolerance
                        ),
                        guard_proxy_scale_floor=float(
                            args.guard_proxy_scale_floor
                        ),
                        guard_smooth_max_temperature=float(
                            args.guard_smooth_max_temperature
                        ),
                        correction_temporal_smoothness_weight=float(
                            args.correction_temporal_smoothness_weight
                        ),
                        guard_minimum_reduction=float(
                            args.guard_minimum_reduction
                        ),
                    )
                if args.activation_aware_g1c:
                    correction_report = _g1c_name_case_physical_diagnostics(
                        correction_report
                    )
                if activation_enabled:
                    global_case = int(sample["case_index"])
                    tangent[global_case:global_case + 1] = corrected
                else:
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
        variant_tangents[key] = tangent.detach().clone()
        variant_correction_reports[key] = correction_reports
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

    activation_summary = None
    activation_aware_supported = False
    if activation_enabled:
        selection_started = time.perf_counter()
        if g1c_family:
            (
                selected_tangent,
                activation_decisions,
                selected_method_counts,
            ) = _g1c_fixed_guard_shadow_selection(
                model=model,
                variants=variant_tangents,
                correction_reports=variant_correction_reports,
                samples=samples,
                domains=domains,
                ownership=ownership,
                baseline_terms=baseline_terms,
                cfg=cfg,
                target_rms=float(args.target_rms),
                workspace_floor=workspace_floor,
                severity_envelope=severity_envelope,
            )
        elif args.activation_aware_g1b:
            (
                selected_tangent,
                activation_decisions,
                selected_method_counts,
            ) = _g1b_contract_margin_selection(
                variants=variant_tangents,
                correction_reports=variant_correction_reports,
                samples=samples,
                domains=domains,
                ownership=ownership,
                baseline_terms=baseline_terms,
                cfg=cfg,
                target_rms=float(args.target_rms),
                severity_envelope=severity_envelope,
                guard_proxy_tolerance=float(
                    args.guard_proxy_nonregression_tolerance
                ),
                guard_safe_interior_margin=float(
                    args.guard_safe_interior_margin
                ),
            )
        elif args.activation_aware_g1:
            (
                selected_tangent,
                activation_decisions,
                selected_method_counts,
            ) = _g1_observable_guard_selection(
                variants=variant_tangents,
                correction_reports=variant_correction_reports,
                samples=samples,
                domains=domains,
                ownership=ownership,
                baseline_terms=baseline_terms,
                cfg=cfg,
                target_rms=float(args.target_rms),
                severity_envelope=severity_envelope,
                severity_scale_floor=float(args.severity_scale_floor),
                guard_proxy_tolerance=float(
                    args.guard_proxy_nonregression_tolerance
                ),
                guard_proxy_scale_floor=float(
                    args.guard_proxy_scale_floor
                ),
            )
        else:
            (
                selected_tangent,
                activation_decisions,
                selected_method_counts,
            ) = _activation_aware_selection(
                variants=variant_tangents,
                correction_reports=variant_correction_reports,
                samples=samples,
                domains=domains,
                ownership=ownership,
                baseline_terms=baseline_terms,
                cfg=cfg,
                target_rms=float(args.target_rms),
                proxy_tolerance=float(args.activation_proxy_tolerance),
                relative_improvement=float(
                    args.activation_relative_improvement
                ),
            )
        selected_audits = adapter._audit_step(
            model,
            batch,
            cfg,
            selected_tangent.detach(),
            selected_tangent.detach(),
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
        selected_summary = _variant_summary(
            selected_audits,
            {},
            time.perf_counter() - selection_started,
        )
        variants["activation_aware_selected"] = selected_summary
        audit_by_uid = {
            str(row["case_uid"]): row for row in selected_audits
        }
        cross_samples = [
            sample for sample in samples
            if sample["teacher_kind"] == "exact_projected_direction"
        ]
        identity_samples = [
            sample for sample in samples
            if sample["teacher_kind"] == "identity_control"
        ]
        required_by_group = Counter(
            str(sample["audit_group"]) for sample in cross_samples
        )
        selected_projected_by_group = Counter(
            str(row["group"])
            for row in selected_audits
            if row["effective_projected_candidate"]
        )
        cross_closure_complete = bool(
            cross_samples
            and all(
                uid in audit_by_uid
                and audit_by_uid[uid]["raw_audit"]["passed"]
                and audit_by_uid[uid]["projector_result"] is not None
                and audit_by_uid[uid]["effective_projected_candidate"]
                for uid in (
                    str(sample["case_uid"])
                    for sample in cross_samples
                )
            )
        )
        identity_rms_by_uid = {}
        for sample in identity_samples:
            case_index = int(sample["case_index"])
            mask = _owned_case_mask(
                ownership, selected_tangent, case_index
            )
            identity_rms_by_uid[str(sample["case_uid"])] = _rms(
                selected_tangent, mask
            )
        identity_rms_max = max(identity_rms_by_uid.values(), default=0.0)
        false_activation_uids = sorted(
            str(sample["case_uid"])
            for sample in identity_samples
            if activation_decisions[str(sample["case_uid"])][
                "selected_nonzero"
            ]
        )
        missed_cross_uids = sorted(
            str(sample["case_uid"])
            for sample in cross_samples
            if not activation_decisions[str(sample["case_uid"])][
                "selected_nonzero"
            ]
        )
        if g1c_family:
            selected_proxy_nonregression_complete = all(
                not decision["selected_nonzero"]
                or (
                    decision["selection"]["candidates"][
                        decision["selected_method"]
                    ]["fixed_guard_shadow_passed"]
                    and decision["selection"]["candidates"][
                        decision["selected_method"]
                    ]["fixed_guard_shadow_matches_exact_guard"]
                )
                for decision in activation_decisions.values()
            )
        else:
            selected_proxy_nonregression_complete = bool(
                not g1_family
                or all(
                    not decision["selected_nonzero"]
                    or decision["selection"]["candidates"][
                        decision["selected_method"]
                    ]["guard_proxy_nonregression"]
                    for decision in activation_decisions.values()
                )
            )
        selected_severity_condition_complete = bool(
            not g1_family
            or all(
                not decision["selected_nonzero"]
                or decision["selection"]["anchor_severity"][
                    "outside_frozen_single_envelope"
                ]
                for decision in activation_decisions.values()
            )
        )
        correction_numeric_failures = sum(
            int(report.get("numeric_failure", False))
            for reports in variant_correction_reports.values()
            for report in reports.values()
        )
        if g1c_family:
            decision_numeric_complete = all(
                all(
                    math.isfinite(float(value))
                    for value in (
                        decision["selection"]["anchor_severity"][
                            "single_conformal_distance"
                        ],
                        decision["selection"]["anchor_severity"][
                            "cross_conformal_distance"
                        ],
                        decision["selection"]["anchor_severity"][
                            "cross_discriminant"
                        ],
                    )
                )
                and all(
                    math.isfinite(float(candidate[
                        "scientific_descent_score"
                    ]))
                    and math.isfinite(float(candidate[
                        "outside_scope_abs_max"
                    ]))
                    and math.isfinite(float(candidate[
                        "maximum_positive_case_physical_signed_margin"
                    ]))
                    and math.isfinite(float(candidate[
                        "maximum_positive_full_transaction_fixed_guard_"
                        "shadow_margin"
                    ]))
                    and all(
                        math.isfinite(float(value))
                        for value in candidate[
                            "case_physical_signed_margin_by_term"
                        ].values()
                    )
                    and all(
                        math.isfinite(float(value))
                        for value in candidate[
                            "full_transaction_fixed_guard_shadow_"
                            "margin_by_term"
                        ].values()
                    )
                    and candidate[
                        "fixed_guard_shadow_matches_exact_guard"
                    ]
                    for candidate in decision["selection"][
                        "candidates"
                    ].values()
                )
                for decision in activation_decisions.values()
            )
        elif args.activation_aware_g1b:
            decision_numeric_complete = all(
                math.isfinite(float(decision["selection"][
                    "severity_conformal_score"
                ]))
                and math.isfinite(float(decision["selection"][
                    "severity_conformal_threshold"
                ]))
                and all(
                    math.isfinite(float(candidate["objective"]))
                    and math.isfinite(float(candidate[
                        "outside_scope_abs_max"
                    ]))
                    and math.isfinite(float(candidate[
                        "maximum_positive_guard_signed_margin"
                    ]))
                    and all(
                        math.isfinite(float(value))
                        for value in candidate[
                            "candidate_guard_signed_margin_by_term"
                        ].values()
                    )
                    for candidate in decision["selection"][
                        "candidates"
                    ].values()
                )
                for decision in activation_decisions.values()
            )
        elif args.activation_aware_g1:
            decision_numeric_complete = all(
                math.isfinite(float(decision["selection"][
                    "anchor_severity"
                ]["maximum_normalized_excess"]))
                and all(
                    math.isfinite(float(candidate["objective"]))
                    and math.isfinite(float(candidate[
                        "outside_scope_abs_max"
                    ]))
                    and all(
                        math.isfinite(float(value))
                        for value in candidate[
                            "guard_proxy_delta"
                        ].values()
                    )
                    for candidate in decision["selection"][
                        "candidates"
                    ].values()
                )
                for decision in activation_decisions.values()
            )
        else:
            decision_numeric_complete = all(
                math.isfinite(float(decision["selection"][
                    "identity_objective"
                ]))
                and all(
                    math.isfinite(float(candidate["objective"]))
                    and math.isfinite(float(candidate[
                        "physical_proxy_excess"
                    ]))
                    for candidate in decision["selection"][
                        "candidates"
                    ].values()
                )
                for decision in activation_decisions.values()
            )
        scope_safe = bool(
            selected_audits
            and all(
                float(row["raw_audit"]["scope"][
                    "outside_case_group_or_ownership_abs_max"
                ]) == 0.0
                for row in selected_audits
            )
        )
        group_coverage_complete = all(
            selected_projected_by_group.get(group, 0) >= count
            for group, count in required_by_group.items()
        )
        single_identity_safe = bool(
            not false_activation_uids
            and identity_rms_max <= 1.0e-7
        )
        numeric_audit_complete = bool(
            decision_numeric_complete
            and correction_numeric_failures == 0
            and selected_audits
        )
        activation_aware_supported = bool(
            cross_closure_complete
            and group_coverage_complete
            and single_identity_safe
            and scope_safe
            and numeric_audit_complete
            and selected_proxy_nonregression_complete
            and selected_severity_condition_complete
        )
        activation_summary = {
            "selection_protocol": (
                (
                    "observable_discriminative_conformal_then_"
                    "full_transaction_shadow_gradient_repair_then_"
                    "complete_fixed_guard_incumbent_lock_v1"
                )
                if args.activation_aware_g1d
                else
                (
                    "observable_discriminative_conformal_then_"
                    "complete_transaction_fixed_guard_shadow_"
                    "incumbent_lock_v1"
                )
                if g1c_family
                else
                (
                    "transaction_conformal_contract_margin_"
                    "active_set_lexicographic_v1"
                )
                if args.activation_aware_g1b
                else "severity_then_guard_proxy_lexicographic_v1"
                if args.activation_aware_g1
                else "identity_or_lowest_observable_anchor_objective_v1"
            ),
            "identity_is_explicit_zero_candidate": True,
            "nonidentity_candidate_radius_rms": float(args.target_rms),
            "activation_proxy_tolerance": float(
                args.activation_proxy_tolerance
            ) if not g1_family else None,
            "activation_relative_improvement": (
                float(args.activation_relative_improvement)
                if not g1_family else None
            ),
            "observable_severity_envelope": (
                str(severity_envelope_path)
                if severity_envelope_path is not None else None
            ),
            "observable_severity_envelope_sha256": (
                _file_sha256(severity_envelope_path)
                if severity_envelope_path is not None else None
            ),
            "severity_envelope_calibration_split": (
                "train" if g1_family else None
            ),
            "severity_envelope_role_used_offline_only": bool(
                g1_family
            ),
            "guard_aligned_proxy_terms": (
                dict(G1_GUARD_PROXY_TERMS)
                if g1_family else None
            ),
            "guard_proxy_nonregression_tolerance": (
                float(args.guard_proxy_nonregression_tolerance)
                if g1_family else None
            ),
            "observable_severity_channels": (
                list(G1_SEVERITY_CHANNELS)
                if g1_family else None
            ),
            "severity_calibration_protocol": (
                (
                    "leave_one_train_transaction_out_single_cross_"
                    "mahalanobis_two_sided"
                )
                if g1c_family
                else
                "leave_one_train_transaction_out_mahalanobis_max"
                if args.activation_aware_g1b else None
            ),
            "severity_conformal_threshold": (
                severity_envelope.get("conformal_threshold")
                if args.activation_aware_g1b else None
            ),
            "severity_activation_threshold": (
                severity_envelope.get("activation_threshold")
                if args.activation_aware_g1b else None
            ),
            "single_conformal_distance_threshold": (
                severity_envelope.get("single_distance_threshold")
                if g1c_family else None
            ),
            "cross_conformal_distance_threshold": (
                severity_envelope.get("cross_distance_threshold")
                if g1c_family else None
            ),
            "activation_discriminant_threshold": (
                severity_envelope.get(
                    "activation_discriminant_threshold"
                )
                if g1c_family else None
            ),
            "conformal_class_overlap_in_calibration": (
                severity_envelope.get("class_overlap_in_calibration")
                if g1c_family else None
            ),
            "conformal_fallback_case_uids": sorted(
                uid
                for uid, decision in activation_decisions.items()
                if decision.get("conformal_fallback", False)
            ),
            "guard_safe_interior_margin": (
                float(args.guard_safe_interior_margin)
                if args.activation_aware_g1b else None
            ),
            "guard_active_set_smooth_max_temperature": (
                float(args.guard_smooth_max_temperature)
                if args.activation_aware_g1b else None
            ),
            "correction_temporal_smoothness_weight": (
                float(args.correction_temporal_smoothness_weight)
                if args.activation_aware_g1b else None
            ),
            "correction_budget_unchanged": bool(
                (args.activation_aware_g1b or g1c_family)
                and budgets == (2, 3, 5)
            ),
            "train_frozen_full_shadow_repair_contract": (
                str(train_shadow_contract_path)
                if train_shadow_contract_path is not None else None
            ),
            "train_frozen_full_shadow_repair_contract_sha256": (
                _file_sha256(train_shadow_contract_path)
                if train_shadow_contract_path is not None else None
            ),
            "repair_parameters_frozen_before_validation": bool(
                args.activation_aware_g1d
            ),
            "held_out_validation_evaluation_passes": (
                1 if args.activation_aware_g1d else None
            ),
            "activation_decision_role_label_consumed": False,
            "activation_decision_teacher_kind_consumed": False,
            "activation_decision_group_consumed": False,
            "activation_decision_hidden_clean_consumed": False,
            "activation_decision_fixed_guard_consumed": False,
            "fixed_guard_shadow_frozen_group_contract_consumed": bool(
                g1c_family
            ),
            "candidate_acceptance_fixed_guard_shadow_consumed": bool(
                g1c_family
            ),
            "fixed_guard_evaluation_only": bool(
                not g1c_family
            ),
            "selected_method_counts": selected_method_counts,
            "required_projected_count_by_group": dict(
                required_by_group
            ),
            "selected_projected_count_by_group": dict(
                selected_projected_by_group
            ),
            "false_activation_case_uids": false_activation_uids,
            "missed_cross_case_uids": missed_cross_uids,
            "single_control_applied_tangent_rms_max": identity_rms_max,
            "single_control_applied_tangent_rms_by_case": (
                identity_rms_by_uid
            ),
            "correction_numeric_failure_count": (
                correction_numeric_failures
            ),
            "cross_exact_closure_complete": cross_closure_complete,
            "group_coverage_complete": group_coverage_complete,
            "single_identity_safe": single_identity_safe,
            "scope_safe": scope_safe,
            "numeric_audit_complete": numeric_audit_complete,
            "selected_guard_proxy_nonregression_complete": (
                selected_proxy_nonregression_complete
                if not g1c_family else None
            ),
            "selected_fixed_guard_shadow_complete": (
                selected_proxy_nonregression_complete
                if g1c_family else None
            ),
            "selected_severity_condition_complete": (
                selected_severity_condition_complete
            ),
            "activation_aware_supported": activation_aware_supported,
            "decisions": activation_decisions,
        }

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
        "schema": (
            G1D_SCHEMA
            if args.activation_aware_g1d
            else G1C_SCHEMA
            if args.activation_aware_g1c
            else G1B_SCHEMA
            if args.activation_aware_g1b
            else G1_SCHEMA
            if args.activation_aware_g1
            else ACTIVATION_AWARE_SCHEMA
            if args.activation_aware
            else SCHEMA
        ),
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "train_teacher_bank": (
            str(train_teacher_path) if train_teacher_path else None
        ),
        "validation_teacher_bank": str(teacher_path),
        "adapter_state": str(state_path),
        "observable_adapter_gate_mode": _gate_mode,
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
        "activation_aware": activation_enabled,
        "observable_severity_guard_aligned": bool(
            g1_family
        ),
        "contract_margin_active_set": bool(
            args.activation_aware_g1b or g1c_family
        ),
        "transaction_conformal_activation": bool(
            args.activation_aware_g1b or g1c_family
        ),
        "discriminative_conformal_activation": bool(
            g1c_family
        ),
        "fixed_guard_shadow_candidate_acceptance": bool(
            g1c_family
        ),
        "adapter_incumbent_lock": bool(g1c_family),
        "activation_severity_source": (
            "train_single_cross_leave_one_transaction_out_mahalanobis"
            if g1c_family
            else
            "train_single_leave_one_transaction_out_mahalanobis"
            if args.activation_aware_g1b
            else "frozen_train_single_observable_envelope"
            if args.activation_aware_g1
            else None
        ),
        "correction_guard_proxy_source": (
            "complete_transaction_fixed_guard_shadow_gradient"
            if args.activation_aware_g1d
            else "per_case_physical_stage_relative_signed_margins"
            if args.activation_aware_g1c
            else "observable_refiner_objective_same_source_signed_margins"
            if g1_family else None
        ),
        "candidate_fixed_guard_shadow_source": (
            "complete_transaction_exact_fixed_guard_details"
            if g1c_family else None
        ),
        "guard_proxy_acceptance_contract": (
            "candidate_signed_margin_le_numeric_tolerance"
            if args.activation_aware_g1b else None
        ),
        "fixed_guard_shadow_acceptance_contract": (
            "full_transaction_fixed_guard_shadow_le_zero"
            if g1c_family else None
        ),
        "guard_active_set_science_feasibility_projection": bool(
            args.activation_aware_g1b or g1c_family
        ),
        "correction_temporal_regularizer": (
            "owned_tangent_second_plus_third_difference_l2"
            if (args.activation_aware_g1b or g1c_family)
            else None
        ),
        "complete_fixed_guard_used_for_candidate_selection": bool(
            g1c_family
        ),
        "complete_fixed_guard_used_for_final_acceptance": True,
        "inference_role_label_consumed": False,
        "inference_teacher_kind_consumed": False,
        "inference_validation_label_consumed": False,
        "inference_hidden_clean_consumed": False,
        "activation_inference_group_label_consumed": False,
        "fixed_guard_shadow_frozen_group_contract_consumed": bool(
            g1c_family
        ),
        "full_transaction_shadow_gradient_repair": bool(
            args.activation_aware_g1d
        ),
        "full_shadow_hard_acceptance_smooth_gradient": bool(
            args.activation_aware_g1d
        ),
        "full_shadow_group_gradient_relaxation": (
            "logsumexp"
            if args.activation_aware_g1d else None
        ),
        "full_shadow_p95_active_index_frozen_within_line_search": bool(
            args.activation_aware_g1d
        ),
        "train_frozen_full_shadow_repair_contract": (
            str(train_shadow_contract_path)
            if train_shadow_contract_path is not None else None
        ),
        "train_frozen_full_shadow_repair_contract_sha256": (
            _file_sha256(train_shadow_contract_path)
            if train_shadow_contract_path is not None else None
        ),
        "repair_parameter_calibration_split": (
            "train" if args.activation_aware_g1d else None
        ),
        "held_out_validation_parameter_selection": False,
        "held_out_validation_evaluation_passes": (
            1 if args.activation_aware_g1d else None
        ),
        "activation_aware_summary": activation_summary,
        "activation_aware_supported": activation_aware_supported,
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
        "numeric_audit_complete": bool(
            all(row["exact_audits"] for row in variants.values())
            and (
                not activation_enabled
                or activation_summary["numeric_audit_complete"]
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = destination / "fixed_budget_correction.report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": (
            "v15_15g1d_full_transaction_shadow_gradient_complete"
            if args.activation_aware_g1d
            else "v15_15g1c_fixed_guard_shadow_complete"
            if args.activation_aware_g1c
            else "v15_15g1b_contract_margin_conformal_complete"
            if args.activation_aware_g1b
            else "v15_15g1_observable_guard_aligned_complete"
            if args.activation_aware_g1
            else "v15_15g_activation_aware_complete"
            if args.activation_aware
            else "v15_15g_fixed_budget_correction_complete"
        ),
        "report": str(report_path),
        "riemannian_correction_supported": riemannian_supported,
        "activation_aware_supported": activation_aware_supported,
        "numeric_audit_complete": report["numeric_audit_complete"],
    }), flush=True)
    if activation_enabled:
        return 0 if activation_aware_supported else 2
    return 0 if report["numeric_audit_complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-teacher-bank", required=True)
    parser.add_argument("--train-teacher-bank")
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--steps", type=int, nargs="+", default=(2, 3, 5))
    parser.add_argument("--target-rms", type=float, default=1.0e-4)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--trust-fraction", type=float, default=0.5)
    parser.add_argument("--activation-aware", action="store_true")
    parser.add_argument("--activation-aware-g1", action="store_true")
    parser.add_argument("--activation-aware-g1b", action="store_true")
    parser.add_argument("--activation-aware-g1c", action="store_true")
    parser.add_argument("--activation-aware-g1d", action="store_true")
    parser.add_argument(
        "--activation-proxy-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--activation-relative-improvement", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--severity-envelope-margin-fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--severity-envelope-absolute-margin", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--severity-scale-floor", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--severity-conformal-shrinkage", type=float, default=0.1
    )
    parser.add_argument(
        "--severity-conformal-uncertainty-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--guard-proxy-nonregression-tolerance",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--guard-proxy-scale-floor", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--guard-smooth-max-temperature", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--correction-temporal-smoothness-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--guard-minimum-reduction", type=float, default=1.0e-12
    )
    parser.add_argument(
        "--guard-safe-interior-margin", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--full-shadow-lse-allowance-fraction",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--full-shadow-lse-temperature-floor",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--full-shadow-projection-damping",
        type=float,
        default=1.0e-12,
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
    if tuple(sorted(set(args.steps))) != (2, 3, 5):
        parser.error("--steps must contain exactly 2 3 5")
    if args.step_size <= 0.0:
        parser.error("--step-size must be positive")
    if not 0.0 < args.trust_fraction <= 1.0:
        parser.error("--trust-fraction must be in (0, 1]")
    if args.activation_proxy_tolerance < 0.0:
        parser.error("--activation-proxy-tolerance must be non-negative")
    if sum(map(int, (
        args.activation_aware,
        args.activation_aware_g1,
        args.activation_aware_g1b,
        args.activation_aware_g1c,
        args.activation_aware_g1d,
    ))) > 1:
        parser.error("choose one activation-aware protocol")
    if (
        args.activation_aware_g1
        or args.activation_aware_g1b
        or args.activation_aware_g1c
        or args.activation_aware_g1d
    ) and not args.train_teacher_bank:
        parser.error(
            "--activation-aware-g1/g1b/g1c/g1d requires --train-teacher-bank"
        )
    if args.severity_envelope_margin_fraction < 0.0:
        parser.error("severity envelope margin fraction must be non-negative")
    if args.severity_envelope_absolute_margin < 0.0:
        parser.error("severity envelope absolute margin must be non-negative")
    if args.severity_scale_floor <= 0.0:
        parser.error("severity scale floor must be positive")
    if not 0.0 < args.severity_conformal_shrinkage <= 1.0:
        parser.error("severity conformal shrinkage must be in (0, 1]")
    if args.severity_conformal_uncertainty_fraction < 0.0:
        parser.error("severity conformal uncertainty must be non-negative")
    if args.guard_proxy_nonregression_tolerance < 0.0:
        parser.error("guard proxy tolerance must be non-negative")
    if args.guard_proxy_scale_floor <= 0.0:
        parser.error("guard proxy scale floor must be positive")
    if args.guard_smooth_max_temperature <= 0.0:
        parser.error("guard smooth-max temperature must be positive")
    if args.correction_temporal_smoothness_weight < 0.0:
        parser.error("correction temporal smoothness must be non-negative")
    if args.guard_minimum_reduction < 0.0:
        parser.error("guard minimum reduction must be non-negative")
    if args.guard_safe_interior_margin < 0.0:
        parser.error("guard safe interior margin must be non-negative")
    if args.full_shadow_lse_allowance_fraction <= 0.0:
        parser.error("full-shadow LSE allowance fraction must be positive")
    if args.full_shadow_lse_temperature_floor <= 0.0:
        parser.error("full-shadow LSE temperature floor must be positive")
    if args.full_shadow_projection_damping < 0.0:
        parser.error("full-shadow projection damping must be non-negative")
    if not 0.0 < args.activation_relative_improvement < 1.0:
        parser.error(
            "--activation-relative-improvement must be in (0, 1)"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
