"""Read-only correction for RPA-LRTA H/I direction reporting.

This audit repairs only the direction-alignment reporting defect in the frozen
RPA-LRTA v2 formal result. It does not train, select, tune, or modify any
model, metric, case, threshold, or production component.

Scientific question preserved from the original formal evaluator:

    Is the frozen RPA raw geometric action better aligned than the frozen RCSP
    raw geometric action with the negative temporal gradient at the exact RCSP
    current point?

Both actions are compared to the SAME RCSP current-point temporal gradient.
The authoritative H/I space remains raw_all_geometry, matching the original
RPA evaluator, which called alignment_stats directly on the raw 75D action and
gradient without applying a support mask.

The gradient is recomputed correctly by treating the frozen RCSP raw output as
a detached leaf tensor and passing that leaf through the unchanged production
decoder and observable temporal objective. No model parameter participates in
autograd.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training import motion_models as m
from training import refiner_boundary_crossing_temporal_reduction_intervention as bctr
from training import refiner_cross_width_normalization_audit as phase2
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_role_phase_anatomy_low_rank_tangent_adaptation as rpa
from training import refiner_safe_start_diagnostics as safe
from training import refiner_support_extent_direction_rotation_intervention as secdr
from training import refiner_temporal_action_alignment_audit as alignment

SCHEMA = "refiner_rpa_lrta_direction_reporting_correction_v1"
CORRECTION_PARENT_COMMIT = "ca8c313bea7870b81cbe4e02ab6ba7c39741764d"
FORMAL_RPA_SCHEMA = (
    "refiner_role_phase_anatomy_low_rank_tangent_adaptation_experiment_v2"
)
FORMAL_RPA_DECISION = "RPA_LRTA_NOT_SUPPORTED"
FORMAL_RPA_NEXT_ACTION = (
    "reject_rpa_lrta_candidate_without_additional_architecture_search"
)
FREEZE_SCHEMA = "refiner_rpa_lrta_v2_result_freeze_v1"

EXPECTED_RPA_REPORT_SHA256 = (
    "08fd36d5bd504a16cb5f18348358e8e236008e0758481e7c5372dddca0c6808e"
)
EXPECTED_RPA_ADAPTER_SHA256 = (
    "2b6a7ae7d08721bcff5b174403a7137ec7494c2f871a21ffc5a63bdc7be70110"
)
EXPECTED_RPA_UPDATES_SHA256 = (
    "aedcf96068976ead5988d055af248b067e641849214365ba9fea3fdee35f0a86"
)

PRIMARY_SPACE = "raw_all_geometry"
SPACES = (
    "raw_all_geometry",
    "raw_supported_geometry",
    "soft_masked_supported_geometry",
)
FINAL_CASES = 64
FINAL_GROUP_CASES = 8
WIDTHS = (10, 28)
SUMMARY_SCOPES = rpa.SUMMARY_SCOPES

LEGACY_CORE_STRENGTH = rpa.LEGACY_CORE_STRENGTH
LEGACY_TRANSITION_STRENGTH = rpa.LEGACY_TRANSITION_STRENGTH
PARITY_ATOL_CPU = alignment.PARITY_ATOL_CPU
PARITY_ATOL_CUDA = alignment.PARITY_ATOL_CUDA


def _finite(value: Any, label: str) -> float:
    result = float(value.detach()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite {label}")
    return result


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _median(values: Iterable[Any]) -> float | None:
    selected = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(selected)) if selected else None


def _max_abs(left: torch.Tensor, right: torch.Tensor, label: str) -> float:
    if left.shape != right.shape:
        raise ValueError(
            f"{label} shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    return _finite((left.detach() - right.detach()).abs().max(), label)


def _state_hash(module: torch.nn.Module) -> str:
    return safe.state_hash(module.state_dict())


def _validate_frozen_inputs(
    report_path: Path,
    adapter_path: Path,
    freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    report_hash = _file_sha256(report_path)
    adapter_hash = _file_sha256(adapter_path)
    freeze_hash = _file_sha256(freeze_path)

    if report_hash != EXPECTED_RPA_REPORT_SHA256:
        raise ValueError(
            "formal RPA report SHA256 mismatch: "
            f"{report_hash} != {EXPECTED_RPA_REPORT_SHA256}"
        )
    if adapter_hash != EXPECTED_RPA_ADAPTER_SHA256:
        raise ValueError(
            "formal RPA adapter SHA256 mismatch: "
            f"{adapter_hash} != {EXPECTED_RPA_ADAPTER_SHA256}"
        )

    report = _load_json(report_path)
    freeze = _load_json(freeze_path)

    if report.get("schema") != FORMAL_RPA_SCHEMA or report.get("completed") is not True:
        raise ValueError("formal RPA report schema/completion mismatch")
    if report.get("decision", {}).get("result") != FORMAL_RPA_DECISION:
        raise ValueError("formal RPA decision is not the frozen NOT_SUPPORTED result")
    if report.get("decision", {}).get("next_action") != FORMAL_RPA_NEXT_ACTION:
        raise ValueError("formal RPA next_action changed")
    if report.get("fixed_final_case_count") != FINAL_CASES:
        raise ValueError("formal RPA report is not final64")
    if report.get("scientific_acceptance") is not False:
        raise ValueError("formal RPA scientific_acceptance must remain false")
    if report.get("pilot_allowed") is not False:
        raise ValueError("formal RPA pilot_allowed must remain false")

    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("completed") is not True:
        raise ValueError("RPA freeze manifest schema/completion mismatch")
    if freeze.get("source_decision") != FORMAL_RPA_DECISION:
        raise ValueError("freeze manifest decision mismatch")
    if freeze.get("source_next_action") != FORMAL_RPA_NEXT_ACTION:
        raise ValueError("freeze manifest next_action mismatch")
    if freeze.get("read_only") is not True:
        raise ValueError("freeze manifest is not read-only")

    artifacts = freeze.get("artifacts", {})
    if artifacts.get("report", {}).get("sha256") != EXPECTED_RPA_REPORT_SHA256:
        raise ValueError("freeze manifest report SHA mismatch")
    if artifacts.get("adapter", {}).get("sha256") != EXPECTED_RPA_ADAPTER_SHA256:
        raise ValueError("freeze manifest adapter SHA mismatch")
    if artifacts.get("updates", {}).get("sha256") != EXPECTED_RPA_UPDATES_SHA256:
        raise ValueError("freeze manifest updates SHA mismatch")

    formal_artifact = report.get("artifacts", {}).get("rpa_adapter_final", {})
    if formal_artifact.get("sha256") != EXPECTED_RPA_ADAPTER_SHA256:
        raise ValueError("formal report adapter SHA mismatch")

    original_conditions = report.get("decision", {}).get("conditions", {})
    required = {
        "A_single_seen_rescue",
        "B_single_new_rescue",
        "C_cross28_seen_effectiveness",
        "D_cross28_new_effectiveness",
        "E_no_temporal_regression",
        "F_no_endpoint_regression",
        "G_no_safety_regression",
        "H_single_direction_improved",
        "I_cross28_direction_improved",
    }
    if not required.issubset(original_conditions):
        raise ValueError("formal RPA report lacks complete A-I conditions")

    # A/B/F/G independently make the method-level rejection invariant to H/I.
    if original_conditions["A_single_seen_rescue"] is not False:
        raise ValueError("frozen A is not false")
    if original_conditions["B_single_new_rescue"] is not False:
        raise ValueError("frozen B is not false")
    if original_conditions["F_no_endpoint_regression"] is not False:
        raise ValueError("frozen F is not false")
    if original_conditions["G_no_safety_regression"] is not False:
        raise ValueError("frozen G is not false")

    # The correction applies because original H/I statistics were undefined,
    # not because measured finite cosines showed no gain.
    for scope in SUMMARY_SCOPES:
        summary = report.get("summaries", {}).get(scope)
        if not isinstance(summary, dict):
            raise TypeError(f"formal RPA report lacks summary scope {scope}")
        if summary.get("median_direction_cosine_RCSP") is not None:
            raise ValueError(f"original RCSP direction cosine defined in {scope}")
        if summary.get("median_direction_cosine_RPA") is not None:
            raise ValueError(f"original RPA direction cosine defined in {scope}")

    if original_conditions["H_single_direction_improved"] is not False:
        raise ValueError("original H is unexpectedly true")
    if original_conditions["I_cross28_direction_improved"] is not False:
        raise ValueError("original I is unexpectedly true")

    return report, freeze, {
        "rpa_report_sha256": report_hash,
        "rpa_adapter_sha256": adapter_hash,
        "freeze_manifest_sha256": freeze_hash,
    }


def _formal_case_identities(report: Mapping[str, Any]) -> list[str]:
    rows = report.get("case_level")
    if not isinstance(rows, list) or len(rows) != FINAL_CASES:
        raise ValueError("formal RPA report case_level is not final64")
    identities = [str(row.get("identity")) for row in rows]
    if len(set(identities)) != FINAL_CASES:
        raise ValueError("formal RPA case identities are not unique")
    return identities


def _load_adapter_checkpoint(
    path: Path,
    model: rpa.FrozenRCSPRPARefiner,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("RPA adapter artifact root must be a dictionary")
    if payload.get("schema") != FORMAL_RPA_SCHEMA:
        raise ValueError("RPA adapter artifact schema mismatch")
    if payload.get("runtime_commit") != CORRECTION_PARENT_COMMIT:
        raise ValueError("RPA adapter artifact runtime commit mismatch")
    if payload.get("parameter_count") != 4692:
        raise ValueError("RPA adapter artifact parameter count mismatch")
    if payload.get("fixed_final_state_not_selected") is not True:
        raise ValueError("RPA adapter is not marked as fixed non-selected final state")

    state = payload.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise TypeError("RPA adapter artifact lacks adapter_state_dict")
    model.adapter.load_state_dict(state, strict=True)

    expected_state_hash = report["state_integrity"]["rpa_adapter_sha256_final"]
    actual_state_hash = _state_hash(model.adapter)
    if actual_state_hash != expected_state_hash:
        raise ValueError(
            "loaded RPA adapter state hash mismatch: "
            f"{actual_state_hash} != {expected_state_hash}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()

    return {
        "schema": payload["schema"],
        "runtime_commit": payload["runtime_commit"],
        "attempted_step": int(payload["step"]),
        "attempt_budget": int(payload["attempt_budget"]),
        "termination_reason": payload["termination_reason"],
        "parameter_count": int(payload["parameter_count"]),
        "state_sha256": actual_state_hash,
        "all_model_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
    }


def _load_lineage_source(
    report: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    provenance = report["provenance"]
    phase21_path = Path(provenance["phase21_source"]).resolve()
    bctr_path = Path(provenance["bctr_source"]).resolve()

    _phase21_report, phase21_hash, lineage_paths, upstream = (
        bctr._validate_phase21_lineage(phase21_path)
    )
    _bctr_report, bctr_hash = secdr._validate_bctr_report(
        bctr_path,
        phase21_path,
        phase21_hash,
    )
    if phase21_hash != provenance["phase21_sha256"]:
        raise ValueError("phase21 SHA differs from formal RPA provenance")
    if bctr_hash != provenance["bctr_sha256"]:
        raise ValueError("BCTR SHA differs from formal RPA provenance")

    (
        trajectory,
        _trajectory_paths,
        _trajectory_hashes,
        trajectory_report,
        experiment,
        _checkpoint,
    ) = failure._load_trajectory(
        lineage_paths["trajectory"],
        failure.TRAJECTORY_COMMIT,
    )

    state, bank, cfg, source_metadata = group_audit.load_frozen_source(
        lineage_paths["source"],
        group_audit.LEGACY_COMMIT,
        legacy_core_strength=LEGACY_CORE_STRENGTH,
        legacy_transition_strength=LEGACY_TRANSITION_STRENGTH,
    )
    if (
        experiment.get("source", {}).get("source_sha256")
        != source_metadata["source_sha256"]
    ):
        raise ValueError("trajectory/source provenance mismatch")

    cfg = dataclasses.replace(cfg, device=str(device))
    return {
        "phase21_path": phase21_path,
        "phase21_sha256": phase21_hash,
        "bctr_path": bctr_path,
        "bctr_sha256": bctr_hash,
        "lineage_paths": lineage_paths,
        "upstream": upstream,
        "trajectory": trajectory,
        "trajectory_report": trajectory_report,
        "state": state,
        "bank": bank,
        "cfg": cfg,
        "source_metadata": source_metadata,
    }


def _load_frozen_models_and_final64(
    context: Mapping[str, Any],
    report: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    lineage_paths = context["lineage_paths"]
    state = context["state"]
    bank = context["bank"]
    cfg = context["cfg"]

    base, rcsp_model, base_hash, rcsp_hash = rpa._load_models(
        lineage_paths,
        context["upstream"],
        context["trajectory_report"],
        lineage_paths["source"],
        state,
        bank,
        cfg,
        device,
    )

    probe, probe_hash = safe.load_probe(
        lineage_paths["source"],
        state,
        bank,
        cfg,
    )
    final_batch, final_metadata = alignment.combine_final_banks(
        failure.final_banks(bank, probe, cfg)
    )
    final_batch = rcsp._move_batch(final_batch, device)
    final_batch["role_id"] = rcsp.role_ids_from_metadata(
        final_metadata,
        device,
    )
    phase2._validate_fixed_metadata(final_metadata)

    if (
        len(final_metadata) != FINAL_CASES
        or int(final_batch["clean"].shape[0]) != FINAL_CASES
    ):
        raise RuntimeError("direction correction reconstructed a non-final64 cohort")

    identities = [phase2._identity_key(meta) for meta in final_metadata]
    if identities != _formal_case_identities(report):
        raise RuntimeError("reconstructed final64 identities/order differ from formal RPA")

    group_counts: dict[str, int] = {}
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            for width in WIDTHS:
                name = f"{split}/{role}/{width}"
                count = sum(
                    meta["split"] == split
                    and meta["role"] == role
                    and int(meta["width"]) == width
                    for meta in final_metadata
                )
                group_counts[name] = count
                if count != FINAL_GROUP_CASES:
                    raise RuntimeError(
                        f"direction correction final group {name} has {count}, not 8"
                    )

    loaded = {
        "base_state_sha256": base_hash,
        "rcsp_adapter_state_sha256": rcsp_hash,
        "probe_bank_sha256": probe_hash,
        "group_counts": group_counts,
    }
    return base, rcsp_model, final_metadata, final_batch, loaded


def _raw_outputs(
    model: rpa.FrozenRCSPRPARefiner,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = int(batch["clean"].shape[0])

    with torch.no_grad():
        rcsp_prediction, _rcsp_identity = rpa.rpa_batch_outputs(
            model,
            batch,
            cfg,
            capture_details=True,
            mode="rcsp",
        )
        rcsp_details = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in model.last_details.items()
        }
        model.clear_last_details()

        _rpa_prediction, _rpa_identity = rpa.rpa_batch_outputs(
            model,
            batch,
            cfg,
            capture_details=True,
            mode="rpa",
        )
        rpa_details = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in model.last_details.items()
        }
        model.clear_last_details()

    raw_rcsp = rcsp_details["raw_rcsp"][:count]
    raw_rpa = rpa_details["raw_rpa"][:count]
    expected_shape = (count, int(batch["bad"].shape[1]), m.PRODUCT_STATE_DIM)
    if raw_rcsp.shape != expected_shape or raw_rpa.shape != expected_shape:
        raise ValueError("captured RCSP/RPA raw output shape mismatch")
    if not bool(torch.isfinite(raw_rcsp).all()):
        raise FloatingPointError("nonfinite frozen RCSP raw output")
    if not bool(torch.isfinite(raw_rpa).all()):
        raise FloatingPointError("nonfinite frozen RPA raw output")
    if not torch.equal(raw_rcsp[..., :4], raw_rpa[..., :4]):
        raise RuntimeError("RPA direction correction observed changed raw contacts")
    return raw_rcsp, raw_rpa, rcsp_prediction


def _rcsp_leaf_temporal_gradient(
    raw_rcsp: torch.Tensor,
    rcsp_prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute d temporal-deficit / d raw geometry at exact frozen RCSP point."""
    raw_leaf = raw_rcsp.detach().clone().requires_grad_(True)
    masks = m._refiner_decode_masks(
        batch["joint"],
        batch["root"],
        batch["contact"],
        batch["seam"],
        cfg,
    )
    decoded = m._decode_product_refiner_output(
        batch["bad"],
        raw_leaf,
        *masks,
        cfg,
    )
    parity_error = _max_abs(
        decoded.detach(),
        rcsp_prediction.detach(),
        "RCSP leaf/manual decoder parity",
    )
    tolerance = PARITY_ATOL_CUDA if decoded.is_cuda else PARITY_ATOL_CPU
    if parity_error > tolerance:
        raise RuntimeError(
            "RCSP leaf/manual decoder parity failed: "
            f"max_abs={parity_error} tolerance={tolerance}"
        )

    _loss, terms = m._observable_refiner_objective(
        decoded,
        batch["bad"],
        batch["seam"],
        cfg,
        reduction="none",
    )
    temporal_values = terms["temporal_scientific_deficit"]
    if temporal_values.shape != (raw_rcsp.shape[0],):
        raise ValueError("temporal scientific deficit is not one scalar per case")

    temporal_total = temporal_values.sum()
    if not temporal_total.requires_grad:
        raise RuntimeError("RCSP leaf temporal objective unexpectedly lacks autograd")

    gradient = torch.autograd.grad(
        temporal_total,
        raw_leaf,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    gradient_geometry = gradient[..., 4:]
    if gradient_geometry.shape != raw_rcsp[..., 4:].shape:
        raise ValueError("RCSP leaf temporal gradient geometry shape mismatch")
    if not bool(torch.isfinite(gradient_geometry).all()):
        raise FloatingPointError("nonfinite RCSP leaf temporal gradient")

    return gradient_geometry.detach(), temporal_values.detach(), {
        "raw_leaf_requires_grad": bool(raw_leaf.requires_grad),
        "temporal_total_requires_grad": bool(temporal_total.requires_grad),
        "decoder_parity_verified": True,
        "decoder_parity_max_abs_error": parity_error,
        "decoder_parity_atol": tolerance,
        "model_parameter_autograd_used": False,
        "gradient_target": "detached RCSP raw output leaf",
        "gradient_definition": (
            "d sum(temporal_scientific_deficit) / d raw_geometry "
            "at exact RCSP current point"
        ),
    }


def _space_vectors(
    action: torch.Tensor,
    gradient: torch.Tensor,
    soft_mask: torch.Tensor,
    space: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action.shape != gradient.shape or action.shape != soft_mask.shape:
        raise ValueError("alignment action/gradient/mask shapes differ")
    if action.shape[-1] != 75:
        raise ValueError("direction correction requires 75D geometry")
    if space == "raw_all_geometry":
        return action, gradient
    support = soft_mask > 0
    if space == "raw_supported_geometry":
        return action * support, gradient * support
    if space == "soft_masked_supported_geometry":
        return action * soft_mask, gradient * soft_mask
    raise ValueError(f"unknown direction space: {space}")


def _alignment_rows(
    action: torch.Tensor,
    gradient: torch.Tensor,
) -> list[dict[str, Any]]:
    if action.shape != gradient.shape or action.ndim != 3:
        raise ValueError("direction vectors must have identical [case,frame,75] shape")
    a = action.detach().double().reshape(action.shape[0], -1)
    g = gradient.detach().double().reshape(gradient.shape[0], -1)
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(g).all()):
        raise FloatingPointError("nonfinite direction vector")

    rows: list[dict[str, Any]] = []
    for index in range(a.shape[0]):
        av = a[index]
        gv = g[index]
        action_norm = _finite(av.norm(), "action norm")
        gradient_norm = _finite(gv.norm(), "gradient norm")
        directional_derivative = _finite(
            torch.dot(gv, av),
            "directional derivative",
        )
        signed_descent_dot = -directional_derivative
        cosine = None
        if action_norm != 0.0 and gradient_norm != 0.0:
            cosine = max(
                -1.0,
                min(1.0, signed_descent_dot / (action_norm * gradient_norm)),
            )
        rows.append(
            {
                "action_norm": action_norm,
                "gradient_norm": gradient_norm,
                "cosine_to_negative_gradient": cosine,
                "directional_derivative": directional_derivative,
                "signed_descent_dot": signed_descent_dot,
                "local_descent": directional_derivative < 0.0,
                "local_ascent": directional_derivative > 0.0,
                "local_flat": directional_derivative == 0.0,
                "exact_zero_action": action_norm == 0.0,
                "exact_zero_gradient": gradient_norm == 0.0,
            }
        )
    return rows


def _scope_rows(
    rows: list[Mapping[str, Any]],
    scope: str,
) -> list[Mapping[str, Any]]:
    if scope == "overall":
        return list(rows)
    if scope == "single_recording":
        return [row for row in rows if row["role"] == "single_recording"]
    if scope == "cross_event":
        return [row for row in rows if row["role"] == "cross_event"]
    if scope == "width10":
        return [row for row in rows if int(row["width"]) == 10]
    if scope == "width28":
        return [row for row in rows if int(row["width"]) == 28]
    if scope == "seen":
        return [row for row in rows if row["split"] == "seen"]
    if scope == "new_position":
        return [row for row in rows if row["split"] == "new_position"]

    selected = list(rows)
    for part in scope.split("/"):
        if part in ("seen", "new_position"):
            selected = [row for row in selected if row["split"] == part]
        elif part in ("single_recording", "cross_event"):
            selected = [row for row in selected if row["role"] == part]
        elif part in ("10", "28"):
            selected = [row for row in selected if int(row["width"]) == int(part)]
        else:
            raise ValueError(f"unknown direction summary scope: {scope}")
    return selected


def _summary(
    rows: list[Mapping[str, Any]],
    scope: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"empty direction summary scope: {scope}")

    spaces: dict[str, Any] = {}
    for space in SPACES:
        rcsp_stats = [row["spaces"][space]["RCSP"] for row in rows]
        rpa_stats = [row["spaces"][space]["RPA"] for row in rows]
        spaces[space] = {
            "cases": len(rows),
            "defined_cosine_RCSP": sum(
                item["cosine_to_negative_gradient"] is not None
                for item in rcsp_stats
            ),
            "defined_cosine_RPA": sum(
                item["cosine_to_negative_gradient"] is not None
                for item in rpa_stats
            ),
            "zero_gradient_cases": sum(
                item["exact_zero_gradient"] for item in rcsp_stats
            ),
            "zero_rcsp_action_cases": sum(
                item["exact_zero_action"] for item in rcsp_stats
            ),
            "zero_rpa_action_cases": sum(
                item["exact_zero_action"] for item in rpa_stats
            ),
            "median_cosine_RCSP": _median(
                item["cosine_to_negative_gradient"] for item in rcsp_stats
            ),
            "median_cosine_RPA": _median(
                item["cosine_to_negative_gradient"] for item in rpa_stats
            ),
            "median_signed_descent_dot_RCSP": _median(
                item["signed_descent_dot"] for item in rcsp_stats
            ),
            "median_signed_descent_dot_RPA": _median(
                item["signed_descent_dot"] for item in rpa_stats
            ),
            "median_action_norm_RCSP": _median(
                item["action_norm"] for item in rcsp_stats
            ),
            "median_action_norm_RPA": _median(
                item["action_norm"] for item in rpa_stats
            ),
            "median_gradient_norm": _median(
                item["gradient_norm"] for item in rcsp_stats
            ),
            "local_descent_RCSP": sum(item["local_descent"] for item in rcsp_stats),
            "local_descent_RPA": sum(item["local_descent"] for item in rpa_stats),
            "local_ascent_RCSP": sum(item["local_ascent"] for item in rcsp_stats),
            "local_ascent_RPA": sum(item["local_ascent"] for item in rpa_stats),
        }

    return {
        "scope": scope,
        "cases": len(rows),
        "primary_space": PRIMARY_SPACE,
        "primary": spaces[PRIMARY_SPACE],
        "spaces": spaces,
    }


def _condition_improved(summary: Mapping[str, Any]) -> bool:
    primary = summary["primary"]
    rcsp_median = primary["median_cosine_RCSP"]
    rpa_median = primary["median_cosine_RPA"]
    return bool(
        primary["defined_cosine_RCSP"] > 0
        and primary["defined_cosine_RPA"] > 0
        and rcsp_median is not None
        and rpa_median is not None
        and rpa_median > rcsp_median
    )


def _decision_with_corrected_hi(
    report: Mapping[str, Any],
    corrected_h: bool,
    corrected_i: bool,
) -> dict[str, Any]:
    conditions = dict(report["decision"]["conditions"])
    conditions["H_single_direction_improved"] = bool(corrected_h)
    conditions["I_cross28_direction_improved"] = bool(corrected_i)

    total_rescues = int(
        report["decision"]["total_temporal_newly_rescued_vs_RCSP"]
    )
    net_gate_improvement = bool(report["decision"]["net_gate_improvement"])

    if all(conditions.values()):
        result = "RPA_LRTA_CANDIDATE_ADVANCE_REVIEW"
    elif conditions["G_no_safety_regression"] and net_gate_improvement:
        result = "RPA_LRTA_PARTIAL_DIAGNOSTIC_SUCCESS"
    elif (
        conditions["H_single_direction_improved"]
        or conditions["I_cross28_direction_improved"]
    ) and total_rescues == 0:
        result = "RPA_LRTA_MECHANISM_ONLY"
    else:
        result = "RPA_LRTA_NOT_SUPPORTED"

    if result != FORMAL_RPA_DECISION:
        raise RuntimeError(
            "corrected H/I unexpectedly change the frozen method-level decision"
        )

    return {
        "conditions_with_corrected_HI": conditions,
        "result": result,
        "decision_invariant": True,
        "invariance_reason": (
            "Frozen A=false, B=false, F=false, G=false independently block "
            "candidate advance; G=false blocks PARTIAL; one temporal rescue "
            "blocks the MECHANISM_ONLY branch."
        ),
    }


def _interpret_direction(
    h_value: bool,
    i_value: bool,
    single_summary: Mapping[str, Any],
    cross28_summary: Mapping[str, Any],
) -> dict[str, Any]:
    single_defined = (
        single_summary["primary"]["defined_cosine_RCSP"] > 0
        and single_summary["primary"]["defined_cosine_RPA"] > 0
    )
    cross_defined = (
        cross28_summary["primary"]["defined_cosine_RCSP"] > 0
        and cross28_summary["primary"]["defined_cosine_RPA"] > 0
    )
    if not single_defined or not cross_defined:
        classification = "RPA_DIRECTION_REPORTING_REMAINS_UNRESOLVED"
    elif h_value or i_value:
        classification = (
            "RPA_DIRECTION_MECHANISM_PRESENT_BUT_METHOD_REMAINS_UNSUPPORTED"
        )
    else:
        classification = "RPA_DIRECTION_REPORTING_CORRECTED_NO_TARGET_ALIGNMENT_GAIN"
    return {
        "classification": classification,
        "single_target_direction_defined": single_defined,
        "cross28_target_direction_defined": cross_defined,
        "H_single_direction_improved": bool(h_value),
        "I_cross28_direction_improved": bool(i_value),
        "formal_candidate_decision": FORMAL_RPA_DECISION,
        "candidate_reopened": False,
        "pilot_authorized": False,
        "new_architecture_search_authorized": False,
    }


def run(args: argparse.Namespace) -> int:
    report_path = Path(args.rpa_report).resolve()
    adapter_path = Path(args.rpa_adapter).resolve()
    freeze_path = Path(args.freeze_manifest).resolve()
    output = Path(args.output_dir).resolve()

    for path in (report_path, adapter_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report, _freeze, artifact_hashes = _validate_frozen_inputs(
        report_path,
        adapter_path,
        freeze_path,
    )

    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(
            "direction correction output must be a fresh empty directory"
        )
    frozen_paths = (report_path, adapter_path, freeze_path)
    if any(output == path or output.is_relative_to(path) for path in frozen_paths):
        raise FileExistsError("direction correction output overlaps a frozen input")

    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")

    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir = output / "result"
    result_dir.mkdir(exist_ok=False)
    failure_path = result_dir / "failure.json"

    frozen_hashes_before = {
        "rpa_report": _file_sha256(report_path),
        "rpa_adapter": _file_sha256(adapter_path),
        "freeze_manifest": _file_sha256(freeze_path),
    }

    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable; no silent CPU fallback"
            )

        context = _load_lineage_source(report, device)
        with group_audit.frozen_environment(
            context["state"]["fingerprint"],
            context["source_metadata"]["decoder_strengths"],
        ):
            base, rcsp_model, final_metadata, final_batch, loaded = (
                _load_frozen_models_and_final64(
                    context,
                    report,
                    device,
                )
            )
            rpa_model = rpa.FrozenRCSPRPARefiner(rcsp_model).to(device)
            adapter_info = _load_adapter_checkpoint(
                adapter_path,
                rpa_model,
                report,
            )

            base_before = _state_hash(base)
            rcsp_before = _state_hash(rcsp_model.adapter)
            rpa_before = _state_hash(rpa_model.adapter)

            all_rows: list[dict[str, Any]] = []
            decoder_parity: list[float] = []
            for start in range(0, FINAL_CASES, rcsp.FINAL_CHUNK_SIZE):
                stop = min(start + rcsp.FINAL_CHUNK_SIZE, FINAL_CASES)
                chunk = {
                    key: value[start:stop]
                    for key, value in final_batch.items()
                }
                metadata = final_metadata[start:stop]
                count = len(metadata)

                raw_rcsp, raw_rpa, rcsp_prediction = _raw_outputs(
                    rpa_model,
                    chunk,
                    context["cfg"],
                )
                gradient, temporal_values, gradient_info = (
                    _rcsp_leaf_temporal_gradient(
                        raw_rcsp,
                        rcsp_prediction,
                        chunk,
                        context["cfg"],
                    )
                )
                decoder_parity.append(
                    gradient_info["decoder_parity_max_abs_error"]
                )

                soft_mask = alignment._effective_geometry_mask(
                    chunk,
                    context["cfg"],
                ).detach()
                rcsp_action = raw_rcsp[..., 4:]
                rpa_action = raw_rpa[..., 4:]
                expected = (count, raw_rcsp.shape[1], 75)
                if (
                    rcsp_action.shape != expected
                    or rpa_action.shape != expected
                    or gradient.shape != expected
                    or soft_mask.shape != expected
                ):
                    raise ValueError("direction correction vector layout mismatch")

                stats: dict[
                    str,
                    tuple[list[dict[str, Any]], list[dict[str, Any]]],
                ] = {}
                for space in SPACES:
                    rcsp_a, rcsp_g = _space_vectors(
                        rcsp_action,
                        gradient,
                        soft_mask,
                        space,
                    )
                    rpa_a, rpa_g = _space_vectors(
                        rpa_action,
                        gradient,
                        soft_mask,
                        space,
                    )
                    if not torch.equal(rcsp_g, rpa_g):
                        raise RuntimeError(
                            "RCSP and RPA were not compared to the same gradient"
                        )
                    stats[space] = (
                        _alignment_rows(rcsp_a, rcsp_g),
                        _alignment_rows(rpa_a, rpa_g),
                    )

                for index, meta in enumerate(metadata):
                    spaces: dict[str, Any] = {}
                    for space in SPACES:
                        rcsp_rows, rpa_rows = stats[space]
                        spaces[space] = {
                            "RCSP": rcsp_rows[index],
                            "RPA": rpa_rows[index],
                        }
                    all_rows.append(
                        {
                            **meta,
                            "identity": phase2._identity_key(meta),
                            "temporal_scientific_deficit_at_RCSP": _finite(
                                temporal_values[index],
                                "temporal scientific deficit",
                            ),
                            "spaces": spaces,
                        }
                    )

            if len(all_rows) != FINAL_CASES:
                raise RuntimeError("direction correction did not produce final64")
            if [row["identity"] for row in all_rows] != _formal_case_identities(report):
                raise RuntimeError(
                    "direction correction final64 row identities changed"
                )

            summaries = {
                scope: _summary(_scope_rows(all_rows, scope), scope)
                for scope in SUMMARY_SCOPES
            }
            corrected_h = _condition_improved(summaries["single_recording"])
            corrected_i = _condition_improved(summaries["cross_event/28"])
            corrected_conditions = {
                "primary_space": PRIMARY_SPACE,
                "H_single_direction_improved": corrected_h,
                "I_cross28_direction_improved": corrected_i,
                "H_rule": (
                    "defined cosines exist and median RPA cosine > "
                    "median RCSP cosine in single_recording"
                ),
                "I_rule": (
                    "defined cosines exist and median RPA cosine > "
                    "median RCSP cosine in cross_event/28"
                ),
                "threshold_added": False,
            }
            decision_check = _decision_with_corrected_hi(
                report,
                corrected_h,
                corrected_i,
            )
            final_interpretation = _interpret_direction(
                corrected_h,
                corrected_i,
                summaries["single_recording"],
                summaries["cross_event/28"],
            )

            base_after = _state_hash(base)
            rcsp_after = _state_hash(rcsp_model.adapter)
            rpa_after = _state_hash(rpa_model.adapter)
            if base_after != base_before:
                raise RuntimeError("direction correction changed frozen base state")
            if rcsp_after != rcsp_before:
                raise RuntimeError("direction correction changed frozen RCSP state")
            if rpa_after != rpa_before:
                raise RuntimeError("direction correction changed frozen RPA state")
            if any(parameter.grad is not None for parameter in rpa_model.parameters()):
                raise RuntimeError(
                    "direction correction left gradient residue on model parameters"
                )

            frozen_hashes_after = {
                "rpa_report": _file_sha256(report_path),
                "rpa_adapter": _file_sha256(adapter_path),
                "freeze_manifest": _file_sha256(freeze_path),
            }
            if frozen_hashes_after != frozen_hashes_before:
                raise RuntimeError(
                    "a frozen direction-correction input changed during audit"
                )

            report_out = {
                "schema": SCHEMA,
                "completed": True,
                "runtime_commit": runtime_commit,
                "correction_parent_commit": CORRECTION_PARENT_COMMIT,
                "scope": {
                    "type": "read_only_reporting_correction",
                    "target": "RPA-LRTA H/I temporal direction statistics only",
                    "formal_method_decision_reopened": False,
                    "training_reopened": False,
                    "candidate_development_reopened": False,
                },
                "provenance": {
                    "rpa_report": str(report_path),
                    "rpa_adapter": str(adapter_path),
                    "freeze_manifest": str(freeze_path),
                    **artifact_hashes,
                    "phase21_report": str(context["phase21_path"]),
                    "phase21_sha256": context["phase21_sha256"],
                    "bctr_report": str(context["bctr_path"]),
                    "bctr_sha256": context["bctr_sha256"],
                    "trajectory": str(context["trajectory"]),
                    "source": str(context["lineage_paths"]["source"]),
                    "probe_bank_sha256": loaded["probe_bank_sha256"],
                    "group_counts": loaded["group_counts"],
                },
                "formal_result": {
                    "schema": report["schema"],
                    "decision": report["decision"]["result"],
                    "next_action": report["decision"]["next_action"],
                    "original_conditions": report["decision"]["conditions"],
                    "original_H_scientifically_interpretable": False,
                    "original_I_scientifically_interpretable": False,
                    "original_H_boolean": report["decision"]["conditions"][
                        "H_single_direction_improved"
                    ],
                    "original_I_boolean": report["decision"]["conditions"][
                        "I_cross28_direction_improved"
                    ],
                    "reason_original_HI_uninterpretable": (
                        "all original summary medians for RCSP and RPA direction "
                        "cosine were null; the formal boolean therefore encoded an "
                        "undefined statistic rather than measured no-gain evidence"
                    ),
                },
                "gradient_correction": {
                    "reference_point": "frozen RCSP current raw output",
                    "comparison_gradient_shared_by_RCSP_and_RPA": True,
                    "gradient_leaf_protocol": (
                        "raw_rcsp.detach().clone().requires_grad_(True)"
                    ),
                    "production_decoder_reused": True,
                    "observable_temporal_objective_reused": True,
                    "model_parameter_autograd_used": False,
                    "primary_space": PRIMARY_SPACE,
                    "primary_space_preserves_original_HI_definition": True,
                    "descriptive_secondary_spaces": [
                        space for space in SPACES if space != PRIMARY_SPACE
                    ],
                    "maximum_decoder_parity_error": max(decoder_parity),
                },
                "adapter": adapter_info,
                "fixed_final_case_count": FINAL_CASES,
                "case_level": all_rows,
                "summaries": summaries,
                "corrected_direction_conditions": corrected_conditions,
                "decision_invariance_check": decision_check,
                "final_interpretation": final_interpretation,
                "state_integrity": {
                    "base_state_sha256_before": base_before,
                    "base_state_sha256_after": base_after,
                    "rcsp_state_sha256_before": rcsp_before,
                    "rcsp_state_sha256_after": rcsp_after,
                    "rpa_state_sha256_before": rpa_before,
                    "rpa_state_sha256_after": rpa_after,
                    "base_unchanged": base_before == base_after,
                    "rcsp_unchanged": rcsp_before == rcsp_after,
                    "rpa_unchanged": rpa_before == rpa_after,
                    "frozen_inputs_unchanged": (
                        frozen_hashes_before == frozen_hashes_after
                    ),
                    "model_parameter_gradients_none": True,
                },
                "read_only": True,
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "parameter_update_performed": False,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "case_selection_performed": False,
                "metric_selection_performed": False,
                "architecture_selection_performed": False,
                "new_loss_introduced": False,
                "formal_candidate_decision": FORMAL_RPA_DECISION,
                "formal_next_action": FORMAL_RPA_NEXT_ACTION,
                "scientific_acceptance": False,
                "publish_allowed": False,
                "pilot_allowed": False,
                "production_model_modified": False,
                "production_inference_modified": False,
            }
            _exclusive_json(result_dir / "report.json", report_out)

            print(
                json.dumps(
                    {
                        "stage": (
                            "rpa_lrta_direction_reporting_correction_complete"
                        ),
                        "report": str(result_dir / "report.json"),
                        "fixed_final_cases": FINAL_CASES,
                        "primary_space": PRIMARY_SPACE,
                        "corrected_H": corrected_h,
                        "corrected_I": corrected_i,
                        "classification": final_interpretation["classification"],
                        "formal_candidate_decision": FORMAL_RPA_DECISION,
                        "pilot_allowed": False,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                flush=True,
            )
            return 0

    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "schema": SCHEMA,
                    "completed": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "read_only": True,
                    "optimizer_constructed": False,
                    "optimizer_steps": 0,
                    "parameter_update_performed": False,
                    "training_performed": False,
                    "formal_candidate_decision": FORMAL_RPA_DECISION,
                    "pilot_allowed": False,
                    "production_model_modified": False,
                    "production_inference_modified": False,
                },
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpa-report", required=True)
    parser.add_argument("--rpa-adapter", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
