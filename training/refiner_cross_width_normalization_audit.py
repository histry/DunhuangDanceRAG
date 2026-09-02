"""Read-only cross-event width and time-normalization audit.

This module evaluates the already frozen step-400 base and RCSP adapter at
alpha=1. It measures the production observable temporal metric, its exact
pre-reduction numerator/denominator terms, the unchanged decoder stages, and
the original temporal gate. It never creates an optimizer, updates a
parameter, changes a production helper, selects a checkpoint, or runs Pilot.

The scientific primary cohort is exactly the 32 cross-event cases in the
fixed final 64-case bank. The 32 single-recording cases are retained only as
an explicitly excluded control for frozen parity and state checks.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from motion_geometry.boundary_observables import boundary_metrics_torch
from training import motion_models as m
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_rcsp_single_direction_attribution as parameter_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_safe_start_diagnostics as safe
from training import refiner_single_direction_decomposition_audit as phase1
from training import refiner_temporal_action_alignment_audit as alignment


SCHEMA = "refiner_cross_width_normalization_audit_v1"
# Frozen upstream reports were produced from this formal baseline.  The
# Phase 2 implementation itself is allowed to live in a later audit commit;
# its current runtime commit is checked against --expected-main-commit.
FROZEN_ARTIFACT_COMMIT = "a9fbff524e46b0e13ab5e902f09c608e43cfb40f"
PARENT_COMMIT = "a33b17a78909bdf7125aa690d672f3991b7e5867"
PRIMARY_CASES = 32
FINAL_CASES = 64
CASES_PER_GROUP = 8
WIDTHS = (10, 28)
PRIMARY_ROLES = ("cross_event",)
EXCLUDED_ROLES = ("single_recording",)
GROUP_ORDER = (
    "seen/cross_event/10",
    "seen/cross_event/28",
    "new_position/cross_event/10",
    "new_position/cross_event/28",
)
PARITY_ATOL = 2.0e-6
PARITY_RTOL = 0.0
ZERO_DENOMINATOR = None


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


def _exclusive_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if float(denominator) != 0.0 else None


def _median(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def temporal_partition(active_frame_indices: Iterable[int]) -> dict[str, list[int]]:
    """Split ordered active frame indices into three deterministic thirds."""
    active = [int(index) for index in active_frame_indices]
    if not active:
        raise ValueError("active_frame_count == 0")
    if active != sorted(active) or len(active) != len(set(active)):
        raise ValueError("active frame indices must be ordered and unique")
    parts = np.array_split(np.asarray(active, dtype=np.int64), 3)
    result = {
        name: [int(value) for value in part.tolist()]
        for name, part in zip(("early", "center", "late"), parts)
    }
    if [value for name in ("early", "center", "late") for value in result[name]] != active:
        raise RuntimeError("temporal thirds are not complete and ordered")
    return result


def effective_weight_count(weights: torch.Tensor | np.ndarray) -> float | None:
    """Return (sum(w)^2)/sum(w^2), or null for a zero-mass weight vector."""
    values = torch.as_tensor(weights, dtype=torch.float64).reshape(-1)
    square_sum = _finite(values.square().sum(), "weight square sum")
    if square_sum == 0.0:
        return ZERO_DENOMINATOR
    return _finite(values.sum().square() / square_sum, "effective weight count")


def tensor_stage_stats(value: torch.Tensor, active_frame_count: int) -> dict[str, float | None]:
    """Describe a decoder stage without introducing a new scientific metric."""
    values = value.detach().double().reshape(-1)
    energy = _finite(values.square().sum(), "decoder stage energy")
    norm = _finite(values.norm(), "decoder stage norm")
    return {
        "l2_norm": norm,
        "rms": norm / math.sqrt(values.numel()) if values.numel() else 0.0,
        "max_abs": _finite(values.abs().max(), "decoder stage max abs") if values.numel() else 0.0,
        "energy_sum": energy,
        "energy_per_active_frame": _ratio(energy, active_frame_count),
    }


def _geometry_soft_mask(joint_weight: torch.Tensor, root_weight: torch.Tensor) -> torch.Tensor:
    if root_weight.ndim == 2:
        root_weight = root_weight.unsqueeze(-1)
    if joint_weight.ndim != 3 or joint_weight.shape[-1] != m.NUM_JOINTS:
        raise ValueError("joint weight must have shape [B,T,24]")
    root = root_weight.expand(root_weight.shape[:-1] + (3,))
    joints = joint_weight[..., None].expand(joint_weight.shape + (3,)).reshape(
        joint_weight.shape[:-1] + (m.NUM_JOINTS * 3,)
    )
    return torch.cat((root, joints), dim=-1).clamp(0.0, 1.0)


def _binary_support(joint_weight: torch.Tensor, root_weight: torch.Tensor) -> torch.Tensor:
    return (_geometry_soft_mask(joint_weight, root_weight) > 0).to(joint_weight.dtype)


def _temporal_component_tensors(
    joints: torch.Tensor,
    seam: torch.Tensor,
    fps: float,
) -> dict[str, dict[str, torch.Tensor | int | float]]:
    """Reproduce the two authoritative temporal reductions exactly."""
    if joints.ndim != 4 or joints.shape[-2:] != (m.NUM_JOINTS, 3):
        raise ValueError("joints must have shape [B,T,24,3]")
    core = seam[..., 0] if seam.ndim == 3 else seam
    core = core >= 0.5
    result: dict[str, dict[str, torch.Tensor | int | float]] = {}
    for order, name, scale in (
        (2, "seam_acceleration", 10.0),
        (3, "seam_jerk", 1000.0),
    ):
        length = joints.shape[1] - order
        if length <= 0:
            values = joints.new_zeros((joints.shape[0], 0), dtype=torch.float64)
            support = core[:, :0]
        else:
            coords = joints.to(torch.float64)
            values = torch.linalg.vector_norm(
                torch.diff(coords, n=order, dim=1) * float(fps) ** order,
                dim=-1,
            ).mean(-1)
            support = torch.stack(
                [core[:, index : index + length] for index in range(order + 1)]
            ).any(0)
        raw = (values * support.to(values.dtype)).sum(1)
        denominator = support.sum(1).clamp_min(1)
        normalized = raw / denominator.to(raw.dtype) / float(scale)
        result[name] = {
            "order": order,
            "scale": float(scale),
            "values": values,
            "support": support,
            "raw_numerator": raw,
            "denominator": denominator,
            "normalized_value": normalized,
        }
    return result


def authoritative_temporal_components(
    joints: torch.Tensor,
    seam: torch.Tensor,
    fps: float,
) -> dict[str, Any]:
    """Return exact numerator/denominator decomposition for one or more cases."""
    terms = _temporal_component_tensors(joints, seam, fps)
    acceleration = terms["seam_acceleration"]["normalized_value"]
    jerk = terms["seam_jerk"]["normalized_value"]
    return {"terms": terms, "temporal_energy": acceleration + jerk}


def temporal_reduction_parity(
    joints: torch.Tensor,
    seam: torch.Tensor,
    fps: float,
) -> dict[str, Any]:
    """Verify exact reconstruction against the authoritative metric helper."""
    components = authoritative_temporal_components(joints, seam, fps)
    official = boundary_metrics_torch(joints, seam, fps)
    reconstructed = components["temporal_energy"]
    max_error = _finite(
        (reconstructed - official["temporal_energy"]).abs().max(),
        "temporal reduction parity",
    )
    term_errors = {}
    for name, official_key in (
        ("seam_acceleration", "seam_acceleration_mps2"),
        ("seam_jerk", "seam_jerk_mps3"),
    ):
        value = components["terms"][name]["normalized_value"]
        expected = official[official_key] / float(components["terms"][name]["scale"])
        term_errors[name] = _finite((value - expected).abs().max(), f"{name} parity")
    return {
        "verified": max_error <= PARITY_ATOL and all(v <= PARITY_ATOL for v in term_errors.values()),
        "max_abs_error": max_error,
        "term_max_abs_error": term_errors,
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
    }


def _taper_from_binary_support(binary_support: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact read-only reproduction of production taper weights."""
    active = binary_support.to(torch.float64).transpose(1, 2)
    eroded = active.clone()
    distance = eroded.clone()
    for _ in range(int(radius)):
        eroded = -F.max_pool1d(-F.pad(eroded, (1, 1), mode="replicate"), 3, stride=1)
        distance = distance + eroded
    phase = distance / float(int(radius) + 1)
    taper = phase.pow(3) * (10.0 - 15.0 * phase + 6.0 * phase.square())
    return taper.transpose(1, 2)


def normalization_inventory(cfg: m.MotionGenerationConfig | None = None) -> dict[str, Any]:
    """Human-readable inventory of the current authoritative formulas."""
    fps = None if cfg is None else float(cfg.fps)
    temporal_gain = None if cfg is None else float(cfg.checkpoint_validation_min_temporal_repair_gain)
    return {
        "temporal_metric": {
            "symbolic_formula": "M_temporal = (sum(v2 * S2) / max(sum(S2),1)) / 10 + (sum(v3 * S3) / max(sum(S3),1)) / 1000",
            "implementation_source": "motion_geometry.boundary_observables.boundary_metrics_torch",
            "reduction_operator": "support-weighted sum divided by clamped support count, then linear weighted sum",
            "subterms": {
                "seam_acceleration": {
                    "symbolic_formula": "A = sum(||diff(J,2)*fps^2||_2.mean(joints) * S2) / max(sum(S2),1) / 10",
                    "numerator_definition": "sum over derivative stencils touching seam core of joint-vector norm mean",
                    "denominator_definition": "max(number of order-2 seam-touching stencils, 1)",
                    "reduction_operator": "sum / clamped count",
                    "order": 2,
                    "scale": "fps^2 / 10",
                },
                "seam_jerk": {
                    "symbolic_formula": "K = sum(||diff(J,3)*fps^3||_2.mean(joints) * S3) / max(sum(S3),1) / 1000",
                    "numerator_definition": "sum over derivative stencils touching seam core of joint-vector norm mean",
                    "denominator_definition": "max(number of order-3 seam-touching stencils, 1)",
                    "reduction_operator": "sum / clamped count",
                    "order": 3,
                    "scale": "fps^3 / 1000",
                },
            },
            "fps_used": fps,
            "duration_used": False,
            "dt_used": False,
            "frame_count_used": False,
            "active_frame_count_used": False,
            "valid_derivative_sample_count_used": True,
            "joint_count_used": True,
            "coordinate_count_used": False,
            "effective_weight_sum_used": False,
            "window_width_used": False,
        },
        "temporal_scientific_deficit": {
            "symbolic_formula": "D = Huber(relu(M - (1-g)*B) / max(abs(B), F), shoulder=g)",
            "implementation_source": "training.motion_models._observable_refiner_objective -> _smooth_observable_margin",
            "numerator_definition": "relu(candidate temporal energy - (1-gain)*degraded temporal energy)",
            "denominator_definition": "max(abs(degraded temporal energy), TRAIN-reference scale floor)",
            "reduction_operator": "per-case one-sided Huber; final batch aggregation is outside this audit",
            "gate_gain": temporal_gain,
            "duration_used": False,
            "dt_used": False,
            "frame_count_used": False,
            "active_frame_count_used": False,
            "effective_weight_sum_used": False,
            "window_width_used": False,
        },
        "gate": {
            "implementation_source": "motion_geometry.boundary_observables.observable_gate",
            "symbolic_formula": "relative temporal gain = (M_before-M_after)/M_before; pass if gain >= configured threshold and jerk non-regression",
            "threshold_source": "cfg.checkpoint_validation_min_temporal_repair_gain",
            "width_dependent": False,
        },
        "decoder_width_dependency": {
            "width_explicitly_used_in_objective": False,
            "width_explicitly_used_in_feature_normalization": False,
            "width_explicitly_used_in_decoder": False,
            "width_explicitly_used_in_support": False,
            "width_explicitly_used_in_gate": False,
            "note": "width is a cohort label and seam-derived case property; no width head or width-specific temporal range is introduced by this audit",
        },
    }


def _case_identity(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["split"]),
        str(row["role"]),
        int(row["width"]),
        int(row.get("bank_case_index", row.get("case_index", -1))),
    )


def _identity_key(row: Mapping[str, Any]) -> str:
    split, role, width, index = _case_identity(row)
    return f"{split}/{role}/{width}/{index}"


def _validate_fixed_metadata(metadata: list[dict[str, Any]]) -> None:
    if len(metadata) != FINAL_CASES:
        raise ValueError("fixed final metadata must contain 64 cases")
    counts = {_name: 0 for _name in GROUP_ORDER}
    excluded = 0
    for row in metadata:
        if row.get("role") not in (*PRIMARY_ROLES, *EXCLUDED_ROLES):
            raise ValueError("fixed final metadata has an unexpected role")
        width = int(row.get("width", -1))
        if width not in WIDTHS:
            raise ValueError("fixed final metadata has an unexpected width")
        key = f"{row['split']}/{row['role']}/{width}"
        if row["role"] == "cross_event":
            if key not in counts:
                raise ValueError("cross-event metadata group is unexpected")
            counts[key] += 1
        else:
            excluded += 1
    if counts != {key: CASES_PER_GROUP for key in GROUP_ORDER}:
        raise ValueError(f"primary cross-event groups are not 4x8: {counts}")
    if excluded != PRIMARY_CASES:
        raise ValueError("single-recording exclusion cohort must contain 32 cases")
    identities = [_identity_key(row) for row in metadata]
    if len(identities) != len(set(identities)):
        raise ValueError("fixed final metadata contains duplicate case identities")


def _metadata_scopes(metadata: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = [row for row in metadata if row["role"] == "cross_event"]
    excluded = [row for row in metadata if row["role"] == "single_recording"]
    if len(primary) != PRIMARY_CASES or len(excluded) != PRIMARY_CASES:
        raise ValueError("primary/excluded cohort counts are not 32/32")
    return primary, excluded


def _active_stats(
    binary_support: torch.Tensor,
    root_weight: torch.Tensor,
    joint_weight: torch.Tensor,
) -> dict[str, Any]:
    binary = binary_support.detach().double()
    root = root_weight.detach().double()
    joints = joint_weight.detach().double()
    active_frames = torch.nonzero(binary.any(dim=-1), as_tuple=False).flatten().tolist()
    root_frames = torch.nonzero((root > 0).any(dim=-1), as_tuple=False).flatten().tolist()
    joint_frames = torch.nonzero((joints > 0).any(dim=-1), as_tuple=False).flatten().tolist()
    frame_count = len(active_frames)
    total_coordinates = int(torch.count_nonzero(binary).item())
    return {
        "binary_active_frame_count": frame_count,
        "binary_active_coordinate_count": total_coordinates,
        "root_active_frame_count": len(root_frames),
        "joint_active_frame_count": len(joint_frames),
        "first_active_frame": int(active_frames[0]) if active_frames else None,
        "last_active_frame": int(active_frames[-1]) if active_frames else None,
        "active_span": int(active_frames[-1] - active_frames[0] + 1) if active_frames else 0,
        "active_frame_density": _ratio(frame_count, binary.shape[0]),
        "active_frame_indices": [int(index) for index in active_frames],
        "temporal_thirds": temporal_partition(active_frames) if active_frames else None,
        "root_active_frame_indices": [int(index) for index in root_frames],
        "joint_active_frame_indices": [int(index) for index in joint_frames],
    }


def _weight_stats(soft_mask: torch.Tensor, binary_support: torch.Tensor, taper: torch.Tensor) -> dict[str, Any]:
    weights = soft_mask.detach().double()
    binary = binary_support.detach().bool()
    selected = weights[binary]
    total = _finite(weights.sum(), "total soft weight mass")
    square = _finite(weights.square().sum(), "total soft weight square mass")
    return {
        "root_weight_sum": _finite(weights[..., :3].sum(), "root weight sum"),
        "joint_weight_sum": _finite(weights[..., 3:].sum(), "joint weight sum"),
        "total_weight_sum": total,
        "weight_square_sum": square,
        "effective_weight_count": effective_weight_count(weights),
        "weight_mean_on_binary_support": _finite(selected.mean(), "support weight mean") if selected.numel() else None,
        "weight_median_on_binary_support": _finite(selected.median(), "support weight median") if selected.numel() else None,
        "weight_max": _finite(selected.max(), "support weight max") if selected.numel() else 0.0,
        "taper_mass": _finite(taper.detach().double().sum(), "taper mass"),
        "taper_mean_on_binary_support": (
            _finite(taper.detach().double()[binary].mean(), "support taper mean")
            if bool(binary.any()) else None
        ),
    }


def _distribution_stats(values: torch.Tensor | np.ndarray, frame_count: int) -> dict[str, Any]:
    array = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if array.numel() == 0:
        return {
            "total_mass": 0.0, "peak_value": 0.0, "peak_index": None,
            "peak_normalized_position": None, "center_of_mass": None,
            "effective_temporal_support": None,
        }
    total = _finite(array.sum(), "temporal contribution total")
    peak_index = int(torch.argmax(array.abs()).item())
    abs_values = array.abs()
    abs_total = _finite(abs_values.sum(), "absolute temporal contribution total")
    center = (
        _finite(
            (torch.arange(array.numel(), dtype=torch.float64) * abs_values).sum()
            / abs_total,
            "temporal contribution center",
        ) if abs_total else None
    )
    square = _finite(array.square().sum(), "temporal contribution square sum")
    return {
        "total_mass": total,
        "peak_value": _finite(array[peak_index], "temporal contribution peak"),
        "peak_index": peak_index,
        "peak_normalized_position": _ratio(peak_index, max(frame_count - 1, 1)),
        "center_of_mass": center,
        "effective_temporal_support": _ratio(total * total, square) if square else None,
    }


def _temporal_distribution(
    components: dict[str, Any],
    active_indices: list[int],
    frame_count: int,
) -> dict[str, Any]:
    """Store native per-stencil terms and a start-index aligned exact sum."""
    aggregate = torch.zeros(frame_count, dtype=torch.float64)
    native: dict[str, Any] = {}
    for name in ("seam_acceleration", "seam_jerk"):
        term = components["terms"][name]
        values = term["values"][0].detach().double()
        support = term["support"][0].detach().double()
        scale = float(term["scale"])
        contribution = values * support / term["denominator"][0].double() / scale
        contribution = contribution.cpu()
        native[name] = {
            "derivative_order": int(term["order"]),
            "index_semantics": "derivative stencil start frame",
            "contribution": [float(value) for value in contribution.tolist()],
            "support": [bool(value) for value in support.bool().tolist()],
            "stats": _distribution_stats(contribution, frame_count),
        }
        aggregate[: contribution.numel()] += contribution
    result = {
        "total_mass": _finite(aggregate.sum(), "aggregate temporal mass"),
        "native_terms": native,
        "aligned_stencil_start_contribution": [float(value) for value in aggregate.tolist()],
        "stats": _distribution_stats(aggregate, frame_count),
        "active_indices_used_for_thirds": list(active_indices),
    }
    thirds = temporal_partition(active_indices) if active_indices else {"early": [], "center": [], "late": []}
    third_mass = {name: 0.0 for name in thirds}
    for name in ("seam_acceleration", "seam_jerk"):
        term = components["terms"][name]
        values = term["values"][0].detach().double().cpu()
        support = term["support"][0].detach().bool().cpu()
        scale = float(term["scale"])
        denominator = float(term["denominator"][0].detach().cpu())
        order = int(term["order"])
        for start, value in enumerate(values.tolist()):
            if not bool(support[start]):
                continue
            touched = [frame for frame in range(start, start + order + 1) if frame in active_indices]
            target = next((name for name, indices in thirds.items() if touched and touched[0] in indices), None)
            if target is not None:
                third_mass[target] += float(value) / denominator / scale
    result["thirds"] = third_mass
    return result


def _repair_distribution(base: dict[str, Any], rcsp_value: dict[str, Any], frame_count: int) -> dict[str, Any]:
    difference = np.asarray(base["aligned_stencil_start_contribution"], dtype=np.float64) - np.asarray(
        rcsp_value["aligned_stencil_start_contribution"], dtype=np.float64
    )
    positive = np.maximum(difference, 0.0)
    negative = np.minimum(difference, 0.0)
    abs_values = np.abs(difference)
    abs_total = float(abs_values.sum())
    square = float(np.square(difference).sum())
    center = (
        float((np.arange(len(difference), dtype=np.float64) * abs_values).sum() / abs_total)
        if abs_total else None
    )
    return {
        "repair_total": _finite(difference.sum(), "repair total"),
        "positive_repair_mass": _finite(positive.sum(), "positive repair mass"),
        "negative_repair_mass": _finite(negative.sum(), "negative repair mass"),
        "peak_positive_repair": _finite(positive.max(), "peak positive repair") if len(positive) else 0.0,
        "peak_negative_repair": _finite(negative.min(), "peak negative repair") if len(negative) else 0.0,
        "repair_center_of_mass": center,
        "repair_effective_support": _ratio(abs_total * abs_total, square) if square else None,
        "aligned_stencil_start_repair": [float(value) for value in difference.tolist()],
        "positive_repair_distribution": _distribution_stats(positive, frame_count),
        "negative_repair_distribution": _distribution_stats(negative, frame_count),
        "thirds": {
            name: float(base["thirds"].get(name, 0.0) - rcsp_value["thirds"].get(name, 0.0))
            for name in ("early", "center", "late")
        },
    }


def _direction_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("direction_alignment", {}).get("case_level", [])
    if len(rows) != FINAL_CASES:
        raise ValueError("RCSP direction_alignment.case_level must contain 64 rows")
    result = {_identity_key(row): row for row in rows}
    if len(result) != FINAL_CASES:
        raise ValueError("RCSP direction rows contain duplicate identities")
    return result


def _support_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = report.get("support_projection_stats", {}).get("case_level", [])
    if len(rows) != FINAL_CASES:
        raise ValueError("RCSP support_projection_stats.case_level must contain 64 rows")
    result = {_identity_key(row): row for row in rows}
    if len(result) != FINAL_CASES:
        raise ValueError("RCSP support rows contain duplicate identities")
    return result


def _report_case_map(report: Mapping[str, Any], state: str) -> dict[str, Mapping[str, Any]]:
    rows = report.get("fixed_final_64", {}).get(state, {}).get("case_level", [])
    if len(rows) != FINAL_CASES:
        raise ValueError(f"RCSP fixed_final_64.{state}.case_level must contain 64 rows")
    return {_identity_key(row): row for row in rows}


def _validate_json_report(path: Path, schema: str, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise ValueError(f"{label} schema mismatch")
    return value, _file_sha256(path)


def _validate_phase1(path: Path, expected_artifact_commit: str) -> tuple[dict[str, Any], str]:
    report, digest = _validate_json_report(path, phase1.SCHEMA, "Phase 1 report")
    if (
        report.get("completed") is not True
        or report.get("provenance", {}).get("runtime_commit") != expected_artifact_commit
        or report.get("optimizer_steps") != 0
        or report.get("parameter_update_performed") is not False
        or report.get("production_model_modified") is not False
        or report.get("pilot_allowed") is not False
    ):
        raise ValueError("Phase 1 lineage or read-only contract mismatch")
    return report, digest


def _validate_rcsp(rcsp_dir: Path) -> dict[str, Any]:
    report_path = rcsp_dir / "report.json"
    review_path = rcsp_dir / "reporting_logic_review_v1.json"
    report, report_hash = _validate_json_report(report_path, parameter_audit.RCSP_SOURCE_SCHEMA, "RCSP report")
    review, review_hash = _validate_json_report(review_path, parameter_audit.RCSP_REVIEW_SCHEMA, "RCSP review")
    false_fields = (
        "checkpoint_selection_performed", "scale_selection_performed",
        "production_model_modified", "production_inference_modified",
        "scientific_acceptance", "publish_allowed", "pilot_allowed",
    )
    if (
        report.get("completed") is not True
        or report.get("base_model_frozen") is not True
        or report.get("adapter_only_training") is not True
        or report.get("optimizer_steps") != rcsp.STEPS
        or any(report.get(field) is not False for field in false_fields)
        or review.get("completed") is not True
        or review.get("source_report", {}).get("sha256") != report_hash
        or review.get("measurement_recomputation_verified") is not True
        or review.get("production_model_modified") is not False
        or review.get("pilot_allowed") is not False
    ):
        raise ValueError("RCSP report/review lineage or read-only contract mismatch")
    descriptor = report.get("parameter_update_scope", {}).get("adapter_checkpoint", {})
    value = descriptor.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("RCSP report lacks adapter checkpoint path")
    adapter_path = Path(value).resolve()
    if adapter_path.parent != rcsp_dir.resolve() or not adapter_path.is_file():
        raise ValueError("RCSP adapter checkpoint is missing or outside result directory")
    adapter_hash = _file_sha256(adapter_path)
    if descriptor.get("sha256") != adapter_hash:
        raise ValueError("RCSP adapter checkpoint hash mismatch")
    checkpoint = m._trusted_torch_load(adapter_path, map_location="cpu")
    if (
        checkpoint.get("schema") != parameter_audit.RCSP_SOURCE_SCHEMA
        or checkpoint.get("completed_steps") != rcsp.STEPS
        or checkpoint.get("formal_checkpoint") is not False
        or checkpoint.get("production_model_modified") is not False
        or checkpoint.get("checkpoint_selection_performed") is not False
        or checkpoint.get("publish_allowed") is not False
        or checkpoint.get("pilot_allowed") is not False
        or checkpoint.get("resume_allowed") is not False
    ):
        raise ValueError("invalid diagnostic-only RCSP adapter checkpoint")
    return {
        "directory": rcsp_dir,
        "report_path": report_path,
        "review_path": review_path,
        "adapter_path": adapter_path,
        "hashes": {
            "report.json": report_hash,
            "reporting_logic_review_v1.json": review_hash,
            "adapter_checkpoint": adapter_hash,
        },
        "report": report,
        "review": review,
        "adapter_checkpoint": checkpoint,
    }


def _validate_parameter_report(path: Path, rcsp_report_hash: str) -> tuple[dict[str, Any], str]:
    report, digest = _validate_json_report(path, parameter_audit.SCHEMA, "parameter attribution report")
    if (
        report.get("completed") is not True
        or report.get("provenance", {}).get("rcsp_sha256", {}).get("report.json") != rcsp_report_hash
        or report.get("optimizer_steps") != 0
        or report.get("gradient_protocol", {}).get("parameter_update_performed") is not False
        or report.get("production_model_modified") is not False
        or report.get("pilot_allowed") is not False
    ):
        raise ValueError("parameter attribution lineage or read-only contract mismatch")
    return report, digest


def _state_file_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in paths.items()}


def _gate_margin(before: float, after: float, threshold: float, valid: bool) -> tuple[float, bool, dict[str, Any]]:
    floor = 1.0e-6
    gain = (before - after) / before if before > floor else (1.0 if after <= floor else -1.0)
    margin = gain - threshold
    return _finite(gain, "temporal gate gain") - threshold, bool(valid and margin >= 0.0), {
        "relative_gain": _finite(gain, "temporal gate gain"),
        "margin": _finite(margin, "temporal gate margin"),
        "threshold": float(threshold),
        "positive_margin_means": "relative temporal gain is at or above the original gate threshold",
        "negative_margin_means": "relative temporal gain remains below the original gate threshold",
        "valid": bool(valid),
        "jerk_non_regression": True,
    }


def _evaluate_chunk(
    base: torch.nn.Module,
    model: rcsp.FrozenBaseRCSPModel,
    batch: Mapping[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    cfg: m.MotionGenerationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks = m._refiner_decode_masks(batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)
    soft_mask = _geometry_soft_mask(masks[0], masks[1])
    binary = (soft_mask > 0).to(soft_mask.dtype)
    count = len(metadata)
    with torch.no_grad():
        # Reuse the authoritative bad+clean batch wrapper used by the frozen
        # RCSP report.  Running only the bad half changes GPU kernel/batch
        # behavior and can perturb the diagnostic adapter norms by a few ppm.
        base_trace: dict[str, Any] = {}
        base_prediction, _ = m._refiner_batch_outputs(
            base, batch, cfg, trace=base_trace
        )
        rcsp_trace: dict[str, Any] = {}
        rcsp_prediction, _ = rcsp.rcsp_batch_outputs(
            model, batch, cfg, trace=rcsp_trace, capture_details=True
        )
        details = model.last_details
        base_raw = base_trace["raw_output"][:count]
        rcsp_raw = details["raw_adapted"][:count]
        if not torch.equal(base_raw[..., :4], rcsp_raw[..., :4]):
            raise RuntimeError("RCSP changed contact channels")
        if _finite(
            (base_raw - details["raw_base"][:count]).abs().max(),
            "base raw parity",
        ) != 0.0:
            raise RuntimeError("RCSP wrapper base raw parity failed")
        traces: dict[str, dict[str, Any]] = {
            "BASE": base_trace["repair"],
            "RCSP": rcsp_trace["repair"],
        }
        predictions: dict[str, torch.Tensor] = {
            "BASE": base_prediction,
            "RCSP": rcsp_prediction,
        }
        _, base_terms = m._observable_refiner_objective(
            predictions["BASE"], batch["bad"], batch["seam"], cfg, reduction="none"
        )
        _, rcsp_terms = m._observable_refiner_objective(
            predictions["RCSP"], batch["bad"], batch["seam"], cfg, reduction="none"
        )
        metric_joints = m._observable_boundary_joints_torch(
            torch.cat([batch["bad"], predictions["BASE"], predictions["RCSP"]])
        )
        before_joints, base_joints, rcsp_joints = metric_joints.split(len(metadata))
        before_metrics = boundary_metrics_torch(before_joints, batch["seam"], cfg.fps)
        base_metrics = boundary_metrics_torch(base_joints, batch["seam"], cfg.fps)
        rcsp_metrics = boundary_metrics_torch(rcsp_joints, batch["seam"], cfg.fps)
        base_components = authoritative_temporal_components(base_joints, batch["seam"], cfg.fps)
        rcsp_components = authoritative_temporal_components(rcsp_joints, batch["seam"], cfg.fps)
    rows = []
    for index, meta in enumerate(metadata):
        active = _active_stats(binary[index], masks[1][index], masks[0][index])
        active_indices = active["active_frame_indices"]
        taper = _taper_from_binary_support(binary[index:index + 1], cfg.product_refiner_residual_taper_frames)[0]
        weights = _weight_stats(soft_mask[index], binary[index], taper)
        components = {}
        for name, source in (("BASE", base_components), ("RCSP", rcsp_components)):
            state_terms = {}
            for term_name in ("seam_acceleration", "seam_jerk"):
                term = source["terms"][term_name]
                state_terms[term_name] = {
                    "raw_numerator": _finite(term["raw_numerator"][index], f"{name} {term_name} numerator"),
                    "denominator": _finite(term["denominator"][index], f"{name} {term_name} denominator"),
                    "normalized_value": _finite(term["normalized_value"][index], f"{name} {term_name} normalized value"),
                    "order": int(term["order"]),
                    "scale": float(term["scale"]),
                }
            state_terms["temporal_energy"] = _finite(source["temporal_energy"][index], f"{name} temporal energy")
            components[name] = state_terms
        component_payload = {
            "terms": {
                key: {
                    **value,
                    "values": value["values"][index:index + 1],
                    "support": value["support"][index:index + 1],
                    "denominator": value["denominator"][index:index + 1],
                }
                for key, value in base_components["terms"].items()
            },
            "temporal_energy": base_components["temporal_energy"][index:index + 1],
        }
        base_distribution = _temporal_distribution(component_payload, active_indices, int(batch["bad"].shape[1]))
        component_payload["terms"] = {
            key: {
                **value,
                "values": value["values"][index:index + 1],
                "support": value["support"][index:index + 1],
                "denominator": value["denominator"][index:index + 1],
            }
            for key, value in rcsp_components["terms"].items()
        }
        component_payload["temporal_energy"] = rcsp_components["temporal_energy"][index:index + 1]
        rcsp_distribution = _temporal_distribution(component_payload, active_indices, int(batch["bad"].shape[1]))
        threshold = float(cfg.checkpoint_validation_min_temporal_repair_gain)
        valid_base = bool(before_metrics["valid"][index] and base_metrics["valid"][index])
        valid_rcsp = bool(before_metrics["valid"][index] and rcsp_metrics["valid"][index])
        base_margin, base_pass, base_gate = _gate_margin(
            _finite(before_metrics["temporal_energy"][index], "BASE before temporal energy"),
            _finite(base_metrics["temporal_energy"][index], "BASE after temporal energy"), threshold, valid_base,
        )
        rcsp_margin, rcsp_pass, rcsp_gate = _gate_margin(
            _finite(before_metrics["temporal_energy"][index], "RCSP before temporal energy"),
            _finite(rcsp_metrics["temporal_energy"][index], "RCSP after temporal energy"), threshold, valid_rcsp,
        )
        base_gate["jerk_non_regression"] = bool(
            base_metrics["seam_jerk_mps3"][index] <= before_metrics["seam_jerk_mps3"][index] * 1.02 + 1.0e-6
        )
        rcsp_gate["jerk_non_regression"] = bool(
            rcsp_metrics["seam_jerk_mps3"][index] <= before_metrics["seam_jerk_mps3"][index] * 1.02 + 1.0e-6
        )
        base_pass = bool(base_pass and base_gate["jerk_non_regression"])
        rcsp_pass = bool(rcsp_pass and rcsp_gate["jerk_non_regression"])
        raw_adapter = details["adapter_raw"][index].detach()
        projected_adapter = details["adapter_projected"][index].detach()
        projected_outside = projected_adapter * (1.0 - binary[index])
        if _finite(projected_outside.abs().max(), "projected outside support") != 0.0:
            raise RuntimeError("projected correction escaped binary support")
        stages = {}
        for state in ("BASE", "RCSP"):
            trace = traces[state]
            state_stages = {
                "raw_action": trace["raw"][index],
                "soft_weighted_action": trace["after_mask"][index],
                "smoothed_action": trace["after_smoothing"][index],
                "tapered_action": trace["after_taper"][index],
                "capped_action": trace["after_cap"][index],
                "final_tangent": trace["after_cap"][index],
                "final_decoded_geometric_displacement": trace["applied"][index],
            }
            stages[state] = {
                name: tensor_stage_stats(value, active["binary_active_frame_count"])
                for name, value in state_stages.items()
            }
        stages["RCSP"]["raw_adapter"] = tensor_stage_stats(raw_adapter, active["binary_active_frame_count"])
        stages["RCSP"]["binary_projected_adapter"] = tensor_stage_stats(projected_adapter, active["binary_active_frame_count"])
        repair_distribution = _repair_distribution(base_distribution, rcsp_distribution, int(batch["bad"].shape[1]))
        row = {
            **meta,
            "pair_key": None,
            "alpha": 1.0,
            "BASE": {
                "temporal_metric": _finite(base_metrics["temporal_energy"][index], "BASE temporal metric"),
                "temporal_deficit": _finite(base_terms["temporal_scientific_deficit"][index], "BASE temporal deficit"),
                "gate_margin": base_margin,
                "gate_pass": base_pass,
                "gate": base_gate,
                "components": components["BASE"],
                "temporal_error_distribution": base_distribution,
                "decoder_stages": stages["BASE"],
            },
            "RCSP": {
                "temporal_metric": _finite(rcsp_metrics["temporal_energy"][index], "RCSP temporal metric"),
                "temporal_deficit": _finite(rcsp_terms["temporal_scientific_deficit"][index], "RCSP temporal deficit"),
                "gate_margin": rcsp_margin,
                "gate_pass": rcsp_pass,
                "gate": rcsp_gate,
                "components": components["RCSP"],
                "temporal_error_distribution": rcsp_distribution,
                "decoder_stages": stages["RCSP"],
            },
            "temporal_metric_base": components["BASE"]["temporal_energy"],
            "temporal_metric_rcsp": components["RCSP"]["temporal_energy"],
            "temporal_deficit_base": _finite(base_terms["temporal_scientific_deficit"][index], "temporal deficit base"),
            "temporal_deficit_rcsp": _finite(rcsp_terms["temporal_scientific_deficit"][index], "temporal deficit rcsp"),
            "repair_gain": _finite(base_terms["temporal_scientific_deficit"][index] - rcsp_terms["temporal_scientific_deficit"][index], "authoritative temporal repair gain"),
            "authoritative_temporal_repair_gain": _finite(base_terms["temporal_scientific_deficit"][index] - rcsp_terms["temporal_scientific_deficit"][index], "authoritative temporal repair gain"),
            "temporal_metric_repair_gain": _finite(base_metrics["temporal_energy"][index] - rcsp_metrics["temporal_energy"][index], "temporal metric repair gain"),
            "gate_threshold": threshold,
            "gate_margin_base": base_margin,
            "gate_margin_rcsp": rcsp_margin,
            "gate_pass_base": base_pass,
            "gate_pass_rcsp": rcsp_pass,
            "raw_temporal_numerator": {
                term: {"BASE": components["BASE"][term]["raw_numerator"], "RCSP": components["RCSP"][term]["raw_numerator"]}
                for term in ("seam_acceleration", "seam_jerk")
            },
            "temporal_denominator": {
                term: {"BASE": components["BASE"][term]["denominator"], "RCSP": components["RCSP"][term]["denominator"]}
                for term in ("seam_acceleration", "seam_jerk")
            },
            "normalized_temporal_subterms": {
                term: {"BASE": components["BASE"][term]["normalized_value"], "RCSP": components["RCSP"][term]["normalized_value"]}
                for term in ("seam_acceleration", "seam_jerk")
            },
            "active_statistics": active,
            "effective_weight_statistics": weights,
            "support_retention_ratio": _ratio(
                # Match RCSP _support_case_rows exactly: cast each case to
                # float64 before the L2 reduction, rather than reducing the
                # original float32 adapter output.
                _finite(projected_adapter.double().norm(), "projected adapter norm"),
                _finite(raw_adapter.double().norm(), "raw adapter norm"),
            ),
            "projected_outside_support_max": _finite(projected_outside.abs().max(), "projected outside support max"),
            "decoder_stage_statistics": stages,
            "temporal_repair_distribution": repair_distribution,
            "temporal_thirds": {
                "BASE_error_mass": base_distribution["thirds"],
                "RCSP_error_mass": rcsp_distribution["thirds"],
                "repair_gain": repair_distribution["thirds"],
            },
            "direction_covariate_pending": {"adapter_cosine": None, "total_cosine": None},
            "authoritative_metric_source": "motion_geometry.boundary_observables.boundary_metrics_torch",
            "authoritative_deficit_source": "training.motion_models._observable_refiner_objective",
        }
        rows.append(row)
    model.clear_last_details()
    return rows, {
        "temporal_reduction_parity": {
            "BASE": temporal_reduction_parity(base_joints, batch["seam"], cfg.fps),
            "RCSP": temporal_reduction_parity(rcsp_joints, batch["seam"], cfg.fps),
        },
        "raw_base_wrapper_parity_max_abs": 0.0,
        "contact_channels_unchanged": True,
    }


def _attach_frozen_covariates(
    rows: list[dict[str, Any]],
    direction_rows: Mapping[str, Any],
    support_rows: Mapping[str, Any],
    tolerance: float = PARITY_ATOL,
) -> dict[str, Any]:
    mismatches = []
    for row in rows:
        key = _identity_key(row)
        direction = direction_rows[key]
        support = support_rows[key]
        adapter = direction.get("projected_adapter_delta_vs_negative_temporal_gradient_cosine")
        total = direction.get("adapted_total_action_vs_negative_temporal_gradient_cosine")
        row["adapter_direction_cosine"] = adapter
        row["total_direction_cosine"] = total
        row["direction_covariates"] = {
            "adapter_cosine": adapter,
            "total_cosine": total,
            "source": "frozen RCSP report direction_alignment.case_level",
        }
        reported = support.get("projection_retention_ratio")
        computed = row["support_retention_ratio"]
        error = (
            abs(float(reported) - float(computed))
            if reported is not None and computed is not None
            else None
        )
        parity_verified = error is not None and error <= tolerance
        if error is not None and not parity_verified:
            mismatch = {
                "identity": key,
                "role": row["role"],
                "reported": float(reported),
                "recomputed": float(computed),
                "abs_error": float(error),
                "tolerance": tolerance,
                "excluded_from_primary_analysis": row["role"] != "cross_event",
            }
            mismatches.append(mismatch)
            if row["role"] == "cross_event":
                raise RuntimeError(
                    "support retention parity failed for "
                    f"{key}: reported={float(reported)!r}, "
                    f"recomputed={float(computed)!r}, abs_error={float(error)!r}, "
                    f"tolerance={tolerance!r}"
                )
        row["support_retention_ratio_recomputed"] = computed
        row["support_retention_parity"] = {
            "reported": reported,
            "recomputed": computed,
            "abs_error": error,
            "verified": parity_verified,
            "source": "RCSP report projection_retention_ratio plus Phase 2 recomputation",
        }
        # The frozen RCSP report is authoritative for this covariate.  The
        # recomputed value remains available for explicit parity reporting.
        row["support_retention_ratio"] = reported if reported is not None else computed
        row["support_projection_report_retention_ratio"] = reported
    return {
        "cases": len(rows),
        "verified": not mismatches,
        "primary_cases_verified": not any(
            item["role"] == "cross_event" for item in mismatches
        ),
        "excluded_control_mismatches_allowed": True,
        "tolerance": tolerance,
        "mismatches": mismatches,
    }


def _validate_rcsp_metric_parity(
    rows: list[dict[str, Any]],
    rcsp_report: Mapping[str, Any],
    tolerance: float = PARITY_ATOL,
) -> dict[str, Any]:
    expected = _report_case_map(rcsp_report, "RCSP")
    errors = []
    for row in rows:
        source = expected[_identity_key(row)]
        errors.extend([
            abs(float(row["temporal_metric_rcsp"]) - float(source["temporal_metric"])),
            abs(float(row["temporal_deficit_rcsp"]) - float(source["temporal_scientific_deficit"])),
        ])
    maximum = max(errors) if errors else 0.0
    return {
        "verified": maximum <= tolerance,
        "max_abs_error": maximum,
        "cases": len(rows),
        "atol": tolerance,
        "rtol": PARITY_RTOL,
    }


def _group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    values = [row for row in rows if f"{row['split']}/{row['role']}/{row['width']}" == group]
    if len(values) != CASES_PER_GROUP:
        raise ValueError(f"{group} must contain exactly 8 cases")
    return values


def _median_nested(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[Any] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(value)
    return _median(values)


def cross_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != PRIMARY_CASES:
        raise ValueError("cross group summary requires 32 primary cases")
    result = {}
    for group in GROUP_ORDER:
        values = _group_rows(rows, group)
        result[group] = {
            "cases": len(values),
            "temporal_pass_count_base": sum(bool(row["gate_pass_base"]) for row in values),
            "temporal_pass_count_rcsp": sum(bool(row["gate_pass_rcsp"]) for row in values),
            "median_temporal_deficit_base": _median(row["temporal_deficit_base"] for row in values),
            "median_temporal_deficit_rcsp": _median(row["temporal_deficit_rcsp"] for row in values),
            "median_repair_gain": _median(row["repair_gain"] for row in values),
            "median_raw_numerator_base": {
                term: _median_nested(values, ("raw_temporal_numerator", term, "BASE"))
                for term in ("seam_acceleration", "seam_jerk")
            },
            "median_raw_numerator_rcsp": {
                term: _median_nested(values, ("raw_temporal_numerator", term, "RCSP"))
                for term in ("seam_acceleration", "seam_jerk")
            },
            "median_authoritative_denominator": {
                term: _median_nested(values, ("temporal_denominator", term, "BASE"))
                for term in ("seam_acceleration", "seam_jerk")
            },
            "median_gate_margin_base": _median(row["gate_margin_base"] for row in values),
            "median_gate_margin_rcsp": _median(row["gate_margin_rcsp"] for row in values),
            "median_active_frame_count": _median(row["active_statistics"]["binary_active_frame_count"] for row in values),
            "median_active_coordinate_count": _median(row["active_statistics"]["binary_active_coordinate_count"] for row in values),
            "median_total_soft_weight_mass": _median(row["effective_weight_statistics"]["total_weight_sum"] for row in values),
            "median_effective_weight_count": _median(row["effective_weight_statistics"]["effective_weight_count"] for row in values),
            "median_adapter_direction_cosine": _median(row["adapter_direction_cosine"] for row in values),
            "median_total_direction_cosine": _median(row["total_direction_cosine"] for row in values),
            "median_support_retention_ratio": _median(row["support_retention_ratio"] for row in values),
            "median_finite_action_efficiency_norm": _median(row["finite_action_efficiency"]["RCSP"]["final_tangent"]["G_over_action_norm"] for row in values),
            "median_finite_action_efficiency_energy": _median(row["finite_action_efficiency"]["RCSP"]["final_tangent"]["G_over_action_energy"] for row in values),
            "median_temporal_error_effective_support_base": _median(row["BASE"]["temporal_error_distribution"]["stats"]["effective_temporal_support"] for row in values),
            "median_temporal_error_effective_support_rcsp": _median(row["RCSP"]["temporal_error_distribution"]["stats"]["effective_temporal_support"] for row in values),
            "median_temporal_repair_effective_support": _median(row["temporal_repair_distribution"]["repair_effective_support"] for row in values),
        }
    return result


def _contrast(left: list[dict[str, Any]], right: list[dict[str, Any]], source: str) -> dict[str, Any]:
    def median(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
        return _median_nested(rows, path)

    def values(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "median_temporal_deficit_base": median(rows, ("temporal_deficit_base",)),
            "median_temporal_deficit_rcsp": median(rows, ("temporal_deficit_rcsp",)),
            "median_repair_gain": median(rows, ("repair_gain",)),
            "median_action_norm": median(rows, ("decoder_stage_statistics", "RCSP", "final_tangent", "l2_norm")),
            "median_action_energy": median(rows, ("decoder_stage_statistics", "RCSP", "final_tangent", "energy_sum")),
            "median_gate_margin_rcsp": median(rows, ("gate_margin_rcsp",)),
            "median_error_effective_support_rcsp": median(rows, ("RCSP", "temporal_error_distribution", "stats", "effective_temporal_support")),
            "median_repair_effective_support": median(rows, ("temporal_repair_distribution", "repair_effective_support")),
            "median_denominator_acceleration": median(rows, ("temporal_denominator", "seam_acceleration", "BASE")),
            "median_denominator_jerk": median(rows, ("temporal_denominator", "seam_jerk", "BASE")),
            "median_total_weight_mass": median(rows, ("effective_weight_statistics", "total_weight_sum")),
            "median_active_coordinate_count": median(rows, ("active_statistics", "binary_active_coordinate_count")),
            "median_adapter_direction_cosine": median(rows, ("adapter_direction_cosine",)),
        }

    return {"source": source, "width10_cases": len(left), "width28_cases": len(right), "width10": values(left), "width28": values(right)}


def width_contrasts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("seen", "new_position"):
        result[split] = _contrast(
            _group_rows(rows, f"{split}/cross_event/10"),
            _group_rows(rows, f"{split}/cross_event/28"),
            split,
        )
    return result


def scientific_answers(rows: list[dict[str, Any]], summary: Mapping[str, Any], contrasts: Mapping[str, Any]) -> dict[str, Any]:
    if len(rows) != PRIMARY_CASES:
        raise ValueError("scientific answers require the 32-case primary cohort")
    c = {split: contrasts[split] for split in ("seen", "new_position")}
    denominator_scaling = all(
        c[split]["width28"]["median_denominator_acceleration"] > c[split]["width10"]["median_denominator_acceleration"]
        and c[split]["width28"]["median_denominator_jerk"] > c[split]["width10"]["median_denominator_jerk"]
        for split in c
    )
    weight_per_active = all(
        c[split]["width28"]["median_total_weight_mass"] / max(c[split]["width28"]["median_active_coordinate_count"], 1.0)
        < c[split]["width10"]["median_total_weight_mass"] / max(c[split]["width10"]["median_active_coordinate_count"], 1.0)
        for split in c
    )
    spread = all(
        c[split]["width28"]["median_error_effective_support_rcsp"] > c[split]["width10"]["median_error_effective_support_rcsp"]
        or c[split]["width28"]["median_repair_effective_support"] > c[split]["width10"]["median_repair_effective_support"]
        for split in c
    )
    efficiency = all(
        (c[split]["width28"]["median_repair_gain"] / max(c[split]["width28"]["median_action_norm"], 1.0e-12))
        < (c[split]["width10"]["median_repair_gain"] / max(c[split]["width10"]["median_action_norm"], 1.0e-12))
        for split in c
    )
    gate_shift = all(
        c[split]["width28"]["median_gate_margin_rcsp"] < c[split]["width10"]["median_gate_margin_rcsp"]
        for split in c
    )
    direction = all(
        abs(c[split]["width28"]["median_adapter_direction_cosine"] or 0.0)
        < abs(c[split]["width10"]["median_adapter_direction_cosine"] or 0.0)
        for split in c
    )
    flags = {
        "temporal_objective_normalization_dilution": denominator_scaling,
        "effective_weight_mass_dilution": weight_per_active,
        "temporal_error_spread": spread,
        "finite_action_efficiency_loss": efficiency,
        "gate_margin_width_shift": gate_shift,
        "direction_quality_remains_contributory": direction,
    }
    supported = [name for name, value in flags.items() if value]
    classification = (
        "WIDTH_MECHANISM_UNRESOLVED" if not supported
        else supported[0] if len(supported) == 1
        else "MULTIPLE_WIDTH_MECHANISMS_SUPPORTED"
    )
    return {
        "width_audit_classification": classification,
        "primary_width_cohort": "32 cross_event cases: seen/cross_event/{10,28} and new_position/cross_event/{10,28}, 8 each",
        "single_excluded_from_primary_analysis": True,
        **flags,
        "hard_support_loss_primary_explanation": "NOT_SUPPORTED_BY_FROZEN_EVIDENCE",
        "seen_width10_vs_28_consistency": contrasts["seen"],
        "new_position_width10_vs_28_consistency": contrasts["new_position"],
        "cross_source_mechanism_consistency": {name: bool(value) for name, value in flags.items()},
        "dominant_width_mechanism": supported[0] if supported else None,
        "secondary_width_mechanisms": supported[1:] if len(supported) > 1 else [],
        "remaining_uncertainty": "All classifications are descriptive fixed-state evidence; no causal root cause or accepted intervention is established.",
        "claim_boundary": "This audit localizes where the width-dependent efficiency gap appears. It cannot prove a causal architectural root cause, select a new model, modify production, or authorize Pilot.",
    }


def _add_efficiency(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        gain = float(row["authoritative_temporal_repair_gain"])
        result = {}
        for state in ("BASE", "RCSP"):
            result[state] = {}
            for stage, stats in row["decoder_stage_statistics"][state].items():
                result[state][stage] = {
                    "G_over_action_norm": _ratio(gain, stats["l2_norm"]),
                    "G_over_action_energy": _ratio(gain, stats["energy_sum"]),
                    "G": gain,
                }
        row["finite_action_efficiency"] = result


def _case_temporal_numerator_denominator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "terms": ("seam_acceleration", "seam_jerk"),
        "cases": [
            {
                "identity": _identity_key(row),
                "BASE": row["BASE"]["components"],
                "RCSP": row["RCSP"]["components"],
                "raw_numerator_repair_gain": {
                    term: row["raw_temporal_numerator"][term]["BASE"] - row["raw_temporal_numerator"][term]["RCSP"]
                    for term in ("seam_acceleration", "seam_jerk")
                },
                "normalized_repair_gain": {
                    term: row["normalized_temporal_subterms"][term]["BASE"] - row["normalized_temporal_subterms"][term]["RCSP"]
                    for term in ("seam_acceleration", "seam_jerk")
                },
            }
            for row in rows
        ],
    }


def _validate_state_integrity(
    before: Mapping[str, str],
    after: Mapping[str, str],
    base_hash_before: str,
    adapter_hash_before: str,
    base: torch.nn.Module,
    model: rcsp.FrozenBaseRCSPModel,
) -> dict[str, Any]:
    base_after = safe.state_hash(base.state_dict())
    adapter_after = safe.state_hash(model.adapter.state_dict())
    all_grads_none = all(parameter.grad is None for parameter in model.parameters())
    unchanged = before == after and base_after == base_hash_before and adapter_after == adapter_hash_before
    if not unchanged or not all_grads_none:
        raise RuntimeError("read-only integrity contract failed")
    return {
        "base_state_sha256_before_after": base_hash_before,
        "adapter_state_sha256_before_after": adapter_hash_before,
        "base_unchanged": base_after == base_hash_before,
        "adapter_unchanged": adapter_after == adapter_hash_before,
        "immutable_artifacts_unchanged": before == after,
        "all_model_parameter_grad_none": all_grads_none,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_update_performed": False,
        "checkpoint_selection_performed": False,
        "scale_selection_performed": False,
        "architecture_selection_performed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
    }


def run(args: argparse.Namespace) -> int:
    root = Path(args.root_dir).resolve()
    source = Path(args.state_dir).resolve()
    trajectory, trajectory_paths, trajectory_hashes, trajectory_report, experiment, base_checkpoint = failure._load_trajectory(
        args.trajectory_dir, args.expected_trajectory_commit
    )
    rcsp_artifacts = _validate_rcsp(Path(args.rcsp_dir).resolve())
    parameter_report, parameter_hash = _validate_parameter_report(
        Path(args.parameter_attribution_report).resolve(), rcsp_artifacts["hashes"]["report.json"]
    )
    phase1_report, phase1_hash = _validate_phase1(
        Path(args.phase1_report).resolve(), FROZEN_ARTIFACT_COMMIT
    )
    single_report, single_hash = _validate_json_report(Path(args.single_decomposition_report).resolve(), phase1.SCHEMA, "single decomposition report")
    if (
        single_report.get("completed") is not True
        or single_report.get("provenance", {}).get("runtime_commit") != FROZEN_ARTIFACT_COMMIT
        or single_report.get("optimizer_steps") != 0
        or single_report.get("parameter_update_performed") is not False
        or single_report.get("pilot_allowed") is not False
    ):
        raise ValueError("single decomposition lineage or read-only contract mismatch")
    if parameter_report.get("completed") is not True or phase1_report.get("completed") is not True:
        raise ValueError("required frozen lineage reports are incomplete")
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    output = Path(args.output_dir).resolve()
    immutable_roots = (source, trajectory, rcsp_artifacts["directory"], Path(args.parameter_attribution_report).resolve().parent)
    if output.exists() or any(output.is_relative_to(path) for path in immutable_roots):
        raise FileExistsError("Phase 2 output must be fresh and outside immutable inputs")
    state, bank, cfg, source_metadata = group_audit.load_frozen_source(
        source,
        group_audit.LEGACY_COMMIT,
        legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength,
    )
    if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
        raise ValueError("trajectory does not reference the supplied frozen source")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    source_paths = {name: source / name for name in ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")}
    immutable_paths = {
        **{f"source/{name}": path for name, path in source_paths.items()},
        **{f"trajectory/{name}": path for name, path in trajectory_paths.items()},
        "rcsp/report.json": rcsp_artifacts["report_path"],
        "rcsp/reporting_logic_review_v1.json": rcsp_artifacts["review_path"],
        "rcsp/adapter_checkpoint": rcsp_artifacts["adapter_path"],
        "parameter_attribution/report.json": Path(args.parameter_attribution_report).resolve(),
        "phase1/report.json": Path(args.phase1_report).resolve(),
        "single_decomposition/report.json": Path(args.single_decomposition_report).resolve(),
    }
    before_files = _state_file_hashes(immutable_paths)
    implementation_paths = {
        "motion_models.py": Path(m.__file__).resolve(),
        "rcsp.py": Path(rcsp.__file__).resolve(),
        "alignment.py": Path(alignment.__file__).resolve(),
        "failure.py": Path(failure.__file__).resolve(),
        "group.py": Path(group_audit.__file__).resolve(),
        "safe.py": Path(safe.__file__).resolve(),
    }
    implementation_before = _state_file_hashes(implementation_paths)
    output.mkdir(parents=True, exist_ok=False)
    failure_path = output / "failure.json"
    try:
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda" else []
        )
        with torch.random.fork_rng(devices=cuda_devices), group_audit.frozen_environment(
            state["fingerprint"], source_metadata["decoder_strengths"]
        ):
            base = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
            base.load_state_dict(base_checkpoint["model_state_dict"], strict=True)
            base.eval()
            base_hash = safe.state_hash(base.state_dict())
            if base_hash != trajectory_report["final_state_sha256"]:
                raise RuntimeError("loaded base differs from immutable trajectory final state")
            model = rcsp.FrozenBaseRCSPModel(base)
            model.adapter.load_state_dict(rcsp_artifacts["adapter_checkpoint"]["adapter_state_dict"], strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            model.eval()
            adapter_hash = safe.state_hash(model.adapter.state_dict())
            if adapter_hash != rcsp_artifacts["report"]["parameter_update_scope"]["adapter_state_sha256"]:
                raise RuntimeError("loaded RCSP adapter state hash mismatch")
            probe, probe_hash = safe.load_probe(source, state, bank, cfg)
            final_batch, final_metadata = alignment.combine_final_banks(failure.final_banks(bank, probe, cfg))
            final_batch = rcsp._move_batch(final_batch, device)
            _validate_fixed_metadata(final_metadata)
            all_rows: list[dict[str, Any]] = []
            parity_rows: list[dict[str, Any]] = []
            for start in range(0, FINAL_CASES, rcsp.FINAL_CHUNK_SIZE):
                stop = start + rcsp.FINAL_CHUNK_SIZE
                chunk = {key: value[start:stop] for key, value in final_batch.items()}
                rows, parity = _evaluate_chunk(base, model, chunk, final_metadata[start:stop], cfg)
                all_rows.extend(rows)
                parity_rows.append(parity)
            if len(all_rows) != FINAL_CASES:
                raise RuntimeError("Phase 2 did not evaluate exactly 64 fixed final cases")
            direction_rows = _direction_map(rcsp_artifacts["report"])
            support_rows = _support_map(rcsp_artifacts["report"])
            support_retention_parity = _attach_frozen_covariates(
                all_rows, direction_rows, support_rows
            )
            _add_efficiency(all_rows)
            primary_rows, excluded_rows = _metadata_scopes(all_rows)
            authoritative_report_parity = _validate_rcsp_metric_parity(all_rows, rcsp_artifacts["report"])
            parity = {
                "cases": FINAL_CASES,
                "parity_cases": FINAL_CASES,
                "authoritative_rcsp_report": authoritative_report_parity,
                "temporal_reduction": {
                    "verified": all(part["temporal_reduction_parity"][state]["verified"] for part in parity_rows for state in ("BASE", "RCSP")),
                    "max_abs_error": max(part["temporal_reduction_parity"][state]["max_abs_error"] for part in parity_rows for state in ("BASE", "RCSP")),
                },
                "rcsp_alpha": 1.0,
                "verified": all(part["temporal_reduction_parity"][state]["verified"] for part in parity_rows for state in ("BASE", "RCSP")) and authoritative_report_parity["verified"] and support_retention_parity["primary_cases_verified"],
                "atol": PARITY_ATOL,
                "rtol": PARITY_RTOL,
                "support_retention": support_retention_parity,
            }
            if not parity["verified"]:
                raise RuntimeError("authoritative metric parity failed")
            primary_summary = cross_group_summary(primary_rows)
            contrasts = width_contrasts(primary_rows)
            answers = scientific_answers(primary_rows, primary_summary, contrasts)
            integrity_before = safe.state_hash(base.state_dict()), safe.state_hash(model.adapter.state_dict())
            del probe
            if _file_sha256(source / "probe_bank.pt") != probe_hash:
                raise RuntimeError("probe artifact changed during Phase 2 audit")
            integrity_after = _validate_state_integrity(
                before_files,
                _state_file_hashes(immutable_paths),
                integrity_before[0], integrity_before[1], base, model,
            )
        implementation_after = _state_file_hashes(implementation_paths)
        if implementation_before != implementation_after:
            raise RuntimeError("production implementation files changed during audit")
        report = {
            "schema": SCHEMA,
            "completed": True,
            "provenance": {
                "runtime_commit": runtime_commit,
                "expected_main_commit": args.expected_main_commit,
                "frozen_artifact_commit": FROZEN_ARTIFACT_COMMIT,
                "parent_commit": PARENT_COMMIT,
                "root": str(root),
                "source": str(source),
                "trajectory": str(trajectory),
                "rcsp_directory": str(rcsp_artifacts["directory"]),
                "phase1_report": str(args.phase1_report),
                "single_decomposition_report": str(args.single_decomposition_report),
                "parameter_attribution_report": str(args.parameter_attribution_report),
                "hashes": {
                    "source": {name: digest for name, digest in before_files.items() if name.startswith("source/")},
                    "trajectory": {name: digest for name, digest in before_files.items() if name.startswith("trajectory/")},
                    "rcsp": {name: digest for name, digest in before_files.items() if name.startswith("rcsp/")},
                    "phase1_report": phase1_hash,
                    "single_decomposition_report": single_hash,
                    "parameter_attribution_report": parameter_hash,
                },
            },
            "lineage": {
                "phase1_schema": phase1.SCHEMA,
                "phase1_completed": True,
                "phase1_optimizer_steps": 0,
                "phase1_pilot_allowed": False,
                "rcsp_report_completed": True,
                "rcsp_review_measurement_recomputation_verified": True,
                "adapter_checkpoint_path_read_from_rcsp_report": str(rcsp_artifacts["adapter_path"]),
                "trajectory_final_state_sha256": trajectory_report["final_state_sha256"],
                "single_report_excluded_from_primary_classification": True,
            },
            "primary_cohort": {
                "cases": PRIMARY_CASES,
                "groups": {group: CASES_PER_GROUP for group in GROUP_ORDER},
                "roles": list(PRIMARY_ROLES),
                "widths": list(WIDTHS),
            },
            "excluded_cohorts": {
                "single_recording": {
                    "cases": len(excluded_rows),
                    "excluded_from_primary_width_analysis": True,
                    "excluded_from_median_ratio_classification_dominant_mechanism": True,
                }
            },
            "normalization_inventory": normalization_inventory(cfg),
            "parity": parity,
            "case_level": primary_rows,
            "excluded_case_level": excluded_rows,
            "cross_group_summary": primary_summary,
            "width_contrasts": contrasts,
            "temporal_numerator_denominator": _case_temporal_numerator_denominator(primary_rows),
            "support_statistics": {
                "projected_outside_support_max": max(row["projected_outside_support_max"] for row in primary_rows),
                "binary_support_source": "RCSP report support_projection_stats plus exact parity recomputation from production effective root/joint masks",
                "primary_case_level": [{"identity": _identity_key(row), "active": row["active_statistics"], "support_retention_ratio": row["support_retention_ratio"]} for row in primary_rows],
            },
            "effective_weight_statistics": {
                "source": "production _refiner_decode_masks root_weight/joint_weight; no invented weights",
                "primary_case_level": [{"identity": _identity_key(row), **row["effective_weight_statistics"]} for row in primary_rows],
            },
            "decoder_stage_statistics": {
                "available_decoder_stages": [
                    "raw_action", "raw_adapter", "binary_projected_adapter", "soft_weighted_action",
                    "smoothed_action", "tapered_action", "capped_action", "final_tangent",
                    "final_decoded_geometric_displacement",
                ],
                "stage_source": "production _decode_product_refiner_output trace and RCSP last_details",
                "primary_case_level": [{"identity": _identity_key(row), **row["decoder_stage_statistics"]} for row in primary_rows],
            },
            "finite_action_efficiency": {
                "definition": "G = temporal_deficit_base - temporal_deficit_rcsp; descriptive G/action_norm and G/action_energy",
                "primary_case_level": [{"identity": _identity_key(row), **row["finite_action_efficiency"]} for row in primary_rows],
            },
            "temporal_error_distribution": [{"identity": _identity_key(row), "BASE": row["BASE"]["temporal_error_distribution"], "RCSP": row["RCSP"]["temporal_error_distribution"]} for row in primary_rows],
            "temporal_repair_distribution": [{"identity": _identity_key(row), **row["temporal_repair_distribution"]} for row in primary_rows],
            "temporal_thirds": [{"identity": _identity_key(row), **row["temporal_thirds"]} for row in primary_rows],
            "gate_margin_analysis": {
                "threshold_source": "cfg.checkpoint_validation_min_temporal_repair_gain",
                "crossing_count": {
                    group: {
                        "BASE": sum(bool(row["gate_pass_base"]) for row in _group_rows(primary_rows, group)),
                        "RCSP": sum(bool(row["gate_pass_rcsp"]) for row in _group_rows(primary_rows, group)),
                    }
                    for group in GROUP_ORDER
                },
                "closest_to_gate_case": min(primary_rows, key=lambda row: abs(row["gate_margin_rcsp"])),
                "farthest_from_gate_case": max(primary_rows, key=lambda row: abs(row["gate_margin_rcsp"])),
                "case_level": [{"identity": _identity_key(row), "base_margin": row["gate_margin_base"], "rcsp_margin": row["gate_margin_rcsp"], "base_pass": row["gate_pass_base"], "rcsp_pass": row["gate_pass_rcsp"]} for row in primary_rows],
            },
            "frozen_direction_covariates": {
                "source": "RCSP report direction_alignment.case_level",
                "recomputed_in_this_audit": False,
                "primary_case_level": [{"identity": _identity_key(row), "adapter_cosine": row["adapter_direction_cosine"], "total_cosine": row["total_direction_cosine"]} for row in primary_rows],
            },
            "scientific_answers": answers,
            "state_integrity": integrity_after,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "parameter_update_performed": False,
            "checkpoint_selection_performed": False,
            "scale_selection_performed": False,
            "architecture_selection_performed": False,
            "production_model_modified": False,
            "production_inference_modified": False,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
            "fake_case_pairing_performed": False,
            "pairing_policy": "UNPAIRED GROUP COMPARISON; pair_key is null because no authoritative same-boundary proof was supplied",
            "next_action": "stop_and_wait_for_manual_intervention_choice",
        }
        _exclusive_json(output / "report.json", report)
        for label in (
            "NORMALIZATION INVENTORY", "SCIENTIFIC ANSWERS", "CROSS GROUP SUMMARY",
            "SEEN CROSS10 VS CROSS28", "NEW_POSITION CROSS10 VS CROSS28",
            "TEMPORAL NUMERATOR / DENOMINATOR", "EFFECTIVE WEIGHT MASS",
            "FINITE-ACTION EFFICIENCY", "TEMPORAL ERROR DISTRIBUTION",
            "TEMPORAL REPAIR DISTRIBUTION", "GATE MARGIN ANALYSIS",
            "FROZEN DIRECTION COVARIATES", "STATE INTEGRITY",
        ):
            print(label, flush=True)
        for payload in (
            report["normalization_inventory"], answers, primary_summary,
            contrasts["seen"], contrasts["new_position"],
            report["temporal_numerator_denominator"], report["effective_weight_statistics"],
            report["finite_action_efficiency"], report["temporal_error_distribution"],
            report["temporal_repair_distribution"], report["gate_margin_analysis"],
            report["frozen_direction_covariates"], report["state_integrity"],
        ):
            print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
        print(json.dumps({"stage": "refiner_cross_width_normalization_audit_complete", "report": str(output / "report.json"), "primary_cases": PRIMARY_CASES, "optimizer_steps": 0, "production_model_modified": False, "scientific_acceptance": False, "pilot_allowed": False}, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(failure_path, {"schema": SCHEMA, "completed": False, "error": {"type": type(error).__name__, "message": str(error)}, "optimizer_steps": 0, "parameter_update_performed": False, "production_model_modified": False, "scientific_acceptance": False, "publish_allowed": False, "pilot_allowed": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--rcsp-dir", required=True)
    parser.add_argument("--parameter-attribution-report", required=True)
    parser.add_argument("--phase1-report", required=True)
    parser.add_argument("--single-decomposition-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--expected-trajectory-commit", default=failure.TRAJECTORY_COMMIT)
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
