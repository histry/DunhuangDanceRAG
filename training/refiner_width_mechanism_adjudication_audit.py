"""Read-only Phase 2.1 adjudication of the frozen cross-width mechanism.

This module is deliberately an audit-only consumer of the already completed
Phase 2 report.  It reuses the frozen source, trajectory and RCSP adapter to
obtain the same decoded tensors, then changes only the seam supplied to the
authoritative temporal metric for the within-case counterfactual.  It never
constructs an optimizer, calls backward, selects a checkpoint or changes a
production implementation.
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

from motion_geometry.boundary_observables import boundary_metrics_torch, observable_gate
from training import motion_models as m
from training import refiner_cross_width_normalization_audit as phase2
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_safe_start_diagnostics as safe
from training import refiner_single_direction_decomposition_audit as phase1
from training import refiner_temporal_action_alignment_audit as alignment


SCHEMA = "refiner_width_mechanism_adjudication_audit_v1"
FROZEN_PHASE2_COMMIT = "8e099944ed07f3550aede952aa1662a50e6e4bbe"
FROZEN_FORMAL_ARTIFACT_COMMIT = phase2.FROZEN_ARTIFACT_COMMIT
PRIMARY_CASES = 32
FINAL_CASES = 64
CASES_PER_GROUP = 8
WIDTHS = (10, 28)
PRIMARY_ROLE = "cross_event"
EXCLUDED_ROLE = "single_recording"
GROUP_ORDER = (
    "seen/cross_event/10",
    "seen/cross_event/28",
    "new_position/cross_event/10",
    "new_position/cross_event/28",
)
PARITY_ATOL = phase2.PARITY_ATOL
PARITY_RTOL = phase2.PARITY_RTOL
MAJOR_GAP_FRACTION = 0.50
LEGACY_CORE_STRENGTH = 0.02
LEGACY_TRANSITION_STRENGTH = 1.0
# Production ``masked_retract_torch`` exposes ``after_cap`` as the geometric
# tangent itself (75D).  The optional 79D branch below is only a defensive
# helper for callers that pass a raw contact+geometry layout.
GEOMETRIC_TANGENT_START = 0
GEOMETRIC_TANGENT_END = 75
SEAM_CORE_VALUE = 1.0
SEAM_HALO_VALUE = 0.35


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


def _median(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _identity(row: Mapping[str, Any]) -> str:
    return phase2._identity_key(row)


def _phase2_value(report: Mapping[str, Any], identity: str) -> Mapping[str, Any]:
    rows = list(report.get("case_level", [])) + list(report.get("excluded_case_level", []))
    values = {_identity(row): row for row in rows}
    if identity not in values:
        raise ValueError(f"Phase 2 case is missing: {identity}")
    return values[identity]


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} sha256 mismatch")
    return actual


def _validate_phase2_lineage(path: Path) -> tuple[dict[str, Any], str, dict[str, Path]]:
    """Validate one explicit Phase 2 report and return only its lineage paths."""
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2 report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    report_hash = _file_sha256(path)
    if report.get("schema") != phase2.SCHEMA:
        raise ValueError("Phase 2 schema mismatch")
    if report.get("completed") is not True:
        raise ValueError("Phase 2 completed=true is required")
    provenance = report.get("provenance", {})
    lineage = report.get("lineage", {})
    integrity = report.get("state_integrity", {})
    if provenance.get("runtime_commit") != FROZEN_PHASE2_COMMIT:
        raise ValueError("Phase 2 runtime commit is not the frozen upstream commit")
    if report.get("primary_cohort", {}).get("cases") != PRIMARY_CASES:
        raise ValueError("Phase 2 primary cohort is not exactly 32 cases")
    if report.get("primary_cohort", {}).get("groups") != {
        group: CASES_PER_GROUP for group in GROUP_ORDER
    }:
        raise ValueError("Phase 2 primary cohort group counts are not the frozen 4x8 contract")
    if report.get("excluded_cohorts", {}).get("single_recording", {}).get(
        "excluded_from_primary_width_analysis"
    ) is not True:
        raise ValueError("Phase 2 single_recording exclusion contract is missing")
    if report.get("parity", {}).get("verified") is not True or report.get("parity", {}).get(
        "temporal_reduction", {}
    ).get("verified") is not True:
        raise ValueError("Phase 2 authoritative parity is not verified")
    if any(
        (
            integrity.get("base_unchanged") is not True,
            integrity.get("adapter_unchanged") is not True,
            integrity.get("optimizer_steps") != 0,
            integrity.get("parameter_update_performed") is not False,
            report.get("optimizer_steps") != 0,
            report.get("parameter_update_performed") is not False,
            report.get("production_model_modified") is not False,
            report.get("production_inference_modified") is not False,
            report.get("scientific_acceptance") is not False,
            report.get("publish_allowed") is not False,
            report.get("pilot_allowed") is not False,
        )
    ):
        raise ValueError("Phase 2 read-only integrity contract is not satisfied")

    required = {
        "source": provenance.get("source"),
        "trajectory": provenance.get("trajectory"),
        "rcsp_directory": provenance.get("rcsp_directory"),
        "phase1_report": provenance.get("phase1_report"),
        "single_decomposition_report": provenance.get("single_decomposition_report"),
        "parameter_attribution_report": provenance.get("parameter_attribution_report"),
        "adapter_checkpoint": lineage.get("adapter_checkpoint_path_read_from_rcsp_report"),
    }
    if any(not isinstance(value, str) or not value for value in required.values()):
        raise ValueError("Phase 2 does not contain complete authoritative lineage paths")
    paths = {name: Path(value).resolve() for name, value in required.items()}
    hashes = provenance.get("hashes", {})
    source_names = ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")
    trajectory_names = ("report.json", "experiment.json", "diagnostic_latest.pt", "updates.jsonl")
    for name in source_names:
        expected = hashes.get("source", {}).get(f"source/{name}")
        if not isinstance(expected, str):
            raise ValueError(f"Phase 2 source hash is missing for {name}")
        _require_hash(paths["source"] / name, expected, f"Phase 2 source/{name}")
    for name in trajectory_names:
        expected = hashes.get("trajectory", {}).get(f"trajectory/{name}")
        if not isinstance(expected, str):
            raise ValueError(f"Phase 2 trajectory hash is missing for {name}")
        _require_hash(paths["trajectory"] / name, expected, f"Phase 2 trajectory/{name}")
    for name, path_value in (
        ("report.json", paths["rcsp_directory"] / "report.json"),
        ("reporting_logic_review_v1.json", paths["rcsp_directory"] / "reporting_logic_review_v1.json"),
    ):
        expected = hashes.get("rcsp", {}).get(f"rcsp/{name}")
        if not isinstance(expected, str):
            raise ValueError(f"Phase 2 RCSP hash is missing for {name}")
        _require_hash(path_value, expected, f"Phase 2 rcsp/{name}")
    expected_adapter = hashes.get("rcsp", {}).get("rcsp/adapter_checkpoint")
    if not isinstance(expected_adapter, str):
        raise ValueError("Phase 2 RCSP adapter hash is missing")
    _require_hash(paths["adapter_checkpoint"], expected_adapter, "Phase 2 adapter checkpoint")
    for key in ("phase1_report", "single_decomposition_report", "parameter_attribution_report"):
        expected = hashes.get(key)
        if not isinstance(expected, str):
            raise ValueError(f"Phase 2 hash is missing for {key}")
        _require_hash(paths[key], expected, f"Phase 2 {key}")
    if paths["adapter_checkpoint"].parent != paths["rcsp_directory"]:
        raise ValueError("Phase 2 adapter path is not inside the authoritative RCSP directory")
    if paths["source"] != Path(provenance["source"]).resolve():
        raise ValueError("Phase 2 source path changed during lineage resolution")
    if lineage.get("phase2_report_path") not in (None, str(path.resolve())):
        raise ValueError("Phase 2 self-lineage path mismatch")
    return report, report_hash, paths


def _validate_upstream_reports(paths: Mapping[str, Path]) -> dict[str, Any]:
    rcsp_artifacts = phase2._validate_rcsp(paths["rcsp_directory"])
    if rcsp_artifacts["adapter_path"] != paths["adapter_checkpoint"]:
        raise ValueError("RCSP adapter path does not match Phase 2 lineage")
    parameter_report, parameter_hash = phase2._validate_parameter_report(
        paths["parameter_attribution_report"], rcsp_artifacts["hashes"]["report.json"]
    )
    phase1_report, phase1_hash = phase2._validate_phase1(
        paths["phase1_report"], FROZEN_FORMAL_ARTIFACT_COMMIT
    )
    single_report, single_hash = phase2._validate_json_report(
        paths["single_decomposition_report"], phase1.SCHEMA, "single decomposition report"
    )
    if (
        single_report.get("completed") is not True
        or single_report.get("provenance", {}).get("runtime_commit") != FROZEN_FORMAL_ARTIFACT_COMMIT
        or single_report.get("optimizer_steps") != 0
        or single_report.get("parameter_update_performed") is not False
        or single_report.get("pilot_allowed") is not False
    ):
        raise ValueError("single decomposition lineage or read-only contract mismatch")
    return {
        "rcsp": rcsp_artifacts,
        "parameter_report": parameter_report,
        "parameter_hash": parameter_hash,
        "phase1_report": phase1_report,
        "phase1_hash": phase1_hash,
        "single_report": single_report,
        "single_hash": single_hash,
    }


def normalized_temporal_spread_fraction(
    contribution: Iterable[float], active_support: Iterable[bool]
) -> float | None:
    """Compute NTSF on the authoritative active support, null for zero energy."""
    values = np.asarray(list(contribution), dtype=np.float64).reshape(-1)
    support = np.asarray(list(active_support), dtype=bool).reshape(-1)
    if values.shape != support.shape:
        raise ValueError("contribution and active support must have equal length")
    selected = values[support]
    if selected.size == 0:
        return None
    if not np.isfinite(selected).all() or bool((selected < 0).any()):
        raise ValueError("NTSF requires finite nonnegative contributions")
    square_sum = float(np.square(selected).sum())
    if square_sum == 0.0:
        return None
    effective = float(np.square(selected.sum()) / square_sum)
    result = effective / float(selected.size)
    if not (result > 0.0 and result <= 1.0 + 1.0e-10):
        raise FloatingPointError(f"NTSF outside theoretical range: {result!r}")
    return result


def _contribution_vectors(joints: torch.Tensor, seam: torch.Tensor, fps: float) -> list[dict[str, Any]]:
    components = phase2.authoritative_temporal_components(joints, seam, fps)
    count, frame_count = int(joints.shape[0]), int(joints.shape[1])
    result = []
    for index in range(count):
        terms = {}
        support_union = np.zeros(frame_count, dtype=bool)
        for name in ("seam_acceleration", "seam_jerk"):
            term = components["terms"][name]
            values = term["values"][index].detach().double().cpu().numpy()
            support = term["support"][index].detach().bool().cpu().numpy()
            denominator = float(term["denominator"][index].detach().cpu())
            contribution = values * support.astype(np.float64) / denominator / float(term["scale"])
            aligned = np.zeros(frame_count, dtype=np.float64)
            aligned[: contribution.size] += contribution
            aligned_support = np.zeros(frame_count, dtype=bool)
            aligned_support[: support.size] = support
            support_union |= aligned_support
            terms[name] = {
                "contribution": [float(value) for value in aligned.tolist()],
                "support": [bool(value) for value in aligned_support.tolist()],
                "raw_numerator": _finite(term["raw_numerator"][index], f"{name} numerator"),
                "denominator": _finite(term["denominator"][index], f"{name} denominator"),
                "normalized_value": _finite(term["normalized_value"][index], f"{name} normalized value"),
            }
        error = np.zeros(frame_count, dtype=np.float64)
        for term in terms.values():
            error += np.asarray(term["contribution"], dtype=np.float64)
        terms["temporal_energy"] = _finite(components["temporal_energy"][index], "temporal energy")
        result.append({"terms": terms, "error": error, "active_support": support_union})
    return result


def positive_repair_contribution(
    base_error: Iterable[float], rcsp_error: Iterable[float]
) -> np.ndarray:
    """Return max(BASE- RCSP, 0); signed repair is never used for NTSF."""
    base = np.asarray(list(base_error), dtype=np.float64)
    rcsp_value = np.asarray(list(rcsp_error), dtype=np.float64)
    if base.shape != rcsp_value.shape:
        raise ValueError("BASE and RCSP contribution vectors must have equal length")
    difference = base - rcsp_value
    if not np.isfinite(difference).all():
        raise FloatingPointError("nonfinite temporal repair contribution")
    return np.maximum(difference, 0.0)


def applied_action_delta_norm(
    final_tangent_rcsp: torch.Tensor, final_tangent_base: torch.Tensor
) -> torch.Tensor:
    """Norm of the 75D final tangent delta; contact logits [0:4] are excluded."""
    if final_tangent_rcsp.shape != final_tangent_base.shape:
        raise ValueError("BASE and RCSP final tangent shapes differ")
    if final_tangent_rcsp.shape[-1] == GEOMETRIC_TANGENT_END:
        delta = final_tangent_rcsp - final_tangent_base
    elif final_tangent_rcsp.shape[-1] == GEOMETRIC_TANGENT_END + 4:
        delta = final_tangent_rcsp[..., 4:79] - final_tangent_base[..., 4:79]
    else:
        raise ValueError("final tangent lacks the 75D geometry or 79D contact+geometry layout")
    return delta.detach().double().reshape(delta.shape[0], -1).norm(dim=1)


def _metric_rows(joints: torch.Tensor, seam: torch.Tensor, fps: float) -> list[dict[str, Any]]:
    metrics = boundary_metrics_torch(joints, seam, fps)
    rows = []
    for index in range(joints.shape[0]):
        rows.append({
            "endpoint_velocity_jump_mps": _finite(metrics["endpoint_velocity_jump_mps"][index], "endpoint metric"),
            "seam_acceleration_mps2": _finite(metrics["seam_acceleration_mps2"][index], "acceleration metric"),
            "seam_jerk_mps3": _finite(metrics["seam_jerk_mps3"][index], "jerk metric"),
            "context_seam_acceleration_mps2": _finite(metrics["context_seam_acceleration_mps2"][index], "context acceleration metric"),
            "context_seam_jerk_mps3": _finite(metrics["context_seam_jerk_mps3"][index], "context jerk metric"),
            "temporal_energy": _finite(metrics["temporal_energy"][index], "temporal metric"),
            "context_temporal_energy": _finite(metrics["context_temporal_energy"][index], "context temporal metric"),
            "valid": bool(metrics["valid"][index]),
        })
    return rows


def _cf_metric_payload(
    before: Mapping[str, Any], base: Mapping[str, Any], rcsp_value: Mapping[str, Any], cfg: m.MotionGenerationConfig,
    before_contrib: Mapping[str, Any], base_contrib: Mapping[str, Any], rcsp_contrib: Mapping[str, Any],
) -> dict[str, Any]:
    base_gate = observable_gate(dict(before), dict(base), cfg)
    rcsp_gate = observable_gate(dict(before), dict(rcsp_value), cfg)
    error = np.asarray(rcsp_contrib["error"], dtype=np.float64)
    active = np.asarray(rcsp_contrib["active_support"], dtype=bool)
    repair = positive_repair_contribution(base_contrib["error"], rcsp_contrib["error"])
    repair_active = np.asarray(base_contrib["active_support"], dtype=bool) | active
    return {
        "M_before": float(before["temporal_energy"]),
        "M_base": float(base["temporal_energy"]),
        "M_rcsp": float(rcsp_value["temporal_energy"]),
        "G_base": _finite(base_gate["temporal_gain"], "BASE relative gate gain"),
        "G_rcsp": _finite(rcsp_gate["temporal_gain"], "RCSP relative gate gain"),
        "gate_margin_base": _finite(base_gate["temporal_gain"], "BASE gate margin")
        - float(cfg.checkpoint_validation_min_temporal_repair_gain),
        "gate_margin_rcsp": _finite(rcsp_gate["temporal_gain"], "RCSP gate margin")
        - float(cfg.checkpoint_validation_min_temporal_repair_gain),
        "base_valid": bool(base_gate["temporal_informative"] or before["valid"] and base["valid"]),
        "rcsp_valid": bool(rcsp_gate["temporal_informative"] or before["valid"] and rcsp_value["valid"]),
        "rcsp_error_spread_fraction": normalized_temporal_spread_fraction(error, active),
        "positive_repair_spread_fraction": normalized_temporal_spread_fraction(repair, repair_active),
        "positive_repair_mass": float(repair.sum()),
        "signed_repair_mass": float(
            np.asarray(base_contrib["error"]).sum() - np.asarray(rcsp_contrib["error"]).sum()
        ),
        "active_temporal_support_count": int(active.sum()),
        "positive_repair_active_support_count": int(repair_active.sum()),
    }


def _reconstruct_seam(actual: torch.Tensor, width: int, fps: float, halo_seconds: float) -> tuple[torch.Tensor, dict[str, Any]]:
    """Rebuild the formal halo/core seam around the center derived from actual core."""
    if actual.ndim != 3 or actual.shape[-1] != 1:
        raise ValueError("seam must have shape [B,T,1]")
    count, frames = actual.shape[:2]
    result = torch.zeros_like(actual)
    records = []
    halo = max(0, int(round(float(halo_seconds) * float(fps))))
    for index in range(count):
        core = (actual[index, :, 0] >= 0.5).detach().cpu().numpy().astype(bool)
        positions = np.flatnonzero(core)
        if positions.size == 0 or not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            raise ValueError("frozen seam core is not one contiguous interval")
        actual_width = int(positions.size)
        if actual_width not in WIDTHS:
            raise ValueError(f"unexpected frozen seam width: {actual_width}")
        center_twice = int(positions[0] + positions[-1])
        numerator = center_twice - int(width) + 1
        if numerator % 2:
            raise ValueError("counterfactual width cannot preserve the frozen seam center")
        start = numerator // 2
        stop = start + int(width)
        if start < 0 or stop > frames or start >= stop:
            raise ValueError("counterfactual seam leaves the fixed window")
        left = max(0, start - halo)
        right = min(frames, stop + halo)
        result[index, left:right, 0] = SEAM_HALO_VALUE
        result[index, start:stop, 0] = SEAM_CORE_VALUE
        records.append({
            "actual_width": actual_width,
            "actual_core_start": int(positions[0]),
            "actual_core_stop_exclusive": int(positions[-1] + 1),
            "center_twice": center_twice,
            "counterfactual_width": int(width),
            "counterfactual_core_start": int(start),
            "counterfactual_core_stop_exclusive": int(stop),
            "halo_frames": halo,
        })
    return result, {"cases": records}


def mask_reconstruction_parity(
    actual: torch.Tensor, fps: float, halo_seconds: float
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    masks = {}
    all_records = []
    for width in WIDTHS:
        mask, details = _reconstruct_seam(actual, width, fps, halo_seconds)
        masks[width] = mask
        all_records.append((width, details["cases"], mask))
    parity_rows = []
    for index in range(actual.shape[0]):
        actual_width = int(((actual[index, :, 0] >= 0.5).sum()).item())
        selected = masks[actual_width][index]
        full_equal = bool(torch.equal(selected, actual[index]))
        core_equal = bool(torch.equal(selected >= 0.5, actual[index] >= 0.5))
        record = next(rows[index] for width, rows, _mask in all_records if width == actual_width)
        parity_rows.append({
            "identity_index": index,
            **record,
            "full_value_equal": full_equal,
            "core_boolean_equal": core_equal,
            "verified": bool(full_equal and core_equal),
        })
    verified = len(parity_rows) == FINAL_CASES and all(row["verified"] for row in parity_rows)
    return masks, {
        "cases": len(parity_rows),
        "verified_cases": sum(bool(row["verified"]) for row in parity_rows),
        "verified": verified,
        "source": "training.motion_models.degrade_for_refiner and make_cross_event_boundary_np: halo=0.35, core=1.0; metric consumes seam >= 0.5",
        "rows": parity_rows,
    }


def _actual_width_parity(
    identity: str,
    actual_width: int,
    cf: Mapping[str, Any],
    p2_row: Mapping[str, Any],
    upstream_rcsp: Mapping[str, Any],
) -> dict[str, Any]:
    base_report = upstream_rcsp["base"]
    rcsp_report = upstream_rcsp["rcsp"]
    expected = {
        "M_before": rcsp_report["observable"]["before"]["temporal_energy"],
        "M_base": p2_row["BASE"]["temporal_metric"],
        "M_rcsp": p2_row["RCSP"]["temporal_metric"],
        "G_base": p2_row["BASE"]["gate"]["relative_gain"],
        "G_rcsp": p2_row["RCSP"]["gate"]["relative_gain"],
        "gate_margin_base": p2_row["BASE"]["gate_margin"],
        "gate_margin_rcsp": p2_row["RCSP"]["gate_margin"],
    }
    expected_source = {
        "M_before": rcsp_report["observable"]["before"]["temporal_energy"],
        "M_base": base_report["observable"]["after"]["temporal_energy"],
        "M_rcsp": rcsp_report["observable"]["after"]["temporal_energy"],
        "G_base": base_report["observable"]["temporal_gain"],
        "G_rcsp": rcsp_report["observable"]["temporal_gain"],
        "gate_margin_base": base_report["observable"]["temporal_gain"] - p2_row["gate_threshold"],
        "gate_margin_rcsp": rcsp_report["observable"]["temporal_gain"] - p2_row["gate_threshold"],
    }
    values = {}
    errors = {}
    for key in expected:
        error = max(abs(float(cf[key]) - float(expected[key])), abs(float(cf[key]) - float(expected_source[key])))
        errors[key] = float(error)
        values[key] = {"counterfactual": float(cf[key]), "phase2": float(expected[key]), "rcsp_upstream": float(expected_source[key])}
    maximum = max(errors.values()) if errors else 0.0
    return {
        "identity": identity,
        "actual_width": actual_width,
        "verified": maximum <= PARITY_ATOL,
        "max_abs_error": maximum,
        "atol": PARITY_ATOL,
        "values": values,
        "errors": errors,
    }


def _capture_outputs(
    base: torch.nn.Module,
    model: rcsp.FrozenBaseRCSPModel,
    batch: Mapping[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    cfg: m.MotionGenerationConfig,
) -> dict[str, Any]:
    count = len(metadata)
    routed = dict(batch)
    routed["role_id"] = rcsp.role_ids_from_metadata(metadata, batch["bad"].device)
    with torch.no_grad():
        base_trace: dict[str, Any] = {}
        base_prediction, _ = m._refiner_batch_outputs(base, routed, cfg, trace=base_trace)
        rcsp_trace: dict[str, Any] = {}
        rcsp_prediction, _ = rcsp.rcsp_batch_outputs(
            model, routed, cfg, trace=rcsp_trace, capture_details=True
        )
        details = model.last_details
        if not torch.equal(base_trace["raw_output"][:count, ..., :4], details["raw_adapted"][:count, ..., :4]):
            raise RuntimeError("RCSP changed contact channels")
        if _finite((base_trace["raw_output"] - details["raw_base"]).abs().max(), "BASE raw parity") != 0.0:
            raise RuntimeError("RCSP base raw wrapper parity failed")
        metric_joints = m._observable_boundary_joints_torch(
            torch.cat((batch["bad"], base_prediction, rcsp_prediction))
        )
        before_joints, base_joints, rcsp_joints = metric_joints.split(count)
    action_norm = applied_action_delta_norm(
        rcsp_trace["repair"]["after_cap"], base_trace["repair"]["after_cap"]
    )
    model.clear_last_details()
    return {
        "before_joints": before_joints,
        "base_joints": base_joints,
        "rcsp_joints": rcsp_joints,
        "base_final_tangent": base_trace["repair"]["after_cap"],
        "rcsp_final_tangent": rcsp_trace["repair"]["after_cap"],
        "applied_action_norm": action_norm,
        "same_output_tensors": True,
    }


def _add_cf_rows(
    rows: list[dict[str, Any]],
    cf_masks: Mapping[int, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    cfg: m.MotionGenerationConfig,
) -> list[dict[str, Any]]:
    count = len(metadata)
    states = torch.cat((outputs["before_joints"], outputs["base_joints"], outputs["rcsp_joints"]))
    for width in WIDTHS:
        seam = cf_masks[width]
        metrics = _metric_rows(states, torch.cat((seam, seam, seam)), cfg.fps)
        contributions = _contribution_vectors(states, torch.cat((seam, seam, seam)), cfg.fps)
        for index in range(count):
            before_metric, base_metric, rcsp_metric = metrics[index], metrics[count + index], metrics[2 * count + index]
            before_contrib, base_contrib, rcsp_contrib = contributions[index], contributions[count + index], contributions[2 * count + index]
            payload = _cf_metric_payload(before_metric, base_metric, rcsp_metric, cfg, before_contrib, base_contrib, rcsp_contrib)
            row = rows[index]
            row.setdefault("counterfactual", {})[f"cf{width}"] = {
                "evaluation_width": width,
                **payload,
                "same_motion_tensor": True,
                "same_decoded_output_tensor": True,
                "same_support_and_repair_output": True,
                "same_fps": True,
                "same_model_and_adapter": True,
                "alpha": 1.0,
            }
    for row in rows:
        cf10 = row["counterfactual"]["cf10"]
        cf28 = row["counterfactual"]["cf28"]
        cf28["metric_ratio_before_28_over_10"] = _ratio(cf28["M_before"], cf10["M_before"])
        cf28["metric_ratio_base_28_over_10"] = _ratio(cf28["M_base"], cf10["M_base"])
        cf28["metric_ratio_rcsp_28_over_10"] = _ratio(cf28["M_rcsp"], cf10["M_rcsp"])
        cf28["delta_G_counterfactual"] = float(cf28["G_rcsp"]) - float(cf10["G_rcsp"])
    return rows


def _scope_rows(rows: list[dict[str, Any]], split: str | None = None) -> list[dict[str, Any]]:
    return [row for row in rows if split is None or row["split"] == split]


def _spread_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    cf10 = [row["counterfactual"]["cf10"][field] for row in rows]
    cf28 = [row["counterfactual"]["cf28"][field] for row in rows]
    ratios = [_ratio(a, b) for a, b in zip(cf28, cf10)]
    ratios = [value for value in ratios if value is not None]
    greater = sum(float(a) > float(b) for a, b in zip(cf28, cf10) if a is not None and b is not None)
    less = sum(float(a) < float(b) for a, b in zip(cf28, cf10) if a is not None and b is not None)
    comparable = sum(a is not None and b is not None for a, b in zip(cf28, cf10))
    return {
        "metric": field,
        "cases": len(rows),
        "median_cf10": _median(cf10),
        "median_cf28": _median(cf28),
        "median_within_case_ratio_cf28_over_cf10": _median(ratios),
        "cases_ratio_gt_1": greater,
        "cases_ratio_lt_1": less,
        "null_cases": len(rows) - comparable,
        "non_null_cases": comparable,
        "cases_cf28_gt_cf10": greater,
        "cases_cf28_lt_cf10": less,
    }


def spread_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != PRIMARY_CASES:
        raise ValueError("spread summaries require exactly 32 primary rows")
    return {
        scope: {
            "rcsp_error_spread_fraction": _spread_summary(_scope_rows(rows, split), "rcsp_error_spread_fraction"),
            "positive_repair_spread_fraction": _spread_summary(_scope_rows(rows, split), "positive_repair_spread_fraction"),
        }
        for scope, split in (("overall", None), ("seen", "seen"), ("new_position", "new_position"))
    }


def _direction_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scope, scope_rows in (("overall", rows), ("seen", _scope_rows(rows, "seen")), ("new_position", _scope_rows(rows, "new_position"))):
        widths = {}
        for width in WIDTHS:
            selected = [row for row in scope_rows if int(row["width"]) == width]
            widths[str(width)] = {
                "cases": len(selected),
                "median_E_gate": _median(row["metric_2"]["E_gate"] for row in selected),
                "median_adapter_direction_cosine": _median(row["metric_2"]["adapter_direction_cosine"] for row in selected),
            }
        widths["28_over_10_E_gate_ratio"] = _ratio(widths["28"]["median_E_gate"], widths["10"]["median_E_gate"])
        result[scope] = widths
    return result


def _gap_explanation_fraction(observed_gap: float | None, counterfactual_gap: float | None) -> float | None:
    if observed_gap is None or abs(float(observed_gap)) <= 0.0:
        return None
    if counterfactual_gap is None or float(observed_gap) * float(counterfactual_gap) <= 0.0:
        return 0.0
    return min(abs(float(counterfactual_gap)) / abs(float(observed_gap)), 1.0)


def _counterfactual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scope, scope_rows in (("overall", rows), ("seen", _scope_rows(rows, "seen")), ("new_position", _scope_rows(rows, "new_position"))):
        deltas = [row["counterfactual"]["delta_G_counterfactual"] for row in scope_rows]
        cf10 = [row["counterfactual"]["cf10"]["G_rcsp"] for row in scope_rows]
        cf28 = [row["counterfactual"]["cf28"]["G_rcsp"] for row in scope_rows]
        actual10 = [row["actual_width"]["G_rcsp"] for row in scope_rows if int(row["width"]) == 10]
        actual28 = [row["actual_width"]["G_rcsp"] for row in scope_rows if int(row["width"]) == 28]
        observed_gap = (_median(actual28) - _median(actual10)) if actual10 and actual28 else None
        counterfactual_gap = _median(deltas)
        result[scope] = {
            "cases": len(scope_rows),
            "median_G_rcsp_cf10": _median(cf10),
            "median_G_rcsp_cf28": _median(cf28),
            "median_delta_G_counterfactual": counterfactual_gap,
            "count_delta_negative": sum(value < 0.0 for value in deltas),
            "count_delta_positive": sum(value > 0.0 for value in deltas),
            "count_delta_equal_within_tolerance": sum(abs(value) <= PARITY_ATOL for value in deltas),
            "observed_gate_gap": observed_gap,
            "counterfactual_gap": counterfactual_gap,
            "gap_explanation_fraction": _gap_explanation_fraction(observed_gap, counterfactual_gap),
        }
    return result


def _spread_increase(summary: Mapping[str, Any]) -> bool:
    error = summary["rcsp_error_spread_fraction"]
    ratio = error["median_within_case_ratio_cf28_over_cf10"]
    non_null = int(error["non_null_cases"])
    return bool(
        ratio is not None
        and float(ratio) > 1.0
        and non_null > 0
        and int(error["cases_cf28_gt_cf10"]) > non_null / 2.0
    )


def _direction_efficiency_loss(summary: Mapping[str, Any]) -> bool:
    width10, width28 = summary["10"], summary["28"]
    return bool(
        width10["median_E_gate"] is not None
        and width28["median_E_gate"] is not None
        and width28["median_E_gate"] < width10["median_E_gate"]
        and width10["median_adapter_direction_cosine"] is not None
        and width28["median_adapter_direction_cosine"] is not None
        and width28["median_adapter_direction_cosine"] < width10["median_adapter_direction_cosine"]
    )


def adjudicate(
    summaries: Mapping[str, Any],
    *,
    counterfactual_mask_parity_verified: bool = True,
) -> dict[str, Any]:
    """Apply the fixed decision tree without post-hoc threshold tuning."""
    cf = summaries["counterfactual_width"]
    direction = summaries["direction_efficiency"]
    if not counterfactual_mask_parity_verified:
        return {
            "adjudicated_primary_mechanism": "MIXED_OR_UNRESOLVED_WITH_THREE_METRICS",
            "counterfactual_construction_available": False,
            "counterfactual_mask_parity_verified": False,
            "cf_explains_major_gap": {"seen": False, "new_position": False},
            "spread_increase": {"seen": False, "new_position": False},
            "direction_efficiency_loss": {
                "seen": _direction_efficiency_loss(direction["seen"]),
                "new_position": _direction_efficiency_loss(direction["new_position"]),
            },
            "primary_intervention_order": [],
            "next_action": "design minimal controlled intervention under the strongest cross-source evidence",
            "causal_root_cause_proven": False,
            "claim_boundary": "Frozen fixed-state mechanism adjudication only; counterfactual parity failed closed.",
        }
    cf_major = {
        split: bool(cf[split]["gap_explanation_fraction"] is not None and cf[split]["gap_explanation_fraction"] >= MAJOR_GAP_FRACTION)
        for split in ("seen", "new_position")
    }
    spread = {split: _spread_increase(summaries["spread"][split]) for split in ("seen", "new_position")}
    direction_loss = {
        split: _direction_efficiency_loss(direction[split]) for split in ("seen", "new_position")
    }
    both_cf = cf_major["seen"] and cf_major["new_position"]
    both_spread = spread["seen"] and spread["new_position"]
    both_direction = direction_loss["seen"] and direction_loss["new_position"]
    if both_cf and both_direction:
        mechanism = "MIXED_WIDTH_MECHANISM"
        order = ["metric/support-time intervention", "direction intervention"]
    elif both_cf and both_spread:
        mechanism = "TEMPORAL_SPREADING_PRIMARY"
        order = ["metric/support-time intervention"]
    elif both_cf:
        mechanism = "WIDTH_NORMALIZATION_PRIMARY"
        order = ["metric/support-time intervention"]
    elif both_direction:
        mechanism = "WIDTH_CONDITIONED_DIRECTION_PRIMARY"
        order = ["direction intervention", "temporal evaluation intervention"]
    else:
        mechanism = "MIXED_OR_UNRESOLVED_WITH_THREE_METRICS"
        order = ["design minimal controlled intervention under the strongest cross-source evidence"]
    return {
        "adjudicated_primary_mechanism": mechanism,
        "counterfactual_construction_available": True,
        "counterfactual_mask_parity_verified": True,
        "cf_explains_major_gap": cf_major,
        "spread_increase": spread,
        "direction_efficiency_loss": direction_loss,
        "normalization_evidence": {split: cf_major[split] and not spread[split] for split in cf_major},
        "temporal_spreading_evidence": spread,
        "width_conditioned_direction_evidence": direction_loss,
        "primary_intervention_order": order,
        "major_gap_threshold": MAJOR_GAP_FRACTION,
        "next_action": "design_minimal_intervention_from_adjudicated_mechanism",
        "causal_root_cause_proven": False,
        "claim_boundary": "Adjudicated primary mechanism under frozen fixed-state evidence; no causal root cause or production solution is established.",
    }


def _attach_case_metrics(
    p2_rows: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    outputs: Mapping[str, Any],
    cf_masks: Mapping[int, torch.Tensor],
    phase2_report: Mapping[str, Any],
    upstream: Mapping[str, Any],
    cfg: m.MotionGenerationConfig,
) -> list[dict[str, Any]]:
    rows = []
    upstream_base = phase2._report_case_map(upstream["rcsp"]["report"], "BASE")
    upstream_rcsp = phase2._report_case_map(upstream["rcsp"]["report"], "RCSP")
    for index, meta in enumerate(metadata):
        identity = _identity(meta)
        source_p2 = _phase2_value(phase2_report, identity)
        source_base = upstream_base[identity]
        source_rcsp = upstream_rcsp[identity]
        rows.append({
            **meta,
            "identity": identity,
            "counterfactual_pair_key": identity,
            "observed_group_pairing": "UNPAIRED",
            "counterfactual_pairing": "WITHIN_CASE_BY_CONSTRUCTION",
            "actual_width": {},
            "metric_1": {},
            "metric_2": {
                "adapter_direction_cosine": source_p2.get("adapter_direction_cosine"),
                "total_direction_cosine": source_p2.get("total_direction_cosine"),
                "applied_action_norm": _finite(outputs["applied_action_norm"][index], "applied action norm"),
            },
            "upstream_parity_identity": {
                "phase2": True,
                "rcsp_base": source_base.get("observable", {}).get("after", {}).get("temporal_energy"),
                "rcsp": source_rcsp.get("observable", {}).get("after", {}).get("temporal_energy"),
            },
            "_source_p2": source_p2,
            "_source_base": source_base,
            "_source_rcsp": source_rcsp,
            "_index": index,
        })
    _add_cf_rows(rows, cf_masks, outputs, metadata, cfg)
    for index, row in enumerate(rows):
        actual_width = int(row["width"])
        actual = row["counterfactual"][f"cf{actual_width}"]
        p2_row = row["_source_p2"]
        parity = _actual_width_parity(
            row["identity"], actual_width, actual, p2_row,
            {"base": row["_source_base"], "rcsp": row["_source_rcsp"]},
        )
        if not parity["verified"] and row["role"] == PRIMARY_ROLE:
            raise RuntimeError(f"actual-width counterfactual metric parity failed for {row['identity']}")
        action_norm = float(row["metric_2"]["applied_action_norm"])
        delta_g = float(actual["G_rcsp"]) - float(actual["G_base"])
        row["actual_width"] = {
            "evaluation_width": actual_width,
            "M_before": actual["M_before"],
            "M_base": actual["M_base"],
            "M_rcsp": actual["M_rcsp"],
            "G_base": actual["G_base"],
            "G_rcsp": actual["G_rcsp"],
            "delta_G_rcsp": delta_g,
            "gate_margin_base": actual["gate_margin_base"],
            "gate_margin_rcsp": actual["gate_margin_rcsp"],
        }
        row["metric_1"] = {
            "rcsp_error_spread_fraction_actual": actual["rcsp_error_spread_fraction"],
            "positive_repair_spread_fraction_actual": actual["positive_repair_spread_fraction"],
        }
        row["metric_2"].update({
            "G_base": actual["G_base"],
            "G_rcsp": actual["G_rcsp"],
            "delta_G_rcsp": delta_g,
            "E_gate": _ratio(delta_g, action_norm),
            "action_zero": action_norm == 0.0,
            "action_zero_returns_null": action_norm == 0.0,
        })
        row["metric_1"].update({
            "positive_repair_uses_max_base_minus_rcsp_zero": True,
            "signed_repair_used_for_ntsf": False,
        })
        row["counterfactual"]["metric_ratios"] = {
            "metric_ratio_before_28_over_10": row["counterfactual"]["cf28"]["metric_ratio_before_28_over_10"],
            "metric_ratio_base_28_over_10": row["counterfactual"]["cf28"]["metric_ratio_base_28_over_10"],
            "metric_ratio_rcsp_28_over_10": row["counterfactual"]["cf28"]["metric_ratio_rcsp_28_over_10"],
        }
        row["actual_width_metric_parity"] = parity
        for key in ("_source_p2", "_source_base", "_source_rcsp", "_index"):
            row.pop(key, None)
    return rows


def _make_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != PRIMARY_CASES:
        raise ValueError("scientific summaries require exactly 32 primary rows")
    return {
        "spread": spread_summaries(rows),
        "direction_efficiency": _direction_group_summary(rows),
        "counterfactual_width": _counterfactual_summary(rows),
    }


def _read_only_integrity(
    before_files: Mapping[str, str],
    immutable_paths: Mapping[str, Path],
    base_hash: str,
    adapter_hash: str,
    base: torch.nn.Module,
    model: rcsp.FrozenBaseRCSPModel,
) -> dict[str, Any]:
    after_files = {name: _file_sha256(path) for name, path in immutable_paths.items()}
    base_after = safe.state_hash(base.state_dict())
    adapter_after = safe.state_hash(model.adapter.state_dict())
    grads_none = all(parameter.grad is None for parameter in model.parameters())
    result = {
        "base_state_sha256_before_after": base_hash,
        "adapter_state_sha256_before_after": adapter_hash,
        "base_unchanged": base_after == base_hash,
        "adapter_unchanged": adapter_after == adapter_hash,
        "immutable_artifacts_unchanged": before_files == after_files,
        "all_model_parameter_grad_none": grads_none,
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
    if not all(result[key] for key in ("base_unchanged", "adapter_unchanged", "immutable_artifacts_unchanged", "all_model_parameter_grad_none")):
        raise RuntimeError("Phase 2.1 read-only integrity contract failed")
    return result


def run(args: argparse.Namespace) -> int:
    phase2_path = Path(args.phase2_report).resolve()
    phase2_report, phase2_hash, paths = _validate_phase2_lineage(phase2_path)
    upstream = _validate_upstream_reports(paths)
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    output = Path(args.output_dir).resolve()
    result_dir = output / "result"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Phase 2.1 output directory must be a fresh empty directory")
    if output.is_relative_to(paths["source"]) or output.is_relative_to(paths["trajectory"]):
        raise FileExistsError("Phase 2.1 output overlaps immutable lineage input")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(exist_ok=False)
    failure_path = result_dir / "failure.json"
    immutable_paths = {
        "phase2/report.json": phase2_path,
        **{f"source/{name}": paths["source"] / name for name in ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")},
        **{f"trajectory/{name}": paths["trajectory"] / name for name in ("report.json", "experiment.json", "diagnostic_latest.pt", "updates.jsonl")},
        "rcsp/report.json": paths["rcsp_directory"] / "report.json",
        "rcsp/reporting_logic_review_v1.json": paths["rcsp_directory"] / "reporting_logic_review_v1.json",
        "rcsp/adapter_checkpoint": paths["adapter_checkpoint"],
        "phase1/report.json": paths["phase1_report"],
        "single_decomposition/report.json": paths["single_decomposition_report"],
        "parameter_attribution/report.json": paths["parameter_attribution_report"],
    }
    before_files = {name: _file_sha256(path) for name, path in immutable_paths.items()}
    implementation_paths = {
        "motion_models.py": Path(m.__file__).resolve(),
        "boundary_observables.py": Path(__import__("motion_geometry.boundary_observables", fromlist=["__name__"]).__file__).resolve(),
        "rcsp.py": Path(rcsp.__file__).resolve(),
        "alignment.py": Path(alignment.__file__).resolve(),
        "failure.py": Path(failure.__file__).resolve(),
        "group.py": Path(group_audit.__file__).resolve(),
        "safe.py": Path(safe.__file__).resolve(),
        "phase2.py": Path(phase2.__file__).resolve(),
    }
    implementation_before = {name: _file_sha256(path) for name, path in implementation_paths.items()}
    try:
        source = paths["source"]
        trajectory, trajectory_paths, _trajectory_hashes, trajectory_report, experiment, checkpoint = failure._load_trajectory(
            paths["trajectory"], failure.TRAJECTORY_COMMIT
        )
        state, bank, cfg, source_metadata = group_audit.load_frozen_source(
            source,
            group_audit.LEGACY_COMMIT,
            legacy_core_strength=LEGACY_CORE_STRENGTH,
            legacy_transition_strength=LEGACY_TRANSITION_STRENGTH,
        )
        if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
            raise ValueError("trajectory does not reference the Phase 2 frozen source")
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        cfg = dataclasses.replace(cfg, device=str(device))
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices), group_audit.frozen_environment(
            state["fingerprint"], source_metadata["decoder_strengths"]
        ):
            base = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
            base.load_state_dict(checkpoint["model_state_dict"], strict=True)
            base.eval()
            base_hash = safe.state_hash(base.state_dict())
            if base_hash != trajectory_report["final_state_sha256"]:
                raise RuntimeError("loaded base differs from immutable trajectory final state")
            model = rcsp.FrozenBaseRCSPModel(base)
            model.adapter.load_state_dict(upstream["rcsp"]["adapter_checkpoint"]["adapter_state_dict"], strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            model.eval()
            adapter_hash = safe.state_hash(model.adapter.state_dict())
            if adapter_hash != upstream["rcsp"]["report"]["parameter_update_scope"]["adapter_state_sha256"]:
                raise RuntimeError("loaded RCSP adapter state hash mismatch")
            probe, probe_hash = safe.load_probe(source, state, bank, cfg)
            final_batch, final_metadata = alignment.combine_final_banks(failure.final_banks(bank, probe, cfg))
            final_batch = rcsp._move_batch(final_batch, device)
            phase2._validate_fixed_metadata(final_metadata)
            cf_masks, mask_parity = mask_reconstruction_parity(
                final_batch["seam"], cfg.fps, cfg.transition_mask_halo_seconds
            )
            if not mask_parity["verified"]:
                raise RuntimeError("64-case counterfactual mask reconstruction parity failed closed")
            all_rows: list[dict[str, Any]] = []
            for start in range(0, FINAL_CASES, rcsp.FINAL_CHUNK_SIZE):
                stop = start + rcsp.FINAL_CHUNK_SIZE
                chunk = {key: value[start:stop] for key, value in final_batch.items()}
                metadata = final_metadata[start:stop]
                p2_rows, _ = phase2._evaluate_chunk(base, model, chunk, metadata, cfg)
                outputs = _capture_outputs(base, model, chunk, metadata, cfg)
                chunk_masks = {width: mask[start:stop] for width, mask in cf_masks.items()}
                all_rows.extend(
                    _attach_case_metrics(
                        p2_rows, metadata, outputs, chunk_masks, phase2_report, upstream, cfg
                    )
                )
            if len(all_rows) != FINAL_CASES:
                raise RuntimeError("Phase 2.1 did not evaluate exactly 64 fixed final cases")
            primary_rows = [row for row in all_rows if row["role"] == PRIMARY_ROLE]
            excluded_rows = [row for row in all_rows if row["role"] == EXCLUDED_ROLE]
            if len(primary_rows) != PRIMARY_CASES or len(excluded_rows) != PRIMARY_CASES:
                raise RuntimeError("Phase 2.1 primary/excluded cohort count mismatch")
            if not all(row["actual_width_metric_parity"]["verified"] for row in primary_rows):
                raise RuntimeError("Phase 2.1 actual-width metric parity failed closed")
            summaries = _make_summaries(primary_rows)
            decisions = adjudicate(summaries, counterfactual_mask_parity_verified=True)
            counterfactual_contract = {
                "same_case": True,
                "same_motion": True,
                "same_degraded_transition": True,
                "same_clean_reference": True,
                "same_output": True,
                "same_seam_center": True,
                "evaluation_width_only_changed": True,
                "same_fps": True,
                "same_model_and_adapter": True,
                "same_alpha": True,
                "same_support_and_repair_output": True,
                "mask_reconstruction_parity_cases": mask_parity["cases"],
                "mask_reconstruction_parity_verified": mask_parity["verified"],
                "fake_case_pairing_performed": False,
                "observed_group_pairing": "UNPAIRED",
                "counterfactual_pairing": "WITHIN_CASE_BY_CONSTRUCTION",
                "mask_reconstruction_source": mask_parity["source"],
                "parity_rows": mask_parity["rows"],
            }
            integrity_before = safe.state_hash(base.state_dict()), safe.state_hash(model.adapter.state_dict())
            del probe
            if _file_sha256(source / "probe_bank.pt") != probe_hash:
                raise RuntimeError("probe artifact changed during Phase 2.1 audit")
            integrity = _read_only_integrity(
                before_files, immutable_paths, integrity_before[0], integrity_before[1], base, model
            )
        implementation_after = {name: _file_sha256(path) for name, path in implementation_paths.items()}
        if implementation_before != implementation_after:
            raise RuntimeError("production implementation files changed during Phase 2.1 audit")
        report = {
            "schema": SCHEMA,
            "completed": True,
            "provenance": {
                "runtime_commit": runtime_commit,
                "expected_main_commit": args.expected_main_commit,
                "phase2_runtime_commit": FROZEN_PHASE2_COMMIT,
                "root": phase2_report.get("provenance", {}).get("root"),
                "phase2_report": str(phase2_path),
                "source": str(paths["source"]),
                "trajectory": str(trajectory),
                "rcsp_directory": str(paths["rcsp_directory"]),
                "adapter_checkpoint": str(paths["adapter_checkpoint"]),
                "phase1_report": str(paths["phase1_report"]),
                "single_decomposition_report": str(paths["single_decomposition_report"]),
                "parameter_attribution_report": str(paths["parameter_attribution_report"]),
                "phase2_report_sha256": phase2_hash,
                "immutable_input_sha256": before_files,
                "implementation_sha256_before": implementation_before,
                "implementation_sha256_after": implementation_after,
            },
            "lineage": {
                "phase2_report_path": str(phase2_path),
                "phase2_report_sha256": phase2_hash,
                "phase2_schema": phase2_report["schema"],
                "phase2_completed": True,
                "phase2_primary_cases": PRIMARY_CASES,
                "phase2_runtime_commit": FROZEN_PHASE2_COMMIT,
                "adapter_checkpoint_path_read_from_phase2_lineage": str(paths["adapter_checkpoint"]),
                "no_latest_artifact_search": True,
            },
            "primary_cohort": {
                "cases": PRIMARY_CASES,
                "groups": {group: CASES_PER_GROUP for group in GROUP_ORDER},
                "role": PRIMARY_ROLE,
                "widths": list(WIDTHS),
            },
            "excluded_cohorts": {
                "single_recording": {
                    "cases": len(excluded_rows),
                    "excluded_from_primary_adjudication": True,
                    "excluded_from_scientific_summaries": True,
                }
            },
            "counterfactual_contract": counterfactual_contract,
            "metric_definitions": {
                "normalized_temporal_spread_fraction": {
                    "formula": "NTSF(c)=((sum_{t in A} c_t)^2/sum_{t in A} c_t^2)/|A|",
                    "contribution_domain": "nonnegative aligned seam_acceleration + seam_jerk contribution at stencil-start indices",
                    "active_support": "union of authoritative derivative supports touching seam core",
                    "zero_square_sum": None,
                    "variants": ["rcsp_error_spread_fraction", "positive_repair_spread_fraction"],
                    "positive_repair": "max(BASE_error_t - RCSP_error_t, 0); signed repair is descriptive only",
                    "implementation_source": "Phase 2 authoritative_temporal_components and boundary_metrics_torch",
                },
                "relative_gate_gain_per_applied_action_norm": {
                    "G_base": "(M_before-M_base)/M_before using production observable_gate",
                    "G_rcsp": "(M_before-M_rcsp)/M_before using production observable_gate",
                    "delta_G_rcsp": "G_rcsp-G_base",
                    "applied_action": "final_tangent_RCSP-final_tangent_BASE before manifold retraction, geometric 75D root(3)+24 joints(72), contact excluded",
                    "E_gate": "(G_rcsp-G_base)/||final_tangent_RCSP-final_tangent_BASE||_2; null when norm is zero",
                    "frozen_direction_covariate": "adapter_direction_cosine from Phase 2/RCSP; no new gradient audit",
                    "implementation_source": "motion_geometry.boundary_observables.observable_gate and production decoder trace after_cap",
                },
                "same_boundary_counterfactual_metric": {
                    "formula": "same frozen motion/output/support tensors; evaluate boundary_metrics_torch with seam_cf10 and seam_cf28",
                    "delta_G_counterfactual": "G_rcsp_cf28-G_rcsp_cf10",
                    "components": ["metric_ratio_before_28_over_10", "metric_ratio_base_28_over_10", "metric_ratio_rcsp_28_over_10"],
                    "pure_scalar_normalization_can_change_relative_gate": False,
                    "cancellation_identity": "(cB-cA)/(cB)=(B-A)/B",
                },
            },
            "case_level": primary_rows,
            "excluded_case_level": excluded_rows,
            "summaries": summaries,
            "adjudication": decisions,
            "state_integrity": integrity,
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
            "next_action": "design_minimal_intervention_from_adjudicated_mechanism",
        }
        _exclusive_json(result_dir / "report.json", report)
        print(json.dumps({
            "stage": "refiner_width_mechanism_adjudication_audit_complete",
            "report": str(result_dir / "report.json"),
            "primary_cases": PRIMARY_CASES,
            "mask_reconstruction_parity_verified": True,
            "adjudicated_primary_mechanism": decisions["adjudicated_primary_mechanism"],
            "optimizer_steps": 0,
            "production_model_modified": False,
            "scientific_acceptance": False,
            "pilot_allowed": False,
        }, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(failure_path, {
                "schema": SCHEMA,
                "completed": False,
                "error": {"type": type(error).__name__, "message": str(error)},
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "parameter_update_performed": False,
                "production_model_modified": False,
                "scientific_acceptance": False,
                "publish_allowed": False,
                "pilot_allowed": False,
            })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
