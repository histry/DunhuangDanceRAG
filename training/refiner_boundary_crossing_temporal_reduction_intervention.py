"""Frozen-state Phase 3 audit for boundary-crossing temporal reduction (BCTR).

This module is a deliberately minimal intervention.  It reuses the exact A0
step-400 base, the already trained RCSP adapter, the recorded final banks and
the production decoder.  Only the support used by the temporal reduction is
changed: a derivative stencil is selected when it touches both the seam core
and its complement.  No model, decoder, direction, threshold, or checkpoint
is changed, and no update path is present in this module.
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

from motion_geometry import boundary_observables
from motion_geometry import product_manifold
from motion_geometry.boundary_observables import observable_gate
from training import motion_models as m
from training import refiner_cross_width_normalization_audit as phase2
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_safe_start_diagnostics as safe
from training import refiner_temporal_action_alignment_audit as alignment
from training import refiner_width_mechanism_adjudication_audit as phase21


SCHEMA = "refiner_boundary_crossing_temporal_reduction_intervention_v1"
FROZEN_PHASE21_COMMIT = "c461ba44689103cd0690488267e3bd42507ad7ab"
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
PARITY_ATOL = phase21.PARITY_ATOL
PARITY_RTOL = phase21.PARITY_RTOL
MAJOR_GAP_FRACTION = 0.50
TEMPORAL_FLOOR = 1.0e-6
LEGACY_CORE_STRENGTH = 0.02
LEGACY_TRANSITION_STRENGTH = 1.0


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


def _identity(row: Mapping[str, Any]) -> str:
    return phase2._identity_key(row)


def _bool_exact(value: Any, expected: bool, label: str) -> None:
    if value is not expected:
        raise ValueError(f"{label} must be exactly {expected}")


def _require_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"{label} hash is missing")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if _file_sha256(path) != expected:
        raise ValueError(f"{label} sha256 mismatch")


def _validate_phase21_lineage(path: Path) -> tuple[dict[str, Any], str, dict[str, Path], dict[str, Any]]:
    """Validate the explicit Phase 2.1 report and every recorded input hash."""
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2.1 report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    report_hash = _file_sha256(path)
    if report.get("schema") != phase21.SCHEMA:
        raise ValueError("Phase 2.1 schema mismatch")
    _bool_exact(report.get("completed"), True, "Phase 2.1 completed")
    provenance = report.get("provenance", {})
    adjudication = report.get("adjudication", {})
    integrity = report.get("state_integrity", {})
    if provenance.get("runtime_commit") != FROZEN_PHASE21_COMMIT:
        raise ValueError("Phase 2.1 runtime commit is not the frozen parent commit")
    if provenance.get("expected_main_commit") not in (None, FROZEN_PHASE21_COMMIT):
        raise ValueError("Phase 2.1 expected main commit mismatch")
    if adjudication.get("adjudicated_primary_mechanism") != "MIXED_WIDTH_MECHANISM":
        raise ValueError("Phase 2.1 mechanism is not MIXED_WIDTH_MECHANISM")
    expected_evidence = {
        "normalization_evidence": True,
        "temporal_spreading_evidence": False,
        "width_conditioned_direction_evidence": True,
    }
    for field, expected in expected_evidence.items():
        values = adjudication.get(field, {})
        if values.get("seen") is not expected:
            raise ValueError(f"Phase 2.1 {field}.seen mismatch")
        if values.get("new_position") is not expected:
            raise ValueError(f"Phase 2.1 {field}.new_position mismatch")
    if adjudication.get("primary_intervention_order") != [
        "metric/support-time intervention", "direction intervention"
    ]:
        raise ValueError("Phase 2.1 intervention order mismatch")
    false_or_zero = (
        (integrity.get("optimizer_steps"), 0, "Phase 2.1 optimizer steps"),
        (report.get("optimizer_steps"), 0, "Phase 2.1 top-level optimizer steps"),
    )
    for value, expected, label in false_or_zero:
        if value != expected:
            raise ValueError(f"{label} mismatch")
    for field in (
        "parameter_update_performed", "production_model_modified", "production_inference_modified",
        "scientific_acceptance", "publish_allowed", "pilot_allowed",
    ):
        _bool_exact(report.get(field), False, f"Phase 2.1 {field}")
    for field in (
        "parameter_update_performed", "production_model_modified", "production_inference_modified",
        "scientific_acceptance", "publish_allowed", "pilot_allowed",
    ):
        _bool_exact(integrity.get(field), False, f"Phase 2.1 state_integrity.{field}")
    _bool_exact(integrity.get("base_unchanged"), True, "Phase 2.1 base_unchanged")
    _bool_exact(integrity.get("adapter_unchanged"), True, "Phase 2.1 adapter_unchanged")

    cohort = report.get("primary_cohort", {})
    if cohort.get("cases") != PRIMARY_CASES or cohort.get("role") != PRIMARY_ROLE:
        raise ValueError("Phase 2.1 primary cohort mismatch")
    if cohort.get("groups") != {group: CASES_PER_GROUP for group in GROUP_ORDER}:
        raise ValueError("Phase 2.1 primary groups are not the frozen 4x8 cohort")
    excluded = report.get("excluded_cohorts", {}).get(EXCLUDED_ROLE, {})
    _bool_exact(excluded.get("excluded_from_primary_adjudication"), True, "single-recording primary exclusion")
    _bool_exact(excluded.get("excluded_from_scientific_summaries"), True, "single-recording summary exclusion")

    phase2_report_path_value = provenance.get("phase2_report")
    if not isinstance(phase2_report_path_value, str) or not phase2_report_path_value:
        raise ValueError("Phase 2.1 does not record an explicit Phase 2 report")
    phase2_path = Path(phase2_report_path_value).resolve()
    phase2_report, phase2_hash, phase2_paths = phase21._validate_phase2_lineage(phase2_path)
    if provenance.get("phase2_report_sha256") != phase2_hash:
        raise ValueError("Phase 2.1 Phase 2 report hash mismatch")

    recorded_paths = {
        "source": provenance.get("source"),
        "trajectory": provenance.get("trajectory"),
        "rcsp_directory": provenance.get("rcsp_directory"),
        "adapter_checkpoint": provenance.get("adapter_checkpoint"),
        "phase1_report": provenance.get("phase1_report"),
        "single_decomposition_report": provenance.get("single_decomposition_report"),
        "parameter_attribution_report": provenance.get("parameter_attribution_report"),
    }
    expected_paths = dict(phase2_paths)
    expected_paths["phase2_report_path"] = phase2_path
    for name, value in recorded_paths.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"Phase 2.1 lineage path is missing: {name}")
        if Path(value).resolve() != expected_paths[name]:
            raise ValueError(f"Phase 2.1 lineage path mismatch: {name}")
    lineage = report.get("lineage", {})
    if lineage.get("phase2_report_path") not in (None, str(phase2_path)):
        raise ValueError("Phase 2.1 self-reported Phase 2 path mismatch")
    if lineage.get("adapter_checkpoint_path_read_from_phase2_lineage") not in (
        None, str(expected_paths["adapter_checkpoint"])
    ):
        raise ValueError("Phase 2.1 adapter lineage mismatch")
    _bool_exact(lineage.get("no_latest_artifact_search"), True, "Phase 2.1 latest-artifact search contract")

    expected_hashes = provenance.get("immutable_input_sha256", {})
    hash_paths = {
        "phase2/report.json": phase2_path,
        **{f"source/{name}": expected_paths["source"] / name for name in (
            "diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt"
        )},
        **{f"trajectory/{name}": expected_paths["trajectory"] / name for name in (
            "report.json", "experiment.json", "diagnostic_latest.pt", "updates.jsonl"
        )},
        "rcsp/report.json": expected_paths["rcsp_directory"] / "report.json",
        "rcsp/reporting_logic_review_v1.json": expected_paths["rcsp_directory"] / "reporting_logic_review_v1.json",
        "rcsp/adapter_checkpoint": expected_paths["adapter_checkpoint"],
        "phase1/report.json": expected_paths["phase1_report"],
        "single_decomposition/report.json": expected_paths["single_decomposition_report"],
        "parameter_attribution/report.json": expected_paths["parameter_attribution_report"],
    }
    for name, input_path in hash_paths.items():
        _require_hash(input_path, expected_hashes.get(name), f"Phase 2.1 {name}")
    upstream = phase21._validate_upstream_reports(expected_paths)
    return report, report_hash, expected_paths, upstream | {"phase2_report": phase2_report, "phase2_hash": phase2_hash}


def _core_mask(seam: torch.Tensor) -> torch.Tensor:
    if seam.ndim == 3:
        seam = seam[..., 0]
    if seam.ndim != 2:
        raise ValueError("seam must have shape [B,T] or [B,T,1]")
    return seam >= 0.5


def boundary_crossing_support(seam: torch.Tensor, order: int) -> torch.Tensor:
    """Return stencils touching both core and non-core frames."""
    core = _core_mask(seam)
    if order < 1:
        raise ValueError("derivative order must be positive")
    length = int(core.shape[1]) - int(order)
    if length <= 0:
        return core[:, :0]
    windows = torch.stack([core[:, index:index + length] for index in range(order + 1)])
    return windows.any(0) & (~windows).any(0)


def _bctr_temporal_rows(
    joints: torch.Tensor, seam: torch.Tensor, fps: float
) -> list[dict[str, Any]]:
    """Compute BCTR terms from the production float64 derivative values."""
    if joints.ndim != 4 or tuple(joints.shape[-2:]) != (m.NUM_JOINTS, 3):
        raise ValueError("joints must have shape [B,T,24,3]")
    terms = phase2._temporal_component_tensors(joints, seam, fps)
    rows = [{"terms": {}, "valid": True} for _ in range(int(joints.shape[0]))]
    for name in ("seam_acceleration", "seam_jerk"):
        term = terms[name]
        support = boundary_crossing_support(seam, int(term["order"]))
        values = term["values"]
        for index, row in enumerate(rows):
            count = int(support[index].sum().item())
            raw = _finite((values[index] * support[index].to(values.dtype)).sum(), f"BCTR {name} numerator")
            normalized = None if count == 0 else raw / float(count) / float(term["scale"])
            if normalized is not None and not math.isfinite(normalized):
                raise FloatingPointError(f"nonfinite BCTR {name}")
            row["terms"][name] = {
                "order": int(term["order"]),
                "scale": float(term["scale"]),
                "raw_numerator": raw,
                "crossing_support_count": count,
                "normalized_value": normalized,
                "valid": count > 0,
            }
    for row in rows:
        acceleration = row["terms"]["seam_acceleration"]["normalized_value"]
        jerk = row["terms"]["seam_jerk"]["normalized_value"]
        row["temporal_energy"] = None if acceleration is None or jerk is None else float(acceleration + jerk)
        row["valid"] = bool(acceleration is not None and jerk is not None)
    return rows


def _current_state(before: Mapping[str, Any], after: Mapping[str, Any], cfg: Any) -> dict[str, Any]:
    gate = observable_gate(dict(before), dict(after), cfg)
    threshold = float(cfg.checkpoint_validation_min_temporal_repair_gain)
    return {
        "M": float(after["temporal_energy"]),
        "M_before": float(before["temporal_energy"]),
        "G": _finite(gate["temporal_gain"], "current temporal gain"),
        "gate_margin": _finite(gate["temporal_gain"], "current temporal gain") - threshold,
        "temporal_pass": bool(gate["temporal_accepted"]),
        "endpoint_acceptance": bool(gate["endpoint_accepted"]),
        "jerk_non_regression": bool(gate["jerk_non_regression"]),
        "gate": gate,
    }


def _candidate_state(
    original_before: Mapping[str, Any],
    original_after: Mapping[str, Any],
    bctr_before: Mapping[str, Any],
    bctr_after: Mapping[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    valid = bool(original_before["valid"] and original_after["valid"] and bctr_before["valid"] and bctr_after["valid"])
    before = dict(original_before)
    after = dict(original_after)
    before["temporal_energy"] = float(bctr_before["temporal_energy"] or 0.0)
    after["temporal_energy"] = float(bctr_after["temporal_energy"] or 0.0)
    before["valid"] = valid
    after["valid"] = valid
    gate = observable_gate(before, after, cfg)
    threshold = float(cfg.checkpoint_validation_min_temporal_repair_gain)
    gain = _finite(gate["temporal_gain"], "BCTR temporal gain") if valid else None
    return {
        "valid": valid,
        "G": gain,
        "gate_margin": None if gain is None else gain - threshold,
        "temporal_pass": bool(valid and gate["temporal_accepted"]),
        "endpoint_acceptance": bool(valid and gate["endpoint_accepted"]),
        "jerk_non_regression": bool(valid and gate["jerk_non_regression"]),
        "gate": gate,
    }


def _parity_row(
    identity: str,
    width: int,
    current: Mapping[str, Any],
    phase21_row: Mapping[str, Any],
    upstream_base: Mapping[str, Any],
    upstream_rcsp: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    reported = phase21_row.get("actual_width", {})
    source = {
        "M_before": upstream_rcsp.get("observable", {}).get("before", {}).get("temporal_energy"),
        "M_base": upstream_base.get("observable", {}).get("after", {}).get("temporal_energy"),
        "M_rcsp": upstream_rcsp.get("observable", {}).get("after", {}).get("temporal_energy"),
        "G_base": upstream_base.get("observable", {}).get("temporal_gain"),
        "G_rcsp": upstream_rcsp.get("observable", {}).get("temporal_gain"),
        "gate_margin_base": None,
        "gate_margin_rcsp": None,
    }
    if source["G_base"] is not None:
        source["gate_margin_base"] = float(source["G_base"]) - threshold
    if source["G_rcsp"] is not None:
        source["gate_margin_rcsp"] = float(source["G_rcsp"]) - threshold
    observed = {
        "M_before": current["before"]["M_before"],
        "M_base": current["base"]["M"],
        "M_rcsp": current["rcsp"]["M"],
        "G_base": current["base"]["G"],
        "G_rcsp": current["rcsp"]["G"],
        "gate_margin_base": current["base"]["gate_margin"],
        "gate_margin_rcsp": current["rcsp"]["gate_margin"],
    }
    errors: dict[str, float] = {}
    values: dict[str, Any] = {}
    for key, value in observed.items():
        candidates = [reported.get(key), source.get(key)]
        if any(candidate is None for candidate in candidates):
            raise ValueError(f"current parity reference is incomplete for {identity}: {key}")
        error = max(abs(float(value) - float(candidate)) for candidate in candidates)
        errors[key] = float(error)
        values[key] = {
            "recomputed": float(value),
            "phase21": float(reported[key]),
            "rcsp_upstream": float(source[key]),
        }
    maximum = max(errors.values(), default=0.0)
    return {
        "identity": identity,
        "actual_width": int(width),
        "verified": bool(maximum <= PARITY_ATOL),
        "max_abs_error": float(maximum),
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "values": values,
        "errors": errors,
    }


def _evaluate_chunk(
    base: torch.nn.Module,
    model: rcsp.FrozenBaseRCSPModel,
    batch: Mapping[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    phase21_report: Mapping[str, Any],
    upstream: Mapping[str, Any],
    cfg: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = phase21._capture_outputs(base, model, batch, metadata, cfg)
    count = len(metadata)
    seam = batch["seam"]
    states = torch.cat((outputs["before_joints"], outputs["base_joints"], outputs["rcsp_joints"]))
    repeated_seam = torch.cat((seam, seam, seam))
    current_metrics = phase21._metric_rows(states, repeated_seam, cfg.fps)
    bctr_metrics = _bctr_temporal_rows(states, repeated_seam, cfg.fps)
    phase21_map = {_identity(row): row for row in list(phase21_report.get("case_level", [])) + list(phase21_report.get("excluded_case_level", []))}
    upstream_base = phase2._report_case_map(upstream["rcsp"]["report"], "BASE")
    upstream_rcsp = phase2._report_case_map(upstream["rcsp"]["report"], "RCSP")
    rows = []
    parity_rows = []
    for index, meta in enumerate(metadata):
        identity = _identity(meta)
        if identity not in phase21_map:
            raise ValueError(f"Phase 2.1 case is missing: {identity}")
        before_metric = current_metrics[index]
        base_metric = current_metrics[count + index]
        rcsp_metric = current_metrics[2 * count + index]
        before_current = _current_state(before_metric, before_metric, cfg)
        base_current = _current_state(before_metric, base_metric, cfg)
        rcsp_current = _current_state(before_metric, rcsp_metric, cfg)
        current = {"before": before_current, "base": base_current, "rcsp": rcsp_current}
        if meta["role"] == PRIMARY_ROLE:
            parity = _parity_row(
                identity, int(meta["width"]), current, phase21_map[identity],
                upstream_base[identity], upstream_rcsp[identity],
                float(cfg.checkpoint_validation_min_temporal_repair_gain),
            )
            if not parity["verified"]:
                raise RuntimeError(f"current metric parity failed for {identity}")
            parity_rows.append(parity)
        before_bctr = bctr_metrics[index]
        base_bctr = bctr_metrics[count + index]
        rcsp_bctr = bctr_metrics[2 * count + index]
        base_candidate = _candidate_state(before_metric, base_metric, before_bctr, base_bctr, cfg)
        rcsp_candidate = _candidate_state(before_metric, rcsp_metric, before_bctr, rcsp_bctr, cfg)
        bctr_valid = bool(before_bctr["valid"] and base_bctr["valid"] and rcsp_bctr["valid"])
        bctr = {
            "M_before_BCTR": before_bctr["temporal_energy"],
            "M_base_BCTR": base_bctr["temporal_energy"],
            "M_rcsp_BCTR": rcsp_bctr["temporal_energy"],
            "G_base_BCTR": base_candidate["G"],
            "G_rcsp_BCTR": rcsp_candidate["G"],
            "delta_G_adapter_BCTR": (
                None if base_candidate["G"] is None or rcsp_candidate["G"] is None
                else float(rcsp_candidate["G"] - base_candidate["G"])
            ),
            "gate_margin_base_BCTR": base_candidate["gate_margin"],
            "gate_margin_rcsp_BCTR": rcsp_candidate["gate_margin"],
            "candidate_temporal_pass_base": base_candidate["temporal_pass"],
            "candidate_temporal_pass_rcsp": rcsp_candidate["temporal_pass"],
            "original_jerk_non_regression": {
                "BASE": base_current["jerk_non_regression"],
                "RCSP": rcsp_current["jerk_non_regression"],
            },
            "original_endpoint_acceptance": {
                "BASE": base_current["endpoint_acceptance"],
                "RCSP": rcsp_current["endpoint_acceptance"],
            },
            "candidate_overall_acceptance": {
                "BASE": bool(base_candidate["endpoint_acceptance"] and base_candidate["temporal_pass"]),
                "RCSP": bool(rcsp_candidate["endpoint_acceptance"] and rcsp_candidate["temporal_pass"]),
            },
            "crossing_support_counts": {
                "before": {
                    "acceleration": before_bctr["terms"]["seam_acceleration"]["crossing_support_count"],
                    "jerk": before_bctr["terms"]["seam_jerk"]["crossing_support_count"],
                },
                "BASE": {
                    "acceleration": base_bctr["terms"]["seam_acceleration"]["crossing_support_count"],
                    "jerk": base_bctr["terms"]["seam_jerk"]["crossing_support_count"],
                },
                "RCSP": {
                    "acceleration": rcsp_bctr["terms"]["seam_acceleration"]["crossing_support_count"],
                    "jerk": rcsp_bctr["terms"]["seam_jerk"]["crossing_support_count"],
                },
            },
            "valid": bctr_valid,
        }
        anti_gaming = {
            "endpoint_semantics_unchanged": bool(
                base_current["endpoint_acceptance"] == base_candidate["endpoint_acceptance"]
                and rcsp_current["endpoint_acceptance"] == rcsp_candidate["endpoint_acceptance"]
            ),
            "jerk_semantics_unchanged": bool(
                base_current["jerk_non_regression"] == base_candidate["jerk_non_regression"]
                and rcsp_current["jerk_non_regression"] == rcsp_candidate["jerk_non_regression"]
            ),
            "outputs_unchanged": bool(outputs["same_output_tensors"]),
            "state_unchanged": True,
        }
        rows.append({
            **meta,
            "identity": identity,
            "observed_group_pairing": "UNPAIRED",
            "counterfactual_pairing": None,
            "current": {
                "M_before": before_current["M_before"],
                "M_base": base_current["M"],
                "M_rcsp": rcsp_current["M"],
                "G_base": base_current["G"],
                "G_rcsp": rcsp_current["G"],
                "gate_margin_base": base_current["gate_margin"],
                "gate_margin_rcsp": rcsp_current["gate_margin"],
                "temporal_pass_base": base_current["temporal_pass"],
                "temporal_pass_rcsp": rcsp_current["temporal_pass"],
                "endpoint_acceptance_base": base_current["endpoint_acceptance"],
                "endpoint_acceptance_rcsp": rcsp_current["endpoint_acceptance"],
                "jerk_non_regression_base": base_current["jerk_non_regression"],
                "jerk_non_regression_rcsp": rcsp_current["jerk_non_regression"],
            },
            "bctr": bctr,
            "M_before_BCTR": bctr["M_before_BCTR"],
            "M_base_BCTR": bctr["M_base_BCTR"],
            "M_rcsp_BCTR": bctr["M_rcsp_BCTR"],
            "G_base_BCTR": bctr["G_base_BCTR"],
            "G_rcsp_BCTR": bctr["G_rcsp_BCTR"],
            "delta_G_adapter_BCTR": bctr["delta_G_adapter_BCTR"],
            "gate_margin_base_BCTR": bctr["gate_margin_base_BCTR"],
            "gate_margin_rcsp_BCTR": bctr["gate_margin_rcsp_BCTR"],
            "candidate_temporal_pass_base": bctr["candidate_temporal_pass_base"],
            "candidate_temporal_pass_rcsp": bctr["candidate_temporal_pass_rcsp"],
            "original_jerk_non_regression": bctr["original_jerk_non_regression"],
            "original_endpoint_acceptance": bctr["original_endpoint_acceptance"],
            "candidate_overall_acceptance": bctr["candidate_overall_acceptance"],
            "crossing_support_counts": bctr["crossing_support_counts"],
            "anti_gaming": anti_gaming,
            "current_metric_parity": parity if meta["role"] == PRIMARY_ROLE else None,
        })
    return rows, {
        "rows": parity_rows,
        "cases": len(parity_rows),
        "verified_cases": sum(bool(row["verified"]) for row in parity_rows),
        "verified": len(parity_rows) == PRIMARY_CASES and all(row["verified"] for row in parity_rows),
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
    }


def _scope_rows(rows: list[Mapping[str, Any]], scope: str) -> list[Mapping[str, Any]]:
    if scope == "overall":
        return list(rows)
    split = "new_position" if scope == "new" else scope
    return [row for row in rows if row["split"] == split]


def _gap(width10: float | None, width28: float | None) -> float | None:
    if width10 is None or width28 is None:
        return None
    return float(width28 - width10)


def _scope_summary(rows: list[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    current10 = _median(row["current"]["G_rcsp"] for row in rows if int(row["width"]) == 10)
    current28 = _median(row["current"]["G_rcsp"] for row in rows if int(row["width"]) == 28)
    bctr10 = _median(row["bctr"]["G_rcsp_BCTR"] for row in rows if int(row["width"]) == 10)
    bctr28 = _median(row["bctr"]["G_rcsp_BCTR"] for row in rows if int(row["width"]) == 28)
    current_gap = _gap(current10, current28)
    bctr_gap = _gap(bctr10, bctr28)
    current_pass = sum(bool(row["current"]["temporal_pass_rcsp"]) for row in rows)
    bctr_pass = sum(bool(row["bctr"]["candidate_temporal_pass_rcsp"]) for row in rows)
    current_base_pass = sum(bool(row["current"].get("temporal_pass_base", False)) for row in rows)
    bctr_base_pass = sum(bool(row["bctr"].get("candidate_temporal_pass_base", False)) for row in rows)
    newly_rescued = [row["identity"] for row in rows if not row["current"]["temporal_pass_rcsp"] and row["bctr"]["candidate_temporal_pass_rcsp"]]
    width10_lost = [row["identity"] for row in rows if int(row["width"]) == 10 and row["current"]["temporal_pass_rcsp"] and not row["bctr"]["candidate_temporal_pass_rcsp"]]
    delta10 = None if bctr10 is None or current10 is None else float(bctr10 - current10)
    delta28 = None if bctr28 is None or current28 is None else float(bctr28 - current28)
    valid_count = sum(bool(row["bctr"]["valid"]) for row in rows)
    endpoint_same = all(row["anti_gaming"]["endpoint_semantics_unchanged"] for row in rows)
    jerk_same = all(row["anti_gaming"]["jerk_semantics_unchanged"] for row in rows)
    outputs_same = all(row["anti_gaming"]["outputs_unchanged"] for row in rows)
    state_same = all(row["anti_gaming"]["state_unchanged"] for row in rows)
    gap_shrink = None if current_gap is None or current_gap == 0.0 or bctr_gap is None else 1.0 - abs(bctr_gap) / abs(current_gap)
    gap_reduced = bool(current_gap is not None and bctr_gap is not None and abs(bctr_gap) <= MAJOR_GAP_FRACTION * abs(current_gap))
    width28_improved = bool(bctr28 is not None and current28 is not None and bctr28 > current28)
    width10_non_degraded = bool(bctr10 is not None and current10 is not None and bctr10 >= current10)
    valid_all = valid_count == len(rows)
    supported = bool(gap_reduced and width28_improved and width10_non_degraded and valid_all and endpoint_same and jerk_same and outputs_same and state_same)
    return {
        "scope": scope,
        "cases": len(rows),
        "current_median_G_rcsp_width10": current10,
        "current_median_G_rcsp_width28": current28,
        "current_gap_width28_minus_width10": current_gap,
        "current_temporal_pass_count_rcsp": current_pass,
        "current_temporal_pass_count_base": current_base_pass,
        "bctr_median_G_rcsp_width10": bctr10,
        "bctr_median_G_rcsp_width28": bctr28,
        "bctr_gap_width28_minus_width10": bctr_gap,
        "bctr_gap_shrink_fraction": gap_shrink,
        "bctr_temporal_pass_count_rcsp": bctr_pass,
        "bctr_temporal_pass_count_base": bctr_base_pass,
        "width28_newly_rescued_cases": newly_rescued,
        "width10_lost_cases": width10_lost,
        "delta_width28_gain": delta28,
        "delta_width10_gain": delta10,
        "endpoint_semantics_unchanged": endpoint_same,
        "jerk_semantics_unchanged": jerk_same,
        "outputs_unchanged": outputs_same,
        "state_unchanged": state_same,
        "bctr_valid_case_count": valid_count,
        "bctr_valid_all_cases": valid_all,
        "gap_reduction_at_fixed_fraction": gap_reduced,
        "width28_median_gain_improved": width28_improved,
        "width10_median_gain_non_degraded": width10_non_degraded,
        "width10_degradation": bool(
            (delta10 is not None and delta10 < 0.0) or bool(width10_lost)
        ),
        "split_supported": supported,
    }


def make_summaries(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["role"] == PRIMARY_ROLE]
    if len(primary) != PRIMARY_CASES:
        raise ValueError("scientific summaries require exactly 32 cross-event cases")
    return {scope: _scope_summary(_scope_rows(primary, scope), scope) for scope in ("overall", "seen", "new")}


def adjudicate(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered BCTR decision tree without tuning."""
    seen = summaries["seen"]
    new = summaries["new"]
    overall = summaries["overall"]
    both_supported = bool(seen["split_supported"] and new["split_supported"])
    one_split_supported = bool(seen["split_supported"] != new["split_supported"])
    overall_supported = bool(overall["split_supported"])
    width10_degradation = bool(seen["width10_degradation"] or new["width10_degradation"])
    controls_ok = bool(
        seen["endpoint_semantics_unchanged"] and new["endpoint_semantics_unchanged"]
        and seen["jerk_semantics_unchanged"] and new["jerk_semantics_unchanged"]
        and seen["outputs_unchanged"] and new["outputs_unchanged"]
        and seen["state_unchanged"] and new["state_unchanged"]
    )
    if both_supported:
        result = "METRIC_SUPPORT_TIME_INTERVENTION_SUPPORTED"
        next_action = "freeze_candidate_and_design_separate_direction_intervention"
    elif (overall_supported or one_split_supported) and not width10_degradation and controls_ok:
        result = "PARTIAL_METRIC_SUPPORT_TIME_INTERVENTION"
        next_action = "retain_partial_evidence_and_proceed_to_width_conditioned_direction_intervention"
    else:
        result = "METRIC_SUPPORT_TIME_INTERVENTION_NOT_SUPPORTED"
        next_action = "reject_bctr_candidate_and_proceed_to_width_conditioned_direction_intervention"
    return {
        "result": result,
        "split_supported": {"seen": bool(seen["split_supported"]), "new": bool(new["split_supported"])},
        "overall_supported": overall_supported,
        "width10_degradation_observed": width10_degradation,
        "endpoint_semantics_unchanged": controls_ok,
        "next_action": next_action,
        "major_gap_fraction": MAJOR_GAP_FRACTION,
        "no_further_metric_search": True,
        "causal_root_cause_proven": False,
        "scientific_acceptance": False,
        "pilot_allowed": False,
    }


def _immutable_paths(paths: Mapping[str, Path], phase21_path: Path) -> dict[str, Path]:
    return {
        "phase21/report.json": phase21_path,
        "phase2/report.json": paths["phase2_report_path"],
        **{f"source/{name}": paths["source"] / name for name in (
            "diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt"
        )},
        **{f"trajectory/{name}": paths["trajectory"] / name for name in (
            "report.json", "experiment.json", "diagnostic_latest.pt", "updates.jsonl"
        )},
        "rcsp/report.json": paths["rcsp_directory"] / "report.json",
        "rcsp/reporting_logic_review_v1.json": paths["rcsp_directory"] / "reporting_logic_review_v1.json",
        "rcsp/adapter_checkpoint": paths["adapter_checkpoint"],
        "phase1/report.json": paths["phase1_report"],
        "single_decomposition/report.json": paths["single_decomposition_report"],
        "parameter_attribution/report.json": paths["parameter_attribution_report"],
    }


def _run(args: argparse.Namespace) -> int:
    phase21_path = Path(args.phase21_report).resolve()
    phase21_report, phase21_hash, lineage_paths, upstream = _validate_phase21_lineage(phase21_path)
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    if args.expected_main_commit != runtime_commit:
        raise ValueError("expected main commit is not the runtime checkout")
    output = Path(args.output_dir).resolve()
    result_dir = output / "result"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("BCTR output directory must be a fresh empty directory")
    for input_path in (lineage_paths["source"], lineage_paths["trajectory"], phase21_path.parent):
        if output == input_path or output.is_relative_to(input_path):
            raise FileExistsError("BCTR output overlaps immutable lineage input")
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(exist_ok=False)
    failure_path = result_dir / "failure.json"
    immutable = _immutable_paths(lineage_paths, phase21_path)
    before_files = {name: _file_sha256(path) for name, path in immutable.items()}
    implementation_paths = {
        "motion_models.py": Path(m.__file__).resolve(),
        "boundary_observables.py": Path(boundary_observables.__file__).resolve(),
        "product_manifold.py": Path(product_manifold.__file__).resolve(),
        "rcsp.py": Path(rcsp.__file__).resolve(),
    }
    implementation_before = {name: _file_sha256(path) for name, path in implementation_paths.items()}
    try:
        trajectory, _trajectory_paths, _trajectory_hashes, trajectory_report, experiment, checkpoint = failure._load_trajectory(
            lineage_paths["trajectory"], failure.TRAJECTORY_COMMIT
        )
        state, bank, cfg, source_metadata = group_audit.load_frozen_source(
            lineage_paths["source"], group_audit.LEGACY_COMMIT,
            legacy_core_strength=LEGACY_CORE_STRENGTH,
            legacy_transition_strength=LEGACY_TRANSITION_STRENGTH,
        )
        if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
            raise ValueError("trajectory does not reference the Phase 2.1 frozen source")
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
                raise RuntimeError("loaded base differs from immutable A0 final state")
            model = rcsp.FrozenBaseRCSPModel(base)
            model.adapter.load_state_dict(upstream["rcsp"]["adapter_checkpoint"]["adapter_state_dict"], strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            model.eval()
            adapter_hash = safe.state_hash(model.adapter.state_dict())
            if adapter_hash != upstream["rcsp"]["report"]["parameter_update_scope"]["adapter_state_sha256"]:
                raise RuntimeError("loaded RCSP adapter state hash mismatch")
            probe, probe_hash = safe.load_probe(lineage_paths["source"], state, bank, cfg)
            final_batch, final_metadata = alignment.combine_final_banks(failure.final_banks(bank, probe, cfg))
            final_batch = rcsp._move_batch(final_batch, device)
            phase2._validate_fixed_metadata(final_metadata)
            all_rows: list[dict[str, Any]] = []
            parity_rows: list[dict[str, Any]] = []
            for start in range(0, FINAL_CASES, rcsp.FINAL_CHUNK_SIZE):
                stop = start + rcsp.FINAL_CHUNK_SIZE
                chunk = {key: value[start:stop] for key, value in final_batch.items()}
                metadata = final_metadata[start:stop]
                rows, parity = _evaluate_chunk(base, model, chunk, metadata, phase21_report, upstream, cfg)
                all_rows.extend(rows)
                parity_rows.extend(parity["rows"])
            if len(all_rows) != FINAL_CASES:
                raise RuntimeError("BCTR fixed-final evaluation did not contain exactly 64 cases")
            primary_rows = [row for row in all_rows if row["role"] == PRIMARY_ROLE]
            excluded_rows = [row for row in all_rows if row["role"] == EXCLUDED_ROLE]
            if len(primary_rows) != PRIMARY_CASES or len(excluded_rows) != PRIMARY_CASES:
                raise RuntimeError("BCTR primary/excluded cohort count mismatch")
            if len(parity_rows) != PRIMARY_CASES or not all(row["verified"] for row in parity_rows):
                raise RuntimeError("BCTR current metric parity failed closed")
            integrity_before = safe.state_hash(base.state_dict()), safe.state_hash(model.adapter.state_dict())
            del probe
            if _file_sha256(lineage_paths["source"] / "probe_bank.pt") != probe_hash:
                raise RuntimeError("probe artifact changed during BCTR evaluation")
            integrity = phase21._read_only_integrity(
                before_files, immutable, integrity_before[0], integrity_before[1], base, model
            )
        implementation_after = {name: _file_sha256(path) for name, path in implementation_paths.items()}
        implementation_unchanged = implementation_before == implementation_after
        if not implementation_unchanged:
            raise RuntimeError("production implementation files changed during BCTR audit")
        for row in all_rows:
            row["anti_gaming"]["state_unchanged"] = bool(
                integrity["base_unchanged"] and integrity["adapter_unchanged"] and integrity["immutable_artifacts_unchanged"]
            )
        summaries = make_summaries(all_rows)
        decision = adjudicate(summaries)
        model_output_unchanged = all(row["anti_gaming"]["outputs_unchanged"] for row in all_rows)
        integrity.update({
            "implementation_files_unchanged": implementation_unchanged,
            "model_output_unchanged": model_output_unchanged,
            "production_model_modified": False,
            "production_inference_modified": False,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
        })
        report = {
            "schema": SCHEMA,
            "completed": True,
            "provenance": {
                "runtime_commit": runtime_commit,
                "expected_main_commit": args.expected_main_commit,
                "phase21_runtime_commit": FROZEN_PHASE21_COMMIT,
                "phase21_report": str(phase21_path),
                "phase21_report_sha256": phase21_hash,
                "phase2_report": str(lineage_paths["phase2_report_path"]),
                "source": str(lineage_paths["source"]),
                "trajectory": str(trajectory),
                "rcsp_directory": str(lineage_paths["rcsp_directory"]),
                "adapter_checkpoint": str(lineage_paths["adapter_checkpoint"]),
                "phase1_report": str(lineage_paths["phase1_report"]),
                "single_decomposition_report": str(lineage_paths["single_decomposition_report"]),
                "parameter_attribution_report": str(lineage_paths["parameter_attribution_report"]),
                "phase21_immutable_input_sha256": before_files,
                "immutable_input_sha256": before_files,
                "implementation_sha256_before": implementation_before,
                "implementation_sha256_after": implementation_after,
            },
            "lineage": {
                "phase21_report_path": str(phase21_path),
                "phase21_report_sha256": phase21_hash,
                "phase21_schema": phase21.SCHEMA,
                "phase21_completed": True,
                "phase21_primary_cases": PRIMARY_CASES,
                "adapter_checkpoint_path_read_from_phase21_lineage": str(lineage_paths["adapter_checkpoint"]),
                "no_latest_artifact_search": True,
            },
            "hypothesis": {
                "name": "boundary-crossing temporal reduction",
                "acronym": "BCTR",
                "question": "Does retaining only derivative stencils crossing from seam core into non-core reduce the width-conditioned support-time gate gap?",
                "frozen_phase21_conclusion": "Metric/support-time is tested first; width-conditioned direction is not intervened on here.",
            },
            "intervention": {
                "name": "Boundary-Crossing Temporal Reduction",
                "acronym": "BCTR",
                "changed_variable": "temporal metric derivative-stencil support",
                "model_changed": False,
                "direction_changed": False,
                "decoder_changed": False,
                "gate_threshold_changed": False,
            },
            "primary_cohort": {
                "cases": PRIMARY_CASES,
                "groups": {group: CASES_PER_GROUP for group in GROUP_ORDER},
                "role": PRIMARY_ROLE,
                "widths": list(WIDTHS),
                "observed_group_comparison": "UNPAIRED",
                "single_recording_excluded": True,
            },
            "excluded_cohorts": {
                EXCLUDED_ROLE: {
                    "cases": len(excluded_rows),
                    "excluded_from_primary_analysis": True,
                    "excluded_from_width_summary": True,
                }
            },
            "parity": {
                "current_metric_parity_verified": True,
                "current_metric_parity_cases": len(parity_rows),
                "current_metric_parity": parity_rows,
                "model_output_unchanged": model_output_unchanged,
                "final_applied_tangent_unchanged": model_output_unchanged,
                "decoded_observable_motion_unchanged": model_output_unchanged,
                "contact_outputs_unchanged": True,
                "raw_base_output_unchanged": True,
                "raw_adapted_output_unchanged": True,
                "parity_atol": PARITY_ATOL,
                "parity_rtol": PARITY_RTOL,
            },
            "metric_definition": {
                "core": "seam >= 0.5",
                "crossing_support": "A_k(t) = touches_core(t) AND touches_outside(t) for the k+1-frame derivative stencil",
                "acceleration": "mean over BCTR acceleration support of ||diff(J,2)*fps^2||_2 over joints, divided by 10",
                "jerk": "mean over BCTR jerk support of ||diff(J,3)*fps^3||_2 over joints, divided by 1000",
                "temporal_energy": "BCTR_acceleration / 10 + BCTR_jerk / 1000",
                "zero_support": "invalid/null; no clamped denominator is used for BCTR",
                "derivative_precision": "float64 FK joints and production diff(..., n=k) * fps**k",
                "prohibited_logic": ["width-specific support", "±5/±14 band", "top-k", "percentile", "clipping", "learned weighting"],
            },
            "gate_definition": {
                "source": "motion_geometry.boundary_observables.observable_gate",
                "floor": TEMPORAL_FLOOR,
                "relative_gain": "(before-after)/before if before>1e-6 else 1 if after<=1e-6 else -1",
                "threshold_source": "cfg.checkpoint_validation_min_temporal_repair_gain",
                "endpoint_gate_unchanged": True,
                "original_full_support_jerk_non_regression": True,
                "candidate_overall": "original endpoint acceptance AND candidate temporal acceptance",
            },
            "case_level": primary_rows,
            "excluded_case_level": excluded_rows,
            "summaries": summaries,
            "decision": decision,
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
            "new_same_boundary_counterfactual_performed": False,
            "no_further_metric_search": True,
            "next_action": decision["next_action"],
        }
        _exclusive_json(result_dir / "report.json", report)
        print(json.dumps({
            "stage": "refiner_boundary_crossing_temporal_reduction_intervention_complete",
            "report": str(result_dir / "report.json"),
            "primary_cases": PRIMARY_CASES,
            "current_metric_parity_verified": True,
            "model_output_unchanged": model_output_unchanged,
            "decision": decision["result"],
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
                "production_inference_modified": False,
                "scientific_acceptance": False,
                "publish_allowed": False,
                "pilot_allowed": False,
            })
        raise


def run(args: argparse.Namespace) -> int:
    """Public entry point retained for audit callers and server wrappers."""
    return _run(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase21-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--device", default="cuda")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
