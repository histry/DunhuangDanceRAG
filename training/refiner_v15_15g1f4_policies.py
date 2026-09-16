"""g1f4 progress policies and the frozen anchor-kinematic metric."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import torch

CURRENT_EQUAL_SHARE = "current_equal_share"
WEIGHTED_DEBT_FILTER = "weighted_debt_filter"
IDENTITY_METRIC = "identity"
ANCHOR_KINEMATIC_METRIC = "anchor_kinematic"
PROGRESS_MODES = (CURRENT_EQUAL_SHARE, WEIGHTED_DEBT_FILTER)
METRIC_MODES = (IDENTITY_METRIC, ANCHOR_KINEMATIC_METRIC)
CALIBRATION_SCHEMA = "refiner_v15_15g1f4_anchor_metric_calibration_v1"
PREREGISTRATION_SCHEMA = "refiner_v15_15g1f4_m_v1_preregistered_contract_v1"


class MetricCalibrationRequired(RuntimeError):
    pass


class MetricKernelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ProgressDecision:
    accepted: bool
    acceptance_mode: str | None
    audit: dict[str, Any]


@dataclass(frozen=True)
class ProgressPolicy:
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in PROGRESS_MODES:
            raise ValueError(f"unsupported progress mode: {self.mode}")

    def decide_intermediate(
        self, *, authoritative_step_closure, quota_progress,
        authoritative_filter_progress,
        both_scientific_terms_strictly_improved,
        new_guard_term_transition, internal_active_witness_transition,
    ) -> ProgressDecision:
        if authoritative_step_closure:
            accepted, acceptance_mode = True, "authoritative_full_closure"
        elif self.mode == CURRENT_EQUAL_SHARE:
            accepted = bool(quota_progress)
            acceptance_mode = "remaining_gap_share" if accepted else None
        else:
            accepted = bool(
                authoritative_filter_progress
                and both_scientific_terms_strictly_improved
                and not new_guard_term_transition
                and not internal_active_witness_transition
            )
            acceptance_mode = "weighted_debt_filter_progress" if accepted else None
        return ProgressDecision(accepted, acceptance_mode, {
            "progress_mode": self.mode,
            "guard_debt_definition": "hard_guard_witness_rows_only",
            "debt_weight_schema": "unit_by_constraint_row_v1",
            "guard_debt_scale_source": "train_frozen_contract",
            "safe_set_rule": "already_safe_rows_must_remain_safe",
            "strict_hard_shadow_decrease_required": True,
            "strict_science_improvement_required": self.mode == WEIGHTED_DEBT_FILTER,
            "new_guard_term_forbidden": self.mode == WEIGHTED_DEBT_FILTER,
            "new_internal_witness_forbidden": self.mode == WEIGHTED_DEBT_FILTER,
        })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise MetricCalibrationRequired("metric contract must be a JSON object")
    return value


def _metric_group(name: str) -> str | None:
    lowered = name.lower()
    if name in {"endpoint", "temporal"}:
        return "science"
    if "boundary" in lowered:
        return "boundary"
    if "jerk" in lowered:
        return "jerk"
    return None


@dataclass
class AnchorMetricKernel:
    """Trace-normalised ``beta I + U.T U`` without an ambient matrix."""
    case_uid: str
    mask: Any
    rows: Any
    beta: float
    trace_scale: float
    rho_g: float | None
    row_names: tuple[str, ...]
    row_groups: tuple[str, ...]
    row_scales: tuple[float, ...]
    anchor_sha256: str

    @property
    def active_count(self) -> int:
        return int(self.mask.sum().detach())

    def _rows_like(self, value):
        return self.rows.to(device=value.device, dtype=value.dtype)

    def apply(self, value):
        scoped = value.masked_fill(~self.mask, 0.0)
        rows = self._rows_like(scoped)
        if rows.numel():
            dots = (rows * scoped.unsqueeze(0)).reshape(rows.shape[0], -1).sum(1)
            low_rank = (rows * dots.reshape((-1,) + (1,) * scoped.ndim)).sum(0)
        else:
            low_rank = torch.zeros_like(scoped)
        return (self.trace_scale * (self.beta * scoped + low_rank)).masked_fill(~self.mask, 0.0)

    def inner(self, left, right):
        return (left.masked_fill(~self.mask, 0.0) * self.apply(right)).sum()

    def solve(self, covector):
        scoped = covector.masked_fill(~self.mask, 0.0)
        rows = self._rows_like(scoped)
        if not rows.numel():
            return scoped / (self.trace_scale * self.beta)
        flat_rows, flat = rows.reshape(rows.shape[0], -1), scoped.reshape(-1)
        gram = flat_rows @ flat_rows.transpose(0, 1)
        system = gram + self.beta * torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        )
        coefficients = torch.linalg.solve(system, flat_rows @ flat)
        result = (flat - flat_rows.transpose(0, 1) @ coefficients) / self.beta
        return (result.reshape_as(scoped) / self.trace_scale).masked_fill(~self.mask, 0.0)

    def norm(self, value):
        return self.inner(value, value).clamp_min(0.0).sqrt()

    def task_norm(self, value):
        """Return the frozen normalized task-Jacobian norm ``||Jd||_W``."""
        scoped = value.masked_fill(~self.mask, 0.0)
        rows = self._rows_like(scoped)
        if not rows.numel():
            return scoped.new_zeros(())
        dots = (rows * scoped.unsqueeze(0)).reshape(rows.shape[0], -1).sum(1)
        return dots.square().sum().clamp_min(0.0).sqrt()

    def rms(self, value) -> float:
        return float((self.norm(value) / math.sqrt(self.active_count)).detach()) if self.active_count else math.nan

    def normalize(self, value, target_rms: float):
        scoped = value.masked_fill(~self.mask, 0.0)
        norm = self.norm(scoped)
        if not bool(torch.isfinite(norm)) or float(norm.detach()) <= 1.0e-12:
            return scoped, False
        result = scoped * (float(target_rms) * math.sqrt(self.active_count) / norm.detach())
        return result.masked_fill(~self.mask, 0.0), True

    def tangent_project(self, vector, current):
        scoped, radial = vector.masked_fill(~self.mask, 0.0), current.masked_fill(~self.mask, 0.0)
        denominator = self.inner(radial, radial).clamp_min(1.0e-30)
        return (scoped - self.inner(scoped, radial) / denominator * radial).masked_fill(~self.mask, 0.0)

    def unit_tangent(self, vector, current, floor):
        projected = self.tangent_project(vector, current)
        norm = self.norm(projected)
        if not bool(torch.isfinite(norm)) or float(norm.detach()) <= float(floor):
            return None
        return projected / norm

    def differentiable_geodesic(self, current, unit_direction, theta):
        radial = current.masked_fill(~self.mask, 0.0)
        unit = unit_direction.masked_fill(~self.mask, 0.0)
        return (torch.cos(theta) * radial + torch.sin(theta) * self.norm(radial) * unit).masked_fill(~self.mask, 0.0)

    def geodesic_update(self, current, direction, theta, floor, *, direction_is_tangent):
        radial = current.detach().masked_fill(~self.mask, 0.0)
        tangent = direction.detach().masked_fill(~self.mask, 0.0)
        radius, tangent_norm = self.norm(radial), self.norm(tangent)
        if (not bool(torch.isfinite(radius) and torch.isfinite(tangent_norm))
                or float(radius) <= float(floor) or float(tangent_norm) <= float(floor)):
            return current.detach(), False, {"geodesic_update_status": "zero_or_nonfinite_metric_geodesic_direction"}
        radial_inner = self.inner(tangent, radial)
        tolerance = max(1.0e-12, float(radius) * float(tangent_norm) * 1.0e-6)
        if direction_is_tangent and abs(float(radial_inner.detach())) > tolerance:
            return current.detach(), False, {
                "geodesic_update_status": "physical_direction_not_metric_tangent",
                "metric_geodesic_radial_inner_product": float(radial_inner.detach()),
                "metric_geodesic_radial_tolerance": tolerance,
            }
        unit = self.unit_tangent(tangent, radial, floor)
        if unit is None:
            return current.detach(), False, {"geodesic_update_status": "zero_or_nonfinite_metric_geodesic_direction"}
        angle = radius.new_tensor(float(theta))
        trial = (torch.cos(angle) * radial + torch.sin(angle) * radius * unit).masked_fill(~self.mask, 0.0)
        return trial.detach(), bool(torch.isfinite(trial).all()), {
            "geodesic_update_status": "exact_anchor_metric_radius_geodesic_update",
            "theta_radians": float(theta),
            "metric_radius_before": self.rms(radial),
            "metric_radius_after": self.rms(trial),
            "euclidean_radius_after": float(torch.sqrt(trial[self.mask].square().mean()).detach()),
            "post_update_normalization_applied": False,
            "outside_scope_abs_max": float(trial.masked_fill(self.mask, 0.0).abs().amax().detach()),
        }

    def audit(self):
        return {
            "case_uid": self.case_uid, "anchor_sha256": self.anchor_sha256,
            "active_coordinate_count": self.active_count,
            "low_rank_row_count": len(self.row_names), "row_names": list(self.row_names),
            "row_groups": list(self.row_groups), "row_scales": list(self.row_scales),
            "beta_identity_regularizer": self.beta,
            "trace_normalization_scale": self.trace_scale,
            "ambient_metric_materialized": False,
            "inverse_method": "exact_low_rank_woodbury",
            "anchor_frozen_across_repair_steps": True,
        }


@dataclass
class MetricOperator:
    mode: str
    calibration_path: str | None = None
    calibration_sha256: str | None = None
    calibration: dict[str, Any] | None = None
    preregistration_path: str | None = None
    preregistration_sha256: str | None = None
    calibration_output: str | None = None
    case_uid: str | None = None
    kernel: AnchorMetricKernel | None = None
    kernels: dict[str, AnchorMetricKernel] = field(default_factory=dict)
    calibration_observations: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def coordinate(self): return "owned_physical_tangent"
    @property
    def collecting_calibration(self): return bool(self.calibration_output)
    @property
    def target_rms(self):
        value = (self.calibration or {}).get("rho_G")
        return float(value) if self.mode != IDENTITY_METRIC and value is not None else None

    def for_case(self, case_uid):
        return MetricOperator(
            self.mode, self.calibration_path, self.calibration_sha256,
            self.calibration, self.preregistration_path,
            self.preregistration_sha256, self.calibration_output, str(case_uid),
            self.kernels.get(str(case_uid)), self.kernels,
            self.calibration_observations,
        )

    def bind_anchor(self, *, current, mask, taper, gradients, row_base_names, row_scales, floor):
        if self.mode == IDENTITY_METRIC and not self.collecting_calibration:
            return
        if not self.case_uid:
            raise MetricKernelUnavailable("anchor metric case UID is not bound")
        if self.case_uid in self.kernels:
            self.kernel = self.kernels[self.case_uid]
            return
        active = mask.to(torch.bool) & (taper.abs() > float(floor))
        groups = {"boundary": [], "jerk": [], "science": []}
        for name, gradient in gradients.items():
            base = str(row_base_names.get(name, name))
            group = _metric_group(name if name in {"endpoint", "temporal"} else base)
            if group is None or gradient is None:
                continue
            physical = torch.zeros_like(gradient, dtype=torch.float64)
            gradient64, taper64 = gradient.detach().to(torch.float64), taper.detach().to(torch.float64)
            physical[active] = gradient64[active] / taper64[active]
            scale = float(row_scales.get(base, row_scales.get(name, 1.0)))
            if not math.isfinite(scale) or scale <= 0.0:
                raise MetricKernelUnavailable(f"invalid frozen metric row scale for {name}")
            normalized = physical / scale
            if not bool(torch.isfinite(normalized).all()):
                raise MetricKernelUnavailable(f"nonfinite anchor metric row {name}")
            if float(torch.linalg.vector_norm(normalized[active]).detach()) > float(floor):
                groups[group].append((str(name), normalized, scale))
        packed, names, labels, scales = [], [], [], []
        weights = ((self.calibration or {}).get("group_weights_v1") or {
            "lambda_boundary": 1.0, "lambda_jerk": 1.0, "lambda_science": 1.0})
        for group in ("boundary", "jerk", "science"):
            entries = groups[group]
            if not entries: continue
            coefficient = math.sqrt(float(weights[f"lambda_{group}"]) / len(entries))
            for name, row, scale in entries:
                packed.append(row * coefficient); names.append(name); labels.append(group); scales.append(scale)
        rows = torch.stack(packed) if packed else torch.empty(
            (0,) + tuple(current.shape), dtype=torch.float64, device=current.device)
        beta = float((self.calibration or {}).get("beta_identity_regularizer", 1.0))
        count = int(active.sum().detach())
        trace_raw = beta * count + float(rows.square().sum().detach())
        if count <= 0 or not math.isfinite(trace_raw) or trace_raw <= 0.0:
            raise MetricKernelUnavailable("invalid anchor metric trace")
        anchor_sha = hashlib.sha256(current.detach().to(torch.float64).cpu().numpy().tobytes()).hexdigest()
        self.kernel = AnchorMetricKernel(
            self.case_uid, active, rows, beta, count / trace_raw, self.target_rms,
            tuple(names), tuple(labels), tuple(scales), anchor_sha)
        self.kernels[self.case_uid] = self.kernel

    def observe_calibration_candidate(self, *, variant, value, mask):
        if not self.collecting_calibration or not self.case_uid or self.case_uid in self.calibration_observations:
            return
        kernel = self.kernels.get(self.case_uid)
        if kernel is None: return
        del mask
        euclidean = float(
            torch.sqrt(value[kernel.mask].square().mean()).detach()
        )
        metric = kernel.rms(value)
        if not (math.isfinite(euclidean) and euclidean > 0 and math.isfinite(metric)): return
        self.calibration_observations[self.case_uid] = {
            "case_uid": self.case_uid, "selected_successful_variant": str(variant),
            "euclidean_rms": euclidean, "metric_rms_at_rho_E": metric,
            "metric_to_euclidean_ratio": metric / euclidean,
            "task_space_norm_Jd_W": float(kernel.task_norm(value).detach()),
            "anchor_metric": kernel.audit(),
        }

    def finalize_calibration(self, *, implementation_commit):
        rows = [self.calibration_observations[k] for k in sorted(self.calibration_observations)]
        if not self.collecting_calibration or not rows:
            raise MetricCalibrationRequired("no successful train correction calibrated M")
        alpha = float(statistics.median(float(r["metric_to_euclidean_ratio"]) for r in rows))
        payload = {
            "schema": CALIBRATION_SCHEMA, "status": "train_only_numerically_calibrated",
            "implementation_commit": implementation_commit,
            "coordinate": "owned_physical_tangent",
            "anchor_definition": "immutable_adapter_physical_tangent_before_first_repair_step",
            "anchor_frozen_across_budgets_and_steps": True,
            "metric_formula": "M=(n_active/trace(M_raw))*(beta*I+sum_group lambda_group*mean((grad(row)/scale) outer (grad(row)/scale)))",
            "beta_identity_regularizer": float(
                (self.calibration or {}).get(
                    "beta_identity_regularizer", 1.0
                )
            ),
            "group_weights_v1": dict(
                (self.calibration or {}).get("group_weights_v1") or {
                    "lambda_boundary": 1.0,
                    "lambda_jerk": 1.0,
                    "lambda_science": 1.0,
                }
            ),
            "trace_normalization": "exact_low_rank_trace_on_owned_active_coordinates",
            "alpha_estimator": "deterministic_median_metric_to_euclidean_rms_ratio_over_first_successful_train_correction_per_case",
            "rho_E": 1.0e-4, "alpha": alpha, "rho_G": alpha * 1.0e-4,
            "calibration_split": "train", "development_or_held_out_consumed": False,
            "observation_count": len(rows), "observations": rows,
            "preregistered_contract_path": self.preregistration_path,
            "preregistered_contract_sha256": self.preregistration_sha256,
            "ambient_metric_materialized": False, "inverse_method": "exact_low_rank_woodbury",
            "metric_radius_tolerance": 1.0e-12,
        }
        destination = Path(self.calibration_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload

    def audit(self):
        return {
            "metric_mode": self.mode, "metric_coordinate": self.coordinate,
            "metric_operator_schema": "g1f4_metric_operator_v2",
            "metric_radius_definition": "euclidean_owned_physical_tangent_rms" if self.mode == IDENTITY_METRIC else "train_calibrated_anchor_kinematic_rms",
            "metric_radius_calibration_required_before_anchor": True,
            "metric_radius_calibration_path": self.calibration_path,
            "metric_radius_calibration_sha256": self.calibration_sha256,
            "metric_target_rms": self.target_rms,
            "anchor_metric_kernel_implemented": True,
            "metric_operator_ready": bool(self.mode == IDENTITY_METRIC or self.calibration),
            "calibration_collection": self.collecting_calibration,
            "anchor_metric": self.kernel.audit() if self.kernel else None,
        }

    def execute(self, stage, identity_kernel, *args, **kwargs):
        if self.mode == IDENTITY_METRIC: return identity_kernel(*args, **kwargs)
        if self.kernel is None: raise MetricKernelUnavailable("anchor metric was not frozen for this case")
        if stage == "normalize_exact_radius": return self.kernel.normalize(args[0], self.target_rms)
        if stage == "radius_rms": return self.kernel.rms(args[0])
        if stage == "exact_radius_geodesic_update":
            return self.kernel.geodesic_update(args[0], args[1], args[3], args[4],
                direction_is_tangent=bool(kwargs.get("direction_is_sphere_tangent", False)))
        if stage == "prepare_second_order_subproblem":
            kwargs["metric_kernel"] = self.kernel
            return identity_kernel(*args, **kwargs)
        raise MetricKernelUnavailable(f"unsupported anchor metric stage {stage}")


def build_metric_operator(mode, *, calibration_path=None, calibration_output=None, preregistration_path=None):
    if mode not in METRIC_MODES: raise ValueError(f"unsupported metric mode: {mode}")
    prereg_sha, prereg_payload = None, None
    if preregistration_path:
        prereg = Path(preregistration_path)
        if not prereg.is_file(): raise MetricCalibrationRequired(f"M preregistration does not exist: {prereg}")
        prereg_payload = _read_json(prereg)
        if prereg_payload.get("schema") != PREREGISTRATION_SCHEMA:
            raise MetricCalibrationRequired("unexpected M preregistration schema")
        if prereg_payload.get("status") != "protocol_frozen_not_numerically_calibrated":
            raise MetricCalibrationRequired("M preregistration status is not frozen")
        prereg_sha = _sha256(prereg)
    if calibration_output:
        if mode != IDENTITY_METRIC or not preregistration_path:
            raise MetricCalibrationRequired("calibration collection requires identity mode and an M preregistration")
        definition = prereg_payload.get("metric_definition") or {}
        collector_config = {
            "beta_identity_regularizer": float(
                definition.get("beta_identity_regularizer", 1.0)
            ),
            "group_weights_v1": dict(
                definition.get("group_weights_v1") or {
                    "lambda_boundary": 1.0,
                    "lambda_jerk": 1.0,
                    "lambda_science": 1.0,
                }
            ),
        }
        return MetricOperator(mode=mode, calibration=collector_config,
            preregistration_path=str(Path(preregistration_path).resolve()),
            preregistration_sha256=prereg_sha,
            calibration_output=str(Path(calibration_output).resolve()))
    if mode == IDENTITY_METRIC:
        if calibration_path: raise ValueError("identity metric must not receive a radius calibration")
        return MetricOperator(mode=mode)
    if not calibration_path: raise MetricCalibrationRequired("anchor_kinematic requires a train-only equivalent-radius calibration")
    path = Path(calibration_path)
    if not path.is_file(): raise MetricCalibrationRequired(f"anchor metric calibration does not exist: {path}")
    payload = _read_json(path)
    if payload.get("schema") != CALIBRATION_SCHEMA or payload.get("status") != "train_only_numerically_calibrated":
        raise MetricCalibrationRequired("anchor metric contract is not a completed train-only calibration")
    for name in ("alpha", "rho_G"):
        value = payload.get(name)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise MetricCalibrationRequired(f"anchor metric calibration has invalid {name}")
    if payload.get("development_or_held_out_consumed") is not False:
        raise MetricCalibrationRequired("anchor metric calibration consumed non-train evidence")
    return MetricOperator(mode=mode, calibration_path=str(path.resolve()),
        calibration_sha256=_sha256(path), calibration=payload)


def mode_grid() -> tuple[Mapping[str, str], ...]:
    return tuple({"progress_mode": p, "metric_mode": m} for p in PROGRESS_MODES for m in METRIC_MODES)
