"""g1f4 progress policies and the frozen anchor-kinematic metric."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
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
METRIC_SHELL_DTYPE = "float64"
_REQUIRED_GROUP_WEIGHT_KEYS = (
    "lambda_boundary", "lambda_jerk", "lambda_science",
)


class MetricCalibrationRequired(RuntimeError):
    pass


class MetricKernelUnavailable(RuntimeError):
    pass


def is_post_selector_projected_calibration_candidate(
    *, selected_method: str, selected_audit: Mapping[str, Any] | None,
    adapter_incumbent_locked: bool,
) -> bool:
    """Return whether a final composite decision is eligible for M alpha."""
    if selected_method in {"identity", "adapter"} or adapter_incumbent_locked:
        return False
    if not isinstance(selected_audit, Mapping):
        return False
    return bool(
        selected_audit.get("raw_audit", {}).get("passed") is True
        and selected_audit.get("projector_result") is not None
        and selected_audit.get("effective_projected_candidate") is True
    )


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


def _strict_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetricCalibrationRequired(f"{description} must be a JSON object")
    return value


def _strict_finite_positive(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricCalibrationRequired(
            f"{description} must be a finite positive number"
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise MetricCalibrationRequired(
            f"{description} must be a finite positive number"
        )
    return result


def _strict_group_weights(value: Any, description: str) -> dict[str, float]:
    mapping = _strict_mapping(value, description)
    if set(mapping) != set(_REQUIRED_GROUP_WEIGHT_KEYS):
        raise MetricCalibrationRequired(
            f"{description} keys must be exactly "
            f"{sorted(_REQUIRED_GROUP_WEIGHT_KEYS)}"
        )
    result = {}
    for name in _REQUIRED_GROUP_WEIGHT_KEYS:
        try:
            weight = float(mapping[name])
        except (TypeError, ValueError) as exc:
            raise MetricCalibrationRequired(
                f"{description}.{name} must be finite and nonnegative"
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise MetricCalibrationRequired(
                f"{description}.{name} must be finite and nonnegative"
            )
        result[name] = weight
    return result


def _exact_float_match(left: Any, right: Any, description: str) -> None:
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError) as exc:
        raise MetricCalibrationRequired(f"invalid {description}") from exc
    if not (math.isfinite(left_value) and math.isfinite(right_value)):
        raise MetricCalibrationRequired(f"nonfinite {description}")
    if left_value != right_value:
        raise MetricCalibrationRequired(f"{description} mismatch")


def _validate_preregistration(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != PREREGISTRATION_SCHEMA:
        raise MetricCalibrationRequired("unexpected M preregistration schema")
    if payload.get("status") != "protocol_frozen_not_numerically_calibrated":
        raise MetricCalibrationRequired("M preregistration status is not frozen")
    definition = _strict_mapping(
        payload.get("metric_definition"), "M preregistration metric_definition"
    )
    beta = _strict_finite_positive(
        definition.get("beta_identity_regularizer"),
        "M preregistration beta_identity_regularizer",
    )
    weights = _strict_group_weights(
        definition.get("group_weights_v1"),
        "M preregistration group_weights_v1",
    )
    train_calibration = _strict_mapping(
        payload.get("train_only_calibration"),
        "M preregistration train_only_calibration",
    )
    rho_e = _strict_finite_positive(
        train_calibration.get("rho_E"),
        "M preregistration train_only_calibration.rho_E",
    )
    if rho_e != 1.0e-4:
        raise MetricCalibrationRequired("M preregistration rho_E must remain 1e-4")
    fairness = _strict_mapping(
        payload.get("fairness_contract"), "M preregistration fairness_contract"
    )
    tolerance = _strict_finite_positive(
        fairness.get("metric_radius_tolerance"),
        "M preregistration fairness_contract.metric_radius_tolerance",
    )
    if tolerance != 1.0e-12:
        raise MetricCalibrationRequired(
            "M preregistration metric_radius_tolerance must remain 1e-12"
        )
    return {
        "beta_identity_regularizer": beta,
        "group_weights_v1": weights,
        "rho_E": rho_e,
        "metric_radius_tolerance": tolerance,
    }


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
        # The shell is deliberately float64 even when a model/FK boundary
        # consumes float32.  Do not reintroduce dtype-dependent geometry here.
        return self.rows.to(device=value.device, dtype=torch.float64)

    def _shell_value(self, value):
        mask = self.mask.to(device=value.device)
        return value.to(dtype=torch.float64).masked_fill(~mask, 0.0)

    def apply(self, value):
        scoped = self._shell_value(value)
        mask = self.mask.to(device=scoped.device)
        rows = self._rows_like(scoped)
        if rows.numel():
            dots = (rows * scoped.unsqueeze(0)).reshape(rows.shape[0], -1).sum(1)
            low_rank = (rows * dots.reshape((-1,) + (1,) * scoped.ndim)).sum(0)
        else:
            low_rank = torch.zeros_like(scoped)
        return (
            self.trace_scale * (self.beta * scoped + low_rank)
        ).masked_fill(~mask, 0.0)

    def inner(self, left, right):
        scoped_left = self._shell_value(left)
        return (scoped_left * self.apply(right)).sum()

    def solve(self, covector):
        scoped = self._shell_value(covector)
        mask = self.mask.to(device=scoped.device)
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
        return (result.reshape_as(scoped) / self.trace_scale).masked_fill(~mask, 0.0)

    def norm(self, value):
        return self.inner(value, value).clamp_min(0.0).sqrt()

    def task_norm(self, value):
        """Return the frozen normalized task-Jacobian norm ``||Jd||_W``."""
        scoped = self._shell_value(value)
        rows = self._rows_like(scoped)
        if not rows.numel():
            return scoped.new_zeros(())
        dots = (rows * scoped.unsqueeze(0)).reshape(rows.shape[0], -1).sum(1)
        return dots.square().sum().clamp_min(0.0).sqrt()

    def rms(self, value) -> float:
        return float((self.norm(value) / math.sqrt(self.active_count)).detach()) if self.active_count else math.nan

    def normalize(self, value, target_rms: float):
        scoped = self._shell_value(value)
        mask = self.mask.to(device=scoped.device)
        norm = self.norm(scoped)
        if not bool(torch.isfinite(norm)) or float(norm.detach()) <= 1.0e-12:
            return scoped, False
        result = scoped * (float(target_rms) * math.sqrt(self.active_count) / norm.detach())
        return result.masked_fill(~mask, 0.0), True

    def tangent_project(self, vector, current):
        scoped, radial = self._shell_value(vector), self._shell_value(current)
        mask = self.mask.to(device=scoped.device)
        denominator = self.inner(radial, radial).clamp_min(1.0e-30)
        return (
            scoped - self.inner(scoped, radial) / denominator * radial
        ).masked_fill(~mask, 0.0)

    def unit_tangent(self, vector, current, floor):
        projected = self.tangent_project(vector, current)
        norm = self.norm(projected)
        if not bool(torch.isfinite(norm)) or float(norm.detach()) <= float(floor):
            return None
        return projected / norm

    def differentiable_geodesic(self, current, unit_direction, theta):
        radial, unit = self._shell_value(current), self._shell_value(unit_direction)
        mask = self.mask.to(device=radial.device)
        angle = theta.to(dtype=torch.float64) if torch.is_tensor(theta) else radial.new_tensor(theta)
        return (
            torch.cos(angle) * radial
            + torch.sin(angle) * self.norm(radial) * unit
        ).masked_fill(~mask, 0.0)

    def geodesic_update(self, current, direction, theta, floor, *, direction_is_tangent):
        radial = self._shell_value(current.detach())
        tangent = self._shell_value(direction.detach())
        mask = self.mask.to(device=radial.device)
        radius, tangent_norm = self.norm(radial), self.norm(tangent)
        if (not bool(torch.isfinite(radius) and torch.isfinite(tangent_norm))
                or float(radius) <= float(floor) or float(tangent_norm) <= float(floor)):
            return current.detach(), False, {
                "geodesic_update_status": (
                    "zero_or_nonfinite_metric_geodesic_direction"
                ),
                "metric_shell_dtype": METRIC_SHELL_DTYPE,
            }
        radial_inner = self.inner(tangent, radial)
        tolerance = max(1.0e-12, float(radius) * float(tangent_norm) * 1.0e-6)
        if direction_is_tangent and abs(float(radial_inner.detach())) > tolerance:
            return current.detach(), False, {
                "geodesic_update_status": "physical_direction_not_metric_tangent",
                "metric_geodesic_radial_inner_product": float(radial_inner.detach()),
                "metric_geodesic_radial_tolerance": tolerance,
                "metric_shell_dtype": METRIC_SHELL_DTYPE,
            }
        unit = self.unit_tangent(tangent, radial, floor)
        if unit is None:
            return current.detach(), False, {
                "geodesic_update_status": (
                    "zero_or_nonfinite_metric_geodesic_direction"
                ),
                "metric_shell_dtype": METRIC_SHELL_DTYPE,
            }
        angle = radius.new_tensor(float(theta))
        trial = (
            torch.cos(angle) * radial + torch.sin(angle) * radius * unit
        ).masked_fill(~mask, 0.0)
        return trial.detach(), bool(torch.isfinite(trial).all()), {
            "geodesic_update_status": "exact_anchor_metric_radius_geodesic_update",
            "theta_radians": float(theta),
            "metric_radius_before": self.rms(radial),
            "metric_radius_after": self.rms(trial),
            "euclidean_radius_after": float(torch.sqrt(trial[mask].square().mean()).detach()),
            "post_update_normalization_applied": False,
            "outside_scope_abs_max": float(
                trial.masked_fill(mask, 0.0).abs().amax().detach()
            ),
            "metric_shell_dtype": METRIC_SHELL_DTYPE,
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
            "metric_shell_dtype": METRIC_SHELL_DTYPE,
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
            if base in row_scales:
                scale = float(row_scales[base])
            elif name in row_scales:
                scale = float(row_scales[name])
            else:
                raise MetricKernelUnavailable(
                    f"missing frozen metric row scale for {name} ({base})"
                )
            if not math.isfinite(scale) or scale <= 0.0:
                raise MetricKernelUnavailable(f"invalid frozen metric row scale for {name}")
            normalized = physical / scale
            if not bool(torch.isfinite(normalized).all()):
                raise MetricKernelUnavailable(f"nonfinite anchor metric row {name}")
            if float(torch.linalg.vector_norm(normalized[active]).detach()) > float(floor):
                groups[group].append((str(name), normalized, scale))
        packed, names, labels, scales = [], [], [], []
        weights = _strict_group_weights(
            (self.calibration or {}).get("group_weights_v1"),
            "anchor metric group_weights_v1",
        )
        for group in ("boundary", "jerk", "science"):
            entries = groups[group]
            if not entries: continue
            coefficient = math.sqrt(float(weights[f"lambda_{group}"]) / len(entries))
            for name, row, scale in entries:
                packed.append(row * coefficient); names.append(name); labels.append(group); scales.append(scale)
        rows = torch.stack(packed) if packed else torch.empty(
            (0,) + tuple(current.shape), dtype=torch.float64, device=current.device)
        beta = _strict_finite_positive(
            (self.calibration or {}).get("beta_identity_regularizer"),
            "anchor metric beta_identity_regularizer",
        )
        count = int(active.sum().detach())
        trace_raw = beta * count + float(rows.square().sum().detach())
        if count <= 0 or not math.isfinite(trace_raw) or trace_raw <= 0.0:
            raise MetricKernelUnavailable("invalid anchor metric trace")
        anchor_sha = hashlib.sha256(current.detach().to(torch.float64).cpu().numpy().tobytes()).hexdigest()
        self.kernel = AnchorMetricKernel(
            self.case_uid, active, rows, beta, count / trace_raw, self.target_rms,
            tuple(names), tuple(labels), tuple(scales), anchor_sha)
        self.kernels[self.case_uid] = self.kernel

    def observe_calibration_candidate(
        self, *, selected_method, value, mask, projector_backtracking_factor,
        projected_candidate,
    ):
        if not self.collecting_calibration:
            return
        if not self.case_uid:
            raise MetricCalibrationRequired(
                "M calibration observation requires a case UID"
            )
        if self.case_uid in self.calibration_observations:
            return
        kernel = self.kernels.get(self.case_uid)
        if kernel is None:
            raise MetricCalibrationRequired(
                "M calibration observation has no frozen metric anchor"
            )
        if selected_method in {"identity", "adapter"}:
            raise MetricCalibrationRequired(
                "identity or Adapter incumbent cannot calibrate anchor metric"
            )
        if projected_candidate is not True:
            raise MetricCalibrationRequired(
                "M calibration requires an accepted projected candidate"
            )
        if projector_backtracking_factor is None:
            raise MetricCalibrationRequired(
                "M calibration requires the accepted Projector factor"
            )
        projector_backtracking_factor = _strict_finite_positive(
            projector_backtracking_factor,
            "M calibration accepted Projector factor",
        )
        if projector_backtracking_factor > 1.0:
            raise MetricCalibrationRequired(
                "M calibration accepted Projector factor exceeds one"
            )
        if value.dtype != torch.float64:
            raise MetricCalibrationRequired(
                "M calibration projected tangent must use float64 shell dtype"
            )
        if not bool(torch.equal(mask.to(torch.bool), kernel.mask.to(torch.bool))):
            raise MetricCalibrationRequired("M calibration projected tangent mask mismatch")
        euclidean = float(
            torch.sqrt(value[kernel.mask].square().mean()).detach()
        )
        metric = kernel.rms(value)
        task_norm = float(kernel.task_norm(value).detach())
        if not (
            math.isfinite(euclidean) and euclidean > 0.0
            and math.isfinite(metric) and metric > 0.0
            and math.isfinite(task_norm) and task_norm >= 0.0
        ):
            raise MetricCalibrationRequired(
                "M calibration selected projected tangent has invalid norms"
            )
        self.calibration_observations[self.case_uid] = {
            "case_uid": self.case_uid,
            "calibration_case_uid": self.case_uid,
            "selected_method": str(selected_method),
            "calibration_source_stage": "post_composite_selector_post_projector",
            "projected_candidate": True,
            "adapter_incumbent_locked": False,
            "projector_backtracking_factor": projector_backtracking_factor,
            "euclidean_rms": euclidean, "metric_rms_at_rho_E": metric,
            "metric_to_euclidean_ratio": metric / euclidean,
            "task_space_norm_Jd_W": task_norm,
            "anchor_metric": kernel.audit(),
        }

    def finalize_calibration(self, *, implementation_commit):
        rows = [self.calibration_observations[k] for k in sorted(self.calibration_observations)]
        if not self.collecting_calibration or not rows:
            raise MetricCalibrationRequired("no successful train correction calibrated M")
        implementation_commit = str(implementation_commit or "").strip().lower()
        if (
            len(implementation_commit) != 40
            or any(character not in "0123456789abcdef" for character in implementation_commit)
        ):
            raise MetricCalibrationRequired(
                "M calibration requires the full implementation commit SHA"
            )
        if (
            not self.preregistration_sha256
            or len(self.preregistration_sha256) != 64
        ):
            raise MetricCalibrationRequired(
                "M calibration requires the bound preregistration SHA256"
            )
        alpha = float(statistics.median(float(r["metric_to_euclidean_ratio"]) for r in rows))
        payload = {
            "schema": CALIBRATION_SCHEMA, "status": "train_only_numerically_calibrated",
            "implementation_commit": implementation_commit,
            "coordinate": "owned_physical_tangent",
            "anchor_definition": "immutable_adapter_physical_tangent_before_first_repair_step",
            "anchor_frozen_across_budgets_and_steps": True,
            "metric_formula": "M=(n_active/trace(M_raw))*(beta*I+sum_group lambda_group*mean((grad(row)/scale) outer (grad(row)/scale)))",
            "beta_identity_regularizer": _strict_finite_positive(
                (self.calibration or {}).get("beta_identity_regularizer"),
                "calibration beta_identity_regularizer",
            ),
            "group_weights_v1": _strict_group_weights(
                (self.calibration or {}).get("group_weights_v1"),
                "calibration group_weights_v1",
            ),
            "trace_normalization": "exact_low_rank_trace_on_owned_active_coordinates",
            "alpha_estimator": (
                "deterministic_median_metric_to_euclidean_rms_ratio_over_"
                "final_composite_selected_projector_accepted_"
                "authoritative_guard_successful_nonidentity_train_"
                "correction_per_case"
            ),
            "rho_E": _strict_finite_positive(
                (self.calibration or {}).get("rho_E"), "calibration rho_E"
            ),
            "alpha": alpha,
            "rho_G": alpha * _strict_finite_positive(
                (self.calibration or {}).get("rho_E"), "calibration rho_E"
            ),
            "calibration_split": "train", "development_or_held_out_consumed": False,
            "calibration_source": (
                "final_composite_selected_projected_train_corrections"
            ),
            "selection_stage": "post_composite_selector",
            "projection_stage": "accepted_projected_candidate",
            "adapter_incumbent_cases_excluded": True,
            "identity_cases_excluded": True,
            "raw_preselector_variants_used": False,
            "projected_tangent_used": True,
            "metric_shell_dtype": METRIC_SHELL_DTYPE,
            "observation_count": len(rows), "observations": rows,
            "preregistered_contract_path": self.preregistration_path,
            "preregistered_contract_sha256": self.preregistration_sha256,
            "ambient_metric_materialized": False, "inverse_method": "exact_low_rank_woodbury",
            "metric_radius_tolerance": _strict_finite_positive(
                (self.calibration or {}).get("metric_radius_tolerance"),
                "calibration metric_radius_tolerance",
            ),
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
            "metric_shell_dtype": METRIC_SHELL_DTYPE,
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


def build_metric_operator(
    mode, *, calibration_path=None, calibration_sha256=None,
    calibration_output=None, preregistration_path=None,
):
    if mode not in METRIC_MODES: raise ValueError(f"unsupported metric mode: {mode}")
    prereg_sha, prereg_payload = None, None
    if preregistration_path:
        prereg = Path(preregistration_path)
        if not prereg.is_file(): raise MetricCalibrationRequired(f"M preregistration does not exist: {prereg}")
        prereg_payload = _read_json(prereg)
        prereg_config = _validate_preregistration(prereg_payload)
        prereg_sha = _sha256(prereg)
    else:
        prereg_config = None
    if calibration_output:
        if mode != IDENTITY_METRIC or not preregistration_path:
            raise MetricCalibrationRequired("calibration collection requires identity mode and an M preregistration")
        collector_config = {
            **prereg_config,
        }
        return MetricOperator(mode=mode, calibration=collector_config,
            preregistration_path=str(Path(preregistration_path).resolve()),
            preregistration_sha256=prereg_sha,
            calibration_output=str(Path(calibration_output).resolve()))
    if mode == IDENTITY_METRIC:
        if calibration_path or calibration_sha256:
            raise ValueError("identity metric must not receive a radius calibration")
        if preregistration_path:
            raise ValueError(
                "identity metric preregistration is valid only for calibration collection"
            )
        return MetricOperator(mode=mode)
    if not calibration_path or not calibration_sha256 or not preregistration_path:
        raise MetricCalibrationRequired(
            "anchor_kinematic requires calibration path, expected SHA256, and M preregistration"
        )
    path = Path(calibration_path)
    if not path.is_file(): raise MetricCalibrationRequired(f"anchor metric calibration does not exist: {path}")
    payload = _read_json(path)
    actual_calibration_sha = _sha256(path)
    if actual_calibration_sha != str(calibration_sha256).strip().lower():
        raise MetricCalibrationRequired("anchor metric calibration SHA256 mismatch")
    if payload.get("schema") != CALIBRATION_SCHEMA or payload.get("status") != "train_only_numerically_calibrated":
        raise MetricCalibrationRequired("anchor metric contract is not a completed train-only calibration")
    if payload.get("preregistered_contract_sha256") != prereg_sha:
        raise MetricCalibrationRequired("anchor calibration preregistration SHA256 mismatch")
    expected_commit = str(os.environ.get("EXPECTED_COMMIT") or "").strip().lower()
    if (
        len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
        or payload.get("implementation_commit") != expected_commit
    ):
        raise MetricCalibrationRequired("anchor calibration implementation commit mismatch")
    if payload.get("coordinate") != "owned_physical_tangent":
        raise MetricCalibrationRequired("anchor calibration coordinate mismatch")
    _exact_float_match(payload.get("rho_E"), prereg_config["rho_E"], "anchor calibration rho_E")
    _exact_float_match(payload.get("rho_E"), 1.0e-4, "anchor calibration rho_E")
    _exact_float_match(
        payload.get("beta_identity_regularizer"),
        prereg_config["beta_identity_regularizer"],
        "anchor calibration beta_identity_regularizer",
    )
    weights = _strict_group_weights(
        payload.get("group_weights_v1"), "anchor calibration group_weights_v1"
    )
    if weights != prereg_config["group_weights_v1"]:
        raise MetricCalibrationRequired("anchor calibration group_weights_v1 mismatch")
    _exact_float_match(
        payload.get("metric_radius_tolerance"),
        prereg_config["metric_radius_tolerance"],
        "anchor calibration metric_radius_tolerance",
    )
    if payload.get("calibration_split") != "train":
        raise MetricCalibrationRequired("anchor metric calibration split is not train")
    if payload.get("development_or_held_out_consumed") is not False:
        raise MetricCalibrationRequired("anchor metric calibration consumed non-train evidence")
    for name in ("alpha", "rho_G"):
        _strict_finite_positive(payload.get(name), f"anchor metric calibration {name}")
    rho_e = _strict_finite_positive(payload.get("rho_E"), "anchor metric calibration rho_E")
    expected_rho_g = float(payload["alpha"]) * rho_e
    if not math.isclose(
        float(payload["rho_G"]), expected_rho_g,
        rel_tol=1.0e-12, abs_tol=1.0e-18,
    ):
        raise MetricCalibrationRequired("anchor calibration rho_G != alpha * rho_E")
    required_top_level = {
        "calibration_source": "final_composite_selected_projected_train_corrections",
        "selection_stage": "post_composite_selector",
        "projection_stage": "accepted_projected_candidate",
        "adapter_incumbent_cases_excluded": True,
        "identity_cases_excluded": True,
        "raw_preselector_variants_used": False,
        "projected_tangent_used": True,
        "metric_shell_dtype": METRIC_SHELL_DTYPE,
    }
    for name, expected in required_top_level.items():
        if payload.get(name) != expected:
            raise MetricCalibrationRequired(f"anchor calibration {name} mismatch")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise MetricCalibrationRequired("anchor calibration observations are missing")
    if int(payload.get("observation_count", -1)) != len(observations):
        raise MetricCalibrationRequired("anchor calibration observation_count mismatch")
    observed_case_uids: set[str] = set()
    observed_ratios: list[float] = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise MetricCalibrationRequired("anchor calibration observation is invalid")
        if (
            observation.get("calibration_source_stage")
            != "post_composite_selector_post_projector"
            or observation.get("projected_candidate") is not True
            or observation.get("selected_method") in {None, "identity", "adapter"}
            or observation.get("adapter_incumbent_locked") is not False
            or observation.get("projector_backtracking_factor") is None
        ):
            raise MetricCalibrationRequired(
                "anchor calibration observation is not a selected projected correction"
            )
        case_uid = observation.get("case_uid")
        if (
            not isinstance(case_uid, str)
            or not case_uid
            or observation.get("calibration_case_uid") != case_uid
            or case_uid in observed_case_uids
        ):
            raise MetricCalibrationRequired(
                "anchor calibration observation case identity is invalid"
            )
        observed_case_uids.add(case_uid)
        projector_factor = _strict_finite_positive(
            observation.get("projector_backtracking_factor"),
            "anchor calibration observation projector_backtracking_factor",
        )
        if projector_factor > 1.0:
            raise MetricCalibrationRequired(
                "anchor calibration Projector factor exceeds one"
            )
        euclidean_rms = _strict_finite_positive(
            observation.get("euclidean_rms"),
            "anchor calibration observation euclidean_rms",
        )
        metric_rms = _strict_finite_positive(
            observation.get("metric_rms_at_rho_E"),
            "anchor calibration observation metric_rms_at_rho_E",
        )
        observed_ratio = _strict_finite_positive(
            observation.get("metric_to_euclidean_ratio"),
            "anchor calibration observation metric_to_euclidean_ratio",
        )
        recomputed_ratio = metric_rms / euclidean_rms
        if not math.isclose(
            observed_ratio, recomputed_ratio,
            rel_tol=1.0e-12, abs_tol=1.0e-18,
        ):
            raise MetricCalibrationRequired(
                "anchor calibration observation ratio is inconsistent"
            )
        try:
            task_norm = float(observation.get("task_space_norm_Jd_W"))
        except (TypeError, ValueError) as exc:
            raise MetricCalibrationRequired(
                "anchor calibration observation task norm is invalid"
            ) from exc
        if not math.isfinite(task_norm) or task_norm < 0.0:
            raise MetricCalibrationRequired(
                "anchor calibration observation task norm is invalid"
            )
        anchor_metric = _strict_mapping(
            observation.get("anchor_metric"),
            "anchor calibration observation anchor_metric",
        )
        if (
            anchor_metric.get("case_uid") != case_uid
            or anchor_metric.get("metric_shell_dtype") != METRIC_SHELL_DTYPE
        ):
            raise MetricCalibrationRequired(
                "anchor calibration observation metric anchor mismatch"
            )
        observed_ratios.append(observed_ratio)
    recomputed_alpha = float(statistics.median(observed_ratios))
    if not math.isclose(
        float(payload["alpha"]), recomputed_alpha,
        rel_tol=1.0e-12, abs_tol=1.0e-18,
    ):
        raise MetricCalibrationRequired(
            "anchor calibration alpha is not the deterministic observation median"
        )
    return MetricOperator(mode=mode, calibration_path=str(path.resolve()),
        calibration_sha256=actual_calibration_sha, calibration=payload,
        preregistration_path=str(Path(preregistration_path).resolve()),
        preregistration_sha256=prereg_sha)


def mode_grid() -> tuple[Mapping[str, str], ...]:
    return tuple({"progress_mode": p, "metric_mode": m} for p in PROGRESS_MODES for m in METRIC_MODES)
