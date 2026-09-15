"""Shared g1f4 progress-policy and tangent-metric interfaces.

The identity/equal-share combination is deliberately a compatibility path.
Anchor-kinematic metric execution is fail-closed until a train-only equivalent
radius calibration and the metric kernel are both available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CURRENT_EQUAL_SHARE = "current_equal_share"
WEIGHTED_DEBT_FILTER = "weighted_debt_filter"
IDENTITY_METRIC = "identity"
ANCHOR_KINEMATIC_METRIC = "anchor_kinematic"

PROGRESS_MODES = (CURRENT_EQUAL_SHARE, WEIGHTED_DEBT_FILTER)
METRIC_MODES = (IDENTITY_METRIC, ANCHOR_KINEMATIC_METRIC)


class MetricCalibrationRequired(RuntimeError):
    """Raised before an uncalibrated anchor metric can execute."""


class MetricKernelUnavailable(RuntimeError):
    """Raised while the staged anchor-metric kernel is not implemented."""


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
        self,
        *,
        authoritative_step_closure: bool,
        quota_progress: bool,
        authoritative_filter_progress: bool,
        both_scientific_terms_strictly_improved: bool,
        new_guard_term_transition: bool,
        internal_active_witness_transition: bool,
    ) -> ProgressDecision:
        if authoritative_step_closure:
            accepted = True
            acceptance_mode = "authoritative_full_closure"
        elif self.mode == CURRENT_EQUAL_SHARE:
            accepted = bool(quota_progress)
            # Keep the v9 observable spelling byte-for-byte on the parity path.
            acceptance_mode = "remaining_gap_share" if accepted else None
        else:
            accepted = bool(
                authoritative_filter_progress
                and both_scientific_terms_strictly_improved
                and not new_guard_term_transition
                and not internal_active_witness_transition
            )
            acceptance_mode = "weighted_debt_filter_progress" if accepted else None

        return ProgressDecision(
            accepted=accepted,
            acceptance_mode=acceptance_mode,
            audit={
                "progress_mode": self.mode,
                "debt_weighting": "unit_by_constraint_row_v1",
                "safe_set_rule": "already_safe_rows_must_remain_safe",
                "strict_hard_shadow_decrease_required": True,
                "strict_science_improvement_required": bool(
                    self.mode == WEIGHTED_DEBT_FILTER
                ),
                "new_guard_term_forbidden": bool(
                    self.mode == WEIGHTED_DEBT_FILTER
                ),
                "new_internal_witness_forbidden": bool(
                    self.mode == WEIGHTED_DEBT_FILTER
                ),
            },
        )


@dataclass(frozen=True)
class MetricOperator:
    mode: str
    calibration_path: str | None = None
    calibration_sha256: str | None = None

    @property
    def coordinate(self) -> str:
        return "owned_physical_tangent"

    def audit(self) -> dict[str, Any]:
        return {
            "metric_mode": self.mode,
            "metric_coordinate": self.coordinate,
            "metric_operator_schema": "g1f4_metric_operator_v1",
            "metric_radius_definition": (
                "euclidean_owned_physical_tangent_rms"
                if self.mode == IDENTITY_METRIC
                else "train_calibrated_anchor_kinematic_rms"
            ),
            "metric_radius_calibration_required_before_anchor": True,
            "metric_radius_calibration_path": self.calibration_path,
            "metric_radius_calibration_sha256": self.calibration_sha256,
            "anchor_metric_kernel_implemented": False,
            "metric_operator_ready": self.mode == IDENTITY_METRIC,
        }

    def execute(self, stage: str, identity_kernel, *args, **kwargs):
        """Route every geometric operation through the selected operator.

        The callback preserves the canonical v9 implementation exactly for
        identity.  The staged anchor operator cannot silently fall through to
        that callback.
        """
        if self.mode != IDENTITY_METRIC:
            raise MetricKernelUnavailable(
                f"anchor_kinematic kernel is unavailable at metric stage {stage}"
            )
        return identity_kernel(*args, **kwargs)


def _sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metric_operator(
    mode: str,
    *,
    calibration_path: str | None = None,
) -> MetricOperator:
    if mode not in METRIC_MODES:
        raise ValueError(f"unsupported metric mode: {mode}")
    if mode == IDENTITY_METRIC:
        if calibration_path:
            raise ValueError("identity metric must not receive a radius calibration")
        return MetricOperator(mode=mode)

    if not calibration_path:
        raise MetricCalibrationRequired(
            "anchor_kinematic requires a train-only equivalent-radius calibration"
        )
    path = Path(calibration_path)
    if not path.is_file():
        raise MetricCalibrationRequired(
            f"anchor metric calibration does not exist: {path}"
        )

    # The interface is intentionally present before the M kernel.  Refusing to
    # run here prevents an identity approximation from being mislabeled as M.
    operator = MetricOperator(
        mode=mode,
        calibration_path=str(path.resolve()),
        calibration_sha256=_sha256(path),
    )
    raise MetricKernelUnavailable(
        "anchor_kinematic interface is staged, but its kernel is not implemented"
    )


def mode_grid() -> tuple[Mapping[str, str], ...]:
    return tuple(
        {"progress_mode": progress, "metric_mode": metric}
        for progress in PROGRESS_MODES
        for metric in METRIC_MODES
    )
