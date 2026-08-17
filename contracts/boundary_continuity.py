"""Strict per-seam boundary-continuity acceptance for final motion output."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite_metric(values: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


@dataclass(frozen=True)
class BoundaryContinuityLimits:
    """C0/C1/C2-inspired hard limits evaluated independently at every seam."""

    boundary_joint_jerk_max_mps3: float = 650.0
    entry_fk_jump_max_m: float = 0.015
    exit_fk_jump_max_m: float = 0.015
    entry_rotation_step_max_rad: float = 0.08
    exit_rotation_step_max_rad: float = 0.08
    foot_slip_max_mps: float = 0.06
    foot_penetration_max_m2: float = 0.001

    @classmethod
    def from_environment(cls) -> "BoundaryContinuityLimits":
        return cls(
            boundary_joint_jerk_max_mps3=_env_float(
                "V46_46_MAX_BOUNDARY_JERK_MPS3", 650.0
            ),
            entry_fk_jump_max_m=_env_float(
                "V46_46_MAX_ENTRY_FK_JUMP_M",
                _env_float("V46_46_MAX_EXIT_FK_JUMP_M", 0.015),
            ),
            exit_fk_jump_max_m=_env_float("V46_46_MAX_EXIT_FK_JUMP_M", 0.015),
            entry_rotation_step_max_rad=_env_float(
                "V46_46_MAX_ENTRY_ROT_RAD",
                _env_float("V46_46_MAX_EXIT_ROT_RAD", 0.08),
            ),
            exit_rotation_step_max_rad=_env_float(
                "V46_46_MAX_EXIT_ROT_RAD", 0.08
            ),
            foot_slip_max_mps=_env_float("V46_46_MAX_FOOT_SLIP_MPS", 0.06),
            foot_penetration_max_m2=_env_float(
                "V46_46_MAX_FOOT_PENETRATION_M2", 0.001
            ),
        )


def boundary_risk_reasons(
    risk: Mapping[str, Any],
    *,
    limits: Optional[BoundaryContinuityLimits] = None,
) -> list[str]:
    """Return failures for one seam; missing/non-finite metrics always fail."""

    lim = limits or BoundaryContinuityLimits.from_environment()
    checks = (
        (
            "boundary_joint_jerk_max",
            lim.boundary_joint_jerk_max_mps3,
            "boundary_joint_jerk_max_mps3",
        ),
        ("entry_fk_jump", lim.entry_fk_jump_max_m, "entry_fk_jump_m"),
        ("exit_fk_jump", lim.exit_fk_jump_max_m, "exit_fk_jump_m"),
        (
            "entry_rotation_step_rad",
            lim.entry_rotation_step_max_rad,
            "entry_rotation_step_rad",
        ),
        (
            "exit_rotation_step_rad",
            lim.exit_rotation_step_max_rad,
            "exit_rotation_step_rad",
        ),
        ("foot_slip", lim.foot_slip_max_mps, "foot_slip_mps"),
        (
            "foot_penetration",
            lim.foot_penetration_max_m2,
            "foot_penetration_m2",
        ),
    )
    reasons: list[str] = []
    for source_key, maximum, report_key in checks:
        value = _finite_metric(risk, source_key)
        if value is None:
            reasons.append(f"missing_or_nonfinite:{report_key}")
        elif value > maximum:
            reasons.append(f"{report_key}_too_high:{value:.9g}>{maximum:.9g}")
    return reasons


def evaluate_boundary_continuity(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_boundaries: Optional[int] = None,
    limits: Optional[BoundaryContinuityLimits] = None,
) -> Dict[str, Any]:
    """Require complete audit coverage and zero unsafe final seams."""

    lim = limits or BoundaryContinuityLimits.from_environment()
    audited_rows = [dict(row) for row in rows]
    reasons: list[str] = []
    boundary_results: list[Dict[str, Any]] = []

    if expected_boundaries is not None and len(audited_rows) != int(expected_boundaries):
        reasons.append(
            "boundary_audit_count_mismatch:"
            f"{len(audited_rows)}!={int(expected_boundaries)}"
        )

    for index, row in enumerate(audited_rows):
        slot = int(row.get("slot", index + 1))
        risk = row.get("risk")
        if not isinstance(risk, Mapping):
            violations = ["missing_or_invalid:risk"]
        else:
            violations = boundary_risk_reasons(risk, limits=lim)
        # Patched runtimes may add anatomy/heading feasibility to their
        # authoritative row-level decision.  Preserve that stricter verdict in
        # addition to recomputing the mandatory C0/C1/C2 metrics here.
        if "safe" in row and not bool(row.get("safe")):
            violations.append("row_marked_unsafe")
        boundary_results.append(
            {
                "slot": slot,
                "ok": not violations,
                "violations": list(violations),
            }
        )
        reasons.extend(f"slot_{slot}:{reason}" for reason in violations)

    return {
        "schema": "strict_boundary_continuity_v1",
        "ok": not reasons,
        "reasons": reasons,
        "expected_boundaries": (
            None if expected_boundaries is None else int(expected_boundaries)
        ),
        "audited_boundaries": int(len(audited_rows)),
        "unsafe_boundaries": int(
            sum(not bool(result["ok"]) for result in boundary_results)
        ),
        "limits": asdict(lim),
        "boundaries": boundary_results,
    }
