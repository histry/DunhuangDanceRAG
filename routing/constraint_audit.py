"""Auditable diagnostics for state-aware probabilistic constraint routing."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def _finite_range(values: Sequence[float]) -> list[Optional[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return [None, None]
    return [float(array.min()), float(array.max())]


def summarize_constraint_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    source_expansion: Optional[Mapping[str, Any]] = None,
    scarcity_context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Summarize exact-safe candidates, state reachability and recovery resources."""
    safe = [row for row in trials if bool(row.get("safe", False))]
    preferred = [row for row in safe if bool(row.get("preferred", False))]
    recovered = [row for row in safe if bool(row.get("recovery_triggered", False))]
    eligible = [row for row in trials if bool(row.get("eligible", False))]
    state_reachable = [
        row
        for row in safe
        if bool(row.get("future_reachability", {}).get("future_reachable", False))
    ]
    soft_reason_counts = Counter(
        str(reason)
        for row in safe
        for reason in row.get("constraint_assessment", {})
        .get("diversity", {})
        .get("soft_reasons", [])
    )
    overrun_reason_counts = Counter(
        str(reason)
        for row in safe
        for reason in row.get("constraint_assessment", {}).get(
            "budget_overrun_reasons", []
        )
    )
    safe_sources = sorted(
        {
            str(row.get("constraint_assessment", {}).get("identity", {}).get("source_uid"))
            for row in safe
        }
    )
    safe_families = sorted(
        {
            str(row.get("constraint_assessment", {}).get("identity", {}).get("family_id"))
            for row in safe
        }
    )
    safe_event_ids = [int(row.get("event_id", -1)) for row in safe]
    observabilities = [float(row.get("observability", np.nan)) for row in safe]
    reachabilities = [
        float(
            row.get("future_reachability", {}).get(
                "future_reachability_probability", np.nan
            )
        )
        for row in safe
    ]
    successor_counts = [
        int(row.get("future_reachability", {}).get("future_safe_successor_count", 0))
        for row in safe
    ]
    recovery_charges = [
        float(row.get("constraint_assessment", {}).get("recovery_charge", np.nan))
        for row in safe
    ]
    first_dead_end_counts = Counter(
        str(row.get("future_reachability", {}).get("future_first_dead_end_slot"))
        for row in safe
        if row.get("future_reachability", {}).get("future_first_dead_end_slot")
        is not None
    )
    scarcity = dict(scarcity_context or {})
    expansion = dict(source_expansion or {})
    return {
        "schema": "state_aware_constraint_collapse_diagnostics",
        "proposals": int(len(trials)),
        "physically_safe": int(len(safe)),
        "state_future_reachable": int(len(state_reachable)),
        "preference_budget_valid": int(len(preferred)),
        "controlled_recovery": int(len(recovered)),
        "eligible": int(len(eligible)),
        "constraint_collapse_detected": bool(len(safe) > 0 and len(eligible) == 0),
        "safe_diversity_reason_counts": dict(soft_reason_counts),
        "safe_budget_overrun_reason_counts": dict(overrun_reason_counts),
        "safe_candidate_event_ids": safe_event_ids,
        "safe_candidate_source_uids": safe_sources,
        "safe_candidate_family_ids": safe_families,
        "safe_source_count": int(len(safe_sources)),
        "safe_observability_range": _finite_range(observabilities),
        "safe_future_reachability_range": _finite_range(reachabilities),
        "safe_future_successor_count_range": (
            [int(min(successor_counts)), int(max(successor_counts))]
            if successor_counts
            else [None, None]
        ),
        "safe_recovery_charge_range": _finite_range(recovery_charges),
        "future_first_dead_end_slot_counts": dict(first_dead_end_counts),
        "source_scarcity": scarcity,
        "safe_source_expansion": expansion,
    }


def controlled_recovery_metadata(
    assessment: Mapping[str, Any],
    *,
    triggered: bool,
    recovery_count_after: int,
    recovery_budget_used_before: float = 0.0,
    recovery_budget_used_after: float = 0.0,
    recovery_budget_total: Optional[float] = None,
) -> dict[str, Any]:
    usage = assessment.get("constraint_usage_after", {})
    budgets = assessment.get("effective_budget", assessment.get("budget", {}))
    ratios = []
    for name, value in usage.items():
        budget = float(budgets.get(name, 0.0))
        ratios.append(float(value) / max(1.0, budget))
    source_scarcity = dict(assessment.get("source_scarcity", {}))
    return {
        "schema": "continuous_controlled_safe_set_recovery",
        "triggered": bool(triggered),
        "physical_constraints_relaxed": False,
        "anatomy_constraints_relaxed": False,
        "severe_heading_constraints_relaxed": False,
        "soft_constraint_budget_used": float(max(ratios, default=0.0)),
        "budget_overrun": dict(assessment.get("budget_overrun", {})),
        "relaxed_preference_constraints": list(
            assessment.get("budget_overrun_reasons", [])
        ),
        "future_reachability_probability": float(
            assessment.get("future_reachability_probability", 0.0)
        ),
        "recovery_charge": float(assessment.get("recovery_charge", 0.0)),
        "recovery_budget_used_before": float(recovery_budget_used_before),
        "recovery_budget_used_after": float(recovery_budget_used_after),
        "recovery_budget_total": (
            None if recovery_budget_total is None else float(recovery_budget_total)
        ),
        "recovery_count_after": int(recovery_count_after),
        "source_scarcity_exemption": bool(
            source_scarcity.get("source_scarcity_exemption", False)
        ),
        "alternative_safe_source_exists": bool(
            source_scarcity.get("alternative_safe_source_exists", True)
        ),
        "source_penalty_scale": float(
            source_scarcity.get("source_penalty_scale", 1.0)
        ),
    }
