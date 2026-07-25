#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime integration for semantic-OT retrieval and pairwise repair policy.

The module is deliberately installed after the repository's existing research
patch stack.  It does not duplicate the closed-loop implementation.  Instead it
wraps two stable extension points:

1. whole-song candidate pre-ordering, where class/source/family quotas reduce
   semantic collapse while preserving the original score order;
2. candidate transition simulation, where event-local intrinsic risk is treated
   only as a prior and the measured event-to-event physical risk remains
   authoritative.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from events.semantic_descriptor import MUSIC_SEMANTIC_LABELS, probs_to_vector
from routing.diversity_constrained_retrieval import (
    select_diverse_candidates,
    selection_audit,
)
from routing.transition_repair_policy import (
    TransitionRiskPolicy,
    transition_decision,
)

_INSTALLED = False


def _env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _db_array(db: Mapping[str, Any], keys: Sequence[str], n: int, default: Any) -> np.ndarray:
    for key in keys:
        if key in db:
            value = np.asarray(db[key], dtype=object)
            if value.ndim == 1 and len(value) == n:
                return value
    return np.asarray([default] * n, dtype=object)


def _slot_probabilities(slot: Mapping[str, Any]) -> np.ndarray:
    raw = slot.get(
        "music_semantic_probs",
        slot.get("slot_music_semantic_probs", slot.get("probabilities", None)),
    )
    top = slot.get(
        "music_semantic_top_label",
        slot.get("slot_music_semantic_top_label", slot.get("music_alignment_label", None)),
    )
    return probs_to_vector(raw, top_label=top, temperature=1.0)


def _diversify_ranked_list(
    slot: Mapping[str, Any],
    ranked_event_ids: Sequence[int],
    db: Mapping[str, Any],
) -> tuple[list[int], Dict[str, Any]]:
    ranked = [int(value) for value in ranked_event_ids]
    if not ranked:
        return [], {"selected": 0, "relaxed": False}
    total_events = len(np.asarray(db.get("paths", [])))
    valid = [event_id for event_id in ranked if 0 <= event_id < total_events]
    if not valid:
        return ranked, {"selected": 0, "relaxed": True, "reason": "no_valid_event_id"}
    sources_all = _db_array(db, ("source_uids", "source_groups"), total_events, "unknown")
    families_all = _db_array(db, ("event_families",), total_events, "unknown")
    semantics_all = _db_array(
        db,
        ("aesd_event_semantics", "music_alignment_labels"),
        total_events,
        "lyrical_flow",
    )
    source = sources_all[valid]
    family = families_all[valid]
    semantic = semantics_all[valid]
    # Strictly monotone scores preserve the incoming rank as the primary signal.
    scores = -np.arange(len(valid), dtype=np.float64)
    top_k = min(
        len(valid),
        max(1, _env_int("SEMANTIC_OT_DIVERSITY_TOP_K", len(valid))),
    )
    chosen_local = select_diverse_candidates(
        scores,
        source,
        family,
        semantic,
        _slot_probabilities(slot),
        top_k=top_k,
        source_cap=max(1, _env_int("SEMANTIC_OT_SOURCE_CAP", 3)),
        family_cap=max(1, _env_int("SEMANTIC_OT_FAMILY_CAP", 6)),
        minimum_class_cap=max(1, _env_int("SEMANTIC_OT_MIN_CLASS_CAP", 2)),
    )
    selected = [valid[int(index)] for index in chosen_local]
    selected_set = set(selected)
    reordered = selected + [event_id for event_id in ranked if event_id not in selected_set]
    audit = selection_audit(chosen_local, source, family, semantic)
    audit.update(
        {
            "schema": "dunhuang_semantic_ot_candidate_diversity",
            "input_candidates": int(len(ranked)),
            "quota_prefix": int(len(selected)),
            "music_probabilities": {
                label: float(value)
                for label, value in zip(MUSIC_SEMANTIC_LABELS, _slot_probabilities(slot))
            },
        }
    )
    return reordered, audit


def _current_intrinsic_prior(db: Optional[Mapping[str, Any]], event_id: int) -> Optional[float]:
    if db is None:
        return None
    for key in ("aesd_intrinsic_transition_prior", "aesd_boundary_risk"):
        try:
            value = float(np.asarray(db[key], dtype=np.float32)[int(event_id)])
            return float(np.clip(value, 0.0, 1.0))
        except Exception:
            continue
    return None


def _policy_from_env() -> TransitionRiskPolicy:
    return TransitionRiskPolicy(
        intrinsic_previous_weight=_env_float("SEMANTIC_OT_PREVIOUS_INTRINSIC_W", 0.10),
        intrinsic_following_weight=_env_float("SEMANTIC_OT_FOLLOWING_INTRINSIC_W", 0.10),
        pairwise_weight=_env_float("SEMANTIC_OT_PAIRWISE_RISK_W", 0.80),
        low_threshold=_env_float("SEMANTIC_OT_REPAIR_LOW", 0.35),
        high_threshold=_env_float("SEMANTIC_OT_REPAIR_HIGH", 0.70),
        residual_inpainting_threshold=_env_float(
            "SEMANTIC_OT_REPAIR_RESIDUAL_HIGH", 0.55
        ),
    )


def install(latest: Any) -> None:
    """Install semantic-OT policies into ``routing.global_path`` exactly once."""
    global _INSTALLED
    if _INSTALLED or not _env_bool("SEMANTIC_OT_ENABLE", False):
        return

    original_route = latest._global_route_preorder
    original_proposal = latest.v52.v4650._build_heading_proposal

    def route_with_diversity(
        slots: Sequence[Mapping[str, Any]],
        candidate_lists: Sequence[Sequence[int]],
        db: Mapping[str, Any],
        banned: Optional[Dict[int, set]] = None,
    ):
        audits = []
        reordered = []
        for index, candidates in enumerate(candidate_lists):
            slot = slots[index] if index < len(slots) else {}
            values, audit = _diversify_ranked_list(slot, candidates, db)
            reordered.append(values)
            audits.append({"slot": int(index), **audit})
        result = original_route(slots, reordered, db, banned=banned)
        report = dict(getattr(latest, "_GLOBAL_ROUTE_REPORT", {}) or {})
        report["semantic_ot_candidate_diversity"] = {
            "enabled": True,
            "slots": audits,
        }
        latest._GLOBAL_ROUTE_REPORT = report
        return result

    def proposal_with_pairwise_policy(*args, **kwargs):
        proposal, extra = original_proposal(*args, **kwargs)
        previous = kwargs.get("prev_motion")
        db = kwargs.get("db")
        cfg = kwargs.get("cfg")
        event_id = int(kwargs.get("event_id", proposal.event_id))
        if previous is None or not len(previous) or not len(proposal.core):
            return proposal, extra
        decision = transition_decision(
            np.asarray(previous, dtype=np.float32)[
                -max(8, _env_int("V46_53_TANGENT_WINDOW", 8)) :
            ],
            np.asarray(proposal.core, dtype=np.float32)[
                : max(8, _env_int("V46_53_TANGENT_WINDOW", 8))
            ],
            previous_intrinsic_prior=None,
            following_intrinsic_prior=_current_intrinsic_prior(db, event_id),
            bridge_motion=np.asarray(proposal.bridge, dtype=np.float32),
            fps=float(getattr(cfg, "fps", 30.0)),
            policy=_policy_from_env(),
        )
        hard = bool(decision["hard_reject"] or decision["action"] == "reroute")
        if hard and proposal.safe:
            proposal.risk_score = float(proposal.risk_score + 1.0e6)
            proposal.safe = False
        proposal.risk["semantic_ot_transition_policy"] = decision
        updated_extra = dict(extra)
        updated_extra["semantic_ot_transition_policy"] = decision
        updated_extra.setdefault("heading_detail", {})["hard_reject"] = bool(
            updated_extra.get("heading_detail", {}).get("hard_reject", False) or hard
        )
        return proposal, updated_extra

    latest._global_route_preorder = route_with_diversity
    latest.v52.v4650._build_heading_proposal = proposal_with_pairwise_policy
    _INSTALLED = True
