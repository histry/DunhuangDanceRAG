#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feasibility-aware whole-song Event-RAG contract.

This module is a runtime patch for the current EDGE whole-song stack.  It does
not replace the mature SO(3), anatomy, heading, Grounder, masked-inpainting, or
dynamic-duration implementations.  It fixes the contract mismatch between:

1. semantic retrieval, which exposes only a small top-k candidate set;
2. the generator, which later applies stricter anatomy/heading/warp/tangent
   hard gates and can therefore exhaust every candidate for a slot.

The research policy is intentionally asymmetric:

* source safety, event anatomy validity, heading validity, performer-group
  policy, and severe physical failures are never relaxed;
* only duration/warp and the V46.53 multiscale tangent gate receive bounded,
  reason-specific rescue tiers;
* a music slot is split, while preserving its total frame budget, when no
  duration-feasible event exists in the expanded candidate pool;
* legacy scheduler event IDs/indices are treated as provenance only.  A stable
  event UID is authoritative only after the Scheduler and Generation DB
  fingerprints have been validated by the base loader.

The implementation is installed after ``v46_53_heading_closed_loop`` has
installed its own patches.  It is API-compatible with the current public main
commit and with the cleaned DunhuangDanceRAG module layout.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

_INSTALLED = False
_ACTIVE_TIER = 0
_RESOLVED_PERFORMER = "mixed"
_LENGTH_CACHE: Dict[str, int] = {}
_LAST_DIAGNOSTICS: Dict[str, Any] = {}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


@dataclass(frozen=True)
class FeasibilityPolicy:
    candidate_pool: int = 160
    global_route_topk: int = 64
    strict_warp_min: float = 0.72
    strict_warp_max: float = 1.32
    relaxed_warp_min: float = 0.68
    relaxed_warp_max: float = 1.42
    rescue_warp_min: float = 0.62
    rescue_warp_max: float = 1.48
    min_transition_frames: int = 10
    max_transition_frames: int = 40
    min_core_frames: int = 24
    min_split_frames: int = 36
    max_slot_split_passes: int = 2
    event_quality_min: float = 0.48
    observability_relaxed_min: float = 0.16
    observability_rescue_min: float = 0.12
    tangent_rescue_score_max: float = 1.35
    max_rescue_tier: int = 2

    @classmethod
    def from_env(cls) -> "FeasibilityPolicy":
        return cls(
            candidate_pool=max(32, env_int("RESEARCH_CANDIDATE_POOL", 160)),
            global_route_topk=max(20, env_int("RESEARCH_GLOBAL_ROUTE_TOPK", 64)),
            strict_warp_min=env_float("RESEARCH_STRICT_WARP_MIN", 0.72),
            strict_warp_max=env_float("RESEARCH_STRICT_WARP_MAX", 1.32),
            relaxed_warp_min=env_float("RESEARCH_RELAXED_WARP_MIN", 0.68),
            relaxed_warp_max=env_float("RESEARCH_RELAXED_WARP_MAX", 1.42),
            rescue_warp_min=env_float("RESEARCH_RESCUE_WARP_MIN", 0.62),
            rescue_warp_max=env_float("RESEARCH_RESCUE_WARP_MAX", 1.48),
            min_transition_frames=max(0, env_int("RESEARCH_MIN_TRANSITION_FRAMES", 10)),
            max_transition_frames=max(1, env_int("RESEARCH_MAX_TRANSITION_FRAMES", 40)),
            min_core_frames=max(1, env_int("RESEARCH_MIN_CORE_FRAMES", 24)),
            min_split_frames=max(8, env_int("RESEARCH_MIN_SPLIT_FRAMES", 36)),
            max_slot_split_passes=max(0, env_int("RESEARCH_MAX_SLOT_SPLIT_PASSES", 2)),
            event_quality_min=env_float("RESEARCH_EVENT_QUALITY_MIN", 0.48),
            observability_relaxed_min=env_float("RESEARCH_OBSERVABILITY_RELAXED_MIN", 0.16),
            observability_rescue_min=env_float("RESEARCH_OBSERVABILITY_RESCUE_MIN", 0.12),
            tangent_rescue_score_max=env_float("RESEARCH_TANGENT_RESCUE_SCORE_MAX", 1.35),
            max_rescue_tier=max(0, min(2, env_int("RESEARCH_MAX_RESCUE_TIER", 2))),
        )

    def warp_range(self, tier: int) -> Tuple[float, float]:
        if int(tier) <= 0:
            return self.strict_warp_min, self.strict_warp_max
        if int(tier) == 1:
            return self.relaxed_warp_min, self.relaxed_warp_max
        return self.rescue_warp_min, self.rescue_warp_max


def _db_array(db: Mapping[str, Any], key: str) -> Optional[np.ndarray]:
    try:
        return np.asarray(db[key])
    except Exception:
        return None


def _db_value(db: Mapping[str, Any], key: str, event_id: int, default: Any) -> Any:
    arr = _db_array(db, key)
    if arr is None or not (0 <= int(event_id) < len(arr)):
        return default
    value = arr[int(event_id)]
    return value.item() if isinstance(value, np.generic) else value


def _event_paths(db: Mapping[str, Any]) -> np.ndarray:
    paths = _db_array(db, "paths")
    if paths is None:
        raise KeyError("Event database does not contain 'paths'")
    return np.asarray(paths, dtype=object)


def event_length(path: Any) -> int:
    text = str(path)
    cached = _LENGTH_CACHE.get(text)
    if cached is not None:
        return cached
    p = Path(text).expanduser()
    if not p.is_file():
        _LENGTH_CACHE[text] = 0
        return 0
    try:
        arr = np.load(str(p), mmap_mode="r", allow_pickle=True)
        length = int(arr.shape[-2]) if arr.ndim >= 2 else 0
    except Exception:
        length = 0
    _LENGTH_CACHE[text] = length
    return length


def performer_groups(db: Mapping[str, Any]) -> np.ndarray:
    paths = _event_paths(db)
    for key in ("performer_groups", "genders"):
        arr = _db_array(db, key)
        if arr is not None and len(arr) == len(paths):
            return np.asarray([str(x).strip().lower() for x in arr], dtype=object)
    out: List[str] = []
    sources = _db_array(db, "source_uids")
    for i, path in enumerate(paths):
        text = str(sources[i]) if sources is not None and i < len(sources) else str(path)
        name = text.lower()
        if "female" in name:
            out.append("female")
        elif "male" in name:
            out.append("male")
        else:
            out.append("unknown")
    return np.asarray(out, dtype=object)


def event_quality(db: Mapping[str, Any], event_id: int) -> float:
    for key in (
        "v46_53_combined_quality",
        "event_quality_scores",
        "anatomy_quality",
        "quality_scores",
    ):
        arr = _db_array(db, key)
        if arr is not None and 0 <= int(event_id) < len(arr):
            try:
                return float(arr[int(event_id)])
            except Exception:
                pass
    return 1.0


def static_event_valid(db: Mapping[str, Any], event_id: int, policy: FeasibilityPolicy) -> bool:
    idx = int(event_id)
    paths = _event_paths(db)
    if not (0 <= idx < len(paths)):
        return False
    for key in ("anatomy_valid", "heading_valid"):
        arr = _db_array(db, key)
        if arr is not None and idx < len(arr) and not bool(arr[idx]):
            return False
    if event_quality(db, idx) < policy.event_quality_min:
        return False
    return event_length(paths[idx]) > 1


def slot_target_frames(slot: Mapping[str, Any], fps: float = 30.0) -> int:
    for key in ("target_frames", "music_length"):
        value = slot.get(key)
        if value is not None:
            try:
                return max(1, int(round(float(value))))
            except Exception:
                pass
    start_frame = slot.get("start_frame", slot.get("music_start"))
    end_frame = slot.get("end_frame", slot.get("music_end"))
    if start_frame is not None and end_frame is not None:
        try:
            return max(1, int(round(float(end_frame) - float(start_frame))))
        except Exception:
            pass
    duration = slot.get("duration", slot.get("duration_sec", 1.0))
    return max(1, int(round(float(duration) * float(fps))))


def duration_feasible(
    source_len: int,
    target_len: int,
    has_previous: bool,
    policy: FeasibilityPolicy,
    tier: int = 2,
) -> bool:
    source_len = max(1, int(source_len))
    target_len = max(1, int(target_len))
    warp_min, warp_max = policy.warp_range(tier)
    if not has_previous:
        warp = float(target_len) / float(source_len)
        return warp_min <= warp <= warp_max

    min_trans = min(policy.min_transition_frames, max(0, target_len - policy.min_core_frames))
    max_trans = min(policy.max_transition_frames, max(0, target_len - policy.min_core_frames))
    lower_core = max(
        policy.min_core_frames,
        target_len - max_trans,
        int(math.ceil(source_len * warp_min)),
    )
    upper_core = min(
        target_len - min_trans,
        int(math.floor(source_len * warp_max)),
    )
    return lower_core <= upper_core


_PROVENANCE_ONLY_KEYS = {
    "event_id", "event_index", "family_id", "motion_event", "natural_duration",
    "v26_event_id", "v26_event_index", "v26_family_id",
    "v26_allocated_content_len", "v26_allocated_phrase_total",
    "v26_time_warp_ratio", "resampling", "time_warp_ratio",
}

_STABLE_IDENTITY_KEYS = ("v26_event_uid", "event_uid")


def sanitize_slot(
    slot: Mapping[str, Any],
    fps: float,
    *,
    aligned_event_db: bool = False,
) -> Dict[str, Any]:
    """Remove legacy identity and retain a contract-validated stable UID.

    ``event_id`` and ``event_index`` are run-local historical values and can
    never identify a row in a rebuilt Generation DB.  ``event_uid`` is stable,
    but is accepted only when the caller has already proved that the MSSD and
    Generation DB ordered fingerprints match.
    """
    source = dict(slot)
    provenance = {key: source.get(key) for key in _PROVENANCE_ONLY_KEYS if key in source}
    out = {key: value for key, value in source.items() if key not in _PROVENANCE_ONLY_KEYS}
    stable_uid = next(
        (
            str(source.get(key)).strip()
            for key in _STABLE_IDENTITY_KEYS
            if source.get(key) is not None and str(source.get(key)).strip()
        ),
        "",
    )
    for key in _STABLE_IDENTITY_KEYS:
        out.pop(key, None)
    if stable_uid and aligned_event_db:
        out["v26_event_uid"] = stable_uid
        out["event_uid"] = stable_uid
    elif stable_uid:
        provenance["event_uid"] = stable_uid
    target = slot_target_frames(out, fps=fps)
    out["target_frames"] = int(target)
    out["duration"] = float(target / max(float(fps), 1e-6))
    out["duration_sec"] = float(out["duration"])
    out["scheduler_event_identity_authoritative"] = bool(
        stable_uid and aligned_event_db
    )
    if provenance:
        out["scheduler_event_provenance"] = provenance
    return out


def _split_slot(slot: Mapping[str, Any], feat: np.ndarray, fps: float) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    original = dict(slot)
    total = slot_target_frames(original, fps)
    left = total // 2
    right = total - left
    start_frame = int(round(float(original.get("start_frame", original.get("music_start", 0)))))
    start_sec = float(original.get("start_sec", original.get("start", start_frame / fps)))

    result_slots: List[Dict[str, Any]] = []
    result_feats: List[np.ndarray] = []
    cursor_frame = start_frame
    cursor_sec = start_sec
    for part, frames in enumerate((left, right)):
        item = dict(original)
        item["target_frames"] = int(frames)
        item["duration"] = float(frames / fps)
        item["duration_sec"] = float(frames / fps)
        item["start_frame"] = int(cursor_frame)
        item["end_frame"] = int(cursor_frame + frames)
        item["music_start"] = int(cursor_frame)
        item["music_end"] = int(cursor_frame + frames)
        item["music_length"] = int(frames)
        item["start"] = float(cursor_sec)
        item["end"] = float(cursor_sec + frames / fps)
        item["start_sec"] = float(cursor_sec)
        item["end_sec"] = float(cursor_sec + frames / fps)
        item["research_feasibility_split"] = {
            "enabled": True,
            "part": int(part),
            "parts": 2,
            "original_target_frames": int(total),
            "reason": "no_duration_feasible_candidate_in_expanded_pool",
        }
        f = np.asarray(feat, dtype=np.float32).copy()
        if f.size:
            f.reshape(-1)[0] = float(frames / fps)
        result_slots.append(item)
        result_feats.append(f)
        cursor_frame += int(frames)
        cursor_sec += float(frames / fps)
    return result_slots, result_feats


def _candidate_group_resolution(
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    requested = str(os.environ.get("PERFORMER_GROUP", "auto")).strip().lower()
    if requested not in {"auto", "female", "male", "mixed"}:
        raise ValueError("PERFORMER_GROUP must be auto, female, male, or mixed")
    if requested != "auto":
        return requested, {"requested": requested, "resolved": requested}

    groups = performer_groups(db)
    summary: Dict[str, Dict[str, float]] = {}
    for group in ("female", "male"):
        missing = 0
        best_quality: List[float] = []
        for row in candidate_lists:
            ids = [int(e) for e in row if 0 <= int(e) < len(groups) and groups[int(e)] == group]
            if not ids:
                missing += 1
            else:
                best_quality.append(max(event_quality(db, event_id) for event_id in ids))
        summary[group] = {
            "missing_slots": float(missing),
            "mean_best_quality": float(np.mean(best_quality)) if best_quality else -1.0,
        }
    resolved = min(
        ("female", "male"),
        key=lambda group: (
            summary[group]["missing_slots"],
            -summary[group]["mean_best_quality"],
            group,
        ),
    )
    return resolved, {"requested": "auto", "resolved": resolved, "summary": summary}


def filter_candidate_lists(
    candidate_lists: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    policy: FeasibilityPolicy,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    global _RESOLVED_PERFORMER
    resolved, performer_report = _candidate_group_resolution(candidate_lists, db)
    _RESOLVED_PERFORMER = resolved
    groups = performer_groups(db)
    allow_cross = env_bool("PERFORMER_ALLOW_CROSS_GROUP_RESCUE", False)
    filtered: List[List[int]] = []
    rescue_slots: List[int] = []

    for slot_id, row in enumerate(candidate_lists):
        valid = [int(e) for e in row if static_event_valid(db, int(e), policy)]
        if resolved in {"female", "male"}:
            same = [event_id for event_id in valid if groups[event_id] == resolved]
            if same:
                valid = same
            elif allow_cross:
                rescue_slots.append(slot_id)
            else:
                raise RuntimeError(
                    "No source-safe %s Event candidate for slot %d in the expanded pool"
                    % (resolved, slot_id)
                )
        if not valid:
            raise RuntimeError("No source-safe Event candidate for slot %d" % slot_id)
        filtered.append(valid[: policy.candidate_pool])

    performer_report.update({
        "cross_group_rescue_enabled": bool(allow_cross),
        "cross_group_rescue_slots": rescue_slots,
    })
    return filtered, performer_report


def _duration_bad_slots(
    slots: Sequence[Mapping[str, Any]],
    candidates: Sequence[Sequence[int]],
    db: Mapping[str, Any],
    policy: FeasibilityPolicy,
    fps: float,
) -> List[int]:
    paths = _event_paths(db)
    bad: List[int] = []
    for slot_id, (slot, row) in enumerate(zip(slots, candidates)):
        target = slot_target_frames(slot, fps)
        has_previous = slot_id > 0
        if not any(
            duration_feasible(
                event_length(paths[int(event_id)]),
                target,
                has_previous,
                policy,
                tier=policy.max_rescue_tier,
            )
            for event_id in row
        ):
            bad.append(slot_id)
    return bad


def _split_bad_slots(
    slots: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    bad: Sequence[int],
    policy: FeasibilityPolicy,
    fps: float,
) -> Tuple[List[Dict[str, Any]], np.ndarray, bool]:
    bad_set = set(map(int, bad))
    out_slots: List[Dict[str, Any]] = []
    out_features: List[np.ndarray] = []
    changed = False
    for slot_id, slot in enumerate(slots):
        feature = np.asarray(features[slot_id], dtype=np.float32)
        target = slot_target_frames(slot, fps)
        if slot_id in bad_set and target >= 2 * policy.min_split_frames:
            split_slots, split_features = _split_slot(slot, feature, fps)
            out_slots.extend(split_slots)
            out_features.extend(split_features)
            changed = True
        else:
            out_slots.append(dict(slot))
            out_features.append(feature)
    for index, item in enumerate(out_slots):
        item["slot_id"] = int(index)
        item["slot"] = int(index)
    return out_slots, np.stack(out_features).astype(np.float32), changed


def _tier_env(policy: FeasibilityPolicy, tier: int) -> Dict[str, str]:
    warp_min, warp_max = policy.warp_range(tier)
    observability = 0.22
    if tier == 1:
        observability = policy.observability_relaxed_min
    elif tier >= 2:
        observability = policy.observability_rescue_min
    return {
        "V46_52_CORE_WARP_MIN": str(warp_min),
        "V46_52_CORE_WARP_MAX": str(warp_max),
        "V46_53_OBSERVABILITY_HARD_MIN": str(observability),
        "V46_50_HEADING_TRIAL_TOPK": str(policy.candidate_pool),
    }


@contextlib.contextmanager
def temporary_environment(values: Mapping[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def install(v53: Any) -> Dict[str, Any]:
    """Install the contract after the current V46.53 patch stack is active."""
    global _INSTALLED, _ACTIVE_TIER, _LAST_DIAGNOSTICS
    if _INSTALLED:
        return dict(_LAST_DIAGNOSTICS)

    policy = FeasibilityPolicy.from_env()
    v52 = v53.v52
    v50 = v52.v4650
    base = v52.base

    original_load = base.load_slots_and_candidates
    original_choose = base.choose_transition_lengths
    original_proposal = v50._build_heading_proposal
    original_assemble = v50.assemble_event_heading_reference

    def research_load_slots_and_candidates(v46: Any, args: Any, cfg: Any):
        # The base loader performs the Scheduler/Generation DB fingerprint
        # check and seeds the exact scheduled event_uid.  Never bypass it.
        try:
            cfg.classification_report_topk = max(
                int(getattr(cfg, "classification_report_topk", 8)),
                int(policy.candidate_pool),
            )
        except Exception:
            pass

        (
            db,
            contrastive,
            slots,
            slot_feat,
            path_idx,
            retrieval_report,
            candidate_lists,
        ) = original_load(
            v46,
            args,
            cfg,
        )
        fps = float(getattr(cfg, "fps", 30.0))
        aligned_event_db = env_bool("V46_54_REQUIRE_ALIGNED_EVENT_DB", True)
        slots = [
            sanitize_slot(
                slot,
                fps,
                aligned_event_db=aligned_event_db,
            )
            for slot in slots
        ]
        features = np.asarray(slot_feat, dtype=np.float32)

        passes: List[Dict[str, Any]] = []
        performer_report: Dict[str, Any] = {}
        for split_pass in range(policy.max_slot_split_passes + 1):
            if split_pass > 0:
                path_idx, retrieval_report = v46.retrieve_schedule(
                    slots, features, db, cfg, contrastive
                )
                candidate_lists = base.extract_candidate_lists(
                    path_idx, retrieval_report, db, cfg
                )
            candidate_lists, performer_report = filter_candidate_lists(
                candidate_lists, db, policy
            )
            bad = _duration_bad_slots(
                slots, candidate_lists, db, policy, fps
            )
            passes.append({
                "pass": int(split_pass),
                "slots": int(len(slots)),
                "duration_infeasible_slots": list(map(int, bad)),
            })
            if not bad:
                _LAST_DIAGNOSTICS.update({
                    "candidate_pool": policy.candidate_pool,
                    "performer_policy": performer_report,
                    "slot_split_passes": passes,
                    "final_slots": len(slots),
                })
                return (
                    db,
                    contrastive,
                    list(slots),
                    features,
                    list(map(int, path_idx)),
                    list(retrieval_report),
                    candidate_lists,
                )
            slots, features, changed = _split_bad_slots(
                slots, features, bad, policy, fps
            )
            if not changed:
                raise RuntimeError(
                    "Expanded retrieval pool still has no duration-feasible "
                    "candidate for slots %s; source/event safety was not relaxed"
                    % bad
                )

        raise RuntimeError("Feasibility slot-splitting passes exhausted")

    def research_choose_transition_lengths(
        v46: Any,
        prev: Optional[np.ndarray],
        source_len: int,
        target_len: int,
        raw_curr: np.ndarray,
        slot: Dict[str, Any],
        cfg: Any,
    ):
        core, transition, info = original_choose(
            v46, prev, source_len, target_len, raw_curr, slot, cfg
        )
        source_len_i = max(1, int(source_len))
        target_len_i = max(1, int(target_len))
        if prev is None or len(prev) == 0:
            info = dict(info)
            info["research_feasibility_tier"] = int(_ACTIVE_TIER)
            info["research_first_slot_warp"] = float(target_len_i / source_len_i)
            return target_len_i, 0, info

        warp_min, warp_max = policy.warp_range(_ACTIVE_TIER)
        min_transition = min(
            policy.min_transition_frames,
            max(0, target_len_i - policy.min_core_frames),
        )
        max_transition = min(
            policy.max_transition_frames,
            max(0, target_len_i - policy.min_core_frames),
        )
        lower_core = max(
            policy.min_core_frames,
            target_len_i - max_transition,
            int(math.ceil(source_len_i * warp_min)),
        )
        upper_core = min(
            target_len_i - min_transition,
            int(math.floor(source_len_i * warp_max)),
        )
        info = dict(info)
        info.update({
            "research_feasibility_tier": int(_ACTIVE_TIER),
            "research_core_interval": [int(lower_core), int(upper_core)],
            "research_warp_range": [float(warp_min), float(warp_max)],
        })
        if lower_core <= upper_core:
            preferred = int(np.clip(int(core), lower_core, upper_core))
            preferred = min(
                max(preferred, lower_core),
                upper_core,
            )
            core = preferred
            transition = target_len_i - core
            info.update({
                "core_frames": int(core),
                "transition_frames": int(transition),
                "core_warp": float(core / source_len_i),
                "research_contract_adjusted": True,
            })
        else:
            info["research_contract_adjusted"] = False
            info["research_contract_infeasible"] = True
        return int(core), int(transition), info

    def research_proposal(*args: Any, **kwargs: Any):
        proposal, extra = original_proposal(*args, **kwargs)
        if _ACTIVE_TIER <= 0:
            return proposal, extra

        db = kwargs.get("db")
        event_id = int(kwargs.get("event_id", proposal.event_id))
        gate = (
            proposal.risk.get("v46_52_event_gate", {})
            if isinstance(proposal.risk, dict)
            else {}
        )
        boundary = (
            proposal.risk.get("v46_53_tangent_boundary")
            if isinstance(proposal.risk, dict)
            else None
        )
        grounding = (
            proposal.risk.get("v46_53_grounding", {})
            if isinstance(proposal.risk, dict)
            else {}
        )

        warp_min, warp_max = policy.warp_range(_ACTIVE_TIER)
        warp = float(gate.get("core_warp", 1.0))
        db_valid = bool(gate.get("db_anatomy_valid", True))
        db_quality = float(gate.get("db_anatomy_quality", 1.0))
        runtime_anatomy = gate.get("runtime_core_anatomy", {})
        runtime_valid = bool(runtime_anatomy.get("anatomy_valid", True))
        heading_penalty = float(extra.get("heading_penalty", 0.0))
        heading_db_valid = True
        try:
            heading_db_valid = bool(v50._heading_valid(db, event_id))
        except Exception:
            pass
        physical_ok = bool(base.risk_safe(proposal.risk))
        observability = float(grounding.get("observability", 1.0))
        boundary_hard = bool(
            isinstance(boundary, Mapping) and boundary.get("hard_reject", False)
        )
        boundary_score = float(
            boundary.get("score", 0.0) if isinstance(boundary, Mapping) else 0.0
        )

        immutable_ok = bool(
            db_valid
            and db_quality >= policy.event_quality_min
            and runtime_valid
            and heading_db_valid
            and heading_penalty < 1e5
            and warp_min <= warp <= warp_max
            and physical_ok
        )
        allow = False
        reason = ""
        if immutable_ok and _ACTIVE_TIER == 1:
            allow = bool(
                (not boundary_hard)
                and observability >= policy.observability_relaxed_min
            )
            reason = "bounded_observability_relaxation"
        elif immutable_ok and _ACTIVE_TIER >= 2:
            allow = bool(
                observability >= policy.observability_rescue_min
                and (
                    not boundary_hard
                    or boundary_score <= policy.tangent_rescue_score_max
                )
            )
            reason = "base_physics_safe_tangent_gate_softening"

        if allow and bool(extra.get("heading_detail", {}).get("hard_reject", False)):
            # Immutable gates above prove that the hard flag came only from the
            # bounded V46.53 observability/tangent layer.  V46.52 anatomy and
            # base physical safety remain mandatory.
            proposal.safe = True
            if proposal.risk_score >= 1e6:
                proposal.risk_score = float(proposal.risk_score - 1e6)
            extra = dict(extra)
            extra["heading_detail"] = dict(extra.get("heading_detail", {}))
            extra["heading_detail"]["hard_reject"] = False
            rescue = {
                "tier": int(_ACTIVE_TIER),
                "reason": reason,
                "immutable_source_event_gates_preserved": True,
                "base_physical_safe": bool(physical_ok),
                "observability": float(observability),
                "boundary_hard_before": bool(boundary_hard),
                "boundary_score": float(boundary_score),
                "warp": float(warp),
                "warp_range": [float(warp_min), float(warp_max)],
            }
            proposal.risk["research_feasibility_rescue"] = rescue
            extra["research_feasibility_rescue"] = rescue
        return proposal, extra

    def research_assemble(*args: Any, **kwargs: Any):
        global _ACTIVE_TIER
        failures: List[Dict[str, Any]] = []
        old_tier = _ACTIVE_TIER
        try:
            for tier in range(policy.max_rescue_tier + 1):
                _ACTIVE_TIER = int(tier)
                with temporary_environment(_tier_env(policy, tier)):
                    try:
                        motion, report, selected = original_assemble(*args, **kwargs)
                        for row in report:
                            row["research_feasibility_tier"] = int(tier)
                            row["research_performer_group"] = _RESOLVED_PERFORMER
                        diagnostics = {
                            "selected_tier": int(tier),
                            "tier_failures": failures,
                            "performer_group": _RESOLVED_PERFORMER,
                            **dict(_LAST_DIAGNOSTICS),
                        }
                        _LAST_DIAGNOSTICS.update(diagnostics)
                        try:
                            current = dict(getattr(v53, "_GLOBAL_ROUTE_REPORT", {}) or {})
                            current["research_feasibility"] = diagnostics
                            v53._GLOBAL_ROUTE_REPORT = current
                        except Exception:
                            pass
                        return motion, report, selected
                    except RuntimeError as exc:
                        text = str(exc)
                        if not any(
                            token in text.lower()
                            for token in (
                                "exhausted candidates",
                                "no candidates remain",
                                "no candidates for slot",
                                "has no candidates",
                            )
                        ):
                            raise
                        failures.append({
                            "tier": int(tier),
                            "error": text,
                            "warp_range": list(policy.warp_range(tier)),
                        })
            raise RuntimeError(
                "Feasibility-aware generation exhausted all bounded tiers. "
                "Source safety, event anatomy, heading validity, performer group, "
                "and severe physical gates were intentionally not relaxed. "
                "Diagnostics=" + json.dumps(failures, ensure_ascii=False)
            )
        finally:
            _ACTIVE_TIER = old_tier

    base.load_slots_and_candidates = research_load_slots_and_candidates
    base.choose_transition_lengths = research_choose_transition_lengths
    v50._build_heading_proposal = research_proposal
    v50.assemble_event_heading_reference = research_assemble

    # Candidate previews and expensive heading trials must use the same pool.
    os.environ.setdefault("V46_46_CANDIDATE_TOPK", str(policy.candidate_pool))
    os.environ.setdefault("V46_46_RESELECT_TOPK", str(policy.candidate_pool))
    os.environ.setdefault("V46_50_HEADING_TRIAL_TOPK", str(policy.candidate_pool))
    os.environ.setdefault("V46_53_GLOBAL_ROUTE_TOPK", str(policy.global_route_topk))
    os.environ.setdefault("V46_52_ALLOW_UNSAFE_RESCUE", "0")

    _INSTALLED = True
    _LAST_DIAGNOSTICS = {
        "installed": True,
        "candidate_pool": policy.candidate_pool,
        "global_route_topk": policy.global_route_topk,
        "max_rescue_tier": policy.max_rescue_tier,
    }
    return dict(_LAST_DIAGNOSTICS)
