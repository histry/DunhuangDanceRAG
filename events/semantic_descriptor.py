#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Descriptor Unified Music Semantic Slot Descriptor (MSSD)
=====================================================

This module owns the final CTSR Router, continuous Planner and Duration Model
schedule descriptor. Generation consumes only a final, FPS-bound JSON
descriptor; it does not discover external semantic sidecars.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

MSSD_SCHEMA_VERSION = "semantic_routing_mssd_aesd_routing_descriptor"

# Keep the same label space as Motion Generation music_alignment labels.  `aerial_curve` is
# accepted as a compatibility label because some Chang-E event metadata uses it
# as an intermediate family/semantic route.
MUSIC_SEMANTIC_LABELS: List[str] = [
    "calm_meditative",
    "pose_hold",
    "lyrical_flow",
    "instrument_phrase",
    "percussive_accent",
    "turning_climax",
    "footwork_flow",
    "aerial_curve",
]

ROLE_MAP: Dict[str, str] = {
    "calm_meditative": "calm",
    "pose_hold": "release",
    "lyrical_flow": "normal",
    "instrument_phrase": "normal",
    "percussive_accent": "climax",
    "turning_climax": "build_up",
    "footwork_flow": "normal",
    "aerial_curve": "normal",
}

ENERGY_RHYTHM_MAP: Dict[str, Tuple[str, str]] = {
    "calm_meditative": ("calm", "sustained"),
    "pose_hold": ("calm", "sustained"),
    "lyrical_flow": ("moderate", "lyrical"),
    "instrument_phrase": ("moderate", "accented"),
    "percussive_accent": ("percussive", "percussive"),
    "turning_climax": ("high", "accented"),
    "footwork_flow": ("moderate", "lyrical"),
    "aerial_curve": ("moderate", "lyrical"),
}

def json_load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return json_safe(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def json_save(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2)


def canonical_music_label(label: Any) -> str:
    text = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "calm": "calm_meditative",
        "calm_flow": "calm_meditative",
        "build_up": "turning_climax",
        "release": "lyrical_flow",
        "section_change": "lyrical_flow",
        "neutral_flow": "lyrical_flow",
        "meditation": "calm_meditative",
        "meditative": "calm_meditative",
        "revelation_meditation": "calm_meditative",
        "pose": "pose_hold",
        "hold": "pose_hold",
        "pose_motif": "pose_hold",
        "thirty_six_postures": "pose_hold",
        "36pose": "pose_hold",
        "lyrical": "lyrical_flow",
        "flow": "lyrical_flow",
        "melodic": "lyrical_flow",
        "aerial": "aerial_curve",
        "aerial_curve": "aerial_curve",
        "ribbon": "lyrical_flow",
        "ribbon_flow": "lyrical_flow",
        "instrument": "instrument_phrase",
        "instrument_motif": "instrument_phrase",
        "pipa": "instrument_phrase",
        "pipa_behind_back": "instrument_phrase",
        "percussive": "percussive_accent",
        "accent": "percussive_accent",
        "drum": "percussive_accent",
        "lei_gong_drum": "percussive_accent",
        "climax": "turning_climax",
        "turn": "turning_climax",
        "turning": "turning_climax",
        "turning_flow": "turning_climax",
        "whirl": "turning_climax",
        "footwork": "footwork_flow",
        "steps": "footwork_flow",
        "step": "footwork_flow",
        "lotus_steps": "footwork_flow",
    }
    if text in aliases:
        return aliases[text]
    if text in MUSIC_SEMANTIC_LABELS:
        return text
    # fuzzy fallback
    if "calm" in text or "meditat" in text:
        return "calm_meditative"
    if "pose" in text or "hold" in text or "36" in text:
        return "pose_hold"
    if "pipa" in text or "instrument" in text:
        return "instrument_phrase"
    if "drum" in text or "accent" in text or "percuss" in text:
        return "percussive_accent"
    if "turn" in text or "climax" in text or "whirl" in text:
        return "turning_climax"
    if "foot" in text or "step" in text or "lotus" in text:
        return "footwork_flow"
    if "aerial" in text:
        return "aerial_curve"
    return "lyrical_flow"


def normalize_probs(probs: Any = None, top_label: Any = None, temperature: float = 0.65) -> Dict[str, float]:
    out = {k: 0.0 for k in MUSIC_SEMANTIC_LABELS}
    if isinstance(probs, dict):
        for k, v in probs.items():
            ck = canonical_music_label(k)
            if ck in out:
                try:
                    out[ck] += max(0.0, float(v))
                except Exception:
                    pass
    elif probs is not None:
        try:
            arr = np.asarray(probs, dtype=np.float32).reshape(-1)
            for i, v in enumerate(arr[: len(MUSIC_SEMANTIC_LABELS)]):
                out[MUSIC_SEMANTIC_LABELS[i]] += max(0.0, float(v))
        except Exception:
            pass
    if sum(out.values()) <= 1e-8 and top_label is not None:
        lab = canonical_music_label(top_label)
        out[lab] = 1.0
    if sum(out.values()) <= 1e-8:
        out["lyrical_flow"] = 1.0
    temp = max(0.05, float(temperature))
    vals = np.asarray([out[k] for k in MUSIC_SEMANTIC_LABELS], dtype=np.float32)
    vals = np.power(np.maximum(vals, 0.0) + 1e-8, 1.0 / temp)
    vals = vals / max(float(vals.sum()), 1e-8)
    return {k: float(vals[i]) for i, k in enumerate(MUSIC_SEMANTIC_LABELS)}


def top_label_from_probs(probs: Dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: float(kv[1]))[0] if probs else "lyrical_flow"


def semantic_fields_from_probs(probs: Dict[str, float], source: str = "", role_hint: Any = None) -> Dict[str, Any]:
    top = top_label_from_probs(probs)
    energy, rhythm = ENERGY_RHYTHM_MAP.get(top, ("moderate", "lyrical"))
    role = str(role_hint or ROLE_MAP.get(top, "normal"))
    return {
        "role": role,
        "slot_role": role,
        "music_alignment_label": top,
        "music_semantic_top_label": top,
        "music_semantic_probs": {k: float(v) for k, v in probs.items()},
        "energy_label": energy,
        "rhythm_label": rhythm,
        "preferred_dance_keys": [],
        "slot_preferred_dance_keys": [],
        "preferred_semantic_roles": [],
        "external_music_semantic_source": str(source),
        "dance_theme_used_as_local_action_truth": False,
        "categorical_music_label_used_as_body_semantics": False,
    }


def is_final_schedule_meta(meta: Dict[str, Any]) -> bool:
    usage = str(meta.get("usage", "")).lower()
    if bool(meta.get("is_final_schedule", False)):
        return True
    if usage in {"generate", "generate_schedule", "final_schedule", "router_schedule"}:
        return True
    return False


def _extract_raw_slots(obj: Any) -> Tuple[List[dict], Dict[str, Any]]:
    if isinstance(obj, dict):
        # Native MSSD / Motion Generation slots JSON.
        for key in ["slots", "segments", "descriptors"]:
            if isinstance(obj.get(key), list):
                return list(obj[key]), dict(obj)
        # Raw Whole-Song Planner schedule report.
        if isinstance(obj.get("schedule"), list):
            meta = dict(obj)
            meta.setdefault("usage", "generate_schedule")
            meta.setdefault("is_final_schedule", True)
            meta.setdefault("slot_source", "music_router_whole_song_planner")
            return list(obj["schedule"]), meta
        # Whole-Song Planner summary file: caller normally resolves report, but support simple form.
        results = obj.get("results")
        if isinstance(results, dict) and len(results) == 1:
            val = next(iter(results.values()))
            if isinstance(val, dict) and val.get("report") and Path(str(val["report"])).exists():
                return _extract_raw_slots(json_load(str(val["report"])))
    return [], {}


def slot_duration_frames(slot: dict, fps: float, default_index: int = 0, default_seconds: float = 4.0) -> Tuple[float, float, float, int]:
    # Whole-Song Planner schedule uses frame-level music_start/music_end/music_length and
    # allocated_phrase_total.  Native MSSD may use seconds and/or frame indices.
    target_frames = slot.get("target_frames", slot.get("allocated_phrase_total", slot.get("whole_song_allocated_phrase_total", None)))
    if target_frames is None:
        target_frames = slot.get("music_length", None)
    if target_frames is not None:
        try:
            target_frames = int(round(float(target_frames)))
        except Exception:
            target_frames = None

    start_frame = slot.get("start_frame", None)
    end_frame = slot.get("end_frame", None)
    if start_frame is None and "music_start" in slot and "music_length" in slot:
        start_frame = slot.get("music_start")
    if end_frame is None and start_frame is not None and target_frames is not None:
        end_frame = int(round(float(start_frame))) + int(target_frames)

    st = slot.get("start_sec", slot.get("start", slot.get("t0", None)))
    ed = slot.get("end_sec", slot.get("end", slot.get("t1", None)))
    dur = slot.get("duration_sec", slot.get("duration", None))

    if st is None and start_frame is not None:
        st = float(start_frame) / fps
    if ed is None and end_frame is not None:
        ed = float(end_frame) / fps
    if dur is None and st is not None and ed is not None:
        dur = float(ed) - float(st)
    if dur is None and target_frames is not None:
        dur = float(target_frames) / fps
    if dur is None:
        dur = float(default_seconds)
    dur = max(0.10, float(dur))

    if st is None:
        st = float(default_index) * float(default_seconds)
    st = float(st)
    if ed is None:
        ed = st + dur
    ed = float(ed)
    dur = max(0.10, ed - st)
    if target_frames is None:
        target_frames = max(1, int(round(dur * fps)))
    return st, ed, dur, int(target_frames)


def normalize_slot(slot0: dict, meta: Dict[str, Any], index: int, fps: float, source_path: str, temperature: float = 0.65) -> Tuple[dict, np.ndarray]:
    slot = dict(slot0)
    st, ed, dur, target_frames = slot_duration_frames(slot, fps=fps, default_index=index)
    top = slot.get("music_semantic_top_label", slot.get("music_alignment_label", slot.get("top_label", slot.get("label", slot.get("music_event", slot.get("motion_event", None))))))
    probs_obj = slot.get("music_semantic_probs", slot.get("probs", slot.get("probabilities", slot.get("slot_probs", None))))
    probs = normalize_probs(probs_obj, top, temperature=temperature)
    sem = semantic_fields_from_probs(probs, source=slot.get("external_music_semantic_source", source_path), role_hint=slot.get("slot_role", slot.get("role", None)))
    feature = np.asarray(slot.get("feature", []), dtype=np.float32).reshape(-1)
    if feature.size < 32 or not np.isfinite(feature[:32]).all():
        feature = np.zeros(32, dtype=np.float32)
    if feature.size < 32:
        feature = np.pad(feature, (0, 32 - feature.size))

    out = {
        **slot,
        "slot_id": int(slot.get("slot_id", slot.get("slot", index))),
        "start": float(st),
        "end": float(ed),
        "start_sec": float(st),
        "end_sec": float(ed),
        "duration": float(dur),
        "duration_sec": float(dur),
        "target_frames": int(target_frames),
        "descriptor_type": "music_semantic_slot",
        "descriptor_schema_version": MSSD_SCHEMA_VERSION,
        "usage": str(meta.get("usage", "generate_schedule" if is_final_schedule_meta(meta) else "train_semantic")),
        "is_final_schedule": bool(is_final_schedule_meta(meta)),
        "slot_source": str(slot.get("slot_source", meta.get("slot_source", "external_sidecar"))),
        "slot_plan_source": str(slot.get("slot_plan_source", meta.get("slot_source", "external_sidecar"))),
        "feature": feature[:32].astype(float).tolist(),
        "feature_contract": "unused_zero_placeholder_formal_motion_conditioning_uses_selected_event_descriptor",
        **sem,
    }
    # Preserve Whole-Song Planner raw fields in a predictable namespace for auditing.
    if "event_id" in slot and "whole_song_event_id" not in out:
        out["whole_song_event_id"] = slot.get("event_id")
    if "event_uid" in slot and "whole_song_event_uid" not in out:
        out["whole_song_event_uid"] = slot.get("event_uid")
    if "event_index" in slot and "whole_song_event_index" not in out:
        out["whole_song_event_index"] = slot.get("event_index")
    if "family_id" in slot and "whole_song_family_id" not in out:
        out["whole_song_family_id"] = slot.get("family_id")
    if "allocated_content_len" in slot and "whole_song_allocated_content_len" not in out:
        out["whole_song_allocated_content_len"] = slot.get("allocated_content_len")
    if "allocated_phrase_total" in slot and "whole_song_allocated_phrase_total" not in out:
        out["whole_song_allocated_phrase_total"] = slot.get("allocated_phrase_total")
    if "time_warp_ratio" in slot and "whole_song_time_warp_ratio" not in out:
        out["whole_song_time_warp_ratio"] = slot.get("time_warp_ratio")
    return out, feature[:32].astype(np.float32)


def parse_descriptor_file(path: str | Path, *, require_final_schedule: bool = False, fps: float = 30.0, temperature: float = 0.65, usage: str = "auto") -> Tuple[List[dict], np.ndarray, Dict[str, Any]]:
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() != ".json":
        raise RuntimeError(f"Formal MSSD must be JSON, got: {p}")
    obj = json_load(p)
    raw_slots, meta = _extract_raw_slots(obj)
    if not raw_slots:
        raise RuntimeError(f"MSSD has no slots/segments/schedule: {p}")
    meta = dict(meta)
    meta.setdefault("descriptor_type", "music_semantic_slot_descriptor")
    meta.setdefault("descriptor_schema_version", MSSD_SCHEMA_VERSION)
    meta.setdefault("slot_source", "music_router_whole_song_planner" if is_final_schedule_meta(meta) else "external_sidecar")
    meta.setdefault("usage", "generate_schedule" if is_final_schedule_meta(meta) else "train_semantic")
    meta.setdefault("is_final_schedule", is_final_schedule_meta(meta))
    if usage != "auto":
        meta["usage_request"] = usage
    final = is_final_schedule_meta(meta)
    declared_fps = meta.get("fps")
    if final:
        if declared_fps is None:
            raise RuntimeError(
                f"Final MSSD descriptor has no FPS contract: {p}"
            )
        try:
            declared_fps = float(declared_fps)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Final MSSD descriptor has invalid FPS metadata: {declared_fps!r}"
            ) from exc
        if not np.isfinite(declared_fps) or declared_fps <= 0.0:
            raise RuntimeError(
                f"Final MSSD descriptor has invalid FPS metadata: {declared_fps!r}"
            )
        if abs(declared_fps - fps) > 1.0e-6:
            raise RuntimeError(
                "Final MSSD FPS contract mismatch: "
                f"descriptor={declared_fps}, runtime={fps}, path={p}"
            )
    meta["fps"] = fps if declared_fps is None else float(declared_fps)
    if require_final_schedule and not final:
        raise RuntimeError(
            f"MSSD strict generation requires final schedule, but descriptor is not final: {p}; "
            f"usage={meta.get('usage')} slot_source={meta.get('slot_source')} is_final_schedule={meta.get('is_final_schedule')}"
        )
    slots: List[dict] = []
    feats: List[np.ndarray] = []
    cursor = 0.0
    for i, raw in enumerate(raw_slots):
        if not isinstance(raw, dict):
            continue
        # If no time coordinate exists, use cursor to keep segments ordered.
        if not any(k in raw for k in ["start", "start_sec", "start_frame", "music_start"]):
            raw = dict(raw)
            raw["start"] = cursor
        slot, feat = normalize_slot(raw, meta, i, fps=fps, source_path=str(p), temperature=temperature)
        cursor = float(slot["end"])
        slots.append(slot)
        feats.append(feat)
    if not feats:
        raise RuntimeError(f"MSSD parsed but produced no usable slots: {p}")
    # For final schedules, ensure target frames sum is explicit and stable.
    meta["num_slots"] = int(len(slots))
    meta["total_target_frames"] = int(sum(int(s.get("target_frames", 0)) for s in slots))
    return slots, np.stack(feats).astype(np.float32), meta




def build_descriptor_object(audio: str, slots: List[dict], meta: Dict[str, Any]) -> Dict[str, Any]:
    total = int(sum(int(s.get("target_frames", 0)) for s in slots))
    final = is_final_schedule_meta(meta)
    if final and meta.get("fps") is None:
        raise RuntimeError("Final MSSD descriptor metadata must declare fps")
    fps = float(meta.get("fps", 30.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"MSSD descriptor metadata has invalid fps: {fps!r}")
    out = {
        "descriptor_type": "music_semantic_slot_descriptor",
        "descriptor_schema_version": MSSD_SCHEMA_VERSION,
        "usage": str(meta.get("usage", "generate_schedule")),
        "is_final_schedule": bool(meta.get("is_final_schedule", True)),
        "slot_source": str(meta.get("slot_source", "music_router_whole_song_planner")),
        "audio": str(audio),
        "fps": fps,
        "num_slots": int(len(slots)),
        "total_target_frames": total,
        "slots": slots,
        # Alias for external semantic readers.
        "segments": slots,
        "provenance": dict(meta.get("provenance", {})),
    }
    for k in [
        "router_ckpt",
        "planner_ckpt",
        "duration_model_ckpt",
        "raw_schedule_json",
        "schedule_summary_json",
        "event_db_contract",
        "transition_budget",
        "music_independent_hard_constraints",
    ]:
        if k in meta and meta[k]:
            out[k] = meta[k]
    return out


# -----------------------------------------------------------------------------
# Semantic Routing Action Event Semantic Descriptor (AESD) and routing helpers
# -----------------------------------------------------------------------------
AESD_SCHEMA_VERSION = "semantic_routing_action_event_semantic_descriptor"

STAGE_AFFORDANCE_BY_LABEL = {
    "calm_meditative": ["intro", "calm", "release", "resolution"],
    "pose_hold": ["intro", "release", "resolution", "motif_recall"],
    "lyrical_flow": ["normal", "development", "motif", "motif_recall"],
    "instrument_phrase": ["normal", "development", "accent", "motif"],
    "percussive_accent": ["accent", "climax", "build_up"],
    "turning_climax": ["build_up", "climax", "accent"],
    "footwork_flow": ["normal", "development", "build_up", "motif"],
    "aerial_curve": ["normal", "development", "climax"],
}


def label_index_map() -> Dict[str, int]:
    return {k: i for i, k in enumerate(MUSIC_SEMANTIC_LABELS)}


def probs_to_vector(probs: Any, top_label: Any = None, temperature: float = 1.0) -> np.ndarray:
    p = normalize_probs(probs, top_label=top_label, temperature=temperature)
    return np.asarray([float(p.get(k, 0.0)) for k in MUSIC_SEMANTIC_LABELS], dtype=np.float32)


def one_hot_music(label: Any, strength: float = 1.0) -> np.ndarray:
    out = np.zeros((len(MUSIC_SEMANTIC_LABELS),), dtype=np.float32)
    lab = canonical_music_label(label)
    if lab in MUSIC_SEMANTIC_LABELS:
        out[MUSIC_SEMANTIC_LABELS.index(lab)] = float(strength)
    return out


def add_hint(vec: np.ndarray, label_or_labels: Any, weight: float) -> None:
    if label_or_labels is None:
        return
    if isinstance(label_or_labels, (list, tuple, set)):
        labs = list(label_or_labels)
    else:
        labs = [label_or_labels]
    for lab in labs:
        cl = canonical_music_label(lab)
        if cl in MUSIC_SEMANTIC_LABELS:
            vec[MUSIC_SEMANTIC_LABELS.index(cl)] += float(weight)


def normalize_vector(vec: np.ndarray, default: str = "lyrical_flow") -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size != len(MUSIC_SEMANTIC_LABELS):
        out = np.zeros((len(MUSIC_SEMANTIC_LABELS),), dtype=np.float32)
        out[: min(len(out), v.size)] = v[: min(len(out), v.size)]
        v = out
    v = np.maximum(v, 0.0)
    if float(v.sum()) <= 1e-8:
        v = one_hot_music(default, 1.0)
    v = v / max(float(v.sum()), 1e-8)
    return v.astype(np.float32)






def semantic_distribution_diagnostics(
    probabilities: np.ndarray,
    ambiguity_margin: float = 0.08,
) -> Dict[str, Any]:
    vector = normalize_vector(probabilities)
    order = np.argsort(-vector, kind="stable")
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    entropy = float(
        -(vector * np.log(vector + 1.0e-8)).sum()
        / max(math.log(len(vector)), 1.0e-8)
    )
    margin = float(vector[top] - vector[second])
    return {
        "top_label": MUSIC_SEMANTIC_LABELS[top],
        "secondary_label": MUSIC_SEMANTIC_LABELS[second],
        "normalized_entropy": entropy,
        "top2_margin": margin,
        "ambiguous": bool(margin < float(ambiguity_margin)),
    }


def class_prior_adjustment(
    probabilities: np.ndarray,
    class_prior: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    vector = normalize_vector(probabilities).astype(np.float64)
    prior = normalize_vector(class_prior).astype(np.float64)
    strength = float(np.clip(alpha, 0.0, 1.0))
    logits = np.log(vector + 1.0e-8) - strength * np.log(prior + 1.0e-8)
    logits -= float(np.max(logits))
    adjusted = np.exp(logits)
    return normalize_vector(adjusted)




def vector_to_prob_dict(vec: np.ndarray) -> Dict[str, float]:
    v = normalize_vector(vec)
    return {k: float(v[i]) for i, k in enumerate(MUSIC_SEMANTIC_LABELS)}


def stage_affordance_from_probs(vec: np.ndarray, explicit_stage: Any = None) -> List[str]:
    out: List[str] = []
    if explicit_stage and str(explicit_stage) != "unknown":
        out.append(str(explicit_stage))
    v = normalize_vector(vec)
    for idx in np.argsort(-v)[:3].tolist():
        lab = MUSIC_SEMANTIC_LABELS[int(idx)]
        out.extend(STAGE_AFFORDANCE_BY_LABEL.get(lab, []))
    # stable order, unique
    return list(dict.fromkeys([x for x in out if x]))


def intrinsic_transition_prior_from_arrays(
    entry: Optional[np.ndarray],
    exit_: Optional[np.ndarray],
    contact_entry: Optional[np.ndarray],
    contact_exit: Optional[np.ndarray],
    duration: float,
    quality: float,
    locomotion_label: Any = "unknown",
    support_label: Any = "unknown",
    low_threshold: float = 0.35,
    high_threshold: float = 0.65,
) -> Tuple[float, str]:
    """Event-local transition prior, not an event-to-event boundary score."""
    risk = 0.0
    try:
        entry_value = np.asarray(entry, dtype=np.float32).reshape(-1)
        exit_value = np.asarray(exit_, dtype=np.float32).reshape(-1)
        if entry_value.size >= 144 and exit_value.size >= 144:
            pose_gap = float(np.mean((exit_value[:72] - entry_value[:72]) ** 2))
            velocity = float(
                np.mean(np.abs(exit_value[72:144]))
                + np.mean(np.abs(entry_value[72:144]))
            )
            risk += float(np.clip(pose_gap * 4.0 + velocity * 2.5, 0.0, 0.55))
    except Exception:
        pass
    try:
        contact0 = np.asarray(contact_entry, dtype=np.float32).reshape(-1)
        contact1 = np.asarray(contact_exit, dtype=np.float32).reshape(-1)
        count = min(len(contact0), len(contact1))
        if count:
            risk += 0.20 * float(np.mean(np.abs(contact1[:count] - contact0[:count])))
    except Exception:
        pass
    if str(locomotion_label) in {"turning_travel", "accented_travel", "traveling_steps"}:
        risk += 0.04
    if str(support_label) in {"low_contact_flight_like", "alternating_or_pivot_support"}:
        risk += 0.04
    if float(duration) < 1.2:
        risk += 0.05
    risk += max(0.0, 0.45 - float(quality)) * 0.12
    risk = float(np.clip(risk, 0.0, 1.0))
    if not 0.0 < float(low_threshold) < float(high_threshold) < 1.0:
        raise ValueError("Intrinsic-risk thresholds must satisfy 0 < low < high < 1")
    if risk < float(low_threshold):
        profile = "low"
    elif risk < float(high_threshold):
        profile = "medium"
    else:
        profile = "high"
    return risk, profile
