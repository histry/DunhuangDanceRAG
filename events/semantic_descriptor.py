#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.35 Unified Music Semantic Slot Descriptor (MSSD)
=====================================================

This module unifies two formerly separated JSON concepts:

1) external music semantic sidecar for unpaired-audio V44 training;
2) final V21/V26/V23 slot plan for V46 whole-song generation.

The format is intentionally backward compatible with existing `slots`,
`segments`, and V26 `schedule` JSON files.  The important distinction is kept
explicit through:

- usage:                train_semantic | generate_schedule
- is_final_schedule:    false | true
- slot_source:          external_sidecar | v21_router_v26_planner | ...

Training may consume weak descriptors without final timing.  Generation in
scientific/strict mode must consume a final schedule descriptor.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MSSD_SCHEMA_VERSION = "v46_38_mssd_aesd_routing_descriptor"

# Keep the same label space as V46 music_alignment labels.  `aerial_curve` is
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

MUSIC_LABEL_TO_ACTION: Dict[str, List[str]] = {
    "calm_meditative": ["revelation_meditation", "thirty_six_postures", "lotus_steps"],
    "pose_hold": ["thirty_six_postures", "revelation_meditation", "lotus_steps"],
    "lyrical_flow": ["lotus_steps", "ribbon_flow", "revelation_meditation", "thirty_six_postures"],
    "instrument_phrase": ["pipa_behind_back", "ribbon_flow", "thirty_six_postures"],
    "percussive_accent": ["lei_gong_drum", "pipa_behind_back", "ribbon_flow"],
    "turning_climax": ["ribbon_flow", "lei_gong_drum", "pipa_behind_back", "sogdian_whirl"],
    "footwork_flow": ["lotus_steps", "ribbon_flow", "lei_gong_drum"],
    "aerial_curve": ["ribbon_flow", "lotus_steps", "thirty_six_postures"],
}

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

# 32D pseudo feature follows V46 descriptor layout sufficiently for V44 OT and
# V46 retrieval.  It deliberately reserves class channels around 22/23/26/28-30.
LABEL_PROTOTYPES: Dict[str, Tuple[float, float, float, float]] = {
    "calm_meditative": (0.020, 0.010, 0.012, 0.85),
    "pose_hold": (0.030, 0.012, 0.010, 0.78),
    "lyrical_flow": (0.055, 0.035, 0.040, 0.42),
    "instrument_phrase": (0.070, 0.065, 0.055, 0.30),
    "percussive_accent": (0.105, 0.125, 0.100, 0.12),
    "turning_climax": (0.095, 0.080, 0.110, 0.08),
    "footwork_flow": (0.065, 0.045, 0.070, 0.34),
    "aerial_curve": (0.060, 0.040, 0.080, 0.36),
}


def env_bool(name: str, default: bool = False) -> bool:
    try:
        return bool(int(os.environ.get(name, "1" if default else "0")))
    except Exception:
        return bool(default)


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
    preferred = MUSIC_LABEL_TO_ACTION.get(top, MUSIC_LABEL_TO_ACTION["lyrical_flow"])
    return {
        "role": role,
        "slot_role": role,
        "music_alignment_label": top,
        "music_semantic_top_label": top,
        "music_semantic_probs": {k: float(v) for k, v in probs.items()},
        "energy_label": energy,
        "rhythm_label": rhythm,
        "preferred_dance_keys": list(preferred),
        "slot_preferred_dance_keys": list(preferred),
        "preferred_semantic_roles": [],
        "external_music_semantic_source": str(source),
    }


def pseudo_feature_from_probs(probs: Dict[str, float], duration: float) -> np.ndarray:
    energy = onset = dyn = calm = 0.0
    for label, p in probs.items():
        e, o, d, c = LABEL_PROTOTYPES.get(label, LABEL_PROTOTYPES["lyrical_flow"])
        energy += float(p) * e
        onset += float(p) * o
        dyn += float(p) * d
        calm += float(p) * c
    feat = np.zeros(32, dtype=np.float32)
    feat[0] = float(duration)
    feat[1] = energy * 2.0
    feat[2] = energy
    feat[3] = max(energy, dyn)
    feat[4] = dyn
    feat[5] = energy + onset
    feat[6] = onset + dyn
    feat[7] = energy + 0.5 * onset
    feat[8] = energy
    feat[9] = 1.0 + onset
    feat[10] = calm
    feat[13] = max(0.02, onset)
    feat[14] = onset
    feat[16] = onset
    feat[17] = dyn
    feat[18] = dyn
    top = top_label_from_probs(probs)
    # These are normalized categorical channels aligned with V46 semantic dims.
    feat[22] = 0.0 if top in {"calm_meditative", "pose_hold"} else (1.0 if top == "percussive_accent" else 0.5)
    feat[23] = 1.0 if top == "percussive_accent" else (0.75 if top in {"turning_climax", "instrument_phrase"} else 0.35)
    feat[26] = MUSIC_SEMANTIC_LABELS.index(top) / max(1, len(MUSIC_SEMANTIC_LABELS) - 1)
    feat[28] = float(probs.get("calm_meditative", 0.0) + 0.5 * probs.get("pose_hold", 0.0))
    feat[29] = float(probs.get("percussive_accent", 0.0) + 0.4 * probs.get("instrument_phrase", 0.0))
    feat[30] = float(probs.get("turning_climax", 0.0) + 0.3 * probs.get("footwork_flow", 0.0))
    feat[31] = 1.0
    return feat.astype(np.float32)


def is_final_schedule_meta(meta: Dict[str, Any]) -> bool:
    usage = str(meta.get("usage", "")).lower()
    src = str(meta.get("slot_source", meta.get("source", ""))).lower()
    if bool(meta.get("is_final_schedule", False)):
        return True
    if usage in {"generate", "generate_schedule", "final_schedule", "router_schedule"}:
        return True
    if any(k in src for k in ["v21", "v23", "v26", "router", "planner", "pretrained"]):
        return True
    return False


def _extract_raw_slots(obj: Any) -> Tuple[List[dict], Dict[str, Any]]:
    if isinstance(obj, dict):
        # Native MSSD / V46 slots JSON.
        for key in ["slots", "segments", "descriptors"]:
            if isinstance(obj.get(key), list):
                return list(obj[key]), dict(obj)
        # Raw V26 schedule report.
        if isinstance(obj.get("schedule"), list):
            meta = dict(obj)
            meta.setdefault("usage", "generate_schedule")
            meta.setdefault("is_final_schedule", True)
            meta.setdefault("slot_source", "v21_router_v26_planner")
            return list(obj["schedule"]), meta
        # V26 summary file: caller normally resolves report, but support simple form.
        results = obj.get("results")
        if isinstance(results, dict) and len(results) == 1:
            val = next(iter(results.values()))
            if isinstance(val, dict) and val.get("report") and Path(str(val["report"])).exists():
                return _extract_raw_slots(json_load(str(val["report"])))
        # Generated V46 report fallback.
        sr = obj.get("stage_reports")
        if isinstance(sr, dict) and isinstance(sr.get("retrieval"), list):
            slots = []
            for r in sr["retrieval"]:
                if isinstance(r, dict):
                    slots.append({
                        "slot_id": r.get("slot", r.get("slot_id", len(slots))),
                        "duration": r.get("duration", 4.0),
                        "music_alignment_label": r.get("slot_music_alignment_label", r.get("music_alignment_label", "lyrical_flow")),
                        "music_semantic_top_label": r.get("slot_music_semantic_top_label", r.get("slot_music_alignment_label", "lyrical_flow")),
                        "music_semantic_probs": r.get("slot_music_semantic_probs", {}),
                        "preferred_dance_keys": r.get("slot_preferred_dance_keys", []),
                    })
            if slots:
                return slots, dict(obj)
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        return list(obj), {"usage": "train_semantic", "is_final_schedule": False, "slot_source": "list_slots"}
    return [], {}


def slot_duration_frames(slot: dict, fps: float, default_index: int = 0, default_seconds: float = 4.0) -> Tuple[float, float, float, int]:
    # V26 schedule uses frame-level music_start/music_end/music_length and
    # allocated_phrase_total.  Native MSSD may use seconds and/or frame indices.
    target_frames = slot.get("target_frames", slot.get("allocated_phrase_total", slot.get("v26_allocated_phrase_total", None)))
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
    if feature.size < 32 or not np.isfinite(feature[:32]).all() or float(np.max(np.abs(feature[:32]))) == 0.0:
        feature = pseudo_feature_from_probs(probs, dur)
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
        **sem,
    }
    # Preserve V26 raw fields in a predictable namespace for auditing.
    if "event_id" in slot and "v26_event_id" not in out:
        out["v26_event_id"] = slot.get("event_id")
    if "event_uid" in slot and "v26_event_uid" not in out:
        out["v26_event_uid"] = slot.get("event_uid")
    if "event_index" in slot and "v26_event_index" not in out:
        out["v26_event_index"] = slot.get("event_index")
    if "family_id" in slot and "v26_family_id" not in out:
        out["v26_family_id"] = slot.get("family_id")
    if "allocated_content_len" in slot and "v26_allocated_content_len" not in out:
        out["v26_allocated_content_len"] = slot.get("allocated_content_len")
    if "allocated_phrase_total" in slot and "v26_allocated_phrase_total" not in out:
        out["v26_allocated_phrase_total"] = slot.get("allocated_phrase_total")
    if "time_warp_ratio" in slot and "v26_time_warp_ratio" not in out:
        out["v26_time_warp_ratio"] = slot.get("time_warp_ratio")
    return out, feature[:32].astype(np.float32)


def parse_descriptor_file(path: str | Path, *, require_final_schedule: bool = False, fps: float = 30.0, temperature: float = 0.65, usage: str = "auto") -> Tuple[List[dict], np.ndarray, Dict[str, Any]]:
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".npz":
        data = np.load(p, allow_pickle=True)
        label_names = data["label_names"].tolist() if "label_names" in data.files else MUSIC_SEMANTIC_LABELS
        probs_arr = np.asarray(data["slot_probs"] if "slot_probs" in data.files else data["probs"], dtype=np.float32)
        starts = np.asarray(data["slot_start"] if "slot_start" in data.files else (data["start"] if "start" in data.files else np.arange(len(probs_arr)) * 4.0), dtype=np.float32)
        ends = np.asarray(data["slot_end"] if "slot_end" in data.files else (data["end"] if "end" in data.files else starts + 4.0), dtype=np.float32)
        labels = data["slot_label"].tolist() if "slot_label" in data.files else [label_names[int(np.argmax(r))] for r in probs_arr]
        raw_slots = []
        for i in range(len(probs_arr)):
            probs = {str(label_names[j]): float(probs_arr[i, j]) for j in range(min(len(label_names), probs_arr.shape[1]))}
            raw_slots.append({"slot_id": i, "start": float(starts[i]), "end": float(ends[i]), "top_label": labels[i], "probs": probs})
        meta: Dict[str, Any] = {"usage": "train_semantic", "is_final_schedule": False, "slot_source": "external_npz", "descriptor_type": "music_semantic_slot_descriptor"}
    else:
        obj = json_load(p)
        raw_slots, meta = _extract_raw_slots(obj)
        if not raw_slots:
            raise RuntimeError(f"MSSD has no slots/segments/schedule: {p}")
    meta = dict(meta)
    meta.setdefault("descriptor_type", "music_semantic_slot_descriptor")
    meta.setdefault("descriptor_schema_version", MSSD_SCHEMA_VERSION)
    meta.setdefault("slot_source", "v21_router_v26_planner" if is_final_schedule_meta(meta) else "external_sidecar")
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


def sidecar_candidate_names(stem: str) -> List[str]:
    return [
        f"{stem}.mssd.json", f"{stem}_mssd.json",
        f"{stem}.music_semantic_slot.json", f"{stem}_music_semantic_slot.json",
        f"{stem}.music_semantic.json", f"{stem}_music_semantic.json",
        f"{stem}.semantic.json", f"{stem}_semantic.json",
        f"{stem}.mssd.npz", f"{stem}_mssd.npz",
        f"{stem}.music_semantic.npz", f"{stem}_music_semantic.npz",
        f"{stem}.semantic.npz", f"{stem}_semantic.npz",
        f"{stem}.json", f"{stem}.npz",
    ]


def split_path_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    out = []
    for chunk in text.replace(";", os.pathsep).split(os.pathsep):
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
    return out


def descriptor_candidates_for_audio(audio_path: str | Path, descriptor_dirs: Any = None) -> List[Path]:
    p = Path(audio_path)
    names = sidecar_candidate_names(p.stem)
    cands = [p.with_name(n) for n in names]
    for d in split_path_list(descriptor_dirs):
        dp = Path(d)
        for n in names:
            cands.append(dp / n)
    # Preserve order and remove duplicates.
    seen = set()
    uniq = []
    for c in cands:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq.append(c)
    return uniq


def load_descriptor_for_audio(audio_path: str | Path, *, descriptor_dirs: Any = None, require_final_schedule: bool = False, fps: float = 30.0, temperature: float = 0.65, usage: str = "auto") -> Optional[Tuple[List[dict], np.ndarray, Dict[str, Any]]]:
    for cand in descriptor_candidates_for_audio(audio_path, descriptor_dirs):
        if cand.exists() and cand.is_file():
            try:
                return parse_descriptor_file(cand, require_final_schedule=require_final_schedule, fps=fps, temperature=temperature, usage=usage)
            except Exception as exc:
                print(f"[V46.35 MSSD WARN] failed descriptor {cand}: {exc}", file=sys.stderr)
    if require_final_schedule:
        raise RuntimeError(f"No final MSSD descriptor found for {audio_path}; searched dirs={descriptor_dirs}")
    return None


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
        "slot_source": str(meta.get("slot_source", "v21_router_v26_planner")),
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
        "v23_ckpt",
        "raw_schedule_json",
        "schedule_summary_json",
        "event_db_contract",
        "transition_budget",
    ]:
        if k in meta and meta[k]:
            out[k] = meta[k]
    return out


# -----------------------------------------------------------------------------
# V46.38 Action Event Semantic Descriptor (AESD) and routing helpers
# -----------------------------------------------------------------------------
AESD_SCHEMA_VERSION = "v46_38_action_event_semantic_descriptor"

DANCE_KEY_TO_MUSIC_LABEL = {
    "revelation_meditation": "calm_meditative",
    "thirty_six_postures": "pose_hold",
    "pipa_behind_back": "instrument_phrase",
    "lei_gong_drum": "percussive_accent",
    "lotus_steps": "footwork_flow",
    "sogdian_whirl": "turning_climax",
    "ribbon_flow": "lyrical_flow",
}
EVENT_FAMILY_TO_MUSIC_LABEL = {
    "calm_flow": "calm_meditative",
    "pose_motif": "pose_hold",
    "aerial_curve": "lyrical_flow",
    "instrument_motif": "instrument_phrase",
    "percussive_accent": "percussive_accent",
    "turning_flow": "turning_climax",
    "footwork_flow": "footwork_flow",
}
ENERGY_TO_MUSIC_HINT = {
    "calm": ["calm_meditative", "pose_hold"],
    "low": ["calm_meditative", "pose_hold"],
    "moderate": ["lyrical_flow", "footwork_flow", "instrument_phrase"],
    "high": ["turning_climax", "percussive_accent", "footwork_flow"],
    "percussive": ["percussive_accent", "turning_climax"],
}
RHYTHM_TO_MUSIC_HINT = {
    "sustained": ["calm_meditative", "pose_hold"],
    "lyrical": ["lyrical_flow", "footwork_flow"],
    "accented": ["instrument_phrase", "turning_climax", "percussive_accent"],
    "percussive": ["percussive_accent"],
}
LOCO_TO_MUSIC_HINT = {
    "in_place_pose": ["pose_hold", "calm_meditative"],
    "slow_weight_shift": ["calm_meditative", "lyrical_flow"],
    "floating_leaning": ["lyrical_flow", "aerial_curve"],
    "upper_body_phrase": ["instrument_phrase", "lyrical_flow"],
    "traveling_steps": ["footwork_flow", "lyrical_flow"],
    "turning_travel": ["turning_climax", "footwork_flow"],
    "accented_travel": ["percussive_accent", "footwork_flow"],
}
SUPPORT_TO_MUSIC_HINT = {
    "stable_support": ["calm_meditative", "pose_hold", "instrument_phrase"],
    "static_or_low_motion_support": ["calm_meditative", "pose_hold"],
    "strong_foot_contact": ["percussive_accent", "footwork_flow"],
    "alternating_foot_support": ["footwork_flow", "lyrical_flow"],
    "alternating_or_pivot_support": ["turning_climax", "footwork_flow"],
    "low_contact_flight_like": ["turning_climax", "aerial_curve"],
}
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


def event_probs_from_fields(
    *,
    dance_key: Any = "unknown",
    event_family: Any = "unknown",
    music_alignment_label: Any = "unknown",
    energy_label: Any = "unknown",
    rhythm_label: Any = "unknown",
    locomotion_label: Any = "unknown",
    support_label: Any = "unknown",
    quality: float = 0.5,
    semantic_confidence: float = 0.5,
    desc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Soft action-to-music response distribution for AESD.

    The vector is not a single hard label.  It fuses cultural class, event family,
    existing music_alignment label, energy/rhythm/locomotion/support, and low-level
    motion descriptor hints.  This keeps the action side symmetric with MSSD.
    """
    vec = np.zeros((len(MUSIC_SEMANTIC_LABELS),), dtype=np.float32)
    dk = str(dance_key or "unknown")
    fam = str(event_family or "unknown")
    align = str(music_alignment_label or "unknown")
    if align and align != "unknown":
        add_hint(vec, align, 1.20)
    if dk in DANCE_KEY_TO_MUSIC_LABEL:
        add_hint(vec, DANCE_KEY_TO_MUSIC_LABEL[dk], 0.95)
    else:
        add_hint(vec, dk, 0.45)
    if fam in EVENT_FAMILY_TO_MUSIC_LABEL:
        add_hint(vec, EVENT_FAMILY_TO_MUSIC_LABEL[fam], 0.90)
    else:
        add_hint(vec, fam, 0.35)
    add_hint(vec, ENERGY_TO_MUSIC_HINT.get(str(energy_label or ""), []), 0.28)
    add_hint(vec, RHYTHM_TO_MUSIC_HINT.get(str(rhythm_label or ""), []), 0.24)
    add_hint(vec, LOCO_TO_MUSIC_HINT.get(str(locomotion_label or ""), []), 0.28)
    add_hint(vec, SUPPORT_TO_MUSIC_HINT.get(str(support_label or ""), []), 0.18)
    if desc is not None:
        d = np.asarray(desc, dtype=np.float32).reshape(-1)
        # Descriptor channels are coarse but stable: duration/root energy/onset/turn bits.
        if d.size > 30:
            if float(d[28]) > 0.35:
                add_hint(vec, ["calm_meditative", "pose_hold"], 0.20 * float(d[28]))
            if float(d[29]) > 0.25:
                add_hint(vec, ["percussive_accent", "instrument_phrase"], 0.22 * float(d[29]))
            if float(d[30]) > 0.25:
                add_hint(vec, ["turning_climax", "footwork_flow"], 0.22 * float(d[30]))
        if d.size > 18:
            if float(d[16]) > 0.08 or float(d[17]) > 0.08:
                add_hint(vec, ["percussive_accent", "turning_climax"], 0.18)
    q = float(np.clip(float(quality), 0.0, 1.0))
    c = float(np.clip(float(semantic_confidence), 0.0, 1.0))
    # Low confidence pushes distribution softer rather than zeroing it.
    vec = vec * (0.55 + 0.30 * q + 0.15 * c)
    return normalize_vector(vec)


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


def boundary_risk_from_arrays(
    entry: Optional[np.ndarray],
    exit_: Optional[np.ndarray],
    contact_entry: Optional[np.ndarray],
    contact_exit: Optional[np.ndarray],
    duration: float,
    quality: float,
    locomotion_label: Any = "unknown",
    support_label: Any = "unknown",
) -> Tuple[float, str]:
    risk = 0.0
    try:
        e = np.asarray(entry, dtype=np.float32).reshape(-1)
        x = np.asarray(exit_, dtype=np.float32).reshape(-1)
        if e.size >= 144 and x.size >= 144:
            pose_gap = float(np.mean((x[:72] - e[:72]) ** 2))
            vel_mag = float(np.mean(np.abs(x[72:144])) + np.mean(np.abs(e[72:144])))
            risk += np.clip(pose_gap * 5.0 + vel_mag * 3.0, 0.0, 0.55)
    except Exception:
        pass
    try:
        c0 = np.asarray(contact_entry, dtype=np.float32).reshape(-1)
        c1 = np.asarray(contact_exit, dtype=np.float32).reshape(-1)
        if c0.size and c1.size:
            risk += 0.25 * float(np.mean(np.abs(c1[: min(len(c0), len(c1))] - c0[: min(len(c0), len(c1))])))
    except Exception:
        pass
    if str(locomotion_label) in {"turning_travel", "accented_travel", "traveling_steps"}:
        risk += 0.12
    if str(support_label) in {"low_contact_flight_like", "alternating_or_pivot_support"}:
        risk += 0.10
    if float(duration) < 1.2:
        risk += 0.08
    risk += max(0.0, 0.45 - float(quality)) * 0.20
    risk = float(np.clip(risk, 0.0, 1.0))
    if risk < 0.28:
        prof = "low"
    elif risk < 0.58:
        prof = "medium"
    else:
        prof = "high"
    return risk, prof


def get_db_array(db: Dict[str, Any], key: str, n: int, default: Any, dtype: Any = object) -> np.ndarray:
    if key in db:
        return np.asarray(db[key], dtype=dtype)
    return np.asarray([default] * n, dtype=dtype)


def get_aesd_prob_matrix(db: Dict[str, Any], n: Optional[int] = None) -> np.ndarray:
    if n is None:
        n = len(db.get("paths", []))
    if "aesd_music_alignment_probs" in db:
        arr = np.asarray(db["aesd_music_alignment_probs"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == n:
            if arr.shape[1] < len(MUSIC_SEMANTIC_LABELS):
                pad = np.zeros((n, len(MUSIC_SEMANTIC_LABELS) - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
            return np.stack([normalize_vector(r) for r in arr[:, : len(MUSIC_SEMANTIC_LABELS)]], axis=0).astype(np.float32)
    desc = np.asarray(db.get("desc", np.zeros((n, 32), dtype=np.float32)), dtype=np.float32)
    dance = get_db_array(db, "dance_keys", n, "unknown", object)
    fam = get_db_array(db, "event_families", n, "unknown", object)
    align = get_db_array(db, "music_alignment_labels", n, "unknown", object)
    energy = get_db_array(db, "energy_labels", n, "unknown", object)
    rhythm = get_db_array(db, "rhythm_labels", n, "unknown", object)
    loco = get_db_array(db, "locomotion_labels", n, "unknown", object)
    support = get_db_array(db, "support_labels", n, "unknown", object)
    qual = np.asarray(db.get("event_quality_scores", np.ones(n, dtype=np.float32) * 0.5), dtype=np.float32)
    conf = np.asarray(db.get("semantic_confidence", np.ones(n, dtype=np.float32) * 0.5), dtype=np.float32)
    rows = []
    for i in range(n):
        rows.append(event_probs_from_fields(
            dance_key=dance[i], event_family=fam[i], music_alignment_label=align[i],
            energy_label=energy[i], rhythm_label=rhythm[i], locomotion_label=loco[i], support_label=support[i],
            quality=float(qual[i]) if i < len(qual) else 0.5,
            semantic_confidence=float(conf[i]) if i < len(conf) else 0.5,
            desc=desc[i] if desc.ndim == 2 and i < desc.shape[0] else None,
        ))
    return np.stack(rows).astype(np.float32)


def slot_prob_vector(slot: Dict[str, Any]) -> np.ndarray:
    return probs_to_vector(slot.get("music_semantic_probs", None), slot.get("music_semantic_top_label", slot.get("music_alignment_label", None)), temperature=1.0)


def dot_compat(slot_vec: np.ndarray, aesd_matrix: np.ndarray) -> np.ndarray:
    s = normalize_vector(slot_vec)
    a = np.asarray(aesd_matrix, dtype=np.float32)
    a = np.stack([normalize_vector(row) for row in a], axis=0)
    return np.clip(a @ s, 0.0, 1.0).astype(np.float32)
