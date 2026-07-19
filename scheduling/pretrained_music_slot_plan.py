#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V46.34 pretrained music-router slot-plan builder.

This script is intentionally independent from README assumptions.  It connects
previously trained V21/V26 music-structure weights to the current V46
MotionRAG-Diff generator by producing a V46-compatible --slots_json file.

Priority order
--------------
1) Strict/preferred: call scheduling/schedule_whole_song.py with --router_ckpt
   and optional --planner_ckpt.  This uses the trained V21 music router and V26
   whole-song planner to segment an unseen song and produce a schedule.
2) Optional controlled fallback: if V46_34_ALLOW_SEMANTIC_FALLBACK=1, build a
   slot plan from an existing music_semantics/<song>.music_semantic.json plus
   acoustic energy/onset.  This is only for debugging and is marked as fallback.

Output schema
-------------
{
  "version": "v46_34_pretrained_router_slot_plan",
  "audio": "...wav",
  "slot_source": "v21_router_v26_planner" | "external_music_semantic_fallback",
  "router_ckpt": "...best.pt",
  "planner_ckpt": "...best.pt",
  "slots": [
    {
      "slot_id": 0,
      "start": 0.0,
      "end": 4.0,
      "duration": 4.0,
      "target_frames": 120,
      "music_alignment_label": "percussive_accent",
      "music_semantic_probs": {...},
      "feature": [32 floats]
    }
  ]
}

The current training/motion_models.py already supports --slots_json.  The
companion patch v46_34_router_slot_patch.py strengthens that loader so missing
features are synthesized and strict router-slot mode can be enforced.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import scipy.io.wavfile as wavfile
except Exception:  # pragma: no cover
    wavfile = None

SEMANTIC_LABELS = [
    "calm_meditative",
    "lyrical_flow",
    "pose_hold",
    "instrument_phrase",
    "percussive_accent",
    "turning_climax",
    "footwork_flow",
]

PREFERRED_DANCE_KEYS = {
    "calm_meditative": ["revelation_meditation", "thirty_six_postures", "lotus_steps"],
    "lyrical_flow": ["lotus_steps", "revelation_meditation", "thirty_six_postures"],
    "pose_hold": ["thirty_six_postures", "revelation_meditation"],
    "instrument_phrase": ["pipa_behind_back", "thirty_six_postures"],
    "percussive_accent": ["lei_gong_drum", "pipa_behind_back"],
    "turning_climax": ["sogdian_whirl", "lei_gong_drum", "pipa_behind_back"],
    "footwork_flow": ["lotus_steps", "revelation_meditation"],
}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_wav_mono(path: str | Path) -> Tuple[int, np.ndarray]:
    if wavfile is None:
        raise RuntimeError("scipy.io.wavfile is required for audio fallback features")
    sr, wav = wavfile.read(str(path))
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    mx = float(np.max(np.abs(wav))) if wav.size else 1.0
    if mx > 1.0:
        wav = wav / mx
    return int(sr), wav


def audio_features_for_segment(seg: np.ndarray, sr: int) -> Dict[str, float]:
    seg = np.asarray(seg, dtype=np.float32)
    if seg.size < 8:
        return {"energy": 0.0, "onset": 0.0, "dynamic": 0.0, "zcr": 0.0}
    frame = max(64, int(0.05 * sr))
    hop = max(16, frame // 2)
    env = []
    zcr = []
    for st in range(0, max(1, seg.size - frame + 1), hop):
        x = seg[st: st + frame]
        if x.size == 0:
            continue
        env.append(float(np.sqrt(np.mean(x * x) + 1e-8)))
        zcr.append(float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.float32))))))
    env = np.asarray(env if env else [0.0], dtype=np.float32)
    onset = np.diff(env, prepend=env[:1])
    onset = np.maximum(onset, 0.0)
    return {
        "energy": float(np.mean(env)),
        "energy_p90": float(np.percentile(env, 90)),
        "onset": float(np.mean(onset)),
        "onset_p90": float(np.percentile(onset, 90)),
        "dynamic": float(np.std(env) + np.std(onset)),
        "zcr": float(np.mean(zcr) if zcr else 0.0),
    }


def normalize_probs(obj: Any, top_label: Optional[str] = None) -> Dict[str, float]:
    if isinstance(obj, dict):
        raw = {k: float(v) for k, v in obj.items() if k in SEMANTIC_LABELS}
    else:
        raw = {}
    if not raw and top_label in SEMANTIC_LABELS:
        raw = {k: 0.02 for k in SEMANTIC_LABELS}
        raw[str(top_label)] = 0.88
    if not raw:
        raw = {k: 1.0 / len(SEMANTIC_LABELS) for k in SEMANTIC_LABELS}
    s = sum(max(0.0, v) for v in raw.values())
    if s <= 1e-8:
        return {k: 1.0 / len(SEMANTIC_LABELS) for k in SEMANTIC_LABELS}
    return {k: float(max(0.0, raw.get(k, 0.0)) / s) for k in SEMANTIC_LABELS}


def feature32(duration: float, probs: Dict[str, float], audio_stats: Optional[Dict[str, float]] = None) -> List[float]:
    audio_stats = dict(audio_stats or {})
    energy = float(audio_stats.get("energy", 0.06))
    onset = float(audio_stats.get("onset", audio_stats.get("onset_p90", 0.02)))
    dyn = float(audio_stats.get("dynamic", 0.04))
    p = normalize_probs(probs)
    # V46 event_descriptor-like coarse layout.  These are not raw CLAP features;
    # they are retrieval-facing pseudo features with explicit semantic logits.
    x = np.zeros(32, dtype=np.float32)
    x[0] = float(duration)
    x[1] = 2.0 * energy + 0.25 * p["footwork_flow"] + 0.25 * p["percussive_accent"]
    x[2] = energy
    x[3] = float(audio_stats.get("energy_p90", energy))
    x[4] = dyn
    x[5] = energy + onset + 0.35 * p["percussive_accent"]
    x[6] = float(audio_stats.get("onset_p90", onset))
    x[7] = energy + 0.5 * onset + 0.25 * p["footwork_flow"]
    x[8] = energy
    x[9] = 1.0 + onset + 0.35 * p["turning_climax"]
    x[10] = 0.65 * p["calm_meditative"] + 0.45 * p["pose_hold"]
    x[11] = p["pose_hold"]
    x[12] = p["calm_meditative"]
    x[13] = max(0.02, onset + 0.2 * p["percussive_accent"])
    x[14] = float(audio_stats.get("zcr", 0.0))
    x[15] = 0.25 * p["turning_climax"]
    x[16] = p["turning_climax"]
    x[17] = p["footwork_flow"]
    x[18] = p["instrument_phrase"]
    x[19] = p["lyrical_flow"]
    x[20] = p["percussive_accent"]
    x[21] = p["calm_meditative"]
    x[22] = p["pose_hold"]
    # Add complete semantic vector in tail for contrastive model / retrieval.
    for i, lab in enumerate(SEMANTIC_LABELS):
        if 23 + i < 32:
            x[23 + i] = p[lab]
    x[30] = SEMANTIC_LABELS.index(max(p, key=p.get)) / max(1, len(SEMANTIC_LABELS) - 1)  # stable category hint
    x[31] = 1.0
    return x.astype(float).tolist()


def find_music_semantic(audio: str | Path, dirs: Iterable[str | Path]) -> Optional[Path]:
    stem = Path(audio).stem
    for d in dirs:
        if not d:
            continue
        dd = Path(d)
        for name in [f"{stem}.music_semantic.json", f"{stem}.json"]:
            p = dd / name
            if p.exists():
                return p
    return None


def slot_label_from_probs(probs: Dict[str, float]) -> str:
    return max(normalize_probs(probs).items(), key=lambda kv: kv[1])[0]


def extract_slots_from_any_json(data: Any) -> Optional[List[Dict[str, Any]]]:
    """Find a slot-like list in arbitrary V21/V26/V34 reports."""
    if isinstance(data, dict):
        if isinstance(data.get("slots"), list) and data["slots"]:
            return list(data["slots"])
        # Some V26 reports store slot info under stage_reports.retrieval.
        sr = data.get("stage_reports")
        if isinstance(sr, dict) and isinstance(sr.get("retrieval"), list) and sr["retrieval"]:
            out = []
            for r in sr["retrieval"]:
                if not isinstance(r, dict):
                    continue
                out.append({
                    "slot_id": r.get("slot", r.get("slot_id", len(out))),
                    "duration": r.get("duration", r.get("slot_duration", 4.0)),
                    "music_alignment_label": r.get("slot_music_alignment_label", r.get("music_alignment_label", r.get("slot_role", "calm_meditative"))),
                    "music_semantic_top_label": r.get("slot_music_semantic_top_label", r.get("slot_music_alignment_label", "calm_meditative")),
                    "music_semantic_probs": r.get("slot_music_semantic_probs", {}),
                    "preferred_dance_keys": r.get("slot_preferred_dance_keys", []),
                })
            if out:
                return out
        # Recurse through likely keys first.
        for k in ["results", "reports", "summary", "songs", "items"]:
            if k in data:
                found = extract_slots_from_any_json(data[k])
                if found:
                    return found
        for v in data.values():
            found = extract_slots_from_any_json(v)
            if found:
                return found
    elif isinstance(data, list):
        if data and all(isinstance(x, dict) for x in data):
            keys = set().union(*(x.keys() for x in data[: min(8, len(data))]))
            if {"duration", "music_alignment_label"} & keys or {"start", "end"} <= keys:
                return list(data)
        for v in data:
            found = extract_slots_from_any_json(v)
            if found:
                return found
    return None


def run_v26_scheduler(args: argparse.Namespace, schedule_dir: Path) -> Optional[Path]:
    if not args.router_ckpt:
        return None
    required = [args.index_json, args.duration_index_npz, args.v23_ckpt]
    if not all(required):
        return None
    if not Path("scheduling/schedule_whole_song.py").exists():
        return None
    cmd = [
        sys.executable, "scheduling/schedule_whole_song.py",
        "--index_json", str(args.index_json),
        "--duration_index_npz", str(args.duration_index_npz),
        "--music", str(args.audio),
        "--out_dir", str(schedule_dir),
        "--router_ckpt", str(args.router_ckpt),
        "--v23_ckpt", str(args.v23_ckpt),
        "--feature_dir", str(args.feature_dir or (schedule_dir / "music_features")),
        "--fps", str(args.fps),
        "--min_phrase_seconds", str(args.min_phrase_seconds),
        "--max_phrase_seconds", str(args.max_phrase_seconds),
        "--max_phrases", str(args.max_phrases),
        "--multi_event_phrases", "1",
        "--lock_music_boundaries", "1",
        "--music_dominant_timing", "1",
    ]
    if args.planner_ckpt:
        cmd += ["--planner_ckpt", str(args.planner_ckpt)]
    if args.hierarchy_index_npz:
        cmd += ["--hierarchy_index_npz", str(args.hierarchy_index_npz)]
    if args.max_seconds and float(args.max_seconds) > 0:
        cmd += ["--max_seconds", str(args.max_seconds)]
    print("[V46.34 SCHEDULE]", " ".join(cmd), flush=True)
    schedule_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    jsons = sorted(schedule_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in jsons:
        try:
            slots = extract_slots_from_any_json(load_json(p))
            if slots:
                return p
        except Exception:
            continue
    return None


def build_fallback_slots(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sem_dirs = list(args.music_semantic_dirs or [])
    sem_path = find_music_semantic(args.audio, sem_dirs)
    if sem_path is None:
        raise RuntimeError(
            "No V21/V26 schedule could be produced and no music semantic JSON found. "
            "Set V26_INDEX_JSON/V26_DURATION_INDEX_NPZ/V26_V23_CKPT/V26_ROUTER_CKPT, "
            "or set V46_34_ALLOW_SEMANTIC_FALLBACK=1 with music_semantics/<song>.music_semantic.json."
        )
    data = load_json(sem_path)
    slots = extract_slots_from_any_json(data)
    sr, wav = read_wav_mono(args.audio)
    if not slots:
        # If semantic file stores probabilities per fixed chunk under a different key,
        # fall back to regular 4s slots while marking it explicitly.
        total = wav.size / sr
        n = max(1, int(math.ceil(total / float(args.slot_seconds))))
        global_probs = normalize_probs(data.get("music_semantic_probs", data.get("probs", {})) if isinstance(data, dict) else {})
        top = slot_label_from_probs(global_probs)
        slots = []
        for i in range(n):
            slots.append({
                "slot_id": i,
                "start": i * float(args.slot_seconds),
                "end": min(total, (i + 1) * float(args.slot_seconds)),
                "duration": min(float(args.slot_seconds), max(0.0, total - i * float(args.slot_seconds))),
                "music_alignment_label": top,
                "music_semantic_top_label": top,
                "music_semantic_probs": global_probs,
            })
    return slots, {"fallback_semantic_json": str(sem_path), "fallback": True}


def finalize_slots(raw_slots: List[Dict[str, Any]], audio: str | Path, args: argparse.Namespace, source_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    sr, wav = read_wav_mono(audio)
    total = wav.size / sr
    out: List[Dict[str, Any]] = []
    cursor = 0.0
    for i, s0 in enumerate(raw_slots):
        s = dict(s0)
        dur = s.get("duration", s.get("duration_sec", None))
        st = s.get("start", s.get("start_sec", s.get("music_start", None)))
        ed = s.get("end", s.get("end_sec", s.get("music_end", None)))
        if dur is None and st is not None and ed is not None:
            dur = float(ed) - float(st)
        if dur is None:
            dur = float(args.slot_seconds)
        dur = max(0.10, float(dur))
        if st is None:
            st = cursor
        st = float(st)
        if ed is None:
            ed = min(total, st + dur)
        ed = float(ed)
        dur = max(0.10, ed - st)
        cursor = ed
        a = max(0, int(round(st * sr)))
        b = min(wav.size, int(round(ed * sr)))
        stats = audio_features_for_segment(wav[a:b], sr)
        probs = normalize_probs(s.get("music_semantic_probs", s.get("probs", {})), s.get("music_semantic_top_label", s.get("music_alignment_label")))
        top = s.get("music_semantic_top_label", s.get("music_alignment_label", slot_label_from_probs(probs)))
        if top not in SEMANTIC_LABELS:
            top = slot_label_from_probs(probs)
        s.update({
            "slot_id": int(s.get("slot_id", s.get("slot", i))),
            "start": float(st),
            "end": float(ed),
            "duration": float(dur),
            "target_frames": int(round(float(dur) * float(args.fps))),
            "music_alignment_label": str(s.get("music_alignment_label", top)),
            "music_semantic_top_label": str(top),
            "music_semantic_probs": probs,
            "preferred_dance_keys": list(s.get("preferred_dance_keys", s.get("slot_preferred_dance_keys", PREFERRED_DANCE_KEYS.get(str(top), [])))),
            "external_music_semantic_source": str(s.get("external_music_semantic_source", source_meta.get("fallback_semantic_json", ""))),
            "slot_plan_source": source_meta.get("slot_source", "unknown"),
            "feature": feature32(dur, probs, stats),
        })
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--router_ckpt", default=os.environ.get("V26_ROUTER_CKPT", os.environ.get("V21_ROUTER_CKPT", "")))
    ap.add_argument("--planner_ckpt", default=os.environ.get("V26_PLANNER_CKPT", ""))
    ap.add_argument("--v23_ckpt", default=os.environ.get("V26_V23_CKPT", ""))
    ap.add_argument("--index_json", default=os.environ.get("V26_INDEX_JSON", ""))
    ap.add_argument("--duration_index_npz", default=os.environ.get("V26_DURATION_INDEX_NPZ", ""))
    ap.add_argument("--hierarchy_index_npz", default=os.environ.get("V26_HIERARCHY_INDEX_NPZ", ""))
    ap.add_argument("--feature_dir", default=os.environ.get("V26_FEATURE_CACHE", ""))
    ap.add_argument("--music_semantic_dirs", nargs="*", default=[x for x in os.environ.get("V46_EXTERNAL_MUSIC_SEMANTIC_DIRS", "music_semantics:external_music_semantics:output/music_semantics").split(":") if x])
    ap.add_argument("--schedule_dir", default="")
    ap.add_argument("--slot_seconds", type=float, default=float(os.environ.get("V46_34_DEFAULT_SLOT_SECONDS", "4.0")))
    ap.add_argument("--fps", type=float, default=float(os.environ.get("V46_FPS", "30")))
    ap.add_argument("--max_seconds", type=float, default=float(os.environ.get("V46_34_MAX_SECONDS", "0")))
    ap.add_argument("--min_phrase_seconds", type=float, default=float(os.environ.get("V26_MIN_PHRASE_SECONDS", "2.5")))
    ap.add_argument("--max_phrase_seconds", type=float, default=float(os.environ.get("V26_MAX_PHRASE_SECONDS", "7.5")))
    ap.add_argument("--max_phrases", type=int, default=int(os.environ.get("V26_MAX_PHRASES", "160")))
    ap.add_argument("--allow_semantic_fallback", action="store_true", default=bool(int(os.environ.get("V46_34_ALLOW_SEMANTIC_FALLBACK", "0"))))
    args = ap.parse_args()

    out_json = Path(args.out_json)
    schedule_dir = Path(args.schedule_dir) if args.schedule_dir else out_json.parent / "v21_v26_schedule_raw"
    source_meta: Dict[str, Any] = {
        "slot_source": "v21_router_v26_planner",
        "router_ckpt": str(args.router_ckpt),
        "planner_ckpt": str(args.planner_ckpt),
        "v23_ckpt": str(args.v23_ckpt),
        "index_json": str(args.index_json),
        "duration_index_npz": str(args.duration_index_npz),
    }
    raw_slots: Optional[List[Dict[str, Any]]] = None
    schedule_json: Optional[Path] = None
    try:
        schedule_json = run_v26_scheduler(args, schedule_dir)
        if schedule_json:
            raw_slots = extract_slots_from_any_json(load_json(schedule_json))
            source_meta["raw_schedule_json"] = str(schedule_json)
    except Exception as exc:
        print(f"[V46.34 WARN] V21/V26 schedule failed: {exc}", file=sys.stderr)
        raw_slots = None
    if not raw_slots:
        if not args.allow_semantic_fallback:
            raise SystemExit(
                "[V46.34 ERROR] Could not obtain slots from trained V21/V26 router/planner. "
                "This run is intentionally strict because unseen-song slotting must use trained music semantics. "
                "Set V46_34_ALLOW_SEMANTIC_FALLBACK=1 only for debugging."
            )
        raw_slots, extra = build_fallback_slots(args)
        source_meta.update(extra)
        source_meta["slot_source"] = "external_music_semantic_fallback"
    slots = finalize_slots(raw_slots, args.audio, args, source_meta)
    obj = {
        "version": "v46_34_pretrained_router_slot_plan",
        "audio": str(args.audio),
        "slot_source": source_meta.get("slot_source"),
        "router_ckpt": str(args.router_ckpt),
        "planner_ckpt": str(args.planner_ckpt),
        "v23_ckpt": str(args.v23_ckpt),
        "raw_schedule_json": source_meta.get("raw_schedule_json", ""),
        "num_slots": len(slots),
        "fps": float(args.fps),
        "total_target_frames": int(sum(int(s.get("target_frames", 0)) for s in slots)),
        "slots": slots,
    }
    save_json(obj, out_json)
    print(json.dumps({
        "out_json": str(out_json),
        "slot_source": obj["slot_source"],
        "num_slots": obj["num_slots"],
        "total_target_frames": obj["total_target_frames"],
        "router_ckpt": obj["router_ckpt"],
        "planner_ckpt": obj["planner_ckpt"],
        "raw_schedule_json": obj.get("raw_schedule_json", ""),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
