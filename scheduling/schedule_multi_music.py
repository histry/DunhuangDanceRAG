#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V21 scalable query-time multi-music scheduler.

Key properties:
- one shared Dunhuang style-safe Event-RAG for any number of songs;
- phrase-level music queries, not per-frame accent majority;
- style-first retrieval with soft family-level MMR diversity;
- optional batch overlap penalty for paper comparison batches;
- optional learned music-motion router;
- optional DPN + endpoint transition refiner;
- no permanent one-song-one-database split.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model.music_motion_router import load_router_checkpoint
from model.transition_model import TRANSITION_LENGTHS
from scheduling.audio_features import extract_audio_features
from scheduling.index_io import load_shared_index, resolve_event_motion_path
from scheduling.music_event_calibration import build_phrase_query as calibrated_phrase_query
from scheduling.retrieval import precompute_music_similarity
from scheduling.transition_builder import (
    load_optional_transition,
    predict_transition_length as predict_transition_len,
    refine_transition,
)
from support.common import (
    CONTACT,
    ROOT_X,
    ROOT_Z,
    ROT,
    apply_start_anchor,
    event_compatibility,
    json_safe,
    load_motion,
    make_geodesic_transition,
    pad_or_trim_motion,
    resample_motion,
    transition_cost_from_arrays,
)


@dataclass
class MusicTask:
    key: str
    source: Path
    audio: Optional[Path]
    features: np.ndarray
    boundaries: List[int]
    queries: List[np.ndarray]
    events: List[str]


@dataclass
class ScheduleResult:
    task: MusicTask
    selected: List[int]
    transition_lengths: List[int]
    score: float
    slot_parts: List[Dict[str, Any]]
    motion: np.ndarray


def music_event_from_window(w: np.ndarray) -> str:
    if len(w) == 0:
        return "neutral_flow"
    arousal = float(w[:, 4].mean())
    delta = float(w[:, 5].mean())
    tension = float(w[:, 6].mean())
    calm = float(w[:, 7].mean())
    section = float(w[:, 10].max())
    accent = float(w[:, 11].mean())
    beat = float(w[:, 2].mean())
    novelty = float(w[:, 8].mean())
    if section > 0.72 or novelty > 0.75:
        return "section_change"
    if calm > 0.68 and tension < 0.50:
        return "calm_flow"
    if arousal > 0.74 and tension > 0.67:
        return "climax"
    if delta > 0.025 or (arousal > 0.62 and tension > 0.57):
        return "build_up"
    if delta < -0.025 and arousal < 0.60:
        return "release"
    if accent > 0.56 or (beat > 0.30 and arousal > 0.50):
        return "accent"
    return "neutral_flow"


def phrase_query(w: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, str]:
    if len(w) == 0:
        w = np.zeros((1, 12), dtype=np.float32)
    event = music_event_from_window(w)
    arousal = float(w[:, 4].mean())
    delta = float(w[:, 5].mean())
    tension = float(w[:, 6].mean())
    calm = float(w[:, 7].mean())
    section = float(w[:, 10].max())
    novelty = float(w[:, 8].mean())
    accent = float(w[:, 11].mean())
    beat = float(w[:, 2].mean())
    desired = {
        "accent": (0.88, 0.65, 0.30),
        "climax": (0.95, 0.78, 0.42),
        "section_change": (0.62, 0.72, 0.78),
        "build_up": (0.76, 0.70, 0.40),
        "release": (0.42, 0.52, 0.32),
        "calm_flow": (0.38, 0.55, 0.25),
        "neutral_flow": (0.55, 0.58, 0.40),
    }[event]
    upper, torso, lower = desired
    query = np.asarray(
        [
            np.clip(arousal, 0, 1),
            upper,
            torso,
            lower,
            np.clip(tension, 0, 1),
            np.clip(calm, 0, 1),
            np.clip(max(section, 0.5 * beat), 0, 1),
            np.clip(max(delta, 0.0) * 8.0, 0, 1),
            np.clip(max(-delta, 0.0) * 8.0, 0, 1),
            np.clip(accent, 0, 1),
            np.clip(novelty, 0, 1),
            np.clip((end - start) / 60.0, 0, 1),
        ],
        dtype=np.float32,
    )
    return query, event


def choose_boundaries(features: np.ndarray, phrase_count: int) -> List[int]:
    t = len(features)
    if phrase_count <= 1:
        return [0, t]
    score = 0.55 * features[:, 10] + 0.25 * features[:, 8] + 0.20 * features[:, 2]
    boundaries = [0]
    min_gap = max(20, int(round(t / phrase_count * 0.55)))
    for k in range(1, phrase_count):
        target = int(round(k * t / phrase_count))
        radius = max(8, int(round(t / phrase_count * 0.30)))
        lo = max(boundaries[-1] + min_gap, target - radius)
        hi = min(t - (phrase_count - k) * min_gap, target + radius)
        if hi <= lo:
            boundary = target
        else:
            local = score[lo : hi + 1]
            boundary = lo + int(np.argmax(local))
        boundaries.append(int(np.clip(boundary, boundaries[-1] + 1, t - 1)))
    boundaries.append(t)
    # Repair monotonicity if novelty peaks were too close.
    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1] + 1)
    boundaries[-1] = t
    return boundaries


def load_music_task(path: Path, phrase_count: int, num_frames: int, feature_dir: Path) -> MusicTask:
    feature_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        features = np.load(path).astype(np.float32)
        audio = None
        key = path.stem.replace("_v21_music", "")
    else:
        key = path.stem
        cache = feature_dir / f"{key}_v21_music.npy"
        if cache.is_file():
            features = np.load(cache).astype(np.float32)
        else:
            features, meta = extract_audio_features(path, num_frames=num_frames)
            np.save(cache, features.astype(np.float32))
            (feature_dir / f"{key}_v21_music.json").write_text(
                json.dumps({"audio": str(path), "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        audio = path
    if features.ndim != 2 or features.shape[1] < 12:
        raise ValueError(f"{path}: expected [T,12], got {features.shape}")
    if len(features) != num_frames:
        old = np.linspace(0.0, 1.0, len(features), dtype=np.float32)
        new = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
        resized = np.empty((num_frames, features.shape[1]), dtype=np.float32)
        for d in range(features.shape[1]):
            resized[:, d] = np.interp(new, old, features[:, d])
        features = resized
    boundaries = choose_boundaries(features, phrase_count)
    queries: List[np.ndarray] = []
    events: List[str] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q, event = calibrated_phrase_query(features[start:end], start, end)
        queries.append(q)
        events.append(event)
    return MusicTask(key=key, source=path, audio=audio, features=features, boundaries=boundaries, queries=queries, events=events)


def load_optional_router(path: str, device: torch.device):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    return load_router_checkpoint(p, device=device)


def schedule_one(
    task: MusicTask,
    items: List[Dict[str, Any]],
    arrays,
    motions: List[np.ndarray],
    router,
    transition_bundle,
    args,
    batch_event_usage: Counter,
    batch_family_usage: Counter,
    batch_selected_indices: Sequence[int],
    device: torch.device,
) -> ScheduleResult:
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    mmr_embed = np.asarray(arrays["mmr_embed"], dtype=np.float32)
    entry_pose = np.asarray(arrays["entry_pose"], dtype=np.float32)
    exit_pose = np.asarray(arrays["exit_pose"], dtype=np.float32)
    entry_vel = np.asarray(arrays["entry_vel"], dtype=np.float32)
    exit_vel = np.asarray(arrays["exit_vel"], dtype=np.float32)
    lengths = np.asarray(arrays["length"], dtype=np.int32)
    style = np.asarray(arrays["style_score"], dtype=np.float32)
    quality = np.asarray(arrays["quality_score"], dtype=np.float32)
    safety = np.asarray(arrays["safety_score"], dtype=np.float32)
    similarities = precompute_music_similarity(router, task.queries, motion_desc, device)
    if batch_selected_indices:
        unique_batch = np.asarray(sorted(set(int(x) for x in batch_selected_indices)), dtype=np.int64)
        batch_mmr_vector = np.max(mmr_embed @ mmr_embed[unique_batch].T, axis=1).astype(np.float32)
    else:
        batch_mmr_vector = np.zeros((len(items),), dtype=np.float32)
    event_types = [str(x.get("event_type", "neutral_flow")) for x in items]
    families = [str(x.get("family_id", "")) for x in items]

    # Beam tuple: score, selected indices, transition lengths, frame_est, slot_parts
    beam: List[Tuple[float, List[int], List[int], int, List[Dict[str, Any]]]] = [(0.0, [], [], 0, [])]

    for slot, (query, music_event) in enumerate(zip(task.queries, task.events)):
        slot_len = task.boundaries[slot + 1] - task.boundaries[slot]
        compat = np.asarray([event_compatibility(music_event, e) for e in event_types], dtype=np.float32)
        duration_match = 1.0 - np.minimum(np.abs(lengths - slot_len) / max(slot_len, 1), 1.0)
        activity_match = 1.0 - np.abs(motion_desc[:, 0] - query[0])
        base = (
            args.style_weight * style
            + args.quality_weight * quality
            + args.safety_weight * safety
            + args.music_weight * similarities[slot]
            + args.event_weight * compat
            + args.activity_weight * activity_match
            + args.duration_weight * duration_match
        )
        shortlist = np.argsort(base)[::-1][: min(args.candidate_top_k, len(items))]
        new_beam: List[Tuple[float, List[int], List[int], int, List[Dict[str, Any]]]] = []

        for state_score, selected, trans_lengths, frame_est, parts in beam:
            for idx in shortlist:
                idx = int(idx)
                if idx in selected:
                    continue
                family = families[idx]
                same_family = sum(1 for s in selected if families[s] == family)
                same_source = sum(1 for s in selected if int(items[s].get("source_id", -1)) == int(items[idx].get("source_id", -2)))
                if same_family > 0 and args.hard_family_unique:
                    continue

                transition_len = 0
                trans_cost = 0.0
                if selected:
                    prev = selected[-1]
                    trans_cost = transition_cost_from_arrays(
                        exit_pose[prev], exit_vel[prev], entry_pose[idx], entry_vel[idx]
                    )
                    transition_len = predict_transition_len(
                        transition_bundle,
                        motions[prev],
                        motions[idx],
                        query,
                        music_event,
                        event_types[idx],
                        device,
                    )

                mmr_penalty = 0.0
                if selected:
                    mmr_penalty = max(float(mmr_embed[idx] @ mmr_embed[s]) for s in selected)
                batch_penalty = (
                    args.batch_overlap_weight * math.log1p(batch_event_usage.get(str(items[idx].get("event_id", idx)), 0))
                    + args.batch_family_overlap_weight * math.log1p(batch_family_usage.get(family, 0))
                    + args.batch_mmr_weight * float(batch_mmr_vector[idx])
                )
                repeat_penalty = args.family_repeat_weight * same_family + args.source_repeat_weight * same_source
                content_len = max(8, slot_len - transition_len)
                ratio = content_len / max(int(lengths[idx]), 1)
                warp_penalty = abs(math.log(max(ratio, 1e-6)))
                if ratio < args.min_time_warp or ratio > args.max_time_warp:
                    continue

                score = (
                    float(base[idx])
                    - args.transition_weight * trans_cost
                    - args.mmr_weight * mmr_penalty
                    - repeat_penalty
                    - batch_penalty
                    - args.time_warp_weight * warp_penalty
                )
                slot_part = {
                    "slot": slot,
                    "slot_start": int(task.boundaries[slot]),
                    "slot_end": int(task.boundaries[slot + 1]),
                    "slot_length": int(slot_len),
                    "music_event": music_event,
                    "query": query.tolist(),
                    "event_index": idx,
                    "event_id": str(items[idx].get("event_id", idx)),
                    "family_id": family,
                    "motion_event": event_types[idx],
                    "transition_len": int(transition_len),
                    "content_len": int(content_len),
                    "style": float(style[idx]),
                    "quality": float(quality[idx]),
                    "safety": float(safety[idx]),
                    "music_similarity": float(similarities[slot, idx]),
                    "event_compatibility": float(compat[idx]),
                    "activity_match": float(activity_match[idx]),
                    "duration_match": float(duration_match[idx]),
                    "transition_cost": float(trans_cost),
                    "mmr_penalty": float(mmr_penalty),
                    "batch_penalty": float(batch_penalty),
                    "time_warp_ratio": float(ratio),
                    "slot_score": float(score),
                }
                new_beam.append(
                    (
                        state_score + score,
                        selected + [idx],
                        trans_lengths + [transition_len],
                        frame_est + slot_len,
                        parts + [slot_part],
                    )
                )

        if not new_beam:
            raise RuntimeError(
                f"No V21 candidate for music={task.key} slot={slot}. "
                "Increase candidate_top_k or relax min/max_time_warp."
            )
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[: args.beam_size]

    best_score, selected, trans_lengths, _, parts = beam[0]
    pieces: List[np.ndarray] = []
    previous_content: Optional[np.ndarray] = None
    for slot, idx in enumerate(selected):
        slot_len = task.boundaries[slot + 1] - task.boundaries[slot]
        k = int(trans_lengths[slot]) if slot > 0 else 0
        content_len = max(8, slot_len - k)
        content = resample_motion(motions[idx], content_len)
        content[:, ROOT_X] = 0.0
        content[:, ROOT_Z] = 0.0
        if previous_content is not None and k > 0:
            rough = make_geodesic_transition(previous_content, content, k)
            transition = refine_transition(
                transition_bundle,
                rough,
                previous_content[-1],
                content[0],
                task.queries[slot],
                device,
            )
            pieces.append(transition)
        pieces.append(content)
        previous_content = content

    motion = np.concatenate(pieces, axis=0).astype(np.float32)
    motion = pad_or_trim_motion(motion, args.num_frames)
    motion[:, ROOT_X] = 0.0
    motion[:, ROOT_Z] = 0.0
    if args.start_pose:
        start_path = Path(args.start_pose)
        if start_path.is_file():
            start_pose = np.load(start_path).astype(np.float32).reshape(-1)
            motion = apply_start_anchor(motion, start_pose, args.start_anchor_blend)

    return ScheduleResult(
        task=task,
        selected=selected,
        transition_lengths=trans_lengths,
        score=float(best_score),
        slot_parts=parts,
        motion=motion,
    )


def usage_from_results(results: Dict[str, ScheduleResult], exclude_key: str = "") -> Tuple[Counter, Counter, List[int]]:
    events: Counter = Counter()
    families: Counter = Counter()
    selected_indices: List[int] = []
    for key, result in results.items():
        if key == exclude_key:
            continue
        selected_indices.extend(int(x) for x in result.selected)
        for part in result.slot_parts:
            events[str(part["event_id"])] += 1
            families[str(part["family_id"])] += 1
    return events, families, selected_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--index_npz", required=True)
    ap.add_argument("--music", action="append", default=[], help="Repeat for .wav or cached [T,12] .npy")
    ap.add_argument("--music_glob", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--feature_dir", default="")
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--phrase_count", type=int, default=3)
    ap.add_argument("--beam_size", type=int, default=48)
    ap.add_argument("--candidate_top_k", type=int, default=1800)
    ap.add_argument("--router_ckpt", default="")
    ap.add_argument("--transition_ckpt", default="")
    ap.add_argument("--start_pose", default="")
    ap.add_argument("--start_anchor_blend", type=int, default=8)
    ap.add_argument("--refine_rounds", type=int, default=2)

    ap.add_argument("--style_weight", type=float, default=1.35)
    ap.add_argument("--quality_weight", type=float, default=0.65)
    ap.add_argument("--safety_weight", type=float, default=0.35)
    ap.add_argument("--music_weight", type=float, default=0.80)
    ap.add_argument("--event_weight", type=float, default=0.65)
    ap.add_argument("--activity_weight", type=float, default=0.25)
    ap.add_argument("--duration_weight", type=float, default=0.25)
    ap.add_argument("--transition_weight", type=float, default=0.55)
    ap.add_argument("--mmr_weight", type=float, default=0.38)
    ap.add_argument("--family_repeat_weight", type=float, default=0.55)
    ap.add_argument("--source_repeat_weight", type=float, default=0.15)
    ap.add_argument("--batch_overlap_weight", type=float, default=0.18)
    ap.add_argument("--batch_family_overlap_weight", type=float, default=0.20)
    ap.add_argument("--batch_mmr_weight", type=float, default=0.18)
    ap.add_argument("--time_warp_weight", type=float, default=0.25)
    ap.add_argument("--min_time_warp", type=float, default=0.65)
    ap.add_argument("--max_time_warp", type=float, default=1.45)
    ap.add_argument("--hard_family_unique", action="store_true")
    args = ap.parse_args()

    paths = [Path(x) for x in args.music]
    if args.music_glob:
        paths.extend(Path(x) for x in sorted(glob.glob(args.music_glob)))
    # Deduplicate while keeping stable order.
    seen = set()
    unique_paths = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)
    paths = unique_paths
    if not paths:
        raise RuntimeError("Provide at least one --music or --music_glob")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(args.feature_dir) if args.feature_dir else out_dir / "music_features"
    index_json = Path(args.index_json).resolve()
    meta, arrays, items = load_shared_index(index_json, Path(args.index_npz))
    motions: List[np.ndarray] = []
    for item in items:
        p = resolve_event_motion_path(item, index_json, metadata=meta)
        motions.append(load_motion(p))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = load_optional_router(args.router_ckpt, device)
    transition_bundle = load_optional_transition(args.transition_ckpt, device)
    tasks = [load_music_task(p, args.phrase_count, args.num_frames, feature_dir) for p in paths]

    results: Dict[str, ScheduleResult] = {}
    for task in tasks:
        results[task.key] = schedule_one(
            task, items, arrays, motions, router, transition_bundle, args, Counter(), Counter(), [], device
        )

    # Synchronous soft cross-music diversification. For deployment with one song,
    # usage counters are empty and this has no effect. With many songs it scales
    # without rebuilding the database.
    for round_idx in range(max(0, args.refine_rounds)):
        # Coordinate descent avoids the symmetric oscillation of synchronous
        # updates, where multiple songs can jump to the same alternative unit.
        working = dict(results)
        if tasks:
            shift = round_idx % len(tasks)
            ordered_tasks = tasks[shift:] + tasks[:shift]
        else:
            ordered_tasks = tasks
        for task in ordered_tasks:
            event_usage, family_usage, batch_indices = usage_from_results(working, exclude_key=task.key)
            working[task.key] = schedule_one(
                task,
                items,
                arrays,
                motions,
                router,
                transition_bundle,
                args,
                event_usage,
                family_usage,
                batch_indices,
                device,
            )
        results = working
        print(f"[V21] coordinate refinement round {round_idx + 1}/{args.refine_rounds} complete", flush=True)

    summary = {
        "version": "v21_query_time_multi_music_router",
        "index_json": str(args.index_json),
        "index_npz": str(args.index_npz),
        "router_ckpt": str(args.router_ckpt),
        "transition_ckpt": str(args.transition_ckpt),
        "num_music": len(tasks),
        "weights": {k: v for k, v in vars(args).items() if k.endswith("_weight")},
        "results": {},
    }

    for task in tasks:
        result = results[task.key]
        out_npy = out_dir / f"{task.key}_v21.npy"
        np.save(out_npy, result.motion[None].astype(np.float32))
        report = {
            "music": task.key,
            "source": str(task.source),
            "audio": str(task.audio) if task.audio else "",
            "boundaries": task.boundaries,
            "music_events": task.events,
            "score": result.score,
            "out_npy": str(out_npy),
            "schedule": [],
        }
        for idx, part in zip(result.selected, result.slot_parts):
            item = dict(items[idx])
            item["v21_slot"] = part
            report["schedule"].append(item)
        report_path = out_dir / f"{task.key}_v21.schedule_report.json"
        report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
        summary["results"][task.key] = {
            "npy": str(out_npy),
            "report": str(report_path),
            "score": result.score,
            "event_ids": [part["event_id"] for part in result.slot_parts],
            "families": [part["family_id"] for part in result.slot_parts],
            "music_events": task.events,
            "motion_events": [part["motion_event"] for part in result.slot_parts],
        }
        print(f"[SAVED] {task.key}: {out_npy}")

    keys = list(summary["results"])
    overlaps = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ae = set(summary["results"][a]["event_ids"])
            be = set(summary["results"][b]["event_ids"])
            af = set(summary["results"][a]["families"])
            bf = set(summary["results"][b]["families"])
            overlaps.append(
                {
                    "a": a,
                    "b": b,
                    "event_overlap": len(ae & be),
                    "family_overlap": len(af & bf),
                }
            )
    summary["pairwise_overlap"] = overlaps
    (out_dir / "V21_MULTI_MUSIC_SUMMARY.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("summary:", out_dir / "V21_MULTI_MUSIC_SUMMARY.json")
    for row in overlaps:
        print(row)


if __name__ == "__main__":
    main()
