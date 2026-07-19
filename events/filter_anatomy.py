#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Posture-aware event-level Anatomy gate for V46.53.1.

Only catastrophic source failures are removed before slicing. This module performs
fine-grained event filtering after slicing, preserving valid cultural poses while
rejecting local collapse, severe rotation excess, collision and floor failures.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from contracts.anatomy import env_float, env_int, event_anatomy_features

SCHEMA = "v46_53_1_posture_aware_event_anatomy_filter"


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    return x


def _load_event(path: str) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1: x = x[0]
    if x.ndim != 2 or x.shape[1] < 151:
        raise ValueError(f"Invalid event motion {path}: {x.shape}")
    return x[:, :151]


def _split_name(path: Path) -> str:
    tokens = {p.lower() for p in path.parts}
    if "train" in tokens: return "train"
    if "val" in tokens or "validation" in tokens: return "val"
    if "test" in tokens: return "test"
    return "unknown"


def _quality_threshold(posture: str) -> float:
    defaults = {
        "standing": 0.46,
        "aerial": 0.46,
        "half_squat": 0.44,
        "deep_squat": 0.41,
        "kneeling": 0.38,
        "floor_pose": 0.38,
    }
    key = str(posture)
    env_key = "V46_52_EVENT_QUALITY_MIN_" + key.upper()
    return env_float(env_key, defaults.get(key, 0.44))


def _source_ids(payload: Dict[str, Any], paths: np.ndarray, n: int) -> np.ndarray:
    for key in ("source_uids", "source_uid", "sources", "source_names"):
        if key in payload:
            a = np.asarray(payload[key], dtype=object)
            if a.ndim >= 1 and len(a) == n:
                return a.reshape(n)
    return np.asarray([Path(str(p)).parent.name or Path(str(p)).stem for p in paths], dtype=object)


def filter_database(db_path: Path, meta_path: Path, audit_path: Path) -> Dict[str, Any]:
    obj = np.load(db_path, allow_pickle=True)
    payload: Dict[str, Any] = {k: obj[k] for k in obj.files}
    paths = np.asarray(payload["paths"], dtype=object)
    n = len(paths)
    source_ids = _source_ids(payload, paths, n)

    meta: List[Dict[str, Any]] = []
    if meta_path.is_file():
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(raw, list): meta = [dict(x) for x in raw]
    if len(meta) != n: meta = [dict() for _ in range(n)]

    features: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    keep = np.zeros(n, dtype=bool)
    hard_valid = np.zeros(n, dtype=bool)
    min_frames = env_int("V46_52_EVENT_MIN_FRAMES", 18)
    rescue_floor = env_float("V46_52_EVENT_RESCUE_QUALITY_FLOOR", 0.30)

    for i, path in enumerate(paths.tolist()):
        try:
            motion = _load_event(str(path))
            feat = event_anatomy_features(motion)
            posture = str(feat.get("posture_mode", "standing"))
            threshold = _quality_threshold(posture)
            hard = bool(feat.get("anatomy_hard_valid", feat.get("anatomy_valid", False))) and len(motion) >= min_frames
            valid = hard and float(feat["anatomy_quality"]) >= threshold
            hard_valid[i] = hard
            keep[i] = valid
            feat.update({
                "event_index_before_filter": int(i),
                "event_path": str(path),
                "source_uid": str(source_ids[i]),
                "frames": int(len(motion)),
                "posture_quality_threshold": float(threshold),
                "selection_mode": "normal" if valid else "rejected",
                "kept": bool(valid),
            })
            features.append(feat)
            meta[i].update({k: v for k, v in feat.items() if k not in {"anatomy_reasons", "anatomy_soft_reasons"}})
        except Exception as exc:
            feat = {
                "event_index_before_filter": int(i), "event_path": str(path), "source_uid": str(source_ids[i]),
                "anatomy_valid": False, "anatomy_hard_valid": False, "anatomy_quality": 0.0,
                "anatomy_reasons": [str(exc)], "selection_mode": "load_failure", "kept": False,
            }
            features.append(feat); failures.append(feat)

    # Preserve minimum safe representation from every accepted training source.
    # Rescue is quality-based and can never override a hard anatomy failure.
    split = _split_name(db_path)
    min_per_source = env_int(
        "V46_52_EVENT_MIN_PER_SOURCE_TRAIN" if split == "train" else "V46_52_EVENT_MIN_PER_SOURCE_EVAL",
        4 if split == "train" else 1,
    )
    rescued: List[int] = []
    by_source: Dict[str, List[int]] = defaultdict(list)
    for i, src in enumerate(source_ids.tolist()): by_source[str(src)].append(i)
    for src, indices in by_source.items():
        current = [i for i in indices if keep[i]]
        need = max(0, min_per_source - len(current))
        if need <= 0: continue
        candidates = [
            i for i in indices
            if not keep[i] and hard_valid[i] and float(features[i].get("anatomy_quality", 0.0)) >= rescue_floor
        ]
        candidates.sort(key=lambda i: (float(features[i].get("anatomy_quality", 0.0)), -i), reverse=True)
        for i in candidates[:need]:
            keep[i] = True
            features[i]["kept"] = True
            features[i]["selection_mode"] = "safe_source_coverage_rescue"
            meta[i]["kept"] = True
            meta[i]["selection_mode"] = "safe_source_coverage_rescue"
            rescued.append(i)

    kept = np.where(keep)[0]
    min_events = env_int(
        "V46_52_EVENT_DB_MIN_EVENTS_TRAIN" if split == "train" else "V46_52_EVENT_DB_MIN_EVENTS_EVAL",
        64 if split == "train" else 12,
    )
    min_ratio = env_float("V46_52_EVENT_DB_MIN_KEEP_RATIO", 0.35)
    if len(kept) < min_events or len(kept) / max(1, n) < min_ratio:
        raise RuntimeError(
            f"V46.53.1 event gate retained {len(kept)}/{n} for split={split}; "
            f"requires >= {min_events} and ratio >= {min_ratio:.3f}"
        )

    out: Dict[str, Any] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        out[key] = arr[keep] if arr.ndim >= 1 and arr.shape[0] == n else value

    kept_features = [features[i] for i in kept]
    out.update({
        "anatomy_contract_schema_version": np.asarray(SCHEMA, dtype=object),
        "anatomy_valid": np.ones(len(kept), dtype=np.bool_),
        "anatomy_hard_valid": np.ones(len(kept), dtype=np.bool_),
        "anatomy_soft_valid": np.asarray([bool(f.get("anatomy_soft_valid", True)) for f in kept_features], dtype=np.bool_),
        "anatomy_quality": np.asarray([f["anatomy_quality"] for f in kept_features], dtype=np.float32),
        "posture_quality_threshold": np.asarray([f["posture_quality_threshold"] for f in kept_features], dtype=np.float32),
        "anatomy_selection_mode": np.asarray([f["selection_mode"] for f in kept_features], dtype=object),
        "posture_entry": np.asarray([f["posture_entry"] for f in kept_features], dtype=object),
        "posture_exit": np.asarray([f["posture_exit"] for f in kept_features], dtype=object),
        "posture_mode": np.asarray([f["posture_mode"] for f in kept_features], dtype=object),
        "pelvis_height_entry_norm": np.asarray([f["pelvis_height_entry_norm"] for f in kept_features], dtype=np.float32),
        "pelvis_height_exit_norm": np.asarray([f["pelvis_height_exit_norm"] for f in kept_features], dtype=np.float32),
        "pelvis_height_median_norm": np.asarray([f["pelvis_height_median_norm"] for f in kept_features], dtype=np.float32),
        "body_height_entry_norm": np.asarray([f["body_height_entry_norm"] for f in kept_features], dtype=np.float32),
        "body_height_exit_norm": np.asarray([f["body_height_exit_norm"] for f in kept_features], dtype=np.float32),
        "body_height_median_norm": np.asarray([f["body_height_median_norm"] for f in kept_features], dtype=np.float32),
        "entry_floor_offset_m": np.asarray([f["entry_floor_offset_m"] for f in kept_features], dtype=np.float32),
        "exit_floor_offset_m": np.asarray([f["exit_floor_offset_m"] for f in kept_features], dtype=np.float32),
        "torso_compression_ratio_p05": np.asarray([f["torso_compression_ratio_p05"] for f in kept_features], dtype=np.float32),
        "local_angle_violation_ratio": np.asarray([f["local_angle_violation_ratio"] for f in kept_features], dtype=np.float32),
        "raw_local_angle_violation_ratio": np.asarray([f.get("raw_local_angle_violation_ratio", f["local_angle_violation_ratio"]) for f in kept_features], dtype=np.float32),
        "local_angle_severe_ratio": np.asarray([f.get("local_angle_severe_ratio", 0.0) for f in kept_features], dtype=np.float32),
        "self_collision_severe_ratio": np.asarray([f["self_collision_severe_ratio"] for f in kept_features], dtype=np.float32),
        "spine_cumulative_angle_p95_deg": np.asarray([f["spine_cumulative_angle_p95_deg"] for f in kept_features], dtype=np.float32),
        "event_index_before_anatomy_filter": kept.astype(np.int32),
    })

    if "desc" in out:
        desc = np.asarray(out["desc"], dtype=np.float32)
        mean = desc.mean(axis=0, keepdims=True)
        std = desc.std(axis=0, keepdims=True) + 1e-6
        out["desc_mean"] = mean.astype(np.float32)
        out["desc_std"] = std.astype(np.float32)
        out["desc_z"] = ((desc - mean) / std).astype(np.float32)

    backup = db_path.with_name(db_path.stem + ".pre_v46_53_1_anatomy.npz")
    if not backup.exists(): shutil.copy2(db_path, backup)
    np.savez_compressed(db_path, **out)
    kept_meta = [meta[i] for i in kept]
    meta_path.write_text(json.dumps(_jsonable(kept_meta), ensure_ascii=False, indent=2), encoding="utf-8")

    rejected = [features[i] for i in np.where(~keep)[0]]
    audit = {
        "schema": SCHEMA,
        "db": str(db_path), "backup": str(backup), "split": split,
        "events_before": int(n), "events_after": int(len(kept)),
        "keep_ratio": float(len(kept) / max(1, n)),
        "quality_min": float(np.min(out["anatomy_quality"])),
        "quality_median": float(np.median(out["anatomy_quality"])),
        "rescued_safe_events": [int(i) for i in rescued],
        "rescue_policy": "hard-valid only; never bypass catastrophic anatomy failures",
        "posture_distribution": {str(k): int(v) for k, v in zip(*np.unique(out["posture_mode"], return_counts=True))},
        "selection_mode_distribution": dict(Counter(out["anatomy_selection_mode"].tolist())),
        "source_event_counts": dict(Counter(str(s) for s in np.asarray(out.get("source_uids", source_ids[keep]), dtype=object).tolist())),
        "rejected": rejected, "load_failures": failures, "ok": True,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(_jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--audit", default=None)
    args = ap.parse_args(argv)
    db = Path(args.db)
    meta = Path(args.meta) if args.meta else db.with_name("events_meta.json")
    audit = Path(args.audit) if args.audit else db.with_name("events.v46_52_anatomy.audit.json")
    result = filter_database(db, meta, audit)
    print(json.dumps({k: result[k] for k in ("events_before", "events_after", "keep_ratio", "quality_min", "ok")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
