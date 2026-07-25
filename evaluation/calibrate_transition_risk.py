#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate pairwise transition-risk thresholds without paired music labels.

The calibration samples event-to-event transitions from a validation Event-DB,
computes the physical multiscale boundary report, and estimates category-aware
low/high thresholds from score quantiles.  It does not force a predetermined
high-risk percentage; the quantiles are initialization values that remain
subject to hard physical rejection contracts.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from contracts.boundary import transition_multiscale_risk
from routing.transition_repair_policy import normalize_pairwise_score


def _array(db: Mapping[str, Any], names: Sequence[str], default: Any = None):
    for name in names:
        if name in db:
            return np.asarray(db[name])
    return default


def _load_motion(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            for key in ("motion", "motions", "data", "arr_0"):
                if key in value.files:
                    return np.asarray(value[key], dtype=np.float32)
        finally:
            value.close()
        raise RuntimeError(f"No motion array found in {path}")
    return np.asarray(value, dtype=np.float32)


def _event_motion_loader(db: Mapping[str, Any], db_path: Path):
    embedded = _array(db, ("motions", "event_motions", "motion_clips"))
    paths = _array(db, ("paths", "motion_paths", "source_paths"))
    starts = _array(db, ("start_frames", "starts", "event_start"))
    ends = _array(db, ("end_frames", "ends", "event_end"))
    cache: Dict[str, np.ndarray] = {}

    def load(index: int) -> np.ndarray:
        if embedded is not None:
            clip = np.asarray(embedded[index], dtype=np.float32)
            if clip.ndim == 3 and clip.shape[0] == 1:
                clip = clip[0]
            return clip
        if paths is None:
            raise RuntimeError(
                "Event-DB has neither embedded motions nor source paths"
            )
        path = Path(str(paths[index])).expanduser()
        if not path.is_absolute():
            path = (db_path.parent / path).resolve()
        else:
            path = path.resolve()
        key = str(path)
        if key not in cache:
            cache[key] = _load_motion(path)
        motion = cache[key]
        if starts is not None and ends is not None:
            start = int(starts[index])
            end = int(ends[index])
            return np.asarray(motion[start:end], dtype=np.float32)
        return np.asarray(motion, dtype=np.float32)

    return load


def _transition_category(left: str, right: str) -> str:
    def coarse(label: str) -> str:
        text = str(label)
        if text in {"calm_meditative", "pose_hold"}:
            return "pose"
        if text in {"turning_climax", "aerial_curve"}:
            return "turn"
        if text in {"lyrical_flow", "footwork_flow", "instrument_phrase"}:
            return "flow"
        if text == "percussive_accent":
            return "accent"
        return "other"

    return f"{coarse(left)}_to_{coarse(right)}"


def calibrate(
    db_path: Path,
    out_path: Path,
    *,
    samples_per_category: int = 250,
    seed: int = 20260724,
    fps: float = 30.0,
    low_quantile: float = 0.60,
    high_quantile: float = 0.90,
) -> Dict[str, Any]:
    if not 0.0 < float(low_quantile) < float(high_quantile) < 1.0:
        raise ValueError("Risk quantiles must satisfy 0 < low < high < 1")
    if int(samples_per_category) < 1:
        raise ValueError("samples_per_category must be positive")
    with np.load(db_path, allow_pickle=True) as data:
        db = {key: data[key] for key in data.files}
    labels = _array(
        db,
        ("aesd_event_semantics", "music_alignment_labels", "event_families"),
    )
    if labels is None:
        raise RuntimeError("Event-DB has no semantic labels for stratification")
    source = _array(db, ("source_uids", "source_groups"), np.asarray(["unknown"] * len(labels)))
    loader = _event_motion_loader(db, db_path)
    rng = np.random.default_rng(int(seed))
    count = len(labels)
    semantic_groups: Dict[str, np.ndarray] = {}
    for semantic in sorted({str(value) for value in labels}):
        coarse = _transition_category(semantic, semantic).split("_to_", 1)[0]
        semantic_groups.setdefault(coarse, np.asarray([], dtype=np.int64))
        semantic_groups[coarse] = np.concatenate(
            [semantic_groups[coarse], np.flatnonzero(np.asarray(labels, dtype=object) == semantic)]
        )
    by_category: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for left_group, left_indices in sorted(semantic_groups.items()):
        for right_group, right_indices in sorted(semantic_groups.items()):
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            category = f"{left_group}_to_{right_group}"
            maximum_attempts = max(int(samples_per_category) * 20, 200)
            seen = set()
            for _ in range(maximum_attempts):
                left = int(rng.choice(left_indices))
                right = int(rng.choice(right_indices))
                if left == right or (left, right) in seen:
                    continue
                seen.add((left, right))
                by_category[category].append((left, right))
                if len(by_category[category]) >= int(samples_per_category):
                    break

    category_reports: Dict[str, Any] = {}
    all_scores: List[float] = []
    hard_rejects = 0
    evaluated = 0
    for category, pairs in sorted(by_category.items()):
        rows = []
        for left, right in pairs:
            try:
                report = transition_multiscale_risk(
                    loader(left),
                    np.zeros((0, 151), dtype=np.float32),
                    loader(right),
                    fps=float(fps),
                )
            except Exception as exc:
                rows.append({"left": left, "right": right, "error": str(exc)})
                continue
            normalized = normalize_pairwise_score(report)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "left_source": str(source[left]),
                    "right_source": str(source[right]),
                    "same_source": bool(str(source[left]) == str(source[right])),
                    "normalized_score": normalized,
                    "hard_reject": bool(report.get("hard_reject", False)),
                    "physical_report": report,
                }
            )
            all_scores.append(normalized)
            hard_rejects += int(bool(report.get("hard_reject", False)))
            evaluated += 1
        valid = np.asarray(
            [row["normalized_score"] for row in rows if "normalized_score" in row],
            dtype=np.float64,
        )
        if len(valid):
            low = float(np.quantile(valid, float(low_quantile)))
            high = float(np.quantile(valid, float(high_quantile)))
        else:
            low = 0.35
            high = 0.70
        if high - low < 0.05:
            center = 0.5 * (low + high)
            low = float(np.clip(center - 0.025, 0.01, 0.94))
            high = float(np.clip(center + 0.025, low + 0.01, 0.99))
        category_reports[category] = {
            "samples": int(len(valid)),
            "low_threshold": low,
            "high_threshold": high,
            "mean": float(valid.mean()) if len(valid) else 0.0,
            "p95": float(np.quantile(valid, 0.95)) if len(valid) else 0.0,
            "rows": rows,
        }
    scores = np.asarray(all_scores, dtype=np.float64)
    global_low = (
        float(np.quantile(scores, float(low_quantile))) if len(scores) else 0.35
    )
    global_high = (
        float(np.quantile(scores, float(high_quantile))) if len(scores) else 0.70
    )
    if global_high - global_low < 0.05:
        center = 0.5 * (global_low + global_high)
        global_low = float(np.clip(center - 0.025, 0.01, 0.94))
        global_high = float(np.clip(center + 0.025, global_low + 0.01, 0.99))
    result = {
        "schema": "dunhuang_transition_risk_calibration_v1",
        "event_db": str(db_path.resolve()),
        "fps": float(fps),
        "quantiles": {
            "low": float(low_quantile),
            "high": float(high_quantile),
        },
        "global_thresholds": {
            "low": global_low,
            "high": global_high,
        },
        "categories": category_reports,
        "evaluated_pairs": int(evaluated),
        "hard_reject_rate": hard_rejects / max(evaluated, 1),
        "human_pair_labels_required": False,
        "thresholds_are_initialization_not_forced_class_ratios": True,
        "ok": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples_per_category", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--low_quantile", type=float, default=0.60)
    parser.add_argument("--high_quantile", type=float, default=0.90)
    args = parser.parse_args(argv)
    report = calibrate(
        Path(args.db),
        Path(args.out),
        samples_per_category=int(args.samples_per_category),
        seed=int(args.seed),
        fps=float(args.fps),
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
