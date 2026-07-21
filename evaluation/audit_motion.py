#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit gravity + anatomy for one motion or a directory of motions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from contracts.gravity import GravityThresholds, evaluate_gravity_contract, gravity_metrics_np
from contracts.anatomy import AnatomyThresholds, anatomy_metrics_np, evaluate_anatomy_contract, env_float


def _audit(path: Path, fps: float) -> Dict[str, Any]:
    x = np.load(path, allow_pickle=True)
    if np.asarray(x).ndim == 3:
        x = np.asarray(x)[0]
    gravity = gravity_metrics_np(x, fps)
    gth = GravityThresholds(
        torso_up_cos_p05_min=env_float("V46_52_GRAVITY_TORSO_P05_MIN", 0.55),
        torso_up_cos_median_min=env_float("V46_52_GRAVITY_TORSO_MEDIAN_MIN", 0.76),
        head_above_pelvis_ratio_min=env_float("V46_52_HEAD_ABOVE_RATIO_MIN", 0.97),
        feet_below_pelvis_ratio_min=env_float("V46_52_FEET_BELOW_RATIO_MIN", 0.94),
        horizontal_body_ratio_max=env_float("V46_52_HORIZONTAL_BODY_RATIO_MAX", 0.04),
    )
    gok, greasons = evaluate_gravity_contract(gravity, gth)
    anatomy = anatomy_metrics_np(x, fps)
    ath = AnatomyThresholds.from_env()
    aok, areasons = evaluate_anatomy_contract(anatomy, ath)
    return {
        "path": str(path),
        "ok": bool(gok and aok),
        "reasons": [*("gravity:" + r for r in greasons), *("anatomy:" + r for r in areasons)],
        "gravity_ok": bool(gok),
        "anatomy_ok": bool(aok),
        "gravity": gravity,
        "anatomy": anatomy,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input")
    group.add_argument("--motion_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--allow_failed", action="store_true")
    args = ap.parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError(f"--fps must be positive and finite, got {args.fps!r}")

    paths = [Path(args.input)] if args.input else sorted(Path(args.motion_dir).rglob("*.npy"))
    # Exclude known generated helper arrays.
    paths = [p for p in paths if not any(t in p.name.lower() for t in ("mask", "motion_ref", "single_test", "jitter"))]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            rows.append(_audit(path, args.fps))
        except Exception as exc:
            rows.append({"path": str(path), "ok": False, "reasons": [str(exc)], "gravity_ok": False, "anatomy_ok": False})
    summary = {
        "schema": "v46_52_combined_motion_contract_audit",
        "fps": float(args.fps),
        "count": len(rows),
        "ok_count": sum(bool(r["ok"]) for r in rows),
        "failed_count": sum(not bool(r["ok"]) for r in rows),
        "all_ok": bool(rows and all(bool(r["ok"]) for r in rows)),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            "path", "ok", "gravity_ok", "anatomy_ok", "reasons",
            "anatomy_quality", "torso_compression_ratio_p05", "local_angle_violation_ratio",
            "self_collision_severe_ratio", "spine_cumulative_angle_p95_deg",
            "pelvis_height_norm_p05", "pelvis_height_norm_p95", "foot_penetration_min_m",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                a = r.get("anatomy", {})
                w.writerow({
                    "path": r.get("path"),
                    "ok": r.get("ok"),
                    "gravity_ok": r.get("gravity_ok"),
                    "anatomy_ok": r.get("anatomy_ok"),
                    "reasons": " | ".join(r.get("reasons", [])),
                    **{k: a.get(k) for k in keys[5:]},
                })
    print(json.dumps({k: summary[k] for k in ("count", "ok_count", "failed_count", "all_ok")}, indent=2))
    return 0 if summary["all_ok"] or args.allow_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
