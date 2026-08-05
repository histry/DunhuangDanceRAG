#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare retrieval, refiner, diffusion and IK motion-activity stages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

# When this file is executed as ``python scripts/...py``, Python places only
# ``scripts/`` on sys.path.  Add the repository root explicitly so sibling
# research packages such as ``evaluation`` and ``motion_geometry`` resolve
# independently of the caller's current working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.motion_activity_analysis import (
    evaluate_final_motion_activity,
    motion_activity_metrics,
    write_activity_report,
)

STAGES = ("retrieval", "refiner", "diffusion", "full_ik")


def _stage_path(final_motion: Path, stage: str) -> Path:
    return final_motion.with_name(f"{final_motion.stem}.stage_{stage}.npy")


def analyze(final_motion: Path, fps: float) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    first_collapsed: Optional[str] = None
    previous_density: Optional[float] = None
    for stage in STAGES:
        path = _stage_path(final_motion, stage)
        if not path.is_file():
            rows[stage] = {"available": False, "path": str(path)}
            continue
        motion = np.load(path, allow_pickle=True)
        metrics = motion_activity_metrics(motion, fps=fps)
        gate = evaluate_final_motion_activity(motion, fps=fps)
        density = float(metrics["motion_density_mean"])
        density_delta = None if previous_density is None else density - previous_density
        rows[stage] = {
            "available": True,
            "path": str(path),
            "metrics": metrics,
            "collapse_detected": bool(gate["collapse_detected"]),
            "collapse_reasons": list(gate["reasons"]),
            "motion_density_delta_from_previous": density_delta,
        }
        if first_collapsed is None and gate["collapse_detected"]:
            first_collapsed = stage
        previous_density = density

    interpretation = {
        None: "No available stage satisfies the static-collapse definition.",
        "retrieval": "Collapse is already present in retrieval, route selection or event resampling.",
        "refiner": "Retrieval is active but the boundary refiner suppresses activity.",
        "diffusion": "Refiner output is active but local diffusion suppresses activity.",
        "full_ik": "Diffusion output is active but IK/contact or heading restoration suppresses activity.",
    }[first_collapsed]
    return {
        "schema": "motion_activity_stage_comparison",
        "final_motion": str(final_motion),
        "fps": float(fps),
        "stage_order": list(STAGES),
        "first_collapsed_stage": first_collapsed,
        "interpretation": interpretation,
        "stages": rows,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="final motion .npy path")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    motion = Path(args.motion)
    report = analyze(motion, fps=float(args.fps))
    output = Path(args.output) if args.output else motion.with_name(
        motion.stem + ".motion_activity_stages.json"
    )
    write_activity_report(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
