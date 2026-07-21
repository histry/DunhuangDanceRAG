#!/usr/bin/env python3
"""Compare two whole-song scheduler runs at route and SO(3) levels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    normalize_rot6d_layout,
    rot6d_to_matrix_layout_np,
    so3_geodesic_np,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_motion(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[1] != 151:
        raise ValueError(f"Expected [T,151] motion in {path}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Motion contains NaN/Inf: {path}")
    return value


def _load_report(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("schedule"), list):
        raise ValueError(f"Invalid schedule report: {path}")
    return value


def _percentiles(values: np.ndarray, points: Sequence[float]) -> Dict[str, float]:
    result = np.percentile(values, points)
    return {f"p{point:g}": float(value) for point, value in zip(points, result)}


def _field_parity(old: Sequence[Dict[str, Any]], new: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    count = min(len(old), len(new))
    exact = sum(old[index].get(field) == new[index].get(field) for index in range(count))
    return {
        "field": field,
        "compared": count,
        "exact": exact,
        "exact_ratio": float(exact / count) if count else 1.0,
        "lengths_equal": len(old) == len(new),
    }


def compare(
    old_motion_path: Path,
    old_report_path: Path,
    new_motion_path: Path,
    new_report_path: Path,
    old_layout: str,
    new_layout: str,
    audio_path: Path | None = None,
) -> Dict[str, Any]:
    old_layout = normalize_rot6d_layout(old_layout)
    new_layout = normalize_rot6d_layout(new_layout)
    old_motion = _load_motion(old_motion_path)
    new_motion = _load_motion(new_motion_path)
    old_report = _load_report(old_report_path)
    new_report = _load_report(new_report_path)
    if old_motion.shape != new_motion.shape:
        raise ValueError(
            f"Cannot make frame-wise comparison: {old_motion.shape} != {new_motion.shape}"
        )

    old_rotations = rot6d_to_matrix_layout_np(
        old_motion[:, 7:151].reshape(-1, 24, 6),
        old_layout,
    )
    new_rotations = rot6d_to_matrix_layout_np(
        new_motion[:, 7:151].reshape(-1, 24, 6),
        new_layout,
    )
    geodesic = so3_geodesic_np(old_rotations, new_rotations)
    per_frame_max = np.max(geodesic, axis=1)
    root_delta = np.abs(old_motion[:, 4:7] - new_motion[:, 4:7])
    contact_delta = np.abs(old_motion[:, :4] - new_motion[:, :4])
    raw_rotation_delta = np.abs(old_motion[:, 7:151] - new_motion[:, 7:151])
    old_schedule = old_report["schedule"]
    new_schedule = new_report["schedule"]
    event_parity = _field_parity(old_schedule, new_schedule, "event_id")
    transition_parity = _field_parity(old_schedule, new_schedule, "transition_len")
    content_parity = _field_parity(old_schedule, new_schedule, "allocated_content_len")
    family_parity = _field_parity(old_schedule, new_schedule, "family_id")

    result: Dict[str, Any] = {
        "schema": "scheduler_end_to_end_differential_v1",
        "inputs": {
            "old_motion": str(old_motion_path.resolve()),
            "old_report": str(old_report_path.resolve()),
            "new_motion": str(new_motion_path.resolve()),
            "new_report": str(new_report_path.resolve()),
            "audio": str(audio_path.resolve()) if audio_path else None,
            "audio_sha256": _sha256(audio_path) if audio_path else None,
            "old_rot6d_layout": old_layout,
            "new_rot6d_layout": new_layout,
        },
        "hashes": {
            "old_motion_sha256": _sha256(old_motion_path),
            "new_motion_sha256": _sha256(new_motion_path),
            "old_report_sha256": _sha256(old_report_path),
            "new_report_sha256": _sha256(new_report_path),
        },
        "contracts": {
            "old_rotation_contract": old_report.get("rotation_contract"),
            "new_rotation_contract": new_report.get("rotation_contract"),
        },
        "route": {
            "old_slots": len(old_schedule),
            "new_slots": len(new_schedule),
            "event_id": event_parity,
            "family_id": family_parity,
            "transition_len": transition_parity,
            "allocated_content_len": content_parity,
            "old_score": float(old_report.get("score", 0.0)),
            "new_score": float(new_report.get("score", 0.0)),
            "score_abs_delta": abs(
                float(old_report.get("score", 0.0))
                - float(new_report.get("score", 0.0))
            ),
        },
        "motion": {
            "shape": list(old_motion.shape),
            "contact_abs_mean": float(contact_delta.mean()),
            "contact_abs_max": float(contact_delta.max()),
            "root_abs_mean_m": float(root_delta.mean()),
            "root_abs_max_m": float(root_delta.max()),
            "rot6d_raw_abs_mean": float(raw_rotation_delta.mean()),
            "rot6d_raw_abs_max": float(raw_rotation_delta.max()),
            "rotation_geodesic_rad": {
                "mean": float(geodesic.mean()),
                **_percentiles(geodesic, (50.0, 95.0, 99.0, 100.0)),
                "joint_ratio_over_1e-4": float(np.mean(geodesic > 1.0e-4)),
                "frame_ratio_over_1e-4": float(np.mean(per_frame_max > 1.0e-4)),
            },
        },
    }
    result["acceptance"] = {
        "exact_frame_count": old_motion.shape[0] == new_motion.shape[0],
        "exact_route": event_parity["exact_ratio"] == 1.0,
        "exact_transition_lengths": transition_parity["exact_ratio"] == 1.0,
        "exact_content_lengths": content_parity["exact_ratio"] == 1.0,
        "rotation_p95_below_1e-4_rad": result["motion"]["rotation_geodesic_rad"]["p95"] < 1.0e-4,
        "rotation_max_below_0_1_rad": result["motion"]["rotation_geodesic_rad"]["p100"] < 0.1,
        "root_max_below_1e-4_m": result["motion"]["root_abs_max_m"] < 1.0e-4,
    }
    result["acceptance"]["passed"] = all(result["acceptance"].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-motion", type=Path, required=True)
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--new-motion", type=Path, required=True)
    parser.add_argument("--new-report", type=Path, required=True)
    parser.add_argument("--old-layout", default=CANONICAL_ROT6D_LAYOUT)
    parser.add_argument("--new-layout", default=CANONICAL_ROT6D_LAYOUT)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = compare(
        args.old_motion,
        args.old_report,
        args.new_motion,
        args.new_report,
        args.old_layout,
        args.new_layout,
        args.audio,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
        print(f"[SAVED] {args.out}")
    print(serialized)
    return 2 if args.require_pass and not result["acceptance"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
