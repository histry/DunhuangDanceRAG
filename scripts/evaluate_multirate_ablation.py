#!/usr/bin/env python3
"""Evaluate native 30, native 60 and SO(3) 30->60 motion ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_geometry.physical import motion_physical_metrics_np, recompute_contacts_np
from motion_geometry.resampling import resample_edge151_np
from motion_geometry.rotations import rot6d_to_matrix_np, so3_geodesic_np
from motion_geometry.smpl24 import ROT6D_END, ROT6D_START, skeleton_contract
from support.event_identity import event_uids_from_generation_db


def _load_motion(path: str | Path) -> np.ndarray:
    value = np.asarray(np.load(path, allow_pickle=True))
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[1] != 151:
        raise ValueError(f"Expected [T,151] motion at {path}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Motion contains NaN/Inf: {path}")
    return value.astype(np.float32)


def _fidelity(reference: np.ndarray, candidate: np.ndarray, fps: float) -> dict[str, float]:
    aligned = resample_edge151_np(candidate, target_frames=len(reference))
    root_error = np.linalg.norm(reference[:, 4:7] - aligned[:, 4:7], axis=-1)
    rr = rot6d_to_matrix_np(reference[:, ROT6D_START:ROT6D_END].reshape(len(reference), 24, 6))
    rc = rot6d_to_matrix_np(aligned[:, ROT6D_START:ROT6D_END].reshape(len(aligned), 24, 6))
    angle = so3_geodesic_np(rr, rc)
    return {
        "fps": float(fps),
        "root_rmse_m": float(np.sqrt(np.mean(root_error ** 2))),
        "joint_geodesic_mean_rad": float(np.mean(angle)),
        "joint_geodesic_p95_rad": float(np.percentile(angle, 95)),
    }


def _uid_comparison(db30: str | None, db60: str | None) -> dict[str, Any] | None:
    if not db30 or not db60:
        return None
    left = np.load(db30, allow_pickle=True)
    right = np.load(db60, allow_pickle=True)
    u30 = set(map(str, event_uids_from_generation_db({key: left[key] for key in left.files})))
    u60 = set(map(str, event_uids_from_generation_db({key: right[key] for key in right.files})))
    return {
        "fps30_count": len(u30),
        "fps60_count": len(u60),
        "intersection": len(u30 & u60),
        "same_uid_set": u30 == u60,
        "fps30_only": sorted(u30 - u60)[:20],
        "fps60_only": sorted(u60 - u30)[:20],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion30", required=True)
    parser.add_argument("--motion60", required=True)
    parser.add_argument("--db30")
    parser.add_argument("--db60")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    motion30 = _load_motion(args.motion30)
    motion60 = _load_motion(args.motion60)
    upsampled = resample_edge151_np(motion30, source_fps=30.0, target_fps=60.0)
    upsampled = recompute_contacts_np(upsampled, fps=60.0)
    contact_report = {
        "schema": "contact_recompute_si_v1",
        "fps": 60.0,
        "contact_ratio": float(np.mean(upsampled[:, :4] > 0.5)),
    }
    upsampled_path = out_dir / "motion_30_to_60_so3.npy"
    np.save(upsampled_path, upsampled.astype(np.float32))

    report = {
        "schema": "dunhuang_30_60_30to60_ablation_v1",
        "skeleton_contract": skeleton_contract(),
        "motions": {
            "native30": motion_physical_metrics_np(motion30, fps=30.0),
            "native60": motion_physical_metrics_np(motion60, fps=60.0),
            "fps30_to_60": motion_physical_metrics_np(upsampled, fps=60.0),
        },
        "contact_recompute_30_to_60": contact_report,
        "fidelity": {
            "native60_vs_native30_time_aligned": _fidelity(motion60, motion30, 60.0),
            "native60_vs_30_to_60": _fidelity(motion60, upsampled, 60.0),
        },
        "event_uid_comparison": _uid_comparison(args.db30, args.db60),
        "outputs": {"motion_30_to_60": str(upsampled_path)},
    }
    report_path = out_dir / "multirate_ablation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
