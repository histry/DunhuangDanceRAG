#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chang-E official-SMPL source-aware preprocessing.

This is the formal source path for direct official fitted SMPL parameters.

Design:
1. Official SMPL is already the authoritative articulated representation, so no
   BVH->SMPL24 optimization/retargeting is performed.
2. The source is first loaded at its authoritative/native Chang-E timebase.
3. Only catastrophic discontinuities are hard-cut in canonical Cartesian FK24
   space (plus a near-pi local-rotation integrity guard).
4. Every retained continuous segment is resampled independently to the target
   FPS, re-anchored by constant X/Z translation, floor-normalized by a constant
   per-segment Y translation, and contacts are recomputed.
5. Source-level anatomy/gravity/physical values are diagnostics, not final
   generation gates. Fine-grained posture/anatomy quality remains deferred to
   the existing event-level filter after event slicing.
6. Final generated-motion physical limits are not modified here.

The produced cache intentionally keeps the historical sibling
``*.retarget.json`` filename so the existing Event-DB stack can consume the
cache without changing its storage layout. The report schema explicitly states
that retargeting was not applied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from contracts.anatomy import anatomy_metrics_np
from contracts.gravity import FOOT_JOINTS, fk24_np, gravity_metrics_np
from data_pipeline.chang_e_smpl_manifest import (
    file_sha256,
    load_manifest as load_smpl_manifest,
    manifest_sha256 as smpl_manifest_sha256,
    match_manifest_entry as match_smpl_manifest_entry,
    validate_source as validate_smpl_source,
)
from motion_geometry.physical import (
    SUPPORT_POLICY_SOURCE,
    motion_physical_metrics_np,
    recompute_contacts_np,
)
from motion_geometry.resampling import positions_for_fps, resample_rotations_so3_np
from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
    so3_geodesic_np,
    validate_rot6d_roundtrip_np,
)
from motion_geometry.smpl24 import (
    MOTION_DIM,
    NUM_JOINTS,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
    ROT6D_END,
    ROT6D_START,
    skeleton_contract,
)
from retargeting.smpl_adapter import load_smpl24_parameters

SCHEMA = "chang_e_official_smpl_source_aware_cache_v1"
SEGMENT_REPORT_SCHEMA = "chang_e_official_smpl_source_aware_preprocess_v1"
VERSION = "chang_e_official_smpl_source_aware_preprocess_event_geometry_2"

SMPL_EXTENSIONS = {".npz", ".pkl", ".pickle"}
_SKIP_TOKENS = ("event", "index", "feature", "cache", "split", "checkpoint")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class HardCutPolicy:
    """Catastrophic-only source continuity policy.

    Floors are intentionally high. Authentic fast dance dynamics must remain in
    the source; this stage only separates obvious teleports/spikes/corruption.
    A robust per-source threshold is combined with the absolute floor.
    """

    min_segment_seconds: float = 2.0
    root_step_floor_m: float = 0.20
    joint_step_floor_m: float = 0.38
    body_relative_step_floor_m: float = 0.32
    body_relative_second_difference_floor_m: float = 0.26
    local_rotation_step_floor_rad: float = 2.60
    robust_quantile: float = 99.7
    robust_quantile_scale: float = 1.50
    robust_mad_scale: float = 14.0

    @classmethod
    def from_environment(cls) -> "HardCutPolicy":
        return cls(
            min_segment_seconds=max(
                0.5, _env_float("OFFICIAL_SMPL_MIN_SEGMENT_SECONDS", 2.0)
            ),
            root_step_floor_m=max(
                1.0e-4, _env_float("OFFICIAL_SMPL_HARD_CUT_ROOT_STEP_M", 0.20)
            ),
            joint_step_floor_m=max(
                1.0e-4, _env_float("OFFICIAL_SMPL_HARD_CUT_JOINT_STEP_M", 0.38)
            ),
            body_relative_step_floor_m=max(
                1.0e-4, _env_float("OFFICIAL_SMPL_HARD_CUT_BODY_STEP_M", 0.32)
            ),
            body_relative_second_difference_floor_m=max(
                1.0e-4, _env_float("OFFICIAL_SMPL_HARD_CUT_BODY_D2_M", 0.26)
            ),
            local_rotation_step_floor_rad=max(
                0.1, _env_float("OFFICIAL_SMPL_HARD_CUT_ROT_STEP_RAD", 2.60)
            ),
            robust_quantile=float(
                np.clip(
                    _env_float("OFFICIAL_SMPL_HARD_CUT_QUANTILE", 99.7),
                    90.0,
                    99.999,
                )
            ),
            robust_quantile_scale=max(
                1.0,
                _env_float("OFFICIAL_SMPL_HARD_CUT_QUANTILE_SCALE", 1.50),
            ),
            robust_mad_scale=max(
                3.0, _env_float("OFFICIAL_SMPL_HARD_CUT_MAD_SCALE", 14.0)
            ),
        )


def discover_official_smpl_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SMPL_EXTENSIONS:
            continue
        name = path.name.lower()
        if any(token in name for token in _SKIP_TOKENS):
            continue
        paths.append(path)
    return sorted(paths)



def load_name_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("--name_map must contain a JSON object")
    return {str(k): str(v) for k, v in payload.items()}


def _robust_limit(values: np.ndarray, absolute_floor: float, policy: HardCutPolicy) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(absolute_floor)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    robust_sigma = 1.4826 * mad
    q = float(np.percentile(x, policy.robust_quantile))
    return float(
        max(
            absolute_floor,
            median + policy.robust_mad_scale * robust_sigma,
            q * policy.robust_quantile_scale,
        )
    )


def _fill_nonfinite_frames(motion: np.ndarray, finite_frames: np.ndarray) -> np.ndarray:
    """Create a finite analysis copy without silently accepting bad frames."""

    x = np.asarray(motion, dtype=np.float32).copy()
    good = np.flatnonzero(finite_frames)
    if good.size == 0:
        raise RuntimeError("Official SMPL contains no finite frame")
    for frame in np.flatnonzero(~finite_frames):
        pos = int(np.searchsorted(good, frame))
        if pos <= 0:
            nearest = int(good[0])
        elif pos >= len(good):
            nearest = int(good[-1])
        else:
            left, right = int(good[pos - 1]), int(good[pos])
            nearest = left if frame - left <= right - frame else right
        x[frame] = x[nearest]
    return x


def hard_cut_analysis(
    motion: np.ndarray,
    *,
    fps: float,
    policy: Optional[HardCutPolicy] = None,
) -> Dict[str, Any]:
    """Find catastrophic discontinuity boundaries in canonical Cartesian space."""

    p = policy or HardCutPolicy.from_environment()
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < MOTION_DIM:
        raise ValueError(f"Expected [T,{MOTION_DIM}], got {x.shape}")
    x = x[:, :MOTION_DIM]
    T = len(x)
    if T < 2:
        return {
            "schema": "official_smpl_cartesian_hard_cut_v1",
            "cut_boundaries": [],
            "bad_frames": [],
            "thresholds": {},
            "observed": {},
            "reasons_by_boundary": {},
        }

    finite_frames = np.isfinite(x).all(axis=1)
    safe = _fill_nonfinite_frames(x, finite_frames)
    joints = fk24_np(safe)
    root = joints[:, 0]
    body = joints - root[:, None, :]

    root_step = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    joint_step = np.linalg.norm(np.diff(joints, axis=0), axis=-1).max(axis=1)
    body_step = np.linalg.norm(np.diff(body, axis=0), axis=-1).max(axis=1)

    if T >= 3:
        body_d2 = np.linalg.norm(
            body[2:] - 2.0 * body[1:-1] + body[:-2], axis=-1
        ).max(axis=1)
    else:
        body_d2 = np.zeros((0,), dtype=np.float32)

    local = rot6d_to_matrix_np(
        safe[:, ROT6D_START:ROT6D_END].reshape(T, NUM_JOINTS, 6)
    )
    rot_step = so3_geodesic_np(local[:-1], local[1:]).max(axis=1)

    thresholds = {
        "root_step_m": _robust_limit(root_step, p.root_step_floor_m, p),
        "joint_step_m": _robust_limit(joint_step, p.joint_step_floor_m, p),
        "body_relative_step_m": _robust_limit(
            body_step, p.body_relative_step_floor_m, p
        ),
        "body_relative_second_difference_m": _robust_limit(
            body_d2, p.body_relative_second_difference_floor_m, p
        ),
        "local_rotation_step_rad": _robust_limit(
            rot_step, p.local_rotation_step_floor_rad, p
        ),
    }

    reasons: Dict[int, set[str]] = {}

    def mark(boundary: int, reason: str) -> None:
        if 0 < boundary < T:
            reasons.setdefault(int(boundary), set()).add(str(reason))

    for idx in np.flatnonzero(root_step > thresholds["root_step_m"]):
        mark(int(idx) + 1, "root_cartesian_step")
    for idx in np.flatnonzero(joint_step > thresholds["joint_step_m"]):
        mark(int(idx) + 1, "joint_cartesian_step")
    for idx in np.flatnonzero(body_step > thresholds["body_relative_step_m"]):
        mark(int(idx) + 1, "body_relative_cartesian_step")
    for idx in np.flatnonzero(rot_step > thresholds["local_rotation_step_rad"]):
        mark(int(idx) + 1, "local_rotation_near_pi_or_jump")

    # A one-frame Cartesian spike often creates a large second difference with
    # two adjacent transition discontinuities. Isolate its center by cutting on
    # both sides rather than smoothing it into neighbouring valid motion.
    for idx in np.flatnonzero(
        body_d2 > thresholds["body_relative_second_difference_m"]
    ):
        center = int(idx) + 1
        mark(center, "body_relative_cartesian_impulse")
        mark(center + 1, "body_relative_cartesian_impulse")

    bad_frames = np.flatnonzero(~finite_frames).astype(int).tolist()
    for frame in bad_frames:
        mark(frame, "nonfinite_frame")
        mark(frame + 1, "nonfinite_frame")

    cut_boundaries = sorted(reasons)
    return {
        "schema": "official_smpl_cartesian_hard_cut_v1",
        "fps": float(fps),
        "frames": int(T),
        "cut_boundaries": cut_boundaries,
        "bad_frames": bad_frames,
        "thresholds": thresholds,
        "observed": {
            "root_step_m_max": float(np.max(root_step)) if root_step.size else 0.0,
            "joint_step_m_max": float(np.max(joint_step)) if joint_step.size else 0.0,
            "body_relative_step_m_max": (
                float(np.max(body_step)) if body_step.size else 0.0
            ),
            "body_relative_second_difference_m_max": (
                float(np.max(body_d2)) if body_d2.size else 0.0
            ),
            "local_rotation_step_rad_max": (
                float(np.max(rot_step)) if rot_step.size else 0.0
            ),
        },
        "reasons_by_boundary": {
            str(boundary): sorted(values) for boundary, values in sorted(reasons.items())
        },
        "policy": asdict(p),
    }


def continuous_intervals(
    frames: int,
    cut_boundaries: Iterable[int],
    *,
    min_frames: int,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    boundaries = [0]
    boundaries.extend(
        sorted({int(v) for v in cut_boundaries if 0 < int(v) < int(frames)})
    )
    boundaries.append(int(frames))
    kept: List[Tuple[int, int, int]] = []
    dropped: List[Tuple[int, int, int]] = []
    for interval_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        row = (int(interval_index), int(start), int(end))
        if end - start >= int(min_frames):
            kept.append(row)
        else:
            dropped.append(row)
    return kept, dropped


def resample_and_normalize_segment(
    native_motion: np.ndarray,
    *,
    start: int,
    end: int,
    source_fps: float,
    target_fps: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    clip = np.asarray(native_motion[start:end, :MOTION_DIM], dtype=np.float32)
    if len(clip) < 2:
        raise ValueError("Segment must contain at least two frames")

    local = rot6d_to_matrix_np(
        clip[:, ROT6D_START:ROT6D_END].reshape(len(clip), NUM_JOINTS, 6)
    )
    root = clip[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]

    if abs(float(source_fps) - float(target_fps)) > 1.0e-8:
        positions = positions_for_fps(len(clip), float(source_fps), float(target_fps))
        local = resample_rotations_so3_np(local, positions)
        source_axis = np.arange(len(root), dtype=np.float32)
        root = np.stack(
            [
                np.interp(positions, source_axis, root[:, dim])
                for dim in range(3)
            ],
            axis=-1,
        ).astype(np.float32)

    motion = np.zeros((len(local), MOTION_DIM), dtype=np.float32)
    motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = root
    motion[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(local).reshape(
        len(local), -1
    )

    x0 = float(motion[0, ROOT_X_IDX])
    z0 = float(motion[0, ROOT_Z_IDX])
    motion[:, ROOT_X_IDX] -= x0
    motion[:, ROOT_Z_IDX] -= z0

    joints = fk24_np(motion)
    floor_y = float(np.percentile(joints[:, list(FOOT_JOINTS), 1], 5))
    motion[:, ROOT_Y_IDX] -= floor_y
    motion = recompute_contacts_np(
        motion,
        fps=float(target_fps),
        height_margin_m=_env_float("RETARGET_CONTACT_HEIGHT_M", 0.055),
        speed_gate_mps=_env_float("RETARGET_CONTACT_SPEED_MPS", 0.75),
        median_seconds=_env_float(
            "RETARGET_CONTACT_MEDIAN_SECONDS", 1.0 / 6.0
        ),
    )

    return motion.astype(np.float32), {
        "root_x_anchor_removed_m": x0,
        "root_z_anchor_removed_m": z0,
        "floor_y_removed_m": floor_y,
    }


def segment_integrity(motion: np.ndarray) -> Tuple[bool, List[str], Dict[str, Any]]:
    x = np.asarray(motion, dtype=np.float32)
    reasons: List[str] = []
    if x.ndim != 2 or x.shape[1] != MOTION_DIM:
        reasons.append(f"bad_shape:{x.shape}")
        return False, reasons, {}
    nonfinite = int((~np.isfinite(x)).sum())
    if nonfinite:
        reasons.append(f"nonfinite_count={nonfinite}")

    roundtrip = validate_rot6d_roundtrip_np(
        x[:, ROT6D_START:ROT6D_END].reshape(len(x), NUM_JOINTS, 6)
    )
    if not bool(roundtrip.get("finite", False)):
        reasons.append("rot6d_roundtrip_nonfinite")
    if float(roundtrip.get("max_geodesic_rad", 1.0)) > 1.0e-4:
        reasons.append(
            "rot6d_roundtrip_max_geodesic_rad="
            f"{float(roundtrip.get('max_geodesic_rad', 1.0)):.6g}"
        )
    return not reasons, reasons, {
        "nonfinite_count": nonfinite,
        "rot6d_roundtrip": roundtrip,
    }


def _source_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": str(row["source_id"]),
        "recording_uid": str(row["recording_uid"]),
        "performer_track_id": row.get("performer_track_id", -1),
        "sequence_index": row.get("sequence_index", -1),
        "performer_group": row.get("performer_group", "unknown"),
        "dance_category": row.get("dance_category", "unknown"),
        "take_id": row.get("take_id"),
        "skeleton_id": "chang_e_official_smpl",
    }


def build_source(
    source: Path,
    *,
    in_dir: Path,
    out_dir: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    target_fps: float,
    policy: HardCutPolicy,
    scaling_mode: str,
    name_map: Mapping[str, str],
    overwrite: bool,
) -> Dict[str, Any]:
    explicit = name_map.get(source.name) or name_map.get(str(source.relative_to(in_dir)))
    row = match_smpl_manifest_entry(source, manifest, explicit_source_id=explicit)
    metadata = _source_metadata(row)
    source_id = str(metadata["source_id"])
    # Formal timebase/provenance comes only from the official-SMPL
    # manifest. The historical BVH manifest is not consulted.
    source_contract = validate_smpl_source(
        source,
        manifest=manifest,
        manifest_file=manifest_path,
        explicit_source_id=source_id,
        verify_hash=True,
    )

    source_fps = float(
        source_contract["source_fps"]
    )

    native_motion, adapter = load_smpl24_parameters(
        source,
        target_fps=source_fps,
        source_fps=source_fps,
        scaling_mode=scaling_mode,
        localize_root_xz=False,
        contact_height_m=_env_float(
            "RETARGET_CONTACT_HEIGHT_M",
            0.055,
        ),
        contact_speed_mps=_env_float(
            "RETARGET_CONTACT_SPEED_MPS",
            0.75,
        ),
        contact_median_seconds=_env_float(
            "RETARGET_CONTACT_MEDIAN_SECONDS",
            1.0 / 6.0,
        ),
    )

    native_motion = np.asarray(
        native_motion,
        dtype=np.float32,
    )

    expected_frames = int(
        source_contract["frames"]
    )

    if len(native_motion) != expected_frames:
        raise RuntimeError(
            f"Official SMPL adapter frame mismatch "
            f"for {source_id}: "
            f"{len(native_motion)}!={expected_frames}"
        )

    source_duration = float(
        source_contract["duration_seconds"]
    )

    analysis = hard_cut_analysis(native_motion, fps=source_fps, policy=policy)
    min_native = max(2, int(math.ceil(policy.min_segment_seconds * source_fps)))
    kept, dropped = continuous_intervals(
        len(native_motion),
        analysis["cut_boundaries"],
        min_frames=min_native,
    )

    source_root = out_dir / source_id
    source_root.mkdir(parents=True, exist_ok=True)
    segment_reports: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for interval_index, start, end in kept:
        dst = source_root / f"segment_{interval_index:03d}.npy"
        rep_path = dst.with_suffix(".retarget.json")
        if dst.exists() and rep_path.exists() and not overwrite:
            old = json.loads(rep_path.read_text(encoding="utf-8"))
            if (
                old.get("schema") == SEGMENT_REPORT_SCHEMA
                and bool(old.get("ok", False))
                and old.get("source_sha256") == file_sha256(source)
                and float(old.get("target_fps", -1.0)) == float(target_fps)
            ):
                segment_reports.append(old)
                continue

        try:
            motion, normalization = resample_and_normalize_segment(
                native_motion,
                start=start,
                end=end,
                source_fps=source_fps,
                target_fps=target_fps,
            )
            min_target = max(
                2, int(math.floor(policy.min_segment_seconds * target_fps))
            )
            if len(motion) < min_target:
                raise RuntimeError(
                    f"resampled segment too short: {len(motion)} < {min_target}"
                )
            integrity_ok, integrity_reasons, integrity = segment_integrity(motion)
            if not integrity_ok:
                raise RuntimeError(
                    "segment integrity failed: " + " | ".join(integrity_reasons)
                )

            # Diagnostics only. They are deliberately not used as a source veto.
            anatomy = anatomy_metrics_np(motion, fps=float(target_fps))
            gravity = gravity_metrics_np(motion, fps=float(target_fps))
            physical = motion_physical_metrics_np(
                motion,
                fps=float(target_fps),
                support_policy=SUPPORT_POLICY_SOURCE,
            )

            start_seconds = float(start / source_fps)
            end_seconds = float(end / source_fps)
            report: Dict[str, Any] = {
                "schema": SEGMENT_REPORT_SCHEMA,
                "version": VERSION,
                "ok": True,
                "source_gate_ok": True,
                "source_preprocess_ok": True,
                "physical_clean_ok": True,
                "fit_ok": True,
                "retargeting_applied": False,
                "source": str(source.resolve()),
                "source_used": str(source.resolve()),
                "source_relative": str(source.relative_to(in_dir)),
                "preferred_source": str(source.resolve()),
                "output": str(dst.resolve()),
                "source_sha256": file_sha256(source),
                "smpl_manifest": str(manifest_path.resolve()),
                "source_manifest": str(manifest_path.resolve()),
                "source_manifest_sha256": smpl_manifest_sha256(manifest_path),
                "source_metadata": metadata,
                "source_fps": float(source_fps),
                "target_fps": float(target_fps),
                "source_duration_seconds": source_duration,
                "smpl_adapter": adapter,
                "skeleton_contract": skeleton_contract(),
                "source_preprocess_contract": {
                    "schema": SEGMENT_REPORT_SCHEMA,
                    "direct_official_smpl": True,
                    "retargeting_applied": False,
                    "optimizer_applied": False,
                    "source_relative_jerk_gate_applied": False,
                    "final_generation_gate_reused": False,
                    "hard_cut_space": "canonical_cartesian_fk24",
                    "hard_cut_policy": asdict(policy),
                    "event_quality_gate_deferred": True,
                    "root_xz_policy": "constant_segment_reanchor_only",
                    "floor_policy": "constant_segment_p05_foot_joint_shift",
                    "heading_policy": (
                        "preserve_official_smpl_orientation_then_apply_"
                        "existing_event_heading_contract_after_slicing"
                    ),
                },
                "preprocess_segment": {
                    "clean": True,
                    "segment_index": int(interval_index),
                    "native_start_frame": int(start),
                    "native_end_frame_exclusive": int(end),
                    "native_frames": int(end - start),
                    "source_start_seconds": start_seconds,
                    "source_end_seconds": end_seconds,
                    "target_frames": int(len(motion)),
                    "normalization": normalization,
                },
                "hard_cut_source_analysis": analysis,
                "integrity": integrity,
                "anatomy_diagnostic": anatomy,
                "gravity_diagnostic": gravity,
                "source_physical_diagnostic": physical,
                "physical_clean_gate": {
                    "ok": True,
                    "policy": (
                        "source_aware_catastrophic_continuity_only;"
                        "event_anatomy_deferred"
                    ),
                    "final_generation_gate_reused": False,
                },
            }
            np.save(dst, motion)
            rep_path.write_text(
                json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            segment_reports.append(report)
        except Exception as exc:
            failures.append(
                {
                    "interval_index": int(interval_index),
                    "start": int(start),
                    "end": int(end),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    return {
        "source_id": source_id,
        "recording_uid": metadata["recording_uid"],
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "native_frames": int(len(native_motion)),
        "duration_seconds": source_duration,
        "hard_cut_analysis": analysis,
        "num_candidate_intervals": int(len(kept) + len(dropped)),
        "num_retained_segments": int(len(segment_reports)),
        "dropped_short_intervals": [
            {"interval_index": i, "start": st, "end": ed, "frames": ed - st}
            for i, st, ed in dropped
        ],
        "segment_reports": segment_reports,
        "segment_failures": failures,
        "ok": bool(segment_reports),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--smpl_manifest",
        default=None,
        help="Authoritative Chang-E official-SMPL manifest",
    )
    # Compatibility alias only; formal launchers use --smpl_manifest.
    ap.add_argument(
        "--source_manifest",
        dest="legacy_source_manifest",
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--target_fps", type=float, default=30.0, choices=(30.0, 60.0))
    ap.add_argument(
        "--scaling_mode",
        choices=("canonical_body", "scale_translation", "inverse_scale_translation"),
        default="canonical_body",
    )
    ap.add_argument("--name_map", default=None)
    ap.add_argument("--min_ok_sources", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    raw_manifest = (
        args.smpl_manifest
        or args.legacy_source_manifest
    )

    if not raw_manifest:
        ap.error("--smpl_manifest is required")

    manifest_path = Path(
        raw_manifest
    ).expanduser().resolve()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Official SMPL directory does not exist: {in_dir}")
    manifest = load_smpl_manifest(manifest_path, required=True)
    if manifest is None:
        raise RuntimeError("Chang-E manifest could not be loaded")

    files = discover_official_smpl_files(in_dir)
    if not files:
        raise RuntimeError(f"No official SMPL npz/pkl/pickle files under {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    name_map = load_name_map(
        Path(args.name_map).expanduser().resolve() if args.name_map else None
    )
    policy = HardCutPolicy.from_environment()

    resolved: Dict[str, Path] = {}
    resolution_errors: List[Dict[str, str]] = []
    for source in files:
        try:
            explicit = name_map.get(source.name) or name_map.get(
                str(source.relative_to(in_dir))
            )
            row = match_smpl_manifest_entry(
                source, manifest, explicit_source_id=explicit
            )
            source_id = str(row["source_id"])
            if source_id in resolved:
                raise RuntimeError(
                    f"Multiple official SMPL files map to source_id={source_id}: "
                    f"{resolved[source_id]} and {source}"
                )
            resolved[source_id] = source
        except Exception as exc:
            resolution_errors.append({"source": str(source), "error": str(exc)})

    min_ok = (
        int(args.min_ok_sources)
        if args.min_ok_sources is not None
        else _env_int("RETARGET_MIN_OK_SOURCES", min(8, len(resolved)))
    )
    min_ok = max(3, min(min_ok, max(3, len(resolved))))

    source_reports: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = list(resolution_errors)
    for index, (source_id, source) in enumerate(sorted(resolved.items()), 1):
        print(
            f"[OFFICIAL-SMPL {index}/{len(resolved)}] {source_id}: {source}",
            flush=True,
        )
        try:
            source_reports.append(
                build_source(
                    source,
                    in_dir=in_dir,
                    out_dir=out_dir,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    target_fps=float(args.target_fps),
                    policy=policy,
                    scaling_mode=str(args.scaling_mode),
                    name_map=name_map,
                    overwrite=bool(args.overwrite),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "source_id": source_id,
                    "source": str(source),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[OFFICIAL-SMPL REJECT] {source_id}: {exc}", flush=True)

    ok_sources = [row for row in source_reports if bool(row.get("ok", False))]
    recording_uids = {
        str(row["recording_uid"]) for row in ok_sources if row.get("recording_uid")
    }
    num_segments = sum(
        int(row.get("num_retained_segments", 0)) for row in ok_sources
    )
    all_ok = len(ok_sources) >= min_ok and len(recording_uids) >= 3 and num_segments >= 3

    summary = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_mode": "chang_e_official_smpl_source_aware",
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "smpl_manifest": str(manifest_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": smpl_manifest_sha256(manifest_path),
        "target_fps": float(args.target_fps),
        "scaling_mode": str(args.scaling_mode),
        "hard_cut_policy": asdict(policy),
        "num_discovered_files": len(files),
        "num_resolved_sources": len(resolved),
        "num_ok_sources": len(ok_sources),
        "num_failed_sources": len(failures),
        "num_retained_segments": int(num_segments),
        "num_recording_groups": len(recording_uids),
        "minimum_ok_sources": int(min_ok),
        "split_feasible": bool(len(recording_uids) >= 3),
        "all_ok": bool(all_ok),
        "policy": {
            "official_smpl_is_authoritative": True,
            "bvh_retarget_optimizer_used": False,
            "v241_direction_regularizer_used": False,
            "whole_source_event_style_anatomy_gate_used": False,
            "cartesian_catastrophic_hard_cut_before_event_slicing": True,
            "recording_uid_preserved_across_segments": True,
            "event_level_anatomy_filter_remains_enabled": True,
            "final_generation_physical_contract_changed": False,
        },
        "sources": source_reports,
        "failures": failures,
    }
    for name in (
        "event_heading_retarget_cache_report.json",
        "anatomy_heading_retarget_cache_report.json",
        "retarget_clean_retarget_cache_report.json",
        "official_smpl_source_aware_cache_report.json",
    ):
        (out_dir / name).write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "num_ok_sources": len(ok_sources),
                "num_retained_segments": num_segments,
                "num_recording_groups": len(recording_uids),
                "minimum_ok_sources": min_ok,
                "all_ok": all_ok,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
