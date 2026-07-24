#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Augment an Anatomy-Gated Event-RAG DB with intrinsic motion geometry.

The augmentation is schema-preserving: every existing array is retained.  New
arrays are event-wise and therefore remain compatible with the V46.38 AESD
copy/enrichment step and the V44/V45/V46 trainers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from motion_geometry.rotations import (
    angular_acceleration_np,
    angular_velocity_np,
    rot6d_to_matrix_np,
    so3_geodesic_np,
)
from contracts.gravity import (
    EDGE_DIM,
    NUM_JOINTS,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
    ROT6D_START,
    ROT6D_END,
    fk24_np,
)

SCHEMA = "v46_53_intrinsic_event_geometry_v3_physical_endpoints"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BODY_PARTS: Dict[str, Tuple[int, ...]] = {
    "root_torso": (0, 3, 6, 9, 12, 15),
    "left_arm": (13, 16, 18, 20, 22),
    "right_arm": (14, 17, 19, 21, 23),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
    "hands": (20, 21, 22, 23),
}
# The Gaussian-Wasserstein retrieval factor uses disjoint anatomical blocks.
# Hands remain represented by their parent arm blocks; keeping the overlapping
# ``hands`` summary above preserves the historical 112D descriptor contract.
GAUSSIAN_BODY_PARTS: Dict[str, Tuple[int, ...]] = {
    name: ids for name, ids in BODY_PARTS.items() if name != "hands"
}
GAUSSIAN_FEATURE_DIM = 8
POSTURE_ORDER = ("floor_pose", "kneeling", "deep_squat", "half_squat", "standing", "aerial")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _load_motion(path: str) -> np.ndarray:
    x = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] < EDGE_DIM:
        raise ValueError(f"Expected [T,{EDGE_DIM}] event, got {x.shape}: {path}")
    return x[:, :EDGE_DIM]


def _resolve_event_motion_path(raw_value: Any, db_path: Path) -> Path:
    raw = Path(str(raw_value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        PROJECT_ROOT / raw,
        db_path.parent / raw,
    ]
    checked: List[str] = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve event motion {raw_value!r} from Event-DB {db_path}; "
        f"checked={checked}"
    )


def _pad_edge(x: np.ndarray, length: int, from_end: bool = False) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros(x.shape[1:], dtype=np.float32)
    n = max(1, min(int(length), x.shape[0]))
    return np.median(x[-n:] if from_end else x[:n], axis=0).astype(np.float32)


def _stats(x: np.ndarray) -> List[float]:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(a)),
        float(np.std(a)),
        float(np.percentile(a, 95)),
        float(np.max(a)),
    ]


def _bodypart_gaussian_statistics(
    omega: np.ndarray,
    alpha: np.ndarray,
    *,
    shrinkage: Optional[float] = None,
    minimum_eigenvalue: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate stable body-part Gaussian dynamics in local SO(3) tangents.

    Each aligned frame contributes eight physical features per anatomical block:
    mean angular velocity (3), RMS angular speed (1), mean angular acceleration
    (3), and RMS angular acceleration (1).  Covariances are shrunk toward their
    diagonal and eigenvalue-floored, which is essential for short event clips.
    """

    w = np.asarray(omega, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64)
    if w.ndim != 3 or a.ndim != 3 or w.shape[1:] != (NUM_JOINTS, 3) or a.shape[1:] != (NUM_JOINTS, 3):
        raise ValueError(
            "body-part Gaussian statistics require omega/alpha shaped "
            f"[T,{NUM_JOINTS},3], got {w.shape} and {a.shape}"
        )
    aligned = min(max(len(w) - 1, 0), len(a))
    shrink = (
        _env_float("V46_53_GAUSSIAN_COV_SHRINKAGE", 0.20)
        if shrinkage is None
        else float(shrinkage)
    )
    floor = (
        _env_float("V46_53_GAUSSIAN_COV_EPS", 1.0e-4)
        if minimum_eigenvalue is None
        else float(minimum_eigenvalue)
    )
    if not np.isfinite(shrink) or not 0.0 <= shrink <= 1.0:
        raise ValueError("Gaussian covariance shrinkage must be in [0,1]")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("Gaussian covariance eigenvalue floor must be positive")

    means: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    sample_counts: List[int] = []
    for ids in GAUSSIAN_BODY_PARTS.values():
        joint_ids = np.asarray(ids, dtype=np.int64)
        if aligned > 0:
            part_w = w[1 : aligned + 1, joint_ids]
            part_a = a[:aligned, joint_ids]
            features = np.concatenate(
                [
                    np.mean(part_w, axis=1),
                    np.sqrt(np.mean(np.sum(part_w * part_w, axis=-1), axis=1))[
                        :, None
                    ],
                    np.mean(part_a, axis=1),
                    np.sqrt(np.mean(np.sum(part_a * part_a, axis=-1), axis=1))[
                        :, None
                    ],
                ],
                axis=-1,
            )
        else:
            # Degenerate clips remain representable but carry an explicit sample
            # count of zero and an isotropic floor covariance.
            features = np.zeros((1, GAUSSIAN_FEATURE_DIM), dtype=np.float64)

        mean = np.mean(features, axis=0)
        centered = features - mean[None]
        denominator = max(len(features) - 1, 1)
        covariance = centered.T @ centered / float(denominator)
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - shrink) * covariance + shrink * diagonal
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, floor)
        covariance = (eigenvectors * eigenvalues[None]) @ eigenvectors.T

        means.append(mean.astype(np.float32))
        covariances.append(covariance.astype(np.float32))
        sample_counts.append(int(aligned))

    return (
        np.stack(means).astype(np.float32),
        np.stack(covariances).astype(np.float32),
        np.asarray(sample_counts, dtype=np.int32),
    )


def _hash_onehot(value: Any, width: int) -> np.ndarray:
    out = np.zeros(int(width), dtype=np.float32)
    token = str(value or "unknown").strip().lower()
    h = int(hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    out[h % int(width)] = 1.0
    return out


def _posture_onehot(value: Any) -> np.ndarray:
    out = np.zeros(len(POSTURE_ORDER), dtype=np.float32)
    token = str(value or "standing")
    out[POSTURE_ORDER.index(token) if token in POSTURE_ORDER else 4] = 1.0
    return out


def _geometry_descriptor(
    motion: np.ndarray,
    posture: str,
    family: str,
    stage_role: str,
    fps: float,
    edge_frames: int,
) -> Dict[str, np.ndarray | float]:
    x = np.asarray(motion, dtype=np.float32)
    t = len(x)
    local = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_END].reshape(t, NUM_JOINTS, 6))
    omega = angular_velocity_np(local, fps=fps)
    alpha = angular_acceleration_np(local, fps=fps)
    gaussian_mean, gaussian_covariance, gaussian_samples = (
        _bodypart_gaussian_statistics(omega, alpha)
    )
    joints = fk24_np(x)

    root = x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    root_v = np.diff(root, axis=0) * float(fps) if t > 1 else np.zeros((0, 3), np.float32)
    root_a = np.diff(root_v, axis=0) * float(fps) if len(root_v) > 1 else np.zeros((0, 3), np.float32)

    desc: List[float] = []
    desc.extend(_stats(np.linalg.norm(root_v, axis=-1)))
    desc.extend(_stats(np.linalg.norm(root_a, axis=-1)))
    desc.extend(_stats(root[:, 1]))

    part_flow = []
    for _, ids in BODY_PARTS.items():
        ids_a = np.asarray(ids, dtype=np.int64)
        ws = np.linalg.norm(omega[:, ids_a], axis=-1) if len(omega) else np.zeros((0, len(ids)), np.float32)
        aa = np.linalg.norm(alpha[:, ids_a], axis=-1) if len(alpha) else np.zeros((0, len(ids)), np.float32)
        wstat = _stats(ws)
        astat = _stats(aa)
        desc.extend(wstat)
        desc.extend(astat)
        part_flow.append(wstat + astat)

    # Long-range trajectory evidence: root, wrists, ankles and hands.
    trajectory_ids = (0, 7, 8, 20, 21, 22, 23)
    trajectory = []
    for jid in trajectory_ids:
        p = joints[:, jid]
        step = np.linalg.norm(np.diff(p, axis=0), axis=-1) if t > 1 else np.zeros(0, np.float32)
        path_len = float(step.sum())
        displacement = float(np.linalg.norm(p[-1] - p[0])) if t else 0.0
        trajectory.extend([path_len, displacement])
    desc.extend(trajectory)

    contacts = np.clip(x[:, :4], 0.0, 1.0)
    desc.extend(np.mean(contacts, axis=0).tolist())
    desc.extend(
        (
            np.mean(np.abs(np.diff(contacts, axis=0)), axis=0)
            * float(fps)
        ).tolist()
        if t > 1
        else [0.0] * 4
    )

    desc.extend(_posture_onehot(posture).tolist())
    desc.extend(_hash_onehot(family, 16).tolist())
    desc.extend(_hash_onehot(stage_role, 8).tolist())

    entry_omega = _pad_edge(omega, edge_frames, False)
    exit_omega = _pad_edge(omega, edge_frames, True)
    entry_alpha = _pad_edge(alpha, edge_frames, False)
    exit_alpha = _pad_edge(alpha, edge_frames, True)
    entry_root_velocity = _pad_edge(root_v, edge_frames, False)
    exit_root_velocity = _pad_edge(root_v, edge_frames, True)

    # Intrinsic structure quality.  This is not a hard anatomy gate; it is a
    # continuous ranking prior that penalizes excessive high-order dynamics.
    omega_p95 = float(np.percentile(np.linalg.norm(omega, axis=-1), 95)) if omega.size else 0.0
    alpha_p95 = float(np.percentile(np.linalg.norm(alpha, axis=-1), 95)) if alpha.size else 0.0
    jerk_scale = _env_float("V46_53_STRUCTURE_ALPHA_SCALE", 180.0)
    speed_scale = _env_float("V46_53_STRUCTURE_OMEGA_SCALE", 8.0)
    structure_quality = float(np.clip(math.exp(-omega_p95 / max(speed_scale, 1e-6) - alpha_p95 / max(jerk_scale, 1e-6)), 0.0, 1.0))

    return {
        "descriptor": np.asarray(desc, dtype=np.float32),
        # Dedicated paper-two routing endpoints.  They are projected SO(3)
        # matrices, so the graph never compares unconstrained Rot6D channels.
        "entry_rotation_matrix": local[0].astype(np.float32),
        "exit_rotation_matrix": local[-1].astype(np.float32),
        "entry_omega": entry_omega.astype(np.float32),
        "exit_omega": exit_omega.astype(np.float32),
        "entry_alpha": entry_alpha.astype(np.float32),
        "exit_alpha": exit_alpha.astype(np.float32),
        "entry_root_velocity_mps": entry_root_velocity.astype(np.float32),
        "exit_root_velocity_mps": exit_root_velocity.astype(np.float32),
        "part_flow": np.asarray(part_flow, dtype=np.float32),
        "bodypart_gaussian_mean": gaussian_mean,
        "bodypart_gaussian_covariance": gaussian_covariance,
        "bodypart_gaussian_samples": gaussian_samples,
        "structure_quality": structure_quality,
        "omega_p95": omega_p95,
        "alpha_p95": alpha_p95,
    }


def _event_array(payload: Mapping[str, Any], key: str, n: int, default: Any) -> np.ndarray:
    if key in payload:
        arr = np.asarray(payload[key])
        if arr.ndim >= 1 and arr.shape[0] == n:
            return arr
    return np.asarray([default] * n, dtype=object)


def _diagonal_w2_barycenter(
    features: np.ndarray,
    sources: np.ndarray,
    quality: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quality-weighted diagonal Gaussian W2 barycenter across source groups."""
    x = np.asarray(features, dtype=np.float64)
    src = np.asarray(sources, dtype=object)
    q = np.clip(np.asarray(quality, dtype=np.float64), 1.0e-3, 1.0)
    unique = np.unique(src)
    means, stds, weights = [], [], []
    source_mean_map: Dict[str, np.ndarray] = {}
    for token in unique:
        idx = np.where(src == token)[0]
        w = q[idx]
        w = w / max(w.sum(), 1.0e-8)
        mu = (x[idx] * w[:, None]).sum(axis=0)
        var = ((x[idx] - mu) ** 2 * w[:, None]).sum(axis=0)
        sd = np.sqrt(np.maximum(var, 1.0e-6))
        means.append(mu)
        stds.append(sd)
        weights.append(float(q[idx].sum()))
        source_mean_map[str(token)] = mu
    sw = np.asarray(weights, dtype=np.float64)
    sw = sw / max(sw.sum(), 1.0e-8)
    bary_mean = np.sum(np.stack(means) * sw[:, None], axis=0)
    # For diagonal Gaussians, the W2 barycenter standard deviation is the
    # weighted average standard deviation.
    bary_std = np.sum(np.stack(stds) * sw[:, None], axis=0)
    bary_std = np.maximum(bary_std, 1.0e-4)
    shared = (x - bary_mean[None]) / bary_std[None]
    private = np.stack([(source_mean_map[str(s)] - bary_mean) / bary_std for s in src])
    return (
        bary_mean.astype(np.float32),
        bary_std.astype(np.float32),
        shared.astype(np.float32),
        private.astype(np.float32),
    )


def _database_fps(payload: Mapping[str, Any]) -> Optional[float]:
    if "canonical_fps" not in payload:
        return None
    values = np.asarray(payload["canonical_fps"], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    unique = np.unique(np.round(values, decimals=6))
    if unique.size != 1 or float(unique[0]) <= 0.0:
        raise RuntimeError(
            "Event-DB canonical_fps must contain exactly one positive rate; "
            f"found={unique.tolist()}"
        )
    return float(unique[0])


def augment_database(
    db_path: Path,
    audit_path: Optional[Path] = None,
    fps: Optional[float] = None,
) -> Dict[str, Any]:
    with np.load(db_path, allow_pickle=True) as data:
        payload: Dict[str, Any] = {k: data[k] for k in data.files}
    declared_fps = _database_fps(payload)
    if fps is None:
        if declared_fps is None:
            raise RuntimeError(
                f"Event-DB has no canonical_fps contract: {db_path}"
            )
        fps = declared_fps
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    if declared_fps is not None and abs(declared_fps - fps) > 1.0e-6:
        raise RuntimeError(
            "Intrinsic geometry FPS mismatch: "
            f"Event-DB={declared_fps}, requested={fps}, path={db_path}"
        )
    paths = np.asarray(payload["paths"], dtype=object)
    n = len(paths)
    posture = _event_array(payload, "posture_mode", n, "standing")
    family = _event_array(payload, "event_families", n, "unknown")
    stage = _event_array(payload, "motion_stage_roles", n, "unknown")
    sources = _event_array(payload, "source_uids", n, "unknown")
    anatomy_q = np.asarray(payload.get("anatomy_quality", np.ones(n, np.float32) * 0.5), dtype=np.float32)
    old_q = np.asarray(payload.get("event_quality_scores", np.ones(n, np.float32) * 0.5), dtype=np.float32)

    edge_frames_30fps = _env_int("V46_53_EVENT_EDGE_FRAMES", 6)
    edge_frames = max(1, int(round(edge_frames_30fps * fps / 30.0)))
    (
        rows,
        w0,
        w1,
        a0,
        a1,
        rv0,
        rv1,
        r0,
        r1,
        parts,
        gaussian_means,
        gaussian_covariances,
        gaussian_sample_counts,
        structure_q,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    diagnostics: List[dict] = []
    for i, path in enumerate(paths.tolist()):
        resolved_path = _resolve_event_motion_path(path, db_path)
        motion = _load_motion(str(resolved_path))
        item = _geometry_descriptor(
            motion,
            posture=str(posture[i]),
            family=str(family[i]),
            stage_role=str(stage[i]),
            fps=float(fps),
            edge_frames=edge_frames,
        )
        rows.append(item["descriptor"])
        w0.append(item["entry_omega"])
        w1.append(item["exit_omega"])
        a0.append(item["entry_alpha"])
        a1.append(item["exit_alpha"])
        rv0.append(item["entry_root_velocity_mps"])
        rv1.append(item["exit_root_velocity_mps"])
        r0.append(item["entry_rotation_matrix"])
        r1.append(item["exit_rotation_matrix"])
        parts.append(item["part_flow"])
        gaussian_means.append(item["bodypart_gaussian_mean"])
        gaussian_covariances.append(item["bodypart_gaussian_covariance"])
        gaussian_sample_counts.append(item["bodypart_gaussian_samples"])
        structure_q.append(float(item["structure_quality"]))
        diagnostics.append({
            "event_id": int(i),
            "path": str(resolved_path),
            "omega_p95": float(item["omega_p95"]),
            "alpha_p95": float(item["alpha_p95"]),
            "structure_quality": float(item["structure_quality"]),
        })

    geometry = np.stack(rows).astype(np.float32)
    mean = geometry.mean(axis=0, keepdims=True)
    std = geometry.std(axis=0, keepdims=True) + 1.0e-6
    geometry_z = ((geometry - mean) / std).astype(np.float32)
    structure_q_arr = np.asarray(structure_q, dtype=np.float32)
    combined_q = np.clip(
        np.power(np.clip(old_q, 1e-4, 1.0), 0.45)
        * np.power(np.clip(anatomy_q, 1e-4, 1.0), 0.35)
        * np.power(np.clip(structure_q_arr, 1e-4, 1.0), 0.20),
        0.0,
        1.0,
    ).astype(np.float32)
    bary_mean, bary_std, shared, private = _diagonal_w2_barycenter(geometry_z, sources, combined_q)

    payload.update({
        "v46_53_geometry_schema_version": np.asarray(SCHEMA, dtype=object),
        "v46_53_geometry_fps": np.asarray(fps, dtype=np.float32),
        "v46_53_geometry_desc": geometry,
        "v46_53_geometry_desc_z": geometry_z,
        "v46_53_geometry_mean": mean.astype(np.float32),
        "v46_53_geometry_std": std.astype(np.float32),
        "v46_53_entry_omega": np.stack(w0).astype(np.float32),
        "v46_53_exit_omega": np.stack(w1).astype(np.float32),
        "v46_53_entry_alpha": np.stack(a0).astype(np.float32),
        "v46_53_exit_alpha": np.stack(a1).astype(np.float32),
        "v46_53_entry_root_velocity_mps": np.stack(rv0).astype(np.float32),
        "v46_53_exit_root_velocity_mps": np.stack(rv1).astype(np.float32),
        "v46_55_route_geometry_schema_version": np.asarray(
            "v46_55_so3_product_event_edge_state_v1", dtype=object
        ),
        "v46_55_entry_rotation_matrix": np.stack(r0).astype(np.float32),
        "v46_55_exit_rotation_matrix": np.stack(r1).astype(np.float32),
        "v46_53_bodypart_flow": np.stack(parts).astype(np.float32),
        "v46_53_bodypart_gaussian_schema_version": np.asarray(
            "v46_53_bodypart_so3_gaussian_bw_v1", dtype=object
        ),
        "v46_53_bodypart_gaussian_mean": np.stack(gaussian_means).astype(
            np.float32
        ),
        "v46_53_bodypart_gaussian_covariance": np.stack(
            gaussian_covariances
        ).astype(np.float32),
        "v46_53_bodypart_gaussian_samples": np.stack(
            gaussian_sample_counts
        ).astype(np.int32),
        "v46_53_structure_quality": structure_q_arr,
        "v46_53_combined_quality": combined_q,
        "v46_53_w2_barycenter_mean": bary_mean,
        "v46_53_w2_barycenter_std": bary_std,
        "v46_53_shared_embedding": shared,
        "v46_53_source_private_embedding": private,
        # Existing routing already consumes this key.  Updating it lets the old
        # V44/V46 path benefit from anatomy and intrinsic dynamics immediately.
        "event_quality_scores": combined_q,
    })

    backup = db_path.with_name(db_path.stem + ".pre_v46_53_geometry.npz")
    if not backup.exists():
        shutil.copy2(db_path, backup)
    np.savez_compressed(db_path, **payload)

    report = {
        "schema": SCHEMA,
        "db": str(db_path),
        "backup": str(backup),
        "num_events": int(n),
        "geometry_dim": int(geometry.shape[1]),
        "fps": fps,
        "edge_frames": int(edge_frames),
        "edge_window_seconds": float(edge_frames / fps),
        "route_edge_geometry": {
            "schema": "v46_55_so3_product_event_edge_state_v1",
            "rotation_shape": [24, 3, 3],
            "distance": "RMS product SO(3) geodesic",
        },
        "body_parts": list(BODY_PARTS),
        "gaussian_body_parts": list(GAUSSIAN_BODY_PARTS),
        "gaussian_feature_dim": int(GAUSSIAN_FEATURE_DIM),
        "gaussian_covariance": {
            "geometry": "Bures-Wasserstein SPD",
            "shrinkage": _env_float(
                "V46_53_GAUSSIAN_COV_SHRINKAGE", 0.20
            ),
            "minimum_eigenvalue": _env_float(
                "V46_53_GAUSSIAN_COV_EPS", 1.0e-4
            ),
        },
        "quality": {
            "min": float(combined_q.min()),
            "median": float(np.median(combined_q)),
            "max": float(combined_q.max()),
        },
        "structure_quality": {
            "min": float(structure_q_arr.min()),
            "median": float(np.median(structure_q_arr)),
            "max": float(structure_q_arr.max()),
        },
        "w2_barycenter": "quality-weighted diagonal Gaussian Wasserstein-2 barycenter",
        "events": diagnostics,
        "ok": True,
    }
    target = audit_path or db_path.with_name("events.v46_53_geometry.audit.json")
    target.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--audit", default=None)
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional assertion; by default FPS is read from Event-DB canonical_fps.",
    )
    args = ap.parse_args(argv)
    report = augment_database(
        Path(args.db),
        Path(args.audit) if args.audit else None,
        fps=args.fps,
    )
    print(json.dumps({k: report[k] for k in ("schema", "num_events", "geometry_dim", "quality", "ok")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
