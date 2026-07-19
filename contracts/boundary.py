#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bidirectional tangent-space boundary risk and body-part masked merge."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from motion_geometry.rotations import (
    angular_acceleration_np,
    angular_velocity_np,
    matrix_to_rot6d_np,
    relative_rotvec_np,
    rot6d_to_matrix_np,
    so3_geodesic_np,
    tangent_blend_np,
)
from contracts.gravity import (
    EDGE_DIM,
    NUM_JOINTS,
    ROOT_X_IDX,
    ROOT_Y_IDX,
    ROOT_Z_IDX,
    ROT6D_START,
    ROT6D_END,
)

SCHEMA = "v46_53_bidirectional_tangent_boundary_v1"
BODY_PARTS: Dict[str, Tuple[int, ...]] = {
    "root_torso": (0, 3, 6, 9, 12, 15),
    "left_arm": (13, 16, 18, 20, 22),
    "right_arm": (14, 17, 19, 21, 23),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
}


def _env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {"1", "true", "yes", "on"}


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


def _motion(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[1] < EDGE_DIM:
        raise ValueError(f"Expected [T,{EDGE_DIM}], got {a.shape}")
    return a[:, :EDGE_DIM]


def _sqrt_psd(m: np.ndarray) -> np.ndarray:
    x = 0.5 * (np.asarray(m, dtype=np.float64) + np.asarray(m, dtype=np.float64).T)
    val, vec = np.linalg.eigh(x)
    return (vec * np.sqrt(np.maximum(val, 0.0))[None]) @ vec.T


def bures_wasserstein_gaussian(mu0: np.ndarray, cov0: np.ndarray, mu1: np.ndarray, cov1: np.ndarray) -> float:
    dm = float(np.sum((np.asarray(mu0) - np.asarray(mu1)) ** 2))
    s0 = _sqrt_psd(cov0)
    middle = _sqrt_psd(s0 @ cov1 @ s0)
    dc = float(np.trace(cov0 + cov1 - 2.0 * middle))
    return float(math.sqrt(max(0.0, dm + dc)))


def _window_features(motion: np.ndarray, from_end: bool, frames: int, fps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _motion(motion)
    n = max(3, min(int(frames), len(x)))
    clip = x[-n:] if from_end else x[:n]
    r = rot6d_to_matrix_np(clip[:, ROT6D_START:ROT6D_END].reshape(len(clip), NUM_JOINTS, 6))
    ref = r[-1] if from_end else r[0]
    tangent = relative_rotvec_np(ref[None], r)
    omega = angular_velocity_np(r, fps=fps)
    alpha = angular_acceleration_np(r, fps=fps)
    if len(omega) < len(r):
        omega = np.concatenate([omega, omega[-1:]], axis=0) if len(omega) else np.zeros((len(r), NUM_JOINTS, 3), np.float32)
    if len(alpha) < len(r):
        pad = np.repeat(alpha[-1:], len(r) - len(alpha), axis=0) if len(alpha) else np.zeros((len(r), NUM_JOINTS, 3), np.float32)
        alpha = np.concatenate([alpha, pad], axis=0) if len(alpha) else pad
    return tangent.astype(np.float32), omega[: len(r)].astype(np.float32), alpha[: len(r)].astype(np.float32)


def transition_multiscale_risk(
    previous: np.ndarray,
    bridge: np.ndarray,
    following: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, Any]:
    p = _motion(previous)
    f = _motion(following)
    b = _motion(bridge) if np.asarray(bridge).size else np.zeros((0, EDGE_DIM), np.float32)
    frames = _env_int("V46_53_TANGENT_WINDOW", 8)
    pt, pw, pa = _window_features(p, True, frames, fps)
    ft, fw, fa = _window_features(f, False, frames, fps)

    r_p = rot6d_to_matrix_np(p[-1, ROT6D_START:ROT6D_END].reshape(NUM_JOINTS, 6))
    r_f = rot6d_to_matrix_np(f[0, ROT6D_START:ROT6D_END].reshape(NUM_JOINTS, 6))
    pose_gap = so3_geodesic_np(r_p, r_f)
    omega_gap = np.linalg.norm(pw[-1] - fw[0], axis=-1)
    alpha_gap = np.linalg.norm(pa[-1] - fa[0], axis=-1)

    part_detail: Dict[str, Any] = {}
    part_scores = []
    for name, ids in BODY_PARTS.items():
        ids_a = np.asarray(ids, dtype=np.int64)
        left_feat = np.concatenate([pt[:, ids_a].mean(axis=1), pw[:, ids_a].mean(axis=1)], axis=-1)
        right_feat = np.concatenate([ft[:, ids_a].mean(axis=1), fw[:, ids_a].mean(axis=1)], axis=-1)
        mu0, mu1 = left_feat.mean(axis=0), right_feat.mean(axis=0)
        cov0 = np.cov(left_feat, rowvar=False) + np.eye(left_feat.shape[1]) * 1e-5
        cov1 = np.cov(right_feat, rowvar=False) + np.eye(right_feat.shape[1]) * 1e-5
        bw = bures_wasserstein_gaussian(mu0, cov0, mu1, cov1)
        pg = float(np.percentile(pose_gap[ids_a], 95))
        wg = float(np.percentile(omega_gap[ids_a], 95))
        ag = float(np.percentile(alpha_gap[ids_a], 95))
        score = (
            _env_float("V46_53_PART_POSE_W", 0.85) * pg
            + _env_float("V46_53_PART_OMEGA_W", 0.10) * wg
            + _env_float("V46_53_PART_ALPHA_W", 0.004) * ag
            + _env_float("V46_53_PART_BURES_W", 0.12) * bw
        )
        part_scores.append(score)
        part_detail[name] = {
            "pose_geodesic_p95_rad": pg,
            "omega_gap_p95_rad_s": wg,
            "alpha_gap_p95_rad_s2": ag,
            "tangent_bures_wasserstein": bw,
            "score": float(score),
        }

    root_idx = [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]
    pv = (p[-1, root_idx] - p[-2, root_idx]) * fps if len(p) > 1 else np.zeros(3, np.float32)
    fv = (f[1, root_idx] - f[0, root_idx]) * fps if len(f) > 1 else np.zeros(3, np.float32)
    root_velocity_gap = float(np.linalg.norm(pv - fv))
    root_y_gap = float(abs(p[-1, ROOT_Y_IDX] - f[0, ROOT_Y_IDX]))
    contact_gap = float(np.mean(np.abs(p[-1, :4] - f[0, :4])))

    # Bidirectional reversal term.  Opposite angular directions should receive a
    # high penalty, while mutually consistent or stationary boundaries remain
    # close to zero.
    wp = pw[-1]
    wf = fw[0]
    npw = np.linalg.norm(wp, axis=-1)
    nwf = np.linalg.norm(wf, axis=-1)
    cosine = np.sum(wp * wf, axis=-1) / np.maximum(npw * nwf, 1.0e-8)
    reversal = np.maximum(0.0, -cosine) * np.sqrt(npw * nwf)
    bidirectional = float(np.mean(reversal))
    score = (
        float(np.mean(part_scores))
        + _env_float("V46_53_ROOT_VEL_W", 0.35) * root_velocity_gap
        + _env_float("V46_53_ROOT_Y_W", 2.2) * root_y_gap
        + _env_float("V46_53_CONTACT_W", 0.55) * contact_gap
        + _env_float("V46_53_BIDIRECTIONAL_W", 0.08) * bidirectional
    )
    hard = bool(
        float(np.max(pose_gap)) > _env_float("V46_53_POSE_GAP_HARD_RAD", 2.75)
        or float(np.percentile(omega_gap, 95)) > _env_float("V46_53_OMEGA_GAP_HARD", 12.0)
        or root_y_gap > _env_float("V46_53_ROOT_Y_GAP_HARD_M", 0.34)
    )
    return {
        "schema": SCHEMA,
        "score": float(score),
        "hard_reject": hard,
        "pose_geodesic_mean_rad": float(np.mean(pose_gap)),
        "pose_geodesic_p95_rad": float(np.percentile(pose_gap, 95)),
        "pose_geodesic_max_rad": float(np.max(pose_gap)),
        "omega_gap_p95_rad_s": float(np.percentile(omega_gap, 95)),
        "alpha_gap_p95_rad_s2": float(np.percentile(alpha_gap, 95)),
        "root_velocity_gap_m_s": root_velocity_gap,
        "root_y_gap_m": root_y_gap,
        "contact_gap": contact_gap,
        "bidirectional_velocity_reversal": bidirectional,
        "bridge_frames": int(len(b)),
        "parts": part_detail,
    }


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if radius <= 0:
        return m
    return np.convolve(m.astype(np.int32), np.ones(radius * 2 + 1, np.int32), mode="same") > 0


def build_frame_joint_risk_mask(
    motion_ref: np.ndarray,
    seam_mask: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, np.ndarray | Dict[str, Any]]:
    x = _motion(motion_ref)
    t = len(x)
    seam = np.asarray(seam_mask)
    if seam.ndim == 1:
        seam_f = seam > 0.01
    else:
        seam_f = seam.reshape(t, -1).max(axis=1) > 0.01
    seam_f = _dilate(seam_f, _env_int("V46_53_MASK_DILATE", 3))

    r = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_END].reshape(t, NUM_JOINTS, 6))
    omega = angular_velocity_np(r, fps=fps)
    alpha = angular_acceleration_np(r, fps=fps)
    w = np.zeros((t, NUM_JOINTS), np.float32)
    a = np.zeros((t, NUM_JOINTS), np.float32)
    if len(omega):
        w[1:] = np.linalg.norm(omega, axis=-1)
    if len(alpha):
        a[2:] = np.linalg.norm(alpha, axis=-1)

    def robust_z(v: np.ndarray) -> np.ndarray:
        med = np.median(v, axis=0, keepdims=True)
        mad = np.median(np.abs(v - med), axis=0, keepdims=True) + 1e-5
        return np.maximum(0.0, (v - med) / (1.4826 * mad))

    risk = 0.55 * np.clip(robust_z(w) / 5.0, 0.0, 1.0) + 0.45 * np.clip(robust_z(a) / 5.0, 0.0, 1.0)
    risk *= seam_f[:, None].astype(np.float32)
    minimum = _env_float("V46_53_MASK_MIN_ON_SEAM", 0.18)
    risk[seam_f] = np.maximum(risk[seam_f], minimum)

    # Promote coherent body-part masks so a dangerous wrist does not leave its
    # elbow/shoulder frozen while still preserving unrelated limbs.
    for ids in BODY_PARTS.values():
        ids_a = np.asarray(ids, dtype=np.int64)
        part = np.max(risk[:, ids_a], axis=1, keepdims=True)
        risk[:, ids_a] = np.maximum(risk[:, ids_a], part * _env_float("V46_53_PART_PROPAGATION", 0.65))

    root_v = np.zeros((t, 3), np.float32)
    root_a = np.zeros((t, 3), np.float32)
    if t > 1:
        root_v[1:] = np.diff(x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]], axis=0) * fps
    if t > 2:
        root_a[2:] = np.diff(root_v[1:], axis=0) * fps
    root_score = np.clip(np.linalg.norm(root_a, axis=-1) / _env_float("V46_53_ROOT_ACC_MASK_SCALE", 8.0), 0.0, 1.0)
    root_score *= seam_f.astype(np.float32)
    contact_score = np.zeros(t, np.float32)
    if t > 1:
        contact_score[1:] = np.max(np.abs(np.diff(np.clip(x[:, :4], 0, 1), axis=0)), axis=-1)
    contact_score = np.maximum(contact_score * seam_f, seam_f.astype(np.float32) * 0.25)

    return {
        "joint": np.clip(risk, 0.0, 1.0).astype(np.float32),
        "root": np.clip(root_score, 0.0, 1.0).astype(np.float32),
        "contact": np.clip(contact_score, 0.0, 1.0).astype(np.float32),
        "frame": seam_f.astype(np.float32),
        "report": {
            "schema": SCHEMA,
            "editable_frame_ratio": float(seam_f.mean()),
            "editable_joint_ratio": float((risk > 0.25).mean()),
            "mean_joint_weight": float(risk.mean()),
            "max_joint_weight": float(risk.max()),
        },
    }


def tangent_masked_merge(
    reference: np.ndarray,
    proposal: np.ndarray,
    mask: Mapping[str, np.ndarray],
) -> np.ndarray:
    ref = _motion(reference)
    pro = _motion(proposal)
    if ref.shape != pro.shape:
        raise ValueError(f"Reference/proposal shape mismatch: {ref.shape} vs {pro.shape}")
    out = ref.copy()
    joint_w = np.asarray(mask["joint"], dtype=np.float32)
    root_w = np.asarray(mask["root"], dtype=np.float32)[:, None]
    contact_w = np.asarray(mask["contact"], dtype=np.float32)[:, None]

    out[:, :4] = np.where(contact_w >= 0.5, np.clip(pro[:, :4], 0.0, 1.0), np.clip(ref[:, :4], 0.0, 1.0))
    out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] = (
        ref[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
        + root_w * (pro[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - ref[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]])
    )
    rr = rot6d_to_matrix_np(ref[:, ROT6D_START:ROT6D_END].reshape(len(ref), NUM_JOINTS, 6))
    rp = rot6d_to_matrix_np(pro[:, ROT6D_START:ROT6D_END].reshape(len(pro), NUM_JOINTS, 6))
    merged = tangent_blend_np(rr, rp, joint_w)
    out[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(merged).reshape(len(out), -1)
    return out.astype(np.float32)


def audit_motion(motion: np.ndarray, fps: float = 30.0) -> Dict[str, Any]:
    x = _motion(motion)
    r = rot6d_to_matrix_np(x[:, ROT6D_START:ROT6D_END].reshape(len(x), NUM_JOINTS, 6))
    w = angular_velocity_np(r, fps=fps)
    a = angular_acceleration_np(r, fps=fps)
    root_v = np.diff(x[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]], axis=0) * fps if len(x) > 1 else np.zeros((0, 3), np.float32)
    root_a = np.diff(root_v, axis=0) * fps if len(root_v) > 1 else np.zeros((0, 3), np.float32)
    return {
        "schema": SCHEMA,
        "frames": int(len(x)),
        "angular_velocity_p95_rad_s": float(np.percentile(np.linalg.norm(w, axis=-1), 95)) if w.size else 0.0,
        "angular_acceleration_p95_rad_s2": float(np.percentile(np.linalg.norm(a, axis=-1), 95)) if a.size else 0.0,
        "root_velocity_p95_m_s": float(np.percentile(np.linalg.norm(root_v, axis=-1), 95)) if root_v.size else 0.0,
        "root_acceleration_p95_m_s2": float(np.percentile(np.linalg.norm(root_a, axis=-1), 95)) if root_a.size else 0.0,
        "ok": bool(np.isfinite(x).all()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args(argv)
    x = np.load(args.input, allow_pickle=True)
    report = audit_motion(x, fps=args.fps)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
