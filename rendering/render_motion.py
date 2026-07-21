#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render EDGE 151D motion with a mandatory gravity-contract preflight.

This replaces the repository root rendering/render_motion.py.  It keeps the corrected
column-concatenated Rot6D convention and prevents a scientifically invalid
sideways/floating sequence from being rendered as a successful result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from vis import SMPLSkeleton, skeleton_render
from contracts.gravity import (
    GravityThresholds,
    evaluate_gravity_contract,
    gravity_metrics_np,
    rot6d_to_matrix_np,
)


def as_batch(arr):
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2 and x.shape[-1] == 151:
        return x[None]
    if x.ndim == 3 and x.shape[-1] == 151:
        return x
    raise ValueError(f"Expected [T,151] or [B,T,151], got {x.shape}")


def sanitize_contacts(c, T):
    c = np.asarray(c, dtype=np.float32)
    if c.ndim != 2:
        return np.zeros((T, 4), dtype=np.float32)
    if c.shape[1] < 4:
        c = np.pad(c, ((0, 0), (0, 4 - c.shape[1])))
    c = c[:, :4]
    if len(c) < T:
        last = c[-1:] if len(c) else np.zeros((1, 4), np.float32)
        c = np.concatenate([c, np.repeat(last, T - len(c), axis=0)], axis=0)
    return c[:T]


def render_one(motion, audio, output, camera_mode, smooth, fps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = len(motion)
    pos = torch.as_tensor(motion[:, 4:7], dtype=torch.float32, device=device).unsqueeze(0)
    rot6 = motion[:, 7:151].reshape(T, 24, 6)
    matrices = rot6d_to_matrix_np(rot6)
    axis_angle = matrix_to_axis_angle(
        torch.as_tensor(matrices, dtype=torch.float32, device=device).unsqueeze(0)
    )
    poses = SMPLSkeleton(device=device).forward(axis_angle, pos).detach().cpu().numpy()[0]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    skeleton_render(
        poses=poses,
        epoch=Path(output).stem,
        out=str(Path(output).parent),
        name=[audio],
        sound=True,
        stitch=False,
        contact=sanitize_contacts(motion[:, :4], T),
        render=True,
        camera_mode=camera_mode,
        output_path=output,
        render_smooth_window=max(1, int(smooth)),
        fps=float(fps),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--camera_mode", choices=["fixed", "follow"], default="fixed")
    ap.add_argument("--render_smooth_window", type=int, default=1)
    ap.add_argument(
        "--fps",
        type=float,
        default=float(os.environ.get("V46_51_FPS", 30.0)),
    )
    ap.add_argument("--gravity_audit_json", default=None)
    ap.add_argument("--allow_invalid_gravity", action="store_true")
    args = ap.parse_args()

    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError(f"--fps must be positive and finite, got {args.fps!r}")

    if not Path(args.motion).is_file():
        raise FileNotFoundError(args.motion)
    if not Path(args.audio).is_file():
        raise FileNotFoundError(args.audio)

    batch = as_batch(np.load(args.motion, allow_pickle=True))
    reports = []
    for i, motion in enumerate(batch):
        metrics = gravity_metrics_np(motion)
        ok, reasons = evaluate_gravity_contract(metrics, GravityThresholds())
        reports.append({"batch": i, "ok": ok, "reasons": reasons, **metrics})
        print(json.dumps(reports[-1], ensure_ascii=False, indent=2))
        if not ok and not args.allow_invalid_gravity:
            raise RuntimeError(
                "Render blocked by V46.49 gravity contract: " + " | ".join(reasons)
            )

    audit_path = (
        Path(args.gravity_audit_json)
        if args.gravity_audit_json
        else Path(args.output).with_name(
            Path(args.output).stem + ".render_gravity.json"
        )
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(batch) == 1:
        render_one(
            batch[0],
            args.audio,
            args.output,
            args.camera_mode,
            args.render_smooth_window,
            args.fps,
        )
    else:
        stem, ext = os.path.splitext(args.output)
        for i, motion in enumerate(batch):
            render_one(
                motion,
                args.audio,
                f"{stem}_b{i:02d}{ext or '.mp4'}",
                args.camera_mode,
                args.render_smooth_window,
                args.fps,
            )
    print(f"[DONE] rendered {len(batch)} sequence(s); audit={audit_path}")


if __name__ == "__main__":
    main()
