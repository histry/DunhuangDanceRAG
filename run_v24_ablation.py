#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from contracts.physical_quality import evaluate_source_physical_clean_audit
from motion_geometry.physical import SUPPORT_POLICY_SOURCE, motion_physical_metrics_np
from retargeting.build_cache import _build_bvh_source_reference_audit
from retargeting import bvh_solver as legacy

PRESETS = {
    "A": (0.10, 0.025),
    "B": (0.25, 0.050),
    "C": (0.50, 0.100),
}
DEFAULT_SOURCES = ("male_36pose_1", "male_drum_2")


def audit_output(root: Path, source: str, motion_path: Path, fps: float) -> dict:
    manifest = root / "assets/motion/bvh/sources.json"
    source_bvh = root / "assets/motion/bvh" / f"{source}.bvh"
    cfg = legacy.RetargetConfig.from_env()
    cfg.target_fps = float(fps)
    cfg.source_manifest_path = str(manifest)
    cfg.require_source_manifest = True
    reference, reference_contract = _build_bvh_source_reference_audit(
        source_bvh,
        cfg=cfg,
        source_manifest=manifest,
        strict_manifest=True,
    )
    motion = np.load(motion_path).astype(np.float32)
    comparison_bones = reference.get("unit_bone_comparison_bones")
    audit = motion_physical_metrics_np(
        motion,
        fps=float(fps),
        support_policy=SUPPORT_POLICY_SOURCE,
        source_comparison_bones=comparison_bones,
    )
    gate = evaluate_source_physical_clean_audit(
        audit,
        source_reference_audit=reference,
    )
    return {
        "gate": gate,
        "audit": audit,
        "source_reference": reference,
        "source_reference_contract": reference_contract,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("PROJECT_ROOT", "/home/disk/lsm/storage/DunhuangDanceRAG"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    ap.add_argument("--presets", nargs="+", choices=sorted(PRESETS), default=list(PRESETS))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--device", default=os.environ.get("SOURCE_RETARGET_DEVICE", "cuda"))
    ap.add_argument("--out-dir", default="output/source_contract_validation_v2/v24_ablation")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for preset in args.presets:
        vel_w, acc_w = PRESETS[preset]
        for source in args.sources:
            source_bvh = root / "assets/motion/bvh" / f"{source}.bvh"
            run_dir = out_root / preset
            run_dir.mkdir(parents=True, exist_ok=True)
            motion_path = run_dir / f"{source}.npy"
            report_path = run_dir / f"{source}.retarget.json"
            env = os.environ.copy()
            env["SOURCE_RETARGET_BONE_DIR_VEL_W"] = str(vel_w)
            env["SOURCE_RETARGET_BONE_DIR_ACC_W"] = str(acc_w)
            env["SOURCE_RETARGET_FPS"] = str(float(args.fps))
            cmd = [
                args.python,
                "-m",
                "retargeting.source_regularized_solver",
                "--input",
                str(source_bvh),
                "--output",
                str(motion_path),
                "--report",
                str(report_path),
                "--device",
                str(args.device),
            ]
            print("[RUN]", preset, source, "VEL=", vel_w, "ACC=", acc_w, flush=True)
            subprocess.run(cmd, cwd=root, env=env, check=True)
            # Audit under the same selected V2.4 environment.
            old_env = os.environ.copy()
            try:
                os.environ.update(env)
                validation = audit_output(root, source, motion_path, args.fps)
            finally:
                os.environ.clear()
                os.environ.update(old_env)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            audit = validation["audit"]
            gate = validation["gate"]
            row = {
                "preset": preset,
                "source": source,
                "velocity_weight": vel_w,
                "acceleration_weight": acc_w,
                "fit_rmse_p95_m": report.get("fit", {}).get("fit_rmse_p95_m"),
                "gate_ok": bool(gate.get("ok")),
                "gate_reasons": list(gate.get("reasons", [])),
                "unit_bone_joint_jerk_s3_p95": audit.get("unit_bone_joint_jerk_s3_p95"),
                "unit_bone_joint_jerk_s3_p99": audit.get("unit_bone_joint_jerk_s3_p99"),
                "unit_bone_joint_jerk_window_p95_max_s3": audit.get("unit_bone_joint_jerk_window_p95_max_s3"),
                "unit_bone_extremity_jerk_s3_p95": audit.get("unit_bone_extremity_jerk_s3_p95"),
                "unit_bone_extremity_jerk_s3_p99": audit.get("unit_bone_extremity_jerk_s3_p99"),
                "unit_bone_extremity_jerk_window_p95_max_s3": audit.get("unit_bone_extremity_jerk_window_p95_max_s3"),
                "foot_penetration_p001_m": audit.get("foot_penetration_p001_m"),
                "joint_rotation_step_rad_max": audit.get("joint_rotation_step_rad_max"),
                "motion": str(motion_path),
                "retarget_report": str(report_path),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    summary = {
        "schema": "source_relative_temporal_regularization_ablation_v1",
        "sources": list(args.sources),
        "presets": {key: {"velocity_weight": PRESETS[key][0], "acceleration_weight": PRESETS[key][1]} for key in args.presets},
        "rows": rows,
    }
    out = out_root / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SAVED]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
