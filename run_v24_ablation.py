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

# V2.4.1: Z proves that the harness reproduces the authoritative research
# solver before any source-direction term is enabled.  A/B/C are deliberately
# small auxiliary weights because the research solver already has SO(3)
# velocity/acceleration regularisation.
PRESETS = {
    "Z": (0.0, 0.0),
    "A": (0.005, 0.0005),
    "B": (0.010, 0.0010),
    "C": (0.020, 0.0020),
}
DEFAULT_SOURCES = ("male_36pose_1", "male_drum_2")

RESEARCH_CONFIG_KEYS = (
    "RETARGET_CLEAN_ITERATIONS",
    "RETARGET_CLEAN_LEARNING_RATE",
    "RETARGET_KEYPOINT_W",
    "RETARGET_ROOT_W",
    "RETARGET_ROOT_VEL_W",
    "RETARGET_POSE_PRIOR_W",
    "RETARGET_CLEAN_ROOT_ANCHOR_W",
    "RETARGET_CLEAN_ROOT_YAW_ANCHOR_MULT",
    "RETARGET_CLEAN_SO3_VEL_W",
    "RETARGET_CLEAN_SO3_ACC_W",
    "RETARGET_UPRIGHT_W",
    "RETARGET_HEAD_ORDER_W",
    "RETARGET_FEET_ORDER_W",
    "RETARGET_FLOOR_W",
    "RETARGET_GRAD_CLIP",
)


def _require_research_environment() -> dict:
    if str(os.environ.get("EXPERIMENT_CONFIG_LOADED", "")).strip() != "1":
        raise RuntimeError(
            "V2.4.1 ablation requires the authoritative research environment. "
            "Run: source configs/experiment.env"
        )
    if str(os.environ.get("EXPERIMENT_ACTIVE_PROFILE", "research")) != "research":
        raise RuntimeError(
            "V2.4.1 source-retarget ablation requires EXPERIMENT_PROFILE=research"
        )
    return {key: os.environ.get(key) for key in RESEARCH_CONFIG_KEYS}


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


def _row_from_validation(
    *,
    preset: str,
    source: str,
    vel_w: float,
    acc_w: float,
    motion_path: Path,
    report_path: Path,
    validation: dict,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = validation["audit"]
    gate = validation["gate"]
    direction_report = report.get("source_direction_regularization", {})
    return {
        "preset": preset,
        "source": source,
        "velocity_weight": vel_w,
        "acceleration_weight": acc_w,
        "fit_rmse_p95_m": report.get("fit", {}).get(
            "fit_rmse_p95_m", report.get("fit_rmse_p95_m")
        ),
        "gate_ok": bool(gate.get("ok")),
        "gate_reasons": list(gate.get("reasons", [])),
        "unit_bone_joint_jerk_s3_p95": audit.get("unit_bone_joint_jerk_s3_p95"),
        "unit_bone_joint_jerk_s3_p99": audit.get("unit_bone_joint_jerk_s3_p99"),
        "unit_bone_joint_jerk_window_p95_max_s3": audit.get(
            "unit_bone_joint_jerk_window_p95_max_s3"
        ),
        "unit_bone_extremity_jerk_s3_p95": audit.get(
            "unit_bone_extremity_jerk_s3_p95"
        ),
        "unit_bone_extremity_jerk_s3_p99": audit.get(
            "unit_bone_extremity_jerk_s3_p99"
        ),
        "unit_bone_extremity_jerk_window_p95_max_s3": audit.get(
            "unit_bone_extremity_jerk_window_p95_max_s3"
        ),
        "foot_penetration_p001_m": audit.get("foot_penetration_p001_m"),
        "foot_penetration_sustained_catastrophic_max_seconds": audit.get(
            "foot_penetration_sustained_catastrophic_max_seconds"
        ),
        "joint_rotation_step_rad_max": audit.get("joint_rotation_step_rad_max"),
        "joint_rotation_step_window_p95_max_rad": audit.get(
            "joint_rotation_step_window_p95_max_rad"
        ),
        "rotation_near_pi_step_ratio": audit.get("rotation_near_pi_step_ratio"),
        "joint_angular_acceleration_rps2_max": audit.get(
            "joint_angular_acceleration_rps2_max"
        ),
        "joint_angular_acceleration_window_p95_max_rps2": audit.get(
            "joint_angular_acceleration_window_p95_max_rps2"
        ),
        "research_solver": direction_report.get("solver"),
        "direction_regularization_enabled": direction_report.get("enabled"),
        "common_direct_mapped_bone_count": direction_report.get(
            "common_direct_mapped_bone_count"
        ),
        "motion": str(motion_path),
        "retarget_report": str(report_path),
    }


def _z_sanity(row: dict) -> tuple[bool, list[str]]:
    reasons = []
    near_pi = row.get("rotation_near_pi_step_ratio")
    if near_pi is not None and float(near_pi) > 0.0:
        reasons.append(f"rotation_near_pi_step_ratio={near_pi}")
    rot_max = row.get("joint_rotation_step_rad_max")
    if rot_max is not None and float(rot_max) > 1.20:
        reasons.append(f"joint_rotation_step_rad_max={rot_max} > 1.20")
    fit = row.get("fit_rmse_p95_m")
    if fit is not None and float(fit) > 0.14:
        reasons.append(f"fit_rmse_p95_m={fit} > 0.14")
    if row.get("research_solver") != "anatomy_retarget_research_so3":
        reasons.append(f"unexpected_solver={row.get('research_solver')}")
    if bool(row.get("direction_regularization_enabled")):
        reasons.append("Z_direction_regularization_must_be_disabled")
    if row.get("common_direct_mapped_bone_count") not in (21, None):
        reasons.append(
            "unexpected_common_direct_mapped_bone_count="
            f"{row.get('common_direct_mapped_bone_count')}"
        )
    return not reasons, reasons


def _attach_delta_vs_z(rows: list[dict]) -> None:
    baseline = {
        row["source"]: row for row in rows if row.get("preset") == "Z"
    }
    metrics = (
        "fit_rmse_p95_m",
        "unit_bone_joint_jerk_s3_p95",
        "unit_bone_joint_jerk_s3_p99",
        "unit_bone_joint_jerk_window_p95_max_s3",
        "unit_bone_extremity_jerk_s3_p95",
        "unit_bone_extremity_jerk_s3_p99",
        "unit_bone_extremity_jerk_window_p95_max_s3",
        "joint_rotation_step_rad_max",
    )
    for row in rows:
        z = baseline.get(row["source"])
        if z is None or row.get("preset") == "Z":
            continue
        delta = {}
        for key in metrics:
            zv = z.get(key)
            cv = row.get(key)
            if zv is None or cv is None:
                continue
            zf = float(zv)
            cf = float(cv)
            delta[key] = {
                "absolute": cf - zf,
                "relative": ((cf / zf) - 1.0) if abs(zf) > 1.0e-12 else None,
            }
        row["delta_vs_Z"] = delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.environ.get(
            "PROJECT_ROOT", "/home/disk/lsm/storage/DunhuangDanceRAG"
        ),
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    ap.add_argument(
        "--presets", nargs="+", choices=sorted(PRESETS), default=list(PRESETS)
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--device", default=os.environ.get("SOURCE_RETARGET_DEVICE", "cuda")
    )
    ap.add_argument(
        "--out-dir",
        default="output/source_contract_validation_v2/v241_ablation",
    )
    ap.add_argument(
        "--allow-nonresearch-env",
        action="store_true",
        help="Diagnostic escape hatch only; formal V2.4.1 ablation must not use it.",
    )
    args = ap.parse_args()

    research_config = (
        {key: os.environ.get(key) for key in RESEARCH_CONFIG_KEYS}
        if args.allow_nonresearch_env
        else _require_research_environment()
    )

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
            # Remove the obsolete V2.4 namespace so it cannot silently affect
            # this controlled V2.4.1 experiment.
            for old_key in (
                "SOURCE_RETARGET_BONE_DIR_VEL_W",
                "SOURCE_RETARGET_BONE_DIR_ACC_W",
                "SOURCE_RETARGET_BONE_DIR_VEL_BETA",
                "SOURCE_RETARGET_BONE_DIR_ACC_BETA",
                "SOURCE_RETARGET_BONE_DIR_REFERENCE_FPS",
            ):
                env.pop(old_key, None)
            env["RETARGET_CLEAN_BONE_DIR_VEL_W"] = str(vel_w)
            env["RETARGET_CLEAN_BONE_DIR_ACC_W"] = str(acc_w)
            env["SOURCE_RETARGET_FPS"] = str(float(args.fps))
            cmd = [
                args.python,
                "-m",
                "retargeting.anatomy_retarget",
                "--input",
                str(source_bvh),
                "--output",
                str(motion_path),
                "--report",
                str(report_path),
                "--device",
                str(args.device),
                "--allow_failed_contract",
            ]
            print(
                "[RUN]",
                preset,
                source,
                "VEL=",
                vel_w,
                "ACC=",
                acc_w,
                "solver=anatomy_retarget_research_so3",
                flush=True,
            )
            subprocess.run(cmd, cwd=root, env=env, check=True)

            old_env = os.environ.copy()
            try:
                os.environ.update(env)
                validation = audit_output(root, source, motion_path, args.fps)
            finally:
                os.environ.clear()
                os.environ.update(old_env)

            row = _row_from_validation(
                preset=preset,
                source=source,
                vel_w=vel_w,
                acc_w=acc_w,
                motion_path=motion_path,
                report_path=report_path,
                validation=validation,
            )
            if preset == "Z":
                z_ok, z_reasons = _z_sanity(row)
                row["z_baseline_sanity_ok"] = bool(z_ok)
                row["z_baseline_sanity_reasons"] = z_reasons
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    _attach_delta_vs_z(rows)
    z_rows = [row for row in rows if row.get("preset") == "Z"]
    z_ok = all(bool(row.get("z_baseline_sanity_ok")) for row in z_rows) if z_rows else None
    summary = {
        "schema": "source_relative_temporal_regularization_ablation_v2_4_1",
        "solver_contract": "anatomy_retarget_research_so3_only",
        "sources": list(args.sources),
        "presets": {
            key: {
                "velocity_weight": PRESETS[key][0],
                "acceleration_weight": PRESETS[key][1],
            }
            for key in args.presets
        },
        "research_environment": research_config,
        "z_baseline_sanity_ok": z_ok,
        "rows": rows,
    }
    out = out_root / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SAVED]", out)
    if z_rows and not z_ok:
        print(
            "[V2.4.1 STOP] Z baseline sanity failed; do not interpret A/B/C.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
