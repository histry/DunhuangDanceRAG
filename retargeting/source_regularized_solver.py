#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entrypoint for V2.4.1 source-direction retargeting.

V2.4 incorrectly owned a second optimizer based on ``retargeting.bvh_solver``.
V2.4.1 removes that fork: all calls delegate to the authoritative
``retargeting.anatomy_retarget.retarget_bvh_research`` SO(3) research solver.

New authoritative environment variables:
    RETARGET_CLEAN_BONE_DIR_VEL_W
    RETARGET_CLEAN_BONE_DIR_ACC_W
    RETARGET_CLEAN_BONE_DIR_VEL_BETA
    RETARGET_CLEAN_BONE_DIR_ACC_BETA
    RETARGET_CLEAN_BONE_DIR_REFERENCE_FPS

The old ``SOURCE_RETARGET_BONE_DIR_*`` weight names are accepted only as
compatibility aliases when the new names are absent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

import retargeting.bvh_solver as _legacy
from retargeting import anatomy_retarget as _research

RetargetConfig = _legacy.RetargetConfig
BVHJoint = _legacy.BVHJoint
BVHMotion = _legacy.BVHMotion
PARENTS = _legacy.PARENTS
NUM_JOINTS = _legacy.NUM_JOINTS
parse_bvh = _legacy.parse_bvh
build_joint_mapping = _legacy.build_joint_mapping

SourceTemporalRegularization = _research.SourceDirectionRegularization
SourceDirectionRegularization = _research.SourceDirectionRegularization
common_direct_mapped_bone_children = _research.common_direct_mapped_bone_children
_bone_direction_derivative_losses = _research._bone_direction_derivative_losses

torch = _research.torch

_COMPAT_ENV = {
    "SOURCE_RETARGET_BONE_DIR_VEL_W": "RETARGET_CLEAN_BONE_DIR_VEL_W",
    "SOURCE_RETARGET_BONE_DIR_ACC_W": "RETARGET_CLEAN_BONE_DIR_ACC_W",
    "SOURCE_RETARGET_BONE_DIR_VEL_BETA": "RETARGET_CLEAN_BONE_DIR_VEL_BETA",
    "SOURCE_RETARGET_BONE_DIR_ACC_BETA": "RETARGET_CLEAN_BONE_DIR_ACC_BETA",
    "SOURCE_RETARGET_BONE_DIR_REFERENCE_FPS": "RETARGET_CLEAN_BONE_DIR_REFERENCE_FPS",
}


def _bridge_legacy_environment() -> list[str]:
    bridged = []
    for old_name, new_name in _COMPAT_ENV.items():
        if new_name in os.environ:
            continue
        value = str(os.environ.get(old_name, "")).strip()
        if not value:
            continue
        os.environ[new_name] = value
        bridged.append(f"{old_name}->{new_name}")
    return bridged


def retarget_bvh(path: str | Path, cfg: Optional[RetargetConfig] = None):
    """Delegate to the authoritative Retarget Clean SO(3) research solver."""
    bridged = _bridge_legacy_environment()
    motion, report = _research.retarget_bvh_research(path, cfg)
    report = dict(report)
    report["source_regularized_solver_compatibility"] = {
        "schema": "v2_4_1_research_solver_delegate",
        "delegated_solver": "retargeting.anatomy_retarget.retarget_bvh_research",
        "legacy_environment_aliases_applied": bridged,
        "owns_optimizer": False,
    }
    return motion, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--allow_failed_contract", action="store_true")
    args = ap.parse_args()

    cfg = RetargetConfig.from_env()
    if args.device:
        cfg.device = args.device
    if args.allow_failed_contract:
        os.environ["RETARGET_HARD_RETARGET_GATE"] = "0"

    motion, report = retarget_bvh(args.input, cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(motion, dtype=np.float32))
    rp = Path(args.report) if args.report else out.with_suffix(".retarget.json")
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "motion": str(out),
                "report": str(rp),
                "frames": int(len(motion)),
                "ok": bool(report.get("ok", False)),
                "delegated_solver": report[
                    "source_regularized_solver_compatibility"
                ]["delegated_solver"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
