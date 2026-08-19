#!/usr/bin/env python3
"""Offline V2.2 source-gate requalification for persisted rejected motions.

This script intentionally does not retarget or train anything.  It rebuilds the
aligned BVH source reference, derives the direct common source/target mapped
bone set, recomputes candidate source-observation metrics on exactly that set,
and reruns the V2.2 source physical-clean gate.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

import retargeting.bvh_solver as legacy
from contracts.physical_quality import evaluate_source_physical_clean_audit
from data_pipeline.chang_e_manifest import find_source_entry
from motion_geometry.physical import (
    SUPPORT_POLICY_SOURCE,
    motion_physical_metrics_np,
)
from retargeting.build_cache import _build_bvh_source_reference_audit

ROOT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        "/home/disk/lsm/storage/DunhuangDanceRAG",
    )
).resolve()
BVH_ROOT = ROOT / "assets/motion/bvh"
MANIFEST = BVH_ROOT / "sources.json"
VALIDATION_ROOT = Path(
    os.environ.get(
        "RETARGET_REQUALIFY_VALIDATION_ROOT",
        str(ROOT / "output/source_contract_validation_v2"),
    )
).resolve()
REJECTED_ROOT = Path(
    os.environ.get(
        "RETARGET_REQUALIFY_REJECTED_ROOT",
        str(VALIDATION_ROOT / "retarget_cache_rejected"),
    )
).resolve()
OUT = Path(
    os.environ.get(
        "RETARGET_REQUALIFY_OUT",
        str(VALIDATION_ROOT / "requalified_gate_v2_3.json"),
    )
).resolve()
FPS = float(os.environ.get("RETARGET_REQUALIFY_FPS", "30.0"))


def _sliding_tokens() -> set[str]:
    return {
        token.strip().lower()
        for token in os.environ.get(
            "CHANG_E_SLIDING_SUPPORT_TOKENS",
            "sogdian_whirl,sogdian,whirl,ribbon,lotus_steps,lotus,"
            "turning_travel,alternating_or_pivot_support",
        ).split(",")
        if token.strip()
    }


def main() -> int:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing source manifest: {MANIFEST}")
    if not REJECTED_ROOT.is_dir():
        raise FileNotFoundError(f"Missing rejected-motion directory: {REJECTED_ROOT}")
    if not np.isfinite(FPS) or FPS <= 0.0:
        raise ValueError(f"RETARGET_REQUALIFY_FPS must be positive, got {FPS!r}")

    cfg = legacy.RetargetConfig.from_env()
    cfg.target_fps = FPS
    cfg.source_manifest_path = str(MANIFEST)
    cfg.require_source_manifest = True
    sliding_tokens = _sliding_tokens()

    rows = []
    for motion_path in sorted(REJECTED_ROOT.glob("*.rejected.npy")):
        stem = motion_path.name.removesuffix(".rejected.npy")
        src = BVH_ROOT / f"{stem}.bvh"
        if not src.is_file():
            rows.append(
                {
                    "source": stem,
                    "ok": False,
                    "error": f"missing source BVH: {src}",
                    "rejected_motion": str(motion_path),
                }
            )
            continue

        motion = np.load(motion_path).astype(np.float32)
        meta = find_source_entry(src, path=MANIFEST, required=True) or {}
        semantic_text = " ".join(
            (
                str(src).lower(),
                json.dumps(meta, ensure_ascii=False, default=str).lower(),
            )
        )
        sliding = any(token in semantic_text for token in sliding_tokens)

        try:
            reference, reference_contract = _build_bvh_source_reference_audit(
                src,
                cfg=cfg,
                source_manifest=MANIFEST,
                strict_manifest=True,
            )
            comparison_bones = reference.get("unit_bone_comparison_bones")
            audit = motion_physical_metrics_np(
                motion,
                fps=FPS,
                sliding_support_eligible=(
                    np.ones(len(motion), dtype=bool) if sliding else None
                ),
                support_policy=SUPPORT_POLICY_SOURCE,
                source_comparison_bones=comparison_bones,
            )
            gate = evaluate_source_physical_clean_audit(
                audit,
                source_reference_audit=reference,
            )
            rows.append(
                {
                    "source": stem,
                    "ok": bool(gate["ok"]),
                    "reasons": list(gate.get("reasons", [])),
                    "gate": gate,
                    "physical_audit": audit,
                    "source_reference_physical_audit": reference,
                    "source_reference_contract": reference_contract,
                    "comparison_bones": list(comparison_bones or []),
                    "comparison_bone_count": len(comparison_bones or []),
                    "rejected_motion": str(motion_path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "source": stem,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "rejected_motion": str(motion_path),
                }
            )

    counter = Counter()
    for row in rows:
        for reason in row.get("reasons", []):
            counter[reason] += 1

    summary = {
        "schema": "retarget_clean_v2_3_requalification_diagnostic",
        "validation_root": str(VALIDATION_ROOT),
        "rejected_root": str(REJECTED_ROOT),
        "fps": float(FPS),
        "num_inputs": len(rows),
        "num_ok": sum(bool(row.get("ok")) for row in rows),
        "num_failed": sum(not bool(row.get("ok")) for row in rows),
        "failure_reason_counts": dict(counter.most_common()),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 88)
    print("V2.3 REJECTED-MOTION REQUALIFICATION")
    print("=" * 88)
    print("num_inputs =", summary["num_inputs"])
    print("num_ok     =", summary["num_ok"])
    print("num_failed =", summary["num_failed"])
    print("\nFAILURE REASON COUNTS")
    for reason, count in counter.most_common():
        print(f"{count:2d}  {reason}")
    print("\nPER SOURCE")
    for row in rows:
        suffix = row.get("reasons", row.get("error", ""))
        bones = row.get("comparison_bone_count", "-")
        print(
            f"{row['source']:24s}  "
            f"{'PASS' if row.get('ok') else 'FAIL'}  "
            f"bones={bones}  {suffix}"
        )
    print("\nSAVED:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
