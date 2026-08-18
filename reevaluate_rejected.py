#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

import retargeting.bvh_solver as legacy
from contracts.physical_quality import evaluate_source_physical_clean_audit
from data_pipeline.chang_e_manifest import find_source_entry
from motion_geometry.physical import SUPPORT_POLICY_SOURCE, motion_physical_metrics_np
from retargeting.build_cache import _build_bvh_source_reference_audit

ROOT = Path(os.environ.get("PROJECT_ROOT", "/home/disk/lsm/storage/DunhuangDanceRAG")).resolve()
BVH_ROOT = ROOT / "assets/motion/bvh"
MANIFEST = BVH_ROOT / "sources.json"
REJECTED_ROOT = ROOT / "output/source_contract_validation_v2/retarget_cache_rejected"
OUT = ROOT / "output/source_contract_validation_v2/requalified_gate_v2_1.json"
FPS = 30.0

cfg = legacy.RetargetConfig.from_env()
cfg.target_fps = FPS
cfg.source_manifest_path = str(MANIFEST)
cfg.require_source_manifest = True

sliding_tokens = {
    token.strip().lower()
    for token in os.environ.get(
        "CHANG_E_SLIDING_SUPPORT_TOKENS",
        "sogdian_whirl,sogdian,whirl,ribbon,lotus_steps,lotus,"
        "turning_travel,alternating_or_pivot_support",
    ).split(",")
    if token.strip()
}

rows = []
for motion_path in sorted(REJECTED_ROOT.glob("*.rejected.npy")):
    stem = motion_path.name.removesuffix(".rejected.npy")
    src = BVH_ROOT / f"{stem}.bvh"
    if not src.is_file():
        rows.append({"source": stem, "ok": False, "error": f"missing source BVH: {src}"})
        continue

    motion = np.load(motion_path).astype(np.float32)
    meta = find_source_entry(src, path=MANIFEST, required=True) or {}
    semantic_text = " ".join((str(src).lower(), json.dumps(meta, ensure_ascii=False, default=str).lower()))
    sliding = any(token in semantic_text for token in sliding_tokens)

    try:
        reference, reference_contract = _build_bvh_source_reference_audit(
            src,
            cfg=cfg,
            source_manifest=MANIFEST,
            strict_manifest=True,
        )
        audit = motion_physical_metrics_np(
            motion,
            fps=FPS,
            sliding_support_eligible=(np.ones(len(motion), dtype=bool) if sliding else None),
            support_policy=SUPPORT_POLICY_SOURCE,
        )
        gate = evaluate_source_physical_clean_audit(
            audit,
            source_reference_audit=reference,
        )
        rows.append({
            "source": stem,
            "ok": bool(gate["ok"]),
            "reasons": list(gate.get("reasons", [])),
            "gate": gate,
            "physical_audit": audit,
            "source_reference_physical_audit": reference,
            "source_reference_contract": reference_contract,
            "rejected_motion": str(motion_path),
        })
    except Exception as exc:
        rows.append({"source": stem, "ok": False, "error": f"{type(exc).__name__}: {exc}", "rejected_motion": str(motion_path)})

counter = Counter()
for row in rows:
    for reason in row.get("reasons", []):
        counter[reason] += 1

summary = {
    "schema": "retarget_clean_v2_1_requalification_diagnostic",
    "num_inputs": len(rows),
    "num_ok": sum(bool(row.get("ok")) for row in rows),
    "num_failed": sum(not bool(row.get("ok")) for row in rows),
    "failure_reason_counts": dict(counter.most_common()),
    "rows": rows,
}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

print("=" * 88)
print("V2.1 REJECTED-MOTION REQUALIFICATION")
print("=" * 88)
print("num_inputs =", summary["num_inputs"])
print("num_ok     =", summary["num_ok"])
print("num_failed =", summary["num_failed"])
print("\nFAILURE REASON COUNTS")
for reason, count in counter.most_common():
    print(f"{count:2d}  {reason}")
print("\nPER SOURCE")
for row in rows:
    print(f"{row['source']:24s}  {'PASS' if row.get('ok') else 'FAIL'}  {row.get('reasons', row.get('error', ''))}")
print("\nSAVED:", OUT)
