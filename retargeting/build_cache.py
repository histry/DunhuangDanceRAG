#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a V46.53.1 source-safe retarget cache.

This replacement separates source-level catastrophic safety from event-level style
quality. A source is retained when it is numerically/physically usable; expressive
low-posture or instrument-specific segments are filtered after event slicing.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import retargeting.bvh_solver as legacy
from contracts.anatomy import env_bool, env_int
from retargeting.legacy_anatomy_adapter import load_official_smpl_motion
from retargeting.anatomy_retarget import retarget_bvh_research

SCHEMA = "v46_53_1_source_safe_retarget_cache"


def _discover(in_dir: Path) -> List[Path]:
    bvh = sorted(in_dir.rglob("*.bvh"))
    smpl = sorted([*in_dir.rglob("*.npz"), *in_dir.rglob("*.pkl"), *in_dir.rglob("*.pickle")])
    smpl = [p for p in smpl if not any(t in p.name.lower() for t in ("event", "index", "feature", "cache", "split"))]
    prefer_smpl = env_bool("V46_52_PREFER_OFFICIAL_SMPL", True)
    grouped: Dict[str, List[Path]] = {}
    for p in [*bvh, *smpl]:
        grouped.setdefault(str(p.relative_to(in_dir).with_suffix("")), []).append(p)
    selected: List[Path] = []
    for _, paths in sorted(grouped.items()):
        paths.sort(key=lambda p: (0 if prefer_smpl and p.suffix.lower() in {".npz", ".pkl", ".pickle"} else 1, str(p)))
        selected.append(paths[0])
    return selected


def _report_valid(rep: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    version = str(rep.get("version", ""))
    if "v46_53_1" not in version and not env_bool("V46_53_1_ALLOW_LEGACY_RETARGET_CACHE", False):
        reasons.append("not_v46_53_1")
    if not bool(rep.get("ok", False)):
        reasons.append("not_ok")
    if not bool(rep.get("source_gate_ok", rep.get("anatomy_ok", False))):
        reasons.append("source_gate_not_ok")
    if not bool(rep.get("gravity_ok", False)):
        reasons.append("gravity_not_ok")
    if not bool(rep.get("fit_ok", False)):
        reasons.append("fit_not_ok")
    return not reasons, reasons


def _split_feasible(num_sources: int) -> bool:
    return int(num_sources) >= 3


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="change")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _discover(in_dir)
    if not files:
        raise RuntimeError(f"No BVH/SMPL source files found under {in_dir}")

    cfg = legacy.RetargetConfig.from_env()
    if args.device:
        cfg.device = args.device
    allow_partial = bool(args.allow_partial or env_bool("V46_52_ALLOW_PARTIAL_RETARGET", True))
    min_ok = max(3, min(len(files), env_int("V46_52_MIN_OK_SOURCES", min(8, len(files)))))

    reports: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    stale: List[Dict[str, Any]] = []

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(".npy")
        rep_path = dst.with_suffix(".retarget.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[V46.53.1 RETARGET {idx}/{len(files)}] {src} -> {dst}", flush=True)
        try:
            if dst.exists() and rep_path.exists() and not args.overwrite:
                old = json.loads(rep_path.read_text(encoding="utf-8"))
                valid, reasons = _report_valid(old)
                if valid:
                    print("[SKIP] existing V46.53.1 source-safe cache", flush=True)
                    reports.append(old)
                    continue
                stale.append({"source": str(src), "reasons": reasons})
                print(f"[REBUILD STALE] {reasons}", flush=True)

            candidates = [src]
            if src.suffix.lower() != ".bvh":
                fallback = src.with_suffix(".bvh")
                if fallback.is_file():
                    candidates.append(fallback)

            candidate_errors: List[Dict[str, str]] = []
            motion = None
            rep = None
            source_used = None
            for candidate in candidates:
                try:
                    if candidate.suffix.lower() == ".bvh":
                        motion, rep = retarget_bvh_research(candidate, cfg)
                    else:
                        # Existing official-SMPL loader remains supported. Its strict
                        # report must still pass the source-safety report validator.
                        motion, rep = load_official_smpl_motion(candidate, target_fps=float(cfg.target_fps))
                        rep = dict(rep)
                        rep.setdefault("source_gate_ok", bool(rep.get("anatomy_ok", False)))
                        rep["version"] = str(rep.get("version", "official_smpl")) + "_v46_53_1"
                    source_used = candidate
                    break
                except Exception as exc:
                    candidate_errors.append({"source": str(candidate), "error": str(exc)})
                    motion = rep = source_used = None

            if motion is None or rep is None or source_used is None:
                raise RuntimeError("All source representations failed: " + json.dumps(candidate_errors, ensure_ascii=False))

            rep = dict(rep)
            rep.update({
                "output": str(dst),
                "source_relative": str(rel.with_suffix(source_used.suffix)),
                "preferred_source": str(src),
                "source_used": str(source_used),
                "representation_fallbacks": candidate_errors,
                "v46_53_1_cache_contract": {
                    "schema": SCHEMA,
                    "source_gate": "catastrophic_only",
                    "event_quality_gate_deferred": True,
                    "requires_gravity_ok": True,
                    "requires_fit_ok": True,
                    "official_smpl_preferred": env_bool("V46_52_PREFER_OFFICIAL_SMPL", True),
                },
            })
            valid, reasons = _report_valid(rep)
            if not valid:
                raise RuntimeError(f"Non-formal V46.53.1 report: {reasons}")
            np.save(dst, np.asarray(motion, dtype=np.float32))
            rep_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            reports.append(rep)
        except Exception as exc:
            fail = {"source": str(src), "output": str(dst), "error": str(exc), "traceback": traceback.format_exc()}
            failures.append(fail)
            print(f"[REJECTED SOURCE] {src}: {exc}", flush=True)
            for p in (dst, rep_path):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            if not allow_partial:
                break

    enough = len(reports) >= min_ok
    split_ok = _split_feasible(len(reports))
    all_ok = enough and split_ok and (allow_partial or not failures)
    summary = {
        "schema": SCHEMA,
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "num_inputs": len(files),
        "num_ok": len(reports),
        "num_failed": len(failures),
        "minimum_ok_sources": int(min_ok),
        "split_feasible": bool(split_ok),
        "allow_partial": bool(allow_partial),
        "all_ok": bool(all_ok),
        "policy": {
            "source_gate": "catastrophic numerical/physical failures only",
            "style_quality": "deferred to event-level posture-aware gate",
            "minimum_split_cardinality": 3,
        },
        "stale_rebuilt": stale,
        "reports": reports,
        "failures": failures,
    }
    for name in ("v46_50_retarget_cache_report.json", "v46_52_retarget_cache_report.json", "v46_53_1_retarget_cache_report.json"):
        (out_dir / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("num_inputs", "num_ok", "num_failed", "minimum_ok_sources", "split_feasible", "all_ok")}, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
