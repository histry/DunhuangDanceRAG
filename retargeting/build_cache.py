#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the Retarget Clean source-safe cache.

This builder intentionally separates two physical contracts:

* source-retarget qualification (this file): compare Retarget Clean against the
  aligned/resampled recorded source trajectory, use source-observation support
  semantics, and preserve authentic high-frequency dance dynamics;
* final-generation qualification: remains the strict absolute fail-closed gate
  implemented by ``evaluate_physical_audit`` and the default physical metrics.

Source motion must still pass anatomy, gravity, fit, rotation integrity and the
source-relative physical-clean gate before it can enter training.  Expressive
style quality remains an event-level decision after slicing.  Intentional travel
is not treated as long-horizon root drift at source-cache time.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
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
from contracts.gravity import fk24_np
from contracts.physical_quality import evaluate_source_physical_clean_audit
from data_pipeline.chang_e_manifest import (
    file_sha256,
    find_source_entry,
    load_manifest,
    manifest_sha256,
    validate_source,
)
from motion_geometry.physical import (
    SUPPORT_POLICY_SOURCE,
    motion_physical_metrics_np,
    source_reference_kinematic_metrics_np,
)
from retargeting.anatomy_retarget import retarget_bvh_research
from retargeting.legacy_anatomy_adapter import load_official_smpl_motion

# Bump the cache schema because physical-clean semantics changed.  Old caches
# are therefore rebuilt rather than silently reused under the new contract.
SCHEMA = "retarget_clean_source_safe_retarget_cache_v3_body_normalized_source_dynamics"


def _discover(in_dir: Path) -> List[Path]:
    bvh = sorted(in_dir.rglob("*.bvh"))
    smpl = sorted(
        [
            *in_dir.rglob("*.npz"),
            *in_dir.rglob("*.pkl"),
            *in_dir.rglob("*.pickle"),
        ]
    )
    smpl = [
        p
        for p in smpl
        if not any(
            t in p.name.lower()
            for t in ("event", "index", "feature", "cache", "split")
        )
    ]
    prefer_smpl = env_bool("RETARGET_PREFER_OFFICIAL_SMPL", True)
    grouped: Dict[str, List[Path]] = {}
    for p in [*bvh, *smpl]:
        grouped.setdefault(
            str(p.relative_to(in_dir).with_suffix("")), []
        ).append(p)

    selected: List[Path] = []
    for _, paths in sorted(grouped.items()):
        paths.sort(
            key=lambda p: (
                0
                if prefer_smpl
                and p.suffix.lower() in {".npz", ".pkl", ".pickle"}
                else 1,
                str(p),
            )
        )
        selected.append(paths[0])
    return selected


def _report_valid(
    rep: Dict[str, Any],
    expected_provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    version = str(rep.get("version", ""))
    if (
        "event_geometry_1" not in version
        and not env_bool("RETARGET_CLEAN_ALLOW_LEGACY_RETARGET_CACHE", False)
    ):
        reasons.append("not_event_geometry_1")
    if not bool(rep.get("ok", False)):
        reasons.append("not_ok")
    if not bool(rep.get("source_gate_ok", rep.get("anatomy_ok", False))):
        reasons.append("source_gate_not_ok")
    if not bool(rep.get("gravity_ok", False)):
        reasons.append("gravity_not_ok")
    if not bool(rep.get("fit_ok", False)):
        reasons.append("fit_not_ok")
    if not bool(rep.get("physical_clean_ok", False)):
        reasons.append("physical_clean_not_ok")

    if expected_provenance is not None:
        actual = rep.get("cache_provenance")
        if not isinstance(actual, dict):
            reasons.append("missing_cache_provenance")
        else:
            for key, expected in expected_provenance.items():
                observed = actual.get(key)
                if isinstance(expected, float):
                    try:
                        matches = abs(float(observed) - expected) <= 1.0e-8
                    except (TypeError, ValueError):
                        matches = False
                else:
                    matches = observed == expected
                if not matches:
                    reasons.append(f"cache_provenance_mismatch:{key}")
    return not reasons, reasons


def _expected_provenance(
    source: Path,
    *,
    target_fps: float,
    source_manifest: Optional[Path],
    strict_manifest: bool,
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "source_sha256": file_sha256(source),
        "target_fps": float(target_fps),
        "cache_schema": SCHEMA,
    }
    if source.suffix.lower() == ".bvh":
        timebase = validate_source(
            source,
            path=source_manifest,
            required=strict_manifest,
            verify_hash=True,
        )
        provenance.update(
            {
                "effective_source_fps": float(timebase["effective_fps"]),
                "declared_source_fps": float(timebase["declared_fps"]),
                "manifest_sha256": timebase.get("manifest_sha256"),
                "source_id": timebase.get("source_id", source.stem),
                "recording_uid": timebase.get("recording_uid", source.stem),
            }
        )
    else:
        provenance.update(
            {
                "effective_source_fps": None,
                "declared_source_fps": None,
                "manifest_sha256": (
                    manifest_sha256(source_manifest)
                    if source_manifest
                    else None
                ),
                "source_id": source.stem,
                "recording_uid": source.stem,
            }
        )
    return provenance


def _split_feasible(num_sources: int) -> bool:
    return int(num_sources) >= 3


def _build_bvh_source_reference_audit(
    source: Path,
    *,
    cfg: Any,
    source_manifest: Optional[Path],
    strict_manifest: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Reconstruct the exact pre-optimization target-space source trajectory.

    This follows the same canonical path as ``legacy.retarget_bvh`` up to, but
    not including, optimization: BVH FK -> similarity alignment -> authoritative
    manifest timebase resampling -> source-heading stabilization -> 24-joint
    target observation layout.  It therefore supplies the correct physical
    baseline for source-relative acceptance without comparing incompatible
    skeleton scales or coordinate systems.
    """

    bvh = legacy.parse_bvh(source)
    timebase = validate_source(
        source,
        path=source_manifest,
        required=bool(strict_manifest),
        verify_hash=bool(strict_manifest),
    )
    source_fps = float(timebase["effective_fps"])

    native_pos, _ = legacy.source_fk(bvh, use_motion=True)
    rest_bvh = legacy.BVHMotion(
        bvh.path,
        bvh.joints,
        np.zeros_like(bvh.values),
        bvh.frame_time,
    )
    rest_pos, _ = legacy.source_fk(rest_bvh, use_motion=False)
    mapping = legacy.build_joint_mapping(bvh.joints)
    target_rest = legacy.target_rest_positions()

    # Match bvh_solver.retarget_bvh calibration exactly: use each source joint
    # once and do not let hand-end fallbacks duplicate calibration points.
    pairs: List[Tuple[int, int]] = []
    used_src: set[int] = set()
    for tgt, src in mapping.items():
        if src in used_src or tgt in {22, 23}:
            continue
        used_src.add(src)
        pairs.append((int(tgt), int(src)))

    X = np.asarray(
        [rest_pos[0, src] for tgt, src in pairs], dtype=np.float32
    )
    Y = np.asarray(
        [target_rest[tgt] for tgt, src in pairs], dtype=np.float32
    )
    W = np.asarray(
        [legacy.TARGET_JOINT_WEIGHTS[tgt] for tgt, src in pairs],
        dtype=np.float32,
    )
    scale, basis_R, trans = legacy.similarity_umeyama(X, Y, W)

    aligned = legacy.apply_similarity(native_pos, scale, basis_R, trans)
    aligned = legacy.resample_global_positions(
        aligned, source_fps, float(cfg.target_fps)
    )
    aligned, heading_report = legacy.stabilize_source_heading_positions(
        aligned,
        mapping,
        float(cfg.target_fps),
        timebase.get("entry", {}),
    )

    T = int(len(aligned))
    target_pos = np.zeros((T, 24, 3), dtype=np.float32)
    observed = np.zeros((24,), dtype=bool)
    for tgt, src in mapping.items():
        target_pos[:, int(tgt)] = aligned[:, int(src)]
        observed[int(tgt)] = True

    # Same virtual observations used by both legacy and research retargeters.
    if not observed[3] and observed[0] and observed[6]:
        target_pos[:, 3] = 0.45 * target_pos[:, 0] + 0.55 * target_pos[:, 6]
        observed[3] = True
    if not observed[6] and observed[3] and observed[9]:
        target_pos[:, 6] = 0.50 * target_pos[:, 3] + 0.50 * target_pos[:, 9]
        observed[6] = True
    if not observed[9] and observed[6] and observed[12]:
        target_pos[:, 9] = 0.55 * target_pos[:, 6] + 0.45 * target_pos[:, 12]
        observed[9] = True

    missing = np.flatnonzero(~observed).astype(int).tolist()
    if missing:
        raise RuntimeError(
            "Cannot build formal source-reference physical audit; "
            f"target observation joints missing: {missing}"
        )

    audit = source_reference_kinematic_metrics_np(
        target_pos,
        fps=float(cfg.target_fps),
    )
    metadata = {
        "mode": "native_bvh_aligned_pre_retarget",
        "source": str(source),
        "source_frames": int(len(bvh.values)),
        "reference_frames": int(T),
        "effective_source_fps": float(source_fps),
        "target_fps": float(cfg.target_fps),
        "similarity": {
            "scale": float(scale),
            "basis_rotation": np.asarray(basis_R).astype(float).tolist(),
            "translation": np.asarray(trans).astype(float).tolist(),
            "det_basis_rotation": float(np.linalg.det(basis_R)),
        },
        "heading_contract": heading_report,
        "mapping": {
            str(tgt): {
                "source_index": int(src),
                "source_name": bvh.joints[int(src)].name,
            }
            for tgt, src in mapping.items()
        },
    }
    return audit, metadata


def _build_source_reference_audit(
    source_used: Path,
    motion: np.ndarray,
    *,
    cfg: Any,
    source_manifest: Optional[Path],
    strict_manifest: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if source_used.suffix.lower() == ".bvh":
        return _build_bvh_source_reference_audit(
            source_used,
            cfg=cfg,
            source_manifest=source_manifest,
            strict_manifest=strict_manifest,
        )

    # Official SMPL has already passed its own loader/safety adapter.  There is
    # no separate BVH observation skeleton to compare against, so its canonical
    # motion is its own dynamic reference.  Source-specific contact/penetration
    # and rotation-integrity checks still remain active.
    reference = source_reference_kinematic_metrics_np(
        fk24_np(np.asarray(motion, dtype=np.float32)),
        fps=float(cfg.target_fps),
    )
    return reference, {
        "mode": "official_smpl_canonical_self_reference",
        "source": str(source_used),
        "reference_frames": int(len(motion)),
        "target_fps": float(cfg.target_fps),
    }


def _rejected_paths(out_dir: Path, rel: Path) -> Tuple[Path, Path]:
    configured = str(os.environ.get("RETARGET_REJECTED_DIR", "")).strip()
    root = Path(configured).expanduser().resolve() if configured else (out_dir.parent / f"{out_dir.name}_rejected").resolve()
    base = (root / rel).with_suffix("")
    return (
        base.with_name(base.name + ".rejected.npy"),
        base.with_name(base.name + ".rejected.retarget.json"),
    )


def _persist_rejected_candidate(*, out_dir: Path, rel: Path, motion: Optional[np.ndarray], report: Optional[Dict[str, Any]], error: str, traceback_text: str) -> Dict[str, Optional[str]]:
    motion_path, report_path = _rejected_paths(out_dir, rel)
    motion_ref = report_ref = None
    if motion is not None:
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(motion_path, np.asarray(motion, dtype=np.float32))
        motion_ref = str(motion_path)
    if report is not None or motion is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(report or {})
        payload["rejected_artifact"] = {
            "accepted_for_training": False,
            "motion": motion_ref,
            "report": str(report_path),
            "error": str(error),
            "traceback": str(traceback_text),
            "persistence_contract": "outside_accepted_cache_tree",
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        report_ref = str(report_path)
    return {"motion": motion_ref, "report": report_ref}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="change")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--target_fps", type=float, choices=(30.0, 60.0), default=None
    )
    ap.add_argument(
        "--source_manifest",
        default=os.environ.get("CHANG_E_SOURCE_MANIFEST", ""),
        help="Authoritative Chang-E sources.json; auto-detected in --in_dir.",
    )
    ap.add_argument(
        "--allow_unmanifested_bvh",
        action="store_true",
        help="Allow generic BVH inputs to use their declared header FPS.",
    )
    ap.add_argument(
        "--smpl_scaling_mode",
        choices=(
            "canonical_body",
            "scale_translation",
            "inverse_scale_translation",
        ),
        default="canonical_body",
    )
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
    if args.target_fps is not None:
        cfg.target_fps = float(args.target_fps)

    manifest_candidate = (
        Path(args.source_manifest).expanduser().resolve()
        if str(args.source_manifest).strip()
        else (in_dir / "sources.json").resolve()
    )
    source_manifest: Optional[Path] = (
        manifest_candidate if manifest_candidate.is_file() else None
    )
    strict_manifest = bool(
        source_manifest is not None and not args.allow_unmanifested_bvh
    )
    if source_manifest is not None:
        load_manifest(source_manifest, required=True)
        cfg.source_manifest_path = str(source_manifest)
        cfg.require_source_manifest = strict_manifest
    elif (
        any(path.suffix.lower() == ".bvh" for path in files)
        and not args.allow_unmanifested_bvh
    ):
        raise RuntimeError(
            "Formal BVH cache construction requires a source manifest. "
            "Pass --source_manifest or --allow_unmanifested_bvh for a non-Chang-E dataset."
        )

    allow_partial = bool(
        args.allow_partial or env_bool("RETARGET_ALLOW_PARTIAL_RETARGET", True)
    )
    min_ok = max(
        3,
        min(
            len(files),
            env_int("RETARGET_MIN_OK_SOURCES", min(8, len(files))),
        ),
    )

    reports: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    stale: List[Dict[str, Any]] = []

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(in_dir)
        dst = (out_dir / rel).with_suffix(".npy")
        rep_path = dst.with_suffix(".retarget.json")
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[Retarget Clean RETARGET {idx}/{len(files)}] {src} -> {dst}",
            flush=True,
        )

        motion = None
        rep = None
        source_used = None
        physical_clean_gate = None
        source_reference_audit = None
        source_reference_contract = None
        try:
            expected_provenance = _expected_provenance(
                src,
                target_fps=float(cfg.target_fps),
                source_manifest=source_manifest,
                strict_manifest=strict_manifest,
            )
            if dst.exists() and rep_path.exists() and not args.overwrite:
                old = json.loads(rep_path.read_text(encoding="utf-8"))
                valid, reasons = _report_valid(old, expected_provenance)
                if valid:
                    print(
                        "[SKIP] existing Retarget Clean source-safe cache",
                        flush=True,
                    )
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
                    candidate_cfg = copy.deepcopy(cfg)
                    if candidate.suffix.lower() == ".bvh":
                        candidate_cfg.source_manifest_path = (
                            str(source_manifest) if source_manifest else ""
                        )
                        candidate_cfg.require_source_manifest = strict_manifest
                        motion, rep = retarget_bvh_research(
                            candidate, candidate_cfg
                        )
                    else:
                        motion, rep = load_official_smpl_motion(
                            candidate,
                            target_fps=float(cfg.target_fps),
                            scaling_mode=str(args.smpl_scaling_mode),
                        )
                        rep = dict(rep)
                        rep.setdefault(
                            "source_gate_ok", bool(rep.get("anatomy_ok", False))
                        )
                        rep["version"] = (
                            str(rep.get("version", "official_smpl"))
                            + "_event_geometry_1"
                        )
                    source_used = candidate
                    break
                except Exception as exc:
                    candidate_errors.append(
                        {"source": str(candidate), "error": str(exc)}
                    )
                    motion = rep = source_used = None

            if motion is None or rep is None or source_used is None:
                raise RuntimeError(
                    "All source representations failed: "
                    + json.dumps(candidate_errors, ensure_ascii=False)
                )

            motion = np.asarray(motion, dtype=np.float32)
            rep = dict(rep)
            source_metadata = find_source_entry(
                source_used,
                path=source_manifest,
                required=bool(
                    strict_manifest and source_used.suffix.lower() == ".bvh"
                ),
            )
            if source_metadata is not None:
                source_metadata = {
                    key: value
                    for key, value in source_metadata.items()
                    if key not in {"manifest_path", "manifest_sha256"}
                }

            semantic_text = " ".join(
                (
                    str(source_used).lower(),
                    json.dumps(
                        source_metadata or {}, ensure_ascii=False
                    ).lower(),
                    json.dumps(
                        rep, ensure_ascii=False, default=str
                    ).lower(),
                )
            )
            sliding_tokens = {
                token.strip().lower()
                for token in os.environ.get(
                    "CHANG_E_SLIDING_SUPPORT_TOKENS",
                    "sogdian_whirl,sogdian,whirl,ribbon,lotus_steps,lotus,"
                    "turning_travel,alternating_or_pivot_support",
                ).split(",")
                if token.strip()
            }
            source_sliding_eligible = any(
                token in semantic_text for token in sliding_tokens
            )

            # Build the recorded-motion reference BEFORE applying any source
            # acceptance threshold.  This is the key semantic separation from
            # final generated motion.
            source_reference_audit, source_reference_contract = (
                _build_source_reference_audit(
                    source_used,
                    motion,
                    cfg=cfg,
                    source_manifest=source_manifest,
                    strict_manifest=strict_manifest,
                )
            )

            physical_audit = motion_physical_metrics_np(
                motion,
                fps=float(cfg.target_fps),
                sliding_support_eligible=(
                    np.full(len(motion), True, dtype=bool)
                    if source_sliding_eligible
                    else None
                ),
                support_policy=SUPPORT_POLICY_SOURCE,
            )
            physical_clean_gate = evaluate_source_physical_clean_audit(
                physical_audit,
                source_reference_audit=source_reference_audit,
            )

            cache_provenance = _expected_provenance(
                source_used,
                target_fps=float(cfg.target_fps),
                source_manifest=source_manifest,
                strict_manifest=strict_manifest,
            )
            rep.update(
                {
                    "output": str(dst),
                    "source_relative": str(
                        rel.with_suffix(source_used.suffix)
                    ),
                    "preferred_source": str(src),
                    "source_used": str(source_used),
                    "representation_fallbacks": candidate_errors,
                    "source_metadata": source_metadata,
                    "cache_provenance": cache_provenance,
                    "source_reference_physical_audit": source_reference_audit,
                    "source_reference_contract": source_reference_contract,
                    "physical_audit": physical_audit,
                    "physical_clean_gate": physical_clean_gate,
                    "physical_clean_ok": bool(physical_clean_gate["ok"]),
                    "retarget_clean_cache_contract": {
                        "schema": SCHEMA,
                        "source_gate": "pretraining_body_normalized_source_physical_clean_v3",
                        "source_support_policy": SUPPORT_POLICY_SOURCE,
                        "final_generation_gate_reused": False,
                        "event_quality_gate_deferred": True,
                        "requires_physical_clean_ok": True,
                        "sliding_support_semantic_eligible": bool(
                            source_sliding_eligible
                        ),
                        "requires_gravity_ok": True,
                        "requires_fit_ok": True,
                        "official_smpl_preferred": env_bool(
                            "RETARGET_PREFER_OFFICIAL_SMPL", True
                        ),
                        "canonical_fps": float(cfg.target_fps),
                        "smpl_scaling_mode": str(args.smpl_scaling_mode),
                    },
                }
            )

            valid, reasons = _report_valid(rep, cache_provenance)
            if not valid:
                gate_reasons = physical_clean_gate.get("reasons", [])
                raise RuntimeError(
                    "Non-formal Retarget Clean report: "
                    f"{reasons}; physical_clean_reasons={gate_reasons}"
                )

            np.save(dst, motion)
            rep_path.write_text(
                json.dumps(rep, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            reports.append(rep)

        except Exception as exc:
            tb = traceback.format_exc()
            rejected = _persist_rejected_candidate(
                out_dir=out_dir, rel=rel,
                motion=None if motion is None else np.asarray(motion, dtype=np.float32),
                report=None if rep is None else dict(rep),
                error=str(exc), traceback_text=tb,
            )
            fail = {
                "source": str(src), "output": str(dst), "error": str(exc),
                "traceback": tb,
                "rejected_motion": rejected["motion"],
                "rejected_report": rejected["report"],
                "physical_clean_reasons": list(physical_clean_gate.get("reasons", [])) if isinstance(physical_clean_gate, dict) else [],
            }
            failures.append(fail)
            print(f"[REJECTED SOURCE] {src}: {exc}; rejected_motion={rejected['motion']} rejected_report={rejected['report']}", flush=True)
            for accepted_path in (dst, rep_path):
                try:
                    accepted_path.unlink(missing_ok=True)
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
        "canonical_fps": float(cfg.target_fps),
        "source_manifest": str(source_manifest) if source_manifest else None,
        "source_manifest_sha256": (
            manifest_sha256(source_manifest) if source_manifest else None
        ),
        "strict_source_manifest": bool(strict_manifest),
        "smpl_scaling_mode": str(args.smpl_scaling_mode),
        "all_ok": bool(all_ok),
        "rejected_artifact_root": str((_rejected_paths(out_dir, Path("placeholder.bvh"))[0]).parent),
        "policy": {
            "source_gate": (
                "pretraining anatomy/gravity/fit + reference-relative "
                "source physical clean"
            ),
            "source_anti_jitter": "body_normalized_root_relative_to_recorded_source",
            "source_support": SUPPORT_POLICY_SOURCE,
            "final_generation_gate_reused": False,
            "style_quality": "deferred to event-level posture-aware gate",
            "minimum_split_cardinality": 3,
        },
        "stale_rebuilt": stale,
        "reports": reports,
        "failures": failures,
    }

    for name in (
        "event_heading_retarget_cache_report.json",
        "anatomy_heading_retarget_cache_report.json",
        "retarget_clean_retarget_cache_report.json",
    ):
        (out_dir / name).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "num_inputs",
                    "num_ok",
                    "num_failed",
                    "minimum_ok_sources",
                    "split_feasible",
                    "all_ok",
                )
            },
            indent=2,
        )
    )
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
