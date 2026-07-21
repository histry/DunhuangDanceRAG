#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a V46.50 heading-aware Event-RAG database from V46.49.4 caches.

Input must be retargeted EDGE151D NPY, not raw BVH.  This guarantees that
bind-pose, gravity, absolute source heading and root-orientation contracts were
resolved before event extraction.

The output preserves the existing V46/V44/V45/V46 32D schema and appends
heading metadata arrays.  Existing AESD enrichment remains compatible because
events/build_semantics.py copies all input NPZ arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.heading import (  # noqa: E402
    EDGE_DIM,
    adaptive_event_segments,
    enforce_event_heading_contract,
)
from support.event_identity import (  # noqa: E402
    EVENT_UID_SCHEMA,
    event_uids_from_generation_db,
    make_event_db_contract,
)
from motion_geometry.smpl24 import skeleton_contract  # noqa: E402


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def collect_npy_inputs(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".npy":
            out.append(p)
        elif p.is_dir():
            for f in p.rglob("*.npy"):
                name = f.name.lower()
                if any(
                    token in name
                    for token in (
                        "motion_ref",
                        "transition_mask",
                        "single_test",
                        "jitter_peak",
                        "spin_interval",
                    )
                ):
                    continue
                out.append(f)
    return sorted(set(out))


def load_motion(path: Path) -> List[np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    arr = np.asarray(obj)
    seqs: List[np.ndarray] = []
    if arr.ndim == 2 and arr.shape[1] >= EDGE_DIM:
        seqs.append(arr[:, :EDGE_DIM].astype(np.float32))
    elif arr.ndim == 3 and arr.shape[-1] >= EDGE_DIM:
        for i in range(arr.shape[0]):
            seqs.append(arr[i, :, :EDGE_DIM].astype(np.float32))
    return seqs


def sibling_retarget_report(path: Path) -> Dict[str, Any]:
    candidates = [
        path.with_suffix(".retarget.json"),
        Path(str(path).replace(".npy", ".retarget.json")),
    ]
    for p in candidates:
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def validate_retarget_contract(path: Path, report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate legacy V46.49.4 caches and V46.53.1 soft-root caches.

    V46.53.1 intentionally replaces the complete root-orientation lock with a
    source-body-frame SO(3) soft geodesic anchor.  The Event-DB builder must not
    reject that newer contract merely because its mode string differs from the
    legacy ``absolute_reference_lock`` contract.  Acceptance remains strict:
    the report must be produced by the V46.53.1 source-safety retargeter and all
    source-level anatomy, gravity and fit gates must have passed.
    """
    reasons: List[str] = []
    strict = str(os.environ.get("V46_50_REQUIRE_V46_49_4_CACHE", "1")).strip().lower() not in {
        "0", "false", "no", "off"
    }
    if not strict:
        return True, reasons

    if not report:
        reasons.append("missing_retarget_report")
        return False, reasons
    if not bool(report.get("ok", False)):
        reasons.append("retarget_report_not_ok")

    pos = report.get("source_position_contract", {})
    if str(pos.get("nonroot_position_mode", "")) != "ignore":
        reasons.append("nonroot_position_mode_not_ignore")

    fit = report.get("fit", {})
    heading = fit.get("heading_contract", {})
    if str(heading.get("mode", "")) != "stabilize":
        reasons.append("heading_contract_not_stabilize")

    root_contract = fit.get("root_orientation_contract", {})
    root_mode = str(root_contract.get("mode", "")).strip()

    if root_mode == "absolute_reference_lock":
        # Preserved V46.49.4/V46.52 contract.
        pass
    elif root_mode == "soft_geodesic_anchor":
        allow_soft = str(
            os.environ.get("V46_53_1_ALLOW_SOFT_ROOT_CONTRACT", "1")
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if not allow_soft:
            reasons.append("v46_53_1_soft_root_contract_disabled")

        if str(report.get("schema", "")) != "v46_53_1_source_safety_retarget":
            reasons.append("soft_root_missing_v46_53_1_source_safety_schema")
        if str(root_contract.get("version", "")) != "v46_53_1_soft_source_body_frame_contract":
            reasons.append("soft_root_contract_version_mismatch")
        if str(root_contract.get("root_orientation", "")) != "optimized_with_source_body_frame_prior":
            reasons.append("soft_root_orientation_prior_missing")

        # Do not weaken the source gate while adding compatibility.
        for key in ("source_gate_ok", "anatomy_ok", "gravity_ok", "fit_ok"):
            if not bool(report.get(key, False)):
                reasons.append(f"soft_root_{key}_false")
    else:
        reasons.append(
            "unsupported_root_orientation_contract_mode="
            + (root_mode if root_mode else "missing")
        )

    return not reasons, reasons


def _field_array(meta: List[dict], key: str, default: Any, dtype=object) -> np.ndarray:
    return np.asarray([m.get(key, default) for m in meta], dtype=dtype)


def _safe_take(value: Any) -> int:
    try:
        return int(value if value is not None else -1)
    except Exception:
        return -1


def save_db(
    out_dir: Path,
    meta: List[dict],
    descs: List[np.ndarray],
    entries: List[np.ndarray],
    exits: List[np.ndarray],
    c0s: List[np.ndarray],
    c1s: List[np.ndarray],
    music_feats: List[np.ndarray],
    music_masks: List[float],
    v46: Any,
    cfg: Any,
) -> Path:
    desc = np.stack(descs).astype(np.float32)
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1e-6
    desc_z = ((desc - mean) / std).astype(np.float32)

    name_semantic = np.stack(
        [v46.filename_semantic_vector_from_meta(m, cfg) for m in meta]
    ).astype(np.float32)
    class_semantic = np.stack(
        [v46.class_semantic_vector_from_meta(m, cfg) for m in meta]
    ).astype(np.float32)

    db_path = out_dir / "events.npz"
    identity_seed = {
        "paths": _field_array(meta, "path", "", object),
        "source_uids": _field_array(meta, "source_uid", "unknown", object),
        "source_files": _field_array(meta, "source_file", "", object),
        "starts": _field_array(meta, "start", 0, np.int32),
        "ends": _field_array(meta, "end", 0, np.int32),
        "frames": _field_array(meta, "frames", 0, np.int32),
        "source_start_seconds": _field_array(meta, "source_start_seconds", 0.0, np.float64),
        "source_end_seconds": _field_array(meta, "source_end_seconds", 0.0, np.float64),
        "canonical_fps": _field_array(meta, "canonical_fps", float(cfg.fps), np.float32),
    }
    event_uids = event_uids_from_generation_db(identity_seed)
    identity_contract = make_event_db_contract(event_uids)
    payload: Dict[str, Any] = {
        "event_uid_schema_version": np.asarray(
            EVENT_UID_SCHEMA, dtype=object
        ),
        "skeleton_contract_json": np.asarray(
            json.dumps(skeleton_contract(), sort_keys=True), dtype=object
        ),
        "event_uids": event_uids,
        "event_db_contract_json": np.asarray(
            json.dumps(identity_contract, sort_keys=True), dtype=object
        ),
        "heading_contract_schema_version": np.asarray(
            "v46_50_event_heading_contract", dtype=object
        ),
        "desc": desc,
        "desc_z": desc_z,
        "desc_mean": mean.astype(np.float32),
        "desc_std": std.astype(np.float32),
        "entry": np.stack(entries).astype(np.float32),
        "exit": np.stack(exits).astype(np.float32),
        "contact_entry": np.stack(c0s).astype(np.float32),
        "contact_exit": np.stack(c1s).astype(np.float32),
        "paths": _field_array(meta, "path", "", object),
        "source_groups": _field_array(meta, "source_group", "unknown", object),
        "source_files": _field_array(meta, "source_file", "", object),
        "source_bvh": _field_array(meta, "source_bvh", "", object),
        "source_uids": _field_array(meta, "source_uid", "unknown", object),
        "genders": _field_array(meta, "gender", "unknown", object),
        "labels": _field_array(meta, "label", "unknown", object),
        "parent_labels": _field_array(meta, "parent_label", "unknown", object),
        "dance_keys": _field_array(meta, "dance_key", "unknown", object),
        "dance_categories": _field_array(meta, "dance_category", "unknown", object),
        "semantic_roles": _field_array(meta, "semantic_role", "unknown", object),
        "semantic_texts": _field_array(meta, "semantic_text", "", object),
        "energy_labels": _field_array(meta, "energy_label", "unknown", object),
        "rhythm_labels": _field_array(meta, "rhythm_label", "unknown", object),
        "body_focus_labels": _field_array(meta, "body_focus_label", "unknown", object),
        "spatial_labels": _field_array(meta, "spatial_label", "unknown", object),
        "music_alignment_labels": _field_array(
            meta, "music_alignment_label", "unknown", object
        ),
        "classification_texts": _field_array(meta, "classification_text", "", object),
        "event_families": _field_array(meta, "event_family", "unknown", object),
        "motion_stage_roles": _field_array(
            meta, "motion_stage_role", "unknown", object
        ),
        "cultural_motifs": _field_array(meta, "cultural_motif", "unknown", object),
        "prop_proxy_labels": _field_array(
            meta, "prop_proxy_label", "unknown", object
        ),
        "locomotion_labels": _field_array(
            meta, "locomotion_label", "unknown", object
        ),
        "support_labels": _field_array(meta, "support_label", "unknown", object),
        "event_position_mid": _field_array(meta, "event_position_mid", 0.5, np.float32),
        "semantic_confidence": _field_array(
            meta, "semantic_confidence", 0.5, np.float32
        ),
        "event_quality_scores": _field_array(
            meta, "event_quality_score", 0.5, np.float32
        ),
        "natural_duration_min": np.asarray(
            [
                float((m.get("natural_duration_range_sec") or [1.5, 4.0])[0])
                for m in meta
            ],
            dtype=np.float32,
        ),
        "natural_duration_max": np.asarray(
            [
                float((m.get("natural_duration_range_sec") or [1.5, 4.0])[-1])
                for m in meta
            ],
            dtype=np.float32,
        ),
        "take_ids": np.asarray(
            [_safe_take(m.get("take_id", -1)) for m in meta], dtype=np.int32
        ),
        "name_semantic": name_semantic,
        "class_semantic": class_semantic,
        "durations": _field_array(meta, "duration", 0.0, np.float32),
        "frames": _field_array(meta, "frames", 0, np.int32),
        "starts": _field_array(meta, "start", 0, np.int32),
        "ends": _field_array(meta, "end", 0, np.int32),
        "source_start_seconds": _field_array(
            meta, "source_start_seconds", 0.0, np.float64
        ),
        "source_end_seconds": _field_array(
            meta, "source_end_seconds", 0.0, np.float64
        ),
        "canonical_fps": _field_array(
            meta, "canonical_fps", float(cfg.fps), np.float32
        ),
        "music": np.stack(music_feats).astype(np.float32),
        "music_mask": np.asarray(music_masks, dtype=np.float32),

        # V46.50 event-level heading state arrays.
        "event_original_entry_heading_rad": _field_array(
            meta, "event_original_entry_heading_rad", 0.0, np.float32
        ),
        "event_entry_heading_rad": _field_array(
            meta, "event_entry_heading_rad", 0.0, np.float32
        ),
        "event_exit_heading_rel_rad": _field_array(
            meta, "event_exit_heading_rel_rad", 0.0, np.float32
        ),
        "event_stage_delta_yaw_rad": _field_array(
            meta, "event_stage_delta_yaw_rad", 0.0, np.float32
        ),
        "event_net_yaw_rad": _field_array(
            meta, "event_net_yaw_rad", 0.0, np.float32
        ),
        "event_abs_yaw_rad": _field_array(
            meta, "event_abs_yaw_rad", 0.0, np.float32
        ),
        "event_yaw_budget_rad": _field_array(
            meta, "event_yaw_budget_rad", 0.0, np.float32
        ),
        "event_turn_intents": _field_array(
            meta, "event_turn_intent", "none", object
        ),
        "event_turn_confidence": _field_array(
            meta, "event_turn_confidence", 0.0, np.float32
        ),
        "event_heading_quality": _field_array(
            meta, "event_heading_quality", 0.0, np.float32
        ),
        "event_heading_valid": _field_array(
            meta, "event_heading_valid", True, np.bool_
        ),
        "event_mechanical_spin_ratio": _field_array(
            meta, "event_mechanical_spin_ratio", 0.0, np.float32
        ),
        "event_longest_same_sign_turn_seconds": _field_array(
            meta, "event_longest_same_sign_turn_seconds", 0.0, np.float32
        ),
        "event_heading_report_json": _field_array(
            meta, "event_heading_report_json", "{}", object
        ),
        "event_segment_start": _field_array(
            meta, "event_segment_start", 0, np.int32
        ),
        "event_segment_end": _field_array(
            meta, "event_segment_end", 0, np.int32
        ),
        "event_source_seq_frames": _field_array(
            meta, "event_source_seq_frames", 0, np.int32
        ),
    }
    np.savez_compressed(db_path, **payload)
    (out_dir / "events_meta.json").write_text(
        json.dumps(jsonable(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return db_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build V46.50 event-heading Event-RAG DB from V46.49.4 NPY"
    )
    ap.add_argument("--motion_dirs", nargs="+", required=True)
    ap.add_argument("--out_db", required=True)
    ap.add_argument("--config", default="configs/motion_model.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--canonical_intervals_in",
        default=None,
        help="Reuse FPS-independent [start_seconds,end_seconds] intervals from the canonical 30 FPS DB.",
    )
    ap.add_argument(
        "--canonical_intervals_out",
        default=None,
        help="Write kept event intervals for a later rate-specific DB build.",
    )
    args = ap.parse_args(argv)

    import training.motion_models as v46  # local latest core

    cfg = v46.V46Config.from_json(args.config).apply_env()
    interval_lookup: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    expected_interval_count: Optional[int] = None
    if args.canonical_intervals_in:
        interval_payload = json.loads(
            Path(args.canonical_intervals_in).read_text(encoding="utf-8")
        )
        if interval_payload.get("schema") != "canonical_event_intervals_v1":
            raise RuntimeError(
                f"Unsupported canonical interval schema: {interval_payload.get('schema')!r}"
            )
        expected_interval_count = int(interval_payload.get("num_intervals", 0))
        for row in interval_payload.get("intervals", []):
            key = (str(row["source_uid"]), int(row.get("seq_id", 0)))
            interval_lookup.setdefault(key, []).append(dict(row))
    kept_intervals: List[Dict[str, Any]] = []
    out_dir = Path(args.out_db)
    if out_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_dir = out_dir / "events"
    event_dir.mkdir(parents=True, exist_ok=True)

    files = collect_npy_inputs(args.motion_dirs)
    if not files:
        raise RuntimeError(f"No retargeted NPY found in {args.motion_dirs}")

    meta: List[dict] = []
    descs: List[np.ndarray] = []
    entries: List[np.ndarray] = []
    exits: List[np.ndarray] = []
    c0s: List[np.ndarray] = []
    c1s: List[np.ndarray] = []
    music_feats: List[np.ndarray] = []
    music_masks: List[float] = []

    source_reports: List[dict] = []
    dropped_events: List[dict] = []
    rejected_sources: List[dict] = []
    event_idx = 0

    for file_idx, path in enumerate(files, 1):
        rep = sibling_retarget_report(path)
        ok_contract, reasons = validate_retarget_contract(path, rep)
        if not ok_contract:
            rejected_sources.append({"path": str(path), "reasons": reasons})
            print(f"[REJECT SOURCE] {path}: {reasons}", file=sys.stderr)
            continue

        seqs = load_motion(path)
        if not seqs:
            rejected_sources.append({"path": str(path), "reasons": ["bad_motion_shape"]})
            continue

        original_source = str(
            rep.get("source")
            or rep.get("source_relative")
            or path
        )
        sem = v46.parse_change_bvh_semantics(original_source)
        strong_base = v46.strong_action_semantics_from_meta(sem)
        semantic_meta = {**sem, **strong_base}
        source_uid = str(sem.get("source_uid") or Path(original_source).stem)

        for seq_id, seq0 in enumerate(seqs):
            seq, contract = v46.enforce_edge151_contract_np(
                seq0,
                cfg,
                source_hint=f"v46_50_source:{path}",
                derive_contact=True,
                project_rot=True,
            )
            interval_key = (source_uid, int(seq_id))
            canonical_segment_rows: List[Optional[Dict[str, Any]]]
            if interval_lookup:
                canonical_rows = interval_lookup.get(interval_key)
                if not canonical_rows:
                    raise RuntimeError(
                        f"Canonical interval manifest has no source/sequence {interval_key!r}"
                    )
                segments = []
                for row in canonical_rows:
                    st = int(round(float(row["start_seconds"]) * float(cfg.fps)))
                    ed = int(round(float(row["end_seconds"]) * float(cfg.fps)))
                    st = max(0, min(st, len(seq) - 1))
                    ed = max(st + 1, min(ed, len(seq)))
                    segments.append((st, ed))
                canonical_segment_rows = [dict(row) for row in canonical_rows]
                seg_report = {
                    "schema": "canonical_event_intervals_v1",
                    "source_fps": float(cfg.fps),
                    "num_segments": len(segments),
                    "manifest": str(Path(args.canonical_intervals_in).resolve()),
                }
            else:
                segments, seg_report = adaptive_event_segments(
                    seq,
                    semantic_meta,
                    fps=float(cfg.fps),
                    min_event_frames=int(cfg.min_event_frames),
                    max_event_frames=int(cfg.max_event_frames),
                )
                canonical_segment_rows = [None] * len(segments)
            kept_here = 0
            dropped_here = 0

            for seg_idx, (st, ed) in enumerate(segments):
                canonical_row = canonical_segment_rows[seg_idx]
                raw_clip = seq[st:ed].astype(np.float32)
                if len(raw_clip) < int(cfg.min_event_frames):
                    if canonical_row is not None:
                        raise RuntimeError(
                            "Canonical event became too short at the target FPS: "
                            f"source={interval_key!r}, interval={canonical_row}, "
                            f"frames={len(raw_clip)}, minimum={cfg.min_event_frames}"
                        )
                    continue

                clip, heading = enforce_event_heading_contract(
                    raw_clip,
                    semantic_meta,
                    fps=float(cfg.fps),
                )
                if not bool(heading.get("valid", False)):
                    if canonical_row is not None:
                        raise RuntimeError(
                            "Canonical 30 FPS event failed the target-rate heading contract: "
                            f"source={interval_key!r}, interval={canonical_row}, "
                            f"reason={heading.get('reason')!r}"
                        )
                    dropped_here += 1
                    dropped_events.append({
                        "source": original_source,
                        "cache": str(path),
                        "seq_id": int(seq_id),
                        "segment_index": int(seg_idx),
                        "start": int(st),
                        "end": int(ed),
                        "reason": heading.get("reason"),
                        "intent": heading.get("intent"),
                        "heading": heading,
                    })
                    continue

                clip, final_contract = v46.enforce_edge151_contract_np(
                    clip,
                    cfg,
                    source_hint=f"v46_50_event:{path}:{st}:{ed}",
                    derive_contact=True,
                    project_rot=True,
                )
                out_path = event_dir / f"event_{event_idx:07d}.npy"
                identity_start_seconds = (
                    float(canonical_row["start_seconds"])
                    if canonical_row is not None
                    else float(st) / float(cfg.fps)
                )
                identity_end_seconds = (
                    float(canonical_row["end_seconds"])
                    if canonical_row is not None
                    else float(ed) / float(cfg.fps)
                )
                base_meta = {
                    **sem,
                    "source_file": original_source,
                    "source_bvh": Path(original_source).name,
                    "load_path": str(path),
                    "source_uid": source_uid,
                    "source_group": source_uid,
                    "seq_id": int(seq_id),
                    "label": sem.get("label", Path(original_source).stem),
                    "parent_label": sem.get("parent_label", sem.get("label", "unknown")),
                    "fragment_index": int(seg_idx),
                    "input_mode": "v46_50_v46_49_4_retarget_cache",
                    "event_start": int(st),
                    "event_end": int(ed),
                    "event_source_frames": int(len(seq)),
                    # Preserve the canonical manifest interval verbatim.  The
                    # local frame bounds are quantized execution details and
                    # must not change event identity across FPS branches.
                    "source_start_seconds": identity_start_seconds,
                    "source_end_seconds": identity_end_seconds,
                    "canonical_fps": float(cfg.fps),
                    "event_position_mid": float((st + ed) * 0.5 / max(len(seq), 1)),
                    "resample_report": {
                        "resampled": False,
                        "native_fps": float(cfg.fps),
                        "target_fps": float(cfg.fps),
                        "source": "v46_49_4_cache",
                    },
                }

                v46.add_event_to_db_lists(
                    clip=clip,
                    event_idx=event_idx,
                    out_path=out_path,
                    cfg=cfg,
                    source=source_uid,
                    matched_audio=None,
                    st=int(st),
                    base_meta=base_meta,
                    descs=descs,
                    entries=entries,
                    exits=exits,
                    c0s=c0s,
                    c1s=c1s,
                    music_feats=music_feats,
                    music_masks=music_masks,
                    meta=meta,
                )

                item = meta[-1]
                before = heading["before_budget"]
                after = heading["after_budget"]
                item.update({
                    "event_original_entry_heading_rad": float(
                        heading["entry"]["entry_heading_before_rad"]
                    ),
                    "event_entry_heading_rad": float(after["entry_heading_rad"]),
                    "event_exit_heading_rel_rad": float(after["net_yaw_rad"]),
                    "event_stage_delta_yaw_rad": float(after["net_yaw_rad"]),
                    "event_net_yaw_rad": float(after["net_yaw_rad"]),
                    "event_abs_yaw_rad": float(after["absolute_yaw_rad"]),
                    "event_yaw_budget_rad": float(heading["yaw_budget_rad"]),
                    "event_turn_intent": str(heading["intent"]),
                    "event_turn_confidence": float(heading["turn_confidence"]),
                    "event_heading_quality": float(heading["heading_quality"]),
                    "event_heading_valid": bool(heading["valid"]),
                    "event_mechanical_spin_ratio": float(
                        after["mechanical_spin_ratio"]
                    ),
                    "event_longest_same_sign_turn_seconds": float(
                        after["longest_same_sign_turn_seconds"]
                    ),
                    "event_heading_report_json": json.dumps(
                        jsonable(heading), ensure_ascii=False, sort_keys=True
                    ),
                    "event_segment_start": int(st),
                    "event_segment_end": int(ed),
                    "event_source_seq_frames": int(len(seq)),
                    "event_segmentation_schema": "v46_50_motion_adaptive_segmentation",
                    "retarget_contract_source": rep.get("version", "v46_49_4"),
                    "edge151_contract_report": {
                        **dict(item.get("edge151_contract_report", {})),
                        "v46_50_final": final_contract,
                    },
                })
                # Heading quality contributes to overall event quality but does not
                # replace motion/semantic quality.
                item["event_quality_score"] = float(
                    np.clip(
                        float(item.get("event_quality_score", 0.5))
                        * (0.75 + 0.25 * float(heading["heading_quality"])),
                        0.0,
                        1.0,
                    )
                )
                kept_intervals.append({
                    "source_uid": source_uid,
                    "source_file": original_source,
                    "seq_id": int(seq_id),
                    "start_seconds": float(item["source_start_seconds"]),
                    "end_seconds": float(item["source_end_seconds"]),
                })

                event_idx += 1
                kept_here += 1

            source_reports.append({
                "cache": str(path),
                "source": original_source,
                "seq_id": int(seq_id),
                "frames": int(len(seq)),
                "segments_proposed": int(len(segments)),
                "events_kept": int(kept_here),
                "events_dropped": int(dropped_here),
                "segmentation": seg_report,
                "source_contract": contract,
            })
            print(
                f"[V46.50 DB {file_idx}/{len(files)}] {path.name}: "
                f"segments={len(segments)} kept={kept_here} dropped={dropped_here}",
                flush=True,
            )

    if not meta:
        raise RuntimeError(
            "No valid V46.50 events built. Check retarget cache contracts and heading filters."
        )
    if expected_interval_count is not None and len(meta) != expected_interval_count:
        raise RuntimeError(
            "Target-rate Event-DB does not preserve the canonical event set: "
            f"expected={expected_interval_count}, built={len(meta)}"
        )

    db_path = save_db(
        out_dir,
        meta,
        descs,
        entries,
        exits,
        c0s,
        c1s,
        music_feats,
        music_masks,
        v46,
        cfg,
    )

    intents = [str(m.get("event_turn_intent", "none")) for m in meta]
    source_uids = [str(m.get("source_uid", "unknown")) for m in meta]
    report = {
        "schema": "v46_50_event_heading_db",
        "input_motion_dirs": args.motion_dirs,
        "output_db": str(db_path),
        "num_input_files": int(len(files)),
        "num_rejected_sources": int(len(rejected_sources)),
        "num_events": int(len(meta)),
        "num_dropped_events": int(len(dropped_events)),
        "num_source_uids": int(len(set(source_uids))),
        "intent_histogram": {
            k: int(sum(x == k for x in intents))
            for k in sorted(set(intents))
        },
        "entry_heading_abs_deg_p95": float(
            np.percentile(
                np.abs(
                    np.degrees(
                        [float(m.get("event_entry_heading_rad", 0.0)) for m in meta]
                    )
                ),
                95,
            )
        ),
        "nonturn_budget_violation_count": int(
            sum(
                abs(float(m.get("event_net_yaw_rad", 0.0)))
                > float(m.get("event_yaw_budget_rad", 0.0)) + np.radians(2.0)
                for m in meta
                if str(m.get("event_turn_intent", "none")) in {
                    "none", "uncertain_turn"
                }
            )
        ),
        "source_reports": source_reports,
        "rejected_sources": rejected_sources,
        "dropped_events": dropped_events,
    }
    report_path = out_dir / "v46_50_event_heading_db_report.json"
    report_path.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.canonical_intervals_out:
        interval_path = Path(args.canonical_intervals_out)
        interval_path.parent.mkdir(parents=True, exist_ok=True)
        interval_path.write_text(
            json.dumps(
                {
                    "schema": "canonical_event_intervals_v1",
                    "canonical_source_fps": float(cfg.fps),
                    "num_intervals": len(kept_intervals),
                    "intervals": kept_intervals,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({
        "db": str(db_path),
        "report": str(report_path),
        "num_events": len(meta),
        "num_dropped_events": len(dropped_events),
        "intent_histogram": report["intent_histogram"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
