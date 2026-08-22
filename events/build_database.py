#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the current SMPL14 heading-aware Event-RAG database.

Input must be a direct official-SMPL EDGE151D cache with an accepted source
preprocess report.  Raw BVH and historical retarget-cache contracts are not
accepted by this business path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """Validate only the direct official-SMPL source-cache contract."""
    reasons: List[str] = []
    if not report:
        reasons.append("missing_retarget_report")
        return False, reasons
    if not bool(report.get("ok", False)):
        reasons.append("retarget_report_not_ok")

    source_preprocess = report.get("source_preprocess_contract", {})
    if str(source_preprocess.get("schema", "")) != "chang_e_official_smpl_source_aware_preprocess_v1":
        reasons.append("not_official_smpl_source_preprocess")
        return False, reasons
    if not bool(report.get("source_gate_ok", False)):
        reasons.append("source_aware_source_gate_not_ok")
    if not bool(report.get("source_preprocess_ok", False)):
        reasons.append("source_aware_preprocess_not_ok")
    if bool(report.get("retargeting_applied", True)):
        reasons.append("source_aware_unexpected_retargeting")
    if not bool(source_preprocess.get("direct_official_smpl", False)):
        reasons.append("source_aware_not_direct_official_smpl")
    if bool(source_preprocess.get("retargeting_applied", True)):
        reasons.append("source_aware_contract_retargeting_applied")
    segment = report.get("preprocess_segment", {})
    if not bool(segment.get("clean", False)):
        reasons.append("source_aware_segment_not_clean")

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
    motion_runtime: Any,
    cfg: Any,
) -> Path:
    desc = np.stack(descs).astype(np.float32)
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1e-6
    desc_z = ((desc - mean) / std).astype(np.float32)


    db_path = out_dir / "events.npz"
    identity_seed = {
        "paths": _field_array(meta, "path", "", object),
        "source_uids": _field_array(meta, "source_uid", "unknown", object),
        "recording_uids": _field_array(meta, "recording_uid", "unknown", object),
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
        "event_semantics_schema_version": np.asarray(
            "chang_e_five_layer_event_semantics_v2", dtype=object
        ),
        "event_descriptor_schema_version": np.asarray(
            "edge151_local_action_descriptor_v2", dtype=object
        ),
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
            "event_heading_contract", dtype=object
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
        "source_assets": _field_array(meta, "source_asset", "", object),
        "source_formats": _field_array(meta, "source_format", "unknown", object),
        "source_uids": _field_array(meta, "source_uid", "unknown", object),
        "recording_uids": _field_array(meta, "recording_uid", "unknown", object),
        "sequence_ids": _field_array(meta, "sequence_id", "unknown", object),
        "dancer_ids": _field_array(meta, "dancer_id", "", object),
        "dancer_id_statuses": _field_array(meta, "dancer_id_status", "unverified", object),
        "manifest_sha256": _field_array(meta, "manifest_sha256", "", object),
        "performer_track_ids": _field_array(meta, "performer_track_id", -1, np.int32),
        "recording_performer_counts": _field_array(
            meta, "recording_performer_count", 1, np.int32
        ),
        "solo_compatibilities": _field_array(
            meta, "solo_compatibility", "unknown", object
        ),
        "solo_compatible": _field_array(meta, "solo_compatible", False, np.bool_),
        "solo_review_statuses": _field_array(
            meta, "solo_review_status", "unknown", object
        ),
        "sequence_indices": _field_array(meta, "sequence_index", -1, np.int32),
        "genders": _field_array(meta, "gender", "unknown", object),
        "performer_groups": _field_array(meta, "performer_group", "unknown", object),
        "labels": _field_array(meta, "label", "unknown", object),
        "parent_labels": _field_array(meta, "parent_label", "unknown", object),
        "dance_keys": _field_array(meta, "dance_key", "unknown", object),
        "dance_categories": _field_array(meta, "dance_category", "unknown", object),
        "dance_themes": _field_array(meta, "dance_theme", "unknown", object),
        "candidate_dance_categories": _field_array(
            meta, "candidate_dance_category", "", object
        ),
        "theme_label_statuses": _field_array(
            meta, "theme_label_status", "unknown", object
        ),
        "source_context_json": np.asarray(
            [json.dumps(m.get("source_context", []), sort_keys=True) for m in meta],
            dtype=object,
        ),
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
        "local_action_labels_json": np.asarray(
            [json.dumps(m.get("local_action_labels", ["unknown"])) for m in meta],
            dtype=object,
        ),
        "local_action_scores_json": _field_array(
            meta, "local_action_scores_json", "{}", object
        ),
        "music_compatibility_top_labels": _field_array(
            meta, "music_compatibility_top_label", "unknown", object
        ),
        "music_compatibility_scores_json": _field_array(
            meta, "music_compatibility_scores_json", "{}", object
        ),
        "music_compatibility_is_ground_truth": _field_array(
            meta, "music_compatibility_is_ground_truth", False, np.bool_
        ),
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

        # Event-Heading event-level heading state arrays.
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
        description="Build Event-Heading event-heading Event-RAG DB from Source-Motion NPY"
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

    import training.motion_models as motion_runtime

    cfg = motion_runtime.MotionGenerationConfig.from_json(args.config).apply_env()
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
    excluded_non_solo_sources: List[dict] = []
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
        report_metadata = rep.get("source_metadata")
        source_format = str(
            rep.get("source_format")
            or (
                report_metadata.get("source_format", "")
                if isinstance(report_metadata, dict)
                else ""
            )
        )
        if source_format != "chang_e_official_smpl":
            raise RuntimeError(
                "Current Event-DB protocol accepts only official SMPL caches; "
                f"got source_format={source_format!r}: {path}"
            )
        if not isinstance(report_metadata, dict):
            raise RuntimeError(f"Formal SMPL report lacks source_metadata: {path}")
        missing_solo_fields = sorted(
                {
                    "recording_performer_count",
                    "solo_compatibility",
                    "solo_compatible",
                    "solo_review_status",
                }
                - set(report_metadata)
            )
        if missing_solo_fields:
            raise RuntimeError(
                "Formal SMPL cache predates the solo-routing metadata contract; "
                f"rebuild it, missing={missing_solo_fields}: {path}"
            )
        formal_metadata = dict(report_metadata)
        formal_metadata["source_format"] = "chang_e_official_smpl"
        formal_metadata["manifest_sha256"] = rep.get("source_manifest_sha256")
        sem = motion_runtime.official_smpl_semantics_from_metadata(formal_metadata)
        strong_base = motion_runtime.strong_action_semantics_from_meta(sem)
        semantic_meta = {**sem, **strong_base}
        source_uid = str(sem.get("source_uid") or Path(original_source).stem)
        require_solo = str(
            os.environ.get("PERFORMER_REQUIRE_SOLO_COMPATIBLE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            require_solo
            and not bool(sem.get("solo_compatible", False))
        ):
            excluded_non_solo_sources.append(
                {
                    "source_uid": source_uid,
                    "recording_uid": sem.get("recording_uid"),
                    "recording_performer_count": sem.get(
                        "recording_performer_count"
                    ),
                    "solo_compatibility": sem.get("solo_compatibility"),
                    "solo_review_status": sem.get("solo_review_status"),
                    "reason": "unreviewed_multi_performer_recording_excluded_from_formal_single_person_db",
                }
            )
            continue
        preprocess_segment = rep.get("preprocess_segment", {})
        source_segment_index = int(preprocess_segment.get("segment_index", 0))
        source_segment_start_seconds = float(preprocess_segment.get("source_start_seconds", 0.0))
        source_duration_seconds = float(rep.get("source_duration_seconds", 0.0) or 0.0)

        for local_seq_id, seq0 in enumerate(seqs):
            seq_id = source_segment_index + int(local_seq_id)
            seq, contract = motion_runtime.enforce_edge151_contract_np(
                seq0,
                cfg,
                source_hint=f"event_heading_source:{path}",
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

                clip, final_contract = motion_runtime.enforce_edge151_contract_np(
                    clip,
                    cfg,
                    source_hint=f"event_heading_event:{path}:{st}:{ed}",
                    derive_contact=True,
                    project_rot=True,
                )
                out_path = event_dir / f"event_{event_idx:07d}.npy"
                identity_start_seconds = (
                    float(canonical_row["start_seconds"])
                    if canonical_row is not None
                    else source_segment_start_seconds + float(st) / float(cfg.fps)
                )
                identity_end_seconds = (
                    float(canonical_row["end_seconds"])
                    if canonical_row is not None
                    else source_segment_start_seconds + float(ed) / float(cfg.fps)
                )
                base_meta = {
                    **sem,
                    "source_file": original_source,
                    "source_asset": original_source,
                    "source_format": "chang_e_official_smpl",
                    "load_path": str(path),
                    "source_uid": source_uid,
                    "source_group": source_uid,
                    "recording_uid": sem.get("recording_uid", source_uid),
                    "seq_id": int(seq_id),
                    "label": sem.get("label", Path(original_source).stem),
                    "parent_label": sem.get("parent_label", sem.get("label", "unknown")),
                    "fragment_index": int(seg_idx),
                    "source_segment_index": int(source_segment_index),
                    "source_segment_start_seconds": float(source_segment_start_seconds),
                    "input_mode": "chang_e_official_smpl_source_aware_cache",
                    "event_start": int(st),
                    "event_end": int(ed),
                    "event_source_frames": int(len(seq)),
                    # Preserve the canonical manifest interval verbatim.  The
                    # local frame bounds are quantized execution details and
                    # must not change event identity across FPS branches.
                    "source_start_seconds": identity_start_seconds,
                    "source_end_seconds": identity_end_seconds,
                    "canonical_fps": float(cfg.fps),
                    "event_position_mid": float(
                        np.clip(
                            (
                                source_segment_start_seconds
                                + (st + ed) * 0.5 / float(cfg.fps)
                            ) / source_duration_seconds,
                            0.0,
                            1.0,
                        )
                        if source_duration_seconds > 0.0
                        else (st + ed) * 0.5 / max(len(seq), 1)
                    ),
                    "resample_report": {
                        "resampled": False,
                        "native_fps": float(cfg.fps),
                        "target_fps": float(cfg.fps),
                        "source": "source_motion_cache",
                    },
                }

                motion_runtime.add_event_to_db_lists(
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
                    "event_segmentation_schema": "event_heading_motion_adaptive_segmentation",
                    "retarget_contract_source": rep.get("version", "source_gravity_4"),
                    "edge151_contract_report": {
                        **dict(item.get("edge151_contract_report", {})),
                        "event_heading_final": final_contract,
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
                f"[Event-Heading DB {file_idx}/{len(files)}] {path.name}: "
                f"segments={len(segments)} kept={kept_here} dropped={dropped_here}",
                flush=True,
            )

    if not meta:
        raise RuntimeError(
            "No valid Event-Heading events built. Check retarget cache contracts and heading filters."
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
        motion_runtime,
        cfg,
    )

    intents = [str(m.get("event_turn_intent", "none")) for m in meta]
    source_uids = [str(m.get("source_uid", "unknown")) for m in meta]
    report = {
        "schema": "event_heading_db",
        "input_motion_dirs": args.motion_dirs,
        "output_db": str(db_path),
        "num_input_files": int(len(files)),
        "num_rejected_sources": int(len(rejected_sources)),
        "num_excluded_non_solo_sources": int(len(excluded_non_solo_sources)),
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
        "excluded_non_solo_sources": excluded_non_solo_sources,
        "dropped_events": dropped_events,
    }
    report_path = out_dir / "event_heading_db_report.json"
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
