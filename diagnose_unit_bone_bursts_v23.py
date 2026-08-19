#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import retargeting.bvh_solver as legacy
from contracts.gravity import fk24_np
from data_pipeline.chang_e_manifest import validate_source
from motion_geometry.physical import EXTREMITY_JOINTS
from motion_geometry.smpl24 import JOINT_NAMES, NUM_JOINTS, PARENTS
from retargeting.build_cache import _common_valid_mapped_bone_contract


def build_reference_trajectory(source: Path, *, manifest: Path, fps: float) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """Reproduce build_cache V2.2 source-reference trajectory exactly."""
    bvh = legacy.parse_bvh(source)
    timebase = validate_source(source, path=manifest, required=True, verify_hash=True)
    source_fps = float(timebase["effective_fps"])

    native_pos, _ = legacy.source_fk(bvh, use_motion=True)
    rest_bvh = legacy.BVHMotion(bvh.path, bvh.joints, np.zeros_like(bvh.values), bvh.frame_time)
    rest_pos, _ = legacy.source_fk(rest_bvh, use_motion=False)
    mapping = legacy.build_joint_mapping(bvh.joints)
    target_rest = legacy.target_rest_positions()

    pairs: list[tuple[int, int]] = []
    used_src: set[int] = set()
    for tgt, src in mapping.items():
        if src in used_src or tgt in {22, 23}:
            continue
        used_src.add(src)
        pairs.append((int(tgt), int(src)))

    X = np.asarray([rest_pos[0, src] for tgt, src in pairs], dtype=np.float32)
    Y = np.asarray([target_rest[tgt] for tgt, src in pairs], dtype=np.float32)
    W = np.asarray([legacy.TARGET_JOINT_WEIGHTS[tgt] for tgt, src in pairs], dtype=np.float32)
    scale, basis_R, trans = legacy.similarity_umeyama(X, Y, W)

    aligned = legacy.apply_similarity(native_pos, scale, basis_R, trans)
    aligned = legacy.resample_global_positions(aligned, source_fps, float(fps))
    aligned, heading_report = legacy.stabilize_source_heading_positions(
        aligned, mapping, float(fps), timebase.get("entry", {})
    )

    bones, contract = _common_valid_mapped_bone_contract(bvh, mapping, aligned, target_rest)

    T = int(len(aligned))
    target_pos = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    observed = np.zeros((NUM_JOINTS,), dtype=bool)
    for tgt, src in mapping.items():
        target_pos[:, int(tgt)] = aligned[:, int(src)]
        observed[int(tgt)] = True
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
        raise RuntimeError(f"Reference target observations missing joints: {missing}")

    meta = {
        "source_fps": source_fps,
        "target_fps": float(fps),
        "similarity_scale": float(scale),
        "heading_contract": heading_report,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
        "common_mapped_bone_contract": contract,
    }
    return target_pos, list(bones), meta


def unit_bone_jerk(joints: np.ndarray, bones: Iterable[int], fps: float) -> np.ndarray:
    bones = [int(b) for b in bones]
    children = np.asarray(bones, dtype=np.int64)
    parents = np.asarray([int(PARENTS[b]) for b in bones], dtype=np.int64)
    pos = np.asarray(joints, dtype=np.float64)
    vectors = pos[:, children] - pos[:, parents]
    lengths = np.linalg.norm(vectors, axis=-1)
    eps = float(os.environ.get("SOURCE_PHYSICAL_UNIT_BONE_LENGTH_EPS_M", "1e-5"))
    unit = vectors / np.maximum(lengths[..., None], eps)
    jerk = np.diff(unit, n=3, axis=0) * float(fps) ** 3
    return np.linalg.norm(jerk, axis=-1)


def window_starts(length: int, fps: float) -> tuple[int, list[int]]:
    window = min(length, max(1, int(round(float(fps)))))
    hop = max(1, window // 2)
    starts = list(range(0, max(1, length - window + 1), hop))
    final_start = max(0, length - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return window, starts


def worst_aggregate_window(values: np.ndarray, columns: list[int], fps: float) -> dict[str, Any]:
    x = np.asarray(values[:, columns], dtype=np.float64)
    window, starts = window_starts(len(x), fps)
    scored = []
    for start in starts:
        end = start + window
        score = float(np.percentile(x[start:end], 95)) if x[start:end].size else 0.0
        scored.append((score, start, end))
    score, start, end = max(scored, key=lambda item: item[0])
    return {"p95": score, "jerk_start": int(start), "jerk_end_exclusive": int(end), "window_jerk_frames": int(window)}


def recursively_find_int(obj: Any, wanted: set[str]) -> int | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower().replace("-", "_")
            if key in wanted:
                try:
                    value = int(v)
                    if value > 0:
                        return value
                except (TypeError, ValueError):
                    pass
        for v in obj.values():
            found = recursively_find_int(v, wanted)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = recursively_find_int(v, wanted)
            if found is not None:
                return found
    return None


def seam_info(report_path: Path, total_frames: int, cfg: legacy.RetargetConfig) -> dict[str, Any]:
    report = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    chunk = recursively_find_int(report, {"chunk_frames", "source_retarget_chunk", "retarget_chunk"}) or int(cfg.chunk_frames)
    overlap = recursively_find_int(report, {"chunk_overlap", "source_retarget_overlap", "retarget_overlap"}) or int(cfg.chunk_overlap)
    if overlap >= chunk:
        overlap = max(0, chunk // 4)
    stride = max(1, chunk - overlap)
    chunk_starts = list(range(0, total_frames, stride))
    # Chunk start is the beginning of an overlap/blend region for all chunks after the first.
    blend_starts = [s for s in chunk_starts[1:] if s < total_frames]
    blend_ends = [min(total_frames - 1, s + overlap - 1) for s in blend_starts]
    return {
        "chunk_frames": int(chunk),
        "chunk_overlap": int(overlap),
        "chunk_stride": int(stride),
        "chunk_starts": chunk_starts,
        "blend_starts": blend_starts,
        "blend_ends": blend_ends,
    }


def nearest_boundary(frame: int, seams: dict[str, Any]) -> dict[str, Any]:
    points: list[tuple[str, int]] = []
    for key in ("blend_starts", "blend_ends"):
        label = "blend_start" if key == "blend_starts" else "blend_end"
        points.extend((label, int(v)) for v in seams.get(key, []))
    if not points:
        return {"type": None, "frame": None, "distance_frames": None}
    label, point = min(points, key=lambda kv: abs(frame - kv[1]))
    return {"type": label, "frame": point, "distance_frames": int(abs(frame - point))}


def analyse_source(root: Path, source_name: str, fps: float) -> dict[str, Any]:
    bvh_path = root / "assets/motion/bvh" / f"{source_name}.bvh"
    manifest = root / "assets/motion/bvh/sources.json"
    rejected_root = root / "output/source_contract_validation_v2/retarget_cache_rejected"
    motion_path = rejected_root / f"{source_name}.rejected.npy"
    report_path = rejected_root / f"{source_name}.rejected.retarget.json"

    if not bvh_path.is_file():
        raise FileNotFoundError(bvh_path)
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)

    cfg = legacy.RetargetConfig.from_env()
    cfg.target_fps = float(fps)
    reference_joints, bones, reference_meta = build_reference_trajectory(bvh_path, manifest=manifest, fps=fps)
    motion = np.load(motion_path).astype(np.float32)
    candidate_joints = fk24_np(motion)
    if abs(len(candidate_joints) - len(reference_joints)) > 1:
        raise RuntimeError(f"frame mismatch candidate={len(candidate_joints)} reference={len(reference_joints)}")
    T = min(len(candidate_joints), len(reference_joints))
    candidate_joints = candidate_joints[:T]
    reference_joints = reference_joints[:T]

    cand = unit_bone_jerk(candidate_joints, bones, fps)
    ref = unit_bone_jerk(reference_joints, bones, fps)
    if cand.shape != ref.shape:
        raise RuntimeError(f"jerk shape mismatch: candidate={cand.shape}, reference={ref.shape}")

    extremity_cols = [i for i, b in enumerate(bones) if int(b) in EXTREMITY_JOINTS]
    all_cols = list(range(len(bones)))
    seams = seam_info(report_path, T, cfg)

    per_bone = []
    for col, child in enumerate(bones):
        parent = int(PARENTS[child])
        c = cand[:, col]
        r = ref[:, col]
        c95 = float(np.percentile(c, 95)) if c.size else 0.0
        r95 = float(np.percentile(r, 95)) if r.size else 0.0
        c99 = float(np.percentile(c, 99)) if c.size else 0.0
        r99 = float(np.percentile(r, 99)) if r.size else 0.0
        peak_idx = int(np.argmax(c)) if c.size else 0
        peak_frame = peak_idx + 3
        per_bone.append({
            "child": int(child),
            "child_name": str(JOINT_NAMES[child]),
            "parent": parent,
            "parent_name": str(JOINT_NAMES[parent]),
            "is_extremity": bool(child in EXTREMITY_JOINTS),
            "candidate_p95_s3": c95,
            "reference_p95_s3": r95,
            "p95_ratio": c95 / max(r95, 1e-12),
            "candidate_p99_s3": c99,
            "reference_p99_s3": r99,
            "p99_ratio": c99 / max(r99, 1e-12),
            "candidate_max_s3": float(np.max(c)) if c.size else 0.0,
            "reference_max_s3": float(np.max(r)) if r.size else 0.0,
            "candidate_peak_frame": peak_frame,
            "nearest_chunk_blend_boundary": nearest_boundary(peak_frame, seams),
        })

    subset_reports: dict[str, Any] = {}
    for label, cols in (("joint", all_cols), ("extremity", extremity_cols)):
        if not cols:
            continue
        cwin = worst_aggregate_window(cand, cols, fps)
        rwin = worst_aggregate_window(ref, cols, fps)
        c_start, c_end = cwin["jerk_start"], cwin["jerk_end_exclusive"]
        orig_start, orig_end = c_start + 3, c_end + 2
        cseg = cand[c_start:c_end, cols]
        rseg_same = ref[c_start:c_end, cols]
        contributors = []
        for local_col, global_col in enumerate(cols):
            child = int(bones[global_col])
            cp95 = float(np.percentile(cseg[:, local_col], 95)) if cseg.size else 0.0
            rp95 = float(np.percentile(rseg_same[:, local_col], 95)) if rseg_same.size else 0.0
            contributors.append({
                "child": child,
                "child_name": str(JOINT_NAMES[child]),
                "parent": int(PARENTS[child]),
                "parent_name": str(JOINT_NAMES[int(PARENTS[child])]),
                "candidate_window_p95_s3": cp95,
                "reference_same_window_p95_s3": rp95,
                "same_window_ratio": cp95 / max(rp95, 1e-12),
            })
        contributors.sort(key=lambda d: d["candidate_window_p95_s3"], reverse=True)
        midpoint = (orig_start + orig_end) // 2
        subset_reports[label] = {
            "candidate_worst_window": {
                **cwin,
                "original_frame_start": int(orig_start),
                "original_frame_end": int(orig_end),
                "nearest_chunk_blend_boundary_to_midpoint": nearest_boundary(midpoint, seams),
                "top_contributors": contributors[:10],
            },
            "reference_own_worst_window": {
                **rwin,
                "original_frame_start": int(rwin["jerk_start"] + 3),
                "original_frame_end": int(rwin["jerk_end_exclusive"] + 2),
            },
        }

    return {
        "schema": "unit_bone_local_burst_diagnostic_v2_3",
        "source": source_name,
        "fps": float(fps),
        "frames": int(T),
        "comparison_bones": list(bones),
        "comparison_bone_names": [str(JOINT_NAMES[b]) for b in bones],
        "reference_meta": reference_meta,
        "chunk_seams": seams,
        "subsets": subset_reports,
        "top_by_candidate_p95": sorted(per_bone, key=lambda d: d["candidate_p95_s3"], reverse=True)[:12],
        "top_by_p95_ratio": sorted(per_bone, key=lambda d: d["p95_ratio"], reverse=True)[:12],
        "top_by_candidate_max": sorted(per_bone, key=lambda d: d["candidate_max_s3"], reverse=True)[:12],
        "per_bone": per_bone,
        "paths": {"source_bvh": str(bvh_path), "candidate_motion": str(motion_path), "rejected_report": str(report_path)},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "child", "child_name", "parent", "parent_name", "is_extremity",
        "candidate_p95_s3", "reference_p95_s3", "p95_ratio",
        "candidate_p99_s3", "reference_p99_s3", "p99_ratio",
        "candidate_max_s3", "reference_max_s3", "candidate_peak_frame",
        "nearest_boundary_type", "nearest_boundary_frame", "nearest_boundary_distance_frames",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            b = row.get("nearest_chunk_blend_boundary", {})
            out = {k: row.get(k) for k in fields if not k.startswith("nearest_boundary_")}
            out.update({
                "nearest_boundary_type": b.get("type"),
                "nearest_boundary_frame": b.get("frame"),
                "nearest_boundary_distance_frames": b.get("distance_frames"),
            })
            w.writerow(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("PROJECT_ROOT", "/home/disk/lsm/storage/DunhuangDanceRAG"))
    ap.add_argument("--sources", nargs="+", default=["male_36pose_1", "male_drum_2"])
    ap.add_argument("--fps", type=float, default=float(os.environ.get("RETARGET_REQUALIFY_FPS", "30.0")))
    ap.add_argument("--out-dir", default="output/source_contract_validation_v2/burst_diagnostics_v2_3")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        result = analyse_source(root, source, args.fps)
        json_path = out_dir / f"{source}.burst.json"
        csv_path = out_dir / f"{source}.per_bone.csv"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        write_csv(csv_path, result["per_bone"])

        print("=" * 110)
        print(source)
        print("=" * 110)
        print("frames:", result["frames"], "fps:", result["fps"])
        print("chunk:", result["chunk_seams"]["chunk_frames"], "overlap:", result["chunk_seams"]["chunk_overlap"], "stride:", result["chunk_seams"]["chunk_stride"])
        for subset in ("joint", "extremity"):
            if subset not in result["subsets"]:
                continue
            win = result["subsets"][subset]["candidate_worst_window"]
            print(f"\n{subset.upper()} candidate worst 1s window:")
            print("  p95_s3 =", win["p95"])
            print("  frames =", win["original_frame_start"], "..", win["original_frame_end"])
            print("  nearest blend boundary =", win["nearest_chunk_blend_boundary_to_midpoint"])
            print("  top contributors:")
            for row in win["top_contributors"][:6]:
                print(
                    f"    {row['parent_name']}->{row['child_name']:<12s} "
                    f"cand_p95={row['candidate_window_p95_s3']:.3f} "
                    f"ref_same={row['reference_same_window_p95_s3']:.3f} "
                    f"ratio={row['same_window_ratio']:.3f}"
                )
        print("\nTop bones by candidate p95:")
        for row in result["top_by_candidate_p95"][:8]:
            print(
                f"  {row['parent_name']}->{row['child_name']:<12s} "
                f"cand={row['candidate_p95_s3']:.3f} ref={row['reference_p95_s3']:.3f} "
                f"ratio={row['p95_ratio']:.3f} peak_frame={row['candidate_peak_frame']} "
                f"seam={row['nearest_chunk_blend_boundary']}"
            )
        print("SAVED:", json_path)
        print("SAVED:", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
