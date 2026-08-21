#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Category-covered, exact-cardinality, recording-disjoint cache split.

Priority order for the local Chang-E subset:
1. no source or synchronized-recording leakage;
2. non-empty train/validation/test;
3. every confirmed held-out dance theme is represented in training;
4. gender-group coverage in validation and test when feasible.

The 14 official SMPL performer tracks form 11 recording groups because three two-person recordings
are exported as separate performer tracks.  Exact split counts therefore apply
to recording groups rather than files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import training.motion_models as motion_api
except Exception:  # package self-test fallback; real projects provide this module
    class _MotionApiFallback:
        @staticmethod
        def parse_change_bvh_semantics(source: str):
            stem = Path(str(source)).stem
            lower = stem.lower()
            performer = (
                "female" if "female" in lower
                else "male" if "male" in lower
                else "unknown"
            )
            return {
                "source_uid": stem,
                "dance_key": stem,
                "performer_group": performer,
            }
    motion_api = _MotionApiFallback()

SPLITS = ("train", "val", "test")
SCHEMA = "category_covered_recording_disjoint_cache_split_v3"


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_int(text: str, seed: int) -> int:
    return int(hashlib.sha256(("%d::%s" % (seed, text)).encode()).hexdigest()[:16], 16)


def exact_split_counts(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    if n < 3:
        raise ValueError("At least three complete sources are required")
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive number")
    ratios = [value / total for value in ratios]
    ideal = [n * value for value in ratios]
    counts = [int(value) for value in ideal]
    remaining = n - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (
            ideal[index] - counts[index],
            ratios[index],
            -index,
        ),
        reverse=True,
    )
    for index in range(remaining):
        counts[order[index % 3]] += 1
    for receiver in (1, 2, 0):
        if counts[receiver] == 0:
            donor = max(
                (index for index in range(3) if counts[index] > 1),
                key=lambda index: counts[index],
            )
            counts[donor] -= 1
            counts[receiver] += 1
    return dict(zip(SPLITS, counts))


def infer_performer_group(semantic: Mapping[str, Any], source: str) -> str:
    raw = str(
        semantic.get("performer_group")
        or semantic.get("gender")
        or semantic.get("genders")
        or ""
    ).strip().lower()
    name = (str(source) + " " + str(semantic.get("source_uid", ""))).lower()
    if raw in {"female", "woman", "women", "f"} or "female" in name:
        return "female"
    if raw in {"male", "man", "men", "m"} or "male" in name:
        return "male"
    return "unknown"


def _compositions(total: int, capacities: Tuple[int, int, int]):
    for train in range(min(total, capacities[0]) + 1):
        for val in range(min(total - train, capacities[1]) + 1):
            test = total - train - val
            if 0 <= test <= capacities[2]:
                yield (train, val, test)


def performer_capacities(
    records: Sequence[Mapping[str, Any]],
    target: Mapping[str, int],
) -> Dict[str, Dict[str, int]]:
    group_counts = Counter(str(row["performer_group"]) for row in records)
    groups = sorted(group_counts)
    capacities = tuple(int(target[split]) for split in SPLITS)
    total_sources = len(records)
    best: Optional[Tuple[float, Dict[str, Tuple[int, int, int]]]] = None

    def search(
        group_index: int,
        remaining: Tuple[int, int, int],
        rows: Dict[str, Tuple[int, int, int]],
    ) -> None:
        nonlocal best
        if group_index == len(groups):
            if remaining != (0, 0, 0):
                return
            score = 0.0
            for group, allocation in rows.items():
                count = group_counts[group]
                ideal = [
                    count * int(target[split]) / float(total_sources)
                    for split in SPLITS
                ]
                score += sum(
                    (allocation[index] - ideal[index]) ** 2
                    for index in range(3)
                )
                # With >=3 sources in a known group, held-out coverage is a
                # scientific requirement rather than a cosmetic preference.
                if group in {"female", "male"} and count >= 3:
                    if allocation[1] == 0:
                        score += 1000.0
                    if allocation[2] == 0:
                        score += 1000.0
            candidate = (score, dict(rows))
            if best is None or candidate[0] < best[0]:
                best = candidate
            return

        group = groups[group_index]
        count = group_counts[group]
        for allocation in _compositions(count, remaining):
            next_remaining = tuple(
                remaining[index] - allocation[index] for index in range(3)
            )
            search(
                group_index + 1,
                next_remaining,
                dict(rows, **{group: allocation}),
            )

    search(0, capacities, {})
    if best is None:
        raise RuntimeError("No feasible performer-stratified split exists")
    return {
        group: dict(zip(SPLITS, allocation))
        for group, allocation in best[1].items()
    }


def assign_records(
    records: Sequence[Mapping[str, Any]],
    target: Mapping[str, int],
    seed: int,
) -> Dict[str, str]:
    capacities = performer_capacities(records, target)
    assignment: Dict[str, str] = {}
    for group in sorted(capacities):
        group_rows = [
            row for row in records if str(row["performer_group"]) == group
        ]
        category_total = Counter(str(row["dance_key"]) for row in group_rows)
        ordered = sorted(
            group_rows,
            key=lambda row: (
                category_total[str(row["dance_key"])],
                stable_int(str(row["source_uid"]), seed),
            ),
        )
        remaining = dict(capacities[group])
        category_counts = {split: Counter() for split in SPLITS}
        for row in ordered:
            category = str(row["dance_key"])
            options = [split for split in SPLITS if remaining[split] > 0]
            if not options:
                raise RuntimeError("Performer split capacity exhausted")
            chosen = min(
                options,
                key=lambda split: (
                    category_counts[split][category]
                    / max(1, capacities[group][split])
                    + 0.25
                    * (capacities[group][split] - remaining[split])
                    / max(1, capacities[group][split]),
                    stable_int(
                        "%s::%s" % (row["source_uid"], split),
                        seed,
                    ),
                ),
            )
            assignment[str(row["source_uid"])] = chosen
            remaining[chosen] -= 1
            category_counts[chosen][category] += 1
    return assignment


def assign_records_category_covered(
    records: Sequence[Mapping[str, Any]],
    target: Mapping[str, int],
    seed: int,
) -> Dict[str, str]:
    """Assign recording units under a hard confirmed-theme coverage contract."""

    rows = [dict(row) for row in records]
    n = len(rows)
    if n > 20:
        raise RuntimeError(
            "Category-covered exhaustive split currently supports at most 20 "
            f"recording groups; got {n}"
        )
    val_n = int(target["val"])
    test_n = int(target["test"])
    all_indices = set(range(n))
    confirmed_categories = {
        str(row["dance_key"])
        for row in rows
        if str(row.get("theme_label_status", "confirmed")) == "confirmed"
        and str(row.get("dance_key", "unknown")) != "unknown"
    }

    best: Optional[Tuple[float, Tuple[str, ...], Dict[str, str]]] = None
    for val_indices_tuple in combinations(range(n), val_n):
        val_indices = set(val_indices_tuple)
        remaining = sorted(all_indices - val_indices)
        for test_indices_tuple in combinations(remaining, test_n):
            test_indices = set(test_indices_tuple)
            train_indices = all_indices - val_indices - test_indices
            split_indices = {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
            train_categories = {
                str(rows[index]["dance_key"])
                for index in train_indices
                if str(rows[index].get("theme_label_status", "confirmed")) == "confirmed"
                and str(rows[index].get("dance_key", "unknown")) != "unknown"
            }
            if not confirmed_categories.issubset(train_categories):
                continue
            heldout_confirmed = {
                str(rows[index]["dance_key"])
                for index in val_indices | test_indices
                if str(rows[index].get("theme_label_status", "confirmed")) == "confirmed"
                and str(rows[index].get("dance_key", "unknown")) != "unknown"
            }
            if not heldout_confirmed.issubset(train_categories):
                continue

            score = 0.0
            # Gender is a reporting stratum, not a dancer identity. Reward its
            # held-out coverage only where both groups have enough recordings.
            for split in ("val", "test"):
                groups = {
                    str(rows[index].get("performer_group", "unknown"))
                    for index in split_indices[split]
                }
                for group in ("female", "male"):
                    total = sum(
                        str(row.get("performer_group", "unknown")) == group
                        for row in rows
                    )
                    if total >= 3 and group not in groups:
                        score += 5.0

            # Prefer distribution of repeatable themes across both held-out
            # sets, without forcing single-recording themes out of training.
            category_counts = Counter(
                str(row["dance_key"])
                for row in rows
                if str(row.get("theme_label_status", "confirmed")) == "confirmed"
            )
            for category, count in category_counts.items():
                if category == "unknown" or count < 2:
                    continue
                split_presence = sum(
                    any(str(rows[index]["dance_key"]) == category for index in split_indices[split])
                    for split in SPLITS
                )
                score += float(3 - split_presence)

            signature = tuple(
                sorted(
                    f"{rows[index]['source_uid']}::{split}"
                    for split, indices in split_indices.items()
                    for index in indices
                )
            )
            tie = stable_int("|".join(signature), seed) / float(2**64)
            assignment = {
                str(rows[index]["source_uid"]): split
                for split, indices in split_indices.items()
                for index in indices
            }
            candidate = (score + tie * 1.0e-6, signature, assignment)
            if best is None or candidate[:2] < best[:2]:
                best = candidate

    if best is None:
        raise RuntimeError(
            "No exact recording-disjoint split satisfies confirmed-theme "
            "training coverage. Use leave_one_theme_out for a unique held-out theme."
        )
    return best[2]


def leave_one_theme_out_assignment(
    records: Sequence[Mapping[str, Any]],
    *,
    heldout_theme: str,
    seed: int,
) -> Dict[str, str]:
    """Isolate one confirmed theme as a zero-shot test protocol."""

    heldout = str(heldout_theme).strip()
    test_rows = [row for row in records if str(row.get("dance_key")) == heldout]
    if not test_rows:
        raise ValueError(f"No recording group for heldout_theme={heldout!r}")
    if any(str(row.get("theme_label_status", "")) != "confirmed" for row in test_rows):
        raise ValueError("leave-one-theme-out requires a confirmed theme")
    remaining = [row for row in records if str(row.get("dance_key")) != heldout]
    if len(remaining) < 2:
        raise RuntimeError("leave-one-theme-out requires at least two non-heldout groups")
    val_row = min(
        remaining,
        key=lambda row: stable_int(str(row["source_uid"]), seed),
    )
    assignment = {str(row["source_uid"]): "train" for row in remaining}
    assignment[str(val_row["source_uid"])] = "val"
    for row in test_rows:
        assignment[str(row["source_uid"])] = "test"
    return assignment


def assign_sources(
    source_to_label: Mapping[str, str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    """Backward-compatible API used by existing source-split tests."""
    records = []
    for source_uid, dance_key in source_to_label.items():
        name = str(source_uid).lower()
        performer = (
            "female" if "female" in name
            else "male" if "male" in name
            else "unknown"
        )
        records.append({
            "source_uid": source_uid,
            "dance_key": dance_key,
            "performer_group": performer,
        })
    target = exact_split_counts(
        len(records), train_ratio, val_ratio, test_ratio
    )
    return assign_records(records, target, seed)


def recording_group_records(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse synchronized performer tracks into indivisible split units."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in records:
        recording_uid = str(row.get("recording_uid") or row["source_uid"])
        grouped.setdefault(recording_uid, []).append(row)

    units: List[Dict[str, Any]] = []
    for recording_uid, rows in sorted(grouped.items()):
        performers = {str(row["performer_group"]) for row in rows}
        categories = {str(row["dance_key"]) for row in rows}
        theme_statuses = {
            str(row.get("theme_label_status", "confirmed")) for row in rows
        }
        if len(performers) != 1:
            raise RuntimeError(
                f"recording_uid {recording_uid!r} mixes performer groups: {performers}"
            )
        if len(categories) != 1:
            raise RuntimeError(
                f"recording_uid {recording_uid!r} mixes dance categories: {categories}"
            )
        if len(theme_statuses) != 1:
            raise RuntimeError(
                f"recording_uid {recording_uid!r} mixes theme statuses: {theme_statuses}"
            )
        units.append({
            # assign_records uses source_uid as its generic assignment key.
            "source_uid": recording_uid,
            "recording_uid": recording_uid,
            "performer_group": next(iter(performers)),
            "dance_key": next(iter(categories)),
            "theme_label_status": next(iter(theme_statuses)),
            "dancer_ids": sorted(
                {
                    str(row["dancer_id"])
                    for row in rows
                    if row.get("dancer_id")
                    and str(row.get("dancer_id_status")) == "verified"
                }
            ),
            "dancer_identity_verified": bool(rows) and all(
                bool(row.get("dancer_id"))
                and str(row.get("dancer_id_status")) == "verified"
                for row in rows
            ),
            "source_uids": sorted({str(row["source_uid"]) for row in rows}),
            "num_performer_tracks": len({str(row["source_uid"]) for row in rows}),
            "num_segments": len(rows),
        })
    return units


def report_path_for_motion(path: Path) -> Path:
    return path.with_suffix(".retarget.json")


def source_record(cache_root: Path, motion_path: Path) -> Dict[str, Any]:
    report_path = report_path_for_motion(motion_path)

    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    if not bool(report.get("ok", False)):
        raise RuntimeError(
            "Source preprocess report is not OK: "
            f"{motion_path}"
        )

    if not bool(
        report.get(
            "source_gate_ok",
            report.get("anatomy_ok", False),
        )
    ):
        raise RuntimeError(
            f"Source-safety gate failed: {motion_path}"
        )

    relative_motion = motion_path.relative_to(
        cache_root
    )

    report_metadata = report.get(
        "source_metadata"
    )

    source_format = str(
        report.get("source_format")
        or (
            report_metadata.get("source_format", "")
            if isinstance(report_metadata, Mapping)
            else ""
        )
    ).strip()

    preprocess_contract = report.get(
        "source_preprocess_contract"
    )

    if (
        isinstance(preprocess_contract, Mapping)
        and bool(preprocess_contract.get("direct_official_smpl", False))
        and source_format != "chang_e_official_smpl"
    ):
        raise RuntimeError(
            "Direct official-SMPL report must declare "
            "source_format=chang_e_official_smpl: "
            f"{report_path}"
        )

    original = str(
        report.get("source_used")
        or report.get("source")
        or report.get("source_relative")
        or relative_motion
    )

    # Formal SMPL path: authoritative metadata comes directly from
    # official_smpl_source_preprocess.py.  No BVH-name parser is involved.
    if source_format == "chang_e_official_smpl":
        if not (
            isinstance(report_metadata, Mapping)
            and report_metadata.get("source_id")
        ):
            raise RuntimeError(
                "Official SMPL report is missing source_metadata.source_id: "
                f"{report_path}"
            )

        semantic = {
            "source_uid": str(
                report_metadata["source_id"]
            ),
            "recording_uid": str(
                report_metadata.get(
                    "recording_uid",
                    report_metadata["source_id"],
                )
            ),
            "sequence_id": str(
                report_metadata.get("sequence_id", report_metadata["source_id"])
            ),
            "dancer_id": report_metadata.get("dancer_id"),
            "dancer_id_status": report_metadata.get(
                "dancer_id_status", "unverified"
            ),
            "performer_track_id": (
                report_metadata.get(
                    "performer_track_id",
                    -1,
                )
            ),
            "sequence_index": (
                report_metadata.get(
                    "sequence_index",
                    -1,
                )
            ),
            "performer_group": (
                report_metadata.get(
                    "performer_group",
                    "unknown",
                )
            ),
            "dance_key": (
                report_metadata.get(
                    "dance_category",
                    "unknown",
                )
            ),
            "dance_category": (
                report_metadata.get(
                    "dance_category",
                    "unknown",
                )
            ),
            "candidate_dance_category": report_metadata.get(
                "candidate_dance_category"
            ),
            "theme_label_status": report_metadata.get(
                "theme_label_status", "confirmed"
            ),
            "source_context": list(report_metadata.get("source_context", [])),
            "manifest_sha256": report.get("source_manifest_sha256"),
            "take_id": report_metadata.get(
                "take_id"
            ),
            "skeleton_id": (
                report_metadata.get(
                    "skeleton_id",
                    "chang_e_official_smpl",
                )
            ),
            "source_format": (
                "chang_e_official_smpl"
            ),
        }

    else:
        # Legacy / ablation-only BVH compatibility.
        legacy_original = str(
            report.get("source_used")
            or report.get("source")
            or report.get("source_relative")
            or relative_motion.with_suffix(".bvh")
        )

        original = legacy_original

        semantic = (
            motion_api.parse_change_bvh_semantics(
                legacy_original
            )
        )

    source_uid = str(
        semantic.get("source_uid")
        or Path(original).stem
    )

    dance_key = str(
        semantic.get("dance_key")
        or semantic.get("dance_category")
        or "unknown"
    )

    performer = infer_performer_group(
        semantic,
        original,
    )

    anatomy_payload = report.get(
        "anatomy_diagnostic"
    )

    if not isinstance(
        anatomy_payload,
        Mapping,
    ):
        anatomy_payload = report.get(
            "anatomy",
            {},
        )

    return {
        "motion": str(
            motion_path.resolve()
        ),
        "report": str(
            report_path.resolve()
        ),
        "relative_motion": str(
            relative_motion
        ),
        "relative_report": str(
            report_path.relative_to(
                cache_root
            )
        ),
        "original_source": original,
        "source_uid": source_uid,
        "recording_uid": str(
            semantic.get(
                "recording_uid",
                source_uid,
            )
            or source_uid
        ),
        "sequence_id": str(
            semantic.get("sequence_id", semantic.get("recording_uid", source_uid))
            or source_uid
        ),
        "dancer_id": semantic.get("dancer_id"),
        "dancer_id_status": semantic.get("dancer_id_status", "unverified"),
        "performer_track_id": (
            semantic.get(
                "performer_track_id",
                -1,
            )
        ),
        "sequence_index": (
            semantic.get(
                "sequence_index",
                -1,
            )
        ),
        "dance_key": dance_key,
        "theme_label_status": semantic.get("theme_label_status", "legacy_or_unknown"),
        "candidate_dance_category": semantic.get("candidate_dance_category"),
        "source_context": semantic.get("source_context", []),
        "manifest_sha256": semantic.get("manifest_sha256"),
        "performer_group": performer,
        "source_anatomy_quality": float(
            anatomy_payload.get(
                "anatomy_quality",
                0.0,
            )
        ),
        "source_gate_reasons": list(
            report.get(
                "source_gate_reasons",
                [],
            )
        ),
        "semantic": semantic,
    }


def materialize(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
        return "copy"
    if mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            shutil.copy2(source, target)
            return "copy_fallback"
    try:
        os.symlink(source.resolve(), target)
        return "symlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy_fallback"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--train_ratio", type=float, default=0.67)
    parser.add_argument("--val_ratio", type=float, default=0.165)
    parser.add_argument("--test_ratio", type=float, default=0.165)
    parser.add_argument(
        "--protocol",
        choices=(
            "category_covered_source_disjoint",
            "leave_one_theme_out",
        ),
        default="category_covered_source_disjoint",
    )
    parser.add_argument(
        "--heldout_theme",
        default=None,
        help="Required only for --protocol leave_one_theme_out",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
    )
    parser.add_argument(
        "--allow_unknown_performer_group",
        action="store_true",
        help=(
            "Allow public datasets without trustworthy gender metadata. "
            "Unknown remains an explicit stratum and is never imputed."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cache_root = Path(args.cache_root).resolve()
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    motions = [
        path
        for path in sorted(cache_root.rglob("*.npy"))
        if not any(
            token in path.name.lower()
            for token in (
                "motion_ref",
                "transition_mask",
                "single_test",
                "spin_interval",
                "jitter",
            )
        )
    ]
    if not motions:
        raise RuntimeError("No retarget-cache motions in %s" % cache_root)

    records = [source_record(cache_root, path) for path in motions]
    # Source-aware official-SMPL preprocessing may hard-cut one source
    # into multiple clean segment files. source_uid therefore identifies
    # one original recording track, not one cache artifact.
    uid_counts = Counter(row["source_uid"] for row in records)
    segments_per_source = {
        str(uid): int(count) for uid, count in sorted(uid_counts.items())
    }
    unknown = [
        row["source_uid"]
        for row in records
        if row["performer_group"] == "unknown"
    ]
    if unknown and not args.allow_unknown_performer_group:
        raise RuntimeError(
            "Unknown performer_group for sources: %s. Pass "
            "--allow_unknown_performer_group for datasets such as AIST++ "
            "whose released motion metadata does not declare gender." % unknown
        )

    recording_units = recording_group_records(records)
    if args.protocol == "leave_one_theme_out":
        if not args.heldout_theme:
            parser.error("--heldout_theme is required for leave_one_theme_out")
        assignment = leave_one_theme_out_assignment(
            recording_units,
            heldout_theme=str(args.heldout_theme),
            seed=int(args.seed),
        )
        target = {
            split: sum(value == split for value in assignment.values())
            for split in SPLITS
        }
    else:
        target = exact_split_counts(
            len(recording_units),
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        assignment = assign_records_category_covered(
            recording_units, target, args.seed
        )
    capacities = performer_capacities(recording_units, target)

    split_records: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    materialization = Counter()
    for record in records:
        split = assignment[record["recording_uid"]]
        motion_target = out_root / split / record["relative_motion"]
        report_target = out_root / split / record["relative_report"]
        materialization[
            materialize(
                Path(record["motion"]), motion_target, args.mode
            )
        ] += 1
        materialization[
            materialize(
                Path(record["report"]), report_target, args.mode
            )
        ] += 1
        row = dict(record)
        row.update({
            "split": split,
            "split_motion": str(motion_target),
            "split_report": str(report_target),
        })
        split_records[split].append(row)

    source_sets = {
        split: {row["source_uid"] for row in rows}
        for split, rows in split_records.items()
    }
    recording_sets = {
        split: {row["recording_uid"] for row in rows}
        for split, rows in split_records.items()
    }
    dancer_identity_available = all(
        bool(row.get("dancer_id"))
        and str(row.get("dancer_id_status")) == "verified"
        for row in records
    )
    dancer_sets = {
        split: {
            str(row["dancer_id"])
            for row in rows
            if row.get("dancer_id")
            and str(row.get("dancer_id_status")) == "verified"
        }
        for split, rows in split_records.items()
    }
    overlap = {
        "train_val": sorted(source_sets["train"] & source_sets["val"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "val_test": sorted(source_sets["val"] & source_sets["test"]),
        "recording_train_val": sorted(
            recording_sets["train"] & recording_sets["val"]
        ),
        "recording_train_test": sorted(
            recording_sets["train"] & recording_sets["test"]
        ),
        "recording_val_test": sorted(
            recording_sets["val"] & recording_sets["test"]
        ),
        "dancer_train_val": sorted(dancer_sets["train"] & dancer_sets["val"]),
        "dancer_train_test": sorted(dancer_sets["train"] & dancer_sets["test"]),
        "dancer_val_test": sorted(dancer_sets["val"] & dancer_sets["test"]),
    }
    reasons: List[str] = []
    if any(overlap[key] for key in ("train_val", "train_test", "val_test")):
        reasons.append("source_overlap")
    if any(
        overlap[key]
        for key in (
            "recording_train_val",
            "recording_train_test",
            "recording_val_test",
        )
    ):
        reasons.append("recording_overlap")
    for split in SPLITS:
        if len(recording_sets[split]) != target[split]:
            reasons.append("count_mismatch_%s" % split)
        if not split_records[split]:
            reasons.append("empty_%s" % split)
    confirmed_categories = {
        row["dance_key"]
        for row in recording_units
        if row.get("theme_label_status") == "confirmed"
        and row["dance_key"] != "unknown"
    }
    train_confirmed_categories = {
        row["dance_key"]
        for row in split_records["train"]
        if row.get("theme_label_status") == "confirmed"
        and row["dance_key"] != "unknown"
    }
    if args.protocol == "category_covered_source_disjoint" and not confirmed_categories.issubset(
        train_confirmed_categories
    ):
        reasons.append("confirmed_theme_missing_from_train")
    for group in ("female", "male"):
        count = sum(
            row["performer_group"] == group for row in recording_units
        )
        if count >= 3:
            for split in ("val", "test"):
                # Exact cardinality can make both performer groups impossible
                # in a one-recording split.  Require coverage only when the
                # optimized capacity allocated this performer to that split.
                if capacities[group][split] > 0 and not any(
                    row["performer_group"] == group
                    for row in split_records[split]
                ):
                    reasons.append(
                        "missing_%s_in_%s" % (group, split)
                    )

    report = {
        "schema": SCHEMA,
        "ok": not reasons,
        "reasons": reasons,
        "cache_root": str(cache_root),
        "out_root": str(out_root),
        "seed": int(args.seed),
        "split_protocol": args.protocol,
        "heldout_theme": args.heldout_theme,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "target_counts": target,
        "gender_group_capacities": capacities,
        "assignment_unit": "recording_uid_before_event_slicing",
        "assignment_algorithm": (
            "exact_recording_disjoint_confirmed_theme_covered_exhaustive"
            if args.protocol == "category_covered_source_disjoint"
            else "leave_one_confirmed_theme_out_zero_shot"
        ),
        "materialization_requested": args.mode,
        "materialization_actual": dict(materialization),
        "num_sources": len(uid_counts),
        "num_segments": len(records),
        "segments_per_source": segments_per_source,
        "num_recording_groups": len(recording_units),
        "recording_groups": recording_units,
        "dancer_identity_available": dancer_identity_available,
        "performer_disjoint_claim": bool(
            dancer_identity_available
            and not any(overlap[key] for key in (
                "dancer_train_val", "dancer_train_test", "dancer_val_test"
            ))
        ),
        "performer_disjoint_unavailable_reason": (
            None
            if dancer_identity_available
            else "global dancer_id is not verified in the released filenames/metadata"
        ),
        "confirmed_theme_coverage": {
            "all_confirmed_themes": sorted(confirmed_categories),
            "train_confirmed_themes": sorted(train_confirmed_categories),
            "all_confirmed_themes_in_train": confirmed_categories.issubset(
                train_confirmed_categories
            ),
        },
        "theme_evaluation_contract": {
            "standard_metrics_include_status": ["confirmed"],
            "pending_theme_labels_excluded": True,
            "pending_source_uids": sorted(
                {
                    row["source_uid"]
                    for row in records
                    if row.get("theme_label_status")
                    == "pending_official_confirmation"
                }
            ),
            "leave_one_theme_out_reported_separately": True,
            "all_data_training_allowed_for_qualitative_generation_only": True,
        },
        "unknown_performer_group_allowed": bool(
            args.allow_unknown_performer_group
        ),
        "unknown_performer_group_sources": sorted(unknown),
        "splits": {
            split: {
                "sources": len({row["source_uid"] for row in rows}),
                "segments": len(rows),
                "source_uids": sorted(
                    {row["source_uid"] for row in rows}
                ),
                "recording_uids": sorted(
                    {row["recording_uid"] for row in rows}
                ),
                "performer_group_histogram": dict(
                    Counter(
                        row["performer_group"] for row in rows
                    )
                ),
                "dance_key_histogram": dict(
                    Counter(row["dance_key"] for row in rows)
                ),
                "recording_group_dance_key_histogram": dict(
                    Counter(
                        row["dance_key"]
                        for row in recording_units
                        if assignment[row["source_uid"]] == split
                    )
                ),
                "records": rows,
            }
            for split, rows in split_records.items()
        },
        "overlap": overlap,
        "policy": {
            "split_before_event_slicing": True,
            "synchronized_performer_tracks_are_indivisible": True,
            "gender_group_is_not_dancer_identity": True,
            "validation_and_test_gender_coverage_is_optimized_when_feasible": True,
            "confirmed_theme_coverage_required_in_train": (
                args.protocol == "category_covered_source_disjoint"
            ),
            "unique_theme_heldout_only_in_leave_one_theme_out": True,
            "training_retrieval_uses_train_motion_only": True,
        },
    }
    manifest = out_root / "source_split_manifest.json"
    save_json(report, manifest)
    print(json.dumps({
        "manifest": str(manifest),
        "ok": report["ok"],
        "reasons": reasons,
        "target_counts": target,
        "gender_group_capacities": capacities,
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
