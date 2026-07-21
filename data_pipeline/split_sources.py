#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performer-aware, exact-cardinality, source-disjoint cache split.

Priority order for the 12-source low-resource setting:
1. no source leakage;
2. non-empty train/validation/test;
3. female and male coverage in validation and test when feasible;
4. dance-category balance within each performer group.

For 4 female + 8 male sources and an 8/2/2 split, the optimal allocation is
train=2F+6M, validation=1F+1M, test=1F+1M.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
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
SCHEMA = "performer_aware_source_disjoint_cache_split"


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


def report_path_for_motion(path: Path) -> Path:
    return path.with_suffix(".retarget.json")


def source_record(cache_root: Path, motion_path: Path) -> Dict[str, Any]:
    report_path = report_path_for_motion(motion_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("ok", False)):
        raise RuntimeError("Retarget report is not OK: %s" % motion_path)
    if not bool(
        report.get(
            "source_gate_ok",
            report.get("anatomy_ok", False),
        )
    ):
        raise RuntimeError("Source-safety gate failed: %s" % motion_path)

    relative_motion = motion_path.relative_to(cache_root)
    original = str(
        report.get("source_used")
        or report.get("source")
        or report.get("source_relative")
        or relative_motion.with_suffix(".bvh")
    )
    semantic = motion_api.parse_change_bvh_semantics(original)
    source_uid = str(
        semantic.get("source_uid") or Path(original).stem
    )
    dance_key = str(
        semantic.get("dance_key")
        or semantic.get("dance_category")
        or "unknown"
    )
    performer = infer_performer_group(semantic, original)
    return {
        "motion": str(motion_path.resolve()),
        "report": str(report_path.resolve()),
        "relative_motion": str(relative_motion),
        "relative_report": str(report_path.relative_to(cache_root)),
        "original_source": original,
        "source_uid": source_uid,
        "dance_key": dance_key,
        "performer_group": performer,
        "source_anatomy_quality": float(
            report.get("anatomy", {}).get("anatomy_quality", 0.0)
        ),
        "source_gate_reasons": list(
            report.get("source_gate_reasons", [])
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
    uid_counts = Counter(row["source_uid"] for row in records)
    duplicates = sorted(
        uid for uid, count in uid_counts.items() if count > 1
    )
    if duplicates:
        raise RuntimeError(
            "source_uid must identify one complete source: %s" % duplicates
        )
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

    target = exact_split_counts(
        len(records),
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    capacities = performer_capacities(records, target)
    assignment = assign_records(records, target, args.seed)

    split_records: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    materialization = Counter()
    for record in records:
        split = assignment[record["source_uid"]]
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
    overlap = {
        "train_val": sorted(source_sets["train"] & source_sets["val"]),
        "train_test": sorted(source_sets["train"] & source_sets["test"]),
        "val_test": sorted(source_sets["val"] & source_sets["test"]),
    }
    reasons: List[str] = []
    if any(overlap.values()):
        reasons.append("source_overlap")
    for split in SPLITS:
        if len(split_records[split]) != target[split]:
            reasons.append("count_mismatch_%s" % split)
        if not split_records[split]:
            reasons.append("empty_%s" % split)
    for group in ("female", "male"):
        count = sum(
            row["performer_group"] == group for row in records
        )
        if count >= 3:
            for split in ("val", "test"):
                if not any(
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
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "target_counts": target,
        "performer_capacities": capacities,
        "assignment_unit": "source_uid_before_event_slicing",
        "assignment_algorithm": (
            "exact_global_capacity_performer_stratified_"
            "dance_category_aware_deterministic"
        ),
        "materialization_requested": args.mode,
        "materialization_actual": dict(materialization),
        "num_sources": len(records),
        "unknown_performer_group_allowed": bool(
            args.allow_unknown_performer_group
        ),
        "unknown_performer_group_sources": sorted(unknown),
        "splits": {
            split: {
                "sources": len(rows),
                "source_uids": sorted(
                    row["source_uid"] for row in rows
                ),
                "performer_group_histogram": dict(
                    Counter(
                        row["performer_group"] for row in rows
                    )
                ),
                "dance_key_histogram": dict(
                    Counter(row["dance_key"] for row in rows)
                ),
                "records": rows,
            }
            for split, rows in split_records.items()
        },
        "overlap": overlap,
        "policy": {
            "split_before_event_slicing": True,
            "validation_and_test_cover_both_known_performer_groups": True,
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
        "performer_capacities": capacities,
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
