#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit weak MSSD-AESD semantic alignment without claiming paired accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from events.semantic_descriptor import MUSIC_SEMANTIC_LABELS
from grounding.semantic_optimal_transport import SCHEMA as OT_DATASET_SCHEMA


def _normalized_rows(value: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(value, dtype=np.float64), 1.0e-8)
    return array / array.sum(axis=-1, keepdims=True)


def _histogram(values: np.ndarray) -> Dict[str, int]:
    rows = np.asarray(values, dtype=object).reshape(-1)
    return {label: int(np.sum(rows == label)) for label in MUSIC_SEMANTIC_LABELS}


def _topk_group_coverage(
    phrase_ids: np.ndarray,
    group_ids: np.ndarray,
    candidate_rank: np.ndarray,
    top_k: int,
) -> float:
    coverages = []
    for phrase in np.unique(phrase_ids):
        mask = phrase_ids == phrase
        order = np.argsort(candidate_rank[mask], kind="stable")[: int(top_k)]
        groups = group_ids[mask][order]
        if len(groups):
            coverages.append(float(len(np.unique(groups))) / len(groups))
    return float(np.mean(coverages)) if coverages else 0.0


def audit_alignment(
    dataset_path: Path,
    *,
    aesd_path: Optional[Path] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    with np.load(dataset_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    schema = str(np.asarray(payload.get("schema", "")).reshape(-1)[0])
    if schema != OT_DATASET_SCHEMA:
        raise RuntimeError(f"Expected {OT_DATASET_SCHEMA}, got {schema!r}")
    if np.any(np.asarray(payload.get("is_ground_truth_pair", []), dtype=bool)):
        raise RuntimeError("Audit refuses a semantic-OT dataset claiming ground truth")

    phrase_ids = np.asarray(payload["phrase_ids"], dtype=np.int64)
    music = _normalized_rows(payload["teacher_music_probs"])
    action = _normalized_rows(payload["teacher_action_probs"])
    pair_weight = np.asarray(payload["teacher_pair_weight"], dtype=np.float64)
    js = np.asarray(payload["teacher_js_divergence"], dtype=np.float64)
    rank = np.asarray(payload["candidate_rank"], dtype=np.int64)
    source_ids = np.asarray(payload["source_ids"], dtype=np.int64)
    family_ids = np.asarray(payload["family_ids"], dtype=np.int64)

    weighted_js = float(np.sum(js * pair_weight) / max(float(pair_weight.sum()), 1.0e-8))
    music_mass = music.mean(axis=0)
    action_mass = np.sum(action * pair_weight[:, None], axis=0) / max(
        float(pair_weight.sum()), 1.0e-8
    )
    top_music = np.asarray(
        [MUSIC_SEMANTIC_LABELS[int(index)] for index in np.argmax(music, axis=1)],
        dtype=object,
    )
    top_action = np.asarray(
        [MUSIC_SEMANTIC_LABELS[int(index)] for index in np.argmax(action, axis=1)],
        dtype=object,
    )
    report: Dict[str, Any] = {
        "schema": "dunhuang_semantic_alignment_audit_v1",
        "dataset": str(dataset_path.resolve()),
        "supervision": "semantic_optimal_transport",
        "is_ground_truth_pair": False,
        "num_rows": int(len(phrase_ids)),
        "num_phrases": int(len(np.unique(phrase_ids))),
        "weighted_mssd_aesd_js_divergence": weighted_js,
        "music_soft_class_mass": {
            label: float(music_mass[index])
            for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
        },
        "transported_action_soft_class_mass": {
            label: float(action_mass[index])
            for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
        },
        "music_top_label_histogram": _histogram(top_music),
        "action_top_label_histogram": _histogram(top_action),
        "topk_source_coverage": _topk_group_coverage(
            phrase_ids, source_ids, rank, int(top_k)
        ),
        "topk_family_coverage": _topk_group_coverage(
            phrase_ids, family_ids, rank, int(top_k)
        ),
        "mean_teacher_entropy": float(np.mean(payload["teacher_entropy"])),
        "mean_teacher_margin": float(np.mean(payload["teacher_margin"])),
        "metric_contract": {
            "retrieval_accuracy_against_human_pairs": False,
            "teacher_agreement_only": True,
            "report_as_weak_supervision": True,
        },
    }
    if aesd_path is not None:
        with np.load(aesd_path, allow_pickle=True) as db:
            report["aesd"] = {
                "path": str(aesd_path.resolve()),
                "num_events": int(len(db["event_uids"])),
                "top_label_histogram": _histogram(db["aesd_event_semantics"]),
                "ambiguous_ratio": float(
                    np.mean(np.asarray(db["aesd_semantic_ambiguous"], dtype=bool))
                ),
                "mean_entropy": float(
                    np.mean(np.asarray(db["aesd_semantic_entropy"], dtype=np.float64))
                ),
                "intrinsic_high_ratio": float(
                    np.mean(
                        np.asarray(
                            db["aesd_intrinsic_transition_profile"], dtype=object
                        )
                        == "high"
                    )
                ),
            }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--aesd", default="")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = audit_alignment(
        Path(args.dataset),
        aesd_path=Path(args.aesd) if args.aesd else None,
        top_k=int(args.top_k),
    )
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
