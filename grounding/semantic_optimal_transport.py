#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse semantic optimal transport for unpaired music-to-motion grounding.

This module constructs an auditable weak-supervision dataset.  It never labels
an automatically selected music-motion relation as a ground-truth pair.  Music
phrases are represented by MSSD probabilities and real CLAP/temporal features;
motion events are represented by calibrated AESD probabilities and intrinsic
geometry.  A sparse entropic optimal-transport plan supplies multi-positive
candidate weights.

The input music and motion splits must already be disjoint.  Run this module
separately for train, validation and test partitions.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from events.semantic_descriptor import (
    MUSIC_SEMANTIC_LABELS,
    parse_descriptor_file,
)
from grounding.paired_data import (
    CONTROL_NAMES,
    DEFAULT_TEMPORAL_FRAMES,
    _load_npz_mapping,
    _row_audio_features,
    validate_paired_payload,
)
from support.event_identity import (
    event_uids_from_generation_db,
    make_event_db_contract,
)


SCHEMA = "dunhuang_semantic_optimal_transport_grounding_v1"
SUPERVISION = "semantic_ot_teacher"


@dataclass(frozen=True)
class TransportWeights:
    semantic: float = 0.55
    duration: float = 0.15
    quality: float = 0.10
    confidence: float = 0.10
    intrinsic_risk: float = 0.05
    source_balance: float = 0.05

    def validate(self) -> None:
        values = np.asarray(
            [
                self.semantic,
                self.duration,
                self.quality,
                self.confidence,
                self.intrinsic_risk,
                self.source_balance,
            ],
            dtype=np.float64,
        )
        if np.any(values < 0.0):
            raise ValueError("Transport weights must be non-negative")
        if not np.isclose(values.sum(), 1.0, atol=1.0e-6):
            raise ValueError(
                "Transport weights must sum to one; got "
                f"{float(values.sum()):.8f}"
            )


@dataclass(frozen=True)
class PhraseRecord:
    phrase_token: str
    song_token: str
    audio_path: Path
    start_sec: float
    end_sec: float
    duration_sec: float
    probabilities: np.ndarray
    entropy: float
    margin: float
    confidence: float
    mssd_path: Path


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    array = np.maximum(array, 1.0e-8)
    return (array / array.sum(axis=-1, keepdims=True)).astype(np.float32)


def _entropy(probabilities: np.ndarray) -> float:
    p = _normalize_rows(np.asarray(probabilities).reshape(1, -1))[0]
    return float(-(p * np.log(p + 1.0e-8)).sum() / math.log(len(p)))


def _margin(probabilities: np.ndarray) -> float:
    p = np.sort(_normalize_rows(np.asarray(probabilities).reshape(1, -1))[0])
    return float(p[-1] - p[-2]) if len(p) > 1 else 1.0


def jensen_shannon_divergence(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Return pairwise Jensen-Shannon divergence in [0, log(2)]."""
    p = _normalize_rows(left).astype(np.float64)
    q = _normalize_rows(right).astype(np.float64)
    p = p[:, None, :]
    q = q[None, :, :]
    midpoint = 0.5 * (p + q)
    left_kl = np.sum(p * (np.log(p + 1.0e-8) - np.log(midpoint + 1.0e-8)), axis=-1)
    right_kl = np.sum(q * (np.log(q + 1.0e-8) - np.log(midpoint + 1.0e-8)), axis=-1)
    return (0.5 * (left_kl + right_kl)).astype(np.float32)


def _read_mssd_files(
    paths: Sequence[Path], fps: float
) -> List[PhraseRecord]:
    records: List[PhraseRecord] = []
    for mssd_path in paths:
        slots, _features, meta = parse_descriptor_file(
            mssd_path,
            require_final_schedule=False,
            fps=float(fps),
            usage="train_semantic",
        )
        raw = json.loads(mssd_path.read_text(encoding="utf-8"))
        audio_text = str(
            raw.get("audio", meta.get("audio", ""))
            if isinstance(raw, Mapping)
            else meta.get("audio", "")
        ).strip()
        if not audio_text:
            raise RuntimeError(f"MSSD has no audio path: {mssd_path}")
        audio_path = Path(audio_text).expanduser()
        if not audio_path.is_absolute():
            audio_path = (mssd_path.parent / audio_path).resolve()
        else:
            audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(str(audio_path))
        song_token = str(
            raw.get("song_uid", raw.get("song_id", audio_path.stem))
            if isinstance(raw, Mapping)
            else audio_path.stem
        )
        for index, slot in enumerate(slots):
            start = float(slot["start_sec"])
            end = float(slot["end_sec"])
            duration = max(end - start, 0.10)
            probabilities_object = slot.get("music_semantic_probs", {})
            probabilities = np.asarray(
                [
                    float(probabilities_object.get(label, 0.0))
                    for label in MUSIC_SEMANTIC_LABELS
                ],
                dtype=np.float32,
            )
            probabilities = _normalize_rows(probabilities[None])[0]
            confidence = float(
                slot.get(
                    "teacher_confidence",
                    max(0.05, 1.0 - _entropy(probabilities)),
                )
            )
            records.append(
                PhraseRecord(
                    phrase_token=f"{song_token}::phrase_{index:05d}",
                    song_token=song_token,
                    audio_path=audio_path,
                    start_sec=start,
                    end_sec=end,
                    duration_sec=duration,
                    probabilities=probabilities,
                    entropy=float(slot.get("teacher_entropy", _entropy(probabilities))),
                    margin=float(slot.get("teacher_margin", _margin(probabilities))),
                    confidence=float(np.clip(confidence, 0.05, 1.0)),
                    mssd_path=mssd_path.resolve(),
                )
            )
    if not records:
        raise RuntimeError("No usable MSSD phrases were found")
    return records


def discover_mssd_files(
    mssd_dirs: Sequence[Path], manifest_path: Optional[Path] = None
) -> List[Path]:
    files: List[Path] = []
    if manifest_path is not None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("songs", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            raise RuntimeError("Music manifest must be a list or contain a songs list")
        for row in rows:
            if not isinstance(row, Mapping) or not row.get("mssd_path"):
                raise RuntimeError("Every music-manifest row requires mssd_path")
            path = Path(str(row["mssd_path"])).expanduser()
            if not path.is_absolute():
                path = manifest_path.parent / path
            files.append(path.resolve())
    for directory in mssd_dirs:
        root = directory.expanduser().resolve()
        files.extend(sorted(root.rglob("*.mssd.json")))
        files.extend(sorted(root.rglob("*_mssd.json")))
    unique: List[Path] = []
    seen = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MSSD files:\n" + "\n".join(missing))
    if not unique:
        raise RuntimeError("No MSSD files discovered")
    return unique


def _source_balanced_candidates(
    cost: np.ndarray,
    source_ids: np.ndarray,
    preselect_k: int,
    per_source: int,
) -> np.ndarray:
    values = np.asarray(cost, dtype=np.float32).reshape(-1)
    sources = np.asarray(source_ids, dtype=np.int64).reshape(-1)
    chosen: List[int] = []
    for source in np.unique(sources):
        indices = np.flatnonzero(sources == source)
        order = indices[np.argsort(values[indices], kind="stable")]
        chosen.extend(map(int, order[: min(per_source, len(order))]))
    global_order = np.argsort(values, kind="stable")
    chosen.extend(map(int, global_order[: min(preselect_k, len(global_order))]))
    unique = list(dict.fromkeys(chosen))
    unique.sort(key=lambda index: float(values[index]))
    return np.asarray(unique[: min(preselect_k, len(unique))], dtype=np.int64)


def sparse_sinkhorn(
    candidate_indices: np.ndarray,
    candidate_costs: np.ndarray,
    event_marginal: np.ndarray,
    *,
    epsilon: float = 0.08,
    iterations: int = 200,
    tolerance: float = 1.0e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Balance a sparse phrase-event graph with entropic Sinkhorn updates."""
    indices = np.asarray(candidate_indices, dtype=np.int64)
    costs = np.asarray(candidate_costs, dtype=np.float64)
    if indices.shape != costs.shape or indices.ndim != 2:
        raise ValueError("candidate indices/costs must be aligned [P,K] matrices")
    phrase_count, candidate_count = indices.shape
    event_count = int(len(event_marginal))
    if phrase_count < 1 or candidate_count < 1:
        raise ValueError("Sparse OT graph must be non-empty")
    epsilon = max(float(epsilon), 1.0e-4)
    kernel = np.exp(-np.clip(costs, 0.0, 100.0) / epsilon)
    kernel = np.maximum(kernel, 1.0e-30)
    active = np.unique(indices)
    target = np.zeros(event_count, dtype=np.float64)
    target[active] = np.asarray(event_marginal, dtype=np.float64)[active]
    target_sum = float(target.sum())
    if target_sum <= 0.0:
        target[active] = 1.0 / len(active)
    else:
        target /= target_sum
    row_target = np.full(phrase_count, 1.0 / phrase_count, dtype=np.float64)
    u = np.ones(phrase_count, dtype=np.float64)
    v = np.ones(event_count, dtype=np.float64)
    residual = float("inf")
    for iteration in range(max(1, int(iterations))):
        row_sum = np.sum(kernel * v[indices], axis=1)
        u = row_target / np.maximum(row_sum, 1.0e-30)
        column_sum = np.zeros(event_count, dtype=np.float64)
        np.add.at(column_sum, indices.reshape(-1), (u[:, None] * kernel).reshape(-1))
        update_mask = target > 0.0
        v[update_mask] = target[update_mask] / np.maximum(
            column_sum[update_mask], 1.0e-30
        )
        if iteration % 10 == 0 or iteration + 1 == int(iterations):
            plan = u[:, None] * kernel * v[indices]
            plan_rows = plan.sum(axis=1)
            plan_columns = np.zeros(event_count, dtype=np.float64)
            np.add.at(plan_columns, indices.reshape(-1), plan.reshape(-1))
            residual = max(
                float(np.max(np.abs(plan_rows - row_target))),
                float(np.max(np.abs(plan_columns - target))),
            )
            if residual <= float(tolerance):
                break
    plan = u[:, None] * kernel * v[indices]
    row_normalized = plan / np.maximum(plan.sum(axis=1, keepdims=True), 1.0e-30)
    if not np.isfinite(row_normalized).all():
        raise RuntimeError("Sparse Sinkhorn produced NaN or Inf")
    return row_normalized.astype(np.float32), {
        "iterations": int(iteration + 1),
        "residual": float(residual),
        "converged": bool(residual <= float(tolerance)),
        "tolerance": float(tolerance),
        "active_events": int(len(active)),
    }


def _event_arrays(db: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    required = (
        "v46_53_geometry_desc",
        "v46_53_bodypart_flow",
        "v46_53_bodypart_gaussian_mean",
        "v46_53_bodypart_gaussian_covariance",
        "aesd_music_alignment_probs",
    )
    missing = [key for key in required if key not in db]
    if missing:
        raise RuntimeError(
            "Event-DB lacks semantic-OT fields; rebuild intrinsic geometry and "
            f"AESD first. missing={missing}"
        )
    event_uids = event_uids_from_generation_db(dict(db))
    count = len(event_uids)
    durations = np.asarray(
        db.get("durations", np.full(count, 2.0, dtype=np.float32)),
        dtype=np.float32,
    )
    quality = np.clip(
        np.asarray(
            db.get(
                "v46_53_combined_quality",
                db.get("event_quality_scores", np.full(count, 0.5)),
            ),
            dtype=np.float32,
        ),
        1.0e-3,
        1.0,
    )
    confidence = np.clip(
        np.asarray(db.get("semantic_confidence", np.full(count, 0.5)), dtype=np.float32),
        0.0,
        1.0,
    )
    risk = np.clip(
        np.asarray(
            db.get(
                "aesd_intrinsic_transition_prior",
                db.get("aesd_boundary_risk", np.zeros(count)),
            ),
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )
    position = np.clip(
        np.asarray(db.get("event_position_mid", np.full(count, 0.5)), dtype=np.float32),
        0.0,
        1.0,
    )
    source_values = np.asarray(db.get("source_uids", ["unknown"] * count), dtype=object)
    family_values = np.asarray(db.get("event_families", ["unknown"] * count), dtype=object)
    source_vocab = {name: index for index, name in enumerate(sorted({str(v) for v in source_values}))}
    family_vocab = {name: index for index, name in enumerate(sorted({str(v) for v in family_values}))}
    source_ids = np.asarray([source_vocab[str(v)] for v in source_values], dtype=np.int64)
    family_ids = np.asarray([family_vocab[str(v)] for v in family_values], dtype=np.int64)
    source_counts = np.bincount(source_ids, minlength=len(source_vocab)).astype(np.float64)
    source_penalty = source_counts[source_ids] / max(float(source_counts.max()), 1.0)
    controls = np.stack(
        [
            np.clip(durations / 6.0, 0.0, 2.0),
            quality,
            confidence,
            position,
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "event_uids": np.asarray(event_uids, dtype=object),
        "probabilities": _normalize_rows(np.asarray(db["aesd_music_alignment_probs"])),
        "durations": durations,
        "quality": quality,
        "confidence": confidence,
        "risk": risk,
        "controls": controls,
        "source_ids": source_ids,
        "family_ids": family_ids,
        "source_penalty": source_penalty.astype(np.float32),
        "geometry": np.asarray(db["v46_53_geometry_desc"], dtype=np.float32),
        "bodypart": np.asarray(db["v46_53_bodypart_flow"], dtype=np.float32)[:, :5],
        "gaussian_mean": np.asarray(db["v46_53_bodypart_gaussian_mean"], dtype=np.float32),
        "gaussian_covariance": np.asarray(db["v46_53_bodypart_gaussian_covariance"], dtype=np.float32),
        "source_values": source_values,
        "family_values": family_values,
        "source_vocab": np.asarray(sorted(source_vocab), dtype=object),
        "family_vocab": np.asarray(sorted(family_vocab), dtype=object),
    }


def build_semantic_ot_dataset(
    event_db_path: Path,
    mssd_files: Sequence[Path],
    out_path: Path,
    *,
    model_name: str = "clap",
    cache_dir: Optional[Path] = None,
    temporal_frames: int = DEFAULT_TEMPORAL_FRAMES,
    temporal_source_frames: int = 2048,
    phrase_fps: float = 30.0,
    top_k: int = 8,
    preselect_k: int = 64,
    preselect_per_source: int = 8,
    sinkhorn_epsilon: float = 0.08,
    sinkhorn_iterations: int = 200,
    weights: TransportWeights = TransportWeights(),
) -> Dict[str, Any]:
    weights.validate()
    if int(top_k) < 2:
        raise ValueError("top_k must be at least two for multi-positive supervision")
    if int(preselect_k) < int(top_k):
        raise ValueError("preselect_k must be no smaller than top_k")
    db = _load_npz_mapping(event_db_path)
    events = _event_arrays(db)
    event_count = int(len(events["event_uids"]))
    if event_count < int(top_k):
        raise RuntimeError(
            f"Event-DB has {event_count} events, fewer than top_k={int(top_k)}"
        )
    phrases = _read_mssd_files(mssd_files, fps=float(phrase_fps))
    phrase_probabilities = np.stack([row.probabilities for row in phrases])
    event_probabilities = events["probabilities"]
    semantic_cost = jensen_shannon_divergence(
        phrase_probabilities, event_probabilities
    ) / math.log(2.0)
    phrase_durations = np.asarray([row.duration_sec for row in phrases], dtype=np.float32)
    event_durations = np.asarray(events["durations"], dtype=np.float32)
    duration_cost = np.abs(
        np.log(
            np.maximum(phrase_durations[:, None], 1.0e-3)
            / np.maximum(event_durations[None, :], 1.0e-3)
        )
    )
    duration_cost = np.clip(duration_cost / 1.5, 0.0, 1.0).astype(np.float32)
    cost_matrix = (
        weights.semantic * semantic_cost
        + weights.duration * duration_cost
        + weights.quality * (1.0 - events["quality"])[None, :]
        + weights.confidence * (1.0 - events["confidence"])[None, :]
        + weights.intrinsic_risk * events["risk"][None, :]
        + weights.source_balance * events["source_penalty"][None, :]
    ).astype(np.float32)

    candidate_indices = np.stack(
        [
            _source_balanced_candidates(
                cost_matrix[index],
                events["source_ids"],
                int(preselect_k),
                int(preselect_per_source),
            )
            for index in range(len(phrases))
        ],
        axis=0,
    )
    candidate_costs = np.take_along_axis(cost_matrix, candidate_indices, axis=1)
    event_marginal = (
        np.asarray(events["quality"], dtype=np.float64)
        * np.asarray(events["confidence"], dtype=np.float64)
        / np.maximum(
            np.bincount(events["source_ids"])[events["source_ids"]], 1
        )
    )
    event_marginal = np.maximum(event_marginal, 1.0e-8)
    event_marginal /= event_marginal.sum()
    transport_weights, sinkhorn_report = sparse_sinkhorn(
        candidate_indices,
        candidate_costs,
        event_marginal,
        epsilon=float(sinkhorn_epsilon),
        iterations=int(sinkhorn_iterations),
    )

    top_order = np.argsort(-transport_weights, axis=1, kind="stable")[:, : int(top_k)]
    top_event_indices = np.take_along_axis(candidate_indices, top_order, axis=1)
    top_weights = np.take_along_axis(transport_weights, top_order, axis=1)
    top_weights = top_weights / np.maximum(top_weights.sum(axis=1, keepdims=True), 1.0e-8)

    clap_rows: List[np.ndarray] = []
    temporal_rows: List[np.ndarray] = []
    event_rows: List[int] = []
    phrase_ids: List[int] = []
    song_ids: List[int] = []
    pair_weights: List[float] = []
    music_prob_rows: List[np.ndarray] = []
    action_prob_rows: List[np.ndarray] = []
    js_rows: List[float] = []
    teacher_entropy: List[float] = []
    teacher_margin: List[float] = []
    candidate_rank: List[int] = []
    phrase_tokens = [row.phrase_token for row in phrases]
    song_tokens = sorted({row.song_token for row in phrases})
    song_vocab = {token: index for index, token in enumerate(song_tokens)}
    audio_reports: List[Dict[str, Any]] = []

    for phrase_index, phrase in enumerate(phrases):
        feature_row = {
            "audio_path": str(phrase.audio_path),
            "start_sec": phrase.start_sec,
            "end_sec": phrase.end_sec,
            "music_event": MUSIC_SEMANTIC_LABELS[int(np.argmax(phrase.probabilities))],
        }
        clap, temporal, audio_report = _row_audio_features(
            feature_row,
            manifest_dir=phrase.mssd_path.parent,
            model_name=model_name,
            cache_dir=cache_dir,
            temporal_frames=int(temporal_frames),
            temporal_source_frames=int(temporal_source_frames),
            phrase_fps=float(phrase_fps),
        )
        audio_reports.append(audio_report)
        for rank, (event_index, pair_weight) in enumerate(
            zip(top_event_indices[phrase_index], top_weights[phrase_index])
        ):
            event_index = int(event_index)
            clap_rows.append(clap)
            temporal_rows.append(temporal)
            event_rows.append(event_index)
            phrase_ids.append(phrase_index)
            song_ids.append(song_vocab[phrase.song_token])
            pair_weights.append(float(pair_weight))
            music_prob_rows.append(phrase.probabilities)
            action_prob_rows.append(event_probabilities[event_index])
            js_rows.append(float(semantic_cost[phrase_index, event_index]))
            teacher_entropy.append(float(phrase.entropy))
            teacher_margin.append(float(phrase.margin))
            candidate_rank.append(rank)

    indices = np.asarray(event_rows, dtype=np.int64)
    payload: Dict[str, Any] = {
        "schema": np.asarray(SCHEMA, dtype=object),
        "clap": np.stack(clap_rows).astype(np.float32),
        "temporal": np.stack(temporal_rows).astype(np.float32),
        "motion_geometry": events["geometry"][indices],
        "bodypart_flow": events["bodypart"][indices],
        "gaussian_mean": events["gaussian_mean"][indices],
        "gaussian_covariance": events["gaussian_covariance"][indices],
        "controls": events["controls"][indices],
        "quality": events["quality"][indices].astype(np.float32),
        "pair_ids": np.asarray(phrase_ids, dtype=np.int64),
        "audio_group_ids": np.asarray(song_ids, dtype=np.int64),
        "family_ids": events["family_ids"][indices],
        "source_ids": events["source_ids"][indices],
        "event_indices": indices,
        "event_uids": events["event_uids"][indices],
        "phrase_ids": np.asarray(phrase_ids, dtype=np.int64),
        "song_ids": np.asarray(song_ids, dtype=np.int64),
        "song_tokens": np.asarray(
            [phrases[index].song_token for index in phrase_ids], dtype=object
        ),
        "phrase_tokens": np.asarray(
            [phrase_tokens[index] for index in phrase_ids], dtype=object
        ),
        "teacher_pair_weight": np.asarray(pair_weights, dtype=np.float32),
        "teacher_music_probs": np.stack(music_prob_rows).astype(np.float32),
        "teacher_action_probs": np.stack(action_prob_rows).astype(np.float32),
        "teacher_js_divergence": np.asarray(js_rows, dtype=np.float32),
        "teacher_entropy": np.asarray(teacher_entropy, dtype=np.float32),
        "teacher_margin": np.asarray(teacher_margin, dtype=np.float32),
        "candidate_rank": np.asarray(candidate_rank, dtype=np.int64),
        "supervision": np.asarray([SUPERVISION] * len(indices), dtype=object),
        "is_ground_truth_pair": np.zeros(len(indices), dtype=np.bool_),
        "event_db_contract_json": np.asarray(
            json.dumps(
                make_event_db_contract(events["event_uids"].tolist()),
                sort_keys=True,
            ),
            dtype=object,
        ),
    }
    dimensions = validate_paired_payload(payload)
    metadata = {
        "schema": SCHEMA,
        "scientific_pairing": "weak_supervision",
        "is_ground_truth_pair": False,
        "supervision": SUPERVISION,
        "event_db": str(event_db_path.resolve()),
        "mssd_files": [str(path.resolve()) for path in mssd_files],
        "num_songs": int(len(song_tokens)),
        "num_phrases": int(len(phrases)),
        "candidates_per_phrase": int(top_k),
        "rows": int(len(indices)),
        "dimensions": dimensions,
        "transport_weights": weights.__dict__,
        "sinkhorn": sinkhorn_report,
        "mean_teacher_js_divergence": float(np.mean(js_rows)),
        "mean_teacher_entropy": float(np.mean(teacher_entropy)),
        "mean_teacher_margin": float(np.mean(teacher_margin)),
        "motion_source_count": int(len(np.unique(events["source_ids"]))),
        "event_family_count": int(len(np.unique(events["family_ids"]))),
        "control_names": list(CONTROL_NAMES),
        "audio_extraction": audio_reports,
        "leakage_contract": {
            "build_after_music_split": True,
            "build_after_motion_source_split": True,
            "train_and_validation_datasets_are_separate": True,
        },
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False), dtype=object
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    report_path = out_path.with_suffix(out_path.suffix + ".json")
    report_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        **metadata,
        "dataset": str(out_path.resolve()),
        "report": str(report_path.resolve()),
        "ok": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build sparse MSSD-AESD semantic optimal-transport supervision"
    )
    parser.add_argument("--event_db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mssd_dirs", nargs="*", default=[])
    parser.add_argument("--music_manifest", default="")
    parser.add_argument("--model_name", default="clap")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--temporal_frames", type=int, default=64)
    parser.add_argument("--temporal_source_frames", type=int, default=2048)
    parser.add_argument("--phrase_fps", type=float, default=30.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--preselect_k", type=int, default=64)
    parser.add_argument("--preselect_per_source", type=int, default=8)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.08)
    parser.add_argument("--sinkhorn_iterations", type=int, default=200)
    args = parser.parse_args(argv)
    files = discover_mssd_files(
        [Path(value) for value in args.mssd_dirs],
        Path(args.music_manifest) if args.music_manifest else None,
    )
    report = build_semantic_ot_dataset(
        Path(args.event_db),
        files,
        Path(args.out),
        model_name=str(args.model_name),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        temporal_frames=int(args.temporal_frames),
        temporal_source_frames=int(args.temporal_source_frames),
        phrase_fps=float(args.phrase_fps),
        top_k=int(args.top_k),
        preselect_k=int(args.preselect_k),
        preselect_per_source=int(args.preselect_per_source),
        sinkhorn_epsilon=float(args.sinkhorn_epsilon),
        sinkhorn_iterations=int(args.sinkhorn_iterations),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
