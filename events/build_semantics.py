#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build calibrated Action Event Semantic Descriptor (AESD) metadata.

The builder enriches an existing Event-RAG ``events.npz`` without modifying
motion arrays.  It performs two passes:

1. construct grouped-evidence action-to-music semantic distributions;
2. optionally apply mild empirical-prior correction and write uncertainty
   diagnostics (top-2 label, entropy, margin and ambiguity flag).

The event-local boundary quantity is explicitly named an *intrinsic transition
prior*.  It is retained under the historical boundary-risk array names only as
an auditable compatibility alias; runtime repair decisions must use pairwise
transition risk from :mod:`contracts.boundary`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from events.semantic_descriptor import (  # noqa: E402
    AESD_SCHEMA_VERSION,
    MUSIC_SEMANTIC_LABELS,
    class_prior_adjustment,
    event_probs_from_fields,
    intrinsic_transition_prior_from_arrays,
    semantic_distribution_diagnostics,
    stage_affordance_from_probs,
    vector_to_prob_dict,
)
from support.event_identity import (  # noqa: E402
    EVENT_UID_SCHEMA,
    event_uids_from_generation_db,
    make_event_db_contract,
)


DEFAULT_PRIOR_ALPHA = 0.35
DEFAULT_AMBIGUITY_MARGIN = 0.08


def _arr(
    db: Mapping[str, Any],
    key: str,
    n: int,
    default: Any,
    dtype: Any = object,
) -> np.ndarray:
    if key in db:
        return np.asarray(db[key], dtype=dtype)
    return np.asarray([default] * n, dtype=dtype)


def _farr(
    db: Mapping[str, Any], key: str, n: int, default: float
) -> np.ndarray:
    if key in db:
        return np.asarray(db[key], dtype=np.float32)
    return np.full((n,), float(default), dtype=np.float32)


def _matrix(
    db: Mapping[str, Any], key: str, n: int, width: int
) -> np.ndarray:
    if key in db:
        value = np.asarray(db[key], dtype=np.float32)
        if value.ndim == 2 and value.shape[0] == n:
            return value
    return np.zeros((n, width), dtype=np.float32)


def _risk_numeric_features(
    entry: np.ndarray,
    exit_: np.ndarray,
    contact_entry: np.ndarray,
    contact_exit: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    try:
        e = np.asarray(entry, dtype=np.float32).reshape(-1)
        x = np.asarray(exit_, dtype=np.float32).reshape(-1)
        result["entry_velocity_norm"] = (
            float(np.linalg.norm(e[72:144]) / 72.0) if e.size >= 144 else 0.0
        )
        result["exit_velocity_norm"] = (
            float(np.linalg.norm(x[72:144]) / 72.0) if x.size >= 144 else 0.0
        )
        result["entry_exit_pose_gap_self"] = (
            float(np.mean((x[:72] - e[:72]) ** 2))
            if e.size >= 72 and x.size >= 72
            else 0.0
        )
    except Exception:
        result.update(
            {
                "entry_velocity_norm": 0.0,
                "exit_velocity_norm": 0.0,
                "entry_exit_pose_gap_self": 0.0,
            }
        )
    try:
        c0 = np.asarray(contact_entry, dtype=np.float32).reshape(-1)
        c1 = np.asarray(contact_exit, dtype=np.float32).reshape(-1)
        count = min(c0.size, c1.size)
        result["contact_jump_self"] = (
            float(np.mean(np.abs(c1[:count] - c0[:count]))) if count else 0.0
        )
    except Exception:
        result["contact_jump_self"] = 0.0
    return result


def _load_prior(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        if "class_prior" in payload:
            payload = payload["class_prior"]
        if isinstance(payload, Mapping):
            vector = np.asarray(
                [float(payload.get(label, 0.0)) for label in MUSIC_SEMANTIC_LABELS],
                dtype=np.float32,
            )
        else:
            vector = np.asarray(payload, dtype=np.float32).reshape(-1)
    else:
        vector = np.asarray(payload, dtype=np.float32).reshape(-1)
    if vector.size != len(MUSIC_SEMANTIC_LABELS):
        raise RuntimeError(
            "Class-prior JSON must contain one value for each AESD label"
        )
    vector = np.maximum(vector, 1.0e-4)
    return (vector / vector.sum()).astype(np.float32)


def _histogram(labels: Sequence[str]) -> Dict[str, int]:
    return {
        label: int(sum(value == label for value in labels))
        for label in MUSIC_SEMANTIC_LABELS
    }


def _soft_mass(matrix: np.ndarray) -> Dict[str, float]:
    mass = np.asarray(matrix, dtype=np.float64).sum(axis=0)
    return {
        label: float(mass[index])
        for index, label in enumerate(MUSIC_SEMANTIC_LABELS)
    }


def build_semantics(
    db_path: Path,
    out_path: Path,
    *,
    report_path: Optional[Path] = None,
    class_prior_path: Optional[Path] = None,
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
    intrinsic_low_threshold: float = 0.35,
    intrinsic_high_threshold: float = 0.65,
) -> Dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(str(db_path))
    if not 0.0 <= float(prior_alpha) <= 1.0:
        raise ValueError("prior_alpha must lie in [0,1]")
    if not 0.0 <= float(ambiguity_margin) <= 1.0:
        raise ValueError("ambiguity_margin must lie in [0,1]")
    if not 0.0 < intrinsic_low_threshold < intrinsic_high_threshold < 1.0:
        raise ValueError("intrinsic risk thresholds must satisfy 0 < low < high < 1")

    with np.load(db_path, allow_pickle=True) as data:
        db = {key: data[key] for key in data.files}
    if "paths" in db:
        n = int(len(db["paths"]))
    else:
        n = int(len(next(value for value in db.values() if np.asarray(value).ndim > 0)))

    desc = _matrix(db, "desc", n, 32)
    entry = _matrix(db, "entry", n, 144)
    exit_ = _matrix(db, "exit", n, 144)
    contact_entry = _matrix(db, "contact_entry", n, 4)
    contact_exit = _matrix(db, "contact_exit", n, 4)
    dance = _arr(db, "dance_keys", n, "unknown", object)
    categories = _arr(db, "dance_categories", n, "unknown", object)
    families = _arr(db, "event_families", n, "unknown", object)
    alignment = _arr(db, "music_alignment_labels", n, "unknown", object)
    stages = _arr(db, "motion_stage_roles", n, "unknown", object)
    energy = _arr(db, "energy_labels", n, "unknown", object)
    rhythm = _arr(db, "rhythm_labels", n, "unknown", object)
    locomotion = _arr(db, "locomotion_labels", n, "unknown", object)
    support = _arr(db, "support_labels", n, "unknown", object)
    source = _arr(db, "source_groups", n, "unknown", object)
    source_uid = _arr(db, "source_uids", n, "unknown", object)
    labels = _arr(db, "labels", n, "unknown", object)
    durations = _farr(db, "durations", n, 2.0)
    natural_min = _farr(db, "natural_duration_min", n, 1.5)
    natural_max = _farr(db, "natural_duration_max", n, 4.0)
    quality = _farr(db, "event_quality_scores", n, 0.5)
    semantic_confidence = _farr(db, "semantic_confidence", n, 0.5)

    raw_probabilities = np.stack(
        [
            event_probs_from_fields(
                dance_key=dance[index],
                event_family=families[index],
                music_alignment_label=alignment[index],
                energy_label=energy[index],
                rhythm_label=rhythm[index],
                locomotion_label=locomotion[index],
                support_label=support[index],
                quality=float(quality[index]),
                semantic_confidence=float(semantic_confidence[index]),
                desc=desc[index],
            )
            for index in range(n)
        ],
        axis=0,
    ).astype(np.float32)

    external_prior = _load_prior(class_prior_path)
    empirical_prior = np.maximum(raw_probabilities.mean(axis=0), 0.02)
    empirical_prior = empirical_prior / empirical_prior.sum()
    class_prior = external_prior if external_prior is not None else empirical_prior
    calibrated_probabilities = np.stack(
        [
            class_prior_adjustment(row, class_prior, alpha=float(prior_alpha))
            if prior_alpha > 0.0
            else row
            for row in raw_probabilities
        ],
        axis=0,
    ).astype(np.float32)

    aesd_rows: List[dict] = []
    event_semantics: List[str] = []
    raw_event_semantics: List[str] = []
    secondary_semantics: List[str] = []
    entropy_rows: List[float] = []
    margin_rows: List[float] = []
    ambiguous_rows: List[bool] = []
    intrinsic_risk_rows: List[float] = []
    intrinsic_profile_rows: List[str] = []
    affordance_rows: List[str] = []
    risk_numeric_rows: List[str] = []

    for index in range(n):
        raw_probs = raw_probabilities[index]
        probs = calibrated_probabilities[index]
        diagnostics = semantic_distribution_diagnostics(
            probs, ambiguity_margin=float(ambiguity_margin)
        )
        raw_top = MUSIC_SEMANTIC_LABELS[int(np.argmax(raw_probs))]
        top = str(diagnostics["top_label"])
        secondary = str(diagnostics["secondary_label"])
        intrinsic_risk, intrinsic_profile = intrinsic_transition_prior_from_arrays(
            entry[index],
            exit_[index],
            contact_entry[index],
            contact_exit[index],
            float(durations[index]),
            float(quality[index]),
            locomotion_label=locomotion[index],
            support_label=support[index],
            low_threshold=float(intrinsic_low_threshold),
            high_threshold=float(intrinsic_high_threshold),
        )
        risk_numeric = _risk_numeric_features(
            entry[index], exit_[index], contact_entry[index], contact_exit[index]
        )
        affordances = stage_affordance_from_probs(
            probs, explicit_stage=stages[index]
        )
        item = {
            "schema": AESD_SCHEMA_VERSION,
            "event_index": int(index),
            "event_id": str(labels[index]),
            "source_group": str(source[index]),
            "source_uid": str(source_uid[index]),
            "dance_key": str(dance[index]),
            "dance_category": str(categories[index]),
            "event_family": str(families[index]),
            "event_semantic": top,
            "raw_event_semantic": raw_top,
            "secondary_semantic": secondary,
            "semantic_entropy": float(diagnostics["normalized_entropy"]),
            "semantic_top2_margin": float(diagnostics["top2_margin"]),
            "semantic_ambiguous": bool(diagnostics["ambiguous"]),
            "music_alignment_label": str(alignment[index]),
            "music_alignment_probs": vector_to_prob_dict(probs),
            "raw_music_alignment_probs": vector_to_prob_dict(raw_probs),
            "route_affordance": affordances,
            "motion_stage_role": str(stages[index]),
            "energy_profile": str(energy[index]),
            "rhythm_profile": str(rhythm[index]),
            "locomotion_profile": str(locomotion[index]),
            "support_profile": str(support[index]),
            "natural_duration_sec": float(durations[index]),
            "natural_duration_range_sec": [
                float(natural_min[index]),
                float(natural_max[index]),
            ],
            "event_quality_score": float(quality[index]),
            "semantic_confidence": float(semantic_confidence[index]),
            "intrinsic_transition_prior": float(intrinsic_risk),
            "intrinsic_transition_profile": str(intrinsic_profile),
            # Compatibility aliases.  They must not be used as the sole runtime
            # inpainting trigger.
            "boundary_risk_score": float(intrinsic_risk),
            "boundary_risk_profile": str(intrinsic_profile),
            "entry_exit_state": risk_numeric,
            "reuse_penalty_key": (
                f"{str(source_uid[index])}::{str(dance[index])}::"
                f"{str(families[index])}"
            ),
        }
        aesd_rows.append(item)
        event_semantics.append(top)
        raw_event_semantics.append(raw_top)
        secondary_semantics.append(secondary)
        entropy_rows.append(float(diagnostics["normalized_entropy"]))
        margin_rows.append(float(diagnostics["top2_margin"]))
        ambiguous_rows.append(bool(diagnostics["ambiguous"]))
        intrinsic_risk_rows.append(float(intrinsic_risk))
        intrinsic_profile_rows.append(str(intrinsic_profile))
        affordance_rows.append(";".join(affordances))
        risk_numeric_rows.append(
            json.dumps(risk_numeric, ensure_ascii=False, sort_keys=True)
        )

    out = dict(db)
    event_uids = event_uids_from_generation_db(out)
    event_contract = make_event_db_contract(event_uids)
    out["event_uid_schema_version"] = np.asarray(EVENT_UID_SCHEMA, dtype=object)
    out["event_uids"] = event_uids
    out["event_db_contract_json"] = np.asarray(
        json.dumps(event_contract, sort_keys=True), dtype=object
    )
    out["aesd_schema_version"] = np.asarray(AESD_SCHEMA_VERSION, dtype=object)
    out["aesd_label_names"] = np.asarray(MUSIC_SEMANTIC_LABELS, dtype=object)
    out["aesd_semantics"] = np.asarray(aesd_rows, dtype=object)
    out["aesd_raw_music_alignment_probs"] = raw_probabilities
    out["aesd_music_alignment_probs"] = calibrated_probabilities
    out["aesd_event_semantics"] = np.asarray(event_semantics, dtype=object)
    out["aesd_raw_event_semantics"] = np.asarray(
        raw_event_semantics, dtype=object
    )
    out["aesd_secondary_semantic"] = np.asarray(
        secondary_semantics, dtype=object
    )
    out["aesd_semantic_entropy"] = np.asarray(entropy_rows, dtype=np.float32)
    out["aesd_top2_margin"] = np.asarray(margin_rows, dtype=np.float32)
    out["aesd_semantic_ambiguous"] = np.asarray(
        ambiguous_rows, dtype=np.bool_
    )
    out["aesd_class_prior"] = np.asarray(class_prior, dtype=np.float32)
    out["aesd_prior_alpha"] = np.asarray(float(prior_alpha), dtype=np.float32)
    out["aesd_route_affordance"] = np.asarray(affordance_rows, dtype=object)
    out["aesd_energy_profile"] = np.asarray(
        [row["energy_profile"] for row in aesd_rows], dtype=object
    )
    out["aesd_rhythm_profile"] = np.asarray(
        [row["rhythm_profile"] for row in aesd_rows], dtype=object
    )
    out["aesd_locomotion_profile"] = np.asarray(
        [row["locomotion_profile"] for row in aesd_rows], dtype=object
    )
    out["aesd_support_profile"] = np.asarray(
        [row["support_profile"] for row in aesd_rows], dtype=object
    )
    out["aesd_intrinsic_transition_prior"] = np.asarray(
        intrinsic_risk_rows, dtype=np.float32
    )
    out["aesd_intrinsic_transition_profile"] = np.asarray(
        intrinsic_profile_rows, dtype=object
    )
    # Historical aliases remain byte-identical to the intrinsic prior.
    out["aesd_boundary_risk"] = out["aesd_intrinsic_transition_prior"]
    out["aesd_boundary_risk_profile"] = out[
        "aesd_intrinsic_transition_profile"
    ]
    out["aesd_entry_exit_state_json"] = np.asarray(
        risk_numeric_rows, dtype=object
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    raw_histogram = _histogram(raw_event_semantics)
    calibrated_histogram = _histogram(event_semantics)
    risk_histogram = {
        profile: int(sum(value == profile for value in intrinsic_profile_rows))
        for profile in ("low", "medium", "high")
    }
    report = {
        "schema": AESD_SCHEMA_VERSION,
        "input_db": str(db_path.resolve()),
        "output_db": str(out_path.resolve()),
        "num_events": int(n),
        "event_db_contract": event_contract,
        "label_names": MUSIC_SEMANTIC_LABELS,
        "raw_event_semantic_histogram": raw_histogram,
        "event_semantic_histogram": calibrated_histogram,
        "raw_semantic_soft_mass": _soft_mass(raw_probabilities),
        "semantic_soft_mass": _soft_mass(calibrated_probabilities),
        "class_prior": vector_to_prob_dict(class_prior),
        "prior_alpha": float(prior_alpha),
        "ambiguity_margin": float(ambiguity_margin),
        "ambiguous_events": int(sum(ambiguous_rows)),
        "ambiguous_ratio": float(np.mean(ambiguous_rows)) if n else 0.0,
        "mean_normalized_entropy": float(np.mean(entropy_rows)) if n else 0.0,
        "mean_top2_margin": float(np.mean(margin_rows)) if n else 0.0,
        "intrinsic_transition_profile_histogram": risk_histogram,
        "intrinsic_risk_thresholds": {
            "low": float(intrinsic_low_threshold),
            "high": float(intrinsic_high_threshold),
        },
        "runtime_contract": {
            "intrinsic_prior_is_not_pairwise_risk": True,
            "masked_inpainting_requires_pairwise_transition_risk": True,
        },
        "arrays_added": [key for key in out if str(key).startswith("aesd_")],
        "first_event": aesd_rows[0] if aesd_rows else {},
        "ok": True,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build calibrated AESD semantics and intrinsic transition priors"
    )
    parser.add_argument("--db", required=True, help="Input events.npz")
    parser.add_argument("--out", required=True, help="Output events_aesd.npz")
    parser.add_argument("--json", default="", help="Optional audit JSON")
    parser.add_argument(
        "--class_prior_json",
        default="",
        help="Optional external 8-class prior JSON; empirical prior is used otherwise",
    )
    parser.add_argument(
        "--prior_alpha", type=float, default=DEFAULT_PRIOR_ALPHA
    )
    parser.add_argument(
        "--ambiguity_margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN
    )
    parser.add_argument("--intrinsic_low_threshold", type=float, default=0.35)
    parser.add_argument("--intrinsic_high_threshold", type=float, default=0.65)
    args = parser.parse_args(argv)
    report = build_semantics(
        Path(args.db),
        Path(args.out),
        report_path=Path(args.json) if args.json else None,
        class_prior_path=(
            Path(args.class_prior_json) if args.class_prior_json else None
        ),
        prior_alpha=float(args.prior_alpha),
        ambiguity_margin=float(args.ambiguity_margin),
        intrinsic_low_threshold=float(args.intrinsic_low_threshold),
        intrinsic_high_threshold=float(args.intrinsic_high_threshold),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
