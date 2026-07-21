#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.38 build complete Action Event Semantic Descriptor (AESD) metadata.

This script enriches an existing Event-RAG `events.npz` without touching motion
npy files.  It builds a routing-aware action-side descriptor for every event:

- cultural action class / dance_key
- event family and music response distribution
- stage role, energy/rhythm, locomotion and support profiles
- natural duration range
- entry/exit/contact state and boundary risk
- quality/safety/semantic confidence proxies

The output remains compatible with training/motion_models.py, but adds AESD
arrays that V46.38 routing and V44 MSSD-AESD Semantic OT explicitly consume.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from events.semantic_descriptor import (  # noqa: E402
    AESD_SCHEMA_VERSION,
    MUSIC_SEMANTIC_LABELS,
    boundary_risk_from_arrays,
    event_probs_from_fields,
    stage_affordance_from_probs,
    vector_to_prob_dict,
)
from support.event_identity import (  # noqa: E402
    EVENT_UID_SCHEMA,
    event_uids_from_generation_db,
    make_event_db_contract,
)


def _arr(db: Dict[str, Any], key: str, n: int, default: Any, dtype=object) -> np.ndarray:
    if key in db:
        return np.asarray(db[key], dtype=dtype)
    return np.asarray([default] * n, dtype=dtype)


def _farr(db: Dict[str, Any], key: str, n: int, default: float) -> np.ndarray:
    if key in db:
        return np.asarray(db[key], dtype=np.float32)
    return np.ones((n,), dtype=np.float32) * float(default)


def _matrix(db: Dict[str, Any], key: str, n: int, width: int) -> np.ndarray:
    if key in db:
        x = np.asarray(db[key], dtype=np.float32)
        if x.ndim == 2 and x.shape[0] == n:
            return x
    return np.zeros((n, width), dtype=np.float32)


def _risk_numeric_features(entry: np.ndarray, exit_: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        e = np.asarray(entry, dtype=np.float32).reshape(-1)
        x = np.asarray(exit_, dtype=np.float32).reshape(-1)
        out["entry_velocity_norm"] = float(np.linalg.norm(e[72:144]) / max(1, 72)) if e.size >= 144 else 0.0
        out["exit_velocity_norm"] = float(np.linalg.norm(x[72:144]) / max(1, 72)) if x.size >= 144 else 0.0
        out["entry_exit_pose_gap_self"] = float(np.mean((x[:72] - e[:72]) ** 2)) if e.size >= 72 and x.size >= 72 else 0.0
    except Exception:
        out.update({"entry_velocity_norm": 0.0, "exit_velocity_norm": 0.0, "entry_exit_pose_gap_self": 0.0})
    try:
        cc0 = np.asarray(c0, dtype=np.float32).reshape(-1)
        cc1 = np.asarray(c1, dtype=np.float32).reshape(-1)
        m = min(cc0.size, cc1.size)
        out["contact_jump_self"] = float(np.mean(np.abs(cc1[:m] - cc0[:m]))) if m else 0.0
    except Exception:
        out["contact_jump_self"] = 0.0
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Input events.npz")
    ap.add_argument("--out", required=True, help="Output events_aesd.npz")
    ap.add_argument("--json", default=None, help="Optional AESD audit JSON")
    args = ap.parse_args(argv)

    src = Path(args.db)
    if not src.exists():
        raise FileNotFoundError(str(src))
    data = np.load(src, allow_pickle=True)
    db = {k: data[k] for k in data.files}
    if "paths" in db:
        n = int(len(db["paths"]))
    else:
        # fall back to first 1D array
        n = int(len(next(iter(db.values()))))

    desc = _matrix(db, "desc", n, 32)
    entry = _matrix(db, "entry", n, 144)
    exit_ = _matrix(db, "exit", n, 144)
    c0 = _matrix(db, "contact_entry", n, 4)
    c1 = _matrix(db, "contact_exit", n, 4)
    dance = _arr(db, "dance_keys", n, "unknown", object)
    cats = _arr(db, "dance_categories", n, "unknown", object)
    fam = _arr(db, "event_families", n, "unknown", object)
    align = _arr(db, "music_alignment_labels", n, "unknown", object)
    stages = _arr(db, "motion_stage_roles", n, "unknown", object)
    energy = _arr(db, "energy_labels", n, "unknown", object)
    rhythm = _arr(db, "rhythm_labels", n, "unknown", object)
    loco = _arr(db, "locomotion_labels", n, "unknown", object)
    support = _arr(db, "support_labels", n, "unknown", object)
    source = _arr(db, "source_groups", n, "unknown", object)
    source_uid = _arr(db, "source_uids", n, "unknown", object)
    labels = _arr(db, "labels", n, "unknown", object)
    durations = _farr(db, "durations", n, 2.0)
    nat_min = _farr(db, "natural_duration_min", n, 1.5)
    nat_max = _farr(db, "natural_duration_max", n, 4.0)
    quality = _farr(db, "event_quality_scores", n, 0.5)
    sem_conf = _farr(db, "semantic_confidence", n, 0.5)

    aesd: List[dict] = []
    prob_rows = []
    event_sem = []
    boundary_risk = []
    boundary_prof = []
    affordance_rows = []
    risk_numeric_rows = []

    for i in range(n):
        probs = event_probs_from_fields(
            dance_key=dance[i],
            event_family=fam[i],
            music_alignment_label=align[i],
            energy_label=energy[i],
            rhythm_label=rhythm[i],
            locomotion_label=loco[i],
            support_label=support[i],
            quality=float(quality[i]),
            semantic_confidence=float(sem_conf[i]),
            desc=desc[i] if desc.ndim == 2 and i < desc.shape[0] else None,
        )
        top = MUSIC_SEMANTIC_LABELS[int(np.argmax(probs))]
        risk, prof = boundary_risk_from_arrays(
            entry[i] if entry.ndim == 2 else None,
            exit_[i] if exit_.ndim == 2 else None,
            c0[i] if c0.ndim == 2 else None,
            c1[i] if c1.ndim == 2 else None,
            float(durations[i]),
            float(quality[i]),
            locomotion_label=loco[i],
            support_label=support[i],
        )
        rn = _risk_numeric_features(entry[i], exit_[i], c0[i], c1[i])
        aff = stage_affordance_from_probs(probs, explicit_stage=stages[i])
        item = {
            "schema": AESD_SCHEMA_VERSION,
            "event_index": int(i),
            "event_id": str(labels[i]),
            "source_group": str(source[i]),
            "source_uid": str(source_uid[i]),
            "dance_key": str(dance[i]),
            "dance_category": str(cats[i]),
            "event_family": str(fam[i]),
            "event_semantic": str(top),
            "music_alignment_label": str(align[i]),
            "music_alignment_probs": vector_to_prob_dict(probs),
            "route_affordance": aff,
            "motion_stage_role": str(stages[i]),
            "energy_profile": str(energy[i]),
            "rhythm_profile": str(rhythm[i]),
            "locomotion_profile": str(loco[i]),
            "support_profile": str(support[i]),
            "natural_duration_sec": float(durations[i]),
            "natural_duration_range_sec": [float(nat_min[i]), float(nat_max[i])],
            "event_quality_score": float(quality[i]),
            "semantic_confidence": float(sem_conf[i]),
            "boundary_risk_score": float(risk),
            "boundary_risk_profile": str(prof),
            "entry_exit_state": rn,
            "reuse_penalty_key": f"{str(source_uid[i])}::{str(dance[i])}::{str(fam[i])}",
        }
        aesd.append(item)
        prob_rows.append(probs.astype(np.float32))
        event_sem.append(top)
        boundary_risk.append(risk)
        boundary_prof.append(prof)
        affordance_rows.append(";".join(aff))
        risk_numeric_rows.append(json.dumps(rn, ensure_ascii=False, sort_keys=True))

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
    out["aesd_semantics"] = np.asarray(aesd, dtype=object)
    out["aesd_music_alignment_probs"] = np.stack(prob_rows).astype(np.float32)
    out["aesd_event_semantics"] = np.asarray(event_sem, dtype=object)
    out["aesd_route_affordance"] = np.asarray(affordance_rows, dtype=object)
    out["aesd_energy_profile"] = np.asarray([x["energy_profile"] for x in aesd], dtype=object)
    out["aesd_rhythm_profile"] = np.asarray([x["rhythm_profile"] for x in aesd], dtype=object)
    out["aesd_locomotion_profile"] = np.asarray([x["locomotion_profile"] for x in aesd], dtype=object)
    out["aesd_support_profile"] = np.asarray([x["support_profile"] for x in aesd], dtype=object)
    out["aesd_boundary_risk"] = np.asarray(boundary_risk, dtype=np.float32)
    out["aesd_boundary_risk_profile"] = np.asarray(boundary_prof, dtype=object)
    out["aesd_entry_exit_state_json"] = np.asarray(risk_numeric_rows, dtype=object)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    hist = {lab: int(sum(x == lab for x in event_sem)) for lab in MUSIC_SEMANTIC_LABELS}
    risk_hist = {p: int(sum(x == p for x in boundary_prof)) for p in ["low", "medium", "high"]}
    report = {
        "schema": AESD_SCHEMA_VERSION,
        "input_db": str(src),
        "output_db": str(out_path),
        "num_events": int(n),
        "event_db_contract": event_contract,
        "label_names": MUSIC_SEMANTIC_LABELS,
        "event_semantic_histogram": hist,
        "boundary_risk_histogram": risk_hist,
        "arrays_added": [k for k in out.keys() if str(k).startswith("aesd_")],
        "first_event": aesd[0] if aesd else {},
    }
    if args.json:
        jp = Path(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
