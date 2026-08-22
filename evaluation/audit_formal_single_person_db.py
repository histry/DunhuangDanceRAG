#!/usr/bin/env python3
"""Fail-closed audit for formal one-body Event-DB inputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def audit_single_person_db(db: Mapping[str, Any]) -> dict[str, Any]:
    count = len(np.asarray(db.get("paths", [])))
    errors: list[str] = []
    required = {
        "solo_compatible",
        "solo_compatibilities",
        "solo_review_statuses",
        "recording_performer_counts",
        "dancer_ids",
        "dancer_id_statuses",
    }
    missing = sorted(required - set(db))
    if missing:
        errors.append(f"missing arrays={missing}")
    solo = np.asarray(db.get("solo_compatible", []), dtype=bool)
    if solo.shape != (count,) or not bool(np.all(solo)):
        errors.append(
            f"solo_compatible must be true for every formal event: shape={solo.shape}, "
            f"events={count}, false_count={int(np.count_nonzero(~solo)) if solo.size else count}"
        )
    statuses = np.asarray(db.get("dancer_id_statuses", []), dtype=object)
    dancer_ids = np.asarray(db.get("dancer_ids", []), dtype=object)
    verified_dancer_identity = bool(
        statuses.shape == (count,)
        and dancer_ids.shape == (count,)
        and count > 0
        and np.all(statuses == "verified")
        and all(bool(str(value).strip()) for value in dancer_ids)
    )
    themes = np.asarray(db.get("dance_categories", db.get("dance_keys", [])), dtype=object)
    sources = np.asarray(db.get("source_uids", []), dtype=object)
    report = {
        "schema": "formal_single_person_event_db_audit_v1",
        "ok": not errors,
        "num_events": count,
        "num_sources": len({str(value) for value in sources}),
        "solo_compatible_events": int(np.count_nonzero(solo)),
        "recording_performer_count_histogram": dict(
            Counter(map(str, np.asarray(db.get("recording_performer_counts", []))))
        ),
        "theme_histogram": dict(Counter(map(str, themes))),
        "verified_dancer_identity": verified_dancer_identity,
        "same_dancer_claim_supported": verified_dancer_identity,
        "identity_limit": (
            None
            if verified_dancer_identity
            else "one-body output only; cross-sequence same-dancer and dancer-disjoint claims are unsupported"
        ),
        "errors": errors,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with np.load(args.db, allow_pickle=True) as source:
        report = audit_single_person_db({key: source[key] for key in source.files})
    report["db"] = str(Path(args.db).resolve())
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
