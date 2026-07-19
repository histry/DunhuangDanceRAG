#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attach explicit performer-group metadata to an Event database."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def infer_group(text: str) -> str:
    value = str(text).strip().lower()
    if "female" in value:
        return "female"
    if "male" in value:
        return "male"
    return "unknown"


def augment_events_npz(path: str, require_known: bool = True) -> Dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)

    db = np.load(str(target), allow_pickle=True)
    payload = {key: db[key] for key in db.files}
    paths = np.asarray(payload["paths"], dtype=object)
    sources = np.asarray(
        payload.get("source_uids", paths),
        dtype=object,
    )
    existing = payload.get("performer_groups", payload.get("genders"))
    groups: List[str] = []
    for index, event_path in enumerate(paths):
        group = ""
        if existing is not None and index < len(existing):
            group = str(existing[index]).strip().lower()
        if group not in {"female", "male"}:
            source = sources[index] if index < len(sources) else event_path
            group = infer_group(str(source) + " " + str(event_path))
        groups.append(group)

    histogram = Counter(groups)
    if require_known and histogram.get("unknown", 0):
        raise RuntimeError(
            "Performer metadata contains unknown events: %s"
            % dict(histogram)
        )

    array = np.asarray(groups, dtype="<U16")
    payload["performer_groups"] = array
    payload["genders"] = array

    tmp = target.with_suffix(target.suffix + ".performer.tmp.npz")
    np.savez_compressed(str(tmp), **payload)
    os.replace(str(tmp), str(target))

    report = {
        "schema": "event_performer_metadata",
        "event_db": str(target),
        "num_events": int(len(groups)),
        "performer_group_histogram": dict(histogram),
        "unknown_is_error": bool(require_known),
    }
    report_path = target.with_suffix(target.suffix + ".performer.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument(
        "--allow_unknown",
        action="store_true",
        help="Do not fail when an event source cannot be classified.",
    )
    args = parser.parse_args(argv)
    report = augment_events_npz(
        args.events,
        require_known=not args.allow_unknown,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
