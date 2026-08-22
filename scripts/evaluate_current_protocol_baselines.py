#!/usr/bin/env python3
"""Evaluate greedy and beam routes on one formal CTSR schedule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from baselines.current_protocol import beam_route, greedy_route
from routing.boundary_closed_loop import formal_candidate_state_from_slots
from support.event_identity import event_uids_from_generation_db


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate_top_k", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=32)
    args = parser.parse_args(argv)

    descriptor = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
    slots = descriptor.get("slots") if isinstance(descriptor, dict) else None
    if not isinstance(slots, list) or not slots:
        raise RuntimeError("Formal schedule has no slots list")
    with np.load(args.db, allow_pickle=True) as payload:
        db = {key: np.asarray(payload[key]) for key in payload.files}
    event_uids = event_uids_from_generation_db(db)
    _, candidate_lists, _ = formal_candidate_state_from_slots(
        slots, event_uids, boundary_top_k=int(args.candidate_top_k)
    )
    report = {
        "schema": "smpl14_ctsr_current_protocol_baseline_evaluation_v1",
        "schedule": str(Path(args.schedule).resolve()),
        "event_db": str(Path(args.db).resolve()),
        "historical_bvh_result_used": False,
        "greedy": greedy_route(slots, candidate_lists, db),
        "beam": beam_route(
            slots, candidate_lists, db, beam_size=int(args.beam_size)
        ),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out)}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
