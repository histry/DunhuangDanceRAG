#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard audit for a V46.50 event-heading Event-RAG database."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.heading import (  # noqa: E402
    heading_metrics_np,
)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--allow_failed", action="store_true")
    args = ap.parse_args(argv)

    data = np.load(args.db, allow_pickle=True)
    db = {k: data[k] for k in data.files}
    required = [
        "heading_contract_schema_version",
        "paths",
        "event_entry_heading_rad",
        "event_stage_delta_yaw_rad",
        "event_yaw_budget_rad",
        "event_turn_intents",
        "event_heading_quality",
        "event_heading_valid",
        "source_uids",
        "starts",
        "ends",
    ]
    missing = [k for k in required if k not in db]
    if missing:
        report = {
            "schema": "v46_50_event_heading_db_audit",
            "ok": False,
            "reasons": [f"missing_arrays:{','.join(missing)}"],
        }
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if args.allow_failed else 2

    n = len(db["paths"])
    paths = np.asarray(db["paths"], dtype=object)
    entry = np.asarray(db["event_entry_heading_rad"], dtype=np.float32)
    delta = np.asarray(db["event_stage_delta_yaw_rad"], dtype=np.float32)
    budget = np.asarray(db["event_yaw_budget_rad"], dtype=np.float32)
    intents = np.asarray(db["event_turn_intents"], dtype=object)
    quality = np.asarray(db["event_heading_quality"], dtype=np.float32)
    valid = np.asarray(db["event_heading_valid"], dtype=bool)
    source_uids = np.asarray(db["source_uids"], dtype=object)

    rows: List[Dict[str, Any]] = []
    missing_motion = 0
    metric_mismatch = 0
    budget_violation = 0
    nonturn_mechanical_fail = 0

    tol = np.radians(2.0)
    for i in range(n):
        p = Path(str(paths[i]))
        row: Dict[str, Any] = {
            "event_id": int(i),
            "path": str(p),
            "source_uid": str(source_uids[i]),
            "intent": str(intents[i]),
            "entry_heading_deg": float(np.degrees(entry[i])),
            "net_yaw_deg_db": float(np.degrees(delta[i])),
            "budget_deg": float(np.degrees(budget[i])),
            "heading_quality": float(quality[i]),
            "valid": bool(valid[i]),
        }
        if not p.is_file():
            missing_motion += 1
            row["motion_ok"] = False
            row["reason"] = "missing_motion_file"
            rows.append(row)
            continue
        x = np.load(p, allow_pickle=True).astype(np.float32)
        if x.ndim == 3:
            x = x[0]
        metrics = heading_metrics_np(x, fps=args.fps)
        row.update({
            "motion_ok": True,
            "net_yaw_deg_recomputed": float(metrics["net_yaw_deg"]),
            "absolute_yaw_deg": float(metrics["absolute_yaw_deg"]),
            "longest_same_sign_turn_seconds": float(
                metrics["longest_same_sign_turn_seconds"]
            ),
            "mechanical_spin_ratio": float(metrics["mechanical_spin_ratio"]),
        })
        if abs(float(metrics["net_yaw_rad"]) - float(delta[i])) > np.radians(3.0):
            metric_mismatch += 1
            row["metric_mismatch"] = True
        else:
            row["metric_mismatch"] = False

        is_nonturn = str(intents[i]) in {"none", "uncertain_turn"}
        violation = abs(float(metrics["net_yaw_rad"])) > float(budget[i]) + tol
        row["budget_violation"] = bool(violation)
        if violation:
            budget_violation += 1
        mech_fail = bool(
            is_nonturn
            and float(metrics["longest_same_sign_turn_seconds"])
            > float(os.environ.get("V46_50_NON_TURN_MAX_MECHANICAL_SECONDS", 3.0))
        )
        row["nonturn_mechanical_fail"] = mech_fail
        if mech_fail:
            nonturn_mechanical_fail += 1
        rows.append(row)

    entry_p95 = float(np.percentile(np.abs(np.degrees(entry)), 95)) if n else 0.0
    max_entry = float(
        os.environ.get("V46_50_ENTRY_HEADING_P95_MAX_DEG", 5.0)
    )
    reasons = []
    if not bool(np.all(valid)):
        reasons.append(f"invalid_heading_events={int(np.sum(~valid))}")
    if missing_motion:
        reasons.append(f"missing_motion_files={missing_motion}")
    if metric_mismatch:
        reasons.append(f"heading_metric_mismatch={metric_mismatch}")
    if budget_violation:
        reasons.append(f"budget_violation={budget_violation}")
    if nonturn_mechanical_fail:
        reasons.append(f"nonturn_mechanical_fail={nonturn_mechanical_fail}")
    if entry_p95 > max_entry:
        reasons.append(
            f"entry_heading_p95_deg={entry_p95:.3f}>limit={max_entry:.3f}"
        )
    if len(set(map(str, source_uids.tolist()))) < 2:
        reasons.append("fewer_than_two_source_uids")

    intent_hist = {
        k: int(np.sum(intents == k))
        for k in sorted(set(map(str, intents.tolist())))
    }
    report = {
        "schema": "v46_50_event_heading_db_audit",
        "db": args.db,
        "ok": not reasons,
        "reasons": reasons,
        "num_events": int(n),
        "num_source_uids": int(len(set(map(str, source_uids.tolist())))),
        "intent_histogram": intent_hist,
        "entry_heading_abs_deg_p95": entry_p95,
        "entry_heading_abs_deg_max": float(
            np.max(np.abs(np.degrees(entry))) if n else 0.0
        ),
        "heading_quality_p05": float(np.percentile(quality, 5)) if n else 0.0,
        "heading_quality_median": float(np.median(quality)) if n else 0.0,
        "missing_motion_files": int(missing_motion),
        "metric_mismatch_count": int(metric_mismatch),
        "budget_violation_count": int(budget_violation),
        "nonturn_mechanical_fail_count": int(nonturn_mechanical_fail),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.csv:
        cp = Path(args.csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            "event_id", "path", "source_uid", "intent",
            "entry_heading_deg", "net_yaw_deg_db", "net_yaw_deg_recomputed",
            "absolute_yaw_deg", "budget_deg", "heading_quality", "valid",
            "longest_same_sign_turn_seconds", "mechanical_spin_ratio",
            "budget_violation", "nonturn_mechanical_fail", "metric_mismatch",
            "motion_ok", "reason",
        ]
        with cp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in keys})

    print(json.dumps({
        "out": str(out),
        "ok": report["ok"],
        "reasons": reasons,
        "num_events": n,
        "num_source_uids": report["num_source_uids"],
        "intent_histogram": intent_hist,
        "entry_heading_abs_deg_p95": entry_p95,
    }, ensure_ascii=False, indent=2))
    return 0 if report["ok"] or args.allow_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
