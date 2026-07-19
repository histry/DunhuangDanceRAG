#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit a V46.50 generated whole-song motion against the heading plan."""
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
    angle_diff,
    heading_metrics_np,
    root_yaw_np,
    slot_turn_policy,
    wrap_angle,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--allow_failed", action="store_true")
    args = ap.parse_args(argv)

    motion = np.load(args.motion, allow_pickle=True).astype(np.float32)
    if motion.ndim == 3:
        motion = motion[0]
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    data = np.load(args.db, allow_pickle=True)
    db = {k: data[k] for k in data.files}

    motion_ref_path = Path(
        report.get("motion_ref_path")
        or str(args.motion).replace(".npy", ".motion_ref.npy")
    )
    if not motion_ref_path.is_file():
        raise FileNotFoundError(motion_ref_path)
    reference = np.load(motion_ref_path, allow_pickle=True).astype(np.float32)
    if reference.ndim == 3:
        reference = reference[0]

    n = min(len(motion), len(reference))
    yaw_final = root_yaw_np(motion[:n])
    yaw_ref = root_yaw_np(reference[:n])
    err = np.abs(np.degrees(wrap_angle(yaw_final - yaw_ref)))
    planned_error_p95 = float(np.percentile(err, 95)) if len(err) else 0.0
    planned_error_max = float(np.max(err)) if len(err) else 0.0

    assembly = (
        report.get("stage_reports", {}).get("closed_loop_concat")
        or report.get("stage_reports", {}).get("concat")
        or []
    )
    slots = report.get("slots", [])
    rows: List[Dict[str, Any]] = []
    nonturn_budget_fail = 0
    intent_mismatch_fail = 0

    intents = np.asarray(db.get("event_turn_intents", []), dtype=object)
    budgets = np.asarray(db.get("event_yaw_budget_rad", []), dtype=np.float32)
    deltas = np.asarray(db.get("event_stage_delta_yaw_rad", []), dtype=np.float32)

    for i, row0 in enumerate(assembly):
        event_id = int(row0.get("event_id", -1))
        core = row0.get("core_span")
        if event_id < 0 or core is None or len(core) < 2:
            continue
        a, b = max(0, int(core[0])), min(len(motion), int(core[1]))
        if b <= a:
            continue
        m = heading_metrics_np(motion[a:b], fps=args.fps)
        intent = str(intents[event_id]) if 0 <= event_id < len(intents) else "unknown"
        budget = float(budgets[event_id]) if 0 <= event_id < len(budgets) else np.pi
        planned_delta = float(deltas[event_id]) if 0 <= event_id < len(deltas) else 0.0
        slot = slots[i] if i < len(slots) else {}
        policy = slot_turn_policy(slot)
        actual_net = float(m["net_yaw_rad"])

        budget_fail = bool(
            intent in {"none", "uncertain_turn"}
            and abs(actual_net) > budget + np.radians(5.0)
        )
        mismatch = bool(
            policy["slot_turn_intent"] == "non_turn_anchor"
            and intent == "explicit_spin"
        )
        if budget_fail:
            nonturn_budget_fail += 1
        if mismatch:
            intent_mismatch_fail += 1

        rows.append({
            "slot": int(i),
            "event_id": event_id,
            "intent": intent,
            "slot_policy": policy["slot_turn_intent"],
            "core_start": a,
            "core_end": b,
            "planned_event_delta_deg": float(np.degrees(planned_delta)),
            "actual_core_net_yaw_deg": float(np.degrees(actual_net)),
            "budget_deg": float(np.degrees(budget)),
            "nonturn_budget_fail": budget_fail,
            "intent_mismatch_fail": mismatch,
            "actual_longest_same_sign_turn_seconds": float(
                m["longest_same_sign_turn_seconds"]
            ),
        })

    reasons: List[str] = []
    max_plan_err = float(
        os.environ.get("V46_50_PLANNED_HEADING_ERROR_P95_MAX_DEG", 2.0)
    )
    if planned_error_p95 > max_plan_err:
        reasons.append(
            f"planned_heading_error_p95={planned_error_p95:.3f}>limit={max_plan_err:.3f}"
        )
    if nonturn_budget_fail:
        reasons.append(f"nonturn_budget_fail={nonturn_budget_fail}")
    if intent_mismatch_fail:
        reasons.append(f"intent_mismatch_fail={intent_mismatch_fail}")
    if not report.get("event_heading_planner"):
        reasons.append("missing_event_heading_planner_report")

    result = {
        "schema": "v46_50_generated_heading_audit",
        "ok": not reasons,
        "reasons": reasons,
        "motion": args.motion,
        "report": args.report,
        "db": args.db,
        "frames": int(len(motion)),
        "planned_heading_error_deg_p95": planned_error_p95,
        "planned_heading_error_deg_max": planned_error_max,
        "nonturn_budget_fail_count": int(nonturn_budget_fail),
        "intent_mismatch_fail_count": int(intent_mismatch_fail),
        "whole_song_heading_metrics": heading_metrics_np(motion, fps=args.fps),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.csv:
        cp = Path(args.csv)
        cp.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            "slot", "event_id", "intent", "slot_policy", "core_start",
            "core_end", "planned_event_delta_deg", "actual_core_net_yaw_deg",
            "budget_deg", "nonturn_budget_fail", "intent_mismatch_fail",
            "actual_longest_same_sign_turn_seconds",
        ]
        with cp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in keys})

    print(json.dumps({
        "out": str(out),
        "ok": result["ok"],
        "reasons": reasons,
        "planned_heading_error_deg_p95": planned_error_p95,
        "nonturn_budget_fail_count": nonturn_budget_fail,
        "intent_mismatch_fail_count": intent_mismatch_fail,
    }, ensure_ascii=False, indent=2))
    return 0 if result["ok"] or args.allow_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
