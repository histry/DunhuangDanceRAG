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
from support.event_identity import (  # noqa: E402
    event_uids_from_generation_db,
    make_event_db_contract,
    normalize_event_db_contract,
)


def _resolve_motion_ref_path(
    report_path: Path,
    motion_path: Path,
    raw_reference: Any,
) -> Path:
    candidates: List[Path] = []
    if raw_reference:
        raw = Path(str(raw_reference)).expanduser()
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend((report_path.parent / raw, ROOT / raw))
    if motion_path.suffix.lower() == ".npy":
        candidates.append(
            motion_path.with_name(motion_path.stem + ".motion_ref.npy")
        )
    checked = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"Cannot resolve motion_ref for report={report_path}; checked={checked}"
    )


def _database_fps(db: Dict[str, Any]) -> float:
    if "canonical_fps" not in db:
        raise RuntimeError("Generation DB has no canonical_fps contract")
    values = np.asarray(db["canonical_fps"], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    unique = np.unique(np.round(values, decimals=6))
    if unique.size != 1 or float(unique[0]) <= 0.0:
        raise RuntimeError(
            f"Generation DB canonical_fps is not unique and positive: {unique.tolist()}"
        )
    return float(unique[0])


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
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError(f"--fps must be positive and finite, got {args.fps!r}")

    motion_path = Path(args.motion).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    motion = np.load(motion_path, allow_pickle=True).astype(np.float32)
    if motion.ndim == 3:
        motion = motion[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(args.db, allow_pickle=True) as data:
        db = {k: data[k] for k in data.files}

    db_fps = _database_fps(db)
    if abs(db_fps - float(args.fps)) > 1.0e-6:
        raise RuntimeError(
            f"Heading audit FPS mismatch: Generation DB={db_fps}, runtime={args.fps}"
        )
    report_fps = report.get("fps", report.get("config", {}).get("fps"))
    try:
        if abs(float(report_fps) - float(args.fps)) > 1.0e-6:
            raise RuntimeError(
                f"Heading audit FPS mismatch: report={report_fps}, runtime={args.fps}"
            )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Generation report has invalid FPS: {report_fps!r}") from exc

    computed_db_contract = make_event_db_contract(
        event_uids_from_generation_db(db)
    )
    report_db_contract = normalize_event_db_contract(
        report.get("event_db_contract")
    )
    if report_db_contract != computed_db_contract:
        raise RuntimeError(
            "Heading audit Event-DB mismatch: "
            f"report={report_db_contract}, Generation DB={computed_db_contract}"
        )

    motion_ref_path = _resolve_motion_ref_path(
        report_path,
        motion_path,
        report.get("motion_ref_path"),
    )
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
    invalid_event_reference_count = 0

    intents = np.asarray(db.get("event_turn_intents", []), dtype=object)
    budgets = np.asarray(db.get("event_yaw_budget_rad", []), dtype=np.float32)
    deltas = np.asarray(db.get("event_stage_delta_yaw_rad", []), dtype=np.float32)

    for i, row0 in enumerate(assembly):
        event_id = int(row0.get("event_id", -1))
        core = row0.get("core_span")
        if event_id < 0 or core is None or len(core) < 2:
            invalid_event_reference_count += 1
            continue
        if event_id >= len(intents):
            invalid_event_reference_count += 1
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
    if invalid_event_reference_count:
        reasons.append(
            f"invalid_event_reference_count={invalid_event_reference_count}"
        )
    if not report.get("event_heading_planner"):
        reasons.append("missing_event_heading_planner_report")

    result = {
        "schema": "v46_50_generated_heading_audit",
        "ok": not reasons,
        "reasons": reasons,
        "motion": args.motion,
        "report": args.report,
        "db": args.db,
        "fps": float(args.fps),
        "event_db_contract": computed_db_contract,
        "frames": int(len(motion)),
        "planned_heading_error_deg_p95": planned_error_p95,
        "planned_heading_error_deg_max": planned_error_max,
        "nonturn_budget_fail_count": int(nonturn_budget_fail),
        "intent_mismatch_fail_count": int(intent_mismatch_fail),
        "invalid_event_reference_count": int(invalid_event_reference_count),
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
