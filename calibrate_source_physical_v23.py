#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

JERK_SPECS = (
    ("unit_bone_joint_jerk_s3_p95", "unit_bone_joint_jerk_p95_regressed_vs_source", "SOURCE_PHYSICAL_UNIT_BONE_JERK_P95_FLOOR_S3", 1900.0),
    ("unit_bone_joint_jerk_s3_p99", "unit_bone_joint_jerk_p99_regressed_vs_source", "SOURCE_PHYSICAL_UNIT_BONE_JERK_P99_FLOOR_S3", 4500.0),
    ("unit_bone_joint_jerk_window_p95_max_s3", "unit_bone_joint_jerk_window_regressed_vs_source", "SOURCE_PHYSICAL_UNIT_BONE_JERK_WINDOW_FLOOR_S3", 11000.0),
    ("unit_bone_extremity_jerk_s3_p95", "unit_bone_extremity_jerk_p95_regressed_vs_source", "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P95_FLOOR_S3", 2800.0),
    ("unit_bone_extremity_jerk_s3_p99", "unit_bone_extremity_jerk_p99_regressed_vs_source", "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_P99_FLOOR_S3", 7500.0),
    ("unit_bone_extremity_jerk_window_p95_max_s3", "unit_bone_extremity_jerk_window_regressed_vs_source", "SOURCE_PHYSICAL_UNIT_BONE_EXTREMITY_JERK_WINDOW_FLOOR_S3", 22000.0),
)
P001_REASON = "foot_penetration_p001_regressed_vs_source"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="output/source_contract_validation_v2/requalified_gate_v2_2.json",
    )
    ap.add_argument(
        "--output",
        default="output/source_contract_validation_v2/source_calibration_v2_3_dry_run.json",
    )
    ap.add_argument(
        "--p001-epsilon-m",
        type=float,
        default=env_float("SOURCE_PHYSICAL_FOOT_PENETRATION_P001_EPS_M", 0.002),
    )
    args = ap.parse_args()

    src = Path(args.input)
    data = json.loads(src.read_text(encoding="utf-8"))
    floors = {key: env_float(env_name, default) for key, _, env_name, default in JERK_SPECS}
    jerk_reasons = {reason for _, reason, _, _ in JERK_SPECS}

    rows: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()

    for row in data.get("rows", []):
        gate = row.get("gate", {})
        checks = gate.get("relative_checks", {})
        old_reasons = list(row.get("reasons", []))
        new_reasons = [r for r in old_reasons if r not in jerk_reasons and r != P001_REASON]
        calibration: dict[str, Any] = {}

        for key, reason, _, _ in JERK_SPECS:
            d = checks.get(key)
            if not isinstance(d, dict):
                new_reasons.append(f"missing_calibration_check:{key}")
                continue
            ref = float(d["reference"])
            cand = float(d["candidate"])
            ratio = float(d["ratio"])
            margin = float(d["margin"])
            relative_allowed = ref * ratio + margin
            floor = float(floors[key])
            allowed = max(relative_allowed, floor)
            passed = cand <= allowed
            calibration[key] = {
                "reference": ref,
                "candidate": cand,
                "ratio": ratio,
                "margin": margin,
                "relative_allowed": relative_allowed,
                "source_only_noise_floor_s3": floor,
                "allowed": allowed,
                "passed": passed,
            }
            if not passed:
                new_reasons.append(reason)

        p001 = checks.get("foot_penetration_p001_m")
        if isinstance(p001, dict):
            cand = float(p001["candidate"])
            allowed_minimum = float(p001["allowed_minimum"])
            eps = float(args.p001_epsilon_m)
            passed = cand >= allowed_minimum - eps
            calibration["foot_penetration_p001_m"] = {
                **p001,
                "comparison_epsilon_m": eps,
                "effective_allowed_minimum": allowed_minimum - eps,
                "passed": passed,
            }
            if not passed:
                new_reasons.append(P001_REASON)
        else:
            new_reasons.append("missing_calibration_check:foot_penetration_p001_m")

        # Stable order, no duplicate reasons.
        seen: set[str] = set()
        new_reasons = [r for r in new_reasons if not (r in seen or seen.add(r))]
        ok = len(new_reasons) == 0
        reason_counter.update(new_reasons)
        rows.append(
            {
                "source": row.get("source"),
                "old_ok": bool(row.get("ok")),
                "new_ok": ok,
                "old_reasons": old_reasons,
                "new_reasons": new_reasons,
                "calibration": calibration,
            }
        )

    payload = {
        "schema": "source_physical_calibration_v2_3_dry_run",
        "input": str(src),
        "source_only_noise_floors_s3": floors,
        "foot_penetration_p001_comparison_epsilon_m": float(args.p001_epsilon_m),
        "num_inputs": len(rows),
        "num_ok": sum(bool(r["new_ok"]) for r in rows),
        "num_failed": sum(not bool(r["new_ok"]) for r in rows),
        "failure_reason_counts": dict(reason_counter.most_common()),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print("V2.3 SOURCE-ONLY CALIBRATION DRY RUN")
    print("=" * 100)
    print("num_inputs =", payload["num_inputs"])
    print("num_ok     =", payload["num_ok"])
    print("num_failed =", payload["num_failed"])
    print("\nFAILURE REASON COUNTS")
    for reason, count in reason_counter.most_common():
        print(f"{count:2d}  {reason}")
    print("\nPER SOURCE")
    for row in rows:
        status = "PASS" if row["new_ok"] else "FAIL"
        print(f"{row['source']:<26s} {status:<4s} {row['new_reasons']}")
    print("\nSAVED:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
